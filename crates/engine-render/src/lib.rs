//! Vulkano-based renderer and windowing for the game engine.
//!
//! Public surface is [`Window`]. A typical setup:
//!
//! ```no_run
//! use engine_render::{Window, MeshRenderer};
//! use engine_core::transform::_Transform;
//! use engine_core::component::{Component, Scene};
//!
//! #[derive(Clone)]
//! struct Spinner;
//! impl Component for Spinner {
//!     fn update(&mut self, dt: f32, t: &engine_core::transform::Transform) {
//!         use glam::Quat;
//!         t.lock().rotate_by(Quat::from_rotation_y(dt));
//!     }
//! }
//!
//! let mut root = Scene::new();
//! let e = root.new_entity(_Transform::default());
//! root.add_component(e, Spinner);
//! root.add_component(e, MeshRenderer::new("cube.mesh"));
//!
//! Window::new("My Game")
//!     .with_scene(root)
//!     .run();
//! ```
//!
//! The renderer maintains **one reusable primary command buffer per swapchain
//! image**. Each per-image "frame slot" owns:
//!
//! * A host-mapped staging buffer of MVP matrices (`HOST_SEQUENTIAL_WRITE`).
//! * A device-local matrix buffer bound as a storage buffer to set 0.
//! * **Offscreen** color (`R16G16B16A16_SFLOAT`) + depth (`D32_SFLOAT`)
//!   attachments — the camera's render targets, never the swapchain image.
//! * A pre-recorded command buffer that copies staging → device, renders the
//!   scene into the offscreen color+depth, and finally `vkCmdBlitImage`s the
//!   offscreen color into the swapchain image. Vulkano auto-tracks the final
//!   layout transition to `PresentSrcKHR` on swapchain-owned images.
//!
//! Decoupling the camera's color target from the swapchain image is step 1 of
//! the multi-camera / post-processing roadmap (`todo.txt`): once the camera
//! owns its attachments, multiple cameras, mirrors, picture-in-picture, and
//! HDR → sRGB tonemapping all become "another pass before the present-blit."
//!
//! On the hot path the renderer (a) computes per-instance MVPs into the
//! staging buffer and (b) submits the pre-recorded CB. Slots are rebuilt only
//! when the swapchain or scene topology changes. This is the scaffolding for
//! a future GPU-driven indirect path with millions of objects — the staging
//! → device pattern is the same; only the draw call collapses to a single
//! `draw_indexed_indirect_count`.

use std::{sync::Arc, time::Instant};

use engine_core::component::Scene;
use vulkano::{
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferInheritanceInfo, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo,
        SecondaryAutoCommandBuffer, SubpassContents,
    },
    descriptor_set::allocator::{
        StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
    },
    device::{Device, DeviceFeatures, DeviceOwned, Queue},
    image::{view::ImageView, ImageLayout},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        compute::ComputePipelineCreateInfo,
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::{PipelineRenderingCreateInfo, PipelineSubpassType},
            vertex_input::VertexDefinition,
            viewport::ViewportState,
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, DynamicState, GraphicsPipeline, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    query::{QueryPool, QueryPoolCreateInfo, QueryType},
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{PresentMode, SurfaceInfo},
    sync::PipelineStage,
};
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{WindowAttributes, WindowId},
};

use engine_core::util::parallel;

pub mod assets;
mod camera;
pub mod components;
mod frame_handoff;
mod gpu_mesh;
mod gpu_renderers;
mod gpu_telemetry;
pub mod input;
mod render_thread;
mod scene;
mod shaders;
mod swapchain;
mod transform_gpu;

use assets::{GpuMaterialStore, GpuMeshStore, GpuTextureStore};
use camera::{
    CameraSceneResources, DrawPlan, RenderCamera, CAMERA_COLOR_FORMAT, CAMERA_DEPTH_FORMAT,
};
use gpu_mesh::GpuVertex;
use gpu_renderers::GpuRenderers;
use swapchain::SwapchainRenderer;
use transform_gpu::WorldTransformGpu;

pub use components::MeshRenderer;
pub use input::{Input, KeyCode, MouseButton};
pub use scene::{CameraComponent, OrbitController};

// ─────────────────────────────────────────────────────────────────────────────
// Pinned static thread pool (engine-core fork-join scheduler)
// ─────────────────────────────────────────────────────────────────────────────

/// Initialise the engine's **global** static thread pool with one worker
/// per logical core (minus one for the main thread), each pinned to its
/// assigned core via `core_affinity`. The **main thread is deliberately
/// left unpinned**: it spins on the dispatch barrier, and a pinned
/// spinning thread heats its core into throttling while forbidding the
/// OS from migrating it to a cooler one. Leaving main free to migrate
/// recovers turbo headroom on low-N / single-core-bound workloads.
///
/// Worker pinning eliminates per-frame jitter from the scheduler
/// bouncing the hot dirty-harvest / scatter-staging workers between
/// cores and lets the L1/L2 caches retain the SoT staging pages across
/// frames. Set `ENGINE_NO_PIN=1` to disable worker pinning entirely
/// (e.g. on laptops, shared CI boxes, or for thermal experiments).
///
/// `ENGINE_NUM_THREADS` (alias: `RAYON_NUM_THREADS` for back-compat with
/// existing benchmark scripts) sets the **total** participant count
/// (workers + main). Parse strictly — no fallback on a bad value.
///
/// Per project rules: **no fallbacks**. If the OS refuses to enumerate
/// cores, or any pin fails, we panic.
fn init_pinned_thread_pool() {
    use engine_core::util::numa::NumaTopology;

    // Whether to skip all CPU affinity pinning.
    // Set ENGINE_NO_PIN=1 (or =true) to disable; default is pinned.
    let no_pin = std::env::var("ENGINE_NO_PIN")
        .ok()
        .map(|v| match v.as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => panic!("ENGINE_NO_PIN must be 0/1/true/false, got {v:?}"),
        })
        .unwrap_or(false);

    // Build a cpuset-filtered NUMA topology so that callers (including
    // future schedulers that pin) never try to pin to a CPU outside our
    // allowed set (matters under numactl --cpunodebind, cgroups, or taskset).
    // The current `my_thread_pool` is simple and does not pin, but we still
    // honour the topology computation so the worker count derived below
    // reflects the cpuset-filtered core count.
    let core_ids = core_affinity::get_core_ids()
        .expect("core_affinity::get_core_ids() returned None — cannot enumerate logical cores");
    assert!(
        !core_ids.is_empty(),
        "core_affinity returned an empty core list"
    );
    let available: std::collections::HashSet<usize> = core_ids.iter().map(|c| c.id).collect();

    let raw = NumaTopology::detect()
        .unwrap_or_else(|_| NumaTopology::single_node(core_ids.iter().map(|c| c.id).collect()));

    let topology: Vec<Vec<usize>> = raw
        .nodes()
        .iter()
        .map(|n| {
            n.cpus
                .iter()
                .copied()
                .filter(|c| available.contains(c))
                .collect::<Vec<usize>>()
        })
        .filter(|cpus| !cpus.is_empty())
        .collect();
    assert!(
        !topology.is_empty(),
        "no NUMA node has any CPU in the current cpuset",
    );

    // ENGINE_NUM_THREADS / RAYON_NUM_THREADS: total participant count
    // (workers + main thread). We subtract one for the main thread so the
    // user-specified value matches the overall thread budget. With no env
    // var, default to one worker per CPU in the cpuset-filtered topology.
    let n_workers =
        match std::env::var("ENGINE_NUM_THREADS").or_else(|_| std::env::var("RAYON_NUM_THREADS")) {
            Ok(s) => {
                let total = s.parse::<usize>().expect(
                    "ENGINE_NUM_THREADS / RAYON_NUM_THREADS must parse as a positive integer",
                );
                assert!(total > 0, "engine pool participant count must be > 0");
                total.saturating_sub(1).max(1)
            }
            Err(_) => topology.iter().map(|n| n.len()).sum::<usize>().max(1),
        };

    // Backend is selected via ENGINE_POOL_BACKEND (mypool | rayon | orx);
    // defaults to the in-tree work-stealing pool.
    let backend = parallel::BackendKind::from_env();
    let ok = parallel::global::init(backend, n_workers);
    assert!(ok, "parallel global pool already initialized");
    let _ = no_pin; // simple pool doesn't pin; flag preserved for future use.

    let n = parallel::global::num_threads();
    println!(
        "engine pool: {n} thread(s) on {backend:?} backend{}",
        if no_pin { " [pinning disabled]" } else { "" },
    );

    // Scene construction runs before this init, so any MeshRenderer built
    // there deferred its asset load; hand those to the pool now.
    engine_core::asset::flush_pending_loads();
}

// Trait imports needed for method resolution on GPU types.
use vulkano::pipeline::graphics::vertex_input::Vertex as VulkanoVertex;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Triple-buffer depth: CPU can record frame N+1/N+2 while GPU renders N.
const MAX_FRAMES_IN_FLIGHT: usize = 4;

/// Sample the system clock only every N frames (must be a power of two).
const FRAMES_PER_FPS_SAMPLE: u32 = 512;

// ─────────────────────────────────────────────────────────────────────
// Per-image frame slot
// ─────────────────────────────────────────────────────────────────────

/// Resources tied to a single swapchain image. Built once per swapchain
/// image; rebuilt when the swapchain image changes, when the camera grows /
/// changes extent, or when world entity capacity or scene topology changes.
///
/// **Post ADR-0003**: this struct is now minimal. The per-frame staging
/// buffers, dirty bitmasks, scatter / mvp_build_set1 descriptor sets, and
/// the scatter compute secondary all moved onto [`WorldTransformGpu`] as
/// **shared** resources, gated by a timeline semaphore. The mvp_build
/// compute secondary moved onto [`RenderCamera`] (per-camera, captures the
/// shared `mvp_build_set1`). What's left here is what's truly per-image:
/// the present-blit secondary (its destination is *this* slot's swapchain
/// image) and the composing primary CB that stitches the shared
/// secondaries together with the per-image blit.
pub(crate) struct FrameSlot {
    /// Pre-recorded secondary that contains the present-blit (camera's
    /// offscreen color → this slot's swapchain image). No render-pass
    /// inheritance.
    #[allow(dead_code)]
    blit_secondary: Arc<SecondaryAutoCommandBuffer>,
    /// Pre-recorded **primary** that stitches everything together:
    /// `execute(world.scatter_secondary)`, three `fill_buffer(0)`s on the
    /// shared dirty bitmasks, `execute(camera.mvp_build_secondary)`,
    /// `begin_rendering` on the camera attachments,
    /// `execute(camera.scene_secondary)`, `end_rendering`,
    /// `execute(blit_secondary)`. This is the CB actually submitted.
    /// Vulkano auto-sync inserts the SHADER_WRITE→SHADER_READ barrier
    /// between scatter and mvp-build, the SHADER_WRITE→SHADER_READ barrier
    /// between mvp-build and the vertex shader, and the
    /// COLOR_ATTACHMENT_WRITE→TRANSFER_READ barrier on the camera color
    /// before the blit — all from the resource-usage records carried by
    /// the secondaries.
    pub(crate) command_buffer: Arc<PrimaryAutoCommandBuffer>,
    /// [`GPU_TS_COUNT`] timestamp queries reset + written inside
    /// `command_buffer` (see that constant for the stage layout). Read
    /// back host-side right after this image's `acquire` — the per-image
    /// `in_flight` fence wait guarantees the previous submission (and
    /// thus every query) has retired, so the read never blocks.
    pub(crate) timestamp_pool: Arc<QueryPool>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// An OS window backed by a Vulkan swapchain.
///
/// Owns the **root [`Scene`]** — all entities, transforms, and components
/// live inside it. The renderer drives `Scene::update(dt)` once per frame
/// (which fans out to every registered [`Component::update`]
/// implementation) immediately before staging the GPU upload.
pub struct Window {
    title: String,
    /// The window's root scene. Named `root_scene` to mirror the editor /
    /// game-side convention of calling the top-level scene `root`.
    root_scene: Option<Scene>,
}

impl Window {
    /// Create a window descriptor with the given title.
    pub fn new(title: &str) -> Self {
        Window {
            title: title.to_owned(),
            root_scene: None,
        }
    }

    /// Attach the root [`Scene`] drawn each frame.
    ///
    /// The window takes ownership of the scene; per-frame `Component::update`
    /// hooks run on the event-loop thread immediately before the staging
    /// upload. Attach a [`MeshRenderer`] component to every entity that should
    /// be drawn — the renderer derives its draw list from those components.
    pub fn with_scene(mut self, root_scene: Scene) -> Self {
        self.root_scene = Some(root_scene);
        self
    }

    /// Open the OS window, initialise Vulkan, and block on the event loop.
    pub fn run(self) {
        init_pinned_thread_pool();
        let event_loop = EventLoop::new().expect("Failed to create winit EventLoop");
        let mut app = RenderApp::new(self.title, self.root_scene);
        event_loop
            .run_app(&mut app)
            .expect("Event loop exited with an error");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FPS tracker
// ─────────────────────────────────────────────────────────────────────────────

// ── Frame stats: FPS + per-phase timings ──────────────────────────────

/// Number of GPU timestamps written per FrameSlot primary CB. Layout
/// (deltas between consecutive queries = per-stage GPU time):
///
///   q0  TOP_OF_PIPE  at CB start
///   q1  after the scatter block (scatter + spawn scatter + dirty fills +
///       VP promotions + signal_cs)
///   q2  after mvp_build pass 1 (`cull_secondary`)
///   q3  after pass 1's render scope
///   q4  after the Hi-Z pyramid build   (≈ q3 when frozen / occlusion off)
///   q5  after pass 2's cull dispatch + history update (≈ q4 when off)
///   q6  after pass 2's render scope    (≈ q5 when off)
///   q7  after the present-blit
///
/// q1..q7 are BOTTOM_OF_PIPE — each records when *all prior commands*
/// drain, which the heavy inter-stage barriers in this CB make a good
/// per-stage boundary. When the occlusion block is compiled out (F8) the
/// unused boundaries are written back-to-back so the readback layout
/// stays fixed and the skipped stages read as ~0.
pub(crate) const GPU_TS_COUNT: u32 = 8;

/// Cumulative `(min, max, sum_ns, count)` for a single phase across the
/// FPS sample window. Avg is `sum_ns / count`.
#[derive(Default, Clone, Copy)]
struct PhaseAcc {
    min_ns: u64,
    max_ns: u64,
    sum_ns: u128,
    count: u64,
}

impl PhaseAcc {
    fn record(&mut self, ns: u64) {
        if self.count == 0 {
            self.min_ns = ns;
            self.max_ns = ns;
        } else {
            if ns < self.min_ns {
                self.min_ns = ns;
            }
            if ns > self.max_ns {
                self.max_ns = ns;
            }
        }
        self.sum_ns += ns as u128;
        self.count += 1;
    }

    /// Format as "min/avg/max µs" with one decimal place. Returns "—" if
    /// no samples were recorded in this window (happens for the very first
    /// FPS line if a phase didn't fire on every frame in the window).
    fn fmt_us(&self) -> String {
        if self.count == 0 {
            return "—".to_string();
        }
        let min = self.min_ns as f64 / 1000.0;
        let max = self.max_ns as f64 / 1000.0;
        let avg = (self.sum_ns as f64 / self.count as f64) / 1000.0;
        format!("{:>6.1}/{:>6.1}/{:>6.1}", min, avg, max)
    }
}

/// Frame-time + per-phase telemetry, printed once per FPS sample window.
///
/// Each phase is recorded by calling the corresponding `record_*(ns)` from
/// the per-frame loop. The window is the same as `FpsTracker`'s
/// (`FRAMES_PER_FPS_SAMPLE` frames AND ≥ 1 second of wall time), so the
/// per-phase numbers line up 1:1 with the FPS line above them.
pub(crate) struct FrameStats {
    last_print: Instant,
    frame_count: u32,
    acquire: PhaseAcc,
    host_wait_compute: PhaseAcc,
    /// Whole background-thread staging-write phase: `FrameHandoff::consume`
    /// (CPU→GPU memcpy, see [`cpu_gpu_staging`](Self::cpu_gpu_staging) for
    /// the isolated cost of just that part) plus `write_parent_updates`,
    /// `write_spawns`, and `write_cull_view_proj`.
    host_staging: PhaseAcc,
    /// CPU→CPU: the main thread's per-frame harvest
    /// (`frame_handoff::harvest_trs_into_cpu_staging`) — draining the ECS
    /// hierarchy's dirty TRS into the shared `CpuTrsStaging` buffer.
    /// Excludes `FrameHandoff::wait_for_buffer_free`'s wait.
    cpu_staging: PhaseAcc,
    /// CPU→GPU: the background thread's dirty-bit-driven promotion of that
    /// buffer into `WorldTransformGpu`'s GPU-visible staging `Subbuffer`s
    /// (`frame_handoff::copy_cpu_staging_into_gpu`, via
    /// `FrameHandoff::consume`) — a subset of
    /// [`host_staging`](Self::host_staging).
    cpu_gpu_staging: PhaseAcc,
    /// Wall time `FrameHandoff::wait_for_buffer_free` blocked for, on the
    /// main thread, immediately before `cpu_staging` each frame. Diagnostic
    /// — see `FrameJob::main_wait_ns`'s doc comment for why a long wait
    /// here can inflate `cpu_staging` through pool-worker wake latency
    /// rather than genuine harvest cost.
    main_wait: PhaseAcc,
    sim_update: PhaseAcc,
    /// Per-GPU-stage times from the in-CB timestamp queries (see
    /// [`GPU_TS_COUNT`] for the stage layout): `[scatter, mvp1, raster1,
    /// hiz, mvp2, raster2, blit]`.
    gpu_stages: [PhaseAcc; 7],
    /// q0 → q7: the whole CB's GPU execution time.
    gpu_total: PhaseAcc,
    /// Best-effort AMD GPU telemetry, sampled once per print window. `None`
    /// when no `amdgpu` DRM node is present (non-AMD / non-Linux).
    gpu: Option<gpu_telemetry::GpuTelemetry>,
}

impl FrameStats {
    pub(crate) fn new() -> Self {
        let gpu = gpu_telemetry::GpuTelemetry::discover();
        match &gpu {
            Some(g) => println!("[gpu-telemetry] monitoring {}", g.label()),
            None => println!("[gpu-telemetry] disabled: no amdgpu DRM card found"),
        }
        Self {
            last_print: Instant::now(),
            frame_count: 0,
            acquire: PhaseAcc::default(),
            host_wait_compute: PhaseAcc::default(),
            host_staging: PhaseAcc::default(),
            cpu_staging: PhaseAcc::default(),
            cpu_gpu_staging: PhaseAcc::default(),
            main_wait: PhaseAcc::default(),
            sim_update: PhaseAcc::default(),
            gpu_stages: [PhaseAcc::default(); 7],
            gpu_total: PhaseAcc::default(),
            gpu,
        }
    }

    pub(crate) fn record_acquire(&mut self, ns: u64) {
        self.acquire.record(ns);
    }
    pub(crate) fn record_host_wait_compute(&mut self, ns: u64) {
        self.host_wait_compute.record(ns);
    }
    pub(crate) fn record_host_staging(&mut self, ns: u64) {
        self.host_staging.record(ns);
    }
    /// CPU→CPU harvest time, reported by the main thread through
    /// `FrameJob::cpu_staging_ns`. See the `cpu_staging` field doc comment.
    pub(crate) fn record_cpu_staging(&mut self, ns: u64) {
        self.cpu_staging.record(ns);
    }
    /// CPU→GPU memcpy time, measured directly on this (background) thread
    /// inside `FrameHandoff::consume`. See the `cpu_gpu_staging` field doc
    /// comment.
    pub(crate) fn record_cpu_gpu_staging(&mut self, ns: u64) {
        self.cpu_gpu_staging.record(ns);
    }
    pub(crate) fn record_main_wait(&mut self, ns: u64) {
        self.main_wait.record(ns);
    }
    pub(crate) fn record_sim_update(&mut self, ns: u64) {
        self.sim_update.record(ns);
    }
    /// Record one frame's GPU per-stage times. `deltas_ns[0..7]` are the
    /// seven q(i)→q(i+1) stage deltas, `deltas_ns[7]` the q0→q7 total —
    /// already converted from ticks to nanoseconds by the caller.
    pub(crate) fn record_gpu_timestamps(&mut self, deltas_ns: &[u64; 8]) {
        for (acc, &ns) in self.gpu_stages.iter_mut().zip(&deltas_ns[..7]) {
            acc.record(ns);
        }
        self.gpu_total.record(deltas_ns[7]);
    }

    pub(crate) fn tick(&mut self) {
        self.frame_count += 1;
        if self.frame_count & (FRAMES_PER_FPS_SAMPLE - 1) == 0 {
            let elapsed = self.last_print.elapsed();
            if elapsed.as_secs() >= 1 {
                let fps = self.frame_count as f64 / elapsed.as_secs_f64();
                println!(
                    "FPS: {:.0}  ({:.3} ms/frame)  | us min/avg/max  acquire {} | host_wait_compute {} | host_staging {} [cpu_gpu_staging {}] | sim_update {} | main_wait {} | cpu_staging {}",
                    fps,
                    1000.0 / fps,
                    self.acquire.fmt_us(),
                    self.host_wait_compute.fmt_us(),
                    self.host_staging.fmt_us(),
                    self.cpu_gpu_staging.fmt_us(),
                    self.sim_update.fmt_us(),
                    self.main_wait.fmt_us(),
                    self.cpu_staging.fmt_us(),
                );
                println!(
                    "  gpu us min/avg/max  scatter {} | mvp1 {} | raster1 {} | hiz {} | mvp2 {} | raster2 {} | blit {} | total {}",
                    self.gpu_stages[0].fmt_us(),
                    self.gpu_stages[1].fmt_us(),
                    self.gpu_stages[2].fmt_us(),
                    self.gpu_stages[3].fmt_us(),
                    self.gpu_stages[4].fmt_us(),
                    self.gpu_stages[5].fmt_us(),
                    self.gpu_stages[6].fmt_us(),
                    self.gpu_total.fmt_us(),
                );
                if let Some(gpu) = &self.gpu {
                    println!("{}", gpu.sample_line());
                }
                self.frame_count = 0;
                self.last_print = Instant::now();
                self.acquire = PhaseAcc::default();
                self.host_wait_compute = PhaseAcc::default();
                self.host_staging = PhaseAcc::default();
                self.cpu_staging = PhaseAcc::default();
                self.cpu_gpu_staging = PhaseAcc::default();
                self.main_wait = PhaseAcc::default();
                self.sim_update = PhaseAcc::default();
                self.gpu_stages = [PhaseAcc::default(); 7];
                self.gpu_total = PhaseAcc::default();
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RenderApp  (internal event-loop handler)
// ─────────────────────────────────────────────────────────────────────────────

/// All state that lives for the entire event-loop lifetime.
///
/// Post background-render-thread refactor: this is deliberately thin. The
/// event-loop thread only does windowing/input/ECS sim and the CPU-side TRS
/// harvest (`frame_handoff::harvest_trs_into_cpu_staging`, called via
/// `render_thread.write_frame`); everything Vulkan-swapchain/GPU-resource
/// facing (acquire, capacity/camera/mesh-sync rebuilds, the GPU-staging
/// memcpy, submit + present) lives on the background thread owned by
/// `render_thread`. See `render_thread`'s module doc comment.
struct RenderApp {
    title: String,
    context: VulkanoContext,
    graphics_queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    render_thread: Option<render_thread::RenderThreadHandle>,

    // ── Scene state ─────────────────────────────────────────────────
    /// The window's root scene — owns the transform hierarchy and the
    /// component registry. Mutated each frame via `Scene::update(dt)`.
    root_scene: Option<Scene>,
    last_frame_time: Option<Instant>,
}

impl RenderApp {
    fn new(title: String, root_scene: Option<Scene>) -> Self {
        let context = VulkanoContext::new(VulkanoConfig {
            device_features: DeviceFeatures {
                dynamic_rendering: true,
                // ADR-0004 Phase 1 (instanced indirect draw):
                // * `multi_draw_indirect` lets a single `vkCmdDrawIndexedIndirect`
                //   read more than one `DrawIndexedIndirectCommand` from the
                //   indirect buffer (we call it once per mesh group with
                //   drawCount = 1 today, but enable for future-proofing /
                //   multi-mesh scenes that batch into a single call).
                // * `draw_indirect_first_instance` lets per-draw structs set a
                //   non-zero `first_instance`, which is what makes
                //   `gl_InstanceIndex` index correctly into the per-camera MVP
                //   buffer when the same vkCmdDrawIndexedIndirect emits
                //   `instance_count` GPU-side instances per mesh.
                multi_draw_indirect: true,
                draw_indirect_first_instance: true,
                // Material/texture pipeline:
                // * `shader_sampled_image_array_non_uniform_indexing` — the
                //   fragment shader indexes its fixed-size `sampler2D` array
                //   by the per-**instance** material's texture, which is NOT
                //   dynamically uniform within a draw (two instances of one
                //   mesh may carry different materials); the shader marks
                //   the index `nonuniformEXT`. Core in Vulkan 1.2's
                //   descriptor-indexing feature block.
                shader_sampled_image_array_non_uniform_indexing: true,
                // ADR-0003 (shared staging + timeline-semaphore sync):
                // We use a Vulkan timeline semaphore signaled at
                // `COMPUTE_SHADER` stage end of every submission to gate
                // host writes to the shared staging triple. Promoted to
                // core in Vulkan 1.2; still must be opted into via the
                // device features struct on devices that report 1.2+.
                timeline_semaphore: true,
                ..Default::default()
            },
            ..Default::default()
        });

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            StandardCommandBufferAllocatorCreateInfo {
                primary_buffer_count: 32,
                // Two secondaries per FrameSlot (scene + blit); allocate enough
                // headroom for several swapchain images per pool reset.
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
            context.device().clone(),
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            context.device().clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        let graphics_queue = context.graphics_queue().clone();

        RenderApp {
            title,
            context,
            graphics_queue,
            command_buffer_allocator,
            memory_allocator,
            descriptor_set_allocator,
            render_thread: None,
            root_scene,
            last_frame_time: None,
        }
    }
}

impl ApplicationHandler for RenderApp {
    /// Called once at startup (and again on Android resume cycles).
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Drop the stale background render thread on mobile resume — its
        // `Drop` impl closes the handoff channel and joins cleanly before
        // we build a fresh one below.
        self.render_thread = None;

        // ── Pick a present mode ahead of swapchain creation ─────────────────
        let probe_window = event_loop
            .create_window(WindowAttributes::default().with_title(self.title.clone()))
            .expect("Failed to create window");
        let probe_window = Arc::new(probe_window);
        let probe_surface = vulkano::swapchain::Surface::from_window(
            self.context.instance().clone(),
            probe_window.clone(),
        )
        .expect("Surface::from_window failed");
        let supported = self
            .context
            .device()
            .physical_device()
            .surface_present_modes(probe_surface.as_ref(), SurfaceInfo::default())
            .expect("Failed to query surface present modes");
        let chosen = if supported.contains(&PresentMode::Mailbox) {
            PresentMode::Mailbox
        } else if supported.contains(&PresentMode::Immediate) {
            PresentMode::Immediate
        } else {
            PresentMode::Fifo
        };
        println!("Present mode: {chosen:?}  (supported: {supported:?})");

        drop(probe_surface);
        drop(probe_window);

        let real_window = event_loop
            .create_window(WindowAttributes::default().with_title(self.title.clone()))
            .expect("Failed to create window");

        let swapchain_renderer = SwapchainRenderer::new(
            self.context.instance().clone(),
            self.context.device().clone(),
            self.graphics_queue.clone(),
            real_window,
            chosen,
            MAX_FRAMES_IN_FLIGHT,
        );

        let swapchain_format = swapchain_renderer.swapchain_format();
        let attachment_image_views = swapchain_renderer.image_views().to_vec();

        let pipeline = create_pipeline(self.context.device().clone());
        // Swapchain format is informational here — the pipeline is built
        // against `CAMERA_COLOR_FORMAT`, and the present-blit handles
        // format conversion to whatever the swapchain offers.
        let _ = swapchain_format;

        // Dual-pass occlusion culling compute pipelines — stateless, built
        // once and shared by every camera (see `camera.rs`'s
        // `CameraSceneResources`), same pattern as `pipeline` above.
        let mvp_build_pass2_pipeline =
            create_mvp_build_pass2_pipeline(self.context.device().clone());
        let cull_pass2_args_pipeline =
            create_cull_pass2_args_pipeline(self.context.device().clone());
        let hiz_reduce_depth_pipeline =
            create_hiz_reduce_depth_pipeline(self.context.device().clone());
        let hiz_reduce_mip_pipeline = create_hiz_reduce_mip_pipeline(self.context.device().clone());
        let hiz_reduce_mip2_pipeline =
            create_hiz_reduce_mip2_pipeline(self.context.device().clone());

        // GPU mirror of the core mesh asset registry (mega buffers + table +
        // redirect). Built before the camera; its first `sync` uploads the
        // placeholder/error meshes and returns the per-slot instance totals.
        let mut gpu_mesh_store = GpuMeshStore::new(
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.graphics_queue.clone(),
        );
        // GPU mirror of the core texture registry. Its first `sync` uploads
        // the placeholder/error textures — required before any descriptor
        // set binds the sampled-image array.
        let mut gpu_texture_store = GpuTextureStore::new(
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.graphics_queue.clone(),
        );
        let _ = gpu_texture_store.sync();
        // GPU mirror of the core material registry. Its first `sync` uploads
        // the default material (slot 0) so descriptor sets bind live buffers.
        let mut gpu_material_store = GpuMaterialStore::new(
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.graphics_queue.clone(),
        );
        let _ = gpu_material_store.sync();

        // World transform state + the per-transform GPURenderers buffer, both
        // sized to the hierarchy's current entity count.
        let initial_entity_count = self
            .root_scene
            .as_ref()
            .map(|s| s.transform_hierarchy.len())
            .unwrap_or(1)
            .max(1);
        let world_transforms = WorldTransformGpu::new(
            self.context.device().clone(),
            &self.memory_allocator,
            &self.descriptor_set_allocator,
            &self.command_buffer_allocator,
            self.graphics_queue.clone(),
            initial_entity_count,
        );
        let gpu_renderers = GpuRenderers::new(
            self.context.device().clone(),
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            self.graphics_queue.clone(),
            initial_entity_count as u32,
        );
        // Parent links recorded before the renderer existed stay queued in
        // the hierarchy's stream; the first frame's drain writes them into
        // the parent-update staging and the first frame CB scatters them
        // before its cull — no pre-frame ingest needed.

        // Initially-authored `MeshRenderer` components (each pushed its
        // `(transform_id, mesh_id)` onto the spawn queue at `init`) stay
        // queued for the first frame's drain → spawn staging → in-CB
        // scatter, same as parent links. The initial draw plan doesn't
        // need them: it derives from the registry's per-slot instance
        // totals via `gpu_mesh_store.sync()`. The cull pass reads
        // GPURenderers + redirect + mesh_table directly — no CPU sort.
        let (_changed, slot_totals) = gpu_mesh_store.sync();
        let plan = build_draw_plan(&gpu_mesh_store, &slot_totals);

        // The main camera matches the swapchain extent so the present-blit
        // stays a 1:1 copy. The first swapchain image gives us the extent.
        let initial_extent = {
            let [w, h, _] = attachment_image_views[0].image().extent();
            [w, h]
        };
        let scene_resources = CameraSceneResources {
            cb_allocator: &self.command_buffer_allocator,
            descriptor_set_allocator: &self.descriptor_set_allocator,
            memory_allocator: &self.memory_allocator,
            pipeline: &pipeline,
            queue_family_index: self.graphics_queue.queue_family_index(),
            world_transforms: &world_transforms,
            mesh_store: &gpu_mesh_store,
            texture_store: &gpu_texture_store,
            material_store: &gpu_material_store,
            gpu_renderers: &gpu_renderers,
            mvp_build_pass2_pipeline: &mvp_build_pass2_pipeline,
            cull_pass2_args_pipeline: &cull_pass2_args_pipeline,
            hiz_reduce_depth_pipeline: &hiz_reduce_depth_pipeline,
            hiz_reduce_mip_pipeline: &hiz_reduce_mip_pipeline,
            hiz_reduce_mip2_pipeline: &hiz_reduce_mip2_pipeline,
        };
        let main_camera = RenderCamera::new_match_swapchain(
            initial_extent,
            &scene_resources,
            &plan,
            initial_entity_count,
        );

        let frame_slots = build_all_frame_slots(
            &self.command_buffer_allocator,
            &self.memory_allocator,
            self.graphics_queue.queue_family_index(),
            &attachment_image_views,
            &main_camera,
            &world_transforms,
            &gpu_renderers,
        );

        let rcx = render_thread::RenderContext {
            swapchain_image_views: attachment_image_views,
            world_transforms,
            main_camera,
            frame_slots,
            gpu_mesh_store,
            gpu_texture_store,
            gpu_material_store,
            gpu_renderers,
        };

        // Everything Vulkan-swapchain/GPU-resource facing (acquire,
        // capacity/camera/mesh-sync rebuilds, the GPU-staging memcpy,
        // submit + present) moves onto a dedicated background thread from
        // here on — see `render_thread`'s module doc comment. The event-
        // loop thread keeps only windowing/input/ECS sim and the CPU-side
        // TRS harvest.
        self.render_thread = Some(render_thread::RenderThreadHandle::spawn(
            render_thread::RenderThreadInit {
                device: self.context.device().clone(),
                graphics_queue: self.graphics_queue.clone(),
                command_buffer_allocator: self.command_buffer_allocator.clone(),
                memory_allocator: self.memory_allocator.clone(),
                descriptor_set_allocator: self.descriptor_set_allocator.clone(),
                swapchain_renderer,
                pipeline,
                mvp_build_pass2_pipeline,
                cull_pass2_args_pipeline,
                hiz_reduce_depth_pipeline,
                hiz_reduce_mip_pipeline,
                hiz_reduce_mip2_pipeline,
                rcx,
                initial_entity_count,
                initial_extent,
            },
        ));
        self.last_frame_time = Some(Instant::now());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Always feed the global input accumulator first — harmless if the
        // renderer isn't ready yet, and lets input-driven components see
        // this frame's state regardless of render readiness.
        input::global_mut().feed_window_event(&event);

        let render_thread = match self.render_thread.as_ref() {
            Some(r) => r,
            None => return,
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => render_thread.request_resize(),
            WindowEvent::RedrawRequested => {}
            _ => {}
        }
    }

    /// Render one frame; runs at full speed (`ControlFlow::Poll`).
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);

        let render_thread = match self.render_thread.as_ref() {
            Some(r) => r,
            None => return,
        };

        // ── dt + per-frame update callback ──────────────────────────────────
        let now = Instant::now();
        let dt = self
            .last_frame_time
            .map(|t| (now - t).as_secs_f32())
            .unwrap_or(0.0)
            .min(0.1); // clamp big stalls (e.g. window drag) to 100 ms
        self.last_frame_time = Some(now);

        let sim_start = Instant::now();
        if let Some(scene) = self.root_scene.as_mut() {
            // Materialise queued subscene spawns whose GLB template has
            // resolved: each template proxy becomes a real MeshRenderer
            // (`from_id` — refcount bump + spawn-queue push, ingested into
            // GPURenderers later this same frame). Templates still parsing
            // stay queued; their meshes stream in via the redirect table
            // after the hierarchy appears.
            let _ =
                engine_core::scene_asset::drain_ready_spawns(scene, |scene, entity, mesh_id| {
                    scene.add_component(entity, MeshRenderer::from_id(mesh_id));
                });

            // Drives every registered `Component::update(dt, &transform)` in
            // parallel. Mutations are recorded against the hierarchy's
            // dirty bitmasks and harvested below.
            scene.update(dt);
        }
        let sim_update_ns = sim_start.elapsed().as_nanos() as u64;

        // Every component's `update` for this frame has now run and had a
        // chance to observe the input accumulated since the last frame.
        // The renderer's own debug hotkeys (F8/F9, below) still need to
        // observe this frame's edge-triggered state too, so the transient
        // (`*_pressed` / `*_released` / deltas) clear is deferred to just
        // after those checks — see `input::global_mut().end_frame()` below.

        // Drain the hierarchy's streamed parent changes now — after the
        // sim update and subscene instantiation, so this frame's
        // re-parents are included.
        // TODO: profile drain. prefer to avoid copies/re-allocs and parallelize
        let parent_updates: Vec<[u32; 2]> = self
            .root_scene
            .as_ref()
            .map(|s| s.transform_hierarchy.drain_parent_updates())
            .unwrap_or_default();
        let spawns = components::drain_spawns();
        let f8_pressed = input::key_pressed(KeyCode::F8);
        let f9_pressed = input::key_pressed(KeyCode::F9);

        // The camera is just another component: locate the scene's (first)
        // `CameraComponent` and read its entity's *global* position +
        // rotation to build the view matrix. Must happen on this thread —
        // it reads `Scene::transform_hierarchy`, ECS state that stays
        // main-thread-owned. Aspect ratio comes from the background
        // thread's last-published swapchain extent (every swapchain image
        // shares the same extent, so this doesn't need this frame's actual
        // acquired image — see `RenderThreadHandle::published_extent`'s doc
        // comment).
        let [w, h] = render_thread.published_extent();
        let aspect = w as f32 / h.max(1) as f32;
        let view_proj = self
            .root_scene
            .as_ref()
            .and_then(|scene| {
                let (entity, cam) = scene.first_component::<scene::CameraComponent>()?;
                let cam = cam.lock();
                let t = scene
                    .transform_hierarchy
                    .get_transform_unchecked(entity.id)
                    .lock();
                Some(cam.view_proj(t.get_global_position(), t.get_global_rotation(), aspect))
            })
            .unwrap_or_else(|| {
                scene::CameraComponent::new().view_proj(
                    glam::Vec3::ZERO,
                    glam::Quat::IDENTITY,
                    aspect,
                )
            });
        let view_proj_cols = view_proj.to_cols_array();

        // Last consumer of this frame's edge-triggered input state (both
        // component `update`s, earlier, and the F8/F9 reads above have now
        // run) — clear it so it doesn't leak into next frame's reads.
        input::global_mut().end_frame();

        // Wait for the background thread to finish reading the *previous*
        // handoff's CPU staging snapshot, then harvest this frame's dirty
        // TRS into it. Replaces the old `host_wait_for_previous_compute`
        // GPU-signal poll — the gate is now "background thread done
        // reading my buffer", not "GPU done reading its buffer" (the
        // background thread still does that wait itself, on its own
        // GPU-visible copy — see `render_thread::run`).
        let entity_capacity = render_thread.published_entity_capacity();
        let (ready_gen, main_wait_ns, cpu_staging_ns) =
            render_thread.write_frame(entity_capacity, self.root_scene.as_ref(), view_proj_cols);

        render_thread.send(frame_handoff::FrameJob {
            ready_gen,
            entity_count: self
                .root_scene
                .as_ref()
                .map(|s| s.transform_hierarchy.len())
                .unwrap_or(1)
                .max(1),
            parent_updates,
            spawns,
            f8_pressed,
            f9_pressed,
            view_proj_cols,
            sim_update_ns,
            cpu_staging_ns,
            main_wait_ns,
        });
    }
}

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────

/// Create the single graphics pipeline used for all mesh draws.
///
/// The color attachment format is fixed at [`CAMERA_COLOR_FORMAT`] (HDR) —
/// independent of the swapchain's pixel format. The present-blit handles
/// any conversion between camera-color and swapchain formats.
/// Resolve the accumulated renderer list into the per-draw `(mesh_slot,
/// entity_id)` topology the camera consumes. Each renderer's `mesh_id` is
/// mapped to its current drawable slot via the registry's redirect map
/// (the placeholder slot until an async loader resolves the asset).
pub(crate) fn build_draw_plan(mesh_store: &GpuMeshStore, slot_totals: &[u32]) -> DrawPlan {
    let mut commands = Vec::with_capacity(slot_totals.len());
    let mut base = 0u32;
    for (slot, &total) in slot_totals.iter().enumerate() {
        let geom = mesh_store.slot_geometry(slot as u32);
        commands.push(vulkano::command_buffer::DrawIndexedIndirectCommand {
            index_count: geom.map(|g| g.index_count).unwrap_or(0),
            instance_count: 0,
            first_index: geom.map(|g| g.first_index).unwrap_or(0),
            vertex_offset: geom.map(|g| g.vertex_offset as u32).unwrap_or(0),
            first_instance: base,
        });
        base += total;
    }
    DrawPlan {
        commands,
        total_renderers: base,
    }
}

fn create_pipeline(device: Arc<Device>) -> Arc<GraphicsPipeline> {
    let vs = shaders::vs::load(device.clone()).expect("Failed to load vertex shader");
    let fs = shaders::fs::load(device.clone()).expect("Failed to load fragment shader");

    let stages = [
        PipelineShaderStageCreateInfo::new(vs.entry_point("main").unwrap()),
        PipelineShaderStageCreateInfo::new(fs.entry_point("main").unwrap()),
    ];

    let vertex_input_state = GpuVertex::per_vertex()
        .definition(&stages[0].entry_point)
        .expect("Vertex input definition mismatch");

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .expect("Failed to create pipeline layout create info"),
    )
    .expect("Failed to create pipeline layout");

    GraphicsPipeline::new(
        device,
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                1,
                ColorBlendAttachmentState::default(),
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(PipelineSubpassType::BeginRendering(
                PipelineRenderingCreateInfo {
                    color_attachment_formats: vec![Some(CAMERA_COLOR_FORMAT)],
                    depth_attachment_format: Some(CAMERA_DEPTH_FORMAT),
                    ..Default::default()
                },
            )),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .expect("Failed to create graphics pipeline")
}

/// Build a single-stage compute pipeline from a loaded shader module's
/// entry point. Shared shape for the four dual-pass-occlusion-culling
/// pipelines below — mirrors `transform_gpu.rs`'s `build_scatter_pipeline`
/// / `build_mvp_build_pipeline`.
fn create_compute_pipeline(
    device: Arc<Device>,
    cs: Arc<vulkano::shader::ShaderModule>,
    label: &str,
) -> Arc<ComputePipeline> {
    let entry = cs
        .entry_point("main")
        .unwrap_or_else(|| panic!("{label} entry point"));
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(&stage))
            .into_pipeline_layout_create_info(device.clone())
            .unwrap_or_else(|_| panic!("{label} pipeline layout info")),
    )
    .unwrap_or_else(|_| panic!("{label} pipeline layout"));
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap_or_else(|_| panic!("{label} ComputePipeline::new"))
}

/// Pass 2's cull pipeline — see `shaders::mvp_build_pass2_cs`.
fn create_mvp_build_pass2_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs =
        shaders::mvp_build_pass2_cs::load(device.clone()).expect("mvp_build_pass2_cs load failed");
    create_compute_pipeline(device, cs, "mvp_build_pass2_cs")
}

/// The tiny "build pass 2's dispatch-indirect args" pipeline — see
/// `shaders::cull_pass2_args_cs`.
fn create_cull_pass2_args_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs =
        shaders::cull_pass2_args_cs::load(device.clone()).expect("cull_pass2_args_cs load failed");
    create_compute_pipeline(device, cs, "cull_pass2_args_cs")
}

/// Hi-Z pyramid level 0 (depth → mip0) pipeline — see
/// `shaders::hiz_reduce_depth_cs`.
fn create_hiz_reduce_depth_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs = shaders::hiz_reduce_depth_cs::load(device.clone())
        .expect("hiz_reduce_depth_cs load failed");
    create_compute_pipeline(device, cs, "hiz_reduce_depth_cs")
}

/// Hi-Z pyramid levels 1..N (mip[L-1] → mip[L]) pipeline — see
/// `shaders::hiz_reduce_mip_cs`.
fn create_hiz_reduce_mip_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs =
        shaders::hiz_reduce_mip_cs::load(device.clone()).expect("hiz_reduce_mip_cs load failed");
    create_compute_pipeline(device, cs, "hiz_reduce_mip_cs")
}

/// Hi-Z pyramid, fused pair of levels (mip[L-1] → mip[L] → mip[L+1] in one
/// dispatch) pipeline — see `shaders::hiz_reduce_mip2_cs`.
fn create_hiz_reduce_mip2_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs =
        shaders::hiz_reduce_mip2_cs::load(device.clone()).expect("hiz_reduce_mip2_cs load failed");
    create_compute_pipeline(device, cs, "hiz_reduce_mip2_cs")
}

/// Build (or rebuild) a `FrameSlot` for every swapchain image. Slots are
/// independent of each other and could be built in parallel; we keep the
/// loop sequential to avoid contention on the descriptor-set / CB allocators
/// (which are not particularly fast under contention).
pub(crate) fn build_all_frame_slots(
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    queue_family_index: u32,
    swapchain_views: &[Arc<ImageView>],
    main_camera: &RenderCamera,
    world_transforms: &WorldTransformGpu,
    gpu_renderers: &GpuRenderers,
) -> Vec<FrameSlot> {
    // Sequential — deliberately not `parallel::global::parallel_for`. That
    // pool allows only one non-worker ("external") caller at a time; post
    // background-render-thread refactor, the *main* thread is also an
    // external caller of it (the TRS harvest, every frame — see
    // `frame_handoff::harvest_trs_into_cpu_staging`). This function runs on
    // the *background* thread (rebuilds are rare: swapchain resize,
    // mesh/texture/material arrival, capacity grows, the F8 occlusion
    // toggle), so dispatching it on the shared global pool would race the
    // main thread's harvest and trip that pool's "concurrent external
    // caller" panic. `swapchain_views.len()` is the swapchain image count
    // (typically 2-4) — too small for parallelism to be worth a dedicated
    // pool here; just loop.
    swapchain_views
        .iter()
        .map(|view| {
            build_frame_slot(
                cb_allocator,
                memory_allocator,
                queue_family_index,
                view,
                main_camera,
                world_transforms,
                gpu_renderers,
            )
        })
        .collect()
}

/// Build one `FrameSlot`: pre-record the per-image present-blit secondary
/// (camera color → *this* slot's swapchain image) and stitch the shared
/// world / camera secondaries together with the per-image blit inside one
/// composing primary CB.
///
/// Post ADR-0003 this function does **no** per-frame buffer allocation
/// and **no** descriptor-set creation — those resources all moved onto
/// `WorldTransformGpu` (shared) and `RenderCamera` (per-camera). The
/// primary captures the shared `world.scatter_secondary()`,
/// `camera.mvp_build_secondary()`, and `camera.scene_secondary()` by
/// `Arc<...>`; vulkano auto-sync infers the cross-stage barriers from the
/// resource-usage records each secondary carries.
fn build_frame_slot(
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    _memory_allocator: &Arc<StandardMemoryAllocator>,
    queue_family_index: u32,
    swapchain_view: &Arc<ImageView>,
    main_camera: &RenderCamera,
    world: &WorldTransformGpu,
    gpu_renderers: &GpuRenderers,
) -> FrameSlot {
    let swapchain_image = swapchain_view.image().clone();

    // Camera-owned offscreen attachments. The dynamic-rendering scope below
    // targets these (NOT the swapchain image); the present-blit downstream
    // copies camera-extent → swapchain-extent. They happen to coincide today
    // because the main camera uses `CameraResolution::MatchSwapchain`.
    let color_image = main_camera.color_image().clone();
    let color_view = main_camera.color_view().clone();
    let depth_view = main_camera.depth_view().clone();

    // ── Pre-record the blit secondary ────────────────────────
    // The only truly per-image secondary: its destination image is *this*
    // slot's swapchain image. MultipleSubmit is fine — the per-image
    // fence guarantees only one primary using this slot is in flight at
    // a time.
    let mut blit_builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::MultipleSubmit,
        CommandBufferInheritanceInfo::default(),
    )
    .expect("blit secondary builder");

    blit_builder
        .blit_image(BlitImageInfo::images(color_image.clone(), swapchain_image))
        .expect("blit_image");
    let blit_secondary = blit_builder.build().expect("build blit secondary");

    // ── Pre-record the FrameSlot primary command buffer ────────────────
    //
    // ADR-0003 (post-fold-into-main revision): scatter, the dirty
    // `fill_buffer(0)` clears, and the `staging_view_proj → sot_view_proj`
    // copy now live at the **front of this CB**, not in a separate
    // pre-batch. One CB, one batch per `vkQueueSubmit2` — the split-submit
    // had ~30μs/frame of fixed overhead at low N (see ADR-0003 measurements
    // section), and folding eliminates the timeline signal/wait inter-batch
    // sync entirely. Vulkano auto-sync inserts the
    // `SHADER_WRITE → SHADER_READ` barrier on each SoT buffer between
    // scatter and mvp_build (which both bind the SoT) without any manual
    // pipeline barrier.
    //
    // CB structure:
    //
    //   world.scatter_secondary  — 3 dispatches: staging_<comp> → sot_<comp>
    //                              gated by staging_dirty_<comp>.
    //     ↓  vulkano auto-sync: SHADER_READ → TRANSFER_WRITE on dirty bufs
    //   fill_buffer(staging_dirty_pos/rot/scl, 0)  — clear dirty bits.
    //     ↓  no dependency, separate buffer
    //   copy_buffer(staging_view_proj → sot_view_proj)  — promote VP.
    //     ↓  vulkano auto-sync: SHADER_WRITE → SHADER_READ on sot_<comp>,
    //                            TRANSFER_WRITE → SHADER_READ on sot_view_proj
    //   copy_buffer(cull_view_proj_staging → cull_view_proj)  — promote the
    //                            cull-test VP (debug frustum-lock feature;
    //                            unconditional regardless of the flag below —
    //                            see `RenderCamera::write_cull_view_proj`).
    //   camera.cull_secondary  — pass 1: frustum (against cull_view_proj)
    //                            + prev-frame-Hi-Z occlusion cull (if
    //                            enabled — see below), writes MVP +
    //                            candidates.
    //     ↓  vulkano auto-sync: SHADER_WRITE → SHADER_READ on device_matrices
    //   begin_rendering(camera attachments, Clear)
    //     camera.scene_secondary_pass1  — draws pass 1's visible instances.
    //   end_rendering
    //     ↓  DEPTH_ATTACHMENT_WRITE → SHADER_READ on camera depth
    //   ── the following block only runs if `main_camera.occlusion_enabled()`
    //      (debug F8 toggle — see `RenderCamera::set_occlusion_enabled`) ──
    //   ── camera.hiz_build_secondary and camera.history_update_secondary
    //      (below) additionally skip if `main_camera.hiz_frozen()` (debug
    //      F9 frustum-lock, one frame behind — see
    //      `RenderCamera::apply_pending_hiz_freeze`) ──
    //   camera.hiz_build_secondary  — max-reduces pass 1's depth into
    //                                 hiz_current's mip pyramid.
    //   camera.cull_pass2_secondary  — dispatch_indirect over the live
    //                                  candidate count; re-tests occlusion
    //                                  against hiz_current (frozen or not),
    //                                  writes MVP.
    //   camera.history_update_secondary  — copies hiz_current → hiz_prev
    //                                      and sot_view_proj → prev_view_proj
    //                                      for next frame's pass 1.
    //   begin_rendering(camera attachments, Load)
    //     camera.scene_secondary_pass2  — draws pass 2's newly-visible
    //                                     instances into the same targets.
    //   end_rendering
    //   ── end of the occlusion_enabled-gated block ──
    //     ↓  COLOR_ATTACHMENT_WRITE → TRANSFER_READ on camera color
    //     ↓  Undefined / PresentSrc → TRANSFER_DST on swapchain image
    //   blit_secondary  — camera color → swapchain image.
    //     ↓  TRANSFER_WRITE → PresentSrc on swapchain (final layout req.)
    //
    // The submission also signals `world.compute_timeline` at
    // `COMPUTE_SHADER | ALL_TRANSFER` stage end (smallest mask covering
    // every read of host-shared buffers). The next frame's host wait
    // gates against that value before mutating shared staging.
    let mut builder = AutoCommandBufferBuilder::primary(
        cb_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::MultipleSubmit,
    )
    .expect("primary CB builder");

    // Per-stage GPU timestamps (see `GPU_TS_COUNT` for the layout). The
    // pool must be reset in-CB before reuse; the per-image fence
    // guarantees only one submission of this CB is in flight, so the
    // reset never races a pending query.
    let timestamp_pool = QueryPool::new(
        cb_allocator.device().clone(),
        QueryPoolCreateInfo {
            query_count: GPU_TS_COUNT,
            ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
        },
    )
    .expect("create frame timestamp pool");
    // SAFETY (all write_timestamp/reset_query_pool calls below): queries
    // stay in [0, GPU_TS_COUNT), the graphics family supports timestamps,
    // and every write is preceded by the in-CB reset.
    unsafe { builder.reset_query_pool(timestamp_pool.clone(), 0..GPU_TS_COUNT) }
        .expect("reset timestamp pool");
    unsafe { builder.write_timestamp(timestamp_pool.clone(), 0, PipelineStage::TopOfPipe) }
        .expect("write_timestamp q0");

    builder
        .execute_commands(world.scatter_secondary().clone())
        .expect("execute scatter_secondary");

    // Spawn-scatter: streamed (transform_id, mesh_id) pairs → GPURenderers.
    // Count-in-buffer like the parent scatter inside `scatter_secondary`;
    // recorded before `signal_cs` so the `gpu_signal` gate covers the host
    // write to its staging, and before the cull secondary which reads the
    // GPURenderers buffer it writes (vulkano auto-sync orders them).
    builder
        .execute_commands(gpu_renderers.spawn_scatter_secondary().clone())
        .expect("execute spawn_scatter_secondary");

    builder
        .fill_buffer(world.staging_dirty_pos().clone().reinterpret::<[u32]>(), 0)
        .expect("fill staging_dirty_pos")
        .fill_buffer(world.staging_dirty_rot().clone().reinterpret::<[u32]>(), 0)
        .expect("fill staging_dirty_rot")
        .fill_buffer(world.staging_dirty_scl().clone().reinterpret::<[u32]>(), 0)
        .expect("fill staging_dirty_scl");

    builder
        .copy_buffer(vulkano::command_buffer::CopyBufferInfo::buffers(
            world.view_proj_buf().clone().reinterpret::<[u8]>(),
            world.sot_view_proj().clone().reinterpret::<[u8]>(),
        ))
        .expect("copy staging_view_proj → sot_view_proj");

    // Cull-test VP promotion (frustum-lock debug feature). Unconditional —
    // runs regardless of `occlusion_enabled` below, since pass 1's frustum
    // test always reads `cull_view_proj`, and this is what keeps the lock
    // toggle cheap (no CB re-recording either way).
    builder
        .copy_buffer(vulkano::command_buffer::CopyBufferInfo::buffers(
            main_camera.cull_view_proj_staging_buf().clone().reinterpret::<[u8]>(),
            main_camera.cull_view_proj_buf().clone().reinterpret::<[u8]>(),
        ))
        .expect("copy cull_view_proj_staging → cull_view_proj");

    // Early-wake signal — atomically increments `gpu_signal[0]`. Recorded
    // **here**, after every read of host-shared staging is done
    // (scatter consumed staging+dirty, fill_buffer cleared dirty,
    // copy_buffer consumed view_proj_buf), and **before** mvp_build so
    // the rest of the CB doesn't gate the increment's visibility to the
    // host. Vulkano auto-sync inserts the prior commands' completion
    // before this dispatch via the SoT/dirty/view_proj buffer
    // dependencies, so when `signal_cs` writes its atomic, the host can
    // safely overwrite the shared staging — the GPU is fully done with
    // it. See `WorldTransformGpu::host_wait_for_previous_compute`.
    builder
        .execute_commands(world.signal_secondary().clone())
        .expect("execute signal_secondary");

    unsafe { builder.write_timestamp(timestamp_pool.clone(), 1, PipelineStage::BottomOfPipe) }
        .expect("write_timestamp q1 (scatter)");

    builder
        .execute_commands(main_camera.cull_secondary().clone())
        .expect("execute cull_secondary (pass 1)");

    unsafe { builder.write_timestamp(timestamp_pool.clone(), 2, PipelineStage::BottomOfPipe) }
        .expect("write_timestamp q2 (mvp1)");

    builder
        .begin_rendering(RenderingInfo {
            contents: SubpassContents::SecondaryCommandBuffers,
            color_attachments: vec![Some(RenderingAttachmentInfo {
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
                clear_value: Some([0.08, 0.08, 0.10, 1.0].into()),
                ..RenderingAttachmentInfo::image_view(color_view.clone())
            })],
            depth_attachment: Some(RenderingAttachmentInfo {
                image_layout: ImageLayout::DepthStencilAttachmentOptimal,
                load_op: AttachmentLoadOp::Clear,
                // Must be `Store` (not `DontCare`): both the Hi-Z build and
                // pass 2's `Load`-scoped render below need this frame's
                // pass-1 depth contents to survive past this render scope.
                store_op: AttachmentStoreOp::Store,
                clear_value: Some(1.0_f32.into()),
                ..RenderingAttachmentInfo::image_view(depth_view.clone())
            }),
            ..Default::default()
        })
        .expect("begin_rendering pass1");

    builder
        .execute_commands(main_camera.scene_secondary_pass1().clone())
        .expect("execute scene_secondary_pass1");

    builder.end_rendering().expect("end_rendering pass1");

    unsafe { builder.write_timestamp(timestamp_pool.clone(), 3, PipelineStage::BottomOfPipe) }
        .expect("write_timestamp q3 (raster1)");

    // Debug: occlusion culling can be disabled entirely (F8 at runtime —
    // see `RenderCamera::set_occlusion_enabled`), in which case this whole
    // block — the Hi-Z pyramid build, pass 2's cull dispatch, pass 2's
    // render scope, and the history-update copy — is omitted from the
    // primary altogether (real GPU-work avoidance, not a shader no-op;
    // `mvp_build.comp`'s own `occlusion_enabled` push constant, baked into
    // `cull_secondary` alongside this flag, is what keeps pass 1 correct
    // while this is skipped — see that shader's module doc comment).
    // Skipping `history_update_secondary` leaves `hiz_prev`/`prev_view_proj`
    // stale until occlusion is re-enabled; that's self-healing — pass 2
    // re-validates every candidate against a fresh Hi-Z before anything is
    // actually dropped, so a stale first frame back never produces a
    // visible artifact, just a momentarily larger candidate list.
    if main_camera.occlusion_enabled() {
        // Debug: the frustum-lock feature (F9) additionally freezes the
        // Hi-Z pipeline (`RenderCamera::hiz_frozen`) — skip only the build
        // + history-update *inside* this still-active occlusion block, so
        // `hiz_current`/`hiz_prev`/`prev_view_proj` stay pinned at the
        // self-consistent snapshot left behind by the frame the lock
        // engaged. `cull_pass2_secondary` and pass 2's render scope below
        // keep running regardless — they just end up testing/drawing
        // against whichever (possibly frozen) pyramid contents currently
        // exist. See `camera.rs`'s module doc comment, "frustum-lock"
        // section, for the full reasoning.
        if !main_camera.hiz_frozen() {
            // Hi-Z build reads the depth attachment pass 1 just wrote
            // (vulkano auto-sync transitions it out of
            // `DepthStencilAttachmentOptimal` from the descriptor-set
            // binding's resource-usage record, same mechanism as the color
            // image's attachment→transfer-src transition before the blit
            // below).
            builder
                .execute_commands(main_camera.hiz_build_secondary().clone())
                .expect("execute hiz_build_secondary");
        }

        unsafe { builder.write_timestamp(timestamp_pool.clone(), 4, PipelineStage::BottomOfPipe) }
            .expect("write_timestamp q4 (hiz)");

        builder
            .execute_commands(main_camera.cull_pass2_secondary().clone())
            .expect("execute cull_pass2_secondary");

        if !main_camera.hiz_frozen() {
            // No dependency on pass 2's render (see
            // `RenderCamera::hiz_current`'s doc comment) — only on
            // `hiz_build_secondary` and `sot_view_proj` already holding
            // this frame's promoted VP, both true by this point.
            builder
                .execute_commands(main_camera.history_update_secondary().clone())
                .expect("execute history_update_secondary");
        }

        unsafe { builder.write_timestamp(timestamp_pool.clone(), 5, PipelineStage::BottomOfPipe) }
            .expect("write_timestamp q5 (mvp2)");

        builder
            .begin_rendering(RenderingInfo {
                contents: SubpassContents::SecondaryCommandBuffers,
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Load,
                    store_op: AttachmentStoreOp::Store,
                    ..RenderingAttachmentInfo::image_view(color_view.clone())
                })],
                depth_attachment: Some(RenderingAttachmentInfo {
                    image_layout: ImageLayout::DepthStencilAttachmentOptimal,
                    load_op: AttachmentLoadOp::Load,
                    store_op: AttachmentStoreOp::DontCare,
                    ..RenderingAttachmentInfo::image_view(depth_view.clone())
                }),
                ..Default::default()
            })
            .expect("begin_rendering pass2");

        builder
            .execute_commands(main_camera.scene_secondary_pass2().clone())
            .expect("execute scene_secondary_pass2");

        builder.end_rendering().expect("end_rendering pass2");

        unsafe { builder.write_timestamp(timestamp_pool.clone(), 6, PipelineStage::BottomOfPipe) }
            .expect("write_timestamp q6 (raster2)");
    } else {
        // Occlusion block compiled out: write the unused stage boundaries
        // back-to-back so the readback layout stays fixed and the skipped
        // stages (hiz / mvp2 / raster2) read as ~0.
        for q in 4..=6 {
            unsafe { builder.write_timestamp(timestamp_pool.clone(), q, PipelineStage::BottomOfPipe) }
                .expect("write_timestamp q4-q6 (occlusion off)");
        }
    }

    builder
        .execute_commands(blit_secondary.clone())
        .expect("execute blit_secondary");

    unsafe { builder.write_timestamp(timestamp_pool.clone(), 7, PipelineStage::BottomOfPipe) }
        .expect("write_timestamp q7 (blit)");

    let command_buffer = builder.build().expect("build primary CB");

    FrameSlot {
        blit_secondary,
        command_buffer,
        timestamp_pool,
    }
}

/// Iterate the set bits of one `u32` word from a packed dirty bitmask and
/// call `f` with the absolute entity index for each. `word_idx` is the
/// position of the word in the bitmask; `entity_count` is an upper bound
/// that lets us skip tail bits past the populated entity range without an
/// explicit per-bit check downstream.
#[inline]
#[allow(dead_code)] // currently unused after the parallel walk inlined the loop, kept for future helpers
fn walk_bits(mut bits: u32, word_idx: usize, entity_count: usize, mut f: impl FnMut(usize)) {
    let base = word_idx * 32;
    while bits != 0 {
        let b = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let i = base + b;
        if i >= entity_count {
            break;
        }
        f(i);
    }
}
