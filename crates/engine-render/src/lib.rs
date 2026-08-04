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

use std::{
    sync::{
        atomic::{self},
        Arc,
    },
    time::{Duration, Instant},
};

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
    query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
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

use engine_core::util::{parallel, thread_pool};

pub mod assets;
mod camera;
pub mod components;
mod gpu_mesh;
mod gpu_renderers;
mod gpu_telemetry;
pub mod input;
mod scene;
mod shaders;
pub mod stats;
mod swapchain;
mod transform_gpu;
pub mod ui;

use assets::{GpuMaterialStore, GpuMeshStore, GpuTextureStore};
use camera::{
    CameraSceneResources, DrawPlan, RenderCamera, CAMERA_COLOR_FORMAT, CAMERA_DEPTH_FORMAT,
};
use gpu_mesh::GpuVertex;
use gpu_renderers::GpuRenderers;
use swapchain::SwapchainRenderer;
use transform_gpu::{dirty_word_count, StagingMemory, WorldTransformGpu};
use ui::UiGpu;

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
/// Worker count for the dedicated TRS-staging pool. Override with
/// `ENGINE_STAGING_THREADS`.
const STAGING_POOL_THREADS: usize = 16;

/// Dedicated pool for the TRS staging drain, with its workers confined to
/// the GPU's NUMA node.
///
/// The scatter compute pulls dirty transforms out of host-**cached**
/// staging, so every read snoops whichever socket's caches hold the lines
/// the staging workers just wrote. Writers spread across both sockets of a
/// 2P box make most of that 8 MB a remote fetch — 673µs versus 320µs for
/// the scatter, measured on this machine. Confining the *whole process* to
/// the GPU's node fixes it but costs every other subsystem half the
/// machine, so instead only the staging drain gets node-local workers.
///
/// Pinning trick: worker threads inherit the creating thread's affinity
/// mask, so this narrows the calling thread's mask, builds the pool, and
/// restores the original mask. No pool-implementation changes needed.
///
/// `None` on single-socket machines and wherever the kernel reports no GPU
/// affinity — callers fall back to the global pool.
fn staging_pool() -> Option<&'static parallel::Pool> {
    use engine_core::util::numa::{self, NumaTopology};
    use std::sync::OnceLock;

    static POOL: OnceLock<Option<parallel::Pool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let node = numa::gpu_numa_node()?;
        let topo = NumaTopology::detect().ok()?;
        if topo.num_nodes() <= 1 {
            return None;
        }
        let cpus = topo.cpus_of_node(node)?.to_vec();
        let n_threads = match std::env::var("ENGINE_STAGING_THREADS") {
            Ok(s) => s
                .parse::<usize>()
                .expect("ENGINE_STAGING_THREADS must parse as a positive integer"),
            Err(_) => STAGING_POOL_THREADS,
        };
        assert!(n_threads > 0, "staging pool needs at least one thread");

        // Narrow → spawn → restore. Failing to restore would silently
        // confine the main thread (and everything it later spawns) to the
        // GPU's node, which is exactly what this is trying to avoid, so
        // both affinity calls are fatal on error.
        let saved = numa::current_affinity().expect("sched_getaffinity failed");
        numa::restrict_affinity_to(&cpus)
            .unwrap_or_else(|e| panic!("sched_setaffinity to GPU node {node} failed: {e}"));
        let pool = parallel::Pool::new(parallel::BackendKind::from_env(), n_threads);
        numa::restrict_affinity_to(&saved)
            .unwrap_or_else(|e| panic!("failed to restore main-thread affinity: {e}"));

        println!(
            "[numa] TRS staging pool: {n_threads} thread(s) confined to the GPU's node {node} \
             ({} cpus); main pool unbound",
            cpus.len(),
        );
        Some(pool)
    })
    .as_ref()
}

fn init_pinned_thread_pool() {
    use engine_core::util::numa::{self, NumaTopology};

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

    // ── Confine every thread to the GPU's NUMA node ─────────────────────
    //
    // Done *first*, so the cpuset-filtered topology and default worker
    // count below simply observe the narrowed mask — exactly as they
    // already do under `numactl --cpunodebind`.
    //
    // The scatter compute pulls dirty transforms out of host-**cached**
    // staging, so each read snoops whichever socket's caches hold the
    // lines the staging workers just wrote. Letting those workers spread
    // across both sockets of a 2P box makes most of that 8 MB a remote
    // fetch: measured on this machine (2×128 CPU, GPU on node 0,
    // `--shapes 1000000`), the scatter runs 673µs unbound versus 320µs
    // confined, i.e. 11.9 GB/s versus 25 GB/s, for 637 → 813 FPS. It also
    // removes a large bimodal spread — which fraction of workers happened
    // to land on the far node varied run to run, so the same build would
    // measure anywhere from 320µs to 700µs.
    //
    // Binding the staging *pages* instead (`ENGINE_STAGING_NUMA_NODE`,
    // `mbind`) does nothing here: the cost is cache residency, not page
    // residency.
    //
    // Defaults **on**: this is the best-measured configuration (803 FPS
    // versus 622 unbound on `--shapes 1000000`). Confining the process does
    // cost every other subsystem half the machine, but the alternative —
    // binding only a separate staging pool — measured worse, because two
    // pools evict each other (see `RenderApp::use_staging_pool`). One pool,
    // node-local, wins.
    //
    // No-op on single-socket machines (one node owns every CPU) and
    // wherever the kernel reports no GPU affinity.
    let gpu_node_affinity = std::env::var("ENGINE_GPU_NODE_AFFINITY")
        .ok()
        .map(|v| match v.as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => panic!("ENGINE_GPU_NODE_AFFINITY must be 0/1/true/false, got {v:?}"),
        })
        .unwrap_or(true);
    if gpu_node_affinity {
        if let (Some(node), Ok(topo)) = (numa::gpu_numa_node(), NumaTopology::detect()) {
            if topo.num_nodes() > 1 {
                let cpus = topo
                    .cpus_of_node(node)
                    .unwrap_or_else(|| panic!("GPU reports NUMA node {node}, absent from topology"))
                    .to_vec();
                numa::restrict_affinity_to(&cpus).unwrap_or_else(|e| {
                    panic!(
                        "sched_setaffinity to GPU node {node} ({} cpus) failed: {e}",
                        cpus.len()
                    )
                });
                println!(
                    "[numa] confined to GPU's node {node} ({} cpus) — \
                     keeps the scatter's host-cached reads socket-local",
                    cpus.len(),
                );
            }
        }
    }

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

/// Number of host-staging slots (double buffering). The host writes slot
/// `k` while the GPU still reads slot `k ^ 1`, which is what lets
/// `WorldTransformGpu::host_wait_for_previous_compute` gate on frame
/// `N-2` instead of `N-1`.
pub(crate) const STAGING_SLOTS: usize = 2;

/// Index into `RenderContext::frame_slots`, which holds one pre-recorded
/// primary per `(swapchain image, staging slot)` pair.
#[inline]
fn frame_slot_index(image_index: usize, staging_slot: usize) -> usize {
    image_index * STAGING_SLOTS + staging_slot
}

/// Sample the system clock only every N frames (must be a power of two).
const FRAMES_PER_FPS_SAMPLE: u32 = 128;

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
struct FrameSlot {
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
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
    /// [`GPU_TS_COUNT`] timestamp queries reset + written inside
    /// `command_buffer` (see that constant for the stage layout). Read
    /// back host-side right after this image's `acquire` — the per-image
    /// `in_flight` fence wait guarantees the previous submission (and
    /// thus every query) has retired, so the read never blocks.
    timestamp_pool: Arc<QueryPool>,
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
///   q0  TOP_OF_PIPE     at CB start
///   q8  BOTTOM_OF_PIPE  at CB start  ("seam", see below)
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
///
/// # Why q8 exists — the pipeline seam
///
/// "All prior commands" spans *submissions*, not just this CB. Nothing
/// makes frame N's submission wait on frame N−1's graphics work (only on
/// its own image-acquire semaphore), so the front-end reaches q0 while
/// frame N−1 is still rastering, while q1 — being BOTTOM_OF_PIPE — cannot
/// latch until frame N−1's blit has retired. q1 − q0 therefore charges
/// the *previous* frame's remaining execution to this frame's "scatter",
/// which is why the scatter block appeared to grow by ~330µs the moment
/// raster went from 7µs to 680µs even though its dirty-word count never
/// moved. q8 is a BOTTOM_OF_PIPE write at CB start, so:
///
///   seam    = q8 − q0  → drain of everything submitted before this frame
///   scatter = q1 − q8  → the scatter block's own cost
///
/// A timestamp write is not a barrier, so q8 does not serialise anything;
/// it only observes when the queue went idle.
///
/// # q9..q11 — inside the scatter block
///
/// q1 − q8 lumps together the TRS scatter, the spawn scatter, the dirty-mask
/// clears, the VP promotions and `signal_cs`. The first-used staging slot is
/// ~13× slower than every other slot for identical work, and knowing *which*
/// of those commands absorbs the difference is the whole question, so the
/// block is subdivided:
///
///   q9   after `scatter_secondary`       (prepass + build_args + TRS + parent)
///   q10  after `spawn_scatter_secondary`
///   q11  after the 3 dirty `fill_buffer`s + 2 VP `copy_buffer`s
///   q1   after `signal_secondary`
const GPU_TS_COUNT: u32 = 12;

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

    /// Same min/avg/max shape for accumulators holding something other
    /// than nanoseconds (e.g. word counts), divided by `scale`.
    fn fmt_scaled(&self, scale: f64) -> String {
        if self.count == 0 {
            return "—".to_string();
        }
        let min = self.min_ns as f64 / scale;
        let max = self.max_ns as f64 / scale;
        let avg = (self.sum_ns as f64 / self.count as f64) / scale;
        format!("{:>7.0}/{:>7.0}/{:>7.0}", min, avg, max)
    }
}

/// Frame-time + per-phase telemetry, printed once per FPS sample window.
///
/// Each phase is recorded by calling the corresponding `record_*(ns)` from
/// the per-frame loop. The window is the same as `FpsTracker`'s
/// (`FRAMES_PER_FPS_SAMPLE` frames AND ≥ 1 second of wall time), so the
/// per-phase numbers line up 1:1 with the FPS line above them.
struct FrameStats {
    last_print: Instant,
    frame_count: u32,
    acquire: PhaseAcc,
    host_wait_compute: PhaseAcc,
    host_staging: PhaseAcc,
    staging_locks: PhaseAcc,
    staging_setup: PhaseAcc,
    staging_parallel: PhaseAcc,
    staging_parents: PhaseAcc,
    staging_renderers: PhaseAcc,
    sim_update: PhaseAcc,
    /// Per-GPU-stage times from the in-CB timestamp queries (see
    /// [`GPU_TS_COUNT`] for the stage layout): `[seam, scatter, mvp1,
    /// raster1, hiz, mvp2, raster2, blit]`. `seam` is not this frame's
    /// work — it is the previous submissions draining.
    gpu_stages: [PhaseAcc; 8],
    /// q8 → q7: this frame's own GPU execution time. Excludes the seam, so
    /// `seam + total` is the full q0→q7 span and `total` is the sum of the
    /// seven real stages.
    gpu_total: PhaseAcc,
    /// Scatter-block time (q8 → q1) split by staging slot. Diagnostic for
    /// whether the scatter's wide spread is per-slot or per-frame.
    scatter_by_slot: [PhaseAcc; STAGING_SLOTS],
    /// The scatter block subdivided: `[trs, spawn, clears+VP, signal]` (see
    /// [`GPU_TS_COUNT`]'s q9..q11), split by staging slot so the first-used
    /// slot's penalty can be attributed to one of the four.
    scatter_parts_by_slot: [[PhaseAcc; 4]; STAGING_SLOTS],
    /// Dirty-word span per frame, attributed to the staging slot that
    /// received it. Companion to `scatter_by_slot`.
    prepass_words_by_slot: [PhaseAcc; STAGING_SLOTS],
    /// Best-effort AMD GPU telemetry, sampled once per print window. `None`
    /// when no `amdgpu` DRM node is present (non-AMD / non-Linux).
    gpu: Option<gpu_telemetry::GpuTelemetry>,
}

impl FrameStats {
    fn new() -> Self {
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
            staging_locks: PhaseAcc::default(),
            staging_setup: PhaseAcc::default(),
            staging_parallel: PhaseAcc::default(),
            staging_parents: PhaseAcc::default(),
            staging_renderers: PhaseAcc::default(),
            sim_update: PhaseAcc::default(),
            gpu_stages: [PhaseAcc::default(); 8],
            gpu_total: PhaseAcc::default(),
            scatter_by_slot: [PhaseAcc::default(); STAGING_SLOTS],
            scatter_parts_by_slot: [[PhaseAcc::default(); 4]; STAGING_SLOTS],
            prepass_words_by_slot: [PhaseAcc::default(); STAGING_SLOTS],
            gpu,
        }
    }

    /// Background shader-clock sampler for [`StagingBalancer`], on the same
    /// card this prints telemetry for. `None` when there is no amdgpu card.
    fn spawn_sclk_monitor(&self) -> Option<gpu_telemetry::SclkMonitor> {
        self.gpu.as_ref().map(|g| g.spawn_sclk_monitor())
    }

    fn record_acquire(&mut self, ns: u64) {
        self.acquire.record(ns);
    }
    fn record_host_wait_compute(&mut self, ns: u64) {
        self.host_wait_compute.record(ns);
    }
    fn record_host_staging(&mut self, ns: u64) {
        self.host_staging.record(ns);
    }
    fn record_staging_locks(&mut self, ns: u64) {
        self.staging_locks.record(ns);
    }
    fn record_staging_parallel(&mut self, ns: u64) {
        self.staging_parallel.record(ns);
    }
    fn record_staging_setup(&mut self, ns: u64) {
        self.staging_setup.record(ns);
    }
    fn record_sim_update(&mut self, ns: u64) {
        self.sim_update.record(ns);
    }
    fn record_staging_parents(&mut self, ns: u64) {
        self.staging_parents.record(ns);
    }
    fn record_staging_renderers(&mut self, ns: u64) {
        self.staging_renderers.record(ns);
    }
    /// Record one frame's GPU per-stage times. `deltas_ns[0..8]` are the
    /// seam + seven stage deltas, `deltas_ns[8]` the q0→q7 total — already
    /// converted from ticks to nanoseconds by the caller.
    fn record_gpu_timestamps(&mut self, deltas_ns: &[u64; 9]) {
        for (acc, &ns) in self.gpu_stages.iter_mut().zip(&deltas_ns[..8]) {
            acc.record(ns);
        }
        self.gpu_total.record(deltas_ns[8]);
    }

    /// `wait_mode` is tagged onto the sample line so an A/B log of the F7
    /// host-sync experiment is unambiguous about which mode produced it.
    fn tick(&mut self, wait_mode: &str, staging_mode: &str) {
        self.frame_count += 1;
        if self.frame_count & (FRAMES_PER_FPS_SAMPLE - 1) == 0 {
            let elapsed = self.last_print.elapsed();
            if elapsed.as_secs() >= 1 {
                let fps = self.frame_count as f64 / elapsed.as_secs_f64();
                println!(
                    "FPS: {:.0}  ({:.3} ms/frame)  [wait={} staging={}] | us min/avg/max  acquire {} | host_wait_compute {} | host_staging {} [locks {} | setup {} | parallel {} | parents {} | renderers {}] | sim_update {}",
                    fps,
                    1000.0 / fps,
                    wait_mode,
                    staging_mode,
                    self.acquire.fmt_us(),
                    self.host_wait_compute.fmt_us(),
                    self.host_staging.fmt_us(),
                    self.staging_locks.fmt_us(),
                    self.staging_setup.fmt_us(),
                    self.staging_parallel.fmt_us(),
                    self.staging_parents.fmt_us(),
                    self.staging_renderers.fmt_us(),
                    self.sim_update.fmt_us(),
                );
                println!(
                    "  gpu us min/avg/max  seam {} | scatter {} | mvp1 {} | raster1 {} | hiz {} | mvp2 {} | raster2 {} | blit {} | total {}",
                    self.gpu_stages[0].fmt_us(),
                    self.gpu_stages[1].fmt_us(),
                    self.gpu_stages[2].fmt_us(),
                    self.gpu_stages[3].fmt_us(),
                    self.gpu_stages[4].fmt_us(),
                    self.gpu_stages[5].fmt_us(),
                    self.gpu_stages[6].fmt_us(),
                    self.gpu_stages[7].fmt_us(),
                    self.gpu_total.fmt_us(),
                );
                let per_slot = |accs: &[PhaseAcc; STAGING_SLOTS], f: fn(&PhaseAcc) -> String| {
                    accs.iter()
                        .enumerate()
                        .map(|(i, a)| format!("slot{i} {}", f(a)))
                        .collect::<Vec<_>>()
                        .join(" | ")
                };
                println!(
                    "  scatter by staging slot  {}",
                    per_slot(&self.scatter_by_slot, PhaseAcc::fmt_us),
                );
                println!(
                    "  dirty words by slot      {}",
                    per_slot(&self.prepass_words_by_slot, |a| a.fmt_scaled(1.0)),
                );
                for (part, label) in ["trs", "spawn", "clears+vp", "signal"]
                    .iter()
                    .enumerate()
                    .map(|(i, l)| (i, l))
                {
                    let accs: [PhaseAcc; STAGING_SLOTS] =
                        std::array::from_fn(|s| self.scatter_parts_by_slot[s][part]);
                    println!(
                        "  {label:<10} by slot     {}",
                        per_slot(&accs, PhaseAcc::fmt_us)
                    );
                }
                if let Some(gpu) = &self.gpu {
                    println!("{}", gpu.sample_line());
                }
                self.frame_count = 0;
                self.last_print = Instant::now();
                self.acquire = PhaseAcc::default();
                self.host_wait_compute = PhaseAcc::default();
                self.host_staging = PhaseAcc::default();
                self.staging_locks = PhaseAcc::default();
                self.staging_setup = PhaseAcc::default();
                self.staging_parallel = PhaseAcc::default();
                self.staging_parents = PhaseAcc::default();
                self.staging_renderers = PhaseAcc::default();
                self.sim_update = PhaseAcc::default();
                self.gpu_stages = [PhaseAcc::default(); 8];
                self.gpu_total = PhaseAcc::default();
                self.scatter_by_slot = [PhaseAcc::default(); STAGING_SLOTS];
                self.scatter_parts_by_slot = [[PhaseAcc::default(); 4]; STAGING_SLOTS];
                self.prepass_words_by_slot = [PhaseAcc::default(); STAGING_SLOTS];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Staging memory-type balancer
// ─────────────────────────────────────────────────────────────────────────────

/// Rolling minimum over the last `WINDOW`..`2 * WINDOW` samples, `None` until
/// the first window closes.
///
/// The balancer wants each side's *steady* cost, and every signal it feeds on
/// is a clean floor with spikes on top: across a window `host_staging` reads
/// `402.7 / 426.6 / 541.7` min/avg/max, `raster1` reads `241.6 / 254.0 /
/// 263.8`. The floors are stable to ~1% and the spikes are asset loading,
/// a scheduler hiccup or a DPM ramp — none of which the choice of staging
/// memory can do anything about. A mean chases them; a minimum ignores them,
/// which is also why it needs no warm-up period and no outlier rejection.
#[derive(Clone, Copy)]
struct RollingMin {
    /// Last closed window's minimum; `INFINITY` until one closes.
    closed: f64,
    open: f64,
    n: u32,
}

impl Default for RollingMin {
    fn default() -> Self {
        Self {
            closed: f64::INFINITY,
            open: f64::INFINITY,
            n: 0,
        }
    }
}

impl RollingMin {
    /// Samples per window. Long enough to step over a hitch, short enough to
    /// follow a camera pan — ~60ms at the frame rates this runs at.
    const WINDOW: u32 = 128;

    fn push(&mut self, v: f64) {
        self.open = self.open.min(v);
        self.n += 1;
        if self.n == Self::WINDOW {
            *self = Self {
                closed: self.open,
                open: f64::INFINITY,
                n: 0,
            };
        }
    }
    fn get(self) -> Option<f64> {
        self.closed.is_finite().then(|| self.closed.min(self.open))
    }
}

/// Runtime CPU↔GPU balancer for the TRS staging memory type.
///
/// [`StagingMemory`] moves the transform upload's cost between the two
/// processors — ~280µs of it at 1M shapes, near 1:1. A frame costs
/// `max(cpu_busy, gpu_busy)`, so the right mode is whichever minimises that
/// max, and which one that is flips with the scene's CPU:GPU ratio.
///
/// Split each side into its staging part and the rest, and predict the mode
/// we are *not* in from the last time we were in it:
///
/// ```text
/// frame(m) = max(cpu_rest + cpu_staging[m], gpu_rest + gpu_scatter[m])
/// ```
///
/// Only the staging terms are mode-dependent, and they are driven by the
/// dirty-word count rather than by the camera, so they stay valid while the
/// view changes. `cpu_rest` / `gpu_rest` are shared and refreshed every frame
/// whichever mode is selected — which is what lets the balancer follow a
/// scene sliding from GPU-bound to CPU-bound. Splitting `gpu_rest` per mode
/// instead breaks exactly that case: the idle mode's raster estimate freezes
/// at the load it had when it was last selected.
///
/// **GPU durations must be clock-normalised before they enter this model.**
/// A GPU that is not the bottleneck has idle gaps, DPM downclocks it, and
/// every stage stretches on identical work — measured across one switch,
/// sclk 1232 → 3139MHz and `raster1` 254 → 116µs. Comparing a downclocked
/// `gpu_rest` against a full-speed `cpu_rest` says the GPU cannot afford the
/// scatter, which is a trap: the CPU-heavy mode manufactures the evidence
/// that keeps it selected. [`SclkMonitor`] supplies the correction, and the
/// model then works in the clock the GPU would run at *if* it were the
/// bottleneck — the only case in which its duration decides the frame.
///
/// A switch costs a full staging + FrameSlot rebuild, hence
/// [`SWITCH_MARGIN`](Self::SWITCH_MARGIN) and [`MIN_DWELL`](Self::MIN_DWELL).
struct StagingBalancer {
    /// `false` pins the mode (`ENGINE_STAGING_MODE=cached|vram`, or F6).
    auto: bool,
    mode: StagingMemory,
    /// `host_staging` / GPU scatter-block minima, indexed by `mode as usize`.
    cpu_staging: [RollingMin; 2],
    gpu_scatter: [RollingMin; 2],
    /// Everything else on each side — mode-independent.
    cpu_rest: RollingMin,
    gpu_rest: RollingMin,
    /// Live shader clock, for normalising the GPU samples. `None` when the
    /// platform exposes no clock, in which case the balancer is disabled
    /// rather than run on timings it cannot compare.
    sclk: Option<gpu_telemetry::SclkMonitor>,
    frames_in_mode: u32,
    last_switch: Instant,
    pending: Option<StagingMemory>,
}

impl StagingBalancer {
    /// Samples discarded after a switch. The GPU timestamps read each frame
    /// come from that (image, slot) pair's *previous* submission — up to
    /// `MAX_FRAMES_IN_FLIGHT * STAGING_SLOTS` frames back — so they describe
    /// the old mode for a while after the buffers change.
    const SETTLE_FRAMES: u32 = (MAX_FRAMES_IN_FLIGHT * STAGING_SLOTS) as u32 * 4;
    /// Minimum time between switches; a switch reallocates every staging
    /// slot and re-records every FrameSlot primary.
    const MIN_DWELL: Duration = Duration::from_millis(250);
    /// Predicted frame-time win required to pay for that.
    const SWITCH_MARGIN: f64 = 0.05;

    fn new(sclk: Option<gpu_telemetry::SclkMonitor>) -> Self {
        let (mut auto, mode) = match std::env::var("ENGINE_STAGING_MODE").as_deref() {
            Ok("cached") => (false, StagingMemory::HostCached),
            Ok("vram") => (false, StagingMemory::DeviceWc),
            Ok("auto") | Err(_) => (true, StagingMemory::HostCached),
            Ok(other) => panic!("ENGINE_STAGING_MODE must be cached|vram|auto, got {other:?}"),
        };
        if auto && sclk.is_none() {
            auto = false;
            println!("[staging] no shader-clock telemetry — GPU timings are not comparable across modes, auto disabled");
        }
        println!(
            "[staging] {} mode, starting on {}",
            if auto { "auto" } else { "pinned" },
            mode.label(),
        );
        Self {
            auto,
            mode,
            cpu_staging: [RollingMin::default(); 2],
            gpu_scatter: [RollingMin::default(); 2],
            cpu_rest: RollingMin::default(),
            gpu_rest: RollingMin::default(),
            sclk,
            frames_in_mode: 0,
            last_switch: Instant::now(),
            pending: None,
        }
    }

    fn settled(&self) -> bool {
        self.auto && self.frames_in_mode >= Self::SETTLE_FRAMES
    }

    /// One frame's GPU timings, from the in-CB timestamp queries, rescaled to
    /// what they would have cost at the reference clock. Without that the two
    /// modes' GPU numbers are measured on what is effectively different
    /// hardware — see the type docs.
    fn record_gpu(&mut self, total_ns: u64, scatter_ns: u64) {
        // A zero delta means the frame never wrote those queries — a rebuild
        // frame, or the first submission into a fresh pool. That is missing
        // data, not free work, and a minimum would latch onto it for good.
        if total_ns == 0 || scatter_ns == 0 {
            return;
        }
        let Some(scale) = self.sclk.as_ref().and_then(|s| s.normalise()) else {
            return;
        };
        if !self.settled() {
            return;
        }
        self.gpu_scatter[self.mode as usize].push(scatter_ns as f64 * scale);
        self.gpu_rest
            .push(total_ns.saturating_sub(scatter_ns) as f64 * scale);
    }

    /// One frame's host timings: total CPU work (frame period minus the
    /// blocking waits on the GPU) and the staging drain's share of it.
    /// Advances the frame counter and re-evaluates the mode.
    fn record_cpu(&mut self, busy_ns: u64, staging_ns: u64) {
        self.frames_in_mode = self.frames_in_mode.saturating_add(1);
        if !self.settled() {
            return;
        }
        self.cpu_staging[self.mode as usize].push(staging_ns as f64);
        self.cpu_rest.push(busy_ns.saturating_sub(staging_ns) as f64);
        self.evaluate();
    }

    fn evaluate(&mut self) {
        if self.pending.is_some() || self.last_switch.elapsed() < Self::MIN_DWELL {
            return;
        }
        let (Some(cpu_rest), Some(gpu_rest)) = (self.cpu_rest.get(), self.gpu_rest.get()) else {
            return;
        };
        let predict = |m: StagingMemory| -> Option<f64> {
            let cpu = cpu_rest + self.cpu_staging[m as usize].get()?;
            let gpu = gpu_rest + self.gpu_scatter[m as usize].get()?;
            Some(cpu.max(gpu))
        };
        let other = self.mode.other();
        let Some(current) = predict(self.mode) else {
            return;
        };
        // Never-measured mode: probe it. One dwell of a possibly-worse mode
        // buys the only numbers that can rule it out.
        let win = match predict(other) {
            None => true,
            Some(o) => o < current * (1.0 - Self::SWITCH_MARGIN),
        };
        if win {
            // Every term the decision rested on, in µs. Only fires on a
            // switch, and without it a wrong choice is indistinguishable from
            // a wrong measurement.
            let us = |v: Option<f64>| match v {
                Some(v) => format!("{:.0}", v / 1000.0),
                None => "-".to_string(),
            };
            println!(
                "[staging] {:?} {}us -> {:?} {}us | cpu_rest {} gpu_rest {} staging {}/{} scatter {}/{}",
                self.mode,
                us(Some(current)),
                other,
                us(predict(other)),
                us(Some(cpu_rest)),
                us(self.gpu_rest.get()),
                us(self.cpu_staging[0].get()),
                us(self.cpu_staging[1].get()),
                us(self.gpu_scatter[0].get()),
                us(self.gpu_scatter[1].get()),
            );
            self.pending = Some(other);
        }
    }

    /// Mode the renderer should switch to this frame, if any. Taking it
    /// commits the switch: the settle window and dwell timer restart.
    fn take_pending(&mut self) -> Option<StagingMemory> {
        let mode = self.pending.take()?;
        self.mode = mode;
        self.frames_in_mode = 0;
        self.last_switch = Instant::now();
        Some(mode)
    }

    /// F6 cycles auto → cached → vram → auto.
    fn cycle(&mut self) {
        let (auto, mode) = match (self.auto, self.mode) {
            (true, _) => (false, StagingMemory::HostCached),
            (false, StagingMemory::HostCached) => (false, StagingMemory::DeviceWc),
            (false, StagingMemory::DeviceWc) => (true, self.mode),
        };
        self.auto = auto;
        if mode != self.mode {
            self.pending = Some(mode);
        }
        println!(
            "[staging] {} mode, {}",
            if auto { "auto" } else { "pinned" },
            mode.label(),
        );
    }

    /// Short tag for the FPS line.
    fn tag(&self) -> &'static str {
        match (self.auto, self.mode) {
            (true, StagingMemory::HostCached) => "auto:cached",
            (true, StagingMemory::DeviceWc) => "auto:vram",
            (false, StagingMemory::HostCached) => "cached",
            (false, StagingMemory::DeviceWc) => "vram",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RenderApp  (internal event-loop handler)
// ─────────────────────────────────────────────────────────────────────────────

/// All state that lives for the entire event-loop lifetime.
struct RenderApp {
    title: String,
    context: VulkanoContext,
    graphics_queue: Arc<Queue>,
    swapchain_renderer: Option<SwapchainRenderer>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    fps: FrameStats,
    /// Picks which side of the PCIe link pays for the transform upload.
    staging_balancer: StagingBalancer,
    pipeline: Option<Arc<GraphicsPipeline>>,
    /// Dual-pass occlusion culling compute pipelines (stateless, shared by
    /// every camera — see `camera.rs`'s `CameraSceneResources`). Built once
    /// in `resumed()`, alongside `pipeline`.
    mvp_build_pass2_pipeline: Option<Arc<ComputePipeline>>,
    cull_pass2_args_pipeline: Option<Arc<ComputePipeline>>,
    hiz_reduce_depth_pipeline: Option<Arc<ComputePipeline>>,
    hiz_reduce_mip_pipeline: Option<Arc<ComputePipeline>>,
    hiz_reduce_mip2_pipeline: Option<Arc<ComputePipeline>>,
    rcx: Option<RenderContext>,

    // ── Scene state ─────────────────────────────────────────────────
    /// The window's root scene — owns the transform hierarchy and the
    /// component registry. Mutated each frame via `Scene::update(dt)`.
    root_scene: Option<Scene>,
    last_frame_time: Option<Instant>,
    /// Total frames rendered. Used for one-shot post-warmup diagnostics
    /// (e.g. NUMA residency verification).
    total_frames: u64,
    /// Host sync mode for the per-frame staging gate. `false` (default) —
    /// the mid-CB early wake (`host_wait_for_previous_compute`): host
    /// resumes as soon as the previous frame's host-shared reads are done
    /// and pipelines its staging writes against the rest of that frame's
    /// GPU work. `true` — the full-retirement wait
    /// ([`SwapchainRenderer::wait_previous_frame`], the previous
    /// submission's `in_flight` fence): host stalls until the previous
    /// frame has entirely retired, giving up CPU/GPU overlap in exchange
    /// for an uncontended staging window.
    ///
    /// `ENGINE_SCATTER_TRACE=1`: dump every frame's raw scatter time.
    scatter_trace: bool,
    /// `ENGINE_STAGING_POOL=1`: run the TRS drain on a separate node-local
    /// pool ([`staging_pool`]) instead of the global one.
    ///
    /// **Defaults off — it measured worse.** It does deliver the full
    /// cache-locality win (scatter 318µs, same as binding the whole
    /// process, and flat from 16 to 128 staging threads), but splitting the
    /// drain off the global pool breaks the worker↔transform-range sharing
    /// that `bitmap_task_layout` exists to provide: `Scene::update` and the
    /// staging drain no longer hand the same range to the same worker, so
    /// each phase runs over data the other just evicted. `sim_update` goes
    /// 240µs → ~800µs and stays there at *every* staging-pool width, which
    /// is what distinguishes a broken locality contract from mere CPU
    /// contention. Net 663 FPS versus 803 for one process-bound pool.
    use_staging_pool: bool,
    /// `ENGINE_UI_TRACE=1`: print the UI's dirty-word count on every frame
    /// that uploads anything. Silence means the retained UI cost zero bytes
    /// that frame, which is the property ADR-0006 exists to deliver.
    ui_trace: bool,
    /// Toggled live with **F7**; initial value from `ENGINE_WAIT_MODE`
    /// (`frame` → `true`, anything else / unset → `false`). Neither mode
    /// changes what gets recorded, so flipping this needs no CB re-record.
    wait_on_frame: bool,
}

/// Swapchain-image-count-sized arrays rebuilt on every swapchain recreation.
struct RenderContext {
    /// Cached swapchain image views. Used as **blit destinations** by each
    /// FrameSlot's pre-recorded CB; refreshed on resize.
    swapchain_image_views: Vec<Arc<ImageView>>,
    /// World-scoped GPU transform state: SoT (pos/rot/scale) buffers +
    /// scatter / mvp-build compute pipelines. Shared by every camera that
    /// targets this scene; sized to the transform hierarchy's entity
    /// count, grown geometrically on demand.
    world_transforms: WorldTransformGpu,
    /// The render-side camera that drives the scene render. Owns its own
    /// offscreen color + depth attachments and a [`CameraResolution`] policy
    /// (currently always `MatchSwapchain`, so the present-blit stays 1:1).
    /// On a swapchain resize the camera decides whether to rebuild its
    /// attachments — future `Fixed` / `ScaleSwapchain` cameras will survive
    /// swapchain resizes untouched without changing the swapchain handler.
    main_camera: RenderCamera,
    /// One `FrameSlot` per swapchain image. Each slot owns the per-frame
    /// staging matrix buffer, the blit secondary, and the composing primary
    /// CB that references `main_camera`'s device matrices + scene secondary
    /// and this slot's swapchain image as the blit destination.
    frame_slots: Vec<FrameSlot>,
    /// GPU mirror of the core mesh asset registry (mega buffers + table +
    /// redirect). `sync()`ed each frame.
    gpu_mesh_store: GpuMeshStore,
    /// GPU mirror of the core texture registry (sampled images + redirect).
    /// `sync()`ed each frame; a texture arrival rides the `force_full`
    /// rebuild path (descriptor set + scene secondary + frame slots).
    gpu_texture_store: GpuTextureStore,
    /// GPU mirror of the core material registry (material SSBO + redirect).
    /// `sync()`ed each frame; a material arrival/edit rides `force_full`.
    gpu_material_store: GpuMaterialStore,
    /// Per-transform `GPURenderers` buffer (`(mesh_id, material_id)` per
    /// transform slot), filled by scattering newly-spawned / re-pointed
    /// `MeshRenderer` components.
    gpu_renderers: GpuRenderers,
    /// Device side of the UI: SoT arrays, staging, the four scatters, the
    /// glyph atlas, and the single-`draw_indirect` graphics secondary.
    ///
    /// The *host* side is not here — it is the global `ui::ui()` store, so
    /// game and editor code can reach it (ADR-0008). The renderer owns the
    /// GPU mirror and nothing about what the UI contains.
    ui_gpu: UiGpu,
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

        let fps = FrameStats::new();
        let staging_balancer = StagingBalancer::new(fps.spawn_sclk_monitor());

        RenderApp {
            title,
            context,
            graphics_queue,
            swapchain_renderer: None,
            command_buffer_allocator,
            memory_allocator,
            descriptor_set_allocator,
            fps,
            staging_balancer,
            pipeline: None,
            mvp_build_pass2_pipeline: None,
            cull_pass2_args_pipeline: None,
            hiz_reduce_depth_pipeline: None,
            hiz_reduce_mip_pipeline: None,
            hiz_reduce_mip2_pipeline: None,
            rcx: None,
            root_scene,
            last_frame_time: None,
            total_frames: 0,
            scatter_trace: std::env::var("ENGINE_SCATTER_TRACE").is_ok_and(|v| v == "1"),
            ui_trace: std::env::var("ENGINE_UI_TRACE").is_ok_and(|v| v == "1"),
            use_staging_pool: std::env::var("ENGINE_STAGING_POOL")
                .is_ok_and(|v| v == "1" || v == "true"),
            wait_on_frame: std::env::var("ENGINE_WAIT_MODE")
                .map(|v| v.eq_ignore_ascii_case("frame"))
                .unwrap_or(false),
        }
    }
}

impl ApplicationHandler for RenderApp {
    /// Called once at startup (and again on Android resume cycles).
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Drop stale renderer on mobile resume.
        self.swapchain_renderer = None;

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

        // `ENGINE_WINDOW_SIZE=WxH` forces the initial size. Benchmark knob:
        // shrinking the viewport makes raster's pixel work negligible while
        // leaving the scene in view and the transform/cull work untouched,
        // which separates "raster is expensive" from "the scene is visible".
        let mut attrs = WindowAttributes::default().with_title(self.title.clone());
        if let Ok(spec) = std::env::var("ENGINE_WINDOW_SIZE") {
            let (w, h) = spec
                .split_once(['x', 'X'])
                .and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)))
                .expect("ENGINE_WINDOW_SIZE must look like 1920x1080");
            attrs = attrs.with_inner_size(winit::dpi::PhysicalSize::<u32>::new(w, h));
        }
        let real_window = event_loop
            .create_window(attrs)
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
        self.pipeline = Some(pipeline.clone());
        // The scene pipeline is built against `CAMERA_COLOR_FORMAT` and the
        // present-blit handles the conversion; `swapchain_format` matters
        // only to the UI pipeline below, which draws straight into the
        // swapchain image after that blit.

        // Dual-pass occlusion culling compute pipelines — stateless, built
        // once and shared by every camera (see `camera.rs`'s
        // `CameraSceneResources`), same pattern as `pipeline` above.
        let mvp_build_pass2_pipeline =
            create_mvp_build_pass2_pipeline(self.context.device().clone());
        self.mvp_build_pass2_pipeline = Some(mvp_build_pass2_pipeline.clone());
        let cull_pass2_args_pipeline =
            create_cull_pass2_args_pipeline(self.context.device().clone());
        self.cull_pass2_args_pipeline = Some(cull_pass2_args_pipeline.clone());
        let hiz_reduce_depth_pipeline =
            create_hiz_reduce_depth_pipeline(self.context.device().clone());
        self.hiz_reduce_depth_pipeline = Some(hiz_reduce_depth_pipeline.clone());
        let hiz_reduce_mip_pipeline = create_hiz_reduce_mip_pipeline(self.context.device().clone());
        self.hiz_reduce_mip_pipeline = Some(hiz_reduce_mip_pipeline.clone());
        let hiz_reduce_mip2_pipeline =
            create_hiz_reduce_mip2_pipeline(self.context.device().clone());
        self.hiz_reduce_mip2_pipeline = Some(hiz_reduce_mip2_pipeline.clone());

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
            self.staging_balancer.mode,
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

        // Retained UI. Built after the texture store's first sync — its
        // bindless array binds `descriptor_array()`, which requires the
        // placeholder slot to be resident.
        let mut ui_gpu = UiGpu::new(
            self.context.device().clone(),
            self.memory_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.graphics_queue.clone(),
            &gpu_texture_store,
            swapchain_format,
            initial_extent,
        );
        // Whatever the game or editor built into the global store before
        // `Window::run` gets its first layout and its device capacity here.
        // An app that built no UI leaves this an empty store, which costs
        // one zero-workgroup draw per frame.
        ui::ui().run_layout([initial_extent[0] as f32, initial_extent[1] as f32]);
        ui_gpu.ensure_capacity(&mut ui::ui());

        let frame_slots = build_all_frame_slots(
            &self.command_buffer_allocator,
            &self.memory_allocator,
            self.graphics_queue.queue_family_index(),
            &attachment_image_views,
            &main_camera,
            &world_transforms,
            &gpu_renderers,
            &ui_gpu,
        );

        self.rcx = Some(RenderContext {
            swapchain_image_views: attachment_image_views,
            world_transforms,
            main_camera,
            frame_slots,
            gpu_mesh_store,
            gpu_texture_store,
            gpu_material_store,
            gpu_renderers,
            ui_gpu,
        });
        self.swapchain_renderer = Some(swapchain_renderer);
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

        let renderer = match self.swapchain_renderer.as_mut() {
            Some(r) => r,
            None => return,
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => renderer.resize(),
            WindowEvent::RedrawRequested => {}
            _ => {}
        }
    }

    /// Render one frame; runs at full speed (`ControlFlow::Poll`).
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);

        let renderer = match self.swapchain_renderer.as_mut() {
            Some(r) => r,
            None => return,
        };
        let rcx = match self.rcx.as_mut() {
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

        // Published before `Scene::update` so a component's `stats::dt()`
        // agrees with the `dt` it was handed.
        {
            let [w, h, _] = rcx.swapchain_image_views[0].image().extent();
            stats::publish(dt, [w, h]);
        }

        // UI hit testing, also before `Scene::update`: a component's
        // `clicked()` must see this frame's press, and `OrbitController` must
        // be able to decline to orbit when the UI took it. Reads the same
        // edge-triggered state components do — the transients are cleared
        // further down, after every `update` has run.
        {
            let inp = input::global();
            let c = inp.cursor_position();
            ui::ui().update_pointer(
                [c.x, c.y],
                inp.mouse_pressed(MouseButton::Left),
                inp.mouse_released(MouseButton::Left),
            );
        }

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
            let inst = Instant::now();
            scene.update(dt);
            self.fps.record_sim_update(inst.elapsed().as_nanos() as u64);
        }

        // Every component's `update` for this frame has now run and had a
        // chance to observe the input accumulated since the last frame.
        // The renderer's own debug hotkeys (F8/F9, below) still need to
        // observe this frame's edge-triggered state too, so the transient
        // (`*_pressed` / `*_released` / deltas) clear is deferred to just
        // after those checks — see `input::global_mut().end_frame()` below.

        // Drain the hierarchy's streamed parent changes now — after the
        // sim update and subscene instantiation, so this frame's
        // re-parents are included. The pairs are *written* into the
        // parent-update staging later, inside the harvest (after the
        // `gpu_signal` wait); draining early lets the staging-capacity
        // check below participate in the rebuild decisions.
        // TODO: profile drain. prefer to avoid copies/re-allocs and parallelize
        let parent_updates: Vec<[u32; 2]> = self
            .root_scene
            .as_ref()
            .map(|s| s.transform_hierarchy.drain_parent_updates())
            .unwrap_or_default();

        // Pre-clone everything the swapchain-recreation closure needs so it
        // doesn't capture `self`.
        let memory_allocator = self.memory_allocator.clone();
        let cb_allocator = self.command_buffer_allocator.clone();
        let descriptor_set_allocator = self.descriptor_set_allocator.clone();
        let pipeline_for_recreate = self.pipeline.clone().expect("Pipeline not initialised");
        let mvp_build_pass2_pipeline = self
            .mvp_build_pass2_pipeline
            .clone()
            .expect("mvp_build_pass2_pipeline not initialised");
        let cull_pass2_args_pipeline = self
            .cull_pass2_args_pipeline
            .clone()
            .expect("cull_pass2_args_pipeline not initialised");
        let hiz_reduce_depth_pipeline = self
            .hiz_reduce_depth_pipeline
            .clone()
            .expect("hiz_reduce_depth_pipeline not initialised");
        let hiz_reduce_mip_pipeline = self
            .hiz_reduce_mip_pipeline
            .clone()
            .expect("hiz_reduce_mip_pipeline not initialised");
        let hiz_reduce_mip2_pipeline = self
            .hiz_reduce_mip2_pipeline
            .clone()
            .expect("hiz_reduce_mip2_pipeline not initialised");
        let queue_family_index = self.graphics_queue.queue_family_index();

        let acquire_start = Instant::now();
        let frame = match renderer.acquire(|swapchain_images| {
            rcx.swapchain_image_views = swapchain_images.to_vec();
            // Inform the main camera of the new swapchain extent. With the
            // current `MatchSwapchain` policy this re-creates the camera's
            // attachments AND re-records its scene secondary (viewport
            // depends on extent). Future cameras with a swapchain-independent
            // policy (`Fixed` / `ScaleSwapchain`) would survive this call
            // untouched, and only the per-image blit secondary + primary
            // would need a rebuild on swapchain change.
            let new_extent = {
                let [w, h, _] = swapchain_images[0].image().extent();
                [w, h]
            };
            let scene_resources = CameraSceneResources {
                cb_allocator: &cb_allocator,
                descriptor_set_allocator: &descriptor_set_allocator,
                memory_allocator: &memory_allocator,
                pipeline: &pipeline_for_recreate,
                queue_family_index,
                world_transforms: &rcx.world_transforms,
                mesh_store: &rcx.gpu_mesh_store,
                texture_store: &rcx.gpu_texture_store,
                material_store: &rcx.gpu_material_store,
                gpu_renderers: &rcx.gpu_renderers,
                mvp_build_pass2_pipeline: &mvp_build_pass2_pipeline,
                cull_pass2_args_pipeline: &cull_pass2_args_pipeline,
                hiz_reduce_depth_pipeline: &hiz_reduce_depth_pipeline,
                hiz_reduce_mip_pipeline: &hiz_reduce_mip_pipeline,
                hiz_reduce_mip2_pipeline: &hiz_reduce_mip2_pipeline,
            };
            let _camera_rebuilt = rcx
                .main_camera
                .on_swapchain_resize(new_extent, &scene_resources);

            // The CBs in every slot reference the *old* swapchain images
            // (as blit destinations) and — if the camera rebuilt — the
            // *old* offscreen color/depth attachments and *old* scene
            // secondary. Rebuild every per-image slot from scratch. The
            // camera's device matrices + descriptor set survive untouched.
            // Drop the old slots BEFORE building new ones. Pre-staging-
            // paradigm refactor this was *required* because each old
            // primary held a `MultipleSubmit` lock on a per-image
            // `mvp_build_secondary[image_index]`. Now `mvp_build_secondary`
            // is `SimultaneousUse` (single shared per camera), so it's
            // not strictly required — but defensive: keeps the rebuild
            // ordering robust if any per-image MultipleSubmit secondary
            // gets added back later.
            // The UI draws straight into the swapchain image, so its
            // viewport and px -> NDC push constant follow the new extent.
            rcx.ui_gpu.on_resize(new_extent);

            rcx.frame_slots.clear();
            rcx.frame_slots = build_all_frame_slots(
                &cb_allocator,
                &memory_allocator,
                queue_family_index,
                &rcx.swapchain_image_views,
                &rcx.main_camera,
                &rcx.world_transforms,
                &rcx.gpu_renderers,
                &rcx.ui_gpu,
            );
        }) {
            Some(f) => f,
            None => return, // out-of-date / minimised — skip frame
        };
        let acquire_ns = acquire_start.elapsed().as_nanos() as u64;
        self.fps.record_acquire(acquire_ns);

        // GPU per-stage timestamps from this image's *previous* submission
        // (fully retired — `acquire` waited its `in_flight` fence, so this
        // never blocks). All-or-nothing: a freshly rebuilt slot's pool has
        // never been written, `get_results` reports not-all-available, and
        // the sample is skipped. 2-3 frames of latency, irrelevant for the
        // 1-second aggregation window.
        {
            // This (image, staging slot) pair's own pool, holding the
            // timings from the last frame that used the same pair — with
            // 4 images × 2 slots that's up to 8 frames of latency rather
            // than 4. Irrelevant against the 1-second aggregation window.
            let ts_slot = rcx.world_transforms.write_slot();
            let pool = &rcx.frame_slots[frame_slot_index(frame.image_index as usize, ts_slot)]
                .timestamp_pool;
            let mut ticks = [0u64; GPU_TS_COUNT as usize];
            if let Ok(true) =
                pool.get_results(0..GPU_TS_COUNT, &mut ticks, QueryResultFlags::empty())
            {
                let period_ns = self
                    .context
                    .device()
                    .physical_device()
                    .properties()
                    .timestamp_period as f64;
                let delta = |a: usize, b: usize| -> u64 {
                    (ticks[b].saturating_sub(ticks[a]) as f64 * period_ns) as u64
                };
                self.fps.record_gpu_timestamps(&[
                    delta(0, 8), // seam: previous submissions draining
                    delta(8, 1), // scatter block
                    delta(1, 2), // mvp_build pass 1
                    delta(2, 3), // raster pass 1
                    delta(3, 4), // Hi-Z build
                    delta(4, 5), // pass-2 cull + history update
                    delta(5, 6), // raster pass 2
                    delta(6, 7), // present blit
                    delta(8, 7), // this frame's own work (seam excluded)
                ]);
                self.staging_balancer.record_gpu(delta(8, 7), delta(8, 1));
                // Same scatter number, but split by which staging slot the
                // frame read. If the two accumulators separate cleanly the
                // spread is a *per-slot* property (page placement, NUMA
                // node, first-touch residency) rather than frame-to-frame
                // noise — the aggregate min/avg/max cannot tell those
                // apart, and here avg sits almost exactly at the midpoint
                // of min and max, which is what a 50/50 bimodal split
                // looks like.
                self.fps.scatter_by_slot[ts_slot].record(delta(8, 1));
                for (acc, ns) in self.fps.scatter_parts_by_slot[ts_slot].iter_mut().zip([
                    delta(8, 9),
                    delta(9, 10),
                    delta(10, 11),
                    delta(11, 1),
                ]) {
                    acc.record(ns);
                }
                // `ENGINE_SCATTER_TRACE=1`: raw per-frame scatter times. The
                // aggregates hide the shape — the scatter is bimodal, and
                // only the raw sequence says whether the fast frames come in
                // bursts, alternate, or track the staging slot.
                if self.scatter_trace {
                    println!(
                        "[trace] slot{ts_slot} scatter_us {:.1}",
                        delta(8, 1) as f64 / 1000.0
                    );
                }
            }
        }

        // ── Retained UI (ADR-0006 / ADR-0008) ───────────────────────────────
        // Components have had their chance to write; solve the tree and push
        // the results into the primitive slots. Runs before the capacity
        // checks below, so a UI that grew past its device arrays this frame
        // is covered by the same rebuild the rest of the engine does.
        //
        // Unconditional and effectively free: `run_layout` returns
        // immediately when nothing marked the tree dirty, which is almost
        // every frame. It lives here rather than in app code because a game
        // that forgot to call it would get a UI that never lays out.
        {
            let [w, h, _] = rcx.swapchain_image_views[0].image().extent();
            ui::ui().run_layout([w as f32, h as f32]);
        }

        // ── World + renderer capacity (per-world axis) ──────────────────────
        // The hierarchy may have grown past the SoT / GPURenderers buffers.
        // Geometric growth keeps this rare.
        let entity_count = self
            .root_scene
            .as_ref()
            .map(|s| s.transform_hierarchy.len())
            .unwrap_or(1)
            .max(1);
        let mut need_frame_slot_rebuild = false;
        let grew_world = rcx
            .world_transforms
            .ensure_capacity(&self.memory_allocator, entity_count);
        if grew_world {
            // SoT re-allocated — its contents are undefined. Re-mark every
            // entity's TRS dirty so the next harvest repopulates the new SoT.
            if let Some(scene) = self.root_scene.as_ref() {
                scene.transform_hierarchy.dirty().mark_all_trs();
            }
        }
        // The cull dispatches over the (geometric) entity capacity, so a spawn
        // within capacity doesn't change its range; grow GPURenderers to match.
        let renderer_capacity = rcx.world_transforms.entity_capacity();
        let grew_renderers = rcx.gpu_renderers.ensure_capacity(renderer_capacity as u32);
        // Parent-update staging must fit this frame's drained burst. A grow
        // re-records the scatter secondary (captured by every FrameSlot
        // primary), so it forces the full rebuild path below. The parents
        // SoT itself grew inside `ensure_capacity` above, copy-preserved.
        let grew_parent_staging = rcx
            .world_transforms
            .ensure_parent_update_capacity(parent_updates.len());
        // Re-home the TRS staging triple when the balancer says the other
        // side of the link is now the cheaper one to charge. Same rebuild
        // class as a capacity grow, and safe for the same reason: the new
        // buffers are untouched by any in-flight frame, and the old ones
        // stay alive through the submissions still holding them.
        //
        // Only the staging slots change, so this needs the FrameSlot
        // primaries (they bake in the slot's scatter secondary) but not the
        // camera rebuild `force_full` drives — the SoT is untouched.
        if self
            .staging_balancer
            .take_pending()
            .is_some_and(|mode| rcx.world_transforms.set_staging_memory(mode))
        {
            need_frame_slot_rebuild = true;
            println!(
                "[staging] switched to {}",
                rcx.world_transforms.staging_memory().label(),
            );
        }

        // ── Mesh sync + renderer scatter (Design B, GPU-driven) ─────────────
        // `sync` uploads any newly-resolved geometry, patches the GPU redirect,
        // and returns the per-slot instance totals (consistent with that
        // redirect). Drain freshly-spawned renderers and scatter them into the
        // GPURenderers buffer. The cull pass reads GPURenderers + redirect +
        // mesh_table directly each frame — there is no CPU topology to derive.
        let (mesh_changed, slot_totals) = rcx.gpu_mesh_store.sync();
        // Texture arrivals (decoded slots / redirect flips) require the
        // graphics texture set + scene secondary to rebind, which the
        // `force_full` path below does. Rare: once per decoded texture.
        let tex_changed = rcx.gpu_texture_store.sync();
        // Material arrivals / in-place edits likewise rebind through
        // `force_full`. Rare: once per created/edited material.
        let mat_changed = rcx.gpu_material_store.sync();
        // The UI's own bindless array is a second copy of the texture
        // store's descriptors, so it rebinds on the same arrivals — and,
        // like the scene's, that re-records a secondary the frame primaries
        // capture.
        if tex_changed {
            rcx.ui_gpu.refresh_textures(&rcx.gpu_texture_store);
            need_frame_slot_rebuild = true;
        }
        // A UI capacity grow reallocates the SoT and re-records both UI
        // secondaries, and re-marks every mirror so the fresh SoT is
        // repopulated. Rare by construction (geometric, never shrinks).
        if rcx.ui_gpu.ensure_capacity(&mut ui::ui()) {
            need_frame_slot_rebuild = true;
        }
        // Drain freshly-spawned renderers now; the pairs are *written* into
        // the spawn staging in the harvest below (after the `gpu_signal`
        // wait) and scattered by the in-CB spawn-scatter secondary. The
        // capacity check here participates in the rebuild decisions — a
        // staging grow re-records the secondary the frame primaries capture.
        let spawns = components::drain_spawns();
        let grew_spawn_staging = rcx.gpu_renderers.ensure_spawn_capacity(spawns.len());

        // Update the camera's draw resources when the topology changed. A
        // within-capacity spawn of an existing mesh only shifts the per-slot
        // bases — the **cheap path**: rewrite the indirect template in place,
        // deferred until after the compute wait (no descriptor / secondary /
        // frame-slot rebuild). A load, a new mesh, or a capacity grow takes the
        // **full path** (`force_full` when a cull-bound buffer reallocated).
        let plan_dirty = !spawns.is_empty() || mesh_changed;
        let force_full = grew_world
            || grew_renderers
            || grew_parent_staging
            || grew_spawn_staging
            || mesh_changed
            || tex_changed
            || mat_changed;
        let mut pending_cheap_plan: Option<DrawPlan> = None;
        if plan_dirty || force_full {
            let plan = build_draw_plan(&rcx.gpu_mesh_store, &slot_totals);
            if rcx
                .main_camera
                .needs_structural_rebuild(&plan, renderer_capacity, force_full)
            {
                let scene_resources = CameraSceneResources {
                    cb_allocator: &self.command_buffer_allocator,
                    descriptor_set_allocator: &self.descriptor_set_allocator,
                    memory_allocator: &self.memory_allocator,
                    pipeline: &self.pipeline.clone().expect("pipeline"),
                    queue_family_index: self.graphics_queue.queue_family_index(),
                    world_transforms: &rcx.world_transforms,
                    mesh_store: &rcx.gpu_mesh_store,
                    texture_store: &rcx.gpu_texture_store,
                    material_store: &rcx.gpu_material_store,
                    gpu_renderers: &rcx.gpu_renderers,
                    mvp_build_pass2_pipeline: &self
                        .mvp_build_pass2_pipeline
                        .clone()
                        .expect("mvp_build_pass2_pipeline"),
                    cull_pass2_args_pipeline: &self
                        .cull_pass2_args_pipeline
                        .clone()
                        .expect("cull_pass2_args_pipeline"),
                    hiz_reduce_depth_pipeline: &self
                        .hiz_reduce_depth_pipeline
                        .clone()
                        .expect("hiz_reduce_depth_pipeline"),
                    hiz_reduce_mip_pipeline: &self
                        .hiz_reduce_mip_pipeline
                        .clone()
                        .expect("hiz_reduce_mip_pipeline"),
                    hiz_reduce_mip2_pipeline: &self
                        .hiz_reduce_mip2_pipeline
                        .clone()
                        .expect("hiz_reduce_mip2_pipeline"),
                };
                rcx.main_camera
                    .ensure_current(&plan, renderer_capacity, &scene_resources);
                need_frame_slot_rebuild = true;
            } else {
                pending_cheap_plan = Some(plan);
            }
        }

        // Debug: bring the Hi-Z freeze state (frustum-lock feature) into
        // line with `cull_lock`, one frame behind by construction (this
        // always runs before this same frame's own F9 check below, so a
        // lock engaged *this* frame is only picked up here *next* frame —
        // see `RenderCamera::apply_pending_hiz_freeze`'s doc comment for
        // why that delay matters).
        if rcx.main_camera.apply_pending_hiz_freeze() {
            need_frame_slot_rebuild = true;
        }

        // Debug: F8 toggles occlusion culling entirely. Rebuilds
        // `cull_secondary` (cheap) and — since `lib.rs::build_frame_slot`
        // decides whether to include the Hi-Z build / pass 2 cull / pass 2
        // render / history-update secondaries in the primary based on this
        // flag — forces a frame-slot rebuild, same cost class as a
        // capacity/extent change.
        if input::key_pressed(KeyCode::F8) {
            let desired = !rcx.main_camera.occlusion_enabled();
            let scene_resources = CameraSceneResources {
                cb_allocator: &self.command_buffer_allocator,
                descriptor_set_allocator: &self.descriptor_set_allocator,
                memory_allocator: &self.memory_allocator,
                pipeline: &self.pipeline.clone().expect("pipeline"),
                queue_family_index: self.graphics_queue.queue_family_index(),
                world_transforms: &rcx.world_transforms,
                mesh_store: &rcx.gpu_mesh_store,
                texture_store: &rcx.gpu_texture_store,
                material_store: &rcx.gpu_material_store,
                gpu_renderers: &rcx.gpu_renderers,
                mvp_build_pass2_pipeline: &self
                    .mvp_build_pass2_pipeline
                    .clone()
                    .expect("mvp_build_pass2_pipeline"),
                cull_pass2_args_pipeline: &self
                    .cull_pass2_args_pipeline
                    .clone()
                    .expect("cull_pass2_args_pipeline"),
                hiz_reduce_depth_pipeline: &self
                    .hiz_reduce_depth_pipeline
                    .clone()
                    .expect("hiz_reduce_depth_pipeline"),
                hiz_reduce_mip_pipeline: &self
                    .hiz_reduce_mip_pipeline
                    .clone()
                    .expect("hiz_reduce_mip_pipeline"),
                hiz_reduce_mip2_pipeline: &self
                    .hiz_reduce_mip2_pipeline
                    .clone()
                    .expect("hiz_reduce_mip2_pipeline"),
            };
            if rcx
                .main_camera
                .set_occlusion_enabled(desired, &scene_resources)
            {
                need_frame_slot_rebuild = true;
            }
        }

        if need_frame_slot_rebuild {
            // See the corresponding `clear()` in the on_recreate closure
            // above for the rationale.
            rcx.frame_slots.clear();
            rcx.frame_slots = build_all_frame_slots(
                &self.command_buffer_allocator,
                &self.memory_allocator,
                self.graphics_queue.queue_family_index(),
                &rcx.swapchain_image_views,
                &rcx.main_camera,
                &rcx.world_transforms,
                &rcx.gpu_renderers,
                &rcx.ui_gpu,
            );
        }

        // ── Sparse staging upload driven by `TransformHierarchy::Dirty` ─────
        let image_index = frame.image_index as usize;
        let [w, h, _] = rcx.swapchain_image_views[image_index].image().extent();
        let aspect = w as f32 / h.max(1) as f32;
        // The camera is just another component: locate the scene's (first)
        // `CameraComponent` and read its entity's *global* position +
        // rotation to build the view matrix. No camera in the scene yet
        // (e.g. the very first frame before the game's setup code runs) —
        // fall back to an identity-posed default so there's still something
        // to render into.
        // The world position comes along for the ride: `scene.frag`'s PBR
        // view vector needs it (see `transform_gpu::CAMERA_BLOCK_MAT4S`).
        let (view_proj, camera_position) = self
            .root_scene
            .as_ref()
            .and_then(|scene| {
                let (entity, cam) = scene.first_component::<scene::CameraComponent>()?;
                let cam = cam.lock();
                let t = scene
                    .transform_hierarchy
                    .get_transform_unchecked(entity.id)
                    .lock();
                let position = t.get_global_position();
                Some((
                    cam.view_proj(position, t.get_global_rotation(), aspect),
                    position,
                ))
            })
            .unwrap_or_else(|| {
                (
                    scene::CameraComponent::new().view_proj(
                        glam::Vec3::ZERO,
                        glam::Quat::IDENTITY,
                        aspect,
                    ),
                    glam::Vec3::ZERO,
                )
            });

        // Debug: F9 toggles the frustum-lock feature. Engaging it snapshots
        // *this* frame's `view_proj` as the frozen cull-test vantage point;
        // the render camera (and `view_proj` above) keeps following live
        // input either way — only `mvp_build.comp`'s frustum test reads the
        // locked value (see `RenderCamera::set_cull_lock`).
        if input::key_pressed(KeyCode::F9) {
            let new_lock = !rcx.main_camera.cull_lock();
            rcx.main_camera
                .set_cull_lock(new_lock, view_proj.to_cols_array());
        }

        // `ENGINE_CULL_AWAY=1` engages the frustum lock at startup on a
        // vantage point shifted 1e7 units off, so every object fails the
        // frustum test and pass 1 rasterises nothing. Benchmark knob: it
        // reproduces the "camera looking away" best case deterministically,
        // with the staging/scatter work byte-for-byte identical, which is
        // the A/B needed to see whether the scatter's cost really depends
        // on how much the frame renders. Idempotent — `set_cull_lock` only
        // snapshots on the engage transition.
        if !rcx.main_camera.cull_lock()
            && std::env::var("ENGINE_CULL_AWAY").is_ok_and(|v| v == "1" || v == "true")
        {
            let away = view_proj * glam::Mat4::from_translation(glam::Vec3::splat(1.0e7));
            rcx.main_camera.set_cull_lock(true, away.to_cols_array());
            println!("[cull-away] frustum locked off-scene; pass 1 should draw nothing");
        }

        // Debug: F6 cycles the staging memory type auto → cached → vram.
        // A pinned mode takes effect on the next frame's rebuild check.
        if input::key_pressed(KeyCode::F6) {
            self.staging_balancer.cycle();
        }

        // Debug: F7 flips the per-frame host sync gate between the mid-CB
        // early wake and the full previous-frame retirement wait (see
        // `RenderApp::wait_on_frame`). Free to toggle live — both GPU
        // counters are signaled every frame either way.
        if input::key_pressed(KeyCode::F7) {
            self.wait_on_frame = !self.wait_on_frame;
            println!(
                "[wait mode] {}",
                if self.wait_on_frame {
                    "previous FRAME (in_flight fence — uncontended staging)"
                } else {
                    "previous COMPUTE (mid-CB early wake — pipelined staging)"
                }
            );
        }

        // Last consumer of this frame's edge-triggered input state (both
        // component `update`s, earlier, and the F8/F9 checks above have now
        // run) — clear it so it doesn't leak into next frame's reads.
        input::global_mut().end_frame();

        let entity_capacity = rcx.world_transforms.entity_capacity();
        let dirty_words = dirty_word_count(entity_capacity);

        // ADR-0003 compute-stage timeline wait.
        //
        // The staging triple, dirty bitmasks, view_proj, and the scatter
        // secondary that consumes them are all **shared** across in-flight
        // frames now. Before the host mutates any of them we host-wait
        // until the GPU has finished the *previous* frame's COMPUTE_SHADER
        // stage — which is when both `scatter` and `mvp_build` have read
        // their last byte from the shared resources, and when the in-CB
        // `vkCmdFillBuffer(0)` for the dirty buffers has fully landed.
        //
        // First call (next_compute_signal_value == 1) waits on value 0 —
        // the semaphore's pre-signaled initial value, so it returns
        // immediately. Steady state: this and the per-image fence wait in
        // `acquire(...)` are both near-zero when the GPU keeps up.
        // ADR-0003 compute-stage timeline wait. The shared scatter
        // secondary, dirty bitmasks, and staging triple are all read by
        // the **previous frame's FrameSlot primary CB** (scatter folded
        // in at front + dirty fill_buffer clears + view_proj copy). We
        // host-wait for that submission's `compute_timeline` signal
        // before overwriting any of the shared host-visible buffers.
        //
        // First call (next_compute_signal_value == 1) waits on value 0 —
        // the semaphore's pre-signaled initial value, returns immediately.
        // Steady state: this and the per-image fence wait in
        // `acquire(...)` are both near-zero when the GPU keeps up.
        // ADR-0003 (post GPU-write early-wake refactor) compute-stage
        // wait. Busy-polls a host-coherent counter that the GPU's
        // `signal_cs` dispatch (recorded mid-CB right after
        // scatter+fill+copy) atomically increments once per frame.
        // Returns the moment every host-shared buffer read is done —
        // even though mvp_build + render + blit are still running.
        // Replaces the previous timeline-semaphore wait, whose
        // `vkWaitSemaphores` syscall added ~30µs/frame at low N.
        //
        // F7 switches this to `SwapchainRenderer::wait_previous_frame` —
        // the previous submission's `in_flight` fence, i.e. an exact
        // end-of-frame gate. That gives up the pipelining above so the
        // host's staging writes and the scatter's reads get an idle GPU,
        // with no contention against a concurrently-rendering frame's
        // memory traffic. Strictly stronger than the mid-CB poll (same
        // submission, later point), so every host-write safety guarantee
        // documented on `host_wait_for_previous_compute` still holds; it
        // costs a `vkWaitForFences` syscall in exchange for exactness.
        let host_wait_start = Instant::now();
        // std::thread::sleep(Duration::from_micros(400)); // give the GPU a chance to signal before busy-polling
        if self.wait_on_frame {
            renderer.wait_previous_frame();
        } else {
            rcx.world_transforms.host_wait_for_previous_compute();
        }
        // std::thread::sleep(Duration::from_micros(1500));
        let host_wait_ns = host_wait_start.elapsed().as_nanos() as u64;
        self.fps.record_host_wait_compute(host_wait_ns);

        // Cheap-path draw-plan update: rewrite the indirect template bases in
        // place. Gated by the compute wait above so no in-flight `template →
        // args` reset copy is mid-read.
        if let Some(plan) = pending_cheap_plan.as_ref() {
            rcx.main_camera.write_template_bases(plan);
        }

        // Drain the per-component dirty bitmasks from the hierarchy into
        // the shared per-frame staging triple. The atomic
        // `swap(0, Relaxed)` makes any concurrent `set_position` /
        // `rotate_by` happening *after* this point on another thread
        // visible to the *next* frame instead of being lost.
        //
        // SAFETY for the host writes below: the timeline wait above
        // guarantees the GPU has finished the previous frame's scatter +
        // mvp_build dispatches AND the in-CB `fill_buffer(0)` on the
        // shared dirty buffers, so the host has exclusive access.
        let host_staging_start = Instant::now();
        {
            let world = &rcx.world_transforms;
            let staging_locks_start = Instant::now();
            let mut pos = world
                .staging_positions()
                .write()
                .expect("staging_positions.write");
            let mut rot = world
                .staging_rotations()
                .write()
                .expect("staging_rotations.write");
            let mut scl = world
                .staging_scales()
                .write()
                .expect("staging_scales.write");
            let mut dirty_pos = world
                .staging_dirty_pos()
                .write()
                .expect("staging_dirty_pos.write");
            let mut dirty_rot = world
                .staging_dirty_rot()
                .write()
                .expect("staging_dirty_rot.write");
            let mut dirty_scl = world
                .staging_dirty_scl()
                .write()
                .expect("staging_dirty_scl.write");
            // view_proj_buf is a single-mat4 staging slot, promoted by
            // `vkCmdCopyBuffer` inside the scatter primary into the
            // stable `sot_view_proj` that mvp_build reads. Same
            // staging→SoT pattern as TRS — gated by the same compute
            // timeline wait above.
            let mut vp = world.view_proj_buf().write().expect("view_proj_buf.write");
            self.fps
                .record_staging_locks(staging_locks_start.elapsed().as_nanos() as u64);

            // Lowest and highest dirty-bitmask word index touched this
            // frame, per component — `max == -1` means untouched (`min` is
            // meaningless in that case). Feeds
            // `write_prepass_dispatch_groups`, which sizes the GPU-side
            // word-compaction prepass's scan to exactly
            // `[min_word, max_word]` instead of `[0, max_word]` — a span,
            // not a compacted live count (dirty bits can be scattered
            // anywhere within it); the prepass is what turns that span into
            // the real scatter dispatch's exact dirty-word count (see
            // `WorldTransformGpu::write_prepass_dispatch_groups`).
            //
            // `min_*_word` inits to `i64::MAX` (not `-1`) so an untouched
            // task never contributes a spurious lower bound to the
            // `fetch_min` fold below — `-1` would sort below every real
            // word index and permanently pin the min at `-1`. This is the
            // mirror image of `max_*_word`'s `-1` init, which works
            // *because* `-1` sorts below every real index for a `fetch_max`.
            let max_pos_word = atomic::AtomicI64::new(-1);
            let max_rot_word = atomic::AtomicI64::new(-1);
            let max_scl_word = atomic::AtomicI64::new(-1);
            let min_pos_word = atomic::AtomicI64::new(i64::MAX);
            let min_rot_word = atomic::AtomicI64::new(i64::MAX);
            let min_scl_word = atomic::AtomicI64::new(i64::MAX);

            if let Some(scene) = self.root_scene.as_ref() {
                let staging_setup_start = Instant::now();
                let dirty = scene.transform_hierarchy.dirty();
                let pw = dirty.position_words();
                let rw = dirty.rotation_words();
                let sw = dirty.scale_words();
                let hier_words = pw.len().min(dirty_words);

                // Raw, lock-free SoA reads. The contract (see
                // `TransformHierarchy::positions_raw`) is that no
                // `TransformGuard` is mutating these arrays right now —
                // satisfied because the scene's per-frame `update` has
                // already returned and the renderer is the sole reader
                // until the next update fires.
                let positions = scene.transform_hierarchy.positions_raw();
                let rotations = scene.transform_hierarchy.rotations_raw();
                let scales = scene.transform_hierarchy.scales_raw();
                let n = positions.len().min(entity_capacity);

                // Multithreaded staging-write path.
                //
                // Split the per-component staging buffers into
                // bitmap-slab tasks along the dirty-bitmask axis.
                // Each task owns one slab — disjoint write regions in
                // the staging value buffers (`words_per_task * 32` entities)
                // and the dirty bitmask buffers (`words_per_task` words),
                // plus an exclusive atomic-swap of its dirty-mask words from
                // the hierarchy. No locks, no false sharing across slabs
                // because each chunk boundary is `words_per_task * 32 * 16`
                // bytes apart — always a multiple of a cache line.
                //
                // The host-visible buffers are HOST_RANDOM_ACCESS (cached),
                // not write-combined, so per-thread sparse / parallel writes
                // don't suffer the WC-flush penalty that single-threaded
                // sequential WC writes optimised for. Without this caching
                // mode the parallel walk would actually be slower than the
                // sequential one at high entity counts.
                //
                // Per-word: drain hierarchy bits via atomic swap (so any
                // concurrent set_position / rotate_by happening *after*
                // this point lands in the next frame), write the drained
                // word into the slot's GPU-visible dirty buffer, walk
                // only the set bits to upload TRS values.
                //
                // NOTE: we upload **local** TRS — matching the granularity
                // of `Dirty` bits. `mvp_build_cs` composes world TRS by
                // walking the per-slot Parents buffer upward each frame
                // (maintained by the streamed parent-scatter pass), so a
                // parent's movement propagates to its children without any
                // child re-upload. A level-ordered global composition pass
                // is the planned faster replacement for the per-slot walk.
                //
                // Share the bitmap slab geometry with `Scene::update` so
                // the static pool keeps the same transform-index ranges
                // on the same workers across sim → staging.
                let bitmap_tasks = parallel::bitmap_task_layout(hier_words);
                let words_per_task = bitmap_tasks.words_per_task;
                let entities_per_task = bitmap_tasks.entities_per_task();
                // NUMA splitting has been removed from TransformHierarchy (Phase 1
                // simplification). Always use the global task dispatcher.

                // Wrap raw mutable pointers in a Sync newtype so the
                // closure can be `Sync`. Each task indexes a disjoint
                // sub-range of every buffer (verified by the chunk
                // arithmetic below), so aliasing is sound.
                struct SyncMut<T>(*mut T);
                unsafe impl<T> Send for SyncMut<T> {}
                unsafe impl<T> Sync for SyncMut<T> {}
                let pos_ptr = SyncMut(pos.as_mut_ptr());
                let rot_ptr = SyncMut(rot.as_mut_ptr());
                let scl_ptr = SyncMut(scl.as_mut_ptr());
                let dpos_ptr = SyncMut(dirty_pos.as_mut_ptr());
                let drot_ptr = SyncMut(dirty_rot.as_mut_ptr());
                let dscl_ptr = SyncMut(dirty_scl.as_mut_ptr());
                let pos_len = pos.len();
                let rot_len = rot.len();
                let scl_len = scl.len();
                let dpos_len = dirty_pos.len();
                let drot_len = dirty_rot.len();
                let dscl_len = dirty_scl.len();

                self.fps
                    .record_staging_setup(staging_setup_start.elapsed().as_nanos() as u64);

                let staging_parallel_start = Instant::now();
                // Per-word body: drains one dirty-bitmap word and
                // copies up to 32 TRS entities. Used by both
                // dispatch flavours below. Returns which of the three
                // components this word touched (`dp`/`dr`/`ds` != 0) so the
                // caller can fold a per-task watermark without an atomic
                // op per word — see the `parallel_for` body below.
                let per_word = |word_idx: usize| -> (bool, bool, bool) {
                    let _ = (
                        &pos_ptr, &rot_ptr, &scl_ptr, &dpos_ptr, &drot_ptr, &dscl_ptr,
                    );
                    let dp = pw[word_idx].swap(0, atomic::Ordering::Relaxed);
                    let dr = rw[word_idx].swap(0, atomic::Ordering::Relaxed);
                    let ds = sw[word_idx].swap(0, atomic::Ordering::Relaxed);
                    if (dp | dr | ds) == 0 {
                        return (false, false, false);
                    }
                    let entity_base = word_idx * 32;
                    if dp != 0 {
                        debug_assert!(word_idx < dpos_len);
                        unsafe {
                            *dpos_ptr.0.add(word_idx) = dp;
                        }
                        let mut bits = dp;
                        while bits != 0 {
                            let bit = bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            let entity = entity_base + bit;
                            if entity >= n {
                                break;
                            }
                            let p = positions[entity];
                            let base = entity * 3;
                            debug_assert!(base + 2 < pos_len);
                            unsafe {
                                *pos_ptr.0.add(base) = p.x;
                                *pos_ptr.0.add(base + 1) = p.y;
                                *pos_ptr.0.add(base + 2) = p.z;
                            }
                        }
                    }
                    if dr != 0 {
                        debug_assert!(word_idx < drot_len);
                        unsafe {
                            *drot_ptr.0.add(word_idx) = dr;
                        }
                        let mut bits = dr;
                        while bits != 0 {
                            let bit = bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            let entity = entity_base + bit;
                            if entity >= n {
                                break;
                            }
                            let q = rotations[entity];
                            // Stage as a `pack_quat_half`-packed pair of
                            // u32s (bit-reinterpreted into the `f32` slots)
                            // — 4×f16 quaternion, two 4-byte words instead
                            // of 3 f32 Euler angles. `scatter_cs` unpacks
                            // via `unpackHalf2x16` before writing the SoT
                            // buffer.
                            let packed = transform_gpu::pack_quat_half(q);
                            let base = entity * 2;
                            debug_assert!(base + 1 < rot_len);
                            unsafe {
                                *rot_ptr.0.add(base) = f32::from_bits(packed[0]);
                                *rot_ptr.0.add(base + 1) = f32::from_bits(packed[1]);
                            }
                        }
                    }
                    if ds != 0 {
                        debug_assert!(word_idx < dscl_len);
                        unsafe {
                            *dscl_ptr.0.add(word_idx) = ds;
                        }
                        let mut bits = ds;
                        while bits != 0 {
                            let bit = bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            let entity = entity_base + bit;
                            if entity >= n {
                                break;
                            }
                            let s = scales[entity];
                            let base = entity * 3;
                            debug_assert!(base + 2 < scl_len);
                            unsafe {
                                *scl_ptr.0.add(base) = s.x;
                                *scl_ptr.0.add(base + 1) = s.y;
                                *scl_ptr.0.add(base + 2) = s.z;
                            }
                        }
                    }
                    (dp != 0, dr != 0, ds != 0)
                };

                {
                    let n_tasks = bitmap_tasks.n_tasks;
                    // Node-local pool when we have one — these writes land
                    // in the caches the scatter is about to snoop, so which
                    // socket runs them dominates the scatter's cost. Note
                    // this gives up the worker↔transform-range sharing with
                    // `Scene::update` that the `bitmap_task_layout` comment
                    // above describes, since the two pools have different
                    // widths.
                    // Short-circuit on the flag, so the second pool is
                    // never even *built* unless it's in use — its workers
                    // would otherwise sit spinning and evicting for nothing.
                    let dispatch =
                        |body: &(dyn Fn(std::ops::Range<usize>) + Sync + Send)| match self
                            .use_staging_pool
                            .then(staging_pool)
                            .flatten()
                        {
                            Some(pool) => pool.parallel_for(0..n_tasks, body),
                            None => parallel::global::parallel_for(0..n_tasks, body),
                        };
                    dispatch(&|task_range: std::ops::Range<usize>| {
                        // Local (non-atomic) watermark for every word this
                        // thread drains across its whole task range — word
                        // indices only increase within the range, so the
                        // last `true` seen for a component is its max and
                        // the first is its min. Folded into the shared
                        // atomics once at the end instead of once per word,
                        // which otherwise contended the atomics on every
                        // dirty word and roughly doubled this loop's wall
                        // time.
                        let mut local_max_pos: i64 = -1;
                        let mut local_max_rot: i64 = -1;
                        let mut local_max_scl: i64 = -1;
                        let mut local_min_pos: i64 = -1;
                        let mut local_min_rot: i64 = -1;
                        let mut local_min_scl: i64 = -1;
                        for task_idx in task_range {
                            let word_base = task_idx * words_per_task;
                            let word_end = (word_base + words_per_task).min(hier_words);
                            for word_idx in word_base..word_end {
                                let (touched_pos, touched_rot, touched_scl) = per_word(word_idx);
                                if touched_pos {
                                    local_max_pos = word_idx as i64;
                                    if local_min_pos < 0 {
                                        local_min_pos = word_idx as i64;
                                    }
                                }
                                if touched_rot {
                                    local_max_rot = word_idx as i64;
                                    if local_min_rot < 0 {
                                        local_min_rot = word_idx as i64;
                                    }
                                }
                                if touched_scl {
                                    local_max_scl = word_idx as i64;
                                    if local_min_scl < 0 {
                                        local_min_scl = word_idx as i64;
                                    }
                                }
                            }
                        }
                        if local_max_pos >= 0 {
                            max_pos_word.fetch_max(local_max_pos, atomic::Ordering::Relaxed);
                            min_pos_word.fetch_min(local_min_pos, atomic::Ordering::Relaxed);
                        }
                        if local_max_rot >= 0 {
                            max_rot_word.fetch_max(local_max_rot, atomic::Ordering::Relaxed);
                            min_rot_word.fetch_min(local_min_rot, atomic::Ordering::Relaxed);
                        }
                        if local_max_scl >= 0 {
                            max_scl_word.fetch_max(local_max_scl, atomic::Ordering::Relaxed);
                            min_scl_word.fetch_min(local_min_scl, atomic::Ordering::Relaxed);
                        }
                    });
                    let _ = entities_per_task;
                }
                self.fps
                    .record_staging_parallel(staging_parallel_start.elapsed().as_nanos() as u64);
            } else if !dirty_pos.is_empty() {
                // Legacy fallback: identity at slot 0 the first time this
                // slot runs. Set the dirty bit so the scatter copies
                // staging[0] → SoT[0]; subsequent frames see no further
                // change so this branch is effectively idempotent.
                pos[0..3].copy_from_slice(&[0.0, 0.0, 0.0]);
                let packed = transform_gpu::pack_quat_half(glam::Quat::IDENTITY);
                rot[0] = f32::from_bits(packed[0]);
                rot[1] = f32::from_bits(packed[1]);
                scl[0..3].copy_from_slice(&[1.0, 1.0, 1.0]);
                dirty_pos[0] = 1;
                dirty_rot[0] = 1;
                dirty_scl[0] = 1;
                max_pos_word.store(0, atomic::Ordering::Relaxed);
                max_rot_word.store(0, atomic::Ordering::Relaxed);
                max_scl_word.store(0, atomic::Ordering::Relaxed);
                min_pos_word.store(0, atomic::Ordering::Relaxed);
                min_rot_word.store(0, atomic::Ordering::Relaxed);
                min_scl_word.store(0, atomic::Ordering::Relaxed);
            }
            vp[0] = view_proj.to_cols_array();
            vp[1][0..3].copy_from_slice(camera_position.as_ref());
            // Cull-test VP staging (frustum-lock debug feature): mirrors
            // `vp[0]` above unless the lock is engaged, in which case it
            // stays frozen at the snapshot taken when the lock last turned
            // on. Same host-write gating as the writes above — see
            // `RenderCamera::write_cull_view_proj`.
            rcx.main_camera
                .write_cull_view_proj(view_proj.to_cols_array());

            // TRS scatter prepass dispatch args: convert this frame's
            // per-component `[min_word, max_word]` dirty-word watermarks
            // into the word-compaction prepass's `dispatch_indirect` group
            // counts + scan bounds. The real scatter dispatch's own args are
            // derived on the GPU from the prepass's output — see
            // `WorldTransformGpu::write_prepass_dispatch_groups`. Same
            // `gpu_signal` gate as every other staging write this frame.
            world.write_prepass_dispatch_groups([
                (
                    min_pos_word.load(atomic::Ordering::Relaxed),
                    max_pos_word.load(atomic::Ordering::Relaxed),
                ),
                (
                    min_rot_word.load(atomic::Ordering::Relaxed),
                    max_rot_word.load(atomic::Ordering::Relaxed),
                ),
                (
                    min_scl_word.load(atomic::Ordering::Relaxed),
                    max_scl_word.load(atomic::Ordering::Relaxed),
                ),
            ]);

            // Parent-update stream: write this frame's drained pairs +
            // live count (0 on quiet frames — retiring last frame's
            // records) into the staging the in-CB parent scatter reads.
            // Same `gpu_signal` gate as every write above, which is what
            // makes a re-parent + local-TRS rewrite land atomically in
            // the same frame.
            let staging_parents = Instant::now();
            world.write_parent_updates(&parent_updates);
            self.fps
                .staging_parents
                .record(staging_parents.elapsed().as_nanos() as u64);

            // Spawn stream: same count-in-buffer pattern for the
            // GPURenderers scatter — new renderers appear in the same
            // frame that uploads their transform.
            let staging_spawns = Instant::now();
            rcx.gpu_renderers.write_spawns(&spawns);
            self.fps
                .staging_renderers
                .record(staging_spawns.elapsed().as_nanos() as u64);
        }
        // UI staging. Same `gpu_signal` gate as everything above it — this
        // is the only point at which the UI's host-visible buffers may be
        // touched. A quiet frame still writes the (zero-workgroup) dispatch
        // args, which is what retires this slot's previous occupant.
        rcx.ui_gpu.write_staging(&mut ui::ui());
        if self.ui_trace {
            let words = rcx.ui_gpu.last_dirty_words();
            if words != 0 {
                println!(
                    "[ui] frame {} dirty_words={words} prims={}",
                    self.total_frames,
                    ui::ui().prim_count(),
                );
            }
        }

        let host_staging_ns = host_staging_start.elapsed().as_nanos() as u64;
        self.fps.record_host_staging(host_staging_ns);
        // Diagnostic: dirty-word span this frame, attributed to the slot
        // that received it. If the two slots' spans match but their
        // scatter times don't, the split is a memory-placement property
        // of the buffers, not a difference in how much work each frame does.
        {
            let (slot, words) = rcx.world_transforms.last_prepass_span_words();
            self.fps.prepass_words_by_slot[slot].record(words as u64);
        }

        // ── Submit + present ──────────────────────────────────────
        //
        // Single CB, single batch per `vkQueueSubmit2`. The FrameSlot
        // primary contains scatter + dirty fills + view_proj copy +
        // signal_cs + mvp_build + render + blit. The host's wait above
        // (`host_wait_for_previous_compute`) busy-polls
        // `gpu_signal[0]`, which the in-CB `signal_cs` dispatch
        // increments right after every read of host-shared staging is
        // done — no kernel sync, no extra batch, no timeline semaphore.
        // Pick the primary recorded against *this frame's* staging slot:
        // it executes that slot's scatter secondary and fills that slot's
        // dirty / view_proj buffers — the ones the host just wrote.
        let staging_slot = rcx.world_transforms.write_slot();
        let cb = rcx.frame_slots[frame_slot_index(image_index, staging_slot)]
            .command_buffer
            .clone();
        renderer.submit_and_present(frame, None, cb, Vec::new(), Vec::new());
        // Increment the expected `gpu_signal` value AFTER submit so the
        // next frame's host wait knows which value the GPU is bringing
        // the counter up to.
        rcx.world_transforms.inc_signal_expected();
        // Flip every host-staging producer to the other slot, in lockstep.
        // All three must advance together: `build_frame_slot` bakes a
        // single slot index into one primary that binds all of them, so a
        // drift would have that CB read one subsystem's fresh slot and
        // another's stale one. Kept adjacent to `inc_signal_expected`
        // because the `N-2` wait target is only correct while slot parity
        // and signal parity advance together.
        rcx.world_transforms.advance_staging_slot();
        rcx.gpu_renderers.advance_staging_slot();
        rcx.main_camera.advance_staging_slot();
        rcx.ui_gpu.advance_staging_slot();
        // CPU busy = this frame's handler span minus the two blocking waits
        // on the GPU (`acquire` on the image fence, `host_wait` on the
        // scatter's signal). Paired with the GPU total, it says which side
        // the frame is actually waiting on.
        self.staging_balancer.record_cpu(
            (now.elapsed().as_nanos() as u64)
                .saturating_sub(acquire_ns)
                .saturating_sub(host_wait_ns),
            host_staging_ns,
        );
        self.fps.tick(
            if self.wait_on_frame {
                "frame"
            } else {
                "compute"
            },
            self.staging_balancer.tag(),
        );
        self.total_frames += 1;
        // One-shot NUMA residency check after the harvest has had a
        // chance to fault every staging page in. Initial bind runs
        // before any writes touch the range, so its verify always
        // reports 0/0; this one reports the real state.
        if self.total_frames == 120 {
            rcx.world_transforms.report_staging_residency();
        }
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
fn build_draw_plan(mesh_store: &GpuMeshStore, slot_totals: &[u32]) -> DrawPlan {
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
fn build_all_frame_slots(
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    queue_family_index: u32,
    swapchain_views: &[Arc<ImageView>],
    main_camera: &RenderCamera,
    world_transforms: &WorldTransformGpu,
    gpu_renderers: &GpuRenderers,
    ui: &UiGpu,
) -> Vec<FrameSlot> {
    // Parallel build across swapchain images. Each task constructs one
    // FrameSlot independently. We pre-allocate the output `Vec` with
    // `MaybeUninit` slots and have each task `ptr::write` its slot —
    // there is no cross-task sharing of either the underlying allocators
    // or the per-slot state, so this is sound.
    use std::mem::MaybeUninit;
    // Two primaries per swapchain image — one per host-staging slot. A
    // primary bakes in which staging slot's scatter secondary it runs and
    // which slot's dirty / view_proj buffers it fills and copies, so it
    // cannot be reused across slots.
    //
    // The pair can't be collapsed by assuming slot parity tracks image
    // parity: `vkAcquireNextImageKHR` is under no obligation to hand back
    // images round-robin, and even where it does in practice, a skipped
    // frame (out-of-date / minimised, which returns before advancing the
    // staging slot) desynchronises the two permanently.
    //
    // Indexing is `image_index * STAGING_SLOTS + staging_slot`.
    let n_images = swapchain_views.len();
    let n = n_images * STAGING_SLOTS;
    let mut out: Vec<MaybeUninit<FrameSlot>> = (0..n).map(|_| MaybeUninit::uninit()).collect();

    struct SyncMut<T>(*mut T);
    unsafe impl<T> Send for SyncMut<T> {}
    unsafe impl<T> Sync for SyncMut<T> {}
    let out_ptr = SyncMut(out.as_mut_ptr());

    parallel::global::parallel_for(0..n, |task_range| {
        let _ = &out_ptr;
        for i in task_range {
            let slot = build_frame_slot(
                cb_allocator,
                memory_allocator,
                queue_family_index,
                &swapchain_views[i / STAGING_SLOTS],
                main_camera,
                world_transforms,
                gpu_renderers,
                ui,
                i % STAGING_SLOTS,
            );
            // SAFETY: each task writes a unique index in [0, n).
            unsafe {
                (*out_ptr.0.add(i)).write(slot);
            }
        }
    });

    // SAFETY: every index was initialised by the loop above.
    unsafe {
        let mut out = std::mem::ManuallyDrop::new(out);
        Vec::from_raw_parts(out.as_mut_ptr() as *mut FrameSlot, n, out.capacity())
    }
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
    ui: &UiGpu,
    staging_slot: usize,
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
    // Pipeline seam: latches when everything submitted *before* this frame
    // has retired. See `GPU_TS_COUNT`'s "Why q8 exists" — without it the
    // previous frame's raster tail is billed to this frame's scatter.
    unsafe { builder.write_timestamp(timestamp_pool.clone(), 8, PipelineStage::BottomOfPipe) }
        .expect("write_timestamp q8 (seam)");

    builder
        .execute_commands(world.scatter_secondary(staging_slot).clone())
        .expect("execute scatter_secondary");

    // Spawn-scatter: streamed (transform_id, mesh_id) pairs → GPURenderers.
    // Count-in-buffer like the parent scatter inside `scatter_secondary`;
    // recorded before `signal_cs` so the `gpu_signal` gate covers the host
    // write to its staging, and before the cull secondary which reads the
    // GPURenderers buffer it writes (vulkano auto-sync orders them).
    unsafe { builder.write_timestamp(timestamp_pool.clone(), 9, PipelineStage::BottomOfPipe) }
        .expect("write_timestamp q9 (trs scatter)");

    builder
        .execute_commands(gpu_renderers.spawn_scatter_secondary(staging_slot).clone())
        .expect("execute spawn_scatter_secondary");

    // UI scatter (ADR-0006): 4 prepasses -> dirty clears -> build-args ->
    // 4 scatters. Recorded **before** `signal_cs` because everything it
    // reads is host-visible staging, and the signal is the host's licence
    // to overwrite that staging for the next frame. Its own draw runs at
    // the very end of this CB, reading only device-local buffers this block
    // produced.
    builder
        .execute_commands(ui.scatter_secondary(staging_slot).clone())
        .expect("execute ui scatter_secondary");

    unsafe { builder.write_timestamp(timestamp_pool.clone(), 10, PipelineStage::BottomOfPipe) }
        .expect("write_timestamp q10 (spawn + ui scatter)");

    builder
        .fill_buffer(
            world
                .staging_dirty_pos_for(staging_slot)
                .clone()
                .reinterpret::<[u32]>(),
            0,
        )
        .expect("fill staging_dirty_pos")
        .fill_buffer(
            world
                .staging_dirty_rot_for(staging_slot)
                .clone()
                .reinterpret::<[u32]>(),
            0,
        )
        .expect("fill staging_dirty_rot")
        .fill_buffer(
            world
                .staging_dirty_scl_for(staging_slot)
                .clone()
                .reinterpret::<[u32]>(),
            0,
        )
        .expect("fill staging_dirty_scl");

    builder
        .copy_buffer(vulkano::command_buffer::CopyBufferInfo::buffers(
            world
                .view_proj_buf_for(staging_slot)
                .clone()
                .reinterpret::<[u8]>(),
            world.sot_view_proj().clone().reinterpret::<[u8]>(),
        ))
        .expect("copy staging_view_proj → sot_view_proj");

    // Cull-test VP promotion (frustum-lock debug feature). Unconditional —
    // runs regardless of `occlusion_enabled` below, since pass 1's frustum
    // test always reads `cull_view_proj`, and this is what keeps the lock
    // toggle cheap (no CB re-recording either way).
    builder
        .copy_buffer(vulkano::command_buffer::CopyBufferInfo::buffers(
            main_camera
                .cull_view_proj_staging_buf(staging_slot)
                .clone()
                .reinterpret::<[u8]>(),
            main_camera
                .cull_view_proj_buf()
                .clone()
                .reinterpret::<[u8]>(),
        ))
        .expect("copy cull_view_proj_staging → cull_view_proj");

    unsafe { builder.write_timestamp(timestamp_pool.clone(), 11, PipelineStage::BottomOfPipe) }
        .expect("write_timestamp q11 (dirty clears + VP promotions)");

    // Early-wake signal — atomically increments `gpu_signal[0]`. Recorded
    // **here**, after every read of host-shared staging is done
    // (scatter consumed staging+dirty, fill_buffer cleared dirty,
    // copy_buffer consumed view_proj_buf), and **before** mvp_build so
    // the rest of the CB doesn't gate the increment's visibility to the
    // host.
    //
    // What puts this dispatch *after* the scatter is NOT a resource
    // hazard — `gpu_signal` is bound by nothing else in the CB, so
    // auto-sync derives no barrier from it. It is vulkano's conservative
    // ALL_COMMANDS first-use barrier, which happens to land at the
    // `fill_buffer` above. That is load-bearing and fragile: read the
    // "What actually orders `signal_cs` after the scatter" section on
    // `WorldTransformGpu::host_wait_for_previous_compute` before moving
    // this dispatch or using `gpu_signal` anywhere else in the CB.
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
            unsafe {
                builder.write_timestamp(timestamp_pool.clone(), q, PipelineStage::BottomOfPipe)
            }
            .expect("write_timestamp q4-q6 (occlusion off)");
        }
    }

    builder
        .execute_commands(blit_secondary.clone())
        .expect("execute blit_secondary");

    // UI, straight into the swapchain image and therefore **after** the
    // blit that tonemaps and encodes the camera's HDR colour — the UI is
    // authored in sRGB and must not be tonemapped. `LoadOp::Load` keeps the
    // scene underneath; the swapchain format is `_SRGB`, so the hardware
    // blends in linear space and encodes on write. One `draw_indirect`,
    // whose instance count lives in a device buffer, so this scope never
    // needs re-recording when the UI's primitive count changes.
    builder
        .begin_rendering(RenderingInfo {
            contents: SubpassContents::SecondaryCommandBuffers,
            color_attachments: vec![Some(RenderingAttachmentInfo {
                load_op: AttachmentLoadOp::Load,
                store_op: AttachmentStoreOp::Store,
                ..RenderingAttachmentInfo::image_view(swapchain_view.clone())
            })],
            ..Default::default()
        })
        .expect("begin_rendering ui");

    builder
        .execute_commands(ui.draw_secondary().clone())
        .expect("execute ui draw_secondary");

    builder.end_rendering().expect("end_rendering ui");

    unsafe { builder.write_timestamp(timestamp_pool.clone(), 7, PipelineStage::BottomOfPipe) }
        .expect("write_timestamp q7 (blit + ui)");

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
