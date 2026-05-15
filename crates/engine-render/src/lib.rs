//! Vulkano-based renderer and windowing for the game engine.
//!
//! Public surface is [`Window`]. A typical setup:
//!
//! ```no_run
//! use engine_render::{Window, RenderInstance};
//! use engine_core::mesh::primitives;
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
//!
//! Window::new("My Game")
//!     .with_meshes(vec![primitives::cube()])
//!     .with_scene(root, vec![RenderInstance::new(0, e.id)])
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
    sync::{Arc, atomic},
    time::Instant,
};

use engine_core::{
    component::Scene,
    mesh::Mesh,
};
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferInheritanceInfo,
        CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderingAttachmentInfo, RenderingInfo, SecondaryAutoCommandBuffer,
        SubpassContents,
        allocator::{
            StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
        },
    },
    descriptor_set::{
        allocator::{
            StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
        },
    },
    device::{Device, DeviceFeatures, Queue},
    image::{ImageLayout, view::ImageView},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::{PipelineRenderingCreateInfo, PipelineSubpassType},
            vertex_input::VertexDefinition,
            viewport::ViewportState,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{PresentMode, SurfaceInfo},
};
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{WindowAttributes, WindowId},
};

use rayon::prelude::*;

mod camera;
mod gpu_mesh;
mod scene;
mod shaders;
mod swapchain;
mod transform_gpu;

use camera::{CAMERA_COLOR_FORMAT, CAMERA_DEPTH_FORMAT, CameraSceneResources, RenderCamera};
use gpu_mesh::{GpuMesh, GpuVertex};
use swapchain::SwapchainRenderer;
use transform_gpu::{WorldTransformGpu, dirty_word_count};

pub use scene::{Camera, OrbitController, RenderInstance};

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
struct FrameSlot {
    /// Pre-recorded secondary that contains the present-blit (camera's
    /// offscreen color → this slot's swapchain image). No render-pass
    /// inheritance.
    #[allow(dead_code)]
    blit_secondary:   Arc<SecondaryAutoCommandBuffer>,
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
    command_buffer:   Arc<PrimaryAutoCommandBuffer>,
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
    title:      String,
    meshes:     Vec<Mesh>,
    /// The window's root scene. Named `root_scene` to mirror the editor /
    /// game-side convention of calling the top-level scene `root`.
    root_scene: Option<Scene>,
    instances:  Vec<RenderInstance>,
}

impl Window {
    /// Create a window descriptor with the given title.
    pub fn new(title: &str) -> Self {
        Window {
            title:      title.to_owned(),
            meshes:     Vec::new(),
            root_scene: None,
            instances:  Vec::new(),
        }
    }

    /// Attach CPU meshes that will be uploaded to the GPU at startup.
    /// The order here defines the `mesh_index` used by [`RenderInstance`].
    pub fn with_meshes(mut self, meshes: Vec<Mesh>) -> Self {
        self.meshes = meshes;
        self
    }

    /// Attach the root [`Scene`] and the list of instances drawn each frame.
    ///
    /// The window takes ownership of the scene; per-frame `Component::update`
    /// hooks run on the event-loop thread immediately before the staging
    /// upload. Each [`RenderInstance::transform_index`] must point at an
    /// entity that was created via `scene.new_entity(...)`.
    pub fn with_scene(
        mut self,
        root_scene: Scene,
        instances: Vec<RenderInstance>,
    ) -> Self {
        self.root_scene = Some(root_scene);
        self.instances  = instances;
        self
    }

    /// Open the OS window, initialise Vulkan, and block on the event loop.
    pub fn run(self) {
        let event_loop = EventLoop::new().expect("Failed to create winit EventLoop");
        let mut app = RenderApp::new(
            self.title,
            self.meshes,
            self.root_scene,
            self.instances,
        );
        event_loop
            .run_app(&mut app)
            .expect("Event loop exited with an error");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FPS tracker
// ─────────────────────────────────────────────────────────────────────────────

// ── Frame stats: FPS + per-phase timings ──────────────────────────────

/// Cumulative `(min, max, sum_ns, count)` for a single phase across the
/// FPS sample window. Avg is `sum_ns / count`.
#[derive(Default, Clone, Copy)]
struct PhaseAcc {
    min_ns: u64,
    max_ns: u64,
    sum_ns: u128,
    count:  u64,
}

impl PhaseAcc {
    fn record(&mut self, ns: u64) {
        if self.count == 0 {
            self.min_ns = ns;
            self.max_ns = ns;
        } else {
            if ns < self.min_ns { self.min_ns = ns; }
            if ns > self.max_ns { self.max_ns = ns; }
        }
        self.sum_ns += ns as u128;
        self.count  += 1;
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
struct FrameStats {
    last_print:        Instant,
    frame_count:       u32,
    acquire:           PhaseAcc,
    host_wait_compute: PhaseAcc,
    host_staging:      PhaseAcc,
}

impl FrameStats {
    fn new() -> Self {
        Self {
            last_print:        Instant::now(),
            frame_count:       0,
            acquire:           PhaseAcc::default(),
            host_wait_compute: PhaseAcc::default(),
            host_staging:      PhaseAcc::default(),
        }
    }

    fn record_acquire(&mut self, ns: u64)           { self.acquire.record(ns); }
    fn record_host_wait_compute(&mut self, ns: u64) { self.host_wait_compute.record(ns); }
    fn record_host_staging(&mut self, ns: u64)      { self.host_staging.record(ns); }

    fn tick(&mut self) {
        self.frame_count += 1;
        if self.frame_count & (FRAMES_PER_FPS_SAMPLE - 1) == 0 {
            let elapsed = self.last_print.elapsed();
            if elapsed.as_secs() >= 1 {
                let fps = self.frame_count as f64 / elapsed.as_secs_f64();
                println!(
                    "FPS: {:.0}  ({:.3} ms/frame)  | us min/avg/max  acquire {} | host_wait_compute {} | host_staging {}",
                    fps,
                    1000.0 / fps,
                    self.acquire.fmt_us(),
                    self.host_wait_compute.fmt_us(),
                    self.host_staging.fmt_us(),
                );
                self.frame_count       = 0;
                self.last_print        = Instant::now();
                self.acquire           = PhaseAcc::default();
                self.host_wait_compute = PhaseAcc::default();
                self.host_staging      = PhaseAcc::default();
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RenderApp  (internal event-loop handler)
// ─────────────────────────────────────────────────────────────────────────────

/// All state that lives for the entire event-loop lifetime.
struct RenderApp {
    title:                       String,
    context:                     VulkanoContext,
    graphics_queue:              Arc<Queue>,
    swapchain_renderer:          Option<SwapchainRenderer>,
    command_buffer_allocator:    Arc<StandardCommandBufferAllocator>,
    memory_allocator:            Arc<StandardMemoryAllocator>,
    descriptor_set_allocator:    Arc<StandardDescriptorSetAllocator>,
    fps:                         FrameStats,
    /// CPU meshes kept around so they can be re-uploaded after a GPU reset.
    meshes:                      Vec<Mesh>,
    pipeline:                    Option<Arc<GraphicsPipeline>>,
    rcx:                         Option<RenderContext>,

    // ── Scene state ─────────────────────────────────────────────────
    /// The window's root scene — owns the transform hierarchy and the
    /// component registry. Mutated each frame via `Scene::update(dt)`.
    root_scene:                  Option<Scene>,
    instances:                   Vec<RenderInstance>,
    orbit:                       OrbitController,
    last_frame_time:             Option<Instant>,
}

/// Swapchain-image-count-sized arrays rebuilt on every swapchain recreation.
struct RenderContext {
    /// Cached swapchain image views. Used as **blit destinations** by each
    /// FrameSlot's pre-recorded CB; refreshed on resize.
    swapchain_image_views: Vec<Arc<ImageView>>,
    /// GPU mesh buffers — uploaded once; kept alive here for the lifetime of
    /// the renderer.
    gpu_meshes:        Vec<GpuMesh>,
    /// World-scoped GPU transform state: SoT (pos/rot/scale) buffers +
    /// scatter / mvp-build compute pipelines. Shared by every camera that
    /// targets this scene; sized to the transform hierarchy's entity
    /// count, grown geometrically on demand.
    world_transforms:  WorldTransformGpu,
    /// The render-side camera that drives the scene render. Owns its own
    /// offscreen color + depth attachments and a [`CameraResolution`] policy
    /// (currently always `MatchSwapchain`, so the present-blit stays 1:1).
    /// On a swapchain resize the camera decides whether to rebuild its
    /// attachments — future `Fixed` / `ScaleSwapchain` cameras will survive
    /// swapchain resizes untouched without changing the swapchain handler.
    main_camera:       RenderCamera,
    /// One `FrameSlot` per swapchain image. Each slot owns the per-frame
    /// staging matrix buffer, the blit secondary, and the composing primary
    /// CB that references `main_camera`'s device matrices + scene secondary
    /// and this slot's swapchain image as the blit destination.
    frame_slots:       Vec<FrameSlot>,
    /// Mesh indices, one per `RenderInstance`, baked into every camera's
    /// scene secondary at build time. Kept here so we can detect topology
    /// changes and rebuild on demand.
    draws_template:    Vec<u32>,
    /// Transform/entity indices, parallel to `draws_template` — one per
    /// `RenderInstance`. Uploaded into each camera's `instance_to_entity`
    /// buffer; read by the mvp-build compute shader to fetch each draw's
    /// TRS from the world's SoT buffers.
    entity_template:   Vec<u32>,
}

impl RenderApp {
    fn new(
        title:      String,
        meshes:     Vec<Mesh>,
        root_scene: Option<Scene>,
        instances:  Vec<RenderInstance>,
    ) -> Self {
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
                multi_draw_indirect:         true,
                draw_indirect_first_instance: true,
                // ADR-0003 (shared staging + timeline-semaphore sync):
                // We use a Vulkan timeline semaphore signaled at
                // `COMPUTE_SHADER` stage end of every submission to gate
                // host writes to the shared staging triple. Promoted to
                // core in Vulkan 1.2; still must be opted into via the
                // device features struct on devices that report 1.2+.
                timeline_semaphore:          true,
                ..Default::default()
            },
            ..Default::default()
        });

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            StandardCommandBufferAllocatorCreateInfo {
                primary_buffer_count:   32,
                // Two secondaries per FrameSlot (scene + blit); allocate enough
                // headroom for several swapchain images per pool reset.
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));

        let memory_allocator =
            Arc::new(StandardMemoryAllocator::new_default(context.device().clone()));

        let descriptor_set_allocator = Arc::new(
            StandardDescriptorSetAllocator::new(
                context.device().clone(),
                StandardDescriptorSetAllocatorCreateInfo::default(),
            ),
        );

        let graphics_queue = context.graphics_queue().clone();

        RenderApp {
            title,
            context,
            graphics_queue,
            swapchain_renderer: None,
            command_buffer_allocator,
            memory_allocator,
            descriptor_set_allocator,
            fps: FrameStats::new(),
            meshes,
            pipeline: None,
            rcx: None,
            root_scene,
            instances,
            orbit: OrbitController::new(),
            last_frame_time: None,
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
        self.pipeline = Some(pipeline.clone());
        // Swapchain format is informational here — the pipeline is built
        // against `CAMERA_COLOR_FORMAT`, and the present-blit handles
        // format conversion to whatever the swapchain offers.
        let _ = swapchain_format;

        // Upload CPU meshes → GPU buffers (once; reused across resizes).
        let gpu_meshes: Vec<GpuMesh> = self
            .meshes
            .iter()
            .map(|m| GpuMesh::upload(m, &self.memory_allocator))
            .collect();

        // Bake the static (mesh_index per draw, entity_index per draw)
        // topology. If `with_scene` wasn't called, fall back to drawing
        // every uploaded mesh once at the origin (legacy test-code
        // behaviour) — entity 0 is the implicit identity slot.
        let (draws_template, entity_template): (Vec<u32>, Vec<u32>) = if self.instances.is_empty() {
            (
                (0..gpu_meshes.len() as u32).collect(),
                vec![0u32; gpu_meshes.len()],
            )
        } else {
            (
                self.instances.iter().map(|i| i.mesh_index).collect(),
                self.instances.iter().map(|i| i.transform_index).collect(),
            )
        };

        // World transform state — sized to the hierarchy's current entity
        // count (or 1 for the legacy-fallback path so the SoT buffers are
        // never zero-sized).
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
            self.graphics_queue.queue_family_index(),
            initial_entity_count,
        );

        // The main camera matches the swapchain extent so the present-blit
        // stays a 1:1 copy. The first swapchain image gives us the extent.
        let initial_extent = {
            let [w, h, _] = attachment_image_views[0].image().extent();
            [w, h]
        };
        let scene_resources = CameraSceneResources {
            cb_allocator:             &self.command_buffer_allocator,
            descriptor_set_allocator: &self.descriptor_set_allocator,
            memory_allocator:         &self.memory_allocator,
            pipeline:                 &pipeline,
            queue_family_index:       self.graphics_queue.queue_family_index(),
            gpu_meshes:               &gpu_meshes,
            draws_template:           &draws_template,
            entity_template:          &entity_template,
            world_transforms:         &world_transforms,
        };
        let main_camera = RenderCamera::new_match_swapchain(
            initial_extent,
            &scene_resources,
        );

        let frame_slots = build_all_frame_slots(
                &self.command_buffer_allocator,
                &self.memory_allocator,
                self.graphics_queue.queue_family_index(),
                &attachment_image_views,
                &main_camera,
                &world_transforms,
            );

        self.rcx = Some(RenderContext {
            swapchain_image_views: attachment_image_views,
            gpu_meshes,
            world_transforms,
            main_camera,
            frame_slots,
            draws_template,
            entity_template,
        });
        self.swapchain_renderer = Some(swapchain_renderer);
        self.last_frame_time = Some(Instant::now());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id:  WindowId,
        event:       WindowEvent,
    ) {
        // Always feed the orbit controller first — it's harmless if the
        // renderer isn't ready yet.
        self.orbit.feed_window_event(&event);

        let renderer = match self.swapchain_renderer.as_mut() {
            Some(r) => r,
            None    => return,
        };
        match event {
            WindowEvent::CloseRequested  => event_loop.exit(),
            WindowEvent::Resized(_)      => renderer.resize(),
            WindowEvent::RedrawRequested => {}
            _ => {}
        }
    }

    /// Render one frame; runs at full speed (`ControlFlow::Poll`).
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);

        let renderer = match self.swapchain_renderer.as_mut() {
            Some(r) => r,
            None    => return,
        };
        let rcx = match self.rcx.as_mut() {
            Some(r) => r,
            None    => return,
        };

        // ── dt + per-frame update callback ──────────────────────────────────
        let now = Instant::now();
        let dt  = self.last_frame_time
            .map(|t| (now - t).as_secs_f32())
            .unwrap_or(0.0)
            .min(0.1); // clamp big stalls (e.g. window drag) to 100 ms
        self.last_frame_time = Some(now);

        if let Some(scene) = self.root_scene.as_mut() {
            // Drives every registered `Component::update(dt, &transform)` in
            // parallel. Mutations are recorded against the hierarchy's
            // dirty bitmasks and harvested below.
            scene.update(dt);
        }

        // Pre-clone everything the swapchain-recreation closure needs so it
        // doesn't capture `self`.
        let memory_allocator        = self.memory_allocator.clone();
        let cb_allocator            = self.command_buffer_allocator.clone();
        let descriptor_set_allocator = self.descriptor_set_allocator.clone();
        let pipeline_for_recreate   = self.pipeline.clone().expect("Pipeline not initialised");
        let queue_family_index      = self.graphics_queue.queue_family_index();

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
                cb_allocator:             &cb_allocator,
                descriptor_set_allocator: &descriptor_set_allocator,
                memory_allocator:         &memory_allocator,
                pipeline:                 &pipeline_for_recreate,
                queue_family_index,
                gpu_meshes:               &rcx.gpu_meshes,
                draws_template:           &rcx.draws_template,
                entity_template:          &rcx.entity_template,
                world_transforms:         &rcx.world_transforms,
            };
            let _camera_rebuilt = rcx.main_camera
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
            rcx.frame_slots.clear();
            rcx.frame_slots = build_all_frame_slots(
                &cb_allocator,
                &memory_allocator,
                queue_family_index,
                &rcx.swapchain_image_views,
                &rcx.main_camera,
                &rcx.world_transforms,
            );
        }) {
            Some(f) => f,
            None    => return, // out-of-date / minimised — skip frame
        };
        self.fps.record_acquire(acquire_start.elapsed().as_nanos() as u64);

        // ── World capacity check (per-world axis): the entity hierarchy
        //    may have grown past what the SoT buffers can hold. When this
        //    fires we re-allocate the SoT buffers, ask every camera to
        //    rebuild its mvp_build_set0 (which captured the old SoT
        //    handles), and rebuild every FrameSlot (whose staging buffers
        //    must match the new entity capacity and whose scatter sets
        //    captured the old SoT handles too). Geometric growth keeps
        //    this rare; it's a strict superset of the camera-capacity
        //    rebuild path below.
        let entity_count = self
            .root_scene
            .as_ref()
            .map(|s| s.transform_hierarchy.len())
            .unwrap_or(1)
            .max(1);
        let mut need_frame_slot_rebuild = false;
        if rcx.world_transforms.ensure_capacity(&self.memory_allocator, entity_count) {
            rcx.main_camera.on_world_capacity_change(
                &self.command_buffer_allocator,
                &self.descriptor_set_allocator,
                self.graphics_queue.queue_family_index(),
                &rcx.world_transforms,
            );
            need_frame_slot_rebuild = true;
            // The SoT was just re-allocated — its contents are undefined.
            // Re-mark every existing entity's TRS dirty so the next frame's
            // harvest re-uploads the full world into the new SoT. Without
            // this, a static scene that was already steady-state would see
            // an empty harvest after grow and never repopulate the SoT.
            if let Some(scene) = self.root_scene.as_ref() {
                scene.transform_hierarchy.dirty().mark_all_trs();
            }
        }

        // ── Camera capacity check: scene topology may have grown past what
        //    the camera's device matrix buffer can hold. Geometric growth
        //    keeps this rare. When it triggers we re-allocate the camera's
        //    device buffer + descriptor set + scene secondary AND every
        //    FrameSlot (whose primaries reference the new device buffer /
        //    scene secondary and whose mvp-build descriptor sets capture
        //    the new mvp output buffer).
        let needed_capacity = rcx.draws_template.len();
        if needed_capacity > rcx.main_camera.allocated_capacity()
            || needed_capacity != rcx.main_camera.draw_count()
        {
            let scene_resources = CameraSceneResources {
                cb_allocator:             &self.command_buffer_allocator,
                descriptor_set_allocator: &self.descriptor_set_allocator,
                memory_allocator:         &self.memory_allocator,
                pipeline:                 &self.pipeline.clone().expect("pipeline"),
                queue_family_index:       self.graphics_queue.queue_family_index(),
                gpu_meshes:               &rcx.gpu_meshes,
                draws_template:           &rcx.draws_template,
                entity_template:          &rcx.entity_template,
                world_transforms:         &rcx.world_transforms,
            };
            if rcx.main_camera.ensure_capacity(needed_capacity, &scene_resources) {
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
            );
        }

        // ── Sparse staging upload driven by `TransformHierarchy::Dirty` ─────
        let image_index = frame.image_index as usize;
        let [w, h, _]   = rcx.swapchain_image_views[image_index].image().extent();
        let aspect      = w as f32 / h.max(1) as f32;
        let view_proj   = self.orbit.camera().view_proj(aspect);

        let entity_capacity = rcx.world_transforms.entity_capacity();
        let dirty_words     = dirty_word_count(entity_capacity);

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
        let host_wait_start = Instant::now();
        rcx.world_transforms.host_wait_for_previous_compute();
        self.fps.record_host_wait_compute(host_wait_start.elapsed().as_nanos() as u64);

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
            let mut pos        = world.staging_positions().write().expect("staging_positions.write");
            let mut rot        = world.staging_rotations().write().expect("staging_rotations.write");
            let mut scl        = world.staging_scales().write().expect("staging_scales.write");
            let mut dirty_pos  = world.staging_dirty_pos().write().expect("staging_dirty_pos.write");
            let mut dirty_rot  = world.staging_dirty_rot().write().expect("staging_dirty_rot.write");
            let mut dirty_scl  = world.staging_dirty_scl().write().expect("staging_dirty_scl.write");
            // view_proj_buf is a single-mat4 staging slot, promoted by
            // `vkCmdCopyBuffer` inside the scatter primary into the
            // stable `sot_view_proj` that mvp_build reads. Same
            // staging→SoT pattern as TRS — gated by the same compute
            // timeline wait above.
            let mut vp         = world.view_proj_buf().write().expect("view_proj_buf.write");

            if let Some(scene) = self.root_scene.as_ref() {
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
                let scales    = scene.transform_hierarchy.scales_raw();
                let n         = positions.len().min(entity_capacity);

                // Multithreaded staging-write path.
                //
                // Split the per-component staging buffers into
                // `WORDS_PER_CHUNK`-word slabs along the dirty-bitmask axis.
                // Each rayon task owns one slab — disjoint write regions in
                // the staging value buffers (`WORDS_PER_CHUNK * 32` entities)
                // and the dirty bitmask buffers (`WORDS_PER_CHUNK` words),
                // plus an exclusive atomic-swap of its dirty-mask words from
                // the hierarchy. No locks, no false sharing across slabs
                // because each chunk boundary is `WORDS_PER_CHUNK * 32 * 16`
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
                // NOTE: we currently upload **local** TRS — `mvp_build_cs`
                // composes the model matrix from these directly without
                // walking the parent chain. This matches the granularity
                // of `Dirty` bits. Multi-level hierarchies will need a
                // GPU-side global composition pass; see todo.txt.
                //
                // Parallel staging-write scheduling — small fixed slabs
                // with `.with_min_len()` on every zipped iterator so rayon
                // **coalesces** small slabs into larger work units when no
                // stealing is happening (matches the steady-state hot path)
                // but can **split down** when one worker is slow and others
                // need work (matches a contended steal scenario).
                //
                // - `WORDS_PER_CHUNK = 8` → 256 entities per slab. The
                //   smallest unit rayon can split a task into when it
                //   needs to steal work.
                //
                // - `MIN_CHUNKS_PER_TASK = 8` → each task processes ≥ 64
                //   words = 2048 entities under no contention. Sweet
                //   spot of the 1–64 sweep: enough tasks (~488 at N=1M,
                //   ~7.6× worker count) to mask a stalled worker via
                //   steal, few enough to keep per-task dispatch noise
                //   below the actual write cost.
                //
                // Sizes at the regimes we care about:
                //
                //   N=100    →    4 words     →    1 task   (no parallelism, fine)
                //   N=1K     →   32 words    →    4 tasks
                //   N=10K    →  313 words   →   ~40 tasks
                //   N=100K   → 3125 words   →  ~390 tasks
                //   N=1M     → 31250 words  → ~488 tasks at min (~3900 slabs total available for steal)
                //
                // Re-tune `MIN_CHUNKS_PER_TASK` if profiling shows the
                // staging walk is CPU-time-variable (search for
                // `host_staging` in the FrameStats print for avg/min/max
                // us). Bigger min = less dispatch overhead but worse
                // load-balance under contention; smaller min = more
                // overhead but better contention recovery.
                //
                // Sweep history (for the next person who profiles this):
                //
                //   `MIN_CHUNKS_PER_TASK = 1`   → worse perf + worse variance (dispatch dominates)
                //   `MIN_CHUNKS_PER_TASK = 8`   → ← here (sweet spot)
                //   `MIN_CHUNKS_PER_TASK = 32`  → mild variance regression
                //   `MIN_CHUNKS_PER_TASK = 64`  → large variance regression (270↔190 at N=1M)
                //
                // The remaining run-to-run variance (~10-15% at N=1M) is
                // CPU power-state / boost-equilibrium dependent (the
                // first ~100 frames after launch determine which thermal
                // regime the CPU settles into), not addressable from
                // user-space without core-affinity pinning.
                use rayon::iter::IndexedParallelIterator;
                const WORDS_PER_CHUNK:     usize = 8;                       //  8 words = 256 entities per slab (steal granularity)
                const ENTITIES_PER_CHUNK:  usize = WORDS_PER_CHUNK * 32;
                const MIN_CHUNKS_PER_TASK: usize = 32;                       // ≥ 64 words / 2048 entities per task under no contention

                pos.par_chunks_mut(ENTITIES_PER_CHUNK).with_min_len(MIN_CHUNKS_PER_TASK)
                    .zip(rot.par_chunks_mut(ENTITIES_PER_CHUNK).with_min_len(MIN_CHUNKS_PER_TASK))
                    .zip(scl.par_chunks_mut(ENTITIES_PER_CHUNK).with_min_len(MIN_CHUNKS_PER_TASK))
                    .zip(dirty_pos.par_chunks_mut(WORDS_PER_CHUNK).with_min_len(MIN_CHUNKS_PER_TASK))
                    .zip(dirty_rot.par_chunks_mut(WORDS_PER_CHUNK).with_min_len(MIN_CHUNKS_PER_TASK))
                    .zip(dirty_scl.par_chunks_mut(WORDS_PER_CHUNK).with_min_len(MIN_CHUNKS_PER_TASK))
                    .enumerate()
                    .for_each(|(chunk_idx, (((((p_chunk, r_chunk), s_chunk), dp_chunk), dr_chunk), ds_chunk))| {
                        let word_base   = chunk_idx * WORDS_PER_CHUNK;
                        let entity_base = chunk_idx * ENTITIES_PER_CHUNK;
                        // The trailing chunk may be shorter than
                        // WORDS_PER_CHUNK; iterate over what we actually got.
                        for local_word in 0..dp_chunk.len() {
                            let word_idx = word_base + local_word;
                            if word_idx >= hier_words {
                                break;
                            }
                            let dp = pw[word_idx].swap(0, atomic::Ordering::Relaxed);
                            let dr = rw[word_idx].swap(0, atomic::Ordering::Relaxed);
                            let ds = sw[word_idx].swap(0, atomic::Ordering::Relaxed);
                            if (dp | dr | ds) == 0 {
                                continue;
                            }
                            // Local entity offset relative to this chunk.
                            let local_entity_base = local_word * 32;
                            if dp != 0 {
                                dp_chunk[local_word] = dp;
                                let mut bits = dp;
                                while bits != 0 {
                                    let bit = bits.trailing_zeros() as usize;
                                    bits &= bits - 1;
                                    let entity = entity_base + local_entity_base + bit;
                                    if entity >= n {
                                        break;
                                    }
                                    let p = positions[entity];
                                    p_chunk[local_entity_base + bit] = [p.x, p.y, p.z, 0.0];
                                }
                            }
                            if dr != 0 {
                                dr_chunk[local_word] = dr;
                                let mut bits = dr;
                                while bits != 0 {
                                    let bit = bits.trailing_zeros() as usize;
                                    bits &= bits - 1;
                                    let entity = entity_base + local_entity_base + bit;
                                    if entity >= n {
                                        break;
                                    }
                                    let q = rotations[entity];
                                    r_chunk[local_entity_base + bit] = [q.x, q.y, q.z, q.w];
                                }
                            }
                            if ds != 0 {
                                ds_chunk[local_word] = ds;
                                let mut bits = ds;
                                while bits != 0 {
                                    let bit = bits.trailing_zeros() as usize;
                                    bits &= bits - 1;
                                    let entity = entity_base + local_entity_base + bit;
                                    if entity >= n {
                                        break;
                                    }
                                    let s = scales[entity];
                                    s_chunk[local_entity_base + bit] = [s.x, s.y, s.z, 0.0];
                                }
                            }
                        }
                    });
            } else if !dirty_pos.is_empty() {
                // Legacy fallback: identity at slot 0 the first time this
                // slot runs. Set the dirty bit so the scatter copies
                // staging[0] → SoT[0]; subsequent frames see no further
                // change so this branch is effectively idempotent.
                pos[0] = [0.0, 0.0, 0.0, 0.0];
                rot[0] = [0.0, 0.0, 0.0, 1.0];
                scl[0] = [1.0, 1.0, 1.0, 0.0];
                dirty_pos[0] = 1;
                dirty_rot[0] = 1;
                dirty_scl[0] = 1;
            }

            vp[0] = view_proj.to_cols_array();
        }
        self.fps.record_host_staging(host_staging_start.elapsed().as_nanos() as u64);

        // ── Submit + present ──────────────────────────────────────
        //
        // Single CB, single batch per `vkQueueSubmit2`. The FrameSlot
        // primary contains scatter + dirty fills + view_proj copy +
        // signal_cs + mvp_build + render + blit. The host's wait above
        // (`host_wait_for_previous_compute`) busy-polls
        // `gpu_signal[0]`, which the in-CB `signal_cs` dispatch
        // increments right after every read of host-shared staging is
        // done — no kernel sync, no extra batch, no timeline semaphore.
        let cb = rcx.frame_slots[image_index].command_buffer.clone();
        renderer.submit_and_present(frame, None, cb, Vec::new(), Vec::new());
        // Increment the expected `gpu_signal` value AFTER submit so the
        // next frame's host wait knows which value the GPU is bringing
        // the counter up to.
        rcx.world_transforms.inc_signal_expected();
        self.fps.tick();
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
                    depth_attachment_format:  Some(CAMERA_DEPTH_FORMAT),
                    ..Default::default()
                },
            )),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .expect("Failed to create graphics pipeline")
}

/// Build (or rebuild) a `FrameSlot` for every swapchain image. Slots are
/// independent of each other and could be built in parallel; we keep the
/// loop sequential to avoid contention on the descriptor-set / CB allocators
/// (which are not particularly fast under contention).
fn build_all_frame_slots(
    cb_allocator:             &Arc<StandardCommandBufferAllocator>,
    memory_allocator:         &Arc<StandardMemoryAllocator>,
    queue_family_index:       u32,
    swapchain_views:          &[Arc<ImageView>],
    main_camera:              &RenderCamera,
    world_transforms:         &WorldTransformGpu,
) -> Vec<FrameSlot> {
    swapchain_views.par_iter().map(|swapchain_view| {
        build_frame_slot(
            cb_allocator,
            memory_allocator,
            queue_family_index,
            swapchain_view,
            main_camera,
            world_transforms,
        )
    }).collect()
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
    cb_allocator:             &Arc<StandardCommandBufferAllocator>,
    _memory_allocator:        &Arc<StandardMemoryAllocator>,
    queue_family_index:       u32,
    swapchain_view:           &Arc<ImageView>,
    main_camera:              &RenderCamera,
    world:                    &WorldTransformGpu,
) -> FrameSlot {
    let swapchain_image = swapchain_view.image().clone();

    // Camera-owned offscreen attachments. The dynamic-rendering scope below
    // targets these (NOT the swapchain image); the present-blit downstream
    // copies camera-extent → swapchain-extent. They happen to coincide today
    // because the main camera uses `CameraResolution::MatchSwapchain`.
    let color_image = main_camera.color_image().clone();
    let color_view  = main_camera.color_view().clone();
    let depth_view  = main_camera.depth_view().clone();

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
    ).expect("blit secondary builder");

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
    //   camera.mvp_build_secondary  — reads stable SoT, writes MVP.
    //     ↓  vulkano auto-sync: SHADER_WRITE → SHADER_READ on device_matrices
    //   begin_rendering(camera attachments)
    //     camera.scene_secondary  — vertex shader reads device_matrices.
    //   end_rendering
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
    ).expect("primary CB builder");

    builder
        .execute_commands(world.scatter_secondary().clone())
        .expect("execute scatter_secondary");

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

    builder
        .execute_commands(main_camera.mvp_build_secondary().clone())
        .expect("execute mvp_build");

    builder
        .begin_rendering(RenderingInfo {
            contents: SubpassContents::SecondaryCommandBuffers,
            color_attachments: vec![Some(RenderingAttachmentInfo {
                load_op:     AttachmentLoadOp::Clear,
                store_op:    AttachmentStoreOp::Store,
                clear_value: Some([0.08, 0.08, 0.10, 1.0].into()),
                ..RenderingAttachmentInfo::image_view(color_view.clone())
            })],
            depth_attachment: Some(RenderingAttachmentInfo {
                image_layout: ImageLayout::DepthStencilAttachmentOptimal,
                load_op:      AttachmentLoadOp::Clear,
                store_op:     AttachmentStoreOp::DontCare,
                clear_value:  Some(1.0_f32.into()),
                ..RenderingAttachmentInfo::image_view(depth_view.clone())
            }),
            ..Default::default()
        }).expect("begin_rendering");

    builder
        .execute_commands(main_camera.scene_secondary().clone())
        .expect("execute scene_secondary");

    builder.end_rendering().expect("end_rendering");

    builder
        .execute_commands(blit_secondary.clone())
        .expect("execute blit_secondary");

    let command_buffer = builder.build().expect("build primary CB");

    FrameSlot {
        blit_secondary,
        command_buffer,
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
        if i >= entity_count { break; }
        f(i);
    }
}
