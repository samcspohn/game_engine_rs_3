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
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
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
        DescriptorSet, WriteDescriptorSet,
        allocator::{
            StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
        },
    },
    device::{Device, DeviceFeatures, Queue},
    image::{ImageLayout, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
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
use transform_gpu::{ComponentSlot, WorldTransformGpu, dirty_word_count};

pub use scene::{Camera, OrbitController, RenderInstance};

// Trait imports needed for method resolution on GPU types.
use vulkano::pipeline::graphics::vertex_input::Vertex as VulkanoVertex;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Triple-buffer depth: CPU can record frame N+1/N+2 while GPU renders N.
const MAX_FRAMES_IN_FLIGHT: usize = 4;

/// Sample the system clock only every N frames (must be a power of two).
const FRAMES_PER_FPS_SAMPLE: u32 = 1024;

// ─────────────────────────────────────────────────────────────────────
// Per-image frame slot
// ─────────────────────────────────────────────────────────────────────

/// Resources tied to a single swapchain image and a single in-flight frame.
/// Built once per swapchain image; rebuilt when the swapchain image changes,
/// when the camera grows / changes extent, or when world entity capacity
/// or scene topology changes.
///
/// What lives here vs. on the camera vs. on the world reflects the three
/// orthogonal invalidation axes:
///
/// - **Per-frame-in-flight (here)** — host-mapped staging buffers
///   (positions / rotations / scales / dirty bitmask + view_proj) plus the
///   compute secondaries that consume them. Sized to match
///   `WorldTransformGpu::entity_capacity()` (staging) and
///   `RenderCamera::draw_count()` (mvp build).
/// - **Per-camera** — attachments, MVP buffer, scene secondary, mvp_build_set0.
/// - **Per-world** — SoT buffers + compute pipelines + scatter set layout.
/// - **Per-swapchain-image** — the blit secondary (its destination is this
///   slot's swapchain image) and the composing primary.
struct FrameSlot {
    // ── Per-frame-in-flight host-visible staging ───────────────────
    /// Host-staged position values (`vec4` per entity slot, `.w` unused).
    /// Sized to `world.entity_capacity()`. Written by the CPU each frame;
    /// consumed by `scatter_secondary` (read together with `staging_dirty`).
    staging_positions: Subbuffer<[ComponentSlot]>,
    /// Host-staged rotation values (quaternion `(x, y, z, w)` per slot).
    staging_rotations: Subbuffer<[ComponentSlot]>,
    /// Host-staged scale values (`vec4` per slot, `.w` unused).
    staging_scales:    Subbuffer<[ComponentSlot]>,
    /// Per-entity-slot dirty bitmask, **per component**. `bit i` set means
    /// the corresponding component of slot `i` is scattered into the SoT
    /// buffer this frame; clear means "SoT already holds the right value".
    /// Sized to `dirty_word_count(entity_capacity)` `u32`s.
    ///
    /// Per-component masks (rather than one shared OR mask) let the
    /// scatter compute skip whole components when only one of the three
    /// changed for a given slot — e.g. a pure-rotation frame writes no
    /// position or scale data on either CPU or GPU side.
    ///
    /// **Lifecycle:** the buffer is zeroed once at slot construction and
    /// thereafter cleared by a `vkCmdFillBuffer(0)` recorded inside the
    /// primary CB immediately after the scatter consumes it. By the time
    /// the per-image fence releases this slot back to the CPU, the GPU's
    /// clear has completed, so each frame the host only writes the words
    /// for entities that were dirtied *this* frame — no fan-out across
    /// other in-flight slots is required (the SoT is shared, so any one
    /// slot's scatter updates the value for every subsequent slot's
    /// mvp-build read).
    staging_dirty_pos: Subbuffer<[u32]>,
    staging_dirty_rot: Subbuffer<[u32]>,
    staging_dirty_scl: Subbuffer<[u32]>,

    /// Host-mapped single-mat4 storage buffer carrying this frame's
    /// `view_proj` for the mvp-build compute. Each frame the host writes
    /// the camera's current `view_proj`; the pre-recorded
    /// `mvp_build_secondary` reads it via `mvp_build_set1`.
    view_proj_buf:     Subbuffer<[[f32; 16]]>,

    // ── Per-frame compute descriptor sets ─────────────────────────
    /// Scatter set 0 for the position component: (dirty, staging_pos, sot_pos).
    /// Captured by buffer handle, so re-allocated whenever staging or SoT
    /// is re-allocated (i.e. world capacity grows).
    #[allow(dead_code)]
    scatter_set_pos:   Arc<DescriptorSet>,
    /// Scatter set 0 for the rotation component.
    #[allow(dead_code)]
    scatter_set_rot:   Arc<DescriptorSet>,
    /// Scatter set 0 for the scale component.
    #[allow(dead_code)]
    scatter_set_scl:   Arc<DescriptorSet>,
    /// MVP-build set 1 — binds `view_proj_buf`.
    #[allow(dead_code)]
    mvp_build_set1:    Arc<DescriptorSet>,

    // ── Pre-recorded secondary command buffers ─────────────────────
    /// Compute secondary: three scatter dispatches (pos, rot, scale), one
    /// after the other. No render-pass inheritance.
    #[allow(dead_code)]
    scatter_secondary: Arc<SecondaryAutoCommandBuffer>,
    /// Compute secondary: bind mvp-build pipeline + sets [0, 1], dispatch
    /// over `draw_count`. No render-pass inheritance.
    #[allow(dead_code)]
    mvp_build_secondary: Arc<SecondaryAutoCommandBuffer>,
    /// Pre-recorded secondary that contains the present-blit (camera's
    /// offscreen color → this slot's swapchain image). No render-pass
    /// inheritance.
    #[allow(dead_code)]
    blit_secondary:   Arc<SecondaryAutoCommandBuffer>,
    /// Pre-recorded **primary** that stitches everything together:
    /// `execute(scatter_secondary)`, `execute(mvp_build_secondary)`,
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


// ─────────────────────────────────────────────────────────────────────────────
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

struct FpsTracker {
    last_print:  Instant,
    frame_count: u32,
}

impl FpsTracker {
    fn new() -> Self {
        Self { last_print: Instant::now(), frame_count: 0 }
    }

    fn tick(&mut self) {
        self.frame_count += 1;
        if self.frame_count & (FRAMES_PER_FPS_SAMPLE - 1) == 0 {
            let elapsed = self.last_print.elapsed();
            if elapsed.as_secs() >= 1 {
                let fps = self.frame_count as f64 / elapsed.as_secs_f64();
                println!("FPS: {:.0}  ({:.3} ms/frame)", fps, 1000.0 / fps);
                self.frame_count = 0;
                self.last_print   = Instant::now();
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
    fps:                         FpsTracker,
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
            fps: FpsTracker::new(),
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
            &self.descriptor_set_allocator,
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
            rcx.frame_slots = build_all_frame_slots(
                &cb_allocator,
                &descriptor_set_allocator,
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
                &self.descriptor_set_allocator,
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
            rcx.frame_slots = build_all_frame_slots(
                &self.command_buffer_allocator,
                &self.descriptor_set_allocator,
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

        // Drain the per-component dirty bitmasks from the hierarchy into a
        // single per-frame harvest. The atomic `swap(0, Relaxed)` makes any
        // concurrent `set_position` / `rotate_by` happening *after* this
        // point on another thread visible to the *next* frame instead of
        // being lost.
        //
        // Unlike the previous design we do **not** fan the harvest out to
        // every in-flight slot's pending mask. The SoT is shared across
        // slots, so once *any* slot's scatter writes `SoT[i] = staging[i]`,
        // every subsequent slot's mvp-build reads the up-to-date value
        // — stale per-slot staging entries for unset bits are never read.
        // The slot's `staging_dirty_*` buffer was zeroed by the GPU after
        // the previous scatter consumed it (see `build_frame_slot`), so we
        // can write the harvest directly into the current frame's slot and
        // be done.
        //
        // SAFETY for the host writes below: `acquire(...)` waited on the
        // per-image fence, so the GPU has finished with this slot's
        // host-visible buffers (including the in-CB `fill_buffer(0)`).
        let slot = &mut rcx.frame_slots[image_index];
        {
            let mut pos        = slot.staging_positions.write().expect("staging_positions.write");
            let mut rot        = slot.staging_rotations.write().expect("staging_rotations.write");
            let mut scl        = slot.staging_scales.write().expect("staging_scales.write");
            let mut dirty_pos  = slot.staging_dirty_pos.write().expect("staging_dirty_pos.write");
            let mut dirty_rot  = slot.staging_dirty_rot.write().expect("staging_dirty_rot.write");
            let mut dirty_scl  = slot.staging_dirty_scl.write().expect("staging_dirty_scl.write");
            let mut vp         = slot.view_proj_buf.write().expect("view_proj_buf.write");

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

                // Per-word: drain the hierarchy bit, write that word into
                // the slot's GPU-visible dirty buffer, and walk only the
                // set bits to upload TRS values. Skipping zero words is
                // the big win on idle frames — a static scene with N=10000
                // entities does ~313 atomic loads and zero memory writes
                // (vs. the old design's ~3756 host writes for the
                // 4-slots-in-flight fan-out).
                //
                // NOTE: we currently upload **local** TRS — `mvp_build_cs`
                // composes the model matrix from these directly without
                // walking the parent chain. This matches the granularity
                // of `Dirty` bits (which fire on local-component writes
                // only). Hierarchies more than one level deep need a GPU-
                // side global composition pass; see todo.txt.
                for word_idx in 0..hier_words {
                    let dp = pw[word_idx].swap(0, atomic::Ordering::Relaxed);
                    let dr = rw[word_idx].swap(0, atomic::Ordering::Relaxed);
                    let ds = sw[word_idx].swap(0, atomic::Ordering::Relaxed);
                    if (dp | dr | ds) == 0 {
                        continue;
                    }
                    if dp != 0 {
                        dirty_pos[word_idx] = dp;
                        walk_bits(dp, word_idx, n, |i| {
                            let p = positions[i];
                            pos[i] = [p.x, p.y, p.z, 0.0];
                        });
                    }
                    if dr != 0 {
                        dirty_rot[word_idx] = dr;
                        walk_bits(dr, word_idx, n, |i| {
                            let q = rotations[i];
                            rot[i] = [q.x, q.y, q.z, q.w];
                        });
                    }
                    if ds != 0 {
                        dirty_scl[word_idx] = ds;
                        walk_bits(ds, word_idx, n, |i| {
                            let s = scales[i];
                            scl[i] = [s.x, s.y, s.z, 0.0];
                        });
                    }
                }
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

        // ── Submit the pre-recorded reusable CB ─────────────────────────
        let cb = slot.command_buffer.clone();
        renderer.submit_and_present(frame, cb);
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
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    memory_allocator:         &Arc<StandardMemoryAllocator>,
    queue_family_index:       u32,
    swapchain_views:          &[Arc<ImageView>],
    main_camera:              &RenderCamera,
    world_transforms:         &WorldTransformGpu,
) -> Vec<FrameSlot> {
    swapchain_views.par_iter().map(|swapchain_view| {
        build_frame_slot(
            cb_allocator,
            descriptor_set_allocator,
            memory_allocator,
            queue_family_index,
            swapchain_view,
            main_camera,
            world_transforms,
        )
    }).collect()
}

/// Build one `FrameSlot`: allocate the per-frame host-visible staging
/// buffers (positions / rotations / scales / dirty mask + view_proj),
/// allocate the per-frame compute descriptor sets that bind them,
/// pre-record the scatter + mvp-build + blit secondaries, and stitch them
/// together inside one composing primary CB.
fn build_frame_slot(
    cb_allocator:             &Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    memory_allocator:         &Arc<StandardMemoryAllocator>,
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

    let entity_capacity = world.entity_capacity();
    let draw_count      = main_camera.draw_count();

    // ── Per-frame host-visible staging buffers ────────────
    let staging_positions = make_host_storage_slice::<ComponentSlot>(memory_allocator, entity_capacity, BufferUsage::empty());
    let staging_rotations = make_host_storage_slice::<ComponentSlot>(memory_allocator, entity_capacity, BufferUsage::empty());
    let staging_scales    = make_host_storage_slice::<ComponentSlot>(memory_allocator, entity_capacity, BufferUsage::empty());
    let dirty_words       = dirty_word_count(entity_capacity);
    // Dirty buffers add `TRANSFER_DST` so the GPU can `vkCmdFillBuffer(0)`
    // them after the scatter compute consumes them (see primary CB below).
    // We tried clearing the bits in the scatter shader itself — it's a
    // tempting simplification — but writing to host-visible memory from
    // compute goes over PCIe on a discrete GPU and produced a ~16× FPS
    // regression in practice. `vkCmdFillBuffer` uses the dedicated
    // transfer engine and is the correct tool for clearing host-visible
    // buffers.
    let staging_dirty_pos = make_host_storage_slice::<u32>(memory_allocator, dirty_words, BufferUsage::TRANSFER_DST);
    let staging_dirty_rot = make_host_storage_slice::<u32>(memory_allocator, dirty_words, BufferUsage::TRANSFER_DST);
    let staging_dirty_scl = make_host_storage_slice::<u32>(memory_allocator, dirty_words, BufferUsage::TRANSFER_DST);
    // Single-mat4 storage buffer for the per-frame view_proj.
    let view_proj_buf     = make_host_storage_slice::<[f32; 16]>(memory_allocator, 1, BufferUsage::empty());

    // One-time CPU zero-init of the three dirty buffers. `Buffer::new_slice`
    // leaves contents undefined; the very first scatter dispatch reads
    // these words before any GPU clear has run, so we must guarantee
    // they're zero up front. Subsequent frames rely on the in-CB
    // `fill_buffer` recorded below to keep them zero between scatter
    // consumption and the next host write.
    for buf in [&staging_dirty_pos, &staging_dirty_rot, &staging_dirty_scl] {
        let mut w = buf.write().expect("zero-init staging_dirty_*.write");
        for word in w.iter_mut() {
            *word = 0;
        }
    }

    // ── Per-component scatter descriptor sets ──────────────────────
    //
    // All three sets share the same layout (set 0 of `scatter_cs`) — only
    // the bound staging + SoT buffers differ.
    let scatter_layout = world.scatter_set_layout().clone();
    let scatter_set_pos = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        scatter_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, staging_dirty_pos.clone()),
            WriteDescriptorSet::buffer(1, staging_positions.clone()),
            WriteDescriptorSet::buffer(2, world.sot_positions().clone()),
        ],
        [],
    ).expect("scatter_set_pos");
    let scatter_set_rot = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        scatter_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, staging_dirty_rot.clone()),
            WriteDescriptorSet::buffer(1, staging_rotations.clone()),
            WriteDescriptorSet::buffer(2, world.sot_rotations().clone()),
        ],
        [],
    ).expect("scatter_set_rot");
    let scatter_set_scl = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        scatter_layout,
        [
            WriteDescriptorSet::buffer(0, staging_dirty_scl.clone()),
            WriteDescriptorSet::buffer(1, staging_scales.clone()),
            WriteDescriptorSet::buffer(2, world.sot_scales().clone()),
        ],
        [],
    ).expect("scatter_set_scl");

    // ── Per-frame mvp-build set 1 (view_proj) ─────────────────────
    let mvp_build_set1 = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        world.mvp_build_set1_layout().clone(),
        [WriteDescriptorSet::buffer(0, view_proj_buf.clone())],
        [],
    ).expect("mvp_build_set1");

    // ── Pre-record the scatter compute secondary ───────────────────
    //
    // Three back-to-back dispatches with the same pipeline + push constant
    // (entity_count) but different descriptor sets. No render-pass
    // inheritance — compute can't run inside a render pass anyway.
    let scatter_pipeline = world.scatter_pipeline().clone();
    let scatter_layout_p = scatter_pipeline.layout().clone();
    let scatter_groups = (entity_capacity as u32).div_ceil(64);
    let scatter_pc = shaders::scatter_cs::PC { entity_count: entity_capacity as u32 };

    let mut scatter_builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        // Per-FrameSlot, so guarded by the per-image fence — only one
        // primary using this slot is in flight at a time. MultipleSubmit
        // suffices.
        CommandBufferUsage::MultipleSubmit,
        CommandBufferInheritanceInfo::default(),
    ).expect("scatter secondary builder");

    scatter_builder
        .bind_pipeline_compute(scatter_pipeline.clone()).expect("bind scatter pipeline")
        .push_constants(scatter_layout_p.clone(), 0, scatter_pc).expect("push scatter pc");
    for set in [&scatter_set_pos, &scatter_set_rot, &scatter_set_scl] {
        scatter_builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                scatter_layout_p.clone(),
                0,
                set.clone(),
            ).expect("bind scatter set");
        // Safety: dispatch counts derived from `entity_capacity`; shader
        // bounds-checks against the push-constant `entity_count`.
        unsafe {
            scatter_builder.dispatch([scatter_groups.max(1), 1, 1]).expect("dispatch scatter");
        }
    }
    let scatter_secondary = scatter_builder.build().expect("build scatter secondary");

    // ── Pre-record the mvp-build compute secondary ─────────────────
    let mvp_build_pipeline = world.mvp_build_pipeline().clone();
    let mvp_build_layout_p = mvp_build_pipeline.layout().clone();
    let mvp_build_groups   = (draw_count as u32).div_ceil(64).max(1);
    let mvp_build_pc = shaders::mvp_build_cs::PC { draw_count: draw_count as u32 };

    let mut mvp_builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::MultipleSubmit,
        CommandBufferInheritanceInfo::default(),
    ).expect("mvp_build secondary builder");

    mvp_builder
        .bind_pipeline_compute(mvp_build_pipeline.clone()).expect("bind mvp pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            mvp_build_layout_p.clone(),
            0,
            (main_camera.mvp_build_set0().clone(), mvp_build_set1.clone()),
        ).expect("bind mvp sets")
        .push_constants(mvp_build_layout_p, 0, mvp_build_pc).expect("push mvp pc");
    unsafe {
        mvp_builder.dispatch([mvp_build_groups, 1, 1]).expect("dispatch mvp");
    }
    let mvp_build_secondary = mvp_builder.build().expect("build mvp_build secondary");

    // ── Pre-record the blit secondary ──────────────────────────
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

    // ── Pre-record the primary command buffer ────────────────────
    //
    // The primary is the only CB actually submitted. It executes the four
    // secondaries in dependency order; vulkano auto-sync infers the
    // barriers from the resource-usage records each secondary carries:
    //
    //   scatter        (writes SoT pos/rot/scl)
    //     ↓  SHADER_WRITE → SHADER_READ on SoT buffers
    //   mvp_build      (reads SoT, writes device_matrices)
    //     ↓  SHADER_WRITE → SHADER_READ on device_matrices
    //   begin_rendering(camera attachments)
    //     scene_secondary  (vertex shader reads device_matrices)
    //   end_rendering
    //     ↓  COLOR_ATTACHMENT_WRITE → TRANSFER_READ on camera color
    //     ↓  Undefined / PresentSrc → TRANSFER_DST on swapchain image
    //   blit_secondary (camera color → swapchain image)
    //     ↓  TRANSFER_WRITE → PresentSrc on swapchain (final layout req.)
    let mut builder = AutoCommandBufferBuilder::primary(
        cb_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::MultipleSubmit,
    ).expect("primary CB builder");

    builder
        .execute_commands(scatter_secondary.clone()).expect("execute scatter");

    // GPU-side dirty-buffer clear, recorded once and replayed every frame.
    // After this fill_buffer completes, `staging_dirty_*` is all-zero again,
    // so the *next* time the host accesses this slot's dirty buffer (after
    // the per-image fence releases) it sees zeros and can safely write
    // only the bits dirtied *this* frame — no per-slot CPU fan-out needed.
    // Vulkano auto-sync infers the SHADER_READ→TRANSFER_WRITE barrier on
    // each dirty buffer between the scatter dispatch and this fill.
    //
    // We tried doing this in the scatter shader instead and saw a ~16×
    // FPS regression — see `shaders/scatter.comp` for the analysis.
    builder
        .fill_buffer(staging_dirty_pos.clone().reinterpret::<[u32]>(), 0).expect("fill staging_dirty_pos")
        .fill_buffer(staging_dirty_rot.clone().reinterpret::<[u32]>(), 0).expect("fill staging_dirty_rot")
        .fill_buffer(staging_dirty_scl.clone().reinterpret::<[u32]>(), 0).expect("fill staging_dirty_scl");

    builder
        .execute_commands(mvp_build_secondary.clone()).expect("execute mvp_build");

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
        staging_positions,
        staging_rotations,
        staging_scales,
        staging_dirty_pos,
        staging_dirty_rot,
        staging_dirty_scl,
        view_proj_buf,
        scatter_set_pos,
        scatter_set_rot,
        scatter_set_scl,
        mvp_build_set1,
        scatter_secondary,
        mvp_build_secondary,
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

/// Allocate a host-visible (sequential-write) STORAGE_BUFFER slice of
/// `count` elements. Used for every per-frame staging buffer (positions,
/// rotations, scales, dirty mask, view_proj).
///
/// `STORAGE_BUFFER` (rather than `UNIFORM_BUFFER`) keeps the same buffer
/// usable for everything from the bitmask to the per-mat4 view_proj — the
/// shaders use `readonly buffer` everywhere for uniformity.
///
/// `extra_usage` lets callers add usage flags on top of `STORAGE_BUFFER`.
/// The dirty bitmask buffers add `TRANSFER_DST` so the GPU can
/// `vkCmdFillBuffer(0)` them after the scatter compute consumes them —
/// see [`build_frame_slot`].
fn make_host_storage_slice<T>(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    count:            usize,
    extra_usage:      BufferUsage,
) -> Subbuffer<[T]>
where
    T: vulkano::buffer::BufferContents,
{
    Buffer::new_slice::<T>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | extra_usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        count.max(1) as u64,
    )
    .expect("Failed to allocate host storage slice")
}
