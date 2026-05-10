//! Vulkano-based renderer and windowing for the game engine.
//!
//! Public surface is [`Window`]. A typical setup:
//!
//! ```no_run
//! use engine_render::{Window, RenderInstance};
//! use engine_core::mesh::primitives;
//! use engine_core::transform::{TransformHierarchy, _Transform};
//! use std::sync::Arc;
//!
//! let mut hierarchy = TransformHierarchy::new();
//! let cube_idx = hierarchy.create_transform(_Transform::default()).get_idx();
//!
//! Window::new("My Game")
//!     .with_meshes(vec![primitives::cube()])
//!     .with_scene(Arc::new(hierarchy), vec![RenderInstance::new(0, cube_idx)])
//!     .on_update(move |h, dt| {
//!         use engine_core::transform::*;
//!         use glam::Quat;
//!         h.get_transform(cube_idx).unwrap().lock()
//!             .rotate_by(Quat::from_rotation_y(dt));
//!     })
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
    sync::Arc,
    time::Instant,
};

use engine_core::{
    mesh::Mesh,
    transform::TransformHierarchy,
};
use glam::Mat4;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferInheritanceInfo,
        CommandBufferInheritanceRenderingInfo, CommandBufferUsage,
        CopyBufferInfo, PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo,
        SecondaryAutoCommandBuffer, SubpassContents,
        allocator::{
            StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
        },
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        layout::DescriptorSetLayout,
    },
    device::{Device, DeviceFeatures, Queue},
    image::{ImageLayout, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
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
            viewport::{Viewport, ViewportState},
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

mod camera;
mod gpu_mesh;
mod scene;
mod shaders;
mod swapchain;

use camera::{CAMERA_COLOR_FORMAT, CAMERA_DEPTH_FORMAT, RenderCamera};
use gpu_mesh::{GpuMesh, GpuVertex};
use swapchain::SwapchainRenderer;

pub use scene::{Camera, OrbitController, RenderInstance};

// Trait imports needed for method resolution on GPU types.
use vulkano::pipeline::graphics::vertex_input::Vertex as VulkanoVertex;
use vulkano::pipeline::Pipeline;

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

/// All resources tied to a single swapchain image. Built once per swapchain
/// image when the swapchain (or scene topology) changes.
struct FrameSlot {
    /// Host-visible matrix staging buffer. The host writes per-instance MVPs
    /// here every frame; the recorded command buffer copies it into
    /// `device_matrices` before drawing.
    staging_matrices: Subbuffer<[[f32; 16]]>,
    /// Device-local matrix buffer. Bound (via `descriptor_set`) as the
    /// storage buffer the vertex shader reads. Held here purely to keep the
    /// allocation alive for the lifetime of the slot — the recorded CB
    /// already references it internally.
    #[allow(dead_code)]
    device_matrices:  Subbuffer<[[f32; 16]]>,
    /// Set 0 — references `device_matrices`. Bound by the recorded CB; held
    /// here only to extend its lifetime to match the CB.
    #[allow(dead_code)]
    descriptor_set:   Arc<DescriptorSet>,
    /// Pre-recorded **secondary** that contains all draw work for this slot's
    /// camera attachments. Inherits the primary's dynamic-rendering scope (so
    /// it cannot call `begin_rendering`/`end_rendering` itself; only viewport,
    /// pipeline binds, vertex/index binds, and `draw_indexed`).
    ///
    /// Invalidation domain: scene topology / pipeline / per-camera attachment
    /// extent. Today every FrameSlot rebuild rebuilds this; the stratification
    /// machinery is in place so future changes can rebuild it independently of
    /// the blit secondary.
    #[allow(dead_code)]
    scene_secondary:  Arc<SecondaryAutoCommandBuffer>,
    /// Pre-recorded **secondary** that contains the present-blit (offscreen
    /// camera color → swapchain image). No render-pass inheritance.
    ///
    /// Invalidation domain: swapchain image identity / extent. Decoupling this
    /// from `scene_secondary` is what unlocks "only rebuild what changed" once
    /// the camera attachments are decoupled from the swapchain.
    #[allow(dead_code)]
    blit_secondary:   Arc<SecondaryAutoCommandBuffer>,
    /// Pre-recorded **primary** that stitches the secondaries together with
    /// the per-frame `copy_buffer` (staging→device matrices) and the dynamic
    /// rendering scope. This is the CB actually submitted to the queue.
    ///
    /// Holds `Arc`s to both secondaries internally; the `scene_secondary` /
    /// `blit_secondary` fields above are kept on `FrameSlot` so future
    /// per-stratum rebuild paths can swap them without rebuilding the primary.
    command_buffer:   Arc<PrimaryAutoCommandBuffer>,
    /// Number of `[f32; 16]` slots in the staging/device buffers.
    capacity:         usize,
}


// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Per-frame closure invoked by the renderer immediately before recording the
/// next command buffer. Receives the live transform hierarchy and the
/// elapsed time since the previous frame in seconds.
pub type UpdateFn = Box<dyn FnMut(&TransformHierarchy, f32) + 'static>;

/// An OS window backed by a Vulkan swapchain.
pub struct Window {
    title:     String,
    meshes:    Vec<Mesh>,
    hierarchy: Option<Arc<TransformHierarchy>>,
    instances: Vec<RenderInstance>,
    on_update: Option<UpdateFn>,
}

impl Window {
    /// Create a window descriptor with the given title.
    pub fn new(title: &str) -> Self {
        Window {
            title:     title.to_owned(),
            meshes:    Vec::new(),
            hierarchy: None,
            instances: Vec::new(),
            on_update: None,
        }
    }

    /// Attach CPU meshes that will be uploaded to the GPU at startup.
    /// The order here defines the `mesh_index` used by [`RenderInstance`].
    pub fn with_meshes(mut self, meshes: Vec<Mesh>) -> Self {
        self.meshes = meshes;
        self
    }

    /// Attach a transform hierarchy and a list of instances drawn each frame.
    ///
    /// The hierarchy is shared via `Arc` so the game / editor can keep a
    /// reference for read-only queries (mutations to existing transforms go
    /// through the interior-mutability guard system on `TransformHierarchy`
    /// itself, so no `&mut` is needed on the hot path).
    pub fn with_scene(
        mut self,
        hierarchy: Arc<TransformHierarchy>,
        instances: Vec<RenderInstance>,
    ) -> Self {
        self.hierarchy = Some(hierarchy);
        self.instances = instances;
        self
    }

    /// Register a per-frame update callback. Invoked before each render pass
    /// with `(hierarchy, dt_seconds)`.
    pub fn on_update<F>(mut self, f: F) -> Self
    where
        F: FnMut(&TransformHierarchy, f32) + 'static,
    {
        self.on_update = Some(Box::new(f));
        self
    }

    /// Open the OS window, initialise Vulkan, and block on the event loop.
    pub fn run(self) {
        let event_loop = EventLoop::new().expect("Failed to create winit EventLoop");
        let mut app = RenderApp::new(
            self.title,
            self.meshes,
            self.hierarchy,
            self.instances,
            self.on_update,
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
    /// Read-only handle held by the renderer; the game may keep its own clone
    /// for queries. Mutations to existing transforms go through the
    /// hierarchy's interior-mutability API.
    hierarchy:                   Option<Arc<TransformHierarchy>>,
    instances:                   Vec<RenderInstance>,
    on_update:                   Option<UpdateFn>,
    orbit:                       OrbitController,
    last_frame_time:             Option<Instant>,

    /// Reusable scratch buffer for per-frame MVP computation — reused so we
    /// don't allocate every frame.
    mvp_scratch:                 Vec<[f32; 16]>,
}

/// Swapchain-image-count-sized arrays rebuilt on every swapchain recreation.
struct RenderContext {
    /// Cached swapchain image views. Used as **blit destinations** by each
    /// FrameSlot's pre-recorded CB; refreshed on resize.
    swapchain_image_views: Vec<Arc<ImageView>>,
    /// GPU mesh buffers — uploaded once; kept alive here for the lifetime of
    /// the renderer.
    gpu_meshes:        Vec<GpuMesh>,
    /// The render-side camera that drives the scene render. Owns its own
    /// offscreen color + depth attachments and a [`CameraResolution`] policy
    /// (currently always `MatchSwapchain`, so the present-blit stays 1:1).
    /// On a swapchain resize the camera decides whether to rebuild its
    /// attachments — future `Fixed` / `ScaleSwapchain` cameras will survive
    /// swapchain resizes untouched without changing the swapchain handler.
    main_camera:       RenderCamera,
    /// One `FrameSlot` per swapchain image. Each slot owns the per-image
    /// matrix staging/device buffers, a descriptor set, and the secondaries
    /// + primary CB that reference `main_camera`'s attachments and the
    /// slot's swapchain image.
    frame_slots:       Vec<FrameSlot>,
    /// Mesh indices, one per `RenderInstance`, baked into every slot's
    /// command buffer at build time. Kept here so we can detect topology
    /// changes and rebuild slots if needed.
    draws_template:    Vec<u32>,
}

impl RenderApp {
    fn new(
        title:     String,
        meshes:    Vec<Mesh>,
        hierarchy: Option<Arc<TransformHierarchy>>,
        instances: Vec<RenderInstance>,
        on_update: Option<UpdateFn>,
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
            hierarchy,
            instances,
            on_update,
            orbit: OrbitController::new(),
            last_frame_time: None,
            mvp_scratch: Vec::new(),
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

        // Bake the static (mesh_index per draw) topology. If `with_scene`
        // wasn't called, fall back to drawing every uploaded mesh once at
        // the origin (legacy test-code behaviour).
        let draws_template: Vec<u32> = if self.instances.is_empty() {
            (0..gpu_meshes.len() as u32).collect()
        } else {
            self.instances.iter().map(|i| i.mesh_index).collect()
        };

        // The main camera matches the swapchain extent so the present-blit
        // stays a 1:1 copy. The first swapchain image gives us the extent.
        let initial_extent = {
            let [w, h, _] = attachment_image_views[0].image().extent();
            [w, h]
        };
        let main_camera = RenderCamera::new_match_swapchain(
            &self.memory_allocator,
            initial_extent,
        );

        let frame_slots = build_all_frame_slots(
            &self.command_buffer_allocator,
            &self.memory_allocator,
            &self.descriptor_set_allocator,
            &pipeline,
            self.graphics_queue.queue_family_index(),
            &attachment_image_views,
            &main_camera,
            &gpu_meshes,
            &draws_template,
        );

        self.rcx = Some(RenderContext {
            swapchain_image_views: attachment_image_views,
            gpu_meshes,
            main_camera,
            frame_slots,
            draws_template,
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

        if let (Some(hierarchy), Some(cb)) = (self.hierarchy.as_ref(), self.on_update.as_mut()) {
            cb(hierarchy.as_ref(), dt);
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
            // attachments. Future cameras with a swapchain-independent
            // policy (`Fixed` / `ScaleSwapchain`) would survive this call
            // untouched, and only the present-blit secondary would need a
            // rebuild on swapchain change.
            let new_extent = {
                let [w, h, _] = swapchain_images[0].image().extent();
                [w, h]
            };
            let _camera_rebuilt = rcx.main_camera
                .on_swapchain_resize(&memory_allocator, new_extent);

            // The CBs in every slot reference the *old* swapchain images
            // (as blit destinations) and — if the camera rebuilt — the
            // *old* offscreen color/depth attachments. Rebuild from scratch.
            // (Per-stratum partial rebuild lands when more cameras with
            // distinct invalidation domains exist; today both strata
            // invalidate together for the main camera.)
            rcx.frame_slots = build_all_frame_slots(
                &cb_allocator,
                &memory_allocator,
                &descriptor_set_allocator,
                &pipeline_for_recreate,
                queue_family_index,
                &rcx.swapchain_image_views,
                &rcx.main_camera,
                &rcx.gpu_meshes,
                &rcx.draws_template,
            );
        }) {
            Some(f) => f,
            None    => return, // out-of-date / minimised — skip frame
        };

        // ── Compute per-instance MVPs into the staging buffer ─────────
        let image_index = frame.image_index as usize;
        let [w, h, _]   = rcx.swapchain_image_views[image_index].image().extent();
        let aspect      = w as f32 / h.max(1) as f32;
        let view_proj   = self.orbit.camera().view_proj(aspect);

        let slot = &rcx.frame_slots[image_index];
        let draw_count = slot.capacity;

        // Reuse the scratch vec to avoid per-frame heap traffic.
        self.mvp_scratch.clear();
        self.mvp_scratch.reserve(draw_count);
        if self.instances.is_empty() {
            // Legacy fallback path — every uploaded mesh at the origin.
            for _ in 0..draw_count {
                self.mvp_scratch.push(view_proj.to_cols_array());
            }
        } else {
            for inst in &self.instances {
                let model = if let Some(h) = self.hierarchy.as_ref() {
                    if let Some(t) = h.get_transform(inst.transform_index) {
                        let g = t.lock();
                        scene::model_matrix(
                            g.get_global_position(),
                            g.get_global_rotation(),
                            g.get_global_scale(),
                        )
                    } else {
                        Mat4::IDENTITY
                    }
                } else {
                    Mat4::IDENTITY
                };
                self.mvp_scratch.push((view_proj * model).to_cols_array());
            }
        }
        debug_assert_eq!(self.mvp_scratch.len(), draw_count);

        // SAFETY: we waited on the per-image fence inside `acquire`, so the
        // GPU is no longer reading this slot's staging buffer.
        {
            let mut guard = slot.staging_matrices.write()
                .expect("staging_matrices.write failed");
            guard.copy_from_slice(&self.mvp_scratch);
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
    cb_allocator:           &Arc<StandardCommandBufferAllocator>,
    memory_allocator:       &Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    pipeline:               &Arc<GraphicsPipeline>,
    queue_family_index:     u32,
    swapchain_views:        &[Arc<ImageView>],
    main_camera:            &RenderCamera,
    gpu_meshes:             &[GpuMesh],
    draws_template:         &[u32],
) -> Vec<FrameSlot> {
    swapchain_views.iter().map(|swapchain_view| {
        build_frame_slot(
            cb_allocator,
            memory_allocator,
            descriptor_set_allocator,
            pipeline,
            queue_family_index,
            swapchain_view,
            main_camera,
            gpu_meshes,
            draws_template,
        )
    }).collect()
}

/// Build one `FrameSlot`: allocate matrix buffers, allocate the camera's
/// offscreen color + depth attachments, build the descriptor set, and
/// pre-record the reusable command buffer (staging→device copy + scene
/// render into the offscreen attachments + blit into the swapchain image).
fn build_frame_slot(
    cb_allocator:             &Arc<StandardCommandBufferAllocator>,
    memory_allocator:         &Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    pipeline:                 &Arc<GraphicsPipeline>,
    queue_family_index:       u32,
    swapchain_view:           &Arc<ImageView>,
    main_camera:              &RenderCamera,
    gpu_meshes:               &[GpuMesh],
    draws_template:           &[u32],
) -> FrameSlot {
    let swapchain_image = swapchain_view.image().clone();

    // Camera-owned offscreen attachments. The viewport / dynamic-rendering
    // scope use the camera's extent, NOT the swapchain image's — they only
    // happen to coincide today because the main camera uses
    // `CameraResolution::MatchSwapchain`. The present-blit copies
    // camera-extent → swapchain-extent (which are equal for the main
    // camera, so it remains a true 1:1 copy).
    let color_image = main_camera.color_image().clone();
    let color_view  = main_camera.color_view().clone();
    let depth_view  = main_camera.depth_view().clone();
    let [cam_w, cam_h] = main_camera.extent();

    // Buffers must be non-zero-sized; clamp capacity up to 1 even if the
    // scene is empty (the empty CB will simply have no draws).
    let capacity_logical = draws_template.len();
    let capacity_alloc   = capacity_logical.max(1);

    let staging_matrices: Subbuffer<[[f32; 16]]> = Buffer::new_slice::<[f32; 16]>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        capacity_alloc as u64,
    )
    .expect("Failed to allocate staging matrix buffer");

    let device_matrices: Subbuffer<[[f32; 16]]> = Buffer::new_slice::<[f32; 16]>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        capacity_alloc as u64,
    )
    .expect("Failed to allocate device matrix buffer");

    // Set 0 — binding 0 — the storage buffer the vertex shader reads.
    let set_layout: Arc<DescriptorSetLayout> = pipeline
        .layout()
        .set_layouts()[0]
        .clone();
    let descriptor_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        set_layout,
        [WriteDescriptorSet::buffer(0, device_matrices.clone())],
        [],
    )
    .expect("Failed to allocate matrices descriptor set");

    // ── Pre-record the scene secondary ──────────────────────────────
    //
    // Inherits the primary's dynamic-rendering scope (color/depth formats
    // must match what the primary's `begin_rendering` will declare). The
    // secondary may NOT call `begin_rendering`/`end_rendering` itself.
    let scene_inheritance = CommandBufferInheritanceInfo {
        render_pass: Some(
            CommandBufferInheritanceRenderingInfo {
                color_attachment_formats: vec![Some(CAMERA_COLOR_FORMAT)],
                depth_attachment_format:  Some(CAMERA_DEPTH_FORMAT),
                ..Default::default()
            }
            .into(),
        ),
        ..Default::default()
    };

    let mut scene_builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::MultipleSubmit,
        scene_inheritance,
    )
    .expect("Failed to create scene secondary builder");

    scene_builder
        .set_viewport(
            0,
            smallvec::smallvec![Viewport {
                offset:      [0.0, 0.0],
                extent:      [cam_w as f32, cam_h as f32],
                depth_range: 0.0..=1.0,
            }],
        )
        .expect("set_viewport failed")
        .bind_pipeline_graphics(pipeline.clone())
        .expect("bind_pipeline_graphics failed")
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            pipeline.layout().clone(),
            0,
            descriptor_set.clone(),
        )
        .expect("bind_descriptor_sets failed");

    // One draw per RenderInstance, with `first_instance = i` so the vertex
    // shader's `gl_InstanceIndex` indexes into `device_matrices`.
    for (i, &mesh_idx) in draws_template.iter().enumerate() {
        let mesh = match gpu_meshes.get(mesh_idx as usize) {
            Some(m) => m,
            None    => continue,
        };
        scene_builder
            .bind_vertex_buffers(0, mesh.vertex_buffer.clone())
            .expect("bind_vertex_buffers failed")
            .bind_index_buffer(mesh.index_buffer.clone())
            .expect("bind_index_buffer failed");
        // Safety: buffers are compatible with the bound pipeline; index
        // count fits within the uploaded index slice; first_instance is
        // bounded by `capacity_logical`.
        unsafe {
            scene_builder
                .draw_indexed(mesh.index_count, 1, 0, 0, i as u32)
                .expect("draw_indexed failed");
        }
    }

    let scene_secondary = scene_builder
        .build()
        .expect("Failed to build scene secondary");

    // ── Pre-record the blit secondary ───────────────────────────────
    //
    // No render-pass inheritance: this secondary executes outside the
    // primary's dynamic-rendering scope and just performs the
    // offscreen-color → swapchain-image blit.
    let mut blit_builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::MultipleSubmit,
        CommandBufferInheritanceInfo::default(),
    )
    .expect("Failed to create blit secondary builder");

    blit_builder
        .blit_image(BlitImageInfo::images(color_image.clone(), swapchain_image))
        .expect("blit_image failed");

    let blit_secondary = blit_builder
        .build()
        .expect("Failed to build blit secondary");

    // ── Pre-record the primary command buffer ───────────────────────
    //
    // The primary is the only CB actually submitted. It:
    //   1. copies staging matrices → device-local matrix buffer,
    //   2. opens a dynamic-rendering scope on the camera attachments and
    //      executes the scene secondary inside it,
    //   3. executes the blit secondary (outside the rendering scope) to
    //      copy the offscreen color image into the swapchain image.
    //
    // AutoCommandBuffer's tracker still sees every resource through the
    // secondaries' usage records, so the TRANSFER_WRITE → SHADER_READ and
    // COLOR_ATTACHMENT_WRITE → TRANSFER_READ barriers are inferred
    // correctly across the secondary/primary boundaries (vulkano's auto-sync
    // covers in-CB *and* primary↔secondary transitions, but NOT cross-CB
    // submissions — which is exactly why we compose secondaries into one
    // primary instead of submitting multiple primaries).
    let mut builder = AutoCommandBufferBuilder::primary(
        cb_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::MultipleSubmit,
    )
    .expect("Failed to create primary command buffer builder");

    builder
        .copy_buffer(CopyBufferInfo::buffers(
            staging_matrices.clone(),
            device_matrices.clone(),
        ))
        .expect("copy_buffer failed");

    builder
        .begin_rendering(RenderingInfo {
            // Tell the primary that draw commands inside this scope will
            // come from secondary CBs (not inline `draw_*` calls).
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
        })
        .expect("begin_rendering failed");

    builder
        .execute_commands(scene_secondary.clone())
        .expect("execute_commands(scene_secondary) failed");

    builder.end_rendering().expect("end_rendering failed");

    builder
        .execute_commands(blit_secondary.clone())
        .expect("execute_commands(blit_secondary) failed");

    let command_buffer = builder.build().expect("Failed to build primary command buffer");

    FrameSlot {
        staging_matrices,
        device_matrices,
        descriptor_set,
        scene_secondary,
        blit_secondary,
        command_buffer,
        capacity: capacity_logical,
    }
}
