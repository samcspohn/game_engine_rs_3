//! Vulkano-based renderer and windowing for the game engine.
//!
//! This crate exposes a single public type, [`Window`], which manages:
//! * An OS window (via `winit 0.30`).
//! * A Vulkan instance, physical device, logical device, and swapchain (via
//!   `vulkano 0.35` and `vulkano-util 0.35`).
//! * A depth buffer (one `D32_SFLOAT` image per swapchain image).
//! * A graphics pipeline with push-constant MVP and a simple diffuse shader.
//! * Per-frame indexed draw calls for every [`Mesh`] passed to
//!   [`Window::with_meshes`].
//!
//! ## Typical usage
//! ```no_run
//! use engine_render::Window;
//! use engine_core::mesh::primitives;
//!
//! Window::new("My Game")
//!     .with_meshes(vec![primitives::cube()])
//!     .run();
//! ```

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use engine_core::mesh::Mesh;
use vulkano::{
    buffer::BufferContents,
    command_buffer::{
        allocator::{
            StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
        },
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderingAttachmentInfo, RenderingInfo,
    },
    device::{Device, DeviceFeatures},
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::{PipelineRenderingCreateInfo, PipelineSubpassType},
            vertex_input::VertexDefinition,
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{PresentMode, SurfaceInfo},
    sync::{future::FenceSignalFuture, GpuFuture},
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::WindowId,
};

mod gpu_mesh;
mod shaders;

use gpu_mesh::{GpuMesh, GpuVertex};

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

// ─────────────────────────────────────────────────────────────────────────────
// Push constants
// ─────────────────────────────────────────────────────────────────────────────

/// 4×4 MVP matrix sent to the vertex shader as a push constant (64 bytes).
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct MvpPushConstants {
    mvp: [f32; 16],
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// An OS window backed by a Vulkan swapchain.
///
/// Build one with [`Window::new`], optionally add renderable meshes via
/// [`Window::with_meshes`], then block on the event loop with [`Window::run`].
pub struct Window {
    title:  String,
    meshes: Vec<Mesh>,
}

impl Window {
    /// Create a window descriptor with the given title.
    pub fn new(title: &str) -> Self {
        Window { title: title.to_owned(), meshes: Vec::new() }
    }

    /// Attach CPU meshes that will be uploaded to the GPU and drawn every frame.
    pub fn with_meshes(mut self, meshes: Vec<Mesh>) -> Self {
        self.meshes = meshes;
        self
    }

    /// Open the OS window, initialise Vulkan, and block on the event loop.
    ///
    /// # Panics
    /// Panics if Vulkan setup fails (no compatible GPU, missing Vulkan 1.3
    /// dynamic-rendering support, etc.).
    pub fn run(self) {
        let event_loop = EventLoop::new().expect("Failed to create winit EventLoop");
        let mut app = RenderApp::new(self.title, self.meshes);
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
    title:                     String,
    context:                   VulkanoContext,
    windows:                   VulkanoWindows,
    command_buffer_allocator:  Arc<StandardCommandBufferAllocator>,
    memory_allocator:          Arc<StandardMemoryAllocator>,
    fps:                       FpsTracker,
    /// CPU meshes kept around so they can be re-uploaded after a GPU reset.
    meshes:                    Vec<Mesh>,
    /// GPU pipeline — created once in `resumed`, lives until the app exits.
    pipeline:                  Option<Arc<GraphicsPipeline>>,
    /// Per-swapchain-image data (rebuilt on resize).
    rcx:                       Option<RenderContext>,
}

/// Swapchain-image-count-sized arrays rebuilt on every swapchain recreation.
struct RenderContext {
    attachment_image_views:  Vec<Arc<ImageView>>,
    depth_image_views:       Vec<Arc<ImageView>>,
    /// GPU mesh buffers — uploaded once; kept here alongside the command
    /// buffers that reference them so we can hand a `&[GpuMesh]` to
    /// `build_command_buffers` inside the swapchain-recreation closure.
    gpu_meshes:              Vec<GpuMesh>,
    cached_command_buffers:  Vec<Arc<PrimaryAutoCommandBuffer>>,
    /// One in-flight fence per swapchain image.
    ///
    /// We re-use a single `SimultaneousUse` command buffer per swapchain image
    /// across multiple frames. Vulkano's host-side resource tracking refuses to
    /// re-submit a command buffer whose resources (notably the depth
    /// attachment) are still marked as in-use by a previous submission.
    ///
    /// Before resubmitting the command buffer for image `i`, we block on this
    /// fence — `FenceSignalFuture::wait` calls `signal_finished` on its inner
    /// chain, releasing the host-side locks on every resource that submission
    /// touched. Because we only block on the per-image fence (not on the
    /// global previous-frame chain), we still get up to `num_swapchain_images`
    /// frames worth of pipelining.
    per_image_fences:        Vec<Option<Arc<FenceSignalFuture<Box<dyn GpuFuture>>>>>,
}

impl RenderApp {
    fn new(title: String, meshes: Vec<Mesh>) -> Self {
        let context = VulkanoContext::new(VulkanoConfig {
            device_features: DeviceFeatures {
                dynamic_rendering: true,
                ..Default::default()
            },
            ..Default::default()
        });

        let windows = VulkanoWindows::default();

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            StandardCommandBufferAllocatorCreateInfo {
                primary_buffer_count:   32,
                secondary_buffer_count: 0,
                ..Default::default()
            },
        ));

        let memory_allocator =
            Arc::new(StandardMemoryAllocator::new_default(context.device().clone()));

        RenderApp {
            title,
            context,
            windows,
            command_buffer_allocator,
            memory_allocator,
            fps: FpsTracker::new(),
            meshes,
            pipeline: None,
            rcx:      None,
        }
    }
}

impl ApplicationHandler for RenderApp {
    /// Called once at startup (and again on Android resume cycles).
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Drop stale renderer on mobile resume.
        if let Some(id) = self.windows.primary_window_id() {
            self.windows.remove_renderer(id);
        }

        self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                title: self.title.clone(),
                ..Default::default()
            },
            |sc| {
                sc.min_image_count =
                    sc.min_image_count.max(MAX_FRAMES_IN_FLIGHT as u32);
            },
        );

        let renderer = self
            .windows
            .get_primary_renderer_mut()
            .expect("Primary renderer must exist right after create_window");

        let swapchain_format      = renderer.swapchain_format();
        let attachment_image_views = renderer.swapchain_image_views().to_vec();
        let [width, height, _]    = attachment_image_views[0].image().extent();

        // Build pipeline once (format only; not swapchain-image-count-dependent).
        let pipeline = create_pipeline(self.context.device().clone(), swapchain_format);
        self.pipeline = Some(pipeline.clone());

        // Upload CPU meshes → GPU buffers (once; reused across resizes).
        let gpu_meshes: Vec<GpuMesh> = self
            .meshes
            .iter()
            .map(|m| GpuMesh::upload(m, &self.memory_allocator))
            .collect();

        let depth_image_views = create_depth_views(
            &self.memory_allocator,
            attachment_image_views.len(),
            [width, height],
        );

        let mvp              = compute_mvp(width, height);
        let cached_command_buffers = build_command_buffers(
            &self.command_buffer_allocator,
            self.context.graphics_queue().queue_family_index(),
            &attachment_image_views,
            &depth_image_views,
            &pipeline,
            &gpu_meshes,
            mvp,
        );

        let per_image_fences = (0..cached_command_buffers.len()).map(|_| None).collect();

        self.rcx = Some(RenderContext {
            attachment_image_views,
            depth_image_views,
            gpu_meshes,
            cached_command_buffers,
            per_image_fences,
        });

        // ── Present-mode selection ────────────────────────────────────────────
        let surface   = renderer.surface();
        let supported = self
            .context
            .device()
            .physical_device()
            .surface_present_modes(surface.as_ref(), SurfaceInfo::default())
            .expect("Failed to query surface present modes");

        let chosen = if supported.contains(&PresentMode::Mailbox) {
            PresentMode::Mailbox
        } else if supported.contains(&PresentMode::Immediate) {
            PresentMode::Immediate
        } else {
            PresentMode::Fifo
        };

        println!("Present mode: {chosen:?}  (supported: {supported:?})");
        renderer.set_present_mode(chosen);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id:  WindowId,
        event:       WindowEvent,
    ) {
        let renderer = match self.windows.get_primary_renderer_mut() {
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

        let renderer = match self.windows.get_primary_renderer_mut() {
            Some(r) => r,
            None    => return,
        };
        let rcx = match self.rcx.as_mut() {
            Some(r) => r,
            None    => return,
        };

        let size = renderer.window().inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        // Pre-clone everything the swapchain-recreation closure needs so it
        // doesn't capture `self` (which would conflict with the `renderer` and
        // `rcx` borrows below).
        let allocator          = self.command_buffer_allocator.clone();
        let memory_allocator   = self.memory_allocator.clone();
        let queue_family_index = self.context.graphics_queue().queue_family_index();
        // Arc clone (cheap) so the closure doesn't need to borrow self.pipeline.
        let pipeline           = self.pipeline.clone().expect("Pipeline not initialised");

        let previous_frame_end = renderer
            .acquire(Some(Duration::from_millis(1_000)), |swapchain_images| {
                let [w, h, _] = swapchain_images[0].image().extent();

                // Rebuild depth images to match new swapchain dimensions.
                let new_depth_views = create_depth_views(
                    &memory_allocator,
                    swapchain_images.len(),
                    [w, h],
                );
                let mvp = compute_mvp(w, h);

                // Build new command buffers.  The immutable borrow of
                // `rcx.gpu_meshes` ends when `build_command_buffers` returns;
                // the subsequent assignments target disjoint fields (NLL).
                let new_bufs = build_command_buffers(
                    &allocator,
                    queue_family_index,
                    swapchain_images,
                    &new_depth_views,
                    &pipeline,
                    &rcx.gpu_meshes,
                    mvp,
                );

                // Drain stale per-image fences. We `wait` on each so its inner
                // chain calls `signal_finished`, releasing host-side locks on
                // the *old* depth views and command buffers before we drop
                // them. Without this, the old `SimultaneousUse` command
                // buffers would still hold write-locks on resources that are
                // about to be dropped.
                for fence in rcx.per_image_fences.drain(..).flatten() {
                    let _ = fence.wait(None);
                }

                rcx.attachment_image_views = swapchain_images.to_vec();
                rcx.depth_image_views      = new_depth_views;
                rcx.cached_command_buffers = new_bufs;
                rcx.per_image_fences       =
                    (0..rcx.cached_command_buffers.len()).map(|_| None).collect();
            })
            .expect("Failed to acquire swapchain image");

        let image_index = renderer.image_index() as usize;

        // Release host-side resource tracking from this image's previous
        // submission (if any). `wait` blocks until the GPU is done, then calls
        // `signal_finished` on the inner chain, unlocking the depth view and
        // every other resource the cached `SimultaneousUse` command buffer
        // touches. Without this we'd hit `AccessError::AlreadyInUse` on the
        // depth attachment as soon as `image_index` repeats.
        if let Some(prev_fence) = rcx.per_image_fences[image_index].take() {
            prev_fence
                .wait(None)
                .expect("per-image fence wait failed");
        }

        // Hot path: clone the Arc — zero recording, zero allocation.
        let command_buffer = rcx.cached_command_buffers[image_index].clone();

        // Build the execution chain and signal our own per-image fence so we
        // can wait on it the next time this `image_index` comes around.
        let exec_future: Box<dyn GpuFuture> = previous_frame_end
            .then_execute(self.context.graphics_queue().clone(), command_buffer)
            .expect("Failed to submit command buffer")
            .boxed();

        let fence = Arc::new(
            exec_future
                .then_signal_fence_and_flush()
                .expect("then_signal_fence_and_flush failed"),
        );
        rcx.per_image_fences[image_index] = Some(fence.clone());

        // The Arc<FenceSignalFuture<…>> implements GpuFuture, so vulkano-util
        // can chain `then_swapchain_present` after it. Both halves observe the
        // same underlying fence object.
        renderer.present(fence.boxed(), false);
        self.fps.tick();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Allocate one `D32_SFLOAT` depth image + view per swapchain image.
///
/// One-per-image prevents data races when `SimultaneousUse` command buffers
/// for different swapchain indices are executing concurrently on the GPU.
fn create_depth_views(
    allocator: &Arc<StandardMemoryAllocator>,
    count:     usize,
    extent:    [u32; 2],
) -> Vec<Arc<ImageView>> {
    (0..count)
        .map(|_| {
            let image = Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format:     Format::D32_SFLOAT,
                    extent:     [extent[0], extent[1], 1],
                    usage:      ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .expect("Failed to create depth image");
            ImageView::new_default(image).expect("Failed to create depth image view")
        })
        .collect()
}

/// Create the single graphics pipeline used for all mesh draws.
///
/// The pipeline uses dynamic viewport state so it does not need to be
/// recreated on window resize — only the command buffers are rebuilt.
fn create_pipeline(device: Arc<Device>, swapchain_format: Format) -> Arc<GraphicsPipeline> {
    let vs = shaders::vs::load(device.clone()).expect("Failed to load vertex shader");
    let fs = shaders::fs::load(device.clone()).expect("Failed to load fragment shader");

    let stages = [
        PipelineShaderStageCreateInfo::new(vs.entry_point("main").unwrap()),
        PipelineShaderStageCreateInfo::new(fs.entry_point("main").unwrap()),
    ];

    // Reflect vertex attribute locations from the shader interface.
    let vertex_input_state = GpuVertex::per_vertex()
        .definition(&stages[0].entry_point)
        .expect("Vertex input definition mismatch");

    // Derive the pipeline layout (including push-constant ranges) from shaders.
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .expect("Failed to create pipeline layout create info"),
    )
    .expect("Failed to create pipeline layout");

    GraphicsPipeline::new(
        device,
        None, // no pipeline cache
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            // Viewport is dynamic so we don't recreate the pipeline on resize.
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
            // Dynamic rendering — no RenderPass / Framebuffer objects needed.
            subpass: Some(PipelineSubpassType::BeginRendering(
                PipelineRenderingCreateInfo {
                    color_attachment_formats: vec![Some(swapchain_format)],
                    depth_attachment_format:  Some(Format::D32_SFLOAT),
                    ..Default::default()
                },
            )),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .expect("Failed to create graphics pipeline")
}

/// Compute a column-major MVP matrix for a fixed perspective camera.
///
/// The Y axis is flipped to convert from OpenGL/glam right-handed NDC to
/// Vulkan's clip space (Y points down in Vulkan NDC).
fn compute_mvp(width: u32, height: u32) -> [f32; 16] {
    let aspect = width as f32 / height.max(1) as f32;

    let mut proj = glam::Mat4::perspective_rh(60_f32.to_radians(), aspect, 0.1, 100.0);
    proj.y_axis.y *= -1.0; // Vulkan Y-flip

    let view = glam::Mat4::look_at_rh(
        glam::Vec3::new(1.5, 1.5, 2.5), // eye
        glam::Vec3::ZERO,               // target (origin = cube centre)
        glam::Vec3::Y,
    );

    (proj * view).to_cols_array()
}

/// Record one [`PrimaryAutoCommandBuffer`] per swapchain image.
///
/// Each buffer:
/// 1. Begins dynamic rendering (dark background + depth clear to 1.0).
/// 2. Sets the dynamic viewport.
/// 3. Binds the pipeline.
/// 4. For every mesh: pushes the MVP constant, binds VB+IB, issues `draw_indexed`.
/// 5. Ends dynamic rendering.
///
/// Built with [`CommandBufferUsage::SimultaneousUse`]: with
/// `MAX_FRAMES_IN_FLIGHT = 4` and a 4-image swapchain the GPU can have two
/// submissions for the same image index in-flight concurrently (one depth image
/// per swapchain image prevents write-after-write hazards on the depth buffer).
fn build_command_buffers(
    allocator:             &Arc<StandardCommandBufferAllocator>,
    queue_family_index:    u32,
    attachment_image_views: &[Arc<ImageView>],
    depth_image_views:     &[Arc<ImageView>],
    pipeline:              &Arc<GraphicsPipeline>,
    gpu_meshes:            &[GpuMesh],
    mvp:                   [f32; 16],
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    attachment_image_views
        .iter()
        .zip(depth_image_views.iter())
        .map(|(color_view, depth_view)| {
            let [width, height, _] = color_view.image().extent();

            let mut builder = AutoCommandBufferBuilder::primary(
                allocator.clone(),
                queue_family_index,
                CommandBufferUsage::SimultaneousUse,
            )
            .expect("Failed to create command buffer builder");

            // ── Dynamic rendering begin ───────────────────────────────────────
            builder
                .begin_rendering(RenderingInfo {
                    color_attachments: vec![Some(RenderingAttachmentInfo {
                        load_op:     AttachmentLoadOp::Clear,
                        store_op:    AttachmentStoreOp::Store,
                        clear_value: Some([0.08, 0.08, 0.10, 1.0].into()), // near-black
                        ..RenderingAttachmentInfo::image_view(color_view.clone())
                    })],
                    depth_attachment: Some(RenderingAttachmentInfo {
                        image_layout: ImageLayout::DepthStencilAttachmentOptimal,
                        load_op:      AttachmentLoadOp::Clear,
                        store_op:     AttachmentStoreOp::DontCare,
                        clear_value:  Some(1.0_f32.into()), // far plane
                        ..RenderingAttachmentInfo::image_view(depth_view.clone())
                    }),
                    ..Default::default()
                })
                .expect("begin_rendering failed");

            // ── Pipeline + viewport ───────────────────────────────────────────
            builder
                .set_viewport(
                0,
                smallvec::smallvec![Viewport {
                    offset:      [0.0, 0.0],
                    extent:      [width as f32, height as f32],
                    depth_range: 0.0..=1.0,
                }],
            )
                .expect("set_viewport failed")
                .bind_pipeline_graphics(pipeline.clone())
                .expect("bind_pipeline_graphics failed");

            // ── Draw each mesh ────────────────────────────────────────────
            for mesh in gpu_meshes {
                builder
                    .push_constants(
                        pipeline.layout().clone(),
                        0,
                        MvpPushConstants { mvp },
                    )
                    .expect("push_constants failed")
                    .bind_vertex_buffers(0, mesh.vertex_buffer.clone())
                    .expect("bind_vertex_buffers failed")
                    .bind_index_buffer(mesh.index_buffer.clone())
                    .expect("bind_index_buffer failed");
                // Safety: vertex/index buffers are compatible with the bound
                // pipeline; index_count fits within the uploaded index slice.
                unsafe {
                    builder
                        .draw_indexed(mesh.index_count, 1, 0, 0, 0)
                        .expect("draw_indexed failed");
                }
            }

            // ── Dynamic rendering end ─────────────────────────────────────────
            builder.end_rendering().expect("end_rendering failed");

            builder.build().expect("Failed to build command buffer")
        })
        .collect()
}
