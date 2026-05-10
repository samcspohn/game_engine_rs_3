//! Vulkano-based renderer and windowing for the game engine.
//!
//! This crate exposes a single public type, [`Window`], which manages:
//! * An OS window (via `winit 0.30`).
//! * A Vulkan instance, physical device, logical device, and swapchain (via
//!   `vulkano 0.35` and `vulkano-util 0.35`).
//! * A per-frame render loop that clears the swapchain image to a solid color.
//!
//! ## Optimizations (ported from `ash_test`)
//!
//! | Optimization | Benefit |
//! |---|---|
//! | **Dynamic rendering** (Vulkan 1.3 / `VK_KHR_dynamic_rendering`) | Eliminates `RenderPass` & `Framebuffer` object allocation/synchronization overhead |
//! | **Triple-buffered swapchain** (`min_image_count = MAX_FRAMES_IN_FLIGHT`) | Keeps GPU pipeline full; CPU never stalls waiting for an available image |
//! | **Present mode: Immediate → Mailbox → Fifo** | Lowest display latency and uncapped frame rate |
//! | **Command-buffer pre-allocation** (`primary_buffer_count = 32`) | Avoids per-frame heap allocation from the pool |
//! | **FPS tracker frame-count gate** | `Instant::now()` called only every [`FRAMES_PER_FPS_SAMPLE`] frames (bitwise AND) instead of every frame |
//! | **Command buffer caching** | Command buffers are recorded once per swapchain image (`MultipleSubmit`) and resubmitted every frame; re-recorded only on swapchain recreation |
//!
//! Typical usage:
//! ```no_run
//! use engine_render::Window;
//! Window::new("My Game").run();
//! ```

use std::{sync::Arc, time::{Duration, Instant}};

use vulkano::{
    command_buffer::{
        allocator::{
            StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
        },
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderingAttachmentInfo, RenderingInfo,
    },
    device::DeviceFeatures,
    image::view::ImageView,
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{PresentMode, SurfaceInfo},
    sync::GpuFuture,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{WindowDescriptor, VulkanoWindows},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::WindowId,
};

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

/// Number of frames that can be in-flight on the GPU concurrently.
///
/// This is also passed as `min_image_count` when creating the swapchain so
/// the swapchain always has enough images to keep the pipeline full: while the
/// GPU is rendering frame N the CPU can freely record frame N+1 and N+2.
const MAX_FRAMES_IN_FLIGHT: usize = 4;

/// How often (in rendered frames) the FPS tracker reads the system clock.
///
/// **Must be a power of two** — the tracker uses a bitwise AND mask instead
/// of a modulo operation, so the compiler can emit a single `AND` instruction.
///
/// At 200 FPS this fires roughly every 2.5 s; at 60 FPS every ~8.5 s — a
/// reasonable diagnostic cadence without per-frame `Instant::now()` overhead.
const FRAMES_PER_FPS_SAMPLE: u32 = 1024;

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

/// An OS window backed by a Vulkan swapchain.
///
/// Build one with [`Window::new`], then drive the event loop by calling
/// [`Window::run`].  The window renders a solid cornflower-blue clear on every
/// frame using dynamic rendering (no `RenderPass`/`Framebuffer` objects).
pub struct Window {
    title: String,
}

impl Window {
    /// Create a window descriptor with the given title string.
    pub fn new(title: &str) -> Self {
        Window {
            title: title.to_string(),
        }
    }

    /// Open the OS window, initialize Vulkan, and block on the event loop.
    ///
    /// # Behavior
    /// * Each frame: acquires a swapchain image and clears it to cornflower
    ///   blue (`#6495ED`) via dynamic rendering.
    /// * `CloseRequested` event: exits the loop cleanly.
    /// * Window resize: recreates the swapchain and refreshes image views.
    ///
    /// # Panics
    /// Panics with a descriptive message if Vulkan setup fails (no compatible
    /// GPU, missing Vulkan 1.3 / `VK_KHR_dynamic_rendering` support, etc.).
    pub fn run(self) {
        let event_loop =
            EventLoop::new().expect("Failed to create winit EventLoop");
        let mut app = RenderApp::new(self.title);
        event_loop
            .run_app(&mut app)
            .expect("Event loop exited with an error");
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Internal implementation
// ────────────────────────────────────────────────────────────────────────────

/// Tracks rendered frames and prints FPS + frame-time to stdout periodically.
///
/// The frame-count gate means [`Instant::now`] — a syscall on many platforms —
/// is only invoked once every [`FRAMES_PER_FPS_SAMPLE`] frames rather than on
/// every single frame.
struct FpsTracker {
    /// Timestamp of the last FPS print.
    last_print: Instant,
    /// Frames rendered since `last_print`.
    frame_count: u32,
}

impl FpsTracker {
    fn new() -> Self {
        Self {
            last_print: Instant::now(),
            frame_count: 0,
        }
    }

    /// Call once per rendered frame.
    ///
    /// Prints `FPS: <n>  (<ms>/frame)` when [`FRAMES_PER_FPS_SAMPLE`] frames
    /// have accumulated **and** at least one second has elapsed since the last
    /// print.
    fn tick(&mut self) {
        self.frame_count += 1;
        // Bit-mask gate: evaluate Instant::now() only every FRAMES_PER_FPS_SAMPLE
        // frames.  FRAMES_PER_FPS_SAMPLE is a power of two so this compiles to
        // a single AND instruction instead of a division.
        if self.frame_count & (FRAMES_PER_FPS_SAMPLE - 1) == 0 {
            let elapsed = self.last_print.elapsed();
            if elapsed.as_secs() >= 1 {
                let fps = self.frame_count as f64 / elapsed.as_secs_f64();
                println!("FPS: {:.0}  ({:.3} ms/frame)", fps, 1000.0 / fps);
                self.frame_count = 0;
                self.last_print = Instant::now();
            }
        }
    }
}

/// State shared across frames, alive for the entire event-loop lifetime.
struct RenderApp {
    title: String,
    context: VulkanoContext,
    windows: VulkanoWindows,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    fps: FpsTracker,
    /// Per-window render state; populated inside `resumed()`.
    rcx: Option<RenderContext>,
}

/// Swapchain image views and cached command buffers for dynamic rendering.
///
/// Dynamic rendering (`VK_KHR_dynamic_rendering` / Vulkan 1.3) removes the
/// need for `RenderPass` and `Framebuffer` objects entirely.  The raw
/// `ImageView` handles and pre-recorded command buffers are both rebuilt
/// whenever the swapchain is recreated (e.g. on window resize), but are
/// otherwise reused every frame with zero re-recording overhead.
struct RenderContext {
    /// One view per swapchain image.  Updated inside the `acquire()` callback
    /// whenever the swapchain is recreated.
    attachment_image_views: Vec<Arc<ImageView>>,
    /// One pre-recorded command buffer per swapchain image.
    ///
    /// Recorded with [`CommandBufferUsage::SimultaneousUse`] so the exact same
    /// `Arc` is passed to `then_execute` on every frame without any builder
    /// allocation, command recording, or `build()` call in the hot path.
    ///
    /// `SimultaneousUse` (rather than `MultipleSubmit`) is required because
    /// with triple buffering the GPU can have two submissions for the same
    /// image index truly in-flight at the same time — the presentation engine
    /// may release image N before the GPU finishes the command buffer that
    /// rendered into it.  `MultipleSubmit` would hit
    /// `VUID-vkQueueSubmit2-commandBuffer-03875`.
    ///
    /// Rebuilt together with `attachment_image_views` on swapchain recreation.
    cached_command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
}

impl RenderApp {
    fn new(title: String) -> Self {
        // ── Vulkan device setup ───────────────────────────────────────────────
        // Enable dynamic rendering (Vulkan 1.3 core promoted from
        // VK_KHR_dynamic_rendering).  This lets us issue draw calls without
        // allocating RenderPass or Framebuffer descriptors, removing that
        // per-resize overhead entirely.
        let context = VulkanoContext::new(VulkanoConfig {
            device_features: DeviceFeatures {
                dynamic_rendering: true,
                ..Default::default()
            },
            ..Default::default()
        });

        let windows = VulkanoWindows::default();

        // Pre-allocate 32 primary command-buffer slots to avoid per-frame pool
        // allocation; secondary buffers are unused at this stage.
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            StandardCommandBufferAllocatorCreateInfo {
                primary_buffer_count: 32,
                secondary_buffer_count: 0,
                ..Default::default()
            },
        ));

        RenderApp {
            title,
            context,
            windows,
            command_buffer_allocator,
            fps: FpsTracker::new(),
            rcx: None,
        }
    }
}

impl ApplicationHandler for RenderApp {
    /// Called when the event loop is ready to display a window.
    ///
    /// On desktop this fires once at startup.  On Android/iOS it may fire
    /// multiple times (resume/pause cycles), so we remove any stale renderer
    /// before creating a fresh one.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Remove stale renderer if resuming from a pause (mobile lifecycle).
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
            // Request at least MAX_FRAMES_IN_FLIGHT swapchain images (triple-
            // buffering) so the CPU can record the next frame while the GPU is
            // still executing the previous one, eliminating pipeline bubbles.
            |sc| {
                sc.min_image_count = sc.min_image_count.max(MAX_FRAMES_IN_FLIGHT as u32);
            },
        );

        // ── Initial image views ───────────────────────────────────────────────
        // With dynamic rendering there are no framebuffers to construct; just
        // snapshot the current swapchain image views.
        let renderer = self
            .windows
            .get_primary_renderer_mut()
            .expect("Primary renderer must exist right after create_window");

        let attachment_image_views = renderer.swapchain_image_views().to_vec();
        let cached_command_buffers = build_command_buffers(
            &self.command_buffer_allocator,
            self.context.graphics_queue().queue_family_index(),
            &attachment_image_views,
        );
        self.rcx = Some(RenderContext {
            attachment_image_views,
            cached_command_buffers,
        });

        // ── Present-mode selection ────────────────────────────────────────────
        // Priority: Immediate (lowest latency, uncapped FPS) → Mailbox
        // (low-latency, tear-free) → Fifo (V-Sync fallback).
        //
        // Note: the original engine preferred Mailbox first; the reference
        // (ash_test) prefers Immediate for maximum throughput.
        let surface = renderer.surface();
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

        println!(
            "Present mode: {:?}  (supported: {:?})",
            chosen, supported
        );
        renderer.set_present_mode(chosen);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let renderer = match self.windows.get_primary_renderer_mut() {
            Some(r) => r,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            // Resize: mark the swapchain as outdated; it will be transparently
            // recreated on the next call to `acquire()`.
            WindowEvent::Resized(_) => {
                renderer.resize();
            }

            WindowEvent::RedrawRequested => {
                // Rendering is driven by `about_to_wait` in Poll mode.
                // OS-generated redraws (window expose, etc.) are satisfied
                // by the next `about_to_wait` tick and need no special handling
                // here.
            }

            _ => {}
        }
    }

    /// Render a frame and keep the loop spinning at full speed.
    ///
    /// `about_to_wait` is called at the end of every event-loop iteration,
    /// after all pending events have been dispatched.  By setting
    /// [`ControlFlow::Poll`] here the OS never blocks the loop waiting for
    /// new input, giving us the lowest possible frame latency without relying
    /// on `request_redraw()` as a synthetic event driver.
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Keep the event loop spinning unconditionally — no OS sleep between
        // frames.  This is set every iteration so it survives any internal
        // winit reset (e.g. on mobile resume/pause cycles).
        event_loop.set_control_flow(ControlFlow::Poll);

        let renderer = match self.windows.get_primary_renderer_mut() {
            Some(r) => r,
            None => return,
        };
        let rcx = match self.rcx.as_mut() {
            Some(r) => r,
            None => return,
        };

        // Skip rendering while the window is zero-sized (minimized on Windows).
        let size = renderer.window().inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        // Pre-capture the allocator and queue family index so the acquire()
        // closure can rebuild the command buffer cache on swapchain recreation
        // without holding a borrow on `self`.
        let allocator          = self.command_buffer_allocator.clone();
        let queue_family_index = self.context.graphics_queue().queue_family_index();

        // Acquire the next swapchain image.  The callback fires only when the
        // swapchain is recreated (resize); we rebuild the image-view slice and
        // command buffer cache so the new images are targeted correctly.
        let previous_frame_end = renderer
            .acquire(Some(Duration::from_millis(1_000)), |swapchain_images| {
                rcx.attachment_image_views = swapchain_images.to_vec();
                rcx.cached_command_buffers = build_command_buffers(
                    &allocator,
                    queue_family_index,
                    swapchain_images,
                );
            })
            .expect("Failed to acquire swapchain image");

        // Hot path: just clone the Arc — no builder, no recording, no build().
        // The command buffer for this image was pre-recorded at startup and
        // after each swapchain recreation.
        let command_buffer =
            rcx.cached_command_buffers[renderer.image_index() as usize].clone();

        let future = previous_frame_end
            .then_execute(self.context.graphics_queue().clone(), command_buffer)
            .expect("Failed to submit command buffer")
            .boxed();

        renderer.present(future, false);

        self.fps.tick();
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/// Record one [`PrimaryAutoCommandBuffer`] per swapchain image.
///
/// Each buffer clears its target image to cornflower blue (`#6495ED`) using
/// dynamic rendering. They are built with [`CommandBufferUsage::MultipleSubmit`]
/// so the same `Arc` values can be passed to `then_execute` on every frame
/// without any re-recording work in the hot path.
///
/// `SimultaneousUse` is chosen over `MultipleSubmit` because with
/// `MAX_FRAMES_IN_FLIGHT = 3` and a triple-buffered swapchain the GPU can have
/// two submissions for the same image index genuinely in-flight concurrently.
/// `MultipleSubmit` would trigger `VUID-vkQueueSubmit2-commandBuffer-03875`.
///
/// This function is called:
/// * Once at startup inside [`RenderApp::resumed`].
/// * Again inside the [`acquire`][vulkano_util::renderer::VulkanoWindowRenderer::acquire]
///   callback whenever the swapchain is recreated (e.g. window resize),
///   because the underlying image handles change.
fn build_command_buffers(
    allocator: &Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
    attachment_image_views: &[Arc<ImageView>],
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    attachment_image_views
        .iter()
        .map(|image_view| {
            let mut builder = AutoCommandBufferBuilder::primary(
                allocator.clone(),
                queue_family_index,
                // SimultaneousUse is required here — not MultipleSubmit.
                //
                // With MAX_FRAMES_IN_FLIGHT = 3 and a triple-buffered swapchain
                // the GPU can genuinely have two submissions for the same image
                // index in-flight at the same time: the presentation engine can
                // release image N back to the acquire pool before the GPU has
                // finished executing the command buffer that was submitted with
                // it.  MultipleSubmit would panic (VUID-vkQueueSubmit2-
                // commandBuffer-03875); SimultaneousUse explicitly opts in to
                // this overlap, which is safe because each buffer only writes
                // to its own dedicated image view.
                CommandBufferUsage::SimultaneousUse,
            )
            .expect("Failed to create command buffer builder");

            builder
                .begin_rendering(RenderingInfo {
                    color_attachments: vec![Some(RenderingAttachmentInfo {
                        load_op:  AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        // Cornflower blue: R=0.39, G=0.58, B=0.93
                        clear_value: Some([0.39, 0.58, 0.93, 1.0].into()),
                        ..RenderingAttachmentInfo::image_view(image_view.clone())
                    })],
                    ..Default::default()
                })
                .expect("begin_rendering failed")
                .end_rendering()
                .expect("end_rendering failed");

            builder.build().expect("Failed to build cached command buffer")
        })
        .collect()
}
