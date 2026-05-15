//! Minimal raw-Vulkan swapchain renderer.
//!
//! This module replaces `vulkano_util::renderer::VulkanoWindowRenderer` on the
//! hot path. The high-level renderer's `acquire` / `present` flow allocates
//! several `Box<dyn GpuFuture>` trampolines per frame, wraps the cmd-buffer
//! submission in a `FenceSignalFuture`, and then performs a *second*
//! `then_signal_fence_and_flush` after `then_swapchain_present` — the latter
//! requires an extra empty `vkQueueSubmit` per frame because
//! `vkQueuePresentKHR` cannot signal a fence in core Vulkan.
//!
//! The implementation here drops down to `Queue::submit_unchecked` and
//! `Queue::present_unchecked`, drives sync with a small pool of pre-allocated
//! fences and semaphores, and emits exactly **one `vkQueueSubmit` + one
//! `vkQueuePresentKHR`** per frame — saving an entire syscall.

use std::sync::Arc;

use vulkano::{
    Validated, VulkanError,
    command_buffer::{
        CommandBufferSubmitInfo, PrimaryAutoCommandBuffer, SemaphoreSubmitInfo, SubmitInfo,
    },
    device::{Device, Queue},
    format::Format,
    image::{Image, ImageUsage, view::ImageView},
    instance::Instance,
    swapchain::{
        AcquireNextImageInfo, AcquiredImage, ColorSpace, PresentInfo, PresentMode,
        SemaphorePresentInfo, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{
        PipelineStages,
        fence::{Fence, FenceCreateFlags, FenceCreateInfo},
        semaphore::Semaphore,
    },
};
use winit::window::Window;

/// Per-acquire handles that are passed back to [`SwapchainRenderer::submit_and_present`].
///
/// `in_flight` is the **per-image** fence — it is reset by `acquire` and will
/// be signaled when the submitted command buffer completes. The host can
/// safely mutate any per-image resources (staging buffers, descriptor sets,
/// depth views, etc.) once this fence is signaled.
pub(crate) struct AcquiredFrame {
    pub image_index:    u32,
    pub image_available: Arc<Semaphore>,
    pub render_finished: Arc<Semaphore>,
    pub in_flight:       Arc<Fence>,
}

/// Optional pre-batch submitted as batch 0 of `submit_and_present`'s
/// single `vkQueueSubmit2`. Used by ADR-0003 to submit the shared
/// scatter-primary CB ahead of the per-image FrameSlot primary, with
/// the compute timeline signal attached.
pub(crate) struct PreBatch {
    pub cmd_buffer:        Arc<PrimaryAutoCommandBuffer>,
    pub signal_semaphores: Vec<SemaphoreSubmitInfo>,
}

pub(crate) struct SwapchainRenderer {
    device:           Arc<Device>,
    queue:            Arc<Queue>,
    window:           Arc<Window>,
    swapchain:        Arc<Swapchain>,
    image_views:      Vec<Arc<ImageView>>,
    present_mode:     PresentMode,
    needs_recreate:   bool,
    /// Pool of `max_frames` semaphores cycled by `next_acquire`. The image
    /// index isn't known before acquire, so this MUST be a separate pool
    /// from anything keyed by image-index.
    image_available:  Vec<Arc<Semaphore>>,
    /// **Per-swapchain-image**: signaled by the submit for that image,
    /// waited on by the host before re-using the image's per-image
    /// resources (staging buffer, reusable command buffer, depth view).
    in_flight:        Vec<Arc<Fence>>,
    /// Per-swapchain-image: signaled by submit, waited on by present. Must be
    /// keyed by image-index because the present queue takes the wait per
    /// presented image.
    render_finished:  Vec<Arc<Semaphore>>,
    /// Cycles through `image_available` only.
    next_acquire:     usize,
    max_frames:       usize,
}

impl SwapchainRenderer {
    /// Build a swapchain + per-frame sync primitives for a freshly-created window.
    pub fn new(
        instance:     Arc<Instance>,
        device:       Arc<Device>,
        queue:        Arc<Queue>,
        window:       Window,
        present_mode: PresentMode,
        max_frames:   usize,
    ) -> Self {
        let window  = Arc::new(window);
        let surface =
            Surface::from_window(instance, window.clone()).expect("Surface::from_window failed");

        let (swapchain, images) =
            create_swapchain(&device, surface, &window, present_mode, max_frames);
        let image_views = make_views(&images);

        // Per-image semaphores for present synchronization.
        let render_finished: Vec<_> = (0..images.len())
            .map(|_| Arc::new(Semaphore::from_pool(device.clone()).unwrap()))
            .collect();

        // Per-image in-flight fences. Pre-signaled so the first acquire
        // returns immediately for every slot.
        let in_flight: Vec<_> = (0..images.len())
            .map(|_| Arc::new(make_signaled_fence(device.clone())))
            .collect();

        // Pool of acquire semaphores. Sized for max_frames-in-flight CPU
        // pipelining; cycled independently of image index.
        let image_available: Vec<_> = (0..max_frames)
            .map(|_| Arc::new(Semaphore::from_pool(device.clone()).unwrap()))
            .collect();

        Self {
            device,
            queue,
            window,
            swapchain,
            image_views,
            present_mode,
            needs_recreate: false,
            image_available,
            in_flight,
            render_finished,
            next_acquire: 0,
            max_frames,
        }
    }

    pub fn image_views(&self) -> &[Arc<ImageView>] { &self.image_views }
    pub fn swapchain_format(&self) -> Format { self.swapchain.image_format() }
    #[allow(dead_code)]
    pub fn image_count(&self) -> usize { self.image_views.len() }
    #[allow(dead_code)]
    pub fn surface(&self) -> &Arc<Surface> { self.swapchain.surface() }
    #[allow(dead_code)]
    pub fn window(&self) -> &Arc<Window> { &self.window }

    #[allow(dead_code)]
    pub fn set_present_mode(&mut self, mode: PresentMode) {
        if self.present_mode != mode {
            self.present_mode  = mode;
            self.needs_recreate = true;
        }
    }

    pub fn resize(&mut self) { self.needs_recreate = true; }

    /// Acquire the next swapchain image. If the swapchain was out-of-date this
    /// transparently recreates it, calling `on_recreate` with the fresh image
    /// views, and returns `None` for this frame (the caller should simply skip
    /// it).
    pub fn acquire(
        &mut self,
        on_recreate: impl FnOnce(&[Arc<ImageView>]),
    ) -> Option<AcquiredFrame> {
        if self.needs_recreate && !self.recreate(on_recreate) {
            return None;
        }

        // Pick the next acquire-semaphore from the pool. This is independent
        // of image_index — the swapchain decides which image we get.
        let image_available = self.image_available[self.next_acquire].clone();
        self.next_acquire = (self.next_acquire + 1) % self.max_frames;

        let acquired = unsafe {
            self.swapchain.acquire_next_image(&AcquireNextImageInfo {
                semaphore: Some(image_available.clone()),
                ..Default::default()
            })
        };
        let AcquiredImage { image_index, is_suboptimal } = match acquired {
            Ok(a) => a,
            Err(Validated::Error(VulkanError::OutOfDate)) => {
                self.needs_recreate = true;
                return None;
            }
            Err(e) => panic!("acquire_next_image failed: {e:?}"),
        };
        if is_suboptimal {
            self.needs_recreate = true;
        }

        let in_flight = self.in_flight[image_index as usize].clone();
        // Wait for this image's previous submission to drain before the
        // host touches any of its per-image resources (staging buffer,
        // reusable CB, depth view).
        in_flight.wait(None).expect("fence wait failed");
        unsafe { in_flight.reset_unchecked() }.expect("fence reset failed");

        let render_finished = self.render_finished[image_index as usize].clone();

        Some(AcquiredFrame { image_index, image_available, render_finished, in_flight })
    }

    /// Submit one or two pre-recorded primary command buffers and present
    /// the resulting swapchain image. Both batches go into a single
    /// `vkQueueSubmit2` call (still one syscall per frame) followed by
    /// one `vkQueuePresentKHR`.
    ///
    /// `pre_batch`, if provided, is submitted as **batch 0** with no
    /// waits and the supplied signal semaphores (typically the
    /// `compute_timeline` signal from ADR-0003). It runs ahead of the
    /// main batch in queue order and any cross-batch memory dependency
    /// must be expressed via the main batch's `extra_main_waits`
    /// (semaphore signal/wait pairs establish both execution and memory
    /// dependency).
    ///
    /// `cmd_buffer` is **batch 1** — the per-image FrameSlot primary.
    /// It always waits on the `image_available` semaphore at
    /// `COLOR_ATTACHMENT_OUTPUT` (so render only starts once the
    /// swapchain image is owned by the queue) plus any `extra_main_waits`
    /// the caller supplies; it always signals `render_finished` at
    /// `COLOR_ATTACHMENT_OUTPUT` plus any `extra_main_signals`. The
    /// per-image `in_flight` fence is signaled at end-of-submit (after
    /// both batches), so the caller can use it to gate reuse of any
    /// per-image resource referenced by either batch.
    pub fn submit_and_present(
        &mut self,
        frame:                  AcquiredFrame,
        pre_batch:              Option<PreBatch>,
        cmd_buffer:             Arc<PrimaryAutoCommandBuffer>,
        extra_main_waits:       Vec<SemaphoreSubmitInfo>,
        extra_main_signals:     Vec<SemaphoreSubmitInfo>,
    ) {
        let AcquiredFrame { image_index, image_available, render_finished, in_flight } = frame;

        // ── Build the (up to two) submit batches ─────────────────────────────
        let mut submit_infos: Vec<SubmitInfo> = Vec::with_capacity(2);

        if let Some(pre) = pre_batch {
            submit_infos.push(SubmitInfo {
                wait_semaphores:   Vec::new(),
                command_buffers:   vec![CommandBufferSubmitInfo::new(pre.cmd_buffer)],
                signal_semaphores: pre.signal_semaphores,
                ..Default::default()
            });
        }

        let mut main_waits = Vec::with_capacity(1 + extra_main_waits.len());
        main_waits.push(SemaphoreSubmitInfo {
            stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
            ..SemaphoreSubmitInfo::new(image_available)
        });
        main_waits.extend(extra_main_waits);

        let mut main_signals = Vec::with_capacity(1 + extra_main_signals.len());
        main_signals.push(SemaphoreSubmitInfo {
            // Signal at end of the color output stage — that's when the
            // image is ready to be presented.
            stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
            ..SemaphoreSubmitInfo::new(render_finished.clone())
        });
        main_signals.extend(extra_main_signals);

        submit_infos.push(SubmitInfo {
            wait_semaphores:   main_waits,
            command_buffers:   vec![CommandBufferSubmitInfo::new(cmd_buffer)],
            signal_semaphores: main_signals,
            ..Default::default()
        });

        // ── Submit ─ single vkQueueSubmit2 with both batches ──────────────────
        self.queue
            .with(|mut g| unsafe { g.submit_unchecked(&submit_infos, Some(&in_flight)) })
            .expect("submit_unchecked failed");

        // ── Present ──────────────────────────────────────────────────────────
        let present_info = PresentInfo {
            wait_semaphores: vec![SemaphorePresentInfo::new(render_finished)],
            swapchain_infos: vec![SwapchainPresentInfo::swapchain_image_index(
                self.swapchain.clone(),
                image_index,
            )],
            ..Default::default()
        };

        let result = self
            .queue
            .with(|mut g| unsafe { g.present_unchecked(&present_info) });

        match result {
            Ok(iter) => {
                for r in iter {
                    match r {
                        Ok(suboptimal) => {
                            if suboptimal {
                                self.needs_recreate = true;
                            }
                        }
                        Err(VulkanError::OutOfDate) => self.needs_recreate = true,
                        Err(e) => panic!("present failed: {e:?}"),
                    }
                }
            }
            Err(VulkanError::OutOfDate) => self.needs_recreate = true,
            Err(e) => panic!("present_unchecked failed: {e:?}"),
        }

    }

    /// Recreate the swapchain to match the window's current size and present
    /// mode. Returns `false` if the window is currently zero-sized
    /// (minimised); the caller should skip this frame.
    fn recreate(&mut self, on_recreate: impl FnOnce(&[Arc<ImageView>])) -> bool {
        let extent: [u32; 2] = self.window.inner_size().into();
        if extent.contains(&0) {
            return false;
        }

        // Drain in-flight work referencing the old images / depth views.
        for fence in &self.in_flight {
            let _ = fence.wait(None);
        }

        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: extent,
                present_mode: self.present_mode,
                ..self.swapchain.create_info()
            })
            .expect("swapchain recreate failed");

        self.swapchain   = new_swapchain;
        self.image_views = make_views(&new_images);

        // If the new swapchain has a different image count, rebuild
        // per-image semaphores AND per-image fences to match.
        if self.render_finished.len() != self.image_views.len() {
            self.render_finished = (0..self.image_views.len())
                .map(|_| Arc::new(Semaphore::from_pool(self.device.clone()).unwrap()))
                .collect();
            self.in_flight = (0..self.image_views.len())
                .map(|_| Arc::new(make_signaled_fence(self.device.clone())))
                .collect();
        }

        self.needs_recreate = false;
        on_recreate(&self.image_views);
        true
    }
}

fn create_swapchain(
    device:       &Arc<Device>,
    surface:      Arc<Surface>,
    window:       &Window,
    present_mode: PresentMode,
    max_frames:   usize,
) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    let caps = device
        .physical_device()
        .surface_capabilities(&surface, Default::default())
        .expect("surface_capabilities failed");

    let (image_format, _color_space) = device
        .physical_device()
        .surface_formats(&surface, Default::default())
        .expect("surface_formats failed")[0];

    let composite_alpha = caps
        .supported_composite_alpha
        .into_iter()
        .next()
        .expect("no supported composite alpha");

    let min_image_count = caps.min_image_count.max(max_frames as u32);

    let create_info = SwapchainCreateInfo {
        min_image_count,
        image_format,
        image_color_space: ColorSpace::SrgbNonLinear,
        image_extent: window.inner_size().into(),
        // The renderer doesn't draw into swapchain images directly today —
        // each frame's CB renders into camera-owned offscreen color/depth
        // attachments and `vkCmdBlitImage`s the camera color into the
        // swapchain image. We still request `COLOR_ATTACHMENT` because
        // (a) `ImageView::new_default` rejects images without one of the
        // "view-compatible" usages, and (b) a future fullscreen present
        // pass (HDR → sRGB tonemap, post-process) will draw into it.
        image_usage: ImageUsage::TRANSFER_DST | ImageUsage::COLOR_ATTACHMENT,
        composite_alpha,
        present_mode,
        ..Default::default()
    };

    Swapchain::new(device.clone(), surface, create_info).expect("Swapchain::new failed")
}

fn make_signaled_fence(device: Arc<Device>) -> Fence {
    Fence::new(
        device,
        FenceCreateInfo {
            flags: FenceCreateFlags::SIGNALED,
            ..Default::default()
        },
    )
    .unwrap()
}

fn make_views(images: &[Arc<Image>]) -> Vec<Arc<ImageView>> {
    images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).expect("ImageView::new_default failed"))
        .collect()
}
