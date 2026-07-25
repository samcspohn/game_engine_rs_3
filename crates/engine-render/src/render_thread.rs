//! The background render thread.
//!
//! Owns everything that used to hang off `RenderApp`/`RenderContext` on the
//! event-loop thread: the [`SwapchainRenderer`], [`WorldTransformGpu`], the
//! main [`RenderCamera`], the per-image [`crate::FrameSlot`]s, the asset
//! stores, [`GpuRenderers`], and [`FrameStats`] (almost all of what it times
//! is background-thread-side work now). It's spawned once from
//! `RenderApp::resumed()` and runs for the lifetime of the window.
//!
//! Per [`FrameJob`] popped off the channel, this is today's
//! `about_to_wait` body (acquire → capacity/camera/mesh-sync rebuilds →
//! GPU-staging memcpy → submit + present) relocated verbatim, just reading
//! `FrameJob` fields (`entity_count`, `f8_pressed`/`f9_pressed`,
//! `view_proj_cols`) instead of `self.root_scene` / `input::key_pressed`,
//! and calling [`FrameHandoff::consume`] instead of writing the harvest
//! itself. The main thread's gate on reusing its CPU staging buffer
//! (`FrameHandoff::wait_for_buffer_free`) is satisfied by `consume` as soon
//! as the memcpy is done — before submit — preserving the early-wake
//! property the old `gpu_signal` poll had.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::{Device, Queue};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline};
use vulkano::query::QueryResultFlags;

use crate::assets::{GpuMaterialStore, GpuMeshStore, GpuTextureStore};
use crate::camera::{CameraSceneResources, DrawPlan, RenderCamera};
use crate::frame_handoff::{FrameHandoff, FrameJob};
use crate::gpu_renderers::GpuRenderers;
use crate::swapchain::SwapchainRenderer;
use crate::transform_gpu::WorldTransformGpu;
use crate::{build_all_frame_slots, build_draw_plan, FrameSlot, FrameStats, GPU_TS_COUNT};

/// Swapchain-image-count-sized arrays rebuilt on every swapchain recreation
/// — exclusively background-thread-owned state (moved here verbatim from
/// `lib.rs`; see that module's history for the per-field rationale).
pub(crate) struct RenderContext {
    pub(crate) swapchain_image_views: Vec<Arc<ImageView>>,
    pub(crate) world_transforms: WorldTransformGpu,
    pub(crate) main_camera: RenderCamera,
    pub(crate) frame_slots: Vec<FrameSlot>,
    pub(crate) gpu_mesh_store: GpuMeshStore,
    pub(crate) gpu_texture_store: GpuTextureStore,
    pub(crate) gpu_material_store: GpuMaterialStore,
    pub(crate) gpu_renderers: GpuRenderers,
}

/// Everything the background thread needs, built on the main thread during
/// `resumed()` (window/device/pipeline/initial-scene setup stays there —
/// one-time cost, not per-frame) and then moved wholesale into the spawned
/// thread's closure.
pub(crate) struct RenderThreadInit {
    pub(crate) device: Arc<Device>,
    pub(crate) graphics_queue: Arc<Queue>,
    pub(crate) command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub(crate) memory_allocator: Arc<StandardMemoryAllocator>,
    pub(crate) descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub(crate) swapchain_renderer: SwapchainRenderer,
    pub(crate) pipeline: Arc<GraphicsPipeline>,
    pub(crate) mvp_build_pass2_pipeline: Arc<ComputePipeline>,
    pub(crate) cull_pass2_args_pipeline: Arc<ComputePipeline>,
    pub(crate) hiz_reduce_depth_pipeline: Arc<ComputePipeline>,
    pub(crate) hiz_reduce_mip_pipeline: Arc<ComputePipeline>,
    pub(crate) hiz_reduce_mip2_pipeline: Arc<ComputePipeline>,
    pub(crate) rcx: RenderContext,
    pub(crate) initial_entity_count: usize,
    pub(crate) initial_extent: [u32; 2],
}

struct RenderThreadState {
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    swapchain_renderer: SwapchainRenderer,
    pipeline: Arc<GraphicsPipeline>,
    mvp_build_pass2_pipeline: Arc<ComputePipeline>,
    cull_pass2_args_pipeline: Arc<ComputePipeline>,
    hiz_reduce_depth_pipeline: Arc<ComputePipeline>,
    hiz_reduce_mip_pipeline: Arc<ComputePipeline>,
    hiz_reduce_mip2_pipeline: Arc<ComputePipeline>,
    rcx: RenderContext,
    fps: FrameStats,
    total_frames: u64,
}

/// Main-thread-side handle: sends [`FrameJob`]s, exposes the
/// [`FrameHandoff`] wait/harvest API, and lets `window_event`'s resize
/// handler flag a pending resize without touching `SwapchainRenderer`
/// directly (it's background-thread-owned now).
pub(crate) struct RenderThreadHandle {
    job_tx: Option<mpsc::Sender<FrameJob>>,
    handoff: Arc<FrameHandoff>,
    pending_resize: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl RenderThreadHandle {
    /// Spawns the render loop on its own dedicated OS thread — it owns
    /// `SwapchainRenderer` (Vulkan requires a single thread to own
    /// `acquire`/`submit_and_present`) and blocks indefinitely on
    /// `job_rx.recv()` between frames, so it does *not* run as a
    /// `parallel::global::spawn_background` task on the shared pool (that
    /// pool's own docs: a task that blocks indefinitely "strands a compute
    /// core" — belongs on a dedicated thread instead). See
    /// `frame_handoff::copy_cpu_staging_into_gpu`'s dedicated
    /// `staging_copy_pool` for how it gets safe, contention-free
    /// parallelism of its own without touching the main thread's
    /// `parallel::global` pool.
    pub(crate) fn spawn(init: RenderThreadInit) -> Self {
        // Eagerly resolve + validate `ENGINE_STAGING_COPY_THREADS` here, on
        // the main thread, instead of lazily on the background thread's
        // first frame — a bad env var then fails immediately and loudly at
        // startup instead of only surfacing once rendering begins.
        let _ = crate::frame_handoff::staging_copy_pool();

        let handoff = Arc::new(FrameHandoff::new(
            init.initial_entity_count,
            init.initial_extent,
        ));
        let pending_resize = Arc::new(AtomicBool::new(false));
        let (job_tx, job_rx) = mpsc::channel::<FrameJob>();

        let state = RenderThreadState {
            device: init.device,
            graphics_queue: init.graphics_queue,
            command_buffer_allocator: init.command_buffer_allocator,
            memory_allocator: init.memory_allocator,
            descriptor_set_allocator: init.descriptor_set_allocator,
            swapchain_renderer: init.swapchain_renderer,
            pipeline: init.pipeline,
            mvp_build_pass2_pipeline: init.mvp_build_pass2_pipeline,
            cull_pass2_args_pipeline: init.cull_pass2_args_pipeline,
            hiz_reduce_depth_pipeline: init.hiz_reduce_depth_pipeline,
            hiz_reduce_mip_pipeline: init.hiz_reduce_mip_pipeline,
            hiz_reduce_mip2_pipeline: init.hiz_reduce_mip2_pipeline,
            rcx: init.rcx,
            fps: FrameStats::new(),
            total_frames: 0,
        };

        let thread_handoff = handoff.clone();
        let thread_pending_resize = pending_resize.clone();
        let join = std::thread::Builder::new()
            .name("render".into())
            .spawn(move || run(state, job_rx, thread_handoff, thread_pending_resize))
            .expect("failed to spawn background render thread");

        Self {
            job_tx: Some(job_tx),
            handoff,
            pending_resize,
            join: Some(join),
        }
    }

    pub(crate) fn published_entity_capacity(&self) -> usize {
        self.handoff.published_entity_capacity()
    }

    pub(crate) fn published_extent(&self) -> [u32; 2] {
        self.handoff.published_extent()
    }

    /// Main-thread-only. Waits for the previous handoff to be consumed and
    /// harvests this frame's dirty TRS into the shared CPU staging buffer.
    /// Returns `(ready_gen, main_wait_ns, cpu_staging_ns)` — the generation
    /// to stamp on this frame's [`FrameJob`], the wait's own wall time, and
    /// the harvest's own wall time.
    pub(crate) fn write_frame(
        &self,
        entity_capacity: usize,
        root_scene: Option<&engine_core::component::Scene>,
        view_proj_cols: [f32; 16],
    ) -> (u64, u64, u64) {
        self.handoff
            .write_frame(entity_capacity, root_scene, view_proj_cols)
    }

    /// Hand a frame's job to the background thread. The channel is
    /// unbounded and the background thread is always draining it, so this
    /// never blocks in practice.
    pub(crate) fn send(&self, job: FrameJob) {
        if let Some(tx) = &self.job_tx {
            let _ = tx.send(job);
        }
    }

    pub(crate) fn request_resize(&self) {
        self.pending_resize.store(true, Ordering::Relaxed);
    }
}

impl Drop for RenderThreadHandle {
    fn drop(&mut self) {
        // Dropping the sender first closes the channel; the background
        // thread's blocking `recv()` then returns `Err`, and its loop
        // exits after finishing whatever submission is already in flight.
        // Only then do we join, so no Vulkan object referenced by the
        // background thread's state is dropped while it might still be in
        // use.
        self.job_tx.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// The background thread's loop. One iteration per [`FrameJob`]; exits
/// cleanly when the channel closes (see [`RenderThreadHandle`]'s `Drop`).
fn run(
    mut state: RenderThreadState,
    job_rx: mpsc::Receiver<FrameJob>,
    handoff: Arc<FrameHandoff>,
    pending_resize: Arc<AtomicBool>,
) {
    while let Ok(job) = job_rx.recv() {
        let RenderThreadState {
            device,
            graphics_queue,
            command_buffer_allocator,
            memory_allocator,
            descriptor_set_allocator,
            swapchain_renderer,
            pipeline,
            mvp_build_pass2_pipeline,
            cull_pass2_args_pipeline,
            hiz_reduce_depth_pipeline,
            hiz_reduce_mip_pipeline,
            hiz_reduce_mip2_pipeline,
            rcx,
            fps,
            total_frames,
        } = &mut state;

        let queue_family_index = graphics_queue.queue_family_index();

        if pending_resize.swap(false, Ordering::Relaxed) {
            swapchain_renderer.resize();
        }

        let acquire_start = Instant::now();
        let frame = match swapchain_renderer.acquire(|swapchain_images| {
            rcx.swapchain_image_views = swapchain_images.to_vec();
            let new_extent = {
                let [w, h, _] = swapchain_images[0].image().extent();
                [w, h]
            };
            handoff.publish_extent(new_extent);
            let scene_resources = CameraSceneResources {
                cb_allocator: command_buffer_allocator,
                descriptor_set_allocator,
                memory_allocator,
                pipeline,
                queue_family_index,
                world_transforms: &rcx.world_transforms,
                mesh_store: &rcx.gpu_mesh_store,
                texture_store: &rcx.gpu_texture_store,
                material_store: &rcx.gpu_material_store,
                gpu_renderers: &rcx.gpu_renderers,
                mvp_build_pass2_pipeline,
                cull_pass2_args_pipeline,
                hiz_reduce_depth_pipeline,
                hiz_reduce_mip_pipeline,
                hiz_reduce_mip2_pipeline,
            };
            let _camera_rebuilt = rcx
                .main_camera
                .on_swapchain_resize(new_extent, &scene_resources);

            rcx.frame_slots.clear();
            rcx.frame_slots = build_all_frame_slots(
                command_buffer_allocator,
                memory_allocator,
                queue_family_index,
                &rcx.swapchain_image_views,
                &rcx.main_camera,
                &rcx.world_transforms,
                &rcx.gpu_renderers,
            );
        }) {
            Some(f) => f,
            None => continue, // out-of-date / minimised — skip this job
        };
        fps.record_acquire(acquire_start.elapsed().as_nanos() as u64);

        // GPU per-stage timestamps from this image's *previous* submission.
        {
            let pool = &rcx.frame_slots[frame.image_index as usize].timestamp_pool;
            let mut ticks = [0u64; GPU_TS_COUNT as usize];
            if let Ok(true) =
                pool.get_results(0..GPU_TS_COUNT, &mut ticks, QueryResultFlags::empty())
            {
                let period_ns = device.physical_device().properties().timestamp_period as f64;
                let delta = |a: usize, b: usize| -> u64 {
                    (ticks[b].saturating_sub(ticks[a]) as f64 * period_ns) as u64
                };
                fps.record_gpu_timestamps(&[
                    delta(0, 1),
                    delta(1, 2),
                    delta(2, 3),
                    delta(3, 4),
                    delta(4, 5),
                    delta(5, 6),
                    delta(6, 7),
                    delta(0, 7),
                ]);
            }
        }

        // ── World + renderer capacity (per-world axis) ──────────────────
        let mut need_frame_slot_rebuild = false;
        let grew_world = rcx
            .world_transforms
            .ensure_capacity(memory_allocator, job.entity_count.max(1));
        // NB: unlike the pre-background-thread version, there is no ECS
        // hierarchy handle here to re-mark dirty on a grow — `mark_all_trs`
        // is called from the main thread's next harvest instead, driven by
        // `handoff.published_entity_capacity()` picking up the larger
        // value published below. See `frame_handoff`'s module doc comment
        // ("Capacity growth").
        let renderer_capacity = rcx.world_transforms.entity_capacity();
        let grew_renderers = rcx.gpu_renderers.ensure_capacity(renderer_capacity as u32);
        let grew_parent_staging = rcx
            .world_transforms
            .ensure_parent_update_capacity(job.parent_updates.len());

        handoff.publish_entity_capacity(renderer_capacity);

        // ── Mesh sync + renderer scatter (Design B, GPU-driven) ──────────
        let (mesh_changed, slot_totals) = rcx.gpu_mesh_store.sync();
        let tex_changed = rcx.gpu_texture_store.sync();
        let mat_changed = rcx.gpu_material_store.sync();
        let grew_spawn_staging = rcx.gpu_renderers.ensure_spawn_capacity(job.spawns.len());

        let plan_dirty = !job.spawns.is_empty() || mesh_changed;
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
                    cb_allocator: command_buffer_allocator,
                    descriptor_set_allocator,
                    memory_allocator,
                    pipeline,
                    queue_family_index,
                    world_transforms: &rcx.world_transforms,
                    mesh_store: &rcx.gpu_mesh_store,
                    texture_store: &rcx.gpu_texture_store,
                    material_store: &rcx.gpu_material_store,
                    gpu_renderers: &rcx.gpu_renderers,
                    mvp_build_pass2_pipeline,
                    cull_pass2_args_pipeline,
                    hiz_reduce_depth_pipeline,
                    hiz_reduce_mip_pipeline,
                    hiz_reduce_mip2_pipeline,
                };
                rcx.main_camera
                    .ensure_current(&plan, renderer_capacity, &scene_resources);
                need_frame_slot_rebuild = true;
            } else {
                pending_cheap_plan = Some(plan);
            }
        }

        // Debug: F9 frustum-lock's one-frame-behind Hi-Z freeze application.
        if rcx.main_camera.apply_pending_hiz_freeze() {
            need_frame_slot_rebuild = true;
        }

        // Debug: F8 toggles occlusion culling entirely.
        if job.f8_pressed {
            let desired = !rcx.main_camera.occlusion_enabled();
            let scene_resources = CameraSceneResources {
                cb_allocator: command_buffer_allocator,
                descriptor_set_allocator,
                memory_allocator,
                pipeline,
                queue_family_index,
                world_transforms: &rcx.world_transforms,
                mesh_store: &rcx.gpu_mesh_store,
                texture_store: &rcx.gpu_texture_store,
                material_store: &rcx.gpu_material_store,
                gpu_renderers: &rcx.gpu_renderers,
                mvp_build_pass2_pipeline,
                cull_pass2_args_pipeline,
                hiz_reduce_depth_pipeline,
                hiz_reduce_mip_pipeline,
                hiz_reduce_mip2_pipeline,
            };
            if rcx
                .main_camera
                .set_occlusion_enabled(desired, &scene_resources)
            {
                need_frame_slot_rebuild = true;
            }
        }

        if need_frame_slot_rebuild {
            rcx.frame_slots.clear();
            rcx.frame_slots = build_all_frame_slots(
                command_buffer_allocator,
                memory_allocator,
                queue_family_index,
                &rcx.swapchain_image_views,
                &rcx.main_camera,
                &rcx.world_transforms,
                &rcx.gpu_renderers,
            );
        }

        let image_index = frame.image_index as usize;

        // Debug: F9 toggles the frustum-lock feature.
        if job.f9_pressed {
            let new_lock = !rcx.main_camera.cull_lock();
            rcx.main_camera
                .set_cull_lock(new_lock, job.view_proj_cols);
        }

        // Cheap-path draw-plan update: rewrite the indirect template bases
        // in place. Gated by the compute wait below so no in-flight
        // `template → args` reset copy is mid-read.
        // (Deferred until after the wait, same ordering as before.)

        let host_wait_start = Instant::now();
        rcx.world_transforms.host_wait_for_previous_compute();
        fps.record_host_wait_compute(host_wait_start.elapsed().as_nanos() as u64);

        if let Some(plan) = pending_cheap_plan.as_ref() {
            rcx.main_camera.write_template_bases(plan);
        }

        let host_staging_start = Instant::now();
        let cpu_gpu_staging_ns = handoff.consume(job.ready_gen, &rcx.world_transforms);
        rcx.world_transforms.write_parent_updates(&job.parent_updates);
        rcx.gpu_renderers.write_spawns(&job.spawns);
        rcx.main_camera.write_cull_view_proj(job.view_proj_cols);
        fps.record_host_staging(host_staging_start.elapsed().as_nanos() as u64);
        fps.record_cpu_gpu_staging(cpu_gpu_staging_ns);
        fps.record_sim_update(job.sim_update_ns);
        fps.record_cpu_staging(job.cpu_staging_ns);
        fps.record_main_wait(job.main_wait_ns);

        // ── Submit + present ─────────────────────────────────────────────
        let cb = rcx.frame_slots[image_index].command_buffer.clone();
        swapchain_renderer.submit_and_present(frame, None, cb, Vec::new(), Vec::new());
        rcx.world_transforms.inc_signal_expected();
        fps.tick();
        *total_frames += 1;
        if *total_frames == 120 {
            rcx.world_transforms.report_staging_residency();
        }
    }
}
