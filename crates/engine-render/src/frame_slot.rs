//! The frame primaries, recorded once per `(swapchain image, staging slot)`
//! and replayed (ADR-0002). What goes in them, in order, is in
//! [frame-loop](../../../docs/notes/frame-loop.md#the-frame-primary).

use std::sync::Arc;

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferInheritanceInfo, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo,
        SecondaryAutoCommandBuffer, SubpassContents,
    },
    device::DeviceOwned,
    image::{ImageLayout, view::ImageView},
    query::{QueryPool, QueryPoolCreateInfo, QueryType},
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    sync::PipelineStage,
};

use engine_core::util::parallel;

use crate::STAGING_SLOTS;
use crate::SyncMut;
use crate::camera::RenderCamera;
use crate::frame::WorldRender;
use crate::gpu::Gpu;
use crate::step::{At, Primary};
use crate::transform_gpu::TransformGpuShared;
use crate::ui::UiGpu;

/// GPU timestamp queries, in the order the primary writes them. The layout and
/// why `SEAM` exists are in [frame-loop](../../../docs/notes/frame-loop.md#gpu-timestamps).
pub(crate) mod ts {
    pub const START: u32 = 0;
    pub const UPLOAD: u32 = 1;
    pub const CULL1: u32 = 2;
    pub const RASTER1: u32 = 3;
    pub const HIZ: u32 = 4;
    pub const CULL2: u32 = 5;
    pub const RASTER2: u32 = 6;
    pub const END: u32 = 7;
    pub const SEAM: u32 = 8;
    pub const TRS: u32 = 9;
    pub const STEPS: u32 = 10;
    pub const CLEARS: u32 = 11;
    pub const COUNT: u32 = 12;
}

pub(crate) struct FrameSlot {
    pub command_buffer: Arc<PrimaryAutoCommandBuffer>,
    /// Read back after this image's next `acquire`, once its fence signalled.
    pub timestamp_pool: Arc<QueryPool>,
}

/// One primary per `(image, staging slot)`, because acquire can return the
/// images in any order.
#[inline]
pub(crate) fn frame_slot_index(image_index: usize, staging_slot: usize) -> usize {
    image_index * STAGING_SLOTS + staging_slot
}

pub(crate) fn build_all_frame_slots(
    gpu: &Gpu,
    shared: &TransformGpuShared,
    swapchain_views: &[Arc<ImageView>],
    cameras: &[&RenderCamera],
    worlds: &[WorldRender],
    ui: &UiGpu,
) -> Vec<FrameSlot> {
    let transforms: Vec<_> = worlds.iter().map(|w| &w.transforms).collect();
    let scatter: [_; STAGING_SLOTS] =
        std::array::from_fn(|s| shared.record_scatter_secondary(s, &transforms));

    let n = swapchain_views.len() * STAGING_SLOTS;
    let mut out: Vec<std::mem::MaybeUninit<FrameSlot>> =
        (0..n).map(|_| std::mem::MaybeUninit::uninit()).collect();
    let out_ptr = SyncMut(out.as_mut_ptr());
    parallel::global::parallel_for(0..n, |range| {
        for i in range {
            let (image, slot) = (i / STAGING_SLOTS, i % STAGING_SLOTS);
            let fs = build_frame_slot(
                gpu,
                shared,
                &swapchain_views[image],
                cameras,
                worlds,
                ui,
                &scatter[slot],
                slot,
            );
            // SAFETY: each task writes a unique index in [0, n).
            unsafe { (*out_ptr.get().add(i)).write(fs) };
        }
    });
    // SAFETY: every index was initialised by the loop above.
    unsafe {
        let mut out = std::mem::ManuallyDrop::new(out);
        Vec::from_raw_parts(out.as_mut_ptr() as *mut FrameSlot, n, out.capacity())
    }
}

#[allow(clippy::too_many_arguments)]
fn build_frame_slot(
    gpu: &Gpu,
    shared: &TransformGpuShared,
    swapchain_view: &Arc<ImageView>,
    cameras: &[&RenderCamera],
    worlds: &[WorldRender],
    ui: &UiGpu,
    scatter: &Arc<SecondaryAutoCommandBuffer>,
    slot: usize,
) -> FrameSlot {
    // Only a camera sized to the window is blitted. A camera shown in a
    // `ui::Viewport` is drawn by that widget.
    let blit = cameras
        .iter()
        .find(|c| c.resolution().blits_to_swapchain())
        .map(|c| record_blit(gpu, c, swapchain_view));

    let mut b = AutoCommandBufferBuilder::primary(
        gpu.cbs.clone(),
        gpu.queue.queue_family_index(),
        CommandBufferUsage::MultipleSubmit,
    )
    .expect("primary CB builder");
    let pool = QueryPool::new(
        gpu.cbs.device().clone(),
        QueryPoolCreateInfo {
            query_count: ts::COUNT,
            ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
        },
    )
    .expect("create frame timestamp pool");
    // SAFETY (every timestamp below): queries stay in [0, COUNT) and the
    // pool is reset in this CB before any is written.
    let stamp = |b: &mut Primary, q: u32| {
        unsafe { b.write_timestamp(pool.clone(), q, PipelineStage::BottomOfPipe) }
            .expect("write_timestamp");
    };
    unsafe { b.reset_query_pool(pool.clone(), 0..ts::COUNT) }.expect("reset timestamp pool");
    unsafe { b.write_timestamp(pool.clone(), ts::START, PipelineStage::TopOfPipe) }
        .expect("write_timestamp");
    stamp(&mut b, ts::SEAM);

    // ── Upload everything that reads host staging ───────────────────────
    b.execute_commands(scatter.clone())
        .expect("execute TRS scatter");
    stamp(&mut b, ts::TRS);
    record_steps(&mut b, worlds, At::Upload, slot);
    b.execute_commands(ui.scatter_secondary(slot).clone())
        .expect("execute ui scatter");
    stamp(&mut b, ts::STEPS);
    for wr in worlds {
        wr.transforms.record_dirty_clears(slot, &mut b);
    }
    for cam in cameras {
        cam.record_promotions(slot, &mut b);
    }
    stamp(&mut b, ts::CLEARS);
    // Lets the host write staging again. Before moving it, read "What
    // actually orders `signal_cs`" on `TransformGpuShared::host_wait`.
    b.execute_commands(shared.signal_secondary().clone())
        .expect("execute signal");
    stamp(&mut b, ts::UPLOAD);

    // ── Compute, then each camera ───────────────────────────────────────
    record_steps(&mut b, worlds, At::Compute, slot);
    for (i, cam) in cameras.iter().enumerate() {
        // Only the first camera writes timestamps, so they show one camera's cost.
        let stamp = |b: &mut Primary, q| {
            if i == 0 {
                stamp(b, q)
            }
        };
        record_camera(&mut b, cam, slot, &stamp);
    }
    if let Some(blit) = &blit {
        b.execute_commands(blit.clone()).expect("execute blit");
    }
    record_ui(&mut b, ui, swapchain_view, blit.is_some());
    stamp(&mut b, ts::END);

    FrameSlot {
        command_buffer: b.build().expect("build primary CB"),
        timestamp_pool: pool,
    }
}

fn record_steps(b: &mut Primary, worlds: &[WorldRender], at: At, slot: usize) {
    for wr in worlds {
        for step in wr.steps() {
            step.record(at, slot, b);
        }
    }
}

/// Dual-pass occlusion cull and draw (ADR-0005), then the editor overlay.
fn record_camera(
    b: &mut Primary,
    cam: &RenderCamera,
    slot: usize,
    stamp: &dyn Fn(&mut Primary, u32),
) {
    b.execute_commands(cam.cull_secondary().clone())
        .expect("execute cull pass 1");
    stamp(b, ts::CULL1);
    b.begin_rendering(scene_scope(cam, true, AttachmentStoreOp::Store))
        .expect("begin pass 1");
    for draw in cam.scene_secondaries_pass1() {
        b.execute_commands(draw.clone())
            .expect("execute pass 1 draw");
    }
    b.end_rendering().expect("end pass 1");
    stamp(b, ts::RASTER1);

    // F8 skips the occlusion block. F9 skips the Hi-Z build, so pass 2 tests
    // against the frozen pyramid.
    if cam.occlusion_enabled() {
        if !cam.hiz_frozen() {
            b.execute_commands(cam.hiz_build_secondary().clone())
                .expect("execute hiz build");
        }
        stamp(b, ts::HIZ);
        b.execute_commands(cam.cull_pass2_secondary().clone())
            .expect("execute cull pass 2");
        if !cam.hiz_frozen() {
            b.execute_commands(cam.history_update_secondary().clone())
                .expect("execute history update");
        }
        stamp(b, ts::CULL2);
        // Store depth so the overlay can depth-test against the finished scene.
        b.begin_rendering(scene_scope(cam, false, AttachmentStoreOp::Store))
            .expect("begin pass 2");
        for draw in cam.scene_secondaries_pass2() {
            b.execute_commands(draw.clone())
                .expect("execute pass 2 draw");
        }
        b.end_rendering().expect("end pass 2");
        stamp(b, ts::RASTER2);
    } else {
        // Stamp the skipped stages anyway so the readback layout stays fixed.
        for q in [ts::HIZ, ts::CULL2, ts::RASTER2] {
            stamp(b, q);
        }
    }

    b.begin_rendering(scene_scope(cam, false, AttachmentStoreOp::DontCare))
        .expect("begin overlay");
    b.execute_commands(cam.overlay_secondary(slot).clone())
        .expect("execute overlay");
    b.end_rendering().expect("end overlay");
}

/// Renders into the camera's colour and depth, cleared for pass 1 and
/// loaded for the rest.
fn scene_scope(cam: &RenderCamera, clear: bool, depth_store: AttachmentStoreOp) -> RenderingInfo {
    let load_op = if clear {
        AttachmentLoadOp::Clear
    } else {
        AttachmentLoadOp::Load
    };
    RenderingInfo {
        contents: SubpassContents::SecondaryCommandBuffers,
        color_attachments: vec![Some(RenderingAttachmentInfo {
            load_op,
            store_op: AttachmentStoreOp::Store,
            clear_value: clear.then(|| [0.08, 0.08, 0.10, 1.0].into()),
            ..RenderingAttachmentInfo::image_view(cam.color_view().clone())
        })],
        depth_attachment: Some(RenderingAttachmentInfo {
            image_layout: ImageLayout::DepthStencilAttachmentOptimal,
            load_op,
            store_op: depth_store,
            clear_value: clear.then(|| 1.0_f32.into()),
            ..RenderingAttachmentInfo::image_view(cam.depth_view().clone())
        }),
        ..Default::default()
    }
}

fn record_blit(
    gpu: &Gpu,
    cam: &RenderCamera,
    view: &Arc<ImageView>,
) -> Arc<SecondaryAutoCommandBuffer> {
    let mut b = AutoCommandBufferBuilder::secondary(
        gpu.cbs.clone(),
        gpu.queue.queue_family_index(),
        CommandBufferUsage::MultipleSubmit,
        CommandBufferInheritanceInfo::default(),
    )
    .expect("blit secondary builder");
    b.blit_image(BlitImageInfo::images(
        cam.color_image().clone(),
        view.image().clone(),
    ))
    .expect("blit_image");
    b.build().expect("build blit secondary")
}

/// The UI draws in sRGB straight to the swapchain, after the blit so it isn't
/// tonemapped. With no blit nothing else fills the window, so it clears.
fn record_ui(b: &mut Primary, ui: &UiGpu, view: &Arc<ImageView>, over_blit: bool) {
    let (load_op, clear_value) = match over_blit {
        true => (AttachmentLoadOp::Load, None),
        false => (AttachmentLoadOp::Clear, Some([0.0, 0.0, 0.0, 1.0].into())),
    };
    b.begin_rendering(RenderingInfo {
        contents: SubpassContents::SecondaryCommandBuffers,
        color_attachments: vec![Some(RenderingAttachmentInfo {
            load_op,
            store_op: AttachmentStoreOp::Store,
            clear_value,
            ..RenderingAttachmentInfo::image_view(view.clone())
        })],
        ..Default::default()
    })
    .expect("begin ui");
    b.execute_commands(ui.draw_secondary().clone())
        .expect("execute ui draw");
    b.end_rendering().expect("end ui");
}
