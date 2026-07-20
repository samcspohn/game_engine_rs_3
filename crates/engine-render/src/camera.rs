//! Render-side camera (Design B — GPU-driven). Owns the GPU attachments
//! (color + depth) a camera renders into, the device-side MVP matrix buffer +
//! graphics descriptor set the draws read from, the per-slot indirect-command
//! buffers, the **cull secondary** (the compute pass that frustum + occlusion
//! tests every renderer and compacts visible MVPs), and the **scene
//! secondary** (the `multiDrawIndexedIndirect`) — now duplicated across two
//! passes for **dual-pass temporal Hi-Z occlusion culling**.
//!
//! There is no CPU-sorted topology. Pass 1's cull dispatches over the whole
//! renderer/transform range and reads the world's `GPURenderers`
//! (`transform → mesh_id`), the registry `redirect` (`mesh_id → slot`), and
//! the `mesh_table` (per-slot bounds), writing each visible instance's MVP
//! into its slot's contiguous region. The CPU only supplies a small per-slot
//! [`DrawPlan`] (geometry + prefix-summed `first_instance` bases) that changes
//! on spawn / load — never an `O(N)` sort.
//!
//! # Dual-pass occlusion culling
//!
//! Pass 1 (`mvp_build.comp`) frustum-tests every slot (authoritative) and,
//! for frustum-visible slots, occlusion-tests against **last frame's** Hi-Z
//! pyramid using **last frame's** `view_proj` (a temporal approximation —
//! camera/objects may have moved). Instances that pass both draw
//! immediately via `scene_secondary_pass1`. Instances the occlusion
//! sub-test rejects become *candidates*, appended (with their resolved
//! world TRS + world-space bounding sphere) to a device-side list.
//!
//! Between the two render passes, `hiz_build_secondary` max-reduces this
//! frame's freshly-drawn (pass-1) depth attachment into `hiz_current`'s
//! mip pyramid. Pass 2 (`mvp_build_pass2.comp`, dispatched indirectly —
//! sized to the live candidate count) re-tests only the candidates against
//! this frame's own accurate Hi-Z; newly-visible ones draw via
//! `scene_secondary_pass2` into the same (still-open, `Load`-not-`Clear`)
//! attachments.
//!
//! At the end of the frame `history_update_secondary` copies
//! `hiz_current → hiz_prev` and the shared `sot_view_proj → prev_view_proj`
//! so next frame's pass 1 sees this frame's data as "last frame's" — the
//! two Hi-Z pyramids and the `prev_view_proj` buffer keep **fixed
//! identities** across frames (never swapped), so no descriptor set ever
//! needs rebinding just because a frame elapsed; only a capacity or extent
//! change triggers a rebuild, per this file's usual invalidation model.
//! Note: `hiz_current` only reflects pass 1's depth contribution (not pass
//! 2's) — an accepted, documented approximation; see the doc comment on
//! [`RenderCamera::hiz_current`].
//!
//! Invalidation axes: per-camera resolution (attachments + both scene
//! secondaries + the Hi-Z pyramids + everything that binds their views),
//! draw plan / capacity (MVP + indirect + cull/scene secondaries for both
//! passes, the candidate list), world capacity (the cull set binds SoT /
//! `GPURenderers` / redirect / mesh_table, so it rebinds when those
//! reallocate), and per-swapchain-image (on `FrameSlot`).

use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferInheritanceInfo, CommandBufferInheritanceRenderingInfo, CommandBufferUsage,
        CopyBufferInfo, CopyImageInfo, DispatchIndirectCommand, DrawIndexedIndirectCommand,
        ImageCopy, SecondaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    format::Format,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::{ImageView, ImageViewCreateInfo},
        Image, ImageCreateInfo, ImageSubresourceLayers, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{graphics::viewport::Viewport, ComputePipeline, GraphicsPipeline, PipelineBindPoint},
};

// `Pipeline` trait is needed for `pipeline.layout()` method resolution.
use vulkano::pipeline::Pipeline;

use crate::assets::{GpuMaterialStore, GpuMeshStore, GpuTextureStore};
use crate::gpu_renderers::GpuRenderers;
use crate::shaders;
use crate::transform_gpu::WorldTransformGpu;

/// Pixel format used for camera-owned offscreen color targets.
pub const CAMERA_COLOR_FORMAT: Format = Format::R16G16B16A16_SFLOAT;

/// Pixel format used for camera-owned depth targets.
pub const CAMERA_DEPTH_FORMAT: Format = Format::D32_SFLOAT;

/// Pixel format used for the Hi-Z occlusion pyramids. Single-channel float
/// so a compute shader can `imageStore`/`imageLoad` it as a plain storage
/// image (unlike the depth attachment format, which isn't guaranteed
/// storage-image-compatible).
const HIZ_FORMAT: Format = Format::R32_SFLOAT;

/// Compute-shader workgroup size (both axes) for the Hi-Z reduce shaders.
/// Must match `local_size_x`/`local_size_y` in `hiz_reduce_depth.comp` /
/// `hiz_reduce_mip.comp`.
const HIZ_WORKGROUP_SIZE: u32 = 8;

/// Compute-shader workgroup size for the cull dispatches. Must match
/// `local_size_x` in `mvp_build.comp` / `mvp_build_pass2.comp`.
const CULL_WORKGROUP_SIZE: u32 = 64;

/// How a camera's attachment extent is determined relative to the swapchain.
#[derive(Clone, Copy, Debug)]
pub enum CameraResolution {
    /// Track the swapchain extent 1:1.
    MatchSwapchain,
}

impl CameraResolution {
    fn resolve(&self, swapchain_extent: [u32; 2]) -> [u32; 2] {
        match self {
            CameraResolution::MatchSwapchain => swapchain_extent,
        }
    }

    /// Does this policy depend on the swapchain extent?
    pub fn depends_on_swapchain(&self) -> bool {
        match self {
            CameraResolution::MatchSwapchain => true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DrawPlan
// ─────────────────────────────────────────────────────────────────────────────

/// The CPU-computed per-frame-static draw description: one indirect command
/// per drawable slot (geometry offsets + prefix-summed `first_instance` base,
/// `instance_count` pre-zeroed for the cull to accumulate), plus the total
/// renderer count (the MVP buffer size). Rebuilt on topology change — `O(#slots)`.
///
/// Shared verbatim by pass 1 and pass 2 (see [`DrawResources`]): both need
/// capacity for the same worst case — "every instance of this mesh slot
/// ends up visible via this pass" — since an instance is drawn by exactly
/// one of the two passes, never both.
#[derive(Clone)]
pub struct DrawPlan {
    pub commands: Vec<DrawIndexedIndirectCommand>,
    pub total_renderers: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// CameraSceneResources
// ─────────────────────────────────────────────────────────────────────────────

/// Per-call bundle of GPU/scene state the camera needs to (re)build its draw
/// resources. Nothing is owned beyond the call.
pub struct CameraSceneResources<'a> {
    pub cb_allocator: &'a Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: &'a Arc<StandardDescriptorSetAllocator>,
    pub memory_allocator: &'a Arc<StandardMemoryAllocator>,
    pub pipeline: &'a Arc<GraphicsPipeline>,
    pub queue_family_index: u32,
    /// SoT TRS + view_proj + the cull (a.k.a. mvp_build) pipeline + set 1.
    pub world_transforms: &'a WorldTransformGpu,
    /// Mega buffers + redirect + mesh table + per-slot authored materials.
    pub mesh_store: &'a GpuMeshStore,
    /// Sampled texture images + texture redirect (graphics set 1).
    pub texture_store: &'a GpuTextureStore,
    /// Material SSBO + material redirect (graphics set 1).
    pub material_store: &'a GpuMaterialStore,
    /// Per-transform `GPURenderers` buffer (`transform → (mesh, material)`).
    pub gpu_renderers: &'a GpuRenderers,
    /// Pass 2's cull pipeline — see `shaders/mvp_build_pass2.comp`.
    pub mvp_build_pass2_pipeline: &'a Arc<ComputePipeline>,
    /// The tiny "build pass 2's dispatch-indirect args" pipeline — see
    /// `shaders/cull_pass2_args.comp`.
    pub cull_pass2_args_pipeline: &'a Arc<ComputePipeline>,
    /// Hi-Z pyramid level 0 (depth → mip0) pipeline — see
    /// `shaders/hiz_reduce_depth.comp`.
    pub hiz_reduce_depth_pipeline: &'a Arc<ComputePipeline>,
    /// Hi-Z pyramid levels 1..N (mip[L-1] → mip[L]) pipeline — see
    /// `shaders/hiz_reduce_mip.comp`. Used only for a trailing odd leftover
    /// level when the remaining level count after level 0 is odd — see
    /// [`hiz_reduce_mip2_pipeline`](Self::hiz_reduce_mip2_pipeline).
    pub hiz_reduce_mip_pipeline: &'a Arc<ComputePipeline>,
    /// Hi-Z pyramid, FUSED pair of levels (mip[L-1] → mip[L] → mip[L+1] in
    /// one dispatch) pipeline — see `shaders/hiz_reduce_mip2.comp`. Used for
    /// every pair of remaining levels; halves the mip-to-mip dispatch count
    /// versus running `hiz_reduce_mip_pipeline` once per level.
    pub hiz_reduce_mip2_pipeline: &'a Arc<ComputePipeline>,
}

// ─────────────────────────────────────────────────────────────────────────────
// DrawResources — one cull pass's compacted output
// ─────────────────────────────────────────────────────────────────────────────

/// Buffers + graphics descriptor set for one cull pass's compacted output:
/// the per-visible-instance MVP + material buffers, the graphics
/// descriptor set (0) that reads them, and the per-slot indirect-command
/// buffers (host template + device args) that pass's cull dispatch
/// atomically accumulates into.
///
/// Pass 1 and pass 2 each own an independent instance, built from the
/// *same* [`DrawPlan`] (see its doc comment for why capacities match) —
/// pass 2 cannot share pass 1's buffers because both passes' `scene_secondary`
/// draws are separately recorded, pre-built command buffers: pass 1's draw
/// executes (and is done reading `instance_count`) before pass 2's cull
/// even starts appending, so sharing one region would require the two
/// draws to somehow agree on non-overlapping sub-ranges of a value only
/// known after both dispatches run. Independent buffers sidestep that
/// entirely at the cost of roughly 2× the per-camera MVP/indirect memory.
struct DrawResources {
    device_matrices: Subbuffer<[[f32; 16]]>,
    inst_material: Subbuffer<[u32]>,
    graphics_set: Arc<DescriptorSet>,
    indirect_template: Subbuffer<[DrawIndexedIndirectCommand]>,
    indirect_args: Subbuffer<[DrawIndexedIndirectCommand]>,
    mvp_capacity: usize,
    slot_capacity: usize,
}

impl DrawResources {
    fn new(scene: &CameraSceneResources<'_>, plan: &DrawPlan) -> Self {
        let slot_count = plan.commands.len();
        let mvp_capacity = (plan.total_renderers as usize).max(1);
        let slot_capacity = slot_count.max(1);

        let (device_matrices, inst_material, graphics_set) = allocate_matrices_and_set(
            scene.memory_allocator,
            scene.descriptor_set_allocator,
            scene.pipeline,
            mvp_capacity,
        );
        let (indirect_template, indirect_args) =
            allocate_indirect_buffers(scene.memory_allocator, slot_capacity);
        write_indirect_template(&indirect_template, &plan.commands);

        Self {
            device_matrices,
            inst_material,
            graphics_set,
            indirect_template,
            indirect_args,
            mvp_capacity,
            slot_capacity,
        }
    }

    /// Grow buffers (geometric) to fit `plan` if needed, and rewrite the
    /// indirect template's per-slot commands unconditionally (the
    /// prefix-summed bases shift on every spawn regardless of whether a
    /// grow happened).
    fn ensure_capacity(&mut self, scene: &CameraSceneResources<'_>, plan: &DrawPlan) {
        let slot_count = plan.commands.len();
        let total = plan.total_renderers as usize;

        if total > self.mvp_capacity {
            self.mvp_capacity = total.max(self.mvp_capacity.saturating_mul(2)).max(1);
            let (dm, im, gs) = allocate_matrices_and_set(
                scene.memory_allocator,
                scene.descriptor_set_allocator,
                scene.pipeline,
                self.mvp_capacity,
            );
            self.device_matrices = dm;
            self.inst_material = im;
            self.graphics_set = gs;
        }
        if slot_count > self.slot_capacity {
            self.slot_capacity = slot_count.max(self.slot_capacity.saturating_mul(2)).max(1);
            let (t, a) = allocate_indirect_buffers(scene.memory_allocator, self.slot_capacity);
            self.indirect_template = t;
            self.indirect_args = a;
        }
        write_indirect_template(&self.indirect_template, &plan.commands);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HizPyramid
// ─────────────────────────────────────────────────────────────────────────────

/// Max-reduction Hi-Z pyramid: one [`HIZ_FORMAT`] image with a full mip
/// chain (level 0 = half the depth buffer's resolution rounded up, each
/// further level halves again down to 1×1). See `hiz_reduce_depth.comp` /
/// `hiz_reduce_mip.comp` for how each level is built.
///
/// `hiz_current` and `hiz_prev` both use this type with **identical**
/// usage flags (`STORAGE | SAMPLED | TRANSFER_SRC | TRANSFER_DST`) even
/// though `hiz_prev` is never a compute write target (only a
/// `copy_image` destination) — keeping them structurally identical means
/// one constructor serves both and nothing prevents swapping their roles
/// later if that ever becomes useful.
struct HizPyramid {
    #[allow(dead_code)] // kept for the copy_image src/dst in history_update
    image: Arc<Image>,
    /// One single-mip view per level — level 0 is the write target for
    /// `hiz_reduce_depth_cs`; each level `L>0` is the read source (as
    /// `mip_views[L-1]`) and write target (as `mip_views[L]`) for one
    /// `hiz_reduce_mip_cs` dispatch.
    mip_views: Vec<Arc<ImageView>>,
    /// Full mip-chain sampled view — bound as the combined image sampler
    /// the cull shaders `texelFetch` an explicit LOD from.
    sampled_view: Arc<ImageView>,
    mip0_extent: [u32; 2],
    mip_count: u32,
}

/// Hi-Z mip-0 extent: half the depth buffer's resolution, rounded up.
fn hiz_mip0_extent(depth_extent: [u32; 2]) -> [u32; 2] {
    [(depth_extent[0] + 1) / 2, (depth_extent[1] + 1) / 2]
}

/// Number of mip levels from `mip0_extent` down to (and including) 1×1.
/// **Must** use the same floor-based halving Vulkan uses to derive each
/// mip's actual dimensions from level 0 (`max(1, extent >> level)`) — an
/// image's `mip_levels` is capped at `floor(log2(max(w,h))) + 1`
/// (`VUID-VkImageCreateInfo-mipLevels-00958`), which is smaller than what
/// ceiling-based halving would suggest (ceiling halving shrinks slower, so
/// it both overcounts levels and disagrees with the extents Vulkan
/// actually assigns each level).
fn hiz_mip_count(mip0_extent: [u32; 2]) -> u32 {
    let (mut w, mut h) = (mip0_extent[0].max(1), mip0_extent[1].max(1));
    let mut count = 1u32;
    while w > 1 || h > 1 {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        count += 1;
    }
    count
}

/// Extent of Hi-Z pyramid level `level`, given the mip-0 extent. Floor-based
/// — see [`hiz_mip_count`]'s doc comment for why. Because this is a floor
/// (not ceiling), an odd-dimensioned source level leaves one source row/
/// column unpaired; `hiz_reduce_mip.comp`'s last dst texel in that
/// dimension explicitly extends its footprint to 3-wide to still include
/// it. (An earlier version relied on clamping the `+1` tap instead —
/// that clamp never actually triggers when `dst_size = floor(src_size/2)`,
/// so the leftover row/column was silently dropped from every level's
/// max-reduction rather than merely duplicated, and the loss compounded
/// across further odd-dimensioned levels — a real occlusion-culling bug,
/// not a harmless approximation. Fixed 2026-07-20.)
fn hiz_level_extent(mip0_extent: [u32; 2], level: u32) -> [u32; 2] {
    let (mut w, mut h) = (mip0_extent[0].max(1), mip0_extent[1].max(1));
    for _ in 0..level {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    [w, h]
}

fn allocate_hiz_pyramid(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    mip0_extent: [u32; 2],
) -> HizPyramid {
    let mip_count = hiz_mip_count(mip0_extent);
    let [w, h] = mip0_extent;
    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: HIZ_FORMAT,
            extent: [w.max(1), h.max(1), 1],
            mip_levels: mip_count,
            usage: ImageUsage::STORAGE
                | ImageUsage::SAMPLED
                | ImageUsage::TRANSFER_SRC
                | ImageUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .expect("Failed to create Hi-Z pyramid image");

    let mip_views: Vec<Arc<ImageView>> = (0..mip_count)
        .map(|m| {
            let mut info = ImageViewCreateInfo::from_image(&image);
            info.subresource_range.mip_levels = m..(m + 1);
            ImageView::new(image.clone(), info).expect("Failed to create Hi-Z mip view")
        })
        .collect();
    let sampled_view =
        ImageView::new_default(image.clone()).expect("Failed to create Hi-Z sampled view");

    HizPyramid {
        image,
        mip_views,
        sampled_view,
        mip0_extent,
        mip_count,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RenderCamera
// ─────────────────────────────────────────────────────────────────────────────

pub struct RenderCamera {
    resolution: CameraResolution,
    extent: [u32; 2],
    color_image: Arc<Image>,
    depth_image: Arc<Image>,
    color_view: Arc<ImageView>,
    depth_view: Arc<ImageView>,

    /// Graphics set 1 — texture redirect + material redirect + material
    /// SSBO + the sampled-image array. Shared by both passes' draws.
    texture_set: Arc<DescriptorSet>,

    /// Pass 1's compacted output — instances visible against last frame's
    /// (reprojected) Hi-Z draw immediately via `scene_secondary_pass1`.
    pass1: DrawResources,
    /// Pass 2's compacted output — instances pass 1's occlusion sub-test
    /// deferred, confirmed against this frame's own Hi-Z, draw via
    /// `scene_secondary_pass2`. See [`DrawResources`]'s doc comment for
    /// why this can't share pass 1's buffers.
    pass2: DrawResources,

    /// Pass 1 cull set 0 — SoT, GPURenderers, redirect, mesh_table, MVP,
    /// indirect, Parents, slot materials, inst material, and (new) the
    /// candidate list + its live counter.
    cull_set: Arc<DescriptorSet>,
    /// Pass 1 cull set 1 (camera-owned occlusion set) — this frame's
    /// `view_proj`, last frame's `view_proj`, and last frame's Hi-Z
    /// (sampled). Replaces the single-buffer `mvp_build_set1` that used to
    /// live on `WorldTransformGpu` — that set is now too narrow for pass
    /// 1's occlusion sub-test and the extra bindings are per-camera data
    /// anyway.
    occlusion_set: Arc<DescriptorSet>,
    /// Pass 1 secondary: reset copy (indirect template → args), reset the
    /// candidate counter, the frustum+occlusion cull dispatch, and the
    /// tiny dispatch-args-builder for pass 2's `dispatch_indirect`.
    cull_secondary: Arc<SecondaryAutoCommandBuffer>,

    /// Pass 2 cull set 0 — candidate list + counter (read), pass 2's own
    /// indirect args (rw), MVP + inst_material (write).
    pass2_cull_set0: Arc<DescriptorSet>,
    /// Pass 2 cull set 1 — this frame's `view_proj` + this frame's own
    /// Hi-Z (sampled).
    pass2_cull_set1: Arc<DescriptorSet>,
    /// Pass 2 secondary: reset copy (pass 2's indirect template → args)
    /// then `dispatch_indirect` over the live candidate count.
    cull_pass2_secondary: Arc<SecondaryAutoCommandBuffer>,

    /// Hi-Z build set for level 0 (depth attachment → `hiz_current` mip 0).
    hiz_level0_set: Arc<DescriptorSet>,
    /// Hi-Z build sets for each FUSED pair of remaining levels (mip[L-1] →
    /// mip[L] → mip[L+1] in one dispatch), indexed `[0] = levels (1,2)'s
    /// set, [1] = levels (3,4)'s set, ...`. See `shaders/hiz_reduce_mip2.comp`.
    hiz_mip2_sets: Vec<Arc<DescriptorSet>>,
    /// Hi-Z build set for a single trailing leftover level (mip[L-1] →
    /// mip[L]), present iff the remaining-level count (`mip_count - 1`) is
    /// odd — one level can't be paired up for fusion. Uses the plain
    /// single-level pipeline/shader (`hiz_reduce_mip_pipeline`).
    hiz_trailing_set: Option<Arc<DescriptorSet>>,
    /// Hi-Z build secondary: level 0, then one dispatch per fused pair of
    /// remaining levels, then (if present) the trailing leftover level —
    /// writing `hiz_current`. Extent-dependent only (mip count/dims derive
    /// from the depth buffer's resolution) — never re-recorded by
    /// [`Self::ensure_current`], only by [`Self::on_swapchain_resize`].
    hiz_build_secondary: Arc<SecondaryAutoCommandBuffer>,
    /// History-update secondary: copies `hiz_current → hiz_prev` (all
    /// mips) and the shared `sot_view_proj → prev_view_proj`, so next
    /// frame's pass 1 sees this frame's data as "last frame's" without
    /// either descriptor set ever rebinding (fixed image/buffer
    /// identities — see the module doc comment). Extent-dependent only,
    /// same rebuild scope as `hiz_build_secondary`.
    history_update_secondary: Arc<SecondaryAutoCommandBuffer>,

    /// Pass 1's `multiDrawIndexedIndirect` over `pass1.indirect_args`.
    scene_secondary_pass1: Arc<SecondaryAutoCommandBuffer>,
    /// Pass 2's `multiDrawIndexedIndirect` over `pass2.indirect_args`,
    /// recorded against a `Load` (not `Clear`) attachment scope — see
    /// `lib.rs`'s `build_frame_slot`.
    scene_secondary_pass2: Arc<SecondaryAutoCommandBuffer>,

    /// This frame's Hi-Z pyramid, built by `hiz_build_secondary` from this
    /// frame's own pass-1 depth output. Read by pass 2's occlusion test
    /// (exact — same frame, no reprojection) and copied into `hiz_prev` at
    /// frame end. **Note:** only reflects pass 1's depth contribution —
    /// pass 2's draws land in the real depth attachment but are not
    /// re-folded into `hiz_current`, so an object confirmed only via pass
    /// 2 this frame won't help occlude anything next frame until pass 1
    /// itself draws it (typically the very next frame, once it's no
    /// longer a "just revealed" edge case). Rebuilding Hi-Z a second time
    /// after pass 2 would close this gap at roughly double the per-frame
    /// Hi-Z build cost; deferred as a planned follow-up if profiling shows
    /// the steady-state candidate count doesn't stay small.
    hiz_current: HizPyramid,
    /// Last frame's Hi-Z pyramid (via the end-of-frame `hiz_current →
    /// hiz_prev` copy). Read by pass 1's occlusion sub-test, reprojected
    /// with `prev_view_proj`.
    hiz_prev: HizPyramid,
    /// Camera-owned `view_proj` history — last frame's value. Copied from
    /// the shared `WorldTransformGpu::sot_view_proj` at the end of every
    /// frame (`history_update_secondary`), *before* next frame's
    /// promotion copy overwrites it.
    prev_view_proj: Subbuffer<[[f32; 16]]>,
    /// Candidate records pass 1 appends, pass 2 consumes — one `[f32; 16]`
    /// (64-byte) slot per record, matching `Candidate`'s 4×vec4 GLSL
    /// layout exactly (see `mvp_build.comp`). Capacity == `cull_range`
    /// (worst case: every dispatched slot becomes a candidate).
    candidate_list: Subbuffer<[[f32; 16]]>,
    /// Live candidate count for this frame — reset to 0 at the front of
    /// `cull_secondary`, accumulated by pass 1's atomics, read by pass 2's
    /// bounds check and by the dispatch-args builder.
    candidate_count: Subbuffer<[u32]>,
    /// `[x, y, z]` group counts for pass 2's `dispatch_indirect`, built by
    /// `cull_pass2_args.comp` from `candidate_count` right after pass 1's
    /// main dispatch (same secondary).
    pass2_dispatch_args: Subbuffer<[DispatchIndirectCommand]>,
    /// Depth-only NEAREST/ClampToEdge sampler shared by every Hi-Z-related
    /// combined-image-sampler binding. `texelFetch` (used throughout the
    /// occlusion tests and the reduce shaders) ignores the sampler's
    /// filter/address mode entirely — a sampler object is still required
    /// to form a combined image sampler, so this exists purely to satisfy
    /// that requirement. Built once; extent/capacity-independent.
    hiz_sampler: Arc<Sampler>,

    /// Number of drawable slots baked into both scene secondaries'
    /// drawCount.
    slot_count: usize,
    /// Renderer range baked into the pass-1 cull dispatch (== world entity
    /// capacity).
    cull_range: usize,
}

impl RenderCamera {
    pub fn new_match_swapchain(
        swapchain_extent: [u32; 2],
        scene: &CameraSceneResources<'_>,
        plan: &DrawPlan,
        renderer_capacity: usize,
    ) -> Self {
        Self::new(
            CameraResolution::MatchSwapchain,
            swapchain_extent,
            scene,
            plan,
            renderer_capacity,
        )
    }

    pub fn new(
        resolution: CameraResolution,
        swapchain_extent: [u32; 2],
        scene: &CameraSceneResources<'_>,
        plan: &DrawPlan,
        renderer_capacity: usize,
    ) -> Self {
        let extent = resolution.resolve(swapchain_extent);
        let (color_image, color_view, depth_image, depth_view) =
            allocate_attachments(scene.memory_allocator, extent);

        let pass1 = DrawResources::new(scene, plan);
        let pass2 = DrawResources::new(scene, plan);

        let (candidate_list, candidate_count) =
            allocate_candidate_buffers(scene.memory_allocator, renderer_capacity);
        let pass2_dispatch_args = allocate_pass2_dispatch_args(scene.memory_allocator);
        let prev_view_proj = allocate_prev_view_proj(scene.memory_allocator);
        let hiz_sampler = build_hiz_sampler(scene.queue_family_index, scene.pipeline.device().clone());

        let hiz_mip0_extent = hiz_mip0_extent(extent);
        let hiz_current = allocate_hiz_pyramid(scene.memory_allocator, hiz_mip0_extent);
        let hiz_prev = allocate_hiz_pyramid(scene.memory_allocator, hiz_mip0_extent);

        let cull_set = build_cull_set(scene, &pass1, &candidate_list, &candidate_count);
        let occlusion_set =
            build_occlusion_set(scene, &prev_view_proj, &hiz_prev, &hiz_sampler);
        let pass2_cull_set0 = build_pass2_cull_set0(scene, &candidate_list, &candidate_count, &pass2);
        let pass2_cull_set1 = build_pass2_cull_set1(scene, &hiz_current, &hiz_sampler);
        let (hiz_level0_set, hiz_mip2_sets, hiz_trailing_set) =
            build_hiz_sets(scene, &depth_view, &hiz_current, &hiz_sampler);

        let cull_secondary = record_cull_secondary(
            scene,
            &pass1,
            &cull_set,
            &occlusion_set,
            &candidate_count,
            &pass2_dispatch_args,
            renderer_capacity as u32,
        );
        let cull_pass2_secondary = record_cull_pass2_secondary(
            scene,
            &pass2,
            &pass2_cull_set0,
            &pass2_cull_set1,
            &pass2_dispatch_args,
        );
        let hiz_build_secondary = record_hiz_build_secondary(
            scene,
            &hiz_level0_set,
            &hiz_mip2_sets,
            &hiz_trailing_set,
            hiz_mip0_extent,
        );
        let history_update_secondary =
            record_history_update_secondary(scene, &hiz_current, &hiz_prev, &prev_view_proj);

        let texture_set = build_texture_set(scene);
        let slot_count = plan.commands.len();
        let scene_secondary_pass1 = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &pass1.graphics_set,
            &texture_set,
            scene.mesh_store,
            &pass1.indirect_args,
            slot_count,
            extent,
        );
        let scene_secondary_pass2 = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &pass2.graphics_set,
            &texture_set,
            scene.mesh_store,
            &pass2.indirect_args,
            slot_count,
            extent,
        );

        RenderCamera {
            resolution,
            extent,
            color_image,
            depth_image,
            color_view,
            depth_view,
            texture_set,
            pass1,
            pass2,
            cull_set,
            occlusion_set,
            cull_secondary,
            pass2_cull_set0,
            pass2_cull_set1,
            cull_pass2_secondary,
            hiz_level0_set,
            hiz_mip2_sets,
            hiz_trailing_set,
            hiz_build_secondary,
            history_update_secondary,
            scene_secondary_pass1,
            scene_secondary_pass2,
            hiz_current,
            hiz_prev,
            prev_view_proj,
            candidate_list,
            candidate_count,
            pass2_dispatch_args,
            hiz_sampler,
            slot_count,
            cull_range: renderer_capacity,
        }
    }

    /// Swapchain resized. Re-creates every extent-dependent resource: the
    /// color/depth attachments, both Hi-Z pyramids (mip0 tracks the depth
    /// buffer's new resolution), the descriptor sets that bind any of
    /// their views, and every secondary that references those sets or
    /// whose recording is extent-shaped (the scene secondaries' viewport,
    /// the Hi-Z build's per-level dispatch dims, the history copy's
    /// per-mip regions). Capacity-dependent resources (pass 1/2's
    /// MVP/indirect buffers, the candidate list, `cull_set`,
    /// `pass2_cull_set0`) are untouched — they don't depend on extent.
    /// Returns `true` if anything was rebuilt.
    pub fn on_swapchain_resize(
        &mut self,
        new_swapchain_extent: [u32; 2],
        scene: &CameraSceneResources<'_>,
    ) -> bool {
        if !self.resolution.depends_on_swapchain() {
            return false;
        }
        let new_extent = self.resolution.resolve(new_swapchain_extent);
        if new_extent == self.extent {
            return false;
        }
        let (color_image, color_view, depth_image, depth_view) =
            allocate_attachments(scene.memory_allocator, new_extent);
        self.extent = new_extent;
        self.color_image = color_image;
        self.color_view = color_view;
        self.depth_image = depth_image;
        self.depth_view = depth_view;

        let hiz_mip0_extent = hiz_mip0_extent(new_extent);
        self.hiz_current = allocate_hiz_pyramid(scene.memory_allocator, hiz_mip0_extent);
        self.hiz_prev = allocate_hiz_pyramid(scene.memory_allocator, hiz_mip0_extent);

        self.occlusion_set =
            build_occlusion_set(scene, &self.prev_view_proj, &self.hiz_prev, &self.hiz_sampler);
        self.pass2_cull_set1 = build_pass2_cull_set1(scene, &self.hiz_current, &self.hiz_sampler);
        let (hiz_level0_set, hiz_mip2_sets, hiz_trailing_set) =
            build_hiz_sets(scene, &self.depth_view, &self.hiz_current, &self.hiz_sampler);
        self.hiz_level0_set = hiz_level0_set;
        self.hiz_mip2_sets = hiz_mip2_sets;
        self.hiz_trailing_set = hiz_trailing_set;

        self.cull_secondary = record_cull_secondary(
            scene,
            &self.pass1,
            &self.cull_set,
            &self.occlusion_set,
            &self.candidate_count,
            &self.pass2_dispatch_args,
            self.cull_range as u32,
        );
        self.cull_pass2_secondary = record_cull_pass2_secondary(
            scene,
            &self.pass2,
            &self.pass2_cull_set0,
            &self.pass2_cull_set1,
            &self.pass2_dispatch_args,
        );
        self.hiz_build_secondary = record_hiz_build_secondary(
            scene,
            &self.hiz_level0_set,
            &self.hiz_mip2_sets,
            &self.hiz_trailing_set,
            hiz_mip0_extent,
        );
        self.history_update_secondary = record_history_update_secondary(
            scene,
            &self.hiz_current,
            &self.hiz_prev,
            &self.prev_view_proj,
        );

        self.scene_secondary_pass1 = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &self.pass1.graphics_set,
            &self.texture_set,
            scene.mesh_store,
            &self.pass1.indirect_args,
            self.slot_count,
            self.extent,
        );
        self.scene_secondary_pass2 = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &self.pass2.graphics_set,
            &self.texture_set,
            scene.mesh_store,
            &self.pass2.indirect_args,
            self.slot_count,
            self.extent,
        );
        true
    }

    /// Rebuild the per-frame-static draw resources for the current draw plan +
    /// renderer capacity. Grows pass 1 and pass 2's MVP / indirect buffers
    /// (geometrically, independently — see [`DrawResources`]) and the
    /// candidate list, rewrites both indirect templates, and re-records the
    /// cull + scene secondaries for both passes (and rebinds `cull_set` /
    /// `pass2_cull_set0` to the current world buffers). Extent-only
    /// resources (Hi-Z pyramids, `occlusion_set`, `pass2_cull_set1`,
    /// `hiz_build_secondary`, `history_update_secondary`) are untouched —
    /// see [`Self::on_swapchain_resize`]. Always returns `true` (the
    /// FrameSlot primaries reference the secondaries, so callers must
    /// rebuild them).
    ///
    /// Called only on topology change / capacity growth — never per frame in
    /// steady state.
    pub fn ensure_current(
        &mut self,
        plan: &DrawPlan,
        renderer_capacity: usize,
        scene: &CameraSceneResources<'_>,
    ) -> bool {
        let slot_count = plan.commands.len();

        self.pass1.ensure_capacity(scene, plan);
        self.pass2.ensure_capacity(scene, plan);

        if renderer_capacity > self.candidate_list.len() as usize {
            let (list, count) =
                allocate_candidate_buffers(scene.memory_allocator, renderer_capacity);
            self.candidate_list = list;
            self.candidate_count = count;
        }

        self.slot_count = slot_count;
        self.cull_range = renderer_capacity;

        self.cull_set = build_cull_set(scene, &self.pass1, &self.candidate_list, &self.candidate_count);
        self.pass2_cull_set0 =
            build_pass2_cull_set0(scene, &self.candidate_list, &self.candidate_count, &self.pass2);

        self.cull_secondary = record_cull_secondary(
            scene,
            &self.pass1,
            &self.cull_set,
            &self.occlusion_set,
            &self.candidate_count,
            &self.pass2_dispatch_args,
            renderer_capacity as u32,
        );
        self.cull_pass2_secondary = record_cull_pass2_secondary(
            scene,
            &self.pass2,
            &self.pass2_cull_set0,
            &self.pass2_cull_set1,
            &self.pass2_dispatch_args,
        );

        // Texture arrivals / redirect-buffer growth reach here via
        // `force_full`; rebind the current views + buffers.
        self.texture_set = build_texture_set(scene);
        self.scene_secondary_pass1 = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &self.pass1.graphics_set,
            &self.texture_set,
            scene.mesh_store,
            &self.pass1.indirect_args,
            slot_count,
            self.extent,
        );
        self.scene_secondary_pass2 = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &self.pass2.graphics_set,
            &self.texture_set,
            scene.mesh_store,
            &self.pass2.indirect_args,
            slot_count,
            self.extent,
        );
        true
    }

    /// Whether the current draw plan / renderer capacity needs a **full**
    /// rebuild (new buffers + descriptor set + secondaries + frame slots) vs.
    /// just an in-place rewrite of the indirect templates' per-slot bases.
    ///
    /// `force` is set by the caller when a cull-bound external buffer (SoT,
    /// `GPURenderers`, redirect, mesh table) reallocated.
    pub fn needs_structural_rebuild(
        &self,
        plan: &DrawPlan,
        renderer_capacity: usize,
        force: bool,
    ) -> bool {
        force
            || plan.total_renderers as usize > self.pass1.mvp_capacity
            || plan.commands.len() > self.pass1.slot_capacity
            || plan.commands.len() != self.slot_count
            || renderer_capacity != self.cull_range
    }

    /// Cheap path: rewrite both passes' indirect templates' per-slot commands
    /// in place (the prefix-summed bases shift on every spawn). The cull /
    /// scene secondaries and the cull sets all stay valid — they bind the
    /// *buffers*, and the per-frame `template → args` copy (inside each
    /// pass's cull secondary) picks up the new contents.
    ///
    /// **The host write must be gated against in-flight reads** — the
    /// templates are read by every in-flight frame's reset copy, so call
    /// this only after `WorldTransformGpu::host_wait_for_previous_compute`.
    pub fn write_template_bases(&self, plan: &DrawPlan) {
        write_indirect_template(&self.pass1.indirect_template, &plan.commands);
        write_indirect_template(&self.pass2.indirect_template, &plan.commands);
    }

    // ── Accessors ───────────────────────────────────────────────────────

    #[allow(dead_code)]
    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }
    pub fn color_image(&self) -> &Arc<Image> {
        &self.color_image
    }
    #[allow(dead_code)]
    pub fn depth_image(&self) -> &Arc<Image> {
        &self.depth_image
    }
    pub fn color_view(&self) -> &Arc<ImageView> {
        &self.color_view
    }
    pub fn depth_view(&self) -> &Arc<ImageView> {
        &self.depth_view
    }
    /// Pass 1's `multiDrawIndexedIndirect` — draws instances visible
    /// against last frame's (reprojected) Hi-Z.
    pub fn scene_secondary_pass1(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.scene_secondary_pass1
    }
    /// Pass 2's `multiDrawIndexedIndirect` — draws instances confirmed
    /// visible against this frame's own Hi-Z. Record against a `Load`
    /// (not `Clear`) attachment scope.
    pub fn scene_secondary_pass2(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.scene_secondary_pass2
    }
    /// Pass 1 cull (mvp-build) compute secondary — executed once per frame
    /// from each FrameSlot primary, before the first scene render.
    pub fn cull_secondary(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.cull_secondary
    }
    /// Hi-Z pyramid build secondary — executed after pass 1's render, before
    /// pass 2's cull (reads the depth attachment pass 1 just wrote).
    pub fn hiz_build_secondary(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.hiz_build_secondary
    }
    /// Pass 2 cull (mvp-build) compute secondary — `dispatch_indirect`,
    /// executed after `hiz_build_secondary`, before the second scene render.
    pub fn cull_pass2_secondary(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.cull_pass2_secondary
    }
    /// History-update secondary — copies this frame's Hi-Z / view_proj into
    /// the "previous frame" slots pass 1 reads next frame. Has no
    /// dependency on pass 2's render, so it can execute any time after
    /// `hiz_build_secondary` (see the module doc comment).
    pub fn history_update_secondary(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.history_update_secondary
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Allocation / recording helpers
// ─────────────────────────────────────────────────────────────────────────────

fn allocate_attachments(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    extent: [u32; 2],
) -> (Arc<Image>, Arc<ImageView>, Arc<Image>, Arc<ImageView>) {
    let [w, h] = extent;
    let color_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: CAMERA_COLOR_FORMAT,
            extent: [w, h, 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .expect("Failed to create offscreen color image");
    let color_view = ImageView::new_default(color_image.clone())
        .expect("Failed to create offscreen color image view");

    let depth_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: CAMERA_DEPTH_FORMAT,
            extent: [w, h, 1],
            // SAMPLED: `hiz_reduce_depth.comp` reads this frame's freshly
            // drawn (pass-1) depth to build the Hi-Z pyramid's first mip.
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .expect("Failed to create offscreen depth image");
    let depth_view = ImageView::new_default(depth_image.clone())
        .expect("Failed to create offscreen depth image view");

    (color_image, color_view, depth_image, depth_view)
}

/// Allocate the per-visible-instance buffers — the device-local `[f32; 16]`
/// MVP buffer and the parallel `u32` concrete-material-id buffer, both of
/// `capacity` slots — plus the graphics descriptor set that points at them.
fn allocate_matrices_and_set(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    pipeline: &Arc<GraphicsPipeline>,
    capacity: usize,
) -> (Subbuffer<[[f32; 16]]>, Subbuffer<[u32]>, Arc<DescriptorSet>) {
    let device_matrices: Subbuffer<[[f32; 16]]> = Buffer::new_slice::<[f32; 16]>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        capacity.max(1) as u64,
    )
    .expect("Failed to allocate device matrix buffer");
    let inst_material: Subbuffer<[u32]> = Buffer::new_slice::<u32>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        capacity.max(1) as u64,
    )
    .expect("Failed to allocate instance material buffer");

    let set_layout = pipeline.layout().set_layouts()[0].clone();
    let graphics_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        set_layout,
        [
            WriteDescriptorSet::buffer(0, device_matrices.clone()),
            WriteDescriptorSet::buffer(1, inst_material.clone()),
        ],
        [],
    )
    .expect("Failed to allocate matrices descriptor set");

    (device_matrices, inst_material, graphics_set)
}

/// Allocate the indirect-command buffers: a host-visible **template** (the
/// CPU writes the per-slot commands with `instance_count` zeroed) and the
/// device-local **args** (reset from the template each frame, written by the
/// cull's atomics, read by the indirect draw).
fn allocate_indirect_buffers(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    capacity: usize,
) -> (
    Subbuffer<[DrawIndexedIndirectCommand]>,
    Subbuffer<[DrawIndexedIndirectCommand]>,
) {
    let cap = capacity.max(1);
    let template = Buffer::new_slice::<DrawIndexedIndirectCommand>(
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
        cap as u64,
    )
    .expect("Failed to allocate indirect template buffer");
    let args = Buffer::new_slice::<DrawIndexedIndirectCommand>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::INDIRECT_BUFFER
                | BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        cap as u64,
    )
    .expect("Failed to allocate indirect args buffer");
    (template, args)
}

/// Write the draw plan's per-slot commands into the host template.
fn write_indirect_template(
    template: &Subbuffer<[DrawIndexedIndirectCommand]>,
    commands: &[DrawIndexedIndirectCommand],
) {
    let mut guard = template.write().expect("indirect_template.write");
    guard[..commands.len()].copy_from_slice(commands);
    // Tail (capacity > commands.len()) left undefined — never read (the draw
    // slices to `slot_count`, the cull only touches slots in range).
}

/// Allocate the candidate record list (capacity == `renderer_capacity`, one
/// `[f32; 16]` slot per record — matches `Candidate`'s 4×vec4 GLSL layout)
/// and its live-count buffer (reset to 0 each frame via `fill_buffer`
/// inside `cull_secondary`, so `TRANSFER_DST` is required).
fn allocate_candidate_buffers(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    capacity: usize,
) -> (Subbuffer<[[f32; 16]]>, Subbuffer<[u32]>) {
    let list = Buffer::new_slice::<[f32; 16]>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        capacity.max(1) as u64,
    )
    .expect("Failed to allocate candidate list buffer");
    let count = Buffer::new_slice::<u32>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        1,
    )
    .expect("Failed to allocate candidate count buffer");
    (list, count)
}

/// Allocate pass 2's single-element `dispatch_indirect` argument buffer.
/// Never reallocated — always exactly one `DispatchIndirectCommand`.
fn allocate_pass2_dispatch_args(
    memory_allocator: &Arc<StandardMemoryAllocator>,
) -> Subbuffer<[DispatchIndirectCommand]> {
    Buffer::new_slice::<DispatchIndirectCommand>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::INDIRECT_BUFFER | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        1,
    )
    .expect("Failed to allocate pass2 dispatch-indirect args buffer")
}

/// Allocate the camera's `prev_view_proj` history buffer (single mat4,
/// fixed identity, overwritten in place each frame by
/// `history_update_secondary`'s `copy_buffer`).
fn allocate_prev_view_proj(
    memory_allocator: &Arc<StandardMemoryAllocator>,
) -> Subbuffer<[[f32; 16]]> {
    Buffer::new_slice::<[f32; 16]>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        1,
    )
    .expect("Failed to allocate prev_view_proj buffer")
}

/// Build the shared depth-only NEAREST/ClampToEdge sampler used by every
/// Hi-Z-related combined-image-sampler binding (`texelFetch` ignores its
/// filter/address mode — see the field doc comment on `RenderCamera::
/// hiz_sampler`).
fn build_hiz_sampler(_queue_family_index: u32, device: Arc<Device>) -> Arc<Sampler> {
    Sampler::new(
        device,
        SamplerCreateInfo {
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            address_mode: [SamplerAddressMode::ClampToEdge; 3],
            ..Default::default()
        },
    )
    .expect("Failed to create Hi-Z sampler")
}

/// Build the graphics material/texture set (set 1): the texture registry's
/// redirect buffer, the material registry's redirect buffer, the material
/// SSBO, and the fixed-size sampled-image array (placeholder-padded — see
/// [`GpuTextureStore`]).
fn build_texture_set(scene: &CameraSceneResources<'_>) -> Arc<DescriptorSet> {
    let set_layout = scene.pipeline.layout().set_layouts()[1].clone();
    DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        set_layout,
        [
            WriteDescriptorSet::buffer(0, scene.texture_store.redirect_buffer().clone()),
            WriteDescriptorSet::buffer(1, scene.material_store.redirect_buffer().clone()),
            WriteDescriptorSet::buffer(2, scene.material_store.materials_buffer().clone()),
            WriteDescriptorSet::image_view_sampler_array(
                3,
                0,
                scene.texture_store.descriptor_array(),
            ),
        ],
        [],
    )
    .expect("Failed to allocate texture descriptor set")
}

/// Build pass 1's cull descriptor set (set 0): SoT, GPURenderers, redirect,
/// mesh table, MVP output, the indirect commands (as a flat `u32[]`), the
/// per-transform Parents buffer the chain walk reads, the per-slot authored
/// materials, the per-visible-instance material output, and the candidate
/// list + its live counter.
fn build_cull_set(
    scene: &CameraSceneResources<'_>,
    pass1: &DrawResources,
    candidate_list: &Subbuffer<[[f32; 16]]>,
    candidate_count: &Subbuffer<[u32]>,
) -> Arc<DescriptorSet> {
    let world = scene.world_transforms;
    DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        world.mvp_build_set0_layout().clone(),
        [
            WriteDescriptorSet::buffer(0, world.sot_positions().clone()),
            WriteDescriptorSet::buffer(1, world.sot_rotations().clone()),
            WriteDescriptorSet::buffer(2, world.sot_scales().clone()),
            WriteDescriptorSet::buffer(3, scene.gpu_renderers.buffer().clone()),
            WriteDescriptorSet::buffer(4, scene.mesh_store.redirect_buffer().clone()),
            WriteDescriptorSet::buffer(5, scene.mesh_store.mesh_table_buffer().clone()),
            WriteDescriptorSet::buffer(6, pass1.device_matrices.clone()),
            WriteDescriptorSet::buffer(7, pass1.indirect_args.clone().reinterpret::<[u32]>()),
            WriteDescriptorSet::buffer(8, world.sot_parents().clone()),
            WriteDescriptorSet::buffer(9, scene.mesh_store.slot_material_buffer().clone()),
            WriteDescriptorSet::buffer(10, pass1.inst_material.clone()),
            WriteDescriptorSet::buffer(11, candidate_list.clone()),
            WriteDescriptorSet::buffer(12, candidate_count.clone()),
        ],
        [],
    )
    .expect("Failed to allocate cull set")
}

/// Build pass 1's camera-owned occlusion set (set 1): this frame's
/// `view_proj` (the shared `sot_view_proj`), last frame's `view_proj`
/// (camera-owned history), and last frame's Hi-Z pyramid (sampled).
fn build_occlusion_set(
    scene: &CameraSceneResources<'_>,
    prev_view_proj: &Subbuffer<[[f32; 16]]>,
    hiz_prev: &HizPyramid,
    hiz_sampler: &Arc<Sampler>,
) -> Arc<DescriptorSet> {
    let world = scene.world_transforms;
    let layout = world.mvp_build_pipeline().layout().set_layouts()[1].clone();
    DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        layout,
        [
            WriteDescriptorSet::buffer(0, world.sot_view_proj().clone()),
            WriteDescriptorSet::buffer(1, prev_view_proj.clone()),
            WriteDescriptorSet::image_view_sampler(
                2,
                hiz_prev.sampled_view.clone(),
                hiz_sampler.clone(),
            ),
        ],
        [],
    )
    .expect("Failed to allocate occlusion set")
}

/// Build pass 2's cull set 0: the candidate list + counter (read), pass 2's
/// own indirect args (rw, as a flat `u32[]`), MVP output, and per-instance
/// material output.
fn build_pass2_cull_set0(
    scene: &CameraSceneResources<'_>,
    candidate_list: &Subbuffer<[[f32; 16]]>,
    candidate_count: &Subbuffer<[u32]>,
    pass2: &DrawResources,
) -> Arc<DescriptorSet> {
    let layout = scene.mvp_build_pass2_pipeline.layout().set_layouts()[0].clone();
    DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        layout,
        [
            WriteDescriptorSet::buffer(0, candidate_list.clone()),
            WriteDescriptorSet::buffer(1, candidate_count.clone()),
            WriteDescriptorSet::buffer(2, pass2.indirect_args.clone().reinterpret::<[u32]>()),
            WriteDescriptorSet::buffer(3, pass2.device_matrices.clone()),
            WriteDescriptorSet::buffer(4, pass2.inst_material.clone()),
        ],
        [],
    )
    .expect("Failed to allocate pass2 cull set0")
}

/// Build pass 2's cull set 1: this frame's `view_proj` + this frame's own
/// Hi-Z pyramid (sampled).
fn build_pass2_cull_set1(
    scene: &CameraSceneResources<'_>,
    hiz_current: &HizPyramid,
    hiz_sampler: &Arc<Sampler>,
) -> Arc<DescriptorSet> {
    let layout = scene.mvp_build_pass2_pipeline.layout().set_layouts()[1].clone();
    DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        layout,
        [
            WriteDescriptorSet::buffer(0, scene.world_transforms.sot_view_proj().clone()),
            WriteDescriptorSet::image_view_sampler(
                1,
                hiz_current.sampled_view.clone(),
                hiz_sampler.clone(),
            ),
        ],
        [],
    )
    .expect("Failed to allocate pass2 cull set1")
}

/// Build the tiny "args builder" set: the candidate counter (read) and
/// pass 2's `dispatch_indirect` args buffer (write).
fn build_args_builder_set(
    scene: &CameraSceneResources<'_>,
    candidate_count: &Subbuffer<[u32]>,
    pass2_dispatch_args: &Subbuffer<[DispatchIndirectCommand]>,
) -> Arc<DescriptorSet> {
    let layout = scene.cull_pass2_args_pipeline.layout().set_layouts()[0].clone();
    DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        layout,
        [
            WriteDescriptorSet::buffer(0, candidate_count.clone()),
            WriteDescriptorSet::buffer(1, pass2_dispatch_args.clone()),
        ],
        [],
    )
    .expect("Failed to allocate cull-pass2-args set")
}

/// Build the Hi-Z build sets: level 0 (depth attachment → `hiz_current`
/// mip 0), one 3-binding set per FUSED pair of remaining levels
/// `(1,2), (3,4), ...` (mip[L-1] → mip[L] → mip[L+1] in one dispatch — see
/// `shaders/hiz_reduce_mip2.comp`), and, iff the remaining-level count
/// (`mip_count - 1`) is odd, one plain 2-binding set for the trailing
/// leftover level that couldn't be paired.
fn build_hiz_sets(
    scene: &CameraSceneResources<'_>,
    depth_view: &Arc<ImageView>,
    hiz_current: &HizPyramid,
    hiz_sampler: &Arc<Sampler>,
) -> (Arc<DescriptorSet>, Vec<Arc<DescriptorSet>>, Option<Arc<DescriptorSet>>) {
    let level0_layout = scene.hiz_reduce_depth_pipeline.layout().set_layouts()[0].clone();
    let level0_set = DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        level0_layout,
        [
            WriteDescriptorSet::image_view_sampler(0, depth_view.clone(), hiz_sampler.clone()),
            WriteDescriptorSet::image_view(1, hiz_current.mip_views[0].clone()),
        ],
        [],
    )
    .expect("Failed to allocate Hi-Z level0 set");

    // Remaining levels are 1..mip_count. Pair them up (1,2), (3,4), ... —
    // an odd remaining count leaves the last level (`mip_count - 1`)
    // trailing, unpaired.
    let mip2_layout = scene.hiz_reduce_mip2_pipeline.layout().set_layouts()[0].clone();
    let remaining = hiz_current.mip_count - 1;
    let pair_count = remaining / 2;
    let mip2_sets: Vec<Arc<DescriptorSet>> = (0..pair_count)
        .map(|i| {
            let l = 1 + 2 * i; // first level of this pair
            DescriptorSet::new(
                scene.descriptor_set_allocator.clone(),
                mip2_layout.clone(),
                [
                    WriteDescriptorSet::image_view(0, hiz_current.mip_views[(l - 1) as usize].clone()),
                    WriteDescriptorSet::image_view(1, hiz_current.mip_views[l as usize].clone()),
                    WriteDescriptorSet::image_view(2, hiz_current.mip_views[(l + 1) as usize].clone()),
                ],
                [],
            )
            .expect("Failed to allocate Hi-Z mip2 set")
        })
        .collect();

    let trailing_set = if remaining % 2 == 1 {
        let last = hiz_current.mip_count - 1;
        let mip_layout = scene.hiz_reduce_mip_pipeline.layout().set_layouts()[0].clone();
        Some(
            DescriptorSet::new(
                scene.descriptor_set_allocator.clone(),
                mip_layout,
                [
                    WriteDescriptorSet::image_view(
                        0,
                        hiz_current.mip_views[(last - 1) as usize].clone(),
                    ),
                    WriteDescriptorSet::image_view(1, hiz_current.mip_views[last as usize].clone()),
                ],
                [],
            )
            .expect("Failed to allocate Hi-Z trailing level set"),
        )
    } else {
        None
    };

    (level0_set, mip2_sets, trailing_set)
}

/// Record pass 1's cull secondary: reset the indirect `instance_count`s and
/// the candidate counter, dispatch the frustum+occlusion cull over the
/// renderer range, then dispatch the tiny args-builder that turns the
/// resulting candidate count into pass 2's `dispatch_indirect` args.
/// Recorded `SimultaneousUse` (shared across FrameSlots).
fn record_cull_secondary(
    scene: &CameraSceneResources<'_>,
    pass1: &DrawResources,
    cull_set: &Arc<DescriptorSet>,
    occlusion_set: &Arc<DescriptorSet>,
    candidate_count: &Subbuffer<[u32]>,
    pass2_dispatch_args: &Subbuffer<[DispatchIndirectCommand]>,
    renderer_capacity: u32,
) -> Arc<SecondaryAutoCommandBuffer> {
    let pipeline = scene.world_transforms.mvp_build_pipeline();
    let layout = pipeline.layout().clone();
    let groups = renderer_capacity.div_ceil(CULL_WORKGROUP_SIZE).max(1);
    let pc = shaders::mvp_build_cs::PC { renderer_capacity };

    let mut builder = AutoCommandBufferBuilder::secondary(
        scene.cb_allocator.clone(),
        scene.queue_family_index,
        CommandBufferUsage::SimultaneousUse,
        CommandBufferInheritanceInfo::default(),
    )
    .expect("cull secondary builder");

    // Reset every slot's instance_count to 0, and the candidate live count.
    // Vulkano auto-syncs these transfer writes against the cull dispatch's
    // atomic read-modify-writes.
    builder
        .copy_buffer(CopyBufferInfo::buffers(
            pass1.indirect_template.clone(),
            pass1.indirect_args.clone(),
        ))
        .expect("reset indirect instance counts");
    builder
        .fill_buffer(candidate_count.clone(), 0)
        .expect("reset candidate count");

    builder
        .bind_pipeline_compute(pipeline.clone())
        .expect("bind cull pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            layout.clone(),
            0,
            (cull_set.clone(), occlusion_set.clone()),
        )
        .expect("bind cull sets")
        .push_constants(layout, 0, pc)
        .expect("push cull constants");
    // Safety: dispatch count derived from `renderer_capacity`; the shader
    // bounds-checks against the push-constant.
    unsafe {
        builder.dispatch([groups, 1, 1]).expect("dispatch cull");
    }

    // Tiny args-builder: converts the candidate count this dispatch just
    // produced into pass 2's `dispatch_indirect` group counts. Same
    // secondary so vulkano auto-sync orders it after the atomic writes
    // above.
    let args_pipeline = scene.cull_pass2_args_pipeline;
    let args_set = build_args_builder_set(scene, candidate_count, pass2_dispatch_args);
    builder
        .bind_pipeline_compute(args_pipeline.clone())
        .expect("bind args-builder pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            args_pipeline.layout().clone(),
            0,
            args_set,
        )
        .expect("bind args-builder set");
    // Safety: 1×1×1 dispatch is unconditionally valid.
    unsafe {
        builder.dispatch([1, 1, 1]).expect("dispatch args-builder");
    }

    builder.build().expect("build cull secondary")
}

/// Record pass 2's cull secondary: reset pass 2's own indirect
/// `instance_count`s, then `dispatch_indirect` the occlusion-only re-test
/// over the live candidate count. Recorded `SimultaneousUse`.
fn record_cull_pass2_secondary(
    scene: &CameraSceneResources<'_>,
    pass2: &DrawResources,
    pass2_cull_set0: &Arc<DescriptorSet>,
    pass2_cull_set1: &Arc<DescriptorSet>,
    pass2_dispatch_args: &Subbuffer<[DispatchIndirectCommand]>,
) -> Arc<SecondaryAutoCommandBuffer> {
    let pipeline = scene.mvp_build_pass2_pipeline;
    let layout = pipeline.layout().clone();

    let mut builder = AutoCommandBufferBuilder::secondary(
        scene.cb_allocator.clone(),
        scene.queue_family_index,
        CommandBufferUsage::SimultaneousUse,
        CommandBufferInheritanceInfo::default(),
    )
    .expect("cull pass2 secondary builder");

    builder
        .copy_buffer(CopyBufferInfo::buffers(
            pass2.indirect_template.clone(),
            pass2.indirect_args.clone(),
        ))
        .expect("reset pass2 indirect instance counts");

    builder
        .bind_pipeline_compute(pipeline.clone())
        .expect("bind cull pass2 pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            layout,
            0,
            (pass2_cull_set0.clone(), pass2_cull_set1.clone()),
        )
        .expect("bind cull pass2 sets");
    // Safety: `pass2_dispatch_args` is written by `cull_secondary`'s
    // args-builder dispatch earlier in the same FrameSlot primary, before
    // this secondary executes (see `lib.rs::build_frame_slot`); the
    // group-count values it contains are `ceil(candidate_count / 64)`,
    // always within the candidate list's allocated capacity.
    unsafe {
        builder
            .dispatch_indirect(pass2_dispatch_args.clone())
            .expect("dispatch_indirect cull pass2");
    }

    builder.build().expect("build cull pass2 secondary")
}

/// Record the Hi-Z pyramid build secondary: level 0 (depth attachment →
/// `hiz_current` mip 0), then one dispatch per FUSED pair of remaining
/// levels (mip[L-1] → mip[L] → mip[L+1] — see `shaders/hiz_reduce_mip2.comp`),
/// then (if the remaining-level count is odd) one final plain single-level
/// dispatch for the trailing leftover level. Recorded `SimultaneousUse`;
/// re-recorded only on extent change (mip count/dims derive from the depth
/// buffer's resolution).
fn record_hiz_build_secondary(
    scene: &CameraSceneResources<'_>,
    hiz_level0_set: &Arc<DescriptorSet>,
    hiz_mip2_sets: &[Arc<DescriptorSet>],
    hiz_trailing_set: &Option<Arc<DescriptorSet>>,
    hiz_mip0_extent: [u32; 2],
) -> Arc<SecondaryAutoCommandBuffer> {
    let mut builder = AutoCommandBufferBuilder::secondary(
        scene.cb_allocator.clone(),
        scene.queue_family_index,
        CommandBufferUsage::SimultaneousUse,
        CommandBufferInheritanceInfo::default(),
    )
    .expect("hiz build secondary builder");

    let depth_pipeline = scene.hiz_reduce_depth_pipeline;
    let [gx, gy] = dispatch_groups_2d(hiz_mip0_extent);
    builder
        .bind_pipeline_compute(depth_pipeline.clone())
        .expect("bind hiz depth pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            depth_pipeline.layout().clone(),
            0,
            hiz_level0_set.clone(),
        )
        .expect("bind hiz level0 set");
    // Safety: dispatch dims derived from `hiz_mip0_extent`; the shader
    // bounds-checks against `imageSize(u_dst)`, which matches.
    unsafe {
        builder.dispatch([gx, gy, 1]).expect("dispatch hiz level0");
    }

    // Each fused-pair dispatch is sized off the PAIR'S FIRST level's
    // extent — one workgroup produces up to an 8x8 tile of that level
    // (same sizing as a plain single-level dispatch would use for it) and
    // opportunistically also produces the second level from its own
    // workgroup-local data. See `shaders/hiz_reduce_mip2.comp`.
    let mip2_pipeline = scene.hiz_reduce_mip2_pipeline;
    for (i, set) in hiz_mip2_sets.iter().enumerate() {
        let l = 1 + 2 * i as u32; // first level of this pair
        let level_extent = hiz_level_extent(hiz_mip0_extent, l);
        let [gx, gy] = dispatch_groups_2d(level_extent);
        builder
            .bind_pipeline_compute(mip2_pipeline.clone())
            .expect("bind hiz mip2 pipeline")
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                mip2_pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .expect("bind hiz mip2 set");
        // Safety: dispatch dims derived from this pair's first level's
        // extent; the shader bounds-checks both `u_mid` and `u_dst`
        // against their own `imageSize`, which matches.
        unsafe {
            builder.dispatch([gx, gy, 1]).expect("dispatch hiz mip2");
        }
    }

    if let Some(set) = hiz_trailing_set {
        let mip_pipeline = scene.hiz_reduce_mip_pipeline;
        // The trailing level is the very last one, `mip_count - 1` — same
        // index the fused-pair loop above would have reached next had the
        // remaining-level count been even.
        let last = 1 + 2 * hiz_mip2_sets.len() as u32;
        let level_extent = hiz_level_extent(hiz_mip0_extent, last);
        let [gx, gy] = dispatch_groups_2d(level_extent);
        builder
            .bind_pipeline_compute(mip_pipeline.clone())
            .expect("bind hiz trailing pipeline")
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                mip_pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .expect("bind hiz trailing set");
        // Safety: dispatch dims derived from the trailing level's extent;
        // the shader bounds-checks against `imageSize(u_dst)`, which matches.
        unsafe {
            builder.dispatch([gx, gy, 1]).expect("dispatch hiz trailing");
        }
    }

    builder.build().expect("build hiz build secondary")
}

fn dispatch_groups_2d(extent: [u32; 2]) -> [u32; 2] {
    [
        extent[0].div_ceil(HIZ_WORKGROUP_SIZE).max(1),
        extent[1].div_ceil(HIZ_WORKGROUP_SIZE).max(1),
    ]
}

/// Record the history-update secondary: copy this frame's Hi-Z pyramid and
/// `view_proj` into the fixed "previous frame" buffer/image identities
/// pass 1 reads next frame. No dependency on pass 2's render (see the
/// module doc comment) — only on `hiz_build_secondary` having produced
/// `hiz_current` and on `sot_view_proj` holding this frame's promoted VP
/// (true from the front of the FrameSlot primary onward). Recorded
/// `SimultaneousUse`; re-recorded only on extent change (the per-mip copy
/// regions depend on the pyramids' dimensions).
fn record_history_update_secondary(
    scene: &CameraSceneResources<'_>,
    hiz_current: &HizPyramid,
    hiz_prev: &HizPyramid,
    prev_view_proj: &Subbuffer<[[f32; 16]]>,
) -> Arc<SecondaryAutoCommandBuffer> {
    let mut builder = AutoCommandBufferBuilder::secondary(
        scene.cb_allocator.clone(),
        scene.queue_family_index,
        CommandBufferUsage::SimultaneousUse,
        CommandBufferInheritanceInfo::default(),
    )
    .expect("history update secondary builder");

    builder
        .copy_buffer(CopyBufferInfo::buffers(
            scene.world_transforms.sot_view_proj().clone(),
            prev_view_proj.clone(),
        ))
        .expect("copy sot_view_proj -> prev_view_proj");

    let regions: Vec<ImageCopy> = (0..hiz_current.mip_count)
        .map(|level| {
            let [w, h] = hiz_level_extent(hiz_current.mip0_extent, level);
            ImageCopy {
                src_subresource: ImageSubresourceLayers {
                    aspects: vulkano::image::ImageAspects::COLOR,
                    mip_level: level,
                    array_layers: 0..1,
                },
                dst_subresource: ImageSubresourceLayers {
                    aspects: vulkano::image::ImageAspects::COLOR,
                    mip_level: level,
                    array_layers: 0..1,
                },
                extent: [w, h, 1],
                ..Default::default()
            }
        })
        .collect();
    builder
        .copy_image(CopyImageInfo {
            regions: regions.into(),
            ..CopyImageInfo::images(hiz_current.image.clone(), hiz_prev.image.clone())
        })
        .expect("copy hiz_current -> hiz_prev");

    builder.build().expect("build history update secondary")
}

/// Record the scene secondary: a single `vkCmdDrawIndexedIndirect` over
/// `indirect_args[0..slot_count]` against the shared mega buffers. Used for
/// both pass 1's and pass 2's draws — identical recording, different
/// (independent) `graphics_set` / `indirect_args` per call.
fn record_scene_secondary(
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
    pipeline: &Arc<GraphicsPipeline>,
    graphics_set: &Arc<DescriptorSet>,
    texture_set: &Arc<DescriptorSet>,
    mesh_store: &GpuMeshStore,
    indirect_args: &Subbuffer<[DrawIndexedIndirectCommand]>,
    slot_count: usize,
    extent: [u32; 2],
) -> Arc<SecondaryAutoCommandBuffer> {
    let [cam_w, cam_h] = extent;

    let inheritance = CommandBufferInheritanceInfo {
        render_pass: Some(
            CommandBufferInheritanceRenderingInfo {
                color_attachment_formats: vec![Some(CAMERA_COLOR_FORMAT)],
                depth_attachment_format: Some(CAMERA_DEPTH_FORMAT),
                ..Default::default()
            }
            .into(),
        ),
        ..Default::default()
    };

    let mut builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::SimultaneousUse,
        inheritance,
    )
    .expect("Failed to create scene secondary builder");

    builder
        .set_viewport(
            0,
            smallvec::smallvec![Viewport {
                offset: [0.0, 0.0],
                extent: [cam_w as f32, cam_h as f32],
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
            (graphics_set.clone(), texture_set.clone()),
        )
        .expect("bind_descriptor_sets failed");

    // All meshes share one mega vertex + one mega index buffer; bind once.
    // Each slot's command carries its own `first_index` / `vertex_offset` /
    // `first_instance` / (cull-written) `instance_count`.
    builder
        .bind_vertex_buffers(0, mesh_store.mega_vertex_buffer().clone())
        .expect("bind mega vertex buffer failed")
        .bind_index_buffer(mesh_store.mega_index_buffer().clone())
        .expect("bind mega index buffer failed");

    // One `vkCmdDrawIndexedIndirect` over all slots (drawCount == slot_count;
    // the `multi_draw_indirect` feature permits > 1). Empty slots have
    // instance_count 0 and draw nothing.
    if slot_count > 0 {
        let draws = indirect_args.clone().slice(0..slot_count as u64);
        // Safety: args buffer is INDIRECT_BUFFER-usable; mega index buffer is
        // bound; `first_instance` bounded by the MVP capacity; the indirect
        // device features are enabled at device creation (see RenderApp::new).
        unsafe {
            builder
                .draw_indexed_indirect(draws)
                .expect("draw_indexed_indirect failed");
        }
    }

    builder.build().expect("Failed to build scene secondary")
}
