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
//! `hiz_current → hiz_prev` and `view_proj → prev_view_proj`
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
//!
//! # Debug: frustum-lock (`cull_lock`) and occlusion enable/disable
//!
//! Two independent, runtime-toggleable debug features (`lib.rs` wires them
//! to F9 / F8 respectively):
//!
//! **Frustum-lock** freezes the cull *frustum* at a snapshot
//! (`locked_view_proj`) while the render camera keeps moving, so an object
//! can be watched from any angle to see whether the frozen frustum draws
//! or culls it. `mvp_build.comp`'s frustum test reads a dedicated
//! `cull_view_proj` buffer (set 1, binding 3) instead of the live render
//! VP; [`RenderCamera::write_cull_view_proj`] writes either the live VP or
//! the locked snapshot into it every frame — cheap, no command-buffer
//! re-recording either way.
//!
//! Naively pointing *only* the frustum test at a locked VP would leave
//! Hi-Z occlusion incoherent the moment the render camera diverges from
//! the lock: the occlusion sub-tests sample a Hi-Z pyramid that's always
//! built from whatever the render camera *actually* renders, so testing
//! it against a different, frozen VP would compare against a screen-space
//! footprint the pyramid was never built for. So frustum-lock also freezes
//! the Hi-Z pipeline (`RenderCamera::hiz_frozen`, one frame behind
//! `cull_lock` via [`RenderCamera::apply_pending_hiz_freeze`] — the extra
//! frame lets the *engage* frame's own Hi-Z build run once more first, so
//! the frozen snapshot it leaves behind is actually consistent with
//! `locked_view_proj`): `lib.rs::build_frame_slot` skips
//! `hiz_build_secondary`/`history_update_secondary` while frozen, pinning
//! `hiz_current`/`hiz_prev`/`prev_view_proj` at that consistent snapshot.
//! `cull_pass2_secondary` and pass 2's render scope keep running against
//! whatever the (possibly frozen) pyramids currently hold. Pass 2's own
//! `view_proj` binding (`build_pass2_cull_set1`) reads the same
//! `cull_view_proj` buffer the frustum test does (not the live render VP)
//! so it too stays paired with the frozen `hiz_current` once frozen —
//! everything in the cull path ends up sharing one `view_proj`, exactly
//! the invariant the debug feature is named for.
//!
//! **Occlusion enable/disable** is coarser and rebuild-gated: disabling it
//! (`RenderCamera::set_occlusion_enabled`) re-records `cull_secondary` with
//! a push constant that forces `mvp_build.comp` to skip the occlusion
//! sub-test entirely (never generate candidates — required for
//! correctness, since pass 2 is the only consumer of that list), and
//! signals the caller to rebuild `FrameSlot`s so `build_frame_slot` omits
//! the Hi-Z build / pass 2 cull / pass 2 render / history-update secondaries
//! from the primary altogether — real GPU-work avoidance, not just a
//! shader no-op.

use crate::STAGING_SLOTS;
use std::sync::Arc;

use engine_core::{Entity, WorldId};
use glam::{Mat4, Quat, Vec3};
use parking_lot::Mutex;

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
    pipeline::{
        graphics::viewport::Viewport, ComputePipeline, GraphicsPipeline, PipelineBindPoint,
    },
};

// `Pipeline` trait is needed for `pipeline.layout()` method resolution.
use vulkano::pipeline::Pipeline;

use crate::assets::{GpuMaterialStore, GpuMeshStore, GpuTextureStore};
use crate::gpu_renderers::GpuRenderers;
use crate::overlay::CameraOverlay;
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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CameraResolution {
    /// Track the swapchain extent 1:1. The present-blit is then a straight
    /// copy, which is how a game with no UI reaches the screen.
    MatchSwapchain,
    /// Sized by whatever is showing it — a [`Viewport`](crate::ui::Viewport)
    /// widget, which re-requests its box whenever the layout moves it.
    ///
    /// A camera this size cannot be blitted to the swapchain: it is not the
    /// swapchain's shape and it belongs inside a panel, so the widget
    /// sampling it is what puts it on screen (see `build_frame_slot`).
    Fixed([u32; 2]),
}

impl CameraResolution {
    fn resolve(&self, swapchain_extent: [u32; 2]) -> [u32; 2] {
        match self {
            CameraResolution::MatchSwapchain => swapchain_extent,
            // Zero is a legal box for a UI node and not for an image.
            CameraResolution::Fixed(e) => [e[0].max(1), e[1].max(1)],
        }
    }

    /// Does this policy depend on the swapchain extent?
    pub fn depends_on_swapchain(&self) -> bool {
        matches!(self, CameraResolution::MatchSwapchain)
    }

    /// Can the present-blit copy this camera onto the swapchain? Only when
    /// it is the swapchain's own size — otherwise something else composites
    /// it and the blit would be a stretch of the wrong thing.
    pub fn blits_to_swapchain(&self) -> bool {
        matches!(self, CameraResolution::MatchSwapchain)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CameraHandle — the half of a camera that owns no Vulkan
// ─────────────────────────────────────────────────────────────────────────────

/// How many cameras a process can show at once. Each costs attachments, a
/// Hi-Z pyramid (ADR-0005) and one reserved bindless slot — so this is a
/// small number on purpose.
pub const MAX_CAMERAS: usize = 8;

/// Perspective parameters plus the aspect of whatever the camera renders
/// into, which only the renderer knows.
#[derive(Clone, Copy)]
struct Projection {
    fov_y_radians: f32,
    z_near: f32,
    z_far: f32,
    aspect: f32,
}

impl Default for Projection {
    fn default() -> Self {
        Self {
            fov_y_radians: 60_f32.to_radians(),
            z_near: 0.1,
            z_far: 10_000.0,
            aspect: 1.0,
        }
    }
}

impl Projection {
    /// Vulkan-NDC projection (Y axis flipped from glam's GL convention).
    fn matrix(&self) -> Mat4 {
        let mut p = Mat4::perspective_rh(
            self.fov_y_radians,
            self.aspect.max(1e-6),
            self.z_near,
            self.z_far,
        );
        p.y_axis.y *= -1.0;
        p
    }
}

/// The camera state anything may write: the matrix the renderer will upload
/// next frame, the box the panel showing it published, and its projection.
///
/// [`RenderCamera`] holds the same `Arc`, so the device half and whoever
/// drives the camera are looking at one object rather than two that have to
/// be kept in step.
pub struct CameraState {
    worlds: Mutex<Vec<WorldId>>,
    slot: usize,
    view: Mutex<(Mat4, Vec3)>,
    rect: Mutex<Option<[f32; 4]>>,
    proj: Mutex<Projection>,
    /// The editor's ground plane: cell size, major multiple and fade
    /// radius, or `None` for a camera nobody asked to show one.
    grid: Mutex<Option<[f32; 4]>>,
}

/// Every camera in the process, in slot order. Strong refs: a camera outlives
/// the component that made it, because the renderer's device half is keyed by
/// slot and slots are never reused.
static CAMERAS: Mutex<Vec<Arc<CameraState>>> = Mutex::new(Vec::new());

/// A camera, by reference. Cloneable and cheap; the thing components,
/// controllers and panel widgets pass around.
#[derive(Clone)]
pub struct CameraHandle(Arc<CameraState>);

impl CameraHandle {
    /// A camera drawing `world`, sized by whatever ends up showing it.
    pub fn new(world: WorldId) -> Self {
        let mut all = CAMERAS.lock();
        assert!(all.len() < MAX_CAMERAS, "at most {MAX_CAMERAS} cameras");
        let state = Arc::new(CameraState {
            worlds: Mutex::new(vec![world]),
            slot: all.len(),
            view: Mutex::new((Mat4::IDENTITY, Vec3::ZERO)),
            rect: Mutex::new(None),
            proj: Mutex::new(Projection::default()),
            grid: Mutex::new(None),
        });
        all.push(state.clone());
        Self(state)
    }

    /// The worlds this camera composites, in draw order — none of them
    /// necessarily the one its driver lives in (ADR-0011 §2).
    pub fn worlds(&self) -> Vec<WorldId> {
        self.0.worlds.lock().clone()
    }

    /// Draw `world` into this camera's image too, on top of what it already
    /// draws: the gizmos-over-document composite (ADR-0011 §5). One depth
    /// buffer, so the layers interleave rather than stack. A repeat is
    /// ignored — a world drawn twice would just z-fight with itself.
    pub fn draw_world(&self, world: WorldId) {
        let mut worlds = self.0.worlds.lock();
        if !worlds.contains(&world) {
            worlds.push(world);
        }
    }

    /// Its reserved bindless slot: what a widget samples to show it.
    pub fn slot(&self) -> usize {
        self.0.slot
    }

    /// The matrix the renderer uploads next frame, and the eye position
    /// `scene.frag`'s PBR view vector needs alongside it.
    pub fn view_proj(&self) -> (Mat4, Vec3) {
        *self.0.view.lock()
    }

    pub fn set_view_proj(&self, view_proj: Mat4, eye: Vec3) {
        *self.0.view.lock() = (view_proj, eye);
    }

    /// Look down the entity's local `-Z` from `pos`, through this camera's
    /// own projection — so no caller has to know the target's aspect.
    pub fn set_from_trs(&self, pos: Vec3, rot: Quat) {
        let view = Mat4::look_to_rh(pos, rot * Vec3::NEG_Z, rot * Vec3::Y);
        self.set_view_proj(self.0.proj.lock().matrix() * view, pos);
    }

    /// Vertical field of view, which is what turns a distance into the
    /// world size a gizmo has to be to cover a fixed slice of the panel.
    pub fn fov_y(&self) -> f32 {
        self.0.proj.lock().fov_y_radians
    }

    /// Show a world grid under this camera: `[cell, major multiple, fade
    /// radius, unused]`. Editor chrome — a game's camera leaves it `None`.
    pub fn set_grid(&self, grid: Option<[f32; 4]>) {
        *self.0.grid.lock() = grid;
    }

    pub fn grid(&self) -> Option<[f32; 4]> {
        *self.0.grid.lock()
    }

    pub fn set_projection(&self, fov_y_radians: f32, z_near: f32, z_far: f32) {
        let mut p = self.0.proj.lock();
        (p.fov_y_radians, p.z_near, p.z_far) = (fov_y_radians, z_near, z_far);
    }

    /// Published by the renderer once the target is sized, so the next
    /// [`set_from_trs`](Self::set_from_trs) projects at the panel's shape.
    pub(crate) fn set_aspect(&self, aspect: f32) {
        self.0.proj.lock().aspect = aspect;
    }

    /// Where the panel showing this camera landed. `None` — nothing shows
    /// it — is the whole window, which is what a game means without saying
    /// it. A zero box is a third thing: shown, but with no box right now.
    pub fn set_rect(&self, rect: Option<[f32; 4]>) {
        *self.0.rect.lock() = rect;
    }

    pub fn rect(&self) -> Option<[f32; 4]> {
        *self.0.rect.lock()
    }

    /// Is `p` over this camera's panel? What keeps a drag in one document
    /// from spinning the camera in the one beside it.
    pub fn contains(&self, p: [f32; 2]) -> bool {
        self.rect()
            .is_none_or(|r| (0..2).all(|k| p[k] >= r[k] && p[k] < r[k] + r[k + 2]))
    }
}

/// How many cameras exist. Zero until something makes one — a game's arrives
/// with its [`CameraComponent`](crate::CameraComponent)'s queued spawn.
pub fn camera_count() -> usize {
    CAMERAS.lock().len()
}

/// The camera in `slot`, if it exists.
pub(crate) fn camera(slot: usize) -> Option<CameraHandle> {
    CAMERAS.lock().get(slot).cloned().map(CameraHandle)
}

/// Cameras driven by a `CameraComponent`, and the entity each takes its pose
/// from.
static BOUND: Mutex<Vec<(WorldId, Entity, CameraHandle)>> = Mutex::new(Vec::new());

pub(crate) fn bind(world: WorldId, entity: Entity, camera: CameraHandle) {
    BOUND.lock().push((world, entity, camera));
}

pub(crate) fn unbind(world: WorldId, entity: Entity) {
    BOUND.lock().retain(|b| (b.0, b.1) != (world, entity));
}

/// Every bound camera takes its entity's settled pose.
///
/// Runs after the sweep and never inside it: component order within a sweep
/// is nondeterministic, so a matrix built mid-sweep races every transform
/// write, including the camera's own parent chain.
pub(crate) fn drive_bound_cameras() {
    for (world, entity, camera) in BOUND.lock().iter() {
        // A world dropped out from under a live binding holds its last
        // matrix rather than snapping to the origin.
        let Some(w) = engine_core::worlds::world(*world) else {
            continue;
        };
        let t = w.hierarchy().get_transform_unchecked(entity.id).lock();
        camera.set_from_trs(t.get_global_position(), t.get_global_rotation());
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

/// One world a camera draws, as the renderer sees it. The `Vec` of these a
/// camera is rebuilt against is its layer stack (ADR-0011 §5).
pub struct WorldSource<'a> {
    pub id: WorldId,
    /// SoT TRS + parents — what the cull dispatch indexes.
    pub transforms: &'a WorldTransformGpu,
    /// Per-transform `GPURenderers` buffer (`transform → (mesh, material)`).
    pub renderers: &'a GpuRenderers,
    /// **This world's** draw plan: the same mesh slots as every other
    /// world's, but `first_instance` bases and `total_renderers` sized to
    /// what this world alone draws (ADR-0011 §4).
    pub plan: DrawPlan,
}

impl WorldSource<'_> {
    /// The cull dispatch covers every transform slot, so this is the world's
    /// entity capacity — and the worst-case size of everything downstream.
    fn capacity(&self) -> usize {
        self.transforms.entity_capacity()
    }
}

/// Per-call bundle of GPU/scene state the camera needs to (re)build its draw
/// resources. Global to the frame — the per-world half is [`WorldSource`],
/// passed alongside. Nothing is owned beyond the call.
pub struct CameraSceneResources<'a> {
    pub cb_allocator: &'a Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: &'a Arc<StandardDescriptorSetAllocator>,
    pub memory_allocator: &'a Arc<StandardMemoryAllocator>,
    pub pipeline: &'a Arc<GraphicsPipeline>,
    pub queue_family_index: u32,
    /// Pass 1's cull pipeline — see `shaders/mvp_build.comp`.
    pub mvp_build_pipeline: &'a Arc<ComputePipeline>,
    /// Mega buffers + redirect + mesh table + per-slot authored materials.
    pub mesh_store: &'a GpuMeshStore,
    /// Sampled texture images + texture redirect (graphics set 1).
    pub texture_store: &'a GpuTextureStore,
    /// Material SSBO + material redirect (graphics set 1).
    pub material_store: &'a GpuMaterialStore,
    /// Pass 2's cull pipeline — see `shaders/mvp_build_pass2.comp`.
    pub mvp_build_pass2_pipeline: &'a Arc<ComputePipeline>,
    /// The tiny "build pass 2's dispatch-indirect args" pipeline — see
    /// `shaders/cull_pass2_args.comp`.
    pub cull_pass2_args_pipeline: &'a Arc<ComputePipeline>,
    pub draw_compact_pipeline: &'a Arc<ComputePipeline>,
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
    /// Per-visible-instance world TRS (packed `InstXform`, 2× `vec4`) — the
    /// world-space shading basis the PBR vertex stage needs, which the
    /// projection-folded MVP can't provide.
    inst_xform: Subbuffer<[[f32; 8]]>,
    graphics_set: Arc<DescriptorSet>,
    indirect_template: Subbuffer<[DrawIndexedIndirectCommand]>,
    indirect_args: Subbuffer<[DrawIndexedIndirectCommand]>,
    /// `indirect_args` minus the slots the cull left empty, plus the
    /// `drawCount` the raster reads — both written by `draw_compact_cs`.
    compact_args: Subbuffer<[DrawIndexedIndirectCommand]>,
    compact_count: Subbuffer<u32>,
    compact_set: Arc<DescriptorSet>,
    mvp_capacity: usize,
    slot_capacity: usize,
}

impl DrawResources {
    fn new(scene: &CameraSceneResources<'_>, plan: &DrawPlan) -> Self {
        let slot_count = plan.commands.len();
        let mvp_capacity = (plan.total_renderers as usize).max(1);
        let slot_capacity = slot_count.max(1);

        let (device_matrices, inst_material, inst_xform, graphics_set) = allocate_matrices_and_set(
            scene.memory_allocator,
            scene.descriptor_set_allocator,
            scene.pipeline,
            mvp_capacity,
        );
        let (indirect_template, indirect_args) =
            allocate_indirect_buffers(scene.memory_allocator, slot_capacity);
        write_indirect_template(&indirect_template, &plan.commands);
        let (compact_args, compact_count) =
            allocate_compact_buffers(scene.memory_allocator, slot_capacity);
        let compact_set = build_compact_set(scene, &indirect_args, &compact_args, &compact_count);

        Self {
            device_matrices,
            inst_material,
            inst_xform,
            graphics_set,
            indirect_template,
            indirect_args,
            compact_args,
            compact_count,
            compact_set,
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
            let (dm, im, ix, gs) = allocate_matrices_and_set(
                scene.memory_allocator,
                scene.descriptor_set_allocator,
                scene.pipeline,
                self.mvp_capacity,
            );
            self.device_matrices = dm;
            self.inst_material = im;
            self.inst_xform = ix;
            self.graphics_set = gs;
        }
        if slot_count > self.slot_capacity {
            self.slot_capacity = slot_count.max(self.slot_capacity.saturating_mul(2)).max(1);
            let (t, a) = allocate_indirect_buffers(scene.memory_allocator, self.slot_capacity);
            self.indirect_template = t;
            self.indirect_args = a;
            let (ca, cc) = allocate_compact_buffers(scene.memory_allocator, self.slot_capacity);
            self.compact_args = ca;
            self.compact_count = cc;
            self.compact_set = build_compact_set(
                scene,
                &self.indirect_args,
                &self.compact_args,
                &self.compact_count,
            );
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

/// One world's contribution to a camera's image: everything the cull and
/// the draws need that is keyed by *which* world, not by which camera.
///
/// A camera holds these in composite order and runs them all into one pair
/// of attachments, so the layers share a depth buffer and interleave
/// (ADR-0011 §5).
struct WorldDraw {
    world: WorldId,
    /// Pass 1's compacted output — instances visible against last frame's
    /// (reprojected) Hi-Z draw immediately via `scene_pass1`.
    pass1: DrawResources,
    /// Pass 2's compacted output — instances pass 1's occlusion sub-test
    /// deferred, confirmed against this frame's own Hi-Z. See
    /// [`DrawResources`]'s doc comment for why this can't share pass 1's.
    pass2: DrawResources,
    /// Pass 1 cull set 0 — this world's SoT, GPURenderers, redirect,
    /// mesh_table, MVP, indirect, Parents, slot materials, inst material,
    /// and the candidate list + its live counter.
    cull_set: Arc<DescriptorSet>,
    /// Pass 2 cull set 0 — candidate list + counter (read), pass 2's own
    /// indirect args (rw), MVP + inst_material (write).
    pass2_cull_set0: Arc<DescriptorSet>,
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
    /// Pass 1's `multiDrawIndexedIndirect` over `pass1.indirect_args`.
    scene_pass1: Arc<SecondaryAutoCommandBuffer>,
    /// Pass 2's, recorded against a `Load` (not `Clear`) attachment scope —
    /// see `lib.rs`'s `build_frame_slot`.
    scene_pass2: Arc<SecondaryAutoCommandBuffer>,
    slot_count: usize,
    cull_range: usize,
}

/// The camera-level state a [`WorldDraw`] binds into but does not own.
struct DrawContext<'a> {
    texture_set: &'a Arc<DescriptorSet>,
    extent: [u32; 2],
    /// `drawCount` baked into the scene secondaries.
    slot_count: usize,
}

impl WorldDraw {
    fn new(scene: &CameraSceneResources<'_>, src: &WorldSource<'_>, ctx: &DrawContext<'_>) -> Self {
        let plan = &src.plan;
        let pass1 = DrawResources::new(scene, plan);
        let pass2 = DrawResources::new(scene, plan);
        let (candidate_list, candidate_count) =
            allocate_candidate_buffers(scene.memory_allocator, src.capacity());
        Self {
            world: src.id,
            cull_set: build_cull_set(scene, src, &pass1, &candidate_list, &candidate_count),
            pass2_cull_set0: build_pass2_cull_set0(
                scene,
                &candidate_list,
                &candidate_count,
                &pass2,
            ),
            pass2_dispatch_args: allocate_pass2_dispatch_args(scene.memory_allocator),
            scene_pass1: record_scene_pass(scene, ctx, &pass1),
            scene_pass2: record_scene_pass(scene, ctx, &pass2),
            slot_count: plan.commands.len(),
            cull_range: src.capacity(),
            pass1,
            pass2,
            candidate_list,
            candidate_count,
        }
    }

    /// Re-derive the sets and secondaries for a new plan or a grown world,
    /// reusing the buffers that are still big enough.
    fn rebuild(
        &mut self,
        scene: &CameraSceneResources<'_>,
        src: &WorldSource<'_>,
        ctx: &DrawContext<'_>,
    ) {
        let plan = &src.plan;
        self.pass1.ensure_capacity(scene, plan);
        self.pass2.ensure_capacity(scene, plan);
        if src.capacity() > self.candidate_list.len() as usize {
            let (list, count) = allocate_candidate_buffers(scene.memory_allocator, src.capacity());
            self.candidate_list = list;
            self.candidate_count = count;
        }
        self.slot_count = plan.commands.len();
        self.cull_range = src.capacity();
        self.cull_set = build_cull_set(
            scene,
            src,
            &self.pass1,
            &self.candidate_list,
            &self.candidate_count,
        );
        self.pass2_cull_set0 = build_pass2_cull_set0(
            scene,
            &self.candidate_list,
            &self.candidate_count,
            &self.pass2,
        );
        self.rerecord(scene, ctx);
    }

    /// Re-record this world's two scene passes. Needed whenever a set they
    /// bake in gets a new identity — a plan change, but also a resize, which
    /// hands the camera a fresh viewport. The cull secondaries are the
    /// camera's, not a world's: see [`RenderCamera::record_cull`].
    fn rerecord(&mut self, scene: &CameraSceneResources<'_>, ctx: &DrawContext<'_>) {
        self.scene_pass1 = record_scene_pass(scene, ctx, &self.pass1);
        self.scene_pass2 = record_scene_pass(scene, ctx, &self.pass2);
    }

    /// Whether this world's plan or capacity outgrew what it was built for.
    fn needs_rebuild(&self, src: &WorldSource<'_>) -> bool {
        let plan = &src.plan;
        plan.total_renderers as usize > self.pass1.mvp_capacity
            || plan.commands.len() > self.pass1.slot_capacity
            || plan.commands.len() != self.slot_count
            || src.capacity() != self.cull_range
    }
}

/// `drawCount` for the scene secondaries. Every world's plan spans the same
/// mesh slots, so this is a camera-level number even though plans are not.
fn slot_count(worlds: &[WorldSource<'_>]) -> usize {
    worlds.first().map_or(1, |src| src.plan.commands.len())
}

/// One pass's `multiDrawIndexedIndirect`, at the camera's extent and
/// against its texture set.
fn record_scene_pass(
    scene: &CameraSceneResources<'_>,
    ctx: &DrawContext<'_>,
    pass: &DrawResources,
) -> Arc<SecondaryAutoCommandBuffer> {
    record_scene_secondary(
        scene.cb_allocator,
        scene.queue_family_index,
        scene.pipeline,
        &pass.graphics_set,
        ctx.texture_set,
        scene.mesh_store,
        &pass.compact_args,
        &pass.compact_count,
        ctx.slot_count,
        ctx.extent,
    )
}

pub struct RenderCamera {
    /// The host half — whoever drives this camera writes its `view_proj`
    /// there, and the panel showing it writes its box.
    state: CameraHandle,
    resolution: CameraResolution,
    extent: [u32; 2],
    color_image: Arc<Image>,
    depth_image: Arc<Image>,
    color_view: Arc<ImageView>,
    depth_view: Arc<ImageView>,

    /// Graphics set 1 — texture redirect + material redirect + material
    /// SSBO + the sampled-image array. Shared by every world's draws.
    texture_set: Arc<DescriptorSet>,

    /// The worlds this camera composites, in draw order.
    draws: Vec<WorldDraw>,

    /// Pass 1 cull, every world in one stage-major secondary; likewise
    /// pass 2. Camera-level rather than per world because a world's own
    /// cull stages are dependent and each barrier between them drains the
    /// GPU — see [`Self::record_cull`].
    cull_secondary: Arc<SecondaryAutoCommandBuffer>,
    cull_pass2_secondary: Arc<SecondaryAutoCommandBuffer>,

    /// Pass 1 cull set 1 — this frame's `view_proj`, last frame's, and last
    /// frame's Hi-Z (sampled). Camera-owned, so every world's pass 1 binds
    /// the same one.
    occlusion_set: Arc<DescriptorSet>,
    /// Pass 2 cull set 1 — the cull-test `view_proj` + this frame's own
    /// Hi-Z (sampled).
    pass2_cull_set1: Arc<DescriptorSet>,

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
    /// mips) and `view_proj → prev_view_proj`, so next
    /// frame's pass 1 sees this frame's data as "last frame's" without
    /// either descriptor set ever rebinding (fixed image/buffer
    /// identities — see the module doc comment). Extent-dependent only,
    /// same rebuild scope as `hiz_build_secondary`.
    history_update_secondary: Arc<SecondaryAutoCommandBuffer>,

    /// This frame's Hi-Z pyramid, built by `hiz_build_secondary` from this
    /// frame's own pass-1 depth output — every world's, since the build
    /// runs after the last of them has drawn. Read by pass 2's occlusion
    /// test (exact — same frame, no reprojection) and copied into
    /// `hiz_prev` at frame end. **Note:** only reflects pass 1's depth
    /// contribution — pass 2's draws land in the real depth attachment but
    /// are not re-folded into `hiz_current`, so an object confirmed only
    /// via pass 2 this frame won't help occlude anything next frame until
    /// pass 1 itself draws it (typically the very next frame, once it's no
    /// longer a "just revealed" edge case). Rebuilding Hi-Z a second time
    /// after pass 2 would close this gap at roughly double the per-frame
    /// Hi-Z build cost; deferred as a planned follow-up if profiling shows
    /// the steady-state candidate count doesn't stay small.
    hiz_current: HizPyramid,
    /// Last frame's Hi-Z pyramid (via the end-of-frame `hiz_current →
    /// hiz_prev` copy). Read by pass 1's occlusion sub-test, reprojected
    /// with `prev_view_proj`.
    hiz_prev: HizPyramid,
    /// The camera block the shaders read: `[0]` this frame's `view_proj`,
    /// `[1][0..3]` the eye position `scene.frag`'s specular term needs.
    /// Device-local with a fixed identity, promoted from
    /// [`Self::view_proj_staging`] by a `copy_buffer` in every FrameSlot
    /// primary — the same staging→SoT pattern as TRS.
    view_proj: Subbuffer<[[f32; 16]]>,
    /// Host-mapped counterpart, double-buffered in lockstep with the TRS
    /// staging slots (see [`Self::cull_view_proj_staging`]).
    view_proj_staging: [Subbuffer<[[f32; 16]]>; STAGING_SLOTS],
    /// Camera-owned `view_proj` history — last frame's value. Copied from
    /// [`Self::view_proj`] at the end of every frame
    /// (`history_update_secondary`), *before* next frame's promotion copy
    /// overwrites it.
    prev_view_proj: Subbuffer<[[f32; 16]]>,
    /// Depth-only NEAREST/ClampToEdge sampler shared by every Hi-Z-related
    /// combined-image-sampler binding. `texelFetch` (used throughout the
    /// occlusion tests and the reduce shaders) ignores the sampler's
    /// filter/address mode entirely — a sampler object is still required
    /// to form a combined image sampler, so this exists purely to satisfy
    /// that requirement. Built once; extent/capacity-independent.
    hiz_sampler: Arc<Sampler>,

    /// Camera-owned cull-VP lock pair — device-local, read by
    /// `mvp_build.comp`'s frustum test only (never the occlusion sub-tests
    /// or the output MVP write, which always stay on the live render VP).
    /// Fixed identity, like `prev_view_proj` — never reallocated, only its
    /// contents change. Host-mapped counterpart is `cull_view_proj_staging`.
    cull_view_proj: Subbuffer<[[f32; 16]]>,
    /// Host-mapped staging the per-frame write (`write_cull_view_proj`)
    /// lands in — promoted into `cull_view_proj` by an unconditional
    /// `copy_buffer` baked into every `FrameSlot` primary (see
    /// `lib.rs::build_frame_slot`).
    /// **Double-buffered**, in lockstep with `WorldTransformGpu`'s
    /// staging slots — the host writes one while the previous frame's
    /// `copy_buffer` still reads the other. A single-buffered host-write
    /// here would re-impose the frame `N-1` gate on the whole engine.
    cull_view_proj_staging: [Subbuffer<[[f32; 16]]>; STAGING_SLOTS],
    /// Slot the host writes both VP staging buffers into this frame;
    /// advanced in lockstep with `WorldTransformGpu::advance_staging_slot`.
    vp_write_slot: usize,
    /// Debug: when true, `write_cull_view_proj` writes `locked_view_proj`
    /// instead of the live render VP every frame — freezes the frustum
    /// test's cull volume while the render camera keeps moving.
    cull_lock: bool,
    /// Snapshot of the live VP taken the moment `cull_lock` last
    /// transitioned off → on (host-only cache; not itself read by the
    /// GPU — `cull_view_proj` is).
    locked_view_proj: [f32; 16],
    /// Debug: when true, `lib.rs::build_frame_slot` skips
    /// `hiz_build_secondary` and `history_update_secondary`, freezing
    /// `hiz_current`/`hiz_prev`/`prev_view_proj` at whatever they held the
    /// moment this became true — a self-consistent snapshot, since
    /// `cull_view_proj` was already pinned to `locked_view_proj` by then.
    /// The pass 2 secondaries and render scope keep running while frozen —
    /// only the data they test against stops updating. Always kept one
    /// frame behind `cull_lock` by `apply_pending_hiz_freeze`, which lets
    /// the *engage* frame's own Hi-Z build run once more first, so the
    /// frozen snapshot is consistent with `locked_view_proj`. See the
    /// module doc comment's "frustum-lock" section.
    hiz_frozen: bool,
    /// Debug: when false, every world's `cull_secondary` push constant
    /// forces frustum-visible instances to draw immediately in pass 1
    /// (skipping the occlusion sub-test in `mvp_build.comp` — required for
    /// correctness, since pass 2 is the only consumer of the candidate list
    /// it would otherwise populate), and `lib.rs::build_frame_slot` skips
    /// the Hi-Z build, pass 2's cull dispatches, pass 2's render scope, and
    /// the history-update secondary entirely. Default `true`.
    occlusion_enabled: bool,

    /// `drawCount` baked into every world's scene secondaries — the plan's,
    /// which is shared until draw plans go per world (ADR-0011 §4).
    slot_count: usize,

    /// Grid + gizmo, drawn in one scope after both scene passes.
    overlay: CameraOverlay,
}

impl RenderCamera {
    pub fn new_match_swapchain(
        state: CameraHandle,
        swapchain_extent: [u32; 2],
        scene: &CameraSceneResources<'_>,
        worlds: &[WorldSource<'_>],
    ) -> Self {
        Self::new(
            state,
            CameraResolution::MatchSwapchain,
            swapchain_extent,
            scene,
            worlds,
        )
    }

    pub fn new(
        state: CameraHandle,
        resolution: CameraResolution,
        swapchain_extent: [u32; 2],
        scene: &CameraSceneResources<'_>,
        worlds: &[WorldSource<'_>],
    ) -> Self {
        let extent = resolution.resolve(swapchain_extent);
        // Every world's plan spans the same mesh slots — only the bases and
        // totals differ — so `drawCount` is the camera's, not a world's.
        let slot_count = slot_count(worlds);
        let (color_image, color_view, depth_image, depth_view) =
            allocate_attachments(scene.memory_allocator, extent);

        let (view_proj, _) = allocate_view_proj(scene.memory_allocator);
        let view_proj_staging: [_; STAGING_SLOTS] =
            std::array::from_fn(|_| allocate_view_proj(scene.memory_allocator).1);
        let prev_view_proj = allocate_prev_view_proj(scene.memory_allocator);
        let (cull_view_proj, _) = allocate_cull_view_proj(scene.memory_allocator);
        let cull_view_proj_staging: [_; STAGING_SLOTS] =
            std::array::from_fn(|_| allocate_cull_view_proj(scene.memory_allocator).1);
        let hiz_sampler =
            build_hiz_sampler(scene.queue_family_index, scene.pipeline.device().clone());

        let hiz_mip0_extent = hiz_mip0_extent(extent);
        let hiz_current = allocate_hiz_pyramid(scene.memory_allocator, hiz_mip0_extent);
        let hiz_prev = allocate_hiz_pyramid(scene.memory_allocator, hiz_mip0_extent);

        let occlusion_set = build_occlusion_set(
            scene,
            &view_proj,
            &prev_view_proj,
            &cull_view_proj,
            &hiz_prev,
            &hiz_sampler,
        );
        let pass2_cull_set1 =
            build_pass2_cull_set1(scene, &cull_view_proj, &hiz_current, &hiz_sampler);
        let (hiz_level0_set, hiz_mip2_sets, hiz_trailing_set) =
            build_hiz_sets(scene, &depth_view, &hiz_current, &hiz_sampler);

        let hiz_build_secondary = record_hiz_build_secondary(
            scene,
            &hiz_level0_set,
            &hiz_mip2_sets,
            &hiz_trailing_set,
            hiz_mip0_extent,
        );
        let history_update_secondary = record_history_update_secondary(
            scene,
            &hiz_current,
            &hiz_prev,
            &view_proj,
            &prev_view_proj,
        );

        let texture_set = build_texture_set(scene, &view_proj);
        let occlusion_enabled = true;
        let ctx = DrawContext {
            texture_set: &texture_set,
            extent,
            slot_count,
        };
        let draws: Vec<WorldDraw> = worlds
            .iter()
            .map(|src| WorldDraw::new(scene, src, &ctx))
            .collect();
        let cull_secondary =
            record_cull_secondary(scene, &draws, &occlusion_set, occlusion_enabled);
        let cull_pass2_secondary = record_cull_pass2_secondary(scene, &draws, &pass2_cull_set1);

        RenderCamera {
            state,
            resolution,
            extent,
            color_image,
            depth_image,
            color_view,
            depth_view,
            texture_set,
            draws,
            cull_secondary,
            cull_pass2_secondary,
            occlusion_set,
            pass2_cull_set1,
            hiz_level0_set,
            hiz_mip2_sets,
            hiz_trailing_set,
            hiz_build_secondary,
            history_update_secondary,
            hiz_current,
            hiz_prev,
            view_proj,
            view_proj_staging,
            prev_view_proj,
            hiz_sampler,
            cull_view_proj,
            cull_view_proj_staging,
            vp_write_slot: 0,
            cull_lock: false,
            locked_view_proj: [0.0; 16],
            hiz_frozen: false,
            occlusion_enabled,
            slot_count,
            overlay: CameraOverlay::new(scene, extent),
        }
    }

    /// Swapchain resized. Re-creates every extent-dependent resource: the
    /// color/depth attachments, both Hi-Z pyramids, the descriptor sets
    /// that bind any of their views, and every secondary that references
    /// those sets or whose recording is extent-shaped. Capacity-dependent
    /// resources are untouched. Returns `true` if anything was rebuilt.
    pub fn on_swapchain_resize(
        &mut self,
        new_swapchain_extent: [u32; 2],
        scene: &CameraSceneResources<'_>,
    ) -> bool {
        if !self.resolution.depends_on_swapchain() {
            return false;
        }
        let new_extent = self.resolution.resolve(new_swapchain_extent);
        self.resize(new_extent, scene)
    }

    /// Adopt a resolution policy, re-allocating if it resolves to a
    /// different extent. This is how a [`Viewport`](crate::ui::Viewport)
    /// hands the camera its panel's box: same rebuild as a window resize,
    /// asked for by the layout instead of by the compositor.
    ///
    /// Switching to or from [`CameraResolution::MatchSwapchain`] also
    /// changes who composites the camera, so the caller must rebuild the
    /// frame slots even when the extent happens not to move.
    pub fn set_resolution(
        &mut self,
        resolution: CameraResolution,
        swapchain_extent: [u32; 2],
        scene: &CameraSceneResources<'_>,
    ) -> bool {
        let was = std::mem::replace(&mut self.resolution, resolution);
        let extent = resolution.resolve(swapchain_extent);
        self.resize(extent, scene) || was.blits_to_swapchain() != resolution.blits_to_swapchain()
    }

    /// Re-create every extent-dependent resource at `new_extent`. Returns
    /// `false` if it is already that size, which is the steady state.
    fn resize(&mut self, new_extent: [u32; 2], scene: &CameraSceneResources<'_>) -> bool {
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
        // Any frozen Hi-Z snapshot is invalidated by the reallocation above
        // (new images, old resolution's content is gone) — fall back to
        // live tracking rather than leaving `hiz_frozen` pointed at
        // undefined data. `apply_pending_hiz_freeze` will re-freeze on a
        // later frame if `cull_lock` is still engaged.
        self.hiz_frozen = false;

        self.occlusion_set = build_occlusion_set(
            scene,
            &self.view_proj,
            &self.prev_view_proj,
            &self.cull_view_proj,
            &self.hiz_prev,
            &self.hiz_sampler,
        );
        self.pass2_cull_set1 = build_pass2_cull_set1(
            scene,
            &self.cull_view_proj,
            &self.hiz_current,
            &self.hiz_sampler,
        );
        let (hiz_level0_set, hiz_mip2_sets, hiz_trailing_set) = build_hiz_sets(
            scene,
            &self.depth_view,
            &self.hiz_current,
            &self.hiz_sampler,
        );
        self.hiz_level0_set = hiz_level0_set;
        self.hiz_mip2_sets = hiz_mip2_sets;
        self.hiz_trailing_set = hiz_trailing_set;

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
            &self.view_proj,
            &self.prev_view_proj,
        );

        let mut draws = std::mem::take(&mut self.draws);
        let ctx = self.draw_context();
        for draw in &mut draws {
            draw.rerecord(scene, &ctx);
        }
        self.draws = draws;
        self.record_cull(scene);
        self.overlay.on_resize(scene, new_extent);
        true
    }

    /// Rebuild the draw resources for the current plan and world list.
    /// Grows each world's MVP / indirect / candidate buffers, rewrites its
    /// indirect templates, and re-records its cull + scene secondaries.
    /// A world that appears in `worlds` for the first time gets a fresh
    /// [`WorldDraw`]; one that disappears is dropped. Extent-only resources
    /// (Hi-Z pyramids, `occlusion_set`, `pass2_cull_set1`) are untouched —
    /// see [`Self::on_swapchain_resize`]. Always returns `true` (the
    /// FrameSlot primaries reference the secondaries, so callers must
    /// rebuild them).
    ///
    /// Called only on topology change / capacity growth — never per frame in
    /// steady state.
    pub fn ensure_current(
        &mut self,
        scene: &CameraSceneResources<'_>,
        worlds: &[WorldSource<'_>],
    ) -> bool {
        // Texture arrivals / redirect-buffer growth reach here via
        // `force_full`; rebind the current views + buffers.
        self.texture_set = build_texture_set(scene, &self.view_proj);
        self.slot_count = slot_count(worlds);
        let mut old = std::mem::take(&mut self.draws);
        let ctx = self.draw_context();
        self.draws = worlds
            .iter()
            .map(|src| match old.iter().position(|d| d.world == src.id) {
                Some(k) => {
                    let mut draw = old.swap_remove(k);
                    draw.rebuild(scene, src, &ctx);
                    draw
                }
                None => WorldDraw::new(scene, src, &ctx),
            })
            .collect();
        self.record_cull(scene);
        true
    }

    /// Both cull passes for every world this camera draws, **stage major**:
    /// all worlds' resets, then all their cull dispatches, then all their
    /// pass-2 args-builders.
    ///
    /// One secondary per world would be the obvious shape. A world's reset →
    /// dispatch → args-builder chain is genuinely dependent, and on AMD each
    /// barrier between those stages drains the whole GPU, so world-major
    /// recording pays N× the drains for the same work. Same reasoning and
    /// the same measurement as the TRS scatter —
    /// `docs/notes/scatter-overlap-bench.md`.
    fn record_cull(&mut self, scene: &CameraSceneResources<'_>) {
        self.cull_secondary = record_cull_secondary(
            scene,
            &self.draws,
            &self.occlusion_set,
            self.occlusion_enabled,
        );
        self.cull_pass2_secondary =
            record_cull_pass2_secondary(scene, &self.draws, &self.pass2_cull_set1);
    }

    /// Whether the current plan / world list needs a **full** rebuild (new
    /// buffers + descriptor sets + secondaries + frame slots) vs. just an
    /// in-place rewrite of the indirect templates' per-slot bases.
    ///
    /// `force` is set by the caller when a cull-bound external buffer (SoT,
    /// `GPURenderers`, redirect, mesh table) reallocated.
    pub fn needs_structural_rebuild(&self, worlds: &[WorldSource<'_>], force: bool) -> bool {
        force
            || worlds.len() != self.draws.len()
            || std::iter::zip(worlds, &self.draws)
                .any(|(src, draw)| draw.world != src.id || draw.needs_rebuild(src))
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
    pub fn write_template_bases(&self, worlds: &[WorldSource<'_>]) {
        for draw in &self.draws {
            let Some(src) = worlds.iter().find(|s| s.id == draw.world) else {
                continue;
            };
            write_indirect_template(&draw.pass1.indirect_template, &src.plan.commands);
            write_indirect_template(&draw.pass2.indirect_template, &src.plan.commands);
        }
    }

    /// The camera-level state every [`WorldDraw`] binds into.
    fn draw_context(&self) -> DrawContext<'_> {
        DrawContext {
            texture_set: &self.texture_set,
            extent: self.extent,
            slot_count: self.slot_count,
        }
    }

    // ── Debug: frustum-lock (cheap, no rebuild) ────────────────────────

    /// Write this frame's cull-test `view_proj`: the live render VP, or —
    /// if the lock is engaged — the frozen snapshot taken when it was
    /// last enabled. No command-buffer re-recording; the value just flows
    /// through `cull_view_proj_staging` → `cull_view_proj` via the
    /// unconditional `copy_buffer` baked into every `FrameSlot` primary.
    ///
    /// **Same host-write gating as [`Self::write_template_bases`]** — call
    /// only after `WorldTransformGpu::host_wait_for_previous_compute`.
    pub fn write_cull_view_proj(&self, live_view_proj: [f32; 16]) {
        let vp = if self.cull_lock {
            self.locked_view_proj
        } else {
            live_view_proj
        };
        let mut w = self.cull_view_proj_staging[self.vp_write_slot]
            .write()
            .expect("cull_view_proj_staging.write");
        w[0] = vp;
    }

    /// Toggle the frustum-lock debug feature. Snapshots `live_view_proj`
    /// into `locked_view_proj` only on the off→on transition, so the lock
    /// always freezes at whatever the camera saw the moment it engaged.
    pub fn set_cull_lock(&mut self, locked: bool, live_view_proj: [f32; 16]) {
        if locked && !self.cull_lock {
            self.locked_view_proj = live_view_proj;
        }
        self.cull_lock = locked;
    }

    pub fn cull_lock(&self) -> bool {
        self.cull_lock
    }

    /// Bring `hiz_frozen` into line with `cull_lock`, if it isn't already —
    /// called once per frame, **before** that frame's own `set_cull_lock`
    /// call (if any). Because of that ordering, a `set_cull_lock` call on
    /// frame N is only ever observed here on frame N+1 or later, which is
    /// what gives the engage transition its required one-frame delay (see
    /// the `hiz_frozen` field doc comment) — this same call handles the
    /// disengage transition too, just without needing that delay for
    /// correctness (unfreezing a frame late is harmless and self-heals,
    /// same as `set_occlusion_enabled`'s re-enable transient).
    ///
    /// Returns whether `hiz_frozen` actually changed — callers must trigger
    /// a `FrameSlot` rebuild (via `build_all_frame_slots`) iff this returns
    /// `true`, since `lib.rs::build_frame_slot` reads
    /// [`Self::hiz_frozen`] to decide whether to include
    /// `hiz_build_secondary`/`history_update_secondary` in the primary.
    pub fn apply_pending_hiz_freeze(&mut self) -> bool {
        if self.hiz_frozen == self.cull_lock {
            return false;
        }
        self.hiz_frozen = self.cull_lock;
        true
    }

    /// Whether the Hi-Z pyramids / `prev_view_proj` are currently frozen
    /// (debug frustum-lock feature, one frame behind `cull_lock` — see
    /// [`Self::apply_pending_hiz_freeze`]).
    pub fn hiz_frozen(&self) -> bool {
        self.hiz_frozen
    }

    // ── Debug: occlusion enable/disable (rebuild-gated) ────────────────

    /// Toggle occlusion culling. Re-records only `cull_secondary` (cheap
    /// relative to [`Self::ensure_current`]) with the new push-constant
    /// flag. Returns whether anything actually changed — callers must
    /// trigger a `FrameSlot` rebuild (via `build_all_frame_slots`) iff this
    /// returns `true`, since `lib.rs::build_frame_slot` reads
    /// [`Self::occlusion_enabled`] to decide whether to include the Hi-Z
    /// build / pass 2 cull / pass 2 render / history-update secondaries in
    /// the primary at all.
    pub fn set_occlusion_enabled(
        &mut self,
        enabled: bool,
        scene: &CameraSceneResources<'_>,
    ) -> bool {
        if enabled == self.occlusion_enabled {
            return false;
        }
        self.occlusion_enabled = enabled;
        self.record_cull(scene);
        true
    }

    pub fn occlusion_enabled(&self) -> bool {
        self.occlusion_enabled
    }

    // ── Accessors ───────────────────────────────────────────────────────

    /// The host half: `view_proj`, the panel's box, the projection.
    pub fn state(&self) -> &CameraHandle {
        &self.state
    }
    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }
    /// Aspect of the target the scene is drawn into — the projection's, now
    /// that the hardware viewport covers the whole of it.
    pub fn aspect(&self) -> f32 {
        self.extent[0] as f32 / self.extent[1].max(1) as f32
    }
    pub fn resolution(&self) -> CameraResolution {
        self.resolution
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
    pub fn scene_secondaries_pass1(
        &self,
    ) -> impl Iterator<Item = &Arc<SecondaryAutoCommandBuffer>> {
        self.draws.iter().map(|d| &d.scene_pass1)
    }
    /// Pass 2's `multiDrawIndexedIndirect` — draws instances confirmed
    /// visible against this frame's own Hi-Z. Record against a `Load`
    /// (not `Clear`) attachment scope.
    pub fn scene_secondaries_pass2(
        &self,
    ) -> impl Iterator<Item = &Arc<SecondaryAutoCommandBuffer>> {
        self.draws.iter().map(|d| &d.scene_pass2)
    }
    /// Pass 1 cull (mvp-build) compute secondary — every world, executed
    /// once per frame from each FrameSlot primary, before the first render.
    /// The overlay's draw for `slot`'s host buffers — see [`Self::write_overlay`].
    pub fn overlay_secondary(&self, slot: usize) -> &Arc<SecondaryAutoCommandBuffer> {
        self.overlay.secondary(slot)
    }

    /// This frame's grid parameters and gizmo triangles, into the staging
    /// slot the host is writing. Same gate as [`Self::write_view_proj`].
    pub fn write_overlay(&self, view_proj: Mat4, eye: Vec3) {
        self.overlay.write(
            self.vp_write_slot,
            self.state.slot(),
            view_proj,
            eye,
            self.state.grid(),
        );
    }

    pub fn cull_secondary(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.cull_secondary
    }
    /// Hi-Z pyramid build secondary — executed after pass 1's render, before
    /// pass 2's cull (reads the depth attachment pass 1 just wrote).
    pub fn hiz_build_secondary(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.hiz_build_secondary
    }
    /// Pass 2 cull (mvp-build) compute secondary — every world, one
    /// `dispatch_indirect` each, after `hiz_build_secondary`.
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
    /// Device-local camera block, read by the cull passes and by
    /// `scene.frag`. Promoted each frame from
    /// [`Self::view_proj_staging_buf`] by a `copy_buffer` in every
    /// FrameSlot primary.
    pub fn view_proj_buf(&self) -> &Subbuffer<[[f32; 16]]> {
        &self.view_proj
    }
    /// Host-mapped counterpart of [`Self::view_proj_buf`], written every
    /// frame by [`Self::write_view_proj`].
    pub fn view_proj_staging_buf(&self, slot: usize) -> &Subbuffer<[[f32; 16]]> {
        &self.view_proj_staging[slot]
    }

    /// Stage this frame's camera block.
    ///
    /// **Same host-write gating as [`Self::write_cull_view_proj`]** — call
    /// only after `WorldTransformGpu::host_wait_for_previous_compute`.
    pub fn write_view_proj(&self, view_proj: Mat4, eye: Vec3) {
        let mut w = self.view_proj_staging[self.vp_write_slot]
            .write()
            .expect("view_proj_staging.write");
        w[0] = view_proj.to_cols_array();
        w[1][0..3].copy_from_slice(eye.as_ref());
    }

    /// Device-local cull-test `view_proj` — `mvp_build.comp`'s frustum test
    /// reads this. Promoted each frame from [`Self::cull_view_proj_staging_buf`]
    /// by an unconditional `copy_buffer` in `lib.rs::build_frame_slot`.
    pub fn cull_view_proj_buf(&self) -> &Subbuffer<[[f32; 16]]> {
        &self.cull_view_proj
    }
    /// Host-mapped staging counterpart of [`Self::cull_view_proj_buf`],
    /// written every frame by [`Self::write_cull_view_proj`].
    pub fn cull_view_proj_staging_buf(&self, slot: usize) -> &Subbuffer<[[f32; 16]]> {
        &self.cull_view_proj_staging[slot]
    }

    /// Flip to the other cull-VP staging slot. Called in lockstep with
    /// `WorldTransformGpu::advance_staging_slot`.
    pub fn advance_staging_slot(&mut self) {
        self.vp_write_slot = (self.vp_write_slot + 1) % STAGING_SLOTS;
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
            // SAMPLED: the UI reads this as a texture, which is how a camera
            // appears inside a panel (`ui::CAMERA_TARGET`).
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
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
/// MVP buffer, the parallel `u32` concrete-material-id buffer and the
/// parallel packed world-TRS buffer, all of `capacity` slots — plus the
/// graphics descriptor set that points at them.
fn allocate_matrices_and_set(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    pipeline: &Arc<GraphicsPipeline>,
    capacity: usize,
) -> (
    Subbuffer<[[f32; 16]]>,
    Subbuffer<[u32]>,
    Subbuffer<[[f32; 8]]>,
    Arc<DescriptorSet>,
) {
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
    let inst_xform: Subbuffer<[[f32; 8]]> = Buffer::new_slice::<[f32; 8]>(
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
    .expect("Failed to allocate instance transform buffer");

    let set_layout = pipeline.layout().set_layouts()[0].clone();
    let graphics_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        set_layout,
        [
            WriteDescriptorSet::buffer(0, device_matrices.clone()),
            WriteDescriptorSet::buffer(1, inst_material.clone()),
            WriteDescriptorSet::buffer(2, inst_xform.clone()),
        ],
        [],
    )
    .expect("Failed to allocate matrices descriptor set");

    (device_matrices, inst_material, inst_xform, graphics_set)
}

/// Allocate the indirect-command buffers: a host-visible **template** (the
/// CPU writes the per-slot commands with `instance_count` zeroed) and the
/// device-local **args** (reset from the template each frame, written by the
/// cull's atomics, read by the indirect draw).
/// The compacted command list and its GPU-written `drawCount`. Both need
/// `INDIRECT_BUFFER` — the count is what `vkCmdDrawIndexedIndirectCount`
/// reads — and `TRANSFER_DST` so the count can be zeroed each frame.
fn allocate_compact_buffers(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    capacity: usize,
) -> (Subbuffer<[DrawIndexedIndirectCommand]>, Subbuffer<u32>) {
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
        capacity.max(1) as u64,
    )
    .expect("Failed to allocate compacted indirect buffer");
    let count = Buffer::new_sized::<u32>(
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
    )
    .expect("Failed to allocate compacted draw count");
    (args, count)
}

/// `draw_compact_cs` set 0: the cull's commands in, the compacted list and
/// its count out.
fn build_compact_set(
    scene: &CameraSceneResources<'_>,
    indirect_args: &Subbuffer<[DrawIndexedIndirectCommand]>,
    compact_args: &Subbuffer<[DrawIndexedIndirectCommand]>,
    compact_count: &Subbuffer<u32>,
) -> Arc<DescriptorSet> {
    DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        scene.draw_compact_pipeline.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, indirect_args.clone().reinterpret::<[u32]>()),
            WriteDescriptorSet::buffer(1, compact_args.clone().reinterpret::<[u32]>()),
            WriteDescriptorSet::buffer(2, compact_count.clone()),
        ],
        [],
    )
    .expect("Failed to allocate draw-compaction set")
}

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
    // Zero the tail rather than leave it undefined. Nothing should read past
    // `slot_count`, but a stray reader that does now issues a draw of nothing
    // instead of a garbage `index_count` that hangs the GPU.
    guard[commands.len()..].fill(DrawIndexedIndirectCommand {
        index_count: 0,
        instance_count: 0,
        first_index: 0,
        vertex_offset: 0,
        first_instance: 0,
    });
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

/// Allocate the camera block pair: the device-local buffer every shader
/// reads, and its host-mapped staging counterpart. `TRANSFER_SRC` on the
/// device side is the end-of-frame copy into `prev_view_proj`.
fn allocate_view_proj(
    memory_allocator: &Arc<StandardMemoryAllocator>,
) -> (Subbuffer<[[f32; 16]]>, Subbuffer<[[f32; 16]]>) {
    let device = Buffer::new_slice::<[f32; 16]>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_DST
                | BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        crate::transform_gpu::CAMERA_BLOCK_MAT4S,
    )
    .expect("Failed to allocate camera view_proj buffer");
    let staging = Buffer::new_slice::<[f32; 16]>(
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
        crate::transform_gpu::CAMERA_BLOCK_MAT4S,
    )
    .expect("Failed to allocate camera view_proj staging buffer");
    (device, staging)
}

/// Allocate the camera's `prev_view_proj` history buffer (fixed identity,
/// overwritten in place each frame by `history_update_secondary`'s
/// `copy_buffer` from `view_proj` — hence the same
/// [`CAMERA_BLOCK_MAT4S`] length; only the leading `view_proj` is read).
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
        crate::transform_gpu::CAMERA_BLOCK_MAT4S,
    )
    .expect("Failed to allocate prev_view_proj buffer")
}

/// Allocate the camera's cull-VP-lock pair: a host-mapped staging slot the
/// host writes every frame (either the live render VP or a frozen
/// snapshot, depending on `RenderCamera::cull_lock`), and its device-local
/// counterpart `mvp_build.comp`'s frustum test reads. Same staging→SoT
/// pattern as [`allocate_view_proj`], one mat4 instead of the camera
/// block — the frustum test reads no eye position.
fn allocate_cull_view_proj(
    memory_allocator: &Arc<StandardMemoryAllocator>,
) -> (Subbuffer<[[f32; 16]]>, Subbuffer<[[f32; 16]]>) {
    let device = Buffer::new_slice::<[f32; 16]>(
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
    .expect("Failed to allocate cull_view_proj buffer");
    let staging = Buffer::new_slice::<[f32; 16]>(
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
        1,
    )
    .expect("Failed to allocate cull_view_proj staging buffer");
    (device, staging)
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
/// SSBO, the fixed-size sampled-image array (placeholder-padded — see
/// [`GpuTextureStore`]), and the shared camera buffer whose world position
/// the PBR specular term needs.
fn build_texture_set(
    scene: &CameraSceneResources<'_>,
    view_proj: &Subbuffer<[[f32; 16]]>,
) -> Arc<DescriptorSet> {
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
            WriteDescriptorSet::buffer(4, view_proj.clone()),
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
    src: &WorldSource<'_>,
    pass1: &DrawResources,
    candidate_list: &Subbuffer<[[f32; 16]]>,
    candidate_count: &Subbuffer<[u32]>,
) -> Arc<DescriptorSet> {
    let world = src.transforms;
    DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        scene.mvp_build_pipeline.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, world.sot_positions().clone()),
            WriteDescriptorSet::buffer(1, world.sot_rotations().clone()),
            WriteDescriptorSet::buffer(2, world.sot_scales().clone()),
            WriteDescriptorSet::buffer(3, src.renderers.buffer().clone()),
            WriteDescriptorSet::buffer(4, scene.mesh_store.redirect_buffer().clone()),
            WriteDescriptorSet::buffer(5, scene.mesh_store.mesh_table_buffer().clone()),
            WriteDescriptorSet::buffer(6, pass1.device_matrices.clone()),
            WriteDescriptorSet::buffer(7, pass1.indirect_args.clone().reinterpret::<[u32]>()),
            WriteDescriptorSet::buffer(8, world.sot_parents().clone()),
            WriteDescriptorSet::buffer(9, scene.mesh_store.slot_material_buffer().clone()),
            WriteDescriptorSet::buffer(10, pass1.inst_material.clone()),
            WriteDescriptorSet::buffer(11, candidate_list.clone()),
            WriteDescriptorSet::buffer(12, candidate_count.clone()),
            WriteDescriptorSet::buffer(13, pass1.inst_xform.clone()),
        ],
        [],
    )
    .expect("Failed to allocate cull set")
}

/// Build pass 1's occlusion set (set 1): this frame's `view_proj`, last
/// frame's, last frame's Hi-Z pyramid (sampled), and the cull-test
/// `view_proj` — normally a mirror of the first, but freezable by the
/// debug frustum-lock (see `RenderCamera::set_cull_lock`). All camera-owned.
fn build_occlusion_set(
    scene: &CameraSceneResources<'_>,
    view_proj: &Subbuffer<[[f32; 16]]>,
    prev_view_proj: &Subbuffer<[[f32; 16]]>,
    cull_view_proj: &Subbuffer<[[f32; 16]]>,
    hiz_prev: &HizPyramid,
    hiz_sampler: &Arc<Sampler>,
) -> Arc<DescriptorSet> {
    let layout = scene.mvp_build_pipeline.layout().set_layouts()[1].clone();
    DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        layout,
        [
            WriteDescriptorSet::buffer(0, view_proj.clone()),
            WriteDescriptorSet::buffer(1, prev_view_proj.clone()),
            WriteDescriptorSet::image_view_sampler(
                2,
                hiz_prev.sampled_view.clone(),
                hiz_sampler.clone(),
            ),
            WriteDescriptorSet::buffer(3, cull_view_proj.clone()),
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
            WriteDescriptorSet::buffer(5, pass2.inst_xform.clone()),
        ],
        [],
    )
    .expect("Failed to allocate pass2 cull set0")
}

/// Build pass 2's cull set 1: the cull-test `view_proj` (camera-owned;
/// mirrors the live render VP unless the debug frustum-lock is engaged —
/// see `RenderCamera::set_cull_lock`) + this frame's own Hi-Z pyramid
/// (sampled). Binding `cull_view_proj` here rather than the live
/// `view_proj` is what lets pass
/// 2's exact re-test stay a self-consistent (VP, Hi-Z) pair with pass 1's
/// occlusion sub-test even while the Hi-Z pyramid is frozen
/// (`RenderCamera::hiz_frozen`) — see the module doc comment's
/// "frustum-lock" section.
fn build_pass2_cull_set1(
    scene: &CameraSceneResources<'_>,
    cull_view_proj: &Subbuffer<[[f32; 16]]>,
    hiz_current: &HizPyramid,
    hiz_sampler: &Arc<Sampler>,
) -> Arc<DescriptorSet> {
    let layout = scene.mvp_build_pass2_pipeline.layout().set_layouts()[1].clone();
    DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        layout,
        [
            WriteDescriptorSet::buffer(0, cull_view_proj.clone()),
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
) -> (
    Arc<DescriptorSet>,
    Vec<Arc<DescriptorSet>>,
    Option<Arc<DescriptorSet>>,
) {
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
                    WriteDescriptorSet::image_view(
                        0,
                        hiz_current.mip_views[(l - 1) as usize].clone(),
                    ),
                    WriteDescriptorSet::image_view(1, hiz_current.mip_views[l as usize].clone()),
                    WriteDescriptorSet::image_view(
                        2,
                        hiz_current.mip_views[(l + 1) as usize].clone(),
                    ),
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
/// Pass 1's cull for every world this camera draws, in one stage-major
/// secondary: all worlds' resets, then all their cull dispatches, then all
/// their pass-2 args-builders. See [`RenderCamera::record_cull`] for why.
fn record_cull_secondary(
    scene: &CameraSceneResources<'_>,
    draws: &[WorldDraw],
    occlusion_set: &Arc<DescriptorSet>,
    occlusion_enabled: bool,
) -> Arc<SecondaryAutoCommandBuffer> {
    let pipeline = scene.mvp_build_pipeline;
    let layout = pipeline.layout().clone();

    let mut builder = AutoCommandBufferBuilder::secondary(
        scene.cb_allocator.clone(),
        scene.queue_family_index,
        CommandBufferUsage::SimultaneousUse,
        CommandBufferInheritanceInfo::default(),
    )
    .expect("cull secondary builder");

    // Stage 1: zero every slot's `instance_count` and every candidate
    // counter. Hoisted ahead of all the dispatches — inline, each reset
    // would collide with its own world's dispatch and buy a barrier.
    for d in draws {
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                d.pass1.indirect_template.clone(),
                d.pass1.indirect_args.clone(),
            ))
            .expect("reset indirect instance counts")
            .fill_buffer(d.candidate_count.clone(), 0)
            .expect("reset candidate count")
            .fill_buffer(d.pass1.compact_count.clone().into_slice(), 0)
            .expect("reset compacted draw count")
            // Zeroed, not just counted: a command the draw reads before the
            // compaction wrote it must be a draw of nothing, never whatever
            // was in device memory. Tens of bytes.
            .fill_buffer(d.pass1.compact_args.clone().reinterpret::<[u32]>(), 0)
            .expect("clear compacted commands");
    }

    // Stage 2: the frustum + occlusion cull, one dispatch per world over
    // that world's own slot range.
    builder
        .bind_pipeline_compute(pipeline.clone())
        .expect("bind cull pipeline");
    for d in draws {
        let pc = shaders::mvp_build_cs::PC {
            renderer_capacity: d.cull_range as u32,
            occlusion_enabled: occlusion_enabled as u32,
        };
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                layout.clone(),
                0,
                (d.cull_set.clone(), occlusion_set.clone()),
            )
            .expect("bind cull sets")
            .push_constants(layout.clone(), 0, pc)
            .expect("push cull constants");
        let groups = (d.cull_range as u32).div_ceil(CULL_WORKGROUP_SIZE).max(1);
        // Safety: group count derives from this world's capacity, and the
        // shader bounds-checks against the push constant.
        unsafe {
            builder.dispatch([groups, 1, 1]).expect("dispatch cull");
        }
    }

    // Stage 3: turn each world's candidate count into pass 2's
    // `dispatch_indirect` group counts.
    let args_pipeline = scene.cull_pass2_args_pipeline;
    let args_layout = args_pipeline.layout().clone();
    builder
        .bind_pipeline_compute(args_pipeline.clone())
        .expect("bind args-builder pipeline");
    for d in draws {
        let args_set = build_args_builder_set(scene, &d.candidate_count, &d.pass2_dispatch_args);
        builder
            .bind_descriptor_sets(PipelineBindPoint::Compute, args_layout.clone(), 0, args_set)
            .expect("bind args-builder set");
        // Safety: 1×1×1 dispatch is unconditionally valid.
        unsafe {
            builder.dispatch([1, 1, 1]).expect("dispatch args-builder");
        }
    }

    // Stage 4: drop the slots this cull left empty, so the raster walks only
    // the ones with instances.
    // `slot_count`, never `slot_capacity`: past the live commands the
    // template is zeroed but nothing writes it, and a garbage `index_count`
    // reaching `vkCmdDrawIndexedIndirectCount` hangs the GPU.
    record_compaction(
        &mut builder,
        scene,
        draws.iter().map(|d| (&d.pass1, d.slot_count as u32)),
    );

    builder.build().expect("build cull secondary")
}

/// Append every pass's compaction dispatch, one stage for all of them —
/// they share no buffer, so the barrier before the stage is paid once.
fn record_compaction<'a>(
    builder: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
    scene: &CameraSceneResources<'_>,
    passes: impl Iterator<Item = (&'a DrawResources, u32)>,
) {
    let pipeline = scene.draw_compact_pipeline;
    let layout = pipeline.layout().clone();
    builder
        .bind_pipeline_compute(pipeline.clone())
        .expect("bind draw-compaction pipeline");
    for (pass, slots) in passes {
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                layout.clone(),
                0,
                pass.compact_set.clone(),
            )
            .expect("bind draw-compaction set")
            .push_constants(
                layout.clone(),
                0,
                shaders::draw_compact_cs::PC { slot_count: slots },
            )
            .expect("push draw-compaction constants");
        // Safety: one invocation per command slot; the shader bounds-checks
        // its trailing wavefront against the push constant.
        unsafe {
            builder
                .dispatch([slots.div_ceil(64).max(1), 1, 1])
                .expect("dispatch draw compaction");
        }
    }
}

/// Pass 2's cull for every world, same stage-major shape: all the indirect
/// resets, then all the occlusion-only re-test dispatches.
fn record_cull_pass2_secondary(
    scene: &CameraSceneResources<'_>,
    draws: &[WorldDraw],
    pass2_cull_set1: &Arc<DescriptorSet>,
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

    for d in draws {
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                d.pass2.indirect_template.clone(),
                d.pass2.indirect_args.clone(),
            ))
            .expect("reset pass2 indirect instance counts")
            .fill_buffer(d.pass2.compact_count.clone().into_slice(), 0)
            .expect("reset pass2 compacted draw count")
            .fill_buffer(d.pass2.compact_args.clone().reinterpret::<[u32]>(), 0)
            .expect("clear pass2 compacted commands");
    }

    builder
        .bind_pipeline_compute(pipeline.clone())
        .expect("bind cull pass2 pipeline");
    for d in draws {
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                layout.clone(),
                0,
                (d.pass2_cull_set0.clone(), pass2_cull_set1.clone()),
            )
            .expect("bind cull pass2 sets");
        // Safety: the args come from pass 1's args-builder earlier in the
        // same primary; the counts are `ceil(candidates / 64)`, within the
        // candidate list's capacity.
        unsafe {
            builder
                .dispatch_indirect(d.pass2_dispatch_args.clone())
                .expect("dispatch_indirect cull pass2");
        }
    }

    record_compaction(
        &mut builder,
        scene,
        draws.iter().map(|d| (&d.pass2, d.slot_count as u32)),
    );

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
            builder
                .dispatch([gx, gy, 1])
                .expect("dispatch hiz trailing");
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
/// `hiz_current` and on `view_proj` holding this frame's promoted VP
/// (true from the front of the FrameSlot primary onward). Recorded
/// `SimultaneousUse`; re-recorded only on extent change (the per-mip copy
/// regions depend on the pyramids' dimensions).
fn record_history_update_secondary(
    scene: &CameraSceneResources<'_>,
    hiz_current: &HizPyramid,
    hiz_prev: &HizPyramid,
    view_proj: &Subbuffer<[[f32; 16]]>,
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
            view_proj.clone(),
            prev_view_proj.clone(),
        ))
        .expect("copy view_proj -> prev_view_proj");

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
    compact_args: &Subbuffer<[DrawIndexedIndirectCommand]>,
    compact_count: &Subbuffer<u32>,
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

    // One `vkCmdDrawIndexedIndirectCount` over the compacted commands:
    // `draw_compact_cs` has already dropped the slots the cull left empty and
    // written how many survived, so the raster never walks an empty draw.
    if slot_count > 0 {
        // Safety: both buffers are INDIRECT_BUFFER-usable; the mega index
        // buffer is bound; `first_instance` is bounded by the MVP capacity;
        // the compaction can only ever write `slot_count` commands, which is
        // `max_draw_count`. Indirect device features are enabled at device
        // creation (see RenderApp::new).
        unsafe {
            builder
                .draw_indexed_indirect_count(
                    compact_args.clone().slice(0..slot_count as u64),
                    compact_count.clone(),
                    slot_count as u32,
                )
                .expect("draw_indexed_indirect_count failed");
        }
    }

    builder.build().expect("Failed to build scene secondary")
}
