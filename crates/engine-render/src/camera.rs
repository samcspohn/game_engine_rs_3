//! Render-side camera: owns the GPU attachments (color + depth) a camera
//! renders into, the device-side MVP matrix buffer + descriptor set the
//! camera's draws read from, the pre-recorded *scene secondary* command
//! buffer that issues those draws, and the resolution / capacity policies
//! that decide when any of the above need to be re-created.
//!
//! This is distinct from [`crate::scene::Camera`], which is a pure
//! view/projection math helper. A `RenderCamera` *uses* a `scene::Camera`
//! (or any other source of `view_proj`) at draw time, but owns the GPU
//! resources independently.
//!
//! # Three invalidation domains
//!
//! `RenderCamera` is the focal point of two of the three orthogonal
//! invalidation axes that drive command-buffer rebuilds:
//!
//! 1. **Per-camera resolution** — color/depth attachment extent.
//!    Changes when [`CameraResolution`] resolves to a new extent (e.g. on
//!    swapchain resize for `MatchSwapchain`). Triggers attachment +
//!    scene-secondary rebuild.
//! 2. **Per-camera capacity** — `device_matrices` slot count.
//!    Changes when the scene asks for more draws than the camera's matrix
//!    buffer can hold. Triggers device-buffer + descriptor-set +
//!    scene-secondary rebuild. Geometric growth keeps amortized cost low.
//! 3. **Per-frame-in-flight** — staging matrix ring. *Not* on the camera —
//!    lives on `FrameSlot`. Sized to match camera capacity so the per-frame
//!    `copy_buffer(staging → device)` works.
//!
//! Plus a fourth axis the camera doesn't own:
//!
//! 4. **Per-swapchain-image** — present-blit destination. Lives on
//!    `FrameSlot::blit_secondary` and the primary CB; rebuilt only when the
//!    swapchain image identity changes.
//!
//! Promoting the device matrices + descriptor set + scene secondary onto
//! the camera (rather than per-`FrameSlot`) means:
//!
//! - We allocate the device matrix buffer **once per camera**, not once per
//!   swapchain image. (Previously N swapchain images × identical device
//!   matrices = N redundant allocations of identical contents.)
//! - The scene secondary survives swapchain resize entirely for cameras
//!   whose policy doesn't depend on the swapchain (future `Fixed` shadow
//!   maps, etc.) — only the per-image blit secondary + primary need
//!   rebuilding in that case.
//! - "Scene capacity grew" becomes a distinct, infrequent event with a
//!   clean rebuild path (camera grows; FrameSlots' staging buffers grow;
//!   primaries re-record), independent of swapchain churn.

use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo,
        CommandBufferInheritanceRenderingInfo, CommandBufferUsage,
        CopyBufferInfo, DrawIndexedIndirectCommand, SecondaryAutoCommandBuffer,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
        layout::DescriptorSetLayout,
    },
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{ComputePipeline, GraphicsPipeline, PipelineBindPoint, graphics::viewport::Viewport},
};

use crate::assets::GpuMeshStore;
use crate::shaders;
use crate::transform_gpu::WorldTransformGpu;

// `Pipeline` trait is needed for `pipeline.layout()` method resolution.
use vulkano::pipeline::Pipeline;

/// Pixel format used for camera-owned offscreen color targets.
///
/// HDR-capable (16-bit float per channel) so future tonemapping / bloom /
/// other post-process passes have headroom; the present-blit converts down
/// to whatever sRGB swapchain format the platform offers.
pub const CAMERA_COLOR_FORMAT: Format = Format::R16G16B16A16_SFLOAT;

/// Pixel format used for camera-owned depth targets.
pub const CAMERA_DEPTH_FORMAT: Format = Format::D32_SFLOAT;

/// How a camera's attachment extent is determined relative to the swapchain.
///
/// This is a *policy* — the camera consults it on every swapchain-resize
/// event to decide whether (and to what size) to re-create its attachments.
#[derive(Clone, Copy, Debug)]
pub enum CameraResolution {
    /// Track the swapchain extent 1:1. Attachments are re-created on every
    /// swapchain resize. Used by the main camera so the present-blit stays a
    /// 1:1 copy.
    MatchSwapchain,
    // Reserved for future variants (`todo.txt` §3):
    //   Fixed { width: u32, height: u32 }
    //       — fixed offscreen size, survives swapchain resize. Shadow maps,
    //         editor thumbnails, render-to-texture for portals/mirrors.
    //   ScaleSwapchain { numerator: u32, denominator: u32 }
    //       — fraction of swapchain extent (e.g. 1/2 for half-res reflections).
}

impl CameraResolution {
    /// Resolve a swapchain extent into the actual extent this camera should
    /// render at, given its policy.
    fn resolve(&self, swapchain_extent: [u32; 2]) -> [u32; 2] {
        match self {
            CameraResolution::MatchSwapchain => swapchain_extent,
        }
    }

    /// Does this policy depend on the swapchain extent? `true` means a
    /// swapchain resize *might* require re-creating the camera's attachments;
    /// `false` means swapchain resizes are irrelevant to this camera.
    pub fn depends_on_swapchain(&self) -> bool {
        match self {
            CameraResolution::MatchSwapchain => true,
        }
    }
}

/// One "group" of instances that share the same mesh: a contiguous range
/// of the per-camera MVP / `instance_to_entity` buffers, plus the index of
/// the GPU mesh whose vertex/index buffers must be bound to draw them.
///
/// Generated by sorting the scene's `(mesh_index, entity_index)` instance
/// pairs by `mesh_index` so each mesh's instances are contiguous (see
/// [`sort_topology_by_mesh`]). The scene secondary records exactly one
/// `vkCmdDrawIndexedIndirect` per group, drawing `instance_count` GPU-side
/// instances with `gl_InstanceIndex == first_instance + i`.
#[derive(Clone, Debug)]
pub struct MeshDrawGroup {
    /// Registry mesh slot (indexes `GpuMeshStore`'s table / mega buffers).
    pub mesh_index:     u32,
    /// First MVP-buffer slot used by this group. Equals
    /// `DrawIndexedIndirectCommand::first_instance`. (The group's total
    /// instance count is implied by the next group's `first_instance`; the
    /// per-frame *visible* count is written by the cull pass.)
    pub first_instance: u32,
}

/// Sort `(draws_template, entity_template)` pairs by `mesh_index` so each
/// mesh's instances form a contiguous run, then bucket the runs into
/// [`MeshDrawGroup`]s.
///
/// Returns `(sorted_entity_template, mesh_groups)`. The MVP buffer slot `i`
/// produced by `mvp_build_cs` will correspond to
/// `sorted_entity_template[i]`, and the vertex shader indexes the same slot
/// via `gl_InstanceIndex` (== `first_instance + i_within_group`) — so the
/// permutation only matters for the buffer layout, not the shader logic.
///
/// O(N log N) on a single u32 key. At N=1M this is sub-millisecond and
/// runs only on topology change, not per frame.
fn sort_topology_by_mesh(
    draws_template:  &[u32],
    entity_template: &[u32],
) -> (Vec<u32>, Vec<MeshDrawGroup>, Vec<u32>) {
    debug_assert_eq!(draws_template.len(), entity_template.len());

    // Build (mesh_index, entity_index) pairs, sort by mesh_index (stable
    // so two instances with the same mesh keep their relative order, which
    // makes debugging predictable).
    let mut pairs: Vec<(u32, u32)> = draws_template
        .iter()
        .copied()
        .zip(entity_template.iter().copied())
        .collect();
    pairs.sort_by_key(|&(mesh, _)| mesh);

    let sorted_entities: Vec<u32> = pairs.iter().map(|&(_, e)| e).collect();

    // Bucket contiguous runs of identical mesh_index into draw groups, and
    // record each sorted draw-slot's group index for the cull pass.
    let mut groups: Vec<MeshDrawGroup> = Vec::new();
    let mut draw_slot_group: Vec<u32> = vec![0u32; pairs.len()];
    let mut i = 0usize;
    while i < pairs.len() {
        let mesh = pairs[i].0;
        let first = i;
        let group_idx = groups.len() as u32;
        while i < pairs.len() && pairs[i].0 == mesh {
            draw_slot_group[i] = group_idx;
            i += 1;
        }
        groups.push(MeshDrawGroup {
            mesh_index:     mesh,
            first_instance: first as u32,
        });
    }
    (sorted_entities, groups, draw_slot_group)
}

/// Per-call bundle of the GPU/scene state the camera needs to (re)build its
/// scene secondary. Bundled into a struct purely to keep the rebuild API's
/// parameter list manageable; nothing is owned beyond the call.
pub struct CameraSceneResources<'a> {
    pub cb_allocator:             &'a Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: &'a Arc<StandardDescriptorSetAllocator>,
    pub memory_allocator:         &'a Arc<StandardMemoryAllocator>,
    pub pipeline:                 &'a Arc<GraphicsPipeline>,
    pub queue_family_index:       u32,
    pub mesh_store:               &'a GpuMeshStore,
    /// Per-instance mesh indices, one entry per draw the scene secondary
    /// should record. `len()` is the **logical** draw count for the next
    /// rebuild; capacity grows to fit it (with headroom).
    pub draws_template:           &'a [u32],
    /// Per-instance entity index, parallel to `draws_template` — one entry
    /// per draw, pointing at a slot in the world's SoT buffers. Uploaded
    /// into `instance_to_entity_buffer` and read by the mvp-build compute
    /// shader to fetch each draw's TRS from the SoT.
    pub entity_template:          &'a [u32],
    /// Per-world transform state (SoT buffers + compute pipelines). Used to
    /// build the camera's `mvp_build_set0` (which captures the SoT buffer
    /// handles) and to look up the mvp-build pipeline + set layouts. Does
    /// **not** need to be the same instance across calls — but if its SoT
    /// buffers were re-allocated since the last time the camera saw it,
    /// `mvp_build_set0` will be silently stale; callers must re-invoke an
    /// invalidation entry point (`on_world_capacity_change`) in that case.
    pub world_transforms:         &'a WorldTransformGpu,
}

/// Render-side camera: owns the offscreen color + depth attachments the
/// camera renders into, the device-side MVP matrix storage buffer + its
/// descriptor set, the pre-recorded scene secondary, and the policies that
/// drive when any of these have to be re-created.
///
/// The host-visible *staging* matrix ring lives on `FrameSlot`
/// (per-frame-in-flight); the per-frame `copy_buffer(staging → device)`
/// runs in the primary CB and bridges those two axes.
pub struct RenderCamera {
    // ── Resolution / attachments ────────────────────────────────────────
    resolution:        CameraResolution,
    extent:            [u32; 2],
    color_image:       Arc<Image>,
    depth_image:       Arc<Image>,
    color_view:        Arc<ImageView>,
    depth_view:        Arc<ImageView>,

    // ── Matrix storage (per-camera, stable across frames) ───────────────
    /// Device-local matrix buffer the vertex shader reads via
    /// `descriptor_set`, **and that the mvp-build compute pass writes**.
    /// Sized to `allocated_capacity` slots; grown geometrically so
    /// amortized rebuild cost is O(1) per added draw.
    device_matrices:   Subbuffer<[[f32; 16]]>,
    /// Graphics set 0 — references `device_matrices`. Re-created together
    /// with `device_matrices` whenever the buffer is re-allocated (the set
    /// captures the buffer by handle; the new buffer needs a new set).
    descriptor_set:    Arc<DescriptorSet>,
    /// Pre-recorded scene secondary that issues exactly `draw_count` draws,
    /// each indexing into `device_matrices` via `first_instance`. Inherits
    /// the primary's dynamic-rendering scope.
    scene_secondary:   Arc<SecondaryAutoCommandBuffer>,

    // ── MVP-build compute (per-camera state) ────────────────────────────
    /// `instance → entity` lookup, length == `draw_count`. The mvp-build
    /// shader does `entity = idx[gl_GlobalInvocationID.x]; mvp[gid] =
    /// view_proj * model_from_trs(sot[entity])`. Re-uploaded whenever
    /// scene topology changes.
    ///
    /// **Sorted by mesh.** ADR-0004 Phase 1: instances are grouped by
    /// `mesh_index` so each mesh's slice of the MVP buffer is contiguous,
    /// which is what lets a single `vkCmdDrawIndexedIndirect` per mesh
    /// fan out to `instance_count` GPU-side instances with `gl_InstanceIndex
    /// == first_instance + i` indexing the per-mesh contiguous range.
    instance_to_entity: Subbuffer<[u32]>,

    // ── Indirect draw (ADR-0004 Phase 1) ────────────────────────────────
    /// Device-readable indirect-args buffer; one
    /// [`DrawIndexedIndirectCommand`] per `mesh_group`. Sized to
    /// `mesh_groups.len()` (typically tiny — one per distinct mesh in the
    /// scene). Re-uploaded whenever scene topology changes.
    ///
    /// Each entry's `instance_count` is the number of instances of that
    /// mesh and `first_instance` is the start offset into the per-camera
    /// MVP buffer for that mesh's range. The vertex shader sees
    /// `gl_InstanceIndex == first_instance + i` and indexes the same MVP
    /// buffer slot it would have under the old per-instance `draw_indexed`
    /// path — no shader change required.
    indirect_args_buf: Subbuffer<[DrawIndexedIndirectCommand]>,
    /// One entry per distinct mesh in the scene: the mesh index (into
    /// `gpu_meshes`) and the slice of `indirect_args_buf` that holds its
    /// command. Today each group's slice has length 1 (one indirect
    /// command per mesh); kept as `Vec` so a future Phase 2 / multi-LOD
    /// path can grow per-mesh slices without changing the loop in the
    /// scene secondary recorder.
    mesh_groups:       Vec<MeshDrawGroup>,
    /// MVP-build descriptor set 0 — binds the world's SoT (pos/rot/scale),
    /// `instance_to_entity`, and `device_matrices` as the output. Captured
    /// by buffer handle, so it must be re-allocated whenever **any** of
    /// those four buffers are re-allocated (camera capacity grows OR world
    /// capacity grows OR topology change re-uploads `instance_to_entity`).
    mvp_build_set0:    Arc<DescriptorSet>,

    /// Pre-recorded compute secondary that runs the per-camera mvp-build
    /// dispatch: binds `mvp_build_pipeline`, `(mvp_build_set0,
    /// mvp_build_set1)`, pushes `draw_count`, and dispatches
    /// `ceil(draw_count / 64)` work-groups.
    ///
    /// Captured by every FrameSlot's primary CB (one shared secondary
    /// across all FrameSlots; recorded `SimultaneousUse` because
    /// multiple FrameSlot primaries may reference it concurrently). Re-
    /// recorded whenever `mvp_build_set0` or `draw_count` change.
    /// `mvp_build_set1` (which binds the world's stable
    /// `sot_view_proj`) is fixed across capacity changes.
    mvp_build_secondary: Arc<SecondaryAutoCommandBuffer>,

    /// Per-draw-slot group index (sorted order); the cull pass reads it to
    /// find each instance's group. Host-visible BAR; rebuilt on topology
    /// change alongside `instance_to_entity`.
    draw_slot_group:   Subbuffer<[u32]>,
    /// Per-group local-space bounding sphere (`center.xyz`, `radius.w`), read
    /// by the cull pass. Host-visible BAR; rebuilt on topology change.
    group_bounds:      Subbuffer<[[f32; 4]]>,
    /// Host-visible template of the indirect commands with `instance_count`
    /// pre-zeroed; copied into `indirect_args_buf` each frame to reset the
    /// counts before the cull pass accumulates the visible counts.
    indirect_template: Subbuffer<[DrawIndexedIndirectCommand]>,

    /// Number of `[f32; 16]` slots actually allocated in `device_matrices`.
    /// Always >= 1 (Vulkan won't allocate zero-sized buffers) and
    /// always >= `draw_count`.
    allocated_capacity: usize,
    /// Number of `DrawIndexedIndirectCommand` slots allocated in
    /// `indirect_args_buf`. Grown geometrically alongside the rest of the
    /// per-camera buffers. Always >= `mesh_groups.len()`.
    indirect_args_capacity: usize,
    /// Number of draws baked into `scene_secondary` (== `draws_template.len()`
    /// at the last rebuild). Distinct from `allocated_capacity` because the
    /// device buffer may have headroom from an earlier geometric grow.
    draw_count:        usize,
}

impl RenderCamera {
    /// Build a camera whose attachments track the swapchain extent.
    pub fn new_match_swapchain(
        swapchain_extent: [u32; 2],
        scene:            &CameraSceneResources<'_>,
    ) -> Self {
        Self::new(CameraResolution::MatchSwapchain, swapchain_extent, scene)
    }

    /// Build a camera with the given resolution policy. `swapchain_extent`
    /// is consulted only if the policy depends on it.
    pub fn new(
        resolution:       CameraResolution,
        swapchain_extent: [u32; 2],
        scene:            &CameraSceneResources<'_>,
    ) -> Self {
        let extent = resolution.resolve(swapchain_extent);
        let (color_image, color_view, depth_image, depth_view) =
            allocate_attachments(scene.memory_allocator, extent);

        let draw_count = scene.draws_template.len();
        let allocated_capacity = draw_count.max(1);
        let (device_matrices, descriptor_set) = allocate_matrices_and_set(
            scene.memory_allocator,
            scene.descriptor_set_allocator,
            scene.pipeline,
            allocated_capacity,
        );

        // Sort instances by mesh so each mesh's instances are contiguous,
        // then upload the reordered entity table + build the indirect-args
        // buffer. The MVP buffer slot `i` in `device_matrices` corresponds
        // to `sorted_entities[i]`, which is what the vertex shader's
        // `gl_InstanceIndex` will index into.
        let (sorted_entities, mesh_groups, draw_slot_group_vec) =
            sort_topology_by_mesh(scene.draws_template, scene.entity_template);
        let instance_to_entity = allocate_and_upload_instance_to_entity(
            scene.memory_allocator,
            &sorted_entities,
            allocated_capacity,
        );
        let draw_slot_group = allocate_and_upload_draw_slot_group(
            scene.memory_allocator,
            &draw_slot_group_vec,
            allocated_capacity,
        );
        let group_bounds = allocate_and_upload_group_bounds(
            scene.memory_allocator,
            scene.mesh_store,
            &mesh_groups,
        );
        let indirect_args_capacity = mesh_groups.len().max(1);
        let (indirect_template, indirect_args_buf) = allocate_indirect_buffers(
            scene.memory_allocator,
            scene.mesh_store,
            &mesh_groups,
            indirect_args_capacity,
        );

        let mvp_build_set0 = build_mvp_build_set0(
            scene.descriptor_set_allocator,
            scene.world_transforms,
            &instance_to_entity,
            &device_matrices,
            &draw_slot_group,
            &group_bounds,
            &indirect_args_buf,
        );
        let mvp_build_secondary = record_mvp_build_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.world_transforms.mvp_build_pipeline(),
            &mvp_build_set0,
            scene.world_transforms.mvp_build_set1(),
            &indirect_template,
            &indirect_args_buf,
            draw_count,
        );
        let scene_secondary = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &descriptor_set,
            scene.mesh_store,
            &mesh_groups,
            &indirect_args_buf,
            extent,
        );

        RenderCamera {
            resolution,
            extent,
            color_image,
            depth_image,
            color_view,
            depth_view,
            device_matrices,
            descriptor_set,
            scene_secondary,
            instance_to_entity,
            indirect_args_buf,
            mesh_groups,
            mvp_build_set0,
            mvp_build_secondary,
            draw_slot_group,
            group_bounds,
            indirect_template,
            allocated_capacity,
            indirect_args_capacity,
            draw_count,
        }
    }

    /// Inform the camera that the swapchain has been re-created with a new
    /// extent. Returns `true` if the camera re-created its attachments
    /// **or** re-recorded its scene secondary (callers must then rebuild any
    /// command buffer that references the camera's color/depth views or its
    /// scene secondary — i.e. every `FrameSlot::command_buffer`).
    ///
    /// For [`CameraResolution::MatchSwapchain`] this re-creates whenever
    /// the extent actually changes (and re-records the scene secondary
    /// because the viewport baked into it depends on the extent); for
    /// swapchain-independent policies (future) this is a no-op.
    pub fn on_swapchain_resize(
        &mut self,
        new_swapchain_extent: [u32; 2],
        scene:                &CameraSceneResources<'_>,
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
        self.extent      = new_extent;
        self.color_image = color_image;
        self.color_view  = color_view;
        self.depth_image = depth_image;
        self.depth_view  = depth_view;

        // The viewport baked into the scene secondary depends on `extent`,
        // so the secondary itself has to be re-recorded. The matrix buffer,
        // descriptor set, and indirect-args buffer are extent-independent
        // and survive untouched.
        self.scene_secondary = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &self.descriptor_set,
            scene.mesh_store,
            &self.mesh_groups,
            &self.indirect_args_buf,
            self.extent,
        );
        self.draw_count = scene.draws_template.len();
        true
    }

    /// Ensure the camera's `device_matrices` can hold at least `needed`
    /// slots and that the scene secondary is recorded for exactly
    /// `scene.draws_template.len()` draws.
    ///
    /// Returns `true` if anything was re-recorded (so callers know to
    /// rebuild dependent CBs — the per-image primaries reference
    /// `device_matrices` via `copy_buffer` and `scene_secondary` via
    /// `execute_commands`, so any rebuild here invalidates them).
    ///
    /// Capacity grows geometrically (≥ 2× current) so amortized cost is
    /// O(1) per added draw. Capacity *never shrinks* — buffers are cheap
    /// and shrinking would just trade memory for future re-allocations.
    pub fn ensure_capacity(
        &mut self,
        needed: usize,
        scene:  &CameraSceneResources<'_>,
        force:  bool,
    ) -> bool {
        let needed_slots   = scene.draws_template.len().max(needed);
        // `force` covers topology *content* changes that don't change the draw
        // count (a redirect flip reassigns slots) and mega-buffer growth (the
        // scene secondary's bound buffers became stale) — neither of which the
        // count comparison below would catch.
        let topology_changed = needed_slots != self.draw_count || force;

        if needed_slots <= self.allocated_capacity && !topology_changed {
            return false;
        }

        if needed_slots > self.allocated_capacity {
            // Geometric growth: at least double, at least what's needed, at
            // least 1. Mirrors `Vec`'s amortized-O(1) growth strategy.
            let new_capacity = needed_slots
                .max(self.allocated_capacity.saturating_mul(2))
                .max(1);
            let (device_matrices, descriptor_set) = allocate_matrices_and_set(
                scene.memory_allocator,
                scene.descriptor_set_allocator,
                scene.pipeline,
                new_capacity,
            );
            self.device_matrices    = device_matrices;
            self.descriptor_set     = descriptor_set;
            self.allocated_capacity = new_capacity;
        }

        // Always re-sort topology and re-upload the (sorted) instance→entity
        // table on topology change. Re-build the indirect-args buffer if the
        // group count exceeds its current capacity (geometric growth) or if
        // any group's contents changed (counts/offsets shifted) — simplest
        // policy: always rebuild on topology change, since the buffer is
        // tiny (one struct per distinct mesh).
        let (sorted_entities, mesh_groups, draw_slot_group_vec) =
            sort_topology_by_mesh(scene.draws_template, scene.entity_template);

        self.instance_to_entity = allocate_and_upload_instance_to_entity(
            scene.memory_allocator,
            &sorted_entities,
            self.allocated_capacity,
        );
        self.draw_slot_group = allocate_and_upload_draw_slot_group(
            scene.memory_allocator,
            &draw_slot_group_vec,
            self.allocated_capacity,
        );
        self.group_bounds = allocate_and_upload_group_bounds(
            scene.memory_allocator,
            scene.mesh_store,
            &mesh_groups,
        );

        if mesh_groups.len() > self.indirect_args_capacity {
            self.indirect_args_capacity = mesh_groups
                .len()
                .max(self.indirect_args_capacity.saturating_mul(2))
                .max(1);
        }
        let (indirect_template, indirect_args_buf) = allocate_indirect_buffers(
            scene.memory_allocator,
            scene.mesh_store,
            &mesh_groups,
            self.indirect_args_capacity,
        );
        self.indirect_template = indirect_template;
        self.indirect_args_buf = indirect_args_buf;
        self.mesh_groups = mesh_groups;

        self.mvp_build_set0 = build_mvp_build_set0(
            scene.descriptor_set_allocator,
            scene.world_transforms,
            &self.instance_to_entity,
            &self.device_matrices,
            &self.draw_slot_group,
            &self.group_bounds,
            &self.indirect_args_buf,
        );

        // mvp_build_secondary captures the new mvp_build_set0 (always);
        // re-record on every capacity / topology change.
        self.mvp_build_secondary = record_mvp_build_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.world_transforms.mvp_build_pipeline(),
            &self.mvp_build_set0,
            scene.world_transforms.mvp_build_set1(),
            &self.indirect_template,
            &self.indirect_args_buf,
            scene.draws_template.len(),
        );

        // Always re-record when topology changed OR capacity grew — the
        // descriptor set may be new, and the indirect-args buffer that the
        // secondary references was just re-allocated.
        self.scene_secondary = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &self.descriptor_set,
            scene.mesh_store,
            &self.mesh_groups,
            &self.indirect_args_buf,
            self.extent,
        );
        self.draw_count = scene.draws_template.len();
        true
    }

    /// Notify the camera that the world's SoT buffers were re-allocated
    /// (capacity grew). Re-builds `mvp_build_set0` (so the per-camera
    /// mvp-build compute pass binds the new SoT buffer handles) AND
    /// re-records `mvp_build_secondary` (so it captures the new
    /// `mvp_build_set0`). Returns `true` so the caller knows that any
    /// pre-recorded primary CB referencing `mvp_build_secondary` (i.e.
    /// every FrameSlot's composing primary) must be re-recorded.
    pub fn on_world_capacity_change(
        &mut self,
        cb_allocator:             &Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        queue_family_index:       u32,
        world_transforms:         &WorldTransformGpu,
    ) -> bool {
        self.mvp_build_set0 = build_mvp_build_set0(
            descriptor_set_allocator,
            world_transforms,
            &self.instance_to_entity,
            &self.device_matrices,
            &self.draw_slot_group,
            &self.group_bounds,
            &self.indirect_args_buf,
        );
        self.mvp_build_secondary = record_mvp_build_secondary(
            cb_allocator,
            queue_family_index,
            world_transforms.mvp_build_pipeline(),
            &self.mvp_build_set0,
            world_transforms.mvp_build_set1(),
            &self.indirect_template,
            &self.indirect_args_buf,
            self.draw_count,
        );
        true
    }

    /// Current attachment extent. Held for future multi-camera UI / debug
    /// readout; no longer read by the renderer hot path now that the
    /// viewport is baked into the camera-owned scene secondary.
    #[allow(dead_code)]
    pub fn extent(&self)              -> [u32; 2]                 { self.extent }
    pub fn color_image(&self)         -> &Arc<Image>              { &self.color_image }
    /// Held for future post-process passes / debug visualizers; not yet read
    /// by the current single-pass renderer.
    #[allow(dead_code)]
    pub fn depth_image(&self)         -> &Arc<Image>              { &self.depth_image }
    pub fn color_view(&self)          -> &Arc<ImageView>          { &self.color_view  }
    pub fn depth_view(&self)          -> &Arc<ImageView>          { &self.depth_view  }
    #[allow(dead_code)]
    pub fn device_matrices(&self)     -> &Subbuffer<[[f32; 16]]>  { &self.device_matrices }
    pub fn scene_secondary(&self)     -> &Arc<SecondaryAutoCommandBuffer> { &self.scene_secondary }
    /// Held so the secondary's binding can be inspected; the primary CB
    /// only references the secondary, not the set directly.
    #[allow(dead_code)]
    pub fn mvp_build_set0(&self)      -> &Arc<DescriptorSet>      { &self.mvp_build_set0 }
    /// Pre-recorded mvp-build compute secondary. Executed once per frame
    /// from each FrameSlot's primary CB. Shared across FrameSlots
    /// (recorded `SimultaneousUse`).
    pub fn mvp_build_secondary(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.mvp_build_secondary
    }
    /// Held so test/debug code can inspect the per-camera lookup; the
    /// renderer hot path never reads this directly (it lives on the GPU).
    #[allow(dead_code)]
    pub fn instance_to_entity(&self)  -> &Subbuffer<[u32]>        { &self.instance_to_entity }
    /// Number of `[f32; 16]` slots in `device_matrices`. Per-frame staging
    /// buffers should match this exactly so the in-primary `copy_buffer`
    /// has same-sized source and destination.
    pub fn allocated_capacity(&self)  -> usize                    { self.allocated_capacity }
    /// Number of draws the scene secondary is currently recorded for. This
    /// is what the host should write into the staging buffer each frame;
    /// the rest of the staging/device buffer (up to `allocated_capacity`)
    /// is unused headroom.
    pub fn draw_count(&self)          -> usize                    { self.draw_count }
}

fn allocate_attachments(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    extent:           [u32; 2],
) -> (Arc<Image>, Arc<ImageView>, Arc<Image>, Arc<ImageView>) {
    let [w, h] = extent;
    let color_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format:     CAMERA_COLOR_FORMAT,
            extent:     [w, h, 1],
            usage:      ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
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
            format:     CAMERA_DEPTH_FORMAT,
            extent:     [w, h, 1],
            usage:      ImageUsage::DEPTH_STENCIL_ATTACHMENT,
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

/// Allocate a device-local `[f32; 16]` storage buffer of `capacity` slots
/// and the descriptor set that points at it. Grouped because the descriptor
/// set captures the buffer by handle, so any new buffer needs a fresh set.
///
/// Usage is `STORAGE_BUFFER` only: the mvp-build compute writes into it
/// and the vertex shader reads from it. There is no longer a host-driven
/// `vkCmdCopyBuffer` into this buffer (replaced by the compute pass), so
/// `TRANSFER_DST` is no longer needed.
fn allocate_matrices_and_set(
    memory_allocator:         &Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    pipeline:                 &Arc<GraphicsPipeline>,
    capacity:                 usize,
) -> (Subbuffer<[[f32; 16]]>, Arc<DescriptorSet>) {
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

    (device_matrices, descriptor_set)
}

/// Allocate the per-camera `instance → entity` lookup buffer of length
/// `capacity`, and immediately upload the contents of `entity_template`
/// (which is `<= capacity` long). Tail past `entity_template.len()` is
/// undefined; the mvp-build shader is dispatched only for `draw_count`
/// invocations, so the tail is never read.
///
/// HOST_SEQUENTIAL_WRITE memory keeps the upload trivial (single `write()`
/// at allocation time, no staging+copy). Topology changes are infrequent;
/// for very large lookup tables a copy-from-staging path may be worth it,
/// but the current scale (one entry per draw) makes that overkill.
fn allocate_and_upload_instance_to_entity(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    entity_template:  &[u32],
    capacity:         usize,
) -> Subbuffer<[u32]> {
    let cap = capacity.max(1).max(entity_template.len());
    let buf: Subbuffer<[u32]> = Buffer::new_slice::<u32>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        cap as u64,
    )
    .expect("Failed to allocate instance_to_entity buffer");

    {
        let mut guard = buf.write().expect("instance_to_entity.write");
        guard[..entity_template.len()].copy_from_slice(entity_template);
        // Tail (if `cap > entity_template.len()`) is left undefined; never
        // read by the dispatch (capped to `draw_count == entity_template.len()`).
    }

    buf
}

/// Build the mvp-build set 0 descriptor set: SoT (pos/rot/scl) + idx + mvp.
/// All buffers are captured by handle, so this set must be re-allocated
/// whenever any of those four buffers is re-allocated.
fn build_mvp_build_set0(
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    world:                    &WorldTransformGpu,
    instance_to_entity:       &Subbuffer<[u32]>,
    device_matrices:          &Subbuffer<[[f32; 16]]>,
    draw_slot_group:          &Subbuffer<[u32]>,
    group_bounds:             &Subbuffer<[[f32; 4]]>,
    indirect_args:            &Subbuffer<[DrawIndexedIndirectCommand]>,
) -> Arc<DescriptorSet> {
    DescriptorSet::new(
        descriptor_set_allocator.clone(),
        world.mvp_build_set0_layout().clone(),
        [
            WriteDescriptorSet::buffer(0, world.sot_positions().clone()),
            WriteDescriptorSet::buffer(1, world.sot_rotations().clone()),
            WriteDescriptorSet::buffer(2, world.sot_scales().clone()),
            WriteDescriptorSet::buffer(3, instance_to_entity.clone()),
            WriteDescriptorSet::buffer(4, device_matrices.clone()),
            WriteDescriptorSet::buffer(5, draw_slot_group.clone()),
            WriteDescriptorSet::buffer(6, group_bounds.clone()),
            // The cull pass views the indirect commands as a flat `u32[]`
            // (reads `first_instance`, atomic-adds `instance_count`).
            WriteDescriptorSet::buffer(7, indirect_args.clone().reinterpret::<[u32]>()),
        ],
        [],
    )
    .expect("Failed to allocate mvp_build_set0")
}

/// Allocate the per-camera indirect-command buffers: a **host-visible
/// template** (geometry + `first_instance` per group, with `instance_count`
/// pre-zeroed) and the **device-local args** buffer the cull pass writes and
/// the draw reads. Each frame the template is `vkCmdCopyBuffer`d into the
/// args buffer (resetting the counts) before the cull accumulates the visible
/// count via atomics — hence the args buffer is device-local (BAR atomics are
/// prohibitively slow) and carries `TRANSFER_DST`.
fn allocate_indirect_buffers(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    mesh_store:       &GpuMeshStore,
    mesh_groups:      &[MeshDrawGroup],
    capacity:         usize,
) -> (
    Subbuffer<[DrawIndexedIndirectCommand]>,
    Subbuffer<[DrawIndexedIndirectCommand]>,
) {
    let cap = capacity.max(1).max(mesh_groups.len());

    let template: Subbuffer<[DrawIndexedIndirectCommand]> =
        Buffer::new_slice::<DrawIndexedIndirectCommand>(
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
    {
        let mut guard = template.write().expect("indirect_template.write");
        for (slot, group) in guard.iter_mut().zip(mesh_groups.iter()) {
            let geom = mesh_store.slot_geometry(group.mesh_index);
            *slot = DrawIndexedIndirectCommand {
                index_count:    geom.map(|g| g.index_count).unwrap_or(0),
                // Reset to 0 — the cull pass accumulates the visible count.
                instance_count: 0,
                first_index:    geom.map(|g| g.first_index).unwrap_or(0),
                vertex_offset:  geom.map(|g| g.vertex_offset as u32).unwrap_or(0),
                first_instance: group.first_instance,
            };
        }
    }

    let args: Subbuffer<[DrawIndexedIndirectCommand]> =
        Buffer::new_slice::<DrawIndexedIndirectCommand>(
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

/// Allocate + upload the per-draw-slot group-index table (host-visible BAR).
/// One `u32` per sorted draw-slot, read by the cull pass.
fn allocate_and_upload_draw_slot_group(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    draw_slot_group:  &[u32],
    capacity:         usize,
) -> Subbuffer<[u32]> {
    let cap = capacity.max(1).max(draw_slot_group.len());
    let buf: Subbuffer<[u32]> = Buffer::new_slice::<u32>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        cap as u64,
    )
    .expect("Failed to allocate draw_slot_group buffer");
    {
        let mut guard = buf.write().expect("draw_slot_group.write");
        guard[..draw_slot_group.len()].copy_from_slice(draw_slot_group);
    }
    buf
}

/// Allocate + upload the per-group bounding spheres (host-visible BAR). One
/// `vec4` per group (`center.xyz`, `radius.w`), copied from the mesh table's
/// CPU mirror; read by the cull pass for the frustum test.
fn allocate_and_upload_group_bounds(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    mesh_store:       &GpuMeshStore,
    mesh_groups:      &[MeshDrawGroup],
) -> Subbuffer<[[f32; 4]]> {
    let cap = mesh_groups.len().max(1);
    let buf: Subbuffer<[[f32; 4]]> = Buffer::new_slice::<[f32; 4]>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        cap as u64,
    )
    .expect("Failed to allocate group_bounds buffer");
    {
        let mut guard = buf.write().expect("group_bounds.write");
        for (slot, group) in guard.iter_mut().zip(mesh_groups.iter()) {
            *slot = match mesh_store.slot_geometry(group.mesh_index) {
                Some(g) => [
                    g.bounds_center[0],
                    g.bounds_center[1],
                    g.bounds_center[2],
                    g.bounds_radius,
                ],
                None => [0.0, 0.0, 0.0, 0.0],
            };
        }
    }
    buf
}

/// Record the scene secondary: viewport + pipeline + descriptor set bind +
/// **a single `vkCmdDrawIndexedIndirect`** over the per-group args buffer
/// (`drawCount == mesh_groups.len()`, ADR-0004).
/// Each call draws `instance_count` GPU-side instances of that group's
/// mesh; `first_instance` is baked into the indirect-args struct so the
/// vertex shader's `gl_InstanceIndex` indexes the contiguous slice of
/// `device_matrices` that the mvp-build compute pass populated for this
/// group.
///
/// Inherits the primary's dynamic-rendering scope (color/depth formats
/// must match what the primary's `begin_rendering` will declare). The
/// secondary may NOT call `begin_rendering`/`end_rendering` itself.
fn record_scene_secondary(
    cb_allocator:        &Arc<StandardCommandBufferAllocator>,
    queue_family_index:  u32,
    pipeline:            &Arc<GraphicsPipeline>,
    descriptor_set:      &Arc<DescriptorSet>,
    mesh_store:          &GpuMeshStore,
    mesh_groups:         &[MeshDrawGroup],
    indirect_args_buf:   &Subbuffer<[DrawIndexedIndirectCommand]>,
    extent:              [u32; 2],
) -> Arc<SecondaryAutoCommandBuffer> {
    let [cam_w, cam_h] = extent;

    let inheritance = CommandBufferInheritanceInfo {
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

    let mut builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        // The camera's scene secondary is shared across **every** FrameSlot's
        // primary CB (one per swapchain image). With `MAX_FRAMES_IN_FLIGHT`
        // primaries potentially in flight at the same time, the same
        // secondary is referenced by multiple in-flight primaries
        // simultaneously — which `MultipleSubmit` (Vulkan: resubmit-after-
        // completion) forbids. `SimultaneousUse` is the Vulkan
        // `SIMULTANEOUS_USE_BIT` that explicitly permits this.
        CommandBufferUsage::SimultaneousUse,
        inheritance,
    )
    .expect("Failed to create scene secondary builder");

    builder
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

    // All meshes share one mega vertex + one mega index buffer; bind once.
    // Each group's indirect command carries its own `first_index` /
    // `vertex_offset` (selecting that mesh's mega-buffer slice), `first_instance`
    // (the contiguous base of its MVP-buffer slice, populated by the mvp-build
    // compute pass earlier in this primary CB), and `instance_count`.
    builder
        .bind_vertex_buffers(0, mesh_store.mega_vertex_buffer().clone())
        .expect("bind mega vertex buffer failed")
        .bind_index_buffer(mesh_store.mega_index_buffer().clone())
        .expect("bind mega index buffer failed");

    // ADR-0004: a single `vkCmdDrawIndexedIndirect` over the whole args buffer
    // (drawCount == number of mesh groups) replaces the old per-group loop.
    // Each command fans out to its `instance_count` GPU-side instances with
    // `gl_InstanceIndex == first_instance + i`. `draw_indexed_indirect` infers
    // drawCount from `subbuffer.len()`; the `multi_draw_indirect` device
    // feature (enabled at device creation) permits drawCount > 1.
    if !mesh_groups.is_empty() {
        let draws = indirect_args_buf
            .clone()
            .slice(0 .. mesh_groups.len() as u64);
        // Safety: the args buffer is INDIRECT_BUFFER-usable; the mega index
        // buffer is bound (above); every command's `first_instance` is bounded
        // by `allocated_capacity` (validated when the groups were built);
        // `multi_draw_indirect` / `draw_indirect_first_instance` are enabled at
        // device creation (see RenderApp::new).
        unsafe {
            builder
                .draw_indexed_indirect(draws)
                .expect("draw_indexed_indirect failed");
        }
    }

    builder.build().expect("Failed to build scene secondary")
}

/// Record the per-camera mvp-build compute secondary: bind the pipeline,
/// bind `(mvp_build_set0, mvp_build_set1)`, push `draw_count`, and
/// dispatch `ceil(draw_count / 64)` work-groups. No render-pass
/// inheritance — compute can't run inside a render pass.
///
/// The secondary is referenced by every FrameSlot's primary CB; with
/// multiple primaries potentially in flight on the GPU at the same time
/// (the host-side timeline wait gates only *CPU* mutations of shared
/// staging, not GPU execution overlap), the secondary is recorded with
/// `SimultaneousUse`.
fn record_mvp_build_secondary(
    cb_allocator:        &Arc<StandardCommandBufferAllocator>,
    queue_family_index:  u32,
    mvp_build_pipeline:  &Arc<ComputePipeline>,
    mvp_build_set0:      &Arc<DescriptorSet>,
    mvp_build_set1:      &Arc<DescriptorSet>,
    indirect_template:   &Subbuffer<[DrawIndexedIndirectCommand]>,
    indirect_args:       &Subbuffer<[DrawIndexedIndirectCommand]>,
    draw_count:          usize,
) -> Arc<SecondaryAutoCommandBuffer> {
    let layout = mvp_build_pipeline.layout().clone();
    let groups = (draw_count as u32).div_ceil(64).max(1);
    let pc     = shaders::mvp_build_cs::PC { draw_count: draw_count as u32 };

    let mut builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::SimultaneousUse,
        CommandBufferInheritanceInfo::default(),
    ).expect("mvp_build secondary builder");

    // Reset every group's `instance_count` to 0 by copying the pre-zeroed
    // host template into the device args buffer. Vulkano auto-syncs this
    // transfer write against the cull dispatch's atomic read-modify-write.
    builder
        .copy_buffer(CopyBufferInfo::buffers(
            indirect_template.clone(),
            indirect_args.clone(),
        ))
        .expect("reset indirect instance counts");

    builder
        .bind_pipeline_compute(mvp_build_pipeline.clone()).expect("bind mvp pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            layout.clone(),
            0,
            (mvp_build_set0.clone(), mvp_build_set1.clone()),
        ).expect("bind mvp sets")
        .push_constants(layout, 0, pc).expect("push mvp pc");
    // Safety: dispatch count derived from `draw_count`; shader bounds-
    // checks against the push-constant `draw_count`.
    unsafe {
        builder.dispatch([groups, 1, 1]).expect("dispatch mvp");
    }
    builder.build().expect("build mvp_build secondary")
}
