//! GPU-side transform pipeline: stable per-component "source of truth"
//! (SoT) buffers, the **shared** per-frame staging mirrors that feed them,
//! the compute pipelines / secondaries / descriptor sets that promote
//! staging → SoT, and the **timeline semaphore** that gates host writes
//! to the shared staging against the previous frame's compute work.
//!
//! # Architecture (post ADR-0003 refactor)
//!
//! Per frame, in this order, all inside the slot's pre-recorded primary CB:
//!
//! 1. **Scatter compute (×3, one per component)**
//!    Reads `staging_<comp>[i]` and `dirty[i]`; if the dirty bit is set,
//!    writes `sot_<comp>[i] = staging_<comp>[i]`. Dispatched over
//!    `entity_capacity` invocations.
//!
//! 2. **MVP-build compute (×1, per camera)**
//!    Reads SoT pos/rot/scale indexed by a per-camera `instance → entity`
//!    lookup, multiplies `view_proj * model`, writes the per-camera MVP
//!    buffer the vertex shader will read.
//!
//! 3. **Graphics**: scene secondary + blit secondary.
//!
//! # What lives where (post ADR-0003)
//!
//! Pre-ADR-0003 every per-frame-in-flight resource was duplicated `N=4`
//! times across [`crate::FrameSlot`]s, costing ~`4×` VRAM on the staging
//! triple. After ADR-0003 the **single** in-flight copy lives here on
//! [`WorldTransformGpu`]; per-frame independence is recovered by host-
//! waiting on a timeline semaphore signaled at `COMPUTE_SHADER` stage end.
//!
//! | Resource                       | Owner          |
//! |--------------------------------|----------------|
//! | SoT pos / rot / scale          | `WorldTransformGpu` |
//! | Staging pos / rot / scale      | `WorldTransformGpu` (this file) |
//! | Dirty bitmask pos / rot / scl  | `WorldTransformGpu` |
//! | `view_proj_buf`                | `WorldTransformGpu` |
//! | Scatter descriptor sets (3)    | `WorldTransformGpu` |
//! | Scatter secondary CB           | `WorldTransformGpu` |
//! | `mvp_build_set1` (view_proj)   | `WorldTransformGpu` |
//! | `mvp_build_secondary`          | [`crate::camera::RenderCamera`] |
//! | `mvp_build_set0` (SoT/idx/mvp) | [`crate::camera::RenderCamera`] |
//! | Blit secondary + composing primary | [`crate::FrameSlot`] (per swapchain image) |
//!
//! # Synchronization
//!
//! Two independent mechanisms:
//!
//! - **Timeline semaphore** ([`compute_timeline`](Self::compute_timeline)):
//!   signaled by every submission at `PipelineStages::COMPUTE_SHADER`
//!   stage end (covers both scatter and mvp_build). The host waits on the
//!   *previous* signaled value before mutating any of the shared staging /
//!   dirty / view_proj buffers for the next frame. Initial value `0` is
//!   pre-signaled, so the first frame's wait is a no-op.
//! - **Per-image fence** (existing, in [`crate::swapchain`]): gates re-
//!   submission of the per-image primary CB and reuse of the swapchain
//!   image. Independent of the timeline; both are part of the same
//!   `vkQueueSubmit2`.
//!
//! Both must be in place. Don't try to fold one into the other.
//!
//! # Invalidation
//!
//! [`WorldTransformGpu::ensure_capacity`] grows the SoT and staging
//! buffers geometrically (≥ 2×). When that fires, the scatter descriptor
//! sets and the scatter secondary are rebuilt internally; every
//! [`crate::camera::RenderCamera`]'s `mvp_build_set0` and
//! `mvp_build_secondary` must be re-allocated for the same reason
//! (`mvp_build_set0` references the SoT buffers; the secondary captures
//! the new `mvp_build_set1` which references the new `view_proj_buf`),
//! and every [`crate::FrameSlot`]'s primary CB must be re-recorded
//! because it captures `scatter_secondary` and the dirty buffers it fills.

use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
        PrimaryAutoCommandBuffer, SecondaryAutoCommandBuffer,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
        layout::DescriptorSetLayout,
    },
    device::Device,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    sync::semaphore::{Semaphore, SemaphoreCreateInfo, SemaphoreType, SemaphoreWaitInfo},
};

use crate::shaders;

/// One `vec4` per entity slot, in either staging (host-visible) or SoT
/// (device-local) form. Layout matches GLSL `vec4` in std430.
pub type ComponentSlot = [f32; 4];

/// Number of `u32` words needed to bitmask `entity_capacity` slots.
#[inline]
pub fn dirty_word_count(entity_capacity: usize) -> usize {
    entity_capacity.div_ceil(32).max(1)
}

/// World-scoped GPU transform state. See module-level docs for the full
/// ownership table; in short, this owns the SoT buffers, the **shared**
/// per-frame staging mirrors, the scatter compute machinery (pipeline,
/// descriptor sets, secondary CB), the per-frame `view_proj` uniform +
/// `mvp_build_set1`, and the timeline semaphore that synchronizes host
/// writes to the shared staging against the GPU's compute work.
pub struct WorldTransformGpu {
    // ── SoT (device-local) ────────────────────────────────────────
    /// Position SoT — `(x, y, z, _)` per slot.
    sot_positions: Subbuffer<[ComponentSlot]>,
    /// Rotation SoT — quaternion `(x, y, z, w)` per slot.
    sot_rotations: Subbuffer<[ComponentSlot]>,
    /// Scale SoT — `(x, y, z, _)` per slot.
    sot_scales:    Subbuffer<[ComponentSlot]>,

    /// Currently-allocated SoT slot count (== capacity of all three SoT
    /// buffers AND all three staging buffers). Always ≥ 1. Grows
    /// geometrically; never shrinks.
    entity_capacity: usize,

    // ── Shared per-frame host-visible staging ─────────────────────
    /// Host-staged position values (`vec4` per entity slot, `.w` unused).
    /// Sized to `entity_capacity`. Written by the CPU each frame after
    /// host-waiting on `compute_timeline`; consumed by `scatter_secondary`.
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
    /// **Lifecycle:** zeroed once at construction and thereafter cleared
    /// by a `vkCmdFillBuffer(0)` recorded inside each FrameSlot's primary
    /// CB immediately after the scatter consumes it. Because the staging
    /// + dirty buffers are now shared across frames, the host wait on
    /// `compute_timeline` (covering the previous frame's
    /// `COMPUTE_SHADER` stage) guarantees that the GPU clear has fully
    /// landed before the host writes the next frame's bits.
    staging_dirty_pos: Subbuffer<[u32]>,
    staging_dirty_rot: Subbuffer<[u32]>,
    staging_dirty_scl: Subbuffer<[u32]>,

    /// Host-mapped storage buffer carrying a **ring** of `view_proj`
    /// matrices (one slot per swapchain image, indexed by
    /// `pc.view_proj_idx` in `mvp_build_cs`). Sized to
    /// `view_proj_ring_size`.
    ///
    /// Decoupled from the timeline-semaphore wait that gates scatter
    /// staging: each slot is reused only when the per-image fence for
    /// the corresponding swapchain image has been signaled (i.e. the
    /// previous frame on that image — including its mvp_build read of
    /// that ring slot — is fully done). This is what lets the scatter
    /// timeline wait fire at end of `TRANSFER` (scatter+fill) instead of
    /// end of `COMPUTE_SHADER` (scatter+fill+mvp_build).
    view_proj_buf:     Subbuffer<[[f32; 16]]>,

    /// Number of `mat4` slots in `view_proj_buf` (== current swapchain
    /// image count). Set at construction and updated by
    /// [`Self::resize_view_proj_ring`] on swapchain recreate.
    view_proj_ring_size: usize,

    // ── Shared compute descriptor sets ────────────────────────────
    /// Scatter set 0 for the position component: (dirty, staging_pos, sot_pos).
    /// Captured by buffer handle, so re-allocated whenever staging or SoT
    /// is re-allocated (i.e. `ensure_capacity` grows).
    scatter_set_pos:   Arc<DescriptorSet>,
    /// Scatter set 0 for the rotation component.
    scatter_set_rot:   Arc<DescriptorSet>,
    /// Scatter set 0 for the scale component.
    scatter_set_scl:   Arc<DescriptorSet>,
    /// MVP-build set 1 — binds the entire `view_proj_buf` ring as a
    /// storage buffer. Shared by every camera's mvp_build_secondaries
    /// (one secondary per ring slot, each pushing its own
    /// `view_proj_idx`).
    mvp_build_set1:    Arc<DescriptorSet>,

    // ── Shared scatter secondary CB ─────────────────────────────
    /// Compute secondary: three scatter dispatches (pos, rot, scale).
    /// Re-recorded by `ensure_capacity` because both the dispatch count
    /// (entity-capacity-sized) and the descriptor sets it captures change.
    scatter_secondary: Arc<SecondaryAutoCommandBuffer>,

    /// **Shared scatter primary CB** — the CB submitted as batch 0 of
    /// each frame's `vkQueueSubmit2`. Contains:
    ///
    /// 1. `execute(scatter_secondary)` — the three scatter dispatches
    ///    that read shared staging+dirty and write SoT.
    /// 2. Three `vkCmdFillBuffer(0)` clears that re-zero the shared dirty
    ///    bitmasks for the next frame's host write.
    /// 3. A trailing `vkCmdPipelineBarrier` with `SHADER_WRITE →
    ///    SHADER_READ` on `COMPUTE_SHADER` stage — establishes the
    ///    GPU-side memory dependency on SoT writes for the *next*
    ///    submit's mvp_build dispatch (which reads SoT). Same-queue
    ///    submission ordering only gives execution dependency, not
    ///    memory visibility, and we use `submit_unchecked` so vulkano's
    ///    cross-CB resource tracking does not insert this barrier for
    ///    us. Without it, mvp_build can legally see stale SoT data.
    ///
    /// The host-side compute timeline semaphore is signaled at
    /// `TRANSFER` stage end of this CB by the per-frame submission — see
    /// `RenderApp::about_to_wait`.
    scatter_primary:   Arc<PrimaryAutoCommandBuffer>,

    // ── Sync primitive (ADR-0003) ─────────────────────────────────
    /// Timeline semaphore signaled at `PipelineStages::ALL_TRANSFER`
    /// stage end of the per-frame **scatter primary** submission (covers
    /// both the scatter dispatches and the trailing `vkCmdFillBuffer(0)`
    /// clears on the dirty bitmasks). The host waits on the previous
    /// frame's signaled value before mutating any of the shared staging /
    /// dirty buffers. Initial value 0 is pre-signaled, so the first
    /// frame's wait is a no-op.
    ///
    /// Note: `view_proj_buf` is **not** gated by this semaphore — it's
    /// a per-image-fence-protected ring (see `view_proj_buf` docs).
    /// Decoupling view_proj from the scatter timeline is what lets the
    /// signal stage be `TRANSFER` (after scatter+fill) instead of
    /// `COMPUTE_SHADER` end (which would also wait for mvp_build to
    /// finish reading view_proj).
    ///
    /// See [`Self::host_wait_for_previous_compute`] /
    /// [`Self::next_compute_signal`].
    compute_timeline: Arc<Semaphore>,

    /// Value the **next** submission will signal `compute_timeline` to.
    /// Starts at `1` (initial semaphore value is `0`); incremented each
    /// time `next_compute_signal()` is called. The previous-frame wait
    /// value is therefore `next_compute_signal_value - 1` — that's what
    /// `host_wait_for_previous_compute` waits on.
    ///
    /// Monotonic across swapchain recreation: do **not** reset this on
    /// resize; the timeline semaphore's value is preserved by Vulkan
    /// across submissions to the same queue.
    next_compute_signal_value: u64,

    // ── Pipelines ─────────────────────────────────────────────────
    /// Scatter compute pipeline — see [`shaders::scatter_cs`]. One pipeline
    /// shared by the per-component scatter dispatches.
    scatter_pipeline:   Arc<ComputePipeline>,
    /// MVP-build compute pipeline — see [`shaders::mvp_build_cs`].
    mvp_build_pipeline: Arc<ComputePipeline>,

    // ── Stash for re-allocation ───────────────────────────────────
    /// Held so `ensure_capacity` can rebuild scatter descriptor sets
    /// without plumbing the allocator through every call site.
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    /// Held so `ensure_capacity` can re-record the scatter secondary.
    cb_allocator:             Arc<StandardCommandBufferAllocator>,
    /// Captured at construction; needed for the secondary builder.
    queue_family_index:       u32,
}

impl WorldTransformGpu {
    /// Build everything: SoT buffers, shared staging triple, dirty + view_proj
    /// buffers, both compute pipelines, scatter/mvp_build descriptor sets,
    /// the scatter secondary CB, and the compute timeline semaphore.
    pub fn new(
        device:                   Arc<Device>,
        memory_allocator:         &Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        cb_allocator:             &Arc<StandardCommandBufferAllocator>,
        queue_family_index:       u32,
        entity_capacity:          usize,
        view_proj_ring_size:      usize,
    ) -> Self {
        let cap = entity_capacity.max(1);
        let ring = view_proj_ring_size.max(1);

        let (sot_positions, sot_rotations, sot_scales) =
            allocate_sot_buffers(memory_allocator, cap);

        let scatter_pipeline   = build_scatter_pipeline(device.clone());
        let mvp_build_pipeline = build_mvp_build_pipeline(device.clone());

        let (
            staging_positions,
            staging_rotations,
            staging_scales,
            staging_dirty_pos,
            staging_dirty_rot,
            staging_dirty_scl,
            view_proj_buf,
        ) = allocate_staging(memory_allocator, cap, ring);

        let (scatter_set_pos, scatter_set_rot, scatter_set_scl) = build_scatter_sets(
            descriptor_set_allocator,
            scatter_pipeline.layout().set_layouts()[0].clone(),
            &staging_positions,
            &staging_rotations,
            &staging_scales,
            &staging_dirty_pos,
            &staging_dirty_rot,
            &staging_dirty_scl,
            &sot_positions,
            &sot_rotations,
            &sot_scales,
        );
        let mvp_build_set1 = build_mvp_build_set1(
            descriptor_set_allocator,
            mvp_build_pipeline.layout().set_layouts()[1].clone(),
            &view_proj_buf,
        );

        let scatter_secondary = record_scatter_secondary(
            cb_allocator,
            queue_family_index,
            &scatter_pipeline,
            &scatter_set_pos,
            &scatter_set_rot,
            &scatter_set_scl,
            cap,
        );

        let scatter_primary = record_scatter_primary(
            cb_allocator,
            queue_family_index,
            &scatter_secondary,
            &staging_dirty_pos,
            &staging_dirty_rot,
            &staging_dirty_scl,
        );

        // Timeline semaphore. Initial value 0 is "already signaled" for
        // the first wait. Vulkano-util enables Vulkan 1.2+ which has
        // timeline_semaphore in core; we still must enable the feature
        // explicitly in the device features (see `lib.rs`).
        let compute_timeline = Arc::new(
            Semaphore::new(
                device,
                SemaphoreCreateInfo {
                    semaphore_type: SemaphoreType::Timeline,
                    initial_value:  0,
                    ..Default::default()
                },
            )
            .expect("create compute timeline semaphore"),
        );

        Self {
            sot_positions,
            sot_rotations,
            sot_scales,
            entity_capacity:    cap,

            staging_positions,
            staging_rotations,
            staging_scales,
            staging_dirty_pos,
            staging_dirty_rot,
            staging_dirty_scl,
            view_proj_buf,
            view_proj_ring_size: ring,

            scatter_set_pos,
            scatter_set_rot,
            scatter_set_scl,
            mvp_build_set1,
            scatter_secondary,
            scatter_primary,

            compute_timeline,
            next_compute_signal_value: 1,

            scatter_pipeline,
            mvp_build_pipeline,

            descriptor_set_allocator: descriptor_set_allocator.clone(),
            cb_allocator:             cb_allocator.clone(),
            queue_family_index,
        }
    }

    /// Ensure the SoT + staging buffers have at least `needed` slots.
    /// Returns `true` if anything was re-allocated (in which case every
    /// dependent FrameSlot primary CB and every camera's `mvp_build_set0`
    /// + `mvp_build_secondary` must be rebuilt — they captured the old
    /// buffer / descriptor-set handles).
    ///
    /// Geometric growth (≥ 2× current) keeps amortized cost O(1) per
    /// added entity. Never shrinks.
    ///
    /// **Sync caveat:** the caller MUST have already host-waited on
    /// `host_wait_for_previous_compute()` for this frame, so the GPU is
    /// no longer reading the old staging / dirty buffers. The buffers
    /// themselves are dropped here; if the GPU were still using them,
    /// vulkano's resource tracking would catch it, but the wait avoids
    /// the panic in the first place.
    pub fn ensure_capacity(
        &mut self,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        needed:           usize,
    ) -> bool {
        if needed <= self.entity_capacity {
            return false;
        }
        let new_cap = needed
            .max(self.entity_capacity.saturating_mul(2))
            .max(1);

        // SoT.
        let (pos, rot, scl) = allocate_sot_buffers(memory_allocator, new_cap);
        self.sot_positions   = pos;
        self.sot_rotations   = rot;
        self.sot_scales      = scl;

        // Staging triple + dirty + view_proj.
        let (
            staging_positions,
            staging_rotations,
            staging_scales,
            staging_dirty_pos,
            staging_dirty_rot,
            staging_dirty_scl,
            view_proj_buf,
        ) = allocate_staging(memory_allocator, new_cap, self.view_proj_ring_size);
        self.staging_positions = staging_positions;
        self.staging_rotations = staging_rotations;
        self.staging_scales    = staging_scales;
        self.staging_dirty_pos = staging_dirty_pos;
        self.staging_dirty_rot = staging_dirty_rot;
        self.staging_dirty_scl = staging_dirty_scl;
        self.view_proj_buf     = view_proj_buf;

        // Scatter sets capture the new staging + SoT handles.
        let (sp, sr, ss) = build_scatter_sets(
            &self.descriptor_set_allocator,
            self.scatter_pipeline.layout().set_layouts()[0].clone(),
            &self.staging_positions,
            &self.staging_rotations,
            &self.staging_scales,
            &self.staging_dirty_pos,
            &self.staging_dirty_rot,
            &self.staging_dirty_scl,
            &self.sot_positions,
            &self.sot_rotations,
            &self.sot_scales,
        );
        self.scatter_set_pos = sp;
        self.scatter_set_rot = sr;
        self.scatter_set_scl = ss;

        // mvp_build_set1 captures the new view_proj_buf.
        self.mvp_build_set1 = build_mvp_build_set1(
            &self.descriptor_set_allocator,
            self.mvp_build_pipeline.layout().set_layouts()[1].clone(),
            &self.view_proj_buf,
        );

        // Scatter secondary captures the new descriptor sets and the new
        // dispatch count.
        self.scatter_secondary = record_scatter_secondary(
            &self.cb_allocator,
            self.queue_family_index,
            &self.scatter_pipeline,
            &self.scatter_set_pos,
            &self.scatter_set_rot,
            &self.scatter_set_scl,
            new_cap,
        );

        // Scatter primary captures the new scatter_secondary and the new
        // dirty buffers (its `fill_buffer`s target them).
        self.scatter_primary = record_scatter_primary(
            &self.cb_allocator,
            self.queue_family_index,
            &self.scatter_secondary,
            &self.staging_dirty_pos,
            &self.staging_dirty_rot,
            &self.staging_dirty_scl,
        );

        self.entity_capacity = new_cap;
        true
    }

    /// Re-allocate `view_proj_buf` and `mvp_build_set1` to a new ring
    /// size. Called from the swapchain-recreate path when the swapchain
    /// image count changes (the ring is sized to the swapchain's image
    /// count so each per-image FrameSlot owns its own ring slot, gated
    /// by that image's per-image fence).
    ///
    /// Returns `true` if the ring was actually resized. When `true`,
    /// every camera's `mvp_build_secondary` array must be re-recorded
    /// (each captures `mvp_build_set1`, which now binds the new ring
    /// buffer; ring length also changes the number of secondaries each
    /// camera needs).
    ///
    /// Does not need a host wait — the caller is expected to have
    /// already drained all in-flight work via per-image fences (the
    /// swapchain renderer does this in `recreate`).
    pub fn resize_view_proj_ring(
        &mut self,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        new_ring_size:    usize,
    ) -> bool {
        let new_ring = new_ring_size.max(1);
        if new_ring == self.view_proj_ring_size {
            return false;
        }
        self.view_proj_buf = make_host_storage_slice::<[f32; 16]>(
            memory_allocator, new_ring, BufferUsage::empty(), false, false,
        );
        self.mvp_build_set1 = build_mvp_build_set1(
            &self.descriptor_set_allocator,
            self.mvp_build_pipeline.layout().set_layouts()[1].clone(),
            &self.view_proj_buf,
        );
        self.view_proj_ring_size = new_ring;
        true
    }

    // ── Host-side sync API ────────────────────────────────────────

    /// Block the calling thread until the GPU has finished the previous
    /// frame's **scatter primary** — i.e. both the scatter dispatches
    /// (which read shared `staging_<comp>` + `dirty_*`) and the trailing
    /// `vkCmdFillBuffer(0)` clears (which write zero into `dirty_*`).
    /// After this returns it is safe for the host to mutate any of the
    /// shared staging / dirty buffers for the next frame.
    ///
    /// # Why this wait does NOT cover mvp_build
    ///
    /// `view_proj_buf` is the only other shared host-writable resource
    /// and it's the only thing `mvp_build` reads from the shared world
    /// state. It lives as a **ring** (one slot per swapchain image)
    /// gated by the per-image fence, **not** by this semaphore. So the
    /// scatter timeline only needs to gate scatter+fill; mvp_build can
    /// continue to run in parallel with the next frame's CPU prep.
    ///
    /// The signal stage is `ALL_TRANSFER` of the scatter primary submit
    /// (which consists of compute dispatches followed by transfer
    /// fill_buffers). That's the smallest stage mask that covers both
    /// scatter (read of staging) and fill (write of dirty).
    ///
    /// First call (no previous submission) waits on value `0`, which is
    /// the semaphore's initial value — returns immediately.
    pub fn host_wait_for_previous_compute(&self) {
        let prev = self.next_compute_signal_value.saturating_sub(1);
        self.compute_timeline
            .wait(
                SemaphoreWaitInfo {
                    value: prev,
                    ..Default::default()
                },
                None,
            )
            .expect("compute timeline wait failed");
    }

    /// Reserve the value the next submission must signal `compute_timeline`
    /// to. The caller is responsible for actually wiring this value into
    /// the corresponding `SemaphoreSubmitInfo`'s `value` field with stage
    /// mask `PipelineStages::COMPUTE_SHADER`.
    pub fn next_compute_signal(&mut self) -> u64 {
        let v = self.next_compute_signal_value;
        self.next_compute_signal_value += 1;
        v
    }

    /// The compute timeline semaphore. Used by the swapchain renderer to
    /// build the per-frame signal-`SemaphoreSubmitInfo`.
    pub fn compute_timeline(&self) -> &Arc<Semaphore> {
        &self.compute_timeline
    }

    // ── Accessors ─────────────────────────────────────────────────

    pub fn entity_capacity(&self)    -> usize                       { self.entity_capacity }
    pub fn view_proj_ring_size(&self) -> usize                      { self.view_proj_ring_size }
    pub fn sot_positions(&self)      -> &Subbuffer<[ComponentSlot]> { &self.sot_positions }
    pub fn sot_rotations(&self)      -> &Subbuffer<[ComponentSlot]> { &self.sot_rotations }
    pub fn sot_scales(&self)         -> &Subbuffer<[ComponentSlot]> { &self.sot_scales }
    pub fn mvp_build_pipeline(&self) -> &Arc<ComputePipeline>       { &self.mvp_build_pipeline }

    /// Shared scatter secondary, executed once per frame from the
    /// shared scatter primary CB.
    #[allow(dead_code)]
    pub fn scatter_secondary(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.scatter_secondary
    }

    /// Shared scatter **primary** CB — submitted as batch 0 of each
    /// frame's `vkQueueSubmit2`. See the field doc for what it contains
    /// and the cross-CB barrier it carries at the end.
    pub fn scatter_primary(&self) -> &Arc<PrimaryAutoCommandBuffer> {
        &self.scatter_primary
    }

    /// Shared `mvp_build_set1` (view_proj uniform). Captured by every
    /// camera's `mvp_build_secondary`.
    pub fn mvp_build_set1(&self) -> &Arc<DescriptorSet> {
        &self.mvp_build_set1
    }

    /// Shared dirty buffers — referenced by the per-FrameSlot primary CB
    /// for the in-CB `vkCmdFillBuffer(0)` that re-zeroes them after the
    /// scatter consumes them.
    pub fn staging_dirty_pos(&self) -> &Subbuffer<[u32]> { &self.staging_dirty_pos }
    pub fn staging_dirty_rot(&self) -> &Subbuffer<[u32]> { &self.staging_dirty_rot }
    pub fn staging_dirty_scl(&self) -> &Subbuffer<[u32]> { &self.staging_dirty_scl }

    /// Shared host-mapped staging triple. Written by the per-frame
    /// harvest in [`crate::RenderApp::about_to_wait`] after the timeline
    /// wait succeeds.
    pub fn staging_positions(&self) -> &Subbuffer<[ComponentSlot]> { &self.staging_positions }
    pub fn staging_rotations(&self) -> &Subbuffer<[ComponentSlot]> { &self.staging_rotations }
    pub fn staging_scales(&self)    -> &Subbuffer<[ComponentSlot]> { &self.staging_scales }

    /// Shared host-mapped view_proj uniform. Written by the per-frame
    /// harvest immediately after the staging triple.
    pub fn view_proj_buf(&self) -> &Subbuffer<[[f32; 16]]> { &self.view_proj_buf }

    /// Convenience: layout of mvp-build set 0 (per-camera SoT/idx/mvp).
    pub fn mvp_build_set0_layout(&self) -> &Arc<DescriptorSetLayout> {
        &self.mvp_build_pipeline.layout().set_layouts()[0]
    }
}

// ─────────────────────────────────────────────────────────────────────
// Allocation helpers
// ─────────────────────────────────────────────────────────────────────

fn allocate_sot_buffers(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    capacity:         usize,
) -> (
    Subbuffer<[ComponentSlot]>,
    Subbuffer<[ComponentSlot]>,
    Subbuffer<[ComponentSlot]>,
) {
    let make = || -> Subbuffer<[ComponentSlot]> {
        Buffer::new_slice::<ComponentSlot>(
            memory_allocator.clone(),
            BufferCreateInfo {
                // STORAGE_BUFFER: read/written by the scatter compute, read
                // by the mvp-build compute. (No TRANSFER_DST: the scatter
                // compute is the only writer; we never `vkCmdCopyBuffer`
                // into these.)
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            capacity as u64,
        )
        .expect("Failed to allocate SoT buffer")
    };
    (make(), make(), make())
}

/// Allocate the shared staging triple (positions / rotations / scales)
/// + the three dirty bitmasks + the single-mat4 view_proj. Memory-type
/// rationale:
///
/// * Staging triple — `PREFER_DEVICE | HOST_RANDOM_ACCESS`. BAR / ReBAR
///   memory so the scatter compute reads at full VRAM bandwidth (falls
///   back to plain host-visible on systems without BAR). Cached host-
///   visible (`HOST_RANDOM_ACCESS`) so the multi-threaded staging-write
///   path can sparse-write disjoint chunks from many cores without WC-
///   buffer flush penalties.
/// * Dirty bitmasks — `PREFER_HOST | HOST_RANDOM_ACCESS`. Tiny (a few KB
///   even at N=1M), not worth BAR heap pressure; cached host-visible to
///   match the parallel writer pattern.
/// * `view_proj` — `PREFER_HOST | HOST_SEQUENTIAL_WRITE`. 64 bytes, one
///   writer per frame, fully sequential. WC is fine.
///
/// Dirty buffers also include `TRANSFER_DST` so the GPU can `vkCmdFillBuffer(0)`
/// them after the scatter consumes them.
///
/// Dirty buffers are zero-initialised on the host before return: the very
/// first scatter dispatch reads them before any GPU clear has run.
fn allocate_staging(
    memory_allocator:    &Arc<StandardMemoryAllocator>,
    entity_capacity:     usize,
    view_proj_ring_size: usize,
) -> (
    Subbuffer<[ComponentSlot]>,
    Subbuffer<[ComponentSlot]>,
    Subbuffer<[ComponentSlot]>,
    Subbuffer<[u32]>,
    Subbuffer<[u32]>,
    Subbuffer<[u32]>,
    Subbuffer<[[f32; 16]]>,
) {
    let pos = make_host_storage_slice::<ComponentSlot>(
        memory_allocator, entity_capacity, BufferUsage::empty(), true, true,
    );
    let rot = make_host_storage_slice::<ComponentSlot>(
        memory_allocator, entity_capacity, BufferUsage::empty(), true, true,
    );
    let scl = make_host_storage_slice::<ComponentSlot>(
        memory_allocator, entity_capacity, BufferUsage::empty(), true, true,
    );

    let dirty_words = dirty_word_count(entity_capacity);
    let dp = make_host_storage_slice::<u32>(
        memory_allocator, dirty_words, BufferUsage::TRANSFER_DST, false, true,
    );
    let dr = make_host_storage_slice::<u32>(
        memory_allocator, dirty_words, BufferUsage::TRANSFER_DST, false, true,
    );
    let ds = make_host_storage_slice::<u32>(
        memory_allocator, dirty_words, BufferUsage::TRANSFER_DST, false, true,
    );

    // One-time CPU zero-init of the dirty buffers. `Buffer::new_slice`
    // leaves contents undefined; the first scatter dispatch reads these
    // words before any GPU clear has run, so we must guarantee they're
    // zero up front. Subsequent frames rely on the in-CB `fill_buffer`
    // to keep them zero between scatter consumption and the next host
    // write.
    for buf in [&dp, &dr, &ds] {
        let mut w = buf.write().expect("zero-init staging_dirty_*.write");
        for word in w.iter_mut() {
            *word = 0;
        }
    }

    let vp = make_host_storage_slice::<[f32; 16]>(
        memory_allocator, view_proj_ring_size.max(1), BufferUsage::empty(), false, false,
    );

    (pos, rot, scl, dp, dr, ds, vp)
}

fn build_scatter_sets(
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    scatter_layout:           Arc<DescriptorSetLayout>,
    staging_positions:        &Subbuffer<[ComponentSlot]>,
    staging_rotations:        &Subbuffer<[ComponentSlot]>,
    staging_scales:           &Subbuffer<[ComponentSlot]>,
    staging_dirty_pos:        &Subbuffer<[u32]>,
    staging_dirty_rot:        &Subbuffer<[u32]>,
    staging_dirty_scl:        &Subbuffer<[u32]>,
    sot_positions:            &Subbuffer<[ComponentSlot]>,
    sot_rotations:            &Subbuffer<[ComponentSlot]>,
    sot_scales:               &Subbuffer<[ComponentSlot]>,
) -> (Arc<DescriptorSet>, Arc<DescriptorSet>, Arc<DescriptorSet>) {
    let pos = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        scatter_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, staging_dirty_pos.clone()),
            WriteDescriptorSet::buffer(1, staging_positions.clone()),
            WriteDescriptorSet::buffer(2, sot_positions.clone()),
        ],
        [],
    ).expect("scatter_set_pos");
    let rot = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        scatter_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, staging_dirty_rot.clone()),
            WriteDescriptorSet::buffer(1, staging_rotations.clone()),
            WriteDescriptorSet::buffer(2, sot_rotations.clone()),
        ],
        [],
    ).expect("scatter_set_rot");
    let scl = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        scatter_layout,
        [
            WriteDescriptorSet::buffer(0, staging_dirty_scl.clone()),
            WriteDescriptorSet::buffer(1, staging_scales.clone()),
            WriteDescriptorSet::buffer(2, sot_scales.clone()),
        ],
        [],
    ).expect("scatter_set_scl");
    (pos, rot, scl)
}

fn build_mvp_build_set1(
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    layout:                   Arc<DescriptorSetLayout>,
    view_proj_buf:            &Subbuffer<[[f32; 16]]>,
) -> Arc<DescriptorSet> {
    DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout,
        [WriteDescriptorSet::buffer(0, view_proj_buf.clone())],
        [],
    ).expect("mvp_build_set1")
}

fn record_scatter_secondary(
    cb_allocator:       &Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
    scatter_pipeline:   &Arc<ComputePipeline>,
    scatter_set_pos:    &Arc<DescriptorSet>,
    scatter_set_rot:    &Arc<DescriptorSet>,
    scatter_set_scl:    &Arc<DescriptorSet>,
    entity_capacity:    usize,
) -> Arc<SecondaryAutoCommandBuffer> {
    let layout = scatter_pipeline.layout().clone();
    let groups = (entity_capacity as u32).div_ceil(64).max(1);
    let pc     = shaders::scatter_cs::PC { entity_count: entity_capacity as u32 };

    let mut builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        // Now world-scoped — referenced only by the shared `scatter_primary`,
        // which is submitted at most once per frame and serialized by the
        // host's timeline-semaphore wait. MultipleSubmit suffices.
        CommandBufferUsage::MultipleSubmit,
        CommandBufferInheritanceInfo::default(),
    ).expect("scatter secondary builder");

    builder
        .bind_pipeline_compute(scatter_pipeline.clone()).expect("bind scatter pipeline")
        .push_constants(layout.clone(), 0, pc).expect("push scatter pc");
    for set in [scatter_set_pos, scatter_set_rot, scatter_set_scl] {
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                layout.clone(),
                0,
                set.clone(),
            ).expect("bind scatter set");
        // Safety: dispatch counts derived from `entity_capacity`; shader
        // bounds-checks against the push-constant `entity_count`.
        unsafe {
            builder.dispatch([groups, 1, 1]).expect("dispatch scatter");
        }
    }
    builder.build().expect("build scatter secondary")
}

/// Record the **shared scatter primary CB** — the CB submitted as batch 0
/// of each frame's `vkQueueSubmit2`. Contents:
///
/// 1. `execute_commands(scatter_secondary)` — the three scatter dispatches.
/// 2. Three `fill_buffer(0)` calls that re-zero the shared dirty bitmasks.
///    Vulkano auto-sync inserts a `SHADER_READ → TRANSFER_WRITE` barrier on
///    each dirty buffer between the scatter dispatch (which read the
///    bits) and this clear.
///
/// **Cross-submit memory dependency on SoT** is supplied by the
/// per-frame submission's *batch 1* (the FrameSlot primary), which adds
/// a wait on `compute_timeline` at `COMPUTE_SHADER` stage for the value
/// signaled at end of this batch's `ALL_TRANSFER` stage. A semaphore
/// signal/wait pair across submits establishes both execution and memory
/// dependency per Vulkan spec, so mvp_build in batch 1 sees the SoT
/// writes from scatter in batch 0 without any manual
/// `vkCmdPipelineBarrier`. (We use `submit_unchecked`, which bypasses
/// vulkano's resource tracking, so we couldn't rely on auto-sync to
/// insert a cross-CB barrier here even if `AutoCommandBufferBuilder`
/// exposed a manual `pipeline_barrier` method, which it doesn't.)
///
/// MultipleSubmit is fine because the host-side timeline wait
/// (`host_wait_for_previous_compute`) serializes consecutive submissions
/// of this CB.
fn record_scatter_primary(
    cb_allocator:       &Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
    scatter_secondary:  &Arc<SecondaryAutoCommandBuffer>,
    staging_dirty_pos:  &Subbuffer<[u32]>,
    staging_dirty_rot:  &Subbuffer<[u32]>,
    staging_dirty_scl:  &Subbuffer<[u32]>,
) -> Arc<PrimaryAutoCommandBuffer> {
    let mut builder = AutoCommandBufferBuilder::primary(
        cb_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::MultipleSubmit,
    ).expect("scatter primary builder");

    builder
        .execute_commands(scatter_secondary.clone()).expect("execute scatter secondary");

    builder
        .fill_buffer(staging_dirty_pos.clone().reinterpret::<[u32]>(), 0).expect("fill staging_dirty_pos")
        .fill_buffer(staging_dirty_rot.clone().reinterpret::<[u32]>(), 0).expect("fill staging_dirty_rot")
        .fill_buffer(staging_dirty_scl.clone().reinterpret::<[u32]>(), 0).expect("fill staging_dirty_scl");

    builder.build().expect("build scatter primary")
}

fn build_scatter_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs    = shaders::scatter_cs::load(device.clone()).expect("scatter_cs load failed");
    let entry = cs.entry_point("main").expect("scatter_cs entry point");
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(&stage))
            .into_pipeline_layout_create_info(device.clone())
            .expect("scatter pipeline layout info"),
    )
    .expect("scatter pipeline layout");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("scatter ComputePipeline::new")
}

fn build_mvp_build_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs    = shaders::mvp_build_cs::load(device.clone()).expect("mvp_build_cs load failed");
    let entry = cs.entry_point("main").expect("mvp_build_cs entry point");
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(&stage))
            .into_pipeline_layout_create_info(device.clone())
            .expect("mvp_build pipeline layout info"),
    )
    .expect("mvp_build pipeline layout");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("mvp_build ComputePipeline::new")
}

/// Allocate a host-visible STORAGE_BUFFER slice of `count` elements. See
/// the doc comment on the [`crate`]-level helper for the parameter
/// rationale; this is the same function but kept here so
/// [`WorldTransformGpu`] is self-contained.
fn make_host_storage_slice<T>(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    count:            usize,
    extra_usage:      BufferUsage,
    prefer_device:    bool,
    random_access:    bool,
) -> Subbuffer<[T]>
where
    T: vulkano::buffer::BufferContents,
{
    let host_access = if random_access {
        MemoryTypeFilter::HOST_RANDOM_ACCESS
    } else {
        MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
    };
    let placement = if prefer_device {
        MemoryTypeFilter::PREFER_DEVICE
    } else {
        MemoryTypeFilter::PREFER_HOST
    };
    let memory_type_filter = placement | host_access;
    Buffer::new_slice::<T>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | extra_usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter,
            ..Default::default()
        },
        count.max(1) as u64,
    )
    .expect("Failed to allocate host storage slice")
}
