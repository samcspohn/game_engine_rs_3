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
//! | SoT pos / rot / scale / parent | `WorldTransformGpu` |
//! | Staging pos / rot / scale      | `WorldTransformGpu` (this file) |
//! | Parent-update stream staging   | `WorldTransformGpu` (count-in-buffer) |
//! | Dirty bitmask pos / rot / scl  | `WorldTransformGpu` |
//! | `view_proj_buf`                | `WorldTransformGpu` |
//! | Scatter descriptor sets (3)    | `WorldTransformGpu` |
//! | Scatter secondary CB           | `WorldTransformGpu` |
//! | `mvp_build_secondary`          | [`crate::camera::RenderCamera`] |
//! | `mvp_build_set0` (SoT/idx/mvp) | [`crate::camera::RenderCamera`] |
//! | `occlusion_set` (view_proj history + Hi-Z) | [`crate::camera::RenderCamera`] |
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
//! (`mvp_build_set0` references the SoT buffers), and every
//! [`crate::FrameSlot`]'s primary CB must be re-recorded because it
//! captures `scatter_secondary` and the dirty buffers it fills.

use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferInheritanceInfo, CommandBufferUsage, CopyBufferInfo, DispatchIndirectCommand,
        PrimaryCommandBufferAbstract, SecondaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayout, DescriptorSet,
        WriteDescriptorSet,
    },
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::GpuFuture,
};

use crate::shaders;

/// Sentinel parent id for a root transform (no parent). Matches
/// `engine_core::transform::NO_PARENT` and the `NO_PARENT` constants in
/// `mvp_build.comp` / `parent_scatter.comp`.
pub const NO_PARENT: u32 = u32::MAX;

/// Initial pair capacity of the parent-update staging buffer. Grows
/// geometrically when a frame's drain exceeds it (e.g. a large subscene
/// instantiation), which forces the usual secondary/frame-slot rebuild.
const INITIAL_PARENT_UPDATE_CAPACITY: usize = 1024;

/// One `vec4` per entity slot, in either staging (host-visible) or SoT
/// (device-local) form. Layout matches GLSL `vec4` in std430.
pub type ComponentSlot = [f32; 4];

/// Number of `u32` words needed to bitmask `entity_capacity` slots.
#[inline]
pub fn dirty_word_count(entity_capacity: usize) -> usize {
    entity_capacity.div_ceil(32).max(1)
}

/// Must match `scatter_prepass.comp`'s `local_size_x`: one prepass
/// workgroup scans this many candidate dirty-bitmask words.
const PREPASS_WORDS_PER_WORKGROUP: u32 = 64;

/// Packs a unit quaternion as 4 IEEE 754 binary16 (`f16`) components into
/// two `u32`s — `[x, y]` then `[z, w]`, each pair combined exactly like
/// GLSL's `packHalf2x16` (first component in the low 16 bits, second in
/// the high 16 bits), so the GPU side decodes with the matching built-in
/// `unpackHalf2x16` instead of a hand-matched bit layout.
///
/// This replaced an earlier "smallest three" 32-bit packed encoding
/// (quantize 3 of 4 components to N bits each, reconstruct the 4th via
/// `sqrt(1 - a² - b² - c²)`). That scheme broke down on real (deeply
/// nested, glTF-sourced) models: quantization error compounds through
/// `mvp_build_cs`'s parent-chain composition, and a leaf's position error
/// scales with lever-arm length, so a fraction of a degree at a root joint
/// became a visible offset — and clipping — many levels down. Widening the
/// 32-bit budget within the smallest-three scheme (e.g. giving 2 of the 3
/// stored components 11 bits instead of 10) doesn't actually buy much: 2
/// of the 32 bits go to the dropped-component index, so there's very
/// little slack to redistribute, and reconstructing the 4th component from
/// only 3 lossy ones amplifies whatever error remains. Storing all 4
/// components directly, each independently quantized via the standard
/// (well-tested, IEEE-754-correct) half-float format, sidesteps both
/// problems: no reconstruction step to amplify error, and a wider,
/// well-understood 11-bit mantissa (10 explicit + implicit leading bit)
/// per component instead of a hand-rolled fixed-point scheme.
///
/// 8 bytes/entity — double the smallest-three scheme's 4 bytes, but still
/// a 33% cut from the original 12-byte (3×f32 Euler angle) staging this
/// whole compression effort started from. `TransformHierarchy` keeps
/// full-precision `Quat`s as the simulation source of truth regardless, so
/// this is a pure rendering-precision/bandwidth tradeoff.
#[inline]
pub fn pack_quat_half(q: glam::Quat) -> [u32; 2] {
    use half::f16;
    let pack2 = |a: f32, b: f32| -> u32 {
        (f16::from_f32(a).to_bits() as u32) | ((f16::from_f32(b).to_bits() as u32) << 16)
    };
    [pack2(q.x, q.y), pack2(q.z, q.w)]
}

/// World-scoped GPU transform state. See module-level docs for the full
/// ownership table; in short, this owns the SoT buffers, the **shared**
/// per-frame staging mirrors, the scatter compute machinery (pipeline,
/// descriptor sets, secondary CB), the per-frame `view_proj` uniform, and
/// the timeline semaphore that synchronizes host writes to the shared
/// staging against the GPU's compute work.
pub struct WorldTransformGpu {
    // ── SoT (device-local) ────────────────────────────────────────
    /// Position SoT — `(x, y, z, _)` per slot.
    sot_positions: Subbuffer<[ComponentSlot]>,
    /// Rotation SoT — quaternion `(x, y, z, w)` per slot.
    sot_rotations: Subbuffer<[ComponentSlot]>,
    /// Scale SoT — `(x, y, z, _)` per slot.
    sot_scales: Subbuffer<[ComponentSlot]>,
    /// **`view_proj` SoT** — a single-mat4 device-local buffer that
    /// `mvp_build_cs` reads via `RenderCamera`'s camera-owned occlusion
    /// set. Promoted from `staging_view_proj` by the `vkCmdCopyBuffer`
    /// recorded inside `scatter_primary`. This makes `view_proj` follow
    /// the same staging→SoT paradigm as TRS — mvp_build reads only stable
    /// SoT buffers, never host-visible staging. Also copied into every
    /// `RenderCamera`'s `prev_view_proj` at the end of each frame (dual-
    /// pass occlusion culling), which is why this buffer needs
    /// `TRANSFER_SRC` in addition to `TRANSFER_DST`.
    sot_view_proj: Subbuffer<[[f32; 16]]>,

    /// **Parents SoT** — one parent transform id per entity slot
    /// ([`NO_PARENT`] = root), the fourth member of the SoT family. Read
    /// by `mvp_build_cs`'s parent-chain walk; updated in-CB by the parent
    /// scatter dispatch folded into `scatter_secondary`. Sentinel-filled
    /// at allocation; **copy-preserved** across `ensure_capacity` grows
    /// (unlike TRS, which is re-uploaded via `mark_all_trs` — parents
    /// have no capacity-sized staging mirror to re-upload from).
    sot_parents: Subbuffer<[u32]>,

    /// Currently-allocated SoT slot count (== capacity of all three SoT
    /// buffers AND all three staging buffers). Always ≥ 1. Grows
    /// geometrically; never shrinks.
    entity_capacity: usize,

    // ── Shared per-frame host-visible staging ─────────────────────
    /// Host-staged position values — flat `float[]`, 3 floats `(x, y, z)`
    /// per entity slot (a true scalar array, not a `vec3[]`/`vec4[]`, so
    /// std430 doesn't pad each entry to 16 bytes). Sized to
    /// `3 * entity_capacity`. Written by the CPU each frame after
    /// host-waiting on `compute_timeline`; consumed by `scatter_secondary`.
    staging_positions: Subbuffer<[f32]>,
    /// Host-staged rotation values — flat `float[]`, **two `f32`-sized
    /// slots per entity**, each the bit-reinterpretation of a
    /// `packHalf2x16`-style pair (`pack_quat_half` packs `(x,y)` into slot
    /// 0 and `(z,w)` into slot 1 — see that function's docs). Sized to
    /// `2 * entity_capacity` (not `3 * entity_capacity` — position/scale
    /// stay 3-wide `f32`). `scatter_cs` unpacks it back to a quaternion via
    /// `unpackHalf2x16` when promoting into the (still `vec4`) SoT rotation
    /// buffer.
    staging_rotations: Subbuffer<[f32]>,
    /// Host-staged scale values — flat `float[]`, 3 floats `(x, y, z)`
    /// per slot.
    staging_scales: Subbuffer<[f32]>,

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

    /// **Device-local indirect dispatch args** for the three *real* TRS
    /// scatter dispatches — `[pos, rot, scl]`, one [`DispatchIndirectCommand`]
    /// each. Written on the GPU, every frame, by `scatter_build_args_cs`
    /// (folded into `scatter_secondary`, right after the word-compaction
    /// prepass) from each component's *exact* compacted dirty-word count —
    /// `ceil(word_count / 2)` workgroups (2 compacted words per 64-wide
    /// workgroup). Unlike the old watermark scheme this tracks the true
    /// number of dirty words, not the span between the lowest and highest
    /// one, so entities dirtied far apart across a huge capacity no longer
    /// force a dispatch covering the gap between them. Never touched by the
    /// host. Fixed at three elements; never reallocated.
    trs_dispatch_args: Subbuffer<[DispatchIndirectCommand]>,

    /// **Host-mapped** `[word_offset, word_count]` pair bounding each
    /// component's word-compaction prepass scan range this frame — one
    /// buffer per component (`prepass_bounds_pos/rot/scl`). Written every
    /// frame by [`Self::write_prepass_dispatch_groups`] from the *lowest and
    /// highest* dirty-word watermark the host staging drain folded this
    /// frame (see `lib.rs`'s per-task `local_min_*`/`local_max_*` fold) — so
    /// the prepass itself only scans the span that could possibly contain a
    /// dirty word, not from word 0 every frame. Read by `scatter_prepass_cs`.
    prepass_bounds_pos: Subbuffer<[u32]>,
    prepass_bounds_rot: Subbuffer<[u32]>,
    prepass_bounds_scl: Subbuffer<[u32]>,

    /// **Host-mapped indirect dispatch args** for the three word-compaction
    /// prepass dispatches — `[pos, rot, scl]`. Written every frame by
    /// [`Self::write_prepass_dispatch_groups`] alongside the bounds above:
    /// `ceil(word_count / 64)` workgroups spanning
    /// `[word_offset, word_offset + word_count)`. Zero-workgroup on a quiet
    /// component. Fixed at three elements; never reallocated.
    prepass_dispatch_args: Subbuffer<[DispatchIndirectCommand]>,

    /// **Device-local** compacted dirty-word lists, one per component —
    /// `{count, pad, uvec2 entries[]}` (`.x` = dirty word index, `.y` = that
    /// word's bitmask), matching `parent_scatter.comp`'s count-in-buffer
    /// convention. Filled every frame by `scatter_prepass_cs`; `count` is
    /// reset to 0 by an in-CB `fill_buffer` immediately before the prepass
    /// dispatch reads/writes it (see `record_scatter_secondary`) — never
    /// touched by the host. Sized to `dirty_word_count(entity_capacity)`
    /// entries worst case (every word dirty this frame). Reallocated by
    /// [`Self::ensure_capacity`].
    compact_words_pos: Subbuffer<[u32]>,
    compact_words_rot: Subbuffer<[u32]>,
    compact_words_scl: Subbuffer<[u32]>,

    /// **Host-mapped parent-update stream staging.** Layout (std430,
    /// matching `parent_scatter.comp`): word 0 = live record count, word 1
    /// = pad, then `[transform_id, new_parent]` pairs from word 2. Written
    /// **every** frame by [`Self::write_parent_updates`] (count 0 when
    /// quiet) after the `gpu_signal` wait — the same gate as the TRS
    /// staging, which is what makes a re-parent + local-TRS rewrite land
    /// **atomically in the same frame**. Sized to
    /// `2 + 2 * parent_update_capacity` u32s.
    staging_parent_updates: Subbuffer<[u32]>,

    /// Pair capacity of `staging_parent_updates`. Grown geometrically by
    /// [`Self::ensure_parent_update_capacity`] when a frame's drain
    /// exceeds it; never shrinks.
    parent_update_capacity: usize,

    /// **Host-mapped indirect dispatch args** for the parent-scatter
    /// dispatch: a single [`DispatchIndirectCommand`] written every frame
    /// by [`Self::write_parent_updates`] from the *live* record count
    /// (`ceil(count / 64)` workgroups), instead of always dispatching over
    /// the full `parent_update_capacity`. Quiet frames write `x = 0` — a
    /// zero-workgroup `vkCmdDispatchIndirect` — instead of a handful of
    /// no-op workgroups. Fixed at one element; never reallocated.
    parent_dispatch_args: Subbuffer<[DispatchIndirectCommand]>,

    /// **Host-mapped staging mat4** carrying this frame's `view_proj`.
    /// Treated like TRS staging: read by the `vkCmdCopyBuffer` inside
    /// `scatter_primary` (which copies it into `sot_view_proj`), never
    /// read directly by `mvp_build_cs`. Single slot, no ring — the
    /// scatter timeline gates host writes to it just like it gates TRS
    /// staging.
    view_proj_buf: Subbuffer<[[f32; 16]]>,

    // ── Shared compute descriptor sets ────────────────────────────
    /// Scatter set 0 for the position component: (dirty, staging_pos, sot_pos).
    /// Captured by buffer handle, so re-allocated whenever staging or SoT
    /// is re-allocated (i.e. `ensure_capacity` grows).
    scatter_set_pos: Arc<DescriptorSet>,
    /// Scatter set 0 for the rotation component.
    scatter_set_rot: Arc<DescriptorSet>,
    /// Scatter set 0 for the scale component.
    scatter_set_scl: Arc<DescriptorSet>,
    /// Parent-scatter set 0: (staging_parent_updates, sot_parents).
    /// Re-allocated when either buffer re-allocates (`ensure_capacity` /
    /// `ensure_parent_update_capacity`).
    parent_scatter_set: Arc<DescriptorSet>,

    /// Prepass set 0 for the position component: (prepass_bounds_pos,
    /// staging_dirty_pos, compact_words_pos). Re-allocated whenever any of
    /// those re-allocate (`ensure_capacity`).
    prepass_set_pos: Arc<DescriptorSet>,
    /// Prepass set 0 for the rotation component.
    prepass_set_rot: Arc<DescriptorSet>,
    /// Prepass set 0 for the scale component.
    prepass_set_scl: Arc<DescriptorSet>,
    /// `scatter_build_args_cs` set 0: (compact_words_pos, compact_words_rot,
    /// compact_words_scl, trs_dispatch_args). Re-allocated whenever the
    /// compact-words buffers re-allocate (`ensure_capacity`).
    build_args_set: Arc<DescriptorSet>,

    // ── Shared scatter secondary CB ─────────────────────────────
    /// Compute secondary: three scatter dispatches (pos, rot, scale).
    /// Re-recorded by `ensure_capacity` because both the dispatch count
    /// (entity-capacity-sized) and the descriptor sets it captures change.
    ///
    /// Executed at the **front** of every FrameSlot primary CB, before
    /// `mvp_build_secondary`. Vulkano auto-sync inserts the
    /// `SHADER_WRITE → SHADER_READ` barrier on each SoT buffer between
    /// this scatter dispatch and mvp_build (which binds the same SoT).
    /// The dirty `fill_buffer(0)` clears and the
    /// `staging_view_proj → sot_view_proj` copy are inlined into the
    /// FrameSlot primary right after this secondary executes (see
    /// `build_frame_slot`).
    scatter_secondary: Arc<SecondaryAutoCommandBuffer>,

    // ── Sync primitive (ADR-0003 — GPU-write early-wake) ─────────────
    /// Host-coherent (HOST_RANDOM_ACCESS), single-`u32` buffer that the
    /// GPU `signal_cs` dispatch atomically increments once per frame.
    /// Recorded into the FrameSlot primary CB **right after**
    /// scatter+fill+copy and **before** mvp_build, so its increment
    /// becomes visible to the host the moment every read of host-shared
    /// staging is done — even though the rest of the CB (mvp_build,
    /// render, blit) is still executing.
    ///
    /// The host busy-polls this counter in
    /// [`Self::host_wait_for_previous_compute`] instead of issuing a
    /// kernel-mode `vkWaitSemaphores`. The poll is a single mapped-memory
    /// load + compare; when the GPU is keeping up it succeeds on the
    /// first read. This gives us the same correctness guarantee as the
    /// previous timeline semaphore (host can't overwrite shared staging
    /// the GPU is still reading) at a fraction of the per-frame cost —
    /// crucial at low N where the scene's GPU work is microseconds and
    /// the timeline syscall dominated the frame budget.
    gpu_signal: Subbuffer<[u32]>,

    /// Bound by `signal_secondary`. Set 0, binding 0 = `gpu_signal`.
    /// Held to keep the descriptor set alive for as long as the
    /// secondary CB references it.
    #[allow(dead_code)]
    signal_set: Arc<DescriptorSet>,

    /// Pre-recorded compute secondary CB — single dispatch of `signal_cs`
    /// (1×1×1). Captured by every FrameSlot primary; recorded
    /// `SimultaneousUse` because multiple in-flight FrameSlot primaries
    /// can be executing it concurrently.
    signal_secondary: Arc<SecondaryAutoCommandBuffer>,

    /// Compute pipeline for `signal_cs`. Kept so `ensure_capacity` (which
    /// today doesn't touch the signal path) can re-record the secondary
    /// if any of its inputs ever need to change.
    #[allow(dead_code)]
    signal_pipeline: Arc<ComputePipeline>,

    /// Frame counter — the value the next frame's `signal_cs` dispatch
    /// will bring `gpu_signal` up to. Host increments this in
    /// [`Self::inc_signal_expected`] right after each submit so the
    /// next frame's [`Self::host_wait_for_previous_compute`] knows what
    /// value to wait for.
    ///
    /// `u32` to match the buffer element type. Wraps every ~4 days at
    /// 11K FPS; `wrapping_sub` in the poll handles wraparound
    /// correctly so a long-running session is fine.
    next_signal_expected: u32,

    // ── Pipelines ─────────────────────────────────────────────────
    /// Scatter compute pipeline — see [`shaders::scatter_cs`]. One pipeline
    /// shared by the per-component scatter dispatches.
    scatter_pipeline: Arc<ComputePipeline>,
    /// Word-compaction prepass pipeline — see [`shaders::scatter_prepass_cs`].
    /// One pipeline shared by the per-component prepass dispatches.
    scatter_prepass_pipeline: Arc<ComputePipeline>,
    /// `scatter_build_args_cs` pipeline — converts the three prepasses'
    /// compacted word counts into the real scatter's indirect dispatch args.
    scatter_build_args_pipeline: Arc<ComputePipeline>,
    /// Parent-scatter compute pipeline — see [`shaders::parent_scatter_cs`].
    /// Streamed count-in-buffer dispatch, folded into `scatter_secondary`.
    parent_scatter_pipeline: Arc<ComputePipeline>,
    /// MVP-build compute pipeline — see [`shaders::mvp_build_cs`].
    mvp_build_pipeline: Arc<ComputePipeline>,

    // ── Stash for re-allocation ───────────────────────────────────
    /// Held so `ensure_capacity` can rebuild scatter descriptor sets
    /// without plumbing the allocator through every call site.
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    /// Held so `ensure_capacity` can re-record the scatter secondary.
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    /// Captured at construction; needed for the secondary builder.
    queue_family_index: u32,
    /// Held for the **rare** one-shot fence-waited submits this struct
    /// issues itself: the initial sentinel-fill of `sot_parents` and the
    /// copy-preserving migration on `ensure_capacity` grows. Never touched
    /// on the per-frame path.
    queue: Arc<Queue>,

    /// Dedicated memory allocator for the **staging triple only**.
    /// Kept separate from the main allocator so `mbind` on the staging
    /// pages can never accidentally migrate pages belonging to unrelated
    /// resources that share a `VkDeviceMemory` chunk via vulkano's
    /// suballocation. Every staging allocation goes through this
    /// instance; everything else (SoT, view_proj, gpu_signal) goes
    /// through the main allocator passed by the caller.
    staging_allocator: Arc<StandardMemoryAllocator>,

    /// If `Some(node)`, every staging allocation is `mbind`'d to that
    /// NUMA node after creation, and the residency is verified.
    /// Sourced from the `ENGINE_STAGING_NUMA_NODE` env var at
    /// construction time (parsed once, cached on the struct so
    /// `ensure_capacity` doesn't re-read the environ).
    staging_numa_node: Option<u32>,
}

impl WorldTransformGpu {
    /// Build everything: SoT buffers, shared staging triple, dirty + view_proj
    /// buffers, both compute pipelines, scatter/mvp_build descriptor sets,
    /// the scatter secondary CB, and the compute timeline semaphore.
    pub fn new(
        device: Arc<Device>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        cb_allocator: &Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
        entity_capacity: usize,
    ) -> Self {
        let cap = entity_capacity.max(1);
        let queue_family_index = queue.queue_family_index();

        let (sot_positions, sot_rotations, sot_scales) =
            allocate_sot_buffers(memory_allocator, cap);
        let sot_view_proj = allocate_sot_view_proj(memory_allocator);
        let sot_parents = allocate_sot_parents(memory_allocator, cap);
        // One-shot sentinel fill: every slot starts as a root. Blocking is
        // fine — construction time, nothing in flight.
        fill_u32_oneshot(cb_allocator, &queue, &sot_parents, NO_PARENT);

        let scatter_pipeline = build_scatter_pipeline(device.clone());
        let scatter_prepass_pipeline = build_scatter_prepass_pipeline(device.clone());
        let scatter_build_args_pipeline = build_scatter_build_args_pipeline(device.clone());
        let parent_scatter_pipeline = build_parent_scatter_pipeline(device.clone());
        let mvp_build_pipeline = build_mvp_build_pipeline(device.clone());
        let signal_pipeline = build_signal_pipeline(device.clone());

        // Dedicated allocator for the staging triple. See the comment
        // on `staging_allocator` in the struct definition.
        let staging_allocator: Arc<StandardMemoryAllocator> =
            Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        // Parse the staging-NUMA env var once. Empty / unset = no mbind.
        // Any non-integer value is a hard error (we'd rather fail
        // loudly than silently fall back to "stripe wherever").
        let staging_numa_node: Option<u32> = match std::env::var("ENGINE_STAGING_NUMA_NODE") {
            Ok(s) if !s.is_empty() => Some(
                s.parse::<u32>()
                    .expect("ENGINE_STAGING_NUMA_NODE must be a non-negative integer"),
            ),
            _ => None,
        };

        let (
            staging_positions,
            staging_rotations,
            staging_scales,
            staging_dirty_pos,
            staging_dirty_rot,
            staging_dirty_scl,
            view_proj_buf,
        ) = allocate_staging(&staging_allocator, cap, staging_numa_node);
        // Real-scatter dispatch args are now GPU-written (by
        // `scatter_build_args_cs`) from the exact compacted dirty-word
        // count, so this lives in device-local memory, not staging.
        let trs_dispatch_args = allocate_trs_dispatch_args(memory_allocator);

        let (prepass_bounds_pos, prepass_bounds_rot, prepass_bounds_scl) =
            allocate_prepass_bounds(&staging_allocator);
        let prepass_dispatch_args = allocate_prepass_dispatch_args(&staging_allocator);
        let compact_words_pos = allocate_compact_words(memory_allocator, cap);
        let compact_words_rot = allocate_compact_words(memory_allocator, cap);
        let compact_words_scl = allocate_compact_words(memory_allocator, cap);

        let parent_update_capacity = INITIAL_PARENT_UPDATE_CAPACITY;
        let staging_parent_updates =
            allocate_parent_update_staging(&staging_allocator, parent_update_capacity);
        let parent_dispatch_args = allocate_parent_dispatch_args(&staging_allocator);

        let (scatter_set_pos, scatter_set_rot, scatter_set_scl) = build_scatter_sets(
            descriptor_set_allocator,
            scatter_pipeline.layout().set_layouts()[0].clone(),
            &staging_positions,
            &staging_rotations,
            &staging_scales,
            &compact_words_pos,
            &compact_words_rot,
            &compact_words_scl,
            &sot_positions,
            &sot_rotations,
            &sot_scales,
        );
        let (prepass_set_pos, prepass_set_rot, prepass_set_scl) = build_prepass_sets(
            descriptor_set_allocator,
            scatter_prepass_pipeline.layout().set_layouts()[0].clone(),
            &prepass_bounds_pos,
            &prepass_bounds_rot,
            &prepass_bounds_scl,
            &staging_dirty_pos,
            &staging_dirty_rot,
            &staging_dirty_scl,
            &compact_words_pos,
            &compact_words_rot,
            &compact_words_scl,
        );
        let build_args_set = build_args_build_set(
            descriptor_set_allocator,
            scatter_build_args_pipeline.layout().set_layouts()[0].clone(),
            &compact_words_pos,
            &compact_words_rot,
            &compact_words_scl,
            &trs_dispatch_args,
        );
        let parent_scatter_set = build_parent_scatter_set(
            descriptor_set_allocator,
            parent_scatter_pipeline.layout().set_layouts()[0].clone(),
            &staging_parent_updates,
            &sot_parents,
        );
        let scatter_secondary = record_scatter_secondary(
            cb_allocator,
            queue_family_index,
            &scatter_prepass_pipeline,
            &prepass_set_pos,
            &prepass_set_rot,
            &prepass_set_scl,
            &prepass_dispatch_args,
            &[&compact_words_pos, &compact_words_rot, &compact_words_scl],
            &scatter_build_args_pipeline,
            &build_args_set,
            &scatter_pipeline,
            &scatter_set_pos,
            &scatter_set_rot,
            &scatter_set_scl,
            &trs_dispatch_args,
            &parent_scatter_pipeline,
            &parent_scatter_set,
            &parent_dispatch_args,
            cap,
        );

        // GPU-write early-wake signal buffer + descriptor set + secondary.
        // Single-u32, host-coherent (HOST_RANDOM_ACCESS so we get a
        // CACHED+COHERENT mapping when ReBAR is available; PREFER_HOST
        // because the GPU writes once and the host reads many times —
        // we want the buffer in system RAM to keep the host load cheap).
        let gpu_signal: Subbuffer<[u32]> = make_host_storage_slice::<u32>(
            memory_allocator,
            1,
            BufferUsage::empty(),
            /* prefer_device = */ false,
            /* random_access = */ true,
        );
        // Pre-zero so the host's first poll doesn't see uninitialised junk.
        if let Ok(mut w) = gpu_signal.write() {
            w[0] = 0;
        }
        let signal_set = build_signal_set(
            descriptor_set_allocator,
            signal_pipeline.layout().set_layouts()[0].clone(),
            &gpu_signal,
        );
        let signal_secondary = record_signal_secondary(
            cb_allocator,
            queue_family_index,
            &signal_pipeline,
            &signal_set,
        );

        // Timeline semaphore. Initial value 0 is "already signaled" for
        // the first wait. Vulkano-util enables Vulkan 1.2+ which has
        // timeline_semaphore in core; we still must enable the feature
        // explicitly in the device features (see `lib.rs`).
        // (Removed: replaced by the `gpu_signal` busy-poll above.)
        let _ = device;

        Self {
            sot_positions,
            sot_rotations,
            sot_scales,
            sot_view_proj,
            sot_parents,
            entity_capacity: cap,

            staging_positions,
            staging_rotations,
            staging_scales,
            staging_dirty_pos,
            staging_dirty_rot,
            staging_dirty_scl,
            trs_dispatch_args,
            prepass_bounds_pos,
            prepass_bounds_rot,
            prepass_bounds_scl,
            prepass_dispatch_args,
            compact_words_pos,
            compact_words_rot,
            compact_words_scl,
            staging_parent_updates,
            parent_update_capacity,
            parent_dispatch_args,
            view_proj_buf,

            scatter_set_pos,
            scatter_set_rot,
            scatter_set_scl,
            parent_scatter_set,
            prepass_set_pos,
            prepass_set_rot,
            prepass_set_scl,
            build_args_set,
            scatter_secondary,

            gpu_signal,
            signal_set,
            signal_secondary,
            signal_pipeline,
            next_signal_expected: 1,

            scatter_pipeline,
            scatter_prepass_pipeline,
            scatter_build_args_pipeline,
            parent_scatter_pipeline,
            mvp_build_pipeline,

            descriptor_set_allocator: descriptor_set_allocator.clone(),
            cb_allocator: cb_allocator.clone(),
            queue_family_index,
            queue,

            staging_allocator,
            staging_numa_node,
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
        needed: usize,
    ) -> bool {
        if needed <= self.entity_capacity {
            return false;
        }
        let new_cap = needed.max(self.entity_capacity.saturating_mul(2)).max(1);

        // SoT.
        let (pos, rot, scl) = allocate_sot_buffers(memory_allocator, new_cap);
        self.sot_positions = pos;
        self.sot_rotations = rot;
        self.sot_scales = scl;

        // Parents SoT: sentinel-fill the new buffer, then **copy-preserve**
        // the old records (one-shot fence-waited submit — grow is rare and
        // already the expensive path). Unlike TRS there is no capacity-
        // sized staging mirror to re-upload parents from, so the old
        // buffer's contents are the only source of truth.
        let new_parents = allocate_sot_parents(memory_allocator, new_cap);
        fill_u32_oneshot(&self.cb_allocator, &self.queue, &new_parents, NO_PARENT);
        copy_u32_oneshot(
            &self.cb_allocator,
            &self.queue,
            self.sot_parents.clone(),
            new_parents.clone(),
            self.entity_capacity as u64,
        );
        self.sot_parents = new_parents;

        // Staging triple + dirty + view_proj. Goes through the dedicated
        // staging allocator (kept across reallocs) so mbind never
        // touches unrelated suballocations.
        let (
            staging_positions,
            staging_rotations,
            staging_scales,
            staging_dirty_pos,
            staging_dirty_rot,
            staging_dirty_scl,
            view_proj_buf,
        ) = allocate_staging(&self.staging_allocator, new_cap, self.staging_numa_node);
        self.staging_positions = staging_positions;
        self.staging_rotations = staging_rotations;
        self.staging_scales = staging_scales;
        self.staging_dirty_pos = staging_dirty_pos;
        self.staging_dirty_rot = staging_dirty_rot;
        self.staging_dirty_scl = staging_dirty_scl;
        self.view_proj_buf = view_proj_buf;

        // Compacted-word lists scale with `dirty_word_count(entity_capacity)`
        // (worst case: every word dirty), same as the dirty bitmasks above.
        self.compact_words_pos = allocate_compact_words(memory_allocator, new_cap);
        self.compact_words_rot = allocate_compact_words(memory_allocator, new_cap);
        self.compact_words_scl = allocate_compact_words(memory_allocator, new_cap);

        // Scatter sets capture the new staging + compact-words + SoT handles.
        let (sp, sr, ss) = build_scatter_sets(
            &self.descriptor_set_allocator,
            self.scatter_pipeline.layout().set_layouts()[0].clone(),
            &self.staging_positions,
            &self.staging_rotations,
            &self.staging_scales,
            &self.compact_words_pos,
            &self.compact_words_rot,
            &self.compact_words_scl,
            &self.sot_positions,
            &self.sot_rotations,
            &self.sot_scales,
        );
        self.scatter_set_pos = sp;
        self.scatter_set_rot = sr;
        self.scatter_set_scl = ss;

        // Prepass sets capture the new dirty + compact-words handles
        // (`prepass_bounds_*` are fixed-size and untouched by a capacity grow).
        let (pp, pr, ps) = build_prepass_sets(
            &self.descriptor_set_allocator,
            self.scatter_prepass_pipeline.layout().set_layouts()[0].clone(),
            &self.prepass_bounds_pos,
            &self.prepass_bounds_rot,
            &self.prepass_bounds_scl,
            &self.staging_dirty_pos,
            &self.staging_dirty_rot,
            &self.staging_dirty_scl,
            &self.compact_words_pos,
            &self.compact_words_rot,
            &self.compact_words_scl,
        );
        self.prepass_set_pos = pp;
        self.prepass_set_rot = pr;
        self.prepass_set_scl = ps;

        // Build-args set captures the new compact-words handles
        // (`trs_dispatch_args` is fixed-size and untouched by a capacity grow).
        self.build_args_set = build_args_build_set(
            &self.descriptor_set_allocator,
            self.scatter_build_args_pipeline.layout().set_layouts()[0].clone(),
            &self.compact_words_pos,
            &self.compact_words_rot,
            &self.compact_words_scl,
            &self.trs_dispatch_args,
        );

        // Parent-scatter set captures the new sot_parents handle (staging
        // side unchanged by an entity-capacity grow).
        self.parent_scatter_set = build_parent_scatter_set(
            &self.descriptor_set_allocator,
            self.parent_scatter_pipeline.layout().set_layouts()[0].clone(),
            &self.staging_parent_updates,
            &self.sot_parents,
        );

        // `sot_view_proj` is **not** re-allocated by capacity-grow (it's a
        // fixed single mat4), so every `RenderCamera`'s occlusion set
        // (which binds it) remains valid — no need to rebuild anything here.

        // Scatter secondary captures the new descriptor sets and the new
        // dispatch count.
        self.scatter_secondary = record_scatter_secondary(
            &self.cb_allocator,
            self.queue_family_index,
            &self.scatter_prepass_pipeline,
            &self.prepass_set_pos,
            &self.prepass_set_rot,
            &self.prepass_set_scl,
            &self.prepass_dispatch_args,
            &[
                &self.compact_words_pos,
                &self.compact_words_rot,
                &self.compact_words_scl,
            ],
            &self.scatter_build_args_pipeline,
            &self.build_args_set,
            &self.scatter_pipeline,
            &self.scatter_set_pos,
            &self.scatter_set_rot,
            &self.scatter_set_scl,
            &self.trs_dispatch_args,
            &self.parent_scatter_pipeline,
            &self.parent_scatter_set,
            &self.parent_dispatch_args,
            new_cap,
        );

        self.entity_capacity = new_cap;
        true
    }

    /// Ensure the parent-update staging can hold `needed` pairs this frame.
    /// Returns `true` if it re-allocated — the scatter secondary was
    /// re-recorded, so every FrameSlot primary must be rebuilt (callers
    /// fold this into `force_full`). Geometric growth; never shrinks.
    ///
    /// Call **before** [`Self::write_parent_updates`] each frame, in the
    /// same pre-wait window as [`Self::ensure_capacity`] (the buffers it
    /// drops are protected by vulkano's resource tracking; the rebuild the
    /// return value forces re-records everything that captured them).
    pub fn ensure_parent_update_capacity(&mut self, needed: usize) -> bool {
        if needed <= self.parent_update_capacity {
            return false;
        }
        let new_cap = needed.max(self.parent_update_capacity.saturating_mul(2));
        self.staging_parent_updates =
            allocate_parent_update_staging(&self.staging_allocator, new_cap);
        self.parent_update_capacity = new_cap;

        self.parent_scatter_set = build_parent_scatter_set(
            &self.descriptor_set_allocator,
            self.parent_scatter_pipeline.layout().set_layouts()[0].clone(),
            &self.staging_parent_updates,
            &self.sot_parents,
        );
        self.scatter_secondary = record_scatter_secondary(
            &self.cb_allocator,
            self.queue_family_index,
            &self.scatter_prepass_pipeline,
            &self.prepass_set_pos,
            &self.prepass_set_rot,
            &self.prepass_set_scl,
            &self.prepass_dispatch_args,
            &[
                &self.compact_words_pos,
                &self.compact_words_rot,
                &self.compact_words_scl,
            ],
            &self.scatter_build_args_pipeline,
            &self.build_args_set,
            &self.scatter_pipeline,
            &self.scatter_set_pos,
            &self.scatter_set_rot,
            &self.scatter_set_scl,
            &self.trs_dispatch_args,
            &self.parent_scatter_pipeline,
            &self.parent_scatter_set,
            &self.parent_dispatch_args,
            self.entity_capacity,
        );
        true
    }

    /// Write this frame's per-component word-compaction prepass bounds +
    /// indirect dispatch args, derived from `[min_word, max_word]` — the
    /// lowest and highest dirty-bitmask word index each component's host
    /// staging drain touched this frame (`max_word < 0` if the component
    /// was untouched; `min_word` is meaningless in that case). This
    /// **replaces** the old direct watermark → real-scatter-dispatch-args
    /// scheme: the prepass this sizes only *scans* `[min_word, max_word]`
    /// for nonzero words (still O(span), same as the old scheme, but now
    /// just the cheap scanning stage); the real TRS scatter's own dispatch
    /// args are written on the GPU by `scatter_build_args_cs` from the
    /// prepass's *exact* compacted dirty-word count, not from this
    /// watermark. Must be called **every** frame (same gate as
    /// [`Self::write_parent_updates`]) — callers pass `(_, -1)` for a
    /// component with no dirty bits this frame, which writes a
    /// zero-workgroup prepass dispatch (and, transitively, a zero-workgroup
    /// real scatter dispatch once the GPU-side chain runs).
    pub fn write_prepass_dispatch_groups(&self, min_max_word: [(i64, i64); 3]) {
        let bounds = [
            &self.prepass_bounds_pos,
            &self.prepass_bounds_rot,
            &self.prepass_bounds_scl,
        ];
        let mut args = self
            .prepass_dispatch_args
            .write()
            .expect("prepass_dispatch_args.write");
        for (i, &(min_word, max_word)) in min_max_word.iter().enumerate() {
            let (offset, count) = if max_word < 0 {
                (0u32, 0u32)
            } else {
                debug_assert!(
                    min_word >= 0 && min_word <= max_word,
                    "min/max dirty-word watermark inconsistent: min={min_word} max={max_word}",
                );
                (min_word as u32, (max_word - min_word + 1) as u32)
            };
            let mut b = bounds[i].write().expect("prepass_bounds.write");
            b[0] = offset;
            b[1] = count;
            args[i] = DispatchIndirectCommand {
                x: count.div_ceil(PREPASS_WORDS_PER_WORKGROUP),
                y: 1,
                z: 1,
            };
        }
    }

    /// Write this frame's drained `[transform_id, new_parent]` pairs (plus
    /// the live count in word 0) into the parent-update staging, and the
    /// matching `ceil(count / 64)` group count into `parent_dispatch_args`
    /// so `scatter_secondary`'s `dispatch_indirect` covers exactly the
    /// live records instead of the full staging capacity. Must be called
    /// **every** frame — count 0 retires the previous frame's records (and
    /// writes a zero-workgroup dispatch) — and only after
    /// [`Self::host_wait_for_previous_compute`] (same gate as the TRS
    /// staging writes, which is what makes a re-parent and its paired
    /// local-TRS rewrite land in the same frame).
    pub fn write_parent_updates(&self, updates: &[[u32; 2]]) {
        assert!(
            updates.len() <= self.parent_update_capacity,
            "parent-update burst ({}) exceeds staging capacity ({}) — \
             ensure_parent_update_capacity must run first",
            updates.len(),
            self.parent_update_capacity,
        );
        static mut last_count: usize = usize::MAX;
        if unsafe { last_count } == 0 && updates.is_empty() {
            return;
        }
        unsafe {
            last_count = updates.len();
        }
        let mut w = self
            .staging_parent_updates
            .write()
            .expect("staging_parent_updates.write");
        w[0] = updates.len() as u32;
        for (i, pair) in updates.iter().enumerate() {
            w[2 + 2 * i] = pair[0];
            w[3 + 2 * i] = pair[1];
        }
        drop(w);

        let groups = (updates.len() as u32).div_ceil(64);
        let mut args = self
            .parent_dispatch_args
            .write()
            .expect("parent_dispatch_args.write");
        args[0] = DispatchIndirectCommand {
            x: groups,
            y: 1,
            z: 1,
        };
    }

    // ── Host-side sync API ────────────────────────────────────────

    /// Block the calling thread until the GPU has finished the previous
    /// frame's **scatter primary** — i.e. the scatter dispatches (which
    /// read shared `staging_<comp>` + `dirty_*`), the trailing
    /// `vkCmdFillBuffer(0)` clears (which write zero into `dirty_*`),
    /// AND the `vkCmdCopyBuffer(staging_view_proj → sot_view_proj)`
    /// (which reads `staging_view_proj`). After this returns it is safe
    /// for the host to mutate any of the shared host-writable buffers
    /// for the next frame.
    ///
    /// # Why this single wait covers everything host-writable
    ///
    /// Post-staging-paradigm refactor, **every** host-writable shared
    /// buffer is read only by the scatter primary:
    ///
    /// | Resource                 | Reader                          |
    /// |--------------------------|---------------------------------|
    /// | `staging_<comp>`         | scatter (compute)               |
    /// | `staging_dirty_*`        | scatter (compute)               |
    /// | `staging_parent_updates` | parent scatter (compute)        |
    /// | `view_proj_buf`          | `vkCmdCopyBuffer` (transfer)    |
    ///
    /// `mvp_build` reads only **stable SoT** (`sot_<comp>` and
    /// Busy-poll the GPU-written `gpu_signal` counter until it reaches
    /// the value `signal_cs` was scheduled to bring it to in the
    /// **previous** frame. After this returns it is safe for the host
    /// to mutate any of the shared host-writable buffers (staging TRS,
    /// dirty bitmasks, staging view_proj) for the next frame.
    ///
    /// # Why a poll instead of `vkWaitSemaphores`
    ///
    /// `vkWaitSemaphores` is a kernel-mode syscall (~tens of
    /// microseconds typical, even on a no-op wait). At low N our
    /// per-frame budget is ~90µs; the syscall consumed ~30% of that.
    /// The `signal_cs` dispatch writes a host-coherent buffer mid-CB,
    /// right after every read of host-shared staging is done, so the
    /// host can wake up in user space without entering the kernel — and
    /// without waiting for the rest of the CB (mvp_build + render +
    /// blit) to complete the way an end-of-CB timeline signal would
    /// require.
    ///
    /// # Why this single poll covers everything host-writable
    ///
    /// Same invariant as the timeline-semaphore version: every
    /// host-writable buffer is read only by the scatter dispatches and
    /// the trailing `fill_buffer` / `copy_buffer` commands inside the
    /// FrameSlot primary CB. `signal_cs` is recorded immediately after
    /// those, so its increment fires the moment they're done. Vulkano
    /// auto-sync inserts the `SHADER_WRITE → HOST_READ` visibility
    /// barrier on `gpu_signal` between the dispatch and the implicit
    /// queue-submit `HOST` stage — with `HOST_COHERENT` memory the host
    /// load sees the updated value without an explicit
    /// `vkInvalidateMappedMemoryRanges`.
    ///
    /// # Wraparound
    ///
    /// `gpu_signal[0]` is `u32`. At 11K FPS it wraps every ~4.5 days.
    /// `wrapping_sub` makes the comparison wraparound-safe so a
    /// long-running session is correct.
    ///
    /// # Polling strategy
    ///
    /// Tight `spin_loop()` for the first ~64 iterations — expected to
    /// succeed on the first or second read when the GPU is keeping up.
    /// Then `yield_now()` to let other threads run. After ~1 ms of total
    /// wait we fall back to a 100µs `sleep`, on the assumption that the
    /// GPU is genuinely overloaded and burning a core would do more harm
    /// than good. This keeps the low-N path syscall-free while bounding
    /// CPU consumption when the GPU falls behind at high N.
    ///
    /// First call (next_signal_expected == 1, target wait = 0) returns
    /// immediately because the buffer was pre-zeroed in [`Self::new`].
    pub fn host_wait_for_previous_compute(&self) {
        let target = self.next_signal_expected.wrapping_sub(1);
        // EXPERIMENT: pure spin_loop, no yield_now/sleep fallback. Tests
        // whether the yield/sleep escape hatch is what's keeping us off
        // the "perfect queue invariance" sweet spot some launches hit.
        loop {
            let v = {
                let r = self.gpu_signal.read().expect("gpu_signal.read");
                r[0]
            };
            let delta = v.wrapping_sub(target);
            if delta < i32::MAX as u32 {
                return;
            }
            std::hint::spin_loop();
            // std::thread::yield_now();
        }
    }

    /// Reserve the value the next frame's `signal_cs` dispatch will
    /// bring `gpu_signal` up to. Call **after** queue-submit so the
    /// next frame's `host_wait_for_previous_compute` sees the right
    /// target value.
    pub fn inc_signal_expected(&mut self) {
        self.next_signal_expected = self.next_signal_expected.wrapping_add(1);
    }

    // ── Accessors ─────────────────────────────────────────────────

    pub fn entity_capacity(&self) -> usize {
        self.entity_capacity
    }
    pub fn sot_positions(&self) -> &Subbuffer<[ComponentSlot]> {
        &self.sot_positions
    }
    pub fn sot_rotations(&self) -> &Subbuffer<[ComponentSlot]> {
        &self.sot_rotations
    }
    pub fn sot_scales(&self) -> &Subbuffer<[ComponentSlot]> {
        &self.sot_scales
    }
    /// Parents SoT — bound at binding 8 of the camera's cull set; read by
    /// `mvp_build_cs`'s parent-chain walk.
    pub fn sot_parents(&self) -> &Subbuffer<[u32]> {
        &self.sot_parents
    }
    /// Stable device-local view_proj buffer, populated by the
    /// `vkCmdCopyBuffer` inside `scatter_primary`. Bound by every
    /// `RenderCamera`'s occlusion set (current VP) and copied into its
    /// `prev_view_proj` at the end of each frame (dual-pass occlusion
    /// culling).
    pub fn sot_view_proj(&self) -> &Subbuffer<[[f32; 16]]> {
        &self.sot_view_proj
    }
    pub fn mvp_build_pipeline(&self) -> &Arc<ComputePipeline> {
        &self.mvp_build_pipeline
    }

    /// Shared scatter secondary, executed once per frame from the
    /// FrameSlot primary CB (front of CB, before mvp_build).
    pub fn scatter_secondary(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.scatter_secondary
    }

    /// Shared signal secondary — single-dispatch `signal_cs` that
    /// atomically increments `gpu_signal`. Captured by every FrameSlot
    /// primary right after scatter+fill+copy and before mvp_build.
    pub fn signal_secondary(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.signal_secondary
    }

    /// Shared dirty buffers — referenced by the per-FrameSlot primary CB
    /// for the in-CB `vkCmdFillBuffer(0)` that re-zeroes them after the
    /// scatter consumes them.
    pub fn staging_dirty_pos(&self) -> &Subbuffer<[u32]> {
        &self.staging_dirty_pos
    }
    pub fn staging_dirty_rot(&self) -> &Subbuffer<[u32]> {
        &self.staging_dirty_rot
    }
    pub fn staging_dirty_scl(&self) -> &Subbuffer<[u32]> {
        &self.staging_dirty_scl
    }

    /// Shared host-mapped staging triple. Written by the per-frame
    /// harvest in [`crate::RenderApp::about_to_wait`] after the timeline
    /// wait succeeds.
    pub fn staging_positions(&self) -> &Subbuffer<[f32]> {
        &self.staging_positions
    }
    pub fn staging_rotations(&self) -> &Subbuffer<[f32]> {
        &self.staging_rotations
    }
    pub fn staging_scales(&self) -> &Subbuffer<[f32]> {
        &self.staging_scales
    }

    /// Shared host-mapped view_proj uniform. Written by the per-frame
    /// harvest immediately after the staging triple.
    pub fn view_proj_buf(&self) -> &Subbuffer<[[f32; 16]]> {
        &self.view_proj_buf
    }

    /// Convenience: layout of mvp-build set 0 (per-camera SoT/idx/mvp).
    pub fn mvp_build_set0_layout(&self) -> &Arc<DescriptorSetLayout> {
        &self.mvp_build_pipeline.layout().set_layouts()[0]
    }

    /// Post-warmup residency diagnostic. Walks every staging buffer's
    /// mapped pages and prints the (checked, off-node) counts. Intended
    /// to be called once after the harvest has run for a handful of
    /// frames so the pages have actually been faulted in — the initial
    /// `bind_staging_to_node` runs before any writes touch the range,
    /// so its verify step always reports 0/0.
    #[cfg(target_os = "linux")]
    pub fn report_staging_residency(&self) {
        use vulkano::buffer::BufferMemory;
        use vulkano::device::DeviceOwned;
        use vulkano::memory::DeviceAlignment;

        // ─── 1. Vulkan memory-type info per buffer ────────────────
        let device = self.staging_allocator.device();
        let mem_props = device.physical_device().memory_properties();

        let describe = |label: &str, buf_mem: &BufferMemory, ptr: *const u8, len: usize| {
            match buf_mem {
                BufferMemory::Normal(rm) => {
                    let dm = rm.device_memory();
                    let idx = dm.memory_type_index();
                    let mt = &mem_props.memory_types[idx as usize];
                    let heap = &mem_props.memory_heaps[mt.heap_index as usize];
                    println!(
                        "[numa-staging-info] {label}: ptr={:p} len={} mem_type_idx={} \
                         flags={:?} heap_idx={} heap_flags={:?} heap_size={}MB \
                         alloc_off={} alloc_size={}",
                        ptr,
                        len,
                        idx,
                        mt.property_flags,
                        mt.heap_index,
                        heap.flags,
                        heap.size / (1024 * 1024),
                        rm.offset(),
                        rm.size(),
                    );
                    let _ = DeviceAlignment::MIN; // keep import live in case of refactor
                }
                other => {
                    println!(
                        "[numa-staging-info] {label}: non-Normal memory: {:?}",
                        other
                    );
                }
            }
        };

        // Stash pointer+len so we can also dump /proc/self/maps + numa_maps below.
        let mut ptrs: Vec<(&'static str, *const u8, usize)> = Vec::new();
        let mut visit =
            |label: &'static str, buf_mem: &BufferMemory, ptr: *const u8, len: usize| {
                describe(label, buf_mem, ptr, len);
                ptrs.push((label, ptr, len));
            };

        for (label, buf) in [
            ("pos", &self.staging_positions),
            ("rot", &self.staging_rotations),
            ("scl", &self.staging_scales),
        ] {
            let m = buf.mapped_slice().expect("staging buffer not host-mapped");
            visit(
                label,
                buf.buffer().memory(),
                m.as_ptr().cast::<u8>(),
                m.len(),
            );
        }
        for (label, buf) in [
            ("dirty_pos", &self.staging_dirty_pos),
            ("dirty_rot", &self.staging_dirty_rot),
            ("dirty_scl", &self.staging_dirty_scl),
        ] {
            let m = buf
                .mapped_slice()
                .expect("staging dirty buffer not host-mapped");
            visit(
                label,
                buf.buffer().memory(),
                m.as_ptr().cast::<u8>(),
                m.len(),
            );
        }

        // ─── 2. /proc/self/maps + numa_maps for each pointer ──────
        let maps = std::fs::read_to_string("/proc/self/maps")
            .unwrap_or_else(|e| format!("(read /proc/self/maps failed: {e})"));
        let numa_maps = std::fs::read_to_string("/proc/self/numa_maps")
            .unwrap_or_else(|e| format!("(read /proc/self/numa_maps failed: {e})"));

        for (label, ptr, _len) in &ptrs {
            let addr = *ptr as usize;
            let map_line = maps
                .lines()
                .find(|l| {
                    let Some((range, _)) = l.split_once(' ') else {
                        return false;
                    };
                    let Some((lo, hi)) = range.split_once('-') else {
                        return false;
                    };
                    let (Ok(lo), Ok(hi)) =
                        (usize::from_str_radix(lo, 16), usize::from_str_radix(hi, 16))
                    else {
                        return false;
                    };
                    (lo..hi).contains(&addr)
                })
                .unwrap_or("(no /proc/self/maps line found)");
            println!("[numa-staging-info] {label} maps:      {}", map_line);

            // numa_maps lines look like:  "7f1234567000 default file=/dev/dri/renderD128 ..."
            let numa_line = numa_maps
                .lines()
                .find(|l| {
                    let Some(first) = l.split_whitespace().next() else {
                        return false;
                    };
                    let Ok(base) = usize::from_str_radix(first, 16) else {
                        return false;
                    };
                    // numa_maps lists only the VMA start; rely on the maps lookup
                    // above to confirm range. We accept a numa_maps line iff its
                    // start address equals the VMA start from /proc/self/maps.
                    map_line.starts_with(&format!("{:x}-", base))
                })
                .unwrap_or("(no /proc/self/numa_maps line found)");
            println!("[numa-staging-info] {label} numa_maps: {}", numa_line);
        }

        // ─── 3. per-page move_pages residency (existing) ──────────
        if let Some(node) = self.staging_numa_node {
            let mut per: Vec<(&'static str, usize, usize)> = Vec::new();
            let mut totals = (0usize, 0usize);
            for (label, ptr, len) in &ptrs {
                match engine_core::util::numa_mem::verify_residency_single_node(*ptr, *len, node) {
                    Ok((c, w)) => {
                        per.push((label, c, w));
                        totals.0 += c;
                        totals.1 += w;
                    }
                    Err(e) => {
                        eprintln!("[numa-staging-verify] {label}: move_pages failed: {e}");
                    }
                }
            }
            println!(
                "[numa-staging-verify] node {node}: {}/{}  pages off-node (per-buf: {:?})",
                totals.1, totals.0, per,
            );
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn report_staging_residency(&self) {}
}

// ─────────────────────────────────────────────────────────────────────
// Allocation helpers
// ─────────────────────────────────────────────────────────────────────

fn allocate_sot_buffers(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    capacity: usize,
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
    memory_allocator: &Arc<StandardMemoryAllocator>,
    entity_capacity: usize,
    numa_node: Option<u32>,
) -> (
    Subbuffer<[f32]>,
    Subbuffer<[f32]>,
    Subbuffer<[f32]>,
    Subbuffer<[u32]>,
    Subbuffer<[u32]>,
    Subbuffer<[u32]>,
    Subbuffer<[[f32; 16]]>,
) {
    // Bind the calling thread's allocation policy to `numa_node` for
    // the duration of these allocations. Driver-internal `mmap`s for
    // staging backing memory pick up the thread's mempolicy at fault
    // time and land on the requested node.
    //
    // This is the **only** working mechanism for staging-buffer NUMA
    // placement that doesn't require `numactl`: the staging buffers
    // are driver-managed DMA mappings, and `mbind` on the post-alloc
    // mapped pointer is a silent no-op for those pages (`move_pages`
    // returns -ENOENT — the kernel doesn't track them as anonymous
    // user pages).
    //
    // Fatal on failure: the entire point of this code path is to
    // control where pages land. EPERM here would typically indicate
    // a broken kernel build (set_mempolicy is unprivileged).
    #[cfg(target_os = "linux")]
    let _mempolicy_guard = numa_node.map(|node| {
        engine_core::util::numa_mem::MempolicyGuard::bind_to_node(node).unwrap_or_else(|e| {
            panic!("[numa-staging] set_mempolicy(BIND, node {node}) failed: {e}",)
        })
    });

    // Staging triple: CPU writes only, GPU reads only — switch to
    // HOST_SEQUENTIAL_WRITE so the allocator picks an uncached/WC
    // memory type. This bypasses the per-socket L3, eliminating the
    // cross-socket coherence snoop storm that otherwise stalls the
    // GPU's scatter-pass reads when CPU writers live on both nodes.
    //
    // Flat `float[]`, 3 floats per entity slot (see the struct-level docs
    // on `staging_positions`) — `3 * entity_capacity` elements, not a
    // `vec3`/`vec4` array, so std430 doesn't pad each entry to 16 bytes.
    let pos = make_host_storage_slice::<f32>(
        memory_allocator,
        entity_capacity * 3,
        BufferUsage::empty(),
        true,
        false,
    );
    // Rotation is packed to two `f32`-sized slots per entity (4×f16
    // quaternion) — see `staging_rotations`'s struct-level docs and
    // `pack_quat_half`.
    let rot = make_host_storage_slice::<f32>(
        memory_allocator,
        entity_capacity * 2,
        BufferUsage::empty(),
        true,
        false,
    );
    let scl = make_host_storage_slice::<f32>(
        memory_allocator,
        entity_capacity * 3,
        BufferUsage::empty(),
        true,
        false,
    );

    let dirty_words = dirty_word_count(entity_capacity);
    let dp = make_host_storage_slice::<u32>(
        memory_allocator,
        dirty_words,
        BufferUsage::TRANSFER_DST,
        false,
        true,
    );
    let dr = make_host_storage_slice::<u32>(
        memory_allocator,
        dirty_words,
        BufferUsage::TRANSFER_DST,
        false,
        true,
    );
    let ds = make_host_storage_slice::<u32>(
        memory_allocator,
        dirty_words,
        BufferUsage::TRANSFER_DST,
        false,
        true,
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

    // Single-mat4 host staging for view_proj. Sequential-write WC is fine
    // (one writer per frame, fully sequential). TRANSFER_SRC so the scatter
    // primary can `vkCmdCopyBuffer` it into `sot_view_proj`.
    let vp = make_host_storage_slice::<[f32; 16]>(
        memory_allocator,
        1,
        BufferUsage::TRANSFER_SRC,
        false,
        false,
    );

    (pos, rot, scl, dp, dr, ds, vp)
}

/// Bind (compact_words, staging_values, sot) at set 0 of the real scatter
/// pipeline for each component. `compact_words_*` replaces the raw dirty
/// bitmask at binding 0 — the real scatter dispatch reads the
/// word-compaction prepass's output, not the bitmask directly.
fn build_scatter_sets(
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    scatter_layout: Arc<DescriptorSetLayout>,
    staging_positions: &Subbuffer<[f32]>,
    staging_rotations: &Subbuffer<[f32]>,
    staging_scales: &Subbuffer<[f32]>,
    compact_words_pos: &Subbuffer<[u32]>,
    compact_words_rot: &Subbuffer<[u32]>,
    compact_words_scl: &Subbuffer<[u32]>,
    sot_positions: &Subbuffer<[ComponentSlot]>,
    sot_rotations: &Subbuffer<[ComponentSlot]>,
    sot_scales: &Subbuffer<[ComponentSlot]>,
) -> (Arc<DescriptorSet>, Arc<DescriptorSet>, Arc<DescriptorSet>) {
    let pos = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        scatter_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, compact_words_pos.clone()),
            WriteDescriptorSet::buffer(1, staging_positions.clone()),
            WriteDescriptorSet::buffer(2, sot_positions.clone()),
        ],
        [],
    )
    .expect("scatter_set_pos");
    let rot = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        scatter_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, compact_words_rot.clone()),
            WriteDescriptorSet::buffer(1, staging_rotations.clone()),
            WriteDescriptorSet::buffer(2, sot_rotations.clone()),
        ],
        [],
    )
    .expect("scatter_set_rot");
    let scl = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        scatter_layout,
        [
            WriteDescriptorSet::buffer(0, compact_words_scl.clone()),
            WriteDescriptorSet::buffer(1, staging_scales.clone()),
            WriteDescriptorSet::buffer(2, sot_scales.clone()),
        ],
        [],
    )
    .expect("scatter_set_scl");
    (pos, rot, scl)
}

/// Bind (prepass_bounds, staging_dirty, compact_words) at set 0 of the
/// word-compaction prepass pipeline, one descriptor set per component.
fn build_prepass_sets(
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    prepass_layout: Arc<DescriptorSetLayout>,
    prepass_bounds_pos: &Subbuffer<[u32]>,
    prepass_bounds_rot: &Subbuffer<[u32]>,
    prepass_bounds_scl: &Subbuffer<[u32]>,
    staging_dirty_pos: &Subbuffer<[u32]>,
    staging_dirty_rot: &Subbuffer<[u32]>,
    staging_dirty_scl: &Subbuffer<[u32]>,
    compact_words_pos: &Subbuffer<[u32]>,
    compact_words_rot: &Subbuffer<[u32]>,
    compact_words_scl: &Subbuffer<[u32]>,
) -> (Arc<DescriptorSet>, Arc<DescriptorSet>, Arc<DescriptorSet>) {
    let pos = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        prepass_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, prepass_bounds_pos.clone()),
            WriteDescriptorSet::buffer(1, staging_dirty_pos.clone()),
            WriteDescriptorSet::buffer(2, compact_words_pos.clone()),
        ],
        [],
    )
    .expect("prepass_set_pos");
    let rot = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        prepass_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, prepass_bounds_rot.clone()),
            WriteDescriptorSet::buffer(1, staging_dirty_rot.clone()),
            WriteDescriptorSet::buffer(2, compact_words_rot.clone()),
        ],
        [],
    )
    .expect("prepass_set_rot");
    let scl = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        prepass_layout,
        [
            WriteDescriptorSet::buffer(0, prepass_bounds_scl.clone()),
            WriteDescriptorSet::buffer(1, staging_dirty_scl.clone()),
            WriteDescriptorSet::buffer(2, compact_words_scl.clone()),
        ],
        [],
    )
    .expect("prepass_set_scl");
    (pos, rot, scl)
}

/// Bind (compact_words_pos, compact_words_rot, compact_words_scl,
/// trs_dispatch_args) at set 0 of `scatter_build_args_cs` — the tiny
/// dispatch that converts the three prepasses' compacted dirty-word counts
/// into the real scatter's indirect dispatch args.
fn build_args_build_set(
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    layout: Arc<DescriptorSetLayout>,
    compact_words_pos: &Subbuffer<[u32]>,
    compact_words_rot: &Subbuffer<[u32]>,
    compact_words_scl: &Subbuffer<[u32]>,
    trs_dispatch_args: &Subbuffer<[DispatchIndirectCommand]>,
) -> Arc<DescriptorSet> {
    DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout,
        [
            WriteDescriptorSet::buffer(0, compact_words_pos.clone()),
            WriteDescriptorSet::buffer(1, compact_words_rot.clone()),
            WriteDescriptorSet::buffer(2, compact_words_scl.clone()),
            WriteDescriptorSet::buffer(3, trs_dispatch_args.clone()),
        ],
        [],
    )
    .expect("build_args_set")
}

/// Allocate the stable device-local `view_proj` SoT buffer (1 mat4).
/// Targeted by the `vkCmdCopyBuffer` inside `scatter_primary` and read by
/// `mvp_build_cs` via `RenderCamera`'s occlusion set. `STORAGE_BUFFER` so
/// it can be bound as such; `TRANSFER_DST` so it can be the destination of
/// the per-frame copy; `TRANSFER_SRC` so `RenderCamera` can copy it into
/// its `prev_view_proj` history at the end of each frame.
fn allocate_sot_view_proj(
    memory_allocator: &Arc<StandardMemoryAllocator>,
) -> Subbuffer<[[f32; 16]]> {
    Buffer::new_slice::<[f32; 16]>(
        memory_allocator.clone(),
        BufferCreateInfo {
            // TRANSFER_DST: the per-frame staging→SoT promotion copy.
            // TRANSFER_SRC: the camera's end-of-frame copy into its
            // `prev_view_proj` (dual-pass occlusion culling — see
            // `camera.rs`), which reads *this* frame's freshly-promoted VP.
            usage: BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_DST
                | BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        1,
    )
    .expect("Failed to allocate sot_view_proj buffer")
}

/// Allocate the device-local Parents SoT buffer: one `u32` parent id per
/// entity slot. `TRANSFER_DST` for the sentinel fill, `TRANSFER_SRC` for
/// the copy-preserving migration on capacity grows.
fn allocate_sot_parents(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    capacity: usize,
) -> Subbuffer<[u32]> {
    Buffer::new_slice::<u32>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_SRC
                | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        capacity.max(1) as u64,
    )
    .expect("Failed to allocate sot_parents buffer")
}

/// Allocate the host-mapped parent-update staging: word 0 = count, word 1
/// = pad (std430 `uvec2[]` starts at offset 8), then `pair_capacity`
/// pairs. Sequential-write WC is the right memory type — one writer per
/// frame, written front-to-back. Count is zeroed so a frame slot recorded
/// before the first `write_parent_updates` scatters nothing.
fn allocate_parent_update_staging(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    pair_capacity: usize,
) -> Subbuffer<[u32]> {
    let buf = make_host_storage_slice::<u32>(
        memory_allocator,
        2 + 2 * pair_capacity.max(1),
        BufferUsage::empty(),
        /* prefer_device = */ false,
        /* random_access = */ false,
    );
    {
        let mut w = buf.write().expect("zero-init staging_parent_updates");
        w[0] = 0;
        w[1] = 0;
    }
    buf
}

/// Allocate the device-local three-`DispatchIndirectCommand` buffer
/// (`[pos, rot, scl]`) the real TRS scatter dispatches indirect over.
/// Written on the GPU every frame by `scatter_build_args_cs` (folded into
/// `scatter_secondary`, right after the word-compaction prepass compacts
/// each component's exact dirty-word count) — never touched by the host,
/// so no zero-init is needed: the very first execution of `scatter_secondary`
/// already runs the args-builder before the real scatter dispatch reads
/// this buffer. Never reallocated — always exactly three commands.
fn allocate_trs_dispatch_args(
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
        3,
    )
    .expect("Failed to allocate trs_dispatch_args buffer")
}

/// Allocate the three host-mapped `[word_offset, word_count]` prepass-bounds
/// pairs (`prepass_bounds_pos/rot/scl`), written every frame by
/// [`WorldTransformGpu::write_prepass_dispatch_groups`]. Never reallocated;
/// zero-initialised so a frame slot recorded before the first write scans
/// nothing.
fn allocate_prepass_bounds(
    memory_allocator: &Arc<StandardMemoryAllocator>,
) -> (Subbuffer<[u32]>, Subbuffer<[u32]>, Subbuffer<[u32]>) {
    let make = || -> Subbuffer<[u32]> {
        let buf = make_host_storage_slice::<u32>(
            memory_allocator,
            2,
            BufferUsage::empty(),
            /* prefer_device = */ false,
            /* random_access = */ false,
        );
        {
            let mut w = buf.write().expect("zero-init prepass_bounds");
            w[0] = 0;
            w[1] = 0;
        }
        buf
    };
    (make(), make(), make())
}

/// Allocate the host-mapped three-`DispatchIndirectCommand` buffer
/// (`[pos, rot, scl]`) that
/// [`WorldTransformGpu::write_prepass_dispatch_groups`] fills every frame
/// with each component's `[min_word, max_word]`-derived prepass group
/// count. Never reallocated — always exactly three commands, zero-
/// initialised so a frame slot recorded before the first
/// `write_prepass_dispatch_groups` dispatches zero workgroups.
fn allocate_prepass_dispatch_args(
    memory_allocator: &Arc<StandardMemoryAllocator>,
) -> Subbuffer<[DispatchIndirectCommand]> {
    let buf = make_host_storage_slice::<DispatchIndirectCommand>(
        memory_allocator,
        3,
        BufferUsage::INDIRECT_BUFFER,
        /* prefer_device = */ false,
        /* random_access = */ false,
    );
    {
        let mut w = buf.write().expect("zero-init prepass_dispatch_args");
        for i in 0..3 {
            w[i] = DispatchIndirectCommand { x: 0, y: 1, z: 1 };
        }
    }
    buf
}

/// Allocate one component's device-local compacted dirty-word list:
/// `{count, pad, uvec2 entries[]}` — `.x` = dirty word index, `.y` = that
/// word's bitmask, matching `parent_scatter.comp`'s count-in-buffer
/// convention. Sized to the worst case (every word dirty this frame):
/// `2 + 2 * dirty_word_count(entity_capacity)` u32s. `count` is reset to 0
/// by an in-CB `fill_buffer` right before the prepass dispatch each frame
/// (see `record_scatter_secondary`) — never touched by the host, hence
/// `TRANSFER_DST` (for the fill) but no zero-init here.
fn allocate_compact_words(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    entity_capacity: usize,
) -> Subbuffer<[u32]> {
    Buffer::new_slice::<u32>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        (2 + 2 * dirty_word_count(entity_capacity)) as u64,
    )
    .expect("Failed to allocate compact_words buffer")
}

/// Allocate the host-mapped single-`DispatchIndirectCommand` buffer that
/// [`WorldTransformGpu::write_parent_updates`] fills every frame with the
/// parent-scatter dispatch's live group count. Never reallocated — always
/// exactly one command, zero-initialised so a frame slot recorded before
/// the first `write_parent_updates` dispatches zero workgroups.
fn allocate_parent_dispatch_args(
    memory_allocator: &Arc<StandardMemoryAllocator>,
) -> Subbuffer<[DispatchIndirectCommand]> {
    let buf = make_host_storage_slice::<DispatchIndirectCommand>(
        memory_allocator,
        1,
        BufferUsage::INDIRECT_BUFFER,
        /* prefer_device = */ false,
        /* random_access = */ false,
    );
    {
        let mut w = buf.write().expect("zero-init parent_dispatch_args");
        w[0] = DispatchIndirectCommand { x: 0, y: 1, z: 1 };
    }
    buf
}

/// Bind (staging_parent_updates, sot_parents) at set 0 of the
/// parent-scatter pipeline.
fn build_parent_scatter_set(
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    layout: Arc<DescriptorSetLayout>,
    staging_parent_updates: &Subbuffer<[u32]>,
    sot_parents: &Subbuffer<[u32]>,
) -> Arc<DescriptorSet> {
    DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout,
        [
            WriteDescriptorSet::buffer(0, staging_parent_updates.clone()),
            WriteDescriptorSet::buffer(1, sot_parents.clone()),
        ],
        [],
    )
    .expect("parent_scatter_set")
}

/// One-shot fence-waited `vkCmdFillBuffer`. Used only off the per-frame
/// path (construction-time sentinel fill, capacity-grow migration).
fn fill_u32_oneshot(
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
    buf: &Subbuffer<[u32]>,
    value: u32,
) {
    let mut builder = AutoCommandBufferBuilder::primary(
        cb_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("one-shot fill CB builder");
    builder
        .fill_buffer(buf.clone(), value)
        .expect("fill_buffer");
    submit_and_wait_oneshot(queue, builder.build().expect("build one-shot fill CB"));
}

/// One-shot fence-waited buffer copy of the first `count` elements.
fn copy_u32_oneshot(
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
    src: Subbuffer<[u32]>,
    dst: Subbuffer<[u32]>,
    count: u64,
) {
    let mut builder = AutoCommandBufferBuilder::primary(
        cb_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("one-shot copy CB builder");
    builder
        .copy_buffer(CopyBufferInfo::buffers(
            src.slice(0..count),
            dst.slice(0..count),
        ))
        .expect("copy_buffer");
    submit_and_wait_oneshot(queue, builder.build().expect("build one-shot copy CB"));
}

fn submit_and_wait_oneshot(
    queue: &Arc<Queue>,
    cb: Arc<impl PrimaryCommandBufferAbstract + 'static>,
) {
    vulkano::sync::now(queue.device().clone())
        .then_execute(queue.clone(), cb)
        .expect("submit one-shot CB")
        .then_signal_fence_and_flush()
        .expect("flush one-shot CB")
        .wait(None)
        .expect("await one-shot CB");
}

#[allow(clippy::too_many_arguments)]
fn record_scatter_secondary(
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
    prepass_pipeline: &Arc<ComputePipeline>,
    prepass_set_pos: &Arc<DescriptorSet>,
    prepass_set_rot: &Arc<DescriptorSet>,
    prepass_set_scl: &Arc<DescriptorSet>,
    prepass_dispatch_args: &Subbuffer<[DispatchIndirectCommand]>,
    compact_words: &[&Subbuffer<[u32]>; 3],
    build_args_pipeline: &Arc<ComputePipeline>,
    build_args_set: &Arc<DescriptorSet>,
    scatter_pipeline: &Arc<ComputePipeline>,
    scatter_set_pos: &Arc<DescriptorSet>,
    scatter_set_rot: &Arc<DescriptorSet>,
    scatter_set_scl: &Arc<DescriptorSet>,
    trs_dispatch_args: &Subbuffer<[DispatchIndirectCommand]>,
    parent_scatter_pipeline: &Arc<ComputePipeline>,
    parent_scatter_set: &Arc<DescriptorSet>,
    parent_dispatch_args: &Subbuffer<[DispatchIndirectCommand]>,
    entity_capacity: usize,
) -> Arc<SecondaryAutoCommandBuffer> {
    let scatter_layout = scatter_pipeline.layout().clone();
    let pc_linear = shaders::scatter_cs::PC {
        entity_count: entity_capacity as u32,
        is_rotation: 0,
    };
    let pc_rotation = shaders::scatter_cs::PC {
        entity_count: entity_capacity as u32,
        is_rotation: 1,
    };

    let mut builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        // SimultaneousUse: this secondary is captured by every FrameSlot
        // primary (one per swapchain image, up to MAX_FRAMES_IN_FLIGHT in
        // flight concurrently). The host-side timeline wait
        // (`host_wait_for_previous_compute`) gates host writes to the
        // shared staging this secondary reads, but the GPU may have
        // multiple in-flight executions of this secondary at any moment
        // (different swapchain images' primaries running concurrently),
        // which `MultipleSubmit` would reject at submit time.
        CommandBufferUsage::SimultaneousUse,
        CommandBufferInheritanceInfo::default(),
    )
    .expect("scatter secondary builder");

    // ── Stage 1: word-compaction prepass (×3: pos, rot, scl) ───────────
    // Resets each component's compacted-word `count` to 0 (vulkano
    // auto-syncs this `fill_buffer` against the prepass's own atomic
    // read-modify-write on the same word — same pattern as
    // `record_cull_secondary`'s `fill_buffer(candidate_count, 0)` in
    // `camera.rs`), then dispatches indirectly over
    // `prepass_dispatch_args[i]` — written every frame by
    // [`WorldTransformGpu::write_prepass_dispatch_groups`] from this
    // component's `[min_word, max_word]` watermark, so a quiet component
    // scans nothing and a component whose dirty range is a small span
    // scans only that span, not the whole capacity.
    builder
        .bind_pipeline_compute(prepass_pipeline.clone())
        .expect("bind scatter prepass pipeline");
    for (i, (set, words)) in [
        (prepass_set_pos, compact_words[0]),
        (prepass_set_rot, compact_words[1]),
        (prepass_set_scl, compact_words[2]),
    ]
    .into_iter()
    .enumerate()
    {
        builder
            .fill_buffer(words.clone().slice(0..1), 0)
            .expect("reset compact_words count");
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                prepass_pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .expect("bind prepass set");
        // Safety: `prepass_dispatch_args[i]` is written by
        // `write_prepass_dispatch_groups` every frame under the same
        // `gpu_signal` gate that protects every other host-shared staging
        // buffer this secondary reads, and always spans every word between
        // this component's lowest and highest dirty word this frame. The
        // shader bounds-checks its own trailing wavefront against the
        // in-buffer `word_count`.
        unsafe {
            builder
                .dispatch_indirect(prepass_dispatch_args.clone().slice(i as u64..i as u64 + 1))
                .expect("dispatch_indirect scatter prepass");
        }
    }

    // ── Stage 2: build the real scatter's indirect dispatch args ───────
    // Single 1×1×1 dispatch, ordered after stage 1's atomic writes to
    // `compact_words[*].count` by vulkano auto-sync (same secondary, same
    // shape as `cull_pass2_args_cs` in the occlusion-cull pipeline).
    builder
        .bind_pipeline_compute(build_args_pipeline.clone())
        .expect("bind scatter build-args pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            build_args_pipeline.layout().clone(),
            0,
            build_args_set.clone(),
        )
        .expect("bind scatter build-args set");
    // Safety: 1×1×1 dispatch is unconditionally valid.
    unsafe {
        builder
            .dispatch([1, 1, 1])
            .expect("dispatch scatter build-args");
    }

    // ── Stage 3: real TRS scatter (×3: pos, rot, scl) ───────────────────
    builder
        .bind_pipeline_compute(scatter_pipeline.clone())
        .expect("bind scatter pipeline");
    // `dispatch_indirect` over this component's slot in `trs_dispatch_args`,
    // written moments earlier (stage 2, same secondary) from the prepass's
    // *exact* compacted dirty-word count — `ceil(word_count / 2)`
    // workgroups, not a span-based watermark. Quiet components dispatch
    // zero workgroups.
    for (i, (set, pc)) in [
        (scatter_set_pos, pc_linear),
        (scatter_set_rot, pc_rotation),
        (scatter_set_scl, pc_linear),
    ]
    .into_iter()
    .enumerate()
    {
        builder
            .push_constants(scatter_layout.clone(), 0, pc)
            .expect("push scatter pc")
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                scatter_layout.clone(),
                0,
                set.clone(),
            )
            .expect("bind scatter set");
        // Safety: `trs_dispatch_args[i]` is written by stage 2's
        // `scatter_build_args_cs` dispatch earlier in this same secondary,
        // always holding this frame's exact compacted dirty-word count for
        // this component. The push-constant `entity_count` additionally
        // bounds-checks the shader's trailing wavefront against
        // `entity_capacity`.
        unsafe {
            builder
                .dispatch_indirect(trs_dispatch_args.clone().slice(i as u64..i as u64 + 1))
                .expect("dispatch_indirect scatter");
        }
    }

    // Parent-update stream scatter. `dispatch_indirect` over
    // `parent_dispatch_args`, which [`WorldTransformGpu::write_parent_updates`]
    // fills each frame with `ceil(live_count / 64)` group counts — quiet
    // frames dispatch zero workgroups instead of walking the full staging
    // capacity. Folded in here so parent updates are (a) covered by the
    // same `gpu_signal` gate as TRS staging (host-write safety + same-frame
    // atomicity with a paired local-TRS rewrite) and (b) ordered before
    // mvp_build's chain walk by vulkano auto-sync on `sot_parents`.
    builder
        .bind_pipeline_compute(parent_scatter_pipeline.clone())
        .expect("bind parent scatter pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            parent_scatter_pipeline.layout().clone(),
            0,
            parent_scatter_set.clone(),
        )
        .expect("bind parent scatter set");
    // Safety: `parent_dispatch_args` is written by `write_parent_updates`
    // every frame under the same `gpu_signal` gate that protects every
    // other host-shared staging buffer this secondary reads, so it always
    // holds the count from the frame that triggered this dispatch. The
    // shader still bounds-checks against the in-buffer live count (the
    // last workgroup can have invocations past it).
    unsafe {
        builder
            .dispatch_indirect(parent_dispatch_args.clone())
            .expect("dispatch_indirect parent scatter");
    }

    builder.build().expect("build scatter secondary")
}

/// Build the compute pipeline for `parent_scatter_cs` — the streamed
/// count-in-buffer parent-update scatter folded into the scatter secondary.
fn build_parent_scatter_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs =
        shaders::parent_scatter_cs::load(device.clone()).expect("parent_scatter_cs load failed");
    let entry = cs
        .entry_point("main")
        .expect("parent_scatter_cs entry point");
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(&stage))
            .into_pipeline_layout_create_info(device.clone())
            .expect("parent scatter pipeline layout info"),
    )
    .expect("parent scatter pipeline layout");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("parent scatter ComputePipeline::new")
}

/// Build the compute pipeline for `signal_cs` — the tiny early-wake
/// dispatch that atomically increments `gpu_signal`.
fn build_signal_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs = shaders::signal_cs::load(device.clone()).expect("signal_cs load failed");
    let entry = cs.entry_point("main").expect("signal_cs entry point");
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(&stage))
            .into_pipeline_layout_create_info(device.clone())
            .expect("signal pipeline layout info"),
    )
    .expect("signal pipeline layout");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("signal ComputePipeline::new")
}

/// Bind `gpu_signal` at set 0, binding 0 of the `signal_cs` pipeline.
fn build_signal_set(
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    layout: Arc<DescriptorSetLayout>,
    gpu_signal: &Subbuffer<[u32]>,
) -> Arc<DescriptorSet> {
    DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout,
        [WriteDescriptorSet::buffer(0, gpu_signal.clone())],
        [],
    )
    .expect("signal_set")
}

/// Pre-record the `signal_cs` secondary — single (1×1×1) dispatch.
/// SimultaneousUse because every in-flight FrameSlot primary captures it.
fn record_signal_secondary(
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
    signal_pipeline: &Arc<ComputePipeline>,
    signal_set: &Arc<DescriptorSet>,
) -> Arc<SecondaryAutoCommandBuffer> {
    let layout = signal_pipeline.layout().clone();
    let mut builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::SimultaneousUse,
        CommandBufferInheritanceInfo::default(),
    )
    .expect("signal secondary builder");
    builder
        .bind_pipeline_compute(signal_pipeline.clone())
        .expect("bind signal pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            layout.clone(),
            0,
            signal_set.clone(),
        )
        .expect("bind signal set");
    // Safety: 1×1×1 dispatch is unconditionally valid; signal_cs is
    // pure-write (atomicAdd), no inputs to bounds-check.
    unsafe {
        builder.dispatch([1, 1, 1]).expect("dispatch signal");
    }
    builder.build().expect("build signal secondary")
}

/// Build the compute pipeline for `scatter_cs`.
fn build_scatter_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs = shaders::scatter_cs::load(device.clone()).expect("scatter_cs load failed");
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

/// Build the compute pipeline for `scatter_prepass_cs` — the word-
/// compaction prepass folded into the scatter secondary ahead of the real
/// scatter dispatch.
fn build_scatter_prepass_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs =
        shaders::scatter_prepass_cs::load(device.clone()).expect("scatter_prepass_cs load failed");
    let entry = cs
        .entry_point("main")
        .expect("scatter_prepass_cs entry point");
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(&stage))
            .into_pipeline_layout_create_info(device.clone())
            .expect("scatter prepass pipeline layout info"),
    )
    .expect("scatter prepass pipeline layout");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("scatter prepass ComputePipeline::new")
}

/// Build the compute pipeline for `scatter_build_args_cs` — converts the
/// three prepasses' compacted dirty-word counts into the real scatter's
/// indirect dispatch args.
fn build_scatter_build_args_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs = shaders::scatter_build_args_cs::load(device.clone())
        .expect("scatter_build_args_cs load failed");
    let entry = cs
        .entry_point("main")
        .expect("scatter_build_args_cs entry point");
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(&stage))
            .into_pipeline_layout_create_info(device.clone())
            .expect("scatter build-args pipeline layout info"),
    )
    .expect("scatter build-args pipeline layout");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("scatter build-args ComputePipeline::new")
}

fn build_mvp_build_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs = shaders::mvp_build_cs::load(device.clone()).expect("mvp_build_cs load failed");
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
    count: usize,
    extra_usage: BufferUsage,
    prefer_device: bool,
    random_access: bool,
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

#[cfg(test)]
mod pack_quat_half_tests {
    use super::pack_quat_half;
    use glam::Quat;
    use half::f16;

    /// Rust mirror of the GLSL decode path (`unpackHalf2x16` × 2 +
    /// `normalize`) in `shaders/scatter.comp` — kept in lockstep so this
    /// test exercises the same bit-level round trip the GPU performs, not
    /// just whatever the encoder intended.
    fn unpack_quat_half(packed: [u32; 2]) -> Quat {
        let unpack2 = |p: u32| -> (f32, f32) {
            let lo = f16::from_bits((p & 0xFFFF) as u16).to_f32();
            let hi = f16::from_bits((p >> 16) as u16).to_f32();
            (lo, hi)
        };
        let (x, y) = unpack2(packed[0]);
        let (z, w) = unpack2(packed[1]);
        Quat::from_xyzw(x, y, z, w).normalize()
    }

    /// Angle (radians) between two unit quaternions, robust to the `q ==
    /// -q` double cover.
    fn angle_between(a: Quat, b: Quat) -> f32 {
        let d = a.dot(b).abs().min(1.0);
        2.0 * d.acos()
    }

    #[test]
    fn round_trip_stays_within_documented_error_bound() {
        // f16 has an 11-bit (10 explicit + implicit leading) mantissa, so
        // relative precision near a unit quaternion's component range is
        // well under 0.1° per component with no reconstruction step to
        // amplify it — 0.1° total leaves a healthy margin.
        let max_allowed = 0.1f32.to_radians();

        let mut max_err = 0.0f32;
        let samples = [
            Quat::IDENTITY,
            Quat::from_rotation_x(0.0),
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            Quat::from_rotation_y(std::f32::consts::PI),
            Quat::from_rotation_z(1.2345),
            Quat::from_euler(glam::EulerRot::XYZ, 0.3, 1.9, -2.7),
            Quat::from_euler(glam::EulerRot::XYZ, -3.0, 0.01, 3.0),
        ];
        for &q in &samples {
            let packed = pack_quat_half(q);
            let decoded = unpack_quat_half(packed);
            let err = angle_between(q, decoded);
            assert!(
                err <= max_allowed,
                "round-trip error {} rad exceeds bound for q={:?}",
                err,
                q
            );
            max_err = max_err.max(err);
        }

        // Dense sweep to catch any axis-dependent worst case, not just the
        // hand-picked samples above.
        for i in 0..2000 {
            let t = i as f32 * 0.01;
            let q = Quat::from_euler(glam::EulerRot::XYZ, t, t * 1.7, t * 0.3).normalize();
            let packed = pack_quat_half(q);
            let decoded = unpack_quat_half(packed);
            let err = angle_between(q, decoded);
            assert!(
                err <= max_allowed,
                "round-trip error {} rad exceeds bound for q={:?}",
                err,
                q
            );
            max_err = max_err.max(err);
        }

        // Sanity: the bound should actually be exercised somewhere in the
        // sweep, otherwise the test isn't a meaningful check of precision.
        assert!(max_err > 0.0);
    }

    #[test]
    fn deeply_nested_chain_stays_visually_stable() {
        // Regression guard for the bug that motivated this rewrite: compose
        // N "joints" the way `mvp_build_cs`'s parent-chain walk does
        // (`rot = quat_mul(parent_rot, rot)`), each joint's rotation
        // round-tripped through the staged encoding exactly as it would be
        // read from SoT, and check the *leaf* transform — after position is
        // carried a unit lever-arm through each joint — hasn't drifted
        // enough to plausibly cause visible clipping.
        let n = 32; // deep hierarchy, well beyond this engine's typical asset depth
        let mut true_rot = Quat::IDENTITY;
        let mut true_pos = glam::Vec3::ZERO;
        let mut approx_rot = Quat::IDENTITY;
        let mut approx_pos = glam::Vec3::ZERO;
        for i in 0..n {
            let joint = Quat::from_euler(
                glam::EulerRot::XYZ,
                0.05 + 0.001 * i as f32,
                0.03,
                -0.02 * i as f32,
            );
            let approx_joint = unpack_quat_half(pack_quat_half(joint));

            // Position carried one unit further out along the local +X
            // lever arm at each level, matching `pos = pp + quat_rotate(pq,
            // pos) * ps` in `mvp_build.comp` (scale = 1 here).
            true_pos += true_rot * glam::Vec3::X;
            approx_pos += approx_rot * glam::Vec3::X;
            true_rot *= joint;
            approx_rot *= approx_joint;
        }

        let leaf_drift = (true_pos - approx_pos).length();
        assert!(
            leaf_drift < 0.01,
            "leaf position drifted {leaf_drift} units over a {n}-joint chain — \
             visible clipping risk at this precision"
        );
    }
}
