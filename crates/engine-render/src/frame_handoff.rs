//! CPU-hosted TRS staging + the main-thread ↔ background-render-thread
//! handoff.
//!
//! The per-frame harvest that used to write dirty TRS values *directly*
//! into `WorldTransformGpu`'s host-visible, GPU-mapped staging `Subbuffer`s
//! (gated by `host_wait_for_previous_compute`'s GPU-signal poll) now writes
//! into a plain heap-allocated [`CpuTrsStaging`] here instead, gated only by
//! [`FrameHandoff::wait_for_buffer_free`] — a wait on the *background*
//! render thread, not the GPU. The background thread's job (see
//! [`FrameHandoff::consume`]) is to memcpy that CPU buffer into the
//! (unchanged) GPU-visible staging `Subbuffer`s and take over from there
//! exactly as the main thread used to: `write_prepass_dispatch_groups`,
//! the existing `gpu_signal` wait/scatter/submit dance.
//!
//! Single buffer, single handoff — a direct structural analog of the GPU
//! staging triple's existing single-buffer + single-wait design, just
//! relocated. See the module docs on `transform_gpu` for the GPU-side half
//! of this pipeline, which is otherwise unchanged.

use std::cell::UnsafeCell;
use std::sync::atomic::{self, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

use engine_core::component::Scene;
use engine_core::util::my_thread_pool;
use engine_core::util::parallel;

use crate::transform_gpu::{self, dirty_word_count, WorldTransformGpu};

#[inline]
fn pack_extent(extent: [u32; 2]) -> u64 {
    ((extent[0] as u64) << 32) | extent[1] as u64
}

#[inline]
fn unpack_extent(packed: u64) -> [u32; 2] {
    [(packed >> 32) as u32, packed as u32]
}

/// Per-entity-slot TRS staged on the CPU — plain heap allocations mirroring
/// the shape of `WorldTransformGpu`'s staging `Subbuffer`s (see that
/// module's docs for the exact layout of each field; unchanged here).
struct CpuTrsStaging {
    positions: Box<[f32]>, // 3 * capacity
    rotations: Box<[f32]>, // 2 * capacity (packed half — `transform_gpu::pack_quat_half`)
    scales: Box<[f32]>,    // 3 * capacity
    dirty_pos: Box<[u32]>, // dirty_word_count(capacity)
    dirty_rot: Box<[u32]>,
    dirty_scl: Box<[u32]>,
    view_proj: [f32; 16],
    /// `[pos, rot, scl]` `(min_word, max_word)` dirty-word watermarks this
    /// frame; `max_word < 0` means untouched (`min_word` meaningless then)
    /// — same convention as `WorldTransformGpu::write_prepass_dispatch_groups`.
    min_max_words: [(i64, i64); 3],
    capacity: usize,
}

impl CpuTrsStaging {
    fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        let words = dirty_word_count(cap);
        Self {
            positions: vec![0.0; 3 * cap].into_boxed_slice(),
            rotations: vec![0.0; 2 * cap].into_boxed_slice(),
            scales: vec![0.0; 3 * cap].into_boxed_slice(),
            dirty_pos: vec![0; words].into_boxed_slice(),
            dirty_rot: vec![0; words].into_boxed_slice(),
            dirty_scl: vec![0; words].into_boxed_slice(),
            view_proj: [0.0; 16],
            min_max_words: [(i64::MAX, -1); 3],
            capacity: cap,
        }
    }

    /// Grow to at least `needed` slots (geometric, never shrinks). Plain
    /// realloc — this buffer is exclusively main-thread-owned memory
    /// (the background thread only ever touches it between a
    /// `wait_for_buffer_free` and the matching `consume`, never during a
    /// grow), so no cross-thread coordination is needed here.
    fn ensure_capacity(&mut self, needed: usize) {
        if needed <= self.capacity {
            return;
        }
        *self = Self::new(needed.max(self.capacity.saturating_mul(2)));
    }
}

/// Per-frame handoff payload, sent from the main thread to the background
/// render thread over an `mpsc` channel after [`FrameHandoff::write_frame`]
/// has harvested this frame's TRS into the shared CPU staging buffer.
///
/// `parent_updates` / `spawns` don't need the shared [`FrameHandoff`]
/// buffer — they're already independently allocated fresh each frame by
/// the hierarchy's drain / the spawn queue, so moving them through the
/// channel is a plain ownership transfer, not a copy.
pub(crate) struct FrameJob {
    /// The `FrameHandoff` generation this job's CPU staging snapshot was
    /// published at — pass back to [`FrameHandoff::consume`] once the
    /// background thread is done reading it.
    pub(crate) ready_gen: u64,
    pub(crate) entity_count: usize,
    pub(crate) parent_updates: Vec<[u32; 2]>,
    pub(crate) spawns: Vec<[u32; 3]>,
    pub(crate) f8_pressed: bool,
    pub(crate) f9_pressed: bool,
    /// This frame's view-projection matrix, columns-major — computed on the
    /// main thread (it reads the camera entity's *global* position/rotation
    /// out of `Scene::transform_hierarchy`, ECS state that stays
    /// main-thread-owned). Already folded into the CPU staging snapshot
    /// this job's `ready_gen` points at (for `sot_view_proj`'s promotion),
    /// but also carried here directly because `RenderCamera::set_cull_lock`
    /// / `write_cull_view_proj` (background-thread-owned) need the raw
    /// value, not a round trip through the GPU staging buffer.
    pub(crate) view_proj_cols: [f32; 16],
    /// Wall time `Scene::update(dt)` took on the main thread this frame —
    /// threaded through so `FrameStats` (which now lives on the background
    /// thread, alongside everything else it times) can keep reporting it.
    pub(crate) sim_update_ns: u64,
    /// Wall time this frame's harvest (`FrameHandoff::write_frame` →
    /// `harvest_trs_into_cpu_staging`, CPU→CPU: ECS → `CpuTrsStaging`) took
    /// on the main thread — same reporting rationale as `sim_update_ns`.
    pub(crate) cpu_staging_ns: u64,
}

/// Single-buffer producer/consumer handoff between the main thread (writer)
/// and the background render thread (reader). Access to `staging` strictly
/// alternates: the main thread only writes after `wait_for_buffer_free`
/// confirms the background thread is done with the previous handoff: the
/// background thread only reads after popping a [`FrameJob`] whose
/// `ready_gen` it's about to hand to `consume`. That alternation is what
/// makes sharing the raw `UnsafeCell` across threads sound — the same
/// argument already used for the `SyncMut<T>` wrapper the per-frame harvest
/// itself uses internally (see `harvest_trs_into_cpu_staging`).
pub(crate) struct FrameHandoff {
    staging: UnsafeCell<CpuTrsStaging>,
    /// Bumped by the main thread after it finishes writing a frame's CPU
    /// staging snapshot.
    ready_gen: AtomicU64,
    /// Bumped by the background thread after it finishes reading
    /// (memcpy-ing out of) that snapshot — the early-wake signal that
    /// lets the main thread overwrite the buffer again.
    consumed_gen: AtomicU64,
    /// Published by the background thread right after `ensure_capacity`
    /// grows `WorldTransformGpu`, so the main thread's next harvest knows
    /// how large a CPU buffer to grow to. One-frame lag by construction —
    /// see the module-level plan doc's "Capacity growth" section.
    entity_capacity: AtomicUsize,
    /// Current swapchain extent `[width, height]`, packed as
    /// `(width << 32) | height`. Published by the background thread on
    /// swapchain (re)creation; read by the main thread to compute this
    /// frame's aspect ratio for `view_proj` *before* that frame's acquire
    /// has even happened on the background thread (acquire doesn't
    /// determine the extent — every swapchain image shares it — so this is
    /// exact, not an approximation, except for the one-frame lag between a
    /// live resize landing and the background thread publishing the new
    /// extent).
    extent: AtomicU64,
}

// SAFETY: see the struct doc comment — access to `staging` strictly
// alternates between the two threads, never concurrent.
unsafe impl Sync for FrameHandoff {}

impl FrameHandoff {
    pub(crate) fn new(initial_capacity: usize, initial_extent: [u32; 2]) -> Self {
        let cap = initial_capacity.max(1);
        Self {
            staging: UnsafeCell::new(CpuTrsStaging::new(cap)),
            ready_gen: AtomicU64::new(0),
            consumed_gen: AtomicU64::new(0),
            entity_capacity: AtomicUsize::new(cap),
            extent: AtomicU64::new(pack_extent(initial_extent)),
        }
    }

    /// Main-thread-only. The GPU entity capacity as of the background
    /// thread's most recent `ensure_capacity` grow.
    pub(crate) fn published_entity_capacity(&self) -> usize {
        self.entity_capacity.load(Ordering::Acquire)
    }

    /// Background-thread-only. Call right after `ensure_capacity` grows.
    pub(crate) fn publish_entity_capacity(&self, cap: usize) {
        self.entity_capacity.store(cap, Ordering::Release);
    }

    /// Main-thread-only. Current swapchain extent, for this frame's aspect
    /// ratio.
    pub(crate) fn published_extent(&self) -> [u32; 2] {
        unpack_extent(self.extent.load(Ordering::Acquire))
    }

    /// Background-thread-only. Call on swapchain (re)creation.
    pub(crate) fn publish_extent(&self, extent: [u32; 2]) {
        self.extent.store(pack_extent(extent), Ordering::Release);
    }

    /// Main-thread-only. Block until the background thread has finished
    /// reading the *previous* handoff's CPU staging snapshot. Mirrors the
    /// spin/yield/sleep polling shape of
    /// `WorldTransformGpu::host_wait_for_previous_compute` — same
    /// rationale: cheap in the steady state where the background thread
    /// keeps up (single relaxed load), no kernel syscall on the hot path.
    ///
    /// First call returns immediately (`ready_gen`/`consumed_gen` both
    /// start at 0), same as that function's first-call behaviour.
    fn wait_for_buffer_free(&self) {
        let target = self.ready_gen.load(Ordering::Relaxed);
        let mut spins: u32 = 0;
        loop {
            if self.consumed_gen.load(Ordering::Acquire) == target {
                return;
            }
            spins += 1;
            if spins < 64 {
                std::hint::spin_loop();
            } else if spins < 2000 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        }
    }

    fn publish_ready(&self) -> u64 {
        let next = self.ready_gen.load(Ordering::Relaxed).wrapping_add(1);
        self.ready_gen.store(next, Ordering::Relaxed);
        next
    }

    fn mark_consumed(&self, ready_gen: u64) {
        self.consumed_gen.store(ready_gen, Ordering::Release);
    }

    /// Main-thread-only. Waits for the previous handoff to be consumed,
    /// grows the CPU staging buffer to `entity_capacity` if needed,
    /// harvests this frame's dirty TRS + `view_proj` into it, and returns
    /// `(ready_gen, cpu_staging_ns)` — the generation to stamp on this
    /// frame's [`FrameJob`], and the wall time the harvest itself took
    /// (CPU→CPU: ECS hierarchy → `CpuTrsStaging`), excluding the
    /// `wait_for_buffer_free` wait above and the capacity check.
    pub(crate) fn write_frame(
        &self,
        entity_capacity: usize,
        root_scene: Option<&Scene>,
        view_proj_cols: [f32; 16],
    ) -> (u64, u64) {
        self.wait_for_buffer_free();
        // SAFETY: `wait_for_buffer_free` just confirmed the background
        // thread is done reading `staging` (see struct doc comment).
        let staging = unsafe { &mut *self.staging.get() };
        staging.ensure_capacity(entity_capacity);
        let harvest_start = std::time::Instant::now();
        harvest_trs_into_cpu_staging(staging, root_scene, entity_capacity, view_proj_cols);
        let cpu_staging_ns = harvest_start.elapsed().as_nanos() as u64;
        (self.publish_ready(), cpu_staging_ns)
    }

    /// Background-thread-only. Memcpys the CPU staging snapshot into
    /// `world`'s GPU-visible staging `Subbuffer`s, writes the
    /// word-compaction prepass dispatch args from this frame's dirty-word
    /// watermarks, and signals the main thread that the buffer is free to
    /// reuse. Call once per popped [`FrameJob`], passing back its
    /// `ready_gen`. Returns the wall time of just the CPU→GPU memcpy
    /// (`copy_cpu_staging_into_gpu`), excluding the prepass-args write and
    /// the `mark_consumed` signal.
    pub(crate) fn consume(&self, ready_gen: u64, world: &WorldTransformGpu) -> u64 {
        // SAFETY: the caller just popped the `FrameJob` carrying
        // `ready_gen` off the handoff channel — the main thread won't
        // touch `staging` again until it observes `consumed_gen ==
        // ready_gen`, which only `mark_consumed` below sets.
        let staging = unsafe { &*self.staging.get() };
        let copy_start = std::time::Instant::now();
        copy_cpu_staging_into_gpu(staging, world);
        let cpu_gpu_staging_ns = copy_start.elapsed().as_nanos() as u64;
        world.write_prepass_dispatch_groups(staging.min_max_words);
        self.mark_consumed(ready_gen);
        cpu_gpu_staging_ns
    }
}

/// Harvest this frame's dirty TRS + `view_proj` into `staging`. Direct port
/// of the per-frame harvest that used to write straight into
/// `WorldTransformGpu`'s host-visible `Subbuffer`s: same parallel
/// bitmap-slab walk (one task per contiguous run of dirty-bitmask words,
/// shared task geometry with `Scene::update` via `parallel::bitmap_task_layout`),
/// same packed-half rotation encoding, same min/max dirty-word watermark
/// fold.
///
/// One behavioural difference from the ported code, required by the move to
/// a CPU buffer: the original wrote a dirty word into the GPU buffer only
/// when it was non-zero, relying on the GPU's own in-CB `vkCmdFillBuffer(0)`
/// (run every frame, right after the scatter consumes it) to keep
/// untouched words at zero. `staging`'s dirty buffers have no such in-CB
/// clear — they're plain heap memory reused frame to frame — so every word
/// visited by the walk (`[0, hier_words)`) is written *unconditionally*
/// (including zero) to avoid a "sticky" dirty bit that would otherwise
/// re-upload an unchanged entity forever after its one real change.
fn harvest_trs_into_cpu_staging(
    staging: &mut CpuTrsStaging,
    root_scene: Option<&Scene>,
    entity_capacity: usize,
    view_proj_cols: [f32; 16],
) {
    let dirty_words = dirty_word_count(entity_capacity);
    staging.view_proj = view_proj_cols;

    let pos = &mut staging.positions;
    let rot = &mut staging.rotations;
    let scl = &mut staging.scales;
    let dirty_pos = &mut staging.dirty_pos;
    let dirty_rot = &mut staging.dirty_rot;
    let dirty_scl = &mut staging.dirty_scl;

    let max_pos_word = atomic::AtomicI64::new(-1);
    let max_rot_word = atomic::AtomicI64::new(-1);
    let max_scl_word = atomic::AtomicI64::new(-1);
    let min_pos_word = atomic::AtomicI64::new(i64::MAX);
    let min_rot_word = atomic::AtomicI64::new(i64::MAX);
    let min_scl_word = atomic::AtomicI64::new(i64::MAX);

    if let Some(scene) = root_scene {
        let dirty = scene.transform_hierarchy.dirty();
        let pw = dirty.position_words();
        let rw = dirty.rotation_words();
        let sw = dirty.scale_words();
        let hier_words = pw.len().min(dirty_words);

        // Raw, lock-free SoA reads — sound because the scene's per-frame
        // `update` has already returned and this harvest is the sole
        // reader until the next `update` fires (same contract as the
        // original lib.rs harvest).
        let positions = scene.transform_hierarchy.positions_raw();
        let rotations = scene.transform_hierarchy.rotations_raw();
        let scales = scene.transform_hierarchy.scales_raw();
        let n = positions.len().min(entity_capacity);

        let bitmap_tasks = parallel::bitmap_task_layout(hier_words);
        let words_per_task = bitmap_tasks.words_per_task;

        struct SyncMut<T>(*mut T);
        unsafe impl<T> Send for SyncMut<T> {}
        unsafe impl<T> Sync for SyncMut<T> {}
        let pos_ptr = SyncMut(pos.as_mut_ptr());
        let rot_ptr = SyncMut(rot.as_mut_ptr());
        let scl_ptr = SyncMut(scl.as_mut_ptr());
        let dpos_ptr = SyncMut(dirty_pos.as_mut_ptr());
        let drot_ptr = SyncMut(dirty_rot.as_mut_ptr());
        let dscl_ptr = SyncMut(dirty_scl.as_mut_ptr());
        let pos_len = pos.len();
        let rot_len = rot.len();
        let scl_len = scl.len();
        let dpos_len = dirty_pos.len();
        let drot_len = dirty_rot.len();
        let dscl_len = dirty_scl.len();

        let per_word = |word_idx: usize| -> (bool, bool, bool) {
            let _ = (
                &pos_ptr, &rot_ptr, &scl_ptr, &dpos_ptr, &drot_ptr, &dscl_ptr,
            );
            let dp = pw[word_idx].swap(0, atomic::Ordering::Relaxed);
            let dr = rw[word_idx].swap(0, atomic::Ordering::Relaxed);
            let ds = sw[word_idx].swap(0, atomic::Ordering::Relaxed);
            // Unconditional — see this function's doc comment for why
            // (no in-CB clear on the CPU side).
            debug_assert!(word_idx < dpos_len && word_idx < drot_len && word_idx < dscl_len);
            unsafe {
                *dpos_ptr.0.add(word_idx) = dp;
                *drot_ptr.0.add(word_idx) = dr;
                *dscl_ptr.0.add(word_idx) = ds;
            }
            if (dp | dr | ds) == 0 {
                return (false, false, false);
            }
            let entity_base = word_idx * 32;
            if dp != 0 {
                let mut bits = dp;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let entity = entity_base + bit;
                    if entity >= n {
                        break;
                    }
                    let p = positions[entity];
                    let base = entity * 3;
                    debug_assert!(base + 2 < pos_len);
                    unsafe {
                        *pos_ptr.0.add(base) = p.x;
                        *pos_ptr.0.add(base + 1) = p.y;
                        *pos_ptr.0.add(base + 2) = p.z;
                    }
                }
            }
            if dr != 0 {
                let mut bits = dr;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let entity = entity_base + bit;
                    if entity >= n {
                        break;
                    }
                    let q = rotations[entity];
                    let packed = transform_gpu::pack_quat_half(q);
                    let base = entity * 2;
                    debug_assert!(base + 1 < rot_len);
                    unsafe {
                        *rot_ptr.0.add(base) = f32::from_bits(packed[0]);
                        *rot_ptr.0.add(base + 1) = f32::from_bits(packed[1]);
                    }
                }
            }
            if ds != 0 {
                let mut bits = ds;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let entity = entity_base + bit;
                    if entity >= n {
                        break;
                    }
                    let s = scales[entity];
                    let base = entity * 3;
                    debug_assert!(base + 2 < scl_len);
                    unsafe {
                        *scl_ptr.0.add(base) = s.x;
                        *scl_ptr.0.add(base + 1) = s.y;
                        *scl_ptr.0.add(base + 2) = s.z;
                    }
                }
            }
            (dp != 0, dr != 0, ds != 0)
        };

        let n_tasks = bitmap_tasks.n_tasks;
        parallel::global::parallel_for(0..n_tasks, |task_range| {
            let mut local_max_pos: i64 = -1;
            let mut local_max_rot: i64 = -1;
            let mut local_max_scl: i64 = -1;
            let mut local_min_pos: i64 = i64::MAX;
            let mut local_min_rot: i64 = i64::MAX;
            let mut local_min_scl: i64 = i64::MAX;
            for task_idx in task_range {
                let word_base = task_idx * words_per_task;
                let word_end = (word_base + words_per_task).min(hier_words);
                for word_idx in word_base..word_end {
                    let (touched_pos, touched_rot, touched_scl) = per_word(word_idx);
                    if touched_pos {
                        local_max_pos = word_idx as i64;
                        if local_min_pos == i64::MAX {
                            local_min_pos = word_idx as i64;
                        }
                    }
                    if touched_rot {
                        local_max_rot = word_idx as i64;
                        if local_min_rot == i64::MAX {
                            local_min_rot = word_idx as i64;
                        }
                    }
                    if touched_scl {
                        local_max_scl = word_idx as i64;
                        if local_min_scl == i64::MAX {
                            local_min_scl = word_idx as i64;
                        }
                    }
                }
            }
            if local_max_pos >= 0 {
                max_pos_word.fetch_max(local_max_pos, atomic::Ordering::Relaxed);
                min_pos_word.fetch_min(local_min_pos, atomic::Ordering::Relaxed);
            }
            if local_max_rot >= 0 {
                max_rot_word.fetch_max(local_max_rot, atomic::Ordering::Relaxed);
                min_rot_word.fetch_min(local_min_rot, atomic::Ordering::Relaxed);
            }
            if local_max_scl >= 0 {
                max_scl_word.fetch_max(local_max_scl, atomic::Ordering::Relaxed);
                min_scl_word.fetch_min(local_min_scl, atomic::Ordering::Relaxed);
            }
        });
    } else if !dirty_pos.is_empty() {
        // Legacy fallback: identity at slot 0 (no scene attached at all —
        // see `crate::Window`'s doc example). Idempotent every frame.
        pos[0..3].copy_from_slice(&[0.0, 0.0, 0.0]);
        let packed = transform_gpu::pack_quat_half(glam::Quat::IDENTITY);
        rot[0] = f32::from_bits(packed[0]);
        rot[1] = f32::from_bits(packed[1]);
        scl[0..3].copy_from_slice(&[1.0, 1.0, 1.0]);
        dirty_pos[0] = 1;
        dirty_rot[0] = 1;
        dirty_scl[0] = 1;
        max_pos_word.store(0, atomic::Ordering::Relaxed);
        max_rot_word.store(0, atomic::Ordering::Relaxed);
        max_scl_word.store(0, atomic::Ordering::Relaxed);
        min_pos_word.store(0, atomic::Ordering::Relaxed);
        min_rot_word.store(0, atomic::Ordering::Relaxed);
        min_scl_word.store(0, atomic::Ordering::Relaxed);
    }

    staging.min_max_words = [
        (
            min_pos_word.load(atomic::Ordering::Relaxed),
            max_pos_word.load(atomic::Ordering::Relaxed),
        ),
        (
            min_rot_word.load(atomic::Ordering::Relaxed),
            max_rot_word.load(atomic::Ordering::Relaxed),
        ),
        (
            min_scl_word.load(atomic::Ordering::Relaxed),
            max_scl_word.load(atomic::Ordering::Relaxed),
        ),
    ];
}

/// Total participant count (background thread + workers) for
/// [`staging_copy_pool`]'s dedicated pool, unless overridden by
/// `ENGINE_STAGING_COPY_THREADS`. Same "total participants" convention as
/// `ENGINE_NUM_THREADS` (see `lib.rs::init_pinned_thread_pool`).
const DEFAULT_STAGING_COPY_THREADS: usize = 8;

/// A dedicated `my_thread_pool::ThreadPool` for [`copy_cpu_staging_into_gpu`]
/// only — deliberately **not** the shared `parallel::global` pool the
/// harvest and sim update use. That pool only allows one non-worker
/// ("external") thread to call `parallel_for` at a time
/// (`my_thread_pool::ThreadPool::parallel_for`'s `external_active` guard);
/// the main thread is already an external caller of it every frame (the
/// TRS harvest, `Scene::update`), and the background render thread is a
/// plain dedicated OS thread (see `render_thread::RenderThreadHandle::spawn`),
/// not a pool worker — so it would be a *second* external caller and could
/// race the main thread's dispatches. A separate pool sidesteps that
/// entirely: this one is only ever driven by the render thread, one
/// dispatch at a time, never concurrently with anything else. Lazily built
/// on first use and reused for the process's lifetime.
///
/// Sized from `ENGINE_STAGING_COPY_THREADS` (total participants, i.e. the
/// calling thread + workers — same convention as `ENGINE_NUM_THREADS`),
/// defaulting to [`DEFAULT_STAGING_COPY_THREADS`]. Parsed strictly: an
/// unset/empty var takes the default, any other unparseable value panics
/// rather than silently falling back.
pub(crate) fn staging_copy_pool() -> &'static my_thread_pool::ThreadPool {
    static POOL: OnceLock<my_thread_pool::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = match std::env::var("ENGINE_STAGING_COPY_THREADS") {
            Ok(s) if !s.is_empty() => s
                .parse::<usize>()
                .expect("ENGINE_STAGING_COPY_THREADS must be a positive integer"),
            _ => DEFAULT_STAGING_COPY_THREADS,
        };
        assert!(n >= 1, "ENGINE_STAGING_COPY_THREADS must be >= 1");
        my_thread_pool::ThreadPool::new(n)
    })
}

/// Copy only the entities `cpu_dirty` marks dirty this frame — mirrors
/// `harvest_trs_into_cpu_staging`'s per-word dirty-bit walk (see that
/// function's doc comment), just for the CPU→GPU direction instead of
/// ECS→CPU. Scans `[min_word, max_word]` (this component's watermark for
/// the frame — `copy_cpu_staging_into_gpu` skips the call entirely when
/// `max_word < 0`, i.e. untouched), and for every non-zero dirty word:
/// copies the word itself into `gpu_dirty`, then copies only the set-bit
/// entities' `stride`-wide component values from `cpu_values` into
/// `gpu_values`. Zero words inside the span (the span can have gaps — see
/// `WorldTransformGpu::write_prepass_dispatch_groups`'s doc comment) are
/// skipped entirely, not zero-written: the GPU dirty buffer is already
/// zero there (cleared every frame by the in-CB `vkCmdFillBuffer(0)` right
/// after the previous frame's scatter consumes it — see
/// `transform_gpu`'s module doc comment), so there is nothing to promote.
///
/// This is why a blind full-buffer copy was wasteful: entities that didn't
/// change this frame keep whatever value is already sitting in their GPU
/// staging slot (never read anyway, since the scatter only promotes
/// staging → SoT for entities whose dirty bit is set) — copying it again
/// is pure bandwidth with no effect.
///
/// Dispatched on [`staging_copy_pool`], one task range per contiguous
/// chunk of `[min_word, max_word]`; `gpu_dirty`/`gpu_values` are shared
/// across worker closures as raw pointers (`SyncMut`) because each word
/// index is visited by exactly one task, so writes into disjoint entities'
/// slots never alias — the same soundness argument as the harvest's own
/// `SyncMut` use.
fn copy_dirty_component(
    pool: &my_thread_pool::ThreadPool,
    min_max_word: (i64, i64),
    cpu_dirty: &[u32],
    cpu_values: &[f32],
    gpu_dirty: &mut [u32],
    gpu_values: &mut [f32],
    stride: usize,
    entity_capacity: usize,
) {
    let (min_word, max_word) = min_max_word;
    if max_word < 0 {
        return;
    }
    let min_word = min_word as usize;
    let max_word = max_word as usize;

    struct SyncMut<T>(*mut T);
    unsafe impl<T> Send for SyncMut<T> {}
    unsafe impl<T> Sync for SyncMut<T> {}
    let gpu_dirty_ptr = SyncMut(gpu_dirty.as_mut_ptr());
    let gpu_values_ptr = SyncMut(gpu_values.as_mut_ptr());

    pool.parallel_for(min_word..max_word + 1, |word_range| {
        let _ = (&gpu_dirty_ptr, &gpu_values_ptr);
        for word_idx in word_range {
            let word = cpu_dirty[word_idx];
            if word == 0 {
                continue;
            }
            // SAFETY: each word index is visited by exactly one task.
            unsafe {
                *gpu_dirty_ptr.0.add(word_idx) = word;
            }
            let entity_base = word_idx * 32;
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let entity = entity_base + bit;
                if entity >= entity_capacity {
                    break;
                }
                let base = entity * stride;
                // SAFETY: disjoint per-entity ranges, same argument as above.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        cpu_values.as_ptr().add(base),
                        gpu_values_ptr.0.add(base),
                        stride,
                    );
                }
            }
        }
    });
}

/// Background-thread-only. Promote `staging`'s dirty entities into
/// `world`'s GPU-visible staging `Subbuffer`s, in parallel on
/// [`staging_copy_pool`]'s dedicated pool — see [`copy_dirty_component`]
/// for why this walks dirty bits instead of copying whole buffers.
fn copy_cpu_staging_into_gpu(staging: &CpuTrsStaging, world: &WorldTransformGpu) {
    let pool = staging_copy_pool();
    let entity_capacity = staging.capacity;
    {
        let mut w = world
            .staging_positions()
            .write()
            .expect("staging_positions.write");
        let mut d = world
            .staging_dirty_pos()
            .write()
            .expect("staging_dirty_pos.write");
        copy_dirty_component(
            pool,
            staging.min_max_words[0],
            &staging.dirty_pos,
            &staging.positions,
            &mut d[..staging.dirty_pos.len()],
            &mut w[..staging.positions.len()],
            3,
            entity_capacity,
        );
    }
    {
        let mut w = world
            .staging_rotations()
            .write()
            .expect("staging_rotations.write");
        let mut d = world
            .staging_dirty_rot()
            .write()
            .expect("staging_dirty_rot.write");
        copy_dirty_component(
            pool,
            staging.min_max_words[1],
            &staging.dirty_rot,
            &staging.rotations,
            &mut d[..staging.dirty_rot.len()],
            &mut w[..staging.rotations.len()],
            2,
            entity_capacity,
        );
    }
    {
        let mut w = world
            .staging_scales()
            .write()
            .expect("staging_scales.write");
        let mut d = world
            .staging_dirty_scl()
            .write()
            .expect("staging_dirty_scl.write");
        copy_dirty_component(
            pool,
            staging.min_max_words[2],
            &staging.dirty_scl,
            &staging.scales,
            &mut d[..staging.dirty_scl.len()],
            &mut w[..staging.scales.len()],
            3,
            entity_capacity,
        );
    }
    {
        // Single mat4, always current (the camera moves basically every
        // frame) — not worth a dirty-bit gate.
        let mut w = world.view_proj_buf().write().expect("view_proj_buf.write");
        w[0] = staging.view_proj;
    }
}
