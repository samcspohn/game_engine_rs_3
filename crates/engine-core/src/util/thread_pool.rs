//! Static fork-join thread pool with collaborative main-thread execution.
//!
//! Replaces rayon for the engine's hot-path workloads. These are all
//! **uniform, fork-join, per-frame** patterns with no nested parallelism:
//!
//! * `ComponentStorage::par_iter`   — bitset walk
//! * `TransformHierarchy` dirty-harvest into staging
//! * `engine_render::WorldTransformGpu` host-staging `par_chunks_mut`
//! * `engine_render::build_all_frame_slots`
//!
//! ## Design
//!
//! * One worker thread per non-main core, pinned via a caller-supplied
//!   start hook (typically `core_affinity::set_for_current`). The main
//!   thread is **always** a participant — work is split across
//!   `num_workers + 1` threads, of which the main thread runs its share
//!   inline. NUMA topology is the caller's concern (used only for worker
//!   pinning); the pool itself is node-agnostic.
//! * One global pool installed once via [`init_global`].
//! * One primitive: [`ThreadPool::parallel_for`]. **Chunk + tail-steal.**
//!   `0..n_tasks` is split into one contiguous slice per participant;
//!   participant `p` is *deterministically* seeded with the `p`-th slice
//!   for a given `n_tasks`, so the worker that touched a slice last frame
//!   touches it again this frame (cache/page locality without explicit
//!   NUMA binding — the OS migrates pages to the touching node). When a
//!   participant drains its own slice it steals the back half of a
//!   neighbour's [`Cursor`] (rotation order via pre-built `steal_order`),
//!   recovering throughput when a worker is OS-preempted mid-frame.
//! * **Worker idle policy.** Three-phase loop:
//!   pure `spin_loop` for [`HOT_SPIN_ITERS`], then
//!   `sleep(Duration::from_nanos(1))` (kernel hrtimer, ~50 µs
//!   ride-out) up to [`PARK_AFTER_SPINS`], then
//!   `parking_lot_core::park` keyed on the worker's parking-shard
//!   address. Dispatch wakes via `unpark_all(key)` — one syscall per
//!   shard regardless of how many workers are parked on it. The middle
//!   `sleep(1ns)` phase is crucial for steady-state throughput: between
//!   dispatches workers sit in the kernel's hrtimer queue rather than
//!   the parking_lot bucket, so main's `unpark_all` is a cheap no-op
//!   token-store rather than a real futex-wake syscall.
//! * No nested parallelism. `parallel_for` called from inside a worker
//!   panics — per project rules, no silent serial fallback. Callers that
//!   need recursive descent must walk sequentially inside a task.
//!
//! ## Safety
//!
//! `parallel_for` publishes a stack-allocated [`Job`] (and a stack
//! `cursors` buffer) via an [`UnsafeCell`] under release/acquire on
//! `epoch`. The borrow lives for the duration of the call; the function
//! blocks until every active worker has signalled completion
//! (`workers_done == active_workers`), so the borrow cannot dangle.

use std::cell::UnsafeCell;
use std::hint::spin_loop;

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::thread::{self, JoinHandle, Thread};
#[cfg(feature = "pool-timing")]
use std::time::Instant;

// ────────────────────────────────────────────────────────────────────────
// Cacheline padding
// ────────────────────────────────────────────────────────────────────────

/// 128-byte aligned wrapper. 128 (not 64) because x86 spatial prefetchers
/// pull adjacent cache-line pairs, so packing two unrelated hot atomics
/// into adjacent 64-byte lines still produces ping-pong traffic. Matches
/// `crossbeam_utils::CachePadded` on x86_64.
#[repr(align(128))]
struct CachePadded<T> {
    value: T,
}

impl<T> CachePadded<T> {
    const fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> std::ops::Deref for CachePadded<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.value
    }
}

// ────────────────────────────────────────────────────────────────────────
// Per-worker timing slots
// ────────────────────────────────────────────────────────────────────────

/// Per-worker timing slot. Cache-line aligned so neighbouring workers'
/// stores to their own timestamps don't ping-pong cache lines.
///
/// `t_seen` is set by the worker the instant it observes a new epoch
/// (before invoking the job thunk). `t_done` is set right before the
/// worker increments `workers_done`. Both are ns since the pool's
/// [`Shared::anchor`] instant.
///
/// Always present even when the `pool-timing` feature is off so the
/// `Shared` layout stays stable; the worker loop only writes to them
/// under the feature gate.
#[repr(align(64))]
struct WorkerTs {
    t_seen: AtomicU64,
    t_done: AtomicU64,
}

impl WorkerTs {
    const fn new() -> Self {
        Self {
            t_seen: AtomicU64::new(0),
            t_done: AtomicU64::new(0),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Work-stealing cursors
// ────────────────────────────────────────────────────────────────────────

/// Grain size for own-cursor consumption. ~hundreds of ns/task ×
/// 256 ≈ tens-of-µs per chunk — large enough to amortise the
/// per-chunk `fetch_add`, small enough that a slow worker's tail
/// can be usefully split by stealers.
const STEAL_GRAIN: usize = 256;

/// A `[start, end)` range packed into a **single** `AtomicU64`
/// (`start` in the high 32 bits, `end` in the low 32 bits).
///
/// The owner advances `start` and thieves lower `end`. Storing them in
/// one word and CAS-ing the whole word is essential for correctness: a
/// split `start`/`end` pair (one atomic each) lets the owner read a
/// stale `end` after its `start` claim while a thief lowers `end` from a
/// stale `start`, so their claimed ranges overlap at the boundary and
/// those tasks get processed twice. The packed CAS serialises the two
/// ends, so owner-claim and steal never disagree.
///
/// Task indices must fit in `u32` (they always do for the engine's
/// bitmap-word / per-entity workloads).
///
/// Cacheline-padded (via [`CachePadded`]) so workers' high-frequency
/// CAS traffic on their own cursor doesn't ping-pong adjacent cursors'
/// cache lines.
struct Cursor {
    packed: AtomicU64,
}

type PaddedCursor = CachePadded<Cursor>;

#[inline(always)]
fn pack_cursor(start: u32, end: u32) -> u64 {
    ((start as u64) << 32) | (end as u64)
}

#[inline(always)]
fn unpack_cursor(v: u64) -> (u32, u32) {
    ((v >> 32) as u32, (v & 0xFFFF_FFFF) as u32)
}

impl Cursor {
    fn new_range(s: usize, e: usize) -> Self {
        // Well-formed empty cursor when the seed is degenerate (s >= e,
        // e.g. a participant whose chunk overran n_tasks).
        let (s, e) = if s < e { (s, e) } else { (0, 0) };
        debug_assert!(e <= u32::MAX as usize, "task index exceeds u32 range");
        Self {
            packed: AtomicU64::new(pack_cursor(s as u32, e as u32)),
        }
    }
}

/// Attempt to steal the back half of `c`. Returns the stolen
/// `[start, end)` (which the caller processes directly) or `None` if the
/// cursor has too little work to bother splitting. The CAS is on the
/// whole packed word, so it cannot race with the owner's claim. Bounded
/// by other stealers shrinking `end`, so it terminates.
#[inline]
fn try_steal(c: &PaddedCursor) -> Option<(usize, usize)> {
    loop {
        let packed = c.packed.load(Ordering::Acquire);
        let (s, e) = unpack_cursor(packed);
        if e <= s || (e - s) < STEAL_GRAIN as u32 {
            return None;
        }
        // `+1` so odd remainder stays with the owner; avoids
        // pointless steals of single trailing tasks.
        let new_end = s + (e - s + 1) / 2;
        if new_end >= e || new_end <= s {
            return None;
        }
        if c.packed
            .compare_exchange_weak(
                packed,
                pack_cursor(s, new_end),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            return Some((new_end as usize, e as usize));
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Job slot
// ────────────────────────────────────────────────────────────────────────

/// Type-erased, lifetime-erased descriptor of a parallel-for job.
///
/// Worker `i` owns `cursors[i]`; main owns `cursors[num_workers]`.
/// Each participant consumes its own cursor in `STEAL_GRAIN` chunks via
/// `start.fetch_add`, then steals the back half of a neighbour's cursor
/// via CAS on `end`. The cursors buffer is stack-allocated for the
/// duration of the dispatch call.
///
/// `invoke_range` runs `for i in start..end { f(i) }` inside a
/// monomorphised thunk so `f(i)` inlines fully into the worker loop.
#[derive(Clone, Copy)]
struct Job {
    /// Workers 0..active_workers participate and contribute to the barrier.
    /// Workers >= active_workers observe the epoch but do no work.
    active_workers: usize,
    data: *const (),
    invoke_range: unsafe fn(*const (), usize, usize),
    /// Points to `n_cursors` `PaddedCursor`s; stack-allocated by main.
    /// Worker `w` owns `cursors[w]`; `cursors[num_workers]` is main's slot.
    cursors: *const PaddedCursor,
    /// Length of the cursors buffer (== num_workers + 1).
    n_cursors: usize,
}

unsafe impl Sync for Job {}
unsafe impl Send for Job {}

// ────────────────────────────────────────────────────────────────────────
// Shared state
// ────────────────────────────────────────────────────────────────────────

struct Shared {
    /// Bumped every time main publishes a new job. Workers compare against
    /// their `last_epoch` to detect new work. Release/Acquire here
    /// synchronises the non-atomic write to `job`. Cacheline-padded
    /// because workers spin-load this on the hot path; without padding
    /// the adjacent `workers_done` store-storm at barrier completion
    /// invalidates the line every worker is polling.
    epoch: CachePadded<AtomicU64>,

    /// Number of workers that have finished their assigned slice for the
    /// current job. Main blocks until this reaches `num_workers`. Reset
    /// to 0 before each new epoch is published. Cacheline-padded
    /// because every worker fetch_adds at barrier-end while every other
    /// worker (and main) is reading `epoch`/spinning.
    workers_done: CachePadded<AtomicUsize>,

    /// Job descriptor. Written by main BEFORE bumping `epoch`. Read by
    /// workers AFTER observing the new epoch. Never written while any
    /// worker is still in its slice (barrier on `workers_done`).
    job: UnsafeCell<Job>,

    /// Set on shutdown. Workers exit their outer loop and the JoinHandles
    /// are joined.
    shutdown: AtomicBool,

    /// Diagnostic mode (enabled via `ENGINE_POOL_VERIFY=1`). When set,
    /// every participant atomically increments [`Shared::tasks_invoked`]
    /// once per `f(task_idx)` invocation. After the barrier, main
    /// compares against the expected `n_tasks` and panics if they
    /// disagree — i.e. a worker silently skipped its slice.
    verify_mode: AtomicBool,

    /// Per-job task-invocation counter for verify mode. Reset to 0 at
    /// the start of each parallel_for, incremented (Relaxed) inside
    /// the dispatcher thunk on every `f` call. Compared against
    /// `n_tasks` after the barrier when `verify_mode` is on.
    tasks_invoked: AtomicUsize,

    /// Per-worker timing slots, indexed by `worker_idx`. Populated by
    /// workers only when the `pool-timing` feature is enabled; main
    /// reads them after the barrier to derive per-worker wake-up and
    /// work-duration statistics. Sized to `num_workers` at pool init.
    worker_ts: Vec<WorkerTs>,

    /// Reference instant for converting per-worker timestamps to ns.
    /// Captured at `init_global` time. Always present (Instant is
    /// cheap to store) so the layout matches with/without
    /// `pool-timing`; only read in the timing-feature paths.
    #[cfg(feature = "pool-timing")]
    anchor: Instant,

    /// Sharded park keys. One entry per shard; each entry's address is a
    /// `parking_lot_core` key. Sharding spreads parked workers across
    /// distinct internal bucket mutexes, avoiding a single-bucket
    /// bottleneck at high thread counts. Shard index for worker w is
    /// `worker_shard[w]`. The byte value is unused; only address matters.
    park_keys: Vec<CachePadded<u8>>,
    /// Per-worker index into `park_keys` (precomputed at init,
    /// = worker_idx / ENGINE_PARK_SHARD_SIZE).
    worker_shard: Vec<u32>,
    /// `steal_order[p]` lists the other participant indices that
    /// participant `p` tries to steal from when its own cursor is
    /// exhausted. Rotation order: p+1, p+2, ..., p-1 (wrapping).
    steal_order: Vec<Vec<usize>>,
}

unsafe impl Sync for Shared {}

// ────────────────────────────────────────────────────────────────────────
// Pool
// ────────────────────────────────────────────────────────────────────────

pub struct ThreadPool {
    shared: &'static Shared,
    /// Retained so the worker threads stay alive for the program's
    /// lifetime. Used to hold Thread handles from the registration
    /// handshake; not used for dispatch (wake is via parking_lot_core).
    #[allow(dead_code)]
    worker_threads: Vec<Thread>,
    /// Kept alive so the worker threads stay joinable. We never join
    /// (the pool is `'static`; shutdown is a process-exit).
    #[allow(dead_code)]
    handles: Vec<JoinHandle<()>>,
    /// Total number of participants in a `parallel_for` call =
    /// `worker_threads.len() + 1` (main thread always participates).
    num_participants: usize,
}

static POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Pool initialisation config.
///
/// `num_workers` is the number of background worker threads to spawn
/// (in addition to the calling thread, which becomes the main participant).
/// CPU pinning is entirely the caller's responsibility inside the
/// `on_worker_start` closure passed to [`init_global`].
pub struct PoolConfig {
    pub num_workers: usize,
}

/// Initialise the global thread pool.
///
/// Spawns `cfg.num_workers` background threads. `on_worker_start(idx)`
/// is invoked once on each newly-spawned worker before it enters its
/// idle loop — use it for CPU pinning / affinity or other per-worker
/// setup. The calling thread becomes the main participant.
///
/// Panics if called more than once or if any worker fails to spawn.
/// No fallbacks.
pub fn init_global<F>(cfg: PoolConfig, on_worker_start: F)
where
    F: Fn(usize) + Send + Sync + 'static,
{
    assert!(
        cfg.num_workers > 0,
        "ThreadPool requires at least one worker",
    );

    let num_workers = cfg.num_workers;
    let num_participants = num_workers + 1; // workers + main

    // Rotation steal order: participant p tries p+1, p+2, ..., p-1 (wrapping).
    let mut steal_order: Vec<Vec<usize>> = Vec::with_capacity(num_participants);
    for p in 0..num_participants {
        let order: Vec<usize> = (1..num_participants)
            .map(|offset| (p + offset) % num_participants)
            .collect();
        steal_order.push(order);
    }

    // Park sharding: node-agnostic, shard by global worker index.
    // Spreading workers across multiple parking_lot bucket keys avoids
    // a single-bucket bottleneck at high thread counts.
    let park_shard_size = std::env::var("ENGINE_PARK_SHARD_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_PARK_SHARD_SIZE);
    let num_shards = ((num_workers + park_shard_size - 1) / park_shard_size).max(1);
    let park_keys: Vec<CachePadded<u8>> = (0..num_shards).map(|_| CachePadded::new(0u8)).collect();
    let worker_shard: Vec<u32> = (0..num_workers)
        .map(|w| (w / park_shard_size) as u32)
        .collect();

    let shared: &'static Shared = Box::leak(Box::new(Shared {
        epoch: CachePadded::new(AtomicU64::new(0)),
        workers_done: CachePadded::new(AtomicUsize::new(0)),
        job: UnsafeCell::new(Job {
            active_workers: 0,
            data: std::ptr::null(),
            invoke_range: noop_invoke_range,
            cursors: std::ptr::null(),
            n_cursors: 0,
        }),
        shutdown: AtomicBool::new(false),
        verify_mode: AtomicBool::new(
            std::env::var("ENGINE_POOL_VERIFY")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
        ),
        tasks_invoked: AtomicUsize::new(0),
        worker_ts: (0..num_workers).map(|_| WorkerTs::new()).collect(),
        #[cfg(feature = "pool-timing")]
        anchor: Instant::now(),
        park_keys,
        worker_shard,
        steal_order,
    }));

    let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Thread)>(num_workers);
    let on_start = std::sync::Arc::new(on_worker_start);

    let mut handles = Vec::with_capacity(num_workers);
    for idx in 0..num_workers {
        let tx = tx.clone();
        let on_start = on_start.clone();
        let handle = thread::Builder::new()
            .name(format!("engine-worker-{idx}"))
            .spawn(move || {
                on_start(idx);
                tx.send((idx, thread::current()))
                    .expect("worker failed to register Thread handle");
                drop(tx);
                worker_loop(shared, idx);
            })
            .expect("failed to spawn engine worker thread");
        handles.push(handle);
    }
    drop(tx);

    // Receive in arrival order, then sort into worker-idx order so
    // `worker_threads[i]` is the handle of worker `i`.
    let mut received: Vec<(usize, Thread)> = (0..num_workers)
        .map(|_| rx.recv().expect("worker thread failed to register"))
        .collect();
    received.sort_by_key(|(i, _)| *i);
    let worker_threads: Vec<Thread> = received.into_iter().map(|(_, t)| t).collect();

    let pool = ThreadPool {
        shared,
        worker_threads,
        handles,
        num_participants,
    };

    POOL.set(pool)
        .ok()
        .expect("engine ThreadPool already initialised");
}

/// Test-only convenience: install the global pool (if not already
/// installed) and return a guard that serialises `parallel_for` calls
/// across the entire test binary. The pool is single-producer by design
/// (the engine's main thread is the only `parallel_for` caller in
/// production); cargo's default test harness runs `#[test]` functions
/// on multiple threads concurrently, which would violate that invariant
/// across test modules, so every test takes this guard before touching
/// the pool.
#[cfg(test)]
pub(crate) fn lock_for_test() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::Mutex;
    static GATE: OnceLock<Mutex<()>> = OnceLock::new();
    let m = GATE.get_or_init(|| Mutex::new(()));
    let g = m.lock().unwrap_or_else(|p| p.into_inner());
    if !is_initialised() {
        init_global(PoolConfig { num_workers: 4 }, |_| {});
    }
    g
}

/// Borrow the global pool. Panics if not initialised.
#[inline]
pub fn global() -> &'static ThreadPool {
    POOL.get()
        .expect("engine ThreadPool not initialised — call util::thread_pool::init_global first")
}

/// Whether the global pool has been initialised.
#[inline]
pub fn is_initialised() -> bool {
    POOL.get().is_some()
}

/// Per-call timing breakdown returned by [`ThreadPool::parallel_for`].
///
/// All fields are wall-clock nanoseconds measured on the main thread.
/// * `dispatch_ns` — from entry to after all worker `unpark()` calls.
///   Includes job setup, epoch publish, and the futex wake per parked worker.
/// * `main_work_ns` — main thread's own slice execution time.
/// * `barrier_ns` — spin-wait from after main's slice until all workers
///   signal completion.
///
/// Note: for `n_tasks <= 1` the fast-path skips dispatch and barrier
/// entirely, so `dispatch_ns` and `barrier_ns` will both be 0 and
/// `main_work_ns` covers the entire call (just `f(0)` for n_tasks == 1,
/// or nothing for n_tasks == 0).
///
/// **Timing is gated on the `pool-timing` Cargo feature.** When the
/// feature is off (the default), every field except `n_tasks` is always
/// `0` and `parallel_for` skips every `Instant::now()` call in the hot
/// path. Downstream crates that consume the breakdown (e.g. the
/// renderer's per-frame staging-walk attribution) must enable the
/// feature on their `engine-core` dependency.
#[derive(Debug, Default, Clone, Copy)]
pub struct DispatchTiming {
    pub n_tasks: usize,
    pub active_workers: usize,
    pub dispatch_ns: u64,
    pub main_work_ns: u64,
    pub barrier_ns: u64,
    /// Min/avg/max across participating workers of `t_seen - t_publish`:
    /// wall time from when main published the new epoch to when the
    /// worker actually observed it and started executing. Captures
    /// wake-up / spin-detection latency variance. Only populated when
    /// the `pool-timing` feature is on.
    pub worker_wake_min_ns: u64,
    pub worker_wake_avg_ns: u64,
    pub worker_wake_max_ns: u64,
    /// Min/avg/max across participating workers of `t_done - t_seen`:
    /// pure work time excluding wake-up. Only populated when the
    /// `pool-timing` feature is on.
    pub worker_work_min_ns: u64,
    pub worker_work_avg_ns: u64,
    pub worker_work_max_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitmapTaskLayout {
    pub words_per_task: usize,
    pub n_tasks: usize,
}

impl BitmapTaskLayout {
    #[inline]
    pub const fn entities_per_task(self) -> usize {
        self.words_per_task * 32
    }
}

const BITMAP_MIN_WORDS_PER_TASK: usize = 8;
const BITMAP_MAX_WORDS_PER_TASK: usize = 256;
const BITMAP_TARGET_TASKS_PER_THREAD: usize = 2;

/// Reset per-worker timestamp slots ahead of a new dispatch and capture
/// the publish timestamp. After the barrier, [`aggregate_worker_stats`]
/// reads back the slots and derives (wake = t_seen - t_publish,
/// work = t_done - t_seen) for every participating worker.
///
/// `participants` is the set of worker indices expected to record
/// timestamps for this dispatch (uniform-chunk mode passes
/// `0..active_workers`; per-worker-slices mode passes `0..num_workers`).
///
/// Only invoked under the `pool-timing` feature.
#[cfg(feature = "pool-timing")]
fn reset_worker_ts(shared: &Shared, participants: std::ops::Range<usize>) -> u64 {
    for w in participants {
        shared.worker_ts[w].t_seen.store(0, Ordering::Relaxed);
        shared.worker_ts[w].t_done.store(0, Ordering::Relaxed);
    }
    shared.anchor.elapsed().as_nanos() as u64
}

/// Aggregate per-worker timestamps into min/avg/max wake & work stats.
///
/// Workers with `t_done == 0` are treated as inactive for this dispatch
/// (uniform-chunk mode leaves the trailing workers idle). Workers with
/// `t_done != 0` but `t_seen <= t_publish` (clock noise / racy reads)
/// contribute a wake time of 0 µs.
///
/// Returns `(wake_min, wake_avg, wake_max, work_min, work_avg, work_max)`
/// in ns. All zero if no worker participated.
#[cfg(feature = "pool-timing")]
fn aggregate_worker_stats(
    shared: &Shared,
    participants: std::ops::Range<usize>,
    t_publish_ns: u64,
) -> (u64, u64, u64, u64, u64, u64) {
    let mut wake_min = u64::MAX;
    let mut wake_max = 0u64;
    let mut wake_sum = 0u128;
    let mut work_min = u64::MAX;
    let mut work_max = 0u64;
    let mut work_sum = 0u128;
    let mut n = 0u64;
    for w in participants {
        let seen = shared.worker_ts[w].t_seen.load(Ordering::Relaxed);
        let done = shared.worker_ts[w].t_done.load(Ordering::Relaxed);
        if done == 0 {
            // Inactive in uniform-chunk mode, or never reached the
            // done store. Skip.
            continue;
        }
        let wake = seen.saturating_sub(t_publish_ns);
        let work = done.saturating_sub(seen.max(t_publish_ns));
        if wake < wake_min {
            wake_min = wake;
        }
        if wake > wake_max {
            wake_max = wake;
        }
        wake_sum += wake as u128;
        if work < work_min {
            work_min = work;
        }
        if work > work_max {
            work_max = work;
        }
        work_sum += work as u128;
        n += 1;
    }
    if n == 0 {
        return (0, 0, 0, 0, 0, 0);
    }
    let wake_avg = (wake_sum / n as u128) as u64;
    let work_avg = (work_sum / n as u128) as u64;
    (wake_min, wake_avg, wake_max, work_min, work_avg, work_max)
}

/// Choose a shared task layout for bitmap-indexed per-entity work.
///
/// The simulation and staging paths both walk transform-indexed bitmaps, so
/// they share this layout helper to keep worker/data slabs consistent across
/// those adjacent phases while still scaling the slab size with the pool width.
#[inline]
pub fn bitmap_task_layout(n_words: usize) -> BitmapTaskLayout {
    if n_words == 0 {
        return BitmapTaskLayout {
            words_per_task: BITMAP_MIN_WORDS_PER_TASK,
            n_tasks: 0,
        };
    }
    let target_tasks = crate::util::numa_pool::global::pool()
        .num_threads()
        .saturating_mul(BITMAP_TARGET_TASKS_PER_THREAD)
        .max(1);
    let words_per_task = n_words
        .div_ceil(target_tasks)
        .clamp(BITMAP_MIN_WORDS_PER_TASK, BITMAP_MAX_WORDS_PER_TASK);
    BitmapTaskLayout {
        words_per_task,
        n_tasks: n_words.div_ceil(words_per_task),
    }
}

impl ThreadPool {
    /// Wake all parked workers via `parking_lot_core::unpark_all`.
    /// One syscall per shard key; workers still spinning observe the new
    /// epoch directly and don't consume a wake.
    /// Must be called AFTER `epoch.fetch_add(_, Release)`.
    #[inline]
    fn wake_workers(&self) {
        for k in &self.shared.park_keys {
            let key = &**k as *const u8 as usize;
            unsafe {
                parking_lot_core::unpark_all(key, parking_lot_core::DEFAULT_UNPARK_TOKEN);
            }
        }
    }

    /// Number of participants (workers + 1 main thread).
    #[inline]
    pub fn num_threads(&self) -> usize {
        self.num_participants
    }

    /// Number of workers (not counting main).
    #[inline]
    pub fn num_workers(&self) -> usize {
        self.worker_threads.len()
    }

    /// Whether `ENGINE_POOL_VERIFY=1` was set at pool init. The env var
    /// is read once in [`init_global`]; callers that want to mirror the
    /// pool's verify mode (e.g. the renderer's staging-walk per-frame
    /// instrumentation) should call this getter instead of re-reading
    /// `env::var` per frame — that takes the process-wide environ
    /// mutex and shows up in hot-path profiles at ~500 ns/call.
    #[inline]
    pub fn verify_enabled(&self) -> bool {
        self.shared.verify_mode.load(Ordering::Relaxed)
    }

    /// Run `f(task_idx)` for every `task_idx` in `0..n_tasks`, distributed
    /// across all participants (workers + main thread) using per-participant
    /// cursors with tail-stealing.
    ///
    /// **Deterministic initial assignment.** Participant `p` is always seeded
    /// with the `p`-th slice of `0..n_tasks` for a given `n_tasks`, so cached
    /// data stays on the worker that first-touched it frame after frame. When
    /// a participant exhausts its own slice it steals the back half of a
    /// neighbour's cursor (rotation order). Main participates inline and
    /// returns only after all active workers have finished.
    ///
    /// **Returns a [`DispatchTiming`] breakdown.** Per-phase timings
    /// (dispatch / main / barrier) are only populated when the `pool-timing`
    /// Cargo feature is enabled.
    pub fn parallel_for<F>(&self, n_tasks: usize, f: F) -> DispatchTiming
    where
        F: Fn(usize) + Sync,
    {
        #[cfg(feature = "pool-timing")]
        use std::time::Instant;

        if n_tasks == 0 {
            return DispatchTiming {
                n_tasks,
                ..Default::default()
            };
        }
        if n_tasks == 1 {
            #[cfg(feature = "pool-timing")]
            let t0 = Instant::now();
            f(0);
            #[cfg(feature = "pool-timing")]
            let main_work_ns = t0.elapsed().as_nanos() as u64;
            #[cfg(not(feature = "pool-timing"))]
            let main_work_ns = 0u64;
            return DispatchTiming {
                n_tasks,
                active_workers: 0,
                dispatch_ns: 0,
                main_work_ns,
                barrier_ns: 0,
                ..Default::default()
            };
        }

        assert!(
            !is_worker(),
            "ThreadPool::parallel_for called from a worker thread — \
             nested parallelism is not supported",
        );

        /// Monomorphised range thunk — runs the inner tight loop inside
        /// this function so `f(i)` inlines fully on workers, matching
        /// the main thread's inline path. One indirect call per slice,
        /// not one per task.
        unsafe fn invoke_range_thunk<F: Fn(usize) + Sync>(
            data: *const (),
            start: usize,
            end: usize,
        ) {
            let f = unsafe { &*(data as *const F) };
            for i in start..end {
                f(i);
            }
        }

        /// Verify variant: batch-bumps `tasks_invoked` once per slice.
        unsafe fn invoke_range_thunk_verify<F: Fn(usize) + Sync>(
            data: *const (),
            start: usize,
            end: usize,
        ) {
            let shared = global().shared;
            let f = unsafe { &*(data as *const F) };
            for i in start..end {
                f(i);
            }
            if end > start {
                shared
                    .tasks_invoked
                    .fetch_add(end - start, Ordering::Relaxed);
            }
        }

        let num_workers = self.worker_threads.len();
        let main_idx = num_workers;

        // When n_tasks < P, only the first active_participants contribute
        // to the barrier; the rest observe the epoch but skip all work.
        let active_participants = self.num_participants.min(n_tasks);
        let active_workers = active_participants.saturating_sub(1);
        let chunk = n_tasks.div_ceil(active_participants);

        // Seed one cursor per participant. Worker p at rank p gets
        // [p*chunk, (p+1)*chunk). Main (rank active_workers) is at cursor
        // index num_workers. Inactive workers get empty cursor [0, 0).
        // If chunk arithmetic overruns n_tasks the cursor start > end;
        // run_with_stealing handles that correctly (start>=end → exit).
        let cursors: Vec<PaddedCursor> = (0..=num_workers)
            .map(|p| {
                let rank = if p < active_workers {
                    p
                } else if p == num_workers {
                    active_workers // main is always the last active rank
                } else {
                    return CachePadded::new(Cursor::new_range(0, 0)); // inactive
                };
                let s = rank * chunk;
                let e = ((rank + 1) * chunk).min(n_tasks);
                CachePadded::new(Cursor::new_range(s, e))
            })
            .collect();

        let verify = self.shared.verify_mode.load(Ordering::Relaxed);
        let invoke_range: unsafe fn(*const (), usize, usize) = if verify {
            invoke_range_thunk_verify::<F>
        } else {
            invoke_range_thunk::<F>
        };

        let new_job = Job {
            active_workers,
            data: &f as *const F as *const (),
            invoke_range,
            cursors: cursors.as_ptr(),
            n_cursors: cursors.len(),
        };

        let shared = self.shared;

        // Reset done counter and optional verify/timing state BEFORE
        // publishing (Release on epoch provides the happens-before edge).
        shared.workers_done.store(0, Ordering::Relaxed);
        if verify {
            shared.tasks_invoked.store(0, Ordering::Relaxed);
        }

        #[cfg(feature = "pool-timing")]
        let t_publish_ns = reset_worker_ts(shared, 0..num_workers);

        // SAFETY: previous parallel_for returned only after workers_done ==
        // active_workers, so no worker is still reading the old job slot.
        unsafe {
            *shared.job.get() = new_job;
        }

        // ── Dispatch phase ─────────────────────────────────────────────
        #[cfg(feature = "pool-timing")]
        let t_dispatch_start = Instant::now();
        shared.epoch.fetch_add(1, Ordering::Release);
        self.wake_workers();
        #[cfg(feature = "pool-timing")]
        let dispatch_ns = t_dispatch_start.elapsed().as_nanos() as u64;
        #[cfg(not(feature = "pool-timing"))]
        let dispatch_ns = 0u64;

        // ── Main work phase: own cursor + stealing ──────────────────────
        #[cfg(feature = "pool-timing")]
        let t_main_start = Instant::now();
        // SAFETY: cursors lives on this stack frame past the barrier.
        unsafe {
            run_with_stealing(
                cursors.as_ptr(),
                cursors.len(),
                main_idx,
                &shared.steal_order[main_idx],
                invoke_range,
                &f as *const F as *const (),
            );
        }
        #[cfg(feature = "pool-timing")]
        let main_work_ns = t_main_start.elapsed().as_nanos() as u64;
        #[cfg(not(feature = "pool-timing"))]
        let main_work_ns = 0u64;

        // ── Barrier phase ───────────────────────────────────────────────
        #[cfg(feature = "pool-timing")]
        let t_barrier_start = Instant::now();
        let mut spins = 0u32;
        loop {
            if shared.workers_done.load(Ordering::Acquire) >= active_workers {
                break;
            }
            spins += 1;
            if spins < 100_000 {
                spin_loop();
            } else if spins < 10_000_000 {
                thread::yield_now();
            } else {
                thread::sleep(std::time::Duration::from_micros(100));
            }
        }
        #[cfg(feature = "pool-timing")]
        let barrier_ns = t_barrier_start.elapsed().as_nanos() as u64;
        #[cfg(not(feature = "pool-timing"))]
        let barrier_ns = 0u64;

        // Now safe to drop cursors — every worker has signalled done.
        drop(cursors);

        if verify {
            let invoked = shared.tasks_invoked.load(Ordering::Acquire);
            assert_eq!(
                invoked, n_tasks,
                "ThreadPool::parallel_for verify FAIL: dispatched {n_tasks} tasks but \
                 {invoked} closure invocations happened. Some participant skipped a task."
            );
        }

        #[cfg(feature = "pool-timing")]
        let (wmin, wavg, wmax, kmin, kavg, kmax) =
            aggregate_worker_stats(shared, 0..active_workers, t_publish_ns);
        #[cfg(not(feature = "pool-timing"))]
        let (wmin, wavg, wmax, kmin, kavg, kmax) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);

        DispatchTiming {
            n_tasks,
            active_workers,
            dispatch_ns,
            main_work_ns,
            barrier_ns,
            worker_wake_min_ns: wmin,
            worker_wake_avg_ns: wavg,
            worker_wake_max_ns: wmax,
            worker_work_min_ns: kmin,
            worker_work_avg_ns: kavg,
            worker_work_max_ns: kmax,
        }
    }
}

/// Set the current thread's CPU-affinity mask to the union of the
/// given Linux CPU ids. Used by `engine-render` (and any custom
/// `on_worker_start` closure) to give NUMA-relay workers a node-wide
/// affinity instead of a single-core pin, so the OS can migrate them
/// off a preempted core without dragging their whole node down with
/// them.
///
/// Returns `true` on success, `false` on a failed `sched_setaffinity`
/// or if any cpu id exceeds `CPU_SETSIZE`. On non-Linux targets this
/// is a no-op that returns `false`.
#[cfg(target_os = "linux")]
pub fn set_current_thread_affinity_mask(cpus: &[usize]) -> bool {
    if cpus.is_empty() {
        return false;
    }
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &cpu in cpus {
            if cpu >= libc::CPU_SETSIZE as usize {
                return false;
            }
            libc::CPU_SET(cpu, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}
#[cfg(not(target_os = "linux"))]
pub fn set_current_thread_affinity_mask(_cpus: &[usize]) -> bool {
    false
}

// ────────────────────────────────────────────────────────────────────────
// Worker loop
// ────────────────────────────────────────────────────────────────────────

thread_local! {
    static IS_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[inline]
fn is_worker() -> bool {
    IS_WORKER.with(|c| c.get())
}

/// Iterations of pure `spin_loop` before a worker falls back to
/// `sleep(Duration::from_nanos(1))` (~50 µs hrtimer ride-out), and
/// total iterations including the sleep phase before the worker
/// finally parks via `parking_lot_core::park`. The 90k sleep
/// iterations in between keep workers out of the parking_lot bucket
/// during the steady-state inter-frame gap, so main's per-dispatch
/// wake is a cheap no-op token-store rather than a real futex-wake.
const HOT_SPIN_ITERS: u32 = 10_000;
const PARK_AFTER_SPINS: u32 = 100_000;

/// Maximum workers that share a single parking_lot_core park key.
/// Each key hashes to a distinct internal bucket, so sharding spreads
/// the wake/park traffic across multiple bucket mutexes. Tunable via
/// `ENGINE_PARK_SHARD_SIZE` env var at pool init.
const DEFAULT_PARK_SHARD_SIZE: usize = 16;

fn worker_loop(shared: &'static Shared, worker_idx: usize) {
    IS_WORKER.with(|c| c.set(true));

    let mut last_epoch: u64 = 0;
    let shard = shared.worker_shard[worker_idx] as usize;
    let park_key = &*shared.park_keys[shard] as *const u8 as usize;

    loop {
        let mut spins: u32 = 0;
        // Spin → sleep(1ns) hrtimer → park. All workers poll shared.epoch
        // directly (no per-node relay epoch).
        let cur = loop {
            let e = shared.epoch.load(Ordering::Acquire);
            if e != last_epoch {
                break e;
            }
            if shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            if spins < HOT_SPIN_ITERS {
                spins = spins.saturating_add(1);
                spin_loop();
                continue;
            }
            if spins < PARK_AFTER_SPINS {
                spins = spins.saturating_add(1);
                // sleep(1ns) → ~50µs hrtimer ride-out; releases the
                // core so SMT siblings keep their turbo budget.
                thread::sleep(std::time::Duration::from_nanos(1));
                continue;
            }
            if shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            // `validate` re-checks under the bucket lock so we don't
            // miss a wake that races with the load above.
            unsafe {
                parking_lot_core::park(
                    park_key,
                    || {
                        shared.epoch.load(Ordering::Acquire) == last_epoch
                            && !shared.shutdown.load(Ordering::Relaxed)
                    },
                    || {},
                    |_, _| {},
                    parking_lot_core::DEFAULT_PARK_TOKEN,
                    None,
                );
            }
            spins = 0;
        };
        last_epoch = cur;

        // Record epoch-observation timestamp BEFORE doing any work.
        // Relaxed is fine: main reads after the barrier (workers_done).
        #[cfg(feature = "pool-timing")]
        {
            let now_ns = shared.anchor.elapsed().as_nanos() as u64;
            shared.worker_ts[worker_idx]
                .t_seen
                .store(now_ns, Ordering::Relaxed);
        }

        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }

        // SAFETY: the Acquire on `epoch` above synchronises with main's
        // Release-publish of the job slot.
        let job = unsafe { *shared.job.get() };

        // Workers 0..active_workers participate and contribute to the
        // barrier; workers >= active_workers observe the epoch but skip
        // all work (and do NOT increment workers_done).
        if worker_idx < job.active_workers {
            // SAFETY: `cursors` points to a stack buffer kept alive by
            // main until after the barrier; steal_order is 'static.
            unsafe {
                run_with_stealing(
                    job.cursors,
                    job.n_cursors,
                    worker_idx,
                    &shared.steal_order[worker_idx],
                    job.invoke_range,
                    job.data,
                );
            }
            #[cfg(feature = "pool-timing")]
            {
                let now_ns = shared.anchor.elapsed().as_nanos() as u64;
                shared.worker_ts[worker_idx]
                    .t_done
                    .store(now_ns, Ordering::Relaxed);
            }
            shared.workers_done.fetch_add(1, Ordering::Release);
        }
        // Inactive workers: t_done stays 0 (sentinel for
        // aggregate_worker_stats's "didn't participate" filter).
    }
}

#[inline]
unsafe fn noop_invoke_range(_data: *const (), _start: usize, _end: usize) {}

/// Drive a single participant through its own cursor and any
/// successful steals until everything reachable is drained.
///
/// SAFETY: `cursors` must be a valid pointer to `n_cursors`
/// `PaddedCursor`s that remain alive for the duration of this call.
/// `own_idx < n_cursors`, every `peer in steal_order` is
/// `< n_cursors`, and `data` plus `invoke_range` together name a
/// valid type-erased closure invocation.
unsafe fn run_with_stealing(
    cursors: *const PaddedCursor,
    _n_cursors: usize,
    own_idx: usize,
    steal_order: &[usize],
    invoke_range: unsafe fn(*const (), usize, usize),
    data: *const (),
) {
    // Drain own cursor in STEAL_GRAIN chunks. Claim `[s, claim_end)` by
    // CAS-ing the whole packed word (so a concurrent thief lowering
    // `end` can't cause an overlap); retry on contention.
    let own = unsafe { &*cursors.add(own_idx) };
    loop {
        let packed = own.packed.load(Ordering::Acquire);
        let (s, e) = unpack_cursor(packed);
        if s >= e {
            break;
        }
        let claim_end = (s + STEAL_GRAIN as u32).min(e);
        if own
            .packed
            .compare_exchange_weak(
                packed,
                pack_cursor(claim_end, e),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            unsafe { (invoke_range)(data, s as usize, claim_end as usize) };
        }
        // else: a thief lowered `end` concurrently — retry with the
        // fresh value.
    }
    // Steal loop: walk peers in `steal_order`. Restart from the
    // beginning after every successful steal so nearest peers
    // (the prefix of the order) are preferred. Terminate when a
    // full sweep finds nothing.
    'outer: loop {
        for &peer in steal_order {
            let cur = unsafe { &*cursors.add(peer) };
            if let Some((s, e)) = try_steal(cur) {
                unsafe { (invoke_range)(data, s, e) };
                continue 'outer;
            }
        }
        break;
    }
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering as O;

    fn test_lock() -> std::sync::MutexGuard<'static, ()> {
        super::lock_for_test()
    }

    fn init_pool_once() {
        // Init happens inside `lock_for_test`; the returned guard is
        // dropped before the test body runs, which is fine because the
        // global pool, once installed, is `'static`. Tests that
        // dispatch parallel_for must call `let _g = test_lock();` to
        // hold the gate for the duration of their dispatch.
        drop(super::lock_for_test());
    }

    #[test]
    fn parallel_for_visits_every_task_exactly_once() {
        init_pool_once();
        let _g = test_lock();
        const N: usize = 10_000;
        let counts: Vec<AtomicUsize> = (0..N).map(|_| AtomicUsize::new(0)).collect();
        global().parallel_for(N, |i| {
            counts[i].fetch_add(1, O::Relaxed);
        });
        for (i, c) in counts.iter().enumerate() {
            let v = c.load(O::Relaxed);
            assert_eq!(v, 1, "task {i} ran {v} times");
        }
    }

    #[test]
    fn parallel_for_back_to_back_jobs() {
        init_pool_once();
        let _g = test_lock();
        for round in 0..50 {
            let n = 100 + round * 17;
            let counter = AtomicUsize::new(0);
            global().parallel_for(n, |_| {
                counter.fetch_add(1, O::Relaxed);
            });
            assert_eq!(counter.load(O::Relaxed), n);
        }
    }

    #[test]
    fn parallel_for_zero_and_one_task() {
        init_pool_once();
        let _g = test_lock();
        global().parallel_for(0, |_| panic!("should not run"));
        let c = AtomicUsize::new(0);
        global().parallel_for(1, |i| {
            assert_eq!(i, 0);
            c.fetch_add(1, O::Relaxed);
        });
        assert_eq!(c.load(O::Relaxed), 1);
    }

    #[test]
    fn parallel_for_n_less_than_workers() {
        // n_tasks=2 with 4 workers+main: chunk=1, only first 2
        // participants do work, rest get empty slices and still
        // increment workers_done.
        init_pool_once();
        let _g = test_lock();
        let counter = AtomicUsize::new(0);
        global().parallel_for(2, |_| {
            counter.fetch_add(1, O::Relaxed);
        });
        assert_eq!(counter.load(O::Relaxed), 2);
    }

    /// Sweep every n from 0 up through 2 × (num_participants · 64).
    /// For each n: every task index in 0..n must be visited exactly once,
    /// no index outside [0, n) may be touched. This proves the static
    /// chunk arithmetic (ceil-divide + last-participant slicing) is
    /// hole-free for the boundary cases that matter (n < participants,
    /// n = participants, n = participants+1, ragged tail, etc.).
    #[test]
    fn parallel_for_sweep_exact_coverage() {
        init_pool_once();
        let _g = test_lock();
        let p = global().num_threads();
        let max_n = (p * 64).max(64) * 2;
        for n in 0..=max_n {
            let visited: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            global().parallel_for(n, |i| {
                // OOB write here would panic on the indexed access.
                visited[i].fetch_add(1, O::Relaxed);
            });
            for (i, c) in visited.iter().enumerate() {
                let v = c.load(O::Relaxed);
                assert_eq!(v, 1, "n={n}: task {i} visited {v} times");
            }
        }
    }

    /// 50 back-to-back jobs each with a different N. The previous
    /// scheme had a race where workers finishing the prior job could
    /// stomp the next job's counters; this is the canary for that.
    #[test]
    fn parallel_for_b2b_coverage_no_drops() {
        init_pool_once();
        let _g = test_lock();
        for round in 0..200 {
            // Mix tiny, around-participant-count, and large jobs.
            let n = match round % 5 {
                0 => 0,
                1 => 1,
                2 => global().num_threads() - 1,
                3 => global().num_threads(),
                _ => 50_000 + round,
            };
            let visited: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            global().parallel_for(n, |i| {
                visited[i].fetch_add(1, O::Relaxed);
            });
            for (i, c) in visited.iter().enumerate() {
                let v = c.load(O::Relaxed);
                assert_eq!(v, 1, "round {round}, n={n}: task {i} visited {v} times");
            }
        }
    }

    /// Verify the same task-index arithmetic the dirty-harvest in
    /// `engine_render` relies on: each task gets a contiguous
    /// [word_base, word_end) range, the union covers [0, hier_words),
    /// the ranges are disjoint, and an inner "entity = word_idx * 32 + bit"
    /// mapping produces unique entity indices across the whole job.
    #[test]
    fn parallel_for_bitmap_walk_arithmetic() {
        init_pool_once();
        let _g = test_lock();
        // Cover the boundary cases: empty, single word, exactly one
        // chunk, multiple full chunks, and a ragged tail.
        for hier_words in [
            0usize,
            1,
            7,
            8,
            9,
            31,
            32,
            33,
            255,
            256,
            257,
            BITMAP_MAX_WORDS_PER_TASK * 3,
            BITMAP_MAX_WORDS_PER_TASK * 3 + 17,
        ] {
            // Simulate: every bit set, then verify every entity in
            // [0, hier_words * 32) is touched exactly once.
            let n_entities = hier_words * 32;
            let entity_hits: Vec<AtomicUsize> =
                (0..n_entities).map(|_| AtomicUsize::new(0)).collect();
            let word_hits: Vec<AtomicUsize> =
                (0..hier_words).map(|_| AtomicUsize::new(0)).collect();

            let layout = bitmap_task_layout(hier_words);
            global().parallel_for(layout.n_tasks, |task_idx| {
                let word_base = task_idx * layout.words_per_task;
                let word_end = (word_base + layout.words_per_task).min(hier_words);
                for word_idx in word_base..word_end {
                    word_hits[word_idx].fetch_add(1, O::Relaxed);
                    // Pretend every bit in this word is set.
                    for bit in 0..32 {
                        let entity = word_idx * 32 + bit;
                        if entity < n_entities {
                            entity_hits[entity].fetch_add(1, O::Relaxed);
                        }
                    }
                }
            });

            for (i, c) in word_hits.iter().enumerate() {
                let v = c.load(O::Relaxed);
                assert_eq!(v, 1, "hier_words={hier_words}: word {i} hit {v} times");
            }
            for (i, c) in entity_hits.iter().enumerate() {
                let v = c.load(O::Relaxed);
                assert_eq!(v, 1, "hier_words={hier_words}: entity {i} hit {v} times");
            }
        }
    }

    #[test]
    fn parallel_for_activates_only_needed_workers() {
        init_pool_once();
        let _g = test_lock();

        let pool = global();

        let small = pool.parallel_for(3, |_| {});
        assert_eq!(small.active_workers, 2);

        let large = pool.parallel_for(pool.num_threads() + 17, |_| {});
        assert_eq!(large.active_workers, pool.num_workers());
    }

    /// Stress test: same pool, many distinct concurrent jobs from main.
    /// Drains the pool ~100× in rapid succession with varying chunk
    /// shapes. Designed to catch any leftover "worker still in flight"
    /// race between consecutive parallel_for calls.
    #[test]
    fn parallel_for_stress_b2b_huge_count() {
        init_pool_once();
        let _g = test_lock();
        for round in 0..100 {
            let n = 100_000 + (round * 1_237) % 50_000;
            let counter = AtomicUsize::new(0);
            global().parallel_for(n, |_| {
                counter.fetch_add(1, O::Relaxed);
            });
            let got = counter.load(O::Relaxed);
            assert_eq!(got, n, "round {round} n={n} got {got}");
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Convenience helpers
// ────────────────────────────────────────────────────────────────────────

/// Run `f(chunk_start, chunk_end)` over every consecutive
/// `[chunk_start, chunk_end)` slice of `0..n` with chunk size `chunk_len`.
/// The trailing chunk may be shorter.
#[inline]
pub fn for_chunks<F>(n: usize, chunk_len: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    assert!(chunk_len > 0);
    if n == 0 {
        return;
    }
    let n_chunks = n.div_ceil(chunk_len);
    global().parallel_for(n_chunks, |chunk_idx| {
        let start = chunk_idx * chunk_len;
        let end = (start + chunk_len).min(n);
        f(start, end);
    });
}
