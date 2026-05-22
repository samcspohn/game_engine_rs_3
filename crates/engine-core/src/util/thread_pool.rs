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
//!   start hook (typically `core_affinity::set_for_current`). Main thread
//!   is **always** a participant — work is split across `num_workers + 1`
//!   threads, of which the main thread runs its share inline.
//! * One global pool installed once via [`init_global`].
//! * One primitive: [`ThreadPool::parallel_for`]. **Static range
//!   partitioning** — every participant gets a contiguous, pre-computed
//!   slice of `0..n_tasks`. No per-task atomic, no work stealing, no
//!   adaptive splitting. For the engine's uniform per-entity workloads
//!   this is faster than rayon because the per-task atomic contention
//!   and the per-task dispatch overhead both vanish; the only price is
//!   loss of load balancing on irregular work (which we don't have).
//! * **Workers do not park inside a frame.** A tight `spin_loop` runs
//!   for ~30 µs, then `thread::sleep(1ns)` (kernel rounds up to one
//!   scheduler tick, ~50 µs on Linux, but releases the core to the
//!   scheduler so neighbouring siblings keep their turbo budget) for
//!   the rest of a ~1 ms idle window, then `thread::park`. The 1 ms
//!   threshold is well above any inter-dispatch gap inside a hot frame
//!   at any realistic FPS, so back-to-back dispatches inside
//!   `Scene::update` and the per-frame staging walk never pay the
//!   futex wake cost. Workers are pinned to their own cores; while
//!   spinning the cost is only the worker's own core which would
//!   otherwise idle. Wakeup after a park is via
//!   `std::thread::Thread::unpark` (futex on Linux, no mutex on the
//!   wake side).
//! * No nested parallelism. `parallel_for` called from inside a worker
//!   panics — per project rules, no silent serial fallback. Callers that
//!   need recursive descent must walk sequentially inside a task.
//!
//! ## Safety
//!
//! `parallel_for` publishes a stack-allocated [`Job`] via an
//! [`UnsafeCell`] under release/acquire on `epoch`. The Job borrow lives
//! for the duration of the call; the function blocks until every worker
//! has signalled completion of its assigned slice (`workers_done ==
//! num_workers`), so the borrow cannot dangle.

use std::cell::UnsafeCell;
use std::hint::spin_loop;
use std::ops::Range;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle, Thread};

// ────────────────────────────────────────────────────────────────────────
// NUMA partitioning trait
// ────────────────────────────────────────────────────────────────────────

/// Data structure whose backing storage is split across NUMA nodes.
///
/// `numa_partitions()` returns one [`Range<usize>`] per NUMA node (in
/// node-id order). Each range is the slice of the structure's logical
/// index space owned by that node. The ranges must:
///
/// * cover the full logical index space (`union == [0, len)`),
/// * be disjoint,
/// * be in node-id order.
///
/// The caller of [`ThreadPool::parallel_for_numa`] supplies a partition
/// in the closure's *task* coordinate (e.g. bitmap word index, slab
/// index) derived from the structure's entity-space partition; the pool
/// only sees the task-space ranges and dispatches each one to the
/// workers pinned to that node.
pub trait NumaPartitioned {
    /// One range per NUMA node, in node-id order. Length must equal
    /// the pool's `num_nodes()`.
    fn numa_partitions(&self) -> &[Range<usize>];
}

// ────────────────────────────────────────────────────────────────────────
// Job slot
// ────────────────────────────────────────────────────────────────────────

/// Type-erased, lifetime-erased descriptor of a parallel-for job.
///
/// Two slicing modes share this slot:
///
/// * **Uniform chunk** (`slices.is_null() == true`): worker `i` runs
///   `[i * chunk, min((i+1) * chunk, n_tasks))`. Used by
///   [`ThreadPool::parallel_for_global`].
/// * **Per-worker slices** (`slices.is_null() == false`): worker `i`
///   runs the `(start, end)` pair stored at `slices.add(i)`. Used by
///   [`ThreadPool::parallel_for_numa`] to give each worker its
///   node-local slab of the per-node partition. The `slices` buffer is
///   stack-allocated for the duration of the dispatch call (same
///   lifetime contract as the closure data pointer).
///
/// `invoke_range` runs the inner `for i in start..end { f(i) }` loop
/// **inside the monomorphised thunk**, so the worker pays a single
/// indirect call per slice instead of per task. This lets the
/// compiler inline `f(i)` into the tight loop body — matching the
/// main thread's inline path — and unlocks LICM / vectorisation of
/// the closure body across iterations.
#[derive(Clone, Copy)]
struct Job {
    /// Total task count. In uniform mode, used with `chunk` to derive
    /// per-worker ranges. In per-worker-slices mode, used only for
    /// verify-mode accounting.
    n_tasks: usize,
    chunk: usize,
    active_workers: usize,
    data: *const (),
    invoke_range: unsafe fn(*const (), usize, usize),
    /// Null in uniform mode; non-null in per-worker-slices mode. Points
    /// to `active_workers` `(start, end)` pairs.
    slices: *const (usize, usize),
}

unsafe impl Sync for Job {}
unsafe impl Send for Job {}

// ────────────────────────────────────────────────────────────────────────
// Shared state
// ────────────────────────────────────────────────────────────────────────

struct Shared {
    /// Bumped every time main publishes a new job. Workers compare against
    /// their `last_epoch` to detect new work. Release/Acquire here
    /// synchronises the non-atomic write to `job`.
    epoch: AtomicU64,

    /// Number of workers that have finished their assigned slice for the
    /// current job. Main blocks until this reaches `num_workers`. Reset
    /// to 0 before each new epoch is published.
    workers_done: AtomicUsize,

    /// Job descriptor. Written by main BEFORE bumping `epoch`. Read by
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
}

unsafe impl Sync for Shared {}

// ────────────────────────────────────────────────────────────────────────
// Pool
// ────────────────────────────────────────────────────────────────────────

pub struct ThreadPool {
    shared: &'static Shared,
    /// Thread handles for `unpark()` on dispatch. Cached separately so
    /// the dispatch loop doesn't need to walk the `JoinHandle`s.
    worker_threads: Vec<Thread>,
    /// Kept alive so the worker threads stay joinable for the program's
    /// lifetime. We never join (the pool is `'static`); shutdown is a
    /// process-exit, not a graceful teardown.
    #[allow(dead_code)]
    handles: Vec<JoinHandle<()>>,
    /// Total number of participants in a `parallel_for_*` =
    /// `worker_threads.len() + 1` (main thread is always a participant).
    num_participants: usize,
    /// NUMA topology metadata. Set at [`init_global`] time and
    /// immutable thereafter. Even on single-node systems we set this to
    /// `[0]` so callers don't need to special-case "no NUMA".
    num_nodes: u32,
    /// NUMA node id of the main thread. Main always participates in
    /// `parallel_for_numa` as a worker of this node.
    main_node: u32,
    /// `worker_nodes[i]` = NUMA node of worker `i`. Length == num_workers.
    worker_nodes: Vec<u32>,
    /// Per-node list of worker indices: `workers_by_node[n]` is the
    /// list of `worker_idx` values whose `worker_nodes[idx] == n`.
    /// Computed once at init for fast dispatch.
    workers_by_node: Vec<Vec<usize>>,
}

static POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Per-worker spawn spec. The caller is responsible for the actual
/// CPU pinning inside the `on_worker_start` closure (typically via
/// `core_affinity::set_for_current`); `node` is just metadata recorded
/// for later NUMA-aware dispatch.
#[derive(Debug, Clone)]
pub struct WorkerSpec {
    /// NUMA node this worker will be pinned to.
    pub node: u32,
}

/// Pool initialisation config.
///
/// * `workers` — one entry per worker thread, in worker-index order.
///   `workers[idx].node` must match the NUMA node the
///   `on_worker_start(idx)` closure pins the worker to. The pool
///   trusts the caller — there is no cross-check at runtime.
/// * `main_node` — NUMA node of the calling thread (which becomes the
///   main participant). Must match wherever the caller pinned main.
/// * `num_nodes` — total node count in the topology. May exceed the
///   set of nodes actually used by workers (an idle node is fine).
pub struct PoolConfig {
    pub workers:   Vec<WorkerSpec>,
    pub main_node: u32,
    pub num_nodes: u32,
}

/// Install the global thread pool. Spawns one worker per entry in
/// `cfg.workers` (in addition to the calling thread, which becomes the
/// main participant). `on_worker_start(idx)` is invoked on each worker
/// before it enters its loop — use this to call
/// `core_affinity::set_for_current` and any other per-worker setup.
///
/// Panics if called more than once, if `cfg.workers` is empty, or if
/// any worker fails to spawn. No fallbacks.
pub fn init_global<F>(cfg: PoolConfig, on_worker_start: F)
where
    F: Fn(usize) + Send + Sync + 'static,
{
    assert!(
        !cfg.workers.is_empty(),
        "ThreadPool requires at least one worker",
    );
    assert!(cfg.num_nodes > 0, "ThreadPool requires num_nodes >= 1");
    assert!(
        cfg.main_node < cfg.num_nodes,
        "main_node {} >= num_nodes {}",
        cfg.main_node,
        cfg.num_nodes,
    );
    for (i, w) in cfg.workers.iter().enumerate() {
        assert!(
            w.node < cfg.num_nodes,
            "worker {i} node {} >= num_nodes {}",
            w.node,
            cfg.num_nodes,
        );
    }

    let num_workers = cfg.workers.len();
    let worker_nodes: Vec<u32> = cfg.workers.iter().map(|w| w.node).collect();
    let mut workers_by_node: Vec<Vec<usize>> = (0..cfg.num_nodes).map(|_| Vec::new()).collect();
    for (idx, n) in worker_nodes.iter().enumerate() {
        workers_by_node[*n as usize].push(idx);
    }

    let shared: &'static Shared = Box::leak(Box::new(Shared {
        epoch: AtomicU64::new(0),
        workers_done: AtomicUsize::new(0),
        job: UnsafeCell::new(Job {
            n_tasks: 0,
            chunk: 0,
            active_workers: 0,
            data: std::ptr::null(),
            invoke_range: noop_invoke_range,
            slices: std::ptr::null(),
        }),
        shutdown: AtomicBool::new(false),
        verify_mode: AtomicBool::new(
            std::env::var("ENGINE_POOL_VERIFY")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
        ),
        tasks_invoked: AtomicUsize::new(0),
    }));

    let (tx, rx) = std::sync::mpsc::sync_channel::<Thread>(num_workers);
    let on_start = std::sync::Arc::new(on_worker_start);

    let mut handles = Vec::with_capacity(num_workers);
    for idx in 0..num_workers {
        let tx = tx.clone();
        let on_start = on_start.clone();
        let handle = thread::Builder::new()
            .name(format!("engine-worker-{idx}"))
            .spawn(move || {
                on_start(idx);
                tx.send(thread::current())
                    .expect("worker failed to register Thread handle");
                drop(tx);
                worker_loop(shared, idx);
            })
            .expect("failed to spawn engine worker thread");
        handles.push(handle);
    }
    drop(tx);

    let mut worker_threads = Vec::with_capacity(num_workers);
    for _ in 0..num_workers {
        worker_threads.push(rx.recv().expect("worker thread failed to register"));
    }

    let pool = ThreadPool {
        shared,
        worker_threads,
        handles,
        num_participants: num_workers + 1,
        num_nodes: cfg.num_nodes,
        main_node: cfg.main_node,
        worker_nodes,
        workers_by_node,
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
        init_global(
            PoolConfig {
                workers: (0..4).map(|_| WorkerSpec { node: 0 }).collect(),
                main_node: 0,
                num_nodes: 1,
            },
            |_| {},
        );
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
    pub n_tasks:      usize,
    pub active_workers: usize,
    pub dispatch_ns:  u64,
    pub main_work_ns: u64,
    pub barrier_ns:   u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitmapTaskLayout {
    pub words_per_task: usize,
    pub n_tasks:        usize,
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
            n_tasks:        0,
        };
    }
    let target_tasks = global().num_threads()
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

    /// Run `f(task_idx)` for every `task_idx` in `0..n_tasks`, statically
    /// partitioned across all participants (workers + main thread). Main
    /// participates inline and returns only after every worker has
    /// finished its slice.
    ///
    /// **Static partitioning.** Each participant owns a contiguous
    /// `[start, end)` of the task range. There is no per-task atomic
    /// claim — once dispatched, each participant's inner loop is a tight
    /// `for i in start..end { f(i) }`. This is faster than a fetch_add
    /// scheme for our workloads (single-digit µs per dispatch at
    /// N=100K, no contended-cache-line ping-pong) at the cost of no
    /// load balancing if `f`'s per-task cost is non-uniform. The
    /// engine's uses are all uniform per-entity work, so the trade is
    /// a win.
    ///
    /// **Returns a [`DispatchTiming`] breakdown.** Per-phase timings
    /// (dispatch / main / barrier) are only captured when the
    /// `pool-timing` Cargo feature is enabled — otherwise every field
    /// except `n_tasks` is `0` and no `Instant::now()` calls happen in
    /// the hot path. Callers that don't care about the breakdown can
    /// freely discard the return value (`DispatchTiming` is not
    /// `#[must_use]`).
    pub fn parallel_for_global<F>(&self, n_tasks: usize, f: F) -> DispatchTiming
    where
        F: Fn(usize) + Sync,
    {
        #[cfg(feature = "pool-timing")]
        use std::time::Instant;

        if n_tasks == 0 {
            return DispatchTiming { n_tasks, ..Default::default() };
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
                dispatch_ns:  0,
                main_work_ns,
                barrier_ns:   0,
            };
        }

        assert!(
            !is_worker(),
            "ThreadPool::parallel_for_global called from a worker thread — \
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

        /// Verify variant: runs the slice then batch-bumps
        /// `tasks_invoked` once (`fetch_add(end - start, Relaxed)`)
        /// instead of once per task. Only installed when `verify_mode`
        /// is on.
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
                shared.tasks_invoked.fetch_add(end - start, Ordering::Relaxed);
            }
        }

        let n_workers             = self.worker_threads.len();
        let n_active_participants = self.num_participants.min(n_tasks);
        let active_workers        = n_active_participants.saturating_sub(1);
        // Round up so the early participants take the extra task when
        // n_tasks doesn't divide evenly. Last participant may have a
        // smaller (or empty) slice.
        let chunk                = n_tasks.div_ceil(n_active_participants);

        let verify = self.shared.verify_mode.load(Ordering::Relaxed);
        let invoke_range: unsafe fn(*const (), usize, usize) = if verify {
            invoke_range_thunk_verify::<F>
        } else {
            invoke_range_thunk::<F>
        };

        let new_job = Job {
            n_tasks,
            chunk,
            active_workers,
            data: &f as *const F as *const (),
            invoke_range,
            slices: std::ptr::null(),
        };

        let shared = self.shared;

        // Reset the worker-done counter BEFORE publishing. Relaxed is
        // fine: the Release on `epoch.fetch_add` below provides the
        // happens-before edge to workers.
        shared.workers_done.store(0, Ordering::Relaxed);
        if verify {
            shared.tasks_invoked.store(0, Ordering::Relaxed);
        }

        // SAFETY: previous parallel_for returned only when
        // workers_done == n_workers, so no worker is touching the slot.
        unsafe { *shared.job.get() = new_job; }

        // ── Dispatch phase: publish + wake workers ──────────────────────
        #[cfg(feature = "pool-timing")]
        let t_dispatch_start = Instant::now();
        shared.epoch.fetch_add(1, Ordering::Release);
        // Wake any parked workers. Cheap no-op if they're spinning.
        // `unpark` is just an atomic + (conditional) futex wake.
        for t in self.worker_threads.iter().take(active_workers) {
            t.unpark();
        }
        #[cfg(feature = "pool-timing")]
        let dispatch_ns = t_dispatch_start.elapsed().as_nanos() as u64;
        #[cfg(not(feature = "pool-timing"))]
        let dispatch_ns = 0u64;

        // ── Main work phase ────────────────────────────────
        #[cfg(feature = "pool-timing")]
        let t_main_start = Instant::now();
        let main_start = active_workers * chunk;
        let main_end   = n_tasks.min(main_start + chunk);
        for i in main_start..main_end {
            f(i);
        }
        if verify && main_end > main_start {
            shared.tasks_invoked.fetch_add(main_end - main_start, Ordering::Relaxed);
        }
        #[cfg(feature = "pool-timing")]
        let main_work_ns = t_main_start.elapsed().as_nanos() as u64;
        #[cfg(not(feature = "pool-timing"))]
        let main_work_ns = 0u64;

        // ── Barrier phase ────────────────────────────────────────────
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

        if verify {
            let invoked = shared.tasks_invoked.load(Ordering::Acquire);
            if invoked != n_tasks {
                eprintln!(
                    "[pool-verify] FAIL n_tasks={n_tasks} chunk={chunk} \
                     n_workers={n_workers} active_workers={active_workers} \
                     active_participants={n_active_participants}"
                );
                eprintln!(
                    "[pool-verify]   expected main slice = [{main_start}, {main_end}) = {} tasks",
                    main_end - main_start,
                );
                for w in 0..active_workers {
                    let s = w * chunk;
                    let e = n_tasks.min(s + chunk);
                    let count = if s < n_tasks { e - s } else { 0 };
                    eprintln!(
                        "[pool-verify]   expected worker {w} slice = [{s}, {e}) = {count} tasks",
                    );
                }
            }
            assert_eq!(
                invoked, n_tasks,
                "ThreadPool::parallel_for_global verify FAIL: dispatched {n_tasks} tasks but\n\
                 only {invoked} closure invocations happened. Some participant skipped a task."
            );
        }

        DispatchTiming {
            n_tasks,
            active_workers,
            dispatch_ns,
            main_work_ns,
            barrier_ns,
        }
    }

    /// NUMA-aware parallel_for. `partitions[n]` is the task-index range
    /// owned by NUMA node `n` (one entry per node, in node-id order;
    /// must have `partitions.len() == self.num_nodes()`). Each worker
    /// runs only over its node's range; main participates as a worker
    /// of `self.main_node()`.
    ///
    /// Within each node, that node's range is statically split among
    /// the node's participants (workers pinned to this node, plus main
    /// if `node == main_node`). Main always takes the last slice of its
    /// node, mirroring [`parallel_for_global`].
    ///
    /// Panics if any node has a non-empty partition but no
    /// participants (would silently drop tasks).
    ///
    /// All workers (across all nodes) park-wake on every dispatch and
    /// each contributes one increment to the barrier counter regardless
    /// of slice size. This is the simplest correct scheme for v1; if
    /// the futex-wake overhead shows up in profiles we can switch to
    /// per-node wake bitmaps later.
    pub fn parallel_for_numa<F>(&self, partitions: &[Range<usize>], f: F) -> DispatchTiming
    where
        F: Fn(usize) + Sync,
    {
        #[cfg(feature = "pool-timing")]
        use std::time::Instant;

        assert_eq!(
            partitions.len(),
            self.num_nodes as usize,
            "parallel_for_numa: partitions.len() ({}) != num_nodes ({})",
            partitions.len(),
            self.num_nodes,
        );
        assert!(
            !is_worker(),
            "ThreadPool::parallel_for_numa called from a worker thread — \
             nested parallelism is not supported",
        );

        let n_tasks_total: usize = partitions.iter().map(|r| r.len()).sum();
        if n_tasks_total == 0 {
            return DispatchTiming { n_tasks: 0, ..Default::default() };
        }

        let num_workers = self.worker_threads.len();
        let mut slices: Vec<(usize, usize)> = vec![(0, 0); num_workers];
        let mut main_slice: (usize, usize) = (0, 0);

        for n in 0..self.num_nodes as usize {
            let range = &partitions[n];
            let node_workers = &self.workers_by_node[n];
            let main_here = n == self.main_node as usize;
            let participants = node_workers.len() + if main_here { 1 } else { 0 };
            if range.is_empty() {
                continue;
            }
            assert!(
                participants > 0,
                "parallel_for_numa: node {n} has non-empty partition {:?} but no participants",
                range,
            );
            let n_node_tasks = range.len();
            let chunk = n_node_tasks.div_ceil(participants);
            let base = range.start;
            for (i, &widx) in node_workers.iter().enumerate() {
                let s = base + i * chunk;
                let e = (base + (i + 1) * chunk).min(range.end);
                slices[widx] = if s < e { (s, e) } else { (s, s) };
            }
            if main_here {
                let i = node_workers.len();
                let s = base + i * chunk;
                let e = (base + (i + 1) * chunk).min(range.end);
                main_slice = if s < e { (s, e) } else { (s, s) };
            }
        }

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
                shared.tasks_invoked.fetch_add(end - start, Ordering::Relaxed);
            }
        }

        let verify = self.shared.verify_mode.load(Ordering::Relaxed);
        let invoke_range: unsafe fn(*const (), usize, usize) = if verify {
            invoke_range_thunk_verify::<F>
        } else {
            invoke_range_thunk::<F>
        };

        let shared = self.shared;
        shared.workers_done.store(0, Ordering::Relaxed);
        if verify {
            shared.tasks_invoked.store(0, Ordering::Relaxed);
        }

        // `slices` is a stack-allocated buffer kept alive until after
        // the barrier; workers read it via raw pointer.
        let new_job = Job {
            n_tasks: n_tasks_total,
            chunk: 0, // unused in slices mode
            active_workers: num_workers,
            data: &f as *const F as *const (),
            invoke_range,
            slices: slices.as_ptr(),
        };
        unsafe { *shared.job.get() = new_job; }

        #[cfg(feature = "pool-timing")]
        let t_dispatch_start = Instant::now();
        shared.epoch.fetch_add(1, Ordering::Release);
        for t in self.worker_threads.iter() {
            t.unpark();
        }
        #[cfg(feature = "pool-timing")]
        let dispatch_ns = t_dispatch_start.elapsed().as_nanos() as u64;
        #[cfg(not(feature = "pool-timing"))]
        let dispatch_ns = 0u64;

        #[cfg(feature = "pool-timing")]
        let t_main_start = Instant::now();
        let (ms, me) = main_slice;
        for i in ms..me {
            f(i);
        }
        if verify && me > ms {
            shared.tasks_invoked.fetch_add(me - ms, Ordering::Relaxed);
        }
        #[cfg(feature = "pool-timing")]
        let main_work_ns = t_main_start.elapsed().as_nanos() as u64;
        #[cfg(not(feature = "pool-timing"))]
        let main_work_ns = 0u64;

        #[cfg(feature = "pool-timing")]
        let t_barrier_start = Instant::now();
        let mut spins = 0u32;
        loop {
            if shared.workers_done.load(Ordering::Acquire) >= num_workers {
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

        // Now safe to drop the slices buffer — every worker has signaled done.
        drop(slices);

        if verify {
            let invoked = shared.tasks_invoked.load(Ordering::Acquire);
            assert_eq!(
                invoked, n_tasks_total,
                "parallel_for_numa verify FAIL: dispatched {n_tasks_total} tasks but \
                 {invoked} closure invocations happened",
            );
        }

        DispatchTiming {
            n_tasks: n_tasks_total,
            active_workers: num_workers,
            dispatch_ns,
            main_work_ns,
            barrier_ns,
        }
    }

    /// NUMA topology getters.
    #[inline] pub fn num_nodes(&self) -> u32 { self.num_nodes }
    #[inline] pub fn main_node(&self) -> u32 { self.main_node }
    #[inline] pub fn worker_node(&self, worker_idx: usize) -> u32 {
        self.worker_nodes[worker_idx]
    }
    #[inline] pub fn workers_on_node(&self, node: u32) -> &[usize] {
        &self.workers_by_node[node as usize]
    }
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

/// Threshold (in spin iterations) before a worker that has seen no new
/// job will park. Tuned for: stay hot across the sub-ms inter-dispatch
/// gap inside a normal frame, but go quiet within ~1 ms when there's
/// nothing to do at all.
///
/// Iteration cost on x86_64 / Linux:
/// * `spin_loop()` → `pause` instruction, ~5-50 cycles ≈ 1.5-15 ns @ 3 GHz
/// * `thread::sleep(1ns)` → kernel-side `nanosleep` rounded up to one
///   scheduler tick (~50-100 µs on stock Linux). Releases the core to
///   the scheduler so sibling cores can keep their turbo budget.
///
/// First ~8 192 iterations are tight spin (~30 µs), the next
/// `PARK_AFTER_SPINS - 8 192` iterations are tick-granular sleeps.
///
/// **Why park at all:** at low workload (e.g. `--cubes 1`) the engine
/// short-circuits `parallel_for` for `n_tasks <= 1` and never dispatches
/// to workers. Workers spinning at 100 % CPU on their pinned cores eat
/// thermal / turbo budget that main needs to push 10 K+ FPS, costing
/// ~40 % on single-core-bound workloads. Parking after ~1 ms cuts CPU
/// usage to near-zero in that case while still being well above any
/// realistic inter-dispatch gap inside a hot frame (sub-µs).
///
/// **Why not park sooner:** parking and waking via futex costs ~5-10 µs.
/// We want to be sure we don't pay that cost between back-to-back
/// dispatches in the same frame.
const PARK_AFTER_SPINS: u32 = 120_000;

fn worker_loop(shared: &'static Shared, worker_idx: usize) {
    IS_WORKER.with(|c| c.set(true));

    let mut last_epoch: u64 = 0;

    loop {
        let mut spins: u32 = 0;
        let cur = loop {
            let e = shared.epoch.load(Ordering::Acquire);
            if e != last_epoch {
                break e;
            }
            if shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            spins = spins.saturating_add(1);
            if spins < 8_192 {
                spin_loop();
            } else if spins < PARK_AFTER_SPINS {
                std::thread::sleep(std::time::Duration::from_nanos(1));
            } else {
                thread::park();
                // After unpark we loop back and reload epoch.
                spins = 0;
            }
        };
        last_epoch = cur;

        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }

        // SAFETY: the Acquire on `epoch` above synchronises with main's
        // Release-publish of the job slot, so the read is well-defined
        // and the slot is immutable until we signal `workers_done`.
        let job = unsafe { *shared.job.get() };

        if job.slices.is_null() {
            // Uniform-chunk mode (parallel_for_global). Only workers
            // with `worker_idx < active_workers` participate.
            if worker_idx < job.active_workers {
                let start = worker_idx * job.chunk;
                let end = job.n_tasks.min(start + job.chunk);
                // SAFETY: see comment on Job::data.
                unsafe { (job.invoke_range)(job.data, start, end) };
                shared.workers_done.fetch_add(1, Ordering::Release);
            }
        } else {
            // Per-worker-slices mode (parallel_for_numa). Every worker
            // reads its slice (may be empty) and always increments
            // `workers_done` so main can wait on `>= num_workers`.
            // SAFETY: `slices` points to a stack array of length
            // `num_workers` kept alive by main until after the barrier.
            let (s, e) = unsafe { *job.slices.add(worker_idx) };
            unsafe { (job.invoke_range)(job.data, s, e) };
            shared.workers_done.fetch_add(1, Ordering::Release);
        }
    }
}

#[inline]
unsafe fn noop_invoke_range(_data: *const (), _start: usize, _end: usize) {}

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
        global().parallel_for_global(N, |i| {
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
            global().parallel_for_global(n, |_| {
                counter.fetch_add(1, O::Relaxed);
            });
            assert_eq!(counter.load(O::Relaxed), n);
        }
    }

    #[test]
    fn parallel_for_zero_and_one_task() {
        init_pool_once();
        let _g = test_lock();
        global().parallel_for_global(0, |_| panic!("should not run"));
        let c = AtomicUsize::new(0);
        global().parallel_for_global(1, |i| {
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
        global().parallel_for_global(2, |_| {
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
            global().parallel_for_global(n, |i| {
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
            global().parallel_for_global(n, |i| {
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
            let entity_hits: Vec<AtomicUsize> = (0..n_entities).map(|_| AtomicUsize::new(0)).collect();
            let word_hits: Vec<AtomicUsize> = (0..hier_words).map(|_| AtomicUsize::new(0)).collect();

            let layout = bitmap_task_layout(hier_words);
            global().parallel_for_global(layout.n_tasks, |task_idx| {
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

        let small = pool.parallel_for_global(3, |_| {});
        assert_eq!(small.active_workers, 2);

        let large = pool.parallel_for_global(pool.num_threads() + 17, |_| {});
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
            global().parallel_for_global(n, |_| {
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
    global().parallel_for_global(n_chunks, |chunk_idx| {
        let start = chunk_idx * chunk_len;
        let end = (start + chunk_len).min(n);
        f(start, end);
    });
}
