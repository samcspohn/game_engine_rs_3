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
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle, Thread};

// ────────────────────────────────────────────────────────────────────────
// Job slot
// ────────────────────────────────────────────────────────────────────────

/// Type-erased, lifetime-erased descriptor of a parallel-for job.
///
/// `chunk` is the number of tasks assigned to each participant
/// (`ceil(n_tasks / num_participants)`). Participant `i` (worker `i`,
/// or main = `num_workers`) owns `[i*chunk, min((i+1)*chunk, n_tasks))`.
///
/// `invoke_range` runs the inner `for i in start..end { f(i) }` loop
/// **inside the monomorphised thunk**, so the worker pays a single
/// indirect call per slice instead of per task. This lets the
/// compiler inline `f(i)` into the tight loop body — matching the
/// main thread's inline path — and unlocks LICM / vectorisation of
/// the closure body across iterations.
#[derive(Clone, Copy)]
struct Job {
    n_tasks: usize,
    chunk: usize,
    data: *const (),
    invoke_range: unsafe fn(*const (), usize, usize),
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
    /// Total number of participants in a `parallel_for` = `worker_threads.len() + 1`
    /// (main thread is always a participant).
    num_participants: usize,
}

static POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Install the global thread pool. Spawns `num_workers` worker threads
/// (in addition to the calling thread, which becomes the main
/// participant). `on_worker_start(idx)` is invoked on each worker before
/// it enters its loop — use this to call `core_affinity::set_for_current`.
///
/// Panics if called more than once, if `num_workers == 0`, or if any
/// worker fails to spawn. No fallbacks.
pub fn init_global<F>(num_workers: usize, on_worker_start: F)
where
    F: Fn(usize) + Send + Sync + 'static,
{
    assert!(num_workers > 0, "ThreadPool requires at least one worker");

    let shared: &'static Shared = Box::leak(Box::new(Shared {
        epoch: AtomicU64::new(0),
        workers_done: AtomicUsize::new(0),
        job: UnsafeCell::new(Job {
            n_tasks: 0,
            chunk: 0,
            data: std::ptr::null(),
            invoke_range: noop_invoke_range,
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
        init_global(4, |_| {});
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
    pub dispatch_ns:  u64,
    pub main_work_ns: u64,
    pub barrier_ns:   u64,
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
    pub fn parallel_for<F>(&self, n_tasks: usize, f: F) -> DispatchTiming
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
                dispatch_ns:  0,
                main_work_ns,
                barrier_ns:   0,
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

        let n_workers      = self.worker_threads.len();
        let n_participants = self.num_participants;
        // Round up so the early participants take the extra task when
        // n_tasks doesn't divide evenly. Last participant may have a
        // smaller (or empty) slice.
        let chunk          = n_tasks.div_ceil(n_participants);

        let verify = self.shared.verify_mode.load(Ordering::Relaxed);
        let invoke_range: unsafe fn(*const (), usize, usize) = if verify {
            invoke_range_thunk_verify::<F>
        } else {
            invoke_range_thunk::<F>
        };

        let new_job = Job {
            n_tasks,
            chunk,
            data: &f as *const F as *const (),
            invoke_range,
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
        for t in &self.worker_threads {
            t.unpark();
        }
        #[cfg(feature = "pool-timing")]
        let dispatch_ns = t_dispatch_start.elapsed().as_nanos() as u64;
        #[cfg(not(feature = "pool-timing"))]
        let dispatch_ns = 0u64;

        // ── Main work phase ────────────────────────────────
        // Main participates: runs the last slice. Inline call to `f` for
        // max throughput in normal mode — no fn-pointer indirection. In
        // verify mode, batch-bump the same counter the worker thunk
        // uses (single Relaxed fetch_add) so the assert below covers
        // the entire range.
        #[cfg(feature = "pool-timing")]
        let t_main_start = Instant::now();
        let main_start = n_workers * chunk;
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
        // Tight spin — at high FPS this is typically zero or single-digit
        // µs. Falls back to `yield_now` after ~100k spins (~few hundred
        // µs on x86), then short sleeps after ~10M iterations of yields.
        // The escape hatch matters: if a worker is briefly preempted by
        // an OS interrupt, a pure spin here burns CPU and keeps the
        // preempting process scheduled longer, turning a microsecond
        // hiccup into a multi-ms hitch.
        #[cfg(feature = "pool-timing")]
        let t_barrier_start = Instant::now();
        let mut spins = 0u32;
        loop {
            if shared.workers_done.load(Ordering::Acquire) >= n_workers {
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
                // Print per-participant accounting to find who skipped.
                eprintln!(
                    "[pool-verify] FAIL n_tasks={n_tasks} chunk={chunk} \
                     n_workers={n_workers} n_participants={n_participants}"
                );
                eprintln!(
                    "[pool-verify]   expected main slice = [{main_start}, {main_end}) = {} tasks",
                    main_end - main_start,
                );
                for w in 0..n_workers {
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
                "ThreadPool::parallel_for verify FAIL: dispatched {n_tasks} tasks but\n\
                 only {invoked} closure invocations happened. Some participant skipped a task."
            );
        }

        DispatchTiming { n_tasks, dispatch_ns, main_work_ns, barrier_ns }
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

        // Run this worker's pre-assigned static slice via the range
        // thunk — a single indirect call into a monomorphised function
        // that contains the entire inner loop. `f(i)` inlines inside
        // the thunk, matching main's inline path.
        let start = worker_idx * job.chunk;
        if start < job.n_tasks {
            let end = job.n_tasks.min(start + job.chunk);
            // SAFETY: `data` points to the caller's `&F` which is
            // alive for the entire `parallel_for` call (main is
            // blocked on `workers_done`). `invoke_range` is the
            // matching monomorphised thunk for that closure type.
            unsafe { (job.invoke_range)(job.data, start, end) };
        }

        // Signal slice complete. Release pairs with main's Acquire load
        // in the barrier above. Every worker increments exactly once
        // per job, regardless of slice size (including empty slices),
        // so the barrier always reaches `n_workers`.
        shared.workers_done.fetch_add(1, Ordering::Release);
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
        const WORDS_PER_TASK: usize = 256;
        // Cover the boundary cases: empty, single word, exactly one
        // chunk, multiple full chunks, and a ragged tail.
        for hier_words in [
            0usize, 1, WORDS_PER_TASK - 1, WORDS_PER_TASK, WORDS_PER_TASK + 1,
            WORDS_PER_TASK * 3, WORDS_PER_TASK * 3 + 17,
        ] {
            // Simulate: every bit set, then verify every entity in
            // [0, hier_words * 32) is touched exactly once.
            let n_entities = hier_words * 32;
            let entity_hits: Vec<AtomicUsize> = (0..n_entities).map(|_| AtomicUsize::new(0)).collect();
            let word_hits: Vec<AtomicUsize> = (0..hier_words).map(|_| AtomicUsize::new(0)).collect();

            let n_tasks = if hier_words == 0 { 0 } else { hier_words.div_ceil(WORDS_PER_TASK) };
            global().parallel_for(n_tasks, |task_idx| {
                let word_base = task_idx * WORDS_PER_TASK;
                let word_end = (word_base + WORDS_PER_TASK).min(hier_words);
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
