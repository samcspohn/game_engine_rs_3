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
//! * **One global pool**, installed once via [`init_global`] or
//!   [`init_global_numa`]. All workers live as one set of OS threads.
//! * Each `parallel_for` dispatch carries a [`Scope`] that filters which
//!   workers participate. Per-worker scope membership is computed at
//!   init time (against a caller-supplied NUMA classifier) and baked
//!   into a per-scope worker map, so per-dispatch lookup is O(1).
//! * Workers outside the dispatch's scope **observe the epoch advance
//!   but do not run the closure and do not signal `done`**. The main
//!   thread's barrier waits only for the in-scope workers. Out-of-scope
//!   parked workers are NOT unparked, so a `Scope::LocalDRAM` dispatch
//!   doesn't burn CPU waking up remote-DRAM workers that won't run.
//! * One worker thread per non-main core, pinned via a caller-supplied
//!   start hook (typically `core_affinity::set_for_current`). Main thread
//!   is **always** a participant — work is split across (in-scope
//!   workers + 1 main) threads, of which the main thread runs its share
//!   inline.
//! * One primitive: [`ThreadPool::parallel_for_mode`]. **Static range
//!   partitioning** — every in-scope participant gets a contiguous,
//!   pre-computed slice of `0..n_tasks`. No per-task atomic, no work
//!   stealing, no adaptive splitting.
//! * **Workers do not park inside a frame.** Spin → `yield_now` loop runs
//!   for ~1 ms (an order of magnitude longer than the inter-dispatch gap
//!   inside a frame) before parking.
//! * No nested parallelism. `parallel_for` called from inside a worker
//!   panics — per project rules, no silent serial fallback.
//!
//! ## Safety
//!
//! `parallel_for_mode` publishes a stack-allocated [`Job`] via an
//! [`UnsafeCell`] under release/acquire on `epoch`. The Job borrow lives
//! for the duration of the call; the function blocks until every in-scope
//! worker has signalled completion (`workers_done == n_active_workers`),
//! so the borrow cannot dangle.

use std::cell::UnsafeCell;
use std::hint::spin_loop;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle, Thread};

// ────────────────────────────────────────────────────────────────────────
// Scope
// ────────────────────────────────────────────────────────────────────────

/// Which subset of workers should participate in a `parallel_for` dispatch.
///
/// All current scopes are evaluated at init time against per-worker
/// NUMA classification; per-dispatch lookup is O(1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Every worker (default — back-compat with `parallel_for`).
    All,
    /// Only workers on NUMA nodes with local DRAM (memory dies).
    /// Use for memory-bandwidth-bound workloads like the host-staging
    /// walk. On non-NUMA systems this is identical to `All`.
    LocalDRAM,
    // Future: GpuLocal — workers on the NUMA node nearest the GPU's PCIe root.
}

/// Number of variants in [`Scope`].
const N_SCOPES: usize = 2;

#[inline]
fn scope_index(scope: Scope) -> usize {
    match scope {
        Scope::All => 0,
        Scope::LocalDRAM => 1,
    }
}

// ────────────────────────────────────────────────────────────────────────
// Job slot
// ────────────────────────────────────────────────────────────────────────

/// Type-erased, lifetime-erased descriptor of a parallel-for job.
///
/// `chunk` is the number of tasks assigned to each participant
/// (`ceil(n_tasks / (n_active_workers + 1))`). Active worker `k` (0-based
/// position among in-scope workers) owns `[(k+1)*chunk, min((k+2)*chunk, n_tasks))`,
/// and main owns `[0, chunk)`.
#[derive(Clone, Copy)]
struct Job {
    n_tasks: usize,
    chunk: usize,
    data: *const (),
    invoke: unsafe fn(*const (), usize),
    scope: Scope,
}

unsafe impl Sync for Job {}
unsafe impl Send for Job {}

// ────────────────────────────────────────────────────────────────────────
// Scope map
// ────────────────────────────────────────────────────────────────────────

/// Pre-computed at init time. For each worker, says whether it
/// participates in this scope and if so, its 0-indexed position among
/// in-scope workers (used for static chunk partitioning).
pub(super) struct ScopeMap {
    pub(super) worker_to_active_idx: Vec<Option<usize>>,
    pub(super) n_active_workers: usize,
}

// ────────────────────────────────────────────────────────────────────────
// Shared state
// ────────────────────────────────────────────────────────────────────────

struct Shared {
    /// Bumped every time main publishes a new job. Workers compare against
    /// their `last_epoch` to detect new work. Release/Acquire here
    /// synchronises the non-atomic write to `job`.
    epoch: AtomicU64,

    /// Number of in-scope workers that have finished their assigned
    /// slice for the current job. Main blocks until this reaches
    /// `n_active_workers` for the current job's scope.
    workers_done: AtomicUsize,

    /// Job descriptor. Written by main BEFORE bumping `epoch`. Read by
    /// workers AFTER observing the new epoch.
    job: UnsafeCell<Job>,

    /// Set on shutdown.
    shutdown: AtomicBool,

    /// Diagnostic mode (enabled via `ENGINE_POOL_VERIFY=1`).
    verify_mode: AtomicBool,

    /// Per-job task-invocation counter for verify mode.
    tasks_invoked: AtomicUsize,

    /// Per-scope worker membership maps. Indexed via `scope_index(scope)`.
    scope_maps: [ScopeMap; N_SCOPES],
}

unsafe impl Sync for Shared {}

// ────────────────────────────────────────────────────────────────────────
// Pool
// ────────────────────────────────────────────────────────────────────────

pub struct ThreadPool {
    shared: &'static Shared,
    worker_threads: Vec<Thread>,
    /// Kept alive so the worker threads stay joinable for the program's
    /// lifetime. We never join (the pool is `'static`); shutdown is a
    /// process-exit, not a graceful teardown.
    #[allow(dead_code)]
    handles: Vec<JoinHandle<()>>,
    /// Total number of participants for `Scope::All` = `worker_threads.len() + 1`.
    num_participants: usize,
}

static POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Install the global pool with explicit per-worker NUMA classification.
///
/// `is_local_dram(worker_idx)` is called once at init (NOT from worker
/// threads) for every worker index `0..num_workers` and the result is
/// baked into the pool's per-scope worker maps.
///
/// `on_worker_start(worker_idx)` is invoked inside each worker thread
/// before it enters its loop — use this to call
/// `core_affinity::set_for_current(...)`.
///
/// Panics if called more than once, if `num_workers == 0`, or if any
/// worker fails to spawn. No fallbacks.
pub fn init_global_numa<F, C>(num_workers: usize, on_worker_start: F, is_local_dram: C)
where
    F: Fn(usize) + Send + Sync + 'static,
    C: Fn(usize) -> bool,
{
    assert!(num_workers > 0, "ThreadPool requires at least one worker");

    // ── Classify every worker into its scope membership ────────────────
    let local_dram_flags: Vec<bool> = (0..num_workers).map(&is_local_dram).collect();

    // Scope::All — every worker participates, active_idx == worker_idx.
    let all_map = ScopeMap {
        worker_to_active_idx: (0..num_workers).map(Some).collect(),
        n_active_workers: num_workers,
    };

    // Scope::LocalDRAM — only local-DRAM workers participate. Active
    // index is assigned in worker-index order.
    let mut ld_vec: Vec<Option<usize>> = Vec::with_capacity(num_workers);
    let mut next_ld_idx: usize = 0;
    for &is_local in &local_dram_flags {
        if is_local {
            ld_vec.push(Some(next_ld_idx));
            next_ld_idx += 1;
        } else {
            ld_vec.push(None);
        }
    }
    let local_dram_map = ScopeMap {
        worker_to_active_idx: ld_vec,
        n_active_workers: next_ld_idx,
    };

    // Order MUST match scope_index(): [All, LocalDRAM].
    let scope_maps: [ScopeMap; N_SCOPES] = [all_map, local_dram_map];

    let shared: &'static Shared = Box::leak(Box::new(Shared {
        epoch: AtomicU64::new(0),
        workers_done: AtomicUsize::new(0),
        job: UnsafeCell::new(Job {
            n_tasks: 0,
            chunk: 0,
            data: std::ptr::null(),
            invoke: noop_invoke,
            scope: Scope::All,
        }),
        shutdown: AtomicBool::new(false),
        verify_mode: AtomicBool::new(
            std::env::var("ENGINE_POOL_VERIFY")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
        ),
        tasks_invoked: AtomicUsize::new(0),
        scope_maps,
    }));

    let on_start = std::sync::Arc::new(on_worker_start);
    let (tx, rx) = std::sync::mpsc::sync_channel::<Thread>(num_workers);

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

/// Install the global pool with all workers assumed to have local DRAM
/// (`Scope::LocalDRAM` collapses to `Scope::All`). Back-compat — used
/// by tests and any caller without NUMA info.
pub fn init_global<F>(num_workers: usize, on_worker_start: F)
where
    F: Fn(usize) + Send + Sync + 'static,
{
    init_global_numa(num_workers, on_worker_start, |_| true);
}

/// Test-only convenience: install the global pool (if not already
/// installed) and return a guard that serialises `parallel_for` calls
/// across the entire test binary.
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
        .expect("engine ThreadPool not initialised — call util::thread_pool::init_global[_numa] first")
}

/// Whether the global pool has been initialised.
#[inline]
pub fn is_initialised() -> bool {
    POOL.get().is_some()
}

// ────────────────────────────────────────────────────────────────────────
// DispatchTiming
// ────────────────────────────────────────────────────────────────────────

/// Per-call timing breakdown returned by [`ThreadPool::parallel_for_timed`].
#[derive(Debug, Default, Clone, Copy)]
pub struct DispatchTiming {
    pub n_tasks:      usize,
    pub dispatch_ns:  u64,
    pub main_work_ns: u64,
    pub barrier_ns:   u64,
}

impl ThreadPool {
    /// Total participants for `Scope::All` (= workers + main). Back-compat.
    #[inline]
    pub fn num_threads(&self) -> usize {
        self.num_participants
    }

    /// Total worker thread count (independent of scope).
    #[inline]
    pub fn num_workers(&self) -> usize {
        self.worker_threads.len()
    }

    /// Number of workers in a particular scope (= main excluded).
    #[inline]
    pub fn num_active_workers(&self, scope: Scope) -> usize {
        self.shared.scope_maps[scope_index(scope)].n_active_workers
    }

    /// Whether `ENGINE_POOL_VERIFY=1` was set at pool init.
    #[inline]
    pub fn verify_enabled(&self) -> bool {
        self.shared.verify_mode.load(Ordering::Relaxed)
    }

    /// Equivalent to `parallel_for_mode(n, Scope::All, f)`.
    pub fn parallel_for<F>(&self, n_tasks: usize, f: F)
    where
        F: Fn(usize) + Sync,
    {
        self.parallel_for_mode(n_tasks, Scope::All, f);
    }

    /// Equivalent to `parallel_for_mode_timed(n, Scope::All, f)`.
    pub fn parallel_for_timed<F>(&self, n_tasks: usize, f: F) -> DispatchTiming
    where
        F: Fn(usize) + Sync,
    {
        self.parallel_for_mode_timed(n_tasks, Scope::All, f)
    }

    /// Dispatch with explicit scope filter. Only workers whose NUMA
    /// classification matches `scope` participate. Workers outside the
    /// scope observe the epoch advance (so they don't get stuck) but
    /// do not execute the closure and do not signal `done`. Main is
    /// always a participant regardless of scope.
    pub fn parallel_for_mode<F>(&self, n_tasks: usize, scope: Scope, f: F)
    where
        F: Fn(usize) + Sync,
    {
        if n_tasks == 0 {
            return;
        }
        if n_tasks == 1 {
            f(0);
            return;
        }

        assert!(
            !is_worker(),
            "ThreadPool::parallel_for_mode called from a worker thread — \
             nested parallelism is not supported",
        );

        let map = &self.shared.scope_maps[scope_index(scope)];
        let n_active_workers = map.n_active_workers;

        // Edge case: only main participates. Run all tasks inline.
        if n_active_workers == 0 {
            for i in 0..n_tasks {
                f(i);
            }
            return;
        }

        unsafe fn invoke_thunk<F: Fn(usize) + Sync>(data: *const (), task_idx: usize) {
            let f = unsafe { &*(data as *const F) };
            f(task_idx);
        }
        unsafe fn invoke_thunk_verify<F: Fn(usize) + Sync>(
            data: *const (),
            task_idx: usize,
        ) {
            let shared_ptr = WORKER_SHARED.with(|c| c.get());
            let shared = unsafe { &*shared_ptr };
            shared.tasks_invoked.fetch_add(1, Ordering::Relaxed);
            let f = unsafe { &*(data as *const F) };
            f(task_idx);
        }

        let n_participants = n_active_workers + 1;
        let chunk = n_tasks.div_ceil(n_participants);

        let verify = self.shared.verify_mode.load(Ordering::Relaxed);
        let invoke: unsafe fn(*const (), usize) = if verify {
            invoke_thunk_verify::<F>
        } else {
            invoke_thunk::<F>
        };

        let new_job = Job {
            n_tasks,
            chunk,
            data: &f as *const F as *const (),
            invoke,
            scope,
        };

        let shared = self.shared;

        shared.workers_done.store(0, Ordering::Relaxed);
        if verify {
            shared.tasks_invoked.store(0, Ordering::Relaxed);
        }

        // SAFETY: previous dispatch returned only when workers_done
        // reached n_active_workers for that dispatch's scope; no
        // in-scope worker is touching the slot. Out-of-scope workers
        // never read the slot.
        unsafe {
            *shared.job.get() = new_job;
        }

        shared.epoch.fetch_add(1, Ordering::Release);

        // Unpark every worker (in-scope AND out-of-scope). Out-of-scope
        // workers must observe the epoch advance and signal `done` so
        // main's barrier can safely move on without racing the single
        // `job` slot — see the comment on the barrier below.
        for t in &self.worker_threads {
            t.unpark();
        }

        // Main takes the first chunk [0, chunk). In-scope worker with
        // active_idx == k owns [(k+1)*chunk, min((k+2)*chunk, n_tasks)).
        let main_start = 0;
        let main_end = n_tasks.min(chunk);
        if verify {
            for i in main_start..main_end {
                shared.tasks_invoked.fetch_add(1, Ordering::Relaxed);
                f(i);
            }
        } else {
            for i in main_start..main_end {
                f(i);
            }
        }

        // Barrier on workers_done — wait for EVERY worker to acknowledge
        // the job, not just in-scope ones. This is required for
        // correctness: there is a single `job` slot, and main may not
        // overwrite it for the next dispatch until every worker has
        // finished reading it. An out-of-scope worker that hasn't yet
        // observed the current epoch would otherwise race main's write
        // of the next Job and either read torn data, double-run, or
        // miss its next in-scope dispatch entirely (manifests as
        // freezes / ballooning barrier times / dispatch spikes).
        // Out-of-scope workers signal `done` without running the
        // closure, so the cost is one atomic per worker per dispatch.
        let n_workers_total = self.worker_threads.len();
        let mut spins = 0u32;
        loop {
            if shared.workers_done.load(Ordering::Acquire) >= n_workers_total {
                break;
            }
            spins += 1;
            if spins < 10_000 {
                spin_loop();
            } else if spins < 1_000_000 {
                thread::yield_now();
            } else {
                thread::sleep(std::time::Duration::from_micros(100));
            }
        }

        if verify {
            let invoked = shared.tasks_invoked.load(Ordering::Acquire);
            if invoked != n_tasks {
                eprintln!(
                    "[pool-verify] FAIL n_tasks={n_tasks} chunk={chunk} \
                     scope={scope:?} n_active_workers={n_active_workers} \
                     n_participants={n_participants}"
                );
                eprintln!(
                    "[pool-verify]   expected main slice = [{main_start}, {main_end}) = {} tasks",
                    main_end - main_start,
                );
                for (worker_idx, active) in map.worker_to_active_idx.iter().enumerate() {
                    let Some(k) = *active else { continue; };
                    let s = (k + 1) * chunk;
                    let e = n_tasks.min(s + chunk);
                    let count = if s < n_tasks { e - s } else { 0 };
                    eprintln!(
                        "[pool-verify]   worker {worker_idx} (active_idx {k}) \
                         expected slice = [{s}, {e}) = {count} tasks",
                    );
                }
            }
            assert_eq!(
                invoked, n_tasks,
                "ThreadPool::parallel_for_mode verify FAIL: dispatched {n_tasks} tasks but \
                 only {invoked} closure invocations happened. Some participant skipped a task."
            );
        }
    }

    /// Like [`parallel_for_mode`] but returns a [`DispatchTiming`] breakdown.
    pub fn parallel_for_mode_timed<F>(
        &self,
        n_tasks: usize,
        scope: Scope,
        f: F,
    ) -> DispatchTiming
    where
        F: Fn(usize) + Sync,
    {
        use std::time::Instant;

        if n_tasks == 0 {
            return DispatchTiming { n_tasks, ..Default::default() };
        }
        if n_tasks == 1 {
            let t0 = Instant::now();
            f(0);
            return DispatchTiming {
                n_tasks,
                dispatch_ns:  0,
                main_work_ns: t0.elapsed().as_nanos() as u64,
                barrier_ns:   0,
            };
        }

        assert!(
            !is_worker(),
            "ThreadPool::parallel_for_mode_timed called from a worker thread — \
             nested parallelism is not supported",
        );

        let map = &self.shared.scope_maps[scope_index(scope)];
        let n_active_workers = map.n_active_workers;

        if n_active_workers == 0 {
            let t0 = Instant::now();
            for i in 0..n_tasks {
                f(i);
            }
            return DispatchTiming {
                n_tasks,
                dispatch_ns:  0,
                main_work_ns: t0.elapsed().as_nanos() as u64,
                barrier_ns:   0,
            };
        }

        unsafe fn invoke_thunk<F: Fn(usize) + Sync>(data: *const (), task_idx: usize) {
            let f = unsafe { &*(data as *const F) };
            f(task_idx);
        }
        unsafe fn invoke_thunk_verify<F: Fn(usize) + Sync>(
            data: *const (),
            task_idx: usize,
        ) {
            let shared_ptr = WORKER_SHARED.with(|c| c.get());
            let shared = unsafe { &*shared_ptr };
            shared.tasks_invoked.fetch_add(1, Ordering::Relaxed);
            let f = unsafe { &*(data as *const F) };
            f(task_idx);
        }

        let n_participants = n_active_workers + 1;
        let chunk          = n_tasks.div_ceil(n_participants);

        let verify = self.shared.verify_mode.load(Ordering::Relaxed);
        let invoke: unsafe fn(*const (), usize) = if verify {
            invoke_thunk_verify::<F>
        } else {
            invoke_thunk::<F>
        };

        let new_job = Job {
            n_tasks,
            chunk,
            data: &f as *const F as *const (),
            invoke,
            scope,
        };

        let shared = self.shared;

        shared.workers_done.store(0, Ordering::Relaxed);
        if verify {
            shared.tasks_invoked.store(0, Ordering::Relaxed);
        }

        unsafe { *shared.job.get() = new_job; }

        // ── Dispatch phase ───────────────────────────────────────────────────────────────
        let t_dispatch_start = Instant::now();
        shared.epoch.fetch_add(1, Ordering::Release);
        // Unpark every worker (in-scope AND out-of-scope). Required for
        // the single-`job`-slot invariant — see the barrier comment
        // below.
        for t in &self.worker_threads {
            t.unpark();
        }
        let dispatch_ns = t_dispatch_start.elapsed().as_nanos() as u64;

        // ── Main work phase ───────────────────────────────────────────────
        let t_main_start = Instant::now();
        let main_start = 0;
        let main_end   = n_tasks.min(chunk);
        if verify {
            for i in main_start..main_end {
                shared.tasks_invoked.fetch_add(1, Ordering::Relaxed);
                f(i);
            }
        } else {
            for i in main_start..main_end {
                f(i);
            }
        }
        let main_work_ns = t_main_start.elapsed().as_nanos() as u64;

        // ── Barrier phase ────────────────────────────────────────────────────────────────
        // Wait for EVERY worker to ack the job (see parallel_for_mode
        // for the full rationale).
        let t_barrier_start = Instant::now();
        let n_workers_total = self.worker_threads.len();
        let mut spins = 0u32;
        loop {
            if shared.workers_done.load(Ordering::Acquire) >= n_workers_total {
                break;
            }
            spins += 1;
            if spins < 10_000 {
                spin_loop();
            } else if spins < 1_000_000 {
                thread::yield_now();
            } else {
                thread::sleep(std::time::Duration::from_micros(100));
            }
        }
        let barrier_ns = t_barrier_start.elapsed().as_nanos() as u64;

        if verify {
            let invoked = shared.tasks_invoked.load(Ordering::Acquire);
            assert_eq!(
                invoked, n_tasks,
                "ThreadPool::parallel_for_mode_timed verify FAIL: dispatched {n_tasks} tasks \
                 but only {invoked} invocations happened.",
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
    static WORKER_SHARED: std::cell::Cell<*const Shared> = const { std::cell::Cell::new(std::ptr::null()) };
}

#[inline]
fn is_worker() -> bool {
    IS_WORKER.with(|c| c.get())
}

/// Threshold (in spin iterations) before a worker that has seen no new
/// job will park. See module docs for rationale.
const SPINS: u32 = 1024*16; //
const PARK_AFTER_SPINS: u32 = SPINS * 4;

fn worker_loop(shared: &'static Shared, worker_idx: usize) {
    IS_WORKER.with(|c| c.set(true));
    WORKER_SHARED.with(|c| c.set(shared as *const _));

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
            if spins < SPINS {
                spin_loop();
            } else if spins < PARK_AFTER_SPINS {
                thread::yield_now();
            } else {
                thread::park();
                spins = 0;
            }
        };
        last_epoch = cur;

        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }

        // SAFETY: the Acquire on `epoch` above synchronises with main's
        // Release-publish of the job slot.
        let job = unsafe { *shared.job.get() };
        let map = &shared.scope_maps[scope_index(job.scope)];

        match map.worker_to_active_idx[worker_idx] {
            Some(active_idx) => {
                // In scope: run our static slice using active_idx for chunk math.
                // Worker with active_idx == k owns
                // [(k+1)*chunk, min((k+2)*chunk, n_tasks)).
                let start = (active_idx + 1) * job.chunk;
                if start < job.n_tasks {
                    let end = job.n_tasks.min(start + job.chunk);
                    for i in start..end {
                        // SAFETY: `data` points to caller's `&F` which is
                        // alive for the entire `parallel_for_mode` call.
                        unsafe { (job.invoke)(job.data, i) };
                    }
                }
                shared.workers_done.fetch_add(1, Ordering::Release);
            }
            None => {
                // Out of scope: don't run the closure, but DO signal
                // done. Main's barrier waits for every worker to ack
                // so it can safely overwrite the single `job` slot for
                // the next dispatch. Skipping the signal here would
                // let main race-overwrite the slot mid-observation,
                // causing torn reads, duplicate execution, missed
                // dispatches, and freezes.
                shared.workers_done.fetch_add(1, Ordering::Release);
            }
        }
    }
}

#[inline]
unsafe fn noop_invoke(_data: *const (), _task_idx: usize) {}

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
        init_pool_once();
        let _g = test_lock();
        let counter = AtomicUsize::new(0);
        global().parallel_for(2, |_| {
            counter.fetch_add(1, O::Relaxed);
        });
        assert_eq!(counter.load(O::Relaxed), 2);
    }

    #[test]
    fn parallel_for_sweep_exact_coverage() {
        init_pool_once();
        let _g = test_lock();
        let p = global().num_threads();
        let max_n = (p * 64).max(64) * 2;
        for n in 0..=max_n {
            let visited: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            global().parallel_for(n, |i| {
                visited[i].fetch_add(1, O::Relaxed);
            });
            for (i, c) in visited.iter().enumerate() {
                let v = c.load(O::Relaxed);
                assert_eq!(v, 1, "n={n}: task {i} visited {v} times");
            }
        }
    }

    #[test]
    fn parallel_for_b2b_coverage_no_drops() {
        init_pool_once();
        let _g = test_lock();
        for round in 0..200 {
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

    #[test]
    fn parallel_for_bitmap_walk_arithmetic() {
        init_pool_once();
        let _g = test_lock();
        const WORDS_PER_TASK: usize = 256;
        for hier_words in [
            0usize, 1, WORDS_PER_TASK - 1, WORDS_PER_TASK, WORDS_PER_TASK + 1,
            WORDS_PER_TASK * 3, WORDS_PER_TASK * 3 + 17,
        ] {
            let n_entities = hier_words * 32;
            let entity_hits: Vec<AtomicUsize> = (0..n_entities).map(|_| AtomicUsize::new(0)).collect();
            let word_hits: Vec<AtomicUsize> = (0..hier_words).map(|_| AtomicUsize::new(0)).collect();

            let n_tasks = if hier_words == 0 { 0 } else { hier_words.div_ceil(WORDS_PER_TASK) };
            global().parallel_for(n_tasks, |task_idx| {
                let word_base = task_idx * WORDS_PER_TASK;
                let word_end = (word_base + WORDS_PER_TASK).min(hier_words);
                for word_idx in word_base..word_end {
                    word_hits[word_idx].fetch_add(1, O::Relaxed);
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

    #[test]
    fn parallel_for_mode_all_covers_every_task() {
        init_pool_once();
        let _g = test_lock();
        const N: usize = 10_000;
        let counts: Vec<AtomicUsize> = (0..N).map(|_| AtomicUsize::new(0)).collect();
        global().parallel_for_mode(N, Scope::All, |i| {
            counts[i].fetch_add(1, O::Relaxed);
        });
        for (i, c) in counts.iter().enumerate() {
            assert_eq!(c.load(O::Relaxed), 1, "task {i}");
        }
    }

    #[test]
    fn parallel_for_mode_local_dram_covers_every_task_when_all_workers_local() {
        // The test pool init goes through init_global which marks every
        // worker as local DRAM, so LocalDRAM should be equivalent to All.
        init_pool_once();
        let _g = test_lock();
        const N: usize = 10_000;
        let counts: Vec<AtomicUsize> = (0..N).map(|_| AtomicUsize::new(0)).collect();
        global().parallel_for_mode(N, Scope::LocalDRAM, |i| {
            counts[i].fetch_add(1, O::Relaxed);
        });
        for (i, c) in counts.iter().enumerate() {
            assert_eq!(c.load(O::Relaxed), 1, "task {i}");
        }
    }

    #[test]
    fn parallel_for_mode_back_to_back_alternating_scopes() {
        // Stress test: alternate All and LocalDRAM dispatches to make sure
        // out-of-scope workers don't get stuck or skip future dispatches.
        init_pool_once();
        let _g = test_lock();
        for round in 0..50 {
            let n = 200 + round * 13;
            let counter = AtomicUsize::new(0);
            let scope = if round % 2 == 0 { Scope::All } else { Scope::LocalDRAM };
            global().parallel_for_mode(n, scope, |_| {
                counter.fetch_add(1, O::Relaxed);
            });
            assert_eq!(counter.load(O::Relaxed), n, "round {round} scope {scope:?}");
        }
    }

    #[test]
    fn num_active_workers_matches_classifier() {
        init_pool_once();
        let _g = test_lock();
        let pool = global();
        let nw = pool.num_workers();
        assert_eq!(pool.num_active_workers(Scope::All), nw);
        assert_eq!(
            pool.num_active_workers(Scope::LocalDRAM),
            nw,
            "default init_global marks every worker as local DRAM",
        );
    }
}

// ────────────────────────────────────────────────────────────────────────
// Convenience helpers
// ────────────────────────────────────────────────────────────────────────

/// Run `f(chunk_start, chunk_end)` over every consecutive
/// `[chunk_start, chunk_end)` slice of `0..n` with chunk size `chunk_len`.
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
