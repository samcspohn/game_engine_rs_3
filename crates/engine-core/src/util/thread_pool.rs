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
//! * **Worker idle policy.** Three-phase loop:
//!   pure `spin_loop` for [`HOT_SPIN_ITERS`], then
//!   `sleep(Duration::from_nanos(1))` (kernel hrtimer, ~50 µs
//!   ride-out) up to [`PARK_AFTER_SPINS`], then
//!   `parking_lot_core::park` keyed on the worker's per-node
//!   parking address. Dispatch wakes via `unpark_all(key)` — one
//!   syscall per active NUMA node bucket regardless of how many
//!   workers are parked on it. The middle `sleep(1ns)` phase is
//!   crucial for steady-state throughput: between dispatches
//!   workers sit in the kernel's hrtimer queue rather than the
//!   parking_lot bucket, so main's `unpark_all` is a cheap no-op
//!   token-store rather than a real futex-wake syscall.
//! * **Work stealing (numa mode).** [`parallel_for_numa`] partitions
//!   work into per-worker [`Cursor`]s (atomic `start`/`end` pair,
//!   cacheline-padded). Workers consume their own cursor in
//!   `STEAL_GRAIN`-sized chunks via `start.fetch_add`; when own
//!   cursor is exhausted, they scan peers (same-node first via
//!   pre-built `steal_order`) and steal the back half via CAS on
//!   `end`. This recovers throughput when a worker is OS-preempted
//!   mid-frame — peers steal the preempted worker's tail and the
//!   barrier no longer waits on the slow worker.
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
// Work-stealing cursors
// ────────────────────────────────────────────────────────────────────────

/// Grain size for own-cursor consumption. ~hundreds of ns/task ×
/// 256 ≈ tens-of-µs per chunk — large enough to amortise the
/// per-chunk `fetch_add`, small enough that a slow worker's tail
/// can be usefully split by stealers.
const STEAL_GRAIN: usize = 256;

/// Owner consumes from `start` (monotonically increasing via
/// `fetch_add(STEAL_GRAIN)`); stealers shrink `end` via CAS to claim
/// the back half. Both halves of the resulting split are processed
/// without overlap because the owner clamps its working window to
/// `min(s + STEAL_GRAIN, end_loaded)` after every `fetch_add`.
///
/// Cacheline-padded (via [`CachePadded`]) so workers' high-frequency
/// `fetch_add`s on their own cursor don't ping-pong adjacent
/// cursors' cache lines.
struct Cursor {
    start: AtomicUsize,
    end: AtomicUsize,
}

type PaddedCursor = CachePadded<Cursor>;

impl Cursor {
    fn new_range(s: usize, e: usize) -> Self {
        Self {
            start: AtomicUsize::new(s),
            end: AtomicUsize::new(e),
        }
    }
}

/// Attempt to steal the back half of `c`. Returns the stolen
/// `[start, end)` (which the caller processes directly) or `None`
/// if the cursor has too little work to bother splitting. CAS loop
/// is bounded by other stealers shrinking `end`, so it terminates.
#[inline]
fn try_steal(c: &PaddedCursor) -> Option<(usize, usize)> {
    loop {
        let s = c.start.load(Ordering::Acquire);
        let e = c.end.load(Ordering::Acquire);
        if e <= s || e - s < STEAL_GRAIN {
            return None;
        }
        // `+1` so odd remainder stays with the owner; avoids
        // pointless steals of single trailing tasks.
        let new_end = s + (e - s + 1) / 2;
        if new_end >= e || new_end <= s {
            return None;
        }
        if c.end
            .compare_exchange(e, new_end, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return Some((new_end, e));
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Job slot
// ────────────────────────────────────────────────────────────────────────

/// Type-erased, lifetime-erased descriptor of a parallel-for job.
///
/// Two slicing modes share this slot:
///
/// * **Uniform chunk** (`cursors.is_null() == true`): worker `i` runs
///   `[i * chunk, min((i+1) * chunk, n_tasks))`. Used by
///   [`ThreadPool::parallel_for_global`].
/// * **Per-worker cursors / stealing** (`cursors.is_null() == false`):
///   worker `i` owns `cursors[i]`; main owns `cursors[num_workers]`.
///   Each participant consumes its own cursor in `STEAL_GRAIN` chunks
///   via `start.fetch_add`, then steals back-half of peer cursors via
///   CAS on `end`. Used by [`ThreadPool::parallel_for_numa`]. The
///   cursors buffer is stack-allocated for the duration of the
///   dispatch call (same lifetime contract as the closure data).
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
    /// per-worker ranges. In cursors mode, used only for verify-mode
    /// accounting.
    n_tasks: usize,
    chunk: usize,
    active_workers: usize,
    data: *const (),
    invoke_range: unsafe fn(*const (), usize, usize),
    /// Null in uniform mode; non-null in per-worker-slices mode. Points
    /// to `n_cursors` `PaddedCursor`s. Worker `w` owns `cursors[w]`;
    /// `cursors[num_workers]` is main's slot. Workers consume their
    /// own cursor in `STEAL_GRAIN` chunks via `start.fetch_add`, then
    /// steal the back half of peers' cursors via CAS on `end`.
    cursors: *const PaddedCursor,
    /// Number of cursors in `cursors` (== num_workers + 1: one per
    /// worker plus one for main).
    n_cursors: usize,
    /// Bitmask of NUMA nodes that participate in this dispatch. Bit
    /// `n` set => node `n`'s workers must wake, do their slice, and
    /// increment `workers_done`. Bit `n` clear => node `n`'s workers
    /// observe the epoch but **skip** the slice and the barrier
    /// increment. `parallel_for_global` always sets this to `!0`
    /// (all nodes active); `parallel_for_numa` computes it from the
    /// partitions. Caps at 64 NUMA nodes — fine in practice (largest
    /// real systems today are 16-node SGI; 64 leaves headroom).
    active_nodes_mask: u64,
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

    /// Sharded parking_lot_core park keys. Flat vector of stable
    /// cacheline-padded bytes (Shared is leaked to `'static`); each
    /// entry's address is one parking_lot_core key. A node's workers
    /// are partitioned into shards of at most `PARK_SHARD_SIZE`
    /// workers; `node_shard_range[n]` gives that node's [start, end)
    /// slice in `park_keys`. Sharding spreads parked workers across
    /// distinct parking_lot internal buckets so a wake doesn't serialise
    /// on a single bucket mutex with hundreds of waiters — critical at
    /// 256-thread scale where a single-key bucket becomes the dominant
    /// wake-path bottleneck. The byte value is unused; only the address
    /// matters. Cacheline padding also keeps unpark/park atomic traffic
    /// on different lines.
    park_keys: Vec<CachePadded<u8>>,
    /// Per-node range [start, end) into `park_keys`.
    node_shard_range: Vec<(u32, u32)>,
    /// Per-worker shard index into `park_keys` (precomputed at init).
    worker_shard: Vec<u32>,
    /// Per-worker relay flag. A relay worker uses a pure-spin idle
    /// policy and is responsible for issuing a node-local
    /// `parking_lot_core::unpark_all` for its node's shards when it
    /// observes a new epoch where its node is active. Designed to
    /// avoid the main thread issuing cross-NUMA unpark syscalls.
    is_relay: Vec<bool>,
    /// Bitmask of NUMA nodes that have a relay worker assigned.
    /// `wake_workers` / `wake_workers_on_node` skip nodes in this
    /// mask — the relay handles its node locally. Always excludes
    /// `main_node` (main itself wakes its own node).
    relay_served_mask: u64,
    /// Per-node "local" epoch. For relay-served nodes the relay
    /// re-publishes the global epoch here AFTER it observes a new
    /// global epoch but BEFORE local unpark, so its peers poll a
    /// node-local cacheline instead of bouncing the global one
    /// across NUMA on every spin iteration. Workers on
    /// non-relay-served nodes (incl. main_node) ignore this and
    /// continue polling `shared.epoch` directly.
    node_epoch: Vec<CachePadded<AtomicU64>>,

    /// `steal_order[i]` lists peers participant `i` (worker idx
    /// 0..num_workers, or main at `num_workers`) tries to steal
    /// from when its own cursor is exhausted. Same-NUMA-node peers
    /// come first, then peers on other nodes. Built once at init.
    steal_order: Vec<Vec<usize>>,
}

unsafe impl Sync for Shared {}

// ────────────────────────────────────────────────────────────────────────
// Pool
// ────────────────────────────────────────────────────────────────────────

pub struct ThreadPool {
    shared: &'static Shared,
    /// Thread handles retained from worker registration; previously
    /// used for `unpark()` on dispatch but now superseded by
    /// `parking_lot_core::unpark_all` on per-node park keys. Kept so
    /// pool init's registration handshake (workers send their
    /// `Thread` over the mpsc channel) stays unchanged.
    #[allow(dead_code)]
    worker_threads: Vec<Thread>,
    /// `worker_threads_by_node[n]` — see `worker_threads`. Retained
    /// for the same reason; the per-node wake path now uses
    /// `shared.park_keys[n]` instead.
    #[allow(dead_code)]
    worker_threads_by_node: Vec<Vec<Thread>>,
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
    pub workers: Vec<WorkerSpec>,
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
/// Initialise the global thread pool.
///
/// `on_worker_start(idx, is_relay)` runs once on each newly-spawned
/// worker thread (before the worker enters its idle loop). Use it for
/// CPU pinning / affinity, scheduler-priority bumps, thread-local
/// setup. The `is_relay` flag is true for the one worker per non-main
/// NUMA node that acts as that node's wake-relay — these workers
/// should typically get a *node-wide affinity mask* (e.g. via
/// `set_current_thread_affinity_mask`) rather than a single-core pin,
/// so the OS can migrate them off a preempted core without their
/// whole node starving.
pub fn init_global<F>(cfg: PoolConfig, on_worker_start: F)
where
    F: Fn(usize, bool) + Send + Sync + 'static,
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

    // Build steal_order: one entry per participant (workers + main
    // at index `num_workers`). Each entry lists peers in
    // same-NUMA-node-first order, then peers on other nodes (cyclic
    // round-robin from `my_node + 1`).
    let main_idx_for_init = num_workers;
    let participant_node = |i: usize| -> u32 {
        if i == main_idx_for_init {
            cfg.main_node
        } else {
            worker_nodes[i]
        }
    };
    let mut steal_order: Vec<Vec<usize>> = Vec::with_capacity(num_workers + 1);
    for i in 0..=num_workers {
        let my_node = participant_node(i);
        let mut order = Vec::with_capacity(num_workers);
        for j in 0..=num_workers {
            if j != i && participant_node(j) == my_node {
                order.push(j);
            }
        }
        for offset in 1..cfg.num_nodes {
            let n = (my_node + offset) % cfg.num_nodes;
            for j in 0..=num_workers {
                if j != i && participant_node(j) == n {
                    order.push(j);
                }
            }
        }
        steal_order.push(order);
    }

    // Build sharded park-keys + relay assignment. Each non-main node
    // with at least one worker gets a relay = the first worker on that
    // node. The relay uses pure-spin idle and does a node-local
    // unpark_all on dispatch; main skips waking relay-served nodes.
    let park_shard_size = std::env::var("ENGINE_PARK_SHARD_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_PARK_SHARD_SIZE);
    let enable_relay = std::env::var("ENGINE_DISABLE_RELAY")
        .map(|v| v != "1" && !v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let mut park_keys: Vec<CachePadded<u8>> = Vec::new();
    let mut node_shard_range: Vec<(u32, u32)> = Vec::with_capacity(cfg.num_nodes as usize);
    let mut worker_shard: Vec<u32> = vec![0u32; num_workers];
    let mut is_relay: Vec<bool> = vec![false; num_workers];
    let mut relay_served_mask: u64 = 0;
    for n in 0..cfg.num_nodes as usize {
        let start = park_keys.len() as u32;
        let node_workers = &workers_by_node[n];
        let shards = if node_workers.is_empty() {
            1
        } else {
            (node_workers.len() + park_shard_size - 1) / park_shard_size
        };
        for _ in 0..shards {
            park_keys.push(CachePadded::new(0u8));
        }
        let end = park_keys.len() as u32;
        node_shard_range.push((start, end));
        for (local_idx, &w) in node_workers.iter().enumerate() {
            worker_shard[w] = start + (local_idx / park_shard_size) as u32;
        }
        if enable_relay && (n as u32) != cfg.main_node && !node_workers.is_empty() {
            is_relay[node_workers[0]] = true;
            relay_served_mask |= 1u64 << n;
        }
    }

    let shared: &'static Shared = Box::leak(Box::new(Shared {
        epoch: CachePadded::new(AtomicU64::new(0)),
        workers_done: CachePadded::new(AtomicUsize::new(0)),
        job: UnsafeCell::new(Job {
            n_tasks: 0,
            chunk: 0,
            active_workers: 0,
            data: std::ptr::null(),
            invoke_range: noop_invoke_range,
            cursors: std::ptr::null(),
            n_cursors: 0,
            active_nodes_mask: 0,
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
        node_shard_range,
        worker_shard,
        is_relay,
        relay_served_mask,
        node_epoch: (0..cfg.num_nodes)
            .map(|_| CachePadded::new(AtomicU64::new(0)))
            .collect(),
        steal_order,
    }));

    let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Thread)>(num_workers);
    let on_start = std::sync::Arc::new(on_worker_start);

    let mut handles = Vec::with_capacity(num_workers);
    for idx in 0..num_workers {
        let tx = tx.clone();
        let on_start = on_start.clone();
        let node = worker_nodes[idx];
        let relay_flag = shared.is_relay[idx];
        let handle = thread::Builder::new()
            .name(format!("engine-worker-{idx}"))
            .spawn(move || {
                on_start(idx, relay_flag);
                tx.send((idx, thread::current()))
                    .expect("worker failed to register Thread handle");
                drop(tx);
                worker_loop(shared, idx, node);
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

    // Build per-node clones of worker handles. `Thread` is cheap to
    // clone (it's an Arc internally). Used by `wake_workers_on_node`
    // to avoid cross-NUMA `unpark` IPIs when only some nodes have work.
    let mut worker_threads_by_node: Vec<Vec<Thread>> =
        (0..cfg.num_nodes).map(|_| Vec::new()).collect();
    for (idx, t) in worker_threads.iter().enumerate() {
        worker_threads_by_node[worker_nodes[idx] as usize].push(t.clone());
    }

    let pool = ThreadPool {
        shared,
        worker_threads,
        worker_threads_by_node,
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
            |_, _| {},
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
    let target_tasks = global()
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
    /// Wake every worker that may be sleeping on the wake-futex.
    ///
    /// On Linux: single `futex_wake_all` syscall. The dispatcher always
    /// wakes ALL workers — extras (e.g. trailing workers in a uniform
    /// dispatch where `active_workers < num_workers`) cheaply re-spin
    /// after observing no epoch change for their slice. The cost of
    /// over-waking is one syscall round-trip plus N short cache reads,
    /// far less than the previous N per-thread `unpark()` loop.
    ///
    /// On non-Linux: falls back to the `unpark()`-per-handle loop.
    /// Must be called AFTER `epoch.fetch_add(_, Release)` so any
    /// worker re-loading wake_seq before going to sleep observes the
    /// new value (else `futex_wait` returns `EAGAIN`).
    /// Wake all parked workers via parking_lot_core. One
    /// `unpark_all` syscall per NUMA node bucket — far cheaper than
    /// the previous N per-thread `unpark()` loop. Workers that are
    /// still in the spin phase observe the new epoch directly; only
    /// truly parked workers consume a wake.
    #[inline]
    fn wake_workers(&self) {
        let served = self.shared.relay_served_mask;
        for n in 0..self.shared.node_shard_range.len() {
            if (served >> n) & 1 != 0 {
                continue; // relay on this node handles its own wake
            }
            let (s, e) = self.shared.node_shard_range[n];
            for k in &self.shared.park_keys[s as usize..e as usize] {
                let key = &**k as *const u8 as usize;
                unsafe {
                    parking_lot_core::unpark_all(key, parking_lot_core::DEFAULT_UNPARK_TOKEN);
                }
            }
        }
    }

    /// Wake only workers parked on `node` via parking_lot_core.
    /// Used by `parallel_for_numa` to skip cross-NUMA wakes when a
    /// node's partition is empty. If a relay is assigned to `node`,
    /// no-op — the relay handles wakes locally on dispatch.
    #[inline]
    fn wake_workers_on_node(&self, node: u32) {
        if (node as usize) >= self.shared.node_shard_range.len() {
            return;
        }
        if (self.shared.relay_served_mask >> node) & 1 != 0 {
            return;
        }
        let (s, e) = self.shared.node_shard_range[node as usize];
        for k in &self.shared.park_keys[s as usize..e as usize] {
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
                shared
                    .tasks_invoked
                    .fetch_add(end - start, Ordering::Relaxed);
            }
        }

        let n_workers = self.worker_threads.len();
        let n_active_participants = self.num_participants.min(n_tasks);
        let active_workers = n_active_participants.saturating_sub(1);
        // Round up so the early participants take the extra task when
        // n_tasks doesn't divide evenly. Last participant may have a
        // smaller (or empty) slice.
        let chunk = n_tasks.div_ceil(n_active_participants);

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
            cursors: std::ptr::null(),
            n_cursors: 0,
            // Global mode: all nodes participate. The `active_workers`
            // gating already serialises which worker idxs do work in
            // uniform mode; the node mask is just a wake hint here.
            active_nodes_mask: !0u64,
        };

        let shared = self.shared;

        // Reset the worker-done counter BEFORE publishing. Relaxed is
        // fine: the Release on `epoch.fetch_add` below provides the
        // happens-before edge to workers.
        shared.workers_done.store(0, Ordering::Relaxed);
        if verify {
            shared.tasks_invoked.store(0, Ordering::Relaxed);
        }

        // Reset per-worker timing slots for participating workers
        // (uniform mode leaves trailing workers idle, but every
        // worker observes the new epoch — so we reset the full
        // participating range plus zero out trailing workers' t_done
        // sentinel below). Doing it before the publish keeps the
        // window between zero-out and worker store as small as
        // possible so spurious "wake=0" reads can't happen.
        #[cfg(feature = "pool-timing")]
        let t_publish_ns = reset_worker_ts(shared, 0..n_workers);
        #[cfg(not(feature = "pool-timing"))]
        let _ = n_workers; // keep variable in non-timing builds

        // SAFETY: previous parallel_for returned only when
        // workers_done == n_workers, so no worker is touching the slot.
        unsafe {
            *shared.job.get() = new_job;
        }

        // ── Dispatch phase: publish + wake workers ──────────────────────
        #[cfg(feature = "pool-timing")]
        let t_dispatch_start = Instant::now();
        shared.epoch.fetch_add(1, Ordering::Release);
        // One syscall on Linux (futex_wake_all) regardless of how many
        // workers are sleeping; falls back to N `unpark()` calls
        // elsewhere. We intentionally wake ALL workers even when
        // `active_workers < num_workers` — inactive workers wake,
        // observe no slice for them, and go back to sleep, far
        // cheaper than the previous per-handle unpark loop.
        self.wake_workers();
        #[cfg(feature = "pool-timing")]
        let dispatch_ns = t_dispatch_start.elapsed().as_nanos() as u64;
        #[cfg(not(feature = "pool-timing"))]
        let dispatch_ns = 0u64;

        // ── Main work phase ────────────────────────────────
        #[cfg(feature = "pool-timing")]
        let t_main_start = Instant::now();
        let main_start = active_workers * chunk;
        let main_end = n_tasks.min(main_start + chunk);
        for i in main_start..main_end {
            f(i);
        }
        if verify && main_end > main_start {
            shared
                .tasks_invoked
                .fetch_add(main_end - main_start, Ordering::Relaxed);
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

        // Same six values get computed in both branches but the
        // non-timing branch uses zeros (no syscalls / atomic loads).
        // The function `aggregate_worker_stats` is itself feature-gated.
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

    /// NUMA-aware parallel_for with work stealing. `partitions[n]`
    /// is the task-index range owned by NUMA node `n` (one entry per
    /// node, in node-id order; must have
    /// `partitions.len() == self.num_nodes()`). Each participant gets
    /// an initial cursor on its node's slab; when a participant's
    /// own cursor is exhausted it steals from peers' cursors
    /// (same-node first, then remote).
    ///
    /// Within each node, that node's range is statically split among
    /// the node's participants (workers pinned to this node, plus
    /// main if `node == main_node`). Main takes the last slice of
    /// its node.
    ///
    /// Panics if any node has a non-empty partition but no
    /// participants (would silently drop tasks).
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
            return DispatchTiming {
                n_tasks: 0,
                ..Default::default()
            };
        }

        let num_workers = self.worker_threads.len();
        let main_idx = num_workers;
        // One cursor per participant (workers + main). Empty cursor
        // for participants whose node got no work; non-active-node
        // workers don't even read it (they skip via active_nodes_mask).
        let mut cursors: Vec<PaddedCursor> = (0..=num_workers)
            .map(|_| CachePadded::new(Cursor::new_range(0, 0)))
            .collect();
        let mut active_nodes_mask: u64 = 0;
        let mut active_workers_count: usize = 0;

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
            assert!(
                n < 64,
                "parallel_for_numa: NUMA node id {n} exceeds active_nodes_mask width (64)"
            );
            active_nodes_mask |= 1u64 << n;
            active_workers_count += node_workers.len();
            let n_node_tasks = range.len();
            let chunk = n_node_tasks.div_ceil(participants);
            let base = range.start;
            for (i, &widx) in node_workers.iter().enumerate() {
                let s = base + i * chunk;
                let e = (base + (i + 1) * chunk).min(range.end);
                let (s, e) = if s < e { (s, e) } else { (s, s) };
                cursors[widx] = CachePadded::new(Cursor::new_range(s, e));
            }
            if main_here {
                let i = node_workers.len();
                let s = base + i * chunk;
                let e = (base + (i + 1) * chunk).min(range.end);
                let (s, e) = if s < e { (s, e) } else { (s, s) };
                cursors[main_idx] = CachePadded::new(Cursor::new_range(s, e));
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
                shared
                    .tasks_invoked
                    .fetch_add(end - start, Ordering::Relaxed);
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

        // Reset per-worker timing slots and capture publish timestamp.
        #[cfg(feature = "pool-timing")]
        let t_publish_ns = reset_worker_ts(shared, 0..num_workers);

        // `cursors` is a stack-allocated buffer kept alive until
        // after the barrier; workers read it via raw pointer.
        let new_job = Job {
            n_tasks: n_tasks_total,
            chunk: 0, // unused in cursors mode
            active_workers: active_workers_count,
            data: &f as *const F as *const (),
            invoke_range,
            cursors: cursors.as_ptr(),
            n_cursors: cursors.len(),
            active_nodes_mask,
        };
        unsafe {
            *shared.job.get() = new_job;
        }

        #[cfg(feature = "pool-timing")]
        let t_dispatch_start = Instant::now();
        shared.epoch.fetch_add(1, Ordering::Release);
        let full_mask = if self.num_nodes as u32 >= 64 {
            !0u64
        } else {
            (1u64 << self.num_nodes as u32) - 1
        };
        if active_nodes_mask == full_mask {
            self.wake_workers();
        } else {
            let mut mask = active_nodes_mask;
            while mask != 0 {
                let n = mask.trailing_zeros();
                self.wake_workers_on_node(n);
                mask &= mask - 1;
            }
        }
        #[cfg(feature = "pool-timing")]
        let dispatch_ns = t_dispatch_start.elapsed().as_nanos() as u64;
        #[cfg(not(feature = "pool-timing"))]
        let dispatch_ns = 0u64;

        // ── Main work phase: own cursor + stealing ────────────────
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

        #[cfg(feature = "pool-timing")]
        let t_barrier_start = Instant::now();
        let mut spins = 0u32;
        loop {
            if shared.workers_done.load(Ordering::Acquire) >= active_workers_count {
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

        // Now safe to drop the cursors buffer — every worker has signaled done.
        drop(cursors);

        if verify {
            let invoked = shared.tasks_invoked.load(Ordering::Acquire);
            assert_eq!(
                invoked, n_tasks_total,
                "parallel_for_numa verify FAIL: dispatched {n_tasks_total} tasks but \
                 {invoked} closure invocations happened",
            );
        }

        #[cfg(feature = "pool-timing")]
        let (wmin, wavg, wmax, kmin, kavg, kmax) =
            aggregate_worker_stats(shared, 0..num_workers, t_publish_ns);
        #[cfg(not(feature = "pool-timing"))]
        let (wmin, wavg, wmax, kmin, kavg, kmax) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);

        DispatchTiming {
            n_tasks: n_tasks_total,
            active_workers: num_workers,
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

    /// NUMA topology getters.
    #[inline]
    pub fn num_nodes(&self) -> u32 {
        self.num_nodes
    }
    #[inline]
    pub fn main_node(&self) -> u32 {
        self.main_node
    }
    #[inline]
    pub fn worker_node(&self, worker_idx: usize) -> u32 {
        self.worker_nodes[worker_idx]
    }
    #[inline]
    pub fn workers_on_node(&self, node: u32) -> &[usize] {
        &self.workers_by_node[node as usize]
    }
    /// Whether worker `idx` is a NUMA-relay worker (pure-spin idle,
    /// responsible for waking its node's peers on dispatch). Useful
    /// for the caller's `on_worker_start` closure to skip CPU pinning
    /// for relays so the OS can migrate them if a core is preempted.
    #[inline]
    pub fn is_relay_worker(&self, idx: usize) -> bool {
        self.shared.is_relay.get(idx).copied().unwrap_or(false)
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

fn worker_loop(shared: &'static Shared, worker_idx: usize, worker_node: u32) {
    IS_WORKER.with(|c| c.set(true));

    let mut last_epoch: u64 = 0;
    let shard = shared.worker_shard[worker_idx] as usize;
    let park_key = &*shared.park_keys[shard] as *const u8 as usize;
    let is_relay = shared.is_relay[worker_idx];
    let node_relay_served = (shared.relay_served_mask >> worker_node) & 1 != 0;
    // Non-relay workers on a relay-served node poll their node's
    // local epoch (republished by the relay) so the hot spin/park
    // loop reads a node-local cacheline. Everyone else polls the
    // global epoch directly (main_node workers, nodes with no
    // relay, and the relay itself which must observe main's bumps).
    let epoch_src: &AtomicU64 = if node_relay_served && !is_relay {
        &*shared.node_epoch[worker_node as usize]
    } else {
        &shared.epoch
    };

    loop {
        let mut spins: u32 = 0;
        let cur = loop {
            let e = epoch_src.load(Ordering::Acquire);
            if e != last_epoch {
                break e;
            }
            if shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            if is_relay {
                // Relay: pure spin so cross-node wake latency is
                // bounded by epoch propagation (one cache miss) +
                // local unpark, not by a futex-wake syscall + IPI.
                spin_loop();
                continue;
            }
            if spins < HOT_SPIN_ITERS {
                spins = spins.saturating_add(1);
                spin_loop();
                continue;
            }
            // Workers on relay-served nodes skip the sleep(1ns)
            // middle phase and park directly — the relay's
            // node-local unpark is the wake mechanism, so the
            // hrtimer dwell of sleep(1ns) would only add latency.
            if !node_relay_served && spins < PARK_AFTER_SPINS {
                spins = spins.saturating_add(1);
                // sleep(1ns) -> ~50µs scheduler tick, releases the core
                // so neighbouring SMT siblings keep their turbo budget.
                thread::sleep(std::time::Duration::from_nanos(1));
                continue;
            }
            if shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            // parking_lot_core park keyed on per-node-shard address.
            // `validate` re-checks under the bucket lock so we don't
            // miss a wake that races with the load above: if the
            // dispatcher (or relay, via node_epoch) has already
            // published, validate returns false and park is a no-op.
            unsafe {
                parking_lot_core::park(
                    park_key,
                    || {
                        epoch_src.load(Ordering::Acquire) == last_epoch
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

        // Relay's first job after observing a new epoch: republish
        // it into the node-local epoch slot (so peers see it on a
        // node-local cacheline), then unpark any parked peers. The
        // store/unpark ordering matters: peers' park-validate
        // re-reads `epoch_src` (== node_epoch[my_node]) under the
        // bucket lock, so the Release store must happen-before the
        // unpark.
        if is_relay {
            // SAFETY: Acquire on epoch above synchronises with main's
            // Release-publish of the job slot; safe to read mask.
            let mask = unsafe { (*shared.job.get()).active_nodes_mask };
            if (mask >> worker_node) & 1 != 0 {
                shared.node_epoch[worker_node as usize].store(cur, Ordering::Release);
                let (s, e) = shared.node_shard_range[worker_node as usize];
                for k in &shared.park_keys[s as usize..e as usize] {
                    let key = &**k as *const u8 as usize;
                    unsafe {
                        parking_lot_core::unpark_all(
                            key,
                            parking_lot_core::DEFAULT_UNPARK_TOKEN,
                        );
                    }
                }
            } else {
                // Node not active this dispatch — still republish so
                // peers' epoch comparison advances and they don't
                // skip an unrelated future dispatch.
                shared.node_epoch[worker_node as usize].store(cur, Ordering::Release);
            }
        }

        // Record epoch-observation timestamp BEFORE doing any work, so
        // (t_seen - t_publish) measures pure wake-up / spin-detection
        // latency. Relaxed is fine: main reads after the barrier, which
        // synchronises via `workers_done`.
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
        // Release-publish of the job slot, so the read is well-defined
        // and the slot is immutable until we signal `workers_done`.
        let job = unsafe { *shared.job.get() };

        if job.cursors.is_null() {
            // Uniform-chunk mode (parallel_for_global). Only workers
            // with `worker_idx < active_workers` participate.
            if worker_idx < job.active_workers {
                let start = worker_idx * job.chunk;
                let end = job.n_tasks.min(start + job.chunk);
                // SAFETY: see comment on Job::data.
                unsafe { (job.invoke_range)(job.data, start, end) };
                #[cfg(feature = "pool-timing")]
                {
                    let now_ns = shared.anchor.elapsed().as_nanos() as u64;
                    shared.worker_ts[worker_idx]
                        .t_done
                        .store(now_ns, Ordering::Relaxed);
                }
                shared.workers_done.fetch_add(1, Ordering::Release);
            } else {
                // Inactive worker still observed the epoch; mark
                // t_done so post-barrier aggregation can skip it
                // cleanly via the (t_done == 0) sentinel filter.
                // (We deliberately do NOT touch t_done here — a
                // zero t_done plus a non-zero t_seen tells the
                // aggregator this worker was inactive this dispatch.)
            }
        } else {
            // Cursors / stealing mode (parallel_for_numa). Workers
            // on nodes whose bit is set in `active_nodes_mask` drain
            // their own cursor and steal from peers; workers on
            // inactive nodes skip everything (main's barrier waits
            // on `active_workers_count`, not `num_workers`).
            let node_active = (job.active_nodes_mask >> worker_node) & 1 != 0;
            if node_active {
                // SAFETY: `cursors` points to a stack buffer of
                // length `n_cursors` kept alive by main until after
                // the barrier; `steal_order[worker_idx]` is built at
                // init and 'static.
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
            // Else: worker is on an idle node — do nothing. t_done
            // stays 0 (sentinel for "didn't participate"); main does
            // not count us in the barrier.
        }
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
    // Drain own cursor in STEAL_GRAIN chunks.
    let own = unsafe { &*cursors.add(own_idx) };
    loop {
        let s = own.start.fetch_add(STEAL_GRAIN, Ordering::AcqRel);
        let e = own.end.load(Ordering::Acquire);
        if s >= e {
            break;
        }
        let chunk_end = (s + STEAL_GRAIN).min(e);
        unsafe { (invoke_range)(data, s, chunk_end) };
    }
    // Steal loop: walk peers in `steal_order`. Restart from the
    // beginning after every successful steal so same-node peers
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
            let entity_hits: Vec<AtomicUsize> =
                (0..n_entities).map(|_| AtomicUsize::new(0)).collect();
            let word_hits: Vec<AtomicUsize> =
                (0..hier_words).map(|_| AtomicUsize::new(0)).collect();

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
