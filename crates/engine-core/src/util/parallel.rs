//! Backend-agnostic parallel dispatch.
//!
//! Thin wrapper over the engine's data-parallel primitive — a blocking
//! `parallel_for` over `Range<usize>` — that lets the executing backend be
//! swapped at pool-init time without touching any call site:
//!
//!   * [`BackendKind::MyPool`] — the in-tree cursor/steal scheduler in
//!     [`my_thread_pool`] (default).
//!   * [`BackendKind::Rayon`] — a dedicated `rayon::ThreadPool`.
//!   * [`BackendKind::Orx`] — `orx-parallel` over a persistent
//!     `rayon-core` pool (its default std-thread runner spawns fresh OS
//!     threads per dispatch; see `Pool::Orx`).
//!
//! Usage mirrors the old `my_thread_pool::global` API:
//! ```ignore
//! parallel::global::init(BackendKind::from_env(), n_workers); // once, at startup
//! parallel::global::parallel_for(0..n, |sub| { /* sub: Range<usize> */ });
//! let n = parallel::global::num_threads();
//! ```
//!
//! Semantics every backend must honour (they are what call sites rely on):
//!   * `parallel_for` returns only after every invocation of `body` has
//!     completed — side effects are visible to the caller on return.
//!   * `body` receives disjoint sub-ranges that exactly cover the input
//!     range; it may be invoked concurrently from many threads.
//!   * A panic in `body` propagates to the caller (no silent loss).
//!   * `num_threads()` counts every thread that may participate in a
//!     dispatch, including the calling thread where the backend uses it.

use std::ops::Range;
use std::sync::OnceLock;

use super::my_thread_pool;

pub use super::my_thread_pool::BitmapTaskLayout;

// ─────────────────────────────────────────────────────────────────────────
// Backend selection
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// In-tree work-stealing pool (`util::my_thread_pool`).
    MyPool,
    /// `rayon` with a dedicated (non-global) thread pool.
    Rayon,
    /// `rayon`, but dispatching via `ThreadPool::broadcast`: chunk `k`
    /// always runs on pool thread `k`. Deterministic static partitioning
    /// — the same chunk→thread mapping every dispatch, like
    /// [`BackendKind::MyPool`] with `work_stealing = false` — which
    /// preserves per-core cache/NUMA locality across frames. No load
    /// balancing: a straggler chunk bounds the dispatch.
    RayonBroadcast,
    /// `orx-parallel` running over a persistent `rayon-core` pool.
    /// Chunk→thread assignment is whichever thread wins the pull from
    /// the shared concurrent iterator — it cannot preserve affinity.
    Orx,
}

impl BackendKind {
    /// Read the backend from `ENGINE_POOL_BACKEND` (`mypool` | `rayon` |
    /// `orx`), defaulting to [`BackendKind::MyPool`] when unset. An
    /// unrecognised value panics rather than silently falling back.
    pub fn from_env() -> Self {
        match std::env::var("ENGINE_POOL_BACKEND") {
            Ok(s) => s.parse().unwrap_or_else(|e: String| panic!("{e}")),
            Err(_) => BackendKind::MyPool,
        }
    }
}

impl std::str::FromStr for BackendKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "mypool" | "my_pool" | "my-thread-pool" => Ok(BackendKind::MyPool),
            "rayon" => Ok(BackendKind::Rayon),
            "rayon-broadcast" | "rayon_broadcast" | "broadcast" => Ok(BackendKind::RayonBroadcast),
            "orx" | "orx-parallel" | "orx_parallel" => Ok(BackendKind::Orx),
            other => Err(format!(
                "unknown pool backend {other:?} (expected mypool | rayon | rayon-broadcast | orx)"
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Pool
// ─────────────────────────────────────────────────────────────────────────

/// How many tasks per participating thread the chunked backends (rayon,
/// orx) split a dispatch into. >1 gives their schedulers slack to balance
/// uneven bodies; my_thread_pool does its own intra-range stealing and
/// ignores this.
const TASKS_PER_THREAD: usize = 4;

pub enum Pool {
    MyPool(my_thread_pool::ThreadPool),
    Rayon(rayon::ThreadPool),
    RayonBroadcast(rayon::ThreadPool),
    /// orx-parallel's default runner (`StdDefaultPool`) spawns fresh OS
    /// threads via `std::thread::scope` on *every* dispatch, which costs
    /// ~250 us/call and doesn't scale. We instead keep a persistent
    /// `rayon_core::ThreadPool` (`rayon::ThreadPool` is the same type)
    /// and hand orx a `RunnerWithPool` borrowing it per dispatch.
    Orx { pool: rayon::ThreadPool },
}

impl Pool {
    /// Build a pool of `num_threads` total participants on `kind`.
    pub fn new(kind: BackendKind, num_threads: usize) -> Self {
        assert!(num_threads >= 1, "pool requires at least 1 thread");
        match kind {
            BackendKind::MyPool => Pool::MyPool(my_thread_pool::ThreadPool::new(num_threads)),
            BackendKind::Rayon => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .thread_name(|i| format!("engine-rayon-{i}"))
                    .build()
                    .expect("failed to build rayon thread pool");
                Pool::Rayon(pool)
            }
            BackendKind::RayonBroadcast => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .thread_name(|i| format!("engine-rayon-bc-{i}"))
                    .build()
                    .expect("failed to build rayon broadcast thread pool");
                Pool::RayonBroadcast(pool)
            }
            BackendKind::Orx => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .thread_name(|i| format!("engine-orx-{i}"))
                    .build()
                    .expect("failed to build orx worker thread pool");
                Pool::Orx { pool }
            }
        }
    }

    /// Backend-specific options. `work_stealing` currently only applies to
    /// [`BackendKind::MyPool`] (see `ThreadPool::with_options`); passing
    /// `false` with any other backend panics so the A/B toggle can't be
    /// silently ignored.
    pub fn with_options(kind: BackendKind, num_threads: usize, work_stealing: bool) -> Self {
        match kind {
            BackendKind::MyPool => Pool::MyPool(my_thread_pool::ThreadPool::with_options(
                num_threads,
                work_stealing,
            )),
            _ => {
                assert!(
                    work_stealing,
                    "work_stealing=false is only supported by the mypool backend"
                );
                Self::new(kind, num_threads)
            }
        }
    }

    pub fn backend(&self) -> BackendKind {
        match self {
            Pool::MyPool(_) => BackendKind::MyPool,
            Pool::Rayon(_) => BackendKind::Rayon,
            Pool::RayonBroadcast(_) => BackendKind::RayonBroadcast,
            Pool::Orx { .. } => BackendKind::Orx,
        }
    }

    /// Total number of threads that may participate in a dispatch.
    #[inline]
    pub fn num_threads(&self) -> usize {
        match self {
            Pool::MyPool(p) => p.num_threads(),
            Pool::Rayon(p) => p.current_num_threads(),
            Pool::RayonBroadcast(p) => p.current_num_threads(),
            Pool::Orx { pool } => pool.current_num_threads(),
        }
    }

    /// Blocking parallel-for over `range`. See the module docs for the
    /// contract all backends share.
    pub fn parallel_for<F>(&self, range: Range<usize>, body: F)
    where
        F: Fn(Range<usize>) + Sync + Send,
    {
        let total = range.end.saturating_sub(range.start);
        if total == 0 {
            return;
        }
        match self {
            Pool::MyPool(p) => p.parallel_for(range, body),
            Pool::Rayon(p) => {
                use rayon::prelude::*;
                let n_tasks = chunk_count(total, p.current_num_threads());
                let start = range.start;
                p.install(|| {
                    (0..n_tasks).into_par_iter().for_each(|k| {
                        body(chunk_range(start, total, k, n_tasks));
                    });
                });
            }
            Pool::RayonBroadcast(p) => {
                // One chunk per pool thread, and chunk `k` always runs on
                // thread `k` — `BroadcastContext::index()` is the stable
                // per-thread id. This is what preserves cross-dispatch
                // cache/NUMA affinity. Threads whose chunk is empty
                // (total < num_threads) return immediately.
                let n_tasks = p.current_num_threads();
                let start = range.start;
                p.broadcast(|ctx: rayon::BroadcastContext<'_>| {
                    let r = chunk_range(start, total, ctx.index(), n_tasks);
                    if !r.is_empty() {
                        body(r);
                    }
                });
            }
            Pool::Orx { pool } => {
                use orx_parallel::*;
                let n_tasks = chunk_count(total, pool.current_num_threads());
                let start = range.start;
                (0..n_tasks)
                    .into_par()
                    .with_runner(RunnerWithPool::from(pool))
                    .for_each(|k| {
                        body(chunk_range(start, total, k, n_tasks));
                    });
            }
        }
    }
}

/// Number of sub-range tasks for the chunked backends: `TASKS_PER_THREAD`
/// per thread, but never more tasks than items.
#[inline]
fn chunk_count(total: usize, threads: usize) -> usize {
    threads.saturating_mul(TASKS_PER_THREAD).clamp(1, total)
}

/// Even split of `[start, start + total)` into `n_tasks` contiguous
/// chunks; chunk `k` is `[start + total*k/n, start + total*(k+1)/n)`.
/// Same arithmetic as `my_thread_pool`'s cursor seeding.
#[inline]
fn chunk_range(start: usize, total: usize, k: usize, n_tasks: usize) -> Range<usize> {
    let s = start + total * k / n_tasks;
    let e = start + total * (k + 1) / n_tasks;
    s..e
}

// ─────────────────────────────────────────────────────────────────────────
// Global pool (process-wide singleton)
// ─────────────────────────────────────────────────────────────────────────

pub mod global {
    use super::*;

    static POOL: OnceLock<Pool> = OnceLock::new();

    /// Initialise the global pool. Returns `false` if it was already
    /// initialised; the caller can `assert!` to surface a double-init bug
    /// loudly (no silent fallback).
    pub fn init(kind: BackendKind, num_threads: usize) -> bool {
        POOL.set(Pool::new(kind, num_threads)).is_ok()
    }

    /// Initialise with backend-specific options (see [`Pool::with_options`]).
    pub fn init_with_options(kind: BackendKind, num_threads: usize, work_stealing: bool) -> bool {
        POOL.set(Pool::with_options(kind, num_threads, work_stealing))
            .is_ok()
    }

    /// Access the global pool. Panics if `init` was not called — a clear
    /// crash rather than a silent default-config pool.
    #[inline]
    pub fn pool() -> &'static Pool {
        POOL.get()
            .expect("parallel::global::init(backend, n) must be called before use")
    }

    pub fn is_initialized() -> bool {
        POOL.get().is_some()
    }

    #[inline]
    pub fn num_threads() -> usize {
        pool().num_threads()
    }

    #[inline]
    pub fn parallel_for<F>(range: Range<usize>, body: F)
    where
        F: Fn(Range<usize>) + Sync + Send,
    {
        pool().parallel_for(range, body)
    }
}

// Convenience re-export so callers can write `parallel::parallel_for`.
pub use global::parallel_for;

// ──────────────────────────────────────────────────────────────────────
// Bitmap-task layout helper
// ──────────────────────────────────────────────────────────────────────
//
// Same geometry as `my_thread_pool::bitmap_task_layout`, but sized from
// the *wrapper's* global pool so it works with every backend. Keeps the
// sim and staging passes on matching slab sizes regardless of backend.

const BITMAP_MIN_WORDS_PER_TASK: usize = 8;
const BITMAP_MAX_WORDS_PER_TASK: usize = 256;
const BITMAP_TARGET_TASKS_PER_THREAD: usize = 2;

/// Choose a shared task layout for bitmap-indexed per-entity work.
#[inline]
pub fn bitmap_task_layout(n_words: usize) -> BitmapTaskLayout {
    if n_words == 0 {
        return BitmapTaskLayout {
            words_per_task: BITMAP_MIN_WORDS_PER_TASK,
            n_tasks: 0,
        };
    }
    let target_tasks = global::num_threads()
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

// ─────────────────────────────────────────────────────────────────────────
// Tests — every backend honours the shared parallel_for contract.
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

    const BACKENDS: [BackendKind; 4] = [
        BackendKind::MyPool,
        BackendKind::Rayon,
        BackendKind::RayonBroadcast,
        BackendKind::Orx,
    ];

    /// Every index visited exactly once, including non-divisible sizes.
    #[test]
    fn coverage_every_index_visited_once_all_backends() {
        for kind in BACKENDS {
            let pool = Pool::new(kind, 4);
            for &n in &[1usize, 7, 64, 1024, 31_337] {
                let visits: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(0)).collect();
                pool.parallel_for(0..n, |r| {
                    for i in r {
                        visits[i].fetch_add(1, Ordering::Relaxed);
                    }
                });
                for (i, v) in visits.iter().enumerate() {
                    let c = v.load(Ordering::Relaxed);
                    assert_eq!(c, 1, "{kind:?}: index {i} visited {c} times (n = {n})");
                }
            }
        }
    }

    /// Non-zero range starts are honoured (offset arithmetic).
    #[test]
    fn offset_range_all_backends() {
        for kind in BACKENDS {
            let pool = Pool::new(kind, 4);
            let total = AtomicUsize::new(0);
            pool.parallel_for(100..1_100, |r| {
                let mut local = 0;
                for i in r {
                    assert!((100..1_100).contains(&i), "{kind:?}: index {i} out of range");
                    local += i;
                }
                total.fetch_add(local, Ordering::Relaxed);
            });
            let expected: usize = (100..1_100).sum();
            assert_eq!(total.load(Ordering::Relaxed), expected, "{kind:?}");
        }
    }

    /// Empty ranges are a no-op on every backend.
    #[test]
    fn empty_range_all_backends() {
        for kind in BACKENDS {
            let pool = Pool::new(kind, 4);
            pool.parallel_for(0..0, |_| panic!("body must not run for an empty range"));
            #[allow(clippy::reversed_empty_ranges)]
            pool.parallel_for(10..5, |_| panic!("body must not run for a reversed range"));
        }
    }

    /// Side effects are visible after `parallel_for` returns (blocking).
    #[test]
    fn parallel_for_blocks_until_done_all_backends() {
        for kind in BACKENDS {
            let pool = Pool::new(kind, 4);
            let n = 10_000usize;
            let total = AtomicUsize::new(0);
            pool.parallel_for(0..n, |r| {
                let mut local = 0usize;
                for i in r {
                    local = local.wrapping_add(i);
                }
                total.fetch_add(local, Ordering::Relaxed);
            });
            let expected: usize = (0..n).sum();
            assert_eq!(total.load(Ordering::Relaxed), expected, "{kind:?}");
        }
    }

    /// Panics inside the body propagate to the caller on every backend.
    #[test]
    fn panic_propagates_all_backends() {
        for kind in BACKENDS {
            let pool = Pool::new(kind, 4);
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.parallel_for(0..64, |_| panic!("intentional"));
            }));
            assert!(r.is_err(), "{kind:?}: panic should have propagated");
        }
    }

    #[test]
    fn backend_kind_parses() {
        assert_eq!("mypool".parse::<BackendKind>(), Ok(BackendKind::MyPool));
        assert_eq!("rayon".parse::<BackendKind>(), Ok(BackendKind::Rayon));
        assert_eq!(
            "rayon-broadcast".parse::<BackendKind>(),
            Ok(BackendKind::RayonBroadcast)
        );
        assert_eq!("orx-parallel".parse::<BackendKind>(), Ok(BackendKind::Orx));
        assert!("threads4days".parse::<BackendKind>().is_err());
    }

    /// The broadcast backend's whole point: chunk `k` runs on pool
    /// thread `k`, every dispatch. Record which OS thread handled each
    /// chunk across repeated dispatches and require a stable mapping.
    #[test]
    fn rayon_broadcast_chunk_to_thread_mapping_is_stable() {
        let threads = 4;
        let pool = Pool::new(BackendKind::RayonBroadcast, threads);
        let n = 1_000usize;
        let baseline: Vec<std::thread::ThreadId> = {
            let slots: Vec<std::sync::Mutex<Option<std::thread::ThreadId>>> =
                (0..threads).map(|_| std::sync::Mutex::new(None)).collect();
            pool.parallel_for(0..n, |r| {
                let k = r.start * threads / n;
                *slots[k].lock().unwrap() = Some(std::thread::current().id());
            });
            slots
                .into_iter()
                .map(|s| s.into_inner().unwrap().expect("chunk not dispatched"))
                .collect()
        };
        for _ in 0..50 {
            let slots: Vec<std::sync::Mutex<Option<std::thread::ThreadId>>> =
                (0..threads).map(|_| std::sync::Mutex::new(None)).collect();
            pool.parallel_for(0..n, |r| {
                let k = r.start * threads / n;
                *slots[k].lock().unwrap() = Some(std::thread::current().id());
            });
            for (k, s) in slots.into_iter().enumerate() {
                let tid = s.into_inner().unwrap().expect("chunk not dispatched");
                assert_eq!(
                    tid, baseline[k],
                    "chunk {k} moved to a different thread between dispatches"
                );
            }
        }
    }
}
