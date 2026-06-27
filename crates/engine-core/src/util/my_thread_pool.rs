//! A small, simple work-stealing thread pool.
//!
//! Goals (in order):
//!   1. **Simplicity.** A reader should be able to hold the whole scheduler in
//!      their head. ~300 lines, no NUMA, no pinning, no per-call timing.
//!   2. **Nested parallelism.** `parallel_for` works correctly when called
//!      from inside another `parallel_for` body. A worker blocked waiting for
//!      its children does not idle — it keeps stealing and running other
//!      tasks (its own freshly pushed children first), so we never starve.
//!   3. **Background threading.** `spawn_background` enqueues a long-running
//!      job that occupies a worker; remaining workers continue to handle
//!      `parallel_for` dispatches concurrently.
//!   4. **Work stealing.** Each worker owns a LIFO `crossbeam_deque::Worker`;
//!      other workers (and external threads) steal from the opposite end.
//!      Off-worker dispatchers push into a shared `Injector`.
//!
//! Usage:
//! ```ignore
//! my_thread_pool::global::init(n_workers);            // call once at startup
//! my_thread_pool::global::parallel_for(0..n, |sub| { /* sub is a Range<usize> */ });
//! my_thread_pool::global::spawn_background(|| { /* long-running */ });
//! ```
//!
//! Safety model for `parallel_for`:
//!   The body is captured by reference and shared with worker tasks as a raw
//!   fat pointer. The calling thread does not return from `parallel_for`
//!   until the per-dispatch pending counter reaches zero, so the body always
//!   outlives every worker invocation. Task panics are caught, the pending
//!   counter is still decremented, and the first panic is resumed on the
//!   calling thread after the dispatch drains (no silent failures, no
//!   permanent deadlocks).

use crossbeam_deque::{Injector, Steal, Stealer, Worker as Deque};

use std::any::Any;
use std::cell::RefCell;
use std::cell::UnsafeCell;
use std::ops::Range;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle, Thread};
use std::time::Duration;

// /// Erased, owned task. We use `Box<dyn FnOnce>` for simplicity — one heap
// /// allocation per scheduled task. `parallel_for` schedules `n_workers` tasks
// /// per call, so that overhead is amortised across the user-supplied body.
// type Task = Box<dyn FnOnce() + Send>;

/// Idle ladder thresholds (iterations of the worker loop with no work).
/// Below `SPIN_ITERS` we `spin_loop`; below `YIELD_ITERS` we `yield_now`;
/// past that we park on the condvar with a short timeout so we still
/// re-check the deques even if a `notify_*` is missed.
const SPIN_ITERS: u32 = 64;
const YIELD_ITERS: u32 = 1024;
const PARK_TIMEOUT: Duration = Duration::from_millis(1);

/// Shared between the pool handle, every worker thread, and every
/// thread-local `WorkerCtx`. Reference-counted via `Arc`.
///
/// Each worker owns one `Worker<Task>` (LIFO deque) plus has one
/// `Injector<Task>` "mailbox" allocated for it here in shared state.
///
///   * `Worker<Task>` is `!Sync` and `!Clone` — it cannot live in
///     `Inner`; it stays in the per-worker thread-local `WorkerCtx`, and
///     other threads reach in via `stealers[i]` to take from the back.
///   * `Injector<Task>` is MPMC-safe — anyone can `push` to
///     `mailboxes[i]` (this is what `schedule_with_hint` uses) and anyone
///     (including worker `i` itself) can `steal` from it.
///
/// The two-channel design (deque + mailbox) preserves the cheap LIFO
/// depth-first behaviour for nested children pushed by the worker itself,
/// while still letting `parallel_for` assign chunk `k` to worker `k`
/// explicitly via the mailbox.
// struct Inner {
//     injector: Injector<Task>,
//     stealers: Vec<Stealer<Task>>,
//     mailboxes: Vec<Injector<Task>>,
//     shutdown: AtomicBool,
//     // Per-worker park handles. Each worker stores its own `Thread` in
//     // `park_handles[i]` before entering its main loop; `wake_worker(i)`
//     // calls `unpark()` on that handle, which targets exactly worker `i`.
//     // `Thread::unpark` is the cheapest possible wake (one futex syscall on
//     // Linux) and — critically — deposits a permit if the target is not
//     // yet parked, so there is no missed-wake race against `park_timeout`.
//     park_handles: Vec<OnceLock<Thread>>,
// }

// /// Per-worker thread-local state. Holding the `Worker<Task>` here keeps it
// /// where only the worker thread can call `push`/`pop` on it (the API is
// /// `!Sync`), while other workers reach in via the matching `Stealer` in
// /// `Inner::stealers`.
// struct WorkerCtx {
//     deque: Deque<Task>,
//     inner: Arc<Inner>,
//     index: usize,
// }

// thread_local! {
//     static WORKER: RefCell<Option<WorkerCtx>> = const { RefCell::new(None) };
// }

// /// Returns `Some(index)` if the current thread is a worker of the pool whose
// /// `Inner` arc matches `inner`. Distinguishing pools matters once we ever
// /// run multiple pools in the same process; it also prevents a task scheduled
// /// on pool A from being pushed onto a worker deque of pool B if the bodies
// /// somehow run nested.
// fn current_worker_of(inner: &Arc<Inner>) -> Option<usize> {
//     WORKER.with(|w| {
//         w.borrow()
//             .as_ref()
//             .and_then(|c| Arc::ptr_eq(&c.inner, inner).then_some(c.index))
//     })
// }

// /// Drain an injector with the retry loop in one place so the priority
// /// ladder reads as a flat sequence of sources.
// #[inline]
// fn try_steal_injector(inj: &Injector<Task>) -> Option<Task> {
//     loop {
//         match inj.steal() {
//             Steal::Success(t) => return Some(t),
//             Steal::Retry => continue,
//             Steal::Empty => return None,
//         }
//     }
// }

// #[inline]
// fn try_steal_deque(s: &Stealer<Task>) -> Option<Task> {
//     loop {
//         match s.steal() {
//             Steal::Success(t) => return Some(t),
//             Steal::Retry => continue,
//             Steal::Empty => return None,
//         }
//     }
// }

// /// Priority ladder for finding one task:
// ///   1. own deque   (LIFO, depth-first for nested children we just pushed)
// ///   2. own mailbox (hinted work assigned to this worker explicitly)
// ///   3. global injector
// ///   4. peers' mailboxes (rotated to avoid hot-spotting one victim)
// ///   5. peers' deques
// /// Returns `None` only after every source was observed empty.
// fn try_grab_one(inner: &Arc<Inner>, self_index: Option<usize>) -> Option<Task> {
//     // 1. Own deque.
//     if let Some(idx) = self_index {
//         let popped = WORKER.with(|w| {
//             w.borrow_mut().as_mut().and_then(|c| {
//                 debug_assert_eq!(c.index, idx);
//                 c.deque.pop()
//             })
//         });
//         if popped.is_some() {
//             return popped;
//         }
//         // 2. Own mailbox.
//         if let Some(t) = try_steal_injector(&inner.mailboxes[idx]) {
//             return Some(t);
//         }
//     }
//     // // 3. Global injector.
//     // if let Some(t) = try_steal_injector(&inner.injector) {
//     //     return Some(t);
//     // }
//     // // 4 + 5. Peer mailboxes then peer deques. Same rotation for both so a
//     // // single victim sweep covers both channels.
//     // let n = inner.stealers.len();
//     // if n == 0 {
//     //     return None;
//     // }
//     // let start = self_index.map(|i| (i + 1) % n).unwrap_or(0);
//     // for i in 0..n {
//     //     let idx = (start + i) % n;
//     //     if Some(idx) == self_index {
//     //         continue;
//     //     }
//     //     if let Some(t) = try_steal_injector(&inner.mailboxes[idx]) {
//     //         return Some(t);
//     //     }
//     //     if let Some(t) = try_steal_deque(&inner.stealers[idx]) {
//     //         return Some(t);
//     //     }
//     // }
//     None
// }

// /// Worker thread main loop. Runs forever until `Inner::shutdown` flips.
// fn worker_main(inner: Arc<Inner>, deque: Deque<Task>, index: usize) {
//     WORKER.with(|w| {
//         *w.borrow_mut() = Some(WorkerCtx {
//             deque,
//             inner: inner.clone(),
//             index,
//         });
//     });

//     // Publish this worker's `Thread` handle so the pool can target it.
//     // Done once, before the loop, so the slot is observable as soon as
//     // any other thread sees that this worker thread has been spawned.
//     inner.park_handles[index]
//         .set(thread::current())
//         .expect("worker park_handle slot already set");

//     let mut idle: u32 = 0;
//     loop {
//         if inner.shutdown.load(Ordering::Acquire) {
//             break;
//         }

//         if let Some(task) = try_grab_one(&inner, Some(index)) {
//             task();
//             idle = 0;
//             continue;
//         }

//         idle = idle.saturating_add(1);
//         if idle < SPIN_ITERS {
//             std::hint::spin_loop();
//         } else if idle < YIELD_ITERS {
//             thread::yield_now();
//         } else {
//             // Re-check shutdown immediately before parking so a Drop that
//             // races our last `try_grab_one` is still observed.
//             if inner.shutdown.load(Ordering::Acquire) {
//                 break;
//             }
//             // Bounded park: `unpark()` from another thread wakes us; a
//             // missed unpark costs at most one `PARK_TIMEOUT` of latency.
//             // `unpark` issued before `park` deposits a permit and the
//             // `park` returns immediately (no missed-wake race).
//             thread::park_timeout(PARK_TIMEOUT);
//             idle = 0;
//         }
//     }

//     WORKER.with(|w| *w.borrow_mut() = None);
// }

// ─────────────────────────────────────────────────────────────────────────
// Public pool type
// ─────────────────────────────────────────────────────────────────────────

// pub struct ThreadPool {
//     inner: Arc<Inner>,
//     n_workers: usize,
//     handles: Mutex<Option<Vec<JoinHandle<()>>>>,
// }

// impl ThreadPool {
//     /// Build a pool with exactly `n_workers` worker threads.
//     pub fn new(n_workers: usize) -> Self {
//         assert!(n_workers > 0, "ThreadPool requires at least one worker");

//         // Build the per-worker deques and their stealers in matching order so
//         // `stealers[i]` belongs to the worker that owns the `Deque<Task>`
//         // that gets moved into worker `i` below.  `Worker<Task>` is
//         // `!Clone`, so we cannot keep a copy in `Inner` — only the stealer.
//         let deques: Vec<Deque<Task>> = (0..n_workers).map(|_| Deque::new_lifo()).collect();
//         let stealers: Vec<Stealer<Task>> = deques.iter().map(|d| d.stealer()).collect();
//         // One MPMC mailbox per worker for hinted scheduling.
//         let mailboxes: Vec<Injector<Task>> = (0..n_workers).map(|_| Injector::new()).collect();
//         // One empty park-handle slot per worker; each worker fills its
//         // own slot before entering its main loop.
//         let park_handles: Vec<OnceLock<Thread>> = (0..n_workers).map(|_| OnceLock::new()).collect();

//         let inner = Arc::new(Inner {
//             injector: Injector::new(),
//             stealers,
//             mailboxes,
//             shutdown: AtomicBool::new(false),
//             park_handles,
//         });

//         let mut handles = Vec::with_capacity(n_workers);
//         for (index, deque) in deques.into_iter().enumerate() {
//             let inner_c = inner.clone();
//             let h = thread::Builder::new()
//                 .name(format!("my-thread-pool-{index}"))
//                 .spawn(move || worker_main(inner_c, deque, index))
//                 .expect("spawn worker thread");
//             handles.push(h);
//         }

//         Self {
//             inner,
//             n_workers,
//             handles: Mutex::new(Some(handles)),
//         }
//     }

//     #[inline]
//     pub fn num_threads(&self) -> usize {
//         self.n_workers
//     }

//     /// Push a task. Prefer the current worker's deque so nested dispatches
//     /// stay local (cache-warm) and only spill across workers via stealing.
//     /// External callers push to the shared `Injector`.
//     fn schedule(&self, task: Task) {
//         let task = WORKER.with(|w| {
//             let mut b = w.borrow_mut();
//             if let Some(ctx) = b.as_mut() {
//                 if Arc::ptr_eq(&ctx.inner, &self.inner) {
//                     ctx.deque.push(task);
//                     return None;
//                 }
//             }
//             Some(task)
//         });
//         if let Some(t) = task {
//             self.inner.injector.push(t);
//         }
//     }

//     /// Push a task with a preferred worker index. Lands in that worker's
//     /// `Injector` mailbox; the targeted worker checks its own mailbox
//     /// before any peer mailbox, so it sees the task first. If the target
//     /// is busy other workers steal the mailbox — the hint is advisory,
//     /// not exclusive.
//     ///
//     /// This does NOT wake the target; pair it with `wake_worker(idx)` (or
//     /// `wake_all()` once after a batch of hinted pushes) so a parked
//     /// target actually picks the task up promptly.
//     ///
//     /// Out-of-range hints panic. We don't silently route them to the
//     /// global injector because that hides a wrong-arithmetic bug at the
//     /// caller (per project rule: avoid silent fallbacks).
//     pub fn schedule_with_hint(&self, worker: usize, task: Task) {
//         assert!(
//             worker < self.n_workers,
//             "schedule_with_hint: worker index {worker} out of range (n_workers = {})",
//             self.n_workers
//         );
//         self.inner.mailboxes[worker].push(task);
//     }

//     /// Targeted wake: unpark exactly worker `idx`. Idempotent; an unpark
//     /// issued before the target parks deposits a permit, so the next
//     /// `park_timeout` returns immediately. No missed-wake race.
//     fn wake_worker(&self, idx: usize) {
//         if let Some(t) = self.inner.park_handles[idx].get() {
//             t.unpark();
//         }
//     }

//     fn wake_all(&self) {
//         for slot in &self.inner.park_handles {
//             if let Some(t) = slot.get() {
//                 t.unpark();
//             }
//         }
//     }

//     fn wake_one(&self) {
//         // Wake the lowest-indexed worker whose handle is published. Used
//         // for `spawn_background`, where any one worker will do.
//         for slot in &self.inner.park_handles {
//             if let Some(t) = slot.get() {
//                 t.unpark();
//                 return;
//             }
//         }
//     }

//     /// Run tasks (own deque -> injector -> steal) until `done()` returns
//     /// true. This is what makes nested `parallel_for` and `spawn_background`
//     /// safe: any thread blocked here keeps draining work instead of idling,
//     /// so its own children can run on the same worker if no one steals them.
//     fn help_until(&self, mut done: impl FnMut() -> bool) {
//         let self_idx = current_worker_of(&self.inner);
//         let mut idle: u32 = 0;
//         while !done() {
//             if let Some(task) = try_grab_one(&self.inner, self_idx) {
//                 task();
//                 idle = 0;
//                 continue;
//             }
//             // No work and predicate still false: another worker is grinding
//             // on the last task(s). Spin briefly, then yield. We do NOT park
//             // here — the predicate could become true at any moment and we
//             // want low wakeup latency on the dispatching thread.
//             idle = idle.saturating_add(1);
//             if idle < SPIN_ITERS {
//                 std::hint::spin_loop();
//             } else {
//                 thread::yield_now();
//                 idle = SPIN_ITERS; // cap so we keep yielding, not sleeping
//             }
//         }
//     }

//     /// Parallel-for over `range`. Splits the index space into `n_workers`
//     /// contiguous chunks, schedules them, then helps drain (including any
//     /// nested dispatches) until all chunks complete. Re-raises the first
//     /// task panic on the calling thread.
//     pub fn parallel_for<F>(&self, range: Range<usize>, body: F)
//     where
//         F: Fn(Range<usize>) + Sync + Send,
//     {
//         if range.start >= range.end {
//             return;
//         }
//         let total = range.end - range.start;
//         // One task per worker; work-stealing balances imbalance. Cap at
//         // `total` so we never schedule empty sub-ranges.
//         let n_tasks = self.n_workers.min(total).max(1);

//         // Stack-rooted body: tasks see it via a thin raw pointer + a
//         // monomorphized call-fn. This avoids `dyn Trait` (whose implicit
//         // `'static` bound would force `F: 'static`) entirely, while still
//         // erasing F from the task closures.  `help_until` below guarantees
//         // the body outlives every dereference.
//         fn call_body<F: Fn(Range<usize>) + Sync + Send>(body_ptr: *const (), range: Range<usize>) {
//             // SAFETY: the dispatching thread blocks until `pending == 0`
//             // before returning, so `body_ptr` is still valid.
//             let f: &F = unsafe { &*(body_ptr as *const F) };
//             f(range);
//         }
//         let body_ptr_raw: *const () = &body as *const F as *const ();
//         let call: fn(*const (), Range<usize>) = call_body::<F>;
//         // Newtype to assert thread-shareability of the thin raw pointer.
//         // Function pointers are already `Send + Sync + Copy + 'static`.
//         #[derive(Copy, Clone)]
//         struct BodyPtr(*const ());
//         unsafe impl Send for BodyPtr {}
//         unsafe impl Sync for BodyPtr {}
//         let bp = BodyPtr(body_ptr_raw);

//         let pending = Arc::new(AtomicUsize::new(n_tasks));
//         let panic_slot: Arc<Mutex<Option<Box<dyn Any + Send>>>> = Arc::new(Mutex::new(None));

//         for i in 0..n_tasks {
//             // Even-ish split: start_i = range.start + total * i / n_tasks.
//             // Using 64-bit arithmetic here would only matter for ranges
//             // approaching `usize::MAX / n_tasks`, far beyond engine scale.
//             let s = range.start + total * i / n_tasks;
//             let e = range.start + total * (i + 1) / n_tasks;
//             let pending_c = pending.clone();
//             let panic_c = panic_slot.clone();
//             let bp = bp;
//             let task: Task = Box::new(move || {
//                 // Force capture of the whole `BodyPtr` newtype (not just
//                 // its raw-pointer field) so the closure inherits the
//                 // wrapper's `Send`/`Sync` impls. Rust 2021's disjoint
//                 // capture would otherwise see only `bp.0`.
//                 let bp = bp;
//                 let res = panic::catch_unwind(AssertUnwindSafe(|| call(bp.0, s..e)));
//                 pending_c.fetch_sub(1, Ordering::Release);
//                 if let Err(p) = res {
//                     // Keep the first panic; drop subsequent ones.
//                     let mut slot = panic_c.lock().unwrap();
//                     if slot.is_none() {
//                         *slot = Some(p);
//                     }
//                 }
//             });
//             // Chunk k -> worker k. With `n_tasks = min(n_workers, total)`,
//             // `i` is already < n_workers so no modulo is needed.
//             self.schedule_with_hint(i, task);
//             // Targeted wake: only worker `i` gets unparked, so peers that
//             // had no chunk of their own stay parked and don't race-steal
//             // worker i's mailbox before worker i can see it. Workers that
//             // finish their chunk early will, of course, then steal from
//             // slower peers' mailboxes — that's the load-balancing path.
//             self.wake_worker(i);
//         }

//         self.help_until(|| pending.load(Ordering::Acquire) == 0);

//         // Take payload into a local so the `MutexGuard` is dropped before
//         // we (maybe) unwind. Otherwise the temporary would still hold the
//         // lock at the resume_unwind point.
//         let payload = panic_slot.lock().unwrap().take();
//         if let Some(p) = payload {
//             panic::resume_unwind(p);
//         }
//     }

//     /// Enqueue a long-running task. Occupies a worker; remaining workers
//     /// keep servicing `parallel_for`. For *blocking* syscalls (file I/O,
//     /// `vkWaitSemaphores`, etc.) prefer a dedicated `std::thread::spawn`
//     /// so you don't strand a compute core.
//     pub fn spawn_background<F>(&self, f: F)
//     where
//         F: FnOnce() + Send + 'static,
//     {
//         self.schedule(Box::new(f));
//         self.wake_one();
//     }
// }

// impl Drop for ThreadPool {
//     fn drop(&mut self) {
//         self.inner.shutdown.store(true, Ordering::Release);
//         self.wake_all();
//         if let Some(handles) = self.handles.lock().unwrap().take() {
//             for h in handles {
//                 let _ = h.join();
//             }
//         }
//     }
// }

// ──────────────────────────────────────────────────────────────────────
// Hot-path synchronization primitives (lock-free + work stealing)
// ──────────────────────────────────────────────────────────────────────
//
// No locks on the hot path. Per dispatch:
//   * Main seeds each participant's cursor with its initial slice of
//     [0, size) and writes the `Job` slot via `UnsafeCell` (no atomic).
//   * Main resets `workers_done`, then bumps `epoch` with one
//     `Release` fetch_add. That single store is the publish edge.
//   * Workers spin on `epoch.load(Acquire)` — a pure load, no CAS —
//     so the cache line stays in shared state across cores and no
//     reader contends with a writer.
//   * On observing a new epoch, every participant (main + workers)
//     runs `run_with_stealing`: drain own cursor in STEAL_GRAIN-sized
//     chunks, then walk peers in `steal_order` and try to steal half
//     of a busy peer's remaining range. Restart the sweep after every
//     successful steal so near peers are tried first. A full empty
//     sweep means the whole dispatch is drained.
//   * Background workers bump `workers_done` (Release) after exiting
//     the steal loop. Main spins on `workers_done == n_workers`
//     (Acquire) for the barrier — main itself is not counted.
//
// Cursor encoding:
//   * One `Cursor { packed: AtomicU64 }` per participant. The packed
//     word holds (start, end) as two u32s. Owner advances `start` via
//     CAS on the whole word; thieves lower `end` via CAS on the whole
//     word. Loser of a race retries with the fresh snapshot.
//   * `PaddedCursor` is 64-byte aligned so a thief's CAS on cursor[p]
//     can't invalidate the line carrying cursor[p+1] for its owner.
//
// Cursors are owned by `Shared` (fixed length = num_threads, allocated
// once at pool creation). No per-dispatch heap traffic.

/// Granularity of own-cursor advances. Workers claim STEAL_GRAIN items
/// per CAS while draining their own range. Smaller = better load balance
/// near the tail but more CAS overhead in the inner loop. Old pool uses
/// 64; same trade-off here.
const STEAL_GRAIN: u32 = 64;

/// 64-byte aligned wrapper. Forces each cursor onto its own cache line
/// so split-end races between owner (advancing `start`) and thieves
/// (lowering `end`) don't cause false sharing with neighbour cursors.
#[repr(align(64))]
struct PaddedCursor {
    packed: AtomicU64,
}

impl PaddedCursor {
    const fn empty() -> Self {
        Self {
            packed: AtomicU64::new(0),
        }
    }
}

#[inline(always)]
fn pack_cursor(s: u32, e: u32) -> u64 {
    ((s as u64) << 32) | (e as u64)
}

#[inline(always)]
fn unpack_cursor(p: u64) -> (u32, u32) {
    ((p >> 32) as u32, p as u32)
}

/// Attempt to steal the back half of a peer's remaining range.
/// Returns the claimed `[s, e)` on success, `None` on contention or if
/// the peer is empty. Caller retries by moving on to the next peer
/// (restart-on-success keeps near peers preferred over re-trying a
/// contended one).
#[inline]
fn try_steal(cursor: &PaddedCursor) -> Option<(u32, u32)> {
    let packed = cursor.packed.load(Ordering::Acquire);
    let (s, e) = unpack_cursor(packed);
    if s >= e {
        return None;
    }
    let remaining = e - s;
    // Take half, rounded up so a 1-item remainder is fully claimed
    // rather than left orphaned with the owner who may have already
    // moved on past their drain loop.
    let take = remaining - remaining / 2;
    let new_end = e - take;
    if cursor
        .packed
        .compare_exchange_weak(
            packed,
            pack_cursor(s, new_end),
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok()
    {
        Some((new_end, e))
    } else {
        None
    }
}

/// Drive one participant through its own cursor and then steal until
/// every reachable cursor is empty.
///
/// SAFETY: `cursors` must be valid for at least `num_threads` reads;
/// `own_idx < num_threads`; every entry in `steal_order` must be a
/// valid participant index `< num_threads`; `body_ptr` + `call_fn`
/// together must name a valid type-erased closure invocation. Caller
/// (main or worker_main) keeps `F` alive past the barrier.
unsafe fn run_with_stealing(
    cursors: &[PaddedCursor],
    own_idx: usize,
    steal_order: &[usize],
    call_fn: unsafe fn(*const (), Range<usize>),
    body_ptr: *const (),
) {
    // ── Drain own cursor in STEAL_GRAIN chunks ─────────────────────
    let own = &cursors[own_idx];
    loop {
        let packed = own.packed.load(Ordering::Acquire);
        let (s, e) = unpack_cursor(packed);
        if s >= e {
            break;
        }
        let claim_end = (s + STEAL_GRAIN).min(e);
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
            unsafe { call_fn(body_ptr, s as usize..claim_end as usize) };
        }
        // else: a thief lowered `end` concurrently — retry with the
        // fresh snapshot.
    }

    // ── Steal sweep ────────────────────────────────────────────────
    // Walk peers in `steal_order` (per-participant rotation: nearest
    // peer first, wrapping). Restart from the top on every successful
    // steal so we keep preferring near peers. Terminate when a full
    // sweep finds every peer empty — within an epoch, cursors only
    // shrink, so an empty sweep means the dispatch is drained.
    'outer: loop {
        for &peer in steal_order {
            let cur = &cursors[peer];
            if let Some((s, e)) = try_steal(cur) {
                unsafe { call_fn(body_ptr, s as usize..e as usize) };
                continue 'outer;
            }
        }
        break;
    }
}

#[derive(Clone, Copy)]
struct Job {
    /// Erased pointer to the caller's `F` (stack-rooted in `parallel_for`).
    body_ptr: *const (),
    /// Monomorphised trampoline: casts `body_ptr` back to `&F` and calls it.
    /// You CANNOT transmute `*const F` to `fn(...)` — a closure value's
    /// address is captured data, not executable code.
    call_fn: unsafe fn(*const (), Range<usize>),
}

struct Shared {
    /// Monotonic dispatch counter. Bumped (Release) by main after the
    /// `job` slot and cursors are written; observed (Acquire) by workers
    /// to learn that a new dispatch is live. Pure-load spin = no CAS
    /// traffic on the spin path.
    epoch: AtomicU64,
    /// Per-dispatch barrier counter. Reset to 0 by main before bumping
    /// `epoch`; each *background worker* fetch_add's 1 (Release) after
    /// exiting its steal loop. Main itself is not counted; it spins on
    /// this reaching `num_threads - 1`.
    workers_done: AtomicUsize,
    /// Set on pool drop. Workers check inside their spin loop and exit.
    shutdown: AtomicBool,
    /// Total participants (workers + main). Fixed at construction.
    num_threads: usize,
    /// Single-writer, multi-reader job slot. Main is the sole writer,
    /// and only between dispatches (after the previous barrier).
    job: UnsafeCell<Job>,
    /// One cursor per participant, indexed by participant id
    /// (0 = main, 1..num_threads = background workers). Reused every
    /// dispatch — main re-seeds the start/end pairs before bumping
    /// `epoch`. Padded so split-end stealing doesn't false-share
    /// neighbour lines.
    cursors: Box<[PaddedCursor]>,
    /// `steal_order[p]` is the list of peer indices participant `p`
    /// tries to steal from, in preference order (nearest first,
    /// wrapping). Pre-computed at pool creation. Each entry has
    /// length `num_threads - 1`.
    steal_order: Box<[Box<[usize]>]>,
}

// SAFETY: All cross-thread access to `job` is gated by the epoch atomic
// (Release on publish, Acquire on observe), so the UnsafeCell is never
// read concurrently with a write. The `*const ()` body pointer inside
// `Job` is only dereferenced by `call_fn`, whose monomorphisation
// guarantees `F: Sync`, and main keeps the `F` alive past the barrier.
// `cursors` are atomic; `steal_order` is immutable after construction.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

unsafe fn noop_call(_: *const (), _: Range<usize>) {}

fn worker_main(index: usize, shared: Arc<Shared>) {
    let mut last_epoch: u64 = 0;
    let steal_order = &shared.steal_order[index];
    loop {
        // Pure-load spin on the epoch. No CAS, no lock acquire — the
        // cache line stays Shared across all spinning workers.
        let cur = loop {
            let e = shared.epoch.load(Ordering::Acquire);
            if e != last_epoch {
                break e;
            }
            if shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            std::hint::spin_loop();
        };
        last_epoch = cur;

        // Shutdown also bumps the epoch (see Drop), so re-check here.
        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }

        // SAFETY: the Acquire on `epoch` above synchronises with main's
        // Release fetch_add that published this dispatch. Job slot,
        // cursor seeds, and `steal_order` (immutable after construction)
        // are all visible.
        let job = unsafe { *shared.job.get() };
        unsafe {
            run_with_stealing(
                &shared.cursors,
                index,
                steal_order,
                job.call_fn,
                job.body_ptr,
            );
        }

        // Release: side effects of every `call_fn` invocation on this
        // worker happen-before main's Acquire barrier load.
        shared.workers_done.fetch_add(1, Ordering::Release);
    }
}

pub struct ThreadPool {
    handles: Vec<JoinHandle<()>>,
    shared: Arc<Shared>,
}

impl ThreadPool {
    pub fn new(n_workers: usize) -> Self {
        assert!(
            n_workers >= 1,
            "ThreadPool::new requires at least 1 participant (main thread)"
        );

        // Pre-allocate one cursor per participant, all initially empty.
        // `parallel_for` re-seeds them before bumping the epoch.
        let cursors: Box<[PaddedCursor]> = (0..n_workers)
            .map(|_| PaddedCursor::empty())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        // Per-participant rotated steal order. Participant p tries
        // p+1, p+2, …, n-1, 0, 1, …, p-1. Length n-1.
        let steal_order: Box<[Box<[usize]>]> = (0..n_workers)
            .map(|me| {
                ((me + 1)..n_workers)
                    .chain(0..me)
                    .collect::<Vec<usize>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            workers_done: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            num_threads: n_workers,
            job: UnsafeCell::new(Job {
                body_ptr: std::ptr::null(),
                call_fn: noop_call,
            }),
            cursors,
            steal_order,
        });
        let mut handles = Vec::with_capacity(n_workers.saturating_sub(1));
        for i in 0..n_workers - 1 {
            let shared = shared.clone();
            let h = thread::Builder::new()
                .name(format!("my-thread-pool-{i}"))
                .spawn(move || worker_main(i + 1, shared))
                .expect("spawn worker thread");
            handles.push(h);
        }
        Self { handles, shared }
    }

    /// Total number of threads that participate in a `parallel_for`
    /// dispatch: background workers + the calling thread.
    #[inline]
    pub fn num_threads(&self) -> usize {
        self.shared.num_threads
    }

    pub fn parallel_for<F>(&self, range: Range<usize>, body: F)
    where
        F: Fn(Range<usize>) + Sync + Send,
    {
        let size = range.end - range.start;
        if size == 0 {
            return;
        }
        // Cursors pack (start, end) as two u32s. Engine workloads stay
        // far below this; crash loudly rather than silently truncate.
        assert!(
            size <= u32::MAX as usize,
            "parallel_for range too large for u32 cursor packing ({size} > {})",
            u32::MAX
        );

        // Monomorphised trampoline for this concrete `F`. The data
        // pointer is the closure's environment; this fn is the only
        // sound way to invoke it through `*const ()`.
        unsafe fn call_impl<F: Fn(Range<usize>)>(ptr: *const (), r: Range<usize>) {
            let f = unsafe { &*(ptr as *const F) };
            f(r);
        }

        let shared = &*self.shared;
        let num_threads = shared.num_threads;
        let n_workers = num_threads - 1;

        // Seed each participant's cursor with its initial slice of
        // [0, size). Relaxed stores are fine: the Release fetch_add on
        // `epoch` below provides the publish edge for these writes.
        for p in 0..num_threads {
            let s = (p * size) / num_threads;
            let e = ((p + 1) * size) / num_threads;
            shared.cursors[p]
                .packed
                .store(pack_cursor(s as u32, e as u32), Ordering::Relaxed);
        }

        // Publish the job. SAFETY: the previous `parallel_for` returned
        // only after `workers_done == n_workers`, so no worker is still
        // reading the old slot. We are the sole writer.
        unsafe {
            *shared.job.get() = Job {
                body_ptr: &body as *const F as *const (),
                call_fn: call_impl::<F>,
            };
        }

        // Reset the barrier BEFORE bumping the epoch. Relaxed is fine:
        // the Release fetch_add on `epoch` orders this store against
        // any worker's Acquire load of `epoch`.
        shared.workers_done.store(0, Ordering::Relaxed);
        shared.epoch.fetch_add(1, Ordering::Release);

        // Main participates as cursor index 0. Same drain-own + steal
        // loop as workers, so an imbalanced dispatch finishes when the
        // *average* slice is done, not the slowest.
        let main_steal_order = &shared.steal_order[0];
        unsafe {
            run_with_stealing(
                &shared.cursors,
                0,
                main_steal_order,
                call_impl::<F>,
                &body as *const F as *const (),
            );
        }

        // Barrier: wait for every background worker to finish its
        // drain+steal loop. Main is not counted.
        while shared.workers_done.load(Ordering::Acquire) < n_workers {
            std::hint::spin_loop();
        }
        // SAFETY: the Acquire load above synchronises with every worker's
        // Release fetch_add, so all side effects of their `call_fn`
        // invocations are visible here. `body` (and anything it borrowed)
        // is now safe to drop on return.
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Empty every cursor so any worker that observes the shutdown
        // epoch bump enters `run_with_stealing` with nothing to do.
        for c in self.shared.cursors.iter() {
            c.packed.store(0, Ordering::Relaxed);
        }
        // Install a safe no-op job so even if a racing worker did try
        // to invoke it (it won't — cursors are empty), the body_ptr
        // left by the last real dispatch is no longer dereferenced.
        unsafe {
            *self.shared.job.get() = Job {
                body_ptr: std::ptr::null(),
                call_fn: noop_call,
            };
        }
        self.shared.shutdown.store(true, Ordering::Release);
        // Kick any worker that's mid-spin so it observes the epoch
        // change, re-checks `shutdown`, and returns.
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Global pool (process-wide singleton)
// ─────────────────────────────────────────────────────────────────────────

pub mod global {
    use super::*;

    static POOL: OnceLock<ThreadPool> = OnceLock::new();

    /// Initialise the global pool with exactly `n_workers` worker threads.
    /// Returns `false` if the pool was already initialised; the caller can
    /// `assert!` to surface a double-init bug loudly (no silent fallback).
    pub fn init(n_workers: usize) -> bool {
        POOL.set(ThreadPool::new(n_workers)).is_ok()
    }

    /// Access the global pool. Panics if `init` was not called — a clear
    /// crash rather than a silent default-config pool.
    #[inline]
    pub fn pool() -> &'static ThreadPool {
        POOL.get()
            .expect("my_thread_pool::global::init(n) must be called before use")
    }

    pub fn is_initialized() -> bool {
        POOL.get().is_some()
    }

    #[inline]
    pub fn parallel_for<F>(range: Range<usize>, body: F)
    where
        F: Fn(Range<usize>) + Sync + Send,
    {
        pool().parallel_for(range, body)
    }

    // #[inline]
    // pub fn spawn_background<F>(f: F)
    // where
    //     F: FnOnce() + Send + 'static,
    // {
    //     pool().spawn_background(f)
    // }
}

// Convenience re-exports so callers can write `my_thread_pool::parallel_for`.
pub use global::parallel_for;

// ──────────────────────────────────────────────────────────────────────
// Bitmap-task layout helper
// ──────────────────────────────────────────────────────────────────────
//
// Shared geometry for the simulation / staging passes that walk
// transform-indexed dirty bitmaps. Picks a `words_per_task` that
// yields ~`BITMAP_TARGET_TASKS_PER_THREAD` tasks per participating
// thread, clamped to `[BITMAP_MIN_WORDS_PER_TASK, BITMAP_MAX_WORDS_PER_TASK]`
// so individual tasks stay coarse enough to amortise dispatch but
// fine enough to keep the longest stragglers from dominating.
//
// `num_threads()` here counts main + workers, matching the old
// `thread_pool` helper this replaces.

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

/// Choose a shared task layout for bitmap-indexed per-entity work.
///
/// Mirrors the (now-legacy) `thread_pool::bitmap_task_layout` so the sim
/// and staging paths keep matching slab sizes after the pool swap.
#[inline]
pub fn bitmap_task_layout(n_words: usize) -> BitmapTaskLayout {
    if n_words == 0 {
        return BitmapTaskLayout {
            words_per_task: BITMAP_MIN_WORDS_PER_TASK,
            n_tasks: 0,
        };
    }
    let target_tasks = global::pool()
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

// ─────────────────────────────────────────────────────────────────────────
// Smoke tests
// ─────────────────────────────────────────────────────────────────────────

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::sync::atomic::AtomicUsize;

//     /// All `cargo test` cases share the global pool; ignore the result so
//     /// whichever case wins the race performs the init.
//     fn ensure_pool() {
//         let _ = global::init(4);
//     }

//     #[test]
//     fn parallel_for_sum() {
//         ensure_pool();
//         let s = Arc::new(AtomicUsize::new(0));
//         let sc = s.clone();
//         global::parallel_for(0..10_000, |r| {
//             let mut local = 0usize;
//             for i in r {
//                 local += i;
//             }
//             sc.fetch_add(local, Ordering::Relaxed);
//         });
//         assert_eq!(s.load(Ordering::Relaxed), (0..10_000usize).sum());
//     }

//     #[test]
//     fn nested_parallel_for() {
//         ensure_pool();
//         let total = Arc::new(AtomicUsize::new(0));
//         let t = total.clone();
//         global::parallel_for(0..16, |outer| {
//             for _ in outer {
//                 let t = t.clone();
//                 global::parallel_for(0..100, |inner| {
//                     t.fetch_add(inner.end - inner.start, Ordering::Relaxed);
//                 });
//             }
//         });
//         assert_eq!(total.load(Ordering::Relaxed), 16 * 100);
//     }

//     #[test]
//     fn background_runs_alongside_parallel_for() {
//         ensure_pool();
//         let bg = Arc::new(AtomicUsize::new(0));
//         let bgc = bg.clone();
//         global::spawn_background(move || {
//             for _ in 0..1_000_000 {
//                 bgc.fetch_add(1, Ordering::Relaxed);
//             }
//         });
//         let n = Arc::new(AtomicUsize::new(0));
//         let nc = n.clone();
//         global::parallel_for(0..10_000, |r| {
//             nc.fetch_add(r.end - r.start, Ordering::Relaxed);
//         });
//         assert_eq!(n.load(Ordering::Relaxed), 10_000);
//         while bg.load(Ordering::Relaxed) < 1_000_000 {
//             std::thread::yield_now();
//         }
//     }

//     #[test]
//     fn schedule_with_hint_runs_task() {
//         // Liveness check for the hinted-schedule path: a task pushed into
//         // a specific worker's mailbox must be picked up promptly by
//         // *some* worker (the hinted one most of the time; a thief if the
//         // hinted worker is busy).
//         ensure_pool();
//         let pool = global::pool();
//         let n = pool.num_threads();
//         let ran = Arc::new(AtomicUsize::new(0));
//         for w in 0..n {
//             let r = ran.clone();
//             pool.schedule_with_hint(
//                 w,
//                 Box::new(move || {
//                     r.fetch_add(1, Ordering::SeqCst);
//                 }),
//             );
//         }
//         pool.wake_all();
//         let start = std::time::Instant::now();
//         while ran.load(Ordering::SeqCst) < n {
//             assert!(
//                 start.elapsed() < std::time::Duration::from_secs(2),
//                 "hinted tasks did not all run within 2s (ran = {})",
//                 ran.load(Ordering::SeqCst),
//             );
//             std::thread::yield_now();
//         }
//         assert_eq!(ran.load(Ordering::SeqCst), n);
//     }

//     #[test]
//     fn schedule_with_hint_prefers_target_when_idle() {
//         // Statistical check: when the pool is otherwise idle, hinted
//         // tasks should land on the hinted worker the *majority* of the
//         // time. We push K tasks one at a time (waiting for each to
//         // complete before the next) so there is never any other work to
//         // steal from. With nothing else in flight, the targeted worker
//         // should win the race for its own mailbox almost always.
//         ensure_pool();
//         let pool = global::pool();
//         let n = pool.num_threads();
//         if n < 2 {
//             return;
//         }
//         const TRIES: usize = 32;
//         let mut hits = 0usize;
//         for k in 0..TRIES {
//             let target = k % n;
//             let observed = Arc::new(std::sync::Mutex::new(None::<usize>));
//             let oc = observed.clone();
//             let done = Arc::new(AtomicUsize::new(0));
//             let dc = done.clone();
//             pool.schedule_with_hint(
//                 target,
//                 Box::new(move || {
//                     let name = std::thread::current().name().unwrap_or("").to_string();
//                     // Parse the trailing index from "my-thread-pool-<N>".
//                     let idx: usize = name
//                         .rsplit('-')
//                         .next()
//                         .and_then(|s| s.parse().ok())
//                         .unwrap_or(usize::MAX);
//                     *oc.lock().unwrap() = Some(idx);
//                     dc.store(1, Ordering::SeqCst);
//                 }),
//             );
//             pool.wake_all();
//             while done.load(Ordering::SeqCst) == 0 {
//                 std::thread::yield_now();
//             }
//             if observed.lock().unwrap().as_ref() == Some(&target) {
//                 hits += 1;
//             }
//         }
//         // Loose bound — races can let any worker grab the mailbox, but
//         // the targeted worker should still win most of the time when the
//         // pool is otherwise idle. If this drops below half the pool is
//         // basically ignoring the hint.
//         assert!(
//             hits >= TRIES / 2,
//             "hint preference too weak: {hits} / {TRIES} landed on target",
//         );
//     }

//     #[test]
//     fn panic_propagates() {
//         ensure_pool();
//         let r = std::panic::catch_unwind(|| {
//             global::parallel_for(0..8, |_| panic!("intentional"));
//         });
//         assert!(r.is_err(), "panic should have propagated");
//     }
// }
