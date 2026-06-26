//! A NUMA-aware, work-stealing fork/join thread pool.
//!
//! Design (as discussed):
//!   * One sub-pool per NUMA node; workers pinned to cores, grouped by node.
//!   * `parallel_for` splits the index space into ONE contiguous block per worker
//!     ("affinity chunks"). Block k -> worker k; nodes own contiguous super-blocks,
//!     so autoNUMA can place pages once and keep them put.
//!   * Each chunk is a single packed AtomicU64 (start:hi32 | end:lo32). The owner
//!     drains the front (`take_front`, CAS); idle thieves bite the rear (`steal_back`,
//!     CAS). One word => no torn `start<=end` invariant, no double counting.
//!   * `find_work` is the whole scheduler — a priority ladder:
//!       own deque -> own chunk -> nearest same-node chunk -> same-node steal
//!       -> cross-node steal -> cross-node chunk help.
//!   * "Wait" never barriers and never just parks: `help_until` loops by calling
//!     `find_work`, so `join`, `scope`, and nested `parallel_for` compose without
//!     deadlock and without spare threads.
//!   * Idle workers spin (bounded) then park; pushers wake by affinity.
//!   * A background task is just a job that occupies a worker; it never touches a
//!     parallel-for counter, so the next `parallel_for` fans out across the
//!     *remaining* workers. CPU-bound long jobs go here; syscall-BLOCKING work
//!     should go on its own OS thread (it would strand a compute core otherwise).
//!
//! Honesty: a faithful, compiling, runnable *reference* skeleton, not a hardened
//! production pool. Orderings are conservative; parking uses std park/unpark (a
//! futex eventcount scales better); topology is Linux-only with a single-node
//! fallback; per-node active parallel tasks live in an Arc-owned, lock-gated slot
//! (so a straggling helper can never deref a freed task). Spots
//! needing hardening are marked `HARDEN:`.
//!
//! Placement: this file is a self-contained module with no `crate::`-absolute
//! references, so it drops in at any path. Placed at `src/util/numa_pool.rs`
//! (i.e. `mod numa_pool;` under `util`), everything is reached as
//! `crate::util::numa_pool::{parallel_for, join, scope, spawn_background, run}`
//! and the global pool as `crate::util::numa_pool::global` (e.g.
//! `crate::util::numa_pool::global::pool()` / `::init(cfg)`).

use crossbeam_deque::{Injector, Steal, Stealer, Worker as Deque};
use crossbeam_utils::Backoff;
use portable_atomic::AtomicU128;
use std::cell::{Cell, RefCell, UnsafeCell};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Arc, OnceLock};
use std::thread::{self, Thread};

/// Idle ladder thresholds (iterations of the worker loop with no work found):
///   < HOT_SPIN_ITERS         : pure spin (stay hot, sub-µs wake)
///   < PARK_AFTER_ITERS       : sleep(1ns) band (release core, stay schedulable)
///   >= PARK_AFTER_ITERS      : park on the futex until unparked
/// Tuned to keep workers warm across a per-frame inter-dispatch gap so the
/// publisher's wake is usually a no-op rather than a real futex wake. Lower
/// HOT_SPIN_ITERS to save power if dispatches are sparse.
const HOT_SPIN_ITERS: u32 = 4_000;
const PARK_AFTER_ITERS: u32 = 100_000;

// ───────────────────────── Packed splittable chunk ─────────────────────────

/// 128-byte aligned so neighboring workers' cursors never share a cache line.
/// Without this, 8 `AtomicU64` cursors pack into one 64-byte line and every
/// worker's claim CAS ping-pongs its neighbors' lines across both sockets — the
/// dominant cost of the claim/steal path. 128 (not 64) defeats x86 adjacent-line
/// prefetch pairing, matching `crossbeam_utils::CachePadded`.
/// Result of attempting to claim work from a gen-tagged chunk.
enum Claim {
    Got(Range<usize>), // claimed this range
    Empty,             // chunk fully drained (for THIS generation)
    Stale,             // generation mismatch: the slot was recycled — abort, do
                       // NOT touch this slot's chunks (they belong to a new task)
}

/// A work cursor packed into a single 128-bit atomic: `(gen:u64, start:u32,
/// end:u32)`. Every claim is a `cmpxchg16b` that validates `gen` in the SAME
/// atomic op, so a helper can never claim out of a slot that was recycled under
/// it — the CAS just fails and we report `Stale`. This is what removes the need
/// for an engaged-counter / hazard-pointer handshake on the discovery path: the
/// generation rides along with the range. 128-byte aligned to keep adjacent
/// chunks (hammered by neighbouring workers) on separate cache lines.
#[repr(align(128))]
struct Chunk(AtomicU128);

#[inline]
fn unpack(w: u128) -> (u64, u32, u32) {
    ((w >> 64) as u64, (w >> 32) as u32, w as u32)
}
#[inline]
fn pack(gen: u64, s: u32, e: u32) -> u128 {
    ((gen as u128) << 64) | ((s as u128) << 32) | e as u128
}

impl Chunk {
    fn new_free() -> Self {
        Chunk(AtomicU128::new(0))
    } // gen 0 = never-published

    /// Publish a fresh range under generation `gen` (called by the owner during
    /// dispatch setup; no concurrent claimers yet because the slot bit isn't set
    /// and `slot.gen` isn't bumped until after all chunks are written).
    #[inline]
    fn reset(&self, gen: u64, start: u32, end: u32) {
        self.0.store(pack(gen, start, end), Relaxed);
    }

    /// Owner: claim up to `grain` from the front, validating `gen`.
    fn take_front(&self, gen: u64, grain: u32) -> Claim {
        let mut w = self.0.load(Acquire);
        loop {
            let (g, s, e) = unpack(w);
            if g != gen {
                return Claim::Stale;
            }
            if s >= e {
                return Claim::Empty;
            }
            let n = e.min(s.saturating_add(grain));
            match self
                .0
                .compare_exchange_weak(w, pack(gen, n, e), AcqRel, Acquire)
            {
                Ok(_) => return Claim::Got(s as usize..n as usize),
                Err(cur) => w = cur,
            }
        }
    }

    /// Claim the entire remaining range in one CAS (no-stealing path).
    fn take_all(&self, gen: u64) -> Claim {
        let mut w = self.0.load(Acquire);
        loop {
            let (g, s, e) = unpack(w);
            if g != gen {
                return Claim::Stale;
            }
            if s >= e {
                return Claim::Empty;
            }
            match self
                .0
                .compare_exchange_weak(w, pack(gen, e, e), AcqRel, Acquire)
            {
                Ok(_) => return Claim::Got(s as usize..e as usize),
                Err(cur) => w = cur,
            }
        }
    }

    /// Thief: claim the back half, validating `gen`. Floor midpoint so a
    /// 1-element chunk is taken whole (no spurious empty CAS).
    fn steal_back(&self, gen: u64) -> Claim {
        let mut w = self.0.load(Acquire);
        loop {
            let (g, s, e) = unpack(w);
            if g != gen {
                return Claim::Stale;
            }
            if s >= e {
                return Claim::Empty;
            }
            let mid = s + (e - s) / 2;
            match self
                .0
                .compare_exchange_weak(w, pack(gen, s, mid), AcqRel, Acquire)
            {
                Ok(_) => return Claim::Got(mid as usize..e as usize),
                Err(cur) => w = cur,
            }
        }
    }
}

// ───────────────────────────── Job references ─────────────────────────────

struct JobRef {
    ptr: *const (),
    exec: unsafe fn(*const ()),
}
unsafe impl Send for JobRef {}

/// Stack job for `join` — no heap alloc on the hot fork path. Sound because
/// `join` does not return until `done`.
struct StackJob<F, R> {
    f: UnsafeCell<Option<F>>,
    r: UnsafeCell<Option<R>>,
    done: AtomicBool,
    waiter: Option<Thread>,
}
impl<F: FnOnce() -> R, R> StackJob<F, R> {
    fn new(f: F, waiter: Option<Thread>) -> Self {
        StackJob {
            f: UnsafeCell::new(Some(f)),
            r: UnsafeCell::new(None),
            done: AtomicBool::new(false),
            waiter,
        }
    }
    fn job_ref(&self) -> JobRef {
        JobRef {
            ptr: self as *const _ as *const (),
            exec: Self::exec,
        }
    }
    unsafe fn exec(p: *const ()) {
        let me = &*(p as *const Self);
        let f = (*me.f.get()).take().expect("job run twice");
        let out = f();
        *me.r.get() = Some(out);
        // Read everything we need from *me BEFORE signaling completion: the moment
        // a waiter observes `done`, it may free this (stack-allocated) StackJob.
        // The store must be the LAST touch of *me by this thread.
        let waiter = me.waiter.clone();
        me.done.store(true, Release);
        if let Some(t) = waiter {
            t.unpark();
        }
    }
    unsafe fn run_inline(&self) -> R {
        let f = (*self.f.get()).take().expect("job run twice");
        let out = f();
        self.done.store(true, Release);
        out
    }
    unsafe fn take_result(&self) -> R {
        (*self.r.get()).take().expect("missing result")
    }
}

/// Heap job for `scope` spawns and background work.
struct HeapJob<F> {
    f: UnsafeCell<Option<F>>,
}
impl<F: FnOnce()> HeapJob<F> {
    fn boxed(f: F) -> Box<Self> {
        Box::new(HeapJob {
            f: UnsafeCell::new(Some(f)),
        })
    }
    fn into_ref(self: Box<Self>) -> JobRef {
        JobRef {
            ptr: Box::into_raw(self) as *const (),
            exec: Self::exec,
        }
    }
    unsafe fn exec(p: *const ()) {
        let b = Box::from_raw(p as *mut Self);
        let f = (*b.f.get()).take().expect("job run twice");
        f();
    }
}

// ───────────────────────────── Parallel task ──────────────────────────────

/// 128-byte aligned wrapper to keep a hot atomic off neighboring lines.
#[repr(align(128))]
struct CachePadded<T>(T);

/// Max parallel-for nesting depth per participant. Each (participant, depth)
/// pair owns a permanent slot in the bank, so a participant can have at most
/// MAX_NEST live dispatches stacked. Deeper nesting falls back to serial
/// execution (see parallel_for_grain) — never an OOB.
const MAX_NEST: usize = 8;

/// A dispatch's shared state, living in a TYPE-STABLE bank that is allocated
/// once and never freed. Publishers reuse their own slot rather than allocating
/// per dispatch. `gen` (even = free, odd = active) tags the contents: a helper
/// reads `gen`, then claims chunks with that gen baked into the 128-bit CAS, so
/// it can never act on a slot that was recycled under it. Because the memory is
/// never freed, a helper dereferencing a slot can never use-after-free — at
/// worst it reads a recycled slot and its gen-tagged claims fail (`Stale`).
struct TaskSlot {
    gen: AtomicU64,
    chunks: Box<[Chunk]>,                // len == num_workers
    processed: CachePadded<AtomicUsize>, // elements processed (per-session adds)
    total: AtomicUsize,
    done: AtomicBool,
    body: AtomicPtr<()>, // &closure on the owner's stack
    call: AtomicUsize,   // call_body::<F> as a fn ptr
    grain: AtomicU64,
    /// Per-dispatch stealing mode. `true` (default): owners take their chunk in
    /// grains and idle participants steal peers — load-balanced and nestable.
    /// `false`: each owner take_all's its whole chunk in one CAS and nobody
    /// steals — matches the static-partition fast path, but is TOP-LEVEL ONLY
    /// (a worker blocked in a nested dispatch would never get its chunk drained).
    steal: AtomicBool,
}
unsafe impl Send for TaskSlot {}
unsafe impl Sync for TaskSlot {}

unsafe fn call_body<F: Fn(Range<usize>) + Sync>(p: *const (), r: Range<usize>) {
    (&*(p as *const F))(r)
}

// ───────────────────────────── Shared state ───────────────────────────────

struct NodeShared {
    workers: Vec<usize>,
    injector: Injector<JobRef>,
}

struct Inner {
    nodes: Vec<NodeShared>,
    stealers: Vec<Stealer<JobRef>>,
    worker_node: Vec<usize>,
    handles: Vec<OnceLock<Thread>>, // each worker registers itself at startup
    sleeping: Vec<AtomicBool>,
    shutdown: AtomicBool,
    num_workers: usize,
    /// Type-stable bank of dispatch slots, indexed `participant * MAX_NEST +
    /// depth`. Allocated once, never freed — the basis of lock-free, UAF-free
    /// discovery. Size = (num_workers + 1) * MAX_NEST; the +1 is main.
    slots: Box<[TaskSlot]>,
    /// Per-node active-slot bitmap (one bit per slot). Publish sets the bit on
    /// every node that owns a chunk; done/unpublish clears it. Idle workers read
    /// these few words (cached-shared when zero) instead of taking a lock.
    node_active: Vec<Box<[AtomicU64]>>,
    /// Fixed chunk→node grouping (chunk k is owned by worker k, so this never
    /// changes per dispatch). Used at publish time to decide which node bitmaps
    /// to flag.
    node_has_chunks: Vec<bool>,
    /// Count of in-flight STEALING dispatches. Cross-node chunk help (find_work
    /// step 5) is only useful while a stealing task exists, so idle workers gate
    /// that scan on this being > 0 — a static-only workload never pays for the
    /// cross-NUMA bitmap scans. Only the owner of a stealing dispatch writes it
    /// (once up, once down); a stale read is harmless because step 5 is a pure
    /// optimization (every chunk is reachable on its own node via step 2).
    stealing_inflight: AtomicUsize,
}
unsafe impl Sync for Inner {}
unsafe impl Send for Inner {}

impl Inner {
    fn handle(&self, w: usize) -> Option<Thread> {
        self.handles[w].get().cloned()
    }
    #[inline]
    fn set_active_bit(&self, node: usize, sidx: usize) {
        self.node_active[node][sidx / 64].fetch_or(1u64 << (sidx % 64), Release);
    }
    #[inline]
    fn clear_active_bit(&self, node: usize, sidx: usize) {
        self.node_active[node][sidx / 64].fetch_and(!(1u64 << (sidx % 64)), Release);
    }
    fn wake_one_on_node(&self, node: usize) {
        for &w in &self.nodes[node].workers {
            if self.sleeping[w].load(Acquire) {
                if let Some(t) = self.handle(w) {
                    t.unpark();
                    return;
                }
            }
        }
        if let Some(&w) = self.nodes[node].workers.first() {
            if let Some(t) = self.handle(w) {
                t.unpark();
            }
        }
    }
    fn wake_all(&self) {
        for w in 0..self.num_workers {
            if let Some(t) = self.handle(w) {
                t.unpark();
            }
        }
    }
}

// ─────────────────────────── Per-worker context ───────────────────────────

struct Ctx {
    index: usize,
    node: usize,
    local: Deque<JobRef>,
    victims_same: Vec<usize>,
    victims_cross: Vec<usize>,
    near_chunks: Vec<usize>, // same-node chunk owners ordered by distance
    inner: Arc<Inner>,
}

thread_local! { static CURRENT: Cell<*const Ctx> = Cell::new(std::ptr::null()); }
#[inline]
fn current() -> Option<&'static Ctx> {
    let p = CURRENT.with(|c| c.get());
    if p.is_null() {
        None
    } else {
        Some(unsafe { &*p })
    }
}

// An external (non-worker) caller — typically the main thread — is lazily
// registered as a participant the first time it dispatches, instead of bouncing
// through `run()` and parking. Its `Ctx` lives here for the thread's lifetime
// (stable heap address via `Box`), and `CURRENT` is pointed at it. Workers never
// touch this; they already have a stack `Ctx`. The box is rebuilt only if the
// caller switches to a different pool (checked by `Arc::ptr_eq` on `inner`).
thread_local! {
    static EXTERNAL_CTX: RefCell<Option<Box<Ctx>>> = const { RefCell::new(None) };
}

// Per-participant parallel-for nesting depth, used to pick this thread's slot
// `participant * MAX_NEST + depth`. Thread-local, so each worker and main track
// their own stack independently.
thread_local! {
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

impl Ctx {
    /// THE scheduler. Make one unit of progress; false if nothing found.
    fn find_work(&self) -> bool {
        // 1. own deque
        if let Some(j) = self.local.pop() {
            unsafe { (j.exec)(j.ptr) };
            return true;
        }
        // 2. active parallel task on my node, discovered lock-free via the
        //    per-node bitmap (no mutex, no Arc clone).
        if self.discover_and_drain(self.node) {
            return true;
        }
        // 3. same-node deque steal (incl. node injector)
        if let Some(j) = self.steal_same() {
            unsafe { (j.exec)(j.ptr) };
            return true;
        }
        // 4. cross-node deque steal (accepts a page migration)
        if let Some(j) = self.steal_cross() {
            unsafe { (j.exec)(j.ptr) };
            return true;
        }
        // 5. cross-node chunk help (last resort) — only meaningful while a
        //    stealing dispatch is live. A static-only workload skips this scan
        //    entirely; every chunk is reachable on its own node via step 2.
        if self.inner.stealing_inflight.load(Relaxed) > 0 {
            for n in 0..self.inner.nodes.len() {
                if n == self.node {
                    continue;
                }
                if self.discover_and_drain(n) {
                    return true;
                }
            }
        }
        false
    }

    fn steal_same(&self) -> Option<JobRef> {
        loop {
            match self.inner.nodes[self.node]
                .injector
                .steal_batch_and_pop(&self.local)
            {
                Steal::Success(j) => return Some(j),
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
        for &v in &self.victims_same {
            loop {
                match self.inner.stealers[v].steal() {
                    Steal::Success(j) => return Some(j),
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }
        None
    }

    fn steal_cross(&self) -> Option<JobRef> {
        for n in 0..self.inner.nodes.len() {
            if n == self.node {
                continue;
            }
            loop {
                match self.inner.nodes[n]
                    .injector
                    .steal_batch_and_pop(&self.local)
                {
                    Steal::Success(j) => return Some(j),
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }
        for &v in &self.victims_cross {
            loop {
                match self.inner.stealers[v].steal() {
                    Steal::Success(j) => return Some(j),
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }
        None
    }

    /// Scan `node`'s active-slot bitmap and drain the first slot that yields
    /// work. Lock-free: a few `AtomicU64` loads instead of a mutex. Returns
    /// whether any work was done.
    fn discover_and_drain(&self, node: usize) -> bool {
        let words = &self.inner.node_active[node];
        for wi in 0..words.len() {
            let mut bits = words[wi].load(Acquire);
            while bits != 0 {
                let sidx = wi * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1; // clear lowest set bit
                let slot = &self.inner.slots[sidx];
                let g = slot.gen.load(Acquire);
                if g & 1 == 0 {
                    continue;
                } // free (stale bit)
                if slot.done.load(Acquire) {
                    continue;
                } // already complete
                if self.drain_slot(slot, g) {
                    return true;
                }
            }
        }
        false
    }

    /// Claim one range from `slot` under generation `gen`: own chunk first, then
    /// (if `steal`) nearest same-node peers, then anywhere. `Stale` means the
    /// slot was recycled — propagate it so the caller stops touching this slot.
    #[inline]
    fn claim_next(&self, slot: &TaskSlot, gen: u64, grain: u32, steal: bool) -> Claim {
        let nchunks = slot.chunks.len();
        let own = self.index;
        if own < nchunks {
            // Stealing: take a grain (leaves the chunk splittable for thieves).
            // Static: take the whole chunk in one CAS — the fast path.
            let c = if steal {
                slot.chunks[own].take_front(gen, grain)
            } else {
                slot.chunks[own].take_all(gen)
            };
            match c {
                Claim::Got(r) => return Claim::Got(r),
                Claim::Stale => return Claim::Stale,
                Claim::Empty => {} // own drained; fall through to stealing
            }
        }
        if !steal {
            return Claim::Empty;
        } // never touch a peer
        for &k in &self.near_chunks {
            if k < nchunks {
                match slot.chunks[k].steal_back(gen) {
                    Claim::Got(r) => return Claim::Got(r),
                    Claim::Stale => return Claim::Stale,
                    Claim::Empty => {}
                }
            }
        }
        for k in 0..nchunks {
            match slot.chunks[k].steal_back(gen) {
                Claim::Got(r) => return Claim::Got(r),
                Claim::Stale => return Claim::Stale,
                Claim::Empty => {}
            }
        }
        Claim::Empty
    }

    /// Drain one session of `slot` under `gen`: run every range we can claim,
    /// summing the count, then publish it to `processed` in a single add. Whoever
    /// pushes `processed` to `total` sets `done`.
    ///
    /// SAFETY of the `Stale` path: if we ever successfully claimed a range under
    /// `gen`, that range is uncounted until the end of this session, so the task
    /// cannot be complete, so the owner cannot have recycled the slot, so `gen`
    /// is still current and we CANNOT observe `Stale`. Therefore `Stale` implies
    /// `my == 0`, and we never add a stale count to a recycled slot's counter.
    fn drain_slot(&self, slot: &TaskSlot, gen: u64) -> bool {
        let total = slot.total.load(Acquire);
        let grain = slot.grain.load(Acquire) as u32;
        let steal = slot.steal.load(Acquire);
        let body = slot.body.load(Acquire) as *const ();
        let call: unsafe fn(*const (), Range<usize>) =
            unsafe { std::mem::transmute(slot.call.load(Acquire)) };
        let mut my = 0usize;
        loop {
            match self.claim_next(slot, gen, grain, steal) {
                Claim::Got(r) => {
                    let n = r.len();
                    if n != 0 {
                        unsafe { call(body, r) };
                    }
                    my += n;
                }
                Claim::Empty | Claim::Stale => break,
            }
        }
        if my != 0 && slot.processed.0.fetch_add(my, AcqRel) + my == total {
            slot.done.store(true, Release);
        }
        my != 0
    }

    /// Help until `pred`. Never barriers, never parks.
    fn help_until(&self, pred: impl Fn() -> bool) {
        let backoff = Backoff::new();
        while !pred() {
            if self.find_work() {
                backoff.reset();
            } else {
                backoff.snooze();
            }
        }
    }
}

// ─────────────────────────────── The pool ─────────────────────────────────

/// Pool configuration.
pub struct Config {
    /// Explicit topology: node -> CPU ids. Default: detected (Linux /sys), else
    /// one node spanning all CPUs.
    pub topology: Option<Vec<Vec<usize>>>,
    /// Total worker count. `None` => one worker per CPU in `topology`. `Some(n)`
    /// => exactly `n` workers spread as evenly as possible across the nodes
    /// (e.g. 128 over 2 nodes = 64 each), each pinned to a distinct CPU on its
    /// node (CPUs are cycled if `n` per node exceeds that node's CPU count).
    pub threads: Option<usize>,
    /// Pin worker threads to their CPU (Linux). Default true.
    pub pin: bool,
}
impl Default for Config {
    fn default() -> Self {
        Config {
            topology: None,
            threads: None,
            pin: true,
        }
    }
}

pub struct ThreadPool {
    inner: Arc<Inner>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl ThreadPool {
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    /// Build a pool with exactly `n` workers, balanced across detected NUMA nodes.
    pub fn with_threads(n: usize) -> Self {
        Self::with_config(Config {
            threads: Some(n),
            ..Config::default()
        })
    }

    pub fn with_config(cfg: Config) -> Self {
        let topo = cfg.topology.unwrap_or_else(detect_topology);
        let num_nodes = topo.len().max(1);
        let mut worker_cpu = Vec::new();
        let mut worker_node = Vec::new();
        let mut node_workers: Vec<Vec<usize>> = vec![Vec::new(); topo.len()];

        match cfg.threads {
            // Default: one worker per CPU in the topology.
            None => {
                for (nid, cpus) in topo.iter().enumerate() {
                    for &cpu in cpus {
                        let gid = worker_cpu.len();
                        worker_cpu.push(cpu);
                        worker_node.push(nid);
                        node_workers[nid].push(gid);
                    }
                }
            }
            // Fixed total: spread evenly across nodes, keeping global ids grouped
            // by node (node 0's workers first) so affinity super-blocks stay
            // contiguous. Within a node, take distinct CPUs (cycle if oversubscribed).
            Some(total) => {
                let total = total.max(1);
                let base = total / num_nodes;
                let rem = total % num_nodes;
                for (nid, cpus) in topo.iter().enumerate() {
                    if cpus.is_empty() {
                        continue;
                    }
                    let count = base + if nid < rem { 1 } else { 0 };
                    for k in 0..count {
                        let cpu = cpus[k % cpus.len()];
                        let gid = worker_cpu.len();
                        worker_cpu.push(cpu);
                        worker_node.push(nid);
                        node_workers[nid].push(gid);
                    }
                }
            }
        }
        let num_workers = worker_cpu.len().max(1);

        let deques: Vec<Deque<JobRef>> = (0..num_workers).map(|_| Deque::new_lifo()).collect();
        let stealers: Vec<Stealer<JobRef>> = deques.iter().map(|d| d.stealer()).collect();

        let nodes: Vec<NodeShared> = (0..topo.len())
            .map(|nid| NodeShared {
                workers: node_workers[nid].clone(),
                injector: Injector::new(),
            })
            .collect();

        // Type-stable slot bank: one slot per (participant, depth). Participants
        // are the `num_workers` workers plus main (index num_workers), hence +1.
        let nslots = (num_workers + 1) * MAX_NEST;
        let slots: Box<[TaskSlot]> = (0..nslots)
            .map(|_| TaskSlot {
                gen: AtomicU64::new(0), // even => free
                chunks: (0..num_workers).map(|_| Chunk::new_free()).collect(),
                processed: CachePadded(AtomicUsize::new(0)),
                total: AtomicUsize::new(0),
                done: AtomicBool::new(false),
                body: AtomicPtr::new(std::ptr::null_mut()),
                call: AtomicUsize::new(0),
                grain: AtomicU64::new(0),
                steal: AtomicBool::new(true),
            })
            .collect();

        // Per-node active-slot bitmap (ceil(nslots/64) words each).
        let words = nslots.div_ceil(64);
        let node_active: Vec<Box<[AtomicU64]>> = (0..topo.len())
            .map(|_| (0..words).map(|_| AtomicU64::new(0)).collect())
            .collect();
        // A node owns chunks iff it has workers (chunk k belongs to worker k).
        let node_has_chunks: Vec<bool> = (0..topo.len())
            .map(|nid| !node_workers[nid].is_empty())
            .collect();

        let inner = Arc::new(Inner {
            nodes,
            stealers,
            worker_node,
            handles: (0..num_workers).map(|_| OnceLock::new()).collect(),
            sleeping: (0..num_workers).map(|_| AtomicBool::new(false)).collect(),
            shutdown: AtomicBool::new(false),
            num_workers,
            slots,
            node_active,
            node_has_chunks,
            stealing_inflight: AtomicUsize::new(0),
        });

        let mut workers = Vec::with_capacity(num_workers);
        let mut deques = deques.into_iter();
        for gid in 0..num_workers {
            let inner_c = inner.clone();
            let local = deques.next().unwrap();
            let node = inner.worker_node[gid];
            let cpu = worker_cpu[gid];
            let pin = cfg.pin;

            let victims_same: Vec<usize> = inner.nodes[node]
                .workers
                .iter()
                .cloned()
                .filter(|&w| w != gid)
                .collect();
            let victims_cross: Vec<usize> = (0..num_workers)
                .filter(|&w| inner.worker_node[w] != node)
                .collect();
            let mut near: Vec<usize> = inner.nodes[node]
                .workers
                .iter()
                .cloned()
                .filter(|&w| w != gid)
                .collect();
            near.sort_by_key(|&w| (w as isize - gid as isize).unsigned_abs());

            let h = thread::Builder::new()
                .name(format!("numa-pool-{gid}"))
                .spawn(move || {
                    if pin {
                        pin_to(cpu);
                    }
                    let _ = inner_c.handles[gid].set(thread::current());
                    let ctx = Ctx {
                        index: gid,
                        node,
                        local,
                        victims_same,
                        victims_cross,
                        near_chunks: near,
                        inner: inner_c,
                    };
                    CURRENT.with(|c| c.set(&ctx as *const Ctx));
                    worker_loop(&ctx);
                    CURRENT.with(|c| c.set(std::ptr::null()));
                })
                .expect("spawn worker");
            workers.push(h);
        }

        ThreadPool { inner, workers }
    }

    pub fn num_threads(&self) -> usize {
        self.inner.num_workers
    }

    /// External entry: on a worker runs `f`; off-pool injects and blocks the
    /// caller until done.
    pub fn run<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        if current().is_some() {
            return f();
        }
        let job = StackJob::new(f, Some(thread::current()));
        self.inner.nodes[0].injector.push(job.job_ref());
        self.inner.wake_one_on_node(0);
        while !job.done.load(Acquire) {
            thread::park_timeout(std::time::Duration::from_micros(200));
        }
        unsafe { job.take_result() }
    }

    /// Fork/join, work-first with inline fallback.
    pub fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        match current() {
            Some(ctx) => join_on(ctx, a, b),
            None => self.run(move || join_on(current().unwrap(), a, b)),
        }
    }

    /// Data-parallel for: one affinity chunk per worker, splittable tails.
    /// Load-balanced (work-stealing) and safe to nest. Default for irregular or
    /// nested work.
    pub fn parallel_for<F>(&self, range: Range<usize>, body: F)
    where
        F: Fn(Range<usize>) + Sync + Send,
    {
        let len = range.len();
        let grain = (len / (self.inner.num_workers * 8))
            .max(1)
            .min(u32::MAX as usize) as u32;
        self.dispatch(range, grain, true, body)
    }

    /// Static (no-steal) data-parallel for: each worker takes its whole chunk in
    /// a single claim and nobody steals — the fast path for *balanced* loops,
    /// matching a static partition's throughput.
    ///
    /// CONTRACT: TOP-LEVEL ONLY. Do not call from inside a parallel body and do
    /// not nest a dispatch inside its body — a worker blocked in a nested
    /// dispatch would never have its chunk drained, deadlocking the section. Use
    /// `parallel_for` whenever nesting or load imbalance is possible.
    pub fn parallel_for_static<F>(&self, range: Range<usize>, body: F)
    where
        F: Fn(Range<usize>) + Sync + Send,
    {
        let len = range.len();
        // Grain is irrelevant in static mode (take_all), but keep it sane.
        let grain = (len / self.inner.num_workers).max(1).min(u32::MAX as usize) as u32;
        self.dispatch(range, grain, false, body)
    }

    /// Return the calling thread's participant `Ctx`. A worker already has one
    /// (set in its loop). An external caller (main) is lazily registered as
    /// participant index `num_workers` with a PRIVATE deque, so it participates
    /// in dispatches inline rather than injecting a job and parking in `run()`.
    ///
    /// Main's deque is intentionally NOT in the pool's `stealers` array and main
    /// is in no node's `workers` list, so: workers never steal from main, never
    /// try to wake main, and never index any pool array at `num_workers`. Main
    /// contributes purely by stealing chunk work and helping — it owns no chunk
    /// (its index ≥ `chunks.len()`, so `claim_next`'s own-chunk path is skipped,
    /// which is OOB-safe by construction). Tradeoff: once registered, `scope`/
    /// `join` issued *from main* run inline-serial (main's deque isn't stealable),
    /// which is fine for the hot path (top-level `parallel_for`).
    fn participant_ctx(&self) -> &'static Ctx {
        if let Some(ctx) = current() {
            return ctx;
        }
        EXTERNAL_CTX.with(|cell| {
            let rebuild = match cell.borrow().as_ref() {
                Some(c) => !Arc::ptr_eq(&c.inner, &self.inner),
                None => true,
            };
            if rebuild {
                *cell.borrow_mut() = Some(Box::new(self.make_external_ctx()));
            }
            // Box gives the Ctx a stable heap address; the pointer stays valid
            // after the borrow ends because the box is not moved again unless a
            // pool switch rebuilds it (which also resets CURRENT here).
            let ptr: *const Ctx = &**cell.borrow().as_ref().unwrap();
            CURRENT.with(|c| c.set(ptr));
        });
        current().unwrap()
    }

    /// Build the external caller's participant context. Index `num_workers`
    /// (distinct from every worker), pinned conceptually to node 0 for steal
    /// ordering, with a private LIFO deque. It steals node-0 chunks/deques first
    /// (locality), then everything else via the all-chunks fallback in
    /// `claim_next` and the cross victims here.
    fn make_external_ctx(&self) -> Ctx {
        let nw = self.inner.num_workers;
        let node0: Vec<usize> = self.inner.nodes[0].workers.clone();
        let victims_cross: Vec<usize> = (0..nw)
            .filter(|&w| self.inner.worker_node[w] != 0)
            .collect();
        Ctx {
            index: nw, // ≥ chunks.len(): no own chunk, steal-only
            node: 0,
            local: Deque::new_lifo(),
            victims_same: node0.clone(),
            victims_cross,
            near_chunks: node0, // chunk k is owned by worker k
            inner: self.inner.clone(),
        }
    }

    /// Explicit-grain, load-balanced data-parallel for (stealing on). Safe to
    /// nest. For the no-steal fast path use `parallel_for_static`.
    pub fn parallel_for_grain<F>(&self, range: Range<usize>, grain: u32, body: F)
    where
        F: Fn(Range<usize>) + Sync + Send,
    {
        self.dispatch(range, grain, true, body)
    }

    fn dispatch<F>(&self, range: Range<usize>, grain: u32, steal: bool, body: F)
    where
        F: Fn(Range<usize>) + Sync + Send,
    {
        // Main participates inline: no run() hop, no park. Workers re-enter here
        // for nested dispatches and already have a Ctx.
        let ctx = self.participant_ctx();
        let len = range.len();
        if len == 0 {
            return;
        }
        assert!(
            range.end <= u32::MAX as usize,
            "u32 index space; use 128-bit CAS for >4G"
        );

        // Slot = (participant, depth). Beyond MAX_NEST we run serially rather than
        // overflow the bank — deep nesting is rare and this is always correct.
        let depth = DEPTH.with(|d| d.get());
        if depth >= MAX_NEST {
            body(range);
            return;
        }
        let sidx = ctx.index * MAX_NEST + depth;
        let slot = &self.inner.slots[sidx];

        // Fill the slot, then publish by bumping `gen` even->odd with Release. All
        // the field writes below are ordinary stores ordered before that Release,
        // so any acquirer of `gen` (or of a node bit, set after) sees them.
        let gen = slot.gen.load(Relaxed) + 1; // was even (free) -> odd (active)
        let n = self.inner.num_workers;
        let base = len / n;
        let rem = len % n;
        let mut off = range.start;
        for k in 0..n {
            let sz = base + if k < rem { 1 } else { 0 };
            slot.chunks[k].reset(gen, off as u32, (off + sz) as u32);
            off += sz;
        }
        slot.processed.0.store(0, Relaxed);
        slot.total.store(len, Relaxed);
        slot.grain.store(grain as u64, Relaxed);
        slot.steal.store(steal, Relaxed);
        slot.done.store(false, Relaxed);
        slot.body.store(&body as *const F as *mut (), Relaxed);
        slot.call.store(call_body::<F> as usize, Relaxed);
        slot.gen.store(gen, Release); // PUBLISH

        // Enable cross-node help for stealing dispatches (gates find_work step 5).
        if steal {
            self.inner.stealing_inflight.fetch_add(1, Release);
        }

        // Flag every node that owns a chunk, then wake any parked workers there.
        for nid in 0..self.inner.nodes.len() {
            if self.inner.node_has_chunks[nid] {
                self.inner.set_active_bit(nid, sidx);
                for &w in &self.inner.nodes[nid].workers {
                    if self.inner.sleeping[w].load(Acquire) {
                        if let Some(t) = self.inner.handle(w) {
                            t.unpark();
                        }
                    }
                }
            }
        }

        // Participate at depth+1 (so a nested parallel_for from a body we run uses
        // the next slot). With stealing ON the caller can finish the slot itself;
        // main owns no chunk and contributes by stealing. Never parks — snooze()
        // yields, keeping the caller hot. `done` ⇒ processed==total ⇒ all bodies
        // ran, so dropping `body` on return is safe.
        DEPTH.with(|d| d.set(depth + 1));
        let backoff = Backoff::new();
        while !slot.done.load(Acquire) {
            if ctx.drain_slot(slot, gen) || ctx.find_work() {
                backoff.reset();
            } else {
                backoff.snooze();
            }
        }
        DEPTH.with(|d| d.set(depth));

        // Unpublish: clear bits, then bump `gen` odd->even (free). A late helper
        // either skips the cleared bit, or claims with the old `gen` and gets
        // `Stale` — it can never act on the slot's NEXT occupant.
        for nid in 0..self.inner.nodes.len() {
            if self.inner.node_has_chunks[nid] {
                self.inner.clear_active_bit(nid, sidx);
            }
        }
        slot.gen.store(gen + 1, Release); // FREE
        if steal {
            self.inner.stealing_inflight.fetch_sub(1, Release);
        }
    }

    /// Structured concurrency; spawns may borrow the scope frame.
    pub fn scope<'s, R: Send>(&self, f: impl FnOnce(&Scope<'s>) -> R + Send) -> R {
        if current().is_none() {
            return self.run(|| self.scope(f));
        }
        let ctx = current().unwrap();
        let scope = Scope {
            remaining: AtomicUsize::new(0),
            ctx: ctx as *const Ctx,
            _marker: std::marker::PhantomData,
        };
        let out = f(&scope);
        ctx.help_until(|| scope.remaining.load(Acquire) == 0);
        out
    }

    /// CPU-bound long job; occupies a worker, which narrows subsequent
    /// `parallel_for` fan-out. NOT for syscall-blocking work.
    pub fn spawn_background(&self, node: usize, f: impl FnOnce() + Send + 'static) {
        let node = node.min(self.inner.nodes.len() - 1);
        self.inner.nodes[node]
            .injector
            .push(HeapJob::boxed(f).into_ref());
        self.inner.wake_one_on_node(node);
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Release);
        self.inner.wake_all();
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

// ───────────────────────────────── Scope ──────────────────────────────────

pub struct Scope<'s> {
    remaining: AtomicUsize,
    ctx: *const Ctx,
    _marker: std::marker::PhantomData<&'s ()>,
}
impl<'s> Scope<'s> {
    pub fn spawn<F: FnOnce(&Scope<'s>) + Send + 's>(&self, f: F) {
        self.remaining.fetch_add(1, AcqRel);
        let scope_ptr = self as *const Scope<'s> as usize;
        let rem = &self.remaining as *const AtomicUsize as usize;
        let job = HeapJob::boxed(move || {
            let s = unsafe { &*(scope_ptr as *const Scope<'s>) };
            f(s);
            let rem = unsafe { &*(rem as *const AtomicUsize) };
            rem.fetch_sub(1, AcqRel);
        });
        let ctx = unsafe { &*self.ctx };
        ctx.local.push(job.into_ref());
        ctx.inner.wake_one_on_node(ctx.node);
    }
}

// ─────────────────────────── join / worker loop ───────────────────────────

fn join_on<A, B, RA, RB>(ctx: &Ctx, a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    let bjob = StackJob::new(b, None);
    let bref = bjob.job_ref();
    let bptr = bref.ptr;
    ctx.local.push(bref);

    let ra = a();

    let rb = match ctx.local.pop() {
        Some(j) if j.ptr == bptr => unsafe { bjob.run_inline() },
        Some(j) => {
            unsafe { (j.exec)(j.ptr) };
            ctx.help_until(|| bjob.done.load(Acquire));
            unsafe { bjob.take_result() }
        }
        None => {
            ctx.help_until(|| bjob.done.load(Acquire));
            unsafe { bjob.take_result() }
        }
    };
    (ra, rb)
}

fn worker_loop(ctx: &Ctx) {
    let inner = &ctx.inner;
    let mut idle: u32 = 0;
    loop {
        if inner.shutdown.load(Acquire) {
            if !ctx.find_work() {
                break;
            }
            continue;
        }
        if ctx.find_work() {
            idle = 0;
            continue;
        }
        idle += 1;
        // Ladder: stay hot first (spin), then a short-sleep band that releases
        // the core to SMT siblings without leaving the run queue, and only then
        // park. Keeping workers warm across the inter-dispatch gap means the
        // publisher's wake is usually a no-op (the worker already sees the new
        // task by spinning) instead of a real futex round-trip every dispatch.
        if idle < HOT_SPIN_ITERS {
            std::hint::spin_loop();
            continue;
        }
        if idle < PARK_AFTER_ITERS {
            // ~hrtimer-granularity yield; cheap, keeps us schedulable.
            thread::sleep(std::time::Duration::from_nanos(1));
            continue;
        }
        // Park: announce, re-check (closes lost-wakeup gap), then sleep.
        // HARDEN: a futex eventcount removes the recheck race window.
        inner.sleeping[ctx.index].store(true, SeqCst);
        if inner.shutdown.load(Acquire) || ctx.find_work() {
            inner.sleeping[ctx.index].store(false, SeqCst);
            idle = 0;
            continue;
        }
        thread::park_timeout(std::time::Duration::from_micros(500));
        inner.sleeping[ctx.index].store(false, SeqCst);
        idle = 0;
    }
}

// ─────────────────────────── topology / pinning ───────────────────────────

fn detect_topology() -> Vec<Vec<usize>> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(rd) = fs::read_dir("/sys/devices/system/node") {
            let mut nodes: Vec<(usize, Vec<usize>)> = Vec::new();
            for e in rd.flatten() {
                let name = e.file_name();
                let s = name.to_string_lossy();
                if let Some(idx) = s.strip_prefix("node") {
                    if let Ok(nid) = idx.parse::<usize>() {
                        let list = fs::read_to_string(e.path().join("cpulist")).unwrap_or_default();
                        let cpus = parse_cpulist(&list);
                        if !cpus.is_empty() {
                            nodes.push((nid, cpus));
                        }
                    }
                }
            }
            if !nodes.is_empty() {
                nodes.sort_by_key(|(n, _)| *n);
                return nodes.into_iter().map(|(_, c)| c).collect();
            }
        }
    }
    let n = thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    vec![(0..n).collect()]
}

#[cfg(target_os = "linux")]
fn parse_cpulist(s: &str) -> Vec<usize> {
    let mut v = Vec::new();
    for part in s.trim().split(',') {
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                v.extend(a..=b);
            }
        } else if let Ok(a) = part.trim().parse::<usize>() {
            v.push(a);
        }
    }
    v
}

#[cfg(target_os = "linux")]
fn pin_to(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}
#[cfg(not(target_os = "linux"))]
fn pin_to(_cpu: usize) {}

// ──────────────────────────── global pool ─────────────────────────────────
//
// A process-wide pool you can use without naming or constructing anything. It
// lazily auto-initializes on first use (`ThreadPool::new`, i.e. detected NUMA
// topology, pinned workers). To override config, call `global::init(cfg)` ONCE
// before any other access.
//
// Intentional leak: a `static` is never dropped, so the global pool's worker
// threads are never joined — they're reclaimed at process exit. This is the
// conventional choice for a global pool (it sidesteps shutdown-ordering hazards)
// and means `Drop for ThreadPool` only ever runs for pools you build yourself.

pub mod global {
    use super::*;
    use std::sync::OnceLock;

    static POOL: OnceLock<ThreadPool> = OnceLock::new();

    /// The global pool, auto-initialized with defaults on first call.
    #[inline]
    pub fn pool() -> &'static ThreadPool {
        POOL.get_or_init(ThreadPool::new)
    }

    /// Initialize the global pool with an explicit config. Returns `false` (and
    /// does nothing) if the pool was already initialized or used. Call before
    /// the first `pool()`/free-function access.
    pub fn init(cfg: Config) -> bool {
        POOL.set(ThreadPool::with_config(cfg)).is_ok()
    }

    /// Initialize the global pool with exactly `n` workers, balanced across nodes.
    /// Returns `false` if the pool was already initialized. Call before first use.
    pub fn init_threads(n: usize) -> bool {
        POOL.set(ThreadPool::with_threads(n)).is_ok()
    }

    pub fn is_initialized() -> bool {
        POOL.get().is_some()
    }

    /// Run on the global pool (no-op wrap if already on a worker).
    #[inline]
    pub fn run<R: Send>(f: impl FnOnce() -> R + Send) -> R {
        pool().run(f)
    }

    #[inline]
    pub fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        pool().join(a, b)
    }

    #[inline]
    pub fn parallel_for<F>(range: Range<usize>, body: F)
    where
        F: Fn(Range<usize>) + Sync + Send,
    {
        pool().parallel_for(range, body)
    }

    #[inline]
    pub fn parallel_for_grain<F>(range: Range<usize>, grain: u32, body: F)
    where
        F: Fn(Range<usize>) + Sync + Send,
    {
        pool().parallel_for_grain(range, grain, body)
    }

    #[inline]
    pub fn scope<'s, R: Send>(f: impl FnOnce(&Scope<'s>) -> R + Send) -> R {
        pool().scope(f)
    }

    #[inline]
    pub fn spawn_background(node: usize, f: impl FnOnce() + Send + 'static) {
        pool().spawn_background(node, f)
    }
}

// Re-exported at the crate root so callers can write `numa_pool::parallel_for(..)`
// etc. Use the longer `global::pool()` / `global::init()` for lifecycle control.
pub use global::{join, parallel_for, parallel_for_grain, run, scope, spawn_background};

// ─────────────────────────────────── tests ────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_for_sums() {
        let pool = ThreadPool::new();
        let n = 1_000_000usize;
        let data: Vec<u64> = (0..n as u64).collect();
        let acc = AtomicU64::new(0);
        pool.run(|| {
            pool.parallel_for(0..n, |r| {
                let mut local = 0u64;
                for i in r {
                    local += data[i];
                }
                acc.fetch_add(local, Relaxed);
            });
        });
        assert_eq!(acc.load(Relaxed), (n as u64 - 1) * (n as u64) / 2);
    }

    #[test]
    fn nested_join_fib() {
        let pool = ThreadPool::new();
        fn fib(p: &ThreadPool, n: u64) -> u64 {
            if n < 2 {
                return n;
            }
            let (a, b) = p.join(|| fib(p, n - 1), || fib(p, n - 2));
            a + b
        }
        assert_eq!(pool.run(|| fib(&pool, 28)), 317811);
    }

    #[test]
    fn nested_parallel_for() {
        let pool = ThreadPool::new();
        let (outer, inner) = (200usize, 5000usize);
        let acc = AtomicU64::new(0);
        pool.run(|| {
            pool.parallel_for(0..outer, |r| {
                for _ in r {
                    pool.parallel_for(0..inner, |ir| {
                        acc.fetch_add(ir.len() as u64, Relaxed);
                    });
                }
            });
        });
        assert_eq!(acc.load(Relaxed), (outer * inner) as u64);
    }

    #[test]
    fn scope_spawns() {
        let pool = ThreadPool::new();
        let acc = AtomicU64::new(0);
        pool.run(|| {
            pool.scope(|s| {
                for i in 0..1000u64 {
                    let acc = &acc;
                    s.spawn(move |_| {
                        acc.fetch_add(i, Relaxed);
                    });
                }
            });
        });
        assert_eq!(acc.load(Relaxed), (0..1000u64).sum());
    }

    #[test]
    fn global_free_functions() {
        // No instantiation: lazy auto-init on first use.
        let n = 500_000usize;
        let data: Vec<u64> = (0..n as u64).collect();
        let acc = AtomicU64::new(0);
        super::run(|| {
            super::parallel_for(0..n, |r| {
                let mut s = 0u64;
                for i in r {
                    s += data[i];
                }
                acc.fetch_add(s, Relaxed);
            });
            // nested join via the global, inside the same global pool
            let (x, y) = super::join(|| 2u64 + 2, || 3u64 * 3);
            acc.fetch_add(x + y, Relaxed);
        });
        assert!(super::global::is_initialized());
        assert_eq!(acc.load(Relaxed), (n as u64 - 1) * (n as u64) / 2 + 13);
    }
}
