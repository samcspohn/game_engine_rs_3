//! A small work-stealing fork-join pool with nested parallelism and
//! long-running background jobs.
//!
//! Goals (in order):
//!   1. **Simplicity.** A reader should be able to hold the whole scheduler
//!      in their head. No NUMA, no pinning, no per-call timing.
//!   2. **Purposeful dispatch.** `parallel_for` claims exactly the workers
//!      it needs from an idle mask, seeds each one's cursor with a
//!      contiguous slice, and wakes them through a binary wake tree of
//!      targeted `unpark`s — never a broadcast.
//!   3. **Nested parallelism with an availability heuristic.** The idle
//!      mask doubles as an *available-thread counter*: a dispatch of 20
//!      items on a 32-thread pool claims 19 workers and leaves 12
//!      claimable. When a body itself calls `parallel_for`, the first
//!      caller to arrive claims those 12; later arrivals find the mask
//!      empty, publish their job in the slot table anyway, seed only
//!      their own cursor, and grind through it in `STEAL_GRAIN` chunks —
//!      any thread that finishes its own job sweeps the slot table and
//!      steals half-ranges from unfinished jobs, so parallelism recovers
//!      as soon as anyone frees up.
//!   4. **Background jobs.** `spawn_background` hands a long-running task
//!      to an idle worker (or queues it for the next worker to go idle).
//!      A worker running a background task never re-enters the idle mask,
//!      so concurrent `parallel_for` dispatches automatically partition
//!      across the threads that are actually available.
//!
//! Usage:
//! ```ignore
//! my_thread_pool::global::init(n_threads);            // once at startup
//! my_thread_pool::global::parallel_for(0..n, |sub| { /* sub: Range<usize> */ });
//! my_thread_pool::global::spawn_background(|| { /* long-running */ });
//! ```
//!
//! Safety model for `parallel_for`:
//!   The body is captured by reference and shared with workers as a raw
//!   pointer + monomorphised trampoline. The calling thread does not
//!   return until the job's `joiners` count reaches zero, which (see the
//!   barrier notes below) implies every seeded cursor is drained and no
//!   thief still holds a reference to the job. Bodies must not panic: a
//!   panicking body unwinds its worker and permanently wedges the
//!   dispatch barrier (same contract as the previous epoch-based pool).

use crossbeam_deque::{Injector, Steal};

use portable_atomic::AtomicU128;
use std::cell::{Cell, UnsafeCell};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle, Thread};

/// Idle ladder threshold (iterations of a wait loop with no progress).
/// Below `SPIN_ITERS` we `spin_loop`; past it, workers `park()` and the
/// dispatch barrier downgrades to `yield_now`. There is no timeout on
/// worker parks: every assignment stores the mailbox *before* the
/// matching `unpark`, and `unpark` deposits a permit even if issued
/// before the target parks, so there is no missed-wake race.
const SPIN_ITERS: u32 = 20_000;

/// Minimum claim size while draining a cursor. An owner claims *half*
/// its remaining range per CAS (geometric drain: few, large body calls,
/// which keeps per-call overhead — and any per-call atomics in user
/// bodies — off the hot path) but never less than this floor, so the
/// tail still splits finely enough for thieves to balance stragglers.
const STEAL_GRAIN: u64 = 256;

/// Maximum `parallel_for` nesting depth per thread. Each thread owns
/// `MAX_DEPTH` cursor rows; a dispatch at depth `d` seeds the caller's
/// row `d` (an outer dispatch's cursor at row `d-1` may still hold
/// unstolen items — those stay stealable by that outer job's peers).
/// Exceeding this panics loudly rather than silently serialising.
const MAX_DEPTH: usize = 8;

/// Cursor start/end are 48-bit; the remaining 32 bits of the 128-bit
/// word carry the owning job's tag (low 32 bits of its unique id), so a
/// single CAS both claims a range and validates which job it belongs to.
const RANGE_MASK: u64 = (1 << 48) - 1;

/// Slot-state bit marking a job slot as allocated but not yet published.
/// Scavengers skip locked slots.
const SLOT_LOCKED: u64 = 1 << 63;

/// Mailbox `pos` sentinel: "drain the background queue" instead of a
/// tree position in a fork-join job.
const BG_ASSIGNMENT: u32 = u32::MAX;

// ──────────────────────────────────────────────────────────────────────
// Scheduler design (lock-free hot path + job table)
// ──────────────────────────────────────────────────────────────────────
//
// Participants: `num_threads` total. Participant 0 is the external
// dispatcher (the engine's main thread — exactly one non-worker thread
// may dispatch at a time, enforced loudly). Participants 1..n are pool
// workers.
//
// Availability: `idle_mask` bit w ⇔ worker participant `w + 1` is idle
// and claimable. Initialised to all-set at construction (workers are
// claimable before their threads even finish spawning — an assignment
// written to a mailbox is picked up by the worker's pre-park check).
// A dispatcher *claims* workers by CASing bits off; a worker re-sets
// its own bit only after it has finished its assignment, swept other
// jobs for stealable work, and drained the background queue. Workers
// running background tasks or scavenging are therefore invisible to
// the partitioning heuristic, exactly as intended.
//
// Per dispatch (`parallel_for`):
//   * Claim up to `min(size - 1, popcount(idle_mask))` workers.
//   * Allocate a slot in the job table (CAS free → LOCKED), wait for
//     any straggling thief of the slot's *previous* job to deregister,
//     write the job fields (body ptr + trampoline, range, member list),
//     pre-count the claimed workers in `joiners`, seed the caller's own
//     cursor, then publish (`state = jid`, Release).
//   * Wake the two roots of a binary tree over the member list: seed
//     child's cursor → store child's mailbox (Release) → unpark. Each
//     woken worker wakes its own two children the same way, so wake
//     latency is O(log n) and only claimed workers are touched.
//   * Every participant runs the same loop: drain own cursor in
//     STEAL_GRAIN chunks, then sweep all cursor rows for ranges tagged
//     with this job and steal half of any it finds. A full empty sweep
//     means the job has no unclaimed work left.
//   * Barrier: claimed workers decrement `joiners` when they exit the
//     sweep; scavenging thieves increment/decrement around their visit.
//     The caller spins until `joiners == 0`. At that point every seeded
//     cursor has been drained (each leaver observed all-empty, and
//     cursors within a job only shrink) and nobody holds a reference to
//     the job, so the caller may free the slot and drop the body.
//
// Stealing across jobs ("scavenging"): a worker that finishes its
// assignment walks the job table; for each published job it registers
// as a joiner, re-validates the slot (monotonic job ids make this
// ABA-proof), sweeps all cursors for that job's tag, and deregisters.
// This is what lets threads that finish their single solo-mode job pile
// onto whichever nested dispatch is still grinding.
//
// Cursor encoding: (start:48 | end:48 | tag:32) in one AtomicU128 —
// still a single `cmpxchg16b` on x86_64. Owner advances `start`;
// thieves lower `end`; the tag makes a steal impossible to land on a
// cursor that was re-seeded for a different job (the CAS's expected
// value carries the old tag). Within one job a cursor's packed value
// never repeats (start only grows, end only shrinks), so the CAS is
// ABA-free there too.

/// 64-byte aligned cursor wrapper. Forces each cursor onto its own cache
/// line so owner/thief races on one participant's cursor don't
/// false-share with its neighbours. `AtomicU128` itself only needs
/// 16-byte alignment (cmpxchg16b); the rest is padding.
#[repr(align(64))]
struct PaddedCursor {
    packed: AtomicU128,
}

impl PaddedCursor {
    const fn empty() -> Self {
        Self {
            packed: AtomicU128::new(0),
        }
    }
}

#[inline(always)]
fn pack_cursor(s: u64, e: u64, tag: u32) -> u128 {
    debug_assert!(s <= RANGE_MASK && e <= RANGE_MASK);
    ((tag as u128) << 96) | ((s as u128) << 48) | (e as u128)
}

#[inline(always)]
fn unpack_cursor(p: u128) -> (u64, u64, u32) {
    (
        ((p >> 48) as u64) & RANGE_MASK,
        (p as u64) & RANGE_MASK,
        (p >> 96) as u32,
    )
}

/// Attempt to steal the back half of a cursor's remaining range, but
/// only if the cursor currently belongs to job `tag`. Returns the
/// claimed `[s, e)` on success; `None` on tag mismatch, empty cursor,
/// or CAS contention (callers just move on to the next victim).
#[inline]
fn try_steal(cursor: &PaddedCursor, tag: u32) -> Option<(u64, u64)> {
    let packed = cursor.packed.load(Ordering::Acquire);
    let (s, e, t) = unpack_cursor(packed);
    if t != tag || s >= e {
        return None;
    }
    let remaining = e - s;
    // Take half, rounded up, so a 1-item remainder is fully claimed
    // rather than left orphaned with an owner who may already be gone.
    let take = remaining - remaining / 2;
    let new_end = e - take;
    if cursor
        .packed
        .compare_exchange_weak(
            packed,
            pack_cursor(s, new_end, tag),
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

/// Type-erased job descriptor. Lives in a `JobSlot`'s `UnsafeCell`;
/// written only by the slot's allocator before publication, read only
/// by threads holding a (pre-counted or registered) joiner reference.
#[derive(Clone, Copy)]
struct Job {
    /// Erased pointer to the caller's `F` (stack-rooted in `parallel_for`).
    body_ptr: *const (),
    /// Monomorphised trampoline: casts `body_ptr` back to `&F` and calls
    /// it. You CANNOT transmute `*const F` to `fn(...)` — a closure
    /// value's address is captured data, not executable code.
    call_fn: unsafe fn(*const (), Range<usize>),
    /// Absolute range start and item count of the dispatch.
    start: usize,
    size: usize,
    /// Caller + claimed workers; slice `v` of the range goes to wake-tree
    /// node `v` in `0..n_participants`.
    n_participants: usize,
}

const EMPTY_JOB: Job = Job {
    body_ptr: std::ptr::null(),
    call_fn: noop_call,
    start: 0,
    size: 0,
    n_participants: 1,
};

unsafe fn noop_call(_: *const (), _: Range<usize>) {}

/// One entry of the job table.
struct JobSlot {
    /// 0 = free; `SLOT_LOCKED | jid` = allocated, not yet published;
    /// `jid` = live. Job ids come from a monotonic counter, so a stale
    /// `state == jid` recheck after registering a joiner is ABA-proof.
    state: AtomicU64,
    /// Number of threads currently attached to this job: claimed workers
    /// are pre-counted at dispatch; scavenging thieves register around
    /// their visit. Never `store`d — a thief racing a free/realloc may
    /// transiently inc/dec across the boundary, and plain fetch ops keep
    /// the count balanced through that.
    joiners: AtomicUsize,
    /// Written by the allocator between the LOCKED CAS and publication;
    /// read by joiners. The allocator waits for `joiners == 0` before
    /// writing, so a straggling thief of the previous job never races
    /// these fields.
    job: UnsafeCell<Job>,
    /// Participant ids of the claimed workers, valid for
    /// `job.n_participants - 1` entries. Wake-tree node `v >= 1` is
    /// `members[v - 1]`.
    members: UnsafeCell<Box<[u32]>>,
}

// SAFETY: `state`/`joiners` are atomics. The `UnsafeCell` fields have a
// single writer (the allocator, pre-publication, after waiting out all
// joiners of the previous occupant) and readers that all hold a joiner
// reference acquired via `state` (Release publish / Acquire validate).
unsafe impl Sync for JobSlot {}
unsafe impl Send for JobSlot {}

/// Per-worker assignment mailbox. Packed (jid:64 | slot:32 | pos:32);
/// job ids are unique, so every new assignment changes the word and the
/// worker can spin on a pure load. `pos == BG_ASSIGNMENT` means "drain
/// the background queue"; otherwise `pos` is the worker's index in the
/// job's member list (wake-tree node `pos + 1`).
#[repr(align(64))]
struct Mailbox {
    packed: AtomicU128,
}

#[inline(always)]
fn pack_mailbox(jid: u64, slot: u32, pos: u32) -> u128 {
    ((jid as u128) << 64) | ((slot as u128) << 32) | pos as u128
}

#[inline(always)]
fn unpack_mailbox(m: u128) -> (u64, u32, u32) {
    ((m >> 64) as u64, (m >> 32) as u32, m as u32)
}

struct Shared {
    /// Total participants (external dispatcher + workers). Fixed.
    num_threads: usize,
    /// Work stealing toggle. Fixed at construction. When false, every
    /// participant drains its own seeded slice in one claim and neither
    /// in-job sweeps nor cross-job scavenging run — dispatch reduces to
    /// pure static partitioning (useful for A/B benchmarking).
    work_stealing: bool,
    shutdown: AtomicBool,
    /// Bit `w` (of word `w / 64`) set ⇔ worker participant `w + 1` is
    /// idle and claimable. Total popcount = the available-thread counter
    /// used to partition work. Claims CAS one word at a time; a claim
    /// that races another dispatcher may see slightly fewer workers,
    /// which is fine — availability is a heuristic, and stealing
    /// recovers any resulting imbalance.
    idle_mask: Box<[AtomicU64]>,
    /// Monotonic job id source (also used to make background-assignment
    /// mailbox words unique). Ids start at 1; 0 means "free slot".
    job_counter: AtomicU64,
    /// High-water mark of nesting depth (+1) ever used; bounds how many
    /// cursor rows per participant a steal sweep must visit. Monotonic.
    depth_hwm: AtomicUsize,
    /// High-water mark of job-slot indices ever allocated; bounds the
    /// scavenge scan. Monotonic (allocation prefers low slots).
    slots_hwm: AtomicUsize,
    /// Exactly one non-worker thread may be inside `parallel_for` at a
    /// time (participant 0's cursor rows and slice are reserved for it).
    external_active: AtomicBool,
    /// `num_threads * MAX_DEPTH` cursors: participant `p`'s cursor for a
    /// dispatch at nesting depth `d` is `cursors[p * MAX_DEPTH + d]`.
    cursors: Box<[PaddedCursor]>,
    /// One mailbox per worker (participant `w + 1` owns `mailboxes[w]`).
    mailboxes: Box<[Mailbox]>,
    /// Per-worker park handle, published once by each worker before it
    /// enters its wait loop. Assignments unpark exactly the target.
    park_handles: Box<[OnceLock<Thread>]>,
    /// The job table ("the stack" that finished threads search for
    /// unfinished jobs). Sized so it cannot overflow under the depth
    /// limit: every live job pins one thread at some depth.
    slots: Box<[JobSlot]>,
    /// Long-running background tasks waiting for a worker.
    bg_queue: Injector<Box<dyn FnOnce() + Send>>,
    /// `steal_order[p]`: peer participants in preference order (nearest
    /// first, wrapping). Sweeps visit each peer's cursor rows in this
    /// order.
    steal_order: Box<[Box<[usize]>]>,
}

// SAFETY: everything cross-thread in `Shared` is either atomic,
// immutable after construction, or covered by `JobSlot`'s invariants.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

/// Per-thread scheduler identity. Workers set this once at startup
/// (depth 0 = "executing my assignment"); an external dispatcher claims
/// participant 0 for the duration of a top-level `parallel_for`. Nested
/// dispatches bump `depth` so each level uses its own cursor row.
#[derive(Clone, Copy)]
struct TlsCtx {
    pool: *const Shared,
    participant: usize,
    depth: usize,
}

thread_local! {
    static CTX: Cell<Option<TlsCtx>> = const { Cell::new(None) };
}

/// Slice of `[start, start + size)` assigned to wake-tree node `v` when
/// `n` participants take part. Same arithmetic on every thread, so a
/// worker can seed its children's cursors without further coordination.
#[inline]
fn node_slice(start: usize, size: usize, n: usize, v: usize) -> (u64, u64) {
    (
        (start + v * size / n) as u64,
        (start + (v + 1) * size / n) as u64,
    )
}

/// Claim up to `want` idle workers (lowest indices first, which keeps
/// the slice→worker mapping stable across back-to-back dispatches for
/// cache affinity). Returns how many were claimed; their participant
/// ids land in `out[..n]`. Claimed workers are removed from the idle
/// mask — the caller now owes each of them exactly one mailbox write.
fn claim_workers(shared: &Shared, want: usize, out: &mut [u32]) -> usize {
    let mut claimed = 0;
    for (wi, word) in shared.idle_mask.iter().enumerate() {
        if claimed >= want {
            break;
        }
        loop {
            let mask = word.load(Ordering::Relaxed);
            let take = (want - claimed).min(mask.count_ones() as usize);
            if take == 0 {
                break;
            }
            let mut bits = 0u64;
            let mut m = mask;
            for _ in 0..take {
                let b = m & m.wrapping_neg(); // lowest set bit
                bits |= b;
                m &= !b;
            }
            if word
                .compare_exchange_weak(mask, mask & !bits, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                let mut b = bits;
                while b != 0 {
                    // word wi, bit z → participant wi * 64 + z + 1
                    out[claimed] = (wi * 64) as u32 + b.trailing_zeros() + 1;
                    claimed += 1;
                    b &= b - 1;
                }
                break;
            }
            // CAS contention: retry this word with a fresh snapshot.
        }
    }
    claimed
}

/// Allocate a free job slot: CAS its state to LOCKED, then wait for any
/// straggling thief of the slot's previous job to deregister so the
/// `UnsafeCell` fields can be rewritten without a reader race (thieves
/// register a joiner *before* validating the slot, so a stale reader
/// always holds a joiner while it might still read `job`).
fn alloc_slot(shared: &Shared) -> (usize, &JobSlot) {
    for (i, s) in shared.slots.iter().enumerate() {
        if s.state.load(Ordering::Relaxed) == 0
            && s.state
                .compare_exchange(0, SLOT_LOCKED, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        {
            shared.slots_hwm.fetch_max(i + 1, Ordering::Relaxed);
            while s.joiners.load(Ordering::Acquire) != 0 {
                std::hint::spin_loop();
            }
            return (i, s);
        }
    }
    panic!(
        "my_thread_pool: job table exhausted — more concurrently active \
         parallel_for dispatches than slots (deep recursion?)"
    );
}

/// Sweep every cursor row for ranges tagged with `tag` and steal half of
/// any found, restarting from the nearest peer after each success.
/// Returns true if at least one range was executed. Terminates because
/// a job's cursors only shrink once seeded, and a full empty pass means
/// no unclaimed work remains (in-flight chunks are tracked by `joiners`,
/// not by this sweep).
///
/// SAFETY: caller must hold a joiner reference on the job identified by
/// `tag` (pre-counted claimed worker, registered scavenger, or the
/// dispatching caller itself), so `job.body_ptr` stays valid.
unsafe fn steal_sweep(shared: &Shared, me: usize, tag: u32, job: &Job) -> bool {
    let depths = shared.depth_hwm.load(Ordering::Acquire).min(MAX_DEPTH);
    let mut did = false;
    'outer: loop {
        for &peer in shared.steal_order[me].iter() {
            for d in 0..depths {
                if let Some((s, e)) = try_steal(&shared.cursors[peer * MAX_DEPTH + d], tag) {
                    unsafe { (job.call_fn)(job.body_ptr, s as usize..e as usize) };
                    did = true;
                    continue 'outer;
                }
            }
        }
        break;
    }
    did
}

/// Drive one participant through a job: drain its own cursor in
/// STEAL_GRAIN chunks, then (if stealing is enabled) sweep the whole
/// cursor table for this job's tag.
///
/// SAFETY: caller must hold a joiner reference on the job (see
/// `steal_sweep`), and `own_cursor` must be this thread's cursor row.
unsafe fn run_job(shared: &Shared, slot: &JobSlot, jid: u64, own_cursor: usize) {
    let job = unsafe { *slot.job.get() };
    let tag = jid as u32;
    let own = &shared.cursors[own_cursor];
    let stealing = shared.work_stealing;
    loop {
        let packed = own.packed.load(Ordering::Acquire);
        let (s, e, t) = unpack_cursor(packed);
        if s >= e {
            break;
        }
        debug_assert_eq!(t, tag, "own cursor retagged mid-drain");
        let claim_end = if stealing {
            // Claim half the remainder (floored at STEAL_GRAIN): the
            // unclaimed back half stays in the cursor for thieves, and
            // the drain finishes in O(log(slice / STEAL_GRAIN)) claims.
            let remaining = e - s;
            s + (remaining / 2).max(STEAL_GRAIN).min(remaining)
        } else {
            e
        };
        if own
            .packed
            .compare_exchange_weak(
                packed,
                pack_cursor(claim_end, e, tag),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            unsafe { (job.call_fn)(job.body_ptr, s as usize..claim_end as usize) };
        }
        // else: a thief lowered `end` concurrently — retry with the
        // fresh snapshot.
    }
    if stealing {
        unsafe { steal_sweep(shared, own_cursor / MAX_DEPTH, tag, &job) };
    }
}

/// Seed and wake the two wake-tree children of `node`. Child cursor and
/// mailbox MUST be written before the unpark: an unpark can race a
/// worker that is between its mailbox check and its `park()`, and since
/// each claimed worker receives exactly one unpark per assignment, a
/// wake that arrives before its mailbox write would strand the worker
/// parked forever. Store-then-unpark (with the Release store) makes any
/// wake caused by this unpark also observe the assignment.
fn wake_children(shared: &Shared, slot_idx: usize, jid: u64, node: usize) {
    // SAFETY: caller holds a joiner reference (dispatcher or pre-counted
    // claimed worker), so the slot fields are stable.
    let job = unsafe { *shared.slots[slot_idx].job.get() };
    let members = unsafe { &**shared.slots[slot_idx].members.get() };
    let k = job.n_participants - 1;
    let tag = jid as u32;
    for c in (node * 2 + 1)..=(node * 2 + 2) {
        if c > k {
            break;
        }
        let w = members[c - 1] as usize;
        let (s, e) = node_slice(job.start, job.size, job.n_participants, c);
        // Claimed workers were idle, so their depth-0 cursor row is free.
        shared.cursors[w * MAX_DEPTH]
            .packed
            .store(pack_cursor(s, e, tag), Ordering::Relaxed);
        shared.mailboxes[w - 1]
            .packed
            .store(pack_mailbox(jid, slot_idx as u32, (c - 1) as u32), Ordering::Release);
        if let Some(t) = shared.park_handles[w - 1].get() {
            t.unpark();
        }
    }
}

/// Walk the job table and steal from every live job found ("threads that
/// finish with their single job search the stack for unfinished jobs").
/// Returns true if any work was executed.
fn scavenge(shared: &Shared, me: usize) -> bool {
    let mut did = false;
    let n_slots = shared.slots_hwm.load(Ordering::Acquire).min(shared.slots.len());
    for slot in shared.slots[..n_slots].iter() {
        if shared.shutdown.load(Ordering::Relaxed) {
            break;
        }
        let jid = slot.state.load(Ordering::Acquire);
        if jid == 0 || jid & SLOT_LOCKED != 0 {
            continue;
        }
        // Register as a joiner *before* validating: once registered, the
        // job's caller cannot pass its barrier (and free the body) until
        // we deregister. If the job completed in the window, the recheck
        // fails (job ids are monotonic — no ABA) and we back off; the
        // transient increment at worst costs the slot's next occupant a
        // few spins in `alloc_slot`.
        slot.joiners.fetch_add(1, Ordering::AcqRel);
        if slot.state.load(Ordering::Acquire) == jid {
            let job = unsafe { *slot.job.get() };
            did |= unsafe { steal_sweep(shared, me, jid as u32, &job) };
        }
        slot.joiners.fetch_sub(1, Ordering::Release);
    }
    did
}

/// Drain the background queue on the current thread.
fn run_background(shared: &Shared) {
    loop {
        match shared.bg_queue.steal() {
            Steal::Success(task) => task(),
            Steal::Retry => continue,
            Steal::Empty => break,
        }
    }
}

fn worker_main(participant: usize, shared: Arc<Shared>) {
    // Publish our `Thread` handle so assignments can target an `unpark`
    // at exactly this worker.
    shared.park_handles[participant - 1]
        .set(thread::current())
        .expect("worker park handle slot already set");
    // Scheduler identity for nested `parallel_for` from job bodies run
    // on this thread: depth 0 is our own assignment's cursor row.
    CTX.set(Some(TlsCtx {
        pool: &*shared as *const Shared,
        participant,
        depth: 0,
    }));

    let mailbox = &shared.mailboxes[participant - 1];
    let mask_word = &shared.idle_mask[(participant - 1) / 64];
    let bit = 1u64 << ((participant - 1) % 64);
    let mut last: u128 = 0;
    loop {
        // Wait for an assignment: pure-load spin (the mailbox line stays
        // Shared across cores), then park indefinitely. Safe because
        // every assignment stores the mailbox before unparking us, and
        // pool Drop unparks everyone after setting `shutdown`.
        let mut idle: u32 = 0;
        let m = loop {
            let m = mailbox.packed.load(Ordering::Acquire);
            if m != last {
                break m;
            }
            if shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            idle = idle.saturating_add(1);
            if idle < SPIN_ITERS {
                std::hint::spin_loop();
            } else {
                thread::park();
            }
        };
        last = m;
        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }

        let (jid, slot_idx, pos) = unpack_mailbox(m);
        if pos == BG_ASSIGNMENT {
            run_background(&shared);
        } else {
            // We are pre-counted in the job's `joiners`, so the slot and
            // body are guaranteed alive until our decrement below.
            let slot = &shared.slots[slot_idx as usize];
            wake_children(&shared, slot_idx as usize, jid, pos as usize + 1);
            unsafe { run_job(&shared, slot, jid, participant * MAX_DEPTH) };
            // Release: side effects of every body invocation on this
            // worker happen-before the caller's Acquire barrier load.
            slot.joiners.fetch_sub(1, Ordering::Release);
        }

        // Steal from any other unfinished job before going idle. We are
        // not in the idle mask here, so no dispatcher can claim us and
        // then stall behind a long stolen chunk.
        if shared.work_stealing {
            while scavenge(&shared, participant) {}
        }

        // Re-idle. The loop closes the race with `spawn_background`:
        // a task pushed after our drain but before our mask bit landed
        // finds no claimable worker, so we re-check the queue after
        // setting the bit and take the bit back to drain it ourselves —
        // unless a dispatcher already claimed us, in which case we fall
        // through to the wait loop and honour the incoming assignment.
        loop {
            run_background(&shared);
            mask_word.fetch_or(bit, Ordering::AcqRel);
            if shared.bg_queue.is_empty() {
                break;
            }
            if mask_word.fetch_and(!bit, Ordering::AcqRel) & bit == 0 {
                break; // claimed in the window; assignment incoming
            }
        }
    }
}

pub struct ThreadPool {
    handles: Vec<JoinHandle<()>>,
    shared: Arc<Shared>,
}

impl ThreadPool {
    /// Convenience constructor with work stealing enabled.
    #[inline]
    pub fn new(n_threads: usize) -> Self {
        Self::with_options(n_threads, true)
    }

    /// Build a pool with `n_threads` total participants (external
    /// dispatcher + workers) and an explicit work-stealing toggle.
    ///
    /// `work_stealing = false` reverts every dispatch to pure static
    /// partitioning: each participant runs its seeded slice in one claim
    /// and neither in-job stealing nor cross-job scavenging run.
    pub fn with_options(n_threads: usize, work_stealing: bool) -> Self {
        assert!(
            n_threads >= 1,
            "ThreadPool requires at least 1 participant (the calling thread)"
        );
        // Cursor CAS is the inner-loop primitive; a software-lock
        // fallback would silently destroy throughput. Crash loudly if
        // the build target lacks 128-bit CAS (x86_64 cmpxchg16b /
        // aarch64 LSE) rather than degrade invisibly.
        assert!(
            AtomicU128::is_lock_free(),
            "portable_atomic::AtomicU128 is not lock-free on this target"
        );

        let n_workers = n_threads - 1;
        let cursors: Box<[PaddedCursor]> = (0..n_threads * MAX_DEPTH)
            .map(|_| PaddedCursor::empty())
            .collect();
        let steal_order: Box<[Box<[usize]>]> = (0..n_threads)
            .map(|me| ((me + 1)..n_threads).chain(0..me).collect())
            .collect();
        let mailboxes: Box<[Mailbox]> = (0..n_workers)
            .map(|_| Mailbox {
                packed: AtomicU128::new(0),
            })
            .collect();
        let park_handles: Box<[OnceLock<Thread>]> =
            (0..n_workers).map(|_| OnceLock::new()).collect();
        // One slot per (participant, depth): a live job pins its caller
        // at one depth level, so the table cannot overflow under the
        // MAX_DEPTH limit.
        let slots: Box<[JobSlot]> = (0..n_threads * MAX_DEPTH)
            .map(|_| JobSlot {
                state: AtomicU64::new(0),
                joiners: AtomicUsize::new(0),
                job: UnsafeCell::new(EMPTY_JOB),
                members: UnsafeCell::new(vec![0u32; n_workers].into_boxed_slice()),
            })
            .collect();
        // All workers are claimable from the start — an assignment made
        // before a worker thread finishes spawning is picked up by its
        // first mailbox check, and the unpark permit covers the race.
        let idle_mask: Box<[AtomicU64]> = (0..n_workers.div_ceil(64).max(1))
            .map(|wi| {
                let bits_here = n_workers.saturating_sub(wi * 64).min(64);
                AtomicU64::new(if bits_here == 64 {
                    u64::MAX
                } else {
                    (1u64 << bits_here) - 1
                })
            })
            .collect();

        let shared = Arc::new(Shared {
            num_threads: n_threads,
            work_stealing,
            shutdown: AtomicBool::new(false),
            idle_mask,
            job_counter: AtomicU64::new(0),
            depth_hwm: AtomicUsize::new(1),
            slots_hwm: AtomicUsize::new(0),
            external_active: AtomicBool::new(false),
            cursors,
            mailboxes,
            park_handles,
            slots,
            bg_queue: Injector::new(),
            steal_order,
        });
        let mut handles = Vec::with_capacity(n_workers);
        for i in 0..n_workers {
            let shared = shared.clone();
            let h = thread::Builder::new()
                .name(format!("my-thread-pool-{i}"))
                .spawn(move || worker_main(i + 1, shared))
                .expect("spawn worker thread");
            handles.push(h);
        }
        Self { handles, shared }
    }

    /// Total number of threads that may participate in a dispatch:
    /// workers + the calling thread.
    #[inline]
    pub fn num_threads(&self) -> usize {
        self.shared.num_threads
    }

    /// Whether work stealing is enabled for this pool. Fixed at
    /// construction.
    #[inline]
    pub fn work_stealing(&self) -> bool {
        self.shared.work_stealing
    }

    /// Snapshot of the available-thread counter: workers that are idle
    /// and claimable right now. Advisory — the value may change before
    /// the caller acts on it.
    #[inline]
    pub fn idle_workers(&self) -> usize {
        self.shared
            .idle_mask
            .iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as usize)
            .sum()
    }

    /// Parallel-for over `range`. Claims up to `min(len - 1, idle)`
    /// workers, splits the range into one contiguous slice per
    /// participant, and blocks (helping) until every item has run.
    ///
    /// May be called from inside another `parallel_for` body (on this
    /// same pool): the nested dispatch partitions across whatever
    /// workers are idle at that moment, and runs caller-only if none
    /// are — its range then stays stealable by threads that free up.
    pub fn parallel_for<F>(&self, range: Range<usize>, body: F)
    where
        F: Fn(Range<usize>) + Sync + Send,
    {
        let size = range.end.saturating_sub(range.start);
        if size == 0 {
            return;
        }
        assert!(
            range.end as u64 <= RANGE_MASK,
            "parallel_for range exceeds 48-bit cursor capacity"
        );
        let shared = &*self.shared;

        // Resolve this thread's scheduler identity and nesting depth.
        let prev = CTX.get();
        let (participant, depth, external) = match prev {
            Some(c) if std::ptr::eq(c.pool, shared as *const Shared) => {
                (c.participant, c.depth + 1, false)
            }
            Some(_) => panic!("parallel_for nested across two different ThreadPools"),
            None => {
                // External (non-worker) dispatcher: claim the reserved
                // participant-0 identity. Loud crash on a second
                // concurrent external dispatcher rather than corruption.
                assert!(
                    shared
                        .external_active
                        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                        .is_ok(),
                    "parallel_for called concurrently from two non-worker threads"
                );
                (0, 0, true)
            }
        };
        assert!(
            depth < MAX_DEPTH,
            "parallel_for nested deeper than MAX_DEPTH = {MAX_DEPTH}"
        );
        CTX.set(Some(TlsCtx {
            pool: shared,
            participant,
            depth,
        }));
        // Publish the new depth before the job becomes visible (the
        // slot-state Release below orders this for every observer), so
        // steal sweeps always cover our cursor row.
        shared.depth_hwm.fetch_max(depth + 1, Ordering::Relaxed);

        // Monomorphised trampoline for this concrete `F`.
        unsafe fn call_impl<F: Fn(Range<usize>)>(ptr: *const (), r: Range<usize>) {
            let f = unsafe { &*(ptr as *const F) };
            f(r);
        }

        // Allocate a job slot, then claim workers directly into its
        // member list. The availability heuristic: claim only workers
        // that are actually idle. Fewer items than threads ⇒ fewer
        // claims; busy (background / mid-job) workers are never claimed.
        let jid = shared.job_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let (slot_idx, slot) = alloc_slot(shared);
        // SAFETY: slot is LOCKED and all joiners of its previous job
        // have left (alloc_slot waited), so we are the sole accessor of
        // its UnsafeCell fields until publication.
        let claimed = claim_workers(
            shared,
            (size - 1).min(shared.num_threads - 1),
            unsafe { &mut *slot.members.get() },
        );
        let n_participants = claimed + 1;
        unsafe {
            *slot.job.get() = Job {
                body_ptr: &body as *const F as *const (),
                call_fn: call_impl::<F>,
                start: range.start,
                size,
                n_participants,
            };
        }
        // Pre-count the claimed workers in `joiners` before anything can
        // observe the job, so the barrier below cannot pass before they
        // all run.
        slot.joiners.fetch_add(claimed, Ordering::AcqRel);

        // Seed our own cursor (wake-tree node 0) with slice 0.
        let tag = jid as u32;
        let own_cursor = participant * MAX_DEPTH + depth;
        let (s0, e0) = node_slice(range.start, size, n_participants, 0);
        shared.cursors[own_cursor]
            .packed
            .store(pack_cursor(s0, e0, tag), Ordering::Relaxed);

        // Publish (scavengers may now find the job), then wake the two
        // roots of the binary wake tree; they wake their own children.
        slot.state.store(jid, Ordering::Release);
        wake_children(shared, slot_idx, jid, 0);

        // Participate: drain own slice, then steal within the job.
        unsafe { run_job(shared, slot, jid, own_cursor) };

        // Barrier: every claimed worker and every scavenging thief has
        // left the job once `joiners == 0`; at that point all seeded
        // cursors are drained and nobody references `body` any more.
        // Spin briefly, then yield — never park; the last leaver could
        // arrive at any moment and we want low wake-up latency here.
        let mut idle: u32 = 0;
        while slot.joiners.load(Ordering::Acquire) != 0 {
            idle = idle.saturating_add(1);
            if idle < SPIN_ITERS {
                std::hint::spin_loop();
            } else {
                thread::yield_now();
            }
        }

        // Free the slot and restore identity. `body` (and anything it
        // borrowed) is safe to drop on return.
        slot.state.store(0, Ordering::Release);
        CTX.set(prev);
        if external {
            shared.external_active.store(false, Ordering::Release);
        }
    }

    /// Enqueue a long-running task. It occupies one worker for its whole
    /// duration; that worker leaves the idle mask, so concurrent
    /// `parallel_for` dispatches partition across the remaining threads.
    /// If every worker is busy, the task waits until one goes idle.
    ///
    /// For *blocking* syscalls (file I/O, `vkWaitSemaphores`, …) prefer
    /// a dedicated `std::thread::spawn` so a compute core isn't stranded.
    ///
    /// Tasks still queued (never started) when the pool is dropped are
    /// discarded; running tasks are joined.
    pub fn spawn_background<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let shared = &*self.shared;
        assert!(
            shared.num_threads >= 2,
            "spawn_background requires at least one worker thread"
        );
        shared.bg_queue.push(Box::new(f));
        // Hand it to an idle worker immediately if there is one; if not,
        // the next worker to go idle drains the queue (see the re-idle
        // double-check in `worker_main`).
        let mut buf = [0u32; 1];
        if claim_workers(shared, 1, &mut buf) == 1 {
            let w = buf[0] as usize;
            let jid = shared.job_counter.fetch_add(1, Ordering::Relaxed) + 1;
            shared.mailboxes[w - 1]
                .packed
                .store(pack_mailbox(jid, 0, BG_ASSIGNMENT), Ordering::Release);
            if let Some(t) = shared.park_handles[w - 1].get() {
                t.unpark();
            }
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        // Spinning workers observe `shutdown` directly; parked ones need
        // the unpark to wake and re-check it.
        for handle in self.shared.park_handles.iter() {
            if let Some(t) = handle.get() {
                t.unpark();
            }
        }
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

    /// Initialise the global pool with `n_threads` total participants
    /// and work stealing enabled. Returns `false` if the pool was
    /// already initialised; the caller can `assert!` to surface a
    /// double-init bug loudly (no silent fallback).
    pub fn init(n_threads: usize) -> bool {
        POOL.set(ThreadPool::new(n_threads)).is_ok()
    }

    /// Initialise the global pool with an explicit work-stealing toggle.
    /// See [`ThreadPool::with_options`] for semantics. Returns `false`
    /// if the pool was already initialised.
    pub fn init_with_options(n_threads: usize, work_stealing: bool) -> bool {
        POOL.set(ThreadPool::with_options(n_threads, work_stealing))
            .is_ok()
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

    #[inline]
    pub fn spawn_background<F>(f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        pool().spawn_background(f)
    }
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

// ──────────────────────────────────────────────────────────────────────
// Tests — correctness of the cursor protocol, observability of stealing
// on imbalanced workloads, nested parallelism, and background jobs.
// ──────────────────────────────────────────────────────────────────────
//
// Each test builds its own `ThreadPool` so they're independent of
// the global singleton and of each other.

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64 as StdAtomicU64, AtomicU8, AtomicUsize};
    use std::time::Duration;

    // ---- Per-thread unique tag (used by the stealing-observation test) ----
    //
    // The pool doesn't expose "which participant am I" to user closures,
    // so we tag whichever OS thread runs the body with a lazily assigned
    // u64. Two different tids on items inside a single participant's
    // seeded range proves a thief ran some of that range.
    thread_local! {
        static TID: Cell<u64> = const { Cell::new(0) };
    }
    static NEXT_TID: StdAtomicU64 = StdAtomicU64::new(1);

    fn my_tid() -> u64 {
        TID.with(|c| {
            let mut t = c.get();
            if t == 0 {
                t = NEXT_TID.fetch_add(1, Ordering::Relaxed);
                c.set(t);
            }
            t
        })
    }

    /// Imbalance window used by the stealing-observation tests.
    /// Sized to dominate any plausible OS scheduling jitter under
    /// `cargo test`'s default parallel runner.
    const STRAGGLE_MS: u64 = 500;

    /// Race-free witness that stealing fired: at least one tid
    /// processed more items than its seeded share `n / num_threads`.
    ///
    /// Why this works regardless of the timing race:
    ///   * Static partitioning + no stealing ⇒ every tid processes
    ///     exactly `seed_share` items. Max == seed_share.
    ///   * Stealing ⇒ some tid claimed items from a cursor that was
    ///     not its own. The total of all per-tid counts is still `n`,
    ///     so if one tid is over `seed_share`, another must be under.
    ///     Max > seed_share.
    ///
    /// Whether the eventual straggler is main, a worker, or whichever
    /// participant won the race to the slow item is irrelevant — the
    /// fingerprint only checks that the distribution is no longer
    /// the static partition.
    fn assert_steal_skew(owners: &[StdAtomicU64], n: usize, num_threads: usize) {
        let mut counts: HashMap<u64, usize> = HashMap::new();
        for o in owners {
            *counts.entry(o.load(Ordering::Relaxed)).or_insert(0) += 1;
        }
        let seed_share = n / num_threads;
        let max_count = *counts.values().max().expect("at least one tid recorded");
        assert!(
            max_count > seed_share,
            "stealing did NOT occur: no tid processed more than its \
             seeded share of {seed_share} items — the distribution \
             is the pure static partition. Per-tid counts: {counts:?}"
        );
    }

    /// Every index in `0..n` is visited exactly once, across a range of
    /// `n` values including ones that don't divide evenly by the pool
    /// size. Catches off-by-one in cursor seeding, double-claims from a
    /// races condition between owner and thief, and dropped tails.
    #[test]
    fn coverage_every_index_visited_once() {
        let pool = ThreadPool::new(4);
        for &n in &[1usize, 7, 64, 1024, 31_337, 100_000] {
            let visits: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(0)).collect();
            pool.parallel_for(0..n, |r| {
                for i in r {
                    visits[i].fetch_add(1, Ordering::Relaxed);
                }
            });
            for (i, v) in visits.iter().enumerate() {
                let c = v.load(Ordering::Relaxed);
                assert_eq!(c, 1, "index {i} visited {c} times (n = {n})");
            }
        }
    }

    /// Regression: `range.start` must be honoured, both in the caller's
    /// cursor seeding and in the worker-side wake-tree seeding (which
    /// re-derives child slices from the job's `size`/`start`). Before
    /// the fix, `parallel_for(100..1100, ..)` dispatched `0..1000`.
    #[test]
    fn offset_range_start_is_honoured() {
        let pool = ThreadPool::new(4);
        for &(start, end) in &[(100usize, 1_100usize), (5, 12), (31_337, 62_674)] {
            let n = end - start;
            let visits: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(0)).collect();
            pool.parallel_for(start..end, |r| {
                for i in r {
                    assert!(
                        (start..end).contains(&i),
                        "index {i} outside dispatched range {start}..{end}"
                    );
                    visits[i - start].fetch_add(1, Ordering::Relaxed);
                }
            });
            for (i, v) in visits.iter().enumerate() {
                let c = v.load(Ordering::Relaxed);
                assert_eq!(c, 1, "index {} visited {c} times", start + i);
            }
        }
    }

    /// Independent correctness check: parallel sum equals serial sum.
    /// A double-executed item would inflate the total; a dropped item
    /// would shrink it.
    #[test]
    fn sum_matches_serial_no_double_or_drop() {
        let pool = ThreadPool::new(4);
        for &n in &[100usize, 10_000, 250_000] {
            let total = AtomicUsize::new(0);
            pool.parallel_for(0..n, |r| {
                let mut local = 0usize;
                for i in r {
                    local = local.wrapping_add(i);
                }
                total.fetch_add(local, Ordering::Relaxed);
            });
            let expected: usize = (0..n).sum();
            assert_eq!(
                total.load(Ordering::Relaxed),
                expected,
                "sum mismatch for n = {n}"
            );
        }
    }

    /// Many back-to-back dispatches on the same pool. Catches stale
    /// cursor state, job-table bookkeeping bugs, and barrier races that
    /// would let a later dispatch start before the previous one drained.
    #[test]
    fn back_to_back_dispatches_all_correct() {
        let pool = ThreadPool::new(4);
        let n = 10_000usize;
        for round in 0..50 {
            let visits: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(0)).collect();
            pool.parallel_for(0..n, |r| {
                for i in r {
                    visits[i].fetch_add(1, Ordering::Relaxed);
                }
            });
            for (i, v) in visits.iter().enumerate() {
                let c = v.load(Ordering::Relaxed);
                assert_eq!(c, 1, "round {round} index {i} visited {c} times");
            }
        }
    }

    /// Regression test for a binary-wake-tree race: assignments store
    /// the child's mailbox and then call `unpark`. If that order were
    /// reversed, a worker could wake, recheck its still-stale mailbox,
    /// find no change, and re-park — and since each claimed worker gets
    /// exactly one `unpark` per assignment, it would then hang forever,
    /// wedging the barrier in `parallel_for` permanently. This only
    /// reproduces when fewer items than workers are dispatched (small
    /// claim sets) and it is a rare, timing-dependent race, so this
    /// hammers many small back-to-back dispatches on a wide pool to
    /// make the race window likely to be hit at least once.
    ///
    /// Runs on a background thread and bounds the wait with
    /// `recv_timeout` so a regression fails the test loudly instead of
    /// hanging the whole suite forever.
    #[test]
    fn many_small_dispatches_do_not_deadlock_binary_wake() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(|| {
                let pool = ThreadPool::new(16);
                let n = 3usize; // fewer items than workers on every round
                for round in 0..5_000 {
                    let visits: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(0)).collect();
                    pool.parallel_for(0..n, |r| {
                        for i in r {
                            visits[i].fetch_add(1, Ordering::Relaxed);
                        }
                    });
                    for (i, v) in visits.iter().enumerate() {
                        let c = v.load(Ordering::Relaxed);
                        assert_eq!(c, 1, "round {round} index {i} visited {c} times");
                    }
                }
            });
            let _ = tx.send(result);
        });
        match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => std::panic::resume_unwind(e),
            Err(_) => panic!(
                "parallel_for deadlocked: binary wake tree left a worker \
                 permanently parked (no result within 30s)"
            ),
        }
    }

    /// Stealing actually fires when work is imbalanced.
    ///
    /// Mechanism: whichever participant processes item 0 stalls for
    /// `STRAGGLE_MS` ms. All other participants finish their tiny
    /// seeded slices in microseconds, find their own cursors empty,
    /// and (if stealing works) sweep the cursor table and pull items
    /// off the stalled participant's cursor.
    ///
    /// The straggle duration is intentionally large (500 ms) so that
    /// under cargo's default parallel test runner — which can
    /// oversubscribe the CPU 4–6× — peer threads still reliably get
    /// scheduled to enter their steal sweep within the window. A
    /// shorter window flakes on oversubscribed runners even though
    /// the protocol is correct.
    #[test]
    fn stealing_fires_on_imbalanced_work() {
        let n_workers = 4;
        let pool = ThreadPool::new(n_workers);
        let n = 4_000usize;
        // `owners[i]` records the tid that last wrote to it; `visits[i]`
        // independently counts how many times the body ran for `i`. The
        // tid is for the stealing observation, the count is the
        // exact-once guarantee — a CAS bug in the claim protocol that
        // lets an owner and a thief both run the same item would inflate
        // `visits[i]` to 2.
        let owners: Vec<StdAtomicU64> = (0..n).map(|_| StdAtomicU64::new(0)).collect();
        let visits: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(0)).collect();

        pool.parallel_for(0..n, |r| {
            for i in r {
                if i == 0 {
                    std::thread::sleep(Duration::from_millis(STRAGGLE_MS));
                }
                owners[i].store(my_tid(), Ordering::Relaxed);
                visits[i].fetch_add(1, Ordering::Relaxed);
            }
        });

        // Exact-once: catches both skips (count == 0) and double-runs
        // (count >= 2). Must hold even when stealing fires.
        for (i, v) in visits.iter().enumerate() {
            let c = v.load(Ordering::Relaxed);
            assert_eq!(
                c, 1,
                "item {i} visited {c} times under imbalanced stealing \
                 — cursor CAS allowed overlap or drop"
            );
        }

        // Race-free stealing fingerprint: if no stealing happened, each
        // tid processes exactly its seeded share `n / num_threads`. If
        // any tid processed more, someone stole work from someone else.
        assert_steal_skew(&owners, n, n_workers);
    }

    /// With work stealing disabled, exact-once coverage still holds:
    /// every participant just drains its own seeded slice. Catches
    /// regressions where the steal toggle accidentally changes the
    /// own-cursor drain behaviour or the barrier accounting.
    #[test]
    fn no_steal_still_covers_every_index() {
        let pool = ThreadPool::with_options(4, false);
        assert!(!pool.work_stealing());
        for &n in &[1usize, 7, 1024, 31_337, 100_000] {
            let visits: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(0)).collect();
            pool.parallel_for(0..n, |r| {
                for i in r {
                    visits[i].fetch_add(1, Ordering::Relaxed);
                }
            });
            for (i, v) in visits.iter().enumerate() {
                let c = v.load(Ordering::Relaxed);
                assert_eq!(c, 1, "no-steal: index {i} visited {c} times (n = {n})");
            }
        }
    }

    /// The disable toggle actually disables. Same imbalanced workload
    /// that proves stealing fires in the on-case; with stealing off,
    /// every participant's seeded slice must carry exactly one tid,
    /// because no thief is allowed to reach into anyone's cursor.
    /// (Single dispatch on a fresh pool, so the claim deterministically
    /// takes every worker and the split is exactly `n / num_threads`.)
    #[test]
    fn no_steal_actually_disables_stealing() {
        let n_workers = 4;
        let pool = ThreadPool::with_options(n_workers, false);
        let n = 4_000usize;
        let owners: Vec<StdAtomicU64> = (0..n).map(|_| StdAtomicU64::new(0)).collect();
        let visits: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(0)).collect();

        pool.parallel_for(0..n, |r| {
            for i in r {
                if i == 0 {
                    std::thread::sleep(Duration::from_millis(STRAGGLE_MS));
                }
                owners[i].store(my_tid(), Ordering::Relaxed);
                visits[i].fetch_add(1, Ordering::Relaxed);
            }
        });

        // Exact-once coverage holds under the imbalanced load too —
        // disabled stealing must not introduce skips or double-runs.
        for (i, v) in visits.iter().enumerate() {
            let c = v.load(Ordering::Relaxed);
            assert_eq!(c, 1, "no-steal: item {i} visited {c} times");
        }

        // With stealing disabled, the distribution collapses to the
        // pure static partition: every tid processes *exactly* its
        // seeded share. Any deviation means either the toggle is not
        // honoured or a non-cursor path is invoking the body.
        let expected_share = n / n_workers;
        let mut counts: HashMap<u64, usize> = HashMap::new();
        for o in &owners {
            *counts.entry(o.load(Ordering::Relaxed)).or_insert(0) += 1;
        }
        for (tid, &c) in &counts {
            assert_eq!(
                c, expected_share,
                "no-steal: tid {tid} processed {c} items (expected exactly \
                 {expected_share}). Per-tid counts: {counts:?}"
            );
        }
    }

    /// Stealing also fires when the *stall* is on a worker's slice
    /// rather than main's. Symmetric counterpart to the test above.
    #[test]
    fn stealing_fires_when_a_worker_is_the_straggler() {
        let n_workers = 4;
        let pool = ThreadPool::new(n_workers);
        let n = 4_000usize;
        // Pick an index that lives in participant 1's seeded range
        // `[n/num_threads, 2*n/num_threads)`.
        let straggler_idx = n / n_workers;
        let owners: Vec<StdAtomicU64> = (0..n).map(|_| StdAtomicU64::new(0)).collect();
        let visits: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(0)).collect();

        pool.parallel_for(0..n, |r| {
            for i in r {
                if i == straggler_idx {
                    std::thread::sleep(Duration::from_millis(STRAGGLE_MS));
                }
                owners[i].store(my_tid(), Ordering::Relaxed);
                visits[i].fetch_add(1, Ordering::Relaxed);
            }
        });

        // Exact-once under stealing.
        for (i, v) in visits.iter().enumerate() {
            let c = v.load(Ordering::Relaxed);
            assert_eq!(
                c, 1,
                "item {i} visited {c} times under imbalanced stealing \
                 — cursor CAS allowed overlap or drop"
            );
        }

        assert_steal_skew(&owners, n, n_workers);
    }

    // ---- Nested parallelism ----

    /// A `parallel_for` inside a `parallel_for` body computes the right
    /// answer. Inner dispatches race for whatever workers are idle;
    /// arrivals that find none run caller-only with their range exposed
    /// to scavengers — either way every item must run exactly once.
    #[test]
    fn nested_parallel_for_sums_correctly() {
        let pool = ThreadPool::new(4);
        let inner_n = 1_000usize;
        let total = AtomicUsize::new(0);
        pool.parallel_for(0..16, |outer| {
            for _ in outer {
                pool.parallel_for(0..inner_n, |inner| {
                    let mut local = 0usize;
                    for i in inner {
                        local = local.wrapping_add(i);
                    }
                    total.fetch_add(local, Ordering::Relaxed);
                });
            }
        });
        let expected = 16 * (0..inner_n).sum::<usize>();
        assert_eq!(total.load(Ordering::Relaxed), expected);
    }

    /// Three levels of nesting, repeated back-to-back to shake out slot
    /// reuse and depth bookkeeping.
    #[test]
    fn deeply_nested_dispatches() {
        let pool = ThreadPool::new(4);
        for round in 0..20 {
            let count = AtomicUsize::new(0);
            pool.parallel_for(0..4, |a| {
                for _ in a {
                    pool.parallel_for(0..4, |b| {
                        for _ in b {
                            pool.parallel_for(0..64, |c| {
                                count.fetch_add(c.len(), Ordering::Relaxed);
                            });
                        }
                    });
                }
            });
            assert_eq!(
                count.load(Ordering::Relaxed),
                4 * 4 * 64,
                "round {round}: nested dispatch lost or duplicated items"
            );
        }
    }

    /// Exact-once coverage for a nested dispatch's items, checked with
    /// per-index visit counters instead of a sum.
    #[test]
    fn nested_coverage_exact_once() {
        let pool = ThreadPool::new(8);
        let inner_n = 5_000usize;
        let visits: Vec<Vec<AtomicU8>> = (0..8)
            .map(|_| (0..inner_n).map(|_| AtomicU8::new(0)).collect())
            .collect();
        pool.parallel_for(0..8, |outer| {
            for o in outer {
                pool.parallel_for(0..inner_n, |inner| {
                    for i in inner {
                        visits[o][i].fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        });
        for (o, row) in visits.iter().enumerate() {
            for (i, v) in row.iter().enumerate() {
                let c = v.load(Ordering::Relaxed);
                assert_eq!(c, 1, "outer {o} inner {i} visited {c} times");
            }
        }
    }

    // ---- Background jobs ----

    /// A background task actually runs, promptly, without any
    /// `parallel_for` traffic to shake it loose.
    #[test]
    fn background_task_runs() {
        let pool = ThreadPool::new(4);
        let done = Arc::new(AtomicUsize::new(0));
        let d = done.clone();
        pool.spawn_background(move || {
            d.store(1, Ordering::Release);
        });
        let start = std::time::Instant::now();
        while done.load(Ordering::Acquire) == 0 {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "background task did not run within 5s"
            );
            thread::yield_now();
        }
    }

    /// Background tasks queued while every worker is busy still all run
    /// (the single worker here must drain the queue serially).
    #[test]
    fn many_background_tasks_all_run() {
        let pool = ThreadPool::new(2);
        let count = Arc::new(AtomicUsize::new(0));
        for _ in 0..16 {
            let c = count.clone();
            pool.spawn_background(move || {
                c.fetch_add(1, Ordering::Relaxed);
            });
        }
        let start = std::time::Instant::now();
        while count.load(Ordering::Relaxed) < 16 {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "only {} of 16 background tasks ran within 5s",
                count.load(Ordering::Relaxed)
            );
            thread::yield_now();
        }
    }

    /// The availability heuristic: a worker pinned by a long background
    /// task leaves the idle mask, and `parallel_for` partitions across
    /// the remaining threads instead of waiting for it.
    #[test]
    fn parallel_for_proceeds_while_background_task_blocks_a_worker() {
        let pool = ThreadPool::new(4);
        let release = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(AtomicUsize::new(0));
        let (r, s) = (release.clone(), started.clone());
        pool.spawn_background(move || {
            s.store(1, Ordering::Release);
            while r.load(Ordering::Acquire) == 0 {
                thread::yield_now();
            }
        });
        while started.load(Ordering::Acquire) == 0 {
            thread::yield_now();
        }
        // One worker is pinned; the dispatch must complete on the rest.
        let n = 10_000usize;
        let total = AtomicUsize::new(0);
        pool.parallel_for(0..n, |rge| {
            total.fetch_add(rge.len(), Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), n);
        release.store(1, Ordering::Release);
    }

    /// With *every* worker pinned by background tasks, a dispatch finds
    /// zero available threads and must complete caller-only (the
    /// solo-mode path: job published, own cursor seeded, no members).
    #[test]
    fn parallel_for_completes_with_every_worker_occupied() {
        let pool = ThreadPool::new(4);
        let release = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(AtomicUsize::new(0));
        for _ in 0..3 {
            let (r, s) = (release.clone(), started.clone());
            pool.spawn_background(move || {
                s.fetch_add(1, Ordering::AcqRel);
                while r.load(Ordering::Acquire) == 0 {
                    thread::yield_now();
                }
            });
        }
        while started.load(Ordering::Acquire) < 3 {
            thread::yield_now();
        }
        let n = 50_000usize;
        let total = AtomicUsize::new(0);
        pool.parallel_for(0..n, |r| {
            total.fetch_add(r.len(), Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), n);
        release.store(1, Ordering::Release);
    }
}
