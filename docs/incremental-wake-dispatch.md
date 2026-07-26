# Incremental (doubling) wake dispatch — design discussion

Status: **Proposal — not yet designed in detail.** Refines the claiming
policy of the scheduler in
[`unified-scheduler-design.md`](unified-scheduler-design.md) (mostly
implemented today as `crates/engine-core/src/util/my_thread_pool.rs`); does
not replace it.

Scope: `crates/engine-core/src/util/my_thread_pool.rs`
(`ThreadPool::parallel_for`, `claim_workers`, `wake_children`, `PaddedCursor`
/ `try_steal`).

---

## 1. Motivation

Today's `parallel_for` (my_thread_pool.rs:838) claims workers **eagerly**:

> "Claims up to `min(len - 1, idle)` workers, splits the range into one
> contiguous slice per participant, and blocks (helping) until every item has
> run." (my_thread_pool.rs:830-832)

`claim_workers` (L378) grabs that many idle workers in one pass and
`wake_children` (L523) wakes all of them via a binary tree of targeted
`unpark`s. The wake-tree makes the *fan-out* cheap (log-depth, not one thread
issuing N syscalls serially), but it does not change *how many* workers get
woken: a 100-item job on a 10-idle-thread pool claims and wakes all 10
threads immediately, no matter how little work each item actually is.

If waking a thread costs ~50us (syscall + scheduler latency) but the
per-thread work is done in ~10us, we've paid for 10 wakeups to save work that
a couple of threads could have finished before the other 8 even got
scheduled. The wake-tree is real value (avoids serial unparks), but it's
solving the *notify* cost, not the *how-many* cost.

## 2. Proposed change

Replace eager `claim_workers`-for-everything with an **incremental,
doubling** ramp, driven by a dispatcher that does not itself run job slices:

1. Dispatcher hands worker 0 the **full** range immediately (no split yet).
   Worker 0 starts draining its cursor right away — no one waits on the
   ramp-up decision.
2. Dispatcher loop: while the job isn't done and there are unwoken candidate
   workers, wake one more, doubling the number of *active* workers each
   iteration (1 → 2 → 4 → 8 → ...).
3. Each newly woken worker is handed the **back half of a donor's remaining
   range** (donor = an already-active worker, read live off its
   `PaddedCursor`, not a static a-priori split) — this reuses the existing
   back-half CAS in `try_steal` (L180) almost verbatim; the difference is the
   *dispatcher* is initiating the split preemptively, not a peer stealing
   reactively after going idle.
4. **Forward/backward alternation**: a donor iterating forward keeps working
   from the front of its range; the new worker takes the back half and
   iterates *backward* (toward the donor). This is not in the current
   scheduler at all — today's `try_steal` always takes the back half but the
   thief still walks it front-to-back. The goal here is that two threads
   converging on the same original range approach each other and finish
   near the same midpoint, rather than one thread's forward walk crossing
   into a region a third thread just stole out from under a second thread
   (the `0,1,2,*steal*,6,7,8` cache-eviction pattern from the original
   discussion).

## 3. Design decisions

- **No dedicated dispatcher thread.** Resolved: the calling thread drives
  the doubling ramp itself (today's "external dispatcher" / participant-0
  path, my_thread_pool.rs:867-879). It's already the thread that would
  otherwise sit blocked waiting for completion, so there's no need to burn
  a standing core on a thread whose only job is bookkeeping.
- **Worker 0 gets an elevated spin threshold, not infinite spin.** Worker 0
  is the target of *every* dispatch's first handoff, so it's the
  highest-value thread to keep hot: if it's already spinning (not parked)
  when the calling thread hands it the full range, that first handoff skips
  the park/unpark round trip entirely, which matters because it's on the
  latency-critical path of every single call. Give worker 0 a much higher
  `SPIN_ITERS`-equivalent than other workers (which spin briefly then park
  per the existing ladder, L58) — but not unconditional/infinite spin: if
  the pool goes genuinely idle for long enough (e.g. a stretch where every
  `parallel_for` call is small enough to hit today's `size <= 1` inline
  fast path at L852-858 and never touches the pool at all), worker 0 should
  still fall through to `park()` like any other worker, just after a much
  longer dwell. This avoids permanently burning a core during genuinely
  idle periods while paying near-zero wake latency during active periods.
  Exact threshold (and whether it should reset/extend on each real
  dispatch vs. decay on a fixed schedule) is still open — start with a
  constant multiple of `SPIN_ITERS` and tune from benchmarks (§6 below).
- **Ramp timing.** "Double every iteration" needs a trigger: is it time-based
  (wake the next tier if the job isn't done after N microseconds), or
  progress-based (wake the next tier once the current tier's cursors are
  each below some remaining-work threshold)? Time-based is simpler but needs
  tuning per workload; progress-based is self-scaling but needs a cheap way
  to read aggregate remaining work across active cursors.
- **Interaction with existing nested/background machinery.** `my_thread_pool.rs`
  already has nested-dispatch depth tracking (`MAX_DEPTH`, `depth_hwm`) and a
  job-slot table for background/nested scavenging (`JobSlot`, `alloc_slot`,
  `steal_sweep`). The doubling ramp needs to coexist with a worker that's
  mid-scavenge on an unrelated job when it gets woken for this one.
- **Where the line is vs. today's reactive stealing.** Today, an idle worker
  that finishes its own slice sweeps the job table and steals half of
  someone else's remaining range (`steal_sweep`, L451) — this is already
  "wake more capacity only if needed," just triggered by a *thief going
  idle* rather than a *dispatcher timer/threshold*. Confirm the doubling
  ramp is solving a case steal-sweep doesn't already cover (it should be:
  steal-sweep only helps once a worker is both awake *and* out of its own
  work, so for very short jobs with few active workers, the other N-k
  workers are never woken at all today — correct, that's the actual gap).

## 4. Relationship to existing docs

- [`thread-pool-redesign-plan.md`](thread-pool-redesign-plan.md) — the
  original NUMA-simplification plan; largely landed (phases 1-4).
- [`unified-scheduler-design.md`](unified-scheduler-design.md) — the
  bisection-tree / targeted-wake / nested-job design; `my_thread_pool.rs` is
  effectively this design implemented (wake tree, stealable packed cursors,
  job-slot table for nesting + background). This document's proposal is a
  further refinement of that scheduler's **claiming policy**, not a new
  scheduler.

## 5. Todo

1. Design the doubling wake-loop precisely: split logic, direction
   alternation, termination condition, and how "remaining work" is tracked
   per worker for splitting.
2. ~~Audit current implementation~~ — done, see §1-2 above and
   my_thread_pool.rs:378 (`claim_workers`), :523 (`wake_children`),
   :145-180 (`PaddedCursor`/`try_steal`).
3. Prototype the ramp: calling thread drives the doubling loop (no
   dedicated dispatcher thread, per §3); worker 0 gets an elevated spin
   threshold so the first handoff of every dispatch avoids park/unpark
   latency, while still parking after a long-enough genuinely-idle stretch.
4. Implement forward/backward alternating range splits on top of the
   existing back-half CAS.
5. Replace/remove the eager `claim_workers`-for-everything path once the
   ramp is validated.
6. Benchmark against current `my_thread_pool` across job sizes and per-task
   costs (short ~10us tasks vs. longer), per §13 of
   `unified-scheduler-design.md`'s A/B methodology (keep the old scheduler
   on a branch, compare at N = 1, 1K, 100K, 1M).
7. Handle edge cases: fewer tasks than threads, doubling exceeding available
   threads, job completing mid-ramp, workers woken with zero remaining work.
