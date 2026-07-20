# Unified scheduler: deterministic bisection work-stealing

Status: **Design — not yet implemented.** Supersedes the "Tier-2 deque" sketch
in [`thread-pool-redesign-plan.md`](thread-pool-redesign-plan.md) §4.5 / Phase 5.

This replaces the current flat `parallel_for` outright (no runtime flag — the
previous scheduler stays available on its own git branch for A/B benchmarking).

Scope: `crates/engine-core/src/util/thread_pool.rs`, plus the asset-loader
integration in `crates/engine-core/src/asset.rs`.

---

## 1. Goals

1. **One scheduler** that hosts three workloads on the same worker threads:
   - per-frame **fork-join** (`parallel_for` — the hot path),
   - **nested** parallelism (`parallel_for` / `join` called from inside a task),
   - **long-running background** jobs (asset decode), fire-and-forget.
2. **Preserve deterministic locality**: thread `i` processes chunk `i` every
   frame, exactly as the current static partition does, so cached/paged data
   stays put. This is non-negotiable — it's the property the whole engine is
   built around.
3. **Wake only as many threads as a job can use**, in log-depth, with no
   thundering-herd `unpark_all` for small jobs.
4. **Tolerate absent workers** (a worker busy on a long background task must not
   stall a concurrent `parallel_for`).
5. No silent fallbacks (project rule); misconfiguration panics.

### Non-goals
- General async / futures. Background jobs are plain `FnOnce`.
- Fairness guarantees for background work. It runs on slack capacity.
- Preempting a running task. Background jobs yield at task boundaries
  (mesh-granular), not mid-task — no fibers.

---

## 2. Why the current scheduler can't host this

Today's `parallel_for` completes when `workers_done == active_workers`: a
worker-**count** barrier that assumes every active worker shows up. A worker off
running a multi-millisecond decode spans *many* dispatches and never increments,
so main would wait forever. Supporting background work therefore requires
replacing the worker-count barrier with **work-completion** signalling — which is
also exactly what nesting needs. That's the core of this redesign.

The insight that makes it cheap *and* locality-preserving: don't adopt rayon's
random-victim stealing (which scatters chunks across threads unpredictably).
Instead distribute work down a **deterministic balanced bisection tree mapped to
the thread topology**, so the steady-state assignment is byte-for-byte the
current static partition, and stealing is only a node-local *repair* mechanism.

---

## 3. Core idea: the bisection tree

`parallel_for(N, f)` is a recursive divide-and-conquer over the chunk range,
where each split hands the right half to a **deterministic** thread — the
midpoint of the responsible thread range — and recurses left on the current
thread.

```
distribute(c_lo, c_hi, t_lo, t_hi, f):   # runs on thread t_lo; synchronous
    if (t_hi - t_lo) <= 1 or (c_hi - c_lo) <= GRAIN:
        run_leaf(c_lo, c_hi, f)          # stealable cursor, NOT a plain loop
        return
    t_mid = t_lo + (t_hi - t_lo) / 2
    c_mid = c_lo + (c_hi - c_lo) * (t_mid - t_lo) / (t_hi - t_lo)   # proportional
    right_latch = Latch::new()
    right = RangeTask { c_mid, c_hi, t_mid, t_hi, &f, &right_latch }
    handoff(right, preferred = t_mid)            # deposit + wake (§5)
    distribute(c_lo, c_mid, t_lo, t_mid, f)      # left, inline
    help_until(right_latch)                       # wait == topology steal (§6)
```

The leaf is **not** a plain `for` loop. It publishes a stealable
`Cursor` (the packed-`AtomicU64` `(start,end)` from the current
scheduler) over `[c_lo, c_hi)` and drains it in `GRAIN` claims via
CAS-the-whole-word. While a thread is draining its leaf, idle peers can
steal the **tail** of that cursor (§6). The bisection tree decides the
*deterministic initial assignment*; the leaf cursors are the *dynamic
rebalancing* mechanism — carried over verbatim from today's tail-steal,
so load imbalance (variable per-task cost, OS preemption) is repaired
exactly as it is now. Dropping the cursor for a plain loop would regress
today's tail-stealing — don't.

Top level: main (thread 0) calls `distribute(0, N, 0, num_threads, f)` and
returns when the root is fully drained — synchronous, same contract as today.

### Why this is exactly the static partition

For `N == num_threads` the proportional split puts `c_mid == t_mid`, so:

- T0 keeps `[0,64)`, hands `[64,128)` → T64;
- T0 keeps `[0,32)`, hands `[32,64)` → T32; T64 keeps `[64,96)`, hands `[96,128)` → T96;
- … after `log2(128)=7` levels, **thread `i` owns chunk `i`.**

Identical to the current contiguous partition, frame after frame → full
locality. For `N != num_threads` the proportional split distributes chunks
evenly across threads; the mapping is still deterministic.

### What falls out for free

- **Wake only what's needed.** Tree depth is `log2(min(chunks, threads))`. A
  4-chunk job wakes 3 threads (depth 2) and stops splitting; a saturating job
  fills the pool in `log2(threads)` wake-latencies. No `unpark_all`.
- **Absent-worker tolerance.** Each handoff carries its own `Latch`; completion
  bubbles up the tree. A busy victim is rerouted (§5) and the latch still fires
  — there is no global worker counter to deadlock.
- **Nesting.** A task calling `parallel_for` roots a fresh tree over its leaf's
  *reserved thread range* (§3a).

### 3a. Nested `parallel_for`

The thread range `[t_lo, t_hi)` carried down the tree is a **capacity
reservation**, and nesting subdivides it recursively. The key is that a leaf
*keeps its full thread range even when it stops splitting early because chunks
ran out*: the leaf condition is `(t_hi - t_lo) <= 1 OR (c_hi - c_lo) <= GRAIN`,
so when threads outlast chunks the leaf retains a **wide** range of idle threads
reserved for it.

A nested `parallel_for(M, g)` is just `distribute(0, M, t_lo, t_hi, g)` where
`[t_lo, t_hi)` is the running leaf's reserved range (held in thread-local state,
set on leaf entry). Two regimes fall out automatically:

- **Outer saturated (`N >= threads`):** every leaf is a single thread
  `[i, i+1)`. A nested call hits `t_hi - t_lo == 1` immediately and runs the
  inner loop **serially on thread `i`** — correct, because the pool is already
  full; spawning would be pure overhead.
- **Outer under-saturated (`N < threads`):** leaves keep wide ranges. A nested
  call recruits the idle threads in its block. Example — `parallel_for(4, f)` on
  128 threads yields four leaves owning `[0,32)`, `[32,64)`, `[64,96)`,
  `[96,128)` (each running one outer chunk). When `f(0)`'s nested
  `parallel_for(M, g)` runs on T0, it roots `distribute(0, M, 0, 32, g)` and fans
  out across the idle threads 0–31.

The reserved ranges are a **disjoint hierarchical partition**, so even if all
four leaves nest simultaneously they recruit non-overlapping thread blocks — no
competition, no double-booking. The only thread present at two levels is the leaf
thread itself, sequentially. Completion is an **independent latch tree** per
`parallel_for` invocation, so the nested `distribute` returns (via its own
`help_until`) before `g`'s caller `f(0)` continues — same synchronous contract as
the outer call, and no deadlock (the outer tree never expects more from the
nesting thread than its leaf, and the recruited block was idle and reserved).

Because each nested tree always uses the same reserved block with the same
midpoint splits, **locality is preserved at every nesting level**, not just the
top — nested-chunk `j` of outer-chunk `0` lands on the same sub-thread every
frame (stable `N`, `M`).

---

## 4. Data structures

Per **thread** (cacheline-padded, `'static` via the leaked `Shared`):

- `incoming: AtomicPtr<RangeTask>` — single-slot mailbox for a handed-off range
  subtree. Producer = the tree-parent (remote); consumers = the owner *or* a
  node-local stealer (both CAS ptr→null to claim). One slot suffices: in the
  happy path a thread receives exactly one handoff per `parallel_for` level.
- `park_key` — sharded `parking_lot_core` key (unchanged from today).
- `status: AtomicU8` ∈ {Parked, Idle, RangeBusy, BackgroundBusy} — drives
  handoff reroute (§5) and sleep management (§7).

Per **NUMA node**:

- `idle: <lock-free stack or bitset of thread ids currently Parked/Idle>` — the
  reroute pool. Handoff claims `t_mid` from it if present, else pops any
  node-local idle thread. (Exact structure — Treiber stack vs atomic bitset — is
  an implementation choice; bitset is appealing because "claim specific
  `t_mid`" is a single bit-clear CAS.)

Global:

- `background: <MPMC queue of Box<dyn FnOnce() + Send + 'static>>` +
  `bg_pending: AtomicUsize`. Asset decodes land here.
- `shutdown: AtomicBool` (unchanged).

`RangeTask` (POD, ~32 B): `{ c_lo:u32, c_hi:u32, t_lo:u16, t_hi:u16,
f: *const (), invoke: unsafe fn(*const(),usize), latch: *const Latch }`. It
lives on the **handing thread's stack frame**, which stays alive because that
thread blocks in `help_until(right_latch)` until the subtree completes — same
lifetime trick as today's `Job`. The mailbox stores a `*const RangeTask` into
that frame.

`Latch`: a one-shot flag. Minimal form is `AtomicBool` + the parking integration
needed by `help_until`. (A counting latch variant is used by `scope`, §9.)

Topology requirement: **threads are numbered NUMA-contiguously** (node 0 =
ids `[0, k)`, node 1 = `[k, 2k)`, …). Then every split below the top is
within one node, so reroutes and "help" stealing stay node-local (§11). The
renderer's pool init already assigns workers per node; it just needs to emit
ids in node-contiguous order.

---

## 5. Handoff + reroute

`handoff(task, preferred)` (steps 2–3 — the reroute — land in Phase 3; see
§14. In pure fork-join, even nested, step 1 always succeeds, so reroute is
inert until background work exists):

1. If `preferred` (== `t_mid`) is in its node's `idle` set, claim it, store
   `&task` into its `incoming` slot (Release), wake it. **Deterministic happy
   path.**
2. Else pop **any** idle thread on `preferred`'s node, deposit + wake it. The
   work lands one core away → an L2/L3 hop, not a remote-DRAM access (§11).
3. Else (no idle thread on the node — the pool is saturated with useful work):
   do **not** spin up anything. The parent processes the right subtree itself
   after its left subtree (`help_until` will pop nothing and fall through to
   running `task` inline). Sequential only for that subtree, only under full
   saturation — acceptable.

Because the `RangeTask` carries its own `(thread range)`, whoever runs it
continues the cascade (waking *their* children). Wake responsibility travels
with the task; there's no separate "transfer the subtree's wake duty" step.

**Why pure fork-join never reroutes (step 1 always hits).** The reserved thread
blocks down the tree are disjoint, so each thread is the target of *exactly one*
handoff, and a thread busy on its own leaf (`RangeBusy`) is never another
handoff's target. Nesting keeps this: a nested `parallel_for`'s targets are the
idle threads inside its leaf's own reserved block (§3a), which got no other
handoff. The *only* thing that makes a target unavailable is a long
`BackgroundBusy` task occupying a block thread — which is precisely why the idle
set + reroute exist, and why they aren't needed before Phase 3.

---

## 6. Help / steal order (locality-preserving wait & rebalance)

`help_until(latch)` is the "waiting == working" loop, and the idle loop (§7)
shares its `find_work` core. Both look for two kinds of stealable work, walked
in **topology order** so the repair phase preserves the cost gradient:

1. an **un-started subtree** (a `RangeTask` sitting in a peer's `incoming`
   mailbox — a handoff target that hasn't woken yet), or
2. the **tail of a peer's in-progress leaf cursor** (§3) — the imbalance-repair
   path, identical to today's tail-steal (CAS the packed `(start,end)` word).

Both are exposed for stealing; `find_work` scans them together, **escalating
outward only when the closer scope is empty**:

```
find_work(self):
    # widen the search ring by ring; stop at the first hit
    for ring in [own_block, parent_block_sibling, rest_of_node, remote_nodes]:
        if t = steal_incoming(ring):   return RangeTask(t)   # un-started subtree
        if t = steal_leaf_tail(ring):  return Chunks(t)      # tail of a running leaf
    return None

help_until(latch):
    while !latch.probe():
        match find_work(self):
            RangeTask(t) => { distribute(t.range, t.threads, t.f); t.latch.set() }
            Chunks(c)    => { run_leaf(c, f) }            # GRAIN claims from stolen tail
            None         => spin/backoff   # owner is awake; don't park inside a help loop
```

**Rings and what they repair:**

- `own_block` (the threads sharing this leaf's `[t_lo, t_hi)`) shares L2/L3 —
  this is where **inner imbalance** (within a nested subtree) is repaired.
- widening to the **parent block's sibling**, then the **rest of the node**,
  repairs **outer imbalance** (unbalanced outer chunks): a block that drains
  early steals a still-busy sibling block's tails — still node-local.
- **remote nodes** are the last resort (remote-DRAM cost), reached only when the
  whole local node is drained but another node is overloaded.

So most steals stay node-local (L2/L3); a cross-node steal happens only under a
genuine global imbalance. The deterministic seed restores each thread's own
chunks next frame, so only the portion a thread truly can't keep up with keeps
migrating — and it stays on-node, so pages don't move. This is exactly today's
tail-stealing behavior, generalized to the topology tree.

**Granularity limit.** A stolen unit is a *chunk* (a `GRAIN` slice of a leaf
cursor). Under full saturation you can take *other* chunks off a slow peer, but
you cannot subdivide a single chunk whose `f(i)` is itself an expensive serial
inner loop — there's no free thread to hand it to. A lone pathologically heavy
chunk at the tail is therefore a tail-latency floor; the mitigation is finer
outer granularity. (To parallelize a giant inner loop you need the pool *not*
saturated, so nesting can recruit — see §3a.)

---

## 7. Idle loop, wake & sleep management

A parked/idle worker's loop:

```
loop:
    if shutdown: return
    if t = try_take_incoming(self):                 # a handoff was deposited for me
        status=RangeBusy; run_range(t); status=Idle; continue
    if w = find_work(self):                          # §6: steal subtree OR leaf tail, topology-ordered
        run(w); continue
    if bg_pending > 0 and t = pop_background():
        status=BackgroundBusy; run_background(t); status=Idle; continue
    idle_staircase()   # spin → sleep(1ns) hrtimer → park on park_key; mark self in node idle set
```

Priority order: **my own handoff → stealable frame work (subtree or leaf tail,
nearest first) → background → sleep.** Frame work always precedes background. A
`BackgroundBusy` worker is out of the idle set, so handoffs reroute around it; it
returns to the loop after its current task (mesh-granular yield).

**Wake.** Handoff wakes its specific target (or reroute target). Background
`spawn` wakes one idle node-local thread. No `unpark_all`.

**Sleep race (the fiddly part).** A worker that finds nothing must not park if
work is in flight. Before parking it re-runs `find_work` (its `incoming` slot +
the node steal rings) and re-checks `bg_pending` under the park `validate`
closure.
This is the classic work-stealing termination problem; we mirror today's
`parking_lot_core::park` validate-recheck. For the structured `parallel_for`
cascade it's mostly moot — a thread only runs when explicitly handed work, and
any *remaining* work has an awake deterministic owner — so the race only really
bites opportunistic background draining, where a missed wake just delays a
background task by one re-check, never deadlocks (main never waits on background).

The steady-state hrtimer trick is retained: between frames workers dwell in
`sleep(1ns)` rather than the park bucket, so wakes stay cheap.

---

## 8. Completion (latch tree replaces `workers_done`)

There is no global `workers_done`. `parallel_for` returns when the **root
subtree** is drained: `distribute` is synchronous on the calling thread and only
returns after its left recursion completes and every `help_until` it issued has
observed its child latch. Completion is thus a property of the latch tree, not a
worker count — which is precisely what makes absent workers harmless.

`DispatchTiming` is recomputed from the tree where still meaningful
(`dispatch`/`main`/`barrier` lose their literal meaning; we keep
`pool-timing`-gated aggregate wake/work stats by sampling per-thread timestamps
as today, or drop the fields that no longer map cleanly — TBD during impl).

---

## 9. `join`, `scope`, and background `spawn`

- `join(a, b)`: degenerate bisection — push `b` as a `RangeTask`-like closure
  task to `t_mid`/reroute, run `a`, `help_until(b.latch)`. Same machinery.
- `scope(|s| …)` + `s.spawn(task)`: a **counting** latch over outstanding
  spawned tasks; scope end = `help_until(count == 0)`. Tasks are boxed closures
  (not POD range tasks) on the background/general queue but with a scope latch
  so the scope can wait.
- `spawn_background(f: FnOnce + Send + 'static)`: box `f`, push to `background`,
  bump `bg_pending`, wake one idle node-local thread. Fire-and-forget; nobody
  waits. This is what the asset loader uses.

---

## 10. Asset-loader integration

Keep a thin blocking-IO thread (the existing `asset-loader`) for `read()` —
blocking IO must **not** occupy a pinned compute worker. Split IO from decode:

```
asset-loader thread:  recv request → read bytes (blocking)
                      → pool.spawn_background(move || { let mesh = decode(bytes);
                                                        registry.resolve(id, mesh); })
```

CPU-bound decode now runs on pool workers in parallel, soaking slack between
frames; the frame loop keeps priority. A single huge mesh decode can itself call
`parallel_for` over vertex ranges (nesting) once that lands. The 30k-node glb
workload (separate feature) is the motivating consumer but is out of scope here.

---

## 11. Locality / NUMA analysis

- **Steady state:** identical to today's static partition (thread `i` ⇄ chunk
  `i`). Full locality, no page migration.
- **Reroute (busy victim):** lands the chunk one core away **within the same
  node** (handoff §5 step 2, guaranteed node-local by contiguous numbering and
  the tree structure — only the *top* split crosses nodes, and that one is
  intentional). Cost ≈ L2 miss → shared-L3 hit; pages do **not** migrate
  off-node even under sustained busy. This is the "off-by-one ⇒ L2/L3 hit,
  self-heals next frame" intuition, made precise.
- **Self-healing:** once the busy thread frees up, the deterministic preference
  routes its chunk back to it next frame.

---

## 12. Safety / lifetimes

- The closure `f` lives on main's stack; main blocks until the root drains, so
  every `RangeTask`'s `*const ()` to `f` stays valid (today's contract).
- Each `RangeTask` lives on its handing thread's frame, kept alive by that
  thread's `help_until` on the corresponding latch.
- `RangeTask`/`Latch` raw pointers crossing threads need the same
  `unsafe impl Send/Sync` newtype discipline as today's `Job`; workers touch
  disjoint chunk ranges so the `f(i)` calls don't alias.
- Background tasks are `'static + Send` and heap-owned by the queue — no
  borrow concerns.

---

## 13. Cost & the A/B baseline

The honest risk: for a **saturating uniform** dispatch the cascade issues
~`threads` staggered handoff/wake atomics down the tree, versus today's single
`unpark_all`. The cascade should win on small/medium N and on staggered (vs
simultaneous) cache traffic, but may lose constant overhead on the big uniform
case. **This must be measured**, which is why the old flat scheduler is retained
on a branch: benchmark both at N = 1, 1K, 100K, 1M (static and animated) on the
2-node box before declaring victory. A regression on the big-uniform hot path is
a blocker; mitigations include a depth cap (stop splitting and let the leaf be
larger) or a hybrid that keeps the flat seed for top-level frame dispatches and
uses the tree only for nesting/background.

---

## 14. Phases

The **per-thread `incoming` mailbox is required from Phase 1** — it is the
handoff channel the bisection is built on, not an optimization. What's deferrable
is the *wake policy* (wake-all → targeted cascade) and the *idle set + reroute*
(needed only once background work can make a handoff target busy; pure fork-join,
even nested, never needs it because the reserved thread blocks are disjoint so
every handoff target is idle at handoff time).

1. **Mailbox + handoff + bisection `distribute` + latch tree**, over **stealable
   leaf cursors** (carry the packed-`AtomicU64` cursor + tail-steal forward from
   the current scheduler) and node-local `find_work` (steal a subtree from a
   peer's mailbox or a peer's leaf tail). **Wake-all** to de-risk: every node
   thread is pre-woken and polls its *own* mailbox first (so distribution stays
   deterministic) then help-steals. No idle set, no reroute, no cascade. This
   validates the recursive distribute, the latch tree, and leaf stealing in
   isolation. Run the existing `parallel_for_*` coverage/stress tests +
   `ENGINE_POOL_VERIFY=1`.
2. **Targeted cascade wake**: per-thread park keys; `handoff` wakes only `t_mid`;
   idle staircase + sleep-race recheck. Identical distribution to Phase 1, only
   `log2` threads wake. Re-run coverage/stress; add a skewed-work test
   (deliberately stall one thread, assert coverage + node-local tail-steal via
   the `find_work` ring escalation).
3. **Background queue + `spawn_background`** + **node idle set + reroute**
   (now a handoff target may be `BackgroundBusy`, so route to another idle
   node-local thread) + status flags. Test: a long background task running
   concurrently with a stream of `parallel_for`s, asserting both complete and
   frame dispatches don't stall.
4. **`join` / `scope`** + nesting (thread-local reserved range) + a
   nested-`parallel_for` test.
5. **Asset-loader repoint** (IO/decode split).

Each phase compiles and passes `cargo test -p engine-core` before the next.

---

## 15. Open questions

- Idle-set structure: atomic bitset (clean "claim specific `t_mid`") vs Treiber
  stack (clean "pop any"). Bitset likely, given the deterministic-preference
  requirement.
- `DispatchTiming` surface under the tree model — keep, reshape, or drop fields.
- Depth cap / leaf granularity tuning vs `STEAL_GRAIN` — one knob or two.
- Whether `scope`'s counting latch and the one-shot `Latch` share an
  implementation.
- Background queue: single global MPMC vs per-node (NUMA-local decode dispatch).
  Start global, revisit if it shows up in profiles.
