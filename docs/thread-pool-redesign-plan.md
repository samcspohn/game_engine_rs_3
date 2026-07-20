# Thread-pool & NUMA simplification — design / migration plan

Status: **Phases 1–4 landed** (containers → `Vec`, unified `parallel_for`,
relays removed, main unpinned + `ENGINE_NO_PIN`). Phase 5 (nesting + background
tasks) is **redesigned** — see
[`unified-scheduler-design.md`](unified-scheduler-design.md), which supersedes
§4.5 below.

> **Implementation note (cursor packing).** During Phase 2 a latent
> double-execution race in the steal engine surfaced: the original `Cursor`
> stored `start` and `end` as two separate atomics, so an owner advancing
> `start` (reading `end` separately) and a thief lowering `end` (reading
> `start` separately) could claim overlapping ranges at the boundary. It was
> never caught before because the cursor/steal path (old `parallel_for_numa`)
> was only used on 2-node machines; single-node test/dev boxes took the
> uniform-chunk `parallel_for_global` path. Fix: pack `(start, end)` into a
> single `AtomicU64` and CAS the whole word in both owner-claim and steal, so
> the two ends can never disagree. Verified by the existing
> `parallel_for_*` coverage/stress tests (which now exercise stealing) plus
> `ENGINE_POOL_VERIFY=1`.

Scope:
- `crates/engine-core/src/util/thread_pool.rs`
- `crates/engine-core/src/util/numa_soa.rs` (to be removed)
- `crates/engine-core/src/util/numa_mem.rs` (trimmed)
- `crates/engine-core/src/util/numa.rs` (kept for pinning topology only)
- `crates/engine-core/src/component/mod.rs` (`ComponentStorage`)
- `crates/engine-core/src/transform/mod.rs` (`TransformHierarchy`, `Dirty`)
- `crates/engine-render/src/lib.rs` (`init_pinned_thread_pool`, per-frame harvest)
- `crates/engine-render/src/transform_gpu.rs` (staging `MempolicyGuard`)

---

## 1. Motivation

The current pool carries a large amount of NUMA-specific machinery that, in
practice, has **not** earned its complexity:

- **NUMA-aware containers** (`NumaSoa<T>`) reserve a virtual range and
  `mbind`/`set_mempolicy` each half to a specific node. This complicates every
  storage type (`ComponentStorage`, `TransformHierarchy`, `Dirty`) and forces a
  hard-coded `entity_split` and a `num_nodes ∈ {1, 2}` restriction.
- **`parallel_for_numa`** dispatches per-node task partitions, requiring the
  `NumaPartitioned` trait, per-node cursor seeding, an `active_nodes_mask`, and
  a node-aware steal order.
- **Relay workers** — one always-spinning worker per non-main node — plus
  `node_epoch` republishing, `relay_served_mask`, and sharded park keys exist
  solely to avoid cross-NUMA wake syscalls.

The observed reality:

1. Letting the OS place pages (first-touch + the kernel's automatic NUMA
   balancing / page migration) is **adequate**. The decisive factor for
   locality is not *where we pre-bind pages* but that **each worker repeatedly
   processes the same chunk of the index space every frame**, so the data it
   touched last frame is already in its cache (and its pages drift to its node
   on their own).
2. The **chunk + tail-steal** dispatch pattern (each participant owns a
   contiguous slice it drains atomically, then steals the back half of a
   neighbour's slice when it runs dry) is the proven optimum for these uniform
   per-frame workloads. This stays.
3. The NUMA containers and relay infrastructure are pure cost: more code, more
   `unsafe`, more env knobs, hard node-count caps — without a measured win over
   "std `Vec` + stable chunk assignment + OS page placement".

We also have two pinning/thermal problems and one missing feature:

- The **main thread is pinned to core 0** and spins on the barrier. A pinned +
  spinning thread heats its core, the core throttles, and the OS is **not
  allowed** to migrate it to a cooler core. Main should be free to migrate.
- There is **no way to disable pinning** entirely (useful for laptops,
  containers, shared CI boxes, and thermal/throughput experiments).
- There is **no nested parallelism**: `parallel_for*` panics when called from a
  worker. For feature completeness we want a job-based API that nests.

---

## 2. Goals

1. Replace all `NumaSoa<T>` fields with plain `Vec<T>` (or `Box<[T]>`). Delete
   `numa_soa.rs` and the `NumaPartitioned` trait.
2. Collapse `parallel_for_global` + `parallel_for_numa` into **one** primitive
   that uses the chunk + tail-steal pattern with **stable, deterministic
   chunk→participant assignment** across calls (preserving the cache/page
   locality that actually matters).
3. Remove relay workers, `node_epoch`, `relay_served_mask`, and per-node wake
   special-casing. One simple wake path.
4. Stop pinning the main thread; let it migrate. Keep worker pinning, but make
   **all pinning opt-out** via an env variable.
5. Add a **job-based API with nested parallelism** (`join` / `scope` /
   `spawn`), so `parallel_for` can be called recursively from inside a worker.
6. Keep the project rule: **no silent fallbacks**. Misconfiguration panics.

### Non-goals

- We are *not* removing the ability to know NUMA topology for pinning
  (`numa.rs` stays — it enumerates nodes/CPUs for affinity).
- We are *not* removing `numa_mem.rs` wholesale. The renderer still uses
  `MempolicyGuard` to influence *driver-internal* DMA staging allocations
  (`transform_gpu.rs::allocate_staging`); that is a genuinely different problem
  (GPU driver `mmap`s we can't first-touch) and stays. The `mbind_to_node` /
  `page_residency` helpers that only existed to support `NumaSoa` can go.

---

## 3. What exists today (baseline)

```
ThreadPool
├── parallel_for_global(n_tasks, f)     uniform chunk, no stealing
│     worker i runs [i*chunk, (i+1)*chunk)
├── parallel_for_numa(partitions, f)    per-node cursors + tail-steal
│     NumaPartitioned-derived ranges, active_nodes_mask, steal_order
└── worker_loop
      ├── relay workers: pure spin, republish node_epoch, node-local unpark_all
      ├── normal workers: spin → sleep(1ns) → parking_lot park (sharded key)
      └── reads Job via epoch Release/Acquire; barrier on workers_done

Containers
├── NumaSoa<T>         mmap + mbind per half, entity_split, num_nodes∈{1,2}
├── ComponentStorage   data/active are NumaSoa
├── TransformHierarchy positions/rotations/scales are NumaSoa
└── Dirty              4× NumaSoa<AtomicU32>

Init (engine-render::init_pinned_thread_pool)
├── probe NumaTopology, intersect with allowed cpuset
├── pin main to node-0 core 0
├── round-robin workers across nodes, build per-worker affinity masks
└── init_global(PoolConfig{workers, main_node, num_nodes}, on_worker_start)
```

Call sites that must keep working:
- `ComponentStorage::par_iter` (component update walk)
- `TransformHierarchy` dirty harvest (`engine-render/src/lib.rs::about_to_wait`)
- `build_all_frame_slots`, `Container::for_each`, `for_chunks`
- `Scene::update` → component updates

---

## 4. New design

### 4.1 Containers: `NumaSoa<T>` → `Vec<T>`

- `ComponentStorage::data: Vec<MaybeUninit<Mutex<T>>>`,
  `active: Vec<AtomicU32>`. Allocate once at `with_layout`, fill to capacity
  (`resize_with`). Drop logic in `ComponentStorage::drop` stays (walk active
  bits, drop live `Mutex<T>` slots) — only the backing accessor changes from
  `NumaSoa::get_unchecked` to slice indexing.
- `TransformHierarchy::{positions,rotations,scales}: Vec<SyncUnsafeCell<...>>`.
- `Dirty`: four `Vec<AtomicU32>`.
- Delete `entity_split` plumbing, `num_nodes` fields on storage,
  `with_split*`, `numa_partitions()`, the `NumaPartitioned` trait, and the
  `num_nodes ∈ {1,2}` asserts.
- `ENGINE_NUMA_NODES` env handling in `TransformHierarchy::new` is removed (or
  kept only as a no-op that warns), since storage is no longer node-split.

Rationale: page placement is delegated to the OS (first-touch by whichever
worker first writes a slot + kernel auto-balancing). Because chunk→worker
assignment is **stable** (see 4.2), the worker that first-touches a slot is the
same worker that revisits it each frame, so pages naturally settle on that
worker's node.

### 4.2 One dispatch primitive: chunk + tail-steal

Replace both `parallel_for_*` with a single:

```rust
pub fn parallel_for<F: Fn(usize) + Sync>(&self, n_tasks: usize, f: F) -> DispatchTiming
```

Behaviour:
- `P = num_participants` (workers + main). Split `0..n_tasks` into `P`
  contiguous slices; participant `p` is **seeded** with cursor `p` covering its
  slice. **Deterministic**: participant `p` always owns the `p`-th slice for a
  given `n_tasks`, frame after frame → stable locality.
- Each participant drains its own cursor in `STEAL_GRAIN` chunks via
  `start.fetch_add` (existing `Cursor` / `try_steal` logic, unchanged).
- When a participant's own cursor is empty, it steals the back half of a
  neighbour's cursor. **Steal order is a simple rotation** `p+1, p+2, …`
  (wrap-around) — no NUMA grouping. (Optional: keep node-affinity-first steal
  order behind the topology we already have, but default to plain rotation for
  simplicity. Decide during impl; plain rotation is the simpler default.)
- Main participates inline, then barriers on `workers_done == active_workers`.

This is exactly today's `parallel_for_numa` cursor engine, minus the per-node
partitioning. The `n_tasks <= 1` fast path and verify-mode accounting carry
over unchanged.

`for_chunks`, `Container::for_each`, `build_all_frame_slots`,
`ComponentStorage::par_iter`, and the dirty harvest all call this single
primitive. The renderer's `use_numa` branch in `about_to_wait` collapses to one
path.

### 4.3 Idle / wake policy without relays

Remove relays entirely. Single wake path:

- Workers: spin (`HOT_SPIN_ITERS`) → `sleep(1ns)` hrtimer dwell
  (`PARK_AFTER_SPINS`) → `parking_lot_core::park` on a sharded key. **Keep park
  sharding** (`ENGINE_PARK_SHARD_SIZE`) — it is cheap and still helps at high
  thread counts — but drop `node_shard_range`'s relay coupling; sharding is now
  purely "spread N waiters across M buckets", node-agnostic (shard by global
  worker index).
- Dispatch wakes via one `unpark_all` per shard key. No `wake_workers_on_node`,
  no `relay_served_mask` skip, no `node_epoch` republish.
- Delete `is_relay`, `node_epoch`, `relay_served_mask`, `node_shard_range`'s
  per-node semantics, and the relay branches in `worker_loop`.
- `init_global`'s `on_worker_start` signature drops the `is_relay: bool`
  argument → `Fn(usize)`.

### 4.4 Pinning policy + opt-out; main not pinned

In `engine-render::init_pinned_thread_pool`:

- **Do not pin the main thread.** Remove the `set_for_current(main_core)` call.
  Main spins on the barrier; leaving it unpinned lets the OS migrate it off a
  hot/throttled core. We still *record* `main_node` only if we keep any
  node-aware steal ordering; with plain-rotation stealing we can drop
  `main_node` entirely.
- **`ENGINE_NO_PIN` (or `ENGINE_PIN=0`)** env var disables *all* affinity
  setting: workers spawn without `set_current_thread_affinity_mask`. Strictly
  parsed (`0`/`1`/`true`/`false`); anything else panics (no fallback).
- When pinning is enabled, workers keep the existing block-affinity mask
  (`ENGINE_AFFINITY_BLOCK`) so the OS can migrate within a block. Worker→CPU
  assignment can stay topology-aware for cache locality, but the pool itself no
  longer needs `WorkerSpec.node` — `PoolConfig` shrinks to
  `{ num_workers, on_worker_start, no_pin }` (topology handling stays in the
  renderer's init, not the pool).

`PoolConfig` after the change (illustrative):

```rust
pub struct PoolConfig {
    pub num_workers: usize,
}
// pinning is entirely the caller's job inside on_worker_start;
// the pool no longer knows about nodes.
```

### 4.5 Job-based system with nested parallelism

> **Superseded.** The two-tier sketch below has been replaced by a single
> unified scheduler — see
> [`unified-scheduler-design.md`](unified-scheduler-design.md) (deterministic
> bisection work-stealing). That design keeps the deterministic chunk→thread
> locality of Tier 1 *and* hosts nesting + long-running background tasks on one
> mechanism, instead of bolting a separate Tier-2 deque alongside the flat
> dispatch. The text below is retained for history.

Add a job layer so parallelism can nest. Two-tier design:

**Tier 1 — top-level flat dispatch (hot path, from main):** the chunk +
tail-steal `parallel_for` from 4.2. Deterministic assignment, full pool
barrier. This is what the per-frame harvest and component update use.

**Tier 2 — nested job API (feature completeness):** for calls *from inside a
worker* (or from main when recursion is desired):

```rust
pub fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where A: FnOnce() -> RA + Send, B: FnOnce() -> RB + Send;

pub fn scope<'s, F>(f: F) where F: FnOnce(&Scope<'s>);
//   scope.spawn(|s| { ... });  // fork tasks, joined at scope end
```

Implementation sketch:
- Each worker owns a **Chase–Lev-style work-stealing deque** of boxed tasks
  (LIFO local pop, FIFO steal). A task is a `FnOnce()` + a pointer to a join
  latch (`AtomicUsize` countdown).
- `join(a, b)`: push `b` onto the current worker's deque, run `a` inline, then
  try to pop `b` back; if a thief took it, help-steal/run other tasks until
  `b`'s latch fires. Classic work-first scheduling. Works on main too (main
  becomes a transient participant in the deque scheduler).
- `parallel_for` called **from a worker** (currently a panic) routes to the
  Tier-2 path: split the range into tasks pushed to the local deque and
  block-help until the latch completes — instead of asserting `!is_worker()`.
- The worker idle loop gains a third source of work: after observing no Tier-1
  epoch, a parked/spinning worker also probes peers' deques for stealable
  nested tasks. (Care: keep Tier-1 the fast path; only fall into deque-steal
  when a nested scope is actually live, gated by a global "nested jobs
  outstanding" counter to avoid steal-scan overhead in the common no-nesting
  case.)

This is the most involved piece; it can land **after** the
simplification (4.1–4.4) so the risky refactor and the new feature are
separable. See phasing below. Nested parallelism is explicitly *for
completeness*, not the hot path, so the simple "help until latch" model is
acceptable even if it doesn't preserve per-chunk locality the way Tier-1 does.

---

## 5. Public API delta

| Before | After |
|--------|-------|
| `parallel_for_global(n, f)` | `parallel_for(n, f)` |
| `parallel_for_numa(parts, f)` | *(removed; callers use `parallel_for`)* |
| `NumaPartitioned` trait | *(removed)* |
| `init_global(PoolConfig{workers, main_node, num_nodes}, Fn(usize,bool))` | `init_global(PoolConfig{num_workers}, Fn(usize))` |
| `ThreadPool::{num_nodes, main_node, worker_node, workers_on_node, is_relay_worker}` | *(removed)* |
| `TransformHierarchy::num_numa_nodes()` | *(removed)* |
| — | `join`, `scope`/`Scope::spawn` (new, Tier 2) |
| `NumaSoa<T>`, `numa_soa.rs` | *(removed)* |
| `numa_mem::{mbind_to_node, mbind_policy_to_node, page_residency, verify_residency_single_node}` | review; keep only what `MempolicyGuard` path needs |

Kept: `MempolicyGuard` (renderer staging), `numa.rs` topology probe (pinning),
`DispatchTiming`, `bitmap_task_layout`, `for_chunks`, verify mode,
`pool-timing` feature.

---

## 6. Env variables

| Var | Status | Meaning |
|-----|--------|---------|
| `ENGINE_NUM_THREADS` / `RAYON_NUM_THREADS` | kept | total participants incl. main |
| `ENGINE_NO_PIN` (new) | new | `1`/`true` → skip *all* affinity pinning (main already unpinned) |
| `ENGINE_AFFINITY_BLOCK` | kept | worker block-affinity width (ignored if `ENGINE_NO_PIN`) |
| `ENGINE_PARK_SHARD_SIZE` | kept | waiters per parking_lot bucket (now node-agnostic) |
| `ENGINE_POOL_VERIFY` | kept | per-task invocation accounting |
| `ENGINE_DISABLE_RELAY` | removed | relays gone |
| `ENGINE_NUMA_NODES` | removed | storage no longer node-split |

All new/kept vars are parsed strictly — bad values panic (no fallback).

---

## 7. Phasing

Land in independently-reviewable, independently-revertable steps. Each phase
must compile and pass `cargo test -p engine-core` before the next.

**Phase 0 — prep / safety net.**
Confirm `ENGINE_POOL_VERIFY=1` passes on current `main` for the workloads we'll
touch. Note baseline FPS from the stress benchmark for later comparison.

**Phase 1 — containers to `Vec`.**
Swap `NumaSoa` → `Vec` in `ComponentStorage`, `TransformHierarchy`, `Dirty`.
Keep `parallel_for_global` for now; route the old NUMA harvest path to
`parallel_for_global` (delete the `use_numa` branch in `about_to_wait`). Delete
`numa_soa.rs`, `NumaPartitioned`, `num_numa_nodes`, `entity_split` plumbing.
*Risk: medium (touches storage + drop logic). Verify with existing
`par_iter_*` tests + `ENGINE_POOL_VERIFY=1`.*

**Phase 2 — unify dispatch on chunk+steal.**
Rename `parallel_for_numa`'s cursor engine into the new flat `parallel_for`
(deterministic per-participant seeding + rotation steal). Delete
`parallel_for_global` and `parallel_for_numa`. Update all call sites.
*Risk: medium. The cursor engine already exists and is tested; the change is
the seeding (flat instead of per-node) and steal order.*

**Phase 3 — remove relays + simplify wake.**
Delete `is_relay`, `node_epoch`, `relay_served_mask`, relay branches in
`worker_loop`, `wake_workers_on_node`. Park sharding becomes node-agnostic.
`on_worker_start` loses the `is_relay` arg. `PoolConfig` drops `main_node` /
`num_nodes` / per-worker `node`.
*Risk: medium (worker loop correctness). Stress-test wake latency.*

**Phase 4 — pinning policy.**
Stop pinning main; add `ENGINE_NO_PIN`. Simplify `init_pinned_thread_pool`
(topology used only for optional worker affinity blocks; otherwise skipped).
Update the README "Static thread pool, pinned at startup" paragraph.
*Risk: low. Mostly init-side.*

**Phase 5 — nested job API (`join`/`scope`).**
Add the Tier-2 deque scheduler; make `parallel_for` from a worker route to it
instead of panicking. Add tests for nesting (e.g. recursive `join` sum, nested
`parallel_for`). This is additive and can be deferred / split out.
*Risk: high (new scheduler). Strongly gate behind tests; keep Tier-1 hot path
untouched when no nested scope is live.*

After each of phases 1–4 the engine is fully functional and faster-to-reason
about; phase 5 is the only net-new feature.

---

## 8. Testing & validation

- `cargo test -p engine-core` — existing `thread_pool` and `component` tests
  (coverage / no-drop / b2b / stress) must pass after every phase.
- `ENGINE_POOL_VERIFY=1` runs of the stress benchmark each phase.
- New tests:
  - `parallel_for` deterministic seeding (participant p sees its slice first).
  - tail-steal still achieves full coverage with a deliberately-skewed `f`.
  - `ENGINE_NO_PIN=1` boots and runs.
  - Phase 5: nested `join`/`scope` coverage + a `parallel_for`-inside-worker
    test.
- Stress-benchmark FPS comparison vs. the Phase-0 baseline at N = 1, 1M static,
  1M animated. The simplification should be **neutral-to-positive** at all N;
  any regression > a few % is a blocker and a signal we lost a locality
  property (revisit steal order / seeding).

---

## 9. Risks & open questions

- **Locality without `mbind`.** Thesis: stable chunk→worker seeding + OS
  auto-NUMA balancing ≈ explicit binding. Must be validated on the 2-node box
  (Phase 1/2 FPS check). If a regression appears, the cheapest mitigation is to
  keep node-affinity-first steal order (we still have `numa.rs`) without
  reintroducing `NumaSoa`.
- **Steal order.** Plain rotation is simplest; node-first rotation is a
  one-line richer alternative if locality needs it. Decide via benchmark.
- **Nested scheduler complexity.** Phase 5 is genuinely complex (Chase–Lev
  deque, latch help-loop). It's isolated and optional; do not let it block
  phases 1–4. Consider whether `parallel_for`-from-worker should remain a panic
  short-term (documented) if `join`/`scope` cover the real need.
- **Main thread migration vs. cache.** Unpinning main trades a little L1/L2
  locality for thermal headroom; expected win on turbo-bound low-N, neutral at
  high N. Confirm with the N=1 benchmark (historically the thermal-sensitive
  case).
- **`numa_mem` surface.** Confirm exactly which functions the renderer's
  `MempolicyGuard` staging path needs before deleting the `mbind_to_node` /
  residency helpers.

---

## 10. README updates required (per project rules)

- "Static thread pool, pinned at startup" paragraph: main is **no longer
  pinned**; add `ENGINE_NO_PIN`.
- "Static range partitioning — no work-stealing" bullet: now **chunk +
  tail-steal with deterministic per-participant seeding**.
- "No nested parallelism" bullet: update once Phase 5 lands (job-based
  `join`/`scope`, nested `parallel_for`).
- Remove NUMA-container / `parallel_for_numa` / relay references.

When the work lands, also consider a short ADR (per `docs/ADR-INDEX.md`
criteria — this locks in a structural pattern) summarising the decision to drop
NUMA containers in favour of OS page placement + stable chunk assignment.
