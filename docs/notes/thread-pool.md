# Work-stealing thread pool

The scheduler every `parallel_for` in the engine runs on, and what the two
older pools still in the tree are for.

**Simple work-stealing pool, initialised at startup.** `Window::run` calls
`init_pinned_thread_pool` before constructing the winit event loop.
`ENGINE_NUM_THREADS` (or `RAYON_NUM_THREADS`) sets the **total** participant
count including the external/main caller; the pool receives `(total - 1)`
worker threads. `ENGINE_NO_PIN=1` is accepted but currently a no-op (the
simple pool does not pin); the flag is preserved on the CLI for a future
pinning-capable scheduler. Per project rules, bad configuration panics rather
than silently falling back.

The active scheduler is
[`engine_core::util::my_thread_pool`](../../crates/engine-core/src/util/my_thread_pool.rs):
a deliberately minimal work-stealing fork-join pool (~350 lines including
tests). Key properties:

* **`crossbeam_deque` per-worker LIFO + shared injector.** Each worker owns a
  `Worker<Task>` (LIFO) and exposes a `Stealer` to peers. External callers
  (including the main thread) push into a shared `Injector`. Worker threads
  run a tight loop: own deque → injector → rotate through peer stealers →
  spin/yield/park-with-timeout.
* **`parallel_for` splits into one contiguous chunk per worker.** The body is
  captured by reference, exposed to tasks as a thin `*const ()` plus a
  monomorphised `call_body::<F>` function pointer (no `dyn Trait`, no
  `'static` bound on `F`). The dispatching thread blocks in `help_until`
  (work-first — it pops/steals while waiting), guaranteeing the body outlives
  every dereference.
* **Nested parallelism.** A worker that calls `parallel_for` inside a task
  pushes the sub-tasks onto its own deque; `help_until` then pops LIFO
  (depth-first) so children run before peers steal them. Idle workers steal
  across deques to load-balance imbalance.
* **Background tasks.** `spawn_background(f)` enqueues a single long-running
  job onto the current worker's deque (or the injector); a worker picks it up
  and stays in it. Remaining workers continue to service `parallel_for`
  dispatches.
* **Panics propagate.** Each chunk task wraps the user closure in
  `catch_unwind`; the *first* panic payload is stored, `pending` is still
  decremented, and the dispatching thread re-raises with `resume_unwind` after
  the dispatch drains. No silent failures, no permanent deadlocks.
* **Lifecycle is explicit.** `my_thread_pool::global::init(n)` must be called
  once at startup. `pool()` panics if invoked before init (no auto-default).

The older `numa_pool` (NUMA-aware, pinning, epoch-directed slots) and
`thread_pool` (legacy static partitioner) source files are still in the tree
under `engine-core::util` but are no longer wired into the engine's
`parallel_for` callsites — they remain as references for the next iteration
that wants pinning + NUMA placement.

