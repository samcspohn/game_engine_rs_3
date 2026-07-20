# Render thread + sparse compact staging — deferred

This document captures a design idea that was discussed and validated
during the NUMA / thread-pool stability sessions but **deferred** to
prioritise tail-latency / variance work first. Pick this back up after
the current variance investigation lands.

## Motivation

The current pipeline is fully synchronous inside a frame:

```
sim_update (parallel_for_numa) → host_staging (parallel_for_numa,
both nodes write directly into the WC/ReBAR staging buffer) → submit
→ host_wait_compute
```

Two observations from benchmarking:

1. With `numactl --cpunodebind=1 --membind=1` (everything on node 1,
   crossing the inter-socket fabric for every staging store), we still
   measure ~880 FPS / ~770 µs `host_staging` parallel time. So the
   cross-socket write itself is **not** the dominant bottleneck for
   the writer side — the inter-socket fabric has enough bandwidth.
2. Single-threaded host_staging is ~4.2 ms; 128-thread is ~750 µs.
   The 5-6× speedup tells us that scaling threads on the staging pass
   is genuinely useful (PCIe write fan-in benefits from multiple
   write-combining buffers).

This suggests the *current* topology is acceptable, but the
opportunity is **overlap**: today the sim threads sit idle during
host_staging and host_wait_compute. If staging is hoisted to a
background thread (or sub-pool), sim threads can begin the next frame
as soon as their per-entity reads are done.

## Design sketch

### Sparse-delta producer (sim threads)

During sim (or harvest), each sim worker writes its dirty entities as
**(entity_id, payload)** pairs to a per-worker (or per-node)
*compact* delta buffer in DRAM:

```rust
struct Delta {
    entity_id: u32,
    payload: [f32; N],  // pos, or rot, or scale
}
```

Properties:
- Pure DRAM writes (no PCIe), so node-local first-touch keeps each
  worker's deltas on its own node.
- The `(entity_id, payload)` form encodes "dirty" implicitly — if a
  pair exists, the entity is dirty. The current per-bitfield
  dirty-bit pass is no longer needed.
- Memory cost: ~50% more than current "always-dense" if dirty rate
  is high, much less if sparse.
- We can keep separate streams per field (positions, rotations,
  scales) so we don't write zero-payload when only one field changed.

### Render thread (background scatter)

A dedicated render thread (or a small "render sub-pool" pinned to
node 0) consumes the per-worker delta vecs and scatters them into
the WC/ReBAR staging triple:

```
for each Delta { staging.position[d.entity_id] = d.payload; }
```

This is itself parallelisable: shard delta vecs across multiple
render-pool workers; use `parallel_for_global` (uniform-chunk static
partitioning) over the union of delta vecs. **The PCIe writer side
benefits from more threads, not fewer** (multiple WC fill buffers).

### Pipeline view

```
Frame N:    sim_update → publish deltas
                ↓ (deltas ready)
Frame N:                   scatter to staging → submit → host_wait_compute
                ↓ (next frame begins on sim threads)
Frame N+1:  sim_update → ...
```

The sim threads block only on **render thread done** at the start of
frame N+1, not on staging or GPU. As long as scatter+submit fits in
sim's window, sim is never blocked.

## Open questions

1. **Two scatter passes vs one.** Current pipeline does
   (cpu→staging) + (gpu scatter into the SoT in VRAM). With deltas
   we can either:
   - Keep two passes: cpu re-densifies into staging, gpu scatter
     unchanged. Pro: zero gpu-side change. Con: cpu does the
     re-scatter work too.
   - Single pass: change the gpu scatter kernel to consume
     `(entity_id, payload)` pairs directly from staging. Pro: cpu
     writes raw deltas. Con: gpu kernel change + a slightly
     different upload format.

   Which is faster depends on dirty rate and staging-bandwidth
   contention. Profile both before committing.

2. **Cross-frame ownership of staging.** Today main writes staging
   between sim and submit; only one frame in flight. If render runs
   in parallel with sim N+1, we either need to triple-buffer staging
   (we already do) and only block sim N+1 on "render thread for
   frame N is done", or accept one extra frame of input lag.

3. **NUMA placement of delta vecs.** Per-worker delta vecs are
   touched by their producing worker (DRAM, local node) and then by
   the render thread (probably on node 0). Either:
   - Allocate on the producer's node (read by render incurs UPI on
     half of them). Likely fine, given the node-1-bound benchmark
     above.
   - Allocate on node 0 (writes from node-1 sim workers go over UPI).
     Likely worse — DRAM writes from node 1 to node 0 vs DRAM reads
     from node 1 to node 0.

4. **Memory overhead.** ~50% more DRAM per dirty field per frame
   (3× delta vecs in addition to the dense staging triple). At 1M
   cubes × 3 fields × ~16 B / pair × triple-buffer ≈ 144 MB peak.
   Acceptable.

5. **Render thread idle policy.** Same hybrid spin/yield/park as
   the engine pool. Wake from main at "sim barrier done"; signal
   "scatter done" via an atomic counter that sim N+1 acquires on
   its first staging touch.

## Why deferred

The current focus is **stability / variance reduction** at the
existing synchronous pipeline. Tail-latency spikes (3-4 ms
`work_max`) are caused by OS scheduler / IRQ noise on the 256-core
saturated configuration, not by lack of pipeline overlap. Fixing
the variance with cacheline padding, NUMA-local wake infrastructure,
and the new spin/yield/park idle policy is a precondition: there's
no point pipelining sim and scatter if either of them still spikes
mid-frame.

Once `pf_barrier max` and `sim_update max` are within ~2× of their
averages, revisit this doc and prototype the delta+render-thread
path.
