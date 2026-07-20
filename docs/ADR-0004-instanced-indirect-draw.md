# ADR-0004: Instanced / Indirect Draw for the Scene Pass

**Status:** Accepted — Phase 1 landed. Frame time at N=100K dropped ~10× (10 ms → 1 ms); N=1M is now feasible (~4.5 ms / ~220 FPS, was previously not measurable). See [Measurements](#measurements-post-phase-1) below.
**Date:** 2025
**Scope:** `crates/engine-render/src/camera.rs` (`scene_secondary` recording), `crates/engine-render/src/lib.rs` (`build_frame_slot`, draw-call submission), `crates/engine-render/shaders/scene.vert`
**Related:** [ADR-0003](ADR-0003-shared-staging-with-compute-sync.md)

## Context

The scene secondary command buffer currently records **one `draw_indexed` call per `RenderInstance`**. At N = 100 000 cubes that's 100 000 draw calls per frame baked into the secondary, which the GPU then walks every frame.

Measurements from the [ADR-0003 stress benchmark](ADR-0003-shared-staging-with-compute-sync.md#measurements-2025-session):

| Cubes  | Frame time | Per-instance cost (assumed linear) |
|---|---|---|
| 1      | ~100 µs   | — |
| 1 000  | ~250 µs   | ~150 ns/instance |
| 10 000 | ~900 µs   | ~80 ns/instance  |
| 100 000 | ~9 900 µs | ~99 ns/instance |

The per-instance cost stays at 80–100 ns whether N is 10K or 100K, which is the signature of a *fixed-cost-per-draw-call* bottleneck, not a vertex/fragment-shading bottleneck. Cubes are 24 verts / 36 indices each — well below the threshold where vertex throughput would dominate.

`Scene::update` (per-component CPU work) is ruled out: a `--static-scene` run at N=100K saved only 250 µs / 9 900 µs ≈ 2.5%. The CPU staging write was likewise ruled out: the multithreaded path landed in this session and showed no measurable effect at any N up to 100K.

So the next big win is a draw-call collapse. The standard GPU-driven-rendering pattern: **emit one `vkCmdDrawIndexedIndirect` (or `vkCmdDrawIndexedIndirectCount`) per mesh, with N indirect-draw structs in a buffer, and let the GPU iterate the instances.**

## Decision (proposed)

Replace the per-instance `draw_indexed` loop in `RenderCamera::scene_secondary` with a **single `draw_indexed_indirect_count` call per mesh**, sourcing the per-draw parameters from a device-local indirect-args buffer the renderer maintains.

### High-level shape

```
                         per-mesh, per-camera
       ┌─────────────────────────────────────────────────┐
       │ vkCmdDrawIndexedIndirectCount(                  │
       │   buffer = indirect_args_buf  ─────────────┐    │
       │   count_buffer = indirect_count_buf  ──┐   │    │
       │ );                                     │   │    │
       └────────────────────────────────────────┼───┼────┘
                                                ▼   ▼
                                        ┌───────────────────────┐
                                        │ indirect_count_buf:   │
                                        │   u32 = N_visible     │
                                        ├───────────────────────┤
                                        │ indirect_args_buf:    │
                                        │   [VkDrawIndexedIndirectCommand;
                                        │      max_visible]     │
                                        │   { index_count, instance_count = 1,
                                        │     first_index, vertex_offset,
                                        │     first_instance = i }            │
                                        └───────────────────────┘
```

The vertex shader continues to look up `instance_to_entity[gl_InstanceIndex]` (or `gl_DrawID` if we go the per-draw-args route) → SoT TRS → MVP. **No vertex-shader change is required for the trivial "all-static, all-visible" version.**

### Two phases, separable

#### Phase 1 — single indirect call, draw-list authored on CPU, no culling

Goal: collapse the draw-call submission cost. Keep everything else identical.

- Add `indirect_args_buf` (device-local, `INDIRECT_BUFFER | TRANSFER_DST`) to `RenderCamera`. Sized to `allocated_capacity` (the same geometric grow that `device_matrices` uses today).
- Add `indirect_count_buf` (device-local, `INDIRECT_BUFFER | TRANSFER_DST`, single u32) to `RenderCamera`.
- At topology change: stage and copy `N` `VkDrawIndexedIndirectCommand` structs into `indirect_args_buf`, and stage the count `N` into `indirect_count_buf`. Same lifecycle as `instance_to_entity` upload today (one-shot, on topology change).
- `scene_secondary` records **one** `draw_indexed_indirect_count(indirect_args_buf, 0, indirect_count_buf, 0, max_count, stride)` call per mesh (today: per `RenderInstance`).
- For the multi-mesh case (when meshes other than the cube exist), record one indirect-count call per mesh, sourcing a slice of the indirect-args buffer.
- Vertex shader: unchanged. `gl_InstanceIndex` indexes the MVP buffer just like today; the indirect-args' `firstInstance` field provides the per-draw offset.

Expected outcome: N=100K frame time drops from ~10 ms to ~1–2 ms (close to GPU vertex/raster cost). The CPU-side staging-write parallelism from ADR-0003 should start showing measurable wins around this point.

#### Phase 2 — GPU-driven draw-list construction (culling)

Once Phase 1 lands and the draw-call cost is gone, the next bottleneck moves to "we're drawing every entity even though most are off-screen / occluded." Phase 2 adds a GPU compute pass that:

1. Reads SoT TRS + per-instance bounding-sphere radius.
2. Tests each instance against the camera frustum (and optionally against a Hi-Z buffer for occlusion).
3. Atomically appends visible instances into `indirect_args_buf` and increments `indirect_count_buf`.
4. The graphics dispatch then draws only the visible instances.

This is a separate ADR-worth of work. Phase 1 is the prerequisite.

### Where things move

| Concept | Today | After Phase 1 |
|---|---|---|
| Per-instance draw call | `draw_indexed` × N in `scene_secondary` | One `draw_indexed_indirect_count` per mesh |
| `instance_to_entity` lookup | One u32 buffer per camera, uploaded on topology change | **Unchanged** — still needed by `mvp_build_cs` and the vertex shader's MVP lookup |
| `mvp_build_cs` | Computes MVP for every instance | **Unchanged** in Phase 1 (still computes for every instance — Phase 2 will scope to visible-only) |
| Per-draw `instance_count` | 1 (one cube per call) | 1 (still — Phase 1 doesn't change the per-instance topology, just the call count) |

### Why `vkCmdDrawIndexedIndirectCount` and not just `vkCmdDrawIndexedIndirect`?

- `vkCmdDrawIndexedIndirect` always issues `drawCount` draws regardless of which are valid.
- `vkCmdDrawIndexedIndirectCount` reads the actual count from a device-local buffer, which Phase 2's GPU culling pass can write atomically. Picking the same primitive in Phase 1 means Phase 2 is drop-in: the count buffer goes from "constant N" to "atomic counter."
- Requires `VK_KHR_draw_indirect_count` (core in 1.2). We're already on Vulkan 1.3-equivalent feature set; this is free.

### Capacity / invalidation

The new per-camera buffers (`indirect_args_buf`, `indirect_count_buf`) join the existing per-camera buffers (`device_matrices`, `instance_to_entity`) under the same geometric-growth policy in `RenderCamera::ensure_capacity`. When camera capacity grows, all four buffers are re-allocated together and the secondary CB is re-recorded. The descriptor sets that mvp-build set 0 binds gain `indirect_args_buf` (Phase 2 only — Phase 1 doesn't read the buffer from compute).

## Implementation plan

### Phase 1 — CPU-authored single indirect call

1. **`RenderCamera`:**
   - Add `indirect_args_buf: Subbuffer<[DrawIndexedIndirectCommand]>` and `indirect_count_buf: Subbuffer<[u32; 1]>`, device-local with appropriate usage flags.
   - In `new` / `ensure_capacity`: allocate sized to `allocated_capacity`. Build the indirect-args contents on the CPU (one struct per `RenderInstance`, `firstInstance = i`) and `vkCmdCopyBuffer` from a host-visible staging into the device-local buffer. Stage `count = N` similarly.
   - One-time copy at topology change. No per-frame upload.
2. **`scene_secondary` recording:** replace the per-instance `draw_indexed` loop with one `draw_indexed_indirect_count` per mesh. For the trivial "all instances use mesh 0" case this is a single call.
3. **Multi-mesh case:** sort `RenderInstance`s by `mesh_index` before building `indirect_args_buf` so each mesh's indirect-args is a contiguous range. Record one indirect-count call per mesh-range with the appropriate `buffer_offset` / `max_count`.
4. **Vertex shader:** unchanged. Indirect's `firstInstance` field becomes the `gl_InstanceIndex` the shader sees — same MVP-buffer lookup pattern.
5. **Benchmark:** re-run the ADR-0003 benchmark suite (`--cubes 1 / 1000 / 10000 / 100000 / 1000000`). Expected: N=100K from ~10 ms → ~1–2 ms; N=1M becomes feasible (was not measured before).

### Phase 2 — GPU culling (separate ADR)

Defer until Phase 1 lands and we can measure the post-collapse bottleneck.

## Consequences

### Wins

- **Frame time at scale.** Expected ~5–10× speedup at N ≥ 100K.
- **Unblocks ADR-0003's full architectural refactor.** Once the draw-call bottleneck is gone, single-shared-staging + timeline-semaphore sync becomes measurable.
- **Standard GPU-driven-rendering shape.** Future features (instance LOD, GPU-side occlusion, frustum culling, multiview) all build on this foundation.
- **Vertex shader stays the same.** The `gl_InstanceIndex → mvp[i]` lookup we already use is exactly the indirection indirect-draw expects.

### Costs

- ~150 lines of new buffer plumbing in `RenderCamera`.
- One additional buffer-copy at topology change (CPU-staged → device-local indirect args). Cheap in absolute terms; rare in frequency.
- Multi-mesh path requires sorting instances by mesh, which changes their relative order in the MVP buffer. The `instance_to_entity` lookup table absorbs this — no shader change needed.

### Caveats

- `multiDrawIndirect` device feature must be enabled at `VulkanoContext` creation. (Already core in our targeted device feature set.)
- `drawIndirectFirstInstance` device feature also required if we use `firstInstance` as the per-draw discriminant — most desktop GPUs support it; check at startup and panic with a clear message if absent.
- Keep `vkCmdDrawIndexedIndirectCount`'s `maxDrawCount` argument at the camera's `allocated_capacity` so the count buffer can drop the value below `maxDrawCount` without exceeding the indirect-args buffer's bounds.

## Revisit if

- Profiling after Phase 1 shows the bottleneck has moved somewhere we don't expect (e.g. if vertex shading dominates at N=1M, we may want to revisit instance LOD before Phase 2).
- The multi-mesh sort shows up as a CPU cost at high mesh-variety scenes (unlikely; sorting 1M u32 keys is sub-millisecond).

## Measurements (post Phase 1)

### Test setup

- `cargo run --release -p test-game -- --cubes N` with the spinning-Rotator scene unless noted.
- `--static-scene` skips the per-entity `Rotator::update` so the CPU-side `Scene::update` cost is removed (isolates draw / staging / compute).
- `RAYON_NUM_THREADS=1` forces a single staging-write thread to A/B the multithreaded staging path that landed under ADR-0003.
- Mailbox present mode (uncapped); FPS sampled in steady state across 10–20 s of run time.

### Headline numbers

| Cubes  | Pre-Phase-1 (per-instance `draw_indexed`) | Post-Phase-1 (one indirect call per mesh) | Speedup |
|---|---|---|---|
| 1         | ~0.10 ms (~10 000 FPS) | ~0.10 ms (~10 000 FPS) | — (already at GPU floor) |
| 1 000     | ~0.26 ms  (~3 800 FPS) | ~0.20 ms  (~5 000 FPS) | ~1.3× |
| 10 000    | ~0.91 ms  (~1 100 FPS) | ~0.69 ms  (~1 450 FPS) | ~1.3× |
| 100 000   | ~9.9 ms   (~100 FPS)   | **~1.01 ms (~990 FPS)** | **~10×** |
| 1 000 000 | not measured (extrapolated ~100 ms) | **~4.55 ms (~220 FPS)** | **N=1M unlocked** |

N=100K matches the ~1–2 ms prediction; N=1M is now interactive. The previously observed flat ~80–100 ns/instance plateau (signature of fixed-cost-per-draw-call) is gone.

### Bottleneck has moved to the CPU side

With the draw-call cost collapsed, the next-largest contributors become visible. `--static-scene` (skip `Rotator::update`) and `RAYON_NUM_THREADS=1` (force single-threaded staging write) isolate them:

| Cubes | animated, multi-thread (default) | `--static-scene`, multi-thread | animated, **single-thread** |
|---|---|---|---|
| 100 000   | ~1.01 ms  (~990 FPS) | ~0.62 ms  (~1 600 FPS) | **~4.0 ms  (~250 FPS)** |
| 1 000 000 | ~4.55 ms  (~220 FPS) | ~1.34 ms  (~745 FPS)   | <33 ms (no FPS sample in 30 s) |

Observations:

- **Per-entity component update (`Rotator::update`) is now a real cost.** At N=1M it consumes ~3.2 ms / frame (4.55 − 1.34); at N=100K ~0.4 ms.
- **Multithreaded staging writes are no longer a no-op.** The rayon `par_chunks_mut` path that landed under ADR-0003 (which showed no measurable signal at the time — see ADR-0003 §Findings) now shows a clear ~4× win at N=100K (~990 FPS multi-thread vs ~250 FPS single-thread). At N=1M the single-threaded run failed to hit even 30 FPS; rayon parallelism is what makes 1M feasible at all on the CPU side.
- **Per-frame staging traffic is now the dominant GPU-readable cost** (the difference between the static and animated columns roughly tracks the dirty-bit count: at 1M the animated case has ~1M rotation entries to upload per frame; at static, only the per-frame `view_proj` write).

### What this unblocks

ADR-0003's full architectural refactor (single shared staging buffer + timeline-semaphore sync on `COMPUTE_SHADER`) was deferred precisely because the per-instance draw-call cost was hiding any potential CPU/staging wins. With Phase 1 landed, that ADR's per-frame VRAM savings (4× at high N) and the CPU/GPU overlap it enables are now both measurable and worth implementing. See [ADR-0003 §Status](ADR-0003-shared-staging-with-compute-sync.md#status-of-the-full-architectural-refactor).

### Phase 2 (GPU culling) status

Still the right next major step after the ADR-0003 refactor lands. At 1M cubes most are off-screen at any given camera position, so frustum culling is expected to give another large multiplicative win on the GPU side. Deferred until the CPU-side bottleneck is addressed.
