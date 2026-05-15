# ADR-0003: Shared Staging Buffers with Compute-Stage Timeline Sync

**Status:** **Landed.** BAR memory placement + multithreaded staging landed in an earlier session; the full single-shared-staging + timeline-semaphore refactor landed in this session. The shared staging+dirty+view_proj+scatter-secondary now live on `WorldTransformGpu`; per-frame mutations are gated by a timeline semaphore signaled at `COMPUTE_SHADER` stage end of every submit.
**Date:** 2025
**Scope:** `crates/engine-render/src/transform_gpu.rs`, `crates/engine-render/src/lib.rs` (`FrameSlot`, `build_frame_slot`, `RenderApp::about_to_wait`), `crates/engine-render/src/swapchain.rs`
**Related:** [ADR-0001](ADR-0001-custom-swapchain.md), [ADR-0002](ADR-0002-per-frame-cb-recording.md), [ADR-0004](ADR-0004-instanced-indirect-draw.md)

## Context

The renderer currently maintains **one set of per-component host-visible staging buffers per `FrameSlot`** (i.e. per swapchain image / per frame in flight). Each frame the CPU writes to its current slot's staging mirror; the GPU's scatter compute then promotes that staging into the world-scoped device-local SoT. CPU writes are gated by the per-image fence the swapchain renderer waits on inside `acquire`.

This works correctly and was designed for two reasons:

1. **No CPU↔GPU race on staging.** Each slot has its own staging buffer that only one writer touches at a time (CPU, gated by the per-image fence; GPU only reads it after the CPU's write).
2. **Independence between in-flight frames.** The CPU can be writing slot 1's staging while the GPU is still consuming slot 0's.

The cost: at `MAX_FRAMES_IN_FLIGHT = 4` we keep four copies of every staging buffer. At small entity counts this is irrelevant. At scale it adds up — for a 1M-entity scene with three components × 16 B per slot, the per-slot staging buffers consume **192 MB of VRAM** purely for in-flight independence, which is `4×` the 48 MB SoT they exist to feed.

The dirty bitmasks have the same shape: per-slot, one per component. Same `4×` multiplier.

## Decision (proposed)

Move the staging buffers, dirty bitmasks, scatter descriptor sets, and scatter secondary CB **out of `FrameSlot` and into `WorldTransformGpu`** as single shared instances. Synchronize CPU writes against GPU reads using a **timeline semaphore signaled at `PipelineStages::COMPUTE_SHADER` end**.

### Key insight

The staging buffer only needs to outlive the scatter pass that reads it. Once scatter has copied `staging → SoT`, the staging contents are dead — every downstream stage (`mvp_build`, scene render, blit, present) reads the SoT, never the staging. So the sync requirement collapses from "wait for the entire previous frame to complete" to **"wait for the previous frame's scatter stage to complete."**

This is exactly what `vkSubmitInfo2`'s per-stage signal mask gives us: signal a timeline semaphore at `COMPUTE_SHADER` stage end, host-wait on the previous frame's value before writing the shared staging for the next frame.

### Synchronization model

```
                                  Wait for compute stage of frame N
                                          ↓
CPU:  write shared │ submit N │           write N+1 │ submit N+1 │
                              ↓                       ↓
GPU:                         scatter N → mvp_build N → render N → blit N → present N
                              ↑                                    scatter N+1 → ...
                              └ Signals timeline value N here
                                (end of COMPUTE_SHADER stage; covers
                                 both scatter and mvp_build)
```

Two independent sync mechanisms after the change:

| Mechanism | Purpose | Wait point |
|---|---|---|
| **Timeline semaphore on compute-stage** | Gates host writes to shared staging / dirty / view_proj | Right before host writes per frame |
| **Per-image fence** (existing) | Gates re-submission of the per-image primary CB and reuse of the swapchain image | Right before submission per frame |

Both are submitted as part of the same `vkQueueSubmit2`. There is no extra syscall.

### Ownership after the change

| Resource | Today | Proposed |
|---|---|---|
| SoT pos / rot / scl | `WorldTransformGpu` (shared) | Unchanged |
| Staging pos / rot / scl | `FrameSlot × 4` | `WorldTransformGpu` (shared, BAR memory if available) |
| Dirty bitmask pos / rot / scl | `FrameSlot × 4` | `WorldTransformGpu` (shared) |
| `view_proj_buf` | `FrameSlot × 4` | `WorldTransformGpu` (shared) |
| Scatter descriptor sets (3) | `FrameSlot × 4` | `WorldTransformGpu` (shared, capture shared buffers) |
| `mvp_build_set1` (view_proj) | `FrameSlot × 4` | `WorldTransformGpu` (shared) |
| Scatter secondary CB | `FrameSlot × 4` | `WorldTransformGpu` (shared) |
| `mvp_build_secondary` | `FrameSlot × 4` | `RenderCamera` (shared per camera; captures shared `mvp_build_set1`) |
| `blit_secondary` | `FrameSlot × N_swapchain_images` | Unchanged — destination is per-image |
| Composing primary | `FrameSlot × N_swapchain_images` | Unchanged — captures the per-image blit secondary |

`FrameSlot` shrinks to just the blit secondary + composing primary.

### Memory placement (BAR / ReBAR)

The shared staging buffers can be allocated with `MemoryTypeFilter::PREFER_DEVICE | HOST_SEQUENTIAL_WRITE`, requesting BAR memory on systems that expose it (most discrete GPUs since 2020 with Resizable BAR enabled). On these systems:

- CPU writes: same speed as classic host-visible (write-combined PCIe writes into VRAM)
- GPU reads from scatter compute: full VRAM bandwidth instead of PCIe per cache line
- `vkCmdFillBuffer` clears: same fast-path as today

On systems without BAR (or where the BAR heap is too small), vulkano transparently falls back to plain host-visible memory. Same correctness, same code path.

**Out of scope for this ADR:** in-shader dirty-bit clearing. We tried that and saw a 16× regression on host-visible buffers ([investigation in `shaders/scatter.comp` header](../crates/engine-render/shaders/scatter.comp)). With BAR memory this might become viable, but the win over `vkCmdFillBuffer` is small and the architecture works fine without it.

### Capacity-grow path

`WorldTransformGpu::ensure_capacity(needed)` (already the entry point for SoT growth) gains responsibility for re-allocating the shared staging + dirty buffers and rebuilding the shared scatter descriptor sets + scatter secondary. After the rebuild, `Dirty::mark_all_trs()` is called so the next frame's harvest re-uploads every existing entity into the new SoT.

## Consequences

### Wins

- **VRAM:** ~`(N_in_flight - 1) × staging_bytes` saved. At 1M entities: ~150 MB.
- **Architectural simplification:** `FrameSlot` shrinks to 2 fields. No per-slot scatter pipeline state. World-scoped resources live in the world-scoped struct.
- **Single source of truth at every layer:** SoT is shared today; staging becomes shared too. The "fan out across slots" mental model goes away entirely.
- **Faster GPU staging reads at scale (with BAR):** scatter no longer pulls staging over PCIe per cache-line miss; it reads at VRAM speed.

### Costs

- **Implementation complexity:** ~300 lines across three files; the sync logic has subtleties (correct timeline-value monotonicity across swapchain recreation, correct first-frame semantics, host-wait timing relative to acquire).
- **One additional host-wait per frame.** The timeline-semaphore wait is in addition to the existing per-image fence wait. In the steady state where the GPU keeps up with the CPU both waits are near-zero, but they're two separate kernel calls.
- **Loses per-slot independence for staging.** If frame N+1's CPU prep happens to be much slower than frame N's GPU scatter, we don't gain anything from queueing N+2 and N+3 ahead of it (because they all share the same staging). In practice this matches our workload — the CPU is always faster than the GPU here — but it removes a degree of freedom that exists today.

### Caveats

- **`view_proj_buf` is read by `mvp_build`, not scatter.** Signaling at `COMPUTE_SHADER` stage end (which covers both scatter and mvp_build dispatches) handles this correctly. Don't move the signal to a finer-grained stage without auditing `view_proj`'s readers.
- **Per-image fence still exists and still has to be waited on.** It gates CB re-submission and swapchain image reuse, neither of which the timeline semaphore covers. Don't try to delete the fence.
- **First frame.** Timeline semaphores start at value 0; the first frame waits on value 0 (already signaled, no-op) and submits with value 1. Document this clearly in the swapchain renderer.

## Revisit if

- We measure that the CPU is consistently faster than the GPU's scatter pass at high entity counts (would mean we're bottlenecked on the timeline wait — at which point we'd want to re-introduce some form of staging multi-buffering).
- A future feature genuinely needs per-frame staging mutation independence (e.g. multiple cameras with different scenes; not currently planned).
- BAR memory turns out to be unavailable on a target platform AND the per-slot architecture's PCIe-bound reads become a measurable cost.

## Implementation plan

1. **Stress benchmark first.** Add `--cubes N` to `test-game` (grid layout) so we can measure VRAM and FPS at the entity counts where the trade-off matters.
2. **Path C — pilot the memory type change.** One-line switch of staging buffers from `PREFER_HOST` to `PREFER_DEVICE | HOST_SEQUENTIAL_WRITE`. Verify no regression on the 1-cube case; measure at the new benchmark scale.
3. **Path A — full architectural refactor** as described above. Land behind the same benchmark to confirm the VRAM win and equivalent FPS.

This ADR is in **Proposed** status until step 3 lands.

### What landed in this session (ADR-0003 Path A)

- **Single shared staging.** `WorldTransformGpu` now owns `staging_positions`, `staging_rotations`, `staging_scales`, the three `staging_dirty_*` bitmasks, the **single-mat4 `staging_view_proj`**, the three scatter descriptor sets, the shared `mvp_build_set1`, and the shared scatter compute secondary. All allocated once per world (and re-allocated on `ensure_capacity` grow), not once per `MAX_FRAMES_IN_FLIGHT` swapchain slot.
- **Uniform staging→SoT paradigm for `view_proj`.** Earlier iterations of this refactor put `view_proj_buf` in either the timeline-gated shared set (cost ~4 ms / frame at N=1M because the host wait then had to cover mvp_build's read of view_proj) or in a per-image-fence-protected ring (worked, but added ring sizing, push-constant indexing, per-ring-slot mvp_build secondaries, and extra plumbing on swapchain image-count change). The actual right answer: `view_proj` follows the **same staging→SoT pattern as TRS**:
  - `view_proj_buf`: host-mapped staging mat4 (`TRANSFER_SRC`).
  - `sot_view_proj`: device-local stable mat4 (`STORAGE_BUFFER | TRANSFER_DST`), bound by `mvp_build_set1`.
  - The scatter primary CB ends with a `vkCmdCopyBuffer(staging_view_proj → sot_view_proj)` — same staging→SoT promotion the TRS scatter dispatches do, just expressed as a 64-byte transfer because there's no per-entity dirty mask to gate on.
  - `mvp_build_cs` reads only stable SoT (`sot_<comp>` and `sot_view_proj`); it never touches a host-shared buffer.
  - This makes the staging→SoT paradigm a uniform invariant of the engine: **anything the host writes per-frame goes through a host staging buffer that a compute or transfer pass promotes into a stable device-local SoT, and shaders only ever read SoT.**
- **Split-submit, single `vkQueueSubmit2`.** Each frame submits **two batches** in one `vkQueueSubmit2` call:
  - **Batch 0** = the shared `world.scatter_primary` (scatter dispatches + 3 dirty `fill_buffer(0)` clears + the view_proj `copy_buffer`). Signals `compute_timeline` at `COMPUTE_SHADER | ALL_TRANSFER` stage end (smallest mask covering all three reads of host-shared buffers — scatter at compute, fill+copy at transfer).
  - **Batch 1** = the per-image FrameSlot primary (mvp_build + scene render + blit). Waits on `compute_timeline` for the value batch 0 just signaled, at `COMPUTE_SHADER` stage. The wait gives mvp_build the SoT memory visibility from scatter + the view_proj copy — a semaphore signal/wait pair establishes both execution and memory dependency per Vulkan spec, so we do **not** need a manual `vkCmdPipelineBarrier` (and we couldn't easily insert one anyway: `AutoCommandBufferBuilder` doesn't expose `pipeline_barrier`, and we use `submit_unchecked` so vulkano's cross-CB resource tracking is bypassed).
- **Per-camera `mvp_build_secondary`** — single secondary per camera (no ring, no per-ring-slot variants, no `view_proj_idx` push constant). Recorded `SimultaneousUse` because multiple FrameSlot primaries can reference it concurrently. Re-recorded by `ensure_capacity` (camera capacity / topology change) and `on_world_capacity_change` (world capacity grow).
- **`FrameSlot` collapsed.** From 14 fields to 2: just the per-image `blit_secondary` (whose destination is *that* slot's swapchain image) and the composing primary CB. The composing primary executes only `mvp_build_secondary`, the scene secondary, and the blit — scatter, the dirty fill_buffers, and the view_proj copy all moved out into `world.scatter_primary`.
- **`SwapchainRenderer::submit_and_present`** rewritten to take an optional `pre_batch: Option<PreBatch>` plus `extra_main_waits` and `extra_main_signals`. Both batches go into one `submit_unchecked(&[batch0, batch1], Some(fence))` call — still one syscall per frame.
- **Frame-slot rebuild ordering.** `frame_slots.clear()` before reassigning. Strictly required only when the captured secondaries are `MultipleSubmit` (which they aren't post-staging-paradigm), but kept defensively to keep rebuild ordering robust.
- **Device feature** `timeline_semaphore: true` opted in at `VulkanoConfig`.

### Measurements (post Path A landing, with view_proj staging→SoT)

Same setup as before (release build, Mailbox present, spinning Rotator scene unless `--static-scene`). Three intermediate iterations are shown so the trade-off curve is visible:

| Cubes     | Pre-refactor | view_proj in timeline-gated set | view_proj as per-image ring + split-submit | **view_proj staging→SoT (final)** |
|---|---:|---:|---:|---:|
| 1         | ~10000 FPS / ~0.10 ms | ~7700 FPS / ~0.13 ms | ~7800 FPS / ~0.13 ms | ~8100 FPS / ~0.12 ms |
| 10000     | ~1450 FPS  / ~0.69 ms | ~1300 FPS / ~0.77 ms | (similar)            | (similar) |
| 100000    | ~990 FPS   / ~1.01 ms | ~880 FPS  / ~1.13 ms | ~785 FPS  / ~1.27 ms | ~800 FPS  / ~1.25 ms |
| 1000000 (animated)         | ~220 FPS / ~4.55 ms | ~120 FPS / ~8.4 ms  | ~177 FPS / ~5.6 ms | **~250 FPS / ~4.0 ms** |
| 1000000 (`--static-scene`) | ~745 FPS / ~1.34 ms | ~326 FPS / ~3.07 ms | ~345 FPS / ~2.9 ms | **~718 FPS / ~1.4 ms** |

**Net result at N=1M: parity with (or slightly faster than) the pre-refactor per-slot-staging implementation, plus the ~144 MB VRAM saving.** The intermediate iterations show that the regression was entirely the host-wait covering mvp_build's read of `view_proj` — once `view_proj` goes through the same staging→SoT path as TRS, the host wait fires the moment scatter+fill+copy are done (microseconds at any N), and mvp_build runs in parallel with the next frame's host prep.

**The lesson the intermediate iterations made expensive:** the right model is *not* "single shared host-writable buffer + clever per-image gating" (the ring) and *not* "shared host-writable buffer + bigger wait" (the original split-submit with view_proj still shared). It's *"compute shaders read only stable SoT; the host writes only host-staging; a per-frame compute/transfer pass promotes staging→SoT."* This is now a uniform invariant of the engine; future host-driven inputs (light positions, camera-extra params, material indices, etc.) should follow the same pattern.

**VRAM savings (the headline win):** at `MAX_FRAMES_IN_FLIGHT = 4` the staging triple + dirty bitmasks + scatter sets + scatter secondary that used to be per-slot are now world-scoped. Roughly:

- Per-component staging triple at N=1M: 16 B × 1M × 3 components = 48 MB. Pre-refactor: 4× that = 192 MB; post-refactor: 48 MB. **~144 MB saved.**
- Dirty bitmasks at N=1M: 4 B × 31250 words × 3 components ≈ 375 KB. Pre-refactor: 4× = 1.5 MB; post-refactor: 375 KB. (Negligible at this scale.)
- Scatter descriptor sets and scatter secondary: 4 of each becomes 1 of each. Negligible bytes, real architectural simplification.
- `view_proj` adds one device-local mat4 (64 B). Negligible.

### Status of the original implementation plan

- Step 1 (`--cubes N` benchmark) — landed in earlier session.
- Step 2 (BAR memory pilot) — landed in earlier session.
- Step 3 (Path A full architectural refactor) — **landed in this session.**

This ADR is now in **Landed** status.

## Measurements (2025 session, deferred Path A)

Steps 1 and 2 were completed. The grid benchmark and BAR-memory pilot landed; the full architectural refactor is deferred behind ADR-0004.

### Test setup

- `cargo run --release -p test-game -- --cubes N` with the spinning-Rotator scene unless noted.
- `--static-scene` flag added to bypass `Scene::update` for isolating CPU update cost.
- Force single-threaded rayon for A/B comparisons by exporting `RAYON_NUM_THREADS=1`.

### Findings

| Cubes  | Baseline (`PREFER_HOST` + WC, single-thread write) | + BAR (`PREFER_DEVICE` + WC) | + BAR + multithreaded write (`HOST_RANDOM_ACCESS`) |
|---|---|---|---|
| 1      | ~10000 FPS | ~9500 FPS  | ~10100 FPS |
| 1000   | ~3800  FPS | ~3850 FPS  | ~3900 FPS  |
| 10000  | ~700   FPS | **~1100 FPS** (+57%) | ~1100 FPS (no further change) |
| 100000 | (not measured — see note) | ~100 FPS | ~101 FPS |

**BAR memory placement is a real win at scale (+57% at 10K), no regression at 1-cube.** This is the change that landed.

**Multithreading the staging write was a no-op at all measured scales.** Cause confirmed via the `--static-scene` probe: at 100K entities, disabling `Scene::update` only saved ~250 µs out of a 9900 µs frame (~2.5%). The CPU is *not* the bottleneck at our current scale — the GPU is, specifically the per-instance `draw_indexed` submission cost (100K calls per frame at N=100K). Until single-instanced indirect draw lands, parallelising the staging write has no measurable effect.

The parallel staging path landed anyway (kept behind `HOST_RANDOM_ACCESS` for the cached-coherent memory it requires) because:

1. It's correct and well-tested.
2. Once ADR-0004 reduces draw-call cost to a single `vkCmdDrawIndexedIndirectCount` per frame, the GPU should clear the bottleneck and the per-frame staging write *will* become a measurable cost at 1M+ entities. The infra is then already in place.
3. The `HOST_RANDOM_ACCESS` memory mode is also what we want for parallel writers regardless of whether they help today.

### Status of the full architectural refactor

The single-shared-staging-buffer + timeline-semaphore design described above is **not yet implemented**. The motivation (4× VRAM savings on staging) is real and remains the right target.

**Update (post ADR-0004 Phase 1):** with the per-instance draw-call bottleneck now gone, the per-frame staging-write cost is exposed:

- N=100K: single-threaded staging walk runs at ~250 FPS (~4 ms) vs ~990 FPS (~1 ms) multi-threaded — the rayon path is doing real work now.
- N=1M: animated frame time is ~4.5 ms; `--static-scene` (which removes per-entity dirtying and so most of the per-frame staging upload) drops it to ~1.3 ms. The 3.2 ms delta is the per-frame staging traffic — exactly what the shared-staging refactor would let us overlap with the previous frame's GPU compute.

Full benchmark in [ADR-0004 §Measurements](ADR-0004-instanced-indirect-draw.md#measurements-post-phase-1). This ADR is unblocked and is the next major refactor.

### What landed in this session

- `make_host_storage_slice` gained `prefer_device: bool` and `random_access: bool` parameters. Big staging buffers use `PREFER_DEVICE | HOST_RANDOM_ACCESS`; dirty bitmasks use `PREFER_HOST | HOST_RANDOM_ACCESS`; view_proj stays `PREFER_HOST | HOST_SEQUENTIAL_WRITE`.
- The per-frame staging write in `RenderApp::about_to_wait` now uses `rayon::par_chunks_mut` with `WORDS_PER_CHUNK = 64` (2048 entities per task) across the three staging buffers and three dirty bitmasks in lockstep.
- `Dirty::mark_all_trs()` added in `engine-core` so capacity-grow paths can re-mark the entire world without per-entity iteration.
- `test-game` got `--cubes N` and `--static-scene` for benchmarking.

### What did not land

- Single-shared-staging in `WorldTransformGpu` (Path A above).
- Timeline-semaphore sync on `COMPUTE_SHADER` stage in the swapchain renderer.
- Migration of `view_proj_buf` / `mvp_build_set1` / `scatter_secondary` / `mvp_build_secondary` out of `FrameSlot`.

All of the above remain the right design for high-N. Implement after ADR-0004.
