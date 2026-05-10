# ADR-0001: Custom (Unchecked) Swapchain Renderer

**Status:** Accepted
**Date:** 2025
**Scope:** `crates/engine-render/src/swapchain.rs`

## Context

The renderer originally used `vulkano-util`'s `VulkanoWindowRenderer` for the
present/submit hot path. Profiling showed this path:

- Allocated several `Box<dyn GpuFuture>` trampolines per frame.
- Used `FenceSignalFuture` + `then_signal_fence_and_flush`, each forcing a
  `vkQueueSubmit` plus mutex/allocation work.
- Effectively issued ~3 `vkQueueSubmit` calls per frame (render + signal +
  the present trampoline's empty submit).

We also hit a vulkano host-side validation panic on cached `SimultaneousUse`
command buffers referencing per-image depth attachments: vulkano's host-side
access tracker kept the resources locked until the future chain's
`signal_finished` ran, and re-submission for the same swapchain image index
was rejected.

## Decision

Replace the `vulkano-util` present path with a manual `SwapchainRenderer` that:

- Calls `Queue::submit_unchecked` and `Queue::present_unchecked` directly.
- Pre-allocates per-frame semaphores + fences:
  - `image_available[frame_slot]` — signaled by `vkAcquireNextImageKHR`.
  - `in_flight[frame_slot]` — host-waited + reset to recycle a frame slot.
  - `render_finished[image_index]` — per-image semaphore for present.
- Issues exactly **one** `vkQueueSubmit2` and **one** `vkQueuePresentKHR` per frame.

This bypasses vulkano's host-side resource tracking *for the render submit and
present only*. Result: ~3→2 Vulkan calls per frame, no per-frame heap
allocations on the hot path, and ~30–40% FPS improvement on the test machine
(~8.7k → ~11.5–11.7k FPS in the editor).

## What This Does NOT Change

The custom swapchain only opts out of tracking **for the submissions it issues
itself**. Everything else in vulkano keeps working normally:

- **`AutoCommandBufferBuilder` still inserts pipeline barriers automatically**
  *within* a recorded command buffer. This logic runs at record time and is
  unaffected. A compute CB with two dependent dispatches will still get the
  right `vkCmdPipelineBarrier` between them.
- **Image layout transitions** within an auto-recorded CB still happen.
- **`GpuFuture`-based submits elsewhere** (e.g. compute work submitted via
  `cb.execute(queue).then_signal_fence_and_flush()`) retain full host-side
  access tracking and automatic semaphore chaining.
- The `Device` and `Queue` themselves are not modified — there is no global
  opt-out.

So a typical compute workflow recorded with the auto builder and submitted via
the standard future API behaves exactly as documented in vulkano. The custom
path is a closed system around the present loop only.

## Caveats — Where You MUST Be Careful

The danger zone is **interaction between the unchecked render submit and any
vulkano-tracked work that touches the same resources.**

### 1. Cross-submit dependencies are invisible to vulkano

If a compute dispatch (tracked submit) writes a resource that the render path
(unchecked submit) reads — e.g. a compute-generated vertex buffer, a texture
sampled by a material, a GPU-driven indirect draw buffer — vulkano's tracker
does not know about the unchecked render submit. Consequences:

- No automatic barrier or semaphore is inserted between the two submits.
- Vulkano won't *block* the compute submit thinking the resource is "still in
  use" by the render — convenient, but correctness is on you.

**Mitigations (pick one):**

- **Preferred:** record the compute dispatch into the same command buffer the
  renderer submits. Auto-barriers inside one CB still work, so RAW/WAW are
  handled automatically.
- **If split submits are required:** add an explicit `Semaphore` — signal it
  from the compute submit, add it to `wait_semaphores` of the swapchain
  renderer's submit info.

### 2. Same-queue ordering assumptions don't hold

Vulkano may believe a resource is free when in fact the unchecked render
submit is still using it on the same queue. Don't rely on submission order
alone for resource visibility across the boundary.

**Mitigation:** semaphore for GPU→GPU ordering, or wait on the swapchain's
`in_flight` fence on the host before issuing tracked work that touches a
shared resource.

### 3. Async transfers / staging uploads

Fine via vulkano's tracked path, **but** you must fence-wait (or semaphore-
gate) before the resource is referenced by an unchecked render frame.

### 4. Resource lifetime

The unchecked submits do not extend `Arc` lifetimes via vulkano's normal
mechanism. The `SwapchainRenderer` owns the per-frame primitives, but any
resource referenced by a pre-recorded CB must be kept alive elsewhere
(currently: in `RenderContext`). If you start dynamically swapping resources,
make sure the old ones live until the relevant `in_flight` fence has signaled.

## Rule of Thumb

> The unchecked path is a closed system around the swapchain present loop.
> Anything that crosses its boundary needs an explicit sync primitive
> (semaphore for GPU↔GPU, fence for GPU→CPU). Anything fully inside
> vulkano's world keeps all its conveniences.

## Suggested API for Future Compute Integration

When adding compute that feeds the renderer, expose a small helper on
`SwapchainRenderer`:

```rust
pub fn add_wait_semaphore(&mut self, sem: Arc<Semaphore>, stage: PipelineStages);
```

This lets external subsystems hook into the render submit's wait list cleanly,
without reaching into the unchecked code.

## Consequences

**Positive:**
- ~30–40% FPS improvement on the present hot path.
- One submit + one present per frame; no per-frame heap allocations.
- Fixed the cached-CB host-tracker validation panic by construction (we no
  longer go through vulkano's tracker for those submits).

**Negative:**
- `unsafe` surface area concentrated in `swapchain.rs`.
- Cross-submit synchronization with the render path is now a manual concern.
- Less help from vulkano if we misuse swapchain image layouts or per-frame
  resource lifetimes — bugs here are likely to surface as driver/validation
  errors rather than friendly Rust-side messages.

## Revisit If

- We need multi-queue rendering (graphics + async compute + transfer) and the
  manual sync becomes error-prone.
- Vulkano gains a lower-overhead tracked path that closes the perf gap.
- We add a debug build mode and want the safer `vulkano-util` path back as a
  toggle for diagnosing rendering bugs.
