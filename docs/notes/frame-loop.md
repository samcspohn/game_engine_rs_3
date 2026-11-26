# Frame loop: sync, command buffers, hot path

One `vkQueueSubmit2` and one `vkQueuePresentKHR` per frame, out of command
buffers recorded once and replayed. What is per swapchain image, what is
per camera, what is per world, and what a frame actually does.

## Frame sync

A custom `SwapchainRenderer` (in `engine_render::swapchain`) drives
`vkAcquireNextImageKHR`, `Queue::submit_unchecked`, and
`Queue::present_unchecked` directly, bypassing `vulkano-util`'s present helper
and the `GpuFuture` trampolines it generates. Each frame uses one
image-available semaphore (cycled from a `MAX_FRAMES_IN_FLIGHT`-sized pool)
plus **per-swapchain-image** in-flight fences and render-finished semaphores.
The per-image fence is what gates host-side writes to that image's reusable
staging buffer (see below). Submission and presentation cost exactly **one
`vkQueueSubmit2` + one `vkQueuePresentKHR`** per frame. Swapchain images are
created with `TRANSFER_DST | COLOR_ATTACHMENT` usage (the latter is required
for `ImageView` validation and reserved for a future fullscreen present-pass)
since the renderer blits into them rather than rendering into them directly.
**See [`docs/ADR-0001-custom-swapchain.md`](../ADR-0001-custom-swapchain.md)
for the full rationale and the synchronization caveats that apply when
integrating compute or other tracked submits with the render path.**

## Reusable command buffers

One `MultipleSubmit` primary command buffer **per swapchain image**
("FrameSlot"). `FrameSlot` is minimal — just the per-image `blit_secondary`
(camera color → *this* slot's swapchain image) and the composing primary CB;
the staging buffers (TRS + dirty) and scatter descriptor sets are **shared**
per world on `WorldTransformGpu`, the scatter secondary is one **stage-major**
recording over every world (rebuilt with the frame slots), and the signal
secondary and `gpu_signal` live on `TransformGpuShared`; `hiz_build_secondary`
and `history_update_secondary` are per camera; the cull secondaries
(`cull_secondary`, `cull_pass2_secondary`) are per camera, each recorded
**stage-major over every world** for the same reason the scatter is; the draw
secondaries (`scene_pass1`, `scene_pass2`) are per camera **per world** (all
`SimultaneousUse`). Each frame's `vkQueueSubmit2` carries **one batch with one
CB**: the FrameSlot primary, which runs `scatter_secondary` →
`spawn_scatter_secondary` → `ui.scatter_secondary` → dirty `fill_buffer` ×3 →
the per-camera block promotions → `signal_secondary` → `cull_secondary` (pass
1, every world) → `begin_rendering(Clear)` → per-world `scene_pass1` →
`end_rendering` → `hiz_build_secondary` → `cull_pass2_secondary` (every world)
→ `history_update_secondary` → `begin_rendering(Load)` → per-world
`scene_pass2` → `end_rendering` → `blit_secondary` →
`begin_rendering(swapchain, Load)` → `ui.draw_secondary` → `end_rendering`.
Vulkano auto-sync inserts every barrier (scatter→cull via SoT, fill→signal via
dirty, copy→cull via `view_proj`, depth-attachment→sampled-image transitions
around the Hi-Z build, etc.). See
[ADR-0005](../ADR-0005-dual-pass-occlusion-culling.md) for why the dual-pass
sequence needs two `begin_rendering` scopes. The earlier Path A split-submit
(scatter primary + FrameSlot primary in two batches with a timeline semaphore
between) was abandoned because the inter-batch sync + extra CB submission cost
~30µs/frame at low N; the GPU-write `signal_cs` mid-CB recovers the early-wake
behavior without the syscall. Slots are rebuilt on swapchain recreation, on
camera extent change, on camera capacity growth, and on **world
entity-capacity growth**. **See
[`docs/ADR-0002-per-frame-cb-recording.md`](../ADR-0002-per-frame-cb-recording.md)
for the history (per-frame recording was tried and superseded due to a ~12k→8k
FPS regression).**

## Per-frame hot path

(1) Acquire image → wait per-image fence. (2) If `hierarchy.len() >
world.entity_capacity()`, grow the SoT + shared staging buffers, rebuild every
camera's cull set + cull secondaries, and rebuild all FrameSlots' primary CBs
(per-world axis). (3) If `draws_template.len() > camera.allocated_capacity()`
(or topology length changed), grow the camera's MVP/candidate buffers
geometrically and rebuild the affected FrameSlots (per-camera axis). (4)
**Busy-poll `WorldTransformGpu::gpu_signal[0]`** until it reaches
`next_signal_expected - 1` (`spin_loop` ×64 → `yield_now` → 100µs `sleep`
after ~1ms) — first frame returns immediately because the buffer is
pre-zeroed. (5) Drain `TransformHierarchy::Dirty`'s per-component atomic
bitmasks (`swap(0, Relaxed)` per word) directly into the **shared**
`staging_dirty_{pos,rot,scl}` + `staging_{pos,rot,scl}` via raw SoA accessors
(`numa_pool::parallel_for`, no per-entity `Mutex`); write each camera's block
into its own staging. (6) Submit the slot's pre-recorded primary CB — plain
submit, no extra waits/signals, one `vkQueueSubmit2` + one `vkQueuePresentKHR`
per frame. The CB runs scatter (uploads dirty TRS into SoT), the dirty
`fill_buffer(0)` clears, the per-camera block `copy_buffer`s, `signal_cs`,
then the dual-pass occlusion cull + render sequence (see the row above), then
blit. Increment `next_signal_expected` after submit so the next frame's poll
knows the new target. No CB recording, no descriptor-set allocation, no buffer
allocation per frame in steady state.


- [transform-gpu](transform-gpu.md)
- [ADR-0001 custom swapchain](../ADR-0001-custom-swapchain.md)
- [ADR-0002 per-frame CB recording](../ADR-0002-per-frame-cb-recording.md)
