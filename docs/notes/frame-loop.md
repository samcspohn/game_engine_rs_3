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
CB**: the FrameSlot primary, which runs `scatter_secondary` → the `At::Upload`
steps → `ui.scatter_secondary` → dirty `fill_buffer` ×3 → the per-camera block
promotions → `signal_secondary` → the `At::Compute` steps → `cull_secondary`
(pass 1, every world) → `begin_rendering(Clear)` → per-world `scene_pass1` →
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

## The frame

`RenderApp::about_to_wait` (`lib.rs`) is the frame, one call per phase, and
`frame.rs` holds the device-side phases as `RenderContext` methods. In order:

1. **Input**: `dispatch_input` folds the pointer, keys and UI handlers, then
   `gizmo::update`. Everything that reads input has decided before any
   component runs.
2. **Boundary**: `retire_worlds`, `worlds::apply_pending`. The only window
   where `&mut World` is sound.
3. **Simulate**: `worlds::sweep_all`, which is every component's `update` and
   then each world's `System`s. Then `adopt_worlds`, for a world a camera
   named this frame.
4. **Acquire**: wait for this image's fence, then read back its last
   timestamps.
5. **Prepare**: `layout_ui`, `prepare` (each world's transforms and steps,
   the asset stores and the UI grow to fit), `sync_cameras`, `drive_cameras`,
   `debug_keys`. Each returns a `Rebuild` (`None < Plans < FrameSlots <
   Cameras`), folded with `max`, and the frame primaries are re-recorded
   once if the result asks for it.
6. **Host wait**: `TransformGpuShared::host_wait` busy-polls `gpu_signal`
   until the frame that last used this staging slot has released it (F7
   waits for the whole previous frame instead).
7. **Stage**: `stage(slot)` writes every host-visible buffer that slot's
   primary reads: camera blocks, each world's TRS drain and steps, the UI.
8. **Submit**, then `TransformGpuShared::end_frame` moves the staging slot
   and the expected signal value on together.

The staging slot is owned by `TransformGpuShared` and passed *in* to every
`stage` and `record`. Nothing else keeps a slot counter, so a camera or
world made mid-run cannot write a slot no frame reads.

## Adding a step

A per-world GPU feature (particles, lights) implements `step::GpuStep`:

- `prepare(&StepCx) -> Rebuild` runs before the host wait. Grow buffers here,
  and say what that invalidated.
- `stage(&StepCx, slot)` runs after the host wait. Write this frame's host
  staging for `slot`.
- `record(At, slot, &mut Primary)` runs **only when the frame primaries are
  rebuilt**. It records into the primary at `At::Upload` (before `signal_cs`,
  because it reads host staging) or `At::Compute` (after every upload, before
  any camera culls).

Command buffers are recorded once and replayed (ADR-0002), so a step's
per-frame work has to reach the GPU as staging writes and indirect dispatch
args, never as a re-record. `GpuRenderers` is the reference: it takes its
world's spawn records in `prepare`, writes them and a `dispatch_indirect`
count in `stage`, and records one scatter at `At::Upload`.

To plug a step in, add a field on `WorldRender` and an entry in its `steps` /
`steps_mut` lists. Everything else in the frame is unchanged. `StepCx` carries
the world's `WorldTransformGpu`, because every per-world buffer is indexed by
transform slot and its capacity follows the SoT's.

A CPU-side per-world step (physics) is an `engine_core::System`, added with
`World::add_system`. It runs after every component's `update`, on simulating
worlds only.

## The frame primary

`frame_slot.rs` records one primary per `(swapchain image, staging slot)`:

- **Upload:** TRS scatter (every world, stage-major), the `At::Upload` steps,
  UI scatter, dirty-mask clears, camera-block promotions, `signal_cs`.
- **Compute:** the `At::Compute` steps.
- **Per camera:** pass-1 cull, pass-1 draw, Hi-Z, pass-2 cull and history,
  pass-2 draw, overlay.
- **Present:** the blit, then the UI.

## GPU timestamps

`frame_slot::ts` names the queries. Each is `BOTTOM_OF_PIPE`, so it marks when
all prior commands drained. Only the first camera stamps its stages. With
occlusion off (F8), the skipped stages are stamped back to back and read as
~0, which keeps the readback layout fixed.

`SEAM` is a `BOTTOM_OF_PIPE` write at the start of the CB. Nothing makes frame
N wait on frame N−1's raster, so `START` (top of pipe) latches while the
previous frame is still drawing. `SEAM − START` is that drain, and the upload
block is measured from `SEAM`. Without `SEAM`, the scatter appeared to grow
~330 µs when raster went from 7 µs to 680 µs. `TRS`, `STEPS` and `CLEARS`
subdivide the upload block. They exist because the first-used staging slot
once ran ~13× slower than the other on identical work.

## Knobs

- `ENGINE_WAIT_MODE=frame` / F7: wait for the previous frame's fence instead
  of the mid-CB signal. This gives up CPU/GPU overlap in exchange for
  uncontended staging.
- `ENGINE_STAGING_POOL=1`: run the TRS drain on a pool confined to the GPU's
  NUMA node. **It measured worse**: scatter 318 µs, but it breaks the
  worker-to-range sharing with the sweep, `sim_update` goes 240 → ~800 µs at
  any pool width, and the frame drops from 803 to 663 FPS.
- `ENGINE_CULL_AWAY=1`: lock the cull frustum 1e7 units off-scene at startup,
  so pass 1 draws nothing while staging and scatter stay identical.
- `ENGINE_SCATTER_TRACE=1`: every frame's raw upload-block time, per slot.
- `ENGINE_UI_TRACE=1`: the UI's dirty-word count on any frame that uploads.

- [transform-gpu](transform-gpu.md)
- [ADR-0001 custom swapchain](../ADR-0001-custom-swapchain.md)
- [ADR-0002 per-frame CB recording](../ADR-0002-per-frame-cb-recording.md)
