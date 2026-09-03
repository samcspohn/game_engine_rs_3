# GPU transform pipeline

The device-local source-of-truth buffers, the staging path that feeds
them, and the host/GPU handshake that gates a frame's writes.

A [`WorldTransformGpu`](../../crates/engine-render/src/transform_gpu.rs) owns
the **device-local SoT** ("source of truth") buffers — one per component
(`vec4` per entity slot for position / rotation / scale, sized to
`entity_capacity`, grown geometrically) — plus the three compute pipelines
(`scatter_cs`, `mvp_build_cs`, `signal_cs`), **and (post ADR-0003 Path C) the
single shared per-frame host-staging buffers (TRS triple + dirty bitmasks),
the three shared scatter descriptor sets, the shared scatter compute
secondary, the host-coherent `gpu_signal` u32 buffer + its descriptor set +
signal compute secondary, and a host-side `next_signal_expected` counter that
gates host writes against the previous frame's GPU scatter completion via a
busy-poll instead of a Vulkan timeline semaphore.** Per frame, in this order,
all inside the slot's pre-recorded primary CB: (1) **scatter** ×3 (one
dispatch per component) reads `staging_<comp>[i]` + `dirty_<comp>[i]` and
writes `sot_<comp>[i]` iff bit `i` is set; (2) three `vkCmdFillBuffer(0)`
clears re-zero the dirty bitmasks; (3) one `vkCmdCopyBuffer` per camera
promotes that camera's block into the device buffer its shaders read; (4)
**`signal_cs`** atomically increments `gpu_signal[0]` — vulkano auto-sync
makes this fire after every read of host-shared staging is done, so the host's
busy-poll on this counter wakes the moment it's safe to overwrite staging for
the next frame, even though the rest of the CB is still running; (5) the
camera's dual-pass occlusion cull + render sequence runs — see
[ADR-0005](../ADR-0005-dual-pass-occlusion-culling.md) for the full pass 1 →
Hi-Z build → pass 2 → history-update breakdown; **mvp_build** (pass 1) still
reads stable SoT pos/rot/scale + a stable `view_proj` the same way — the
camera's, since a world two cameras draw is drawn twice from two vantage
points. **Uniform staging→SoT paradigm:** mvp_build (and any future shader)
reads only stable SoT — it never touches a host-shared buffer. Host writes go
into staging; a per-frame compute/transfer pass promotes staging→SoT.
**Dirty-only sparse upload is live:** each frame the host first calls
`host_wait_for_previous_compute()` (busy-polls `gpu_signal[0]` with
`spin_loop` → `yield_now` → 100µs `sleep` fallback after ~1ms; returns
immediately on the first frame because the buffer is pre-zeroed), then drains
`TransformHierarchy::Dirty`'s three per-component `AtomicU32` bitmasks (atomic
`swap(0, Relaxed)`) directly into the **shared** staging triple + dirty
buffers in one parallel `numa_pool::parallel_for` walk (256-word tasks), and
writes each camera's block into that camera's staging; the host then submits
the FrameSlot primary CB and increments `next_signal_expected` so the next
frame's wait knows what value to poll for. Per-component masks mean a
pure-rotation frame writes zero pos/scale data on either CPU or GPU. The SoT
stores **local** TRS; `mvp_build_cs` composes world-space TRS by walking the
parent chain upward per slot, terminating at the root (slot 0, its own parent
— see [ADR-0009](../ADR-0009-hierarchy-root-entity.md)) or at
`MAX_PARENT_DEPTH`. Hoisting that walk into a shared, slot-indexed pass so
lights and UI anchors can read world transforms too is
[ADR-0007](../ADR-0007-global-transform-pass.md). **See
[`docs/ADR-0003`](../ADR-0003-shared-staging-with-compute-sync.md) for the
shared-staging refactor, the abandoned timeline-semaphore intermediates (Paths
A and B), and the final GPU-write early-wake design (Path C) that beats the
previous timeline version at every measured N (+36% at N=1, +27% at N=1M
static, +6% at N=1M animated) while keeping the ~144 MB VRAM saving at N=1M.**


- [ADR-0003 shared staging](../ADR-0003-shared-staging-with-compute-sync.md)
- [staging balancer](staging-balancer.md)
- [frame loop](frame-loop.md)
