# ADR-0005: Dual-Pass Temporal Hi-Z Occlusion Culling

**Status:** Accepted — landed.
**Date:** 2026-07-19
**Scope:** `crates/engine-render/src/camera.rs`, `crates/engine-render/src/lib.rs` (`build_frame_slot`), `crates/engine-render/src/transform_gpu.rs` (removed `mvp_build_set1`), `crates/engine-render/shaders/{mvp_build,mvp_build_pass2,cull_pass2_args,hiz_reduce_depth,hiz_reduce_mip}.comp`
**Related:** [ADR-0004](ADR-0004-instanced-indirect-draw.md) (Phase 2's "optionally against a Hi-Z buffer for occlusion" is what this ADR implements)

## Context

ADR-0004 landed GPU-side frustum culling: one compute dispatch per frame
tests every renderer's world bounding sphere against `view_proj` and
compacts visible instances into per-mesh-slot indirect draws. Frustum
culling alone still draws everything the *camera* can see, including
objects fully hidden behind other geometry — the next bottleneck ADR-0004
flagged for scenes with real depth complexity (buildings, terrain,
dense props).

## Decision

Add **temporal two-phase Hi-Z occlusion culling**, the standard GPU-driven
technique (Frostbite/Ubisoft "GPU-driven rendering pipelines", and
similar in most modern engines' culling pipelines):

1. **Pass 1 cull** (`mvp_build.comp`, extended): every renderer slot still
   gets the authoritative frustum test against *this frame's* `view_proj`.
   Frustum-visible slots additionally get an occlusion test against **last
   frame's** Hi-Z pyramid, reprojected with **last frame's** `view_proj`
   (an approximation — camera/objects may have moved). Not-occluded →
   draw now (unchanged compaction path). Possibly-occluded → append a
   compact record (world TRS + resolved mesh slot + material id + the
   already-computed world-space bounding sphere) to a device-side
   candidate list.
2. **Render pass 1** draws pass 1's visible set — this frame's first depth
   contribution.
3. **Hi-Z build** (`hiz_reduce_depth.comp` + `hiz_reduce_mip.comp`)
   max-reduces that depth into a full mip pyramid (`hiz_current`).
   MAX-reduction is the standard conservative choice: a texel records the
   *farthest* depth anything in its footprint reached, so "my nearest
   point is closer than the farthest thing recorded here" is the only
   safe basis for culling.
4. **Pass 2 cull** (`mvp_build_pass2.comp`) re-tests only the candidates,
   against *this frame's own* Hi-Z and *this frame's* `view_proj` — exact,
   not an approximation, since it's the same frame. Dispatched via
   `vkCmdDispatchIndirect`, sized to the live candidate count by a tiny
   helper (`cull_pass2_args.comp`) rather than the full renderer capacity,
   so quiet frames (few/no candidates) cost close to nothing.
5. **Render pass 2** draws pass 2's newly-visible set into the same
   (`Load`, not `Clear`) attachments.
6. **History update**: `hiz_current` and this frame's `view_proj` are
   copied into fixed-identity `hiz_prev` / `prev_view_proj` buffers so
   next frame's pass 1 sees them as "last frame's" data — **no descriptor
   set is ever rebuilt just because a frame elapsed**; only a capacity or
   extent change triggers a rebuild, preserving the reusable-secondary /
   rebuild-only-on-change paradigm every other camera resource already
   follows.

### Why this belongs on `RenderCamera`, not `WorldTransformGpu`

The Hi-Z pyramid is a function of one camera's depth buffer and view — a
second camera needs its own history. `RenderCamera` already owns the
depth attachment, the cull secondary, and the scene secondary, so
occlusion culling is a natural extension of state it already has, not a
new ownership boundary. The one piece that could have stayed on the
(today still camera-agnostic) `WorldTransformGpu` — `prev_view_proj` — was
deliberately moved to the camera too, for symmetry with the Hi-Z pyramids
and so the whole feature is forward-compatible with multiple cameras
without a later split. This did require removing `WorldTransformGpu`'s
old `mvp_build_set1` (a single-buffer set bound to the *current*
`view_proj` only) — pass 1's occlusion sub-test needs a richer set (current
VP + previous VP + previous Hi-Z), and since that's per-camera data, the
whole set moved to `RenderCamera::build_occlusion_set`.

### Why independent MVP/indirect buffers per pass

Pass 2's compacted output could not reuse pass 1's per-mesh-slot regions:
the two passes' draws are two separately pre-recorded, `SimultaneousUse`
command buffers, and pass 1's draw has already executed (reading
`instance_count`) before pass 2 even starts appending. Sharing one region
would need the two draws to agree on non-overlapping sub-ranges of a
count only known after both dispatches run. Independent buffers (built
from the *same* [`DrawPlan`] — both need capacity for the same worst case,
"every instance of this mesh slot draws via this pass", since an instance
is drawn by exactly one of the two passes) sidestep this at the cost of
roughly 2× the per-camera MVP/indirect memory.

### Why the candidate list isn't a bitmask, and why it uses `dispatch_indirect`

The candidate list follows this codebase's existing "count-in-buffer"
streamed pattern (`gpu_renderers_scatter.comp` / `parent_scatter.comp`):
pass 1 atomically appends records + a live count. Unlike those, pass 2 is
dispatched via `vkCmdDispatchIndirect` (sized to `ceil(live_count / 64)`,
built by `cull_pass2_args.comp`) rather than a fixed-capacity dispatch
with a per-invocation early-out — deliberately, since a full
renderer-capacity-sized dispatch every frame (even one that mostly
early-outs) is real, avoidable work at scale, and this is exactly the
kind of `O(N)`-every-frame cost this codebase has consistently designed
around elsewhere (the `DrawPlan` rebuild, the parent-chain-walk note in
`mvp_build.comp`, etc.).

### Known approximations (documented in-shader, same convention as the
existing "planned follow-up" comments in `mvp_build.comp`)

- **Projected footprint**: the occlusion test uses the bounding sphere's
  world-space AABB corners projected into clip space, not the tighter
  analytic sphere→ellipse projection. Simpler, slightly conservative
  (over-estimates the screen footprint, never under-estimates).
- **Behind-eye-plane candidates**: if any AABB corner has `clip.w <= 0`,
  the occlusion test bails out and treats the instance as visible —
  a stale reprojection (pass 1) or an object straddling the near plane
  (either pass) can't safely bound that corner's screen position.
- **`hiz_current` only reflects pass 1's depth.** Pass 2's draws land in
  the real depth attachment but are not re-folded into the Hi-Z pyramid
  that becomes next frame's `hiz_prev` — rebuilding Hi-Z a second time
  after pass 2 would close this gap at roughly double the per-frame Hi-Z
  build cost. An object confirmed only via pass 2 this frame won't help
  occlude anything next frame until pass 1 itself draws it (typically the
  very next frame). Deferred as a follow-up if profiling shows the
  steady-state candidate count doesn't stay small.

## Consequences

### Wins

- Scenes with real depth complexity draw substantially fewer fragments —
  occluded geometry never reaches the rasterizer.
- Pass 2's `dispatch_indirect` sizing means the occlusion re-test's cost
  scales with *disocclusion churn* (objects becoming newly visible), not
  scene size — a mostly-static camera in a mostly-static scene pays
  almost nothing for pass 2.
- No new per-frame descriptor-set rebuilds: the Hi-Z ping-pong is realized
  via fixed-identity buffers/images mutated in place (`history_update_secondary`'s
  `copy_image`/`copy_buffer`) rather than swapping which resource a
  descriptor set points at.

### Costs

- Roughly 2× the per-camera MVP/indirect memory (independent pass 1/pass 2
  buffers).
- Two Hi-Z pyramid images per camera (`hiz_current`, `hiz_prev`), each with
  a full mip chain — modest (single-channel `R32_SFLOAT`, roughly 1.33× a
  half-resolution image's footprint per pyramid).
- Two `begin_rendering`/`end_rendering` scopes per frame instead of one
  (Hi-Z build needs the depth attachment out of attachment layout between
  passes).
- `WorldTransformGpu::sot_view_proj` needed `TRANSFER_SRC` added to its
  usage flags (previously only `STORAGE_BUFFER | TRANSFER_DST`), and its
  cross-frame read pattern (written frame N, read again in frame N+1
  without an explicit GPU-GPU barrier) is the same trust-the-queue's-
  submission-order approach this codebase already relies on for the SoT
  buffers' cross-frame persistence — not a new risk class, but worth
  naming since it's easy to assume Vulkan guarantees this for free (it
  doesn't, by the letter of the spec; this codebase already depends on it
  working in practice on the tested driver).

## Verification

No automated test harness exercises the render path (same as every prior
render ADR). Verified by running `test-game --shapes 2000` and `--shapes
5000` under `VK_LAYER_KHRONOS_validation` with
`VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT` explicitly
enabled — thousands of frames at each scale, zero validation or
synchronization-validation errors/warnings, stable frame time and VRAM.
This exercises every new synchronization surface (the depth
attachment↔sampled-image layout transitions around the Hi-Z build, the
two `begin_rendering` scopes sharing one depth image, the
`dispatch_indirect` argument buffer, the mip-chain image copy) but does
**not** visually confirm the occlusion test itself is culling/revealing
the geometry a human would expect — no screenshot tooling was available
in the verification environment. Recommend an interactive pass (orbit the
camera around dense/occluding geometry, watch for pop-in) before treating
the occlusion *behavior* — as opposed to its Vulkan-usage correctness —
as fully proven.

## Revisit if

- Profiling shows pass 2's candidate count doesn't stay small in
  steady-state scenes (the `hiz_current`-only-reflects-pass-1 gap above
  would be the first thing to fix).
- A second camera is added — `prev_view_proj`/Hi-Z are already per-camera,
  but `WorldTransformGpu::sot_view_proj` (the *current*-frame VP pass 1/2
  both read) is still world-shared/single-camera; that pre-existing
  assumption, not anything from this ADR, would need to change first.
- The AABB-corner footprint projection proves too conservative (culls too
  little) at high object counts — the tighter analytic sphere→ellipse
  projection is the documented follow-up.
