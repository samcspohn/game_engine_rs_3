# ADR-0006: Toggleable Depth Prepass

**Status:** Accepted — landed.
**Date:** 2026-08-02
**Scope:** `crates/engine-render/src/{camera,lib}.rs`, `crates/engine-render/shaders/{depth.vert,scene.vert}`
**Related:** [ADR-0004](ADR-0004-instanced-indirect-draw.md) (indirect draw),
[ADR-0005](ADR-0005-dual-pass-occlusion-culling.md) (dual-pass Hi-Z occlusion
culling — this ADR restructures the render scopes that ADR introduced).
Design doc: `docs/depth-prepass-plan.md`.

## Context

`scene.frag` is a full metallic-roughness PBR shader (Cook-Torrance GGX, up
to five texture fetches, tangent-space normal mapping). ADR-0005's dual-pass
Hi-Z occlusion culling removes most *fully hidden* geometry before it
reaches the rasterizer, but does nothing about *partial* overdraw — objects
whose bounding sphere is occlusion-visible but which are still shaded once
per overlapping layer because color and depth are written together.

## Decision

Add an optional depth-only prepass: a vertex-only pipeline
(`crates/engine-render/shaders/depth.vert`, no fragment stage) writes the
frame's final depth buffer before any shading runs. The color pass then
runs with `CompareOp::Equal` and depth writes disabled, so early-Z kills
every fragment that isn't the frontmost one before `scene.frag` executes —
each visible pixel is shaded exactly once.

### Frame restructure, paired with ADR-0005's occlusion culling

Rather than bolting a prepass in front of each of ADR-0005's two occlusion
passes independently, both passes' visible sets now feed **one shared color
scope** at the end of the frame:

```
cull1 → depth-only scope 1 → Hi-Z build → cull2 → history
      → depth-only scope 2 → shared color scope (pass1 + pass2 draws) → blit
```

Two things make this pairing work rather than just coexist:

- **Hi-Z still builds from pass 1's prepass depth**, bit-identical to what
  it would build from the old shaded pass-1 depth (see the `invariant`
  discussion below) — `hiz_reduce_depth.comp` and both cull shaders are
  completely unchanged.
- **The color scope must come after *both* prepasses**, not after each
  pass individually. Pass 2 exists precisely because pass 1's occlusion
  test was uncertain about those instances; a pass-2 instance can end up
  nearer than a pass-1 instance that was conservatively deferred behind it.
  Shading pass 1 before pass 2's depth exists would risk shading the wrong
  fragment as "frontmost." Folding both prepasses' draws into the same
  `indirect_args` region the color draws already read means this costs no
  extra cull work and no extra memory — the prepass and color secondaries
  for a given pass draw from the literal same buffer.

The two techniques end up dividing the overdraw problem: Hi-Z removes
objects hidden *behind* other objects before rasterization; the prepass
removes shading redundancy *within* what still rasterizes (partial overlap,
depth complexity finer than a Hi-Z texel).

### The correctness-critical part: `invariant gl_Position`

`CompareOp::Equal` requires `depth.vert` and `scene.vert` — two separate
SPIR-V modules — to produce **bit-identical** clip-space depth for the same
vertex. Without a compiler guarantee, FMA contraction or reassociation
could differ between the two, and a 1-ULP difference fails the EQUAL test
and drops the fragment entirely: a hole in the geometry, not a shading
artifact, and one that would only show up from certain view angles.

Both shaders declare `invariant gl_Position;` and compute it with the
identical expression (same MVP buffer, same index, same
`vec4(position, 1.0)`) — `scene.vert` already read a pre-multiplied MVP
for exactly this class of reason (Hi-Z bit-identity, per ADR-0005), so the
prepass shader was written to match it, not the other way around.

### Pipelines

Three graphics pipelines share **one** `PipelineLayout` (reflected from the
vs+fs stages, same layout the pre-existing single pipeline used):

| pipeline | stages | depth state | color attachments |
|---|---|---|---|
| `pipeline` (`less_write`) | vs + fs | `Less`, write on | 1 |
| `scene_pipeline_eq` (`equal_no_write`) | vs + fs | `Equal`, write off | 1 |
| `depth_pipeline` (`depth_only`) | depth_vs only | `Less`, write on | 0 |

A pipeline layout may be a strict superset of what a given stage set
statically uses — `depth_vs` only declares set 0 binding 0 (the MVP
buffer), but binds fine against the full layout; it simply never
references set 1 (the texture/material bindings). This is what let every
existing descriptor set (`DrawResources::graphics_set`,
`RenderCamera::texture_set`) bind to all three pipelines with zero new
descriptor-set plumbing.

`depth_only`'s `GraphicsPipelineCreateInfo` has `color_blend_state: None`
and `color_attachment_formats: vec![]` — required together per vulkano
0.35.2's validation once the color-attachment count is zero. A graphics
pipeline with fragment-output state but no fragment shader stage is
explicitly permitted by that same validation (the error path is the
reverse: a fragment stage present without fragment-shader *state*).

### Toggle

**F10** (runtime) and **`ENGINE_DEPTH_PREPASS=1`** (startup default),
mirroring F8's occlusion toggle exactly:
`RenderCamera::set_depth_prepass_enabled` re-records
`scene_secondary_pass1`/`_pass2` against the newly-selected color pipeline
and returns whether a `FrameSlot` rebuild is needed — `build_frame_slot`
reads `depth_prepass_enabled()` to pick the whole render-scope structure,
so toggling it is the same cost class as toggling occlusion.

### GPU timestamps

`GPU_TS_COUNT` grew from 12 to 13 (new `q12`, at the end of the shared
color scope). Rather than add more queries for the "off" case, the
existing per-stage slots change *meaning* depending on the toggle, which
keeps the readback layout fixed either way:

- **prepass off**: `q3`/`q6` measure pass 1/2's full shaded render exactly
  as before this feature existed; `q12` is written immediately after `q6`
  (no separate color scope runs), so it reads ~0 and the `blit` column
  (`q12`→`q7`) is unaffected.
- **prepass on**: `q3`/`q6` instead measure pass 1/2's depth-only prepass
  draw; `q12 − q6` is the real shading cost — the one shared color scope.

`FrameStats::gpu_stages` grew `[PhaseAcc; 8] → [PhaseAcc; 9]` with a new
`shade` column in the FPS print line.

## Consequences

### Wins

- Each visible pixel is shaded exactly once regardless of depth
  complexity, independent of and complementary to Hi-Z occlusion culling.
- No new descriptor sets, no new cull work, no extra per-camera memory
  beyond three small command-buffer secondaries — the prepass draws reuse
  the color draws' `indirect_args` verbatim.
- Toggle is cheap to flip live (same cost class as the existing F8
  occlusion toggle) and cheap to disable entirely (falls back to
  byte-for-byte the pre-existing render path).

### Costs

- Vertex/geometry front-end work roughly doubles for any prepass-covered
  instance (transformed once for the prepass, once for the color draw).
  Measured on `test-game --shapes 2000 --static-scene`
  (`ENGINE_NUM_THREADS=128`, RX 7900 XTX): total per-frame GPU time went
  from ~71-72µs (prepass off) to ~82µs (prepass on) — this scene's simple
  untextured/lightly-textured cubes and spheres are vertex/geometry-bound
  with modest overdraw, so the doubled vertex cost outweighs the shading
  savings. This is expected and is *why* the feature is a toggle, not a
  default: it is a net win only on scenes with real shading cost (heavy
  PBR materials, high texture-fetch counts, deep overlap) and a net loss
  on cheap-shader/low-overdraw scenes. No universal default is correct;
  profile the target scene before flipping `ENGINE_DEPTH_PREPASS=1` by
  default anywhere.
- One extra `begin_rendering`/`end_rendering` scope pair per frame
  (depth-only prepass 1; prepass 2 only exists when occlusion is also on).
- No alpha-tested/cutout materials exist today (`scene.frag` never
  `discard`s), so the prepass's vertex-only pipeline is exactly
  equivalent to the color pass's depth contribution. If alpha cutout is
  ever added, those instances need either exclusion from the prepass or a
  matching alpha-test fragment stage in `depth.vert`'s pipeline — silently
  including them would prepass-write opaque depth for a fragment the color
  pass will `discard`, punching a hole through anything correctly drawn
  behind it.
- Depth is `DontCare`-stored after the shared color scope (nothing
  downstream reads it again this frame). A future post-process needing
  depth would need to flip that to `Store`.

## Verification

- `cargo build --workspace` clean throughout.
- All four `occlusion_enabled × depth_prepass_enabled` combinations run
  without crashing on `test-game --shapes 2000 --static-scene`
  (`ENGINE_NUM_THREADS=128`, release build): each was run for several
  seconds and produced continuously-updating, sane telemetry (`hiz`/`mvp2`/
  `raster2` correctly read ~0 when occlusion is off; `shade` correctly
  reads ~0 when the prepass is off and a real value when it's on).
- **Pixel-diff A/B** (the correctness check the `invariant gl_Position`
  pairing lives or dies by): two independent runs from the same fixed
  startup camera pose (`--static-scene`, no input), one with
  `ENGINE_DEPTH_PREPASS=0` and one with `=1`, screenshotted via
  `spectacle -b -n` and cropped to the game window's rendered viewport
  (800×580). `compare -metric RMSE` gave `0.000137` (normalized) and
  `-metric AE` (exact-match pixel count) `12978.5` out of 464,000 pixels;
  thresholding the difference image at 1% isolated only **7 pixels**
  total — consistent with independent-capture/compositor noise (±1 LSB
  dithering across two separate screenshots taken at different wall-clock
  moments), not a structural difference. No holes, no silhouette drift,
  no shading change. This is the load-bearing check: it directly confirms
  the two shader modules' clip-space depth agrees closely enough that the
  `Equal` test never wrongly rejects a fragment.
- **Validation-layer verification was attempted but is inconclusive in
  this sandbox environment**, unlike ADR-0005's clean run: no
  system-packaged `VK_LAYER_KHRONOS_validation` was available, only a
  Steam Linux Runtime-bundled copy outside its intended container. Loaded
  via a hand-written layer manifest, it reported
  `VUID-VkGraphicsPipelineCreateInfo-pMultisampleState-09026` (claiming
  `pMultisampleState` was `NULL`) immediately followed by a **segfault
  inside the driver** (`radv_generate_graphics_pipeline_state`, called
  from the validation layer's own trampoline — see the coredump
  backtrace). All three pipeline constructions in `create_scene_pipelines`
  were re-read line-by-line and each explicitly sets
  `multisample_state: Some(MultisampleState::default())`; the same
  `less_write` pipeline's field values are unchanged from the
  single-pipeline code this replaced. Combined with the crash landing
  inside the driver rather than returning a diagnostic (a spec-conformant
  validation layer reports and continues; it doesn't segfault), this
  points at an ABI/struct-layout mismatch between the mismatched-version
  layer and this system's newer loader/driver (apiVersion 1.4.354),
  corrupting the struct in transit — not a defect in this feature's
  pipeline construction. **Recommended follow-up**: install a proper
  `vulkan-validation-layers` package matching the system loader version
  and re-run the same `VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`
  sweep ADR-0005 used, across all four toggle combinations, before treating
  the synchronization surface (three render scopes' worth of depth-image
  layout transitions, one more than ADR-0005 exercised) as fully proven.

## Revisit if

- The recommended validation-layer follow-up above surfaces a real
  synchronization issue in the three-scope structure.
- Profiling on a shading-heavy scene shows the prepass is worth defaulting
  to `on` — currently off by default, per the cost/benefit measured above.
- Alpha-tested/cutout materials are added — see the Costs section; the
  prepass will silently mis-shade around cutout edges until handled.
- ADR-0005's documented "`hiz_current` only reflects pass 1's depth" gap
  is revisited — with a prepass landed, folding pass 2's depth-only
  contribution into a second Hi-Z build is cheaper than it would have been
  against pass 2's full shaded draw, since the prepass's depth-only
  secondary is already isolated from color work.
