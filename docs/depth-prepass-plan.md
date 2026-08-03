# Plan: Toggleable Depth Prepass (Z-Prepass)

**Status:** Proposed — not implemented.
**Scope:** `crates/engine-render/shaders/{depth.vert,scene.vert}`,
`crates/engine-render/src/{shaders,camera,lib}.rs`
**Related:** [ADR-0004](ADR-0004-instanced-indirect-draw.md) (indirect draw),
[ADR-0005](ADR-0005-dual-pass-occlusion-culling.md) (dual-pass Hi-Z occlusion
culling — this plan restructures the render scopes that ADR introduced)

---

## 1. Goal

Add a depth-only prepass that lays down the frame's final depth buffer before
any shading runs, so `scene.frag` (metallic-roughness PBR + up to five texture
fetches per fragment) is invoked **exactly once per visible pixel**. The color
pass becomes `depth_compare == EQUAL` with depth writes disabled; every
occluded fragment is killed by early-Z before the fragment shader starts.

Toggleable at runtime, in the same style as the existing F8 (occlusion) / F9
(frustum lock) debug toggles, because a prepass is not unconditionally a win —
it roughly doubles vertex/raster front-end work in exchange for removing
shading overdraw, so the trade depends on the scene.

---

## 2. Where this lands in the current frame

Today's primary CB (`lib.rs::build_frame_slot`) is:

```
cull pass 1  →  render pass 1 (Clear, PBR, depth LESS+write)
             →  Hi-Z build (from pass-1 depth)
             →  cull pass 2 (indirect)  →  history update
             →  render pass 2 (Load, PBR, depth LESS+write)
             →  blit
```

Both render scopes shade with full PBR *while* depth is still being resolved,
so a fragment later overwritten by a nearer one has already paid for its
material fetches and BRDF.

### The pairing with Hi-Z occlusion culling

This is the part worth getting right, and it is what makes the restructure
better than "add a prepass in front of each of the two passes":

- The Hi-Z pyramid is built from **pass 1's depth attachment**. A depth-only
  prepass produces *bit-identical* depth content at a fraction of the cost, so
  Hi-Z quality is unchanged and its input gets cheaper. Nothing in
  `hiz_reduce_depth.comp` or either cull shader changes.
- Because occlusion culling already removes most hidden geometry *before*
  rasterization, the prepass's job is the residual overdraw that Hi-Z cannot
  see: partially-occluded objects, and depth complexity finer than a Hi-Z
  texel. The two techniques attack different halves of the same problem and
  compose.
- Crucially, the two cull passes' visible sets should feed **one shared color
  pass**, not two. Pass 2's draws are the ones Hi-Z was least sure about — they
  are exactly the fragments most likely to be overwritten. Shading them in
  their own scope before pass 1's depth is complete… is fine today (pass 1's
  depth *is* complete by then), but shading pass **1** before pass **2**'s
  depth exists is not: a pass-2 instance can be nearer than a pass-1 instance
  it was (wrongly, conservatively) deferred behind. So the single color scope
  must come after *both* prepasses.

### Proposed structure (prepass enabled, occlusion enabled)

```
cull pass 1
begin_rendering(depth only, Clear)          ← no color attachment at all
  depth_secondary_pass1                       (vertex-only pipeline)
end_rendering
Hi-Z build (from that depth — unchanged)
cull pass 2 (indirect)  →  history update
begin_rendering(depth only, Load)
  depth_secondary_pass2
end_rendering
begin_rendering(color Clear, depth Load)    ← ONE shading scope
  scene_secondary_pass1                       (EQUAL, no depth write)
  scene_secondary_pass2                       (EQUAL, no depth write)
end_rendering
blit
```

Three rendering scopes instead of two. The depth-only scopes bind **no color
attachment**, which is what lets the driver take its depth-only fast path
(double-rate depth on most AMD/NVIDIA parts) — the reason for keeping them
separate rather than folding the pass-2 prepass into the color scope.

### The other three toggle combinations

| occlusion | prepass | frame structure |
|---|---|---|
| on  | on  | as above |
| on  | off | today's path, unchanged |
| off | on  | cull1 → depth prepass 1 → color scope (pass-1 draws only) |
| off | off | today's path minus the occlusion block, unchanged |

---

## 3. The depth-only shader

New `crates/engine-render/shaders/depth.vert`:

```glsl
#version 450
layout(location = 0) in vec3 position;          // location 0 only
layout(set = 0, binding = 0) readonly buffer Matrices { mat4 mvp[]; } u_matrices;
invariant gl_Position;
void main() {
    gl_Position = u_matrices.mvp[gl_InstanceIndex] * vec4(position, 1.0);
}
```

**No fragment shader.** Verified against vulkano 0.35.2's validation
(`pipeline/graphics/mod.rs`: the `(false, true)` arm of the fragment-stage
match is explicitly permitted), so a vertex-only graphics pipeline builds. This
is the fastest form — zero fragment invocations, pure fixed-function depth
write. Fallback if a driver objects: an empty `depth.frag` with no outputs.

It declares only vertex attribute location 0, so
`GpuVertex::per_vertex().definition(&depth_vs_entry)` produces a vertex-input
state that fetches position only — normal/uv/tangent are never read. Stride
still comes from `GpuVertex`, so the same mega vertex buffer binds unchanged.

### The correctness-critical part: `invariant gl_Position`

`CompareOp::Equal` demands that `depth.vert` and `scene.vert` compute
**bit-identical** clip-space depth for the same vertex. They are separate
SPIR-V modules; without the `Invariant` decoration a compiler is free to
contract `mvp * vec4(position, 1.0)` into FMAs in one module and not the other,
or reassociate the dot products — and a 1-ULP difference turns into a **hole in
the geometry** (the fragment fails EQUAL and is never shaded, leaving the clear
color). This is the single most likely way this feature ships broken, and it is
intermittent and view-dependent, so it will not show up in a static screenshot
test.

Mitigations, all three:

1. `invariant gl_Position;` in **both** `depth.vert` and `scene.vert`
   (the latter is a one-line addition; harmless when the prepass is off).
2. Byte-identical source expression for `gl_Position` in both shaders — same
   buffer, same index, same `vec4(position, 1.0)` construction. `scene.vert`
   already reads the pre-multiplied MVP for exactly this class of reason
   (its comment: "so clip-space depth stays bit-identical to what the cull's
   Hi-Z was built against").
3. A `PREPASS_COMPARE_OP` constant in `lib.rs` defaulting to
   `CompareOp::Equal`, switchable to `CompareOp::LessOrEqual`. LEQUAL is
   strictly more forgiving — it differs from EQUAL only when the prepass depth
   came out *greater* than the color pass's, i.e. exactly the failure mode
   above — while still rejecting every genuinely-occluded fragment via early-Z.
   If holes ever appear in the field, this is the one-line escape hatch.

Both pipelines must also match in every state that affects which fragments are
generated: same viewport/depth range, `RasterizationState::default()` (no
culling, no depth bias, fill mode), same `MultisampleState`. Keep them
constructed side by side in one function so they cannot drift.

`scene.frag` has no `discard` and never writes `gl_FragDepth`, so the color
pass remains fully early-Z-eligible under EQUAL. Worth a comment in the shader
so a future alpha-cutout addition doesn't silently break it.

---

## 4. Pipelines

Three graphics pipelines, **one shared `PipelineLayout`**:

| pipeline | stages | depth state | color attachments |
|---|---|---|---|
| `pipeline` (existing) | vs + fs | `Less`, write on | 1 |
| `pipeline_eq` (new) | vs + fs | `Equal`, write **off** | 1 |
| `depth_pipeline` (new) | depth_vs only | `Less`, write on | **0** |

Building all three from `pipeline.layout().clone()` (reflected once from the
vs+fs stages) means every existing descriptor set — `DrawResources::graphics_set`
(set 0) and `texture_set` (set 1), both allocated from that layout's
`set_layouts()` — binds to all three without change. A pipeline layout that is
a superset of what a shader statically uses is legal, and the depth pipeline
simply never binds set 1.

Note for the depth-only pipeline: with `color_attachment_formats: vec![]`,
vulkano requires `color_blend_state: None` (verified in the same validation
source). Everything else mirrors `create_pipeline`.

Refactor `create_pipeline` → `create_scene_pipelines(device) -> (Arc<GraphicsPipeline>, Arc<GraphicsPipeline>, Arc<GraphicsPipeline>)`
returning `(less, equal, depth_only)`, and store all three on `RenderApp`
alongside the existing `pipeline` field.

---

## 5. `camera.rs` changes

`RenderCamera` gains:

- `depth_secondary_pass1: Arc<SecondaryAutoCommandBuffer>`
- `depth_secondary_pass2: Arc<SecondaryAutoCommandBuffer>`
- `depth_prepass_enabled: bool`

New `record_depth_secondary(...)` — a near-copy of `record_scene_secondary`
with:
- inheritance `CommandBufferInheritanceRenderingInfo { color_attachment_formats: vec![], depth_attachment_format: Some(CAMERA_DEPTH_FORMAT), .. }`
- binds `depth_pipeline` and set 0 only (no `texture_set`)
- same `bind_vertex_buffers` / `bind_index_buffer` / `draw_indexed_indirect`
  over the **same** `indirect_args` buffer as the corresponding color pass.

The shared `indirect_args` is the key simplification: the prepass and the color
pass draw the identical instance set from the identical buffer, both read-only,
so no extra barrier, no extra cull work, no extra memory. Pass 1's args are
written by `cull_secondary` before either reads them; pass 2's by
`cull_pass2_secondary`.

`CameraSceneResources` gains two fields:
```rust
pub scene_pipeline_eq: &'a Arc<GraphicsPipeline>,
pub depth_pipeline: &'a Arc<GraphicsPipeline>,
```
(`pipeline` stays as-is — it is also the source of the shared layout, so the
existing `scene.pipeline.layout()` / `scene.pipeline.device()` uses keep
working.)

A private helper picks the color pipeline:
```rust
fn color_pipeline<'a>(&self, scene: &'a CameraSceneResources<'a>) -> &'a Arc<GraphicsPipeline> {
    if self.depth_prepass_enabled { scene.scene_pipeline_eq } else { scene.pipeline }
}
```
called at every existing `record_scene_secondary` site — in `new`,
`ensure_current`, and `on_swapchain_resize`. Those three functions also gain
the two `record_depth_secondary` calls (extent-dependent viewport → both
rebuild paths must re-record them, same as the scene secondaries).

New toggle, mirroring `set_occlusion_enabled` exactly:
```rust
/// Returns whether anything changed — callers must rebuild the FrameSlots
/// iff true, since `build_frame_slot` reads this flag to decide the render
/// scope structure, and the color secondaries bake in which pipeline
/// (LESS+write vs EQUAL+no-write) they bind.
pub fn set_depth_prepass_enabled(&mut self, enabled: bool, scene: &CameraSceneResources<'_>) -> bool
```
It re-records `scene_secondary_pass1`/`_pass2` with the newly-selected color
pipeline (the depth secondaries are pipeline-invariant, so they are recorded
once and simply not executed when the prepass is off). Plus an
`depth_prepass_enabled()` accessor and `depth_secondary_pass1/2()` accessors,
matching the file's existing accessor block.

---

## 6. `lib.rs` changes

### `build_frame_slot`

Restructure per §2. Concretely, the pass-1 render scope becomes conditional:

- **prepass on** — `begin_rendering` with `color_attachments: vec![]` and the
  depth attachment `Clear`/`Store`; execute `depth_secondary_pass1`.
- **prepass off** — today's `Clear` color + `Clear` depth scope executing
  `scene_secondary_pass1`.

Inside the `occlusion_enabled` block, pass 2's render scope likewise becomes
either the depth-only `Load`/`Store` scope executing `depth_secondary_pass2`,
or today's `Load` color + `Load` depth scope executing `scene_secondary_pass2`.

Then, **only when the prepass is on**, a final shading scope:
`color_attachments: [Clear]` (the color image has not been touched this frame —
the prepasses had no color attachment), `depth_attachment: Load` /
`store_op: DontCare` (Hi-Z was built before this point and nothing downstream
reads depth again), executing `scene_secondary_pass1` and then
`scene_secondary_pass2` if occlusion is on.

Vulkano's auto-sync handles all of it from the secondaries' resource-usage
records — the depth image's attachment→sampled transition before the Hi-Z
build, and back to attachment for the pass-2 prepass, is the same mechanism
already in use today.

Update the long CB-structure comment at the top of `build_frame_slot`; it is
the map people actually read.

### Timestamps

`GPU_TS_COUNT: 12 → 13`, adding **q12** at the end of the shading scope.
Semantics of the existing slots shift when the prepass is on, which is cheaper
than adding two more queries and keeps the readback layout fixed:

| query | prepass off | prepass on |
|---|---|---|
| q3 (`raster1`) | pass-1 shaded render | pass-1 depth prepass |
| q6 (`raster2`) | pass-2 shaded render | pass-2 depth prepass |
| q12 (`shade`) | written back-to-back with q6 → reads ~0 | the single color scope |

`FrameStats::gpu_stages` grows `[PhaseAcc; 8] → [PhaseAcc; 9]`, the
`record_gpu_timestamps` call gets `delta(6, 12)`, and the gpu line gets a
`shade {}` column between `raster2` and `blit`. `delta(8, 7)` (frame total)
and the staging balancer's inputs are unaffected — q7 is still last. Update
`GPU_TS_COUNT`'s doc comment with the table above.

### Toggle plumbing

- **F10** toggles the prepass (F6/F7/F8/F9 are taken). Same shape as the F8
  block at `lib.rs:1866`: build a `CameraSceneResources`, call
  `set_depth_prepass_enabled`, set `need_frame_slot_rebuild` if it returns
  true. Both toggles funnel into the same rebuild at `lib.rs:1907`, so
  toggling both in one frame costs one rebuild.
- **`ENGINE_DEPTH_PREPASS=1`** sets the startup default (matching the
  `ENGINE_CULL_AWAY` / `ENGINE_STAGING_MODE` convention), so A/B benchmark runs
  are scriptable without keystrokes. Default **off** initially — flip to
  on-by-default only once measured on the real scenes.

---

## 7. Costs and known limitations (to document in the ADR)

- **Vertex work roughly doubles.** Every visible instance is transformed twice.
  For vertex-dense, low-overdraw scenes the prepass is a net loss. This is the
  entire reason it is a toggle rather than unconditional.
- **One extra `begin_rendering`/`end_rendering` pair** per frame, plus one
  extra depth-image layout transition round trip.
- **No alpha-tested or blended materials today.** `scene.frag` has no
  `discard`, so the prepass's vertex-only pipeline is exactly equivalent. If
  alpha cutout is ever added, those materials must be excluded from the prepass
  and drawn with the LESS+write pipeline in a separate scope, and the prepass
  will need its own fragment shader doing the same alpha test. Call this out in
  both `depth.vert`'s header and the ADR.
- **Depth is `DontCare`-stored** after the color scope. Fine today; a future
  post-process reading depth would need `Store`.

## 8. Follow-up this unlocks (not in scope)

ADR-0005's documented gap — "`hiz_current` only reflects pass 1's depth", so an
object first confirmed by pass 2 does not help occlude anything until pass 1
draws it next frame — becomes much cheaper to close with a prepass: a second
Hi-Z build after the pass-2 *depth prepass* (not after a full shaded pass) would
fold pass 2's contribution into next frame's `hiz_prev`. Still roughly doubles
Hi-Z build cost, so it stays gated on profiling showing a non-small
steady-state candidate count. Worth a line in the new ADR pointing back at
ADR-0005.

---

## 9. Implementation order

1. `depth.vert` + `invariant gl_Position;` in `scene.vert` + `shaders.rs`
   module. Build — confirms the vertex-only SPIR-V reflects cleanly.
2. `create_scene_pipelines` (three pipelines, one layout) + `RenderApp` fields
   + `CameraSceneResources` fields. Build — confirms the vertex-only
   `GraphicsPipeline::new` is accepted, the one real API risk.
3. `camera.rs`: depth secondaries, `depth_prepass_enabled`,
   `set_depth_prepass_enabled`, `color_pipeline` helper threaded through
   `new`/`ensure_current`/`on_swapchain_resize`.
4. `build_frame_slot` restructure + CB-structure comment.
5. Timestamps + telemetry column.
6. F10 + `ENGINE_DEPTH_PREPASS`.
7. Verify, then `docs/ADR-0006-depth-prepass.md` + an ADR-INDEX row.

## 10. Verification

Same bar as ADR-0005 (no automated harness covers the render path):

- `cargo build --workspace` clean.
- `test-game --shapes 2000` and `--shapes 5000` under
  `VK_LAYER_KHRONOS_validation` with
  `VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`, exercising all
  **four** occlusion×prepass combinations (toggle F8/F10 live during the run)
  plus a window resize in each — zero validation/sync errors.
- **Visual A/B is mandatory here**, not optional: screenshot with the prepass
  off and on from an identical camera pose and diff them. They must be
  pixel-identical (the EQUAL test changes *which fragments are shaded*, not
  *what they shade to*). Any difference is the `invariant` problem from §3, and
  the diff is what will catch it — a "looks fine" eyeball pass will not.
  Screenshot tooling in this sandbox: `spectacle -b -n -o <path>`.
- Perf A/B with `ENGINE_NUM_THREADS=128`, comparing the `gpu us` line's
  `raster1 / raster2 / shade / total` columns between
  `ENGINE_DEPTH_PREPASS=0` and `=1` on both a dense-occlusion scene and a
  low-overdraw one — the numbers that decide whether the default flips.
