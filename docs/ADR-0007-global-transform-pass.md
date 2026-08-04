# ADR-0007: Global Transform Composition as a Shared Pass

**Status:** Proposed
**Date:** 2026
**Scope:** `crates/engine-render/shaders/{global_transform.comp,mvp_build.comp}`, `crates/engine-render/src/transform_gpu.rs`, one new block in `build_frame_slot`
**Related:** [ADR-0003](ADR-0003-shared-staging-with-compute-sync.md) (staging → scatter → SoT), [ADR-0005](ADR-0005-dual-pass-occlusion-culling.md) (the cull kernel that owns the walk today), [ADR-0008](ADR-0008-ui-integration.md) (first external consumer)

## Context

The transform SoT holds **local** TRS. World-space TRS is composed inside
`mvp_build.comp`, which walks the parent chain upward per slot:

```glsl
// mvp_build.comp:380
pos = u_pos.p[i].xyz;  rot = u_rot.q[i];  scl = u_scl.s[i].xyz;
uint parent = u_parents.parent[i];
for (uint depth = 0u; parent != NO_PARENT && depth < MAX_PARENT_DEPTH; ++depth) {
    pos = pp + quat_rotate(pq, pos) * ps;
    rot = quat_mul(pq, rot);
    scl *= ps;
    parent = u_parents.parent[parent];
}
```

That shader's own comment already flags this as provisional: *"This walk is
the straightforward kernel; a level-ordered global composition pass is the
planned faster replacement."*

Three properties of the current arrangement matter:

1. **The walk is `O(depth)` per slot**, re-derived every frame, inside the
   cull kernel's hot loop. A five-deep glTF hierarchy pays five dependent
   buffer reads per node per frame, every frame, whether or not anything
   moved.
2. **It runs per *renderer* slot** — only transforms carrying a
   `MeshRenderer` reach it.
3. **Its output is not addressable.** The composed result lands in
   `candidate_list` (model matrices) and `InstXform`, both **compacted per
   visible instance**. Pass 2 reuses pass 1's candidates, so the walk itself
   happens once per frame — but there is no buffer anywhere that answers
   "what is the world position of transform slot `i`".

Point 3 is the one that has become blocking. Every GPU-side consumer of
world position that is not the mesh draw needs exactly that query:

* **World-anchored UI** ([ADR-0008](ADR-0008-ui-integration.md)) — project
  an entity's world position to screen and write a `ui_group` offset.
* **Lights** — cull and bin by world position; a light attached to a moving
  entity has its position defined by the hierarchy, not by its own record.
* **Particle emitters** — spawn in the emitter's world frame.

Each of these would otherwise re-implement the parent walk against the same
buffers, and each would carry its own copy of `MAX_PARENT_DEPTH`, its own
composition-order comment, and its own opportunity to diverge from
`TransformHierarchy::get_global_transform`. Three copies of a subtle kernel
is how composition-order bugs get shipped.

## Decision

**Extract composition into its own pass** — `global_transform.comp` —
running after the scatter and before `mvp_build`, writing **slot-indexed**
device-local world TRS:

```
sot_global_pos[slot]    vec4   // xyz = world position
sot_global_rot[slot]    vec4   // world rotation quaternion
sot_global_scale[slot]  vec4   // xyz = world scale
```

`mvp_build.comp` then reads these instead of walking. Every future GPU
component reads the same three buffers. Composition order lives in exactly
one kernel.

The buffers mirror the local SoT's slot indexing and `ComponentSlot`
layout, so they grow on the same world-capacity axis and need no separate
capacity logic.

### Stage 1: extract, unchanged

Move the loop verbatim into `global_transform.comp`, dispatched over the
transform count. `mvp_build.comp` loses the loop and gains three reads.
Behaviour is identical by construction — the same arithmetic on the same
inputs — which makes this stage bisectable against a visual diff rather
than something to reason about.

This alone unblocks ADR-0008, lights and emitters. It is not yet faster.

### Stage 2: level ordering

Once composition is its own pass, the walk is removable. Process slots
**grouped by hierarchy depth**, shallowest first: when level `L` runs, every
level `< L` already holds *global* TRS, so each node composes against its
parent's finished value in **one** read and never loops.

```
pos = parent_global_pos + quat_rotate(parent_global_rot, local_pos) * parent_global_scale
rot = quat_mul(parent_global_rot, local_rot)
scl = local_scale * parent_global_scale
```

`O(N·depth)` → `O(N)`, and the dependent-read chain that stalls the current
kernel disappears.

This needs two host-maintained arrays:

* **`level[slot]`** — depth in the hierarchy. Cheap to maintain: a new
  transform's level is its parent's plus one, and the existing
  `drain_parent_updates` stream is exactly the set of slots whose level (and
  whose subtree's levels) may have changed.
* **`level_order[]`** — slot ids bucketed by level, with a `level_offsets[]`
  table. A counting sort over `level[]`, rebuilt **only on structural
  change**, which is the same discipline the pre-recorded command buffers
  already follow.

Dispatch is `MAX_LEVELS` pre-recorded `dispatch_indirect` calls with a
compute→compute barrier between each, args read from a device buffer.
Unused levels resolve to `(0,0,0)` and cost nothing — the same trick
`ui_build_args.comp` already uses to keep a static CB in front of a dynamic
count. Hierarchies are shallow (glTF scenes run 5–10 levels), so the barrier
count is small and fixed.

### Where it lands in the frame

After the scatter block (local TRS and `sot_parents` must be current — note
`parent_scatter.comp` feeds the latter) and before `mvp_build` pass 1. One
compute→compute barrier on the three new buffers separates it from its
consumers.

Pass 2 is unaffected: it already consumes pass 1's candidate list rather
than composing anything itself.

## Consequences

### Wins

* **One kernel owns composition order.** The rule stated in
  `mvp_build.comp`'s comment and implemented in
  `TransformHierarchy::get_global_transform` now has exactly one GPU
  implementation, and new consumers physically cannot diverge from it.
* **World TRS becomes addressable by slot** — the prerequisite for every
  non-mesh GPU component. This is the actual reason to do it now.
* **The cull kernel gets shorter and flatter.** `mvp_build` loses a
  variable-length dependent-read loop from its inner path.
* **Stage 2 makes composition `O(N)`** and removes the per-node
  serialization that a pointer-chase imposes on a wide GPU.
* **`MAX_PARENT_DEPTH` stops being a silent correctness cliff.** Today a
  65-deep chain composes *wrongly* — the loop exits and the node renders at
  a partially-composed transform. Level ordering has no depth bound at all.

### Costs

* **Composition now runs for every transform, not just renderer slots.**
  glTF scenes carry many mesh-less intermediate nodes, so this is a real
  increase in composed nodes — though those intermediates are precisely what
  the current walk already traverses repeatedly, so stage 2 should still net
  out ahead. Worth measuring on the `--glb` path specifically, since that is
  where the ratio is worst.
* **Three new device buffers sized by world capacity.** 48 B/transform at
  1M transforms is 48 MB. Not free, and it grows on the same axis as the
  existing SoT.
* **`level[]` and `level_order[]` are new host state that reparenting must
  maintain**, including the subtree-wide level shift a reparent implies.
  Getting this wrong is a silent wrong-transform bug, so it needs a test
  that reparents a deep subtree and checks every descendant's level.
* **One more barrier** in an already heavily-barriered primary.

### Caveats

* Stage 1 must land and be verified visually before stage 2 touches
  ordering. The two are independently reversible only in that order.
* Level ordering assumes an acyclic hierarchy. A cycle currently produces a
  bounded-garbage transform via `MAX_PARENT_DEPTH`; under level ordering a
  cycle has no consistent level assignment at all, so the host's level
  maintenance is where a cycle must be rejected — loudly, at reparent time.

## Implementation plan

**Stage 1 — extract.** `global_transform.comp` with the existing walk;
three slot-indexed buffers; `mvp_build.comp` reads instead of walks. Verify
by visual diff against the current build on `--shapes` and `--glb` scenes.

**Stage 2 — level ordering.** Host `level[]` + `level_order[]` +
`level_offsets[]`, counting-sorted on structural change; `MAX_LEVELS`
indirect dispatches; the loop deleted. Measure the scatter→cull span before
and after on a deep `--glb` hierarchy.

## Revisit if

* **Composition shows up in the GPU timings on mostly-static scenes** —
  then make it dirty-driven. The scatter already knows which transforms
  changed; what is missing is downward propagation, since a moved parent
  invalidates its whole subtree. The `has_children` bitmask and a
  subtree-range encoding are the ingredients. This is the natural third
  stage and is deliberately not in scope: it is only a win once the constant
  factor is already low, and it reintroduces exactly the dirty-propagation
  complexity that stage 2 exists to keep out of the kernel.
* **A consumer needs world *matrices* rather than TRS** — then add a fourth
  buffer rather than making every consumer recompose. `InstXform` already
  demonstrates the packing tradeoff.
* **Hierarchies get deep enough that `MAX_LEVELS` dispatches dominate** —
  then a single persistent-workgroup kernel with device-side level barriers,
  which trades portability for dispatch count.
