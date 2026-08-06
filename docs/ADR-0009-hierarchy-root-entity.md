# ADR-0009 — The hierarchy owns a root entity

**Status:** Accepted, implemented.
**Supersedes:** the `NO_PARENT` sentinel introduced alongside the parent SoT
in ADR-0003.
**Related:** [ADR-0007](ADR-0007-global-transform-pass.md) (the composition
walk this changes the terminator of), [ADR-0008](ADR-0008-ui-integration.md)
(the hierarchy panel this exists to simplify).

## Context

`TransformMeta::parent` encoded "no parent" as `u32::MAX`. That sentinel was
duplicated in **three** places that had to be kept in agreement by hand:

| Site | Was |
|---|---|
| `engine_core::transform::NO_PARENT` | `u32::MAX` |
| `engine_render::transform_gpu::NO_PARENT` | `u32::MAX` |
| `mvp_build.comp` | `const uint NO_PARENT = 0xFFFFFFFFu` |

`transform_gpu.rs` carried a doc comment whose entire job was warning that
the three must match. The renderer had to *fill* its per-slot parent buffer
with the sentinel at allocation and again on every capacity grow, because a
zeroed slot would otherwise have claimed slot 0 as its parent.

Two further costs showed up while designing the hierarchy panel
(ADR-0008):

1. **Enumerating roots was an O(N) scan** of `metadata` for
   `parent == u32::MAX`. A panel showing 40 rows had to scan a million
   entities to find where to start.
2. **"Detached" was a reachable state.** `remove_transform` set orphaned
   children to `u32::MAX`, which made them invisible to a tree walk that
   starts from roots — present in the simulation, absent from the panel.

## Decision

**Slot 0 is the hierarchy root, created with the hierarchy and present for
the whole session. `_Transform::parent == None` means "child of the root",
not "detached". The root is its own parent.**

The sentinel disappears. It is not replaced by a different constant — it is
replaced by `0`, whose meaning is true by construction:

```glsl
for (uint depth = 0u; parent != ROOT && depth < MAX_PARENT_DEPTH; ++depth)
```

### Why this deletes more than it adds

* **"Never written" and "parented to the root" are the same value.** The
  renderer zero-fills `sot_parents` and is *correct*, rather than filling a
  sentinel to avoid being wrong. Both `fill_u32_oneshot` call sites become
  zero-fills.
* **Roots are free to enumerate.** They are `children` of slot 0, a `Vec<u32>`
  the existing add / remove / re-parent paths already maintain. The
  `roots: Vec<u32>` this ADR was originally going to add is not needed.
* **A panel has one place to start** — no "for each root" loop, no scan.
* **There is no detached state to lose an entity in.** `remove_transform`
  re-homes orphans onto the root, so they stay reachable.

### The root is an anchor, not a transform

The composition walks terminate *at* the root without composing its TRS. It
is a structural node whose TRS slot exists only because the SoA arrays are
indexed by entity.

This is a deliberate exception and the alternative was considered: composing
it would make a scene-wide transform work "for free", at the cost of one
extra iteration per renderer per frame in `mvp_build`'s hot loop. Parent a
container under the root instead — same result, paid for only by the scenes
that want it.

### Invariants, enforced loudly

Per the project's no-fallback rule, each of these panics rather than
degrading:

* `remove_transform(ROOT)` — the root cannot be removed.
* `set_parent(ROOT, _)` — the root cannot be re-parented.
* `set_parent(t, p)` where `p` is a descendant of `t` — a cycle. Previously
  this produced a chain the GPU walk could not terminate; it bottomed out at
  `MAX_PARENT_DEPTH` and rendered garbage. The check is O(depth) on a cold
  path, and it is the same host-side check ADR-0007 requires once level
  ordering removes the GPU's depth bound.

### Sibling order is now stable

`swap_remove` became `remove` in both detach paths. The cost is an
O(siblings) memmove on a cold path. It buys two things:

* A hierarchy panel does not visually scramble a parent's children when an
  unrelated sibling is deleted.
* **Undo of a re-parent becomes exact.** The inverse of
  `set_parent(child, new, j)` is `(child, old_parent, old_index)` — 12 bytes
  — but only if removal preserves the order of the remaining siblings.
  An editor with drag-re-parent and inexact undo is not finished, so this
  stopped being optional.

## Consequences

### Wins

* Three duplicated constants and the comment policing them: gone.
* Sentinel fills become zero fills.
* Roots enumerable in O(1); no new bookkeeping.
* Cycles and root mutation are now loud failures rather than silent garbage.
* Orphans stay reachable.

### Costs

* **`TransformHierarchy::len()` is one larger**, and it sizes the transform
  SoT, the scatter, `GPURenderers` and the cull dispatch. One slot.
* **Entity indices no longer start at 0 for game entities.** Two
  `scene_asset` tests asserted absolute counts and were corrected; anything
  else assuming index 0 is a game entity would be wrong.
* `MAX_PARENT_DEPTH` spends one level on the root, so user-visible depth is
  63 rather than 64.

### Caveats

* The root's TRS is silently unused. Mitigated by documenting it on the
  constant and by `root_transform_is_not_composed`, which pins the behaviour
  so it cannot drift into being half-applied.
* Slot recycling (`Avail`) is populated by `remove_transform` but never
  consumed — `create_transform` always appends. Once recycling is turned on,
  a stale `u32` index will silently resolve to a different entity, which
  matters for a panel that stores a selection. That wants per-slot
  generation tags and is out of scope here.

## Verification

* `root_exists_and_is_the_default_parent`, `removal_preserves_sibling_order`,
  `reparenting_under_a_descendant_panics`, `removing_the_root_panics`,
  `root_transform_is_not_composed`, `nested_chain_composes_through_to_the_root`,
  and the updated `parent_stream_drains_current_values`.
* GPU: the parent-chain walk was exercised end-to-end by temporarily chaining
  `test-game`'s grid into stacks of 7 with a `(0, 2, 0)` local offset, which
  composes correctly only if the walk runs. Confirmed visually, 11 000 FPS,
  then reverted. No asset in the repository has a deep hierarchy, so this is
  not covered by a standing test.
