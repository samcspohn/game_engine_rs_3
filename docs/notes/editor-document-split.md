# The editor's entities vs. the document's

The editor is a game built on the engine (ADR-0008): `Chrome` is a
`Component`, the hierarchy panel reads the live `TransformHierarchy`, the
viewport camera is an ordinary `CameraComponent` + `OrbitController` pair.
That is deliberate and is not what this note changes.

What it changes is that the editor was also a game **editing itself**. One
`Scene`, one hierarchy rooted at `ROOT`, and `editor camera` a sibling of the
content — so the tree listed it, an inspector could drive it (moving the
editor's own view), and a save would have serialised it.

## The split

`TransformHierarchy` gains a `scene_root`, which is what `parent: None`
resolves to. It is `ROOT` on construction, so a game sees no change at all —
"the top level" and "the hierarchy root" stay the same place.

The editor creates its rig (camera, and later gizmos, grid, selection
outlines) parented to `ROOT` *explicitly*, then creates a `document` entity
and calls `set_scene_root(document)`. Everything after that — project
loading, `spawn_subscene`, a component spawning at runtime, a drop-to-top-
level in the tree — resolves `None` to the document.

The hierarchy panel roots its `TreeView` at the document rather than at
`ROOT`, which is a one-argument change because `TreeView` already took a root
id. That single change is what removes the camera from the tree, from
selection, from every `EntityRef` an inspector can be handed, and from a save
walk.

## Why `None` and not a per-entity flag

The alternative was Unity's `HideFlags`: a bit per transform, filtered in the
panel's children closure. Cheaper today, but then every document operation is
an inverted predicate ("everything not flagged"), and each new piece of editor
furniture has to remember to flag itself. A subtree makes "the document" a
thing you can address — save it, clear it, reload it, snapshot it for play
mode — and makes the default for anything that forgets to say be *inside* it,
which is the safe direction to fail.

## Consequences

* `None` means the scene root **everywhere**, not just at creation:
  `set_parent(None)` follows it too, so a drop-to-top-level lands in the
  document rather than beside the editor's camera.
* The scene root cannot be removed, and neither can any ancestor of it
  (asserted) — `remove_transform` takes the whole subtree, so an ancestor
  would take the document with it.
* Document entities gain one level in the GPU parent walk
  (`mvp_build.comp`). Games pay nothing, since `scene_root == ROOT` there.
* The zero-fill invariant in `transform_gpu` is untouched: zero still means
  `ROOT`, and a transform parented anywhere else already emitted a
  `parent_stream` record at creation — the path GLB subscene children have
  always taken.

## Known limitation: the hierarchy panel's entity count

The count reads `hierarchy.len() - editor_entities`, where `editor_entities`
is `len()` snapshotted in `load_project_scene` at the instant the rig and the
empty document exist and nothing has been loaded yet. Taken rather than
hardcoded, so adding a gizmo to the rig keeps it correct.

It is arithmetic on a length, not a count of the document, and it is wrong in
two ways that happen to cancel out today:

1. **`len()` is a high-water mark, not a live count.** `avail` is only ever
   pushed to (`transform/mod.rs`, in `remove_transform`); `create_transform`
   always appends and never pops it, so a removed entity still occupies a
   slot and still counts. Deleting would not decrease the number. Invisible
   only because the panel has no delete yet.
2. **It measures the whole hierarchy, not the document subtree.** Anything
   the editor creates *after* setup — a selection outline, a drag preview,
   a transform gizmo — lands outside the snapshot and is counted as if the
   project had added it.

### The fix

Walk the document subtree **once** when the panel is built to seed a real
count, then maintain it incrementally: `+1` when an entity is created inside
the document, `-1` when one is destroyed. Exact regardless of slot reuse,
regardless of what the editor spawns later, and O(1) per frame instead of
O(entities).

What is missing is the signal. The panel already learns about one kind of
creation — `scene_asset::drain_instantiated`, which is what drives
`TreeView::invalidate` — but a plain `new_entity` announces nothing, and
there is no destroy event at all. That same create/destroy stream is what
undo, dirty-flagging and save-on-change all need, so it is worth building
once and deliberately rather than growing a counter-shaped hole for each.

Until then the count is honest for the case the editor actually supports:
load a project, spawn subscenes, never delete.

## The active camera

`first_component::<CameraComponent>()` picked the camera by lowest transform
index. The editor camera won only because the stub scene has no other, and a
project shipping its own camera would have made it a creation-order coin
flip. The renderer now reads an explicitly published entity:
`CameraComponent::init` publishes itself (so a one-camera game still says
nothing), and `set_active_camera` overrides — which is the edit/play switch:
edit mode points at the editor camera, play mode at the document's.
