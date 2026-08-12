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
  `set_parent(None)` and orphan adoption in `remove_transform` follow it too.
  An entity whose parent is deleted stays in the document instead of escaping
  to sit beside the editor's camera.
* The scene root cannot be removed (asserted) — its orphans would be adopted
  by itself.
* Document entities gain one level in the GPU parent walk
  (`mvp_build.comp`). Games pay nothing, since `scene_root == ROOT` there.
* The zero-fill invariant in `transform_gpu` is untouched: zero still means
  `ROOT`, and a transform parented anywhere else already emitted a
  `parent_stream` record at creation — the path GLB subscene children have
  always taken.

## The active camera

`first_component::<CameraComponent>()` picked the camera by lowest transform
index. The editor camera won only because the stub scene has no other, and a
project shipping its own camera would have made it a creation-order coin
flip. The renderer now reads an explicitly published entity:
`CameraComponent::init` publishes itself (so a one-camera game still says
nothing), and `set_active_camera` overrides — which is the edit/play switch:
edit mode points at the editor camera, play mode at the document's.
