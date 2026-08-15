# The editor's entities vs. the document's

The editor is a game built on the engine (ADR-0008): `Chrome` is a
`Component`, the hierarchy panel reads a live `TransformHierarchy`, the
viewport camera is an ordinary `CameraComponent` + `OrbitController` pair.
That is deliberate and is not what this note changes.

What it changes is that the editor was also a game **editing itself**. One
`Scene`, one hierarchy rooted at `ROOT`, and `editor camera` a sibling of the
content — so the tree listed it, an inspector could drive it (moving the
editor's own view), and a save would have serialised it.

## The split

The editor runs **one world per document plus its own rig** (ADR-0011) —
camera(s), and later gizmos, grid, selection outlines. A world owns its
hierarchy, so these are separate graphs with separate `ROOT`s and no
relationship at all, rather than one graph with a boundary drawn through it.
It currently opens two documents, each in its own `ui::Viewport` panel with a
camera of its own, which is what step 3 + step 4 were for.

`parent: None` therefore means the document's own root whenever a document
operation resolves it — project loading, `spawn_subscene`, a component
spawning at runtime, a drop-to-top-level in the tree. It cannot name the rig,
because the rig is not in that hierarchy.

Both worlds are built by queued spawn (`world.spawn(t, |e| …)`, ADR-0011 §3),
so the camera, its controller and `Chrome` are attached inside one builder
callback at the first frame boundary rather than by a `&mut World` `main`
never has.

The hierarchy panel roots its `TreeView` at the document world's `ROOT`. That
is what keeps the camera out of the tree, out of selection, out of every
`EntityRef` an inspector can be handed, and out of a save walk.

An earlier attempt did this with a `scene_root` field: `parent: None` aimed at
a `document` entity inside one shared hierarchy. It worked, and it is gone —
per-world hierarchies delete the field, the "the scene root cannot be removed"
assert, and the extra level document entities paid in the GPU parent walk.

## Why a world and not a per-entity flag

The alternative was Unity's `HideFlags`: a bit per transform, filtered in the
panel's children closure. Cheaper today, but then every document operation is
an inverted predicate ("everything not flagged"), and each new piece of editor
furniture has to remember to flag itself. A world makes "the document" a thing
you can address — save it, clear it, reload it, snapshot it for play mode —
and makes the default for anything that forgets to say be *inside* it, which
is the safe direction to fail. ADR-0011 §1 has the rest of the argument, which
is mostly about GPU buffer sizing.

## Consequences

* The document is a world that does not simulate, so its components live in a
  registry the update loop skips entirely — edit mode runs no behaviour, and
  the rig, being a different world, keeps running. See the Worlds section of
  `Readme.md`.
* Chrome lives in the rig and inspects the document, so it holds the
  document's `WorldHandle` — beside every id it points at, which is the
  discipline a bare `Entity` asks for (ADR-0011 §2). A handle and not an id:
  the world it edits has to stay alive because the editor is looking at it.
* Every world is drawn through buffers of its own (ADR-0011 step 3), and each
  viewport has its own camera (step 4). The rig's gizmos still wait, but on
  step 5 now — drawing *two* worlds into *one* viewport — not on the buffers.
* The zero-fill invariant in `transform_gpu` is untouched: zero means `ROOT`,
  and every hierarchy has one at slot 0.

## Known limitation: the hierarchy panel's entity count

The count reads `hierarchy.len() - 1` — the document world's whole hierarchy
minus its root. That is exact for what the editor supports today and wrong in
one way: **`len()` is a high-water mark, not a live count.** `avail` is only
ever pushed to (`transform/mod.rs`, in `remove_transform`); `create_transform`
always appends and never pops it, so a removed entity still occupies a slot
and still counts. Deleting would not decrease the number. Invisible only
because the panel has no delete yet.

### The fix

Maintain a real count incrementally: `+1` when an entity is created, `-1` when
one is destroyed. Exact regardless of slot reuse, and O(1) per frame.

What is missing is the signal. The panel already learns about one kind of
creation — `scene_asset::drain_instantiated`, which is what drives
`TreeView::invalidate` — but a plain `new_entity` announces nothing, and there
is no destroy event at all. That same create/destroy stream is what undo,
dirty-flagging and save-on-change all need, so it is worth building once and
deliberately rather than growing a counter-shaped hole for each.

## The active camera

`first_component::<CameraComponent>()` picked the camera by lowest transform
index. The editor camera won only because the stub scene has no other, and a
project shipping its own camera would have made it a creation-order coin
flip. There is no "the" camera any more: each one is a `CameraHandle` naming
the world it draws, and the editor owns its own rather than attaching a
`CameraComponent` it would then have to out-vote. Edit versus play is which
cameras exist, not which of them is named.
