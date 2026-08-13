# ADR-0010 — Scene authoring: documents, reflection, and in-process play

**Status:** Proposed.
**Related:** [ADR-0008](ADR-0008-ui-integration.md) (the editor is a game built
on the public API), [ADR-0009](ADR-0009-hierarchy-root-entity.md) (`parent ==
None` and the root anchor this builds on),
[`docs/notes/editor-document-split.md`](notes/editor-document-split.md) (the
`scene_root` split this generalises).

## Context

The editor renders a viewport and shows a hierarchy. It cannot save, inspect,
or play. Four questions were open, and they resolve to one:

1. **GLB subscenes.** A `.glb` placed in a document already *is* a hierarchy on
   disk. Re-serialising its nodes duplicates the asset, and the copy stops
   matching the moment the artist re-exports.
2. **Inspection.** A Unity-style inspector must enumerate a component's fields.
   `Component` exposes none. Separately, `ComponentRegistry` is keyed by
   `TypeId`, which is not stable across builds and therefore cannot appear in a
   file.
3. **Play mutates what it plays.** `Spinner` writes its transform every frame.
   Any save-after-play bakes simulation results into the document.
4. **In-process play can take the editor down.** Game code runs inside the
   process that holds the unsaved work.

The first three are all "what is the document, and how is it addressed". The
fourth is the price of the answer.

## Decision

### 1. Addresses inside an instance are file-native keys, and always a path

`SceneTemplate` node order is a **parser artifact**: `build_template` walks a
stack with `pop()` (reverse sibling order) and splices one extra node per
primitive for multi-primitive meshes. Keying overrides by array position means
a future change to that traversal silently retargets every saved override — no
error, wrong node moved.

Address by the glTF's own indices instead, which are already trusted enough to
mint mesh paths (`file.glb#mesh{i}/prim{j}`):

```rust
enum NodeKey { Node(u32), Prim { node: u32, prim: u32 } }
```

`TemplateNode` gains the key it was built from. **The address is a
`Vec<NodeKey>`, not a single key,** from the first version — an inherited scene
overriding something inside a nested instance needs the path, and adding it
later is a file-format break.

### 2. A subscene instance is a component

```rust
struct SubsceneInstance { source: SceneId, nodes: HashMap<NodeKey, Entity> }
```

The save walk descends the document subtree and, on an entity carrying this,
writes a reference plus a delta and **does not recurse**. The GLB's hierarchy
is not filtered out of the output — it is never reached.

`nodes` is built by `instantiate` and is also the reverse lookup the inspector
needs: "which instance owns this entity" is a parent walk to the nearest
`SubsceneInstance`, which is the same walk that decides whether an edit is an
override or a plain edit.

### 3. One `#[export]` derive, four consumers

The derive emits a property list (name, type, get, set) and a stable
`TYPE_NAME`. That single mechanism serves:

| Consumer | What it takes |
|---|---|
| Inspector | rows, and the typed drop target per field |
| Serialisation | the property list, written by name |
| `ComponentRegistry` | `TYPE_NAME` → constructor, replacing `TypeId` in files |
| Deltas | per-property granularity |

The last one reverses an earlier call. Without reflection, component-level
deltas were the YAGNI answer; with it, per-property is the *same walk* and it
is what makes overrides usable.

Keep the value enum small: `f32 / i32 / bool / String / Vec3 / Quat / Color`,
plus `AssetRef` and `EntityRef`. Those two are what the existing drop zones
feed, and typing the drop target by the reflected field type is what makes a
wrong drop a rejection rather than a crash — the user-friendliness fallback the
project requires, implemented once instead of per widget.

### 4. Documents, and play, are sibling subtrees of `ROOT`

`scene_root` — added for the editor/document split — already is this mechanism.
N documents are children of `ROOT`, the hierarchy panel points at one, and
`set_scene_root` follows focus. **The play instance is another sibling**,
instantiated from the document rather than run in place.

This replaces the snapshot-and-restore alternative outright rather than on
points: the document is never mutated, so there is nothing to restore and no
"did the restore restore everything" bug class. Stop-play is deleting a
subtree.

During play, `scene_root` points at the **play** root — otherwise a game
spawning bullets appends them to the document being edited — and
`set_active_camera` points at the game's camera. Both mechanisms already exist.

### 5. Edit mode runs no behaviour; the boundary is construction vs. `init`

`HAS_UPDATE` is not the whole line. `CameraComponent::init` publishes
`active_camera`; attaching one in the editor would steal the viewport.

**Construction is edit-time. `init` and `update` are play-time.** That is
nearly the split already in place — `CameraComponent` and `MeshRenderer` are
`HAS_UPDATE = false` data the renderer reads directly each frame, and must stay
live in edit mode or the viewport goes black. It becomes a stated rule so the
next data component knows which side it is on.

The `update` half **landed** as `Scene::set_simulating`, a second switch
beside §6's `enabled` rather than a reuse of it. Two axes and not one because
`enabled` off also scatters `NO_RENDERER` — exactly right for the unfocused
document, exactly wrong for the one being edited, which has to be *seen* to be
authored. The editor turns `simulating` off over its document at startup, so
the project's `Spinner` sits still while its cube renders.

The `init` half is not built: the case that needs it (a document carrying a
`CameraComponent`) cannot arise until documents are deserialised, and
`MeshRenderer` wants its `init` to run in edit mode anyway, so the rule is
narrower than "init is play-time" makes it sound.

### 6. Activation is a bitset ANDed into two sweeps that already exist

A scene-level `enabled` bitset over transform slots, `activeSelf` +
`activeInHierarchy` recomputed on toggle (rare, O(subtree)) so per-frame reads
stay free.

* **CPU.** `ComponentStorage::par_iter` already sweeps a per-slot `active`
  bitmap word by word through `bitmap_task_layout`. AND `enabled` into the word
  load: one extra load per 32 entities.
* **GPU.** `GPURenderers` is one `(mesh_id, material_id)` per slot with
  `NO_RENDERER = u32::MAX`, fed by the scatter queue `MeshRenderer` already
  pushes to. Deactivating scatters sentinels; reactivating scatters the real
  ids back. **No new GPU code.**

Visibility is not optional here — two documents open, only the focused one
should be on screen.

The rejected alternative was per-type serialisable *proxies* on edited scenes
(the `MeshRendererProxy` pattern). It doubles every component type, makes the
inspector edit a stand-in, and needs conversion both ways. The proxy exists for
templates because a `SceneTemplate` is data with no ECS storage; a live edited
scene is not that, and with §3 the proxy's only job is already done.

This is not play-mode scaffolding: it is `SetActive`, which the engine needs on
its own. Play mode is its first consumer.

### 7. Panics are caught per component call, not per frame

`std::panic::catch_unwind` around `f(&mut *guard, &transform)` **inside**
`par_iter`'s per-word loop.

* **Per call, so the slot is known.** `current_idx` is in scope; the recovery
  is to clear that entity's `enabled` bit and log to the editor console. A
  frame-level catch yields a payload and no culprit.
* **Per call, so the frame stays coherent.** Component update runs before any
  scatter or CB recording, so a panic leaves one transform half-updated and
  self-correcting. A frame-level catch can land mid-way through the staging-slot
  protocol (`write_spawns`, the `gpu_signal` gates, `last_spawn_count`), where
  "recovered" would mean resynchronising GPU state with no way to inspect it.

Three facts make this viable rather than theatre: `ComponentStorage` holds
components in `parking_lot::Mutex`, which does not poison; `parallel_for`
already propagates panics to the caller on every backend
(`parallel.rs::panic_propagates_all_backends`); and no profile sets
`panic = "abort"`.

A `panic::set_hook` capturing message, location and backtrace into a ring
buffer is what the console displays — the hook runs before unwinding and is the
only place a backtrace exists.

**What this does not catch**, and therefore what a process boundary would still
buy: stack overflow, abort from a double panic, UB from `unsafe` (including
this crate's own `MaybeUninit` + `SyncPtr` storage), deadlock, and device-lost.
The real safety net is stop-play; per-component catching is what keeps an
`unwrap` on a `None` from needing it.

### 8. In-process now; the process boundary is a later cut along this seam

The play instance is already an isolated subtree with its own root, camera and
enable bit. Moving it out of process later changes *who feeds that subtree*,
not the structure. The genuine argument for the boundary is crash isolation,
and it gets strong only once there is saved work to lose — after save and undo,
not before.

## Consequences

### Wins

* The GLB hierarchy is never serialised, because the save walk stops at the
  instance rather than filtering its contents.
* Inherited scenes are not a second mechanism: a scene file is
  `{ base: Option<AssetRef>, delta, extra }`, and `base: None` is an ordinary
  scene. A `.glb` is a base that cannot be written to, so "edit this GLB and
  save" *is* an inherited scene, for free.
* Play needs no snapshot and no restore.
* Diff-on-save needs no command layer, no undo stack, and no create/destroy
  event stream — none of which exist (see the entity-count limitation in the
  document-split note). It is exact precisely because edit mode runs no
  `update`, and it detects removals correctly: the template says node `k` has a
  renderer, the live entity does not.
* Activation costs one AND in the CPU sweep and zero new GPU code.

### Costs

* A proc-macro crate, and every serialisable component must derive.
* `instantiate` carries a per-instance `HashMap<NodeKey, Entity>`.
* Deltas must ride *with* the spawn request: `spawn_subscene` queues and
  `drain_ready_spawns` materialises frames later, so a caller has nothing to
  apply an override to when the call returns.
* Per-call `catch_unwind` adds landing pads to the hottest component loop. If
  that shows up in a profile, gate it behind a flag only the editor sets —
  measure first.

### Prerequisites this exposes

These were gaps in current code, not future work items. The first three are
**done** (they landed before step 1 of the build order):

* ~~**`remove_subtree` does not exist**, and `remove_transform` *adopts*
  orphans onto the scene root.~~ `remove_transform` now takes the whole
  subtree and returns the removed slots, which is what lets
  `Scene::remove_entity` drop their components. No separate primitive: the
  delta's "removed children" and stop-play both want the same default.
  Removing the scene root or any ancestor of it panics, checked over the
  collected subtree before anything is unlinked.
* ~~**Deferred spawns resolve `parent: None` at drain time**~~, which is still
  the semantics — `spawn_subscene` has no `Scene` to resolve against — but it
  is now documented as a hazard rather than a feature, and the editor pins
  `parent: Some(scene_root())` when queuing.
* ~~**The global registries poison.**~~ Every `global()` (meshes, materials,
  textures, scene templates, the UI store, `ACTIVE_CAMERA`, `VIEWPORT`, the
  renderer spawn queue) is `parking_lot` now, and no `.expect("… poisoned")`
  remains.
* **`ACTIVE_CAMERA` is a single global.** Two viewports showing two documents
  need it per-viewport; the `Viewport` widget is where it belongs. Still open.

### Caveats

* Node keys survive a re-export only as far as the exporter keeps node indices
  stable. Storing the node *name* alongside the key lets a mismatch be reported
  loudly instead of silently rotating the wrong bone.
* Diff-on-save cannot distinguish "deliberately set to the template's value"
  from "not overridden". This has no observable consequence and is the reason
  recording intent is not worth its infrastructure yet.
* Slot recycling remains off (ADR-0009's caveat). A delta keyed by `NodeKey`
  is immune, but `SubsceneInstance::nodes` holds live `Entity` values and will
  need generation tags when recycling lands.

## Build order

~~`#[export]` derive + value model~~ → ~~`enabled` bitset (AND in `par_iter`,
sentinel scatter)~~ → ~~`remove_subtree`~~ → play root as a sibling → node keys
and the delta → inheritance (by then a `base` field and a recursive call).

The bitset landed as **two** — `enabled` (the switch) and
`enabled_in_hierarchy` (it resolved against every ancestor) — because one
cannot survive the round trip: a child switched off in its own right must stay
off when its parent comes back. §6 said as much (`activeSelf` +
`activeInHierarchy`); the names avoid a third meaning of "active" in a file
that already has two. The sweep ANDs `enabled_in_hierarchy` and the GPU learns
through a new `Component::set_enabled` hook, which `MeshRenderer` implements by
scattering `NO_RENDERER` — the same path that also closes the older bug where
a *deleted* entity kept drawing.

The derive landed as `crates/engine-derive` plus `engine_core::reflect`, and
it was prototyped against `MeshRenderer` as this said to be. The verdict: a
naïve value model breaks not on the *types* but on the *access* — writing
`Option<MaterialId>` as a field skips the registry refcount and the GPU
record, so the derive routes through methods (`#[export(get = …, set = …)]`)
and `Export::set` takes the entity's `Transform`. `MeshRenderer::set_mesh`
was added; it had a getter and no setter. See
[`docs/notes/reflection.md`](notes/reflection.md).

## Revisit if

* Saved work becomes valuable enough that an in-process crash is unacceptable —
  then §8, and the play subtree is the seam.
* Two people edit the same component often enough that per-property overrides
  stop being fine-grained enough.
* Node-index churn from re-exports turns out to be common, which would move
  addressing from indices to name paths.
