# Undo

The editor keeps one timeline (`crates/editor/src/history.rs`) for every open
asset: each document's world, and each resource file. Every step is tagged
with its asset. *Edit > undo / redo*, Ctrl+Z, and Ctrl+Y or Ctrl+Shift+Z undo
or redo the newest step, wherever it is. The asset it changed is then brought
to the front: the document's tab is selected, or the resource is shown in
the inspector. A document that is playing refuses the undo and says so,
because its panels show the running scene. Closing or reloading a document
drops its steps.

One timeline rather than one history per asset, because an edit can span
assets. Moving an entity is a copy into one document and a delete from
another; *Make unique* writes a resource and re-points an emitter. With a
single timeline such an action undoes as one thing. There is also no need to
work out which panel has focus to decide where Ctrl+Z goes.

## A step holds the other side

A step stores only the value the asset does not have right now. That is a
property's previous value, the old name, the old parent and index, or the
deleted subtree. Undo and redo are the same `flip`: it swaps what the asset
holds with what the step holds.

| Step | Recorded by | Flip |
|---|---|---|
| `Set` | inspector write, gizmo release | swap a property, or position, rotation or scale |
| `Name` | rename | swap the name |
| `Place` | drag in the hierarchy | swap the parent and sibling index |
| `Component` | *Add Component* | remove when present, add a default when absent |
| `Life` | spawn, instance, delete | snapshot and remove, or restore at the old index |
| `Resource` | inspector write on a resource | swap a property |

The panels make their own edits and then *record* them. The exception is
delete, which goes through `history::destroy`, because the snapshot has to be
taken before the subtree is gone.

A scrub writes on every frame it moves. Those writes are marked as a
gesture, and writes from one gesture merge into the first step, which holds
the value from before the scrub. Releasing the scrub seals the step. The gizmo
needs no merging, because it reports once, on release, with the transform the
drag started from (`gizmo::on_release`).

## Unsaved until saved

Each asset remembers which of its steps was the newest when it was last saved.
An asset is unsaved whenever its newest done step differs from that one.
Comparing step ids rather than counting edits means "undo, then make a
different edit" still reads as unsaved.

- **Marker:** a document's tab shows `*` while it is unsaved, and so does a
  resource's title in the inspector.
- **Saving a document:** *File > save scene* saves the one in front.
- **Saving everything:** *File > save all* and Ctrl+S save every unsaved
  document and resource.
- **Undo touches memory only.** Undoing never writes a file, so undoing past
  a save makes the asset unsaved again.
- **Quitting** does not ask about unsaved changes yet.

## When a flip runs

A world's flips are queued through `World::mutate` and run at its frame
boundary, which is the only point where the world can be changed directly.
A resource's flip runs at once. The step moves to the other stack as soon as
the key is pressed, so a second press in the same frame sees the order the
flips will run in.

Steps on different assets can therefore run in a different order within one
frame. That is harmless, because changes to different assets don't depend on
each other.

## Entities are paths, not ids

A deleted subtree comes back in new slots, so a step cannot hold a slot. It
holds the entity's path from the root instead, which names one entity
because siblings' names are unique ([names](names.md)). The path a step
holds is the one in the world it will flip from next. The timeline is
linear, so when a step flips, every newer step has been flipped back, and
the world is the one the step was made in: the path still resolves, to
whatever slot the entity is in now. A flip that changes a path, a rename or
a move, stores the path the entity has afterwards. A restored subtree takes
back its old names, because nothing has taken them since.

That covers a sub-scene's children the same way. The snapshot stops at a
sub-scene root, storing the model and its edits rather than the children,
so a large model costs only its overrides to hold. The children are not in
the snapshot, but their paths come back with the model.

That needs the children present when the next queued flip runs, so a
restored or loaded sub-scene is built inline by `inherit::build`, and the
build its `init` queues sees `built` and does nothing.

An anonymous entity has no path, so an edit to it is not recorded. The
editor names everything it creates, and an unnamed glTF node is `node`.

A step that no longer resolves is dropped instead of being retried.

## What pointed into a deleted subtree

The snapshot is `scene_file::snapshot`, the save walk run on one entity
including itself, and it is held in memory. Its references are paths, so
one inside the subtree comes back pointing at the restored entity, and one
out of it at the entity it named before.

A reference from outside into the subtree is not in the snapshot. Deleting
inspects every other entity for properties that can hold one, and keeps
those that point into the subtree, with their values and the paths of the
entities they named. Restoring writes each back with those paths resolved
to the new slots. Without this, a reference held the old slot, and it was
right only when the restore happened to land in the same one.
