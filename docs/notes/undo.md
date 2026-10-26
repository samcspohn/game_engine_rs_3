# Undo

Each open document keeps one history (`crates/editor/src/history.rs`), keyed
by its world id. *Edit > undo / redo*, Ctrl+Z, and Ctrl+Y or Ctrl+Shift+Z act
on the document in front. Play pauses it: the panels show the running scene
then, and the history belongs to the file. *Reload scene* clears it.

## A step holds the other side

A step stores only the value the world does not have right now: a property's
previous value, the old name, the old parent and index, or the deleted
subtree. Undo and redo are the same `flip`. It swaps what the world holds with
what the step holds, so after an undo the step holds the redo.

| Step | Recorded by | Flip |
|---|---|---|
| `Set` | inspector write, gizmo release | swap a property, or position, rotation or scale |
| `Name` | rename | swap the name |
| `Place` | drag in the hierarchy | swap the parent and sibling index |
| `Component` | *Add Component* | remove when present, add a default when absent |
| `Life` | spawn, instance, delete | snapshot and remove, or restore at the old index |

The panels make their own edits and then *record* them. The exception is
delete, which goes through `history::destroy`, because the snapshot has to be
taken before the subtree is gone.

A scrub writes on every frame it moves. The inspector marks those writes as a
gesture, and writes from one gesture merge into the first step, which holds
the value from before the scrub. Releasing the scrub seals the step. The gizmo
needs no merging, because it reports once, on release, with the transform the
drag started from (`gizmo::on_release`).

## Every flip runs at the frame boundary

Undo and redo queue their flip through `World::mutate`, which hands the
closure `&mut World` in queue order alongside spawns and destroys. Two undos
in one frame therefore run in sequence. A restore also needs the mutable world
to build entities. Recording happens straight away, under the history's lock
only.

## Entities are keys, not ids

A deleted subtree comes back in new slots. A step names its entity by *key*,
an index into the history's table of `EntityRef`s. When a `Life` step
removes a subtree, it notes which of the snapshot's entities had keys. On
restore it re-binds those keys to the new slots, so a step recorded against
the old entity lands on its successor.

A step that no longer resolves is dropped instead of being retried. This
happens when its entity's key was never re-bound, such as a child that an
instance rebuilds from its model.

The snapshot is `scene_file::snapshot`, the save walk run on one entity
including itself, and it is held in memory. An entity reference that points
outside the subtree comes back empty.
