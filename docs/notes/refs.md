# Ids and refs

An **id** is a raw slot index: `Entity`, `NodeId`'s index, the `u64` a
`TreeView` row names. It is what the stores work in, and it says nothing about
whether the slot still holds what it held when you took it.

A **ref** is a handle that outlives the frame it was made in. Every ref is
`Copy + Send`, owns nothing, and resolves with `with(…) -> Option<R>`, which
answers `None` once what it names is gone — including when its slot has been
reused since.

| Ref | Names | Resolves through | Checked by |
|---|---|---|---|
| `AssetRef` | an asset | the asset registry | the registry |
| `EntityRef` | an entity in a world | `worlds::world` | the slot's generation |
| `ComponentRef<T>` | `T` on an entity | its `EntityRef`, then the storage | the same generation |
| `UiRef<T>` | state `T` a UI node owns | `&mut UiCore` | the node's generation |

The rule of thumb: **store a ref, pass an id.** A listener closure, a drag
payload, anything queued for a later frame holds a ref. A function handed a
world and asked about one of its entities takes an id.

## Why refs carry a generation

`TransformHierarchy` reuses freed slots, lowest first, in `create_transform`
— otherwise every delete leaks a slot in every per-slot array, CPU and GPU.
Reuse is what makes a bare id dangerous: within one frame a destroy can free
slot 7 and a spawn take it, and anything still holding `Entity(7)` now names
the newcomer. So each slot has a generation, bumped by `remove_transform`,
and a ref compares it before it resolves.

`Entity` itself stays a bare `u32`. The hierarchy, scene files and the tree
views all work in slot ids, and threading a generation through every one of
them would buy nothing an id is not already used for — an id is only valid
for the moment it is handed over. The handles that *do* outlive a frame are
the refs, and the world's queued edits: `World::edit` and `World::destroy`
capture the generation when queued, so an edit or a destroy cannot land on an
entity that took the slot in between.

## What reuse asks of the GPU side

A fresh slot parented to the root records nothing, because the renderer
zero-fills its per-slot parent buffer and zero is the root. A reused slot's
entry still names its last occupant's parent, so `create_transform` records
the parent of every reused slot. Per-slot renderer state is already reset by
`World::remove_entity`, which deinitialises components while they can still
scatter their own sentinels.
