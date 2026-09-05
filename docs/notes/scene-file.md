# Scene files

`engine_core::scene_file` saves a subtree and loads it back: the shape of the
hierarchy, the names and transforms on it, and the components attached to each
entity. It is the save half of
[ADR-0010](../ADR-0010-scene-authoring-and-play.md) §3, without the delta and
inheritance halves — a file is a whole scene, not an override of one.

`save(world, root)` writes everything *under* `root`, not `root` itself, so a
document's world root is the file and `load(world, parent, text)` can put it
back under anything. `save_to` / `load_from` are the same thing against a
path.

## What persists is what `#[export]` marks

The save walk reads the same reflection the inspector does. A component
implements nothing for this module's sake — no `Serialize`, no `save`/`load`
override, no second list of fields to keep in step with the first. `#[export]`
is the one place a type says what it is made of, and both consumers read it
(ADR-0010 §3).

### The value model grows by deriving

The scalar `Value` variants are closed on purpose: an inspector row has to
render a kind and a drop target has to reject a wrong one. Composite values
are not closed:

```rust
Struct(Vec<(&'static str, Value)>)   // kind: Struct(&'static [PropertyInfo])
List(Vec<Value>)                     // kind: List(&'static ValueKind)
```

`#[derive(Export)]` emits `Exportable` as well as `Export`, so **an exported
type is a value type**: it can be a field of another one, and a `Vec` of it
works through one blanket impl. A project adds a saveable field type by
deriving on it, not by adding a variant to an enum in `engine-core`.

```rust
#[derive(Clone, Default, Export)]
struct Leg { #[export] at: Vec3, #[export] wait: f32 }

#[derive(Clone, Export)]
struct Patrol {
    #[export] speed: f32,
    #[export] route: Vec<Leg>,   // no serde anywhere in this file
}
```

Nesting needs `Default`, because a nested value is rebuilt by setting the
properties the file carried onto a fresh one — so a field added later reads an
older file, and the derive's `where Self: Default` is what says so.

`Exportable::from_value` takes the entity's `Transform` for the same reason
`Export::set` does: rebuilding a nested value sets its properties, and one of
those may be a setter that publishes GPU state.

**What is not covered yet:** enums and maps. Both are a `Value` variant and a
derive arm, in the shape `Struct` and `List` already have. Neither has a
consumer, so neither is built.

## The format

```json
{
  "version": 1,
  "entities": [
    {
      "name": "hull",
      "pos": [1.0, 2.0, 3.0],
      "components": {
        "MeshRenderer": { "mesh_id": "assets/cube/cube.obj" },
        "Rotator": { "speed": 0.7853982 }
      },
      "children": [{ "name": "turret", "rot": [0.0, 0.247, 0.0, 0.969] }]
    }
  ]
}
```

Entities nest, so the file's shape is the scene's shape. `components` is an
object keyed by `TYPE_NAME` because a storage holds one component of a type
per entity, and a `BTreeMap` so two saves of one scene are the same bytes.

**Values are untagged.** `"speed": 0.5`, not `"speed": {"f32": 0.5}` — the
property's declared kind is what a value is read against, and a composite kind
carries the kinds inside it, so the file never has to describe itself. A
property whose type changed since the save fails to read, is reported, and the
rest of the component still loads.

**Absent means default.** An empty asset or entity slot is left out, because
that is what a fresh component already has; so is a transform at the identity,
and a nested field the file does not mention. This is the loader's rule rather
than a per-field attribute, so a component author gets it for free.

**Arrays of numbers stay on one line.** serde_json's pretty printer gives
every array element its own line, which costs each transform fifteen of them.
`Pretty` in `scene_file.rs` is that printer with an array kept inline until
something inside it opens a block of its own — what makes a file with a
hundred transforms in it readable.

## What an id becomes

A registry id means nothing outside the process that minted it, so each one
has its own `Serialize`/`Deserialize` writing what it can be *asked for*
again. Those impls live at the bottom of `scene_file.rs`, one per id — the one
place serde is written by hand here, and the reason a handle nested inside an
exported struct still round-trips.

| Id | In the file | Read back by |
|---|---|---|
| `MeshId` | its requested path (`""` = no mesh chosen) | `request` + `request_load` |
| `TextureId` | path *and* color space, since the pair is the key | `request` |
| `SceneId` | its requested path | `request_scene` |
| `MaterialId` | its `MaterialData`, inline | `get_or_create` |
| `Entity` | its position in the file | that position's new entity |

Refcounts come out right with nothing to compensate: `request` and
`get_or_create` each take the reference the component then holds, which is
exactly what `MeshRenderer::new` and `with_material` rely on.

**A material is its data.** Nothing requests one by name — the registry dedups
on content — so there is no path to write, and the data goes in whole. This is
the case the previous design could not represent at all: a closed value enum
could hold `AssetRef::Material(id)` and had nowhere to put the eleven fields
behind it, so a material override was dropped on save. Reading it back interns
to the same id, because that is what `get_or_create` does.

**An entity reference needs the file, not a registry.** `save` and `load` put
the file's entities in a thread-local for the duration, and `Entity`'s impls
read it. A reference to something outside the saved subtree serialises as
`null`, which `Option<Entity>` reads back as `None` — the same empty slot it
would have had. A bare `Entity` field fails loudly instead.

## What is skipped, and why it is not a crash

A component type that no registry names cannot be written (the file would name
something `load` could not construct) and cannot be loaded (there is nothing
to construct). Both are skipped, and the load reports the name. This is the
case where a project's script dylib is missing — see [scripts](scripts.md) —
and it has to give back the scene minus its behaviour, not an error. A
property is skipped the same way and on its own, so one field that no longer
reads does not cost the component the rest of them.

Only the file's own frame is fatal: malformed JSON, and a `version` this build
does not read.

The one thing this loses is a round trip through a process that could not
build a component: save, and it is gone, since nothing keeps its unparsed
value.

## In the editor

*File > save scene* writes the document in front to
`<project>/scenes/<title>.json`; *File > reload scene* clears that document
and reads the file back. The reload is what makes a save checkable — it is the
only way to see that what came back is what was there.

The load runs against `&mut World` taken from the document's handle rather
than through the deferred queue, which is sound for the same reason *new
scene* is: a document does not simulate, so no sweep is reading it.
