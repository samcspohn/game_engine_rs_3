# Scene files

`engine_core::scene_file` saves a subtree and loads it back: the shape of the
hierarchy, the names and transforms on it, and the components attached to each
entity. It is the save half of
[ADR-0010](../ADR-0010-scene-authoring-and-play.md) §3, without the delta
half. A file is a whole scene — except where it inherits one, which is an
entity written as a model plus what this scene changed about it; see
[inherited-scenes](inherited-scenes.md).

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

```toml
version = 4

["/"]
Spawner = { rate = 2.0 }

["/hull"]
pos = [1.0, 2.0, 3.0]
MeshRenderer = { mesh_id = "assets/cube/cube.obj" }
Rotator = { speed = 0.7853982 }

["/hull/turret"]
rot = [0.0, 0.247, 0.0, 0.969]
Aim = { target = "/hull" }
Idle = {}
```

The file is TOML, read with `toml_edit`, which keeps sections in file order;
that order is the siblings' order. Each node is a section headed by its path
from the scene's root. Siblings' names are unique ([names](names.md)), so the
path is the node's whole identity: there is no `name`, `parent`, `index` or
id to keep in step with it. A parent's section comes before its children's.
A name cannot contain `/`: renaming refuses one, and a glTF name has it
replaced with `_`.

`["/"]` is the scene's **root**. The root has no transform worth authoring,
because it is the identity the hierarchy composes from. It can still carry
behaviour, and an instance of this scene puts these components on the entity
that plays the root's part there. So the same file means the same thing
whether it is opened or instanced; see
[inherited-scenes](inherited-scenes.md).

Inside a section, `pos`, `rot` and `scale` are the node's transform, and
every inline table is a component, keyed by `TYPE_NAME`. `instance`,
`remove` and `added` are the sub-scene fields. Components are sorted, so two
saves of one scene are the same bytes.

**Long values wrap.** A value stays on one line up to 100 columns. Past that
it wraps one entry a line, which a wrapped inline table needs TOML 1.1 for:

```toml
Patrol = {
    route = [
        { at = [0.0, 0.0, 0.0], wait = 1.0 },
        { at = [4.0, 0.0, 0.0], wait = 0.5 },
    ],
    speed = 2.0,
}
```

A reader that only knows TOML 1.0 refuses that; one-line tables are TOML 1.0.
The writer could as well have written components as `["/hull".Patrol]`
tables. TOML reads that as the same data, so a hand-written file may use
either.

**Only version 4 reads.** The JSON format before it was ported, not
translated.

**Values are untagged.** `speed = 0.5`, not a tagged value. The property's
declared kind is what a value is read against, and a composite kind carries
the kinds inside it, so the file never has to describe itself. A property
whose type changed since the save fails to read, is reported, and the rest of
the component still loads. A number is an `f32`, and is written as the
shortest one that reads back the same.

**Absent means default.** A property equal to a fresh component's is left
out, and so is an empty asset or entity slot and a transform at the identity.
The component itself is still written, as `{}` if nothing on it differs. The
default comes from `ComponentType::fresh`, so a component author gets this
for free. A field added later no longer appears in every scene the next time
it is saved. The other side is that changing a `Default` changes every scene
that never set the field. `scene_file::changes` is the one rule. A plain
entity's base is the defaults; a sub-scene node's base is the model's values
on top of them. TOML has no null: a field holding none is left out, and an
empty slot inside a list is `""`.

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
| `Entity` | its path from the scene's root, `""` for none | the entity at that path |

Every one of those paths is relative, and the project is the working
directory — which is what makes a saved scene mean the same thing in the
editor, under `cargo run` and in a packaged bundle. See
[packaging](packaging.md).

Refcounts come out right with nothing to compensate: `request` and
`get_or_create` each take the reference the component then holds, which is
exactly what `MeshRenderer::new` and `with_material` rely on.

**A material is its data.** Nothing requests one by name — the registry dedups
on content — so there is no path to write, and the data goes in whole. This is
the case the previous design could not represent at all: a closed value enum
could hold `AssetRef::Material(id)` and had nowhere to put the eleven fields
behind it, so a material override was dropped on save. Reading it back interns
to the same id, because that is what `get_or_create` does.

**An entity reference is a path.** The file writes it from the scene's root,
the same as a header, so `target = "/hull"` names the section `["/hull"]`.
It can name a node inside an instance, which the file never writes down,
because a path crosses into one like any other node. One that leaves the
saved subtree is written `""`.

In memory, a node holds it from the node itself: `..` up to where the two
meet, then names down, the way `TransformHierarchy::find` reads it. That
form means the same wherever the node lands. A template grafts another
scene's nodes in, and an edit from a containing file is written against that
file's root. Both stay right without being rewritten. Reading turns the
file's form into this one and writing turns it back. The properties' kinds
say which strings are entities, so this needs the component types registered,
which reading already does.

A reference is set last, once everything it could name is built:
`scene_file::whole` holds those properties back until the outermost load,
restore or instance build has finished.

## What is skipped, and why it is not a crash

A component type that no registry names cannot be written (the file would name
something `load` could not construct) and cannot be loaded (there is nothing
to construct). Both are skipped, and the load reports the name. This is the
case where a project's script dylib is missing — see [scripts](scripts.md) —
and it has to give back the scene minus its behaviour, not an error. A
property is skipped the same way and on its own, so one field that no longer
reads does not cost the component the rest of them.

Only the file's own frame is fatal: malformed TOML, and a `version` this build
does not read.

The one thing this loses is a round trip through a process that could not
build a component: save, and it is gone, since nothing keeps its unparsed
value.

## In the editor

*File > save scene* writes the document in front to
`<project>/scenes/<title>.scene`; *File > reload scene* clears that document
and reads the file back. The reload is what makes a save checkable — it is the
only way to see that what came back is what was there.

The load runs against `&mut World` taken from the document's handle rather
than through the deferred queue, which is sound for the same reason *new
scene* is: a document does not simulate, so no sweep is reading it.

`snapshot` and `restore` are the same walk, run on one entity and including
that entity, and kept in memory. Undoing a delete puts the subtree back with
them; see [undo](undo.md).

## Component types have to be registered before a file is read

A component is named by its `TYPE_NAME`, so a name the registry cannot
resolve is reported and skipped — the file loads, minus everything on it. That
is right for a project whose script dylib is missing, and a trap everywhere
else, because the first symptom is a black window rather than an error.

Both fills therefore happen before anything can load a scene:
`engine::new_world` registers the engine's own types (earlier than
`Window::new`, which is where that used to live alone), and `declare_scripts!`
emits a `register()` a game binary calls for the project's. See
[play-mode](play-mode.md) and [scripts](scripts.md).
