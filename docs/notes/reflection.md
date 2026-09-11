# `#[derive(Export)]` and the value model

Implements ADR-0010 §3: one derive, four consumers — the inspector's rows and
typed drop targets, the save walk, the `TYPE_NAME` a scene file names a
component by, and per-property deltas. The mechanism lives in
`engine-core/src/reflect.rs`; the derive is `crates/engine-derive`.

## The attribute grammar

Fields **opt in**. A struct can derive `Export` and expose nothing, which is
what transient state wants (`OrbitController::dragging` is mid-gesture, not
authored).

```rust
#[derive(Clone, Export)]
pub struct CameraComponent {
    #[export] pub fov_y_radians: f32,
    #[export] pub z_near: f32,
    #[export] pub z_far: f32,
}
```

`#[export]` alone reads and writes the field directly. `#[export(get = f, set
= g)]` routes through methods instead: `get` calls `self.f()`, `set` calls
`self.g(transform, value)`. The getter must return the field's own type **by
value** — the field's declared type is the property type either way, which is
what keeps one `Exportable` impl serving both forms.

Giving only one of `get`/`set` is a compile error. Half a routing is the
dangerous shape: a method-backed setter paired with a raw field read reports a
value the setter's side effects never produced.

## Why `set` takes a transform

`Export::set(&mut self, name, value, transform)`. The transform is not
decoration — `MeshRenderer::set_material` refcounts in the material registry
*and* pushes a `(transform_id, mesh_id, material_word)` record the renderer
scatters into `GPURenderers` next frame. Without the entity there is no
`transform_id` and the GPU never learns. Every other component hook
(`init`/`update`/`deinit`) already takes one, so this is the existing shape
rather than a new one.

`get` does not take a transform. Reading a property never needs the entity,
and pretending otherwise would make the save walk carry one for nothing.

## `MeshRenderer` is why method routing exists

ADR-0010's build order says to prototype the derive against `MeshRenderer`
before committing the value enum, because "its `Option<MaterialId>` and asset
handle are where a naïve value model breaks". They break it in three ways:

1. **Both fields are private.** Not actually an obstacle — the generated impl
   sits in the same module — but it signals the next two.
2. **A plain write leaks.** `self.material = Some(id)` skips
   `MaterialRegistry::retain`/`release`, so the new material can be evicted
   while a renderer still points at it and the old one never falls out.
3. **A plain write is invisible.** The GPU reads `GPURenderers`, not the
   component. Only the setter's `push_spawn` reaches it.

`mesh_id` had a getter but no setter at all; `MeshRenderer::set_mesh` was
added for this, mirroring `set_material`.

## The scalars are closed; the composites are not

`f32 / i32 / bool / String / Vec3 / Quat / Color`, plus `AssetRef` and
`EntityRef`. A field type that is not `Exportable` fails to compile rather
than degrading to a string.

That set stays closed because an inspector row has to render a kind and a drop
target has to reject a wrong one — see below. It is not what a *file* needs,
though, and a project cannot add a variant to an enum in `engine-core`. So
`Value::Struct` and `Value::List` carry an exported type's own properties, and
the derive emits `Exportable` alongside `Export`: **a type becomes a value
type by deriving**, and a `Vec` of it works through one blanket impl. Nesting
needs `Default`, since a nested value is rebuilt by setting the properties a
caller had onto a fresh one — which is also what makes a field added later
read an older scene file. `Exportable::from_value` takes the `Transform` for
the same reason `Export::set` does: those property writes may be setters that
publish GPU state.

The inspector has no widget for a composite yet, so its row is the read-only
line an asset slot already uses (`3 fields`, `2 items`). Enums and maps are
the two shapes still missing, and both are a variant plus a derive arm in the
shape these have.

`ValueKind` is carried in `PropertyInfo` **separately from any value**, which
is the point: an empty material slot still types its drop target as
`Asset(Material)`, so a texture dragged onto it is declined before any id is
dereferenced. That rejection is `Exportable::from_value` returning `None` and
`Export::set` returning `false` — the user-friendliness fallback the project
requires, written once instead of per widget.

`Value::Asset(Option<AssetRef>)` and `Value::Entity(Option<Entity>)` carry the
`Option` at the value level, and `T` and `Option<T>` share a `ValueKind`. Only
the `Option` form accepts `None`, so clearing a non-optional handle is a
rejection rather than a silent no-op. Scalars have no optional form — nothing
needs one, and adding it later is additive.

## Path resolution

The derive emits absolute paths, and the crate they should point at differs:
`engine-core` and `engine-render` reach `engine_core` directly, a game reaches
it only through the `engine` facade. `proc-macro-crate` resolves which, and
`engine` re-exports `engine_core` so the two prefixes address the same items.
`engine-core` carries `extern crate self as engine_core` so the paths resolve
inside it too.

The alternative — every game adding `engine-core` to its manifest — would put
an implementation crate in the dependency list the facade exists to keep out.

## The Inspector, the first consumer

`Component` requires `Export`, so anything attachable is inspectable and
`impl Export for T {}` is the honest "nothing to author here" (the editor's
own `Chrome` and `HierarchyPanel` say exactly that). The alternative — an
optional bound — cannot work: `ComponentStorage<T>` is generic, and without
the supertrait there is no way for a type-erased walk to ask whether `T`
opted in.

`world.entity(e).inspect(f)` hands every component on one entity to a
`&mut dyn Export` callback. `Component::update` takes `&World`, so a panel
that *is* a component can read another entity's components — the engine's
missing `GetComponent`.

Its own world and no wider (ADR-0011 §3). The editor's chrome lives in the
rig's world and the document it inspects is another, so it reaches that one
through `engine_editor_api::world(id)` — the frame's world list, ambient
because chrome runs *inside* the sweep over both.

The editor's panel names no component type. It walks `inspect`, builds a row
per `PropertyInfo`, gives the scalar kinds a `TextField` and everything else a
read-only line, and **reads values back through `get` every frame rather than
echoing what was typed** — so a `set` that refuses is visible as the field
reverting to the live value. Rows are keyed by `(type_name, prop)` and not by
position, because `inspect` walks a `HashMap`.

## From extension to kind

A drop target asks for an `AssetKind`. A file in the browser is an extension.
`engine-core::asset_type` is the table between them, and the only place that
reads an extension forward — every loader dispatches on its own, but until a
drag needed it nothing could answer "what *is* this file".

```rust
pub trait AssetType: Send + Sync {
    fn extensions(&self) -> &[&str];
    fn kind(&self) -> AssetKind;
    fn reference(&self, path: &Path) -> AssetRef;
}
```

`reference` is infallible because it reads no bytes: it asks the registry that
owns the kind for a handle, and that handle draws the placeholder until an
async load replaces it — or the error mesh when the file will not decode. A
bad drop is a mesh that looks wrong, not a dialog, which is the same path a
scene file's mesh path has always taken.

Built in: `.obj` is a `Mesh`; `.glb` / `.gltf` are a `Scene`, because a glTF
is a node tree that instantiates and one of its primitives is not addressable
on its own; the image extensions are a `Texture`, requested as sRGB — a normal
map is the exception, and the material that wants one linear is where that
belongs, not the file's extension. `Material` claims no extension at all,
because no format makes one yet, so a material slot accepts nothing and says
so by never lighting up.

`register` takes a `Box<dyn AssetType>` and leaks it: a registration is a
startup singleton, and a lookup that borrowed out of the table would hold its
lock for as long as the caller held the type. Lookup searches newest first, so
a game registering `.obj` replaces the built-in rather than being shadowed by
it. Nothing is keyed on the extension being unique.

`path_of` is the inverse, and is what makes an asset row legible: the
inspector shows `sphere.obj` rather than `mesh 4`, and `no mesh` for a handle
no file minted — which is exactly what `MeshRenderer::default()` produces.

## Not yet

`TYPE_NAME` is emitted (as an inherent const, so a `name → constructor` table
can use it without an instance) but nothing consumes it: `ComponentRegistry`
is still keyed by `TypeId`. It is unqualified — just the struct's name — which
is readable in a file and collides only across crates that ship components of
the same name. Rekeying the registry is the serialisation step, not this one.

An asset row is now a drop zone; an entity row is still a read-only line and
wants the hierarchy's `EntityRef` the same way — that payload already exists,
the zone does not. Neither ever got a text box, because one you cannot type a
`MeshId` into would be a worse lie than showing the value. `Color` is four
numbers rather than a picker.
