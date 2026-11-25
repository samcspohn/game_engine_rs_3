# Resources

A resource is an exported struct stored as a file of its own. It is the
definition many things share: a particle system that emitters point at, or a
light profile. The instance holds a handle plus its own overrides. Editing
the file changes every instance at once, and a setting the instance does not
expose cannot be overridden from it.

```rust
#[derive(Clone, Default, Export)]
#[export(resource = "spin")]
pub struct Spin { #[export] axis: Vec3, #[export] speed: f32 }

#[derive(Clone, Export)]
pub struct Rotator {
    #[export] spin: Option<Res<Spin>>,  // a slot that takes .spin files only
    #[export] rate: f32,                // the instance's override
}

engine::declare_scripts!(Rotator; resources: Spin);
```

`#[export(resource = "ext")]` implements `Resource` with the derive's
`TYPE_NAME` and that extension. `resource::register::<T>()` is called for each
type listed after `resources:`. It adds the type to the registry and claims
its extension in `asset_type`, which is how a Browser drag knows what the
file is.

## One kind for every resource type

`AssetKind::Resource(name)` covers every resource type, keyed by the type's
name. `Res<T>`'s value kind therefore names `T`. A slot lights up only for
that type's files, and `from_value` refuses a handle to any other type, so a
game adds a resource type by deriving and needs no engine change.
`AssetRef::Resource` holds an untyped `ResId`, and the registry knows its
type.

## The file

A resource file is the struct's properties as a JSON object. It uses the
same value format a scene file uses for a component's properties
(`scene_file::props_of` / `set_props_on`), so the same rules hold:

- an absent field means its default;
- nested structs and lists of structs work;
- asset fields are written as their paths.

An entity reference has no meaning outside a scene, so it is written as
`null`.

`Export::set` takes a `Transform`, because a component's setter may publish
GPU state. A resource is on no entity. Its setters get
`transform::detached()`, a transform in a hierarchy that belongs to no world.

## The registry

Requests are deduplicated by path, so every handle to a file shares one value.
The registry handles files in these ways:

- **A file that does not exist yet** reads as the type's defaults. That is
  what `create` writes.
- **A file that fails to read** is reported and held at the defaults. It is
  never saved over, because that would replace what the file says.
- **Edits** go through `resource::edit` and bump a version, so a renderer can
  re-upload only the resources that changed.
- **Reads** use `Res::with`, which takes a read lock, so a sweep can read from
  every worker at once. A closure inside `with` or `edit` must not request a
  resource.

## Materials

`MaterialData` is a resource (`.mat`) that also feeds a store of its own: its
`Resource` impl sets `KIND` to `AssetKind::Material` and makes `reference`
return the `MaterialId` the file maps to, so a `.mat` file drops onto a
renderer's material slot like any material. A file is one id, keyed by its
path in the material registry. `edited` writes each edit into that id's
slot, and the GPU material store re-uploads the slot through the dirty-slot
drain it already had. Every renderer on the file follows.

A scene writes a file material as its path and reads it back with
`material::request_path`. A material an importer made has no file, so it is
still written as its data and interned by content.

## In the editor

- **Opening:** double-clicking a resource file in the Browser shows it in the
  inspector of the document in front, with the same rows a component gets.
- **Saving:** an edit is unsaved until *save all* (Ctrl+S), and the
  inspector's title shows `*` until then. Edits undo on the editor's one
  timeline; see [undo](undo.md).
- **Creating:** right-clicking a Browser row offers *new &lt;Type&gt;* for
  each registered type. It writes the defaults into that folder, or beside the
  file under the pointer, opens the folder and selects the new file.

## Not yet

- **Errors:** a read error is only printed on stderr. A save error appears in
  the console.
- **Make unique:** copying a resource and re-pointing one instance at the
  copy is not built yet.
