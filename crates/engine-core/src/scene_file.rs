//! A scene file: the shape of a subtree, and the components on it.
//!
//! What persists is what `#[export]` marks — the same reflection the
//! inspector reads, so a field a user can edit is a field that is saved, and
//! neither is declared twice (ADR-0010 §3). A component implements nothing
//! for this module's sake.
//!
//! serde carries it to JSON, but the file is *untagged*: a value is written
//! as itself, and read back against the kind the property declares. That is
//! what keeps `"speed": 0.5` from having to be `"speed": {"f32": 0.5}`.
//!
//! Ids are the exception that needs code here, because none of them survive
//! the process that minted them: a component is named by the derive's
//! `TYPE_NAME`, a registry handle by what it can be requested with again, and
//! an entity by its position in the file. Those live on the types, one impl
//! each, at the bottom.
//!
//! See [`docs/notes/scene-file.md`](../../../docs/notes/scene-file.md).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use glam::{Quat, Vec3};
use serde::de::{DeserializeOwned, Error as _};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value as Json;

use crate::asset::{self, MeshId};
use crate::component::{Entity, World};
use crate::material::{self, MaterialId};
use crate::reflect::{AssetKind, AssetRef, Value, ValueKind};
use crate::scene_asset::{self, SceneId};
use crate::script;
use crate::texture::{self, ColorSpace, TextureId};
use crate::transform::_Transform;

/// Bumped when a reader of the previous number would get this one wrong.
const VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct Scene {
    version: u32,
    entities: Vec<Node>,
}

/// One entity. Children nest, so the file's shape is the scene's shape.
#[derive(Serialize, Deserialize)]
struct Node {
    name: String,
    /// Absent means the default a fresh entity already has, which is why
    /// these are `Option` rather than zeroes nobody reads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pos: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rot: Option<Quat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scale: Option<Vec3>,
    /// Keyed by `TYPE_NAME`, then by property name — one component of a type
    /// per entity because a storage holds one. Sorted, so two saves of one
    /// scene are the same bytes.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    components: BTreeMap<String, BTreeMap<String, Json>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    children: Vec<Node>,
}

// ─── Writing ─────────────────────────────────────────────────────────────

/// Everything under `root`, not including `root` itself — so a document's
/// world root is the file, and [`load`] puts it back under any parent.
pub fn save(world: &World, root: u32) -> String {
    // The whole of it inside the file's entities: a property turns into JSON
    // during the walk below, which is where an entity reference resolves.
    in_file(walk(world, root), || {
        let scene = Scene {
            version: VERSION,
            entities: world
                .hierarchy()
                .children(root)
                .iter()
                .map(|&c| node(world, c))
                .collect(),
        };
        let mut out = Vec::new();
        let mut json = serde_json::Serializer::with_formatter(&mut out, Pretty::default());
        scene
            .serialize(&mut json)
            .expect("a scene holds only what JSON does");
        out.push(b'\n');
        String::from_utf8(out).expect("serde_json writes utf-8")
    })
}

fn node(world: &World, idx: u32) -> Node {
    let h = world.hierarchy();
    let t = h.get_transform_(idx);
    let mut components = BTreeMap::new();
    world.entity(Entity::new(idx)).inspect(|c| {
        // A type no registry names cannot be written down at all — the file
        // would name something `load` could not construct.
        if script::find(c.type_name()).is_none() {
            return;
        }
        let mut props = BTreeMap::new();
        for p in c.properties() {
            match c.get(p.name).as_ref().and_then(to_json) {
                Some(Ok(json)) => drop(props.insert(p.name.to_string(), json)),
                Some(Err(e)) => eprintln!("scene: {}.{}: {e}", c.type_name(), p.name),
                None => {}
            }
        }
        components.insert(c.type_name().to_string(), props);
    });
    Node {
        name: t.name,
        pos: (t.position != Vec3::ZERO).then_some(t.position),
        rot: (t.rotation != Quat::IDENTITY).then_some(t.rotation),
        scale: (t.scale != Vec3::ONE).then_some(t.scale),
        components,
        children: h.children(idx).iter().map(|&c| node(world, c)).collect(),
    }
}

/// `None` drops the property: an empty slot is what a fresh component
/// already has, so writing it down would say nothing.
fn to_json(v: &Value) -> Option<Result<Json, String>> {
    match v {
        Value::Asset(None) | Value::Entity(None) => None,
        v => Some(serde_json::to_value(v).map_err(|e| e.to_string())),
    }
}

/// Depth-first in child order — the same order the nesting above emits, and
/// so the ordinal an entity-valued property refers to.
fn walk(world: &World, root: u32) -> Vec<u32> {
    let h = world.hierarchy();
    let mut out = Vec::new();
    let mut stack: Vec<u32> = h.children(root).iter().rev().copied().collect();
    while let Some(idx) = stack.pop() {
        out.push(idx);
        stack.extend(h.children(idx).iter().rev());
    }
    out
}

/// A value writes as itself; what it *is* comes from the property's declared
/// kind on the way back in ([`from_json`]).
impl Serialize for Value {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::F32(x) => s.serialize_f32(*x),
            Value::I32(x) => s.serialize_i32(*x),
            Value::Bool(b) => s.serialize_bool(*b),
            Value::String(v) => s.serialize_str(v),
            Value::Vec3(v) => v.serialize(s),
            Value::Quat(q) => q.serialize(s),
            Value::Color(c) => c.serialize(s),
            Value::Asset(None) | Value::Entity(None) => s.serialize_none(),
            Value::Asset(Some(AssetRef::Mesh(id))) => id.serialize(s),
            Value::Asset(Some(AssetRef::Material(id))) => id.serialize(s),
            Value::Asset(Some(AssetRef::Texture(id))) => id.serialize(s),
            Value::Asset(Some(AssetRef::Scene(id))) => id.serialize(s),
            Value::Entity(Some(e)) => e.serialize(s),
            Value::Struct(fields) => s.collect_map(fields.iter().map(|(name, v)| (name, v))),
            Value::List(items) => s.collect_seq(items),
        }
    }
}

/// serde_json's pretty printer gives every array element its own line, which
/// costs each transform fifteen of them. This is that output with an array
/// kept on one line until something inside it opens a block of its own — so
/// `pos` is three numbers and `children` is still a list of entities.
#[derive(Default)]
struct Pretty {
    indent: usize,
    /// Whether each open array is still on one line.
    arrays: Vec<bool>,
    /// Whether each open object has a member yet, so an empty one stays `{}`.
    filled: Vec<bool>,
    /// What is about to be written is an array element, and so is what
    /// decides whether that array can stay inline.
    element: bool,
}

impl Pretty {
    fn newline<W: ?Sized + io::Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(b"\n")?;
        w.write_all("  ".repeat(self.indent).as_bytes())
    }

    /// About to write a block. If it is an element of an array still hoping
    /// to be one line, that hope ends here — and the break belongs *before*
    /// this element, which has not been written yet.
    fn opening<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        if self.element && self.arrays.last() == Some(&true) {
            *self.arrays.last_mut().expect("just read") = false;
            self.newline(w)?;
        }
        Ok(())
    }
}

impl serde_json::ser::Formatter for Pretty {
    fn begin_array<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.opening(w)?;
        self.arrays.push(true);
        self.indent += 1;
        w.write_all(b"[")
    }

    fn end_array<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.indent -= 1;
        if self.arrays.pop() == Some(false) {
            self.newline(w)?;
        }
        w.write_all(b"]")
    }

    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.element = true;
        match (first, self.arrays.last() == Some(&true)) {
            (true, _) => Ok(()),
            (false, true) => w.write_all(b", "),
            (false, false) => {
                w.write_all(b",")?;
                self.newline(w)
            }
        }
    }

    fn begin_object<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.opening(w)?;
        self.filled.push(false);
        self.indent += 1;
        w.write_all(b"{")
    }

    fn end_object<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.indent -= 1;
        if self.filled.pop() == Some(true) {
            self.newline(w)?;
        }
        w.write_all(b"}")
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if !first {
            w.write_all(b",")?;
        }
        self.newline(w)
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.element = false;
        w.write_all(b": ")
    }

    fn end_object_value<W: ?Sized + io::Write>(&mut self, _w: &mut W) -> io::Result<()> {
        *self.filled.last_mut().expect("inside an object") = true;
        Ok(())
    }
}

pub fn save_to(path: impl AsRef<Path>, world: &World, root: u32) -> std::io::Result<()> {
    let path = path.as_ref();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, save(world, root))
}

// ─── Reading ─────────────────────────────────────────────────────────────

/// Read `text` into `world` under `parent`, returning the new entities in
/// file order.
///
/// A component this process has no type for, and a property that no longer
/// exists or no longer reads, are reported and skipped: opening a project
/// whose script dylib is missing must give back its scene, not an error.
pub fn load(world: &mut World, parent: u32, text: &str) -> Result<Vec<Entity>, String> {
    let scene: Scene = serde_json::from_str(text).map_err(|e| e.to_string())?;
    if scene.version != VERSION {
        return Err(format!(
            "scene version {}, and this build reads {VERSION}",
            scene.version
        ));
    }

    // Flattened in the order the ordinals count, and every entity is made
    // before any component, so a reference can point forwards.
    let mut flat = Vec::new();
    flatten(&scene.entities, None, &mut flat);

    let mut ids: Vec<Entity> = Vec::with_capacity(flat.len());
    for &(n, under) in &flat {
        let e = world.new_entity(_Transform {
            name: n.name.clone(),
            parent: Some(under.map_or(parent, |i| ids[i].id)),
            position: n.pos.unwrap_or(Vec3::ZERO),
            rotation: n.rot.unwrap_or(Quat::IDENTITY),
            scale: n.scale.unwrap_or(Vec3::ONE),
        });
        ids.push(e);
    }

    in_file(ids.iter().map(|e| e.id).collect(), || {
        for (&(n, _), &e) in flat.iter().zip(&ids) {
            for (name, props) in &n.components {
                let Some(ty) = script::find(name) else {
                    eprintln!("scene: no component type named {name} — skipped");
                    continue;
                };
                world.build(e, |mut entity| (ty.add)(&mut entity));
                apply(world, e, name, props);
            }
        }
    });
    Ok(ids)
}

pub fn load_from(
    path: impl AsRef<Path>,
    world: &mut World,
    parent: u32,
) -> Result<Vec<Entity>, String> {
    let text = std::fs::read_to_string(path.as_ref()).map_err(|e| e.to_string())?;
    load(world, parent, &text)
}

/// One value through its own `Deserialize` — the ids' impls are what turn a
/// path back into a handle.
fn of<T: DeserializeOwned>(json: &Json) -> Option<T> {
    serde_json::from_value(json.clone()).ok()
}

/// Pre-order, each node paired with its parent's position in the list.
fn flatten<'a>(nodes: &'a [Node], under: Option<usize>, out: &mut Vec<(&'a Node, Option<usize>)>) {
    for n in nodes {
        let me = out.len();
        out.push((n, under));
        flatten(&n.children, Some(me), out);
    }
}

/// Write `props` onto the component named `component` of `e`.
fn apply(world: &World, e: Entity, component: &str, props: &BTreeMap<String, Json>) {
    let transform = world.hierarchy().get_transform_unchecked(e.id);
    world.entity(e).inspect(|c| {
        if c.type_name() != component {
            return;
        }
        for (name, json) in props {
            let Some(p) = c.properties().iter().find(|p| p.name == name) else {
                eprintln!("scene: {component} has no property {name} — skipped");
                continue;
            };
            let Some(value) = from_json(json, p.kind) else {
                eprintln!("scene: {component}.{name}: cannot read {json}");
                continue;
            };
            if !c.set(name, value, &transform) {
                eprintln!("scene: {component}.{name}: declined");
            }
        }
    });
}

/// The file against the kind the property declares — which is what lets the
/// file hold a bare `0.5` and still land in the right variant. A composite
/// kind carries the kinds inside it, so this recurses without the file
/// having to describe itself.
fn from_json(json: &Json, kind: ValueKind) -> Option<Value> {
    Some(match kind {
        ValueKind::F32 => Value::F32(json.as_f64()? as f32),
        ValueKind::I32 => Value::I32(json.as_i64()? as i32),
        ValueKind::Bool => Value::Bool(json.as_bool()?),
        ValueKind::String => Value::String(json.as_str()?.to_owned()),
        ValueKind::Vec3 => Value::Vec3(of(json)?),
        ValueKind::Quat => Value::Quat(of(json)?),
        ValueKind::Color => Value::Color(of(json)?),
        ValueKind::Asset(_) if json.is_null() => Value::Asset(None),
        ValueKind::Asset(AssetKind::Mesh) => Value::Asset(Some(AssetRef::Mesh(of(json)?))),
        ValueKind::Asset(AssetKind::Material) => Value::Asset(Some(AssetRef::Material(of(json)?))),
        ValueKind::Asset(AssetKind::Texture) => Value::Asset(Some(AssetRef::Texture(of(json)?))),
        ValueKind::Asset(AssetKind::Scene) => Value::Asset(Some(AssetRef::Scene(of(json)?))),
        ValueKind::Entity if json.is_null() => Value::Entity(None),
        ValueKind::Entity => Value::Entity(Some(of(json)?)),
        // A name the file does not carry is left out, and the nested
        // default stands — which is what reads an older file after a field
        // is added.
        ValueKind::Struct(props) => Value::Struct(
            props
                .iter()
                .filter_map(|p| Some((p.name, from_json(json.get(p.name)?, p.kind)?)))
                .collect(),
        ),
        ValueKind::List(inner) => Value::List(
            json.as_array()?
                .iter()
                .map(|j| from_json(j, *inner))
                .collect::<Option<_>>()?,
        ),
    })
}

// ─── Ids, as a file holds them ─────────────────────────────────────────────

thread_local! {
    /// The entities of the file being written or read, in file order. An
    /// entity reference is the one id no registry can resolve — it means a
    /// position in *this* file, which is context serde has nowhere to put.
    static REFS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Run `f` with `entities` as the file being read or written.
fn in_file<T>(entities: Vec<u32>, f: impl FnOnce() -> T) -> T {
    REFS.with_borrow_mut(|r| *r = entities);
    let out = f();
    REFS.with_borrow_mut(Vec::clear);
    out
}

/// A position in the file, or `null` for an entity the file does not contain
/// — which `Option<Entity>` reads back as `None`, the same empty slot it
/// would have had.
impl Serialize for Entity {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match REFS.with_borrow(|r| r.iter().position(|&idx| idx == self.id)) {
            Some(n) => s.serialize_some(&n),
            None => s.serialize_none(),
        }
    }
}

impl<'de> Deserialize<'de> for Entity {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let n = Option::<usize>::deserialize(d)?
            .ok_or_else(|| D::Error::custom("an entity outside the file"))?;
        REFS.with_borrow(|r| r.get(n).copied())
            .map(Entity::new)
            .ok_or_else(|| D::Error::custom(format!("no entity at position {n}")))
    }
}

/// The path it was requested with. Empty is the id [`asset::AssetRegistry::empty`]
/// mints — no mesh chosen — and comes back as that, since it is requested the
/// same way.
impl Serialize for MeshId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let reg = asset::global().lock();
        s.serialize_str(
            &reg.path_of(*self)
                .unwrap_or(Path::new(""))
                .to_string_lossy(),
        )
    }
}

impl<'de> Deserialize<'de> for MeshId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let path = PathBuf::from(String::deserialize(d)?);
        let (id, needs_load) = asset::global().lock().request(&path);
        // The request is the reference this component keeps, exactly as
        // `MeshRenderer::new` uses it — nothing to hand back.
        if needs_load {
            asset::request_load(id, path);
        }
        Ok(id)
    }
}

/// Path and color space both: the registry keys on the pair, and one image
/// wanted in two spaces is two ids.
#[derive(Serialize, Deserialize)]
struct TexturePath {
    path: String,
    color: ColorSpace,
}

impl Serialize for TextureId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let reg = texture::global().lock();
        let (path, color) = reg
            .path_of(*self)
            .ok_or_else(|| serde::ser::Error::custom("a texture with no path"))?;
        TexturePath {
            path: path.to_string_lossy().into_owned(),
            color,
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for TextureId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let t = TexturePath::deserialize(d)?;
        let path = PathBuf::from(t.path);
        let (id, needs_load) = texture::global().lock().request(&path, t.color);
        if needs_load {
            texture::request_load(id, path);
        }
        Ok(id)
    }
}

/// A material has no path, because nothing requests one by name — it *is*
/// its data, deduped on exactly that. So the data is what a file holds, and
/// reading it back interns to the same id this process already had.
impl Serialize for MaterialId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let reg = material::global().lock();
        reg.slot(reg.slot_of(*self)).serialize(s)
    }
}

impl<'de> Deserialize<'de> for MaterialId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let data = crate::material::MaterialData::deserialize(d)?;
        // `get_or_create` retains, which is the reference the component holds.
        Ok(material::global().lock().get_or_create(data).0)
    }
}

impl Serialize for SceneId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let path = scene_asset::path_of(*self)
            .ok_or_else(|| serde::ser::Error::custom("a scene with no path"))?;
        s.serialize_str(&path.to_string_lossy())
    }
}

impl<'de> Deserialize<'de> for SceneId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(scene_asset::request_scene(PathBuf::from(
            String::deserialize(d)?,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::reflect::Export;
    use crate::script::ComponentType;
    use crate::transform::ROOT;
    use crate::MeshId;

    #[derive(Clone, Default, Export)]
    struct Widget {
        #[export]
        speed: f32,
        #[export]
        label: String,
        #[export]
        mesh: Option<MeshId>,
        #[export]
        target: Option<Entity>,
        #[export]
        tint: Option<MaterialId>,
    }

    impl Component for Widget {}

    /// A component with a field of a type the inspector's value model has no
    /// variant for — and could not be given one from outside this crate.
    #[derive(Clone, Default, Export)]
    struct Patrol {
        #[export]
        speed: f32,
        /// A list of a type this module has no variant for — it nests
        /// because it derives `Export` too, which is the whole point.
        #[export]
        route: Vec<Leg>,
    }

    #[derive(Clone, Default, Export, PartialEq, Debug)]
    struct Leg {
        #[export]
        at: Vec3,
        #[export]
        wait: f32,
    }

    impl Component for Patrol {}

    /// The registry is process-wide, so registering once is enough and
    /// registering twice is a no-op.
    fn registered() {
        script::register(&[ComponentType::of::<Widget>(), ComponentType::of::<Patrol>()]);
    }

    /// No frame runs in a test, so `&mut` on a fresh world is sound. The
    /// handle is leaked because a world lives only while one exists.
    fn world() -> &'static mut World {
        unsafe { Box::leak(Box::new(crate::new_world())).get_mut() }
    }

    fn spawn(w: &mut World, name: &str, parent: Option<u32>) -> Entity {
        w.new_entity(_Transform {
            name: name.into(),
            parent,
            .._Transform::default()
        })
    }

    fn named(w: &World, e: Entity) -> String {
        w.hierarchy().name(e.id).to_string()
    }

    #[test]
    fn the_shape_of_a_subtree_survives() {
        let w = world();
        let parent = w.new_entity(_Transform {
            name: "hull".into(),
            position: Vec3::new(1.0, 2.0, 3.0),
            scale: Vec3::splat(2.0),
            .._Transform::default()
        });
        let child = w.new_entity(_Transform {
            name: "turret".into(),
            parent: Some(parent.id),
            rotation: Quat::from_rotation_y(0.5),
            .._Transform::default()
        });
        spawn(w, "muzzle", Some(child.id));
        spawn(w, "sibling", None);

        let text = save(w, ROOT);
        let out = world();
        let ids = load(out, ROOT, &text).expect("round trip");

        assert_eq!(ids.len(), 4);
        let names: Vec<_> = ids.iter().map(|&e| named(out, e)).collect();
        assert_eq!(names, ["hull", "turret", "muzzle", "sibling"]);
        let parent_of = |e: Entity| out.hierarchy().get_transform_(e.id).parent;
        assert_eq!(parent_of(ids[0]), Some(ROOT));
        assert_eq!(parent_of(ids[1]), Some(ids[0].id));
        assert_eq!(parent_of(ids[2]), Some(ids[1].id));
        assert_eq!(parent_of(ids[3]), Some(ROOT));
        let t = out.hierarchy().get_transform_(ids[0].id);
        assert_eq!(t.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.scale, Vec3::splat(2.0));
        assert_eq!(
            out.hierarchy().get_transform_(ids[1].id).rotation,
            Quat::from_rotation_y(0.5)
        );
    }

    /// A default transform is what a loaded entity already has, so the file
    /// carries no key for it.
    #[test]
    fn an_untouched_transform_is_not_written() {
        let w = world();
        spawn(w, "plain", None);
        let text = save(w, ROOT);
        assert!(!text.contains("pos"), "{text}");
        assert!(!text.contains("scale"), "{text}");
    }

    #[test]
    fn components_come_back_with_their_properties() {
        registered();
        let w = world();
        let e = spawn(w, "widget", None);
        let mesh = asset::global()
            .lock()
            .request(Path::new("assets/hull.obj"))
            .0;
        w.add_component(
            e,
            Widget {
                speed: 2.5,
                label: "fast \"one\"".into(),
                mesh: Some(mesh),
                ..Widget::default()
            },
        );

        let text = save(w, ROOT);
        let out = world();
        let ids = load(out, ROOT, &text).expect("round trip");

        let mut seen = None;
        out.entity(ids[0]).get_component::<Widget>(|c| {
            seen = Some((c.speed, c.label.clone(), c.mesh));
        });
        let (speed, label, mesh_back) = seen.expect("the component came back");
        assert_eq!(speed, 2.5);
        assert_eq!(label, "fast \"one\"");
        assert_eq!(
            asset::global().lock().path_of(mesh_back.expect("a mesh")),
            Some(Path::new("assets/hull.obj")),
            "the path is what crossed the file, and it deduped back to an id"
        );
    }

    /// `skip_serializing_if` is how a component says an empty slot is the
    /// default — the file then carries only what was authored.
    #[test]
    fn an_empty_slot_is_absent() {
        registered();
        let w = world();
        let e = spawn(w, "widget", None);
        w.add_component(e, Widget::default());
        let text = save(w, ROOT);
        assert!(!text.contains("mesh"), "{text}");
        assert!(!text.contains("target"), "{text}");
        assert!(text.contains("\"speed\""), "{text}");
    }

    /// An id is process-local, so a reference is written as a position in
    /// the file and read back against what that position became.
    #[test]
    fn an_entity_reference_is_remapped() {
        registered();
        let w = world();
        let first = spawn(w, "a", None);
        let second = spawn(w, "b", None);
        w.add_component(
            first,
            Widget {
                target: Some(second),
                ..Widget::default()
            },
        );

        let text = save(w, ROOT);
        assert!(text.contains("\"target\": 1"), "{text}");
        let out = world();
        // Slots already taken, so the file's ordinals cannot be read as ids.
        spawn(out, "in the way", None);
        let ids = load(out, ROOT, &text).expect("round trip");

        let mut target = None;
        out.entity(ids[0])
            .get_component::<Widget>(|c| target = c.target);
        assert_eq!(target, Some(ids[1]));
        assert_ne!(
            target,
            Some(second),
            "not the ids the file was written from"
        );
    }

    /// Opening a project whose script dylib is missing must give back the
    /// scene, minus what it cannot build.
    #[test]
    fn an_unknown_component_does_not_lose_the_scene() {
        let text = r#"{
            "version": 1,
            "entities": [
                { "name": "a", "components": { "NotRegistered": { "speed": 1 } } },
                { "name": "b" }
            ]
        }"#;
        let out = world();
        let ids = load(out, ROOT, text).expect("the rest still loads");
        assert_eq!(ids.len(), 2);
        assert_eq!(named(out, ids[1]), "b");
    }

    /// The whole of why the value model is structural: a field type this
    /// module has never heard of persists, with no variant added anywhere —
    /// `Leg` only derives `Export`, like any other authored type.
    #[test]
    fn a_type_the_value_model_has_no_variant_for_round_trips() {
        registered();
        let w = world();
        let e = spawn(w, "patroller", None);
        let route = vec![
            Leg {
                at: Vec3::new(1.0, 0.0, 2.0),
                wait: 0.5,
            },
            Leg {
                at: Vec3::Z,
                wait: 1.5,
            },
        ];
        w.add_component(
            e,
            Patrol {
                speed: 3.0,
                route: route.clone(),
            },
        );

        let text = save(w, ROOT);
        let out = world();
        let ids = load(out, ROOT, &text).expect("round trip");

        let mut back = None;
        out.entity(ids[0])
            .get_component::<Patrol>(|c| back = Some(c.route.clone()));
        assert_eq!(back.as_deref(), Some(route.as_slice()));
    }

    /// A material has no path to be named by. Serialised as its data, it
    /// interns back to the id this process already had.
    #[test]
    fn a_material_round_trips_as_its_data() {
        registered();
        let data = crate::MaterialData {
            metallic: 0.75,
            roughness: 0.125,
            ..Default::default()
        };
        let id = material::global().lock().get_or_create(data).0;
        let w = world();
        let e = spawn(w, "widget", None);
        w.add_component(
            e,
            Widget {
                tint: Some(id),
                ..Widget::default()
            },
        );

        let text = save(w, ROOT);
        assert!(text.contains("\"metallic\": 0.75"), "{text}");
        let out = world();
        let ids = load(out, ROOT, &text).expect("round trip");

        let mut back = None;
        out.entity(ids[0])
            .get_component::<Widget>(|c| back = c.tint);
        assert_eq!(back, Some(id), "deduped to the same material");
    }

    #[test]
    fn a_second_save_is_the_same_text() {
        registered();
        let w = world();
        let e = w.new_entity(_Transform {
            name: "widget".into(),
            position: Vec3::X,
            .._Transform::default()
        });
        w.add_component(
            e,
            Widget {
                speed: 0.25,
                ..Widget::default()
            },
        );
        let once = save(w, ROOT);
        let out = world();
        load(out, ROOT, &once).expect("round trip");
        assert_eq!(once, save(out, ROOT));
    }

    #[test]
    fn a_file_that_is_not_one_is_refused() {
        let out = world();
        assert!(load(out, ROOT, "").is_err());
        assert!(load(out, ROOT, "scene 1\n").is_err(), "not json");
        assert!(
            load(out, ROOT, r#"{"version": 99, "entities": []}"#).is_err(),
            "version"
        );
    }
}
