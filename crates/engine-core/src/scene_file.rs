//! A scene file: the shape of a subtree, and the components on it.
//!
//! What persists is what `#[export]` marks — the same reflection the
//! inspector reads, so a field a user can edit is a field that is saved, and
//! neither is declared twice (ADR-0010 §3). The file is TOML, one section per
//! node keyed by its path, and *untagged*: a value is written as itself and
//! read back against the kind its property declares.
//!
//! See [`docs/notes/scene-file.md`](../../../docs/notes/scene-file.md).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use glam::{Quat, Vec3};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value as Json};

use crate::asset::{self, MeshId};
use crate::component::{Entity, World};
use crate::inherit::{self, Edit, SceneInstance};
use crate::material::{self, MaterialId};
use crate::reflect::{AssetKind, AssetRef, Export, Value, ValueKind};
use crate::resource;
use crate::scene_asset::{self, SceneId};
use crate::script;
use crate::texture::{self, ColorSpace, TextureId};
use crate::transform::{_Transform, Name, Transform, TransformHierarchy};

/// Only this one reads: files from before it were ported, not translated.
const VERSION: i64 = 4;

/// One component's properties, by name, as the file holds them.
pub(crate) type Props = BTreeMap<String, Json>;

/// One entity's components, keyed by `TYPE_NAME`.
pub(crate) type Components = BTreeMap<String, Props>;

#[derive(Clone)]
pub(crate) struct Scene {
    /// What the scene's own root carries: no transform worth authoring, but
    /// behaviour, which an instance puts on the entity that plays the root's
    /// part there. See [`crate::inherit`].
    pub(crate) components: Components,
    pub(crate) entities: Vec<Node>,
}

/// One entity, and under an instance only what this scene changed about the
/// model, whose own nodes come back from it. An entity reference in
/// `components` is a path from this node, so a node means the same wherever
/// it is grafted; the file writes it from the scene's root.
#[derive(Clone)]
pub(crate) struct Node {
    pub(crate) name: String,
    /// `None` is the default a fresh entity already has.
    pub(crate) pos: Option<Vec3>,
    pub(crate) rot: Option<Quat>,
    pub(crate) scale: Option<Vec3>,
    pub(crate) components: Components,
    /// The model this entity instantiates, read from its file every time.
    pub(crate) instance: Option<SceneId>,
    pub(crate) edits: Vec<Edit>,
    pub(crate) children: Vec<Node>,
}

// ─── From a world ────────────────────────────────────────────────────────

/// Everything under `root`, not including `root` itself — so a document's
/// world root is the file, and [`load`] puts it back under any parent.
pub fn save(world: &World, root: u32) -> String {
    write(&Scene {
        components: changes(components_of(world, root), &Components::new()),
        entities: world
            .hierarchy()
            .children(root)
            .map(|c| node(world, c))
            .collect(),
    })
}

pub(crate) fn node(world: &World, idx: u32) -> Node {
    let h = world.hierarchy();
    let t = h.get_transform_(idx);
    let mut components = components_of(world, idx);
    // An instance writes as the model plus what this scene changed, and its
    // nodes are not children to write down: they come back from the model.
    let instance = components
        .remove(inherit::TYPE_NAME)
        .and(inherit::of(world, idx));
    let (components, edits, children) = match &instance {
        Some(i) => (
            i.own_for_file(components),
            i.for_file(world, Entity::new(idx)),
            Vec::new(),
        ),
        None => (
            changes(components, &Components::new()),
            Vec::new(),
            h.children(idx).map(|c| node(world, c)).collect(),
        ),
    };
    Node {
        name: t.name.to_string(),
        pos: (t.position != Vec3::ZERO).then_some(t.position),
        rot: (t.rotation != Quat::IDENTITY).then_some(t.rotation),
        scale: (t.scale != Vec3::ONE).then_some(t.scale),
        components,
        instance: instance.and_then(|i| i.model),
        edits,
        children,
    }
}

/// Every `#[export]`ed property of every component on `idx`, as the file
/// holds them.
pub(crate) fn components_of(world: &World, idx: u32) -> Components {
    let h = world.hierarchy();
    let mut components = BTreeMap::new();
    world.entity(Entity::new(idx)).inspect(|c| {
        // A type no registry names cannot be written down at all — the file
        // would name something `load` could not construct.
        if script::find(c.type_name()).is_none() {
            return;
        }
        let at = |e: Entity| between(h, idx, e.id);
        components.insert(c.type_name().to_string(), props_of(c, &at));
    });
    components
}

/// `live` less what `base` says, a property `base` leaves out being its
/// type's default. A component `base` lacks stays, even with nothing left.
pub(crate) fn changes(mut live: Components, base: &Components) -> Components {
    live.retain(|ty, props| {
        let fresh = script::find(ty).map(|t| props_of(&*(t.fresh)(), &|_| None));
        let had = base.get(ty);
        props.retain(|name, v| {
            let was = had.and_then(|h| h.get(name));
            was.or(fresh.as_ref().and_then(|f| f.get(name))) != Some(v)
        });
        had.is_none() || !props.is_empty()
    });
    live
}

/// Every `#[export]`ed property of `c`, an entity in it as `at` names it. An
/// empty slot is left out: it is what a fresh component already has.
pub(crate) fn props_of(c: &dyn Export, at: &dyn Fn(Entity) -> Option<String>) -> Props {
    let mut props = BTreeMap::new();
    for p in c.properties() {
        let Some(mut v) = c.get(p.name) else { continue };
        if matches!(v, Value::Asset(None) | Value::Entity(None)) {
            continue;
        }
        named(&mut v, at);
        match serde_json::to_value(&v) {
            Ok(json) => drop(props.insert(p.name.to_string(), json)),
            Err(e) => eprintln!("scene: {}.{}: {e}", c.type_name(), p.name),
        }
    }
    props
}

/// Each entity in `v` as its path, `""` for none.
fn named(v: &mut Value, at: &dyn Fn(Entity) -> Option<String>) {
    match v {
        Value::Entity(e) => {
            let path = (*e).and_then(|e| at(e)).unwrap_or_default();
            *v = Value::String(path);
        }
        Value::Struct(fields) => fields.iter_mut().for_each(|(_, v)| named(v, at)),
        Value::List(items) => items.iter_mut().for_each(|v| named(v, at)),
        _ => {}
    }
}

/// The path from `from` to `to`: `..` up to where they meet, names down.
fn between(h: &TransformHierarchy, from: u32, to: u32) -> Option<String> {
    if !h.is_active(to) {
        return None;
    }
    let chain = |mut idx: u32| {
        let mut out = vec![idx];
        while let Some(p) = h.get_transform_unchecked(idx).lock().get_parent() {
            out.push(p);
            idx = p;
        }
        out.reverse();
        out
    };
    let (a, b) = (chain(from), chain(to));
    let same = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let mut path = vec!["..".to_string(); a.len() - same];
    for &idx in &b[same..] {
        let name = h.name(idx);
        path.push(Some(name).filter(|n| !n.is_none())?.to_string());
    }
    Some(match path.is_empty() {
        true => ".".into(),
        false => path.join("/"),
    })
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
            // Note: an entity is a path by now; see `named`.
            Value::Asset(None) | Value::Entity(_) => s.serialize_none(),
            Value::Asset(Some(AssetRef::Mesh(id))) => id.serialize(s),
            Value::Asset(Some(AssetRef::Material(id))) => id.serialize(s),
            Value::Asset(Some(AssetRef::Texture(id))) => id.serialize(s),
            Value::Asset(Some(AssetRef::Scene(id))) => id.serialize(s),
            Value::Asset(Some(AssetRef::Resource(id))) => resource::path_of(*id).serialize(s),
            Value::Struct(fields) => s.collect_map(fields.iter().map(|(name, v)| (name, v))),
            Value::List(items) => s.collect_seq(items),
        }
    }
}

pub fn save_to(path: impl AsRef<Path>, world: &World, root: u32) -> std::io::Result<()> {
    let path = path.as_ref();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, save(world, root))
}

/// `e` and its subtree, held in memory to put back — what undoing a delete
/// needs. Its references are paths, so one out of the subtree comes back too.
pub struct Snapshot {
    node: Node,
}

pub fn snapshot(world: &World, e: Entity) -> Snapshot {
    Snapshot {
        node: node(world, e.id),
    }
}

// ─── The text ────────────────────────────────────────────────────────────

/// Columns a value stays on one line within.
const WIDE: usize = 100;

fn write(scene: &Scene) -> String {
    let mut out = format!("version = {VERSION}\n");
    if !scene.components.is_empty() {
        section(&mut out, "/", &[], scene.components.clone());
    }
    for (n, name) in scene.entities.iter().zip(names(&scene.entities)) {
        write_node(&mut out, "", &name, n, false);
    }
    out
}

/// Siblings' names, an anonymous one given the next free `entity`: a path
/// needs a name, and a header names one node.
fn names(nodes: &[Node]) -> Vec<String> {
    let mut taken: HashSet<String> = nodes.iter().map(|n| n.name.clone()).collect();
    let free = |taken: &mut HashSet<String>| {
        let k = (0..)
            .find(|k| !taken.contains(&suffixed(*k)))
            .expect("one is free");
        taken.insert(suffixed(k));
        suffixed(k)
    };
    let pick = |n: &Node| match n.name.is_empty() {
        true => free(&mut taken),
        false => n.name.clone(),
    };
    nodes.iter().map(pick).collect()
}

fn suffixed(k: u32) -> String {
    match k {
        0 => "entity".into(),
        k => format!("entity{k}"),
    }
}

fn trs(pos: Option<Vec3>, rot: Option<Quat>, scale: Option<Vec3>) -> Vec<(&'static str, Json)> {
    fn json(v: &impl Serialize) -> Json {
        serde_json::to_value(v).expect("numbers are json")
    }
    let mut out = Vec::new();
    pos.inspect(|v| out.push(("pos", json(v))));
    rot.inspect(|v| out.push(("rot", json(v))));
    scale.inspect(|v| out.push(("scale", json(v))));
    out
}

fn write_node(out: &mut String, under: &str, name: &str, n: &Node, added: bool) {
    let path = format!("{under}/{name}");
    let mut fields = trs(n.pos, n.rot, n.scale);
    if let Some(model) = n.instance.and_then(|m| serde_json::to_value(m).ok()) {
        fields.push(("instance", model));
    }
    if added {
        fields.push(("added", Json::Bool(true)));
    }
    if n.instance.is_none() {
        section(out, &path, &fields, n.components.clone());
        for (c, name) in n.children.iter().zip(names(&n.children)) {
            write_node(out, &path, &name, c, false);
        }
        return;
    }
    let mut edits = merged(&n.edits);
    // An edit at the instance itself is said once, in its own section.
    let here = edits.remove("").unwrap_or_default();
    let mut components = n.components.clone();
    for (ty, props) in &here.components {
        components
            .entry(ty.clone())
            .or_default()
            .extend(props.clone());
    }
    fields.extend(trs(here.pos, here.rot, here.scale));
    section(out, &path, &fields, components);
    for (c, name) in here.children.iter().zip(names(&here.children)) {
        write_node(out, &path, &name, c, true);
    }
    for (at, e) in edits {
        let at = format!("{path}/{at}");
        let mut fields = trs(e.pos, e.rot, e.scale);
        if e.remove {
            fields.push(("remove", Json::Bool(true)));
        }
        if !fields.is_empty() || !e.components.is_empty() || e.children.is_empty() {
            section(out, &at, &fields, e.components.clone());
        }
        for (c, name) in e.children.iter().zip(names(&e.children)) {
            write_node(out, &at, &name, c, true);
        }
    }
}

/// Edits to one node as one: a file has one section per path.
fn merged(edits: &[Edit]) -> BTreeMap<String, Edit> {
    let mut out: BTreeMap<String, Edit> = BTreeMap::new();
    for e in edits {
        let into = out.entry(e.at.clone()).or_default();
        into.remove |= e.remove;
        into.pos = e.pos.or(into.pos);
        into.rot = e.rot.or(into.rot);
        into.scale = e.scale.or(into.scale);
        for (ty, props) in &e.components {
            into.components
                .entry(ty.clone())
                .or_default()
                .extend(props.clone());
        }
        into.children.extend(e.children.iter().cloned());
    }
    out
}

fn section(out: &mut String, path: &str, fields: &[(&str, Json)], mut components: Components) {
    refs(&mut components, &mut |rel| absolute(path, rel));
    out.push_str(&format!("\n[{}]\n", quoted(path)));
    for (k, v) in fields {
        line(out, k, v);
    }
    for (ty, props) in components {
        line(out, &ty, &Json::Object(props.into_iter().collect()));
    }
}

/// `key = value`, wrapped one entry a line once it outgrows [`WIDE`].
fn line(out: &mut String, key: &str, v: &Json) {
    let key = bare(key);
    out.push_str(&format!("{key} = {}\n", block(v, 0, key.len() + 3)));
}

fn block(v: &Json, indent: usize, used: usize) -> String {
    let one = inline(v);
    if used + one.len() <= WIDE {
        return one;
    }
    let pad = " ".repeat(indent + 4);
    let (open, close, entries): (_, _, Vec<String>) = match v {
        Json::Array(items) if !items.is_empty() => (
            '[',
            ']',
            (items.iter())
                .map(|v| block(v, indent + 4, indent + 4))
                .collect(),
        ),
        Json::Object(m) if !m.is_empty() => (
            '{',
            '}',
            (m.iter().filter(|(_, v)| !v.is_null()))
                .map(|(k, v)| {
                    let k = bare(k);
                    format!("{k} = {}", block(v, indent + 4, indent + k.len() + 7))
                })
                .collect(),
        ),
        _ => return one,
    };
    let body: String = entries.iter().map(|e| format!("{pad}{e},\n")).collect();
    format!("{open}\n{body}{}{close}", " ".repeat(indent))
}

fn inline(v: &Json) -> String {
    match v {
        // A file has no null: a field without one is left out, and an empty
        // slot in a list is the empty path.
        Json::Null => "\"\"".into(),
        Json::Bool(b) => b.to_string(),
        Json::Number(n) => match n.as_f64() {
            Some(x) if n.is_f64() => float(x as f32),
            _ => n.to_string(),
        },
        Json::String(s) => quoted(s),
        Json::Array(items) => {
            let items: Vec<String> = items.iter().map(inline).collect();
            format!("[{}]", items.join(", "))
        }
        Json::Object(m) => {
            let entries: Vec<String> = (m.iter().filter(|(_, v)| !v.is_null()))
                .map(|(k, v)| format!("{} = {}", bare(k), inline(v)))
                .collect();
            match entries.is_empty() {
                true => "{}".into(),
                false => format!("{{ {} }}", entries.join(", ")),
            }
        }
    }
}

/// Every number a scene holds is an `f32`, so it prints as the shortest one.
fn float(x: f32) -> String {
    match x {
        x if x.is_nan() => "nan".into(),
        x if x.is_infinite() => (if x > 0.0 { "inf" } else { "-inf" }).into(),
        x if x.fract() == 0.0 => format!("{x:.1}"),
        x => x.to_string(),
    }
}

/// A TOML basic string. JSON's escapes are a subset of TOML's, bar DEL.
fn quoted(s: &str) -> String {
    let json = serde_json::to_string(s).expect("a string is json");
    json.replace('\u{7f}', "\\u007F")
}

fn bare(key: &str) -> String {
    let plain = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-';
    match !key.is_empty() && key.chars().all(plain) {
        true => key.to_string(),
        false => quoted(key),
    }
}

/// `rel`, a path from the node at `holder`, as one from the scene's root.
/// `""` for none, and for one that leaves the file.
fn absolute(holder: &str, rel: &str) -> String {
    if rel.is_empty() {
        return String::new();
    }
    let mut at: Vec<&str> = holder.split('/').filter(|s| !s.is_empty()).collect();
    for s in rel.split('/').filter(|s| !s.is_empty() && *s != ".") {
        match s {
            ".." if at.pop().is_none() => return String::new(),
            ".." => {}
            s => at.push(s),
        }
    }
    format!("/{}", at.join("/"))
}

/// `abs`, a path from the scene's root, as one from the node at `holder`.
fn relative(holder: &str, abs: &str) -> String {
    let Some(abs) = abs.strip_prefix('/') else {
        if !abs.is_empty() {
            eprintln!("scene: {holder}: {abs:?} is not a path from the root — dropped");
        }
        return String::new();
    };
    let split = |p: &str| -> Vec<String> {
        (p.split('/').filter(|s| !s.is_empty()))
            .map(str::to_string)
            .collect()
    };
    let (from, to) = (split(holder), split(abs));
    let same = from.iter().zip(&to).take_while(|(a, b)| a == b).count();
    let mut path = vec!["..".to_string(); from.len() - same];
    path.extend(to[same..].iter().cloned());
    match path.is_empty() {
        true => ".".into(),
        false => path.join("/"),
    }
}

/// Map every entity reference in `components`, which the properties' kinds
/// pick out: the file does not mark them.
fn refs(components: &mut Components, f: &mut impl FnMut(&str) -> String) {
    for (ty, props) in components.iter_mut() {
        let Some(t) = script::find(ty) else { continue };
        let kinds = (t.fresh)().properties();
        for (name, json) in props.iter_mut() {
            if let Some(p) = kinds.iter().find(|p| p.name == name && p.kind.refers()) {
                refs_in(json, p.kind, f);
            }
        }
    }
}

fn refs_in(json: &mut Json, kind: ValueKind, f: &mut impl FnMut(&str) -> String) {
    match (kind, json) {
        (ValueKind::Entity, Json::String(s)) => *s = f(s),
        (ValueKind::Struct(props), Json::Object(m)) => {
            for p in props {
                if let Some(json) = m.get_mut(p.name) {
                    refs_in(json, p.kind, f);
                }
            }
        }
        (ValueKind::List(inner), Json::Array(items)) => {
            items.iter_mut().for_each(|json| refs_in(json, *inner, f))
        }
        _ => {}
    }
}

/// A scene file's entities, without a world to put them in — what a
/// template is resolved from.
pub(crate) fn parse(text: &str) -> Result<Scene, String> {
    let doc: toml_edit::DocumentMut = text.parse().map_err(|e| format!("{e}"))?;
    match doc.get("version").and_then(toml_edit::Item::as_integer) {
        Some(VERSION) => {}
        Some(v) => return Err(format!("scene version {v}, and this build reads {VERSION}")),
        None => return Err("not a scene: it has no version".into()),
    }
    let mut sections = Vec::new();
    for (key, item) in doc.iter().filter(|(k, _)| *k != "version") {
        match (key.starts_with('/'), item) {
            (true, toml_edit::Item::Table(t)) => sections.push((key.to_string(), table(t))),
            _ => eprintln!("scene: {key:?} is not a node's path — skipped"),
        }
    }
    Ok(Sections::new(sections).scene())
}

/// A file's sections, each under the nearest one above it: under an
/// instance, a node of the model needs no section to be edited below.
struct Sections {
    all: Vec<(String, Map<String, Json>)>,
    /// By the section above, `None` for the root.
    under: HashMap<Option<usize>, Vec<usize>>,
}

struct Fields {
    pos: Option<Vec3>,
    rot: Option<Quat>,
    scale: Option<Vec3>,
    instance: Option<SceneId>,
    remove: bool,
    added: bool,
    components: Components,
}

impl Sections {
    fn new(all: Vec<(String, Map<String, Json>)>) -> Self {
        let at: HashMap<&str, usize> = (all.iter().enumerate())
            .map(|(i, (p, _))| (p.as_str(), i))
            .collect();
        let mut under: HashMap<Option<usize>, Vec<usize>> = HashMap::new();
        for (i, (path, _)) in all.iter().enumerate().filter(|(_, (p, _))| p != "/") {
            let mut up = parent(path);
            let above = loop {
                match at.get(up) {
                    _ if up == "/" => break None,
                    Some(&j) => break Some(j),
                    None => up = parent(up),
                }
            };
            under.entry(above).or_default().push(i);
        }
        Self { all, under }
    }

    fn scene(&self) -> Scene {
        let root = self.all.iter().find(|(p, _)| p == "/");
        let components = root.map(|(p, m)| self.fields(p, m).components);
        Scene {
            components: components.unwrap_or_default(),
            entities: self.nodes(None, "/"),
        }
    }

    fn below(&self, i: Option<usize>) -> impl Iterator<Item = usize> + '_ {
        self.under.get(&i).into_iter().flatten().copied()
    }

    /// The plain nodes under `path`, which each need a section of their own.
    fn nodes(&self, i: Option<usize>, path: &str) -> Vec<Node> {
        let child = |&c: &usize| {
            let ok = parent(&self.all[c].0) == path;
            if !ok {
                eprintln!("scene: {} has no parent section — skipped", self.all[c].0);
            }
            ok
        };
        let kids: Vec<usize> = self.below(i).filter(child).collect();
        kids.into_iter().map(|c| self.node(c)).collect()
    }

    fn node(&self, i: usize) -> Node {
        let (path, m) = &self.all[i];
        let f = self.fields(path, m);
        let mut edits = Vec::new();
        if f.instance.is_some() {
            self.edits(path, i, &mut edits);
        }
        Node {
            name: path.rsplit('/').next().unwrap_or_default().to_string(),
            pos: f.pos,
            rot: f.rot,
            scale: f.scale,
            components: f.components,
            instance: f.instance,
            children: match f.instance {
                Some(_) => Vec::new(),
                None => self.nodes(Some(i), path),
            },
            edits,
        }
    }

    /// Everything under section `i` of the instance at `root`: an edit to a
    /// model node, or a node `added` below one.
    fn edits(&self, root: &str, i: usize, out: &mut Vec<Edit>) {
        let at = |p: &str| {
            p.strip_prefix(root)
                .unwrap_or(p)
                .trim_start_matches('/')
                .to_string()
        };
        for c in self.below(Some(i)) {
            let (path, m) = &self.all[c];
            let f = self.fields(path, m);
            if f.added {
                out.push(Edit {
                    at: at(parent(path)),
                    children: vec![self.node(c)],
                    ..Edit::default()
                });
                continue;
            }
            out.push(Edit {
                at: at(path),
                remove: f.remove,
                pos: f.pos,
                rot: f.rot,
                scale: f.scale,
                components: f.components,
                children: Vec::new(),
            });
            self.edits(root, c, out);
        }
    }

    fn fields(&self, path: &str, m: &Map<String, Json>) -> Fields {
        let mut f = Fields {
            pos: None,
            rot: None,
            scale: None,
            instance: None,
            remove: false,
            added: false,
            components: Components::new(),
        };
        for (k, v) in m {
            match (k.as_str(), v) {
                ("pos", v) => f.pos = of(v),
                ("rot", v) => f.rot = of(v),
                ("scale", v) => f.scale = of(v),
                ("instance", v) => f.instance = of(v),
                ("remove", v) => f.remove = v.as_bool().unwrap_or(false),
                ("added", v) => f.added = v.as_bool().unwrap_or(false),
                (ty, Json::Object(props)) => drop(f.components.insert(
                    ty.to_string(),
                    props.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                )),
                (k, _) => eprintln!("scene: {path} has no field {k} — skipped"),
            }
        }
        refs(&mut f.components, &mut |abs| relative(path, abs));
        f
    }
}

fn parent(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some(("", _)) | None => "/",
        Some((p, _)) => p,
    }
}

fn table(t: &toml_edit::Table) -> Map<String, Json> {
    t.iter().map(|(k, i)| (k.to_string(), item(i))).collect()
}

fn item(i: &toml_edit::Item) -> Json {
    match i {
        toml_edit::Item::None => Json::Null,
        toml_edit::Item::Value(v) => value(v),
        toml_edit::Item::Table(t) => Json::Object(table(t)),
        toml_edit::Item::ArrayOfTables(a) => a.iter().map(|t| Json::Object(table(t))).collect(),
    }
}

fn value(v: &toml_edit::Value) -> Json {
    use toml_edit::Value as V;
    match v {
        V::String(s) => Json::String(s.value().clone()),
        V::Integer(i) => (*i.value()).into(),
        V::Float(f) => serde_json::Number::from_f64(*f.value()).map_or(Json::Null, Json::Number),
        V::Boolean(b) => (*b.value()).into(),
        V::Datetime(d) => Json::String(d.value().to_string()),
        V::Array(a) => a.iter().map(value).collect(),
        V::InlineTable(t) => {
            Json::Object(t.iter().map(|(k, v)| (k.to_string(), value(v))).collect())
        }
    }
}

/// A resource file: its properties as a bare table.
pub(crate) fn props_text(props: &Props) -> String {
    let mut out = String::new();
    for (k, v) in props {
        line(&mut out, k, v);
    }
    out
}

pub(crate) fn props_from_text(text: &str) -> Result<Props, String> {
    let doc: toml_edit::DocumentMut = text.parse().map_err(|e| format!("{e}"))?;
    Ok(doc.iter().map(|(k, i)| (k.to_string(), item(i))).collect())
}

// ─── Into a world ────────────────────────────────────────────────────────

/// Read `text` into `world` under `parent`, returning the new entities in
/// file order.
///
/// A component this process has no type for, and a property that no longer
/// exists or no longer reads, are reported and skipped: opening a project
/// whose script dylib is missing must give back its scene, not an error.
pub fn load(world: &mut World, parent: u32, text: &str) -> Result<Vec<Entity>, String> {
    let scene = parse(text)?;
    Ok(instantiate(
        world,
        parent,
        &scene.entities,
        &scene.components,
    ))
}

/// Put `s` back as the last child of `parent`, returning its entities, its
/// own first.
pub fn restore(world: &mut World, parent: u32, s: &Snapshot) -> Vec<Entity> {
    instantiate_nodes(world, parent, std::slice::from_ref(&s.node))
}

/// Make `nodes` children of `parent`, with no root components to put on
/// `parent` itself — what an edit's added entities are.
pub(crate) fn instantiate_nodes(world: &mut World, parent: u32, nodes: &[Node]) -> Vec<Entity> {
    instantiate(world, parent, nodes, &Components::new())
}

/// Make `nodes` and everything under them children of `parent`, put `root` on
/// `parent` itself, and return the new entities in file order.
fn instantiate(world: &mut World, parent: u32, nodes: &[Node], root: &Components) -> Vec<Entity> {
    whole(world, |world| {
        let mut flat = Vec::new();
        flatten(nodes, None, &mut flat);
        let mut ids: Vec<Entity> = Vec::with_capacity(flat.len());
        for &(n, under) in &flat {
            let e = world.new_entity(_Transform {
                name: Name::new(&n.name),
                parent: Some(under.map_or(parent, |i| ids[i].id)),
                position: n.pos.unwrap_or(Vec3::ZERO),
                rotation: n.rot.unwrap_or(Quat::IDENTITY),
                scale: n.scale.unwrap_or(Vec3::ONE),
            });
            ids.push(e);
        }
        set_components(world, Entity::new(parent), root);
        for (&(n, _), &e) in flat.iter().zip(&ids) {
            match n.instance {
                // The instance carries them instead: the base scene's root
                // components go on first and these win over them.
                Some(model) => {
                    let it = SceneInstance::from_file(model, n.edits.clone(), &n.components);
                    world.add_component(e, it);
                }
                None => set_components(world, e, &n.components),
            }
        }
        // Now rather than at the boundary, so the whole subtree is here on return.
        for (&(n, _), &e) in flat.iter().zip(&ids) {
            if n.instance.is_some() {
                inherit::build(world, e);
            }
        }
        ids
    })
}

thread_local! {
    /// Properties holding an entity, set once everything they could name is
    /// built; `None` outside [`whole`].
    static LATER: RefCell<Option<Vec<(Entity, String, Props)>>> = const { RefCell::new(None) };
}

/// Run `f`, then set the references it read: a path may name a node an
/// instance built after the property was read.
pub(crate) fn whole<T>(world: &mut World, f: impl FnOnce(&mut World) -> T) -> T {
    let outer = LATER.with_borrow_mut(|l| l.is_none() && l.replace(Vec::new()).is_none());
    let out = f(world);
    if outer {
        for (e, ty, props) in LATER.take().unwrap_or_default() {
            set_props(world, e, &ty, &props);
        }
    }
    out
}

/// Put `components` on `e`, making each one that is not there yet.
pub(crate) fn set_components(world: &mut World, e: Entity, components: &Components) {
    for (name, props) in components {
        let Some(ty) = script::find(name) else {
            eprintln!("scene: no component type named {name} — skipped");
            continue;
        };
        // Only when there is none: a second `add` hands the entity a default
        // over whatever put the first one there.
        if !inherit::has(world, e, name) {
            world.build(e, |mut entity| (ty.add)(&mut entity));
        }
        set_props(world, e, name, props);
    }
}

/// Components made once, off any entity, and copied onto each one they go
/// on. A property holding an entity stays written: each copy resolves its own.
#[derive(Default)]
pub(crate) struct Prebuilt(Vec<(Box<dyn script::Prototype>, Props)>);

impl Prebuilt {
    pub(crate) fn new(components: &Components) -> Self {
        let detached = crate::transform::detached();
        let mut out = Vec::with_capacity(components.len());
        for (name, props) in components {
            let Some(ty) = script::find(name) else {
                eprintln!("scene: no component type named {name} — skipped");
                continue;
            };
            let mut proto = (ty.fresh)();
            let kinds = proto.properties();
            let (refs, plain): (Props, Props) = (props.clone().into_iter())
                .partition(|(n, _)| (kinds.iter()).any(|p| p.name == n && p.kind.refers()));
            set_props_on(&mut *proto, &plain, &detached);
            out.push((proto, refs));
        }
        Self(out)
    }

    /// Put these on `e`. A type `e` already carries is written over instead,
    /// the way [`set_components`] does, from `written`.
    pub(crate) fn put(&self, world: &mut World, e: Entity, written: &Components) {
        for (proto, refs) in &self.0 {
            let name = proto.type_name();
            match proto.add_to(world, e) {
                false => set_props(world, e, name, &written[name]),
                true if !refs.is_empty() => set_props(world, e, name, refs),
                true => {}
            }
        }
    }
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

/// Write `props` onto the component named `component` of `e`. Inside
/// [`whole`], one holding an entity waits for the end of it.
pub(crate) fn set_props(world: &World, e: Entity, component: &str, props: &Props) {
    let transform = world.hierarchy().get_transform_unchecked(e.id);
    let later = LATER.with_borrow(Option::is_some);
    world.entity(e).inspect(|c| {
        if c.type_name() != component {
            return;
        }
        let kinds = c.properties();
        let refers = |n: &str| kinds.iter().any(|p| p.name == n && p.kind.refers());
        if !later || !props.keys().any(|n| refers(n)) {
            return set_props_on(c, props, &transform);
        }
        let (refs, now): (Props, Props) = props.clone().into_iter().partition(|(n, _)| refers(n));
        set_props_on(c, &now, &transform);
        let wait = (e, component.to_string(), refs);
        LATER.with_borrow_mut(|l| l.as_mut().expect("inside whole").push(wait));
    });
}

/// Write `props` onto `c`, reporting and skipping what does not fit. An
/// entity is found from `t`'s.
pub(crate) fn set_props_on(c: &mut dyn Export, props: &Props, t: &Transform) {
    let ty = c.type_name();
    for (name, json) in props {
        let Some(p) = c.properties().iter().find(|p| p.name == name) else {
            eprintln!("scene: {ty} has no property {name} — skipped");
            continue;
        };
        let Some(value) = from_json(json, p.kind, t) else {
            eprintln!("scene: {ty}.{name}: cannot read {json}");
            continue;
        };
        if !c.set(name, value, t) {
            eprintln!("scene: {ty}.{name}: declined");
        }
    }
}

/// The file against the kind the property declares — which is what lets the
/// file hold a bare `0.5` and still land in the right variant. A composite
/// kind carries the kinds inside it, so this recurses without the file
/// having to describe itself.
fn from_json(json: &Json, kind: ValueKind, t: &Transform) -> Option<Value> {
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
        ValueKind::Asset(AssetKind::Resource(name)) => Value::Asset(Some(AssetRef::Resource(
            resource::request(name, Path::new(json.as_str()?))?,
        ))),
        ValueKind::Entity => match json.as_str() {
            None | Some("") => Value::Entity(None),
            Some(path) => Value::Entity(Some(Entity::new(t.hierarchy().find(t.get_idx(), path)?))),
        },
        // A name the file does not carry is left out, and the nested
        // default stands — which is what reads an older file after a field
        // is added.
        ValueKind::Struct(props) => Value::Struct(
            props
                .iter()
                .filter_map(|p| Some((p.name, from_json(json.get(p.name)?, p.kind, t)?)))
                .collect(),
        ),
        ValueKind::List(inner) => Value::List(
            json.as_array()?
                .iter()
                .map(|j| from_json(j, *inner, t))
                .collect::<Option<_>>()?,
        ),
    })
}

// ─── Ids, as a file holds them ─────────────────────────────────────────────

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

/// A material read from a file writes as that file's path. One an importer
/// made has no path — it *is* its data, deduped on exactly that — so the data
/// is what the scene holds, and reading it back interns to the same id.
impl Serialize for MaterialId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let reg = material::global().lock();
        match reg.path_of(*self) {
            Some(path) => s.serialize_str(&path.to_string_lossy()),
            None => reg.slot(reg.slot_of(*self)).serialize(s),
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum WrittenMaterial {
    File(PathBuf),
    Data(crate::material::MaterialData),
}

impl<'de> Deserialize<'de> for MaterialId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        // Either way it takes the reference the component holds.
        Ok(match WrittenMaterial::deserialize(d)? {
            WrittenMaterial::File(path) => material::request_path(&path),
            WrittenMaterial::Data(data) => material::global().lock().get_or_create(data).0,
        })
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
    use crate::MeshId;
    use crate::component::Component;
    use crate::reflect::Export;
    use crate::script::ComponentType;
    use crate::transform::ROOT;

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

    /// A file carries only what differs from a fresh component, and the
    /// component itself even when nothing does.
    #[test]
    fn a_default_is_absent() {
        registered();
        let w = world();
        let e = spawn(w, "widget", None);
        w.add_component(e, Widget::default());
        let text = save(w, ROOT);
        assert!(text.contains("Widget = {}"), "{text}");
        assert!(!text.contains("speed") && !text.contains("mesh"), "{text}");
    }

    /// An id is process-local, so a reference is written as a position in
    /// the file and read back against what that position became.
    #[derive(Clone, Default, Export)]
    #[export(resource = "look")]
    struct Look {
        #[export]
        gloss: f32,
    }

    #[derive(Clone, Default, Export)]
    struct Styled {
        #[export]
        look: Option<crate::resource::Res<Look>>,
    }

    impl Component for Styled {}

    /// A resource field writes as its file's path and reads back as the same
    /// handle, since the registry dedups by path.
    #[test]
    fn a_resource_reference_is_its_path() {
        script::register(&[ComponentType::of::<Styled>()]);
        let path = std::env::temp_dir().join(format!("engine_look_{}.look", std::process::id()));
        let look = crate::resource::Res::<Look>::request(&path);
        let w = world();
        let e = spawn(w, "styled", None);
        w.add_component(e, Styled { look: Some(look) });

        let text = save(w, ROOT);
        assert!(
            text.contains(&format!("{:?}", path.display().to_string())),
            "{text}"
        );
        let out = world();
        let ids = load(out, ROOT, &text).expect("round trip");
        let mut back = None;
        out.entity(ids[0])
            .get_component::<Styled>(|c| back = c.look);
        assert_eq!(back, Some(look));
    }

    #[derive(Clone, Default, Export)]
    struct Painted {
        #[export]
        material: Option<crate::material::MaterialId>,
    }

    impl Component for Painted {}

    /// A material from a file writes as the file, so a scene keeps following
    /// it; one an importer made still writes as its data.
    #[test]
    fn a_file_material_is_its_path() {
        script::register(&[ComponentType::of::<Painted>()]);
        let path = std::env::temp_dir().join(format!("engine_paint_{}.mat", std::process::id()));
        let file = material::request_path(&path);
        let w = world();
        let e = spawn(w, "painted", None);
        w.add_component(
            e,
            Painted {
                material: Some(file),
            },
        );

        let text = save(w, ROOT);
        assert!(
            text.contains(&format!("{:?}", path.display().to_string())),
            "{text}"
        );
        let out = world();
        let ids = load(out, ROOT, &text).expect("round trip");
        let mut back = None;
        out.entity(ids[0])
            .get_component::<Painted>(|c| back = c.material);
        assert_eq!(back, Some(file));
    }

    /// The undo of a delete: the subtree comes back with its own root, and a
    /// reference inside it points at the successor.
    #[test]
    fn a_snapshot_puts_a_subtree_back() {
        registered();
        let w = world();
        let a = spawn(w, "a", None);
        let b = spawn(w, "b", Some(a.id));
        w.add_component(
            a,
            Widget {
                target: Some(b),
                ..Widget::default()
            },
        );
        let held = snapshot(w, a);
        w.remove_entity(a);

        let ids = restore(w, ROOT, &held);
        assert_eq!(
            (named(w, ids[0]), named(w, ids[1])),
            ("a".into(), "b".into())
        );
        assert_eq!(
            w.hierarchy().children(ids[0].id).collect::<Vec<_>>(),
            [ids[1].id]
        );
        let mut target = None;
        w.entity(ids[0])
            .get_component::<Widget>(|c| target = c.target);
        assert_eq!(target, Some(ids[1]));
    }

    /// A template's components are made once and copied, so each instance
    /// gets the values, and an entity reference points into its own copy.
    #[test]
    fn each_instance_copies_its_template() {
        registered();
        let path = std::env::temp_dir().join(format!("prebuilt_{}.scene", std::process::id()));
        std::fs::write(
            &path,
            "version = 4\n[\"/a\"]\nWidget = { speed = 2.0, target = \"/b\" }\n[\"/b\"]\n",
        )
        .expect("write");
        let scene = scene_asset::request_scene(&path);
        let w = world();
        for root in [spawn(w, "one", None), spawn(w, "two", None)] {
            assert!(scene_asset::instantiate_into(w, scene, root));
            let kids: Vec<u32> = w.hierarchy().children(root.id).collect();
            let (mut speed, mut target) = (0.0, None);
            w.entity(Entity::new(kids[0]))
                .get_component::<Widget>(|c| (speed, target) = (c.speed, c.target));
            assert_eq!((speed, target), (2.0, Some(Entity::new(kids[1]))));
        }
        std::fs::remove_file(&path).ok();
    }

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
        assert!(text.contains("target = \"/b\""), "{text}");
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
        let text = "version = 4\n[\"/a\"]\nNotRegistered = { speed = 1 }\n[\"/b\"]\n";
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
        assert!(text.contains("metallic = 0.75"), "{text}");
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
        assert!(load(out, ROOT, "scene 1\n").is_err(), "not toml");
        assert!(load(out, ROOT, "version = 99\n").is_err(), "version");
    }

    /// The file's own words for a scene: a path a section, a component a
    /// line, and an entity the path from the scene's root.
    #[test]
    fn a_file_reads_as_its_tree() {
        registered();
        let w = world();
        let hull = spawn(w, "hull", None);
        let turret = w.new_entity(_Transform {
            name: "turret".into(),
            parent: Some(hull.id),
            position: Vec3::X,
            .._Transform::default()
        });
        w.add_component(
            turret,
            Widget {
                speed: 0.5,
                target: Some(hull),
                ..Widget::default()
            },
        );
        let text = save(w, ROOT);
        let want = "version = 4\n\n[\"/hull\"]\n\n[\"/hull/turret\"]\n\
                    pos = [1.0, 0.0, 0.0]\nWidget = { speed = 0.5, target = \"/hull\" }\n";
        assert_eq!(text, want);
    }

    /// A long component wraps a property a line, and still reads.
    #[test]
    fn a_long_component_wraps() {
        registered();
        let w = world();
        let e = spawn(w, "patroller", None);
        let route = (0..6)
            .map(|i| Leg {
                at: Vec3::splat(i as f32),
                wait: 0.5,
            })
            .collect();
        w.add_component(e, Patrol { speed: 1.0, route });
        let text = save(w, ROOT);
        assert!(text.contains("Patrol = {\n    route = [\n"), "{text}");
        let out = world();
        let ids = load(out, ROOT, &text).expect("TOML 1.1 reads it back");
        let mut legs = 0;
        out.entity(ids[0])
            .get_component::<Patrol>(|c| legs = c.route.len());
        assert_eq!(legs, 6);
    }

    /// A reference out of a deleted subtree is a path, so it comes back.
    #[test]
    fn a_snapshot_keeps_a_reference_out_of_it() {
        registered();
        let w = world();
        let keep = spawn(w, "keep", None);
        let gone = spawn(w, "gone", None);
        let target = Some(keep);
        w.add_component(
            gone,
            Widget {
                target,
                ..Widget::default()
            },
        );
        let held = snapshot(w, gone);
        w.remove_entity(gone);
        let back = restore(w, ROOT, &held);
        let mut now = None;
        w.entity(back[0])
            .get_component::<Widget>(|c| now = c.target);
        assert_eq!(now, target);
    }

    /// An instance's node is addressable like any other: a reference into
    /// one survives a save, and an edit to it is a section under its path.
    #[test]
    fn a_reference_into_an_instance_survives_a_save() {
        registered();
        let base = std::env::temp_dir().join(format!("into_{}.scene", std::process::id()));
        script::register(&[ComponentType::of::<SceneInstance>()]);
        std::fs::write(&base, "version = 4\n[\"/arm\"]\n[\"/arm/hand\"]\n").expect("write");
        let text = format!(
            "version = 4\n\n[\"/it\"]\ninstance = {base:?}\n\n\
             [\"/it/arm/hand\"]\npos = [0.0, 2.0, 0.0]\n\n\
             [\"/aim\"]\nWidget = {{ target = \"/it/arm/hand\" }}\n"
        );
        let w = world();
        let ids = load(w, ROOT, &text).expect("loads");
        let hand = w.hierarchy().find(ids[0].id, "arm/hand").expect("built");
        let mut target = None;
        w.entity(ids[1])
            .get_component::<Widget>(|c| target = c.target);
        assert_eq!(target, Some(Entity::new(hand)));
        assert_eq!(w.hierarchy().get_transform_(hand).position, Vec3::Y * 2.0);
        assert_eq!(save(w, ROOT), text);
        std::fs::remove_file(&base).ok();
    }
}
