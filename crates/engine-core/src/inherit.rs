//! An inherited scene: a model, and what a scene changed about it.
//!
//! The model's nodes are never written down — they are read out of the
//! `.glb` every time — so a re-export reaches every scene built on it. What
//! a scene stores is the difference: a transform the model does not have, a
//! component it does not carry, an entity it does not contain, a node it
//! should not instantiate at all. Each is addressed by the `/`-separated
//! node names from the instance root, `""` being the root itself.
//!
//! See [`docs/notes/inherited-scenes.md`](../../../docs/notes/inherited-scenes.md).

use std::collections::{BTreeMap, HashSet};

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::component::{Component, Entity, World};
use crate::reflect::Export;
use crate::scene_asset::{self, BaseNode, SceneId, SceneTemplate};
use crate::scene_file::{self, Components, Node};
use crate::transform::Transform;

/// The component a *model's* node turns into, and the property naming its
/// mesh. A scene file's nodes carry theirs in the component map like
/// everything else; only glTF keeps it outside one.
///
/// Named rather than asked for: the render crate hands
/// [`scene_asset::drain_ready_spawns`] the step that attaches it, and this is
/// that step's inverse — the one thing here that has to know what it made.
const MODEL_RENDERER: (&str, &str) = ("MeshRenderer", "mesh_id");

/// What [`scene_file`] calls the instance component, so it can take it out of
/// the components map it would otherwise write twice.
pub(crate) const TYPE_NAME: &str = "SceneInstance";

/// One change an inherited scene makes to the model it is built on.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Edit {
    /// `/`-separated node names from the instance root; `""` is the root.
    pub(crate) at: String,
    /// Instantiate neither this node nor anything under it.
    #[serde(default, skip_serializing_if = "not")]
    pub(crate) remove: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) pos: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) rot: Option<Quat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) scale: Option<Vec3>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) components: BTreeMap<String, BTreeMap<String, Json>>,
    /// Entities this scene adds under the node, written in full.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) children: Vec<Node>,
}

fn not(b: &bool) -> bool {
    !b
}

impl Edit {
    /// The node it addresses, for a panel reporting one the model has no
    /// place for.
    pub fn at(&self) -> &str {
        &self.at
    }

    /// Whether it takes the node out rather than changing it.
    pub fn removes(&self) -> bool {
        self.remove
    }

    fn nothing(&self) -> bool {
        !self.remove
            && self.pos.is_none()
            && self.rot.is_none()
            && self.scale.is_none()
            && self.components.is_empty()
            && self.children.is_empty()
    }
}

/// A model instantiated as part of a scene, and that scene's changes to it.
#[derive(Clone, Default, Export)]
pub struct SceneInstance {
    #[export(get = model, set = set_model)]
    pub(crate) model: Option<SceneId>,
    /// Before the model parses, every edit the file carried; after, the ones
    /// its nodes had no place for.
    edits: Vec<Edit>,
    /// The components the instancing scene put on this entity, kept until
    /// the base's root components can go on underneath them — and kept after
    /// that as what to write back if the base never arrives.
    own: Components,
    /// Whether the instantiation has happened. A save before it has nothing
    /// to diff, and writing what it found would empty the file.
    built: bool,
}

impl SceneInstance {
    /// An instance with nothing changed about it yet — what dropping a scene
    /// into a document makes.
    pub fn new(model: SceneId) -> Self {
        Self {
            model: Some(model),
            ..Self::default()
        }
    }

    pub(crate) fn from_file(model: SceneId, edits: Vec<Edit>, own: &Components) -> Self {
        Self {
            model: Some(model),
            edits,
            own: own.clone(),
            built: false,
        }
    }

    pub fn model(&self) -> Option<SceneId> {
        self.model
    }

    /// Refused. The model's nodes are already entities, and a setter has no
    /// world to take them out of — so a scene changes model by being written
    /// and reopened, and the inspector shows the refusal by reading back.
    pub fn set_model(&mut self, _t: &Transform, _model: Option<SceneId>) {
        eprintln!("scene: an instance keeps the model it was made with");
    }

    /// What this scene puts on the instance entity itself, as the file should
    /// hold it: the live components minus everything the base's own root
    /// already puts there. Before the base arrives there is nothing to
    /// subtract, and the file's own are written back untouched.
    pub(crate) fn own_for_file(&self, mut live: Components) -> Components {
        let Some(base) = self
            .model
            .filter(|_| self.built)
            .and_then(scene_asset::template)
        else {
            return self.own.clone();
        };
        let base = base.root();
        live.retain(|ty, props| {
            let Some(had) = base.get(ty) else { return true };
            props.retain(|name, v| had.get(name) != Some(&*v));
            !props.is_empty()
        });
        live
    }

    /// The edits the model had no node for. Kept rather than dropped, and
    /// written back out, so a rename in the model costs a warning instead of
    /// the work that pointed at the old name.
    pub fn unresolved(&self) -> &[Edit] {
        &self.edits
    }
}

impl Component for SceneInstance {
    const HAS_UPDATE: bool = false;
    /// A document has to show the model it inherits from — the same reason a
    /// `MeshRenderer` publishes its record in one.
    const INIT_IN_EDIT: bool = true;

    fn init(&mut self, transform: &Transform) {
        let Some(model) = self.model else { return };
        let root = Entity::new(transform.get_idx());
        let edits = std::mem::take(&mut self.edits);
        let own = self.own.clone();
        scene_asset::spawn_subscene_into(model, transform.world(), root, move |world, root| {
            // The base's root components are already on, put there with the
            // rest of its nodes. This scene's win over them, and an edit at
            // `""` — the same thing said the other way — wins over both.
            scene_file::set_components(world, root, &own);
            let left = apply(world, root, edits);
            if let Some(c) = world.get_component::<SceneInstance>(root) {
                let mut c = c.lock();
                c.edits = left;
                c.built = true;
            }
        });
    }
}

/// The instance component on `idx`, cloned out of its lock — what it holds is
/// read while walking the entity's other components, which takes that lock
/// again.
pub(crate) fn of(world: &World, idx: u32) -> Option<SceneInstance> {
    let c = world.get_component::<SceneInstance>(Entity::new(idx))?;
    let c = c.lock().clone();
    Some(c)
}

impl SceneInstance {
    /// What this scene changed, as the file should hold it.
    ///
    /// Diffed against the model rather than logged as the user types: a value
    /// put back to the model's own stops being an edit, and a node nobody
    /// touched follows the model when the model changes. Before the instance
    /// is built there is nothing to diff, and the file's own edits are
    /// written back untouched.
    pub(crate) fn for_file(&self, world: &World, root: Entity) -> Vec<Edit> {
        let Some(model) = self.model.filter(|_| self.built) else {
            return self.edits.clone();
        };
        let Some(template) = scene_asset::template(model) else {
            return self.edits.clone();
        };
        let mut out = diff(world, root, &template);
        out.extend(self.edits.iter().cloned());
        out.sort_by(|a, b| a.at.cmp(&b.at));
        out
    }
}

// ─── Applying ────────────────────────────────────────────────────────────

/// Apply `edits` to the instance rooted at `root`, returning the ones no node
/// answered to.
pub fn apply(world: &mut World, root: Entity, edits: Vec<Edit>) -> Vec<Edit> {
    let mut unresolved = Vec::new();
    for edit in edits {
        let Some(idx) = resolve(world, root.id, &edit.at) else {
            eprintln!("scene: the model has no node {:?} — edit kept", edit.at);
            unresolved.push(edit);
            continue;
        };
        let e = Entity::new(idx);
        if edit.remove {
            world.remove_entity(e);
            continue;
        }
        {
            let g = world.hierarchy().get_transform_unchecked(idx).lock();
            edit.pos.inspect(|&p| g.set_position(p));
            edit.rot.inspect(|&r| g.set_rotation(r));
            edit.scale.inspect(|&s| g.set_scale(s));
        }
        scene_file::set_components(world, e, &edit.components);
        scene_file::instantiate_nodes(world, idx, &edit.children);
    }
    unresolved
}

/// The entity `path` names under `root`. `None` when nothing answers to a
/// name — or when two things do, which is a path that names neither.
fn resolve(world: &World, root: u32, path: &str) -> Option<u32> {
    let h = world.hierarchy();
    let mut at = root;
    for name in path.split('/').filter(|s| !s.is_empty()) {
        let mut named = (h.children(at).iter()).filter(|&&c| h.name(c) == name);
        let one = *named.next()?;
        if named.next().is_some() {
            return None;
        }
        at = one;
    }
    Some(at)
}

pub(crate) fn has(world: &World, e: Entity, name: &str) -> bool {
    let mut found = false;
    world.entity(e).inspect(|c| found |= c.type_name() == name);
    found
}

// ─── Diffing ─────────────────────────────────────────────────────────────

/// What `root`'s subtree has that `template` does not.
fn diff(world: &World, root: Entity, template: &SceneTemplate) -> Vec<Edit> {
    let mut out = Vec::new();
    let mut added: BTreeMap<String, Vec<Node>> = BTreeMap::new();
    let mut live = HashSet::new();
    walk(
        world, root.id, "", template, &mut out, &mut added, &mut live,
    );

    // A node the model has and the world does not. Only the topmost one:
    // removing it takes everything under it, which is why the descendants
    // are missing too.
    for path in template.paths() {
        let parent = path.rsplit_once('/').map_or("", |(p, _)| p);
        if !live.contains(path) && (parent.is_empty() || live.contains(parent)) {
            out.push(Edit {
                at: path.to_string(),
                remove: true,
                ..Edit::default()
            });
        }
    }
    for (at, children) in added {
        out.push(Edit {
            at,
            children,
            ..Edit::default()
        });
    }
    out.sort_by(|a, b| a.at.cmp(&b.at));
    out
}

fn walk(
    world: &World,
    idx: u32,
    path: &str,
    template: &SceneTemplate,
    out: &mut Vec<Edit>,
    added: &mut BTreeMap<String, Vec<Node>>,
    live: &mut HashSet<String>,
) {
    let h = world.hierarchy();
    for &c in h.children(idx) {
        let sub = match path.is_empty() {
            true => h.name(c).to_string(),
            false => format!("{path}/{}", h.name(c)),
        };
        // Not a node of the model, so the whole subtree is this scene's and
        // is written whole rather than as a difference from nothing.
        let Some(base) = template.node_at(&sub) else {
            added
                .entry(path.to_string())
                .or_default()
                .push(scene_file::node(world, c));
            continue;
        };
        live.insert(sub.clone());
        if let Some(edit) = changed(world, c, &sub, base) {
            out.push(edit);
        }
        walk(world, c, &sub, template, out, added, live);
    }
}

/// One live node against the base's, or `None` when they are the same.
fn changed(world: &World, idx: u32, path: &str, base: BaseNode<'_>) -> Option<Edit> {
    let t = world.hierarchy().get_transform_(idx);
    let mut components = scene_file::components_of(world, idx);
    let mut had = base.components.clone();
    let (renderer, mesh) = MODEL_RENDERER;
    if let Some(m) = base.mesh.and_then(|m| serde_json::to_value(m).ok()) {
        (had.entry(renderer.to_string()).or_default()).insert(mesh.to_string(), m);
    }
    // Per property, not per component: a scene that overrode the material
    // must still follow the base when the base's mesh changes.
    components.retain(|ty, props| {
        let Some(had) = had.get(ty) else { return true };
        props.retain(|name, v| had.get(name) != Some(&*v));
        !props.is_empty()
    });
    let edit = Edit {
        at: path.to_string(),
        pos: (t.position != base.position).then_some(t.position),
        rot: (t.rotation != base.rotation).then_some(t.rotation),
        scale: (t.scale != base.scale).then_some(t.scale),
        components,
        ..Edit::default()
    };
    (!edit.nothing()).then_some(edit)
}
