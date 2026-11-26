//! Undo and redo: one timeline across every open asset — a document's world,
//! a resource file. A step holds the other side of its change, so undoing
//! and redoing are the same `flip`. A world's flips run at its frame
//! boundary, in queue order, with the world held mutably.
//! See [undo](../../../docs/notes/undo.md).

use crate::inspector::{TRANSFORM, set_trs, trs_values};
use engine::{
    Entity, Value, WorldId,
    component::{EntityRef, World},
    glam::{Quat, Vec3},
    resource::{self, ResId},
    scene_file::{self, Snapshot},
    transform::{Name, Path, ROOT},
    ui::defer,
    worlds,
};
use std::sync::Mutex;

static TIMELINE: Mutex<Timeline> = Mutex::new(Timeline {
    done: Vec::new(),
    undone: Vec::new(),
    worlds: Vec::new(),
    saved: Vec::new(),
    gesture: false,
    next: 0,
});

/// What a step changed, and what saving clears.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Asset {
    Document(WorldId),
    Resource(ResId),
}

struct Timeline {
    done: Vec<Entry>,
    undone: Vec<Entry>,
    /// The documents' worlds.
    worlds: Vec<WorldId>,
    /// The newest step each asset had done when it was last saved.
    saved: Vec<(Asset, Option<u64>)>,
    /// A scrub in progress: its writes merge into one step.
    gesture: bool,
    next: u64,
}

struct Entry {
    id: u64,
    change: Change,
}

enum Change {
    World(WorldId, Step),
    /// A property of a resource, and the value on the other side.
    Resource(ResId, &'static str, Value),
}

impl Change {
    fn asset(&self) -> Asset {
        match self {
            Change::World(w, _) => Asset::Document(*w),
            Change::Resource(id, ..) => Asset::Resource(*id),
        }
    }

    /// A scrub's next write to the same property.
    fn same(&self, other: &Change) -> bool {
        match (self, other) {
            (Change::World(a, s), Change::World(b, t)) => a == b && s.same(t),
            (Change::Resource(a, p, _), Change::Resource(b, q, _)) => (a, p) == (b, q),
            _ => false,
        }
    }
}

/// Each names its entity by `path`, as it is in the world the step flips
/// from next. The timeline is linear, so that world is the one it was made
/// in, wherever the entity's slot went meanwhile.
enum Step {
    /// `ty` is [`TRANSFORM`] for the entity's own position, rotation, scale.
    Set {
        path: Path,
        ty: &'static str,
        prop: &'static str,
        value: Value,
    },
    Name {
        path: Path,
        name: Name,
    },
    /// A parent, the index among its children, and the name it had there,
    /// which a sibling's may have suffixed since.
    Place {
        path: Path,
        parent: Path,
        at: usize,
        name: Name,
    },
    /// Toggles: added when absent, removed when present.
    Component {
        path: Path,
        ty: &'static str,
    },
    /// Alive while `held` is `None`.
    Life {
        path: Path,
        held: Option<Held>,
    },
}

struct Held {
    parent: Path,
    at: usize,
    snapshot: Snapshot,
    pointers: Pointers,
}

/// Properties outside a deleted subtree that pointed into it, with their
/// values then, and each entity those named: its slot then, and its path.
struct Pointers {
    from: Vec<(Path, &'static str, &'static str, Value)>,
    to: Vec<(u32, Path)>,
}

fn path_to(w: &World, e: Entity) -> Option<Path> {
    w.hierarchy().path(e.id)
}

fn path_of(e: EntityRef) -> Option<Path> {
    e.with(path_to).flatten()
}

fn live(w: &World, path: &Path) -> Option<Entity> {
    w.hierarchy().find(ROOT, path).map(Entity::new)
}

impl Timeline {
    fn push(&mut self, change: Change, gesture: bool) {
        self.undone.clear();
        let merge =
            gesture && self.gesture && self.done.last().is_some_and(|e| e.change.same(&change));
        if !merge {
            self.next += 1;
            self.done.push(Entry {
                id: self.next,
                change,
            });
        }
        self.gesture = gesture;
        changed();
    }

    /// The newest done step on `asset`.
    fn head(&self, asset: Asset) -> Option<u64> {
        let e = self.done.iter().rev().find(|e| e.change.asset() == asset)?;
        Some(e.id)
    }

    fn saved_at(&self, asset: Asset) -> Option<u64> {
        let s = self.saved.iter().find(|(a, _)| *a == asset);
        s.and_then(|(_, at)| *at)
    }
}

/// Unsaved markers read the timeline; the root hears it moved.
fn changed() {
    defer(|ui| {
        let root = ui.root();
        ui.notify(root);
    });
}

fn alive(w: &World, e: EntityRef) -> Option<Entity> {
    (EntityRef::new(w, e.entity())? == e).then_some(e.entity())
}

fn under(w: &World, mut idx: u32, root: u32) -> bool {
    let h = w.hierarchy();
    loop {
        if idx == root {
            return true;
        }
        match h.get_transform_unchecked(idx).lock().get_parent() {
            Some(p) => idx = p,
            None => return false,
        }
    }
}

/// Every entity slot `v` holds, nested ones included.
fn each(v: &mut Value, f: &mut impl FnMut(&mut Option<Entity>)) {
    match v {
        Value::Entity(e) => f(e),
        Value::Struct(fields) => fields.iter_mut().for_each(|(_, v)| each(v, f)),
        Value::List(vs) => vs.iter_mut().for_each(|v| each(v, f)),
        _ => {}
    }
}

/// Note: inspects every entity outside `root`, once per delete.
fn pointers(w: &World, root: u32) -> Pointers {
    let h = w.hierarchy();
    let into =
        |e: &Option<Entity>| e.filter(|e| EntityRef::new(w, *e).is_some() && under(w, e.id, root));
    let (mut from, mut to) = (Vec::new(), Vec::new());
    let mut stack = vec![ROOT];
    while let Some(idx) = stack.pop() {
        if idx == root {
            continue;
        }
        stack.extend(h.children(idx));
        let mut found = Vec::new();
        w.entity(Entity::new(idx)).inspect(|c| {
            for p in c.properties().iter().filter(|p| p.kind.refers()) {
                let Some(mut v) = c.get(p.name) else { continue };
                let mut hit = false;
                each(&mut v, &mut |e| {
                    if let Some(e) = into(e) {
                        hit = true;
                        if !to.iter().any(|(id, _)| *id == e.id) {
                            to.extend(path_to(w, e).map(|at| (e.id, at)));
                        }
                    }
                });
                if hit {
                    found.push((c.type_name(), p.name, v));
                }
            }
        });
        if let Some(at) = path_to(w, Entity::new(idx)) {
            from.extend(found.into_iter().map(|(ty, p, v)| (at.clone(), ty, p, v)));
        }
    }
    Pointers { from, to }
}

impl Pointers {
    /// Point them at the entities now at the paths theirs had.
    fn restore(self, w: &World) {
        let now: Vec<(u32, Option<Entity>)> = (self.to.iter())
            .map(|(was, at)| (*was, live(w, at)))
            .collect();
        for (holder, ty, prop, mut v) in self.from {
            let Some(e) = live(w, &holder) else { continue };
            each(&mut v, &mut |r| {
                if let Some((_, n)) = now.iter().find(|(was, _)| Some(*was) == r.map(|r| r.id)) {
                    *r = *n;
                }
            });
            let t = w.hierarchy().get_transform_unchecked(e.id);
            w.entity(e).inspect(|c| {
                if c.type_name() == ty {
                    c.set(prop, v.clone(), &t);
                }
            });
        }
    }
}

/// Remove `e`, holding what puts it back, and hand back its parent.
fn take(w: &mut World, e: Entity) -> Option<(Entity, Held)> {
    let h = w.hierarchy();
    let was = h.get_transform_unchecked(e.id).lock().get_parent()?;
    let held = Held {
        at: h.children(was).position(|c| c == e.id)?,
        parent: path_to(w, Entity::new(was))?,
        snapshot: scene_file::snapshot(w, e),
        pointers: pointers(w, e.id),
    };
    w.remove_entity(e);
    Some((Entity::new(was), held))
}

impl Step {
    /// A scrub's next write to the same property.
    fn same(&self, other: &Step) -> bool {
        matches!((self, other), (
            Step::Set { path, ty, prop, .. },
            Step::Set { path: q, ty: t, prop: p, .. },
        ) if (path, ty, prop) == (q, t, p))
    }

    /// Swap the world's side of the change with the one held here, and say
    /// which entity to select after: `None` if it no longer resolves.
    fn flip(&mut self, w: &mut World) -> Option<Entity> {
        let h = w.hierarchy();
        match self {
            Step::Set {
                path,
                ty,
                prop,
                value,
            } => {
                let e = live(w, path)?;
                let t = h.get_transform_unchecked(e.id);
                if *ty == TRANSFORM {
                    let (_, _, now) = trs_values(&t).into_iter().find(|v| v.1 == *prop)?;
                    set_trs(&t, prop, value);
                    *value = now;
                    return Some(e);
                }
                let mut now = None;
                w.entity(e).inspect(|c| {
                    if c.type_name() == *ty {
                        now = c.get(prop).filter(|_| c.set(prop, value.clone(), &t));
                    }
                });
                *value = now?;
                Some(e)
            }
            Step::Name { path, name } => {
                let e = live(w, path)?;
                let t = h.get_transform_unchecked(e.id).lock();
                let now = t.get_name();
                h.set_name(&t, *name);
                (*path, *name) = (path_to(w, e)?, now);
                Some(e)
            }
            Step::Place {
                path,
                parent,
                at,
                name,
            } => {
                let (e, p) = (live(w, path)?, live(w, parent)?);
                let t = h.get_transform_unchecked(e.id).lock();
                let was = t.get_parent()?;
                let index = h.children(was).position(|c| c == e.id)?;
                let named = t.get_name();
                match was == p.id {
                    true => h.move_child(&t, *at),
                    false => h.set_parent_at(&t, Some(p.id), *at),
                }
                h.set_name(&t, *name);
                *path = path_to(w, e)?;
                (*parent, *at, *name) = (path_to(w, Entity::new(was))?, index, named);
                Some(e)
            }
            Step::Component { path, ty } => {
                let e = live(w, path)?;
                let kind = engine::script::find(ty)?;
                let mut present = false;
                w.entity(e).inspect(|c| present |= c.type_name() == *ty);
                w.build(e, |mut m| match present {
                    true => (kind.remove)(&mut m),
                    false => (kind.add)(&mut m),
                });
                Some(e)
            }
            Step::Life {
                path,
                held: held @ None,
            } => {
                let (was, took) = take(w, live(w, path)?)?;
                *held = Some(took);
                Some(was)
            }
            Step::Life { path, held } => {
                let p = live(w, &held.as_ref()?.parent)?;
                let Held {
                    at,
                    snapshot,
                    pointers,
                    ..
                } = held.take()?;
                let new = scene_file::restore(w, p.id, &snapshot);
                let h = w.hierarchy();
                h.move_child(&h.get_transform_unchecked(new[0].id).lock(), at);
                *path = path_to(w, new[0])?;
                pointers.restore(w);
                Some(new[0])
            }
        }
    }
}

/// Start keeping `world`'s steps, as saved.
pub(crate) fn track(world: WorldId) {
    TIMELINE.lock().unwrap().worlds.push(world);
}

/// Drop `world` and every step on it — its document closed or reloaded.
pub(crate) fn forget(world: WorldId) {
    let mut t = TIMELINE.lock().unwrap();
    let doc = Asset::Document(world);
    t.worlds.retain(|w| *w != world);
    t.saved.retain(|(a, _)| *a != doc);
    t.done.retain(|e| e.change.asset() != doc);
    t.undone.retain(|e| e.change.asset() != doc);
    drop(t);
    changed();
}

/// Whether `asset` has changed since it was last saved.
pub(crate) fn unsaved(asset: Asset) -> bool {
    let t = TIMELINE.lock().unwrap();
    t.head(asset) != t.saved_at(asset)
}

/// Every resource with changes not yet saved.
pub(crate) fn unsaved_resources() -> Vec<ResId> {
    let t = TIMELINE.lock().unwrap();
    let mut out: Vec<ResId> = Vec::new();
    for e in &t.done {
        if let Asset::Resource(id) = e.change.asset()
            && !out.contains(&id)
            && t.head(Asset::Resource(id)) != t.saved_at(Asset::Resource(id))
        {
            out.push(id);
        }
    }
    out
}

/// `asset` was just written out as it stands.
pub(crate) fn saved(asset: Asset) {
    let mut t = TIMELINE.lock().unwrap();
    let head = t.head(asset);
    t.saved.retain(|(a, _)| *a != asset);
    t.saved.push((asset, head));
    drop(t);
    changed();
}

/// Record a change already made, if `world` is a document's and its entity
/// has a path: an anonymous one is not addressable.
fn record(world: WorldId, step: Option<Step>, gesture: bool) {
    let mut t = TIMELINE.lock().unwrap();
    if let Some(s) = step
        && t.worlds.contains(&world)
    {
        t.push(Change::World(world, s), gesture);
    }
}

/// A property of `e` was `before`. `gesture` merges a scrub's frames.
pub(crate) fn set(
    e: EntityRef,
    ty: &'static str,
    prop: &'static str,
    before: Value,
    gesture: bool,
) {
    let step = path_of(e).map(|path| Step::Set {
        path,
        ty,
        prop,
        value: before,
    });
    record(e.world(), step, gesture);
}

/// A property of resource `id` was `before`.
pub(crate) fn set_resource(id: ResId, prop: &'static str, before: Value, gesture: bool) {
    let mut t = TIMELINE.lock().unwrap();
    t.push(Change::Resource(id, prop, before), gesture);
}

/// End a scrub: its next write is a step of its own.
pub(crate) fn seal() {
    TIMELINE.lock().unwrap().gesture = false;
}

/// The gizmo moved `e` from `start`; one step per part it changed.
pub(crate) fn dragged(world: WorldId, e: Entity, start: (Vec3, Quat, Vec3)) {
    let Some((e, now)) = worlds::world(world).and_then(|w| {
        let r = EntityRef::new(&w, e)?;
        Some((r, trs_values(&w.hierarchy().get_transform_unchecked(e.id))))
    }) else {
        return;
    };
    let was = [
        Value::Vec3(start.0),
        Value::Quat(start.1),
        Value::Vec3(start.2),
    ];
    for ((_, prop, now), was) in now.into_iter().zip(was) {
        if now != was {
            set(e, TRANSFORM, prop, was, false);
        }
    }
}

pub(crate) fn renamed(e: EntityRef, before: Name) {
    let step = path_of(e).map(|path| Step::Name { path, name: before });
    record(e.world(), step, false);
}

/// `e` was the `at`th child of `parent`, named `name`.
pub(crate) fn moved(e: EntityRef, parent: EntityRef, at: usize, name: Name) {
    let step = path_of(e)
        .zip(path_of(parent))
        .map(|(path, parent)| Step::Place {
            path,
            parent,
            at,
            name,
        });
    record(e.world(), step, false);
}

/// `ty` was added to `e`.
pub(crate) fn added(e: EntityRef, ty: &'static str) {
    let step = path_of(e).map(|path| Step::Component { path, ty });
    record(e.world(), step, false);
}

pub(crate) fn spawned(w: &World, e: Entity) {
    let step = path_to(w, e).map(|path| Step::Life { path, held: None });
    record(w.id(), step, false);
}

/// Delete `e` at the frame boundary, holding it to put back. `after` hears
/// its parent, which is what to select now.
pub(crate) fn destroy(e: EntityRef, after: impl FnOnce(Option<Entity>) + Send + 'static) {
    let Some(w) = worlds::world(e.world()) else {
        return;
    };
    w.mutate(move |w| {
        let Some((path, (parent, held))) =
            alive(w, e).and_then(|e| Some((path_to(w, e), take(w, e)?)))
        else {
            return after(None);
        };
        let step = path.map(|path| Step::Life {
            path,
            held: Some(held),
        });
        record(e.world(), step, false);
        after(Some(parent));
    });
}

/// The asset the next undo (`back`) or redo would change.
pub(crate) fn next(back: bool) -> Option<Asset> {
    let t = TIMELINE.lock().unwrap();
    let from = if back { &t.done } else { &t.undone };
    Some(from.last()?.change.asset())
}

/// Undo (`back`) or redo the newest step, wherever it is. `after` hears its
/// asset and the entity it touched, if it was a document's.
pub(crate) fn shift(back: bool, after: impl FnOnce(Asset, Option<Entity>) + Send + 'static) {
    let mut t = TIMELINE.lock().unwrap();
    t.gesture = false;
    let Some(mut entry) = (if back { t.done.pop() } else { t.undone.pop() }) else {
        return;
    };
    let (id, asset) = (entry.id, entry.change.asset());
    let world = match &mut entry.change {
        Change::Resource(res, prop, value) => {
            let prop = *prop;
            let flipped = resource::edit(*res, |c, tr| {
                let now = c.get(prop).filter(|_| c.set(prop, value.clone(), tr));
                now.map(|now| *value = now)
            });
            if flipped.is_some() {
                to(&mut t, back).push(entry);
            }
            drop(t);
            changed();
            return after(asset, None);
        }
        Change::World(world, _) => *world,
    };
    // On the other side now, so a second press before the boundary sees the
    // order the flips will run in.
    to(&mut t, back).push(entry);
    drop(t);
    let Some(w) = worlds::world(world) else {
        return;
    };
    w.mutate(move |w| {
        let mut t = TIMELINE.lock().unwrap();
        let Timeline { done, undone, .. } = &mut *t;
        let entry = done
            .iter_mut()
            .chain(undone.iter_mut())
            .find(|e| e.id == id);
        let touched = match entry {
            Some(Entry {
                change: Change::World(_, step),
                ..
            }) => step.flip(w),
            _ => None,
        };
        // A step whose entity is gone is dropped rather than retried forever.
        if touched.is_none() {
            done.retain(|e| e.id != id);
            undone.retain(|e| e.id != id);
        }
        drop(t);
        changed();
        after(asset, touched);
    });
}

fn to(t: &mut Timeline, back: bool) -> &mut Vec<Entry> {
    if back { &mut t.undone } else { &mut t.done }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Export, Res, WorldHandle, inherit, transform::_Transform};

    /// The timeline is one per process, so tests that use it take turns.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn spawn(w: &WorldHandle, name: &str, parent: u32) -> EntityRef {
        let e = unsafe { w.get_mut() }.new_entity(_Transform {
            name: name.into(),
            parent: Some(parent),
            .._Transform::default()
        });
        EntityRef::new(w, e).expect("just made")
    }

    fn boundary(w: &WorldHandle) {
        unsafe { worlds::apply_pending(std::slice::from_ref(w)) }
    }

    fn pos(e: EntityRef) -> Value {
        e.with(|w, e| {
            trs_values(&w.hierarchy().get_transform_unchecked(e.id))[0]
                .2
                .clone()
        })
        .expect("alive")
    }

    fn move_to(e: EntityRef, x: f32, gesture: bool) {
        let before = pos(e);
        e.with(|w, e| {
            set_trs(
                &w.hierarchy().get_transform_unchecked(e.id),
                "position",
                &Value::Vec3(Vec3::X * x),
            )
        });
        set(e, TRANSFORM, "position", before, gesture);
    }

    fn undo() {
        shift(true, |_, _| {});
    }

    fn redo() {
        shift(false, |_, _| {});
    }

    #[test]
    fn a_scrub_undoes_in_one_step_and_redoes() {
        let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let w = engine::new_world();
        track(w.id());
        let e = spawn(&w, "a", engine::transform::ROOT);
        for x in [1.0, 2.0, 3.0] {
            move_to(e, x, true);
        }
        seal();
        move_to(e, 4.0, false);

        undo();
        undo();
        boundary(&w);
        assert_eq!(pos(e), Value::Vec3(Vec3::ZERO));
        redo();
        boundary(&w);
        assert_eq!(pos(e), Value::Vec3(Vec3::X * 3.0));
        forget(w.id());
    }

    /// An entity that comes back is a new slot, and a step recorded against
    /// the old one still lands on it.
    #[test]
    fn an_undone_delete_comes_back_where_it_was() {
        let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let w = engine::new_world();
        track(w.id());
        let root = engine::transform::ROOT;
        let [a, b, c] = ["a", "b", "c"].map(|n| spawn(&w, n, root));
        let child = spawn(&w, "child", b.entity().id);
        move_to(child, 5.0, false);
        destroy(b, |_| {});
        boundary(&w);
        assert!(b.with(|_, _| ()).is_none());

        undo();
        undo();
        boundary(&w);
        let h = w.hierarchy();
        let kids: Vec<u32> = h.children(root).collect();
        assert_eq!(kids.len(), 3);
        assert_eq!([kids[0], kids[2]], [a.entity().id, c.entity().id]);
        assert_eq!(h.name(kids[1]), "b");
        let back = h.children(kids[1]).next().unwrap();
        assert_eq!(
            h.get_transform_unchecked(back).lock().get_position(),
            Vec3::ZERO
        );

        redo();
        boundary(&w);
        assert_eq!(
            h.get_transform_unchecked(back).lock().get_position(),
            Vec3::X * 5.0
        );
        forget(w.id());
    }

    #[test]
    fn an_undone_move_takes_back_the_suffix_it_got() {
        let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let w = engine::new_world();
        track(w.id());
        let root = engine::transform::ROOT;
        let [a, _] = ["a", "e"].map(|n| spawn(&w, n, root));
        let e = spawn(&w, "e", a.entity().id);
        let h = w.hierarchy();
        let t = h.get_transform_unchecked(e.entity().id).lock();
        let named = t.get_name();
        h.set_parent_at(&t, None, 0);
        drop(t);
        assert_eq!(h.name(e.entity().id), "e1");
        moved(e, a, 0, named);

        undo();
        boundary(&w);
        assert_eq!(h.name(e.entity().id), "e");
        redo();
        boundary(&w);
        assert_eq!(h.name(e.entity().id), "e1");
        forget(w.id());
    }

    /// A sub-scene's nodes come back from its scene, not the snapshot, and a
    /// step recorded against one still finds it by path.
    #[test]
    fn an_undone_delete_finds_a_sub_scenes_node() {
        let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let base = std::env::temp_dir().join(format!("undo_base_{}.json", std::process::id()));
        std::fs::write(
            &base,
            r#"{"version":3,"entities":[{"name":"arm","children":[{"name":"hand"}]}]}"#,
        )
        .expect("write base");
        let w = engine::new_world();
        track(w.id());
        let it = spawn(&w, "it", engine::transform::ROOT);
        let scene = engine::scene_asset::request_scene(&base);
        unsafe { w.get_mut() }.add_component(it.entity(), inherit::SceneInstance::new(scene));
        boundary(&w);
        let hand = |root: u32| {
            let at = inherit::resolve(&w, root, "arm/hand").expect("instantiated");
            EntityRef::new(&w, Entity::new(at)).expect("alive")
        };
        move_to(hand(it.entity().id), 5.0, false);
        destroy(it, |_| {});
        boundary(&w);

        undo();
        undo();
        boundary(&w);
        let back = w
            .hierarchy()
            .children(engine::transform::ROOT)
            .next()
            .unwrap();
        assert_eq!(pos(hand(back)), Value::Vec3(Vec3::ZERO));
        redo();
        boundary(&w);
        assert_eq!(pos(hand(back)), Value::Vec3(Vec3::X * 5.0));
        forget(w.id());
        std::fs::remove_file(&base).ok();
    }

    #[derive(Clone, Default, Export)]
    struct Aim {
        #[export]
        at: Option<Entity>,
    }
    impl engine::component::Component for Aim {}

    fn aim(w: &WorldHandle, e: EntityRef) -> Option<Value> {
        let mut at = None;
        w.entity(e.entity())
            .inspect(|c| at = at.take().or(c.get("at")));
        at
    }

    /// What pointed into a deleted subtree points at what comes back, though
    /// its old slots were taken meanwhile.
    #[test]
    fn an_undone_delete_is_pointed_at_again() {
        let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let w = engine::new_world();
        track(w.id());
        let [a, b] = ["a", "b"].map(|n| spawn(&w, n, ROOT));
        let child = spawn(&w, "child", b.entity().id);
        let at = Some(child.entity());
        unsafe { w.get_mut() }.add_component(a.entity(), Aim { at });
        destroy(b, |_| {});
        boundary(&w);
        let squatters = ["s", "t"].map(|n| spawn(&w, n, a.entity().id).entity());
        assert!(squatters.contains(&child.entity()), "the slot was taken");

        undo();
        boundary(&w);
        let back = w.hierarchy().find(ROOT, "b/child").map(Entity::new);
        assert!(back.is_some() && back != at);
        assert_eq!(aim(&w, a), Some(Value::Entity(back)));
        forget(w.id());
    }

    #[derive(Clone, Default, Export)]
    #[export(resource = "testgloss")]
    struct Gloss {
        #[export]
        amount: f32,
    }

    /// One timeline: the newest step is undone first whichever asset it is
    /// on, and each asset knows on its own whether it is saved.
    #[test]
    fn undo_crosses_assets_in_the_order_they_changed() {
        let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let w = engine::new_world();
        track(w.id());
        let doc = Asset::Document(w.id());
        let e = spawn(&w, "a", engine::transform::ROOT);
        let path = std::env::temp_dir().join(format!("undo_{}.testgloss", std::process::id()));
        let gloss = Res::<Gloss>::request(&path).id();
        let res = Asset::Resource(gloss);

        move_to(e, 1.0, false);
        resource::edit(gloss, |c, t| c.set("amount", Value::F32(0.5), t));
        set_resource(gloss, "amount", Value::F32(0.0), false);
        saved(res);
        assert!(unsaved(doc) && !unsaved(res));

        assert_eq!(next(true), Some(res));
        undo();
        assert_eq!(
            resource::read(gloss, |c| c.get("amount")),
            Some(Value::F32(0.0))
        );
        assert!(unsaved(res), "undone past its save");
        assert_eq!(next(true), Some(doc));
        undo();
        boundary(&w);
        assert_eq!(pos(e), Value::Vec3(Vec3::ZERO));
        assert!(!unsaved(doc), "back where it was opened");

        forget(w.id());
        redo();
        assert_eq!(next(false), None, "a closed document's steps are gone");
        assert_eq!(
            resource::read(gloss, |c| c.get("amount")),
            Some(Value::F32(0.5))
        );
    }
}
