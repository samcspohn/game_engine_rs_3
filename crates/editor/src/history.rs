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

/// An entity as steps name it: an index into its world's keys, which an
/// undone delete re-binds to the slots the subtree comes back in.
type Key = usize;

struct Timeline {
    done: Vec<Entry>,
    undone: Vec<Entry>,
    /// Each document's world, with the keys its steps name entities by.
    worlds: Vec<(WorldId, Vec<EntityRef>)>,
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

enum Step {
    /// `ty` is [`TRANSFORM`] for the entity's own position, rotation, scale.
    Set {
        key: Key,
        ty: &'static str,
        prop: &'static str,
        value: Value,
    },
    Name {
        key: Key,
        name: String,
    },
    /// A parent, and the index among its children.
    Place {
        key: Key,
        parent: Key,
        at: usize,
    },
    /// Toggles: added when absent, removed when present.
    Component {
        key: Key,
        ty: &'static str,
    },
    /// Alive while `held` is `None`.
    Life {
        key: Key,
        held: Option<Held>,
    },
}

struct Held {
    parent: Key,
    at: usize,
    snapshot: Snapshot,
    /// Which of the snapshot's entities a key named, by position.
    keyed: Vec<(usize, Key)>,
}

fn key(keys: &mut Vec<EntityRef>, e: EntityRef) -> Key {
    keys.iter().position(|&k| k == e).unwrap_or_else(|| {
        keys.push(e);
        keys.len() - 1
    })
}

/// The key of slot `idx`'s current occupant.
fn key_at(w: &World, keys: &mut Vec<EntityRef>, idx: u32) -> Option<Key> {
    Some(key(keys, EntityRef::new(w, Entity::new(idx))?))
}

impl Timeline {
    fn keys(&mut self, world: WorldId) -> Option<&mut Vec<EntityRef>> {
        Some(&mut self.worlds.iter_mut().find(|(w, _)| *w == world)?.1)
    }

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

/// The live entity `k` names in `w`.
fn live(w: &World, keys: &[EntityRef], k: Key) -> Option<Entity> {
    let e = keys[k];
    let now = EntityRef::new(w, e.entity())?;
    (now == e).then_some(e.entity())
}

impl Step {
    /// A scrub's next write to the same property.
    fn same(&self, other: &Step) -> bool {
        matches!((self, other), (
            Step::Set { key, ty, prop, .. },
            Step::Set { key: k, ty: t, prop: p, .. },
        ) if (key, ty, prop) == (k, t, p))
    }

    /// Swap the world's side of the change with the one held here, and say
    /// which entity to select after: `None` if it no longer resolves.
    fn flip(&mut self, w: &mut World, keys: &mut Vec<EntityRef>) -> Option<Entity> {
        let h = w.hierarchy();
        match self {
            Step::Set {
                key,
                ty,
                prop,
                value,
            } => {
                let e = live(w, keys, *key)?;
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
            Step::Name { key, name } => {
                let e = live(w, keys, *key)?;
                let t = h.get_transform_unchecked(e.id).lock();
                let now = t.get_name();
                h.set_name(&t, name);
                *name = now;
                Some(e)
            }
            Step::Place { key, parent, at } => {
                let (e, p) = (live(w, keys, *key)?, live(w, keys, *parent)?);
                let t = h.get_transform_unchecked(e.id).lock();
                let was = t.get_parent()?;
                let index = h.children(was).iter().position(|&c| c == e.id)?;
                match was == p.id {
                    true => h.move_child(&t, *at),
                    false => h.set_parent_at(&t, Some(p.id), *at),
                }
                (*parent, *at) = (key_at(w, keys, was)?, index);
                Some(e)
            }
            Step::Component { key, ty } => {
                let e = live(w, keys, *key)?;
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
                key,
                held: held @ None,
            } => {
                let e = live(w, keys, *key)?;
                let was = h.get_transform_unchecked(e.id).lock().get_parent()?;
                let at = h.children(was).iter().position(|&c| c == e.id)?;
                let snapshot = scene_file::snapshot(w, e);
                let keyed = (snapshot.entities().iter().enumerate())
                    .filter_map(|(i, s)| {
                        let k = (0..keys.len()).find(|&k| live(w, keys, k) == Some(*s))?;
                        Some((i, k))
                    })
                    .collect();
                let parent = key_at(w, keys, was)?;
                w.remove_entity(e);
                *held = Some(Held {
                    parent,
                    at,
                    snapshot,
                    keyed,
                });
                Some(Entity::new(was))
            }
            Step::Life { held, .. } => {
                let p = live(w, keys, held.as_ref()?.parent)?;
                let Held {
                    at,
                    snapshot,
                    keyed,
                    ..
                } = held.take()?;
                let new = scene_file::restore(w, p.id, &snapshot);
                let h = w.hierarchy();
                h.move_child(&h.get_transform_unchecked(new[0].id).lock(), at);
                for (i, k) in keyed {
                    keys[k] = EntityRef::new(w, new[i])?;
                }
                Some(new[0])
            }
        }
    }
}

/// Start keeping `world`'s steps, as saved.
pub(crate) fn track(world: WorldId) {
    TIMELINE.lock().unwrap().worlds.push((world, Vec::new()));
}

/// Drop `world` and every step on it — its document closed or reloaded.
pub(crate) fn forget(world: WorldId) {
    let mut t = TIMELINE.lock().unwrap();
    let doc = Asset::Document(world);
    t.worlds.retain(|(w, _)| *w != world);
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

/// Record a change already made, if `world` is a document's. Under the lock:
/// a parent that is a key has to be one before the step names it.
fn record(world: WorldId, step: impl FnOnce(&mut Vec<EntityRef>) -> Step, gesture: bool) {
    let mut t = TIMELINE.lock().unwrap();
    if let Some(keys) = t.keys(world) {
        let s = step(keys);
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
    record(
        e.world(),
        |k| Step::Set {
            key: key(k, e),
            ty,
            prop,
            value: before,
        },
        gesture,
    );
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

pub(crate) fn renamed(e: EntityRef, before: String) {
    record(
        e.world(),
        |k| Step::Name {
            key: key(k, e),
            name: before,
        },
        false,
    );
}

/// `e` was the `at`th child of `parent`.
pub(crate) fn moved(e: EntityRef, parent: EntityRef, at: usize) {
    record(
        e.world(),
        |k| Step::Place {
            key: key(k, e),
            parent: key(k, parent),
            at,
        },
        false,
    );
}

/// `ty` was added to `e`.
pub(crate) fn added(e: EntityRef, ty: &'static str) {
    record(e.world(), |k| Step::Component { key: key(k, e), ty }, false);
}

pub(crate) fn spawned(e: EntityRef) {
    record(
        e.world(),
        |k| Step::Life {
            key: key(k, e),
            held: None,
        },
        false,
    );
}

/// Delete `e` at the frame boundary, holding it to put back. `after` hears
/// its parent, which is what to select now.
pub(crate) fn destroy(e: EntityRef, after: impl FnOnce(Option<Entity>) + Send + 'static) {
    let Some(w) = worlds::world(e.world()) else {
        return;
    };
    w.mutate(move |w| {
        let mut t = TIMELINE.lock().unwrap();
        let mut scratch = Vec::new();
        let tracked = t.keys(e.world()).is_some();
        let keys = t.keys(e.world()).unwrap_or(&mut scratch);
        let mut step = Step::Life {
            key: key(keys, e),
            held: None,
        };
        let parent = step.flip(w, keys);
        if parent.is_some() && tracked {
            t.push(Change::World(e.world(), step), false);
        }
        drop(t);
        after(parent);
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
        let Timeline {
            done,
            undone,
            worlds,
            ..
        } = &mut *t;
        let keys = worlds
            .iter_mut()
            .find(|(id, _)| *id == world)
            .map(|(_, k)| k);
        let entry = done
            .iter_mut()
            .chain(undone.iter_mut())
            .find(|e| e.id == id);
        let touched = match (keys, entry) {
            (
                Some(keys),
                Some(Entry {
                    change: Change::World(_, step),
                    ..
                }),
            ) => step.flip(w, keys),
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
    use engine::{Export, Res, WorldHandle, transform::_Transform};

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
        let kids = h.children(root).to_vec();
        assert_eq!(kids.len(), 3);
        assert_eq!([kids[0], kids[2]], [a.entity().id, c.entity().id]);
        assert_eq!(h.name(kids[1]), "b");
        let back = h.children(kids[1])[0];
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
