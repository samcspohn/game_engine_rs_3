//! Undo and redo, one history per document world. A step holds the other side
//! of its change, so undoing and redoing are the same `flip`. Every flip runs
//! at the frame boundary, in queue order, with the world held mutably.
//! See [undo](../../../docs/notes/undo.md).

use crate::inspector::{TRANSFORM, set_trs, trs_values};
use engine::{
    Entity, Value, WorldId,
    component::{EntityRef, World},
    glam::{Quat, Vec3},
    scene_file::{self, Snapshot},
    worlds,
};
use std::sync::Mutex;

static HISTORIES: Mutex<Vec<(WorldId, History)>> = Mutex::new(Vec::new());

/// An entity as steps name it: an index into `keys`, which an undone delete
/// re-binds to the slots the subtree comes back in.
type Key = usize;

#[derive(Default)]
struct History {
    done: Vec<Step>,
    undone: Vec<Step>,
    keys: Vec<EntityRef>,
    /// A scrub in progress: its writes merge into one step.
    gesture: bool,
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

impl History {
    fn key(&mut self, e: EntityRef) -> Key {
        key(&mut self.keys, e)
    }

    fn push(&mut self, step: Step, gesture: bool) {
        self.undone.clear();
        let merge = gesture && self.gesture && self.done.last().is_some_and(|s| s.same(&step));
        if !merge {
            self.done.push(step);
        }
        self.gesture = gesture;
    }
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

/// Start keeping `world`'s history.
pub(crate) fn track(world: WorldId) {
    HISTORIES.lock().unwrap().push((world, History::default()));
}

pub(crate) fn forget(world: WorldId) {
    HISTORIES.lock().unwrap().retain(|(w, _)| *w != world);
}

/// Record a change already made, if `world` keeps a history. Run under the
/// lock: a parent that is a key has to be one before the step names it.
fn record(world: WorldId, step: impl FnOnce(&mut History) -> Step, gesture: bool) {
    let mut all = HISTORIES.lock().unwrap();
    if let Some((_, h)) = all.iter_mut().find(|(w, _)| *w == world) {
        let s = step(h);
        h.push(s, gesture);
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
        |h| Step::Set {
            key: h.key(e),
            ty,
            prop,
            value: before,
        },
        gesture,
    );
}

/// End a scrub: its next write is a step of its own.
pub(crate) fn seal(world: WorldId) {
    let mut all = HISTORIES.lock().unwrap();
    if let Some((_, h)) = all.iter_mut().find(|(w, _)| *w == world) {
        h.gesture = false;
    }
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
        |h| Step::Name {
            key: h.key(e),
            name: before,
        },
        false,
    );
}

/// `e` was the `at`th child of `parent`.
pub(crate) fn moved(e: EntityRef, parent: EntityRef, at: usize) {
    record(
        e.world(),
        |h| Step::Place {
            key: h.key(e),
            parent: h.key(parent),
            at,
        },
        false,
    );
}

/// `ty` was added to `e`.
pub(crate) fn added(e: EntityRef, ty: &'static str) {
    record(e.world(), |h| Step::Component { key: h.key(e), ty }, false);
}

pub(crate) fn spawned(e: EntityRef) {
    record(
        e.world(),
        |h| Step::Life {
            key: h.key(e),
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
        let mut all = HISTORIES.lock().unwrap();
        let mut scratch = History::default();
        let h = match all.iter_mut().find(|(id, _)| *id == e.world()) {
            Some((_, h)) => h,
            None => &mut scratch,
        };
        let mut step = Step::Life {
            key: h.key(e),
            held: None,
        };
        let parent = step.flip(w, &mut h.keys);
        if parent.is_some() {
            h.push(step, false);
        }
        drop(all);
        after(parent);
    });
}

/// Undo `world`'s last step at the frame boundary; `after` hears what it
/// touched.
pub(crate) fn undo(world: WorldId, after: impl FnOnce(Option<Entity>) + Send + 'static) {
    shift(world, true, after);
}

pub(crate) fn redo(world: WorldId, after: impl FnOnce(Option<Entity>) + Send + 'static) {
    shift(world, false, after);
}

fn shift(world: WorldId, back: bool, after: impl FnOnce(Option<Entity>) + Send + 'static) {
    let Some(w) = worlds::world(world) else {
        return;
    };
    w.mutate(move |w| {
        let mut all = HISTORIES.lock().unwrap();
        let Some((_, h)) = all.iter_mut().find(|(id, _)| *id == world) else {
            return;
        };
        h.gesture = false;
        let (from, to) = match back {
            true => (&mut h.done, &mut h.undone),
            false => (&mut h.undone, &mut h.done),
        };
        let Some(mut step) = from.pop() else { return };
        // A step whose entity is gone is dropped rather than retried forever.
        let touched = step.flip(w, &mut h.keys);
        if touched.is_some() {
            to.push(step);
        }
        drop(all);
        after(touched);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{WorldHandle, transform::_Transform};

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

    #[test]
    fn a_scrub_undoes_in_one_step_and_redoes() {
        let w = engine::new_world();
        track(w.id());
        let e = spawn(&w, "a", engine::transform::ROOT);
        for x in [1.0, 2.0, 3.0] {
            move_to(e, x, true);
        }
        seal(w.id());
        move_to(e, 4.0, false);

        undo(w.id(), |_| {});
        undo(w.id(), |_| {});
        boundary(&w);
        assert_eq!(pos(e), Value::Vec3(Vec3::ZERO));
        redo(w.id(), |_| {});
        boundary(&w);
        assert_eq!(pos(e), Value::Vec3(Vec3::X * 3.0));
        forget(w.id());
    }

    /// An entity that comes back is a new slot, and a step recorded against
    /// the old one still lands on it.
    #[test]
    fn an_undone_delete_comes_back_where_it_was() {
        let w = engine::new_world();
        track(w.id());
        let root = engine::transform::ROOT;
        let [a, b, c] = ["a", "b", "c"].map(|n| spawn(&w, n, root));
        let child = spawn(&w, "child", b.entity().id);
        move_to(child, 5.0, false);
        destroy(b, |_| {});
        boundary(&w);
        assert!(b.with(|_, _| ()).is_none());

        undo(w.id(), |_| {});
        undo(w.id(), |_| {});
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

        redo(w.id(), |_| {});
        boundary(&w);
        assert_eq!(
            h.get_transform_unchecked(back).lock().get_position(),
            Vec3::X * 5.0
        );
        forget(w.id());
    }
}
