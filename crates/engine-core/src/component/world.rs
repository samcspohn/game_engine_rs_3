//! A world: a hierarchy, a registry over it, and whether it runs.
//!
//! ADR-0011 §1–2. A world owns its graph outright, so its slot indices are
//! dense and zero-based and an [`Entity`] means nothing outside the world it
//! came from — which is why every lookup goes through [`EntityView`] rather
//! than a bare id. Worlds are engine-owned and refcounted; make one with
//! [`worlds::new_world`].
//!
//! Growing a world reallocates its SoA, so [`spawn`](World::spawn),
//! [`duplicate`](World::duplicate) and [`destroy`](World::destroy) queue and
//! run at the frame boundary, handing the new entity to a builder callback
//! once it exists.

use std::cell::SyncUnsafeCell;
use std::collections::HashMap;
use std::sync::Weak;

use parking_lot::Mutex;

use crate::reflect::Export;
use crate::transform::{
    _Transform, compute::PerfCounter, Transform, TransformHierarchy, WorldId, ROOT,
};
use crate::util::parallel;
use crate::worlds::{self, WorldHandle};

use super::{Component, ComponentRegistry, Entity};

/// What a queued spawn hands its new entity to.
type Build = Box<dyn FnOnce(EntityMut) + Send>;

/// A structural change waiting for the frame boundary.
enum Pending {
    Spawn(_Transform, Build),
    /// The source world is held rather than named: it has to outlive the
    /// frame that queued the copy.
    Duplicate(WorldHandle, Entity, Build),
    Destroy(Entity),
}

/// One world: a scene graph, the components over it, and whether it simulates.
pub struct World {
    hierarchy: TransformHierarchy,
    registry: ComponentRegistry,
    simulating: bool,
    /// Per-type update timings; `Some` to profile. Behind a lock because the
    /// sweep runs through `&self`.
    perf: Mutex<Option<HashMap<String, PerfCounter>>>,
    /// This world's own handle, for queued work that outlives the frame.
    me: Weak<SyncUnsafeCell<World>>,
    pending: Mutex<Vec<Pending>>,
}

impl World {
    /// `id` is the registry slot; [`worlds::new_world`] is the way in.
    pub(crate) fn new(id: WorldId) -> Self {
        Self {
            hierarchy: TransformHierarchy::new(id),
            registry: ComponentRegistry::new(),
            simulating: true,
            perf: Mutex::new(None),
            me: Weak::new(),
            pending: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn set_handle(&mut self, me: Weak<SyncUnsafeCell<World>>) {
        self.me = me;
    }

    /// A refcounted handle to this world, for anything that outlives the frame.
    pub fn handle(&self) -> WorldHandle {
        WorldHandle::new(
            self.me
                .upgrade()
                .expect("a world is only reachable through a handle"),
        )
    }

    pub fn id(&self) -> WorldId {
        self.hierarchy.world()
    }

    /// Whether [`worlds::sweep_all`] visits this world at all.
    pub fn simulating(&self) -> bool {
        self.simulating
    }

    /// `false` is edit mode: the world still renders — renderers are data the
    /// renderer reads directly — but nothing in it is swept (ADR-0010 §5).
    pub fn set_simulating(&mut self, simulating: bool) {
        self.simulating = simulating;
    }

    pub fn hierarchy(&self) -> &TransformHierarchy {
        &self.hierarchy
    }

    pub fn registry(&self) -> &ComponentRegistry {
        &self.registry
    }

    /// Collect per-type update timings from here on, or stop.
    pub fn profile(&self, on: bool) {
        *self.perf.lock() = on.then(HashMap::new);
    }

    /// An entity **in this world** — a bare index means nothing anywhere else.
    pub fn entity(&self, entity: Entity) -> EntityView<'_> {
        EntityView {
            world: self,
            id: entity,
        }
    }

    // ── Structural changes: queued to the frame boundary ────────────────

    /// Spawn an entity here and hand it to `build` once it exists.
    ///
    /// Deferred because growing the hierarchy reallocates the SoA a running
    /// sweep is reading — see [`worlds::apply_pending`].
    pub fn spawn(&self, t: _Transform, build: impl FnOnce(EntityMut) + Send + 'static) {
        self.pending.lock().push(Pending::Spawn(t, Box::new(build)));
    }

    /// Copy `src` and everything under it into this world, at the root.
    ///
    /// A copy and not a move (ADR-0011 §6): the new subtree is a new identity,
    /// and `src`'s world is untouched.
    pub fn duplicate(&self, src: EntityView, build: impl FnOnce(EntityMut) + Send + 'static) {
        self.pending.lock().push(Pending::Duplicate(
            src.world.handle(),
            src.id,
            Box::new(build),
        ));
    }

    /// Remove `entity` and its subtree at the next frame boundary.
    pub fn destroy(&self, entity: Entity) {
        self.pending.lock().push(Pending::Destroy(entity));
    }

    /// Run every queued change; `true` if there was any. A builder may queue
    /// more, which is why the caller loops.
    pub(crate) fn apply_pending(&mut self) -> bool {
        let queued = std::mem::take(&mut *self.pending.lock());
        let any = !queued.is_empty();
        for op in queued {
            let (id, build) = match op {
                Pending::Spawn(t, build) => (self.new_entity(t), build),
                Pending::Duplicate(src, e, build) => (self.copy_subtree(&src, e.id, ROOT), build),
                Pending::Destroy(e) => {
                    self.remove_entity(e);
                    continue;
                }
            };
            build(EntityMut { world: self, id });
        }
        any
    }

    // ── Direct edits: `&mut self`, so only the frame boundary ───────────

    /// Spawn a new entity from a transform descriptor. Returns a handle.
    pub fn new_entity(&mut self, t: _Transform) -> Entity {
        Entity::new(self.hierarchy.create_transform(t).get_idx())
    }

    /// Attach component `T` to `entity`, calling [`Component::init`].
    ///
    /// On first use for type `T` *in this world* the storage is registered
    /// with `T::HAS_UPDATE` — so the same type can be swept in the play world
    /// and dormant in the document, which is the granularity `HAS_UPDATE`
    /// alone cannot express.
    pub fn add_component<T>(&mut self, entity: Entity, mut component: T)
    where
        T: Component + Clone + Send + Sync + 'static,
    {
        let t = self.hierarchy.get_transform_unchecked(entity.id);
        component.init(&t);
        self.registry
            .register::<T>(T::HAS_UPDATE)
            .set(entity.id, component);
    }

    /// Remove component `T` from `entity`, calling [`Component::deinit`]
    /// first.
    pub fn remove_component<T>(&mut self, entity: Entity)
    where
        T: Component + Clone + Send + Sync + 'static,
    {
        let t = self.hierarchy.get_transform_unchecked(entity.id);
        if let Some(storage) = self.registry.get_storage_mut::<T>() {
            if let Some(mutex) = storage.get(entity.id) {
                mutex.lock().deinit(&t);
            }
            storage.drop(entity.id);
        }
    }

    /// Remove `entity`, its descendants, and all of their components,
    /// calling [`Component::deinit`] on each.
    pub fn remove_entity(&mut self, entity: Entity) {
        let t = self.hierarchy.get_transform_unchecked(entity.id).lock();
        // Transforms first: a storage sweep runs against live transforms, so
        // a slot must stop being one before its component stops existing.
        let removed = self.hierarchy.remove_transform(t);
        for &idx in &removed {
            let t = self.hierarchy.get_transform_unchecked(idx);
            // `deinit` while the component is still there: a dropped
            // `MeshRenderer` cannot scatter its own `NO_RENDERER`, and
            // without that the mesh keeps drawing at a dead slot.
            for storage in self.registry.components.values() {
                storage.deinit(idx, &t);
            }
            for storage in self.registry.components.values_mut() {
                storage.remove(idx);
            }
        }
    }

    /// Borrow the `Mutex<T>` for `entity`'s component `T`, or `None`.
    ///
    /// The owner's API. Inside a frame the world hands components out through
    /// a closure instead — see [`EntityView::get_component`].
    pub fn get_component<T>(&self, entity: Entity) -> Option<&Mutex<T>>
    where
        T: Component + Send + Sync + 'static,
    {
        self.registry.get_storage::<T>()?.get(entity.id)
    }

    /// Copy `root`'s subtree out of `src` under `into`, returning its new root.
    fn copy_subtree(&mut self, src: &World, root: u32, into: u32) -> Entity {
        let mut map: HashMap<u32, u32> = HashMap::new();
        for idx in src.hierarchy.subtree(root) {
            let s = src.hierarchy.get_transform_(idx);
            let parent = match idx == root {
                true => into,
                false => {
                    map[&s
                        .parent
                        .expect("only ROOT has none, and it is not in a subtree")]
                }
            };
            let new = self.new_entity(_Transform {
                parent: Some(parent),
                ..s
            });
            map.insert(idx, new.id);
        }
        self.copy_components(src, &map);
        Entity::new(map[&root])
    }

    /// Clone every component from `src`'s slots onto the slots they map to.
    fn copy_components(&mut self, src: &World, map: &HashMap<u32, u32>) {
        for (type_id, storage) in src.registry.components.iter() {
            let into = self
                .registry
                .components
                .entry(*type_id)
                .or_insert_with(|| storage.empty_like());
            for (&s_idx, &d_idx) in map {
                let t = self.hierarchy.get_transform_unchecked(d_idx);
                into.clone_from_other(storage.as_ref(), s_idx, d_idx, &t);
            }
        }
    }

    /// Deep-clone this world into a new one.
    ///
    /// Play mode's shape (ADR-0010 §4): source and destination are separate
    /// hierarchies and separate registries, so the document keeps being
    /// edited while its copy runs.
    pub fn duplicate_world(&self, simulating: bool) -> WorldHandle {
        let out = worlds::new_world();
        // SAFETY: nothing else holds this handle yet, so no sweep can be
        // reading the world it names.
        let w = unsafe { out.get_mut() };
        w.simulating = simulating;
        for &child in self.hierarchy.children(ROOT) {
            w.copy_subtree(self, child, ROOT);
        }
        out
    }

    /// `&self`: the registry sweeps through it and components mutate through
    /// their own `Mutex<T>`, so a frame never needs `&mut`.
    pub(crate) fn sweep(&self, dt: f32) {
        let bitmap_tasks = parallel::bitmap_task_layout(self.hierarchy.len().div_ceil(32));
        let mut perf = self.perf.lock();
        self.registry.update_all(dt, self, bitmap_tasks, &mut perf);
    }
}

/// An entity paired with the world that gives its index meaning.
#[derive(Clone, Copy)]
pub struct EntityView<'a> {
    world: &'a World,
    id: Entity,
}

impl<'a> EntityView<'a> {
    pub fn id(&self) -> Entity {
        self.id
    }

    pub fn world(&self) -> &'a World {
        self.world
    }

    pub fn transform(&self) -> Transform<'a> {
        self.world.hierarchy().get_transform_unchecked(self.id.id)
    }

    /// Run `f` against this entity's `T`; `false` if it has none. A closure
    /// rather than a `&Mutex<T>`, so the lock never leaves the world.
    pub fn get_component<T>(&self, f: impl FnOnce(&mut T)) -> bool
    where
        T: Component + Send + Sync + 'static,
    {
        match self.world.get_component::<T>(self.id) {
            Some(m) => {
                f(&mut m.lock());
                true
            }
            None => false,
        }
    }

    /// Hand every component on this entity to `f`, in no particular order —
    /// what an inspector walks to build its rows.
    pub fn inspect(&self, f: impl FnMut(&mut dyn Export)) {
        self.world.registry().inspect(self.id.id, f);
    }
}

/// A just-created entity, at the boundary where its world is still `&mut`.
///
/// The only place [`World::add_component`] is reachable from game code, which
/// is why every spawn takes a builder.
pub struct EntityMut<'a> {
    world: &'a mut World,
    id: Entity,
}

impl EntityMut<'_> {
    pub fn id(&self) -> Entity {
        self.id
    }

    pub fn world(&self) -> &World {
        self.world
    }

    pub fn transform(&self) -> Transform<'_> {
        self.world.hierarchy().get_transform_unchecked(self.id.id)
    }

    pub fn add_component<T>(&mut self, component: T) -> &mut Self
    where
        T: Component + Clone + Send + Sync + 'static,
    {
        self.world.add_component(self.id, component);
        self
    }

    /// Run `f` against this entity's `T`; `false` if it has none.
    pub fn get_component<T>(&self, f: impl FnOnce(&mut T)) -> bool
    where
        T: Component + Send + Sync + 'static,
    {
        self.world.entity(self.id).get_component(f)
    }

    /// Spawn a child of this entity. Queued like any other spawn, so it lands
    /// in the same boundary's next pass.
    pub fn spawn_child(&self, t: _Transform, build: impl FnOnce(EntityMut) + Send + 'static) {
        self.world.spawn(
            _Transform {
                parent: Some(self.id.id),
                ..t
            },
            build,
        );
    }
}
