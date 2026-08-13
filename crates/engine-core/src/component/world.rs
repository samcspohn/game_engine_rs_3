//! A world: a hierarchy, a registry over it, and whether it runs.
//!
//! ADR-0011 §1–2. A world owns its graph outright, so its slot indices are
//! dense and zero-based and an [`Entity`] means nothing outside the world it
//! came from — which is why every lookup goes through [`EntityView`] rather
//! than a bare id. Worlds are ordinary owned values; the list of them the
//! frame is running is published by [`World::sweep_all`] (see [`worlds`]).

use std::collections::HashMap;

use parking_lot::Mutex;

use crate::reflect::Export;
use crate::transform::{
    compute::PerfCounter, Transform, TransformHierarchy, WorldId, _Transform, ROOT,
};
use crate::util::parallel;
use crate::worlds;

use super::{Component, ComponentRegistry, Entity};

/// One world: a scene graph, the components over it, and whether it simulates.
pub struct World {
    hierarchy: TransformHierarchy,
    registry: ComponentRegistry,
    simulating: bool,
    /// Per-type update timings; `Some` to profile. Behind a lock because the
    /// sweep runs through `&self` — see [`World::sweep_all`].
    perf: Mutex<Option<HashMap<String, PerfCounter>>>,
}

impl World {
    /// `id` is this world's position in the list handed to the window.
    pub fn new(id: WorldId) -> Self {
        Self {
            hierarchy: TransformHierarchy::new(id),
            registry: ComponentRegistry::new(),
            simulating: true,
            perf: Mutex::new(None),
        }
    }

    pub fn id(&self) -> WorldId {
        self.hierarchy.world()
    }

    /// Whether [`sweep_all`](Self::sweep_all) visits this world at all.
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

    /// Deep-clone this world into a new one with id `id`.
    ///
    /// Play mode's shape (ADR-0010 §4): source and destination are separate
    /// hierarchies and separate registries, so the document keeps being
    /// edited while its copy runs.
    pub fn instantiate(&self, id: WorldId, simulating: bool) -> World {
        let mut out = World::new(id);
        out.simulating = simulating;
        let mut map: HashMap<u32, u32> = HashMap::from([(ROOT, ROOT)]);
        for idx in self.hierarchy.subtree(ROOT).into_iter().skip(1) {
            let s = self.hierarchy.get_transform_(idx);
            let parent = map[&s.parent.expect("only ROOT has none, and it is skipped")];
            let new = out.new_entity(_Transform {
                parent: Some(parent),
                ..s
            });
            map.insert(idx, new.id);
        }
        for (type_id, storage) in self.registry.components.iter() {
            let into = out
                .registry
                .components
                .entry(*type_id)
                .or_insert_with(|| storage.empty_like());
            for (&s_idx, &d_idx) in &map {
                let t = out.hierarchy.get_transform_unchecked(d_idx);
                into.clone_from_other(storage.as_ref(), s_idx, d_idx, &t);
            }
        }
        out
    }

    /// Advance every **simulating** world in `worlds` by `dt` seconds.
    ///
    /// A world that does not simulate is not visited at all — no sweep, no
    /// per-entity test. That is the whole point of the split.
    ///
    /// The list is published for the duration ([`worlds::world`]) so chrome
    /// running *inside* the sweep can reach a world it is not in.
    pub fn sweep_all(worlds: &[World], dt: f32) {
        let _live = worlds::publish(worlds);
        for w in worlds.iter().filter(|w| w.simulating) {
            w.sweep(dt);
        }
    }

    /// `&self`: the registry sweeps through it, components mutate through
    /// their own `Mutex<T>`, and the published list would alias a `&mut`.
    fn sweep(&self, dt: f32) {
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
