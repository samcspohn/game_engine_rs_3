//! Entity-Component System (ECS) for the engine core.
//!
//! This module provides:
//! - [`Component`] — the trait every game-logic component must implement.
//! - [`ComponentStorage<T>`] — a dense, parallel-friendly per-type store.
//! - [`ComponentRegistry`] — the type-erased collection of all storages.
//! - [`Entity`] — a handle to a transform slot (its id equals the transform
//!   index in [`TransformHierarchy`]).
//! - [`Scene`] — the root object that owns a hierarchy + registry and drives
//!   the update loop.
//!
//! Renderer-specific components (`RendererComponent`, etc.) and GPU resources
//! live in `engine-render` and depend on this crate via `engine-core`.

#![allow(dead_code)]

use std::{
    any::TypeId,
    collections::HashMap,
    sync::atomic::AtomicU32,
};

use parking_lot::Mutex;

use crate::{
    transform::{
        Transform, TransformHierarchy, _Transform,
        compute::PerfCounter,
    },
    util::seg_storage::{SegStorage, get_from_slice_unchecked},
    util::thread_pool,
};

// ---------------------------------------------------------------------------
// Component trait
// ---------------------------------------------------------------------------

/// Trait that every component type must implement.
///
/// All methods have empty default implementations so that components only need
/// to override what they care about.
pub trait Component {
    /// Whether this component type wants its [`Component::update`] hook
    /// called every frame. Defaults to `true` — set to `false` for pure
    /// data components (saves the per-frame storage iteration).
    ///
    /// Read by [`Scene::add_component`] when it lazily creates the
    /// per-type [`ComponentStorage`].
    const HAS_UPDATE: bool = true;

    /// Called once after the component is attached to an entity.
    fn init(&mut self, _transform: &Transform) {}

    /// Called once just before the component is detached / the entity is
    /// destroyed.
    fn deinit(&mut self, _transform: &Transform) {}

    /// Called every frame (only if [`Component::HAS_UPDATE`] is `true`).
    fn update(&mut self, _dt: f32, _transform: &Transform) {}
}

// ---------------------------------------------------------------------------
// ComponentStorage<T>
// ---------------------------------------------------------------------------

/// Dense, parallel-friendly storage for a single component type `T`.
///
/// Internally backed by [`SegStorage`] so that pointers into the storage stay
/// valid as the collection grows (useful for the parallel iterator which holds
/// raw references).
pub struct ComponentStorage<T> {
    data: SegStorage<Mutex<T>>,
    extent: usize,
    active: Vec<AtomicU32>,
    has_update: bool,
}

impl<T> ComponentStorage<T>
where
    T: Component + Send + Sync,
{
    pub fn new(has_update: bool) -> Self {
        Self {
            data: SegStorage::new(),
            extent: 0,
            active: Vec::new(),
            has_update,
        }
    }

    /// Insert or overwrite the component at slot `t_idx` (the entity's
    /// transform index).  Returns `t_idx` for convenience.
    pub fn set(&mut self, t_idx: u32, item: T) -> u32 {
        let idx = t_idx as usize;
        self.data.set(idx, Mutex::new(item));

        let required_active_len = (idx >> 5) + 1;
        if required_active_len > self.active.len() {
            self.active
                .resize_with(required_active_len, || AtomicU32::new(0));
        }
        if idx >= self.extent {
            self.extent = idx + 1;
        }

        let atomic_idx = idx >> 5;
        let bit_idx = idx & 31;
        self.active[atomic_idx]
            .fetch_or(1 << bit_idx, std::sync::atomic::Ordering::Relaxed);
        t_idx
    }

    #[inline]
    fn is_active(&self, idx: u32) -> bool {
        let atomic_idx = (idx >> 5) as usize;
        let bit_idx = idx & 31;
        (self.active[atomic_idx].load(std::sync::atomic::Ordering::Relaxed) & (1 << bit_idx)) != 0
    }

    /// Remove the component at `idx`, calling the storage-level drop (does
    /// **not** call [`Component::deinit`]; the caller is responsible for that).
    pub fn drop(&mut self, idx: u32) {
        if (idx as usize) < self.data.len() && self.is_active(idx) {
            let atomic_idx = (idx >> 5) as usize;
            let bit_idx = idx & 31;
            self.active[atomic_idx]
                .fetch_and(!(1 << bit_idx), std::sync::atomic::Ordering::Relaxed);
            self.data.drop(idx as usize);
        }
    }

    /// Borrow the mutex for the component at `idx`, or `None` if absent.
    pub fn get(&self, idx: u32) -> Option<&Mutex<T>> {
        if (idx as usize) < self.data.len() && self.is_active(idx) {
            Some(self.data.get_unchecked(idx as usize))
        } else {
            None
        }
    }

    /// Iterate over all active components in parallel, calling `f` with a
    /// mutable reference to the component and the corresponding transform.
    fn par_iter<F>(
        &self,
        f: F,
        transform_hierarchy: &TransformHierarchy,
        bitmap_tasks: thread_pool::BitmapTaskLayout,
    )
    where
        F: Fn(&mut T, &Transform) + Sync + Send + Copy,
    {
        let extent = self.extent;
        let active = &self.active;
        let n_words = active.len();
        if n_words == 0 {
            return;
        }
        let words_per_task = bitmap_tasks.words_per_task;
        let n_tasks = n_words.div_ceil(words_per_task);
        thread_pool::global().parallel_for_global(n_tasks, |task_idx| {
            let word_start = task_idx * words_per_task;
            let word_end = (word_start + words_per_task).min(n_words);
            for atomic_idx in word_start..word_end {
                let atomic = unsafe { active.get_unchecked(atomic_idx) };
                let mut bits = atomic.load(std::sync::atomic::Ordering::Relaxed);
                if bits == 0 {
                    continue;
                }
                let base_idx = atomic_idx << 5;
                let seg_chunk = self.data.get_segment_chunk_unchecked(base_idx);
                while bits != 0 {
                    let bit_idx = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let current_idx = base_idx + bit_idx;
                    if current_idx >= extent {
                        break;
                    }
                    let component = get_from_slice_unchecked(seg_chunk, bit_idx);
                    let transform = transform_hierarchy
                        .get_transform_unchecked(current_idx as u32);
                    let mut guard = component.lock();
                    f(&mut *guard, &transform);
                }
            }
        });
    }

    /// Drive the `update` callback on every active component.  No-op if the
    /// storage was created with `has_update = false`.
    pub fn _update(
        &self,
        dt: f32,
        transform_hierarchy: &TransformHierarchy,
        bitmap_tasks: thread_pool::BitmapTaskLayout,
    ) {
        if self.has_update {
            self.par_iter(|c, t| c.update(dt, t), transform_hierarchy, bitmap_tasks);
        }
    }
}

// ---------------------------------------------------------------------------
// ComponentStorageTrait (type-erased)
// ---------------------------------------------------------------------------

impl<T: Component + Clone + Send + Sync + 'static> ComponentStorageTrait
    for ComponentStorage<T>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn remove(&mut self, idx: u32) {
        self.drop(idx);
    }
    fn update(
        &self,
        dt: f32,
        transform_hierarchy: &TransformHierarchy,
        bitmap_tasks: thread_pool::BitmapTaskLayout,
        perf: &mut Option<HashMap<String, PerfCounter>>,
    ) {
        let name = std::any::type_name::<T>();
        if let Some(p) = perf.as_mut() {
            p.entry(name.into()).or_insert_with(PerfCounter::new).start();
        }
        self._update(dt, transform_hierarchy, bitmap_tasks);
        if let Some(p) = perf.as_mut() {
            p.get_mut(name).unwrap().stop();
        }
    }
    fn clone_from_other(
        &mut self,
        other: &dyn ComponentStorageTrait,
        src_idx: u32,
        dst_idx: u32,
        t: &Transform,
    ) {
        if let Some(other_storage) = other.as_any().downcast_ref::<ComponentStorage<T>>() {
            if let Some(other_mutex) = other_storage.get(src_idx) {
                let other_component = other_mutex.lock();
                let mut new_component = (*other_component).clone();
                new_component.init(t);
                self.set(dst_idx, new_component);
            }
        }
    }
}

/// Object-safe, type-erased interface over [`ComponentStorage<T>`].
trait ComponentStorageTrait {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    /// Drop the component at `idx` without calling `deinit`.
    fn remove(&mut self, idx: u32);
    fn update(
        &self,
        dt: f32,
        transform_hierarchy: &TransformHierarchy,
        bitmap_tasks: thread_pool::BitmapTaskLayout,
        perf: &mut Option<HashMap<String, PerfCounter>>,
    );
    /// Clone component `src_idx` from `other` into slot `dst_idx` of `self`,
    /// then call `init` on the clone.
    fn clone_from_other(
        &mut self,
        other: &dyn ComponentStorageTrait,
        src_idx: u32,
        dst_idx: u32,
        t: &Transform,
    );
}

// ---------------------------------------------------------------------------
// ComponentRegistry
// ---------------------------------------------------------------------------

/// Type-erased registry of all component storages in a [`Scene`].
pub struct ComponentRegistry {
    components: HashMap<TypeId, Box<dyn ComponentStorageTrait + Send + Sync>>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
        }
    }

    /// Ensure a storage exists for `T` and return a mutable handle to it.
    ///
    /// If the storage already exists this is a pure lookup — `has_update`
    /// is **only** consulted on first registration. Use this when you want
    /// register-or-get semantics in a single call (e.g. from
    /// [`Scene::add_component`]).
    pub fn register<T: Component + Clone + Send + Sync + 'static>(
        &mut self,
        has_update: bool,
    ) -> &mut ComponentStorage<T> {
        self.components
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(ComponentStorage::<T>::new(has_update)))
            .as_any_mut()
            .downcast_mut::<ComponentStorage<T>>()
            .expect("TypeId collision: storage exists but for a different T")
    }

    /// Borrow the typed storage for `T`, or `None` if it was never registered.
    pub fn get_storage<T: Component + Send + Sync + 'static>(
        &self,
    ) -> Option<&ComponentStorage<T>> {
        let type_id = TypeId::of::<T>();
        self.components
            .get(&type_id)
            .and_then(|s| s.as_any().downcast_ref::<ComponentStorage<T>>())
    }

    /// Borrow the typed storage for `T` mutably, or `None` if `T` was never
    /// registered. **Does not** create a storage on miss — call
    /// [`register`](Self::register) first if you want register-or-get
    /// semantics. Keeping this strict means `remove_component` for an
    /// unregistered type is a no-op rather than a silent allocation.
    pub fn get_storage_mut<T: Component + Send + Sync + 'static>(
        &mut self,
    ) -> Option<&mut ComponentStorage<T>> {
        self.components
            .get_mut(&TypeId::of::<T>())
            .and_then(|s| s.as_any_mut().downcast_mut::<ComponentStorage<T>>())
    }

    /// Drive the `update` callback on every registered storage.
    pub fn update_all(
        &self,
        dt: f32,
        transform_hierarchy: &TransformHierarchy,
        bitmap_tasks: thread_pool::BitmapTaskLayout,
        perf: &mut Option<HashMap<String, PerfCounter>>,
    ) {
        for storage in self.components.values() {
            storage.update(dt, transform_hierarchy, bitmap_tasks, perf);
        }
    }
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Entity
// ---------------------------------------------------------------------------

/// A handle to a living entity.
///
/// The `id` is the index of the entity's [`Transform`] in the scene's
/// [`TransformHierarchy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    pub id: u32,
}

impl Entity {
    pub fn new(id: u32) -> Self {
        Entity { id }
    }
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// The top-level game-world object.
///
/// A `Scene` owns a [`TransformHierarchy`] (all entity transforms) and a
/// [`ComponentRegistry`] (all typed component storages).  Renderer resources
/// are **not** stored here; they live in `engine-render`.
pub struct Scene {
    pub components: ComponentRegistry,
    pub transform_hierarchy: TransformHierarchy,
    /// Optional per-type timing data.  Set to `Some(HashMap::new())` to
    /// enable component-update profiling.
    pub perf: Option<HashMap<String, PerfCounter>>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            components: ComponentRegistry::new(),
            transform_hierarchy: TransformHierarchy::new(),
            perf: None,
        }
    }

    /// Advance all components by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        let bitmap_tasks =
            thread_pool::bitmap_task_layout(self.transform_hierarchy.len().div_ceil(32));
        self.components
            .update_all(dt, &self.transform_hierarchy, bitmap_tasks, &mut self.perf);
    }

    /// Spawn a new entity from a transform descriptor.  Returns a handle.
    pub fn new_entity(&mut self, t: _Transform) -> Entity {
        Entity::new(self.transform_hierarchy.create_transform(t).get_idx())
    }

    /// Attach component `T` to `entity`, calling [`Component::init`].
    ///
    /// On first use for type `T`, the per-type storage is registered with
    /// `T::HAS_UPDATE`; subsequent calls reuse the same storage. The
    /// `register → set` chain is a single hash lookup with no `Option`
    /// dance and no silent fallback path.
    pub fn add_component<T>(&mut self, entity: Entity, mut component: T)
    where
        T: Component + Clone + Send + Sync + 'static,
    {
        let t = self.transform_hierarchy.get_transform_unchecked(entity.id);
        component.init(&t);
        self.components
            .register::<T>(T::HAS_UPDATE)
            .set(entity.id, component);
    }

    /// Remove component `T` from `entity`, calling [`Component::deinit`]
    /// first.
    pub fn remove_component<T>(&mut self, entity: Entity)
    where
        T: Component + Clone + Send + Sync + 'static,
    {
        // Call deinit before dropping.
        if let Some(storage) = self.components.get_storage_mut::<T>() {
            if let Some(mutex) = storage.get(entity.id) {
                let t = self.transform_hierarchy.get_transform_unchecked(entity.id);
                mutex.lock().deinit(&t);
            }
            storage.drop(entity.id);
        }
    }

    /// Remove an entity and all of its components from the scene.
    ///
    /// `deinit` is **not** called on individual components by this path — use
    /// [`remove_component`](Self::remove_component) for each type first if
    /// you need orderly teardown.
    pub fn remove_entity(&mut self, entity: Entity) {
        for storage in self.components.components.values_mut() {
            storage.remove(entity.id);
        }
        let t = self
            .transform_hierarchy
            .get_transform_unchecked(entity.id)
            .lock();
        self.transform_hierarchy.remove_transform(t);
    }

    /// Borrow the `Mutex<T>` for `entity`'s component `T`, or `None`.
    pub fn get_component<T>(&self, entity: Entity) -> Option<&Mutex<T>>
    where
        T: Component + Send + Sync + 'static,
    {
        self.components.get_storage::<T>()?.get(entity.id)
    }

    /// Deep-clone `other` into `self`.
    ///
    /// All transforms are duplicated (respecting the parent hierarchy), then
    /// all component storages are cloned slot-by-slot.  Returns a
    /// [`Transform`] handle to the root entity (index 0 in `other`).
    pub fn instantiate(&mut self, other: &Scene) -> Transform<'_> {
        let mut entity_map: HashMap<u32, u32> = HashMap::new();

        // --- duplicate transforms preserving parent links -----------------
        for t_idx in 0..other.transform_hierarchy.len() as u32 {
            let src = other.transform_hierarchy.get_transform_(t_idx);
            let new_t = _Transform {
                position: src.position,
                rotation: src.rotation,
                scale: src.scale,
                name: src.name.clone(),
                parent: src.parent.map(|p| {
                    *entity_map.get(&p).unwrap_or_else(|| {
                        panic!("instantiate: parent transform {} not yet mapped", p)
                    })
                }),
            };
            let new_entity = self.new_entity(new_t);
            entity_map.insert(t_idx, new_entity.id);
        }

        // --- clone component storages -------------------------------------
        for (type_id, other_storage) in &other.components.components {
            if let Some(self_storage) = self.components.components.get_mut(type_id) {
                for t_idx in 0..other.transform_hierarchy.len() as u32 {
                    let dst_idx = *entity_map.get(&t_idx).unwrap();
                    let t = self
                        .transform_hierarchy
                        .get_transform_unchecked(dst_idx);
                    self_storage.clone_from_other(
                        other_storage.as_ref(),
                        t_idx,
                        dst_idx,
                        &t,
                    );
                }
            }
        }

        self.transform_hierarchy
            .get_transform_unchecked(entity_map[&0])
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::_Transform;
    use crate::util::thread_pool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering as O;

    /// Test component that records visits via a shared atomic.
    struct Probe {
        id: u32,
    }
    impl Component for Probe {}

    fn init_pool_once() {
        drop(thread_pool::lock_for_test());
    }

    fn test_lock() -> std::sync::MutexGuard<'static, ()> {
        thread_pool::lock_for_test()
    }

    /// Build a hierarchy with `n` transforms and a storage with `n` Probes
    /// (one per entity), then drive `par_iter` and assert every probe is
    /// visited exactly once with the correct `id`. Covers boundary
    /// values around the shared bitmap chunking policy and the
    /// participant count of the pool.
    #[test]
    fn par_iter_visits_every_active_component_exactly_once() {
        init_pool_once();
        let _g = test_lock();

        // Mix of edge cases: empty, sub-word, exactly one word, several
        // words, ragged across a task boundary, and a large run.
        let test_sizes = [
            0usize, 1, 2, 31, 32, 33, 63, 64, 65,
            255, 256, 257,           // 8-word task boundary (256 entities)
            511, 512, 513,           // 16-word boundary
            1_000, 4_096, 10_000,
        ];

        for n in test_sizes {
            let mut hier = TransformHierarchy::new();
            for i in 0..n {
                let _t = hier.create_transform(_Transform {
                    position: glam::Vec3::ZERO,
                    rotation: glam::Quat::IDENTITY,
                    scale:    glam::Vec3::ONE,
                    name:     String::new(),
                    parent:   None,
                });
                let _ = i;
            }

            let mut storage: ComponentStorage<Probe> = ComponentStorage::new(true);
            for i in 0..n as u32 {
                storage.set(i, Probe { id: i });
            }

            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            let bitmap_tasks = thread_pool::bitmap_task_layout(hier.len().div_ceil(32));
            storage.par_iter(
                |probe: &mut Probe, _t: &Transform| {
                    // Indexed access panics on OOB, which catches any
                    // index-arithmetic bug in the par_iter chunking.
                    hits[probe.id as usize].fetch_add(1, O::Relaxed);
                },
                &hier,
                bitmap_tasks,
            );

            for (i, c) in hits.iter().enumerate() {
                let v = c.load(O::Relaxed);
                assert_eq!(v, 1, "n={n}: probe {i} visited {v} times");
            }
        }
    }

    /// Sparse activation: only every k-th entity has a component. The
    /// par_iter walk iterates every word in the active bitset but only
    /// dispatches for set bits — verify both the "only set bits run" and
    /// "every set bit runs" properties.
    #[test]
    fn par_iter_skips_inactive_and_hits_every_active() {
        init_pool_once();
        let _g = test_lock();

        let n: u32 = 5_000;
        let stride: u32 = 7; // co-prime with 32 to cross word boundaries irregularly

        let mut hier = TransformHierarchy::new();
        for _ in 0..n {
            hier.create_transform(_Transform {
                position: glam::Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                scale:    glam::Vec3::ONE,
                name:     String::new(),
                parent:   None,
            });
        }

        let mut storage: ComponentStorage<Probe> = ComponentStorage::new(true);
        let mut expected_active: Vec<u32> = Vec::new();
        for i in (0..n).step_by(stride as usize) {
            storage.set(i, Probe { id: i });
            expected_active.push(i);
        }

        let hits: Vec<AtomicUsize> = (0..n as usize).map(|_| AtomicUsize::new(0)).collect();
        let bitmap_tasks = thread_pool::bitmap_task_layout(hier.len().div_ceil(32));
        storage.par_iter(
            |probe: &mut Probe, _t: &Transform| {
                hits[probe.id as usize].fetch_add(1, O::Relaxed);
            },
            &hier,
            bitmap_tasks,
        );

        for i in 0..n {
            let v = hits[i as usize].load(O::Relaxed);
            let expected = if i % stride == 0 { 1 } else { 0 };
            assert_eq!(v, expected, "probe {i} visited {v} times (expected {expected})");
        }
    }
}
