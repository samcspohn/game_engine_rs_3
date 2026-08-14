//! Entity-Component System (ECS) for the engine core.
//!
//! This module provides:
//! - [`Component`] — the trait every game-logic component must implement.
//! - [`ComponentStorage<T>`] — a dense, parallel-friendly per-type store.
//! - [`ComponentRegistry`] — the type-erased collection of all storages.
//! - [`Entity`] — a handle to a transform slot (its id equals the transform
//!   index in [`TransformHierarchy`]).
//! - [`World`] — a hierarchy + a registry over it, and the update sweep.
//!
//! Renderer-specific components (`RendererComponent`, etc.) and GPU resources
//! live in `engine-render` and depend on this crate via `engine-core`.

#![allow(dead_code)]

use std::{
    any::TypeId,
    collections::HashMap,
    mem::MaybeUninit,
    sync::atomic::{AtomicU32, AtomicUsize, Ordering},
};

use parking_lot::Mutex;

use crate::{
    reflect::Export,
    transform::{compute::PerfCounter, Transform, TransformHierarchy},
    util::{parallel, parallel::BitmapTaskLayout},
};

mod world;
pub use world::{EntityMut, EntityView, World};

// ---------------------------------------------------------------------------
// Component trait
// ---------------------------------------------------------------------------

/// Trait that every component type must implement.
///
/// All methods have empty default implementations so that components only need
/// to override what they care about.
///
/// [`Export`] is a supertrait so an inspector or a save walk can read any
/// component without asking whether it opted in; `impl Export for T {}` is
/// the "nothing to author" case.
pub trait Component: Export {
    /// Whether this component type wants its [`Component::update`] hook
    /// called every frame. Defaults to `true` — set to `false` for pure
    /// data components (saves the per-frame storage iteration).
    ///
    /// Read by [`World::add_component`] when it lazily creates the
    /// per-type [`ComponentStorage`].
    const HAS_UPDATE: bool = true;

    /// Called once after the component is attached to an entity.
    fn init(&mut self, _transform: &Transform) {}

    /// Called once just before the component is detached / the entity is
    /// destroyed.
    fn deinit(&mut self, _transform: &Transform) {}

    /// Called every frame (only if [`Component::HAS_UPDATE`] is `true`).
    ///
    /// `world` is this component's own and nothing wider; reach another by
    /// holding its [`WorldHandle`](crate::WorldHandle). Locking another
    /// component from here is fine — two locking *each other* is not.
    fn update(&mut self, _dt: f32, _transform: &Transform, _world: &World) {}
}

// ---------------------------------------------------------------------------
// ComponentStorage<T>
// ---------------------------------------------------------------------------

/// Dense, parallel-friendly storage for a single component type `T`.
pub struct ComponentStorage<T> {
    /// One slot per entity index. `active` is the source of truth for
    /// which slots are initialized.
    data: Vec<MaybeUninit<Mutex<T>>>,
    /// 1 bit per entity; word index `i` covers entities `[i*32, i*32+32)`.
    active: Vec<AtomicU32>,
    /// Highest-set entity-bit + 1, in entity units.
    extent: AtomicUsize,
    /// Lowest-ever-set entity-bit, in entity units. Monotonically
    /// decreasing (never bumped back up on `drop`), so it's a safe lower
    /// bound: no active bit can exist below it.
    start: AtomicUsize,
    has_update: bool,
}

impl<T> ComponentStorage<T>
where
    T: Component + Send + Sync,
{
    /// Construct an empty, fully-dynamic storage. `data` and `active`
    /// start empty and grow on the first `set` call for each new index.
    pub fn new(has_update: bool) -> Self {
        Self {
            data: Vec::new(),
            active: Vec::new(),
            extent: AtomicUsize::new(0),
            start: AtomicUsize::new(usize::MAX),
            has_update,
        }
    }

    /// Insert or overwrite the component at slot `t_idx` (the entity's
    /// transform index). Grows storage automatically. Returns `t_idx`.
    pub fn set(&mut self, t_idx: u32, item: T) -> u32 {
        let idx = t_idx as usize;
        let atomic_idx = idx >> 5;
        let bit_idx = idx & 31;

        // Grow the data buffer if this index is beyond current allocation.
        // Geometric doubling (min 64 slots) keeps amortised cost O(1).
        // SAFETY: MaybeUninit<T> has no validity invariant so uninitialised
        // bytes are fine; `active` is the source of truth for which slots
        // are live.
        if idx >= self.data.len() {
            let new_len = (idx + 1).max(self.data.len() * 2).max(64);
            let additional = new_len - self.data.len();
            self.data.reserve(additional);
            unsafe {
                self.data.set_len(new_len);
            }
        }

        // Grow the active bitmap if this word index is beyond current length.
        if atomic_idx >= self.active.len() {
            self.active
                .resize_with(atomic_idx + 1, || AtomicU32::new(0));
        }

        let active_word = &self.active[atomic_idx];
        let was_set = (active_word.load(Ordering::Relaxed) & (1u32 << bit_idx)) != 0;

        // Drop the old value in-place before overwriting, if any.
        unsafe {
            let slot = self.data.as_mut_ptr().add(idx);
            if was_set {
                (*slot).assume_init_drop();
            }
            slot.write(MaybeUninit::new(Mutex::new(item)));
        }

        // Publish liveness after the slot is fully constructed so a
        // concurrent reader observing the bit sees a valid Mutex.
        active_word.fetch_or(1u32 << bit_idx, Ordering::Release);

        // Bump extent monotonically.
        // let new_extent = idx + 1;
        // let mut cur = self.extent.load(Ordering::Relaxed);
        // while cur < new_extent {
        //     match self.extent.compare_exchange_weak(
        //         cur,
        //         new_extent,
        //         Ordering::Relaxed,
        //         Ordering::Relaxed,
        //     ) {
        //         Ok(_) => break,
        //         Err(actual) => cur = actual,
        //     }
        // }
        self.extent.fetch_max(idx + 1, Ordering::Relaxed);
        self.start.fetch_min(idx, Ordering::Relaxed);

        t_idx
    }

    #[inline]
    fn is_active(&self, idx: u32) -> bool {
        let atomic_idx = (idx >> 5) as usize;
        let bit_idx = idx & 31;
        if atomic_idx >= self.active.len() {
            return false;
        }
        let word = &self.active[atomic_idx];
        (word.load(Ordering::Acquire) & (1u32 << bit_idx)) != 0
    }

    /// Remove the component at `idx`, calling the storage-level drop (does
    /// **not** call [`Component::deinit`]; the caller is responsible for that).
    pub fn drop(&mut self, idx: u32) {
        if !self.is_active(idx) {
            return;
        }
        let atomic_idx = (idx >> 5) as usize;
        let bit_idx = idx & 31;
        let active_word = &self.active[atomic_idx];
        active_word.fetch_and(!(1u32 << bit_idx), Ordering::AcqRel);
        unsafe {
            let slot = self.data.as_mut_ptr().add(idx as usize);
            (*slot).assume_init_drop();
        }
    }

    /// Borrow the mutex for the component at `idx`, or `None` if absent.
    pub fn get(&self, idx: u32) -> Option<&Mutex<T>> {
        if !self.is_active(idx) {
            return None;
        }
        // SAFETY: active bit is set ⇒ slot has been initialized via
        // `set` and is not yet dropped. idx < data.len(), so the raw
        // pointer is valid.
        unsafe {
            let slot = self.data.as_ptr().add(idx as usize);
            Some((*slot).assume_init_ref())
        }
    }

    /// Entity index of the first active component in the storage, or `None`
    /// if the storage is empty. Linear scan over the active bitmap — fine
    /// for the rare "there is exactly one of these" queries (e.g. the
    /// renderer locating the scene's main camera); not meant for hot paths.
    pub fn first_index(&self) -> Option<u32> {
        let extent = self.extent.load(Ordering::Relaxed);
        let start_word = self.start.load(Ordering::Relaxed) >> 5;
        let extent_words = extent.div_ceil(32);
        for w in start_word..extent_words {
            let bits = self.active[w].load(Ordering::Acquire);
            if bits != 0 {
                let idx = (w << 5) + bits.trailing_zeros() as usize;
                if idx < extent {
                    return Some(idx as u32);
                }
            }
        }
        None
    }

    /// Iterate over all active components in parallel, calling `f` with a
    /// mutable reference to the component and the corresponding transform.
    fn par_iter<F>(
        &self,
        f: F,
        transform_hierarchy: &TransformHierarchy,
        bitmap_tasks: BitmapTaskLayout,
    ) where
        F: Fn(&mut T, &Transform) + Sync + Send + Copy,
    {
        let extent = self.extent.load(Ordering::Relaxed);
        if extent == 0 {
            return;
        }
        let start_word = self.start.load(Ordering::Relaxed) >> 5;
        let extent_words = extent.div_ceil(32);
        // Wrap raw pointers in a Sync newtype so the per-word closure
        // can satisfy `parallel_for`'s `Sync` bound. Workers touch
        // disjoint word ranges (and therefore disjoint Mutex slots) so
        // aliasing is sound.
        struct SyncPtr<T>(*const T);
        unsafe impl<T> Send for SyncPtr<T> {}
        unsafe impl<T> Sync for SyncPtr<T> {}
        let active_ptr = SyncPtr(self.active.as_ptr());
        let data_ptr = SyncPtr(self.data.as_ptr());

        // Per-word body: drains one bitmap word, dispatching `f` for
        // each set bit.
        let per_word = |atomic_idx: usize| {
            let _ = (&active_ptr, &data_ptr);
            // SAFETY: atomic_idx < extent_words ≤ active.len().
            let atomic = unsafe { &*active_ptr.0.add(atomic_idx) };
            let mut bits = atomic.load(Ordering::Acquire);
            if bits == 0 {
                return;
            }
            let base_idx = atomic_idx << 5;
            while bits != 0 {
                let bit_idx = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let current_idx = base_idx + bit_idx;
                if current_idx >= extent {
                    break;
                }
                // SAFETY: active bit was set ⇒ slot is initialized.
                let component = unsafe { (*data_ptr.0.add(current_idx)).assume_init_ref() };
                let transform = transform_hierarchy.get_transform_unchecked(current_idx as u32);
                let mut guard = component.lock();
                f(&mut *guard, &transform);
            }
        };

        let words_per_task = bitmap_tasks.words_per_task.max(1);
        let n_tasks = (extent_words - start_word).div_ceil(words_per_task);
        parallel::global::parallel_for(0..n_tasks, |task_range| {
            for task_idx in task_range {
                let word_start = start_word + task_idx * words_per_task;
                let word_end = (word_start + words_per_task).min(extent_words);
                for atomic_idx in word_start..word_end {
                    per_word(atomic_idx);
                }
            }
        });
        // parallel::global::parallel_for(0..extent_words, |atomic_idx| {
        //     per_word(atomic_idx.0);
        // });
    }

    /// Drive the `update` callback on every active component.  No-op if the
    /// storage was created with `has_update = false`.
    pub fn _update(&self, dt: f32, world: &World, bitmap_tasks: BitmapTaskLayout) {
        if self.has_update {
            self.par_iter(
                |c, t| c.update(dt, t, world),
                world.hierarchy(),
                bitmap_tasks,
            );
        }
    }
}

impl<T> Drop for ComponentStorage<T> {
    fn drop(&mut self) {
        // Walk the active bitmap and drop each live slot in place.
        // Vec<MaybeUninit<...>> does not run element destructors, so
        // this is the only place `Mutex<T>` destructors fire.
        let extent = *self.extent.get_mut();
        let start_word = *self.start.get_mut() >> 5;
        let extent_words = extent.div_ceil(32);
        for w in start_word..extent_words {
            // SAFETY: w < extent_words ≤ active.len().
            let word = &self.active[w];
            let mut bits = word.load(Ordering::Relaxed);
            let base = w << 5;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let idx = base + bit;
                // SAFETY: bit set ⇒ slot was initialized via `set`.
                unsafe {
                    let slot = self.data.as_mut_ptr().add(idx);
                    (*slot).assume_init_drop();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ComponentStorageTrait (type-erased)
// ---------------------------------------------------------------------------

impl<T: Component + Clone + Send + Sync + 'static> ComponentStorageTrait for ComponentStorage<T> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn remove(&mut self, idx: u32) {
        self.drop(idx);
    }
    fn deinit(&self, idx: u32, t: &Transform) {
        if let Some(m) = self.get(idx) {
            m.lock().deinit(t);
        }
    }
    fn empty_like(&self) -> Box<dyn ComponentStorageTrait + Send + Sync> {
        Box::new(ComponentStorage::<T>::new(self.has_update))
    }
    fn inspect(&self, idx: u32, f: &mut dyn FnMut(&mut dyn Export)) {
        if let Some(m) = self.get(idx) {
            f(&mut *m.lock());
        }
    }
    fn update(
        &self,
        dt: f32,
        world: &World,
        bitmap_tasks: BitmapTaskLayout,
        perf: &mut Option<HashMap<String, PerfCounter>>,
    ) {
        let name = std::any::type_name::<T>();
        if let Some(p) = perf.as_mut() {
            p.entry(name.into())
                .or_insert_with(PerfCounter::new)
                .start();
        }
        self._update(dt, world, bitmap_tasks);
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
    /// Call [`Component::deinit`] on the component at `idx`, if any.
    fn deinit(&self, idx: u32, t: &Transform);
    /// An empty storage of the same concrete type, for a destination registry
    /// that has never seen it. Only this storage knows its own `T`, which is
    /// what makes a type-erased move between registries possible at all.
    fn empty_like(&self) -> Box<dyn ComponentStorageTrait + Send + Sync>;
    /// Hand the component at `idx`, if any, to `f` as `&mut dyn Export`.
    /// A callback rather than a return, because the component is behind a
    /// `Mutex` this storage owns.
    fn inspect(&self, idx: u32, f: &mut dyn FnMut(&mut dyn Export));
    fn update(
        &self,
        dt: f32,
        world: &World,
        bitmap_tasks: BitmapTaskLayout,
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

/// Type-erased registry of all component storages in a [`World`].
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
    /// is **only** consulted on first registration.
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

    /// Hand every component attached to `idx` to `f`, in no particular
    /// order — what an inspector walks to build its rows.
    pub fn inspect(&self, idx: u32, mut f: impl FnMut(&mut dyn Export)) {
        for storage in self.components.values() {
            storage.inspect(idx, &mut f);
        }
    }

    /// Drive the `update` callback on every registered storage.
    pub fn update_all(
        &self,
        dt: f32,
        world: &World,
        bitmap_tasks: BitmapTaskLayout,
        perf: &mut Option<HashMap<String, PerfCounter>>,
    ) {
        for storage in self.components.values() {
            storage.update(dt, world, bitmap_tasks, perf);
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::{DEFAULT_WORLD, _Transform};
    use crate::worlds::{self, WorldHandle};
    use crate::util::thread_pool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering as O;

    /// Test component that records visits via a shared atomic.
    #[derive(Clone)]
    struct Probe {
        id: u32,
    }
    impl Export for Probe {}
    impl Component for Probe {}

    /// The components of a deleted subtree must go with it, or the next
    /// `par_iter` sweeps a component whose transform slot is dead.
    #[test]
    fn remove_entity_takes_the_subtree_s_components() {
        let mut w = World::new(DEFAULT_WORLD);
        let top = w.new_entity(_Transform {
            name: "top".into(),
            .._Transform::default()
        });
        let child = w.new_entity(_Transform {
            name: "child".into(),
            parent: Some(top.id),
            .._Transform::default()
        });
        let bystander = w.new_entity(_Transform::default());
        for e in [top, child, bystander] {
            w.add_component(e, Probe { id: e.id });
        }

        w.remove_entity(top);
        assert!(w.get_component::<Probe>(top).is_none());
        assert!(w.get_component::<Probe>(child).is_none(), "child's went too");
        assert!(w.get_component::<Probe>(bystander).is_some());
    }

    /// Counts `update` calls, and records each `deinit`.
    #[derive(Clone, Default)]
    struct Watcher {
        ticks: std::sync::Arc<AtomicUsize>,
        gone: std::sync::Arc<AtomicUsize>,
    }
    impl Export for Watcher {}
    impl Component for Watcher {
        fn update(&mut self, _dt: f32, _t: &Transform, _w: &World) {
            self.ticks.fetch_add(1, O::Relaxed);
        }
        fn deinit(&mut self, _t: &Transform) {
            self.gone.fetch_add(1, O::Relaxed);
        }
    }

    /// No frame runs in a test, so `&mut` is always sound here.
    fn edit(w: &WorldHandle) -> &mut World {
        unsafe { w.get_mut() }
    }

    /// One entity in each of two worlds, at the same index — which is the
    /// normal case, because each world's slots start at zero.
    fn two_worlds() -> (WorldHandle, WorldHandle, Entity) {
        let (doc, rig) = (worlds::new_world(), worlds::new_world());
        edit(&doc).set_simulating(false);
        let e = edit(&doc).new_entity(_Transform::default());
        assert_eq!(edit(&rig).new_entity(_Transform::default()), e);
        (doc, rig, e)
    }

    /// The point of worlds: a non-simulating one is not swept at all, so the
    /// cost of an edited scene is zero rather than a filtered walk.
    #[test]
    fn a_non_simulating_world_never_updates() {
        init_pool_once();
        let _g = test_lock();

        let (doc, rig, e) = two_worlds();
        let w: Vec<Watcher> = (0..2).map(|_| Watcher::default()).collect();
        edit(&doc).add_component(e, w[0].clone());
        edit(&rig).add_component(e, w[1].clone());

        worlds::sweep_all(&[doc, rig.clone()], 0.0);
        worlds::sweep_all(&[rig], 0.0);
        assert_eq!(w[0].ticks.load(O::Relaxed), 0, "the document stays still");
        assert_eq!(w[1].ticks.load(O::Relaxed), 2, "the rest does not");
    }

    /// ADR-0011 §2: an entity id is meaningless without its world, so the
    /// same index asked of the wrong one has to answer "no", not someone
    /// else's component.
    #[test]
    fn a_world_only_answers_for_its_own() {
        let (doc, rig, e) = two_worlds();
        edit(&doc).add_component(e, Watcher::default());

        let mut seen = 0;
        doc.entity(e).inspect(|_| seen += 1);
        assert_eq!(seen, 1);
        assert!(doc.entity(e).get_component(|_: &mut Watcher| {}));
        assert!(
            !rig.entity(e).get_component(|_: &mut Watcher| {}),
            "the same index, asked of the world it is not in"
        );
    }

    /// Editor chrome in one world inspecting a document in another, from
    /// inside the sweep that is running it — an ordinary capability now, and
    /// the world is held by handle rather than looked up (ADR-0011 §3).
    #[derive(Clone)]
    struct Reacher {
        document: WorldHandle,
        found: std::sync::Arc<AtomicUsize>,
    }
    impl Export for Reacher {}
    impl Component for Reacher {
        fn update(&mut self, _dt: f32, t: &Transform, _w: &World) {
            self.document
                .entity(Entity::new(t.get_idx()))
                .get_component(|_: &mut Watcher| {})
                .then(|| self.found.fetch_add(1, O::Relaxed));
        }
    }

    #[test]
    fn chrome_reaches_another_world_by_handle() {
        init_pool_once();
        let _g = test_lock();

        let (doc, rig, e) = two_worlds();
        edit(&doc).add_component(e, Watcher::default());
        let r = Reacher {
            document: doc.clone(),
            found: Default::default(),
        };
        edit(&rig).add_component(e, r.clone());

        worlds::sweep_all(&[doc, rig], 0.0);
        assert_eq!(r.found.load(O::Relaxed), 1);
    }

    /// The world lives exactly as long as a handle to it does — what ends an
    /// overlay world made by a component that has since been dropped.
    #[test]
    fn a_world_dies_with_its_last_handle() {
        let w = worlds::new_world();
        let id = w.id();
        assert!(worlds::world(id).is_some());
        drop(w);
        assert!(worlds::world(id).is_none());
    }

    /// The queued path a component uses: the entity does not exist until the
    /// frame boundary, and its builder runs there.
    #[test]
    fn a_queued_spawn_lands_at_the_boundary() {
        let w = worlds::new_world();
        let seen = std::sync::Arc::new(AtomicUsize::new(0));
        let probe = Watcher {
            gone: seen.clone(),
            ..Default::default()
        };
        w.spawn(_Transform::default(), move |mut e| {
            e.add_component(probe);
        });
        assert_eq!(w.hierarchy().len(), 1, "nothing yet — only the root");

        unsafe { worlds::apply_pending(&[w.clone()]) };
        assert_eq!(w.hierarchy().len(), 2);
        assert!(w.entity(Entity::new(1)).get_component(|_: &mut Watcher| {}));

        w.destroy(Entity::new(1));
        unsafe { worlds::apply_pending(&[w.clone()]) };
        assert_eq!(seen.load(O::Relaxed), 1, "and `deinit` ran on the way out");
    }

    /// ADR-0011 §6: crossing worlds is copy-and-delete, so the source keeps
    /// its subtree and the copy is a new identity in the destination.
    #[test]
    fn duplicate_copies_a_subtree_into_another_world() {
        let (src, dst, _) = two_worlds();
        let top = edit(&src).new_entity(_Transform::default());
        let child = edit(&src).new_entity(_Transform {
            parent: Some(top.id),
            .._Transform::default()
        });
        edit(&src).add_component(child, Watcher::default());

        let landed = std::sync::Arc::new(AtomicUsize::new(0));
        let l = landed.clone();
        dst.duplicate(src.entity(top), move |e| {
            l.store(e.id().id as usize, O::Relaxed);
        });
        unsafe { worlds::apply_pending(&[dst.clone()]) };

        let root = Entity::new(landed.load(O::Relaxed) as u32);
        let copied = dst.hierarchy().children(root.id).to_vec();
        assert_eq!(copied.len(), 1, "the child came too");
        assert!(dst
            .entity(Entity::new(copied[0]))
            .get_component(|_: &mut Watcher| {}));
        assert!(
            src.get_component::<Watcher>(child).is_some(),
            "and the source is untouched"
        );
    }

    /// Play mode's shape (ADR-0010 §4): the copy is a world of its own, so
    /// running it cannot touch the document it came from.
    #[test]
    fn duplicate_world_deep_copies_into_a_world_of_its_own() {
        init_pool_once();
        let _g = test_lock();

        let doc = worlds::new_world();
        edit(&doc).set_simulating(false);
        let top = edit(&doc).new_entity(_Transform::default());
        let child = edit(&doc).new_entity(_Transform {
            parent: Some(top.id),
            .._Transform::default()
        });
        let w = Watcher::default();
        edit(&doc).add_component(child, w.clone());

        let play = doc.duplicate_world(true);
        assert_eq!(play.hierarchy().len(), doc.hierarchy().len());
        assert_eq!(play.hierarchy().children(top.id).to_vec(), vec![child.id]);

        worlds::sweep_all(&[play], 0.0);
        assert!(w.ticks.load(O::Relaxed) > 0, "the clone shares the Arc");
        assert!(doc.get_component::<Watcher>(child).is_some(), "and the original stays");
    }

    /// A dropped component cannot announce its own disappearance, so
    /// `remove_entity` calls `deinit` while it is still there — this is what
    /// puts `NO_RENDERER` over a deleted entity's GPU slot.
    #[test]
    fn removal_deinits_the_subtree_before_dropping_it() {
        let mut world = World::new(DEFAULT_WORLD);
        let top = world.new_entity(_Transform::default());
        let child = world.new_entity(_Transform {
            parent: Some(top.id),
            .._Transform::default()
        });
        let w = Watcher::default();
        world.add_component(top, w.clone());
        world.add_component(child, w.clone());

        world.remove_entity(top);
        assert_eq!(w.gone.load(O::Relaxed), 2, "both, before the drop");
        assert!(world.get_component::<Watcher>(child).is_none());
    }

    fn init_pool_once() {
        drop(thread_pool::lock_for_test());
        // The dispatch wrapper has its own global; whichever test wins
        // the race performs the init.
        let _ = parallel::global::init(parallel::BackendKind::MyPool, 4);
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
            0usize, 1, 2, 31, 32, 33, 63, 64, 65, 255, 256,
            257, // 8-word task boundary (256 entities)
            511, 512, 513, // 16-word boundary
            1_000, 4_096, 10_000,
        ];

        for n in test_sizes {
            let mut hier = TransformHierarchy::new(0);
            for i in 0..n {
                let _t = hier.create_transform(_Transform {
                    position: glam::Vec3::ZERO,
                    rotation: glam::Quat::IDENTITY,
                    scale: glam::Vec3::ONE,
                    name: String::new(),
                    parent: None,
                });
                let _ = i;
            }

            let mut storage: ComponentStorage<Probe> = ComponentStorage::new(true);
            for i in 0..n as u32 {
                storage.set(i, Probe { id: i });
            }

            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            let bitmap_tasks = parallel::bitmap_task_layout(hier.len().div_ceil(32));
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

        let mut hier = TransformHierarchy::new(0);
        for _ in 0..n {
            hier.create_transform(_Transform {
                position: glam::Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
                name: String::new(),
                parent: None,
            });
        }

        let mut storage: ComponentStorage<Probe> = ComponentStorage::new(true);
        let mut expected_active: Vec<u32> = Vec::new();
        for i in (0..n).step_by(stride as usize) {
            storage.set(i, Probe { id: i });
            expected_active.push(i);
        }

        let hits: Vec<AtomicUsize> = (0..n as usize).map(|_| AtomicUsize::new(0)).collect();
        let bitmap_tasks = parallel::bitmap_task_layout(hier.len().div_ceil(32));
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
            assert_eq!(
                v, expected,
                "probe {i} visited {v} times (expected {expected})"
            );
        }
    }
}
