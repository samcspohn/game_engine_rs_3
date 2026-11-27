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
    transform::{Transform, TransformHierarchy, compute::PerfCounter},
    util::{parallel, parallel::BitmapTaskLayout},
};

mod refs;
mod world;
pub use refs::{ComponentRef, EntityRef};
pub use world::{EntityMut, EntityView, System, World};

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

    /// Whether [`Component::init`] runs when the world is not simulating.
    ///
    /// Construction is edit-time and `init` is play-time (ADR-0010 §5), but
    /// not for everything: a `MeshRenderer` has to publish its GPU record in
    /// a document or the viewport goes black, while a `CameraComponent`
    /// minting its camera in one would put a second view on screen. So the
    /// default is the permissive one and the rule is stated per type.
    const INIT_IN_EDIT: bool = true;

    /// Called once after the component is attached to an entity — unless the
    /// world is not simulating and [`Component::INIT_IN_EDIT`] is `false`.
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

        // `&mut self`: no reader can be looking, so plain writes publish.
        let was_set = (*self.active[atomic_idx].get_mut() & (1u32 << bit_idx)) != 0;

        // Drop the old value in-place before overwriting, if any.
        unsafe {
            let slot = self.data.as_mut_ptr().add(idx);
            if was_set {
                (*slot).assume_init_drop();
            }
            slot.write(MaybeUninit::new(Mutex::new(item)));
        }
        *self.active[atomic_idx].get_mut() |= 1u32 << bit_idx;
        let extent = self.extent.get_mut();
        *extent = (*extent).max(idx + 1);
        let start = self.start.get_mut();
        *start = (*start).min(idx);

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

    /// Every component with its entity index, in index order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &Mutex<T>)> {
        let first = self
            .start
            .load(Ordering::Relaxed)
            .min(self.active.len() << 5)
            >> 5;
        let words = self.extent.load(Ordering::Relaxed).div_ceil(32);
        (first..words).flat_map(move |w| {
            let mut bits = self.active[w].load(Ordering::Acquire);
            std::iter::from_fn(move || {
                let b = (bits != 0).then(|| bits.trailing_zeros())?;
                bits &= bits - 1;
                let idx = (w << 5) as u32 | b;
                // SAFETY: an active bit means `set` initialized the slot.
                Some((idx, unsafe {
                    (*self.data.as_ptr().add(idx as usize)).assume_init_ref()
                }))
            })
        })
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
    ) -> bool {
        let other = other.as_any().downcast_ref::<ComponentStorage<T>>();
        let Some(other_mutex) = other.and_then(|o| o.get(src_idx)) else {
            return false;
        };
        let mut new_component = other_mutex.lock().clone();
        new_component.init(t);
        self.set(dst_idx, new_component);
        true
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
    ) -> bool;
}

// ---------------------------------------------------------------------------
// ComponentRegistry
// ---------------------------------------------------------------------------

/// A `TypeId` is already a hash, so it is passed through, not hashed again.
#[derive(Default)]
struct TypeIdHasher(u64);

impl std::hash::Hasher for TypeIdHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ b as u64;
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

type ByType<V> = HashMap<TypeId, V, std::hash::BuildHasherDefault<TypeIdHasher>>;

/// Type-erased registry of all component storages in a [`World`].
pub struct ComponentRegistry {
    /// In the order their types were first added.
    storages: Vec<Box<dyn ComponentStorageTrait + Send + Sync>>,
    by_type: ByType<usize>,
    /// A bit per storage a slot has a component in, `words` a slot, so an
    /// entity's removal visits what it has and not every type there is.
    has: Vec<u64>,
    words: usize,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self {
            storages: Vec::new(),
            by_type: ByType::default(),
            has: Vec::new(),
            words: 1,
        }
    }

    /// `T`'s storage and its index, made on first use — the only time
    /// `has_update` is read.
    pub(crate) fn register<T: Component + Clone + Send + Sync + 'static>(
        &mut self,
        has_update: bool,
    ) -> (usize, &mut ComponentStorage<T>) {
        let ty = TypeId::of::<T>();
        let i = match self.by_type.get(&ty) {
            Some(&i) => i,
            None => self.push(Box::new(ComponentStorage::<T>::new(has_update)), ty),
        };
        let s = self.storages[i].as_any_mut().downcast_mut();
        (i, s.expect("a storage holds the type it is keyed by"))
    }

    fn push(&mut self, storage: Box<dyn ComponentStorageTrait + Send + Sync>, ty: TypeId) -> usize {
        let i = self.storages.len();
        self.storages.push(storage);
        self.by_type.insert(ty, i);
        if i == self.words * 64 {
            self.widen();
        }
        i
    }

    /// One more word a slot, for a 65th type or a 129th.
    fn widen(&mut self) {
        let w = self.words;
        let mut has = vec![0; self.has.len() / w * (w + 1)];
        for (to, from) in has.chunks_mut(w + 1).zip(self.has.chunks(w)) {
            to[..w].copy_from_slice(from);
        }
        (self.has, self.words) = (has, w + 1);
    }

    /// Slot `idx` has a component in storage `i` now, or no longer does.
    fn mark(&mut self, idx: u32, i: usize, on: bool) {
        let at = idx as usize * self.words + i / 64;
        if at >= self.has.len() {
            if !on {
                return;
            }
            self.has.resize((idx as usize + 1) * self.words, 0);
        }
        let bit = 1u64 << (i % 64);
        match on {
            true => self.has[at] |= bit,
            false => self.has[at] &= !bit,
        }
    }

    /// The storages slot `idx` has a component in.
    fn present(&self, idx: u32) -> impl Iterator<Item = usize> + '_ {
        let at = idx as usize * self.words;
        let words = self.has.get(at..at + self.words).unwrap_or(&[]);
        words.iter().enumerate().flat_map(|(w, &bits)| {
            let mut bits = bits;
            std::iter::from_fn(move || {
                let b = (bits != 0).then(|| bits.trailing_zeros() as usize)?;
                bits &= bits - 1;
                Some(w * 64 + b)
            })
        })
    }

    /// Borrow the typed storage for `T`, or `None` if it was never registered.
    pub fn get_storage<T: Component + Send + Sync + 'static>(
        &self,
    ) -> Option<&ComponentStorage<T>> {
        let &i = self.by_type.get(&TypeId::of::<T>())?;
        self.storages[i].as_any().downcast_ref()
    }

    /// Deinit and drop slot `idx`'s `T`, if it has one. An unregistered `T`
    /// is a no-op rather than a storage made to be empty.
    pub(crate) fn remove<T: Component + Send + Sync + 'static>(&mut self, idx: u32, t: &Transform) {
        let Some(&i) = self.by_type.get(&TypeId::of::<T>()) else {
            return;
        };
        let s: &mut ComponentStorage<T> = self.storages[i].as_any_mut().downcast_mut().unwrap();
        if let Some(m) = s.get(idx) {
            m.lock().deinit(t);
        }
        s.drop(idx);
        self.mark(idx, i, false);
    }

    /// Deinit every component on slot `idx`, then drop them all.
    pub(crate) fn remove_all(&mut self, idx: u32, t: &Transform) {
        for i in self.present(idx) {
            self.storages[i].deinit(idx, t);
        }
        let at = idx as usize * self.words;
        let Some(words) = self.has.get_mut(at..at + self.words) else {
            return;
        };
        for (w, bits) in words.iter_mut().enumerate() {
            while *bits != 0 {
                self.storages[w * 64 + bits.trailing_zeros() as usize].remove(idx);
                *bits &= *bits - 1;
            }
        }
    }

    /// Hand every component attached to `idx` to `f`, in the order their
    /// types registered — what an inspector walks to build its rows.
    pub fn inspect(&self, idx: u32, mut f: impl FnMut(&mut dyn Export)) {
        for i in self.present(idx) {
            self.storages[i].inspect(idx, &mut f);
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
        for storage in &self.storages {
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
    use crate::transform::{_Transform, DEFAULT_WORLD};
    use crate::util::thread_pool;
    use crate::worlds::{self, WorldHandle};
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
        assert!(
            w.get_component::<Probe>(child).is_none(),
            "child's went too"
        );
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
        let _g = test_lock();
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
    ///
    /// Serialized: a freed slot is handed to the next world made, so asking
    /// whether one is empty only means something while nothing else in the
    /// process is allocating.
    #[test]
    fn a_world_dies_with_its_last_handle() {
        let _g = test_lock();
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
        let _g = test_lock();
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
        let _g = test_lock();
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
        let copied: Vec<u32> = dst.hierarchy().children(root.id).collect();
        assert_eq!(copied.len(), 1, "the child came too");
        assert!(
            dst.entity(Entity::new(copied[0]))
                .get_component(|_: &mut Watcher| {})
        );
        assert!(
            src.get_component::<Watcher>(child).is_some(),
            "and the source is untouched"
        );
    }

    /// A component whose `init` is play-time only. The engine's
    /// `CameraComponent` is the one that needs this: minting its camera in a
    /// document would put a second view on screen (ADR-0010 §5).
    #[derive(Clone, Default)]
    struct PlayOnly {
        started: std::sync::Arc<AtomicUsize>,
    }
    impl Export for PlayOnly {}
    impl Component for PlayOnly {
        const HAS_UPDATE: bool = false;
        const INIT_IN_EDIT: bool = false;
        fn init(&mut self, _t: &Transform) {
            self.started.fetch_add(1, O::Relaxed);
        }
    }

    /// Edit-time construction, play-time `init` — the same component added to
    /// a document and to a running scene, and only one of them starts.
    #[test]
    fn a_play_time_init_does_not_run_in_a_document() {
        let _g = test_lock();
        let (doc, play) = (worlds::new_world(), worlds::new_world());
        edit(&doc).set_simulating(false);

        let c = PlayOnly::default();
        for w in [&doc, &play] {
            let e = edit(w).new_entity(_Transform::default());
            edit(w).add_component(e, c.clone());
        }
        assert_eq!(
            c.started.load(O::Relaxed),
            1,
            "attached to both, started only in the one that runs"
        );
    }

    /// Stop-play is dropping the handle. Nothing walks the copy taking it
    /// apart, so what has to be true is that the last handle is the world's
    /// life — and that a holder can tell it is the last one, which is how
    /// the renderer knows to retire the buffers it drew the copy through.
    /// Says when it is dropped, which is when the world holding it is.
    #[derive(Clone)]
    struct Freed(std::sync::Arc<std::sync::atomic::AtomicBool>);
    impl Export for Freed {}
    impl Component for Freed {}
    impl Drop for Freed {
        fn drop(&mut self) {
            self.0.store(true, O::Relaxed);
        }
    }

    #[test]
    fn stopping_play_is_the_last_handle_going_away() {
        init_pool_once();
        let _g = test_lock();

        let play = worlds::new_world();
        let freed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let e = edit(&play).new_entity(_Transform::default());
        edit(&play).add_component(e, Freed(freed.clone()));

        assert!(play.is_orphan(), "the play world is its maker's alone");
        let drawn = play.clone();
        assert!(!play.is_orphan(), "a second holder is what keeps it alive");
        drop(play);
        assert!(drawn.is_orphan(), "and dropping the maker's leaves one");
        assert!(!freed.load(O::Relaxed));
        drop(drawn);
        // Not by its id: a dead world's id is the next new world's, which a
        // test running alongside may already have made.
        assert!(freed.load(O::Relaxed), "the world went with it");
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

    /// A distinct type per `N`, for a registry with more than one word of them.
    #[derive(Clone, Default)]
    struct Many<const N: usize>;
    impl<const N: usize> Export for Many<N> {}
    impl<const N: usize> Component for Many<N> {
        const HAS_UPDATE: bool = false;
    }

    #[test]
    fn every_component_of_a_type_comes_with_its_entity() {
        let mut w = World::new(DEFAULT_WORLD);
        let top = w.new_entity(_Transform {
            name: "top".into(),
            .._Transform::default()
        });
        let es: Vec<Entity> = (0..40)
            .map(|_| {
                w.new_entity(_Transform {
                    name: "e".into(),
                    parent: Some(top.id),
                    .._Transform::default()
                })
            })
            .collect();
        for &e in es.iter().step_by(3) {
            w.add_component(e, Probe { id: e.id });
        }
        w.remove_entity(es[3]);
        let all: Vec<(u32, u32)> = (w.get_all_components::<Probe>())
            .map(|(e, p)| (e.id().id, p.lock().id))
            .collect();
        let want: Vec<(u32, u32)> = (es.iter().step_by(3))
            .filter(|&&e| e != es[3])
            .map(|e| (e.id, e.id))
            .collect();
        assert_eq!(all, want);
        let e9 = w.get("/root/top/e9").expect("named and suffixed").id();
        assert_eq!(e9, es[9]);
        assert_eq!(w.entity(top).get("e9").map(|e| e.id()), Some(e9));
    }

    /// Past 64 types a slot's mask is two words, and what an entity had
    /// before the widening is still what removing it takes.
    #[test]
    fn removal_takes_what_the_entity_has_past_64_types() {
        let mut world = World::new(DEFAULT_WORLD);
        let (gone, kept) = (
            world.new_entity(_Transform::default()),
            world.new_entity(_Transform::default()),
        );
        let w = Watcher::default();
        world.add_component(gone, w.clone());
        macro_rules! every {
            ($($n:literal)*) => {$(world.add_component(kept, Many::<$n>);)*};
        }
        every!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69);
        world.add_component(gone, Many::<69>);
        world.add_component(kept, w.clone());

        let mut count = 0;
        world.entity(kept).inspect(|_| count += 1);
        assert_eq!(count, 71);
        world.remove_entity(gone);
        assert_eq!(
            w.gone.load(O::Relaxed),
            1,
            "its watcher, not the kept one's"
        );
        assert!(world.get_component::<Many<69>>(gone).is_none());
        assert!(world.get_component::<Many<69>>(kept).is_some());
        assert!(world.get_component::<Watcher>(kept).is_some());
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
                    name: Default::default(),
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
                name: Default::default(),
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
