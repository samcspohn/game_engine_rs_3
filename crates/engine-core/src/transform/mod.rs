#![allow(dead_code)]
use std::{
    cell::SyncUnsafeCell,
    ops::BitOr,
    sync::atomic::{AtomicU32, AtomicUsize, Ordering},
};

use glam::{Quat, Vec3};
use parking_lot::{Mutex, RwLock};

use crate::util::Avail;

pub mod compute;
mod name;

pub(crate) use name::Names;
pub use name::{Lookup, Name, Path, Symbol};

/// The hierarchy's root, created with it and present for the whole session.
///
/// Every transform descends from it: `_Transform::parent == None` means this,
/// never "detached". There is no detached state and no sentinel — **"never
/// written" and "parented to the root" are the same value**, which is why the
/// renderer can zero-fill its per-slot parent buffer and be correct, and why
/// the GPU walk terminates on `parent == 0` with no magic constant to keep in
/// agreement across three files.
///
/// The root is a structural anchor, not a transform: it is skipped by the
/// composition walks, so its own TRS is never applied. Parent a container
/// under it if you want a scene-wide transform.
pub const ROOT: u32 = 0;

// ─────────────────────────────────────────────────────────────────────────────
// Parent-update stream
// ─────────────────────────────────────────────────────────────────────────────

/// Number of per-thread accumulation shards. Power of two so the thread →
/// shard mapping is a mask. 64 shards × one cache line each = 4 KB per
/// hierarchy — enough that concurrent writers virtually never share a shard.
const PARENT_STREAM_SHARDS: usize = 64;

/// One accumulation shard, padded to a cache line so two threads recording
/// into adjacent shards never false-share.
#[repr(align(64))]
struct ParentShard(Mutex<Vec<u32>>);

/// Per-thread accumulation of re-parented transform indices, drained once
/// per frame by the renderer into `[trs_index, new_parent]` pairs for the
/// GPU parent-scatter pass.
///
/// Unlike TRS (sparse bitmask + full staging mirror), parent changes are
/// **streamed**: they are rare, so an O(changes) record list beats another
/// capacity-sized staging buffer + capacity-wide scatter dispatch (see
/// todo.txt — "a streamed buffer for parent updates since that happens
/// much less frequently").
///
/// Each thread appends to its own shard (uncontended `parking_lot` lock),
/// so a parallel `Component::update` burst of `set_parent` calls never
/// serialises on a global queue.
///
/// **Why shards store indices, not `[index, parent]` pairs:** two
/// re-parents of the *same* transform from different threads land in
/// different shards, and drain-time concatenation would replay them in
/// arbitrary order — a stale parent could win. Recording only "this index
/// changed" and snapshotting the *current* parent at drain time makes
/// duplicate records idempotent: every record of index `i` yields the same
/// `[i, current_parent(i)]` pair, so replay order (and the GPU scatter's
/// undefined write order between duplicate records) cannot matter.
struct ParentStream {
    shards: Vec<ParentShard>,
}

impl ParentStream {
    fn new() -> Self {
        Self {
            shards: (0..PARENT_STREAM_SHARDS)
                .map(|_| ParentShard(Mutex::new(Vec::new())))
                .collect(),
        }
    }

    /// Record that `idx`'s parent changed. Called under the transform's
    /// mutex by `create_transform` / `set_parent` / `remove_transform`.
    #[inline]
    fn record(&self, idx: u32) {
        let shard = thread_shard_slot() & (PARENT_STREAM_SHARDS - 1);
        self.shards[shard].0.lock().push(idx);
    }

    /// Take every recorded index, leaving all shards empty.
    fn drain_indices(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for shard in &self.shards {
            let mut v = shard.0.lock();
            if !v.is_empty() {
                out.append(&mut v);
            }
        }
        out
    }
}

/// Stable per-thread slot for shard selection: a monotonically-assigned
/// small integer per OS thread (first `record` on a thread claims the next
/// one). Distributes pool workers across shards evenly and permanently.
#[inline]
fn thread_shard_slot() -> usize {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    thread_local! {
        static SLOT: usize = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    SLOT.with(|s| *s)
}

/// No slot: the end of a sibling or child list.
const NONE: u32 = u32::MAX;

/// Children are a list linked through their siblings, so one leaves in O(1)
/// and the order the hierarchy panel shows holds. Written under the parent's
/// lock.
/// Note: 32-byte aligned, as 28 bytes straddled cache lines: 5 ns a spawn.
#[repr(align(32))]
struct TransformMeta {
    parent: u32,
    first: u32,
    last: u32,
    prev: u32,
    next: u32,
    name: Name,
}

/// A slot's children in order. Same aliasing contract as
/// [`TransformHierarchy::children`].
#[derive(Clone)]
pub struct Children<'a> {
    h: &'a TransformHierarchy,
    front: u32,
    back: u32,
}

impl Iterator for Children<'_> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        let at = self.front;
        if at == NONE {
            return None;
        }
        (self.front, self.back) = match at == self.back {
            true => (NONE, NONE),
            false => (self.h._meta(at).next, self.back),
        };
        Some(at)
    }
}

impl DoubleEndedIterator for Children<'_> {
    fn next_back(&mut self) -> Option<u32> {
        let at = self.back;
        if at == NONE {
            return None;
        }
        (self.front, self.back) = match at == self.front {
            true => (NONE, NONE),
            false => (self.front, self.h._meta(at).prev),
        };
        Some(at)
    }
}

pub struct Transform<'a> {
    hierarchy: &'a TransformHierarchy,
    idx: u32,
}

/// A transform on no entity and in no world: what a setter sees on a value
/// that is not a component's, such as a resource's.
pub fn detached() -> Transform<'static> {
    static NOWHERE: std::sync::OnceLock<TransformHierarchy> = std::sync::OnceLock::new();
    Transform::new(
        NOWHERE.get_or_init(|| TransformHierarchy::new(WorldId::MAX)),
        ROOT,
    )
}

impl<'a> Transform<'a> {
    fn new(hierarchy: &'a TransformHierarchy, idx: u32) -> Self {
        Self { hierarchy, idx }
    }
    pub fn lock(&self) -> TransformGuard<'a> {
        let lock = self.hierarchy.mutexes[self.idx as usize].lock();
        TransformGuard {
            hierarchy: self.hierarchy,
            idx: self.idx as usize,
            _lock: lock,
        }
    }
    pub fn get_idx(&self) -> u32 {
        self.idx
    }
    pub fn generation(&self) -> u32 {
        self.hierarchy
            .generation(self.idx)
            .expect("a transform names a slot")
    }
    /// The world this transform is in — its hierarchy's, since a hierarchy
    /// belongs to exactly one.
    pub fn world(&self) -> WorldId {
        self.hierarchy.world()
    }
    /// The hierarchy this transform belongs to — the access path a component
    /// has to the scene graph, since `Component::update` is handed a
    /// `Transform` and nothing else.
    pub fn hierarchy(&self) -> &'a TransformHierarchy {
        self.hierarchy
    }
}

pub struct TransformGuard<'a> {
    hierarchy: &'a TransformHierarchy,
    idx: usize,
    _lock: parking_lot::MutexGuard<'a, ()>,
}

impl<'a> TransformGuard<'a> {
    pub fn scale_by(&self, scale: Vec3) {
        self.hierarchy.scale_by(&self, scale);
    }
    pub fn set_scale(&self, scale: Vec3) {
        self.hierarchy.set_scale(&self, scale);
    }
    pub fn translate_by(&self, translation: Vec3) {
        self.hierarchy.translate_by(&self, translation);
    }
    pub fn set_position(&self, position: Vec3) {
        self.hierarchy.set_position(&self, position);
    }
    pub fn rotate_by(&self, rotation: Quat) {
        self.hierarchy.rotate_by(&self, rotation);
    }
    pub fn set_rotation(&self, rotation: Quat) {
        self.hierarchy.set_rotation(&self, rotation);
    }
    pub fn get_position(&self) -> Vec3 {
        self.hierarchy.get_position(&self)
    }
    pub fn get_rotation(&self) -> Quat {
        self.hierarchy.get_rotation(&self)
    }
    pub fn get_scale(&self) -> Vec3 {
        self.hierarchy.get_scale(&self)
    }
    pub fn get_parent(&self) -> Option<u32> {
        self.hierarchy.get_parent(&self)
    }
    /// Read-only by design: mutating the child list from outside would
    /// bypass `parent`, `has_children` and the parent stream at once.
    pub fn get_children(&self) -> Children<'a> {
        self.hierarchy.children(self.idx as u32)
    }
    pub fn get_name(&self) -> Name {
        self.hierarchy.get_meta(&self).name
    }
    // pub fn get_meta(&self) -> &mut TransformMeta {
    //     self.hierarchy.get_meta(&self)
    // }
    pub fn get_idx(&self) -> u32 {
        self.idx as u32
    }
    pub fn shift(&self, delta: Vec3) {
        self.hierarchy.shift(&self, delta);
    }
    pub fn get_global_position(&self) -> Vec3 {
        self.hierarchy.get_global_position(&self)
    }
    pub fn get_global_rotation(&self) -> Quat {
        self.hierarchy.get_global_rotation(&self)
    }
    pub fn get_global_scale(&self) -> Vec3 {
        self.hierarchy.get_global_scale(&self)
    }
}
#[repr(u8)]
enum TransformComponent {
    Position = 1 << 0,
    Rotation = 1 << 1,
    Scale = 1 << 2,
    Parent = 1 << 3,
}

// New: Flags type for combining components
#[derive(Copy, Clone)]
struct TransformComponentFlags(u8);

impl TransformComponentFlags {
    const NONE: Self = Self(0);
    const ALL: Self = Self(0b1111);
}

impl From<TransformComponent> for TransformComponentFlags {
    fn from(component: TransformComponent) -> Self {
        Self(component as u8)
    }
}

impl BitOr<TransformComponent> for TransformComponent {
    type Output = TransformComponentFlags;

    fn bitor(self, rhs: TransformComponent) -> Self::Output {
        TransformComponentFlags(self as u8 | rhs as u8)
    }
}

impl BitOr<TransformComponent> for TransformComponentFlags {
    type Output = TransformComponentFlags;

    fn bitor(self, rhs: TransformComponent) -> Self::Output {
        TransformComponentFlags(self.0 | rhs as u8)
    }
}

impl BitOr for TransformComponentFlags {
    type Output = TransformComponentFlags;

    fn bitor(self, rhs: Self) -> Self::Output {
        TransformComponentFlags(self.0 | rhs.0)
    }
}

pub struct _Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub name: Name,
    pub parent: Option<u32>,
}

impl _Transform {
    pub fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            name: Name::NONE,
            parent: None,
        }
    }
}

pub struct Dirty {
    position: Vec<AtomicU32>,
    rotation: Vec<AtomicU32>,
    scale: Vec<AtomicU32>,
    parent: Vec<AtomicU32>,
}

impl Dirty {
    pub fn new() -> Self {
        Self {
            position: Vec::new(),
            rotation: Vec::new(),
            scale: Vec::new(),
            parent: Vec::new(),
        }
    }
    #[inline]
    pub fn pos(&self, idx: u32) {
        let mask = 1 << (idx & 31);
        self.position[idx as usize >> 5].fetch_or(mask, Ordering::Relaxed);
    }
    #[inline]
    pub fn rot(&self, idx: u32) {
        let mask = 1 << (idx & 31);
        self.rotation[idx as usize >> 5].fetch_or(mask, Ordering::Relaxed);
    }
    #[inline]
    pub fn pos_rot(&self, idx: u32) {
        let mask = 1 << (idx & 31);
        let i = idx as usize >> 5;
        self.position[i].fetch_or(mask, Ordering::Relaxed);
        self.rotation[i].fetch_or(mask, Ordering::Relaxed);
    }
    #[inline]
    pub fn scale(&self, idx: u32) {
        let mask = 1 << (idx & 31);
        self.scale[idx as usize >> 5].fetch_or(mask, Ordering::Relaxed);
    }
    #[inline]
    pub fn parent(&self, idx: u32) {
        let mask = 1 << (idx & 31);
        self.parent[idx as usize >> 5].fetch_or(mask, Ordering::Relaxed);
    }
    #[inline]
    pub fn all(&self, idx: u32) {
        let mask = 1 << (idx & 31);
        self.position[idx as usize >> 5].fetch_or(mask, Ordering::Relaxed);
        self.rotation[idx as usize >> 5].fetch_or(mask, Ordering::Relaxed);
        self.scale[idx as usize >> 5].fetch_or(mask, Ordering::Relaxed);
        self.parent[idx as usize >> 5].fetch_or(mask, Ordering::Relaxed);
    }
    pub fn push(&mut self) {
        self.position.push(AtomicU32::new(0));
        self.rotation.push(AtomicU32::new(0));
        self.scale.push(AtomicU32::new(0));
        self.parent.push(AtomicU32::new(0));
    }
    pub fn len(&self) -> usize {
        self.position.len()
    }

    // ── Drain helpers for the renderer ────────────────────────────────────
    //
    // The renderer wants to read the current dirty state and then atomically
    // clear it so newly-dirtied entries set after the read are kept for the
    // next frame. Returning `&[AtomicU32]` lets the caller `swap(0, …)` each
    // word inline without having to clone the bitmask first.

    /// Position-dirty bitmask, one bit per entity slot. Bit `i & 31` of word
    /// `i >> 5` is set iff `set_position` / `translate_*` was called on
    /// entity `i` since the last drain.
    #[inline]
    pub fn position_words(&self) -> &[AtomicU32] {
        self.position.as_slice()
    }
    /// Rotation-dirty bitmask. See [`position_words`](Self::position_words).
    #[inline]
    pub fn rotation_words(&self) -> &[AtomicU32] {
        self.rotation.as_slice()
    }
    /// Scale-dirty bitmask. See [`position_words`](Self::position_words).
    #[inline]
    pub fn scale_words(&self) -> &[AtomicU32] {
        self.scale.as_slice()
    }
    /// Parent-dirty bitmask (re-parenting / removal). Not yet consumed by
    /// the renderer.
    #[inline]
    pub fn parent_words(&self) -> &[AtomicU32] {
        self.parent.as_slice()
    }

    /// Mark every TRS slot dirty (position + rotation + scale).
    ///
    /// Used by the renderer when the SoT is freshly (re-)allocated — e.g.
    /// after a world-capacity grow — so the next frame's harvest re-uploads
    /// every existing entity into the new SoT, regardless of whether the
    /// game just happened to call `set_position` / `rotate_by` recently.
    /// Per-bit `Relaxed` writes match the rest of `Dirty`'s ordering: the
    /// only synchronizing edge is the renderer's per-image fence.
    pub fn mark_all_trs(&self) {
        let pos = self.position.as_slice();
        let rot = self.rotation.as_slice();
        let scl = self.scale.as_slice();
        for ((p, r), s) in pos.iter().zip(rot.iter()).zip(scl.iter()) {
            p.store(u32::MAX, Ordering::Relaxed);
            r.store(u32::MAX, Ordering::Relaxed);
            s.store(u32::MAX, Ordering::Relaxed);
        }
    }
}

/// A [`World`](crate::World)'s position in the frame's world list. One per
/// hierarchy, not per slot: a world owns its graph outright (ADR-0011 §1).
pub type WorldId = u16;

/// The world a game that never makes a second one is in.
pub const DEFAULT_WORLD: WorldId = 0;

pub struct TransformHierarchy {
    mutexes: Vec<Mutex<()>>,
    positions: Vec<SyncUnsafeCell<Vec3>>,
    rotations: Vec<SyncUnsafeCell<Quat>>,
    scales: Vec<SyncUnsafeCell<Vec3>>,
    metadata: Vec<SyncUnsafeCell<TransformMeta>>,
    // dirty: Vec<AtomicU8>,
    dirty: Dirty,
    dirty_l2: Vec<AtomicU32>, // one bit for every 32 transforms 1024 total per u32
    has_children: Vec<AtomicU32>,
    active: Vec<AtomicU32>,
    /// Bumped when a slot is removed, so a ref to its old occupant goes stale.
    generations: Vec<AtomicU32>,
    /// Which world owns this hierarchy — its index in the frame's list.
    world: WorldId,
    avail: Avail,
    /// Per-thread accumulation of re-parented indices — see [`ParentStream`].
    parent_stream: ParentStream,
    names: RwLock<name::Names>,
}

impl TransformHierarchy {
    pub fn new(world: WorldId) -> Self {
        let mut h = Self {
            mutexes: Vec::new(),
            positions: Vec::new(),
            rotations: Vec::new(),
            scales: Vec::new(),
            metadata: Vec::new(),
            dirty: Dirty::new(),
            dirty_l2: Vec::new(),
            has_children: Vec::new(),
            active: Vec::new(),
            generations: Vec::new(),
            world,
            avail: Avail::new(),
            parent_stream: ParentStream::new(),
            names: RwLock::default(),
        };
        // Slot 0 is [`ROOT`], so it exists before any caller can ask for a
        // parent and every later `create_transform` can default to it.
        h.create_transform(_Transform {
            name: "root".into(),
            .._Transform::default()
        });
        h
    }

    pub fn len(&self) -> usize {
        self.mutexes.len()
    }

    /// Live slots. [`len`](Self::len) is the high-water mark instead —
    /// removing a transform frees its slot without shrinking the arrays.
    pub fn active_len(&self) -> usize {
        let live = |w: &AtomicU32| w.load(Ordering::Relaxed).count_ones() as usize;
        self.active.iter().map(live).sum()
    }

    /// The world this hierarchy belongs to.
    pub fn world(&self) -> WorldId {
        self.world
    }

    // ── Raw component access (no-lock fast path) ─────────────────────────
    //
    // These accessors hand out plain `&[T]` views over the SoA component
    // arrays. They exist for hot paths (today: the renderer's per-frame
    // staging upload) that:
    //
    //   * already hold the system-level invariant that no other thread is
    //     mutating the hierarchy for the duration of the borrow (i.e. the
    //     update callback has returned and the renderer is the sole reader
    //     until the next callback), and
    //   * want to amortise the per-entity `Mutex` lock + parent chain
    //     traversal that a `TransformGuard` implies.
    //
    // The compile-time signature is `&self`, so concurrent reads are fine.
    // The runtime contract is that no `TransformGuard` is mutating any of
    // these arrays concurrently — same contract as `get_transform_unchecked`
    // and the existing `Dirty::*_words` atomics.
    //
    // `SyncUnsafeCell<T>` is `#[repr(transparent)]` over `T`, so a slice of
    // `SyncUnsafeCell<T>` has the same layout as a slice of `T` and the
    // pointer-cast below is sound.

    /// This slot's children, in order. Guard-free for the same reason
    /// [`Self::positions_raw`] is: a hierarchy panel walks thousands of these
    /// per structural edit and the sim is not mutating while it does.
    #[inline]
    pub fn children(&self, idx: u32) -> Children<'_> {
        let m = self._meta(idx);
        Children {
            h: self,
            front: m.first,
            back: m.last,
        }
    }

    /// This slot's name. Same aliasing contract as [`Self::children`].
    #[inline]
    pub fn name(&self, idx: u32) -> Name {
        unsafe { &*self.metadata[idx as usize].get() }.name
    }

    /// `parent`'s child called `name`.
    pub fn child(&self, parent: u32, name: Name) -> Option<u32> {
        self.names.read().child(parent, name)
    }

    /// The slot `path` names below `from`, or from the root for `/root/...`.
    pub fn find(&self, from: u32, path: &(impl Lookup + ?Sized)) -> Option<u32> {
        let (absolute, mut names) = path.parts();
        if absolute && names.next().flatten() != Some(self._meta(ROOT).name) {
            return None;
        }
        let by = self.names.read();
        let mut at = if absolute { ROOT } else { from };
        for name in names {
            at = by.child(at, name?)?;
        }
        Some(at)
    }

    /// Read-only view of the local-space position component array.
    ///
    /// One entry per entity slot in insertion order; index with the entity
    /// `u32` from `Transform::get_idx`. See the section comment above for
    /// the aliasing contract.
    #[inline]
    pub fn positions_raw(&self) -> &[Vec3] {
        // SAFETY: see section comment.
        unsafe {
            std::slice::from_raw_parts(self.positions.as_ptr() as *const Vec3, self.positions.len())
        }
    }
    /// Read-only view of the local-space rotation component array.
    #[inline]
    pub fn rotations_raw(&self) -> &[Quat] {
        // SAFETY: see section comment.
        unsafe {
            std::slice::from_raw_parts(self.rotations.as_ptr() as *const Quat, self.rotations.len())
        }
    }
    /// Read-only view of the local-space scale component array.
    #[inline]
    pub fn scales_raw(&self) -> &[Vec3] {
        // SAFETY: see section comment.
        unsafe {
            std::slice::from_raw_parts(self.scales.as_ptr() as *const Vec3, self.scales.len())
        }
    }

    /// Borrow the per-component dirty bitmasks. The renderer drains these
    /// per frame (atomic `swap(0, …)`) to discover which entity slots'
    /// position / rotation / scale changed since the last frame.
    #[inline]
    pub fn dirty(&self) -> &Dirty {
        &self.dirty
    }

    /// Drain every parent change recorded since the last drain into
    /// `[trs_index, new_parent]` pairs ([`NO_PARENT`] = detached), ready to
    /// be written into the renderer's parent-update staging buffer and
    /// scattered by the GPU parent-scatter pass.
    ///
    /// The new-parent value is snapshotted **now**, from the authoritative
    /// metadata — not at record time — so duplicate records for the same
    /// index (multiple re-parents in one frame, possibly from different
    /// threads/shards) all yield the same pair and any replay order is
    /// correct. Aliasing contract: same as [`positions_raw`]
    /// (Self::positions_raw) — no `TransformGuard` may be mutating the
    /// hierarchy concurrently; the renderer calls this after the sim
    /// update has returned.
    pub fn drain_parent_updates(&self) -> Vec<[u32; 2]> {
        self.parent_stream
            .drain_indices()
            .into_iter()
            .map(|idx| {
                // SAFETY: see aliasing contract above.
                let parent = unsafe { &*self.metadata[idx as usize].get() }.parent;
                [idx, parent]
            })
            .collect()
    }

    pub fn create_transform<'a>(&'a mut self, t: _Transform) -> Transform<'a> {
        // The root is its own parent, which is what makes `parent == ROOT`
        // the loop terminator everywhere.
        let parent = t.parent.unwrap_or(ROOT);
        let meta = TransformMeta {
            parent,
            first: NONE,
            last: NONE,
            prev: NONE,
            next: NONE,
            name: t.name,
        };
        // The lowest freed slot first, so the hierarchy stays dense.
        self.avail.commit();
        let (idx, reused) = match self.avail.pop() {
            Some(i) => {
                let i = i as usize;
                *self.positions[i].get_mut() = t.position;
                *self.rotations[i].get_mut() = t.rotation;
                *self.scales[i].get_mut() = t.scale;
                *self.metadata[i].get_mut() = meta;
                (i, true)
            }
            None => {
                let i = self.mutexes.len();
                self.mutexes.push(Mutex::new(()));
                self.positions.push(SyncUnsafeCell::new(t.position));
                self.rotations.push(SyncUnsafeCell::new(t.rotation));
                self.scales.push(SyncUnsafeCell::new(t.scale));
                self.metadata.push(SyncUnsafeCell::new(meta));
                self.generations.push(AtomicU32::new(0));
                (i, false)
            }
        };
        if idx as u32 != ROOT && !t.name.is_none() {
            // `claim` without the lock, which `&mut self` makes free.
            let got = self.names.get_mut().claim(parent, t.name, idx as u32);
            self.metadata[idx].get_mut().name = got;
        }
        if idx as u32 != ROOT {
            self.link(parent, idx as u32, NONE);
            self.has_children[parent as usize >> 5].fetch_or(1 << (parent & 31), Ordering::Relaxed);
            // A fresh slot parented to the root needs no record — the renderer
            // zero-fills its per-slot parent buffer, and zero *is* the root.
            // A reused one still holds its last occupant's parent there.
            if parent != ROOT || reused {
                self.parent_stream.record(idx as u32);
            }
        }
        // if idx >> 1 >= self.dirty.len() {
        //     self.dirty.push(AtomicU8::new(0b1111)); // one u8 for every 2 transforms
        // } else {
        //     self.dirty[idx >> 1].fetch_or(0b1111 << 4, Ordering::Relaxed);
        // }
        // if idx >> 10 >= self.dirty_l2.len() {
        //     self.dirty_l2.push(AtomicU32::new(0));
        // }
        // self.dirty_l2[idx >> 10].fetch_or(1 << ((idx >> 5) & 0b11111), Ordering::Relaxed);
        if idx >> 5 >= self.dirty.len() {
            self.has_children.push(AtomicU32::new(0));
            self.dirty.push();
            self.active.push(AtomicU32::new(0));
        }
        self.active[idx >> 5].fetch_or(1 << (idx & 31), Ordering::Relaxed);
        //       self.mark_dirty(
        // 	&self._lock_internal(idx as u32),
        // 	TransformComponent::Parent
        // 		| TransformComponent::Position
        // 		| TransformComponent::Rotation
        // 		| TransformComponent::Scale,
        // );
        self.dirty.all(idx as u32);
        // self.dirty.push(AtomicU8::new(0b1111));

        Transform::new(self, idx as u32)
    }
    /// Remove `t` **and everything below it**, returning the removed slots
    /// (`t` first, parents before children).
    ///
    /// Crate-private, and [`World::remove_entity`] is the only caller: those
    /// slots still hold components, which the hierarchy cannot see and only
    /// the world can drop. A public entry point here would be a way to kill a
    /// transform and leak its component behind it.
    ///
    /// The subtree and not the node alone: re-homing orphans puts them
    /// somewhere the user never placed them, and there is no delete a user
    /// can ask for that means "keep the children".
    ///
    /// [`World::remove_entity`]: crate::component::World::remove_entity
    pub(crate) fn remove_transform(&self, t: TransformGuard) -> Vec<u32> {
        let t_idx = t.idx as u32;
        assert_ne!(t_idx, ROOT, "the hierarchy root cannot be removed");
        if !self.get_active(t_idx) {
            return Vec::new();
        }
        let removed = self.subtree(t_idx);

        let parent = self
            .get_parent(&t)
            .expect("only ROOT has no parent, and it cannot be removed");
        drop(t);
        let guard = self._lock_internal(parent);
        self.unlink(t_idx);
        drop(guard);

        let mut names = None;
        for &idx in &removed {
            let m = self._meta(idx);
            if !m.name.is_none() {
                let names = names.get_or_insert_with(|| self.names.write());
                names.release(m.parent, m.name);
            }
        }
        drop(names);
        for &idx in &removed {
            // Held one at a time, and `t`'s was dropped above: nothing here
            // takes a second slot's lock while holding the first.
            let g = self._lock_internal(idx);
            // Not recorded as an enable change: the slot is about to lose its
            // components, and `World::remove_entity` notifies them itself
            // while they are still there to scatter their own sentinel.
            self.active[idx as usize >> 5].fetch_and(!(1 << (idx & 31)), Ordering::Relaxed);
            self.has_children[idx as usize >> 5].fetch_and(!(1 << (idx & 31)), Ordering::Relaxed);
            self.dirty.all(idx);
            let m = self.get_meta(&g);
            (m.first, m.last) = (NONE, NONE);
            self.generations[idx as usize].fetch_add(1, Ordering::Relaxed);
            self.avail.push(idx);
        }
        removed
    }

    /// `idx` and its descendants, parents before children. Iterative — a
    /// deep GLB rig is exactly the shape that would blow a recursive walk.
    pub(crate) fn subtree(&self, idx: u32) -> Vec<u32> {
        let mut out = vec![idx];
        let mut i = 0;
        while i < out.len() {
            out.extend(self.children(out[i]));
            i += 1;
        }
        out
    }

    /// Whether slot `idx` is a live entity. Freed slots are never reused, so
    /// this is how a deferred edit tells "still there" from "destroyed since".
    #[inline]
    /// How many times slot `idx` has been freed; `None` past the end.
    pub fn generation(&self, idx: u32) -> Option<u32> {
        let g = self.generations.get(idx as usize)?;
        Some(g.load(Ordering::Relaxed))
    }

    pub fn is_active(&self, idx: u32) -> bool {
        self.get_active(idx)
    }

    #[inline]
    fn get_active(&self, idx: u32) -> bool {
        let mask = 1 << (idx & 0b11111);
        (self.active[idx as usize >> 5].load(Ordering::Relaxed) & mask) != 0
    }

    #[inline]
    fn get_has_children(&self, idx: u32) -> bool {
        let mask = 1 << (idx & 0b11111);
        (self.has_children[idx as usize >> 5].load(Ordering::Relaxed) & mask) != 0
    }
    // #[inline]
    // fn mark_dirty(&self, t: &TransformGuard, component: impl Into<TransformComponentFlags>) {
    //     let flags: TransformComponentFlags = component.into();
    //     let shift = (t.idx & 1) * 4;
    //     let flag = flags.0 << shift;
    //     unsafe {
    //         self.dirty
    //             .get_unchecked(t.idx >> 1)
    //             .fetch_or(flag, Ordering::Relaxed)
    //     };
    // }

    // fn get_dirty(&self, idx: u32) -> u8 {
    //     let shift = (idx & 1) * 4; // Fixed: Added parentheses
    //     let mask = 0b1111 << shift;
    //     // Fixed: Use load to read without modifying; shift back to return only the 4 bits
    //     ((unsafe { self.dirty.get_unchecked((idx >> 1) as usize) }).load(Ordering::Relaxed) & mask)
    //         >> shift
    // }

    fn get_dirty_l2(&self, chunk_id: usize) -> u32 {
        self.dirty_l2[chunk_id].swap(0, Ordering::Relaxed)
    }

    // fn mark_clean(&self, idx: u32) {
    //     let shift = (idx & 1) * 4; // Fixed: Added parentheses
    //     let mask = !(0b1111 << shift); // Fixed: Use NOT of the mask to clear the bits
    //     unsafe {
    //         self.dirty
    //             .get_unchecked((idx >> 1) as usize)
    //             .fetch_and(mask, Ordering::Relaxed)
    //     };
    // }

    /// Rename `t`, suffixed if a sibling has the name, and return what it
    /// got. Nothing is dirtied: the render path never reads a name.
    pub fn set_name(&self, t: &TransformGuard, name: impl Into<Name>) -> Name {
        let (idx, name) = (t.idx as u32, name.into());
        let meta = self.get_meta(t);
        if meta.name != name {
            self.names.write().release(meta.parent, meta.name);
            meta.name = name;
            if idx != ROOT {
                self.claim(meta.parent, idx);
            }
        }
        self.get_meta(t).name
    }

    /// Index `idx` under `parent`, suffixing its name to be unique there.
    fn claim(&self, parent: u32, idx: u32) {
        let m = self._meta(idx);
        if m.name.is_none() {
            return;
        }
        m.name = self.names.write().claim(parent, m.name, idx);
    }

    /// Re-parent `t`. `None` means [`ROOT`].
    ///
    /// Panics on a cycle rather than producing a hierarchy whose composition
    /// walk cannot terminate — the GPU walk would otherwise silently bottom
    /// out at `MAX_PARENT_DEPTH` and render garbage.
    pub fn set_parent(&self, t: &TransformGuard, parent: Option<u32>) {
        self.reparent(t, parent, None);
    }

    /// Re-parent `t` and place it at `at` among the new parent's children.
    ///
    /// `at` counts those children **after `t` has left its old parent**, so a
    /// caller that computed an index from what it saw on screen needs no
    /// off-by-one when the move is within one parent. Out of range panics —
    /// an index nobody can point at is a caller bug, not something to clamp.
    pub fn set_parent_at(&self, t: &TransformGuard, parent: Option<u32>, at: usize) {
        self.reparent(t, parent, Some(at));
    }

    /// Move `t` to index `at` among its **current** parent's children.
    ///
    /// Ordering only. The parent link does not change, so nothing is dirtied
    /// and nothing reaches the GPU: the composition walk runs child → parent
    /// and never reads sibling order, which is why this is not
    /// [`set_parent_at`](Self::set_parent_at) aimed at the parent a node
    /// already has — that would record a parent change that did not happen
    /// and push it down the parent stream for nothing.
    ///
    /// Order is still worth maintaining because it is what the user sees: a
    /// hierarchy panel lists children in this order, and [`remove_transform`]
    /// is order-preserving for the same reason.
    ///
    /// [`remove_transform`]: Self::remove_transform
    pub fn move_child(&self, t: &TransformGuard, at: usize) {
        let t_idx = t.idx as u32;
        assert_ne!(
            t_idx, ROOT,
            "the hierarchy root has no siblings to move among"
        );
        let parent = self._meta(t_idx).parent;
        let _guard = self._lock_internal(parent);
        self.unlink(t_idx);
        self.link_at(parent, t_idx, at);
    }

    /// Put `idx` into `parent`'s children before `before`, or last for
    /// [`NONE`].
    fn link(&self, parent: u32, idx: u32, before: u32) {
        let prev = match before {
            NONE => self._meta(parent).last,
            b => self._meta(b).prev,
        };
        let m = self._meta(idx);
        (m.prev, m.next) = (prev, before);
        match prev {
            NONE => self._meta(parent).first = idx,
            p => self._meta(p).next = idx,
        }
        match before {
            NONE => self._meta(parent).last = idx,
            b => self._meta(b).prev = idx,
        }
    }

    /// [`link`](Self::link) at index `at`. Past the end panics, as
    /// `Vec::insert` did: an index nobody can point at is a caller bug.
    fn link_at(&self, parent: u32, idx: u32, at: usize) {
        let mut kids = self.children(parent);
        let before = kids.nth(at).unwrap_or_else(|| {
            let len = self.children(parent).count();
            assert_eq!(at, len, "child index {at} past {len} children");
            NONE
        });
        self.link(parent, idx, before);
    }

    /// Take `idx` out of its parent's children, which keep their order.
    fn unlink(&self, idx: u32) {
        let m = self._meta(idx);
        let (parent, prev, next) = (m.parent, m.prev, m.next);
        (m.prev, m.next) = (NONE, NONE);
        match prev {
            NONE => self._meta(parent).first = next,
            p => self._meta(p).next = next,
        }
        match next {
            NONE => self._meta(parent).last = prev,
            n => self._meta(n).prev = prev,
        }
        if self._meta(parent).first == NONE {
            self.has_children[parent as usize >> 5]
                .fetch_and(!(1 << (parent & 0b11111)), Ordering::Relaxed);
        }
    }

    fn reparent(&self, t: &TransformGuard, parent: Option<u32>, at: Option<usize>) {
        let t_idx = t.idx as u32;
        assert_ne!(t_idx, ROOT, "the hierarchy root cannot be re-parented");
        let new_parent = parent.unwrap_or(ROOT);

        let mut p = new_parent;
        while p != ROOT {
            assert_ne!(p, t_idx, "re-parenting {t_idx} under its own descendant");
            p = self._meta(p).parent;
        }

        if let Some(old_parent) = self.get_parent(t) {
            let _guard = self._lock_internal(old_parent);
            self.unlink(t_idx);
            self.names
                .write()
                .release(old_parent, self._meta(t_idx).name);
        }
        self.get_meta(t).parent = new_parent;
        self.claim(new_parent, t_idx);
        let guard = self._lock_internal(new_parent);
        match at {
            Some(at) => self.link_at(new_parent, t_idx, at),
            None => self.link(new_parent, t_idx, NONE),
        }
        drop(guard);
        self.has_children[new_parent as usize >> 5]
            .fetch_or(1 << (new_parent & 0b11111), Ordering::Relaxed);

        self.dirty.parent(t.idx as u32);
        self.parent_stream.record(t.idx as u32);
    }
    fn _lock_internal<'a>(&'a self, idx: u32) -> TransformGuard<'a> {
        let lock = self.mutexes[idx as usize].lock();
        TransformGuard {
            hierarchy: self,
            idx: idx as usize,
            _lock: lock,
        }
    }
    fn _scale(&self, idx: u32) -> &mut Vec3 {
        unsafe { &mut *self.scales.get_unchecked(idx as usize).get() }
    }
    fn _position(&self, idx: u32) -> &mut Vec3 {
        unsafe { &mut *self.positions.get_unchecked(idx as usize).get() }
    }
    fn _rotation(&self, idx: u32) -> &mut Quat {
        unsafe { &mut *self.rotations.get_unchecked(idx as usize).get() }
    }
    fn _meta(&self, idx: u32) -> &mut TransformMeta {
        unsafe { &mut *self.metadata.get_unchecked(idx as usize).get() }
    }
    fn scale_by(&self, t: &TransformGuard, scale: Vec3) {
        let s = self._scale(t.idx as u32);
        *s *= scale;
        self.dirty.scale(t.idx as u32);
        // self.mark_dirty(t, TransformComponent::Scale);
    }
    fn set_scale(&self, t: &TransformGuard, scale: Vec3) {
        let s = self._scale(t.idx as u32);
        *s = scale;
        // if self.get_has_children(t.idx as u32) {
        //     let base_pos = self._position(t.idx as u32);
        //     self.scale_children(t, scale, base_pos);
        // }
        self.dirty.scale(t.idx as u32);
        // self.mark_dirty(t, TransformComponent::Scale);
    }
    pub(crate) fn scale_children(&self, t: &TransformGuard, scale: Vec3, base_pos: &Vec3) {
        for child in self.children(t.idx as u32) {
            let child = self._lock_internal(child);
            let s = self._scale(child.idx as u32);
            let p = self._position(child.idx as u32);
            *s *= scale;
            *p = base_pos + (*p - base_pos) * scale;
            self.dirty.scale(child.idx as u32);
            // self.mark_dirty(&child, TransformComponent::Scale);
            // if self.get_has_children(child.idx as u32) {
            //     self.scale_children(&child, scale, base_pos);
            // }
        }
    }
    fn shift(&self, t: &TransformGuard, delta: Vec3) {
        let p = self._position(t.idx as u32);
        *p += delta;
        self.dirty.pos(t.idx as u32);
        // self.mark_dirty(t, TransformComponent::Position);
        // if self.get_has_children(t.idx as u32) {
        //     self.translate_children(t, delta);
        // }
    }
    fn translate_by(&self, t: &TransformGuard, translation: Vec3) {
        let p = self._position(t.idx as u32);
        let r = self._rotation(t.idx as u32);
        let translation = *r * translation;
        *p += translation;
        self.dirty.pos(t.idx as u32);
        // self.mark_dirty(t, TransformComponent::Position);
        // if self.get_has_children(t.idx as u32) {
        //     self.translate_children(t, translation);
        // }
    }
    pub(crate) fn translate_children(&self, t: &TransformGuard, translation: Vec3) {
        for child in self.children(t.idx as u32) {
            let child = self._lock_internal(child);
            let p = self._position(child.idx as u32);
            *p += translation;
            self.dirty.pos(child.idx as u32);
            // self.mark_dirty(&child, TransformComponent::Position);
            // if self.get_has_children(child.idx as u32) {
            //     self.translate_children(&child, translation);
            // }
        }
    }
    fn set_position(&self, t: &TransformGuard, position: Vec3) {
        let p = self._position(t.idx as u32);
        // if self.get_has_children(t.idx as u32) {
        //     let delta = position.sub(*p);
        //     self.translate_children(t, delta);
        // }
        *p = position;
        self.dirty.pos(t.idx as u32);
        // self.mark_dirty(t, TransformComponent::Position);
    }
    fn rotate_by(&self, t: &TransformGuard, rotation: Quat) {
        let r = self._rotation(t.idx as u32);
        *r = rotation * *r;
        self.dirty.rot(t.idx as u32);
        // self.mark_dirty(t, TransformComponent::Rotation);
        // if self.get_has_children(t.idx as u32) {
        //     self.rotate_children(t, rotation, *self._position(t.idx as u32));
        // }
    }
    pub(crate) fn rotate_children(&self, t: &TransformGuard, rotation: Quat, position: Vec3) {
        // Sequential walk: the new static thread pool does not support
        // nested parallelism, and `rotate_children` recurses into itself.
        // The top-level callers (currently all commented out in the
        // setters above) would dispatch the outer level via the pool;
        // each task then walks its subtree sequentially.
        for child in self.children(t.idx as u32) {
            let child = self._lock_internal(child);
            let r = self._rotation(child.idx as u32);
            let p = self._position(child.idx as u32);
            *p = rotation * (*p - position) + position;
            *r = rotation * *r;
            self.dirty.pos_rot(child.idx as u32);
            if self.get_has_children(child.idx as u32) {
                self.rotate_children(&child, rotation, position);
            }
        }
    }
    fn set_rotation(&self, t: &TransformGuard, rotation: Quat) {
        let r = self._rotation(t.idx as u32);
        // if self.get_has_children(t.idx as u32) {
        //     let delta = rotation * r.conjugate();
        //     self.rotate_children(t, delta, *self._position(t.idx as u32));
        // }
        *r = rotation;
        self.dirty.rot(t.idx as u32);
        // self.mark_dirty(t, TransformComponent::Rotation);
    }
    pub fn get_transform_unchecked(&self, idx: u32) -> Transform<'_> {
        Transform::new(self, idx)
    }
    pub fn get_transform(&self, idx: u32) -> Option<Transform<'_>> {
        if (idx as usize) < self.mutexes.len() {
            if self.get_active(idx) {
                Some(Transform::new(self, idx))
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn get_transform_(&self, idx: u32) -> _Transform {
        let guard = self._lock_internal(idx);
        _Transform {
            position: self.get_position(&guard),
            rotation: self.get_rotation(&guard),
            scale: self.get_scale(&guard),
            name: self.get_meta(&guard).name,
            parent: self.get_parent(&guard),
        }
    }
    fn get_position(&self, t: &TransformGuard) -> Vec3 {
        unsafe { *self.positions[t.idx as usize].get() }
    }
    fn get_rotation(&self, t: &TransformGuard) -> Quat {
        unsafe { *self.rotations[t.idx as usize].get() }
    }
    fn get_scale(&self, t: &TransformGuard) -> Vec3 {
        unsafe { *self.scales[t.idx as usize].get() }
    }
    /// `None` only for the [`ROOT`], which is its own parent.
    fn get_parent(&self, t: &TransformGuard) -> Option<u32> {
        let parent = unsafe { &*self.metadata[t.idx as usize].get() }.parent;
        (t.idx as u32 != ROOT).then_some(parent)
    }
    fn get_meta(&self, t: &TransformGuard) -> &mut TransformMeta {
        unsafe { &mut *self.metadata[t.idx as usize].get() }
    }
    fn get_global_transform(&self, t: &TransformGuard) -> _Transform {
        let mut global_position = self.get_position(t);
        let mut global_rotation = self.get_rotation(t);
        let mut global_scale = self.get_scale(t);
        let mut parent = self._meta(t.idx as u32).parent;
        while parent != ROOT {
            let parent_position = self._position(parent);
            let parent_rotation = self._rotation(parent);
            let parent_scale = self._scale(parent);
            global_position =
                *parent_position + (*parent_rotation * global_position) * *parent_scale;
            global_rotation = *parent_rotation * global_rotation;
            global_scale *= *parent_scale;
            parent = self._meta(parent).parent;
        }
        _Transform {
            position: global_position,
            rotation: global_rotation,
            scale: global_scale,
            name: self.get_meta(t).name,
            parent: None,
        }
    }

    fn get_global_position(&self, t: &TransformGuard) -> Vec3 {
        let mut global_position = self.get_position(t);
        let mut parent = self._meta(t.idx as u32).parent;
        // let mut _parent = self.get_parent(t);
        // while let Some(parent) = _parent {
        //     let parent = self._lock_internal(parent);
        //     let parent_position = self.get_position(&parent);
        //     let parent_rotation = self.get_rotation(&parent);
        //     global_position = parent_position + parent_rotation * global_position;
        //     _parent = self.get_parent(&parent);
        // }
        while parent != ROOT {
            let parent_position = self._position(parent);
            let parent_rotation = self._rotation(parent);
            let parent_scale = self._scale(parent);
            global_position =
                *parent_position + (*parent_rotation * global_position) * *parent_scale;
            parent = self._meta(parent).parent;
        }
        global_position
    }
    fn get_global_rotation(&self, t: &TransformGuard) -> Quat {
        let mut global_rotation = self.get_rotation(t);
        let mut parent = self._meta(t.idx as u32).parent;
        while parent != ROOT {
            global_rotation = *self._rotation(parent) * global_rotation;
            parent = self._meta(parent).parent;
        }
        global_rotation
    }
    fn get_global_scale(&self, t: &TransformGuard) -> Vec3 {
        let mut global_scale = self.get_scale(t);
        let mut parent = self._meta(t.idx as u32).parent;
        while parent != ROOT {
            global_scale *= *self._scale(parent);
            parent = self._meta(parent).parent;
        }
        global_scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(name: &str, parent: Option<u32>) -> _Transform {
        _Transform {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            name: name.into(),
            parent,
        }
    }

    fn children_of(h: &TransformHierarchy, idx: u32) -> Vec<u32> {
        h.children(idx).collect()
    }

    /// `a → b → c`.
    fn chain() -> (TransformHierarchy, u32, u32, u32) {
        let mut h = TransformHierarchy::new(0);
        let a = h.create_transform(plain("a", None)).get_idx();
        let b = h.create_transform(plain("b", Some(a))).get_idx();
        let c = h.create_transform(plain("c", Some(b))).get_idx();
        (h, a, b, c)
    }

    /// A reused slot parented to the root must still tell the renderer so:
    /// its GPU parent entry names whatever its last occupant hung under.
    #[test]
    fn a_reused_root_level_slot_records_its_parent() {
        let (mut h, a, b, _) = chain();
        let t = h.get_transform_unchecked(b).lock();
        h.remove_transform(t);
        h.drain_parent_updates();

        let again = h.create_transform(plain("again", None)).get_idx();
        assert_eq!(again, b, "the lowest freed slot");
        assert_eq!(h.drain_parent_updates(), [[b, ROOT]]);
        assert_eq!(h.generation(b), Some(1));
        assert!(children_of(&h, a).is_empty(), "nothing re-hung under a");
    }

    /// A hierarchy is one world's, so every transform in it is in that world
    /// — the whole of what used to be a per-slot id and a migration queue.
    #[test]
    fn every_slot_is_in_the_hierarchy_s_world() {
        let (h, a, b, c) = chain();
        for idx in [ROOT, a, b, c] {
            assert_eq!(h.get_transform_unchecked(idx).world(), 0);
        }
        assert_eq!(TransformHierarchy::new(3).world(), 3);
    }

    #[test]
    fn root_exists_and_is_the_default_parent() {
        let mut h = TransformHierarchy::new(DEFAULT_WORLD);
        assert_eq!(h.len(), 1, "the root occupies slot 0 from construction");

        let a = h.create_transform(plain("a", None)).get_idx();
        assert_ne!(a, ROOT);
        assert_eq!(h._meta(a).parent, ROOT);
        assert_eq!(children_of(&h, ROOT), vec![a]);
        assert!(h.get_has_children(ROOT));

        // The root is its own parent, which is what terminates every walk.
        assert_eq!(h._meta(ROOT).parent, ROOT);
        assert_eq!(h.get_parent(&h.get_transform_unchecked(ROOT).lock()), None);
    }

    /// An explicitly-parented spawn must land in its parent's child list —
    /// the path `scene_asset::instantiate_template` takes for every glTF node.
    #[test]
    fn explicit_parent_populates_the_child_list() {
        let mut h = TransformHierarchy::new(0);
        let a = h.create_transform(plain("a", None)).get_idx();
        let b = h.create_transform(plain("b", Some(a))).get_idx();
        let c = h.create_transform(plain("c", Some(a))).get_idx();
        assert_eq!(children_of(&h, a), vec![b, c]);
        assert!(h.get_has_children(a));
        assert_eq!(children_of(&h, ROOT), vec![a]);
    }

    /// Sibling order survives a removal from the middle — what makes an undo
    /// of that removal exact, and what stops a hierarchy panel reordering
    /// itself when an unrelated entity dies.
    #[test]
    fn removal_preserves_sibling_order() {
        let mut h = TransformHierarchy::new(0);
        let kids: Vec<u32> = (0..4)
            .map(|i| h.create_transform(plain(&format!("k{i}"), None)).get_idx())
            .collect();
        assert_eq!(children_of(&h, ROOT), kids);

        h.remove_transform(h.get_transform_unchecked(kids[1]).lock());
        assert_eq!(children_of(&h, ROOT), vec![kids[0], kids[2], kids[3]]);
    }

    /// A drop from a hierarchy panel names a position among the new parent's
    /// children, not "somewhere under it" — so the index has to land.
    #[test]
    fn set_parent_at_places_the_child_where_it_was_dropped() {
        let mut h = TransformHierarchy::new(0);
        let a = h.create_transform(plain("a", None)).get_idx();
        let kids: Vec<u32> = (0..3)
            .map(|i| {
                h.create_transform(plain(&format!("k{i}"), Some(a)))
                    .get_idx()
            })
            .collect();
        let moved = h.create_transform(plain("moved", None)).get_idx();

        h.set_parent_at(&h.get_transform_unchecked(moved).lock(), Some(a), 1);
        assert_eq!(children_of(&h, a), vec![kids[0], moved, kids[1], kids[2]]);
        assert!(
            !children_of(&h, ROOT).contains(&moved),
            "left its old parent"
        );

        // Appending is the same call with the end index, which is what
        // `set_parent` reduces to.
        let last = h.create_transform(plain("last", None)).get_idx();
        h.set_parent_at(&h.get_transform_unchecked(last).lock(), Some(a), 4);
        assert_eq!(children_of(&h, a).last(), Some(&last));
    }

    /// `at` counts the destination's children *after* the node has left its
    /// old parent. Within one parent that is the whole subtlety, and getting
    /// it wrong shifts every backwards move by one.
    #[test]
    fn move_child_reorders_within_one_parent() {
        let mut h = TransformHierarchy::new(0);
        let kids: Vec<u32> = (0..4)
            .map(|i| h.create_transform(plain(&format!("k{i}"), None)).get_idx())
            .collect();
        let _ = h.drain_parent_updates();

        h.move_child(&h.get_transform_unchecked(kids[0]).lock(), 2);
        assert_eq!(
            children_of(&h, ROOT),
            vec![kids[1], kids[2], kids[0], kids[3]]
        );

        h.move_child(&h.get_transform_unchecked(kids[3]).lock(), 0);
        assert_eq!(
            children_of(&h, ROOT),
            vec![kids[3], kids[1], kids[2], kids[0]]
        );

        // Ordering is invisible to the GPU — the composition walk runs
        // child → parent — so a reorder must not reach the parent stream.
        assert!(
            h.drain_parent_updates().is_empty(),
            "a sibling reorder is not a parent change"
        );
    }

    /// A cycle would make the composition walk non-terminating; on the GPU it
    /// would bottom out at `MAX_PARENT_DEPTH` and render garbage instead.
    #[test]
    #[should_panic(expected = "own descendant")]
    fn reparenting_under_a_descendant_panics() {
        let mut h = TransformHierarchy::new(0);
        let a = h.create_transform(plain("a", None)).get_idx();
        let b = h.create_transform(plain("b", Some(a))).get_idx();
        let t = h.get_transform_unchecked(a);
        h.set_parent(&t.lock(), Some(b));
    }

    #[test]
    #[should_panic(expected = "root cannot be removed")]
    fn removing_the_root_panics() {
        let h = TransformHierarchy::new(0);
        h.remove_transform(h.get_transform_unchecked(ROOT).lock());
    }

    /// The recurrence `mvp_build.comp` transcribes, walked to the root. Guards
    /// the loop terminator: a chain that stopped one level early would still
    /// look plausible for depth 1.
    #[test]
    fn nested_chain_composes_through_to_the_root() {
        let mut h = TransformHierarchy::new(0);
        let mut prev = None;
        let chain: Vec<u32> = (0..4)
            .map(|i| {
                let idx = h.create_transform(plain(&format!("n{i}"), prev)).get_idx();
                h.get_transform_unchecked(idx)
                    .lock()
                    .set_position(Vec3::new(0.0, 2.0, 0.0));
                prev = Some(idx);
                idx
            })
            .collect();

        for (i, &idx) in chain.iter().enumerate() {
            let t = h.get_transform_unchecked(idx);
            assert_eq!(
                t.lock().get_global_position(),
                Vec3::new(0.0, 2.0 * (i + 1) as f32, 0.0),
                "depth {i} did not compose the whole chain"
            );
        }
    }

    /// The root is a structural anchor: its own TRS is never composed in.
    #[test]
    fn root_transform_is_not_composed() {
        let mut h = TransformHierarchy::new(0);
        let a = h.create_transform(plain("a", None)).get_idx();
        {
            let r = h.get_transform_unchecked(ROOT);
            r.lock().set_position(Vec3::new(100.0, 0.0, 0.0));
        }
        let t = h.get_transform_unchecked(a);
        assert_eq!(t.lock().get_global_position(), Vec3::ZERO);
    }

    #[test]
    fn siblings_names_stay_unique_through_every_way_in() {
        let (mut h, a, b, c) = chain();
        let dup = h.create_transform(plain("b", Some(a))).get_idx();
        assert_eq!(h.name(dup), "b1");
        h.set_name(&h.get_transform_unchecked(dup).lock(), "b");
        assert_eq!(h.name(dup), "b2", "a rename past a taken name");
        h.set_parent(&h.get_transform_unchecked(c).lock(), Some(a));
        assert_eq!(h.name(c), "c", "nothing to collide with");
        let other = h.create_transform(plain("c", Some(b))).get_idx();
        h.set_parent(&h.get_transform_unchecked(other).lock(), Some(a));
        assert_eq!(h.name(other), "c1", "a move into a taken name");
        let anon =
            [plain("", Some(a)), plain("", Some(a))].map(|t| h.create_transform(t).get_idx());
        assert!(anon.iter().all(|&i| h.name(i).is_none()));
    }

    #[test]
    fn a_path_finds_by_name_and_a_freed_name_is_free() {
        let (mut h, a, b, c) = chain();
        assert_eq!(h.find(ROOT, "/root/a/b/c"), Some(c));
        assert_eq!(h.find(ROOT, "a/b"), Some(b));
        assert_eq!(h.find(a, "b/c"), Some(c));
        assert_eq!(h.find(a, &Path::new("b/c")), Some(c));
        assert_eq!(
            h.find(ROOT, "/a/b"),
            None,
            "an absolute path starts at root"
        );
        assert_eq!(h.find(ROOT, "a/nobody"), None);
        let dup = h.create_transform(plain("b", Some(a))).get_idx();
        h.remove_transform(h.get_transform_unchecked(b).lock());
        assert_eq!(h.find(a, "b1"), Some(dup));
        assert_eq!(h.find(a, "b/c"), None, "gone with its parent");
        let again = h.create_transform(plain("b", Some(a))).get_idx();
        assert_eq!(h.name(again), "b");
    }

    #[test]
    fn a_freed_parent_s_suffixes_do_not_reach_its_slot_s_next_occupant() {
        let (mut h, a, b, _) = chain();
        for _ in 0..3 {
            h.create_transform(plain("c", Some(b)));
        }
        h.remove_transform(h.get_transform_unchecked(b).lock());
        let fresh = h.create_transform(plain("fresh", Some(a))).get_idx();
        assert_eq!(fresh, b, "the slot came back");
        let kids = [plain("c", Some(fresh)), plain("c", Some(fresh))]
            .map(|t| h.create_transform(t).get_idx());
        assert_eq!(kids.map(|k| h.name(k).to_string()), ["c", "c1"]);
    }
}
