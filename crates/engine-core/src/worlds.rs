//! Every world in the process, and the handle that names one.
//!
//! ADR-0011 §3, revised: a world is engine-owned and refcounted rather than a
//! value the app hands to the window. [`new_world`] registers one and it lives
//! until the last [`WorldHandle`] drops, so an overlay world dies with the
//! component that made it. Reaching another world is an ordinary capability
//! here, not the editor's privilege.
//!
//! `&mut World` exists only *between* frames. Structural changes asked for
//! during one queue on the world and land in [`apply_pending`] — which is what
//! lets the hierarchy's SoA stay contiguous `Vec`s while a sweep reads it.

use std::cell::SyncUnsafeCell;
use std::ops::Deref;
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::component::World;
use crate::transform::WorldId;

pub(crate) type WorldCell = SyncUnsafeCell<World>;

/// Every world ever registered, by id. A dead entry's slot is reused.
static REGISTRY: Mutex<Vec<Weak<WorldCell>>> = Mutex::new(Vec::new());

/// A refcounted world. Derefs to `&World`, which is all a frame ever needs.
#[derive(Clone)]
pub struct WorldHandle(Arc<WorldCell>);

impl Deref for WorldHandle {
    type Target = World;

    fn deref(&self) -> &World {
        // SAFETY: `&mut` is handed out only by `get_mut`, whose contract is
        // that no frame is running.
        unsafe { &*self.0.get() }
    }
}

impl WorldHandle {
    pub(crate) fn new(cell: Arc<WorldCell>) -> Self {
        Self(cell)
    }

    /// # Safety
    /// Between frames only: this aliases every `&World` a sweep hands out.
    pub unsafe fn get_mut(&self) -> &mut World {
        &mut *self.0.get()
    }

    /// Whether both handles name the same world.
    pub fn is(&self, other: &WorldHandle) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Whether this is the last handle: whoever owned the world has dropped
    /// it and what still names it is bookkeeping. Stop-play is that drop, so
    /// this is what tells the renderer to retire a world it drew.
    pub fn is_orphan(&self) -> bool {
        Arc::strong_count(&self.0) == 1
    }
}

/// Register a new, empty world.
///
/// Callable mid-frame — a fresh world is reachable through nothing else, so
/// filling it in touches no memory a sweep is reading.
pub fn new_world() -> WorldHandle {
    let mut reg = REGISTRY.lock();
    let id = reg
        .iter()
        .position(|w| w.strong_count() == 0)
        .unwrap_or(reg.len());
    let cell = Arc::new(SyncUnsafeCell::new(World::new(id as WorldId)));
    // SAFETY: sole owner — the registry holds only a `Weak`, and the handle
    // this returns has not been handed out yet.
    unsafe { &mut *cell.get() }.set_handle(Arc::downgrade(&cell));
    match reg.get_mut(id) {
        Some(slot) => *slot = Arc::downgrade(&cell),
        None => reg.push(Arc::downgrade(&cell)),
    }
    WorldHandle(cell)
}

/// The world `id`, if it still exists.
///
/// A dropped world's id is handed to the next world made, so an id kept across
/// frames can name a different world than the one it came from. Hold a
/// [`WorldHandle`] for anything that has to outlive a frame.
pub fn world(id: WorldId) -> Option<WorldHandle> {
    REGISTRY.lock().get(id as usize)?.upgrade().map(WorldHandle)
}

/// Every live world, as strong handles — so one dropped mid-frame outlives the
/// frame that is sweeping it.
pub fn live() -> Vec<WorldHandle> {
    REGISTRY
        .lock()
        .iter()
        .filter_map(Weak::upgrade)
        .map(WorldHandle)
        .collect()
}

/// Apply every queued structural change.
///
/// The frame boundary, and the only place `&mut World` exists. A builder
/// callback may queue more work, so this runs to a fixed point.
///
/// # Safety
/// No sweep may be running, and nothing may hold a `&World` from one.
pub unsafe fn apply_pending(worlds: &[WorldHandle]) {
    while worlds
        .iter()
        .fold(false, |any, w| w.get_mut().apply_pending() || any)
    {}
}

/// Advance every **simulating** world by `dt` seconds.
///
/// A world that does not simulate is not visited at all — no sweep, no
/// per-entity test. That is the whole point of the split.
pub fn sweep_all(worlds: &[WorldHandle], dt: f32) {
    for w in worlds.iter().filter(|w| w.simulating()) {
        w.sweep(dt);
    }
}
