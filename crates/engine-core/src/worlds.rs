//! The worlds the running frame is sweeping — ADR-0011 §3.
//!
//! Ambient rather than a parameter, beside `asset::global()` and `ui()`:
//! reaching across worlds is the editor's privilege, so `engine-editor-api`
//! re-exports this and `engine` does not.
//!
//! Read without a lock. `Chrome::update` runs *inside* the sweep, so an outer
//! lock taken by both would be a re-entrant acquire on a `parking_lot::Mutex`
//! and deadlock the editor against the loop running it. The list is instead
//! stable for the duration of a frame; creating or dropping a world is a
//! structural change that queues and drains between frames.

use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use crate::component::World;
use crate::transform::WorldId;

static LIVE: AtomicPtr<World> = AtomicPtr::new(ptr::null_mut());
/// Written before `LIVE` goes non-null and never while it is, so a reader
/// that sees a pointer sees the length that came with it.
static LEN: AtomicUsize = AtomicUsize::new(0);

/// Unpublishes the list when the sweep returns.
pub(crate) struct Live;

impl Drop for Live {
    fn drop(&mut self) {
        LIVE.store(ptr::null_mut(), Ordering::Release);
    }
}

/// Publish `worlds` for as long as the guard lives.
pub(crate) fn publish(worlds: &[World]) -> Live {
    LEN.store(worlds.len(), Ordering::Relaxed);
    LIVE.store(worlds.as_ptr() as *mut World, Ordering::Release);
    Live
}

/// The worlds the frame is sweeping; empty outside one.
///
/// The `'static` is the frame, not the process — the same aliasing contract
/// as `TransformHierarchy::positions_raw`, and the callers run in the sweep.
pub fn all() -> &'static [World] {
    let p = LIVE.load(Ordering::Acquire);
    match p.is_null() {
        // SAFETY: see the contract above.
        false => unsafe { std::slice::from_raw_parts(p, LEN.load(Ordering::Relaxed)) },
        true => &[],
    }
}

/// The world `id`, or `None` outside a sweep and for an id nobody made.
pub fn world(id: WorldId) -> Option<&'static World> {
    all().get(id as usize)
}

/// How many worlds the frame is sweeping; zero outside one.
pub fn count() -> usize {
    all().len()
}
