//! Renderer-side ECS components.
//!
//! These implement the core [`engine_core::Component`] trait but live in
//! `engine-render` because they bridge the ECS to GPU state. Today that's just
//! [`MeshRenderer`].

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use engine_core::asset::{self, MeshId};
use engine_core::{Component, Transform};

/// A drawable mesh attached to an entity.
///
/// The component stores only a stable [`MeshId`] — never a path. Its
/// constructor resolves the path against the global [`asset`] registry
/// (deduped), so the returned id points at the placeholder until an async load
/// completes (or the error mesh if it fails). The renderer's `MeshId` never
/// changes; the registry's redirect map handles the placeholder→real swap.
///
/// At [`Component::init`] time — once the entity (hence its `transform_id`)
/// exists — the component pushes `(transform_id, mesh_id)` onto the spawn queue
/// the renderer drains and scatters into the `GPURenderers` buffer each frame.
#[derive(Clone)]
pub struct MeshRenderer {
    mesh_id: MeshId,
}

impl MeshRenderer {
    /// Request `path` from the global asset registry and store the resulting
    /// (deduped) [`MeshId`]. The mesh resolves to the placeholder until a
    /// loader resolves it.
    pub fn new(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        let (mesh_id, needs_load) = asset::global()
            .lock()
            .expect("asset registry mutex poisoned")
            .request(path);
        if needs_load {
            // First request of this path — kick the async load. The mesh draws
            // as the placeholder until the loader resolves it (or the error
            // mesh if the load fails).
            asset::request_load(mesh_id, path);
        }
        Self { mesh_id }
    }

    /// Build a renderer directly from an existing [`MeshId`] — no path
    /// lookup. Used when instantiating a subscene template (each template
    /// proxy already minted its id) or wherever a handle is shared without
    /// re-requesting the path. Bumps the registry refcount, so this
    /// renderer counts toward the id's instance total like a `new` would.
    pub fn from_id(mesh_id: MeshId) -> Self {
        asset::global()
            .lock()
            .expect("asset registry mutex poisoned")
            .retain(mesh_id);
        Self { mesh_id }
    }

    /// The mesh this renderer draws (via the registry's redirect map).
    pub fn mesh_id(&self) -> MeshId {
        self.mesh_id
    }
}

impl Component for MeshRenderer {
    // Pure data — no per-frame `update`. The renderer pulls its state via the
    // GPURenderers buffer, not a component hook.
    const HAS_UPDATE: bool = false;

    fn init(&mut self, transform: &Transform) {
        push_spawn(transform.get_idx(), self.mesh_id.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spawn queue
// ─────────────────────────────────────────────────────────────────────────────

/// `(transform_id, mesh_id)` pairs queued by [`MeshRenderer::init`], drained
/// by the renderer once per frame and scattered into the `GPURenderers`
/// buffer. Bounded by the per-frame spawn rate, not the entity count.
///
/// Global (like [`engine_core::asset::global`]) because `Component::init` can
/// reach a static but not the renderer's `RenderContext`. `init` runs
/// single-threaded at `add_component` time, so contention is negligible.
static SPAWN_QUEUE: OnceLock<Mutex<Vec<[u32; 2]>>> = OnceLock::new();

fn spawn_queue() -> &'static Mutex<Vec<[u32; 2]>> {
    SPAWN_QUEUE.get_or_init(|| Mutex::new(Vec::new()))
}

/// Enqueue a newly-attached renderer's `(transform_id, mesh_id)` pair.
fn push_spawn(transform_id: u32, mesh_id: u32) {
    spawn_queue()
        .lock()
        .expect("spawn queue mutex poisoned")
        .push([transform_id, mesh_id]);
}

/// Take all queued spawns, leaving the queue empty. Called once per frame by
/// the renderer's ingest pass.
pub(crate) fn drain_spawns() -> Vec<[u32; 2]> {
    std::mem::take(&mut *spawn_queue().lock().expect("spawn queue mutex poisoned"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::asset::MeshSlot;

    #[test]
    fn new_requests_and_resolves_to_placeholder() {
        // Unique path so this test doesn't depend on other tests' requests.
        let r = MeshRenderer::new("components_test_unique_a.mesh");
        let slot = asset::global()
            .lock()
            .expect("registry")
            .redirect_of(r.mesh_id());
        assert_eq!(slot, MeshSlot::PLACEHOLDER);
    }

    #[test]
    fn spawn_queue_round_trips() {
        // Drain any prior state, then push a known batch and drain it.
        let _ = drain_spawns();
        push_spawn(5, 7);
        push_spawn(9, 2);
        let drained = drain_spawns();
        assert!(drained.contains(&[5, 7]));
        assert!(drained.contains(&[9, 2]));
        assert!(drain_spawns().is_empty(), "queue must be empty after drain");
    }
}
