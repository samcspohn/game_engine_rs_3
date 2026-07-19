//! Renderer-side ECS components.
//!
//! These implement the core [`engine_core::Component`] trait but live in
//! `engine-render` because they bridge the ECS to GPU state. Today that's just
//! [`MeshRenderer`].

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use engine_core::asset::{self, MeshId};
use engine_core::material::{self, MaterialId};
use engine_core::{Component, Transform};

use crate::gpu_renderers::MATERIAL_INHERIT;

/// A drawable mesh attached to an entity.
///
/// The component stores only a stable [`MeshId`] — never a path. Its
/// constructor resolves the path against the global [`asset`] registry
/// (deduped), so the returned id points at the placeholder until an async load
/// completes (or the error mesh if it fails). The renderer's `MeshId` never
/// changes; the registry's redirect map handles the placeholder→real swap.
///
/// # Materials
///
/// By default a renderer **inherits** the mesh's authored material (whatever
/// the OBJ MTL / glTF primitive assigned, or the engine default) — resolved
/// GPU-side, so it tracks the mesh through its placeholder→real swap with no
/// component involvement. [`with_material`](Self::with_material) /
/// [`set_material`](Self::set_material) override it with an explicit
/// [`MaterialId`]; swapping back to [`None`] restores inheritance.
///
/// At [`Component::init`] time — once the entity (hence its `transform_id`)
/// exists — the component pushes `(transform_id, mesh_id, material_word)`
/// onto the record queue the renderer drains and scatters into the
/// `GPURenderers` buffer each frame; `set_material` on a live entity pushes
/// a fresh record over the same slot.
#[derive(Clone)]
pub struct MeshRenderer {
    mesh_id: MeshId,
    /// Explicit material override; `None` = inherit the mesh's authored
    /// material (scattered as [`MATERIAL_INHERIT`]).
    material: Option<MaterialId>,
}

impl MeshRenderer {
    /// Request `path` from the global asset registry and store the resulting
    /// (deduped) [`MeshId`]. The mesh resolves to the placeholder until a
    /// loader resolves it. The renderer inherits the mesh's authored
    /// material.
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
        Self {
            mesh_id,
            material: None,
        }
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
        Self {
            mesh_id,
            material: None,
        }
    }

    /// Builder-style explicit material override (bumps the material
    /// refcount). Apply before the component is added to an entity.
    pub fn with_material(mut self, material_id: MaterialId) -> Self {
        material::global()
            .lock()
            .expect("material registry mutex poisoned")
            .retain(material_id);
        self.material = Some(material_id);
        self
    }

    /// Swap this renderer's material on a live entity: `Some(id)` overrides,
    /// `None` restores inheritance of the mesh's authored material. Takes
    /// the entity's transform to locate the GPU record; the change lands via
    /// the next frame's scatter.
    pub fn set_material(&mut self, transform: &Transform, material: Option<MaterialId>) {
        {
            let mut reg = material::global().lock().expect("material registry mutex poisoned");
            if let Some(id) = material {
                reg.retain(id);
            }
            if let Some(old) = self.material {
                reg.release(old);
            }
        }
        self.material = material;
        push_spawn(transform.get_idx(), self.mesh_id.0, self.material_word());
    }

    /// The mesh this renderer draws (via the registry's redirect map).
    pub fn mesh_id(&self) -> MeshId {
        self.mesh_id
    }

    /// The explicit material override, if any (`None` = inheriting).
    pub fn material(&self) -> Option<MaterialId> {
        self.material
    }

    /// The material word scattered into the GPU record.
    fn material_word(&self) -> u32 {
        self.material.map_or(MATERIAL_INHERIT, |m| m.0)
    }
}

impl Component for MeshRenderer {
    // Pure data — no per-frame `update`. The renderer pulls its state via the
    // GPURenderers buffer, not a component hook.
    const HAS_UPDATE: bool = false;

    fn init(&mut self, transform: &Transform) {
        push_spawn(transform.get_idx(), self.mesh_id.0, self.material_word());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spawn queue
// ─────────────────────────────────────────────────────────────────────────────

/// `(transform_id, mesh_id, material_word)` records queued by
/// [`MeshRenderer::init`] / [`MeshRenderer::set_material`], drained by the
/// renderer once per frame and scattered into the `GPURenderers` buffer.
/// Bounded by the per-frame spawn/swap rate, not the entity count.
///
/// Global (like [`engine_core::asset::global`]) because `Component::init` can
/// reach a static but not the renderer's `RenderContext`. `init` runs
/// single-threaded at `add_component` time, so contention is negligible.
static SPAWN_QUEUE: OnceLock<Mutex<Vec<[u32; 3]>>> = OnceLock::new();

fn spawn_queue() -> &'static Mutex<Vec<[u32; 3]>> {
    SPAWN_QUEUE.get_or_init(|| Mutex::new(Vec::new()))
}

/// Enqueue a renderer's `(transform_id, mesh_id, material_word)` record.
fn push_spawn(transform_id: u32, mesh_id: u32, material_word: u32) {
    spawn_queue()
        .lock()
        .expect("spawn queue mutex poisoned")
        .push([transform_id, mesh_id, material_word]);
}

/// Take all queued records, leaving the queue empty. Called once per frame by
/// the renderer's ingest pass.
pub(crate) fn drain_spawns() -> Vec<[u32; 3]> {
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
        assert_eq!(r.material(), None, "fresh renderers inherit");
    }

    #[test]
    fn spawn_queue_round_trips() {
        // Drain any prior state, then push a known batch and drain it.
        let _ = drain_spawns();
        push_spawn(5, 7, MATERIAL_INHERIT);
        push_spawn(9, 2, 3);
        let drained = drain_spawns();
        assert!(drained.contains(&[5, 7, MATERIAL_INHERIT]));
        assert!(drained.contains(&[9, 2, 3]));
        assert!(drain_spawns().is_empty(), "queue must be empty after drain");
    }

    #[test]
    fn with_material_overrides_and_retains() {
        let id = material::global()
            .lock()
            .expect("material registry")
            .create(engine_core::MaterialData::default());
        let r = MeshRenderer::new("components_test_unique_b.mesh").with_material(id);
        assert_eq!(r.material(), Some(id));
        assert!(
            material::global().lock().expect("registry").refcount_of(id) >= 2,
            "with_material must retain"
        );
    }
}
