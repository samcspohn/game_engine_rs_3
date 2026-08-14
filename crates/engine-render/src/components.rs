//! Renderer-side ECS components.
//!
//! These implement the core [`engine_core::Component`] trait but live in
//! `engine-render` because they bridge the ECS to GPU state. Today that's just
//! [`MeshRenderer`].

use std::path::Path;
use std::sync::OnceLock;

use parking_lot::Mutex;

use std::collections::HashMap;

use engine_core::asset::{self, MeshId};
use engine_core::material::{self, MaterialId};
use engine_core::reflect::Export;
use engine_core::{Component, Transform, WorldId};

use crate::gpu_renderers::{MATERIAL_INHERIT, NO_RENDERER};

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
/// exists — the component pushes `(world, transform_id, mesh_id,
/// material_word)` onto the record queue the renderer drains and scatters
/// into the `GPURenderers` buffer each frame; `set_material` on a live entity
/// pushes a fresh record over the same slot.
///
/// # Reflection
///
/// Both properties route through methods rather than the fields behind them:
/// a plain write would skip the registry refcount and the GPU record, which
/// is why the derive has method routing at all (ADR-0010 §3).
#[derive(Clone, Export)]
pub struct MeshRenderer {
    #[export(get = mesh_id, set = set_mesh)]
    mesh_id: MeshId,
    /// Explicit material override; `None` = inherit the mesh's authored
    /// material (scattered as [`MATERIAL_INHERIT`]).
    #[export(get = material, set = set_material)]
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
            .retain(material_id);
        self.material = Some(material_id);
        self
    }

    /// Swap the mesh this renderer draws on a live entity; the change lands
    /// via the next frame's scatter. Refcounts move with it, so the old mesh
    /// can be evicted once nothing draws it.
    pub fn set_mesh(&mut self, transform: &Transform, mesh_id: MeshId) {
        {
            let mut reg = asset::global().lock();
            reg.retain(mesh_id);
            reg.release(self.mesh_id);
        }
        self.mesh_id = mesh_id;
        self.publish(transform);
    }

    /// Swap this renderer's material on a live entity: `Some(id)` overrides,
    /// `None` restores inheritance of the mesh's authored material. Takes
    /// the entity's transform to locate the GPU record; the change lands via
    /// the next frame's scatter.
    pub fn set_material(&mut self, transform: &Transform, material: Option<MaterialId>) {
        {
            let mut reg = material::global().lock();
            if let Some(id) = material {
                reg.retain(id);
            }
            if let Some(old) = self.material {
                reg.release(old);
            }
        }
        self.material = material;
        self.publish(transform);
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

    /// Queue this renderer's current record. Every write goes through here,
    /// so there is one place that decides what the GPU is told.
    fn publish(&self, transform: &Transform) {
        push_spawn(
            transform.world(),
            transform.get_idx(),
            self.mesh_id.0,
            self.material_word(),
        );
    }
}

impl Component for MeshRenderer {
    // Pure data — no per-frame `update`. The renderer pulls its state via the
    // GPURenderers buffer, not a component hook.
    const HAS_UPDATE: bool = false;

    fn init(&mut self, transform: &Transform) {
        self.publish(transform);
    }

    /// The GPU reads `GPURenderers`, not this component, so a deleted entity
    /// keeps drawing at its dead slot unless the sentinel is scattered over
    /// it. The cull kernel already skips [`NO_RENDERER`] — no new GPU code.
    fn deinit(&mut self, transform: &Transform) {
        push_spawn(
            transform.world(),
            transform.get_idx(),
            NO_RENDERER,
            self.material_word(),
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spawn queue
// ─────────────────────────────────────────────────────────────────────────────

/// `(world, transform_id, mesh_id, material_word)` records queued by
/// [`MeshRenderer::init`] / [`MeshRenderer::set_material`], drained by the
/// renderer once per frame and scattered into the `GPURenderers` buffer.
/// Bounded by the per-frame spawn/swap rate, not the entity count.
///
/// Global (like [`engine_core::asset::global`]) because `Component::init` can
/// reach a static but not the renderer's `RenderContext`. `init` runs
/// single-threaded at `add_component` time, so contention is negligible.
static SPAWN_QUEUE: OnceLock<Mutex<Vec<[u32; 4]>>> = OnceLock::new();

fn spawn_queue() -> &'static Mutex<Vec<[u32; 4]>> {
    SPAWN_QUEUE.get_or_init(|| Mutex::new(Vec::new()))
}

/// Enqueue a renderer's record, tagged with the world its index belongs to.
fn push_spawn(world: WorldId, transform_id: u32, mesh_id: u32, material_word: u32) {
    spawn_queue()
        .lock()
        .push([world as u32, transform_id, mesh_id, material_word]);
}

/// Take every queued record, grouped by the world it belongs to. Called once
/// per frame by the renderer's ingest pass.
///
/// Grouped rather than filtered: each world scatters into its own
/// `GPURenderers` buffer (ADR-0011 step 3), so a record is no longer either
/// this world's or discarded.
pub(crate) fn drain_spawns() -> HashMap<WorldId, Vec<[u32; 3]>> {
    let mut out: HashMap<WorldId, Vec<[u32; 3]>> = HashMap::new();
    for r in std::mem::take(&mut *spawn_queue().lock()) {
        out.entry(r[0] as WorldId).or_default().push([r[1], r[2], r[3]]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::asset::MeshSlot;

    /// `SPAWN_QUEUE` is process-global and `drain_spawns` takes *everything*,
    /// so two tests asserting on their own records will steal each other's.
    /// Hold this for the whole of any test that pushes or drains.
    static QUEUE: Mutex<()> = Mutex::new(());

    #[test]
    fn new_requests_and_resolves_to_placeholder() {
        // Unique path so this test doesn't depend on other tests' requests.
        let r = MeshRenderer::new("components_test_unique_a.mesh");
        let slot = asset::global()
            .lock()
            .redirect_of(r.mesh_id());
        assert_eq!(slot, MeshSlot::PLACEHOLDER);
        assert_eq!(r.material(), None, "fresh renderers inherit");
    }

    #[test]
    fn spawn_queue_round_trips() {
        let _q = QUEUE.lock();
        // Drain any prior state, then push a known batch and drain it.
        let _ = drain_spawns().remove(&0).unwrap_or_default();
        push_spawn(0, 5, 7, MATERIAL_INHERIT);
        push_spawn(0, 9, 2, 3);
        push_spawn(1, 5, 4, 4);
        let drained = drain_spawns().remove(&0).unwrap_or_default();
        assert!(drained.contains(&[5, 7, MATERIAL_INHERIT]));
        assert!(drained.contains(&[9, 2, 3]));
        assert!(
            !drained.contains(&[5, 4, 4]),
            "another world's index would land on whatever slot shares it"
        );
        assert!(drain_spawns().remove(&0).unwrap_or_default().is_empty(), "queue must be empty after drain");
    }

    /// The case ADR-0010 says a naïve value model breaks on: the property is
    /// private, its setter refcounts, and the write has to reach the GPU.
    #[test]
    fn reflected_material_write_refcounts_and_scatters() {
        let _q = QUEUE.lock();
        use engine_core::reflect::{AssetRef, Value};
        use engine_core::transform::{TransformHierarchy, _Transform};

        let mut h = TransformHierarchy::new(0);
        let idx = h.create_transform(_Transform::default()).get_idx();
        let t = h.get_transform_unchecked(idx);

        let id = material::global()
            .lock()
            .create(engine_core::MaterialData::default());
        let before = material::global().lock().refcount_of(id);

        let mut r = MeshRenderer::new("components_test_unique_c.mesh");
        let _ = drain_spawns().remove(&0).unwrap_or_default();
        assert!(r.set("material", Value::Asset(Some(AssetRef::Material(id))), &t));

        assert_eq!(r.material(), Some(id));
        assert!(material::global().lock().refcount_of(id) > before, "retained");
        assert!(
            drain_spawns().remove(&0).unwrap_or_default().contains(&[idx, r.mesh_id().0, id.0]),
            "a field write would not have reached the GPU"
        );
        assert_eq!(r.get("material"), Some(Value::Asset(Some(AssetRef::Material(id)))));
    }

    /// Deleting an entity used to leave its mesh drawing at a dead slot.
    /// `remove_entity` calls `deinit` while the component is still there,
    /// which is the only moment it can say so.
    #[test]
    fn removal_scatters_the_sentinel() {
        let _q = QUEUE.lock();
        use engine_core::transform::_Transform;
        // No frame is running in a test, so `&mut` is sound.
        let h = engine_core::new_world();
        let world = unsafe { h.get_mut() };
        let top = world.new_entity(_Transform::default());
        let child = world.new_entity(_Transform {
            parent: Some(top.id),
            .._Transform::default()
        });
        world.add_component(child, MeshRenderer::new("components_test_unique_f.mesh"));
        let _ = drain_spawns().remove(&h.id()).unwrap_or_default();

        world.remove_entity(top);
        assert!(drain_spawns().remove(&h.id()).unwrap_or_default().contains(&[child.id, NO_RENDERER, MATERIAL_INHERIT]));
    }

    /// A renderer in a non-simulating world still reaches the GPU: edit mode
    /// stops behaviour, not drawing, or the viewport would go black.
    #[test]
    fn an_edited_world_still_publishes_its_renderer() {
        let _q = QUEUE.lock();
        use engine_core::transform::_Transform;
        let h = engine_core::new_world();
        // No frame is running in a test, so `&mut` is sound.
        let doc = unsafe { h.get_mut() };
        doc.set_simulating(false);
        let e = doc.new_entity(_Transform::default());
        let _ = drain_spawns().remove(&h.id()).unwrap_or_default();

        let r = MeshRenderer::new("components_test_unique_h.mesh");
        let mesh = r.mesh_id().0;
        doc.add_component(e, r);
        assert!(drain_spawns().remove(&h.id()).unwrap_or_default().contains(&[e.id, mesh, MATERIAL_INHERIT]));
    }

    /// A texture dragged onto the material slot: declined, and nothing moved.
    #[test]
    fn a_wrong_asset_kind_is_declined() {
        use engine_core::reflect::{AssetRef, Value};
        use engine_core::texture::TextureId;
        use engine_core::transform::{TransformHierarchy, _Transform};

        let mut h = TransformHierarchy::new(0);
        let idx = h.create_transform(_Transform::default()).get_idx();
        let t = h.get_transform_unchecked(idx);
        let mut r = MeshRenderer::new("components_test_unique_d.mesh");
        let wrong = Value::Asset(Some(AssetRef::Texture(TextureId(1))));
        assert!(!r.set("material", wrong, &t));
        assert_eq!(r.material(), None);
    }

    #[test]
    fn with_material_overrides_and_retains() {
        let id = material::global()
            .lock()
            .create(engine_core::MaterialData::default());
        let r = MeshRenderer::new("components_test_unique_b.mesh").with_material(id);
        assert_eq!(r.material(), Some(id));
        assert!(
            material::global().lock().refcount_of(id) >= 2,
            "with_material must retain"
        );
    }
}
