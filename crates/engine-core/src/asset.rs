//! Mesh asset registry — the GPU-agnostic source of truth.
//!
//! Lives in `engine-core` (not `engine-render`) so that **mesh data is shared
//! across subsystems**: the renderer uploads it to GPU mega-buffers, and a
//! future physics system can read the same retained `Arc<Mesh>` for collision
//! geometry. The registry owns no GPU resources; `engine-render`'s
//! `GpuMeshStore` mirrors it into device buffers.
//!
//! # The redirect indirection
//!
//! A renderer component stores only a stable, write-once [`MeshId`]. The
//! registry maps `mesh_id → MeshSlot` through a **redirect** table:
//!
//! * while the asset loads, the id resolves to [`MeshSlot::PLACEHOLDER`];
//! * once [`AssetRegistry::resolve`]d, to the real slot;
//! * on [`AssetRegistry::fail`], to [`MeshSlot::ERROR`].
//!
//! Load completion is therefore a single redirect write — no renderer record
//! is ever patched, and no per-renderer "pending" state is tracked.
//!
//! # Identifier spaces
//!
//! * [`MeshId`] indexes `redirect` / `refcount` — one per *unique requested
//!   path* (deduped), stable for a renderer's lifetime.
//! * [`MeshSlot`] indexes the retained mesh + bounds tables. Slots `0` and `1`
//!   are permanently reserved for the placeholder and error meshes.
//!
//! # Global access
//!
//! [`global`] returns a lazily-initialized `Mutex<AssetRegistry>` (mirroring
//! [`crate::util::thread_pool`]'s global), so a renderer component's
//! constructor can `request` a mesh and immediately receive a [`MeshId`]
//! without threading a context through the ECS.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex, OnceLock};

use glam::{Vec2, Vec3};

use crate::mesh::{Mesh, Vertex};

// ─────────────────────────────────────────────────────────────────────────────
// Identifiers
// ─────────────────────────────────────────────────────────────────────────────

/// Stable, write-once handle held by a renderer component. Allocated per
/// unique requested path (deduped) and indexes the redirect map.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MeshId(pub u32);

/// Physical drawable slot — indexes the retained mesh + bounds tables (and,
/// on the GPU side, the mega vertex/index buffers and per-frame indirect
/// commands). Slots [`MeshSlot::PLACEHOLDER`] and [`MeshSlot::ERROR`] are
/// reserved.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MeshSlot(pub u32);

impl MeshSlot {
    /// Resolved-to by any `MeshId` whose asset is still loading.
    pub const PLACEHOLDER: MeshSlot = MeshSlot(0);
    /// Resolved-to by any `MeshId` whose load failed.
    pub const ERROR: MeshSlot = MeshSlot(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bounds
// ─────────────────────────────────────────────────────────────────────────────

/// Local-space bounding sphere of a mesh, used for GPU frustum culling (and
/// available to CPU systems such as physics broad-phase).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshBounds {
    pub center: [f32; 3],
    pub radius: f32,
}

impl MeshBounds {
    /// Derive from a mesh's AABB. Empty meshes collapse to a zero-radius
    /// sphere at the origin.
    pub fn of(mesh: &Mesh) -> Self {
        match mesh.aabb() {
            Some(aabb) => MeshBounds {
                center: aabb.center().to_array(),
                radius: aabb.half_extent().length(),
            },
            None => MeshBounds {
                center: [0.0; 3],
                radius: 0.0,
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AssetRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// Retained per-slot data: the CPU mesh (shared with physics / re-upload) and
/// its bounds.
struct SlotData {
    mesh: Arc<Mesh>,
    bounds: MeshBounds,
}

/// GPU-agnostic mesh registry. See the module docs for the redirect model.
pub struct AssetRegistry {
    /// Dedup cache: path hash → already-allocated `MeshId`.
    by_hash: HashMap<u64, MeshId>,
    /// `mesh_id → drawable slot`. New ids default to
    /// [`MeshSlot::PLACEHOLDER`]; `resolve`/`fail` repoint them.
    redirect: Vec<MeshSlot>,
    /// Reference count per `MeshId`.
    refcount: Vec<u32>,
    /// Retained mesh + bounds per drawable slot.
    slots: Vec<SlotData>,
    /// `MeshId`s whose redirect entry changed since the last
    /// [`take_redirect_updates`](Self::take_redirect_updates) drain — consumed
    /// by the GPU mirror to patch its redirect buffer.
    dirty_redirect: Vec<MeshId>,
}

impl AssetRegistry {
    /// Build a registry with `placeholder` at slot 0 and `error` at slot 1.
    pub fn new(placeholder: Arc<Mesh>, error: Arc<Mesh>) -> Self {
        let mut reg = Self {
            by_hash: HashMap::new(),
            redirect: Vec::new(),
            refcount: Vec::new(),
            slots: Vec::new(),
            dirty_redirect: Vec::new(),
        };
        let ph = reg.alloc_slot(placeholder);
        let er = reg.alloc_slot(error);
        assert_eq!(ph, MeshSlot::PLACEHOLDER, "placeholder must be slot 0");
        assert_eq!(er, MeshSlot::ERROR, "error must be slot 1");
        reg
    }

    /// Build with the engine's default placeholder (cube) and error
    /// (tetrahedron) meshes. Used by [`global`].
    pub fn with_defaults() -> Self {
        Self::new(Arc::new(placeholder_mesh()), Arc::new(error_mesh()))
    }

    /// Allocate the next drawable slot, retaining the mesh and computing its
    /// bounds. Does **not** touch the redirect map (callers do).
    fn alloc_slot(&mut self, mesh: Arc<Mesh>) -> MeshSlot {
        let slot = MeshSlot(self.slots.len() as u32);
        let bounds = MeshBounds::of(&mesh);
        self.slots.push(SlotData { mesh, bounds });
        slot
    }

    /// Deduped request for `path`. On a cache miss, allocate a fresh
    /// [`MeshId`] pointing at [`MeshSlot::PLACEHOLDER`] and return
    /// `needs_load = true`. On a hit, bump the refcount and return the
    /// existing id with `needs_load = false`.
    pub fn request(&mut self, path: &Path) -> (MeshId, bool) {
        let hash = hash_path(path);
        if let Some(&id) = self.by_hash.get(&hash) {
            self.refcount[id.0 as usize] += 1;
            return (id, false);
        }
        let id = MeshId(self.redirect.len() as u32);
        self.redirect.push(MeshSlot::PLACEHOLDER);
        self.refcount.push(1);
        self.by_hash.insert(hash, id);
        (id, true)
    }

    /// A load finished: retain the mesh in a fresh slot and flip
    /// `redirect[id]` to it. Returns the new slot.
    pub fn resolve(&mut self, id: MeshId, mesh: Arc<Mesh>) -> MeshSlot {
        let slot = self.alloc_slot(mesh);
        self.redirect[id.0 as usize] = slot;
        self.dirty_redirect.push(id);
        slot
    }

    /// A load failed: point `redirect[id]` at the error slot.
    pub fn fail(&mut self, id: MeshId) {
        self.redirect[id.0 as usize] = MeshSlot::ERROR;
        self.dirty_redirect.push(id);
    }

    /// Drop one reference. Slot reclamation on zero is deferred (geometry is
    /// retained append-only for now).
    pub fn release(&mut self, id: MeshId) {
        let rc = &mut self.refcount[id.0 as usize];
        debug_assert!(*rc > 0, "release of MeshId({}) with zero refcount", id.0);
        *rc = rc.saturating_sub(1);
    }

    // ── Reads (physics / GPU mirror) ────────────────────────────────────

    /// Current drawable slot a `MeshId` resolves to.
    pub fn redirect_of(&self, id: MeshId) -> MeshSlot {
        self.redirect[id.0 as usize]
    }

    /// Retained mesh + bounds for a slot (clones the `Arc`).
    pub fn slot(&self, slot: MeshSlot) -> (Arc<Mesh>, MeshBounds) {
        let d = &self.slots[slot.0 as usize];
        (d.mesh.clone(), d.bounds)
    }

    /// Retained CPU mesh for a slot (e.g. for physics).
    pub fn mesh(&self, slot: MeshSlot) -> Arc<Mesh> {
        self.slots[slot.0 as usize].mesh.clone()
    }

    /// Number of drawable slots.
    pub fn slot_count(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Number of allocated `MeshId`s (→ redirect buffer sizing).
    pub fn mesh_id_count(&self) -> u32 {
        self.redirect.len() as u32
    }

    /// Reference count for a `MeshId`.
    pub fn refcount_of(&self, id: MeshId) -> u32 {
        self.refcount[id.0 as usize]
    }

    /// Drain the `(MeshId, MeshSlot)` redirect changes accumulated since the
    /// last call. Consumed by the GPU mirror to patch its redirect buffer.
    pub fn take_redirect_updates(&mut self) -> Vec<(MeshId, MeshSlot)> {
        self.dirty_redirect
            .drain(..)
            .map(|id| (id, self.redirect[id.0 as usize]))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global instance
// ─────────────────────────────────────────────────────────────────────────────

static REGISTRY: OnceLock<Mutex<AssetRegistry>> = OnceLock::new();

/// The process-wide mesh registry, lazily initialized with the default
/// placeholder/error meshes on first access. Component constructors call
/// `global().lock().request(...)`; the async loader calls `resolve`/`fail`;
/// the renderer's `GpuMeshStore` syncs from it each frame.
pub fn global() -> &'static Mutex<AssetRegistry> {
    REGISTRY.get_or_init(|| Mutex::new(AssetRegistry::with_defaults()))
}

// ─────────────────────────────────────────────────────────────────────────────
// Async loader
// ─────────────────────────────────────────────────────────────────────────────

/// A queued mesh load: decode `path` off-thread, then `resolve`/`fail` the id.
struct LoadRequest {
    mesh_id: MeshId,
    path: PathBuf,
}

static LOADER: OnceLock<mpsc::Sender<LoadRequest>> = OnceLock::new();

/// Lazily spawn the background loader thread and return its request channel.
///
/// The loader runs on a **dedicated** thread doing blocking file IO + decode —
/// kept off the fork-join [`crate::util::thread_pool`], which is tuned for
/// short, hot, non-blocking per-frame work and panics on nested parallelism.
fn loader() -> &'static mpsc::Sender<LoadRequest> {
    LOADER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<LoadRequest>();
        std::thread::Builder::new()
            .name("asset-loader".to_string())
            .spawn(move || loader_loop(rx))
            .expect("spawn asset-loader thread");
        tx
    })
}

fn loader_loop(rx: mpsc::Receiver<LoadRequest>) {
    while let Ok(req) = rx.recv() {
        match decode_mesh(&req.path) {
            Ok(mesh) => {
                global()
                    .lock()
                    .expect("asset registry mutex poisoned")
                    .resolve(req.mesh_id, Arc::new(mesh));
            }
            Err(e) => {
                // Per the project's no-silent-fallback rule, surface the
                // failure loudly and swap to the visible error mesh.
                eprintln!("asset load failed for {}: {e}", req.path.display());
                global()
                    .lock()
                    .expect("asset registry mutex poisoned")
                    .fail(req.mesh_id);
            }
        }
    }
}

/// Queue an asynchronous load of `path` for `mesh_id`. On completion the loader
/// thread calls [`AssetRegistry::resolve`] (success, flipping the redirect to
/// the real mesh) or [`AssetRegistry::fail`] (→ error mesh). Call once per id —
/// the renderer component does this only when `request` reports a new path.
pub fn request_load(mesh_id: MeshId, path: impl Into<PathBuf>) {
    loader()
        .send(LoadRequest {
            mesh_id,
            path: path.into(),
        })
        .expect("asset-loader thread has hung up");
}

/// Decode a mesh file into a CPU [`Mesh`]. Dispatches on file extension.
fn decode_mesh(path: &Path) -> Result<Mesh, String> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("obj") => decode_obj(path),
        other => Err(format!("unsupported mesh format: {other:?}")),
    }
}

/// Decode a Wavefront OBJ into a single [`Mesh`]. All sub-objects are merged
/// (materials / sub-meshes are ignored for now). Quads / n-gons are
/// triangulated and vertices deduplicated via `tobj`'s single-index option.
fn decode_obj(path: &Path) -> Result<Mesh, String> {
    let (models, _materials) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .map_err(|e| format!("OBJ parse error: {e}"))?;

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for model in &models {
        let m = &model.mesh;
        let base = vertices.len() as u32;
        let vcount = m.positions.len() / 3;
        for i in 0..vcount {
            let position = Vec3::new(
                m.positions[3 * i],
                m.positions[3 * i + 1],
                m.positions[3 * i + 2],
            );
            let normal = if m.normals.len() >= 3 * i + 3 {
                Vec3::new(m.normals[3 * i], m.normals[3 * i + 1], m.normals[3 * i + 2])
            } else {
                Vec3::ZERO
            };
            let uv = if m.texcoords.len() >= 2 * i + 2 {
                Vec2::new(m.texcoords[2 * i], m.texcoords[2 * i + 1])
            } else {
                Vec2::ZERO
            };
            vertices.push(Vertex {
                position,
                normal,
                uv,
            });
        }
        indices.extend(m.indices.iter().map(|&idx| base + idx));
    }

    if vertices.is_empty() {
        return Err(format!("OBJ {} contained no geometry", path.display()));
    }
    // std::thread::sleep(std::time::Duration::from_millis(1000)); // Simulate a slow load for testing.
    Ok(Mesh::new(vertices, indices))
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers + default meshes
// ─────────────────────────────────────────────────────────────────────────────

/// Hash a path to the `u64` dedup-cache key.
fn hash_path(path: &Path) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut h);
    h.finish()
}

/// Default placeholder mesh: a unit cube, drawn while an asset loads.
pub fn placeholder_mesh() -> Mesh {
    crate::mesh::primitives::cube()
}

/// Default error mesh: a tetrahedron — a deliberately *distinct* silhouette
/// from the cube placeholder so a failed load is visually obvious. Per the
/// project's no-silent-fallback rule, a failed load is surfaced loudly.
///
/// Flat per-face normals are computed via cross product and oriented away from
/// the origin, so lighting is correct regardless of vertex ordering.
pub fn error_mesh() -> Mesh {
    let a = Vec3::new(0.0, 0.5, 0.0);
    let b = Vec3::new(-0.5, -0.5, 0.5);
    let c = Vec3::new(0.5, -0.5, 0.5);
    let d = Vec3::new(0.0, -0.5, -0.5);

    let faces = [[a, b, c], [a, c, d], [a, d, b], [b, d, c]];

    let mut vertices = Vec::with_capacity(12);
    let mut indices = Vec::with_capacity(12);
    for [p0, p1, p2] in faces {
        let face_center = (p0 + p1 + p2) / 3.0;
        let mut normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        if normal.dot(face_center) < 0.0 {
            normal = -normal;
        }
        let base = vertices.len() as u32;
        for p in [p0, p1, p2] {
            vertices.push(Vertex {
                position: p,
                normal,
                uv: Vec2::ZERO,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    Mesh::new(vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> AssetRegistry {
        AssetRegistry::new(Arc::new(placeholder_mesh()), Arc::new(error_mesh()))
    }

    #[test]
    fn reserved_slots_take_zero_and_one() {
        let reg = fresh();
        assert_eq!(reg.slot_count(), 2);
        assert_eq!(reg.mesh_id_count(), 0);
    }

    #[test]
    fn request_dedups_and_refcounts() {
        let mut reg = fresh();
        let (a, load_a) = reg.request(Path::new("a.mesh"));
        assert!(load_a, "first request of a path must trigger a load");
        assert_eq!(reg.redirect_of(a), MeshSlot::PLACEHOLDER);

        let (a2, load_a2) = reg.request(Path::new("a.mesh"));
        assert_eq!(a, a2, "same path must dedup to the same MeshId");
        assert!(!load_a2, "repeat request must not trigger a second load");
        assert_eq!(reg.refcount_of(a), 2);

        let (b, load_b) = reg.request(Path::new("b.mesh"));
        assert!(load_b);
        assert_ne!(a, b, "distinct paths must get distinct MeshIds");
    }

    #[test]
    fn resolve_retains_mesh_and_flips_redirect() {
        let mut reg = fresh();
        let (a, _) = reg.request(Path::new("a.mesh"));
        assert_eq!(reg.redirect_of(a), MeshSlot::PLACEHOLDER);

        let mesh = Arc::new(crate::mesh::primitives::cube());
        let slot = reg.resolve(a, mesh.clone());
        assert_eq!(slot, MeshSlot(2), "first real mesh lands in slot 2");
        assert_eq!(reg.redirect_of(a), MeshSlot(2));

        // Mesh retained and queryable (for physics / re-upload).
        let (retained, bounds) = reg.slot(slot);
        assert_eq!(retained.vertices.len(), mesh.vertices.len());
        assert!((bounds.radius - 3_f32.sqrt() * 0.5).abs() < 1e-5);

        // The redirect change is reported exactly once.
        let updates = reg.take_redirect_updates();
        assert_eq!(updates, vec![(a, MeshSlot(2))]);
        assert!(reg.take_redirect_updates().is_empty());
    }

    #[test]
    fn fail_points_redirect_at_error_slot() {
        let mut reg = fresh();
        let (a, _) = reg.request(Path::new("missing.mesh"));
        reg.fail(a);
        assert_eq!(reg.redirect_of(a), MeshSlot::ERROR);
        // Failing allocates no new slot.
        assert_eq!(reg.slot_count(), 2);
        assert_eq!(reg.take_redirect_updates(), vec![(a, MeshSlot::ERROR)]);
    }
}
