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
//! [`crate::util::parallel`]'s global pool), so a renderer component's
//! constructor can `request` a mesh and immediately receive a [`MeshId`]
//! without threading a context through the ECS. Loads decode on that pool
//! as background tasks (see [`request_load`]) rather than on a dedicated
//! loader thread.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use glam::{Vec2, Vec3};

use crate::mesh::{Mesh, Vertex};
use crate::texture::{self, TextureId};

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

/// Retained per-slot data: the CPU mesh (shared with physics / re-upload),
/// its bounds, and its base-color texture (if the source material has one —
/// resolved against the [`texture`] registry's own redirect model).
struct SlotData {
    mesh: Arc<Mesh>,
    bounds: MeshBounds,
    texture: Option<TextureId>,
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
        self.alloc_slot_textured(mesh, None)
    }

    fn alloc_slot_textured(&mut self, mesh: Arc<Mesh>, texture: Option<TextureId>) -> MeshSlot {
        let slot = MeshSlot(self.slots.len() as u32);
        let bounds = MeshBounds::of(&mesh);
        self.slots.push(SlotData {
            mesh,
            bounds,
            texture,
        });
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
        self.resolve_textured(id, mesh, None)
    }

    /// [`resolve`](Self::resolve) with the mesh's base-color texture (the
    /// loader requested it from the [`texture`] registry during decode).
    pub fn resolve_textured(
        &mut self,
        id: MeshId,
        mesh: Arc<Mesh>,
        texture: Option<TextureId>,
    ) -> MeshSlot {
        let slot = self.alloc_slot_textured(mesh, texture);
        self.redirect[id.0 as usize] = slot;
        self.dirty_redirect.push(id);
        slot
    }

    /// A load failed: point `redirect[id]` at the error slot.
    pub fn fail(&mut self, id: MeshId) {
        self.redirect[id.0 as usize] = MeshSlot::ERROR;
        self.dirty_redirect.push(id);
    }

    /// Add one reference to an already-allocated `MeshId`. Used when a
    /// handle is duplicated without going through [`request`](Self::request)
    /// — e.g. `MeshRenderer::from_id` cloning a subscene template's proxy.
    pub fn retain(&mut self, id: MeshId) {
        self.refcount[id.0 as usize] += 1;
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

    /// Base-color texture of a slot, if its source material had one.
    pub fn slot_texture(&self, slot: MeshSlot) -> Option<TextureId> {
        self.slots[slot.0 as usize].texture
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

    /// Per-drawable-slot total instance count: for each slot, the sum of the
    /// refcounts of every `MeshId` currently redirecting to it. Returned
    /// length is [`slot_count`](Self::slot_count). Used by the renderer to
    /// prefix-sum the per-slot `first_instance` bases (the cull pass writes
    /// each visible instance into its slot's region). `O(#mesh_ids)`.
    pub fn slot_instance_totals(&self) -> Vec<u32> {
        let mut totals = vec![0u32; self.slots.len()];
        for (mesh_id, slot) in self.redirect.iter().enumerate() {
            totals[slot.0 as usize] += self.refcount[mesh_id];
        }
        totals
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

/// Loads requested before the global pool exists. Games build their scene
/// (constructing `MeshRenderer`s → [`request_load`], or requesting GLB
/// subscenes) *before* handing it to the engine, which only then
/// initialises the pool — so early requests park here as deferred spawn
/// closures until [`flush_pending_loads`] runs them.
static PENDING_LOADS: Mutex<Vec<Box<dyn FnOnce() + Send>>> = Mutex::new(Vec::new());

/// Run `f` as a pool background task, or defer it until the pool exists.
/// The shared deferral path for every asset-load flavour (OBJ meshes,
/// GLB scene templates).
pub(crate) fn spawn_when_pool_ready(f: impl FnOnce() + Send + 'static) {
    // Take the pending lock *around* the initialized check so a request
    // can't slip between `flush_pending_loads` draining and the pool
    // becoming visible: while the lock is held the flush can't drain, and
    // once the pool reads as initialised we spawn directly.
    let mut pending = PENDING_LOADS.lock().expect("pending-load mutex poisoned");
    if crate::util::parallel::global::is_initialized() {
        drop(pending);
        crate::util::parallel::global::spawn_background(f);
    } else {
        pending.push(Box::new(f));
    }
}

/// Queue an asynchronous load of `path` for `mesh_id`. On completion the load
/// task calls [`AssetRegistry::resolve`] (success, flipping the redirect to
/// the real mesh) or [`AssetRegistry::fail`] (→ error mesh). Call once per id —
/// the renderer component does this only when `request` reports a new path.
///
/// Decoding runs as a [`crate::util::parallel::global::spawn_background`]
/// task on the engine pool: the load occupies one worker for its duration,
/// and (on the mypool backend) that worker leaves the pool's availability
/// mask, so per-frame `parallel_for` dispatches automatically partition
/// across the threads that remain free instead of stalling behind the
/// decode. Independent loads run concurrently on however many workers are
/// idle; when none are, queued loads start as workers free up.
///
/// Requests made before the pool is initialised (scene construction runs
/// before engine init) are deferred and spawned by
/// [`flush_pending_loads`] once the pool is up.
pub fn request_load(mesh_id: MeshId, path: impl Into<PathBuf>) {
    let path: PathBuf = path.into();
    spawn_when_pool_ready(move || match decode_mesh(&path) {
        Ok((mesh, texture)) => {
            global()
                .lock()
                .expect("asset registry mutex poisoned")
                .resolve_textured(mesh_id, Arc::new(mesh), texture);
        }
        Err(e) => {
            // Per the project's no-silent-fallback rule, surface the
            // failure loudly and swap to the visible error mesh.
            eprintln!("asset load failed for {}: {e}", path.display());
            global()
                .lock()
                .expect("asset registry mutex poisoned")
                .fail(mesh_id);
        }
    });
}

/// Spawn every load deferred by [`request_load`] (or a scene-template
/// request) before the pool existed. Called once by engine init immediately
/// after the global pool is built; panics if the pool still isn't
/// initialised (init-order bug — no silent re-deferral).
pub fn flush_pending_loads() {
    assert!(
        crate::util::parallel::global::is_initialized(),
        "flush_pending_loads called before parallel::global::init"
    );
    let pending = std::mem::take(&mut *PENDING_LOADS.lock().expect("pending-load mutex poisoned"));
    for spawn in pending {
        crate::util::parallel::global::spawn_background(spawn);
    }
}

/// Decode a mesh file into a CPU [`Mesh`] plus its base-color texture (if
/// the source material references one). Dispatches on file extension.
fn decode_mesh(path: &Path) -> Result<(Mesh, Option<TextureId>), String> {
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
/// (sub-meshes are ignored for now). Quads / n-gons are triangulated and
/// vertices deduplicated via `tobj`'s single-index option.
///
/// The first `map_Kd` (diffuse texture) among the models' materials becomes
/// the mesh's base-color texture, requested from the [`texture`] registry
/// (path resolved relative to the OBJ's directory). Because sub-objects
/// merge into one mesh, a second *distinct* diffuse map cannot be honoured —
/// reported loudly, per the no-silent-fallback rule.
fn decode_obj(path: &Path) -> Result<(Mesh, Option<TextureId>), String> {
    let (models, materials) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .map_err(|e| format!("OBJ parse error: {e}"))?;

    // Pick the merged mesh's base-color texture from the MTL materials.
    // A failed MTL load is loud but non-fatal: the geometry still decodes.
    let mut diffuse_map: Option<String> = None;
    match materials {
        Ok(mats) => {
            for model in &models {
                let Some(mid) = model.mesh.material_id else { continue };
                let Some(map) = mats.get(mid).and_then(|m| m.diffuse_texture.clone()) else {
                    continue;
                };
                match &diffuse_map {
                    None => diffuse_map = Some(map),
                    Some(first) if *first != map => eprintln!(
                        "OBJ {}: multiple diffuse textures ({first:?}, {map:?}) — \
                         sub-objects merge into one mesh, using the first",
                        path.display()
                    ),
                    _ => {}
                }
            }
        }
        Err(e) => eprintln!("OBJ {}: MTL load failed ({e}) — no textures", path.display()),
    }
    let texture = diffuse_map.map(|map| {
        let tex_path = path.parent().unwrap_or(Path::new("")).join(map);
        let (texture_id, needs_load) = texture::global()
            .lock()
            .expect("texture registry mutex poisoned")
            .request(&tex_path);
        if needs_load {
            texture::request_load(texture_id, tex_path);
        }
        texture_id
    });

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
    Ok((Mesh::new(vertices, indices), texture))
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

    /// Poll the *global* registry until `id` redirects away from the
    /// placeholder (background loads complete asynchronously).
    fn wait_for_redirect(id: MeshId) -> MeshSlot {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let slot = global()
                .lock()
                .expect("asset registry mutex poisoned")
                .redirect_of(id);
            if slot != MeshSlot::PLACEHOLDER {
                return slot;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "background load of MeshId({}) never resolved",
                id.0
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// End-to-end: `request_load` decodes on a pool background task and
    /// resolves the redirect to a real slot.
    #[test]
    fn request_load_resolves_via_pool_background_task() {
        let _ = crate::util::parallel::global::init(
            crate::util::parallel::BackendKind::MyPool,
            4,
        );
        let path = std::env::temp_dir().join(format!(
            "engine_asset_test_{}_ok.obj",
            std::process::id()
        ));
        std::fs::write(&path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            .expect("write test obj");

        let (id, needs_load) = global()
            .lock()
            .expect("asset registry mutex poisoned")
            .request(&path);
        assert!(needs_load);
        request_load(id, &path);

        let slot = wait_for_redirect(id);
        assert_ne!(slot, MeshSlot::ERROR, "valid OBJ must not fail");
        let (mesh, _) = global()
            .lock()
            .expect("asset registry mutex poisoned")
            .slot(slot);
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.indices.len(), 3);
        std::fs::remove_file(&path).ok();
    }

    /// A missing file redirects to the error slot via the same path.
    #[test]
    fn request_load_missing_file_redirects_to_error() {
        let _ = crate::util::parallel::global::init(
            crate::util::parallel::BackendKind::MyPool,
            4,
        );
        let path = std::env::temp_dir().join(format!(
            "engine_asset_test_{}_missing.obj",
            std::process::id()
        ));

        let (id, needs_load) = global()
            .lock()
            .expect("asset registry mutex poisoned")
            .request(&path);
        assert!(needs_load);
        request_load(id, &path);
        assert_eq!(wait_for_redirect(id), MeshSlot::ERROR);
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
