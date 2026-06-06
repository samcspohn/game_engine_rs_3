//! GPU mesh asset registry.
//!
//! The registry decouples a renderer's **stable, write-once `MeshId`** from
//! the **physical drawable slot** it currently resolves to. A `MeshRenderer`
//! component stores only a `MeshId`; a redirection map (`mesh_id → MeshSlot`)
//! points it at the placeholder while an asset loads, the real geometry once
//! it lands, or a distinct error mesh if the load fails. Because the renderer
//! never changes, load completion is a single map write — no renderer
//! patching, no tracking of which renderers are pending.
//!
//! # Layers
//!
//! | Type | Role |
//! |------|------|
//! | [`MeshCatalog`] | Pure-CPU bookkeeping: dedup cache (`path-hash → MeshId`), the redirect mirror, per-slot [`MeshTableEntry`]s, and the mega-buffer offset cursors. Unit-testable without a GPU. |
//! | [`MeshRegistry`] | Owns the device-local GPU buffers (mega vertex/index, mesh table, redirect) and drives staging→copy uploads, keeping the GPU mirrors in sync with the catalog. |
//!
//! # Identifier spaces
//!
//! Two distinct `u32` index spaces meet at the redirect map:
//!
//! * [`MeshId`] indexes `redirect` / `refcount` — allocated per *unique
//!   requested path* (deduped), stable for the renderer's lifetime.
//! * [`MeshSlot`] indexes the [`MeshTableEntry`] table, the per-frame indirect
//!   command array, and (via the table) the mega vertex/index buffers. Slots
//!   `0` and `1` are permanently reserved for the placeholder and error
//!   meshes.

mod catalog;
mod registry;

pub use catalog::{MeshCatalog, Placement};
pub use registry::MeshRegistry;

use std::hash::{Hash, Hasher};
use std::path::Path;

use engine_core::mesh::{Mesh, Vertex};
use glam::{Vec2, Vec3};
use vulkano::buffer::BufferContents;

// ─────────────────────────────────────────────────────────────────────────────
// Identifiers
// ─────────────────────────────────────────────────────────────────────────────

/// Stable, write-once handle held by a `MeshRenderer`. Allocated by the
/// registry on first request of a path (deduped) and indexes the redirect
/// map. Never changes once handed out — load completion only repoints the
/// redirect entry it refers to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MeshId(pub u32);

/// Physical drawable slot. Indexes the [`MeshTableEntry`] table, the
/// per-frame indirect command array, and (via the table) the mega
/// vertex/index buffers. Slots [`MeshSlot::PLACEHOLDER`] and
/// [`MeshSlot::ERROR`] are reserved.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MeshSlot(pub u32);

impl MeshSlot {
    /// Drawn for any `MeshId` whose asset is still loading.
    pub const PLACEHOLDER: MeshSlot = MeshSlot(0);
    /// Drawn for any `MeshId` whose load failed (distinct shape — see
    /// [`error_mesh`]).
    pub const ERROR: MeshSlot = MeshSlot(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// MeshTableEntry
// ─────────────────────────────────────────────────────────────────────────────

/// One per drawable slot. Mirrors the *static* fields of a
/// `VkDrawIndexedIndirectCommand` (the cull kernel writes the dynamic
/// `instance_count` / `first_instance` per frame) plus a local-space
/// bounding sphere for frustum culling.
///
/// Laid out for std430 as two 16-byte rows (32 bytes total, no internal
/// padding), so the GPU side can read it as a tightly-packed array.
#[repr(C)]
#[derive(Clone, Copy, Debug, BufferContents)]
pub struct MeshTableEntry {
    /// → `DrawIndexedIndirectCommand::index_count`.
    pub index_count: u32,
    /// → `DrawIndexedIndirectCommand::first_index` (offset into mega index).
    pub first_index: u32,
    /// → `DrawIndexedIndirectCommand::vertex_offset` (base into mega vertex).
    pub vertex_offset: i32,
    pub _pad0: u32,
    /// Local-space bounding-sphere center.
    pub bounds_center: [f32; 3],
    /// Local-space bounding-sphere radius.
    pub bounds_radius: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Hash a path to the `u64` key used by the dedup cache. Two requests for
/// the same path collapse onto one [`MeshId`] (and one GPU upload).
pub fn hash_path(path: &Path) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut h);
    h.finish()
}

/// Local-space bounding sphere `(center, radius)` of a mesh, derived from its
/// AABB. Empty meshes collapse to a zero-radius sphere at the origin.
pub(crate) fn bounding_sphere(mesh: &Mesh) -> ([f32; 3], f32) {
    match mesh.aabb() {
        Some(aabb) => {
            let c = aabb.center();
            ([c.x, c.y, c.z], aabb.half_extent().length())
        }
        None => ([0.0; 3], 0.0),
    }
}

/// Default placeholder mesh: a unit cube. Drawn while an asset loads.
pub fn placeholder_mesh() -> Mesh {
    engine_core::mesh::primitives::cube()
}

/// Default error mesh: a tetrahedron — a deliberately *distinct* silhouette
/// from the cube placeholder so a failed load is visually obvious in a scene
/// of normal (often cube-ish) geometry. Per the project's no-silent-fallback
/// rule, a failed load is surfaced loudly rather than hidden behind a
/// look-alike.
///
/// Flat per-face normals are computed via cross product and oriented away
/// from the origin, so lighting is correct regardless of vertex ordering.
pub fn error_mesh() -> Mesh {
    // Four corners of an origin-centered tetrahedron (apex up).
    let a = Vec3::new(0.0, 0.5, 0.0);
    let b = Vec3::new(-0.5, -0.5, 0.5);
    let c = Vec3::new(0.5, -0.5, 0.5);
    let d = Vec3::new(0.0, -0.5, -0.5);

    let faces = [[a, b, c], [a, c, d], [a, d, b], [b, d, c]];

    let mut vertices = Vec::with_capacity(12);
    let mut indices = Vec::with_capacity(12);
    for tri in faces {
        let [p0, p1, p2] = tri;
        let face_center = (p0 + p1 + p2) / 3.0;
        let mut normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        // Orient outward (away from the origin / mesh center).
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
