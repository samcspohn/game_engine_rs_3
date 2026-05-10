//! Procedural generators for common primitive meshes.
//!
//! Every function returns a [`super::Mesh`] with positions, normals, and UVs
//! fully populated and indices laid out as CCW triangle lists (right-handed,
//! Y-up).  All primitives are unit-sized and centred at the origin so they
//! can be scaled uniformly via the [`TransformHierarchy`].
//!
//! [`TransformHierarchy`]: crate::transform::TransformHierarchy

use super::{Mesh, Vertex};

// ---------------------------------------------------------------------------
// Cube
// ---------------------------------------------------------------------------

/// Build a unit cube centred at the origin.
///
/// The cube spans `[-0.5, 0.5]` on every axis.  Each of the 6 faces is
/// tessellated as an independent quad (4 vertices, 2 triangles) so that
/// **flat per-face normals** are correct without averaging:
///
/// | Face  | Normal        |
/// |-------|---------------|
/// | +Z    | `(0, 0, 1)`   |
/// | −Z    | `(0, 0, −1)`  |
/// | +X    | `(1, 0, 0)`   |
/// | −X    | `(−1, 0, 0)`  |
/// | +Y    | `(0, 1, 0)`   |
/// | −Y    | `(0, −1, 0)`  |
///
/// **Vertex count:** 24 (6 faces × 4)  
/// **Index count:** 36 (6 faces × 2 triangles × 3)  
///
/// UV origin `(0, 0)` is at the **top-left** of each face; `(1, 1)` is at
/// the **bottom-right**.
pub fn cube() -> Mesh {
    // Each row: position [x,y,z], normal [nx,ny,nz], uv [u,v]
    // Winding is CCW when the face is viewed from outside.
    #[rustfmt::skip]
    let vertices: Vec<Vertex> = vec![
        // ── +Z face (front, normal 0 0 1) ───────────────────────────────
        Vertex::new([-0.5, -0.5,  0.5], [0.0,  0.0,  1.0], [0.0, 1.0]),
        Vertex::new([ 0.5, -0.5,  0.5], [0.0,  0.0,  1.0], [1.0, 1.0]),
        Vertex::new([ 0.5,  0.5,  0.5], [0.0,  0.0,  1.0], [1.0, 0.0]),
        Vertex::new([-0.5,  0.5,  0.5], [0.0,  0.0,  1.0], [0.0, 0.0]),
        // ── −Z face (back,  normal 0 0 −1) ──────────────────────────────
        Vertex::new([ 0.5, -0.5, -0.5], [0.0,  0.0, -1.0], [0.0, 1.0]),
        Vertex::new([-0.5, -0.5, -0.5], [0.0,  0.0, -1.0], [1.0, 1.0]),
        Vertex::new([-0.5,  0.5, -0.5], [0.0,  0.0, -1.0], [1.0, 0.0]),
        Vertex::new([ 0.5,  0.5, -0.5], [0.0,  0.0, -1.0], [0.0, 0.0]),
        // ── +X face (right, normal 1 0 0) ───────────────────────────────
        Vertex::new([ 0.5, -0.5,  0.5], [1.0,  0.0,  0.0], [0.0, 1.0]),
        Vertex::new([ 0.5, -0.5, -0.5], [1.0,  0.0,  0.0], [1.0, 1.0]),
        Vertex::new([ 0.5,  0.5, -0.5], [1.0,  0.0,  0.0], [1.0, 0.0]),
        Vertex::new([ 0.5,  0.5,  0.5], [1.0,  0.0,  0.0], [0.0, 0.0]),
        // ── −X face (left,  normal −1 0 0) ──────────────────────────────
        Vertex::new([-0.5, -0.5, -0.5], [-1.0, 0.0,  0.0], [0.0, 1.0]),
        Vertex::new([-0.5, -0.5,  0.5], [-1.0, 0.0,  0.0], [1.0, 1.0]),
        Vertex::new([-0.5,  0.5,  0.5], [-1.0, 0.0,  0.0], [1.0, 0.0]),
        Vertex::new([-0.5,  0.5, -0.5], [-1.0, 0.0,  0.0], [0.0, 0.0]),
        // ── +Y face (top,    normal 0 1 0) ──────────────────────────────
        Vertex::new([-0.5,  0.5,  0.5], [0.0,  1.0,  0.0], [0.0, 1.0]),
        Vertex::new([ 0.5,  0.5,  0.5], [0.0,  1.0,  0.0], [1.0, 1.0]),
        Vertex::new([ 0.5,  0.5, -0.5], [0.0,  1.0,  0.0], [1.0, 0.0]),
        Vertex::new([-0.5,  0.5, -0.5], [0.0,  1.0,  0.0], [0.0, 0.0]),
        // ── −Y face (bottom, normal 0 −1 0) ─────────────────────────────
        Vertex::new([-0.5, -0.5, -0.5], [0.0, -1.0,  0.0], [0.0, 1.0]),
        Vertex::new([ 0.5, -0.5, -0.5], [0.0, -1.0,  0.0], [1.0, 1.0]),
        Vertex::new([ 0.5, -0.5,  0.5], [0.0, -1.0,  0.0], [1.0, 0.0]),
        Vertex::new([-0.5, -0.5,  0.5], [0.0, -1.0,  0.0], [0.0, 0.0]),
    ];

    // Each face is a quad at base index `face * 4`.
    // The two triangles per quad share the diagonal v0→v2.
    let mut indices: Vec<u32> = Vec::with_capacity(36);
    for face in 0..6u32 {
        let b = face * 4;
        indices.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }

    Mesh::new(vertices, indices)
}
