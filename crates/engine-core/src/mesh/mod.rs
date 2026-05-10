//! CPU-side mesh types and primitive generators.
//!
//! This module is deliberately GPU-free: it holds only plain data (`Vec`,
//! `glam` math types) so that every workspace crate — packager, editor,
//! game-logic — can use it without pulling in Vulkan.
//!
//! # Types
//!
//! | Type | Description |
//! |------|-------------|
//! [`Vertex`] | A single vertex: position, normal, and UV. |
//! [`Mesh`]   | Indexed triangle list (vertices + u32 indices). |
//! [`Aabb`]   | Axis-aligned bounding box computed from a [`Mesh`]. |
//!
//! # Primitives
//!
//! Ready-made unit meshes live in the [`primitives`] submodule:
//!
//! ```rust
//! use engine_core::mesh::primitives;
//! let cube = primitives::cube(); // unit cube centred at the origin
//! ```

pub mod primitives;

use glam::{Vec2, Vec3};

// ---------------------------------------------------------------------------
// Vertex
// ---------------------------------------------------------------------------

/// A single mesh vertex.
///
/// The layout is `#[repr(C)]` so that a `&[Vertex]` can be cast directly to
/// a byte slice for GPU upload without any transformation.
///
/// All vectors are in **local (model) space**.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    /// Position in local space.
    pub position: Vec3,
    /// Surface normal in local space (should be unit-length).
    pub normal: Vec3,
    /// Texture / UV coordinate (origin at top-left, (1,1) at bottom-right).
    pub uv: Vec2,
}

impl Vertex {
    /// Construct a vertex from plain array literals — handy in const / test
    /// contexts where you want to avoid repeating `Vec3::new(…)`.
    #[inline]
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            position: Vec3::from(position),
            normal:   Vec3::from(normal),
            uv:       Vec2::from(uv),
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

/// An indexed triangle-list mesh stored entirely on the CPU.
///
/// Indices are `u32` and are laid out as sequential triangles:
/// `[i0, i1, i2,  i3, i4, i5, …]`.  Winding order is **counter-clockwise**
/// when viewed from outside the surface (right-handed, Y-up convention).
///
/// # GPU upload
///
/// The actual Vulkano vertex/index buffers live in `engine-render`.  A
/// `GpuMesh` in that crate is constructed by passing a reference to this
/// type.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Vertex data.
    pub vertices: Vec<Vertex>,
    /// Triangle indices into `vertices`.
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Create a new mesh from pre-built vertex and index data.
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    /// Return the number of triangles in this mesh.
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Compute the axis-aligned bounding box of this mesh.
    ///
    /// Returns `None` when the vertex list is empty.
    pub fn aabb(&self) -> Option<Aabb> {
        if self.vertices.is_empty() {
            return None;
        }
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for v in &self.vertices {
            min = min.min(v.position);
            max = max.max(v.position);
        }
        Some(Aabb { min, max })
    }
}

// ---------------------------------------------------------------------------
// Aabb
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box in local (model) space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// Minimum corner (smallest X, Y, Z).
    pub min: Vec3,
    /// Maximum corner (largest X, Y, Z).
    pub max: Vec3,
}

impl Aabb {
    /// Centre of the bounding box.
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Full extent along each axis (`max - min`).
    #[inline]
    pub fn extent(&self) -> Vec3 {
        self.max - self.min
    }

    /// Half-extent along each axis (i.e. the "radius" in each dimension).
    #[inline]
    pub fn half_extent(&self) -> Vec3 {
        self.extent() * 0.5
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mesh_has_no_aabb() {
        let m = Mesh::new(vec![], vec![]);
        assert!(m.aabb().is_none());
    }

    #[test]
    fn single_vertex_aabb_is_a_point() {
        let m = Mesh::new(
            vec![Vertex::new([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0])],
            vec![],
        );
        let aabb = m.aabb().unwrap();
        assert_eq!(aabb.min, aabb.max);
        assert_eq!(aabb.center(), glam::Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn cube_aabb_is_unit() {
        let cube = primitives::cube();
        let aabb = cube.aabb().unwrap();
        assert!((aabb.min - glam::Vec3::splat(-0.5)).length() < 1e-6);
        assert!((aabb.max - glam::Vec3::splat( 0.5)).length() < 1e-6);
        assert!((aabb.center()).length() < 1e-6);
        assert!((aabb.extent() - glam::Vec3::ONE).length() < 1e-6);
    }

    #[test]
    fn cube_has_correct_counts() {
        let cube = primitives::cube();
        // 6 faces × 4 verts = 24 vertices; 6 faces × 6 indices = 36 indices
        assert_eq!(cube.vertices.len(), 24);
        assert_eq!(cube.indices.len(), 36);
        assert_eq!(cube.triangle_count(), 12);
    }

    #[test]
    fn cube_indices_in_bounds() {
        let cube = primitives::cube();
        let n = cube.vertices.len() as u32;
        for &i in &cube.indices {
            assert!(i < n, "index {i} out of bounds (vertex count = {n})");
        }
    }

    #[test]
    fn cube_normals_are_unit() {
        let cube = primitives::cube();
        for v in &cube.vertices {
            let len = v.normal.length();
            assert!(
                (len - 1.0).abs() < 1e-6,
                "normal {:?} has length {len}",
                v.normal
            );
        }
    }
}
