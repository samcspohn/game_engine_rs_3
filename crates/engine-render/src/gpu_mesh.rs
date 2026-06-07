//! GPU-side vertex type.
//!
//! The CPU-side [`engine_core::mesh::Vertex`] is mirrored by [`GpuVertex`],
//! which derives vulkano's `BufferContents` (for slice-buffer writes) and
//! `Vertex` (so the pipeline can auto-reflect attribute locations from the
//! vertex shader interface).
//!
//! Mesh geometry is uploaded into the shared mega vertex/index buffers owned
//! by [`crate::assets::GpuMeshStore`]; there is no per-mesh GPU buffer type.

use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

/// GPU-side vertex that mirrors [`engine_core::mesh::Vertex`] exactly.
///
/// The `#[format(...)]` annotations tell vulkano which Vulkan format to use
/// for each attribute, matching the locations declared in the vertex shader.
#[derive(BufferContents, Vertex, Clone, Copy, Debug)]
#[repr(C)]
pub struct GpuVertex {
    /// `layout(location = 0) in vec3 position`
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    /// `layout(location = 1) in vec3 normal`
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    /// `layout(location = 2) in vec2 uv`
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
}

impl From<engine_core::mesh::Vertex> for GpuVertex {
    fn from(v: engine_core::mesh::Vertex) -> Self {
        GpuVertex {
            position: v.position.into(),
            normal: v.normal.into(),
            uv: v.uv.into(),
        }
    }
}
