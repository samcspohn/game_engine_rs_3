//! GPU-resident mesh buffers.
//!
//! [`GpuMesh`] uploads an [`engine_core::mesh::Mesh`] into Vulkan vertex and
//! index buffers.  The CPU-side [`engine_core::mesh::Vertex`] is mirrored by
//! [`GpuVertex`], which derives vulkano's `BufferContents` (for slice-buffer
//! writes) and `Vertex` (so the pipeline can auto-reflect attribute locations
//! from the vertex shader interface).

use std::sync::Arc;

use engine_core::mesh::Mesh;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::graphics::vertex_input::Vertex,
};

// ─────────────────────────────────────────────────────────────────────────────
// GpuVertex
// ─────────────────────────────────────────────────────────────────────────────

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
            normal:   v.normal.into(),
            uv:       v.uv.into(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuMesh
// ─────────────────────────────────────────────────────────────────────────────

/// Uploaded vertex + index buffers ready for `draw_indexed` calls.
pub struct GpuMesh {
    pub vertex_buffer: Subbuffer<[GpuVertex]>,
    pub index_buffer:  Subbuffer<[u32]>,
    pub index_count:   u32,
}

impl GpuMesh {
    /// Upload a CPU [`Mesh`] into device-accessible Vulkan buffers.
    ///
    /// Uses `HOST_SEQUENTIAL_WRITE` for a simple, one-shot upload path.
    /// For frequently-updated or very large meshes a staging-buffer approach
    /// (host-visible → device-local copy) would be preferable.
    pub fn upload(mesh: &Mesh, allocator: &Arc<StandardMemoryAllocator>) -> Self {
        let gpu_verts: Vec<GpuVertex> =
            mesh.vertices.iter().copied().map(GpuVertex::from).collect();

        let vertex_buffer = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            gpu_verts,
        )
        .expect("Failed to create vertex buffer");

        let index_buffer = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            mesh.indices.iter().copied(),
        )
        .expect("Failed to create index buffer");

        GpuMesh {
            index_count: mesh.indices.len() as u32,
            vertex_buffer,
            index_buffer,
        }
    }
}
