//! GPU mirror of the core mesh asset registry.
//!
//! The CPU source of truth — dedup, redirect map, retained `Arc<Mesh>` — lives
//! in [`engine_core::asset`]. This module holds only the **device-local** GPU
//! data: the mega vertex/index buffers, the per-slot draw/cull table, and the
//! redirect buffer the cull kernel reads. [`GpuMeshStore::sync`] drains the
//! core registry's deltas each frame and uploads them.

mod gpu_store;
mod texture_store;

pub use gpu_store::GpuMeshStore;
pub use texture_store::{GpuTextureStore, MAX_TEXTURES};

use vulkano::buffer::BufferContents;

/// Per drawable slot, as the GPU sees it. Mirrors the *static* fields of a
/// `VkDrawIndexedIndirectCommand` (the cull kernel writes the dynamic
/// `instance_count` / `first_instance` per frame) plus a local-space bounding
/// sphere for frustum culling.
///
/// Laid out for std430 as two 16-byte rows (32 bytes total, no internal
/// padding). The mega-buffer offsets are assigned by [`GpuMeshStore`] (a
/// render-side concern); the bounds come from [`engine_core::asset::MeshBounds`].
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
