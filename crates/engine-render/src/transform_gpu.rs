//! GPU-side transform pipeline: stable per-component "source of truth"
//! (SoT) buffers + the compute pipelines that read / write them.
//!
//! # The pipeline
//!
//! Per frame, in this order, all inside the slot's pre-recorded primary CB:
//!
//! 1. **Scatter compute (×3, one per component)**
//!    Reads `staging_<comp>[i]` and `dirty[i]`; if the dirty bit is set,
//!    writes `sot_<comp>[i] = staging_<comp>[i]`. Dispatched over
//!    `entity_capacity` invocations. The per-frame staging buffer + dirty
//!    bitmask live on the [`crate::FrameSlot`]; the SoT buffers live here.
//!
//! 2. **MVP-build compute (×1)**
//!    Reads SoT pos/rot/scale indexed by a per-camera `instance → entity`
//!    lookup, multiplies `view_proj * model`, writes the per-camera MVP
//!    buffer the vertex shader will read. Lives partially on the camera
//!    (set 0: SoT + idx + mvp) and partially on the FrameSlot (set 1:
//!    view_proj uniform).
//!
//! 3. **Graphics: scene secondary**
//!    Indexed draws read the MVP buffer via `gl_InstanceIndex`.
//!
//! 4. **Graphics: blit secondary**
//!    Camera color → swapchain image.
//!
//! # Why a single shader for all three components?
//!
//! Position, rotation, and scale all upload as `vec4` (rotation is a quat,
//! pos/scale are `vec3` padded to `vec4`). Same descriptor-set layout works
//! for all three — only the bound buffers differ. Three dispatches with the
//! same scatter pipeline but different descriptor sets is one fewer pipeline
//! to manage and lets the driver fuse them efficiently.
//!
//! # Invalidation
//!
//! [`WorldTransformGpu::ensure_capacity`] grows the SoT buffers
//! geometrically (≥ 2×). When that fires, every FrameSlot needs to rebuild
//! (its staging buffers must match the new entity capacity, and its scatter
//! descriptor sets reference the SoT buffers by handle), and every
//! `RenderCamera`'s `mvp_build_set0` must be re-allocated for the same
//! reason.

use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::layout::DescriptorSetLayout,
    device::Device,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
};

use crate::shaders;

/// One `vec4` per entity slot, in either staging (host-visible) or SoT
/// (device-local) form. Layout matches GLSL `vec4` in std430.
pub type ComponentSlot = [f32; 4];

/// Number of `u32` words needed to bitmask `entity_capacity` slots.
#[inline]
pub fn dirty_word_count(entity_capacity: usize) -> usize {
    entity_capacity.div_ceil(32).max(1)
}

/// World-scoped GPU transform state: the three SoT buffers (one per
/// component) plus the two compute pipelines that act on them.
///
/// "World-scoped" because there is exactly one transform hierarchy per
/// scene; cameras are independent of it. Multiple cameras may all read
/// these SoT buffers from their own `mvp_build_set0`s simultaneously.
pub struct WorldTransformGpu {
    /// Position SoT — `(x, y, z, _)` per slot.
    sot_positions: Subbuffer<[ComponentSlot]>,
    /// Rotation SoT — quaternion `(x, y, z, w)` per slot.
    sot_rotations: Subbuffer<[ComponentSlot]>,
    /// Scale SoT — `(x, y, z, _)` per slot.
    sot_scales:    Subbuffer<[ComponentSlot]>,

    /// Currently-allocated SoT slot count (== capacity of all three buffers
    /// above, same value). Always ≥ 1. Grows geometrically; never shrinks.
    entity_capacity: usize,

    /// Scatter compute pipeline — see [`shaders::scatter_cs`]. One pipeline
    /// shared by the per-component scatter dispatches.
    scatter_pipeline:   Arc<ComputePipeline>,
    /// MVP-build compute pipeline — see [`shaders::mvp_build_cs`].
    mvp_build_pipeline: Arc<ComputePipeline>,
}

impl WorldTransformGpu {
    /// Build the SoT buffers for `entity_capacity` slots and create both
    /// compute pipelines.
    pub fn new(
        device:           Arc<Device>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        entity_capacity:  usize,
    ) -> Self {
        let cap = entity_capacity.max(1);

        let (sot_positions, sot_rotations, sot_scales) =
            allocate_sot_buffers(memory_allocator, cap);

        let scatter_pipeline   = build_scatter_pipeline(device.clone());
        let mvp_build_pipeline = build_mvp_build_pipeline(device);

        Self {
            sot_positions,
            sot_rotations,
            sot_scales,
            entity_capacity:    cap,
            scatter_pipeline,
            mvp_build_pipeline,
        }
    }

    /// Ensure the SoT buffers have at least `needed` slots. Returns `true`
    /// if the buffers were re-allocated (in which case every dependent
    /// descriptor set / FrameSlot / `RenderCamera::mvp_build_set0` must be
    /// rebuilt — they captured the old buffer handles).
    ///
    /// Geometric growth (≥ 2× current) keeps amortized cost O(1) per added
    /// entity. Never shrinks.
    pub fn ensure_capacity(
        &mut self,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        needed:           usize,
    ) -> bool {
        if needed <= self.entity_capacity {
            return false;
        }
        let new_cap = needed
            .max(self.entity_capacity.saturating_mul(2))
            .max(1);
        let (pos, rot, scl) = allocate_sot_buffers(memory_allocator, new_cap);
        self.sot_positions   = pos;
        self.sot_rotations   = rot;
        self.sot_scales      = scl;
        self.entity_capacity = new_cap;
        true
    }

    pub fn entity_capacity(&self)    -> usize                       { self.entity_capacity }
    pub fn sot_positions(&self)      -> &Subbuffer<[ComponentSlot]> { &self.sot_positions }
    pub fn sot_rotations(&self)      -> &Subbuffer<[ComponentSlot]> { &self.sot_rotations }
    pub fn sot_scales(&self)         -> &Subbuffer<[ComponentSlot]> { &self.sot_scales }
    pub fn scatter_pipeline(&self)   -> &Arc<ComputePipeline>       { &self.scatter_pipeline }
    pub fn mvp_build_pipeline(&self) -> &Arc<ComputePipeline>       { &self.mvp_build_pipeline }

    /// Convenience: the descriptor-set layout the per-component scatter
    /// passes bind to (set 0 of [`shaders::scatter_cs`]).
    pub fn scatter_set_layout(&self) -> &Arc<DescriptorSetLayout> {
        &self.scatter_pipeline.layout().set_layouts()[0]
    }

    /// Convenience: layout of mvp-build set 0 (per-camera SoT/idx/mvp).
    pub fn mvp_build_set0_layout(&self) -> &Arc<DescriptorSetLayout> {
        &self.mvp_build_pipeline.layout().set_layouts()[0]
    }

    /// Convenience: layout of mvp-build set 1 (per-frame view_proj uniform).
    pub fn mvp_build_set1_layout(&self) -> &Arc<DescriptorSetLayout> {
        &self.mvp_build_pipeline.layout().set_layouts()[1]
    }
}

fn allocate_sot_buffers(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    capacity:         usize,
) -> (
    Subbuffer<[ComponentSlot]>,
    Subbuffer<[ComponentSlot]>,
    Subbuffer<[ComponentSlot]>,
) {
    let make = || -> Subbuffer<[ComponentSlot]> {
        Buffer::new_slice::<ComponentSlot>(
            memory_allocator.clone(),
            BufferCreateInfo {
                // STORAGE_BUFFER: read/written by the scatter compute, read
                // by the mvp-build compute. (No TRANSFER_DST: the scatter
                // compute is the only writer; we never `vkCmdCopyBuffer`
                // into these.)
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            capacity as u64,
        )
        .expect("Failed to allocate SoT buffer")
    };
    (make(), make(), make())
}

fn build_scatter_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs    = shaders::scatter_cs::load(device.clone()).expect("scatter_cs load failed");
    let entry = cs.entry_point("main").expect("scatter_cs entry point");
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(&stage))
            .into_pipeline_layout_create_info(device.clone())
            .expect("scatter pipeline layout info"),
    )
    .expect("scatter pipeline layout");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("scatter ComputePipeline::new")
}

fn build_mvp_build_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs    = shaders::mvp_build_cs::load(device.clone()).expect("mvp_build_cs load failed");
    let entry = cs.entry_point("main").expect("mvp_build_cs entry point");
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(&stage))
            .into_pipeline_layout_create_info(device.clone())
            .expect("mvp_build pipeline layout info"),
    )
    .expect("mvp_build pipeline layout");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("mvp_build ComputePipeline::new")
}
