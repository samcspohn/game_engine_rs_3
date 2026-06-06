//! Per-transform GPU renderer records.
//!
//! [`GpuRenderers`] owns the device-local `GPURenderers` buffer: one `u32`
//! `mesh_id` per transform slot (indexed by `transform_id`, parallel to the
//! SoT TRS buffers). Newly-spawned renderers are scattered in via a compute
//! dispatch from a list of `(transform_id, mesh_id)` pairs that the
//! `MeshRenderer` component pushes at `init` time.
//!
//! This is the instance-side input the future GPU cull kernel consumes
//! (`mesh_id → redirect → drawable slot`). Nothing reads it yet; it is
//! groundwork, like the [`crate::assets::GpuMeshStore`].
//!
//! # Synchronization (first slice)
//!
//! The scatter uses a **one-shot fence-waited submit**, fired only on frames
//! where renderers actually spawned (the drained queue is non-empty).
//! Steady-state frames do no GPU work here. Frame-loop folding (recording the
//! scatter into the per-frame primary CB) comes with the cull kernel.

use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::GpuFuture,
};

use crate::shaders;

/// Sentinel `mesh_id` for a transform slot with no renderer attached. The
/// future cull kernel skips slots holding this value.
pub const NO_RENDERER: u32 = u32::MAX;

/// Device-local `GPURenderers` buffer + the scatter pipeline that fills it.
pub struct GpuRenderers {
    /// One `mesh_id` per transform slot; [`NO_RENDERER`] where empty.
    renderers: Subbuffer<[u32]>,
    capacity: u32,
    pipeline: Arc<ComputePipeline>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    queue: Arc<Queue>,
}

impl GpuRenderers {
    /// Allocate the renderers buffer sized to `capacity` transform slots,
    /// initialized to [`NO_RENDERER`].
    pub fn new(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        cb_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        queue: Arc<Queue>,
        capacity: u32,
    ) -> Self {
        let capacity = capacity.max(1);
        let pipeline = build_scatter_pipeline(device);
        let renderers = alloc_renderers(&memory_allocator, capacity);

        let store = Self {
            renderers,
            capacity,
            pipeline,
            memory_allocator,
            cb_allocator,
            descriptor_set_allocator,
            queue,
        };
        store.fill_sentinel(&store.renderers);
        store
    }

    /// The device-local `GPURenderers` buffer (read by the future cull kernel).
    #[allow(dead_code)] // consumed by the cull kernel (next slice)
    pub fn buffer(&self) -> &Subbuffer<[u32]> {
        &self.renderers
    }

    /// Number of transform slots the buffer can hold.
    #[allow(dead_code)] // consumed by the cull kernel (next slice)
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Grow the buffer to hold at least `needed` transform slots, preserving
    /// existing records and sentinel-filling the new tail. Geometric (≥ 2×).
    /// Returns whether it grew.
    pub fn ensure_capacity(&mut self, needed: u32) -> bool {
        if needed <= self.capacity {
            return false;
        }
        let new_cap = self.capacity.saturating_mul(2).max(needed);
        let new = alloc_renderers(&self.memory_allocator, new_cap);
        self.fill_sentinel(&new);

        let mut builder = self.primary_builder();
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.renderers.clone().slice(0..self.capacity as u64),
                new.clone().slice(0..self.capacity as u64),
            ))
            .expect("copy old GPURenderers into grown buffer");
        self.submit_and_wait(builder.build().expect("build GPURenderers grow CB"));

        self.renderers = new;
        self.capacity = new_cap;
        true
    }

    /// Scatter newly-spawned `(transform_id, mesh_id)` pairs into the buffer.
    /// No-op when `spawns` is empty. Callers must have grown capacity to cover
    /// every `transform_id` first (via [`ensure_capacity`](Self::ensure_capacity)).
    pub fn ingest(&self, spawns: &[[u32; 2]]) {
        if spawns.is_empty() {
            return;
        }
        debug_assert!(
            spawns.iter().all(|p| p[0] < self.capacity),
            "spawn transform_id out of GPURenderers capacity",
        );

        let staging = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            spawns.iter().copied(),
        )
        .expect("create spawn staging buffer");

        let layout = self.pipeline.layout().clone();
        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, staging.clone()),
                WriteDescriptorSet::buffer(1, self.renderers.clone()),
            ],
            [],
        )
        .expect("GPURenderers scatter descriptor set");

        let pc = shaders::gpu_renderers_scatter_cs::PC {
            spawn_count: spawns.len() as u32,
        };
        let groups = (spawns.len() as u32).div_ceil(64).max(1);

        let mut builder = self.primary_builder();
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .expect("bind scatter pipeline")
            .bind_descriptor_sets(PipelineBindPoint::Compute, layout.clone(), 0, set)
            .expect("bind scatter set")
            .push_constants(layout, 0, pc)
            .expect("push scatter constants");
        unsafe {
            builder.dispatch([groups, 1, 1]).expect("dispatch scatter");
        }
        self.submit_and_wait(builder.build().expect("build GPURenderers scatter CB"));
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn fill_sentinel(&self, buf: &Subbuffer<[u32]>) {
        let mut builder = self.primary_builder();
        builder
            .fill_buffer(buf.clone(), NO_RENDERER)
            .expect("sentinel-fill GPURenderers");
        self.submit_and_wait(builder.build().expect("build GPURenderers fill CB"));
    }

    fn primary_builder(&self) -> PrimaryBuilder {
        AutoCommandBufferBuilder::primary(
            self.cb_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("create one-shot primary command buffer")
    }

    fn submit_and_wait<C>(&self, cb: Arc<C>)
    where
        C: vulkano::command_buffer::PrimaryCommandBufferAbstract + 'static,
    {
        vulkano::sync::now(self.queue.device().clone())
            .then_execute(self.queue.clone(), cb)
            .expect("submit GPURenderers op")
            .then_signal_fence_and_flush()
            .expect("flush GPURenderers op")
            .wait(None)
            .expect("await GPURenderers op");
    }
}

/// Builder type for the store's one-shot command buffers.
type PrimaryBuilder = AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>;

fn alloc_renderers(allocator: &Arc<StandardMemoryAllocator>, count: u32) -> Subbuffer<[u32]> {
    Buffer::new_slice::<u32>(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_SRC
                | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        count as u64,
    )
    .expect("allocate GPURenderers buffer")
}

fn build_scatter_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs = shaders::gpu_renderers_scatter_cs::load(device.clone())
        .expect("gpu_renderers_scatter_cs load failed");
    let entry = cs.entry_point("main").expect("scatter entry point");
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
