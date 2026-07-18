//! Per-transform GPU parent records.
//!
//! [`GpuParents`] owns the device-local `Parents` buffer: one `u32` parent
//! transform id per transform slot (indexed by `transform_id`, parallel to
//! the SoT TRS buffers), [`NO_PARENT`] for roots. `mvp_build_cs` walks this
//! buffer upward each frame to compose world TRS from the **local** TRS held
//! in the SoT — the GPU side of the parent hierarchy.
//!
//! # Streamed, not bitmask-driven
//!
//! Unlike TRS (sparse dirty bitmask over a capacity-sized staging mirror),
//! parent changes arrive as a **stream** of `[transform_id, new_parent]`
//! pairs (see `TransformHierarchy::drain_parent_updates`): re-parenting is
//! rare, so an O(changes) staging upload + scatter dispatch beats another
//! O(capacity) staging buffer. Steady-state frames do no GPU work here.
//!
//! # Synchronization
//!
//! Same first-slice model as [`crate::gpu_renderers::GpuRenderers`]: a
//! one-shot fence-waited submit, fired only on frames with parent changes.
//! Frame-loop folding (recording the scatter into the per-frame primary CB)
//! is future work, shared with the GPURenderers scatter.

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

/// Sentinel parent id for a root transform (no parent). Matches
/// `engine_core::transform::NO_PARENT` and the `NO_PARENT` constant in
/// `mvp_build.comp` / `parent_scatter.comp`.
pub const NO_PARENT: u32 = u32::MAX;

/// Device-local `Parents` buffer + the scatter pipeline that updates it.
pub struct GpuParents {
    /// One parent transform id per transform slot; [`NO_PARENT`] for roots.
    parents: Subbuffer<[u32]>,
    capacity: u32,
    pipeline: Arc<ComputePipeline>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    queue: Arc<Queue>,
}

impl GpuParents {
    /// Allocate the parents buffer sized to `capacity` transform slots,
    /// initialized to [`NO_PARENT`].
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
        let parents = alloc_parents(&memory_allocator, capacity);

        let store = Self {
            parents,
            capacity,
            pipeline,
            memory_allocator,
            cb_allocator,
            descriptor_set_allocator,
            queue,
        };
        store.fill_sentinel(&store.parents);
        store
    }

    /// The device-local `Parents` buffer (read by `mvp_build_cs`'s chain walk).
    pub fn buffer(&self) -> &Subbuffer<[u32]> {
        &self.parents
    }

    /// Number of transform slots the buffer can hold.
    #[allow(dead_code)]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Grow the buffer to hold at least `needed` transform slots, preserving
    /// existing records and sentinel-filling the new tail. Geometric (≥ 2×).
    /// Returns whether it grew — the caller must then rebuild everything that
    /// captured the old buffer handle (the camera's cull set).
    pub fn ensure_capacity(&mut self, needed: u32) -> bool {
        if needed <= self.capacity {
            return false;
        }
        let new_cap = self.capacity.saturating_mul(2).max(needed);
        let new = alloc_parents(&self.memory_allocator, new_cap);
        self.fill_sentinel(&new);

        let mut builder = self.primary_builder();
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.parents.clone().slice(0..self.capacity as u64),
                new.clone().slice(0..self.capacity as u64),
            ))
            .expect("copy old Parents into grown buffer");
        self.submit_and_wait(builder.build().expect("build Parents grow CB"));

        self.parents = new;
        self.capacity = new_cap;
        true
    }

    /// Scatter streamed `(transform_id, new_parent)` pairs into the buffer.
    /// No-op when `updates` is empty. Callers must have grown capacity to
    /// cover every `transform_id` first (via
    /// [`ensure_capacity`](Self::ensure_capacity)).
    pub fn ingest(&self, updates: &[[u32; 2]]) {
        if updates.is_empty() {
            return;
        }
        debug_assert!(
            updates.iter().all(|p| p[0] < self.capacity),
            "parent-update transform_id out of Parents capacity",
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
            updates.iter().copied(),
        )
        .expect("create parent-update staging buffer");

        let layout = self.pipeline.layout().clone();
        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, staging.clone()),
                WriteDescriptorSet::buffer(1, self.parents.clone()),
            ],
            [],
        )
        .expect("Parents scatter descriptor set");

        let pc = shaders::parent_scatter_cs::PC {
            update_count: updates.len() as u32,
        };
        let groups = (updates.len() as u32).div_ceil(64).max(1);

        let mut builder = self.primary_builder();
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .expect("bind parent scatter pipeline")
            .bind_descriptor_sets(PipelineBindPoint::Compute, layout.clone(), 0, set)
            .expect("bind parent scatter set")
            .push_constants(layout, 0, pc)
            .expect("push parent scatter constants");
        unsafe {
            builder
                .dispatch([groups, 1, 1])
                .expect("dispatch parent scatter");
        }
        self.submit_and_wait(builder.build().expect("build Parents scatter CB"));
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn fill_sentinel(&self, buf: &Subbuffer<[u32]>) {
        let mut builder = self.primary_builder();
        builder
            .fill_buffer(buf.clone(), NO_PARENT)
            .expect("sentinel-fill Parents");
        self.submit_and_wait(builder.build().expect("build Parents fill CB"));
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
            .expect("submit Parents op")
            .then_signal_fence_and_flush()
            .expect("flush Parents op")
            .wait(None)
            .expect("await Parents op");
    }
}

/// Builder type for the store's one-shot command buffers.
type PrimaryBuilder = AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>;

fn alloc_parents(allocator: &Arc<StandardMemoryAllocator>, count: u32) -> Subbuffer<[u32]> {
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
    .expect("allocate Parents buffer")
}

fn build_scatter_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs = shaders::parent_scatter_cs::load(device.clone())
        .expect("parent_scatter_cs load failed");
    let entry = cs.entry_point("main").expect("parent scatter entry point");
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(&stage))
            .into_pipeline_layout_create_info(device.clone())
            .expect("parent scatter pipeline layout info"),
    )
    .expect("parent scatter pipeline layout");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("parent scatter ComputePipeline::new")
}
