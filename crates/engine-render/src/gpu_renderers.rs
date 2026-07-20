//! Per-transform GPU renderer records.
//!
//! [`GpuRenderers`] owns the device-local `GPURenderers` buffer: one
//! `(mesh_id, material_id)` `uvec2` per transform slot (indexed by
//! `transform_id`, parallel to the SoT TRS buffers), read by the cull kernel
//! each frame (`mesh_id → redirect → drawable slot`; `material_id` — or the
//! mesh slot's authored material when the word is [`MATERIAL_INHERIT`] — is
//! forwarded per visible instance to the fragment shader). Newly-spawned
//! renderers, and material swaps on live renderers, are scattered in from
//! the record queue the `MeshRenderer` component pushes.
//!
//! # Folded into the frame CB (streamed, count-in-buffer)
//!
//! The spawn scatter is a **pre-recorded** compute secondary
//! ([`Self::spawn_scatter_secondary`]) executed at the front of every
//! FrameSlot primary, exactly like `WorldTransformGpu`'s scatters. Because
//! the primary is pre-recorded, the dispatch covers the spawn staging's
//! fixed pair *capacity*; the *live* per-frame count is word 0 of the
//! staging buffer, written by [`Self::write_spawns`] each frame (0 when
//! quiet) under the same `gpu_signal` gate as the TRS staging. Quiet
//! frames cost a handful of early-out workgroups — no submit, no fence, no
//! allocation. This replaced the first-slice one-shot fence-waited
//! `ingest` submit, which paid a host-blocking pipeline bubble on every
//! burst frame.
//!
//! One-shot submits remain only on the **rare** paths: sentinel fill at
//! construction and the copy-preserving migration in
//! [`Self::ensure_capacity`].

use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferInheritanceInfo, CommandBufferUsage, CopyBufferInfo,
        SecondaryAutoCommandBuffer,
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
/// cull kernel skips slots holding this value.
pub const NO_RENDERER: u32 = u32::MAX;

/// Sentinel `material_id` meaning "use the mesh slot's authored material"
/// (the default for renderers that never set an explicit material). Resolved
/// by the cull kernel against the mesh store's slot-material table.
pub const MATERIAL_INHERIT: u32 = u32::MAX;

/// Initial record capacity of the spawn staging buffer. Grows geometrically
/// when a frame's spawn burst exceeds it (e.g. initial scene population),
/// which forces the usual secondary/frame-slot rebuild.
const INITIAL_SPAWN_CAPACITY: usize = 1024;

/// Device-local `GPURenderers` buffer + the folded spawn-scatter machinery.
pub struct GpuRenderers {
    /// Two words — `(mesh_id, material_id)` — per transform slot;
    /// `(NO_RENDERER, MATERIAL_INHERIT)` where empty (both bit-patterns are
    /// `0xFFFFFFFF`, so the sentinel fill covers the whole buffer).
    renderers: Subbuffer<[u32]>,
    capacity: u32,
    pipeline: Arc<ComputePipeline>,

    /// Host-mapped spawn stream staging. Layout (std430, matching
    /// `gpu_renderers_scatter.comp`): word 0 = live record count, word 1 =
    /// pad, then `[transform_id, mesh_id, material_id]` triples from word 2.
    spawn_staging: Subbuffer<[u32]>,
    /// Record capacity of `spawn_staging`.
    spawn_capacity: usize,
    /// Set 0: (spawn_staging, renderers). Rebuilt when either reallocates.
    scatter_set: Arc<DescriptorSet>,
    /// Pre-recorded SimultaneousUse compute secondary — one dispatch over
    /// the spawn staging capacity. Captured by every FrameSlot primary;
    /// re-recorded on either buffer's growth.
    scatter_secondary: Arc<SecondaryAutoCommandBuffer>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    queue: Arc<Queue>,
}

impl GpuRenderers {
    /// Allocate the renderers buffer sized to `capacity` transform slots
    /// (initialized to [`NO_RENDERER`]) plus the spawn staging + scatter
    /// secondary.
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

        let spawn_capacity = INITIAL_SPAWN_CAPACITY;
        let spawn_staging = alloc_spawn_staging(&memory_allocator, spawn_capacity);
        let scatter_set = build_scatter_set(
            &descriptor_set_allocator,
            &pipeline,
            &spawn_staging,
            &renderers,
        );
        let scatter_secondary = record_scatter_secondary(
            &cb_allocator,
            queue.queue_family_index(),
            &pipeline,
            &scatter_set,
            spawn_capacity,
        );

        let store = Self {
            renderers,
            capacity,
            pipeline,
            spawn_staging,
            spawn_capacity,
            scatter_set,
            scatter_secondary,
            memory_allocator,
            cb_allocator,
            descriptor_set_allocator,
            queue,
        };
        store.fill_sentinel(&store.renderers);
        store
    }

    /// The device-local `GPURenderers` buffer (read by the cull kernel).
    pub fn buffer(&self) -> &Subbuffer<[u32]> {
        &self.renderers
    }

    /// Number of transform slots the buffer can hold.
    #[allow(dead_code)]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Pre-recorded spawn-scatter secondary, executed at the front of every
    /// FrameSlot primary (before `signal_cs`, so the `gpu_signal` wait
    /// covers the staging read; before the cull, which reads the buffer it
    /// writes).
    pub fn spawn_scatter_secondary(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.scatter_secondary
    }

    /// Grow the buffer to hold at least `needed` transform slots, preserving
    /// existing records and sentinel-filling the new tail. Geometric (≥ 2×).
    /// Returns whether it grew — the caller must then rebuild everything
    /// that captured the old handles (cull set, FrameSlot primaries).
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
                self.renderers.clone().slice(0..2 * self.capacity as u64),
                new.clone().slice(0..2 * self.capacity as u64),
            ))
            .expect("copy old GPURenderers into grown buffer");
        self.submit_and_wait(builder.build().expect("build GPURenderers grow CB"));

        self.renderers = new;
        self.capacity = new_cap;
        self.rebuild_scatter();
        true
    }

    /// Ensure the spawn staging can hold `needed` records this frame. Returns
    /// `true` if it re-allocated — the scatter secondary was re-recorded,
    /// so every FrameSlot primary must be rebuilt (callers fold this into
    /// `force_full`). Geometric growth; never shrinks.
    pub fn ensure_spawn_capacity(&mut self, needed: usize) -> bool {
        if needed <= self.spawn_capacity {
            return false;
        }
        self.spawn_capacity = needed.max(self.spawn_capacity.saturating_mul(2));
        self.spawn_staging = alloc_spawn_staging(&self.memory_allocator, self.spawn_capacity);
        self.rebuild_scatter();
        true
    }

    /// Write this frame's drained `(transform_id, mesh_id, material_id)`
    /// records (plus the live count in word 0) into the spawn staging. Must
    /// be called **every** frame — count 0 retires the previous frame's
    /// records — and only after
    /// `WorldTransformGpu::host_wait_for_previous_compute` (the `gpu_signal`
    /// gate covers this buffer's in-CB read).
    pub fn write_spawns(&self, spawns: &[[u32; 3]]) {
        assert!(
            spawns.len() <= self.spawn_capacity,
            "spawn burst ({}) exceeds staging capacity ({}) — \
             ensure_spawn_capacity must run first",
            spawns.len(),
            self.spawn_capacity,
        );
        debug_assert!(
            spawns.iter().all(|r| r[0] < self.capacity),
            "spawn transform_id out of GPURenderers capacity",
        );
        let mut w = self.spawn_staging.write().expect("spawn_staging.write");
        w[0] = spawns.len() as u32;
        for (i, rec) in spawns.iter().enumerate() {
            w[2 + 3 * i] = rec[0];
            w[3 + 3 * i] = rec[1];
            w[4 + 3 * i] = rec[2];
        }
    }

    // ── Internal ────────────────────────────────────────────────────────

    /// Rebuild the descriptor set + secondary after either buffer moved.
    fn rebuild_scatter(&mut self) {
        self.scatter_set = build_scatter_set(
            &self.descriptor_set_allocator,
            &self.pipeline,
            &self.spawn_staging,
            &self.renderers,
        );
        self.scatter_secondary = record_scatter_secondary(
            &self.cb_allocator,
            self.queue.queue_family_index(),
            &self.pipeline,
            &self.scatter_set,
            self.spawn_capacity,
        );
    }

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
        // Two words per transform slot: (mesh_id, material_id).
        2 * count as u64,
    )
    .expect("allocate GPURenderers buffer")
}

/// Allocate the host-mapped spawn staging: word 0 = count, word 1 = pad,
/// then `record_capacity` `(transform_id, mesh_id, material_id)` triples.
/// Sequential-write WC — one writer per frame, front-to-back. Count is
/// zeroed so frame slots recorded before the first `write_spawns` scatter
/// nothing.
fn alloc_spawn_staging(
    allocator: &Arc<StandardMemoryAllocator>,
    record_capacity: usize,
) -> Subbuffer<[u32]> {
    let buf = Buffer::new_slice::<u32>(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        (2 + 3 * record_capacity.max(1)) as u64,
    )
    .expect("allocate spawn staging buffer");
    {
        let mut w = buf.write().expect("zero-init spawn_staging");
        w[0] = 0;
        w[1] = 0;
    }
    buf
}

fn build_scatter_set(
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    pipeline: &Arc<ComputePipeline>,
    spawn_staging: &Subbuffer<[u32]>,
    renderers: &Subbuffer<[u32]>,
) -> Arc<DescriptorSet> {
    DescriptorSet::new(
        descriptor_set_allocator.clone(),
        pipeline.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, spawn_staging.clone()),
            WriteDescriptorSet::buffer(1, renderers.clone()),
        ],
        [],
    )
    .expect("GPURenderers scatter descriptor set")
}

/// Pre-record the spawn-scatter secondary: one dispatch over the staging
/// pair capacity; the shader early-outs past the in-buffer live count.
/// SimultaneousUse — captured by every in-flight FrameSlot primary.
fn record_scatter_secondary(
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
    pipeline: &Arc<ComputePipeline>,
    scatter_set: &Arc<DescriptorSet>,
    spawn_capacity: usize,
) -> Arc<SecondaryAutoCommandBuffer> {
    let groups = (spawn_capacity as u32).div_ceil(64).max(1);
    let mut builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::SimultaneousUse,
        CommandBufferInheritanceInfo::default(),
    )
    .expect("spawn scatter secondary builder");
    builder
        .bind_pipeline_compute(pipeline.clone())
        .expect("bind spawn scatter pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            scatter_set.clone(),
        )
        .expect("bind spawn scatter set");
    // Safety: dispatch derived from the staging capacity; shader bounds-
    // checks against the in-buffer count (host guarantees count ≤ capacity).
    unsafe {
        builder.dispatch([groups, 1, 1]).expect("dispatch spawn scatter");
    }
    builder.build().expect("build spawn scatter secondary")
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
