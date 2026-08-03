//! Device side of the retained-mode UI (ADR-0006).
//!
//! Four slot-indexed device-local SoT arrays — quad, style, group, order —
//! each with its own dirty bitmask, its own word-compaction prepass and its
//! own scatter dispatch. The prepass is `scatter_prepass.comp`, reused
//! verbatim from the transform pipeline: it only ever looks at bitmasks and
//! counts, never at element type. The scatter is `ui_scatter.comp`, a
//! sibling of `scatter.comp` rather than a refactor of it — see that
//! shader's header.
//!
//! # Frame shape
//!
//! One compute secondary per staging slot, executed at the front of the
//! FrameSlot primary alongside the transform scatters and **before**
//! `signal_cs`:
//!
//! ```text
//!   fill_buffer(compact_words[i].count, 0)  ×4
//!   prepass  ×4        — compact each array's nonzero dirty words
//!   fill_buffer(dirty[i], 0)  ×4            — bitmask consumed, clear it
//!   ui_build_args      — 4 dispatch args + the draw's instance count
//!   ui_scatter ×4      — dispatch_indirect, exact dirty-word count
//! ```
//!
//! and one graphics secondary, executed after the present blit inside a
//! `LoadOp::Load` render scope on the swapchain image, containing exactly
//! one `vkCmdDrawIndirect`. Both are recorded once and rebuilt only on a
//! capacity grow, a swapchain resize, or a texture arrival.
//!
//! Placement before `signal_cs` is load-bearing: the signal is the host's
//! guarantee that every read of host-visible staging has retired, and UI
//! staging is host-visible staging. The draw, by contrast, runs *after* the
//! signal — which is exactly why its instance count is promoted into a
//! device-local `VkDrawIndirectCommand` by `ui_build_args.comp` instead of
//! being read from the host buffer the count was written into.

use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferInheritanceInfo, CommandBufferInheritanceRenderingInfo, CommandBufferUsage,
        CopyBufferToImageInfo, DispatchIndirectCommand, DrawIndirectCommand,
        SecondaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    format::Format,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo,
        graphics::{
            color_blend::{
                AttachmentBlend, BlendFactor, ColorBlendAttachmentState, ColorBlendState,
            },
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::{PipelineRenderingCreateInfo, PipelineSubpassType},
            vertex_input::VertexInputState,
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint,
        PipelineLayout, PipelineShaderStageCreateInfo,
    },
    sync::GpuFuture,
};

use super::{font, OrderEntry, Record, UiCore, UiGroup, UiQuad, UiStyle};
use crate::{assets::GpuTextureStore, shaders, transform_gpu::dirty_word_count, STAGING_SLOTS};

/// Array indices, shared by every `[_; N_ARRAYS]` on this type. The order is
/// also the order of `ui_build_args.comp`'s bindings 0..3.
const QUAD: usize = 0;
const STYLE: usize = 1;
const GROUP: usize = 2;
const ORDER: usize = 3;
const N_ARRAYS: usize = 4;

/// `u32` words per element of each array, pushed to `ui_scatter.comp`.
const STRIDE: [usize; N_ARRAYS] = [
    UiQuad::STRIDE,
    UiStyle::STRIDE,
    UiGroup::STRIDE,
    OrderEntry::STRIDE,
];

/// Must match `scatter_prepass.comp`'s `local_size_x`.
const PREPASS_WORDS_PER_WORKGROUP: u32 = 64;

const INITIAL_PRIM_CAPACITY: usize = 1024;
const INITIAL_GROUP_CAPACITY: usize = 32;

/// One host-staging slot. Everything the host writes and the GPU reads in
/// the UI scatter block lives here, for the same reason as
/// `transform_gpu::StagingSlot`: a single-buffered member would re-impose
/// the frame `N-1` wait and give back the whole benefit of double buffering.
struct StagingSlot {
    stage: [Subbuffer<[u32]>; N_ARRAYS],
    dirty: [Subbuffer<[u32]>; N_ARRAYS],
    /// `[word_offset, word_count]` bounding each prepass's scan range.
    bounds: [Subbuffer<[u32]>; N_ARRAYS],
    prepass_args: Subbuffer<[DispatchIndirectCommand]>,
    /// Word 0: the live primitive count `ui_build_args.comp` promotes into
    /// the device-local draw command.
    counts: Subbuffer<[u32]>,

    /// Held only to keep the sets alive for as long as `secondary`, which
    /// baked them in, references them.
    #[allow(dead_code)]
    scatter_set: [Arc<DescriptorSet>; N_ARRAYS],
    #[allow(dead_code)]
    prepass_set: [Arc<DescriptorSet>; N_ARRAYS],
    #[allow(dead_code)]
    build_args_set: Arc<DescriptorSet>,
    secondary: Arc<SecondaryAutoCommandBuffer>,
}

pub struct UiGpu {
    /// Device-local source of truth: quad, style, group, order.
    sot: [Subbuffer<[u32]>; N_ARRAYS],
    /// Device-local scratch, produced and consumed inside one frame's
    /// scatter block, so *not* duplicated per staging slot.
    compact_words: [Subbuffer<[u32]>; N_ARRAYS],
    dispatch_args: Subbuffer<[DispatchIndirectCommand]>,
    /// Device-local `VkDrawIndirectCommand`, written every frame by
    /// `ui_build_args.comp`. The draw reads this and nothing host-visible.
    draw_args: Subbuffer<[DrawIndirectCommand]>,

    staging: [StagingSlot; STAGING_SLOTS],
    write_slot: usize,

    prim_capacity: usize,
    group_capacity: usize,

    scatter_pipeline: Arc<ComputePipeline>,
    prepass_pipeline: Arc<ComputePipeline>,
    build_args_pipeline: Arc<ComputePipeline>,
    draw_pipeline: Arc<GraphicsPipeline>,

    glyph_view: Arc<ImageView>,
    sampler: Arc<Sampler>,

    draw_set0: Arc<DescriptorSet>,
    draw_set1: Arc<DescriptorSet>,
    draw_secondary: Arc<SecondaryAutoCommandBuffer>,
    color_format: Format,
    extent: [u32; 2],

    /// Diagnostic: dirty-word span across the four arrays for the most
    /// recent frame. Reads zero on an idle frame, which is the property
    /// worth instrumenting.
    last_dirty_words: std::sync::atomic::AtomicU32,

    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
}

impl UiGpu {
    pub fn new(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        cb_allocator: Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
        texture_store: &GpuTextureStore,
        swapchain_format: Format,
        extent: [u32; 2],
    ) -> Self {
        let scatter_pipeline = compute_pipeline(
            device.clone(),
            shaders::ui_scatter_cs::load(device.clone()).expect("ui_scatter_cs load"),
        );
        let prepass_pipeline = compute_pipeline(
            device.clone(),
            shaders::scatter_prepass_cs::load(device.clone()).expect("scatter_prepass_cs load"),
        );
        let build_args_pipeline = compute_pipeline(
            device.clone(),
            shaders::ui_build_args_cs::load(device.clone()).expect("ui_build_args_cs load"),
        );
        let draw_pipeline = build_draw_pipeline(device.clone(), swapchain_format);

        let prim_capacity = INITIAL_PRIM_CAPACITY;
        let group_capacity = INITIAL_GROUP_CAPACITY;
        let counts = element_counts(prim_capacity, group_capacity);

        let sot = std::array::from_fn(|i| device_words(&memory_allocator, counts[i] * STRIDE[i]));
        let compact_words = std::array::from_fn(|i| alloc_compact_words(&memory_allocator, counts[i]));
        let dispatch_args = alloc_dispatch_args(&memory_allocator);
        let draw_args = alloc_draw_args(&memory_allocator);

        let (glyph_view, sampler) = build_glyph_atlas(
            &memory_allocator,
            &cb_allocator,
            &queue,
            device.clone(),
        );

        let draw_set0 = build_draw_set0(&descriptor_set_allocator, &draw_pipeline, &sot);
        let draw_set1 = build_draw_set1(
            &descriptor_set_allocator,
            &draw_pipeline,
            &glyph_view,
            &sampler,
            texture_store,
        );
        let draw_secondary = record_draw_secondary(
            &cb_allocator,
            queue.queue_family_index(),
            &draw_pipeline,
            &draw_set0,
            &draw_set1,
            &draw_args,
            swapchain_format,
            extent,
        );

        let staging = make_staging_slots(
            &memory_allocator,
            &descriptor_set_allocator,
            &cb_allocator,
            queue.queue_family_index(),
            &counts,
            &sot,
            &compact_words,
            &dispatch_args,
            &draw_args,
            &scatter_pipeline,
            &prepass_pipeline,
            &build_args_pipeline,
        );

        Self {
            sot,
            compact_words,
            dispatch_args,
            draw_args,
            staging,
            write_slot: 0,
            prim_capacity,
            group_capacity,
            scatter_pipeline,
            prepass_pipeline,
            build_args_pipeline,
            draw_pipeline,
            glyph_view,
            sampler,
            draw_set0,
            draw_set1,
            draw_secondary,
            color_format: swapchain_format,
            extent,
            last_dirty_words: std::sync::atomic::AtomicU32::new(0),
            memory_allocator,
            descriptor_set_allocator,
            cb_allocator,
            queue,
        }
    }

    /// The UI scatter secondary for a staging slot. Executed at the front of
    /// that slot's FrameSlot primary, before `signal_cs`.
    pub fn scatter_secondary(&self, slot: usize) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.staging[slot].secondary
    }

    /// The single-`draw_indirect` graphics secondary. Executed inside a
    /// `LoadOp::Load` render scope on the swapchain image, after the blit.
    pub fn draw_secondary(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.draw_secondary
    }

    pub fn write_slot(&self) -> usize {
        self.write_slot
    }

    /// Flip to the other staging slot. Must be called in lockstep with
    /// `WorldTransformGpu::advance_staging_slot` — one FrameSlot primary
    /// binds both, so a drift would have that CB read one subsystem's fresh
    /// slot and another's stale one.
    pub fn advance_staging_slot(&mut self) {
        self.write_slot = (self.write_slot + 1) % STAGING_SLOTS;
    }

    /// Grow the device arrays to cover `core`'s live counts. Returns `true`
    /// if anything was reallocated, in which case every FrameSlot primary
    /// must be rebuilt (they capture the scatter and draw secondaries).
    ///
    /// Everything in `core` is re-marked on a grow: the fresh SoT holds
    /// nothing, so the mirror's "already equal" answers would be lies.
    pub fn ensure_capacity(&mut self, core: &mut UiCore) -> bool {
        let prims = core.prim_count() as usize;
        let groups = core.group_count() as usize;
        if prims <= self.prim_capacity && groups <= self.group_capacity {
            return false;
        }
        // Per axis: a UI that added a panel shouldn't double its primitive
        // capacity as a side effect.
        if prims > self.prim_capacity {
            self.prim_capacity = prims.max(self.prim_capacity * 2);
        }
        if groups > self.group_capacity {
            self.group_capacity = groups.max(self.group_capacity * 2);
        }

        let counts = element_counts(self.prim_capacity, self.group_capacity);
        self.sot =
            std::array::from_fn(|i| device_words(&self.memory_allocator, counts[i] * STRIDE[i]));
        self.compact_words =
            std::array::from_fn(|i| alloc_compact_words(&self.memory_allocator, counts[i]));

        self.draw_set0 =
            build_draw_set0(&self.descriptor_set_allocator, &self.draw_pipeline, &self.sot);
        self.rebuild_staging();
        self.rebuild_draw_secondary();
        core.mark_all();
        true
    }

    /// Re-record the draw secondary against a new swapchain extent (the
    /// px → NDC push constant and the viewport are both baked into it).
    pub fn on_resize(&mut self, extent: [u32; 2]) {
        if self.extent == extent {
            return;
        }
        self.extent = extent;
        self.rebuild_draw_secondary();
    }

    /// Re-bind the bindless texture array after a `GpuTextureStore::sync`
    /// reported an arrival. Callers fold the resulting rebuild into the
    /// engine's existing `force_full` path.
    pub fn refresh_textures(&mut self, texture_store: &GpuTextureStore) {
        self.draw_set1 = build_draw_set1(
            &self.descriptor_set_allocator,
            &self.draw_pipeline,
            &self.glyph_view,
            &self.sampler,
            texture_store,
        );
        self.rebuild_draw_secondary();
    }

    /// Publish this frame's changes into the current staging slot: the dirty
    /// elements themselves, the dirty bitmasks, each prepass's scan bounds
    /// and group count, and the live primitive count.
    ///
    /// Must run **every** frame (a quiet frame writes zero-workgroup
    /// dispatch args, retiring the previous occupant of this slot) and only
    /// after `WorldTransformGpu::host_wait_for_previous_compute` — the
    /// `gpu_signal` gate is what covers these buffers' in-CB reads.
    pub fn write_staging(&self, core: &mut UiCore) {
        let slot = &self.staging[self.write_slot];
        let mut bounds = [(0i64, -1i64); N_ARRAYS];

        {
            let mut s = slot.stage[QUAD].write().expect("ui stage quad");
            let mut d = slot.dirty[QUAD].write().expect("ui dirty quad");
            bounds[QUAD] = core.quad.upload(&mut s, &mut d);
        }
        {
            let mut s = slot.stage[STYLE].write().expect("ui stage style");
            let mut d = slot.dirty[STYLE].write().expect("ui dirty style");
            bounds[STYLE] = core.style.upload(&mut s, &mut d);
        }
        {
            let mut s = slot.stage[GROUP].write().expect("ui stage group");
            let mut d = slot.dirty[GROUP].write().expect("ui dirty group");
            bounds[GROUP] = core.group.upload(&mut s, &mut d);
        }
        {
            let mut s = slot.stage[ORDER].write().expect("ui stage order");
            let mut d = slot.dirty[ORDER].write().expect("ui dirty order");
            bounds[ORDER] = core.order.upload(&mut s, &mut d);
        }

        let mut args = slot.prepass_args.write().expect("ui prepass_args");
        for (i, &(min_word, max_word)) in bounds.iter().enumerate() {
            let (offset, count) = if max_word < 0 {
                (0, 0)
            } else {
                (min_word as u32, (max_word - min_word + 1) as u32)
            };
            let mut b = slot.bounds[i].write().expect("ui prepass bounds");
            b[0] = offset;
            b[1] = count;
            args[i] = DispatchIndirectCommand {
                x: count.div_ceil(PREPASS_WORDS_PER_WORKGROUP),
                y: 1,
                z: 1,
            };
        }
        drop(args);

        slot.counts.write().expect("ui counts")[0] = core.prim_count();
        self.last_dirty_words.store(
            bounds
                .iter()
                .map(|&(mn, mx)| if mx < 0 { 0 } else { (mx - mn + 1) as u32 })
                .sum(),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Dirty-word span across the four arrays for the frame most recently
    /// staged. Zero on an idle frame, which is the property worth watching.
    pub fn last_dirty_words(&self) -> u32 {
        self.last_dirty_words
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    // ── Internal ────────────────────────────────────────────────────

    /// Rebuild **both** staging slots. Every grow path touches both: the two
    /// are used on alternating frames and must stay interchangeable, so
    /// rebuilding one would leave the next frame's secondary bound to
    /// dropped buffers.
    fn rebuild_staging(&mut self) {
        self.staging = make_staging_slots(
            &self.memory_allocator,
            &self.descriptor_set_allocator,
            &self.cb_allocator,
            self.queue.queue_family_index(),
            &element_counts(self.prim_capacity, self.group_capacity),
            &self.sot,
            &self.compact_words,
            &self.dispatch_args,
            &self.draw_args,
            &self.scatter_pipeline,
            &self.prepass_pipeline,
            &self.build_args_pipeline,
        );
    }

    fn rebuild_draw_secondary(&mut self) {
        self.draw_secondary = record_draw_secondary(
            &self.cb_allocator,
            self.queue.queue_family_index(),
            &self.draw_pipeline,
            &self.draw_set0,
            &self.draw_set1,
            &self.draw_args,
            self.color_format,
            self.extent,
        );
    }
}

/// Elements per array at a given capacity — quad / style / order are
/// per-primitive, group is per-group.
fn element_counts(prims: usize, groups: usize) -> [usize; N_ARRAYS] {
    [prims.max(1), prims.max(1), groups.max(1), prims.max(1)]
}

// ─────────────────────────────────────────────────────────────────────
// Allocation
// ─────────────────────────────────────────────────────────────────────

fn device_words(allocator: &Arc<StandardMemoryAllocator>, words: usize) -> Subbuffer<[u32]> {
    Buffer::new_slice::<u32>(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        words.max(1) as u64,
    )
    .expect("allocate UI SoT buffer")
}

/// Host-mapped staging. Sequential-write WC is right for all of it: one
/// writer per frame, and the dirty bitmask is written whole-word (never
/// read-modify-written on the host — `SlotArray` keeps its own mirror of the
/// bits and publishes them, so WC memory is never read back).
fn host_words(
    allocator: &Arc<StandardMemoryAllocator>,
    words: usize,
    extra: BufferUsage,
) -> Subbuffer<[u32]> {
    let buf = Buffer::new_slice::<u32>(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | extra,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        words.max(1) as u64,
    )
    .expect("allocate UI staging buffer");
    buf.write().expect("zero-init UI staging").fill(0);
    buf
}

/// `{count, pad, uvec2 entries[]}`, sized for the worst case of every word
/// dirty. `count` is reset by an in-CB `fill_buffer` each frame, so no
/// host-side zero-init is needed.
fn alloc_compact_words(
    allocator: &Arc<StandardMemoryAllocator>,
    elements: usize,
) -> Subbuffer<[u32]> {
    Buffer::new_slice::<u32>(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        (2 + 2 * dirty_word_count(elements)) as u64,
    )
    .expect("allocate UI compact_words buffer")
}

fn alloc_dispatch_args(
    allocator: &Arc<StandardMemoryAllocator>,
) -> Subbuffer<[DispatchIndirectCommand]> {
    Buffer::new_slice::<DispatchIndirectCommand>(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::INDIRECT_BUFFER | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        N_ARRAYS as u64,
    )
    .expect("allocate UI dispatch args")
}

/// Device-local so nothing the draw reads lives in memory the host is free
/// to overwrite the moment `signal_cs` fires. Never touched by the host and
/// never zero-initialised: `ui_build_args.comp` runs earlier in the same
/// primary than the draw does, so this is always written before it is read.
fn alloc_draw_args(
    allocator: &Arc<StandardMemoryAllocator>,
) -> Subbuffer<[DrawIndirectCommand]> {
    Buffer::new_slice::<DrawIndirectCommand>(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::INDIRECT_BUFFER | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        1,
    )
    .expect("allocate UI draw args")
}

fn alloc_prepass_args(
    allocator: &Arc<StandardMemoryAllocator>,
) -> Subbuffer<[DispatchIndirectCommand]> {
    let buf = Buffer::new_slice::<DispatchIndirectCommand>(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::INDIRECT_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        N_ARRAYS as u64,
    )
    .expect("allocate UI prepass args");
    {
        let mut w = buf.write().expect("zero-init UI prepass args");
        w.fill(DispatchIndirectCommand { x: 0, y: 1, z: 1 });
    }
    buf
}

// ─────────────────────────────────────────────────────────────────────
// Glyph atlas
// ─────────────────────────────────────────────────────────────────────

/// Upload the built-in font's atlas once, via a fence-waited one-shot. The
/// image is a **dedicated** `R8_UNORM` sampled image with a stable handle,
/// deliberately not a slot in `GpuTextureStore`: that store's `sync`
/// returning `changed` triggers a descriptor-set + secondary + frame-slot
/// rebuild, and a font has no business dragging every command buffer in the
/// engine through one.
fn build_glyph_atlas(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
    device: Arc<Device>,
) -> (Arc<ImageView>, Arc<Sampler>) {
    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8_UNORM,
            extent: [font::ATLAS_W, font::ATLAS_H, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .expect("allocate glyph atlas image");

    let staging = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        font::rasterize_atlas(),
    )
    .expect("allocate glyph atlas staging");

    let mut builder = AutoCommandBufferBuilder::primary(
        cb_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("glyph atlas upload CB");
    builder
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, image.clone()))
        .expect("record glyph atlas upload");
    vulkano::sync::now(device.clone())
        .then_execute(queue.clone(), builder.build().expect("build atlas CB"))
        .expect("submit glyph atlas upload")
        .then_signal_fence_and_flush()
        .expect("flush glyph atlas upload")
        .wait(None)
        .expect("await glyph atlas upload");

    let sampler = Sampler::new(
        device,
        SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::ClampToEdge; 3],
            ..Default::default()
        },
    )
    .expect("glyph atlas sampler");

    (
        ImageView::new_default(image).expect("glyph atlas view"),
        sampler,
    )
}

// ─────────────────────────────────────────────────────────────────────
// Descriptor sets
// ─────────────────────────────────────────────────────────────────────

fn build_draw_set0(
    allocator: &Arc<StandardDescriptorSetAllocator>,
    pipeline: &Arc<GraphicsPipeline>,
    sot: &[Subbuffer<[u32]>; N_ARRAYS],
) -> Arc<DescriptorSet> {
    DescriptorSet::new(
        allocator.clone(),
        pipeline.layout().set_layouts()[0].clone(),
        (0..N_ARRAYS).map(|i| WriteDescriptorSet::buffer(i as u32, sot[i].clone())),
        [],
    )
    .expect("UI draw set 0")
}

fn build_draw_set1(
    allocator: &Arc<StandardDescriptorSetAllocator>,
    pipeline: &Arc<GraphicsPipeline>,
    glyph_view: &Arc<ImageView>,
    sampler: &Arc<Sampler>,
    texture_store: &GpuTextureStore,
) -> Arc<DescriptorSet> {
    DescriptorSet::new(
        allocator.clone(),
        pipeline.layout().set_layouts()[1].clone(),
        [
            WriteDescriptorSet::image_view_sampler(0, glyph_view.clone(), sampler.clone()),
            WriteDescriptorSet::image_view_sampler_array(1, 0, texture_store.descriptor_array()),
        ],
        [],
    )
    .expect("UI draw set 1")
}

// ─────────────────────────────────────────────────────────────────────
// Staging slot construction
// ─────────────────────────────────────────────────────────────────────

/// Build **both** staging slots. Every construction and grow path goes
/// through here: the two are used on alternating frames and must stay
/// interchangeable, so building one and not the other would leave the next
/// frame's secondary bound to dropped buffers.
#[allow(clippy::too_many_arguments)]
fn make_staging_slots(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
    counts: &[usize; N_ARRAYS],
    sot: &[Subbuffer<[u32]>; N_ARRAYS],
    compact_words: &[Subbuffer<[u32]>; N_ARRAYS],
    dispatch_args: &Subbuffer<[DispatchIndirectCommand]>,
    draw_args: &Subbuffer<[DrawIndirectCommand]>,
    scatter_pipeline: &Arc<ComputePipeline>,
    prepass_pipeline: &Arc<ComputePipeline>,
    build_args_pipeline: &Arc<ComputePipeline>,
) -> [StagingSlot; STAGING_SLOTS] {
    std::array::from_fn(|_| {
        build_staging_slot(
            memory_allocator,
            descriptor_set_allocator,
            cb_allocator,
            queue_family_index,
            counts,
            sot,
            compact_words,
            dispatch_args,
            draw_args,
            scatter_pipeline,
            prepass_pipeline,
            build_args_pipeline,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn build_staging_slot(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
    counts: &[usize; N_ARRAYS],
    sot: &[Subbuffer<[u32]>; N_ARRAYS],
    compact_words: &[Subbuffer<[u32]>; N_ARRAYS],
    dispatch_args: &Subbuffer<[DispatchIndirectCommand]>,
    draw_args: &Subbuffer<[DrawIndirectCommand]>,
    scatter_pipeline: &Arc<ComputePipeline>,
    prepass_pipeline: &Arc<ComputePipeline>,
    build_args_pipeline: &Arc<ComputePipeline>,
) -> StagingSlot {
    let stage: [_; N_ARRAYS] =
        std::array::from_fn(|i| host_words(memory_allocator, counts[i] * STRIDE[i], BufferUsage::empty()));
    let dirty: [_; N_ARRAYS] = std::array::from_fn(|i| {
        host_words(
            memory_allocator,
            dirty_word_count(counts[i]),
            BufferUsage::TRANSFER_DST,
        )
    });
    let bounds: [_; N_ARRAYS] =
        std::array::from_fn(|_| host_words(memory_allocator, 2, BufferUsage::empty()));
    let prepass_args = alloc_prepass_args(memory_allocator);
    let counts_buf = host_words(memory_allocator, 1, BufferUsage::empty());

    let scatter_set: [_; N_ARRAYS] = std::array::from_fn(|i| {
        DescriptorSet::new(
            descriptor_set_allocator.clone(),
            scatter_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, compact_words[i].clone()),
                WriteDescriptorSet::buffer(1, stage[i].clone()),
                WriteDescriptorSet::buffer(2, sot[i].clone()),
            ],
            [],
        )
        .expect("UI scatter set")
    });
    let prepass_set: [_; N_ARRAYS] = std::array::from_fn(|i| {
        DescriptorSet::new(
            descriptor_set_allocator.clone(),
            prepass_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, bounds[i].clone()),
                WriteDescriptorSet::buffer(1, dirty[i].clone()),
                WriteDescriptorSet::buffer(2, compact_words[i].clone()),
            ],
            [],
        )
        .expect("UI prepass set")
    });
    let build_args_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        build_args_pipeline.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, compact_words[QUAD].clone()),
            WriteDescriptorSet::buffer(1, compact_words[STYLE].clone()),
            WriteDescriptorSet::buffer(2, compact_words[GROUP].clone()),
            WriteDescriptorSet::buffer(3, compact_words[ORDER].clone()),
            WriteDescriptorSet::buffer(4, counts_buf.clone()),
            WriteDescriptorSet::buffer(5, dispatch_args.clone()),
            WriteDescriptorSet::buffer(6, draw_args.clone()),
        ],
        [],
    )
    .expect("UI build-args set");

    let secondary = record_scatter_secondary(
        cb_allocator,
        queue_family_index,
        counts,
        &dirty,
        compact_words,
        &prepass_args,
        dispatch_args,
        scatter_pipeline,
        prepass_pipeline,
        build_args_pipeline,
        &scatter_set,
        &prepass_set,
        &build_args_set,
    );

    StagingSlot {
        stage,
        dirty,
        bounds,
        prepass_args,
        counts: counts_buf,
        scatter_set,
        prepass_set,
        build_args_set,
        secondary,
    }
}

#[allow(clippy::too_many_arguments)]
fn record_scatter_secondary(
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
    counts: &[usize; N_ARRAYS],
    dirty: &[Subbuffer<[u32]>; N_ARRAYS],
    compact_words: &[Subbuffer<[u32]>; N_ARRAYS],
    prepass_args: &Subbuffer<[DispatchIndirectCommand]>,
    dispatch_args: &Subbuffer<[DispatchIndirectCommand]>,
    scatter_pipeline: &Arc<ComputePipeline>,
    prepass_pipeline: &Arc<ComputePipeline>,
    build_args_pipeline: &Arc<ComputePipeline>,
    scatter_set: &[Arc<DescriptorSet>; N_ARRAYS],
    prepass_set: &[Arc<DescriptorSet>; N_ARRAYS],
    build_args_set: &Arc<DescriptorSet>,
) -> Arc<SecondaryAutoCommandBuffer> {
    // SimultaneousUse: captured by every FrameSlot primary, several of which
    // can be in flight at once. Same reasoning as the TRS scatter secondary.
    let mut builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::SimultaneousUse,
        CommandBufferInheritanceInfo::default(),
    )
    .expect("UI scatter secondary builder");

    // ── Stage 1: word-compaction prepass, one per array ────────────────
    builder
        .bind_pipeline_compute(prepass_pipeline.clone())
        .expect("bind UI prepass pipeline");
    for i in 0..N_ARRAYS {
        builder
            .fill_buffer(compact_words[i].clone().slice(0..1), 0)
            .expect("reset UI compact_words count")
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                prepass_pipeline.layout().clone(),
                0,
                prepass_set[i].clone(),
            )
            .expect("bind UI prepass set");
        // Safety: `prepass_args[i]` is written by `write_staging` every
        // frame under the same `gpu_signal` gate as every other host-shared
        // buffer this secondary reads, and always spans every word between
        // this array's lowest and highest dirty word. The shader
        // bounds-checks its own trailing wavefront against `word_count`.
        unsafe {
            builder
                .dispatch_indirect(prepass_args.clone().slice(i as u64..i as u64 + 1))
                .expect("dispatch_indirect UI prepass");
        }
    }

    // ── Stage 2: the bitmask has been consumed; clear it ───────────────
    // vulkano auto-syncs each `fill_buffer` against the prepass that read
    // the same buffer (SHADER_READ → TRANSFER_WRITE). Recorded here rather
    // than in the FrameSlot primary — unlike the TRS masks, nothing outside
    // this secondary touches them.
    for d in dirty {
        builder
            .fill_buffer(d.clone(), 0)
            .expect("clear UI dirty bitmask");
    }

    // ── Stage 3: dispatch args + the draw's instance count ─────────────
    builder
        .bind_pipeline_compute(build_args_pipeline.clone())
        .expect("bind UI build-args pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            build_args_pipeline.layout().clone(),
            0,
            build_args_set.clone(),
        )
        .expect("bind UI build-args set");
    // Safety: 1×1×1 dispatch is unconditionally valid.
    unsafe {
        builder.dispatch([1, 1, 1]).expect("dispatch UI build-args");
    }

    // ── Stage 4: the real scatter, one per array ───────────────────────
    let layout = scatter_pipeline.layout().clone();
    builder
        .bind_pipeline_compute(scatter_pipeline.clone())
        .expect("bind UI scatter pipeline");
    for i in 0..N_ARRAYS {
        builder
            .push_constants(
                layout.clone(),
                0,
                shaders::ui_scatter_cs::PC {
                    elem_count: counts[i] as u32,
                    stride_words: STRIDE[i] as u32,
                },
            )
            .expect("push UI scatter constants")
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                layout.clone(),
                0,
                scatter_set[i].clone(),
            )
            .expect("bind UI scatter set");
        // Safety: `dispatch_args[i]` was written by stage 3 earlier in this
        // same secondary, from this frame's exact compacted dirty-word
        // count. The `elem_count` push constant bounds-checks the shader's
        // trailing wavefront.
        unsafe {
            builder
                .dispatch_indirect(dispatch_args.clone().slice(i as u64..i as u64 + 1))
                .expect("dispatch_indirect UI scatter");
        }
    }

    builder.build().expect("build UI scatter secondary")
}

fn record_draw_secondary(
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
    pipeline: &Arc<GraphicsPipeline>,
    set0: &Arc<DescriptorSet>,
    set1: &Arc<DescriptorSet>,
    draw_args: &Subbuffer<[DrawIndirectCommand]>,
    color_format: Format,
    extent: [u32; 2],
) -> Arc<SecondaryAutoCommandBuffer> {
    let inheritance = CommandBufferInheritanceInfo {
        render_pass: Some(
            CommandBufferInheritanceRenderingInfo {
                color_attachment_formats: vec![Some(color_format)],
                ..Default::default()
            }
            .into(),
        ),
        ..Default::default()
    };

    let mut builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::SimultaneousUse,
        inheritance,
    )
    .expect("UI draw secondary builder");

    builder
        .set_viewport(
            0,
            smallvec::smallvec![Viewport {
                offset: [0.0, 0.0],
                extent: [extent[0] as f32, extent[1] as f32],
                depth_range: 0.0..=1.0,
            }],
        )
        .expect("UI set_viewport")
        .bind_pipeline_graphics(pipeline.clone())
        .expect("bind UI pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            pipeline.layout().clone(),
            0,
            (set0.clone(), set1.clone()),
        )
        .expect("bind UI descriptor sets")
        .push_constants(
            pipeline.layout().clone(),
            0,
            shaders::ui_vs::PC {
                screen: [extent[0] as f32, extent[1] as f32],
            },
        )
        .expect("push UI screen extent");

    // Safety: `draw_args` is device-local and written by `ui_build_args.comp`
    // earlier in the same primary, always to a count ≤ the SoT capacity the
    // bound descriptor set covers. `vertex_count` is the constant 4.
    unsafe {
        builder
            .draw_indirect(draw_args.clone())
            .expect("UI draw_indirect");
    }

    builder.build().expect("build UI draw secondary")
}

// ─────────────────────────────────────────────────────────────────────
// Pipelines
// ─────────────────────────────────────────────────────────────────────

fn compute_pipeline(
    device: Arc<Device>,
    cs: Arc<vulkano::shader::ShaderModule>,
) -> Arc<ComputePipeline> {
    let stage = PipelineShaderStageCreateInfo::new(cs.entry_point("main").expect("entry point"));
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(&stage))
            .into_pipeline_layout_create_info(device.clone())
            .expect("UI compute pipeline layout info"),
    )
    .expect("UI compute pipeline layout");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("UI ComputePipeline::new")
}

/// The UI graphics pipeline: no vertex input, four-vertex triangle strips,
/// no depth, premultiplied alpha.
///
/// `(ONE, ONE_MINUS_SRC_ALPHA)` is the only blend mode that composites text
/// coverage and nested translucent panels correctly, and the attachment
/// format is the swapchain's `_SRGB`, so the hardware blends in linear space
/// and encodes on write.
fn build_draw_pipeline(device: Arc<Device>, color_format: Format) -> Arc<GraphicsPipeline> {
    let vs = shaders::ui_vs::load(device.clone()).expect("ui_vs load");
    let fs = shaders::ui_fs::load(device.clone()).expect("ui_fs load");
    let stages = [
        PipelineShaderStageCreateInfo::new(vs.entry_point("main").expect("ui_vs entry")),
        PipelineShaderStageCreateInfo::new(fs.entry_point("main").expect("ui_fs entry")),
    ];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .expect("UI pipeline layout info"),
    )
    .expect("UI pipeline layout");

    GraphicsPipeline::new(
        device,
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(VertexInputState::default()),
            input_assembly_state: Some(InputAssemblyState {
                topology: PrimitiveTopology::TriangleStrip,
                ..Default::default()
            }),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                1,
                ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend {
                        src_color_blend_factor: BlendFactor::One,
                        dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                        src_alpha_blend_factor: BlendFactor::One,
                        dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(PipelineSubpassType::BeginRendering(
                PipelineRenderingCreateInfo {
                    color_attachment_formats: vec![Some(color_format)],
                    ..Default::default()
                },
            )),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .expect("UI GraphicsPipeline::new")
}
