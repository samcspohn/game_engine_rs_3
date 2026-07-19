//! Render-side camera (Design B — GPU-driven). Owns the GPU attachments
//! (color + depth) a camera renders into, the device-side MVP matrix buffer +
//! graphics descriptor set the draws read from, the per-slot indirect-command
//! buffers, the **cull secondary** (the compute pass that frustum-tests every
//! renderer and compacts visible MVPs), and the **scene secondary** (the
//! single `multiDrawIndexedIndirect`).
//!
//! There is no CPU-sorted topology. The cull pass dispatches over the whole
//! renderer/transform range and reads the world's `GPURenderers`
//! (`transform → mesh_id`), the registry `redirect` (`mesh_id → slot`), and the
//! `mesh_table` (per-slot bounds), writing each visible instance's MVP into its
//! slot's contiguous region. The CPU only supplies a small per-slot
//! [`DrawPlan`] (geometry + prefix-summed `first_instance` bases) that changes
//! on spawn / load — never an `O(N)` sort.
//!
//! Invalidation axes: per-camera resolution (attachments + scene secondary),
//! draw plan / capacity (MVP + indirect + cull/scene secondaries), world
//! capacity (the cull set binds SoT / `GPURenderers` / redirect / mesh_table,
//! so it rebinds when those reallocate), and per-swapchain-image (on
//! `FrameSlot`).

use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferInheritanceInfo, CommandBufferInheritanceRenderingInfo, CommandBufferUsage,
        CopyBufferInfo, DrawIndexedIndirectCommand, SecondaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{graphics::viewport::Viewport, GraphicsPipeline, PipelineBindPoint},
};

// `Pipeline` trait is needed for `pipeline.layout()` method resolution.
use vulkano::pipeline::Pipeline;

use crate::assets::{GpuMeshStore, GpuTextureStore};
use crate::gpu_renderers::GpuRenderers;
use crate::shaders;
use crate::transform_gpu::WorldTransformGpu;

/// Pixel format used for camera-owned offscreen color targets.
pub const CAMERA_COLOR_FORMAT: Format = Format::R16G16B16A16_SFLOAT;

/// Pixel format used for camera-owned depth targets.
pub const CAMERA_DEPTH_FORMAT: Format = Format::D32_SFLOAT;

/// How a camera's attachment extent is determined relative to the swapchain.
#[derive(Clone, Copy, Debug)]
pub enum CameraResolution {
    /// Track the swapchain extent 1:1.
    MatchSwapchain,
}

impl CameraResolution {
    fn resolve(&self, swapchain_extent: [u32; 2]) -> [u32; 2] {
        match self {
            CameraResolution::MatchSwapchain => swapchain_extent,
        }
    }

    /// Does this policy depend on the swapchain extent?
    pub fn depends_on_swapchain(&self) -> bool {
        match self {
            CameraResolution::MatchSwapchain => true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DrawPlan
// ─────────────────────────────────────────────────────────────────────────────

/// The CPU-computed per-frame-static draw description: one indirect command
/// per drawable slot (geometry offsets + prefix-summed `first_instance` base,
/// `instance_count` pre-zeroed for the cull to accumulate), plus the total
/// renderer count (the MVP buffer size). Rebuilt on topology change — `O(#slots)`.
#[derive(Clone)]
pub struct DrawPlan {
    pub commands: Vec<DrawIndexedIndirectCommand>,
    pub total_renderers: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// CameraSceneResources
// ─────────────────────────────────────────────────────────────────────────────

/// Per-call bundle of GPU/scene state the camera needs to (re)build its draw
/// resources. Nothing is owned beyond the call.
pub struct CameraSceneResources<'a> {
    pub cb_allocator: &'a Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: &'a Arc<StandardDescriptorSetAllocator>,
    pub memory_allocator: &'a Arc<StandardMemoryAllocator>,
    pub pipeline: &'a Arc<GraphicsPipeline>,
    pub queue_family_index: u32,
    /// SoT TRS + view_proj + the cull (a.k.a. mvp_build) pipeline + set 1.
    pub world_transforms: &'a WorldTransformGpu,
    /// Mega buffers + redirect + mesh table.
    pub mesh_store: &'a GpuMeshStore,
    /// Sampled texture images + texture redirect (graphics set 1).
    pub texture_store: &'a GpuTextureStore,
    /// Per-transform `GPURenderers` buffer (`transform → mesh_id`).
    pub gpu_renderers: &'a GpuRenderers,
}

// ─────────────────────────────────────────────────────────────────────────────
// RenderCamera
// ─────────────────────────────────────────────────────────────────────────────

pub struct RenderCamera {
    resolution: CameraResolution,
    extent: [u32; 2],
    color_image: Arc<Image>,
    depth_image: Arc<Image>,
    color_view: Arc<ImageView>,
    depth_view: Arc<ImageView>,

    /// Device-local MVP buffer (cull writes, vertex shader reads via
    /// `descriptor_set`). Sized to the total renderer count.
    device_matrices: Subbuffer<[[f32; 16]]>,
    /// Graphics set 0 — references `device_matrices`.
    descriptor_set: Arc<DescriptorSet>,
    /// Graphics set 1 — texture redirect + per-slot texture ids + the
    /// sampled-image array. Rebuilt by `ensure_current` (texture arrivals
    /// ride the `force_full` path).
    texture_set: Arc<DescriptorSet>,
    /// Single `multiDrawIndexedIndirect` over `indirect_args[0..slot_count]`.
    scene_secondary: Arc<SecondaryAutoCommandBuffer>,

    /// Host-visible template: per-slot commands with `instance_count` zeroed.
    /// Copied into `indirect_args` each frame to reset the counts.
    indirect_template: Subbuffer<[DrawIndexedIndirectCommand]>,
    /// Device-local indirect commands (cull-written `instance_count`, draw
    /// source).
    indirect_args: Subbuffer<[DrawIndexedIndirectCommand]>,

    /// Cull set 0 — SoT, GPURenderers, redirect, mesh_table, MVP, indirect.
    cull_set: Arc<DescriptorSet>,
    /// Cull secondary: reset copy (`template → args`) + the cull dispatch over
    /// the renderer range.
    cull_secondary: Arc<SecondaryAutoCommandBuffer>,

    /// Allocated `[f32; 16]` MVP slots (≥ total renderers).
    mvp_capacity: usize,
    /// Allocated indirect-command slots (≥ `slot_count`).
    slot_capacity: usize,
    /// Number of drawable slots baked into the scene secondary's drawCount.
    slot_count: usize,
    /// Renderer range baked into the cull dispatch (== world entity capacity).
    cull_range: usize,
}

impl RenderCamera {
    pub fn new_match_swapchain(
        swapchain_extent: [u32; 2],
        scene: &CameraSceneResources<'_>,
        plan: &DrawPlan,
        renderer_capacity: usize,
    ) -> Self {
        Self::new(
            CameraResolution::MatchSwapchain,
            swapchain_extent,
            scene,
            plan,
            renderer_capacity,
        )
    }

    pub fn new(
        resolution: CameraResolution,
        swapchain_extent: [u32; 2],
        scene: &CameraSceneResources<'_>,
        plan: &DrawPlan,
        renderer_capacity: usize,
    ) -> Self {
        let extent = resolution.resolve(swapchain_extent);
        let (color_image, color_view, depth_image, depth_view) =
            allocate_attachments(scene.memory_allocator, extent);

        let slot_count = plan.commands.len();
        let mvp_capacity = (plan.total_renderers as usize).max(1);
        let slot_capacity = slot_count.max(1);

        let (device_matrices, descriptor_set) = allocate_matrices_and_set(
            scene.memory_allocator,
            scene.descriptor_set_allocator,
            scene.pipeline,
            mvp_capacity,
        );
        let (indirect_template, indirect_args) =
            allocate_indirect_buffers(scene.memory_allocator, slot_capacity);
        write_indirect_template(&indirect_template, &plan.commands);

        let cull_set = build_cull_set(scene, &device_matrices, &indirect_args);
        let cull_secondary = record_cull_secondary(
            scene,
            &indirect_template,
            &indirect_args,
            &cull_set,
            renderer_capacity as u32,
        );
        let texture_set = build_texture_set(scene);
        let scene_secondary = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &descriptor_set,
            &texture_set,
            scene.mesh_store,
            &indirect_args,
            slot_count,
            extent,
        );

        RenderCamera {
            resolution,
            extent,
            color_image,
            depth_image,
            color_view,
            depth_view,
            device_matrices,
            descriptor_set,
            texture_set,
            scene_secondary,
            indirect_template,
            indirect_args,
            cull_set,
            cull_secondary,
            mvp_capacity,
            slot_capacity,
            slot_count,
            cull_range: renderer_capacity,
        }
    }

    /// Swapchain resized. Re-creates attachments + re-records the scene
    /// secondary (its viewport depends on extent). The cull resources are
    /// extent-independent and survive. Returns `true` if anything was rebuilt.
    pub fn on_swapchain_resize(
        &mut self,
        new_swapchain_extent: [u32; 2],
        scene: &CameraSceneResources<'_>,
    ) -> bool {
        if !self.resolution.depends_on_swapchain() {
            return false;
        }
        let new_extent = self.resolution.resolve(new_swapchain_extent);
        if new_extent == self.extent {
            return false;
        }
        let (color_image, color_view, depth_image, depth_view) =
            allocate_attachments(scene.memory_allocator, new_extent);
        self.extent = new_extent;
        self.color_image = color_image;
        self.color_view = color_view;
        self.depth_image = depth_image;
        self.depth_view = depth_view;

        self.scene_secondary = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &self.descriptor_set,
            &self.texture_set,
            scene.mesh_store,
            &self.indirect_args,
            self.slot_count,
            self.extent,
        );
        true
    }

    /// Rebuild the per-frame-static draw resources for the current draw plan +
    /// renderer capacity. Grows the MVP / indirect buffers geometrically,
    /// rewrites the indirect template, and re-records the cull + scene
    /// secondaries (and rebinds the cull set to the current world buffers).
    /// Always returns `true` (the FrameSlot primaries reference the
    /// secondaries, so callers must rebuild them).
    ///
    /// Called only on topology change / capacity growth — never per frame in
    /// steady state.
    pub fn ensure_current(
        &mut self,
        plan: &DrawPlan,
        renderer_capacity: usize,
        scene: &CameraSceneResources<'_>,
    ) -> bool {
        let slot_count = plan.commands.len();
        let total = plan.total_renderers as usize;

        if total > self.mvp_capacity {
            self.mvp_capacity = total.max(self.mvp_capacity.saturating_mul(2)).max(1);
            let (dm, ds) = allocate_matrices_and_set(
                scene.memory_allocator,
                scene.descriptor_set_allocator,
                scene.pipeline,
                self.mvp_capacity,
            );
            self.device_matrices = dm;
            self.descriptor_set = ds;
        }
        if slot_count > self.slot_capacity {
            self.slot_capacity = slot_count.max(self.slot_capacity.saturating_mul(2)).max(1);
            let (t, a) = allocate_indirect_buffers(scene.memory_allocator, self.slot_capacity);
            self.indirect_template = t;
            self.indirect_args = a;
        }

        write_indirect_template(&self.indirect_template, &plan.commands);
        self.slot_count = slot_count;
        self.cull_range = renderer_capacity;

        self.cull_set = build_cull_set(scene, &self.device_matrices, &self.indirect_args);
        self.cull_secondary = record_cull_secondary(
            scene,
            &self.indirect_template,
            &self.indirect_args,
            &self.cull_set,
            renderer_capacity as u32,
        );
        // Texture arrivals / redirect-buffer growth reach here via
        // `force_full`; rebind the current views + buffers.
        self.texture_set = build_texture_set(scene);
        self.scene_secondary = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &self.descriptor_set,
            &self.texture_set,
            scene.mesh_store,
            &self.indirect_args,
            slot_count,
            self.extent,
        );
        true
    }

    /// Whether the current draw plan / renderer capacity needs a **full**
    /// rebuild (new buffers + descriptor set + secondaries + frame slots) vs.
    /// just an in-place rewrite of the indirect template's per-slot bases.
    ///
    /// `force` is set by the caller when a cull-bound external buffer (SoT,
    /// `GPURenderers`, redirect, mesh table) reallocated.
    pub fn needs_structural_rebuild(
        &self,
        plan: &DrawPlan,
        renderer_capacity: usize,
        force: bool,
    ) -> bool {
        force
            || plan.total_renderers as usize > self.mvp_capacity
            || plan.commands.len() > self.slot_capacity
            || plan.commands.len() != self.slot_count
            || renderer_capacity != self.cull_range
    }

    /// Cheap path: rewrite the indirect template's per-slot commands in place
    /// (the prefix-summed bases shift on every spawn). The cull / scene
    /// secondaries and the cull set all stay valid — they bind the *buffers*,
    /// and the per-frame `template → args` copy picks up the new contents.
    ///
    /// **The host write must be gated against in-flight reads** — the template
    /// is read by every in-flight frame's reset copy, so call this only after
    /// `WorldTransformGpu::host_wait_for_previous_compute`.
    pub fn write_template_bases(&self, plan: &DrawPlan) {
        write_indirect_template(&self.indirect_template, &plan.commands);
    }

    // ── Accessors ───────────────────────────────────────────────────────

    #[allow(dead_code)]
    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }
    pub fn color_image(&self) -> &Arc<Image> {
        &self.color_image
    }
    #[allow(dead_code)]
    pub fn depth_image(&self) -> &Arc<Image> {
        &self.depth_image
    }
    pub fn color_view(&self) -> &Arc<ImageView> {
        &self.color_view
    }
    pub fn depth_view(&self) -> &Arc<ImageView> {
        &self.depth_view
    }
    pub fn scene_secondary(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.scene_secondary
    }
    /// The cull (mvp-build) compute secondary — executed once per frame from
    /// each FrameSlot primary, before the scene render.
    pub fn cull_secondary(&self) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.cull_secondary
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Allocation / recording helpers
// ─────────────────────────────────────────────────────────────────────────────

fn allocate_attachments(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    extent: [u32; 2],
) -> (Arc<Image>, Arc<ImageView>, Arc<Image>, Arc<ImageView>) {
    let [w, h] = extent;
    let color_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: CAMERA_COLOR_FORMAT,
            extent: [w, h, 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .expect("Failed to create offscreen color image");
    let color_view = ImageView::new_default(color_image.clone())
        .expect("Failed to create offscreen color image view");

    let depth_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: CAMERA_DEPTH_FORMAT,
            extent: [w, h, 1],
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .expect("Failed to create offscreen depth image");
    let depth_view = ImageView::new_default(depth_image.clone())
        .expect("Failed to create offscreen depth image view");

    (color_image, color_view, depth_image, depth_view)
}

/// Allocate a device-local `[f32; 16]` MVP buffer of `capacity` slots + the
/// graphics descriptor set that points at it.
fn allocate_matrices_and_set(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    pipeline: &Arc<GraphicsPipeline>,
    capacity: usize,
) -> (Subbuffer<[[f32; 16]]>, Arc<DescriptorSet>) {
    let device_matrices: Subbuffer<[[f32; 16]]> = Buffer::new_slice::<[f32; 16]>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        capacity.max(1) as u64,
    )
    .expect("Failed to allocate device matrix buffer");

    let set_layout = pipeline.layout().set_layouts()[0].clone();
    let descriptor_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        set_layout,
        [WriteDescriptorSet::buffer(0, device_matrices.clone())],
        [],
    )
    .expect("Failed to allocate matrices descriptor set");

    (device_matrices, descriptor_set)
}

/// Allocate the indirect-command buffers: a host-visible **template** (the
/// CPU writes the per-slot commands with `instance_count` zeroed) and the
/// device-local **args** (reset from the template each frame, written by the
/// cull's atomics, read by the indirect draw).
fn allocate_indirect_buffers(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    capacity: usize,
) -> (
    Subbuffer<[DrawIndexedIndirectCommand]>,
    Subbuffer<[DrawIndexedIndirectCommand]>,
) {
    let cap = capacity.max(1);
    let template = Buffer::new_slice::<DrawIndexedIndirectCommand>(
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
        cap as u64,
    )
    .expect("Failed to allocate indirect template buffer");
    let args = Buffer::new_slice::<DrawIndexedIndirectCommand>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::INDIRECT_BUFFER
                | BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        cap as u64,
    )
    .expect("Failed to allocate indirect args buffer");
    (template, args)
}

/// Write the draw plan's per-slot commands into the host template.
fn write_indirect_template(
    template: &Subbuffer<[DrawIndexedIndirectCommand]>,
    commands: &[DrawIndexedIndirectCommand],
) {
    let mut guard = template.write().expect("indirect_template.write");
    guard[..commands.len()].copy_from_slice(commands);
    // Tail (capacity > commands.len()) left undefined — never read (the draw
    // slices to `slot_count`, the cull only touches slots in range).
}

/// Build the graphics texture set (set 1): the texture registry's redirect
/// buffer, the mesh store's per-slot texture ids, and the fixed-size
/// sampled-image array (placeholder-padded — see [`GpuTextureStore`]).
fn build_texture_set(scene: &CameraSceneResources<'_>) -> Arc<DescriptorSet> {
    let set_layout = scene.pipeline.layout().set_layouts()[1].clone();
    DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        set_layout,
        [
            WriteDescriptorSet::buffer(0, scene.texture_store.redirect_buffer().clone()),
            WriteDescriptorSet::buffer(1, scene.mesh_store.slot_texture_buffer().clone()),
            WriteDescriptorSet::image_view_sampler_array(
                2,
                0,
                scene.texture_store.descriptor_array(),
            ),
        ],
        [],
    )
    .expect("Failed to allocate texture descriptor set")
}

/// Build the cull descriptor set (set 0): SoT, GPURenderers, redirect, mesh
/// table, MVP output, the indirect commands (as a flat `u32[]`), and the
/// per-transform Parents buffer the chain walk reads.
fn build_cull_set(
    scene: &CameraSceneResources<'_>,
    device_matrices: &Subbuffer<[[f32; 16]]>,
    indirect_args: &Subbuffer<[DrawIndexedIndirectCommand]>,
) -> Arc<DescriptorSet> {
    let world = scene.world_transforms;
    DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        world.mvp_build_set0_layout().clone(),
        [
            WriteDescriptorSet::buffer(0, world.sot_positions().clone()),
            WriteDescriptorSet::buffer(1, world.sot_rotations().clone()),
            WriteDescriptorSet::buffer(2, world.sot_scales().clone()),
            WriteDescriptorSet::buffer(3, scene.gpu_renderers.buffer().clone()),
            WriteDescriptorSet::buffer(4, scene.mesh_store.redirect_buffer().clone()),
            WriteDescriptorSet::buffer(5, scene.mesh_store.mesh_table_buffer().clone()),
            WriteDescriptorSet::buffer(6, device_matrices.clone()),
            WriteDescriptorSet::buffer(7, indirect_args.clone().reinterpret::<[u32]>()),
            WriteDescriptorSet::buffer(8, world.sot_parents().clone()),
        ],
        [],
    )
    .expect("Failed to allocate cull set")
}

/// Record the cull compute secondary: reset the indirect `instance_count`s by
/// copying the template into the args buffer, then dispatch the cull over the
/// renderer range. Recorded `SimultaneousUse` (shared across FrameSlots).
fn record_cull_secondary(
    scene: &CameraSceneResources<'_>,
    indirect_template: &Subbuffer<[DrawIndexedIndirectCommand]>,
    indirect_args: &Subbuffer<[DrawIndexedIndirectCommand]>,
    cull_set: &Arc<DescriptorSet>,
    renderer_capacity: u32,
) -> Arc<SecondaryAutoCommandBuffer> {
    let pipeline = scene.world_transforms.mvp_build_pipeline();
    let layout = pipeline.layout().clone();
    let groups = renderer_capacity.div_ceil(64).max(1);
    let pc = shaders::mvp_build_cs::PC { renderer_capacity };

    let mut builder = AutoCommandBufferBuilder::secondary(
        scene.cb_allocator.clone(),
        scene.queue_family_index,
        CommandBufferUsage::SimultaneousUse,
        CommandBufferInheritanceInfo::default(),
    )
    .expect("cull secondary builder");

    // Reset every slot's instance_count to 0. Vulkano auto-syncs this transfer
    // write against the cull dispatch's atomic read-modify-write.
    builder
        .copy_buffer(CopyBufferInfo::buffers(
            indirect_template.clone(),
            indirect_args.clone(),
        ))
        .expect("reset indirect instance counts");

    builder
        .bind_pipeline_compute(pipeline.clone())
        .expect("bind cull pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            layout.clone(),
            0,
            (
                cull_set.clone(),
                scene.world_transforms.mvp_build_set1().clone(),
            ),
        )
        .expect("bind cull sets")
        .push_constants(layout, 0, pc)
        .expect("push cull constants");
    // Safety: dispatch count derived from `renderer_capacity`; the shader
    // bounds-checks against the push-constant.
    unsafe {
        builder.dispatch([groups, 1, 1]).expect("dispatch cull");
    }
    builder.build().expect("build cull secondary")
}

/// Record the scene secondary: a single `vkCmdDrawIndexedIndirect` over
/// `indirect_args[0..slot_count]` against the shared mega buffers.
fn record_scene_secondary(
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
    pipeline: &Arc<GraphicsPipeline>,
    descriptor_set: &Arc<DescriptorSet>,
    texture_set: &Arc<DescriptorSet>,
    mesh_store: &GpuMeshStore,
    indirect_args: &Subbuffer<[DrawIndexedIndirectCommand]>,
    slot_count: usize,
    extent: [u32; 2],
) -> Arc<SecondaryAutoCommandBuffer> {
    let [cam_w, cam_h] = extent;

    let inheritance = CommandBufferInheritanceInfo {
        render_pass: Some(
            CommandBufferInheritanceRenderingInfo {
                color_attachment_formats: vec![Some(CAMERA_COLOR_FORMAT)],
                depth_attachment_format: Some(CAMERA_DEPTH_FORMAT),
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
    .expect("Failed to create scene secondary builder");

    builder
        .set_viewport(
            0,
            smallvec::smallvec![Viewport {
                offset: [0.0, 0.0],
                extent: [cam_w as f32, cam_h as f32],
                depth_range: 0.0..=1.0,
            }],
        )
        .expect("set_viewport failed")
        .bind_pipeline_graphics(pipeline.clone())
        .expect("bind_pipeline_graphics failed")
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            pipeline.layout().clone(),
            0,
            (descriptor_set.clone(), texture_set.clone()),
        )
        .expect("bind_descriptor_sets failed");

    // All meshes share one mega vertex + one mega index buffer; bind once.
    // Each slot's command carries its own `first_index` / `vertex_offset` /
    // `first_instance` / (cull-written) `instance_count`.
    builder
        .bind_vertex_buffers(0, mesh_store.mega_vertex_buffer().clone())
        .expect("bind mega vertex buffer failed")
        .bind_index_buffer(mesh_store.mega_index_buffer().clone())
        .expect("bind mega index buffer failed");

    // One `vkCmdDrawIndexedIndirect` over all slots (drawCount == slot_count;
    // the `multi_draw_indirect` feature permits > 1). Empty slots have
    // instance_count 0 and draw nothing.
    if slot_count > 0 {
        let draws = indirect_args.clone().slice(0..slot_count as u64);
        // Safety: args buffer is INDIRECT_BUFFER-usable; mega index buffer is
        // bound; `first_instance` bounded by the MVP capacity; the indirect
        // device features are enabled at device creation (see RenderApp::new).
        unsafe {
            builder
                .draw_indexed_indirect(draws)
                .expect("draw_indexed_indirect failed");
        }
    }

    builder.build().expect("Failed to build scene secondary")
}
