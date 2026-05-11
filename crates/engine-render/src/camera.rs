//! Render-side camera: owns the GPU attachments (color + depth) a camera
//! renders into, the device-side MVP matrix buffer + descriptor set the
//! camera's draws read from, the pre-recorded *scene secondary* command
//! buffer that issues those draws, and the resolution / capacity policies
//! that decide when any of the above need to be re-created.
//!
//! This is distinct from [`crate::scene::Camera`], which is a pure
//! view/projection math helper. A `RenderCamera` *uses* a `scene::Camera`
//! (or any other source of `view_proj`) at draw time, but owns the GPU
//! resources independently.
//!
//! # Three invalidation domains
//!
//! `RenderCamera` is the focal point of two of the three orthogonal
//! invalidation axes that drive command-buffer rebuilds:
//!
//! 1. **Per-camera resolution** — color/depth attachment extent.
//!    Changes when [`CameraResolution`] resolves to a new extent (e.g. on
//!    swapchain resize for `MatchSwapchain`). Triggers attachment +
//!    scene-secondary rebuild.
//! 2. **Per-camera capacity** — `device_matrices` slot count.
//!    Changes when the scene asks for more draws than the camera's matrix
//!    buffer can hold. Triggers device-buffer + descriptor-set +
//!    scene-secondary rebuild. Geometric growth keeps amortized cost low.
//! 3. **Per-frame-in-flight** — staging matrix ring. *Not* on the camera —
//!    lives on `FrameSlot`. Sized to match camera capacity so the per-frame
//!    `copy_buffer(staging → device)` works.
//!
//! Plus a fourth axis the camera doesn't own:
//!
//! 4. **Per-swapchain-image** — present-blit destination. Lives on
//!    `FrameSlot::blit_secondary` and the primary CB; rebuilt only when the
//!    swapchain image identity changes.
//!
//! Promoting the device matrices + descriptor set + scene secondary onto
//! the camera (rather than per-`FrameSlot`) means:
//!
//! - We allocate the device matrix buffer **once per camera**, not once per
//!   swapchain image. (Previously N swapchain images × identical device
//!   matrices = N redundant allocations of identical contents.)
//! - The scene secondary survives swapchain resize entirely for cameras
//!   whose policy doesn't depend on the swapchain (future `Fixed` shadow
//!   maps, etc.) — only the per-image blit secondary + primary need
//!   rebuilding in that case.
//! - "Scene capacity grew" becomes a distinct, infrequent event with a
//!   clean rebuild path (camera grows; FrameSlots' staging buffers grow;
//!   primaries re-record), independent of swapchain churn.

use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo,
        CommandBufferInheritanceRenderingInfo, CommandBufferUsage,
        SecondaryAutoCommandBuffer,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
        layout::DescriptorSetLayout,
    },
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{GraphicsPipeline, PipelineBindPoint, graphics::viewport::Viewport},
};

use crate::gpu_mesh::GpuMesh;
use crate::transform_gpu::WorldTransformGpu;

// `Pipeline` trait is needed for `pipeline.layout()` method resolution.
use vulkano::pipeline::Pipeline;

/// Pixel format used for camera-owned offscreen color targets.
///
/// HDR-capable (16-bit float per channel) so future tonemapping / bloom /
/// other post-process passes have headroom; the present-blit converts down
/// to whatever sRGB swapchain format the platform offers.
pub const CAMERA_COLOR_FORMAT: Format = Format::R16G16B16A16_SFLOAT;

/// Pixel format used for camera-owned depth targets.
pub const CAMERA_DEPTH_FORMAT: Format = Format::D32_SFLOAT;

/// How a camera's attachment extent is determined relative to the swapchain.
///
/// This is a *policy* — the camera consults it on every swapchain-resize
/// event to decide whether (and to what size) to re-create its attachments.
#[derive(Clone, Copy, Debug)]
pub enum CameraResolution {
    /// Track the swapchain extent 1:1. Attachments are re-created on every
    /// swapchain resize. Used by the main camera so the present-blit stays a
    /// 1:1 copy.
    MatchSwapchain,
    // Reserved for future variants (`todo.txt` §3):
    //   Fixed { width: u32, height: u32 }
    //       — fixed offscreen size, survives swapchain resize. Shadow maps,
    //         editor thumbnails, render-to-texture for portals/mirrors.
    //   ScaleSwapchain { numerator: u32, denominator: u32 }
    //       — fraction of swapchain extent (e.g. 1/2 for half-res reflections).
}

impl CameraResolution {
    /// Resolve a swapchain extent into the actual extent this camera should
    /// render at, given its policy.
    fn resolve(&self, swapchain_extent: [u32; 2]) -> [u32; 2] {
        match self {
            CameraResolution::MatchSwapchain => swapchain_extent,
        }
    }

    /// Does this policy depend on the swapchain extent? `true` means a
    /// swapchain resize *might* require re-creating the camera's attachments;
    /// `false` means swapchain resizes are irrelevant to this camera.
    pub fn depends_on_swapchain(&self) -> bool {
        match self {
            CameraResolution::MatchSwapchain => true,
        }
    }
}

/// Per-call bundle of the GPU/scene state the camera needs to (re)build its
/// scene secondary. Bundled into a struct purely to keep the rebuild API's
/// parameter list manageable; nothing is owned beyond the call.
pub struct CameraSceneResources<'a> {
    pub cb_allocator:             &'a Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: &'a Arc<StandardDescriptorSetAllocator>,
    pub memory_allocator:         &'a Arc<StandardMemoryAllocator>,
    pub pipeline:                 &'a Arc<GraphicsPipeline>,
    pub queue_family_index:       u32,
    pub gpu_meshes:               &'a [GpuMesh],
    /// Per-instance mesh indices, one entry per draw the scene secondary
    /// should record. `len()` is the **logical** draw count for the next
    /// rebuild; capacity grows to fit it (with headroom).
    pub draws_template:           &'a [u32],
    /// Per-instance entity index, parallel to `draws_template` — one entry
    /// per draw, pointing at a slot in the world's SoT buffers. Uploaded
    /// into `instance_to_entity_buffer` and read by the mvp-build compute
    /// shader to fetch each draw's TRS from the SoT.
    pub entity_template:          &'a [u32],
    /// Per-world transform state (SoT buffers + compute pipelines). Used to
    /// build the camera's `mvp_build_set0` (which captures the SoT buffer
    /// handles) and to look up the mvp-build pipeline + set layouts. Does
    /// **not** need to be the same instance across calls — but if its SoT
    /// buffers were re-allocated since the last time the camera saw it,
    /// `mvp_build_set0` will be silently stale; callers must re-invoke an
    /// invalidation entry point (`on_world_capacity_change`) in that case.
    pub world_transforms:         &'a WorldTransformGpu,
}

/// Render-side camera: owns the offscreen color + depth attachments the
/// camera renders into, the device-side MVP matrix storage buffer + its
/// descriptor set, the pre-recorded scene secondary, and the policies that
/// drive when any of these have to be re-created.
///
/// The host-visible *staging* matrix ring lives on `FrameSlot`
/// (per-frame-in-flight); the per-frame `copy_buffer(staging → device)`
/// runs in the primary CB and bridges those two axes.
pub struct RenderCamera {
    // ── Resolution / attachments ────────────────────────────────────────
    resolution:        CameraResolution,
    extent:            [u32; 2],
    color_image:       Arc<Image>,
    depth_image:       Arc<Image>,
    color_view:        Arc<ImageView>,
    depth_view:        Arc<ImageView>,

    // ── Matrix storage (per-camera, stable across frames) ───────────────
    /// Device-local matrix buffer the vertex shader reads via
    /// `descriptor_set`, **and that the mvp-build compute pass writes**.
    /// Sized to `allocated_capacity` slots; grown geometrically so
    /// amortized rebuild cost is O(1) per added draw.
    device_matrices:   Subbuffer<[[f32; 16]]>,
    /// Graphics set 0 — references `device_matrices`. Re-created together
    /// with `device_matrices` whenever the buffer is re-allocated (the set
    /// captures the buffer by handle; the new buffer needs a new set).
    descriptor_set:    Arc<DescriptorSet>,
    /// Pre-recorded scene secondary that issues exactly `draw_count` draws,
    /// each indexing into `device_matrices` via `first_instance`. Inherits
    /// the primary's dynamic-rendering scope.
    scene_secondary:   Arc<SecondaryAutoCommandBuffer>,

    // ── MVP-build compute (per-camera state) ────────────────────────────
    /// `instance → entity` lookup, length == `draw_count`. The mvp-build
    /// shader does `entity = idx[gl_GlobalInvocationID.x]; mvp[gid] =
    /// view_proj * model_from_trs(sot[entity])`. Re-uploaded whenever
    /// scene topology changes.
    instance_to_entity: Subbuffer<[u32]>,
    /// MVP-build descriptor set 0 — binds the world's SoT (pos/rot/scale),
    /// `instance_to_entity`, and `device_matrices` as the output. Captured
    /// by buffer handle, so it must be re-allocated whenever **any** of
    /// those four buffers are re-allocated (camera capacity grows OR world
    /// capacity grows OR topology change re-uploads `instance_to_entity`).
    mvp_build_set0:    Arc<DescriptorSet>,

    /// Number of `[f32; 16]` slots actually allocated in `device_matrices`.
    /// Always >= 1 (Vulkan won't allocate zero-sized buffers) and
    /// always >= `draw_count`.
    allocated_capacity: usize,
    /// Number of draws baked into `scene_secondary` (== `draws_template.len()`
    /// at the last rebuild). Distinct from `allocated_capacity` because the
    /// device buffer may have headroom from an earlier geometric grow.
    draw_count:        usize,
}

impl RenderCamera {
    /// Build a camera whose attachments track the swapchain extent.
    pub fn new_match_swapchain(
        swapchain_extent: [u32; 2],
        scene:            &CameraSceneResources<'_>,
    ) -> Self {
        Self::new(CameraResolution::MatchSwapchain, swapchain_extent, scene)
    }

    /// Build a camera with the given resolution policy. `swapchain_extent`
    /// is consulted only if the policy depends on it.
    pub fn new(
        resolution:       CameraResolution,
        swapchain_extent: [u32; 2],
        scene:            &CameraSceneResources<'_>,
    ) -> Self {
        let extent = resolution.resolve(swapchain_extent);
        let (color_image, color_view, depth_image, depth_view) =
            allocate_attachments(scene.memory_allocator, extent);

        let draw_count = scene.draws_template.len();
        let allocated_capacity = draw_count.max(1);
        let (device_matrices, descriptor_set) = allocate_matrices_and_set(
            scene.memory_allocator,
            scene.descriptor_set_allocator,
            scene.pipeline,
            allocated_capacity,
        );
        let instance_to_entity = allocate_and_upload_instance_to_entity(
            scene.memory_allocator,
            scene.entity_template,
            allocated_capacity,
        );
        let mvp_build_set0 = build_mvp_build_set0(
            scene.descriptor_set_allocator,
            scene.world_transforms,
            &instance_to_entity,
            &device_matrices,
        );
        let scene_secondary = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &descriptor_set,
            scene.gpu_meshes,
            scene.draws_template,
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
            scene_secondary,
            instance_to_entity,
            mvp_build_set0,
            allocated_capacity,
            draw_count,
        }
    }

    /// Inform the camera that the swapchain has been re-created with a new
    /// extent. Returns `true` if the camera re-created its attachments
    /// **or** re-recorded its scene secondary (callers must then rebuild any
    /// command buffer that references the camera's color/depth views or its
    /// scene secondary — i.e. every `FrameSlot::command_buffer`).
    ///
    /// For [`CameraResolution::MatchSwapchain`] this re-creates whenever
    /// the extent actually changes (and re-records the scene secondary
    /// because the viewport baked into it depends on the extent); for
    /// swapchain-independent policies (future) this is a no-op.
    pub fn on_swapchain_resize(
        &mut self,
        new_swapchain_extent: [u32; 2],
        scene:                &CameraSceneResources<'_>,
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
        self.extent      = new_extent;
        self.color_image = color_image;
        self.color_view  = color_view;
        self.depth_image = depth_image;
        self.depth_view  = depth_view;

        // The viewport baked into the scene secondary depends on `extent`,
        // so the secondary itself has to be re-recorded. The matrix buffer
        // and descriptor set are extent-independent and survive untouched.
        self.scene_secondary = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &self.descriptor_set,
            scene.gpu_meshes,
            scene.draws_template,
            self.extent,
        );
        self.draw_count = scene.draws_template.len();
        true
    }

    /// Ensure the camera's `device_matrices` can hold at least `needed`
    /// slots and that the scene secondary is recorded for exactly
    /// `scene.draws_template.len()` draws.
    ///
    /// Returns `true` if anything was re-recorded (so callers know to
    /// rebuild dependent CBs — the per-image primaries reference
    /// `device_matrices` via `copy_buffer` and `scene_secondary` via
    /// `execute_commands`, so any rebuild here invalidates them).
    ///
    /// Capacity grows geometrically (≥ 2× current) so amortized cost is
    /// O(1) per added draw. Capacity *never shrinks* — buffers are cheap
    /// and shrinking would just trade memory for future re-allocations.
    pub fn ensure_capacity(
        &mut self,
        needed: usize,
        scene:  &CameraSceneResources<'_>,
    ) -> bool {
        let needed_slots   = scene.draws_template.len().max(needed);
        let topology_changed = needed_slots != self.draw_count;

        if needed_slots <= self.allocated_capacity && !topology_changed {
            return false;
        }

        if needed_slots > self.allocated_capacity {
            // Geometric growth: at least double, at least what's needed, at
            // least 1. Mirrors `Vec`'s amortized-O(1) growth strategy.
            let new_capacity = needed_slots
                .max(self.allocated_capacity.saturating_mul(2))
                .max(1);
            let (device_matrices, descriptor_set) = allocate_matrices_and_set(
                scene.memory_allocator,
                scene.descriptor_set_allocator,
                scene.pipeline,
                new_capacity,
            );
            self.device_matrices    = device_matrices;
            self.descriptor_set     = descriptor_set;
            self.allocated_capacity = new_capacity;
        }

        // Always re-upload the instance→entity table on topology change
        // (lengths or contents may have shifted), and always re-build
        // mvp_build_set0 since at least one of `device_matrices` /
        // `instance_to_entity` was re-allocated when we got here.
        self.instance_to_entity = allocate_and_upload_instance_to_entity(
            scene.memory_allocator,
            scene.entity_template,
            self.allocated_capacity,
        );
        self.mvp_build_set0 = build_mvp_build_set0(
            scene.descriptor_set_allocator,
            scene.world_transforms,
            &self.instance_to_entity,
            &self.device_matrices,
        );

        // Always re-record when topology changed OR capacity grew — the
        // descriptor set may be new, and the draw count baked into the
        // secondary must match the template.
        self.scene_secondary = record_scene_secondary(
            scene.cb_allocator,
            scene.queue_family_index,
            scene.pipeline,
            &self.descriptor_set,
            scene.gpu_meshes,
            scene.draws_template,
            self.extent,
        );
        self.draw_count = scene.draws_template.len();
        true
    }

    /// Notify the camera that the world's SoT buffers were re-allocated
    /// (capacity grew). Re-builds `mvp_build_set0` so the per-camera
    /// mvp-build compute pass binds the new buffer handles. Cheap — just a
    /// descriptor-set allocation, no buffer churn. Returns `true` so the
    /// caller knows that any pre-recorded CB referencing
    /// `mvp_build_set0` (i.e. every FrameSlot's mvp_build secondary +
    /// composing primary) must be re-recorded.
    pub fn on_world_capacity_change(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        world_transforms:         &WorldTransformGpu,
    ) -> bool {
        self.mvp_build_set0 = build_mvp_build_set0(
            descriptor_set_allocator,
            world_transforms,
            &self.instance_to_entity,
            &self.device_matrices,
        );
        true
    }

    /// Current attachment extent. Held for future multi-camera UI / debug
    /// readout; no longer read by the renderer hot path now that the
    /// viewport is baked into the camera-owned scene secondary.
    #[allow(dead_code)]
    pub fn extent(&self)              -> [u32; 2]                 { self.extent }
    pub fn color_image(&self)         -> &Arc<Image>              { &self.color_image }
    /// Held for future post-process passes / debug visualizers; not yet read
    /// by the current single-pass renderer.
    #[allow(dead_code)]
    pub fn depth_image(&self)         -> &Arc<Image>              { &self.depth_image }
    pub fn color_view(&self)          -> &Arc<ImageView>          { &self.color_view  }
    pub fn depth_view(&self)          -> &Arc<ImageView>          { &self.depth_view  }
    #[allow(dead_code)]
    pub fn device_matrices(&self)     -> &Subbuffer<[[f32; 16]]>  { &self.device_matrices }
    pub fn scene_secondary(&self)     -> &Arc<SecondaryAutoCommandBuffer> { &self.scene_secondary }
    pub fn mvp_build_set0(&self)      -> &Arc<DescriptorSet>      { &self.mvp_build_set0 }
    /// Held so test/debug code can inspect the per-camera lookup; the
    /// renderer hot path never reads this directly (it lives on the GPU).
    #[allow(dead_code)]
    pub fn instance_to_entity(&self)  -> &Subbuffer<[u32]>        { &self.instance_to_entity }
    /// Number of `[f32; 16]` slots in `device_matrices`. Per-frame staging
    /// buffers should match this exactly so the in-primary `copy_buffer`
    /// has same-sized source and destination.
    pub fn allocated_capacity(&self)  -> usize                    { self.allocated_capacity }
    /// Number of draws the scene secondary is currently recorded for. This
    /// is what the host should write into the staging buffer each frame;
    /// the rest of the staging/device buffer (up to `allocated_capacity`)
    /// is unused headroom.
    pub fn draw_count(&self)          -> usize                    { self.draw_count }
}

fn allocate_attachments(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    extent:           [u32; 2],
) -> (Arc<Image>, Arc<ImageView>, Arc<Image>, Arc<ImageView>) {
    let [w, h] = extent;
    let color_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format:     CAMERA_COLOR_FORMAT,
            extent:     [w, h, 1],
            usage:      ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
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
            format:     CAMERA_DEPTH_FORMAT,
            extent:     [w, h, 1],
            usage:      ImageUsage::DEPTH_STENCIL_ATTACHMENT,
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

/// Allocate a device-local `[f32; 16]` storage buffer of `capacity` slots
/// and the descriptor set that points at it. Grouped because the descriptor
/// set captures the buffer by handle, so any new buffer needs a fresh set.
///
/// Usage is `STORAGE_BUFFER` only: the mvp-build compute writes into it
/// and the vertex shader reads from it. There is no longer a host-driven
/// `vkCmdCopyBuffer` into this buffer (replaced by the compute pass), so
/// `TRANSFER_DST` is no longer needed.
fn allocate_matrices_and_set(
    memory_allocator:         &Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    pipeline:                 &Arc<GraphicsPipeline>,
    capacity:                 usize,
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

    let set_layout: Arc<DescriptorSetLayout> = pipeline
        .layout()
        .set_layouts()[0]
        .clone();
    let descriptor_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        set_layout,
        [WriteDescriptorSet::buffer(0, device_matrices.clone())],
        [],
    )
    .expect("Failed to allocate matrices descriptor set");

    (device_matrices, descriptor_set)
}

/// Allocate the per-camera `instance → entity` lookup buffer of length
/// `capacity`, and immediately upload the contents of `entity_template`
/// (which is `<= capacity` long). Tail past `entity_template.len()` is
/// undefined; the mvp-build shader is dispatched only for `draw_count`
/// invocations, so the tail is never read.
///
/// HOST_SEQUENTIAL_WRITE memory keeps the upload trivial (single `write()`
/// at allocation time, no staging+copy). Topology changes are infrequent;
/// for very large lookup tables a copy-from-staging path may be worth it,
/// but the current scale (one entry per draw) makes that overkill.
fn allocate_and_upload_instance_to_entity(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    entity_template:  &[u32],
    capacity:         usize,
) -> Subbuffer<[u32]> {
    let cap = capacity.max(1).max(entity_template.len());
    let buf: Subbuffer<[u32]> = Buffer::new_slice::<u32>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        cap as u64,
    )
    .expect("Failed to allocate instance_to_entity buffer");

    {
        let mut guard = buf.write().expect("instance_to_entity.write");
        guard[..entity_template.len()].copy_from_slice(entity_template);
        // Tail (if `cap > entity_template.len()`) is left undefined; never
        // read by the dispatch (capped to `draw_count == entity_template.len()`).
    }

    buf
}

/// Build the mvp-build set 0 descriptor set: SoT (pos/rot/scl) + idx + mvp.
/// All buffers are captured by handle, so this set must be re-allocated
/// whenever any of those four buffers is re-allocated.
fn build_mvp_build_set0(
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    world:                    &WorldTransformGpu,
    instance_to_entity:       &Subbuffer<[u32]>,
    device_matrices:          &Subbuffer<[[f32; 16]]>,
) -> Arc<DescriptorSet> {
    DescriptorSet::new(
        descriptor_set_allocator.clone(),
        world.mvp_build_set0_layout().clone(),
        [
            WriteDescriptorSet::buffer(0, world.sot_positions().clone()),
            WriteDescriptorSet::buffer(1, world.sot_rotations().clone()),
            WriteDescriptorSet::buffer(2, world.sot_scales().clone()),
            WriteDescriptorSet::buffer(3, instance_to_entity.clone()),
            WriteDescriptorSet::buffer(4, device_matrices.clone()),
        ],
        [],
    )
    .expect("Failed to allocate mvp_build_set0")
}

/// Record the scene secondary: viewport + pipeline + descriptor set bind +
/// one `draw_indexed` per entry of `draws_template` (with `first_instance`
/// stepping through `device_matrices`).
///
/// Inherits the primary's dynamic-rendering scope (color/depth formats
/// must match what the primary's `begin_rendering` will declare). The
/// secondary may NOT call `begin_rendering`/`end_rendering` itself.
fn record_scene_secondary(
    cb_allocator:        &Arc<StandardCommandBufferAllocator>,
    queue_family_index:  u32,
    pipeline:            &Arc<GraphicsPipeline>,
    descriptor_set:      &Arc<DescriptorSet>,
    gpu_meshes:          &[GpuMesh],
    draws_template:      &[u32],
    extent:              [u32; 2],
) -> Arc<SecondaryAutoCommandBuffer> {
    let [cam_w, cam_h] = extent;

    let inheritance = CommandBufferInheritanceInfo {
        render_pass: Some(
            CommandBufferInheritanceRenderingInfo {
                color_attachment_formats: vec![Some(CAMERA_COLOR_FORMAT)],
                depth_attachment_format:  Some(CAMERA_DEPTH_FORMAT),
                ..Default::default()
            }
            .into(),
        ),
        ..Default::default()
    };

    let mut builder = AutoCommandBufferBuilder::secondary(
        cb_allocator.clone(),
        queue_family_index,
        // The camera's scene secondary is shared across **every** FrameSlot's
        // primary CB (one per swapchain image). With `MAX_FRAMES_IN_FLIGHT`
        // primaries potentially in flight at the same time, the same
        // secondary is referenced by multiple in-flight primaries
        // simultaneously — which `MultipleSubmit` (Vulkan: resubmit-after-
        // completion) forbids. `SimultaneousUse` is the Vulkan
        // `SIMULTANEOUS_USE_BIT` that explicitly permits this.
        CommandBufferUsage::SimultaneousUse,
        inheritance,
    )
    .expect("Failed to create scene secondary builder");

    builder
        .set_viewport(
            0,
            smallvec::smallvec![Viewport {
                offset:      [0.0, 0.0],
                extent:      [cam_w as f32, cam_h as f32],
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
            descriptor_set.clone(),
        )
        .expect("bind_descriptor_sets failed");

    // One draw per RenderInstance, with `first_instance = i` so the vertex
    // shader's `gl_InstanceIndex` indexes into `device_matrices`.
    for (i, &mesh_idx) in draws_template.iter().enumerate() {
        let mesh = match gpu_meshes.get(mesh_idx as usize) {
            Some(m) => m,
            None    => continue,
        };
        builder
            .bind_vertex_buffers(0, mesh.vertex_buffer.clone())
            .expect("bind_vertex_buffers failed")
            .bind_index_buffer(mesh.index_buffer.clone())
            .expect("bind_index_buffer failed");
        // Safety: buffers are compatible with the bound pipeline; index
        // count fits within the uploaded index slice; first_instance is
        // bounded by `draws_template.len()` <= `allocated_capacity`.
        unsafe {
            builder
                .draw_indexed(mesh.index_count, 1, 0, 0, i as u32)
                .expect("draw_indexed failed");
        }
    }

    builder.build().expect("Failed to build scene secondary")
}
