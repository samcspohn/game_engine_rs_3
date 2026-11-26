//! The editor overlay: a world grid and the TRS gizmo's geometry, drawn per
//! camera in one render scope after the scene's two passes.
//!
//! Two pipelines over one host-written vertex buffer and one camera block:
//! the grid is a fullscreen triangle that intersects the ground plane per
//! pixel (depth-tested against the scene, writing its own depth), the gizmo
//! is world-space triangles with no depth test at all.
//!
//! Everything is sized once and drawn indirectly, because the primary
//! command buffer that executes this is recorded once and replayed —
//! nothing here may change a command, only the bytes a command reads.
//! Host writes are double-buffered by staging slot for the same reason
//! every other host-visible buffer here is.
//!
//! [`publish`] is how the gizmo hands its triangles over: it names a camera
//! slot rather than a `RenderCamera`, which does not exist yet when the
//! gizmo runs.

use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo,
        CommandBufferInheritanceRenderingInfo, CommandBufferUsage, DrawIndirectCommand,
        SecondaryAutoCommandBuffer,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    device::Device,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{
                AttachmentBlend, BlendFactor, ColorBlendAttachmentState, ColorBlendState,
            },
            depth_stencil::{CompareOp, DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::{PipelineRenderingCreateInfo, PipelineSubpassType},
            vertex_input::{Vertex, VertexDefinition, VertexInputState},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
};

use vulkano::buffer::BufferContents;

use crate::camera::{CameraSceneResources, CAMERA_COLOR_FORMAT, CAMERA_DEPTH_FORMAT, MAX_CAMERAS};
use crate::{shaders, STAGING_SLOTS};

/// Triangles per camera per frame. The gizmo's worst case (three rotate
/// rings at full tessellation) is under a third of this.
const VERTEX_CAPACITY: usize = 8192;

/// One overlay triangle corner, in world space.
#[derive(BufferContents, Vertex, Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct OverlayVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}

impl OverlayVertex {
    pub fn new(position: glam::Vec3, color: [f32; 4]) -> Self {
        Self {
            position: position.into(),
            color,
        }
    }
}

/// The per-camera block both stages read. Matches the `Overlay` uniform in
/// `overlay.vert` / `grid.vert` / `grid.frag`.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct OverlayParams {
    view_proj: [f32; 16],
    inv_view_proj: [f32; 16],
    eye: [f32; 4],
    grid: [f32; 4],
}

/// What the gizmo published for each camera this frame, by camera slot.
static PUBLISHED: Mutex<Vec<Vec<OverlayVertex>>> = Mutex::new(Vec::new());

/// Hand `camera_slot`'s overlay triangles to the renderer. Replaces the
/// previous frame's — the gizmo rebuilds from scratch every frame, so an
/// unpublished camera correctly shows nothing.
pub(crate) fn publish(camera_slot: usize, vertices: Vec<OverlayVertex>) {
    let mut all = PUBLISHED.lock();
    if all.len() <= camera_slot {
        all.resize_with(MAX_CAMERAS, Vec::new);
    }
    all[camera_slot] = vertices;
}

/// Both pipelines, built once for the device. They differ only in depth
/// state and vertex input, and neither depends on anything per-camera.
struct Pipelines {
    grid: Arc<GraphicsPipeline>,
    gizmo: Arc<GraphicsPipeline>,
}

static PIPELINES: OnceLock<Pipelines> = OnceLock::new();

fn pipelines(device: &Arc<Device>) -> &'static Pipelines {
    PIPELINES.get_or_init(|| Pipelines {
        grid: build_grid_pipeline(device.clone()),
        gizmo: build_gizmo_pipeline(device.clone()),
    })
}

/// One camera's overlay: the host buffers it writes each frame and the
/// secondary that draws them.
pub(crate) struct CameraOverlay {
    vertices: [Subbuffer<[OverlayVertex]>; STAGING_SLOTS],
    params: [Subbuffer<OverlayParams>; STAGING_SLOTS],
    /// Two commands: the grid's fullscreen triangle, then the gizmo's
    /// triangles. Both counts are host-written, so an empty overlay costs
    /// two zero-instance draws rather than a re-record.
    args: [Subbuffer<[DrawIndirectCommand]>; STAGING_SLOTS],
    sets: [(Arc<DescriptorSet>, Arc<DescriptorSet>); STAGING_SLOTS],
    secondaries: [Arc<SecondaryAutoCommandBuffer>; STAGING_SLOTS],
}

impl CameraOverlay {
    pub(crate) fn new(scene: &CameraSceneResources<'_>, extent: [u32; 2]) -> Self {
        let p = pipelines(scene.pipeline.device());
        let vertices = std::array::from_fn(|_| {
            host_buffer(
                scene.memory_allocator,
                BufferUsage::VERTEX_BUFFER,
                VERTEX_CAPACITY,
            )
        });
        let params: [Subbuffer<OverlayParams>; STAGING_SLOTS] = std::array::from_fn(|_| {
            Buffer::new_sized(
                scene.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                host_alloc(),
            )
            .expect("overlay params buffer")
        });
        let args = std::array::from_fn(|_| {
            host_buffer(scene.memory_allocator, BufferUsage::INDIRECT_BUFFER, 2)
        });
        let sets = std::array::from_fn(|i| {
            (
                params_set(scene, &p.grid, &params[i]),
                params_set(scene, &p.gizmo, &params[i]),
            )
        });
        let secondaries =
            std::array::from_fn(|i| record(scene, p, &vertices[i], &args[i], &sets[i], extent));
        Self {
            vertices,
            params,
            args,
            sets,
            secondaries,
        }
    }

    /// The viewport is baked into the secondary, so a resized camera needs
    /// its overlay re-recorded — nothing else here depends on extent.
    pub(crate) fn on_resize(&mut self, scene: &CameraSceneResources<'_>, extent: [u32; 2]) {
        let p = pipelines(scene.pipeline.device());
        self.secondaries = std::array::from_fn(|i| {
            record(
                scene,
                p,
                &self.vertices[i],
                &self.args[i],
                &self.sets[i],
                extent,
            )
        });
    }

    /// This frame's camera block and geometry, into `slot`'s buffers.
    ///
    /// Gated by the same compute wait as every other host-visible write —
    /// see the call site in `lib.rs`.
    pub(crate) fn write(
        &self,
        slot: usize,
        camera_slot: usize,
        view_proj: glam::Mat4,
        eye: glam::Vec3,
        grid: Option<[f32; 4]>,
    ) {
        let published = PUBLISHED.lock();
        let verts: &[OverlayVertex] = published.get(camera_slot).map_or(&[], |v| v.as_slice());
        let count = verts.len().min(VERTEX_CAPACITY);

        let mut dst = self.vertices[slot].write().expect("overlay vertices write");
        dst[..count].copy_from_slice(&verts[..count]);
        drop(dst);

        *self.params[slot].write().expect("overlay params write") = OverlayParams {
            view_proj: view_proj.to_cols_array(),
            inv_view_proj: view_proj.inverse().to_cols_array(),
            eye: [eye.x, eye.y, eye.z, 0.0],
            grid: grid.unwrap_or_default(),
        };

        let mut args = self.args[slot].write().expect("overlay args write");
        args[0] = DrawIndirectCommand {
            vertex_count: 3,
            instance_count: grid.is_some() as u32,
            first_vertex: 0,
            first_instance: 0,
        };
        args[1] = DrawIndirectCommand {
            vertex_count: count as u32,
            instance_count: (count > 0) as u32,
            first_vertex: 0,
            first_instance: 0,
        };
    }

    pub(crate) fn secondary(&self, slot: usize) -> &Arc<SecondaryAutoCommandBuffer> {
        &self.secondaries[slot]
    }
}

fn host_alloc() -> AllocationCreateInfo {
    AllocationCreateInfo {
        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        ..Default::default()
    }
}

fn host_buffer<T: BufferContents>(
    allocator: &Arc<StandardMemoryAllocator>,
    usage: BufferUsage,
    len: usize,
) -> Subbuffer<[T]> {
    Buffer::new_slice(
        allocator.clone(),
        BufferCreateInfo {
            usage,
            ..Default::default()
        },
        host_alloc(),
        len as u64,
    )
    .expect("overlay host buffer")
}

fn params_set(
    scene: &CameraSceneResources<'_>,
    pipeline: &Arc<GraphicsPipeline>,
    params: &Subbuffer<OverlayParams>,
) -> Arc<DescriptorSet> {
    DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        pipeline.layout().set_layouts()[0].clone(),
        [WriteDescriptorSet::buffer(0, params.clone())],
        [],
    )
    .expect("overlay params set")
}

/// Grid then gizmo, in that order: the grid is depth-tested against the
/// scene, the gizmo is drawn over both.
fn record(
    scene: &CameraSceneResources<'_>,
    p: &Pipelines,
    vertices: &Subbuffer<[OverlayVertex]>,
    args: &Subbuffer<[DrawIndirectCommand]>,
    sets: &(Arc<DescriptorSet>, Arc<DescriptorSet>),
    extent: [u32; 2],
) -> Arc<SecondaryAutoCommandBuffer> {
    let mut builder = AutoCommandBufferBuilder::secondary(
        scene.cb_allocator.clone(),
        scene.queue_family_index,
        CommandBufferUsage::SimultaneousUse,
        CommandBufferInheritanceInfo {
            render_pass: Some(
                CommandBufferInheritanceRenderingInfo {
                    color_attachment_formats: vec![Some(CAMERA_COLOR_FORMAT)],
                    depth_attachment_format: Some(CAMERA_DEPTH_FORMAT),
                    ..Default::default()
                }
                .into(),
            ),
            ..Default::default()
        },
    )
    .expect("overlay secondary builder");

    builder
        .set_viewport(
            0,
            smallvec::smallvec![Viewport {
                offset: [0.0, 0.0],
                extent: [extent[0] as f32, extent[1] as f32],
                depth_range: 0.0..=1.0,
            }],
        )
        .expect("overlay set_viewport");

    for (pipeline, set, i) in [(&p.grid, &sets.0, 0u64), (&p.gizmo, &sets.1, 1)] {
        builder
            .bind_pipeline_graphics(pipeline.clone())
            .expect("bind overlay pipeline")
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .expect("bind overlay params");
        if i == 1 {
            builder
                .bind_vertex_buffers(0, vertices.clone())
                .expect("bind overlay vertices");
        }
        // SAFETY: both commands are host-written every frame with a
        // `vertex_count` inside the capacity of the buffers bound above.
        unsafe {
            builder
                .draw_indirect(args.clone().slice(i..i + 1))
                .expect("overlay draw_indirect");
        }
    }

    builder.build().expect("build overlay secondary")
}

fn stages(
    device: &Arc<Device>,
    vs: Arc<vulkano::shader::ShaderModule>,
    fs: Arc<vulkano::shader::ShaderModule>,
) -> ([PipelineShaderStageCreateInfo; 2], Arc<PipelineLayout>) {
    let stages = [
        PipelineShaderStageCreateInfo::new(vs.entry_point("main").expect("overlay vs entry")),
        PipelineShaderStageCreateInfo::new(fs.entry_point("main").expect("overlay fs entry")),
    ];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .expect("overlay pipeline layout info"),
    )
    .expect("overlay pipeline layout");
    (stages, layout)
}

fn blended(depth: Option<DepthState>) -> (Option<ColorBlendState>, Option<DepthStencilState>) {
    (
        Some(ColorBlendState::with_attachment_states(
            1,
            ColorBlendAttachmentState {
                blend: Some(AttachmentBlend {
                    src_color_blend_factor: BlendFactor::SrcAlpha,
                    dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                    src_alpha_blend_factor: BlendFactor::One,
                    dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )),
        Some(DepthStencilState {
            depth,
            ..Default::default()
        }),
    )
}

/// The grid tests against the scene's depth but writes none of its own: a
/// translucent plane that occluded the gizmo drawn after it would be worse
/// than one that never occludes anything.
fn build_grid_pipeline(device: Arc<Device>) -> Arc<GraphicsPipeline> {
    let (stages, layout) = stages(
        &device,
        shaders::grid_vs::load(device.clone()).expect("grid_vs load"),
        shaders::grid_fs::load(device.clone()).expect("grid_fs load"),
    );
    let (color_blend_state, depth_stencil_state) = blended(Some(DepthState {
        write_enable: false,
        compare_op: CompareOp::Less,
    }));
    GraphicsPipeline::new(
        device,
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(VertexInputState::default()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state,
            color_blend_state,
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(PipelineSubpassType::BeginRendering(
                PipelineRenderingCreateInfo {
                    color_attachment_formats: vec![Some(CAMERA_COLOR_FORMAT)],
                    depth_attachment_format: Some(CAMERA_DEPTH_FORMAT),
                    ..Default::default()
                },
            )),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .expect("grid GraphicsPipeline::new")
}

/// No depth state at all: a handle buried in the mesh it moves is a handle
/// you cannot grab, and the gizmo is the last thing drawn either way.
fn build_gizmo_pipeline(device: Arc<Device>) -> Arc<GraphicsPipeline> {
    let (stages, layout) = stages(
        &device,
        shaders::overlay_vs::load(device.clone()).expect("overlay_vs load"),
        shaders::overlay_fs::load(device.clone()).expect("overlay_fs load"),
    );
    let vertex_input_state = OverlayVertex::per_vertex()
        .definition(&stages[0].entry_point)
        .expect("overlay vertex input definition");
    let (color_blend_state, depth_stencil_state) = blended(None);
    GraphicsPipeline::new(
        device,
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState {
                cull_mode: vulkano::pipeline::graphics::rasterization::CullMode::None,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state,
            color_blend_state,
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(PipelineSubpassType::BeginRendering(
                PipelineRenderingCreateInfo {
                    color_attachment_formats: vec![Some(CAMERA_COLOR_FORMAT)],
                    depth_attachment_format: Some(CAMERA_DEPTH_FORMAT),
                    ..Default::default()
                },
            )),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .expect("overlay GraphicsPipeline::new")
}
