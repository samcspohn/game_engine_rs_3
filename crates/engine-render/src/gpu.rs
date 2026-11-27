//! Device handles, and the pipelines every command-buffer rebuild uses.

use std::sync::Arc;

use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, Queue},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        ComputePipeline, DynamicState, GraphicsPipeline, PipelineLayout,
        PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::{PipelineRenderingCreateInfo, PipelineSubpassType},
            vertex_input::{Vertex as VulkanoVertex, VertexDefinition},
            viewport::ViewportState,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
};

use crate::assets::{GpuMaterialStore, GpuMeshStore, GpuTextureStore};
use crate::camera::{CAMERA_COLOR_FORMAT, CAMERA_DEPTH_FORMAT, CameraSceneResources};
use crate::gpu_mesh::GpuVertex;
use crate::shaders;
use crate::transform_gpu::TransformGpuShared;

pub(crate) struct Gpu {
    pub queue: Arc<Queue>,
    pub memory: Arc<StandardMemoryAllocator>,
    pub cbs: Arc<StandardCommandBufferAllocator>,
    pub sets: Arc<StandardDescriptorSetAllocator>,
    pub pipelines: Pipelines,
}

/// Dual-pass occlusion culling and the scene draw, shared by every camera.
pub(crate) struct Pipelines {
    pub scene: Arc<GraphicsPipeline>,
    pub mvp_build_pass2: Arc<ComputePipeline>,
    pub cull_pass2_args: Arc<ComputePipeline>,
    pub draw_compact: Arc<ComputePipeline>,
    pub hiz_reduce_depth: Arc<ComputePipeline>,
    pub hiz_reduce_mip: Arc<ComputePipeline>,
    pub hiz_reduce_mip2: Arc<ComputePipeline>,
}

impl Pipelines {
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            scene: create_pipeline(device.clone()),
            mvp_build_pass2: create_mvp_build_pass2_pipeline(device.clone()),
            cull_pass2_args: create_cull_pass2_args_pipeline(device.clone()),
            draw_compact: create_draw_compact_pipeline(device.clone()),
            hiz_reduce_depth: create_hiz_reduce_depth_pipeline(device.clone()),
            hiz_reduce_mip: create_hiz_reduce_mip_pipeline(device.clone()),
            hiz_reduce_mip2: create_hiz_reduce_mip2_pipeline(device.clone()),
        }
    }
}

/// The GPU copies of the asset registries, synced each frame.
pub(crate) struct Stores {
    pub mesh: GpuMeshStore,
    pub texture: GpuTextureStore,
    pub material: GpuMaterialStore,
}

impl Gpu {
    pub fn scene_resources<'a>(
        &'a self,
        shared: &'a TransformGpuShared,
        stores: &'a Stores,
    ) -> CameraSceneResources<'a> {
        let p = &self.pipelines;
        CameraSceneResources {
            cb_allocator: &self.cbs,
            descriptor_set_allocator: &self.sets,
            memory_allocator: &self.memory,
            pipeline: &p.scene,
            queue_family_index: self.queue.queue_family_index(),
            mvp_build_pipeline: shared.mvp_build_pipeline(),
            mesh_store: &stores.mesh,
            texture_store: &stores.texture,
            material_store: &stores.material,
            mvp_build_pass2_pipeline: &p.mvp_build_pass2,
            cull_pass2_args_pipeline: &p.cull_pass2_args,
            draw_compact_pipeline: &p.draw_compact,
            hiz_reduce_depth_pipeline: &p.hiz_reduce_depth,
            hiz_reduce_mip_pipeline: &p.hiz_reduce_mip,
            hiz_reduce_mip2_pipeline: &p.hiz_reduce_mip2,
        }
    }
}

/// Create the single graphics pipeline used for all mesh draws.
///
/// The color attachment format is fixed at [`CAMERA_COLOR_FORMAT`] (HDR) —
/// independent of the swapchain's pixel format. The present-blit handles
/// any conversion between camera-color and swapchain formats.
fn create_pipeline(device: Arc<Device>) -> Arc<GraphicsPipeline> {
    let vs = shaders::vs::load(device.clone()).expect("Failed to load vertex shader");
    let fs = shaders::fs::load(device.clone()).expect("Failed to load fragment shader");

    let stages = [
        PipelineShaderStageCreateInfo::new(vs.entry_point("main").unwrap()),
        PipelineShaderStageCreateInfo::new(fs.entry_point("main").unwrap()),
    ];

    let vertex_input_state = GpuVertex::per_vertex()
        .definition(&stages[0].entry_point)
        .expect("Vertex input definition mismatch");

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .expect("Failed to create pipeline layout create info"),
    )
    .expect("Failed to create pipeline layout");

    GraphicsPipeline::new(
        device,
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                1,
                ColorBlendAttachmentState::default(),
            )),
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
    .expect("Failed to create graphics pipeline")
}

/// Build a single-stage compute pipeline from a loaded shader module's
/// entry point. Shared shape for the four dual-pass-occlusion-culling
/// pipelines below — mirrors `transform_gpu.rs`'s `build_scatter_pipeline`
/// / `build_mvp_build_pipeline`.
fn create_compute_pipeline(
    device: Arc<Device>,
    cs: Arc<vulkano::shader::ShaderModule>,
    label: &str,
) -> Arc<ComputePipeline> {
    let entry = cs
        .entry_point("main")
        .unwrap_or_else(|| panic!("{label} entry point"));
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(&stage))
            .into_pipeline_layout_create_info(device.clone())
            .unwrap_or_else(|_| panic!("{label} pipeline layout info")),
    )
    .unwrap_or_else(|_| panic!("{label} pipeline layout"));
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap_or_else(|_| panic!("{label} ComputePipeline::new"))
}

/// Pass 2's cull pipeline — see `shaders::mvp_build_pass2_cs`.
fn create_mvp_build_pass2_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs =
        shaders::mvp_build_pass2_cs::load(device.clone()).expect("mvp_build_pass2_cs load failed");
    create_compute_pipeline(device, cs, "mvp_build_pass2_cs")
}

/// The tiny "build pass 2's dispatch-indirect args" pipeline — see
/// `shaders::cull_pass2_args_cs`.
/// Empty-draw compaction pipeline — see `shaders::draw_compact_cs`.
fn create_draw_compact_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs = shaders::draw_compact_cs::load(device.clone()).expect("draw_compact_cs load failed");
    create_compute_pipeline(device, cs, "draw_compact_cs")
}

fn create_cull_pass2_args_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs =
        shaders::cull_pass2_args_cs::load(device.clone()).expect("cull_pass2_args_cs load failed");
    create_compute_pipeline(device, cs, "cull_pass2_args_cs")
}

/// Hi-Z pyramid level 0 (depth → mip0) pipeline — see
/// `shaders::hiz_reduce_depth_cs`.
fn create_hiz_reduce_depth_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs = shaders::hiz_reduce_depth_cs::load(device.clone())
        .expect("hiz_reduce_depth_cs load failed");
    create_compute_pipeline(device, cs, "hiz_reduce_depth_cs")
}

/// Hi-Z pyramid levels 1..N (mip[L-1] → mip[L]) pipeline — see
/// `shaders::hiz_reduce_mip_cs`.
fn create_hiz_reduce_mip_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs =
        shaders::hiz_reduce_mip_cs::load(device.clone()).expect("hiz_reduce_mip_cs load failed");
    create_compute_pipeline(device, cs, "hiz_reduce_mip_cs")
}

/// Hi-Z pyramid, fused pair of levels (mip[L-1] → mip[L] → mip[L+1] in one
/// dispatch) pipeline — see `shaders::hiz_reduce_mip2_cs`.
fn create_hiz_reduce_mip2_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs =
        shaders::hiz_reduce_mip2_cs::load(device.clone()).expect("hiz_reduce_mip2_cs load failed");
    create_compute_pipeline(device, cs, "hiz_reduce_mip2_cs")
}
