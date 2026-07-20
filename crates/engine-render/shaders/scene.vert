#version 450

// Vertex shader — looks up an MVP and a material id per instance from
// storage buffers written by `mvp_build.comp` earlier in the same primary.
//
// `gl_InstanceIndex` (== firstInstance + intra-draw instance) indexes both
// per-visible-instance buffers: the compacted MVP and the concrete material
// id the cull resolved (renderer override or the mesh slot's authored
// material). The material id is passed flat to the fragment shader, which
// resolves it through the material redirect.

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

// One MVP per visible instance, filled by the MVP-build compute pass.
layout(set = 0, binding = 0) readonly buffer Matrices {
    mat4 mvp[];
} u_matrices;
// One concrete material id per visible instance, parallel to Matrices.
layout(set = 0, binding = 1) readonly buffer InstMaterial {
    uint mat_id[];
} u_inst_mat;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 v_uv;
layout(location = 2) flat out uint v_material;

void main() {
    gl_Position = u_matrices.mvp[gl_InstanceIndex] * vec4(position, 1.0);
    v_normal    = normal;
    v_uv        = uv;
    v_material  = u_inst_mat.mat_id[gl_InstanceIndex];
}
