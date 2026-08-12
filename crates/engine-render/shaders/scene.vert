#version 450

// Vertex shader — looks up an MVP, a world TRS and a material id per
// instance from storage buffers written by `mvp_build.comp` earlier in the
// same primary.
//
// `gl_InstanceIndex` (== firstInstance + intra-draw instance) indexes all
// three per-visible-instance buffers. `gl_Position` uses the pre-multiplied
// MVP so clip-space depth stays bit-identical to what the cull's Hi-Z was
// built against; the world TRS exists only to put the shading basis
// (position, normal, tangent) in world space for the PBR fragment stage,
// which the projection-folded MVP cannot do.
//
// The normal matrix is free here: for a TRS the inverse-transpose of the
// upper 3×3 is just `R · S⁻¹`, so normals divide by the scale and rotate,
// no matrix inverse anywhere.

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 tangent; // xyz = tangent, w = bitangent sign

// One MVP per visible instance, filled by the MVP-build compute pass.
layout(set = 0, binding = 0) readonly buffer Matrices {
    mat4 mvp[];
} u_matrices;
// One concrete material id per visible instance, parallel to Matrices.
layout(set = 0, binding = 1) readonly buffer InstMaterial {
    uint mat_id[];
} u_inst_mat;
// One world TRS per visible instance — must match `mvp_build.comp`'s
// `InstXform` (the quaternion rides the `w` lanes as 4×f16).
struct InstXform {
    vec4 pos_rot_xy;
    vec4 scale_rot_zw;
};
layout(set = 0, binding = 2) readonly buffer InstXforms {
    InstXform x[];
} u_inst_xform;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec4 v_tangent;
layout(location = 3) out vec2 v_uv;
layout(location = 4) flat out uint v_material;

// Rotate v by unit quaternion q (matches glam's `Quat * Vec3`).
vec3 quat_rotate(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main() {
    InstXform xf = u_inst_xform.x[gl_InstanceIndex];
    vec3 t = xf.pos_rot_xy.xyz;
    vec3 s = xf.scale_rot_zw.xyz;
    vec4 q = vec4(
        unpackHalf2x16(floatBitsToUint(xf.pos_rot_xy.w)),
        unpackHalf2x16(floatBitsToUint(xf.scale_rot_zw.w))
    );

    gl_Position  = u_matrices.mvp[gl_InstanceIndex] * vec4(position, 1.0);
    v_world_pos  = t + quat_rotate(q, position * s);
    v_normal     = quat_rotate(q, normal / s);
    v_tangent    = vec4(quat_rotate(q, tangent.xyz * s), tangent.w);
    v_uv         = uv;
    v_material   = u_inst_mat.mat_id[gl_InstanceIndex];
}
