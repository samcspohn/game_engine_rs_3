#version 450

// Vertex shader — looks up an MVP per instance from a storage buffer.
//
// Same shader as the previous "CPU computes MVPs" path; the only thing
// that changed is who *writes* the storage buffer (now `mvp_build.comp`).

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

// One MVP per visible instance. Indexed by gl_InstanceIndex == firstInstance
// (since every draw uses instance_count = 1). Filled by the MVP-build
// compute pass earlier in the same primary command buffer.
layout(set = 0, binding = 0) readonly buffer Matrices {
    mat4 mvp[];
} u_matrices;

layout(location = 0) out vec3 v_normal;

void main() {
    gl_Position = u_matrices.mvp[gl_InstanceIndex] * vec4(position, 1.0);
    v_normal    = normal;
}
