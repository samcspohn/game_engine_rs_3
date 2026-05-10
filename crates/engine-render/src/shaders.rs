//! Compile-time GLSL shaders for the mesh rendering pipeline.
//!
//! The vertex shader expects three per-vertex attributes (position, normal, uv)
//! and reads a per-instance pre-computed MVP from a storage buffer indexed by
//! `gl_InstanceIndex`. Each `RenderInstance` is drawn with a separate
//! `vkCmdDrawIndexed` whose `firstInstance` is the instance's slot in that
//! buffer (`gl_InstanceIndex == firstInstance`, since `instance_count = 1`).
//!
//! This is the scaffolding for a future GPU-driven indirect path
//! (`vkCmdDrawIndexedIndirectCount`) where each draw will index into a
//! mega-buffer of transforms — the storage-buffer indirection is the same.
//!
//! The fragment shader applies a simple diffuse + ambient directional light
//! so the cube faces are shaded differently and the shape reads clearly.

/// Vertex shader — looks up an MVP per instance from a storage buffer.
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r#"
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

// One MVP per instance. Indexed by gl_InstanceIndex == firstInstance
// (since every draw uses instance_count = 1).
layout(set = 0, binding = 0) readonly buffer Matrices {
    mat4 mvp[];
} u_matrices;

layout(location = 0) out vec3 v_normal;

void main() {
    gl_Position = u_matrices.mvp[gl_InstanceIndex] * vec4(position, 1.0);
    v_normal    = normal;
}
        "#
    }
}

/// Fragment shader — warm-orange base colour with a single directional light.
pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r#"
#version 450

layout(location = 0) in  vec3 v_normal;
layout(location = 0) out vec4 f_color;

void main() {
    vec3  light_dir = normalize(vec3(1.0, 2.0, 3.0));
    float ambient   = 0.20;
    float diffuse   = max(dot(normalize(v_normal), light_dir), 0.0);
    vec3  base      = vec3(0.85, 0.55, 0.20);   // warm orange
    f_color = vec4(base * (ambient + diffuse), 1.0);
}
        "#
    }
}
