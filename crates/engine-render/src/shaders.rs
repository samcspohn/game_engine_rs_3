//! Compile-time GLSL shaders for the mesh rendering pipeline.
//!
//! The vertex shader expects three per-vertex attributes (position, normal, uv)
//! and a single push-constant block containing the combined MVP matrix.
//! The fragment shader applies a simple diffuse + ambient directional light
//! so the cube faces are shaded differently and the shape reads clearly.

/// Vertex shader — transforms positions by MVP, passes normals to the FS.
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r#"
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(push_constant) uniform PC {
    mat4 mvp;
} pc;

layout(location = 0) out vec3 v_normal;

void main() {
    gl_Position = pc.mvp * vec4(position, 1.0);
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
