#version 450

// Editor overlay geometry: world-space triangles the host rebuilds every
// frame (the TRS gizmo). The camera block is a buffer and not a push
// constant because this pass's secondary is recorded once and replayed.

layout(set = 0, binding = 0) uniform Overlay {
    mat4 view_proj;
    mat4 inv_view_proj;
    vec4 eye;  // xyz = eye position
    vec4 grid; // cell, major multiple, fade radius, unused
} o;

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 v_color;

void main() {
    gl_Position = o.view_proj * vec4(position, 1.0);
    v_color = color;
}
