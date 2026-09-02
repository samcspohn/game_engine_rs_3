#version 450

// The world grid's ground plane, as a fullscreen triangle: the fragment
// stage intersects y = 0 itself, so the grid is unbounded without any
// geometry to bound.

layout(set = 0, binding = 0) uniform Overlay {
    mat4 view_proj;
    mat4 inv_view_proj;
    vec4 eye;
    vec4 grid;
} o;

layout(location = 0) out vec3 v_near;
layout(location = 1) out vec3 v_far;

vec3 unproject(vec2 ndc, float z) {
    vec4 p = o.inv_view_proj * vec4(ndc, z, 1.0);
    return p.xyz / p.w;
}

void main() {
    vec2 ndc = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_near = unproject(ndc, 0.0);
    v_far = unproject(ndc, 1.0);
}
