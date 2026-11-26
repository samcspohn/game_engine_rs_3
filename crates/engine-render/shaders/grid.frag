#version 450

layout(set = 0, binding = 0) uniform Overlay {
    mat4 view_proj;
    mat4 inv_view_proj;
    vec4 eye;
    vec4 grid; // cell, major multiple, fade radius, unused
} o;

layout(location = 0) in vec3 v_near;
layout(location = 1) in vec3 v_far;

layout(location = 0) out vec4 f_color;

/// Line coverage for one cell size, faded out as the cell shrinks under a
/// pixel — without that term a grid seen edge-on turns into solid fill.
float coverage(vec2 xz, float cell) {
    vec2 uv = xz / cell;
    vec2 d = fwidth(uv);
    vec2 g = abs(fract(uv - 0.5) - 0.5) / max(d, 1e-8);
    return (1.0 - min(min(g.x, g.y), 1.0)) * clamp(1.0 - max(d.x, d.y), 0.0, 1.0);
}

void main() {
    vec3 dir = v_far - v_near;
    float t = -v_near.y / dir.y;
    if (t < 0.0 || t > 1.0) {
        discard;
    }
    vec3 p = v_near + dir * t;

    float minor = coverage(p.xz, o.grid.x);
    float major = coverage(p.xz, o.grid.x * o.grid.y);
    float a = max(minor * 0.35, major * 0.7);

    // The two world axes through the origin, in the same red/blue the gizmo
    // uses for X and Z.
    vec3 rgb = vec3(0.55);
    if (abs(p.z) < fwidth(p.z)) {
        rgb = vec3(0.85, 0.25, 0.3);
        a = max(a, major);
    } else if (abs(p.x) < fwidth(p.x)) {
        rgb = vec3(0.25, 0.45, 0.9);
        a = max(a, major);
    }

    a *= 1.0 - smoothstep(0.6, 1.0, distance(p.xz, o.eye.xz) / o.grid.z);
    if (a <= 0.0) {
        discard;
    }

    vec4 clip = o.view_proj * vec4(p, 1.0);
    gl_FragDepth = clip.z / clip.w;
    f_color = vec4(rgb, a);
}
