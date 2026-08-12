#version 450

// UI vertex stage (ADR-0006). No vertex buffer and no index buffer exist:
// one `vkCmdDrawIndirect` with `vertex_count = 4` and `TRIANGLE_STRIP`
// topology draws every primitive in the UI, and this shader generates each
// quad's corners from `gl_VertexIndex` and its geometry from three
// slot-indexed storage buffers.
//
// `ui_order[gl_InstanceIndex]` is the draw list, maintained back-to-front on
// the host. Vulkan defines primitive order within one draw by instance
// index and blending respects primitive order, so the order array *is* the
// painter's algorithm — no depth buffer, no GPU sort, no multi-draw.
//
// Screen extent arrives as a push constant rather than a per-frame buffer:
// it changes only on resize, and a resize already re-records this
// secondary. Nothing else this pass reads varies between rebuilds, which is
// what keeps the draw side free of any promotion step.

layout(push_constant) uniform PC {
    vec2 screen; // swapchain extent in physical px
} pc;

struct UiQuad {
    vec4 rect; // x, y, w, h — group-local physical px
    vec4 uv;   // u0, v0, u1, v1 — glyph cell or texture sub-rect
};

struct UiStyle {
    // a = (fill rgba8, border rgba8, 4x u8 corner radius, f16 border width | f16 softness)
    uvec4 a;
    // b = (kind | flags, bindless TextureId, reserved, reserved)
    uvec4 b;
};

struct UiGroup {
    vec4 clip; // x0, y0, x1, y1 — already intersected with parents on the CPU
    vec4 xf;   // offset.xy, opacity, flags
};

layout(set = 0, binding = 0) readonly buffer Quads { UiQuad q[]; } u_quad;
layout(set = 0, binding = 1) readonly buffer Styles { UiStyle s[]; } u_style;
layout(set = 0, binding = 2) readonly buffer Groups { UiGroup g[]; } u_group;
layout(set = 0, binding = 3) readonly buffer Order { uvec2 e[]; } u_order;

layout(location = 0) out vec2 v_uv;
layout(location = 1) flat out uvec4 v_style_a;
layout(location = 2) flat out uvec4 v_style_b;
// Fragment position relative to the *unclipped* rect's centre, in px. The
// rounded-box SDF is evaluated against the full rect, so clipping never
// deforms the corners.
layout(location = 3) out vec2 v_local;
layout(location = 4) flat out vec2 v_half;
layout(location = 5) flat out float v_opacity;

void main() {
    uvec2 e = u_order.e[gl_InstanceIndex];
    UiQuad q = u_quad.q[e.x];
    UiStyle s = u_style.s[e.x];
    UiGroup g = u_group.g[e.y];

    vec2 p0 = q.rect.xy + g.xf.xy;
    vec2 p1 = p0 + q.rect.zw;

    // Clip by shrinking the quad, not by discarding fragments. The
    // zero-area early-out below is free per-primitive culling for three
    // cases at once: fully-clipped, fully off-screen, and **freed slots** —
    // a freed slot is just `rect.zw = 0`, so the order array does not have
    // to be compacted the instant a widget dies.
    vec2 c0 = max(p0, g.clip.xy);
    vec2 c1 = min(p1, g.clip.zw);
    if (any(greaterThanEqual(c0, c1))) {
        gl_Position = vec4(-2.0, -2.0, 0.0, 1.0); // degenerate + off-screen
        return;
    }

    vec2 t = vec2(float(gl_VertexIndex & 1), float((gl_VertexIndex >> 1) & 1));
    vec2 pos = mix(c0, c1, t);

    // uv follows the clip: interpolate against the *unclipped* rect so a
    // half-covered glyph shows its correct half.
    vec2 f = (pos - p0) / q.rect.zw;
    v_uv = mix(q.uv.xy, q.uv.zw, f);

    v_half = q.rect.zw * 0.5;
    v_local = pos - p0 - v_half;
    v_style_a = s.a;
    v_style_b = s.b;
    v_opacity = g.xf.z;

    gl_Position = vec4(pos / pc.screen * 2.0 - 1.0, 0.0, 1.0);
}
