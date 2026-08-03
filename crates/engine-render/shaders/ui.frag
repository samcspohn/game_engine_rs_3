#version 450
#extension GL_EXT_nonuniform_qualifier : require

// UI fragment stage (ADR-0006). Three primitive kinds share one pipeline;
// the branch is on `kind_flags`, uniform per instance (declared `flat`).
//
// Output is **premultiplied** and **linear**. The UI draws straight into the
// swapchain image, which is an `_SRGB` format, so the hardware does the
// linear->sRGB encode on write and blends in linear space — which is the
// only way nested translucent panels and subpixel text coverage composite
// correctly. Authored colours are non-premultiplied sRGB rgba8, so this
// shader decodes then premultiplies. The blend state is
// `(ONE, ONE_MINUS_SRC_ALPHA)`.
//
// No MSAA: rounded rects are antialiased analytically from the box SDF,
// which is exact at any corner radius and costs one `clamp`.

layout(location = 0) in vec2 v_uv;
layout(location = 1) flat in uvec4 v_style_a;
layout(location = 2) flat in uvec4 v_style_b;
layout(location = 3) in vec2 v_local;
layout(location = 4) flat in vec2 v_half;
layout(location = 5) flat in float v_opacity;

layout(location = 0) out vec4 f_color;

// Dedicated R8 glyph atlas — a stable handle, deliberately *not* a slot in
// `GpuTextureStore` (whose `sync` returning `changed` triggers a full
// descriptor-set + secondary + frame-slot rebuild).
layout(set = 1, binding = 0) uniform sampler2D u_glyphs;
// The engine's bindless array; must match `GpuTextureStore::MAX_TEXTURES`.
layout(set = 1, binding = 1) uniform sampler2D u_textures[1024];

const uint KIND_RECT = 0u;
const uint KIND_TEXT = 1u;
const uint KIND_IMAGE = 2u;

vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

// rgba8 (little-endian: r in the low byte), non-premultiplied sRGB.
vec4 unpack_color(uint c) {
    vec4 s = vec4(uvec4(c, c >> 8, c >> 16, c >> 24) & 0xFFu) / 255.0;
    return vec4(srgb_to_linear(s.rgb), s.a);
}

vec4 premul(vec4 c) {
    return vec4(c.rgb * c.a, c.a);
}

// Signed distance to a rounded box with per-corner radii, after iq. `r` is
// ordered (top-right, bottom-right, top-left, bottom-left) — the selection
// below collapses it to the one radius this fragment's quadrant uses.
float sd_round_box(vec2 p, vec2 b, vec4 r) {
    r.xy = (p.x > 0.0) ? r.xy : r.zw;
    r.x = (p.y > 0.0) ? r.x : r.y;
    vec2 q = abs(p) - b + r.x;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - r.x;
}

void main() {
    uint kind = v_style_b.x & 0xFFu;
    vec4 fill = unpack_color(v_style_a.x);

    vec4 color;
    if (kind == KIND_TEXT) {
        color = premul(fill) * texture(u_glyphs, v_uv).r;
    } else if (kind == KIND_IMAGE) {
        color = premul(fill * texture(u_textures[nonuniformEXT(v_style_b.y)], v_uv));
    } else {
        uint packed = v_style_a.z;
        // Host packs (top-left, top-right, bottom-right, bottom-left) in
        // ascending byte order; reorder into the SDF's quadrant convention.
        vec4 bytes = vec4(uvec4(packed, packed >> 8, packed >> 16, packed >> 24) & 0xFFu);
        vec4 radius = vec4(bytes.y, bytes.z, bytes.x, bytes.w);

        vec2 bw = unpackHalf2x16(v_style_a.w); // .x = border width, .y = edge softness
        float d = sd_round_box(v_local, v_half, radius);
        float aa = 0.5 + bw.y;
        // Coverage of the whole shape, then of the shape inset by the border
        // width; the ring between them is the border.
        float outer = clamp((aa - d) / (2.0 * aa), 0.0, 1.0);
        float inner = clamp((aa - (d + bw.x)) / (2.0 * aa), 0.0, 1.0);
        color = premul(fill) * inner + premul(unpack_color(v_style_a.y)) * (outer - inner);
    }

    f_color = color * v_opacity;
}
