#version 450

// Fragment shader — base colour (per-slot texture, or warm orange when the
// mesh has none) with a single directional light.
//
// Texture lookup chain, all GPU-side (mirroring the mesh redirect model):
//   v_slot (== gl_DrawID == drawable mesh slot, flat from the vertex stage)
//     → u_slot_tex.tex_id[slot]      (mesh slot → TextureId; 0xFFFFFFFF = none)
//     → u_tex_redirect.slot[tex_id]  (TextureId → texture slot; 0 = white
//                                     placeholder while loading, 1 = loud
//                                     magenta/black error checkerboard)
//     → u_textures[texture_slot]     (descriptor array of sampled images)
//
// The array index is dynamically uniform (same for every fragment of a
// draw), so this needs only `shader_sampled_image_array_dynamic_indexing`,
// not full descriptor indexing. MAX_TEXTURES here must match
// `GpuTextureStore::MAX_TEXTURES`; unused array elements are bound to the
// placeholder view.

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_uv;
layout(location = 2) flat in uint v_slot;
layout(location = 0) out vec4 f_color;

// TextureId → texture slot (device mirror of the texture registry redirect).
layout(set = 1, binding = 0) readonly buffer TexRedirect {
    uint slot[];
} u_tex_redirect;
// Drawable mesh slot → TextureId (0xFFFFFFFF == untextured).
layout(set = 1, binding = 1) readonly buffer SlotTexture {
    uint tex_id[];
} u_slot_tex;
layout(set = 1, binding = 2) uniform sampler2D u_textures[1024];

const uint NO_TEXTURE = 0xFFFFFFFFu;

void main() {
    vec3 base = vec3(0.85, 0.55, 0.20); // warm orange (untextured)
    uint tex_id = u_slot_tex.tex_id[v_slot];
    if (tex_id != NO_TEXTURE) {
        uint tex_slot = u_tex_redirect.slot[tex_id];
        base = texture(u_textures[tex_slot], v_uv).rgb;
    }

    vec3 light_dir = normalize(vec3(1.0, 2.0, 3.0));
    float ambient = 0.20;
    float diffuse = max(dot(normalize(v_normal), light_dir), 0.0);
    f_color = vec4(base * (ambient + diffuse), 1.0);
}
