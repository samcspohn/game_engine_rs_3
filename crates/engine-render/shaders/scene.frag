#version 450
#extension GL_EXT_nonuniform_qualifier : require

// Fragment shader — per-instance material (base-color factor × optional
// base-color texture, plus emissive) with a single directional light.
//
// Material lookup chain, all GPU-side (mirroring the mesh redirect model):
//   v_material (concrete material id, flat from the vertex stage — the cull
//              already resolved renderer-override vs. mesh-authored)
//     → u_mat_redirect.slot[id]     (MaterialId → material slot; 0 = engine
//                                    default until the slot's data is
//                                    GPU-resident)
//     → u_materials.m[slot]         (factors + the raw base-color TextureId)
//     → u_tex_redirect.slot[tex_id] (TextureId → texture slot; 0 = white
//                                    placeholder while decoding, 1 = loud
//                                    magenta/black error checkerboard)
//     → u_textures[texture_slot]    (descriptor array of sampled images)
//
// Because materials are per **instance**, the texture array index is *not*
// dynamically uniform within a draw — hence `nonuniformEXT` (requires the
// `shader_sampled_image_array_non_uniform_indexing` device feature).
// MAX_TEXTURES here must match `GpuTextureStore::MAX_TEXTURES`; unused array
// elements are bound to the placeholder view.
//
// Shading is deliberately minimal (ambient + lambert + emissive); metallic /
// roughness are uploaded and reserved for the planned PBR pass.

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_uv;
layout(location = 2) flat in uint v_material;
layout(location = 0) out vec4 f_color;

// Must match `GpuMaterial` in material_store.rs (std430, 48 bytes).
struct Material {
    vec4 base_color;
    vec3 emissive;
    float roughness;
    float metallic;
    uint base_color_tex; // TextureId; 0xFFFFFFFF == none
    uint pad0;
    uint pad1;
};

// TextureId → texture slot (device mirror of the texture registry redirect).
layout(set = 1, binding = 0) readonly buffer TexRedirect {
    uint slot[];
} u_tex_redirect;
// MaterialId → material slot (device mirror of the material registry;
// entries lag behind the registry until their slot's data is uploaded).
layout(set = 1, binding = 1) readonly buffer MatRedirect {
    uint slot[];
} u_mat_redirect;
// Material data per slot.
layout(set = 1, binding = 2) readonly buffer Materials {
    Material m[];
} u_materials;
layout(set = 1, binding = 3) uniform sampler2D u_textures[1024];

const uint NO_TEXTURE = 0xFFFFFFFFu;

void main() {
    Material m = u_materials.m[u_mat_redirect.slot[v_material]];

    vec3 base = m.base_color.rgb;
    if (m.base_color_tex != NO_TEXTURE) {
        uint tex_slot = u_tex_redirect.slot[m.base_color_tex];
        base *= texture(u_textures[nonuniformEXT(tex_slot)], v_uv).rgb;
    }

    vec3 light_dir = normalize(vec3(1.0, 2.0, 3.0));
    float ambient = 0.20;
    float diffuse = max(dot(normalize(v_normal), light_dir), 0.0);
    f_color = vec4(base * (ambient + diffuse) + m.emissive, m.base_color.a);
}
