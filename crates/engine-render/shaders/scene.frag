#version 450
#extension GL_EXT_nonuniform_qualifier : require

// Fragment shader — metallic-roughness PBR (Cook-Torrance GGX) with
// tangent-space normal mapping, lit by one hardcoded directional light plus
// a flat ambient term. Output is linear HDR; the camera target is
// `R16G16B16A16_SFLOAT` and the present blit does the encode.
//
// Material lookup chain, all GPU-side (mirroring the mesh redirect model):
//   v_material (concrete material id, flat from the vertex stage — the cull
//              already resolved renderer-override vs. mesh-authored)
//     → u_mat_redirect.slot[id]     (MaterialId → material slot; 0 = engine
//                                    default until the slot's data is
//                                    GPU-resident)
//     → u_materials.m[slot]         (factors + the raw map TextureIds)
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
// The white 1×1 placeholder is the right stand-in for a still-decoding
// *color* map (it multiplies to the untinted factor), but not for data maps:
// white decodes to a 45°-tilted normal and to fully-metallic. Those two
// therefore test the redirect for PLACEHOLDER and skip the map entirely
// until its slot is resident. Occlusion needs no such test — white is
// exactly "unoccluded".

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec4 v_tangent;
layout(location = 3) in vec2 v_uv;
layout(location = 4) flat in uint v_material;
layout(location = 0) out vec4 f_color;

// Must match `GpuMaterial` in material_store.rs (std430, 64 bytes).
struct Material {
    vec4 base_color;
    vec3 emissive;
    float roughness;
    float metallic;
    float normal_scale;
    float occlusion_strength;
    uint base_color_tex; // TextureIds; 0xFFFFFFFF == none
    uint normal_tex;
    uint metallic_roughness_tex;
    uint occlusion_tex;
    uint emissive_tex;
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
// The shared per-frame camera buffer: element 0 is the `view_proj` the cull
// passes read (unused here), followed by the camera's world position — the
// view vector every specular term needs.
layout(set = 1, binding = 4) readonly buffer Camera {
    mat4 view_proj;
    vec4 position;
} u_camera;

const uint NO_TEXTURE = 0xFFFFFFFFu;
const uint TEX_PLACEHOLDER = 0u;
const float PI = 3.14159265359;

// Hardcoded key light. `LIGHT_COLOR` carries a factor of π so a fully rough
// dielectric matches the brightness of the lambert shading this replaced.
const vec3 LIGHT_DIR = vec3(0.267261, 0.534522, 0.801784); // normalize(1,2,3)
const vec3 LIGHT_COLOR = vec3(PI);
const float AMBIENT = 0.2;

// Slot for a material's map, or 0 (PLACEHOLDER) when it has none / isn't
// resident yet — callers treat both the same for data maps.
uint map_slot(uint tex_id) {
    return tex_id == NO_TEXTURE ? TEX_PLACEHOLDER : u_tex_redirect.slot[tex_id];
}

vec4 sample_map(uint slot) {
    return texture(u_textures[nonuniformEXT(slot)], v_uv);
}

// Trowbridge-Reitz GGX normal distribution.
float distribution_ggx(float n_dot_h, float a) {
    float a2 = a * a;
    float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Smith height-correlated visibility (the G / (4·NdotV·NdotL) denominator
// folded in, so the specular term below has no explicit divide).
float visibility_smith(float n_dot_v, float n_dot_l, float a) {
    float a2 = a * a;
    float v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    float l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(v + l, 1e-7);
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

void main() {
    Material m = u_materials.m[u_mat_redirect.slot[v_material]];

    vec3 albedo = m.base_color.rgb;
    if (m.base_color_tex != NO_TEXTURE) {
        albedo *= sample_map(u_tex_redirect.slot[m.base_color_tex]).rgb;
    }

    float metallic = m.metallic;
    float roughness = m.roughness;
    uint mr_slot = map_slot(m.metallic_roughness_tex);
    if (mr_slot != TEX_PLACEHOLDER) {
        vec4 mr = sample_map(mr_slot);
        roughness *= mr.g;
        metallic *= mr.b;
    }
    roughness = clamp(roughness, 0.03, 1.0);

    // The pipeline doesn't cull back faces, so every material is effectively
    // double-sided: flip the shading normal per the rasterizer's own facing
    // determination (glTF's rule). Keying this off `dot(n, v)` instead would
    // put a shading seam along every silhouette, where that sign flips.
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) {
        n = -n;
    }
    // The bitangent below follows n, so it flips with it — as it must.
    uint normal_slot = map_slot(m.normal_tex);
    if (normal_slot != TEX_PLACEHOLDER) {
        // Gram-Schmidt: the interpolated tangent drifts off-perpendicular.
        vec3 t = normalize(v_tangent.xyz - n * dot(n, v_tangent.xyz));
        vec3 b = cross(n, t) * v_tangent.w;
        vec3 ts = sample_map(normal_slot).xyz * 2.0 - 1.0;
        ts.xy *= m.normal_scale;
        n = normalize(mat3(t, b, n) * ts);
    }

    float occlusion = 1.0;
    uint ao_slot = map_slot(m.occlusion_tex);
    if (ao_slot != TEX_PLACEHOLDER) {
        occlusion = mix(1.0, sample_map(ao_slot).r, m.occlusion_strength);
    }

    vec3 emissive = m.emissive;
    uint emissive_slot = map_slot(m.emissive_tex);
    if (emissive_slot != TEX_PLACEHOLDER) {
        emissive *= sample_map(emissive_slot).rgb;
    }

    vec3 v = normalize(u_camera.position.xyz - v_world_pos);
    vec3 l = LIGHT_DIR;
    vec3 h = normalize(v + l);
    float n_dot_v = max(dot(n, v), 1e-4);
    float n_dot_l = max(dot(n, l), 0.0);
    float n_dot_h = max(dot(n, h), 0.0);
    float v_dot_h = max(dot(v, h), 0.0);

    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    vec3 f = fresnel_schlick(v_dot_h, f0);
    float a = roughness * roughness;
    vec3 specular = f * distribution_ggx(n_dot_h, a) * visibility_smith(n_dot_v, n_dot_l, a);
    // Metals have no diffuse lobe; what the specular reflects isn't
    // transmitted.
    vec3 diffuse = (1.0 - f) * (1.0 - metallic) * albedo / PI;

    vec3 color = (diffuse + specular) * LIGHT_COLOR * n_dot_l
               + AMBIENT * albedo * occlusion
               + emissive;
    f_color = vec4(color, m.base_color.a);
}
