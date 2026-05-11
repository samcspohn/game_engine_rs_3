//! Compile-time GLSL shaders for the renderer.
//!
//! Three pipelines, two stages each:
//!
//! * [`vs`] / [`fs`] — graphics. Draws indexed meshes with one *pre-computed*
//!   MVP per instance read from a storage buffer (set 0, binding 0). The MVP
//!   buffer is filled by the [`mvp_build_cs`] compute pass below; the vertex
//!   shader does not see TRS components directly.
//! * [`scatter_cs`] — compute. Promotes a per-frame host-visible component
//!   staging buffer into a device-local "source of truth" (SoT) buffer. One
//!   GLSL invocation per entity slot; copies `staging[i] → sot[i]` iff the
//!   `i`-th bit of the per-frame `dirty` bitmask is set.
//! * [`mvp_build_cs`] — compute. Reads SoT position / rotation / scale
//!   indexed via a per-camera `instance → entity` lookup table, builds the
//!   model matrix, multiplies by `view_proj`, and writes the result into the
//!   per-camera device-local MVP buffer that [`vs`] reads.
//!
//! This is the staging-only-on-the-host, "everything on the GPU" pipeline
//! sketched in `todo.txt`. The pre-recorded primary CB executes the three
//! compute secondaries (scatter ×3 component types, then mvp build) before
//! `begin_rendering`, so vulkano's auto-sync inserts the
//! `SHADER_WRITE → SHADER_READ` barriers between them automatically.

/// Vertex shader — looks up an MVP per instance from a storage buffer.
///
/// Same shader as the previous "CPU computes MVPs" path; the only thing
/// that changed is who *writes* the storage buffer (now [`mvp_build_cs`]).
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r#"
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

// One MVP per visible instance. Indexed by gl_InstanceIndex == firstInstance
// (since every draw uses instance_count = 1). Filled by the MVP-build
// compute pass earlier in the same primary command buffer.
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

/// Scatter compute — copy host-staged component values into the device-local
/// SoT buffer for entries whose dirty bit is set.
///
/// Layout is one `vec4` per entity slot for **all three** components
/// (position / rotation / scale) so a single shader + descriptor-set layout
/// works for all of them — only the bound buffers differ. Position and scale
/// pack into `vec4` with an unused `.w`; rotation is a quaternion and uses
/// all four lanes naturally.
///
/// The dirty buffer is a packed `uint`-bitset, one bit per entity slot.
/// Bit `i` set means "scatter slot `i`". The CPU is expected to clear or
/// re-set this every frame (currently we just set every bit and re-upload
/// everything — see `lib.rs` for the upload path; the dirty-only fast path
/// will land once the CPU side is wired up to `TransformHierarchy::Dirty`).
///
/// `entity_count` arrives via push constant so the shader can early-out for
/// the trailing wavefront in non-multiple-of-64 dispatches.
pub mod scatter_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r#"
#version 450

layout(local_size_x = 64) in;

layout(push_constant) uniform PC {
    uint entity_count;
} pc;

layout(set = 0, binding = 0) readonly buffer Dirty {
    uint bits[];
} u_dirty;

layout(set = 0, binding = 1) readonly buffer Stage {
    vec4 v[];
} u_stage;

layout(set = 0, binding = 2) buffer Sot {
    vec4 v[];
} u_sot;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.entity_count) return;
    uint word = u_dirty.bits[i >> 5];
    uint bit  = 1u << (i & 31u);
    if ((word & bit) != 0u) {
        u_sot.v[i] = u_stage.v[i];
    }
}
        "#
    }
}

/// MVP-build compute — read TRS from per-world SoT buffers, indexed via the
/// per-camera `instance → entity` lookup, multiply by the per-frame
/// `view_proj` and write the result into the per-camera MVP buffer the
/// vertex shader will read.
///
/// Two descriptor sets:
/// * **Set 0** — per-camera (stable across frames given fixed scene topology
///   + fixed world capacity): SoT buffers (pos/rot/scale, set/binding-stable
///   against the world), the `instance → entity` lookup, and the output MVP
///   buffer.
/// * **Set 1** — per-frame: a single-element host-visible storage buffer
///   carrying the camera's `view_proj` for this frame.
///
/// Writes one `mat4` per draw (== visible instance, today equal to the full
/// `RenderInstance` list); `gl_InstanceIndex` in [`vs`] indexes into the
/// resulting buffer.
pub mod mvp_build_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r#"
#version 450

layout(local_size_x = 64) in;

layout(push_constant) uniform PC {
    uint draw_count;
} pc;

layout(set = 0, binding = 0) readonly buffer Pos { vec4 p[]; } u_pos;
layout(set = 0, binding = 1) readonly buffer Rot { vec4 q[]; } u_rot;
layout(set = 0, binding = 2) readonly buffer Scl { vec4 s[]; } u_scl;
layout(set = 0, binding = 3) readonly buffer Idx { uint  e[]; } u_idx;
layout(set = 0, binding = 4) writeonly buffer Mvp { mat4  m[]; } u_mvp;

layout(set = 1, binding = 0) readonly buffer ViewProj {
    mat4 view_proj;
} u_vp;

mat4 trs_to_model(vec3 t, vec4 q, vec3 s) {
    float xx = q.x*q.x, yy = q.y*q.y, zz = q.z*q.z;
    float xy = q.x*q.y, xz = q.x*q.z, yz = q.y*q.z;
    float wx = q.w*q.x, wy = q.w*q.y, wz = q.w*q.z;

    // Column-major: each column is a basis vector scaled by the matching
    // scale lane, then translation in the last column.
    vec3 c0 = vec3(1.0 - 2.0*(yy + zz),  2.0*(xy + wz),       2.0*(xz - wy)) * s.x;
    vec3 c1 = vec3(2.0*(xy - wz),        1.0 - 2.0*(xx + zz), 2.0*(yz + wx)) * s.y;
    vec3 c2 = vec3(2.0*(xz + wy),        2.0*(yz - wx),       1.0 - 2.0*(xx + yy)) * s.z;
    return mat4(
        vec4(c0, 0.0),
        vec4(c1, 0.0),
        vec4(c2, 0.0),
        vec4(t,  1.0)
    );
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.draw_count) return;
    uint e = u_idx.e[i];
    mat4 model = trs_to_model(u_pos.p[e].xyz, u_rot.q[e], u_scl.s[e].xyz);
    u_mvp.m[i] = u_vp.view_proj * model;
}
        "#
    }
}
