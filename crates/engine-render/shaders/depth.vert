#version 450

// Depth-only prepass vertex shader — see docs/depth-prepass-plan.md. Writes
// clip-space position only; no fragment stage is bound to this pipeline (a
// vertex-only graphics pipeline is legal — see `create_scene_pipelines`'s
// doc comment). Reads the same per-instance MVP buffer `mvp_build.comp`
// writes and `scene.vert` reads, over the same indirect-args buffer, so a
// visible instance in a given pass gets an identical clip-space transform
// here and in the color pass that follows.
//
// Only vertex attribute location 0 (position) is declared, so
// `GpuVertex::per_vertex().definition(...)` produces a vertex-input state
// that fetches position alone — normal/uv/tangent are never read, even
// though the same interleaved mega vertex buffer (full `GpuVertex` stride)
// stays bound.
//
// `invariant gl_Position` + the byte-identical expression below is REQUIRED
// for the color pass's `CompareOp::Equal` depth test — see this shader's
// pairing note in `scene.vert` for why a compiler is otherwise free to
// produce a 1-ULP-different clip depth between the two modules, which
// would silently drop fragments (holes in the geometry) rather than fail
// loudly.
//
// No alpha-tested/cutout materials exist yet (`scene.frag` never
// `discard`s). If one is added, its instances must be excluded from this
// prepass (or given a matching alpha-test fragment stage here) — otherwise
// this shader would incorrectly write opaque depth for a fragment the
// color pass will discard.

layout(location = 0) in vec3 position;

layout(set = 0, binding = 0) readonly buffer Matrices {
    mat4 mvp[];
} u_matrices;

invariant gl_Position;

void main() {
    gl_Position = u_matrices.mvp[gl_InstanceIndex] * vec4(position, 1.0);
}
