# Shaders and the graphics pipeline

Where the GLSL lives, how it is compiled, and what the scene pipeline is
configured with.

GLSL sources live as standalone files under
[`crates/engine-render/shaders/`](../../crates/engine-render/shaders/)
(`scene.vert`, `scene.frag`, `scatter.comp`, `mvp_build.comp`,
`mvp_build_pass2.comp`, `cull_pass2_args.comp`, `hiz_reduce_depth.comp`,
`hiz_reduce_mip.comp`, `signal.comp`, `ui.vert`, `ui.frag`, `ui_scatter.comp`,
`ui_build_args.comp`, `grid.vert`, `grid.frag`, `overlay.vert`,
`overlay.frag`) and are compiled to SPIR-V at build time by the
`vulkano-shaders` macro via `path:` (each macro registers
`cargo:rerun-if-changed` for its source). Splitting them out of
`src/shaders.rs` enables editor / GLSL-LSP support, scoped recompiles when
iterating on a shader, and reuse by a future SPIR-V on-disk cache. Graphics:
vertex shader looks up a per-instance MVP from a storage buffer (set 0,
binding 0) using `gl_InstanceIndex`; fragment shader does metallic-roughness
PBR (Cook-Torrance GGX + Smith visibility + Schlick Fresnel) with
tangent-space normal mapping, under one hardcoded directional light plus a
flat ambient term; it resolves the per-instance material through the material
redirect and each of its five maps (base color, normal, metallic-roughness,
occlusion, emissive) through the texture redirect. The vertex shader also
unpacks a per-instance **world TRS** (`InstXform`: world position + scale in
two `vec4`s, the quaternion packed as 4×f16 in their `w` lanes) written by the
cull pass, and uses it to hand the fragment stage a world-space
position/normal/tangent — the projection-folded MVP alone cannot, and for a
TRS the normal matrix is just `R · S⁻¹`, so no matrix inverse is involved. The
camera's world position (needed for every view-dependent term) rides in the
camera's own `view_proj` buffer, whose second `mat4` element carries it — a
pre-recorded scene secondary rules out a push constant. Compute
(`scatter_cs`): one shader, three dispatches per frame — reads a per-frame
staging buffer (`vec4` per entity slot) and a per-frame `dirty` bitmask,
writes the world-scoped device-local SoT buffer for that component (position /
rotation / scale share the descriptor-set layout, only the bound buffers
differ). Compute (`mvp_build_cs`, pass 1 of the dual-pass occlusion cull):
reads the three SoT buffers, indexed via a per-camera `instance → entity`
lookup, frustum- and (against last frame's Hi-Z) occlusion-tests each,
multiplies survivors by the camera's stable device-local `view_proj` and
writes that world's MVP buffer, or appends occlusion-test candidates for pass
2. Compute (`cull_pass2_args_cs`): converts pass 1's live candidate count into
pass 2's `dispatch_indirect` args. Compute (`hiz_reduce_depth_cs` /
`hiz_reduce_mip_cs`): max-reduce this frame's freshly-drawn depth into a full
Hi-Z mip pyramid. Compute (`mvp_build_pass2_cs`): re-tests pass 1's candidates
against this frame's own Hi-Z, dispatched indirectly. Compute (`signal_cs`):
trivial 1×1×1 dispatch — atomically increments a host-coherent `u32` so the
host can busy-poll for early-wake instead of issuing a `vkWaitSemaphores`
syscall (see ADR-0003 Path C). Graphics (`grid_*` / `overlay_*`): the editor
overlay pass — see the row below. **See
[`docs/ADR-0005`](../ADR-0005-dual-pass-occlusion-culling.md) for the
dual-pass occlusion design.**

## Pipeline

Single `GraphicsPipeline` created once at startup with dynamic viewport, depth
testing (`D32_SFLOAT`), and `PipelineRenderingCreateInfo` for dynamic
rendering (no `RenderPass`/`Framebuffer`). Color attachment format is fixed at
HDR `R16G16B16A16_SFLOAT` — independent of the swapchain pixel format.

## Vertex shader

`gl_InstanceIndex` (== `firstInstance + i_within_group`, where `firstInstance`
is the per-mesh base offset baked into each `DrawIndexedIndirectCommand`)
indexes a `readonly buffer Matrices { mat4 mvp[]; }` storage buffer that the
**mvp-build compute** populated earlier in the same primary CB. Because
instances are sorted by mesh on the CPU side at topology-change time, each
mesh's MVP-buffer slice is contiguous and one indirect call fans out to all of
that mesh's instances via HW instancing. No push constants.

## Present-blit

After the scene render, the recorded CB issues `vkCmdBlitImage` to copy the
camera's color image into the acquired swapchain image (1:1,
`Filter::Nearest`, with format conversion HDR → sRGB). Vulkano auto-tracks
barriers (`COLOR_ATTACHMENT_WRITE → TRANSFER_READ` on the offscreen color,
`Undefined/PresentSrc → TransferDstOptimal` on the swapchain image) and —
because swapchain images report a `final_layout_requirement` of `PresentSrc` —
emits the final transition back to `PresentSrc` at end-of-CB so
`vkQueuePresentKHR` is satisfied.

