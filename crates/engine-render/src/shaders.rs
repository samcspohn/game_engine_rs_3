//! Compile-time-compiled GLSL shaders for the renderer.
//!
//! Each `vulkano_shaders::shader!` macro reads a `.glsl`/`.vert`/`.frag`/
//! `.comp` file from the crate's `shaders/` directory at build time and
//! emits a Rust module exposing `load(device)` plus any push-constant /
//! interface types reflected from SPIR-V. The macro registers a
//! `cargo:rerun-if-changed` for the source file, so editing a shader
//! triggers a recompile of just this crate.
//!
//! * [`vs`] / [`fs`] (`shaders/scene.vert`, `shaders/scene.frag`) — graphics.
//!   Draws indexed meshes with one *pre-computed* MVP per instance read
//!   from a storage buffer (set 0, binding 0). The MVP buffer is filled by
//!   [`mvp_build_cs`] / [`mvp_build_pass2_cs`] below; the vertex shader
//!   does not see TRS components directly.
//! * [`scatter_cs`] (`shaders/scatter.comp`) — compute. Promotes a
//!   per-frame host-visible component staging buffer into a device-local
//!   "source of truth" (SoT) buffer. One GLSL invocation per entity slot;
//!   copies `staging[i] → sot[i]` iff the `i`-th bit of the per-frame
//!   `dirty` bitmask is set.
//! * [`mvp_build_cs`] (`shaders/mvp_build.comp`) — compute, pass 1 of the
//!   dual-pass occlusion cull. Reads SoT position / rotation / scale
//!   indexed via a per-camera `instance → entity` lookup table, builds the
//!   model matrix, frustum-tests it, occlusion-tests it against last
//!   frame's Hi-Z pyramid, and either writes the MVP [`vs`] reads or
//!   appends a candidate record for pass 2.
//! * [`cull_pass2_args_cs`] (`shaders/cull_pass2_args.comp`) — compute.
//!   Converts pass 1's live candidate count into the
//!   `VkDispatchIndirectCommand` pass 2 is dispatched with.
//! * [`hiz_reduce_depth_cs`] / [`hiz_reduce_mip_cs`]
//!   (`shaders/hiz_reduce_depth.comp`, `shaders/hiz_reduce_mip.comp`) —
//!   compute. Max-reduce this frame's freshly-drawn (pass-1) depth
//!   attachment into a full Hi-Z mip pyramid, one dispatch per level.
//! * [`mvp_build_pass2_cs`] (`shaders/mvp_build_pass2.comp`) — compute,
//!   pass 2. Dispatched indirectly (sized to the live candidate count),
//!   re-tests pass 1's candidates against this frame's own accurate Hi-Z,
//!   and writes MVP/material for the newly-revealed instances.
//!
//! This is the staging-only-on-the-host, "everything on the GPU" pipeline
//! sketched in `todo.txt`. The pre-recorded primary CB executes these
//! compute secondaries (scatter ×3 component types, mvp-build pass 1,
//! Hi-Z build, mvp-build pass 2) interleaved with two `begin_rendering`
//! scopes (see `camera.rs`), so vulkano's auto-sync inserts the
//! `SHADER_WRITE → SHADER_READ` barriers between them automatically.
//!
//! ## Why files instead of inline `src:`
//!
//! Splitting the GLSL out of the Rust source has three benefits:
//!
//! 1. Editor support — IDE plugins, syntax highlighting, formatters and
//!    the GLSL language server all work on real files but not on string
//!    literals.
//! 2. Faster iteration — touching a `.comp` file invalidates only this
//!    crate; the macro's `rerun-if-changed` does the right thing.
//! 3. A future shader cache (SPIR-V on disk, keyed by a hash of source
//!    + macros) can share the same on-disk artefact between shipped
//!    binaries and tools.

/// Vertex shader — looks up an MVP per instance from a storage buffer.
pub mod vs {
    vulkano_shaders::shader! {
        ty:   "vertex",
        path: "shaders/scene.vert",
    }
}

/// Fragment shader — warm-orange base colour with a single directional light.
pub mod fs {
    vulkano_shaders::shader! {
        ty:   "fragment",
        path: "shaders/scene.frag",
    }
}

/// Scatter compute — copy host-staged component values into the device-local
/// SoT buffer for entries whose dirty bit is set. See `shaders/scatter.comp`
/// for the GLSL and the descriptor-set layout comments.
pub mod scatter_cs {
    vulkano_shaders::shader! {
        ty:   "compute",
        path: "shaders/scatter.comp",
    }
}

/// TRS-scatter word-compaction prepass — scans a host-bounded range of a
/// dirty bitmask and compacts every nonzero word (index + value) into a
/// list `scatter_cs` reads instead of walking the raw bitmask itself. See
/// `shaders/scatter_prepass.comp`.
pub mod scatter_prepass_cs {
    vulkano_shaders::shader! {
        ty:   "compute",
        path: "shaders/scatter_prepass.comp",
    }
}

/// Tiny compute that converts the three TRS components' compacted
/// dirty-word counts (written by `scatter_prepass_cs`) into the real
/// scatter dispatch's `VkDispatchIndirectCommand`s. See
/// `shaders/scatter_build_args.comp`.
pub mod scatter_build_args_cs {
    vulkano_shaders::shader! {
        ty:   "compute",
        path: "shaders/scatter_build_args.comp",
    }
}

/// MVP-build compute, pass 1 of 2 — frustum cull (authoritative) + a
/// temporal occlusion sub-test against last frame's Hi-Z. Visible
/// instances get MVP/material written same as before; frustum-visible but
/// (stale-)occluded instances get appended to the candidate list for pass
/// 2. See `shaders/mvp_build.comp`.
pub mod mvp_build_cs {
    vulkano_shaders::shader! {
        ty:   "compute",
        path: "shaders/mvp_build.comp",
    }
}

/// MVP-build compute, pass 2 of 2 — re-tests pass 1's occlusion candidates
/// against this frame's own freshly-built Hi-Z. Dispatched indirectly,
/// sized to the live candidate count. See `shaders/mvp_build_pass2.comp`.
pub mod mvp_build_pass2_cs {
    vulkano_shaders::shader! {
        ty:   "compute",
        path: "shaders/mvp_build_pass2.comp",
    }
}

/// Tiny compute that converts pass 1's live candidate count into a
/// `VkDispatchIndirectCommand` for pass 2's indirect dispatch. See
/// `shaders/cull_pass2_args.comp`.
pub mod cull_pass2_args_cs {
    vulkano_shaders::shader! {
        ty:   "compute",
        path: "shaders/cull_pass2_args.comp",
    }
}

/// Hi-Z pyramid build, level 0 — max-reduces the real depth attachment
/// into the pyramid's first mip. See `shaders/hiz_reduce_depth.comp`.
pub mod hiz_reduce_depth_cs {
    vulkano_shaders::shader! {
        ty:   "compute",
        path: "shaders/hiz_reduce_depth.comp",
    }
}

/// Hi-Z pyramid build, levels 1..N — max-reduces the previous pyramid
/// level into the next. See `shaders/hiz_reduce_mip.comp`.
pub mod hiz_reduce_mip_cs {
    vulkano_shaders::shader! {
        ty:   "compute",
        path: "shaders/hiz_reduce_mip.comp",
    }
}

/// Hi-Z pyramid build, FUSED pair of levels — max-reduces one pyramid
/// level into the next two in a single dispatch via workgroup shared
/// memory, halving the mip-to-mip dispatch/barrier count versus running
/// [`hiz_reduce_mip_cs`] twice. See `shaders/hiz_reduce_mip2.comp`.
pub mod hiz_reduce_mip2_cs {
    vulkano_shaders::shader! {
        ty:   "compute",
        path: "shaders/hiz_reduce_mip2.comp",
    }
}

/// Early-wake signal compute — atomically increments a host-coherent u32
/// once per frame, recorded right after scatter+fill+copy in the
/// FrameSlot primary CB. The host busy-polls this counter instead of
/// blocking on a kernel-mode timeline-semaphore wait. See
/// `shaders/signal.comp` and the host poll in
/// `WorldTransformGpu::host_wait_for_previous_compute`.
pub mod signal_cs {
    vulkano_shaders::shader! {
        ty:   "compute",
        path: "shaders/signal.comp",
    }
}

/// GPURenderers scatter compute — writes newly-spawned `(transform_id,
/// mesh_id)` pairs into the per-transform `GPURenderers` buffer
/// (`gpu_renderers[transform_id] = mesh_id`). One invocation per spawn; see
/// `shaders/gpu_renderers_scatter.comp`.
pub mod gpu_renderers_scatter_cs {
    vulkano_shaders::shader! {
        ty:   "compute",
        path: "shaders/gpu_renderers_scatter.comp",
    }
}

/// Parent scatter compute — writes streamed `(transform_id, new_parent)`
/// pairs into the per-transform `Parents` buffer
/// (`parents[transform_id] = new_parent`). One invocation per parent change
/// this frame — O(changes), never O(N); see `shaders/parent_scatter.comp`.
pub mod parent_scatter_cs {
    vulkano_shaders::shader! {
        ty:   "compute",
        path: "shaders/parent_scatter.comp",
    }
}
