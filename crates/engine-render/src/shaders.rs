//! Compile-time-compiled GLSL shaders for the renderer.
//!
//! Each `vulkano_shaders::shader!` macro reads a `.glsl`/`.vert`/`.frag`/
//! `.comp` file from the crate's `shaders/` directory at build time and
//! emits a Rust module exposing `load(device)` plus any push-constant /
//! interface types reflected from SPIR-V. The macro registers a
//! `cargo:rerun-if-changed` for the source file, so editing a shader
//! triggers a recompile of just this crate.
//!
//! Three pipelines, two stages each:
//!
//! * [`vs`] / [`fs`] (`shaders/scene.vert`, `shaders/scene.frag`) — graphics.
//!   Draws indexed meshes with one *pre-computed* MVP per instance read
//!   from a storage buffer (set 0, binding 0). The MVP buffer is filled
//!   by the [`mvp_build_cs`] compute pass below; the vertex shader does
//!   not see TRS components directly.
//! * [`scatter_cs`] (`shaders/scatter.comp`) — compute. Promotes a
//!   per-frame host-visible component staging buffer into a device-local
//!   "source of truth" (SoT) buffer. One GLSL invocation per entity slot;
//!   copies `staging[i] → sot[i]` iff the `i`-th bit of the per-frame
//!   `dirty` bitmask is set.
//! * [`mvp_build_cs`] (`shaders/mvp_build.comp`) — compute. Reads SoT
//!   position / rotation / scale indexed via a per-camera
//!   `instance → entity` lookup table, builds the model matrix,
//!   multiplies by `view_proj`, and writes the result into the per-camera
//!   device-local MVP buffer that [`vs`] reads.
//!
//! This is the staging-only-on-the-host, "everything on the GPU" pipeline
//! sketched in `todo.txt`. The pre-recorded primary CB executes the three
//! compute secondaries (scatter ×3 component types, then mvp build) before
//! `begin_rendering`, so vulkano's auto-sync inserts the
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

/// MVP-build compute — read TRS from per-world SoT buffers, indexed via
/// the per-camera `instance → entity` lookup, multiply by the per-frame
/// `view_proj` and write the result into the per-camera MVP buffer the
/// vertex shader will read. See `shaders/mvp_build.comp`.
pub mod mvp_build_cs {
    vulkano_shaders::shader! {
        ty:   "compute",
        path: "shaders/mvp_build.comp",
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
