# Engine

A Rust game engine using Vulkan (via [vulkano](https://github.com/vulkano-rs/vulkano)) for rendering. Organized as a Cargo workspace with a strict separation between game-facing APIs, editor-only APIs, and tooling.

## Architecture

```
crates/
├── engine-core/          # Core types and traits. Math/concurrency only — no GPU deps.
│   ├── transform/        # Hierarchical transform system (TransformHierarchy, Transform, …)
│   ├── component/        # ECS (Component, ComponentStorage, ComponentRegistry, Entity, Scene)
│   ├── mesh/             # CPU-side mesh data (Vertex, Mesh, Aabb) + primitive generators
│   └── util/             # Internal containers (Avail, Storage, SegStorage, Container)
├── engine-render/        # Vulkan renderer and windowing (vulkano + winit).
├── engine-editor-api/    # Editor-only engine APIs. Not on the game's dep path.
├── engine/               # Umbrella crate. Public game-facing API surface.
├── editor/               # Editor application.
├── packager/             # CLI tool that builds and bundles a game project.
└── test-game/            # Example game using the engine.
```

## engine-core in detail

### Transform system (`engine_core::transform`)

`TransformHierarchy` is a flat, SoA (struct-of-arrays) store of positions, rotations, and scales.  Each slot maps 1:1 to an entity.  Transforms support:

- Parent/child hierarchy with automatic dirty-flag propagation.
- Lock-free parallel reads via `SyncUnsafeCell`; per-slot `Mutex<()>` guards mutable access.
- `Dirty` bitsets (one `AtomicU32` per 32 slots, for position / rotation / scale / parent) that the GPU-side `TransformCompute` (in `engine-render`) can consume to upload only changed data.

`_Transform` is a plain data builder passed to `TransformHierarchy::create_transform`.  `Transform<'a>` is a borrowing handle; calling `.lock()` returns a `TransformGuard` that exposes all mutating operations.

`transform::compute` (engine-core) contains only CPU timing helpers (`PerfCounter`, `StaticPerfCounters`).  The GPU compute pipeline (`TransformCompute`, Vulkano shaders) will live in `engine-render`.

### ECS (`engine_core::component`)

| Type | Role |
|------|------|
| `Component` | Trait with default-empty `init`, `deinit`, `update` hooks. |
| `ComponentStorage<T>` | Per-type dense store backed by `SegStorage<Mutex<T>>` with an `AtomicU32` active-bitset.  Parallel update via `rayon`. |
| `ComponentRegistry` | Type-erased map of `TypeId → Box<dyn ComponentStorageTrait>`. |
| `Entity` | Newtype `u32` that indexes directly into `TransformHierarchy`. |
| `Scene` | Owns a `TransformHierarchy` + `ComponentRegistry`.  Drives `update`, `new_entity`, `add_component`, `remove_component`, `remove_entity`, `get_component`, and `instantiate` (deep-clone). |

Renderer-specific components (`RendererComponent`) will live in `engine-render` and be registered into the same `ComponentRegistry` through the existing type-erased interface.

### Mesh system (`engine_core::mesh`)

CPU-side mesh data with no GPU dependencies, following the same split as the transform system.

| Type | Role |
|------|------|
| `Vertex` | `#[repr(C)]` struct holding `position: Vec3`, `normal: Vec3`, and `uv: Vec2`. The repr makes byte-casting for GPU upload zero-cost. |
| `Mesh` | Indexed triangle-list: `vertices: Vec<Vertex>` + `indices: Vec<u32>`. Winding is CCW (right-handed, Y-up). Provides `triangle_count()` and `aabb() -> Option<Aabb>`. |
| `Aabb` | Axis-aligned bounding box computed from a `Mesh`. Provides `center()`, `extent()`, and `half_extent()`. |

`mesh::primitives` contains procedural generators for common shapes.  All primitives are unit-sized (spanning `[-0.5, 0.5]`) and centred at the origin.

| Function | Description |
|----------|-------------|
| `primitives::cube()` | Unit cube, 24 vertices / 36 indices, flat per-face normals. |

The actual Vulkano vertex/index buffers (`GpuMesh`) will live in `engine-render`, constructed from a `&Mesh`.

### Renderer (`engine_render`)

The renderer draws indexed meshes with a full Vulkan graphics pipeline:

| Component | Details |
|-----------|--------|
| `GpuMesh` | Uploads a CPU `Mesh` to device-local vertex/index `Subbuffer`s via `Buffer::from_iter`. |
| `GpuVertex` | `#[repr(C)]` mirror of `Vertex`; derives vulkano's `BufferContents` + `Vertex` for attribute location reflection. |
| Shaders | Compile-time GLSL via `vulkano-shaders`: vertex shader looks up a per-instance MVP from a storage buffer (set 0, binding 0) using `gl_InstanceIndex`; fragment shader does diffuse + ambient shading on the face normal. |
| Pipeline | Single `GraphicsPipeline` created once at startup with dynamic viewport, depth testing (`D32_SFLOAT`), and `PipelineRenderingCreateInfo` for dynamic rendering (no `RenderPass`/`Framebuffer`). Color attachment format is fixed at HDR `R16G16B16A16_SFLOAT` — independent of the swapchain pixel format. |
| Camera render targets | A [`RenderCamera`](crates/engine-render/src/camera.rs) owns the offscreen color image (`R16G16B16A16_SFLOAT`, `COLOR_ATTACHMENT \| TRANSFER_SRC`) and depth image (`D32_SFLOAT`) the scene render writes into. The swapchain image is never used as a color attachment. Each camera carries a `CameraResolution` policy that decides how its extent is derived from (or independent of) the swapchain extent: today the only variant is `MatchSwapchain` (used by the main camera so the present-blit stays a true 1:1 copy), with `Fixed { w, h }` and `ScaleSwapchain { num, denom }` reserved for future shadow maps / half-res reflection probes / editor thumbnails. On a swapchain resize the swapchain handler informs every camera of the new extent and each one decides whether to rebuild its attachments — swapchain-independent cameras (when added) survive resizes untouched without changing the swapchain handler. This is step 1 of the multi-camera / post-processing roadmap (`todo.txt`): once cameras own their own attachments and resolution policies, multiple cameras, picture-in-picture, mirrors, and HDR → sRGB tonemapping all become "another pass before the present-blit." |
| Present-blit | After the scene render, the recorded CB issues `vkCmdBlitImage` to copy the camera's color image into the acquired swapchain image (1:1, `Filter::Nearest`, with format conversion HDR → sRGB). Vulkano auto-tracks barriers (`COLOR_ATTACHMENT_WRITE → TRANSFER_READ` on the offscreen color, `Undefined/PresentSrc → TransferDstOptimal` on the swapchain image) and — because swapchain images report a `final_layout_requirement` of `PresentSrc` — emits the final transition back to `PresentSrc` at end-of-CB so `vkQueuePresentKHR` is satisfied. |
| Frame sync | A custom `SwapchainRenderer` (in `engine_render::swapchain`) drives `vkAcquireNextImageKHR`, `Queue::submit_unchecked`, and `Queue::present_unchecked` directly, bypassing `vulkano-util`'s present helper and the `GpuFuture` trampolines it generates. Each frame uses one image-available semaphore (cycled from a `MAX_FRAMES_IN_FLIGHT`-sized pool) plus **per-swapchain-image** in-flight fences and render-finished semaphores. The per-image fence is what gates host-side writes to that image's reusable staging buffer (see below). Submission and presentation cost exactly **one `vkQueueSubmit2` + one `vkQueuePresentKHR`** per frame. Swapchain images are created with `TRANSFER_DST | COLOR_ATTACHMENT` usage (the latter is required for `ImageView` validation and reserved for a future fullscreen present-pass) since the renderer blits into them rather than rendering into them directly. **See [`docs/ADR-0001-custom-swapchain.md`](docs/ADR-0001-custom-swapchain.md) for the full rationale and the synchronization caveats that apply when integrating compute or other tracked submits with the render path.** |
| Reusable command buffers | One `MultipleSubmit` primary command buffer **per swapchain image** ("FrameSlot"), composed from two `MultipleSubmit` **secondary** command buffers stratified by invalidation domain: a **scene secondary** (inherits the primary's dynamic-rendering scope; contains viewport, pipeline/descriptor binds, and one `draw_indexed` per `RenderInstance`) and a **blit secondary** (no inheritance; contains the offscreen-color → swapchain-image `vkCmdBlitImage`). The primary records `copy_buffer(staging → device)`, opens a `begin_rendering(SubpassContents::SecondaryCommandBuffers)` scope, `execute_commands`s the scene secondary, ends rendering, then `execute_commands`s the blit secondary. Vulkano's auto-sync covers in-CB *and* primary↔secondary transitions, so all barriers (`TRANSFER_WRITE → SHADER_READ`, `COLOR_ATTACHMENT_WRITE → TRANSFER_READ`, swapchain layout transitions) are inferred correctly. Each FrameSlot owns a host-mapped staging matrix buffer (`HOST_SEQUENTIAL_WRITE`, `TRANSFER_SRC`), a device-local matrix buffer (`DEVICE_LOCAL`, `STORAGE_BUFFER \| TRANSFER_DST`), a descriptor set bound to the device buffer, and the camera's offscreen color + depth images. Slots are rebuilt only on swapchain recreation. The secondary split is the foundation for per-stratum partial rebuild — when scene topology changes, only the scene secondary is rebuilt; when the swapchain image changes, only the blit secondary is rebuilt — without falling into the multi-submit-with-semaphores trap (vulkano's auto-sync is per-CB and does not insert barriers across submission boundaries). **See [`docs/ADR-0002-per-frame-cb-recording.md`](docs/ADR-0002-per-frame-cb-recording.md) for the history (per-frame recording was tried and superseded due to a ~12k→8k FPS regression).** |
| Per-frame hot path | (1) Acquire image → wait per-image fence. (2) Compute per-instance MVPs (`view_proj * model`) into a reusable scratch `Vec<[f32;16]>`. (3) `memcpy` into the slot's mapped staging buffer. (4) Submit the slot's pre-recorded CB. No CB recording, no descriptor-set allocation, no buffer allocation per frame. |
| Vertex shader | `gl_InstanceIndex` (== `firstInstance` since `instance_count = 1`) indexes a `readonly buffer Matrices { mat4 mvp[]; }` storage buffer. This is the same indirection the future `draw_indexed_indirect_count` mega-buffer path will use. No push constants. |
| Camera | Built-in [`OrbitController`](crates/engine-render/src/scene.rs) drives an [`engine_render::Camera`] each frame. Left-button drag orbits, right-button drag pans, scroll zooms. Pitch is clamped to avoid the gimbal flip; distance is clamped to a non-zero minimum. |
| Scene API | `Window::with_scene(Arc<TransformHierarchy>, Vec<RenderInstance>)` attaches a CPU scene graph; `Window::on_update(\|hierarchy, dt\| { … })` registers a per-frame callback that runs on the event-loop thread immediately before the staging-buffer write. |

`Window::with_meshes(vec![...])` defines the mesh table; each `RenderInstance { mesh_index, transform_index }` then pairs a mesh with an entity in the hierarchy. Without `with_scene` the renderer falls back to drawing every uploaded mesh at the origin (legacy behaviour for trivial test code).

### Dependency tree

```
engine-render  ──depends on──▶  engine-core  (transform + ECS)
    │
    └── vulkano, winit, GPU resources

engine  ──depends on──▶  engine-core + engine-render
```


- Games depend on `engine` only.
- The editor depends on `engine` **and** `engine-editor-api`.
- `engine` does **not** depend on `engine-editor-api`.

This is what gives the editor "privileged" access to the engine without bloating shipped game binaries. Editor-only capabilities live in a crate the game's dependency graph never touches, so the compiler enforces the boundary.

### Why crates, not cargo features?

Cargo features unify across a workspace build — if any crate in the graph enables a feature, every crate sees it enabled for that build. Putting editor-only APIs behind a feature would mean `cargo build --workspace` silently enables them for shipped games. A dedicated crate cannot leak: if the game doesn't depend on it, the symbols don't exist.

## Workflow (Makefile)

A top-level `Makefile` wraps the common `cargo` commands for quick access:

```sh
make editor   # cargo run -p editor -- --project crates/test-game
make game     # cargo run -p test-game
make build    # cargo build --workspace
make test     # cargo test --workspace
make fmt      # cargo fmt --all
make clippy   # cargo clippy --workspace -- -D warnings
```

## Commands

Build everything:

```sh
cargo build --workspace
```

Run the editor (opens the test-game project, renders the cube, prints the editor-only hello message):

```sh
cargo run -p editor
# or with an explicit project path:
cargo run -p editor -- --project crates/test-game
# or via the Makefile:
make editor
```

Run the test game standalone (cube rendered, no editor overlay):

```sh
cargo run -p test-game
# or via the Makefile:
make game
```

Invoke the packager (stub):

```sh
cargo run -p packager -- --project crates/test-game --out target/dist
```

Tests, format, lint:

```sh
cargo test --workspace
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

## Adding a new game project

1. Create a new binary crate (e.g. `crates/my-game/`).
2. Add `engine = { path = "../engine" }` to its `Cargo.toml`.
3. Add the crate to `members` in the workspace `Cargo.toml`.
4. Run with `cargo run -p my-game`.

## Adding editor-only APIs

If you need an API that only the editor should call — asset import, hot-reload, scene serialization in editor format, runtime introspection — put it in `engine-editor-api`. It will be unreachable from any game crate by construction.

## Packaging a game

`packager` is the export tool. It is invoked by the editor when the user clicks "export," and is also runnable standalone for CI builds. It is independent of the editor binary so headless builds work without a display.

Current implementation is a stub. Planned steps:

1. `cargo build --release` on the target game crate.
2. Cook assets (texture compression, mesh optimization, shader/pipeline pre-bake).
3. Bundle the binary and asset pack into the output directory.

## Documentation

- [`docs/ADR-INDEX.md`](docs/ADR-INDEX.md) — Architecture Decision Records. Start here for the *why* behind structural choices (e.g. the custom swapchain).

## Status

The renderer draws a lit unit cube (warm-orange, diffuse + ambient shading) whose transform lives in an `engine_core::transform::TransformHierarchy`. The default `on_update` spins it around its Y axis at ~45°/sec.
Mouse controls (left-drag orbit, right-drag pan, scroll zoom) are wired through the renderer's built-in `OrbitController`.
The editor opens the test-game project by default (`--project crates/test-game`) and shows the same animated cube in its viewport.
The packager prints its intended steps without performing them.
