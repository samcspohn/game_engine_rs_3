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
| Shaders | Compile-time GLSL via `vulkano-shaders`: vertex shader applies a push-constant MVP matrix; fragment shader does diffuse + ambient shading on the face normal. |
| Pipeline | Single `GraphicsPipeline` created once at startup with dynamic viewport, depth testing (`D32_SFLOAT`), and `PipelineRenderingCreateInfo` for dynamic rendering (no `RenderPass`/`Framebuffer`). |
| Depth buffer | One `D32_SFLOAT` image per swapchain image to avoid write-after-write hazards across `SimultaneousUse` command buffers. |
| Frame sync | Pre-recorded `SimultaneousUse` command buffers (one per swapchain image) are re-submitted every frame. Before re-submitting image `i`'s command buffer, the renderer waits on a per-image `FenceSignalFuture` so vulkano's host-side resource tracking releases the depth view's write-lock. The render-then-present chain runs through that same `Arc<FenceSignalFuture<…>>`, preserving up to `num_swapchain_images` frames of CPU/GPU pipelining. |
| MVP | Perspective camera (60° FOV) looking at the origin from `(1.5, 1.5, 2.5)`, Y-axis flipped for Vulkan NDC.  Recomputed on every resize. |

`Window::with_meshes(vec![...])` is the public entry point for passing CPU meshes to the renderer.

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

## Status

The renderer draws a lit unit cube (warm-orange, diffuse + ambient shading) at the origin.
The editor opens the test-game project by default (`--project crates/test-game`) and shows the same cube in its viewport.
The packager prints its intended steps without performing them.
