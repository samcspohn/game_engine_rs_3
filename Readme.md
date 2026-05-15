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
| `Component` | Trait with default-empty `init`, `deinit`, `update` hooks plus a `const HAS_UPDATE: bool = true` that controls whether the per-frame `update` is dispatched. |
| `ComponentStorage<T>` | Per-type dense store backed by `SegStorage<Mutex<T>>` with an `AtomicU32` active-bitset.  Parallel update via `rayon`. |
| `ComponentRegistry` | Type-erased map of `TypeId → Box<dyn ComponentStorageTrait>`. |
| `Entity` | Newtype `u32` that indexes directly into `TransformHierarchy`. |
| `Scene` | Owns a `TransformHierarchy` + `ComponentRegistry`.  Drives `update`, `new_entity`, `add_component` (which lazily registers the storage using `T::HAS_UPDATE`), `remove_component`, `remove_entity`, `get_component`, and `instantiate` (deep-clone). |

The canonical authoring paradigm is:

```rust
let mut root = Scene::new();
let e = root.new_entity(_Transform::default());
root.add_component(e, Rotator::new());
```

No explicit `register::<T>()` call is required — `add_component` registers the storage on first use, honouring the component's `HAS_UPDATE` constant.

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
| Shaders | GLSL sources live as standalone files under [`crates/engine-render/shaders/`](crates/engine-render/shaders/) (`scene.vert`, `scene.frag`, `scatter.comp`, `mvp_build.comp`, `signal.comp`) and are compiled to SPIR-V at build time by the `vulkano-shaders` macro via `path:` (each macro registers `cargo:rerun-if-changed` for its source). Splitting them out of `src/shaders.rs` enables editor / GLSL-LSP support, scoped recompiles when iterating on a shader, and reuse by a future SPIR-V on-disk cache. **Four pipelines.** Graphics: vertex shader looks up a per-instance MVP from a storage buffer (set 0, binding 0) using `gl_InstanceIndex`; fragment shader does diffuse + ambient shading on the face normal. Compute (`scatter_cs`): one shader, three dispatches per frame — reads a per-frame staging buffer (`vec4` per entity slot) and a per-frame `dirty` bitmask, writes the world-scoped device-local SoT buffer for that component (position / rotation / scale share the descriptor-set layout, only the bound buffers differ). Compute (`mvp_build_cs`): reads the three SoT buffers, indexed via a per-camera `instance → entity` lookup, multiplies by a stable device-local `sot_view_proj` (set 1, single mat4 storage buffer) and writes the per-camera MVP buffer the vertex shader will read. Compute (`signal_cs`): trivial 1×1×1 dispatch — atomically increments a host-coherent `u32` so the host can busy-poll for early-wake instead of issuing a `vkWaitSemaphores` syscall (see ADR-0003 Path C). |
| Pipeline | Single `GraphicsPipeline` created once at startup with dynamic viewport, depth testing (`D32_SFLOAT`), and `PipelineRenderingCreateInfo` for dynamic rendering (no `RenderPass`/`Framebuffer`). Color attachment format is fixed at HDR `R16G16B16A16_SFLOAT` — independent of the swapchain pixel format. |
| Camera render targets & matrices | A [`RenderCamera`](crates/engine-render/src/camera.rs) owns the offscreen color image (`R16G16B16A16_SFLOAT`, `COLOR_ATTACHMENT \| TRANSFER_SRC`), the depth image (`D32_SFLOAT`), the **device-local MVP storage buffer** (`STORAGE_BUFFER` only — written by the mvp-build compute, read by the vertex shader), the **graphics descriptor set** that points at it, the per-camera **`instance_to_entity` lookup** (one `u32` per draw, **sorted by mesh** so each mesh's instances form a contiguous range — uploaded once per topology change), the **mvp-build set 0** (binds SoT pos/rot/scl + the lookup + MVP output), the per-camera **indirect-args buffer** (one `VkDrawIndexedIndirectCommand` per distinct mesh, `INDIRECT_BUFFER`, BAR/ReBAR-resident, populated on topology change), and the pre-recorded **scene secondary** that issues all draws against them. The swapchain image is never used as a color attachment. Each camera carries a `CameraResolution` policy (`MatchSwapchain` today; `Fixed { w, h }` and `ScaleSwapchain { num, denom }` reserved for shadow maps / half-res reflections / editor thumbnails) plus an `allocated_capacity` (matrix-buffer slot count, grown geometrically). Four orthogonal invalidation domains drive rebuilds: **(a) per-camera resolution** (extent change → rebuild attachments + scene secondary), **(b) per-camera capacity** (scene topology grows past the current slot count → re-allocate device buffer + graphics set + `instance_to_entity` + indirect-args + mvp-build set 0 + scene secondary, geometric ≥ 2× growth), **(c) per-world capacity** (entity hierarchy grows past SoT capacity → re-allocate SoT buffers + every camera's mvp-build set 0 + every FrameSlot, since FrameSlots' scatter sets capture SoT handles), **(d) per-frame-in-flight** (host-mapped staging buffers + dirty mask + view_proj uniform — lives on `FrameSlot`). Promoting the device matrices and scene secondary onto the camera (rather than per-`FrameSlot`) means resources allocate once per camera and survive swapchain resize for swapchain-independent cameras. |
| GPU transform pipeline | A [`WorldTransformGpu`](crates/engine-render/src/transform_gpu.rs) owns the **device-local SoT** ("source of truth") buffers — one per component (`vec4` per entity slot for position / rotation / scale, sized to `entity_capacity`, grown geometrically), **plus a single-mat4 `sot_view_proj`** — plus the three compute pipelines (`scatter_cs`, `mvp_build_cs`, `signal_cs`), **and (post ADR-0003 Path C) the single shared per-frame host-staging buffers (TRS triple + dirty bitmasks + view_proj), the three shared scatter descriptor sets, the shared `mvp_build_set1`, the shared scatter compute secondary, the host-coherent `gpu_signal` u32 buffer + its descriptor set + signal compute secondary, and a host-side `next_signal_expected` counter that gates host writes against the previous frame's GPU scatter completion via a busy-poll instead of a Vulkan timeline semaphore.** Per frame, in this order, all inside the slot's pre-recorded primary CB: (1) **scatter** ×3 (one dispatch per component) reads `staging_<comp>[i]` + `dirty_<comp>[i]` and writes `sot_<comp>[i]` iff bit `i` is set; (2) three `vkCmdFillBuffer(0)` clears re-zero the dirty bitmasks; (3) `vkCmdCopyBuffer` promotes `staging_view_proj → sot_view_proj`; (4) **`signal_cs`** atomically increments `gpu_signal[0]` — vulkano auto-sync makes this fire after every read of host-shared staging is done, so the host's busy-poll on this counter wakes the moment it's safe to overwrite staging for the next frame, even though the rest of the CB is still running; (5) **mvp_build** reads stable SoT pos/rot/scale + `sot_view_proj`, indexed via the per-camera `instance → entity` lookup, and writes the per-camera MVP buffer; (6) the graphics scene secondary draws using that MVP buffer. **Uniform staging→SoT paradigm:** mvp_build (and any future shader) reads only stable SoT — it never touches a host-shared buffer. Host writes go into staging; a per-frame compute/transfer pass promotes staging→SoT. **Dirty-only sparse upload is live:** each frame the host first calls `host_wait_for_previous_compute()` (busy-polls `gpu_signal[0]` with `spin_loop` → `yield_now` → 100µs `sleep` fallback after ~1ms; returns immediately on the first frame because the buffer is pre-zeroed), then drains `TransformHierarchy::Dirty`'s three per-component `AtomicU32` bitmasks (atomic `swap(0, Relaxed)`) directly into the **shared** staging triple + dirty buffers in one parallel `rayon::par_chunks_mut` walk, and writes view_proj into the shared `staging_view_proj`; the host then submits the FrameSlot primary CB and increments `next_signal_expected` so the next frame's wait knows what value to poll for. Per-component masks mean a pure-rotation frame writes zero pos/scale data on either CPU or GPU. The SoT currently stores **local** TRS — `mvp_build_cs` composes the model matrix directly without parent-chain composition; multi-level hierarchies await a GPU-side global-composition pass. **See [`docs/ADR-0003`](docs/ADR-0003-shared-staging-with-compute-sync.md) for the shared-staging refactor, the abandoned timeline-semaphore intermediates (Paths A and B), and the final GPU-write early-wake design (Path C) that beats the previous timeline version at every measured N (+36% at N=1, +27% at N=1M static, +6% at N=1M animated) while keeping the ~144 MB VRAM saving at N=1M.** |
| Present-blit | After the scene render, the recorded CB issues `vkCmdBlitImage` to copy the camera's color image into the acquired swapchain image (1:1, `Filter::Nearest`, with format conversion HDR → sRGB). Vulkano auto-tracks barriers (`COLOR_ATTACHMENT_WRITE → TRANSFER_READ` on the offscreen color, `Undefined/PresentSrc → TransferDstOptimal` on the swapchain image) and — because swapchain images report a `final_layout_requirement` of `PresentSrc` — emits the final transition back to `PresentSrc` at end-of-CB so `vkQueuePresentKHR` is satisfied. |
| Frame sync | A custom `SwapchainRenderer` (in `engine_render::swapchain`) drives `vkAcquireNextImageKHR`, `Queue::submit_unchecked`, and `Queue::present_unchecked` directly, bypassing `vulkano-util`'s present helper and the `GpuFuture` trampolines it generates. Each frame uses one image-available semaphore (cycled from a `MAX_FRAMES_IN_FLIGHT`-sized pool) plus **per-swapchain-image** in-flight fences and render-finished semaphores. The per-image fence is what gates host-side writes to that image's reusable staging buffer (see below). Submission and presentation cost exactly **one `vkQueueSubmit2` + one `vkQueuePresentKHR`** per frame. Swapchain images are created with `TRANSFER_DST | COLOR_ATTACHMENT` usage (the latter is required for `ImageView` validation and reserved for a future fullscreen present-pass) since the renderer blits into them rather than rendering into them directly. **See [`docs/ADR-0001-custom-swapchain.md`](docs/ADR-0001-custom-swapchain.md) for the full rationale and the synchronization caveats that apply when integrating compute or other tracked submits with the render path.** |
| Reusable command buffers | One `MultipleSubmit` primary command buffer **per swapchain image** ("FrameSlot"). Post ADR-0003 Path C `FrameSlot` is minimal — just the per-image `blit_secondary` (camera color → *this* slot's swapchain image) and the composing primary CB; the staging buffers (TRS + dirty + view_proj), scatter descriptor sets, scatter secondary, signal secondary, mvp_build_set1, stable `sot_view_proj`, and `gpu_signal` are all **shared** on `WorldTransformGpu`, and `mvp_build_secondary` is per-camera on `RenderCamera` (single secondary, recorded `SimultaneousUse`). Each frame's `vkQueueSubmit2` carries **one batch with one CB**: the FrameSlot primary, which runs `scatter_secondary` → dirty `fill_buffer` ×3 → `copy_buffer(staging_view_proj → sot_view_proj)` → `signal_secondary` → `mvp_build_secondary` → `begin_rendering` → `scene_secondary` → `end_rendering` → `blit_secondary`. Vulkano auto-sync inserts every barrier (scatter→mvp_build via SoT, fill→signal via dirty, copy→mvp_build via sot_view_proj, etc.). The earlier Path A split-submit (scatter primary + FrameSlot primary in two batches with a timeline semaphore between) was abandoned because the inter-batch sync + extra CB submission cost ~30µs/frame at low N; the GPU-write `signal_cs` mid-CB recovers the early-wake behavior without the syscall. Slots are rebuilt on swapchain recreation, on camera extent change, on camera capacity growth, and on **world entity-capacity growth**. **See [`docs/ADR-0002-per-frame-cb-recording.md`](docs/ADR-0002-per-frame-cb-recording.md) for the history (per-frame recording was tried and superseded due to a ~12k→8k FPS regression).** |
| Per-frame hot path | (1) Acquire image → wait per-image fence. (2) If `hierarchy.len() > world.entity_capacity()`, grow the SoT + shared staging buffers, rebuild every camera's mvp-build set 0 + mvp_build_secondary, and rebuild all FrameSlots' primary CBs (per-world axis). (3) If `draws_template.len() > camera.allocated_capacity()` (or topology length changed), grow the camera's MVP buffer geometrically and rebuild the affected FrameSlots (per-camera axis). (4) **Busy-poll `WorldTransformGpu::gpu_signal[0]`** until it reaches `next_signal_expected - 1` (`spin_loop` ×64 → `yield_now` → 100µs `sleep` after ~1ms) — first frame returns immediately because the buffer is pre-zeroed. (5) Drain `TransformHierarchy::Dirty`'s per-component atomic bitmasks (`swap(0, Relaxed)` per word) directly into the **shared** `staging_dirty_{pos,rot,scl}` + `staging_{pos,rot,scl}` via raw SoA accessors (`rayon::par_chunks_mut`, no per-entity `Mutex`); write `view_proj` into the shared `staging_view_proj`. (6) Submit the slot's pre-recorded primary CB — plain submit, no extra waits/signals, one `vkQueueSubmit2` + one `vkQueuePresentKHR` per frame. The CB runs scatter (uploads dirty TRS into SoT), the dirty `fill_buffer(0)` clears, the `view_proj` `copy_buffer`, `signal_cs` (atomically increments `gpu_signal`), mvp_build, scene render, blit. Increment `next_signal_expected` after submit so the next frame's poll knows the new target. No CB recording, no descriptor-set allocation, no buffer allocation per frame in steady state. |
| Vertex shader | `gl_InstanceIndex` (== `firstInstance + i_within_group`, where `firstInstance` is the per-mesh base offset baked into each `DrawIndexedIndirectCommand`) indexes a `readonly buffer Matrices { mat4 mvp[]; }` storage buffer that the **mvp-build compute** populated earlier in the same primary CB. Because instances are sorted by mesh on the CPU side at topology-change time, each mesh's MVP-buffer slice is contiguous and one indirect call fans out to all of that mesh's instances via HW instancing. No push constants. |
| Camera | Built-in [`OrbitController`](crates/engine-render/src/scene.rs) drives an [`engine_render::Camera`] each frame. Left-button drag orbits, right-button drag pans, scroll zooms. Pitch is clamped to avoid the gimbal flip; distance is clamped to a non-zero minimum. |
| Scene API | `Window::with_scene(Scene, Vec<RenderInstance>)` hands the window an owned root [`Scene`] (the convention is to call it `root` / `root_scene`); the renderer drives `Scene::update(dt)` once per frame on the event-loop thread immediately before the staging-buffer write. Per-frame game logic lives in `Component::update(&mut self, dt, &Transform)` implementations registered against that scene — there is no separate `on_update` callback. |

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

The renderer draws lit cubes (warm-orange, diffuse + ambient shading) whose transforms live in an `engine_core::transform::TransformHierarchy` owned by the window's root [`Scene`]. The `test-game` defines a `Rotator` component (implementing `Component::update`) that spins each cube around its Y axis at ~45°/sec; the renderer drives `Scene::update(dt)` once per frame, which dispatches every active `update` in parallel via `rayon`.

Mouse controls (left-drag orbit, right-drag pan, scroll zoom) are wired through the renderer's built-in `OrbitController`.

The editor opens the test-game project by default (`--project crates/test-game`) and shows the same animated cube in its viewport, animated by an editor-side `Spinner` component until project-scene deserialisation lands.

The packager prints its intended steps without performing them.

### Stress benchmark

`test-game` accepts `--cubes N` to spawn an `N`-cube grid (and `--static-scene` to skip the per-frame `Rotator` updates). Use `RAYON_NUM_THREADS=1` to compare single- vs multi-threaded staging writes.

```sh
cargo run --release -p test-game -- --cubes 100000
RAYON_NUM_THREADS=1 cargo run --release -p test-game -- --cubes 100000
```

Current measured frame times (release build, multi-threaded staging, animated `Rotator` scene; **post ADR-0003 shared-staging refactor** — see [ADR-0003 §Measurements](docs/ADR-0003-shared-staging-with-compute-sync.md#measurements-post-path-a-landing) for the full pre-/post-refactor comparison and the throughput trade-off at very large N, plus [ADR-0004 §Measurements](docs/ADR-0004-instanced-indirect-draw.md#measurements-post-phase-1) for the original per-instance-vs-indirect-draw comparison):

| Cubes     | Frame time | Notes |
|---|---|---|
| 1         | ~0.12 ms  (~8 100 FPS) | GPU floor; single mesh, single instance. |
| 10 000    | ~0.77 ms  (~1 300 FPS) | |
| 100 000   | **~1.25 ms (~800 FPS)** | |
| 1 000 000 | **~4.0 ms (~250 FPS)** | At parity with (slightly faster than) the pre-refactor per-slot-staging baseline (4.55 ms), with the ~144 MB VRAM saving still banked. The uniform staging→SoT paradigm (host writes only staging, mvp_build reads only stable SoT) is what made this work — see [ADR-0003](docs/ADR-0003-shared-staging-with-compute-sync.md). |

The N≥10K baseline wins came from moving the per-component staging buffers into BAR / ReBAR memory (`MemoryTypeFilter::PREFER_DEVICE | HOST_RANDOM_ACCESS`) so the GPU's scatter compute reads them at full VRAM bandwidth instead of PCIe per cache line. The CPU staging-write loop runs in parallel via `rayon::par_chunks_mut` (64 dirty-words / 2048 entities per task).

**ADR-0004 Phase 1 (instanced indirect draw) landed and was measured.** The scene secondary now records exactly **one `vkCmdDrawIndexedIndirect` per distinct mesh** instead of one `draw_indexed` per `RenderInstance`. Instances are sorted by `mesh_index` on the CPU at topology-change time so each mesh's MVP-buffer slice is contiguous; the indirect command's `instance_count` and `first_instance` fields then drive HW instancing for the entire group in one call. Required `multi_draw_indirect` and `draw_indirect_first_instance` device features are enabled at device creation. The vertex / compute shaders are unchanged — `gl_InstanceIndex` still indexes the same MVP buffer, just with a non-zero base from `first_instance`. Result: ~10× speedup at N=100K (~10 ms → ~1 ms), and N=1M is now interactive at ~4.5 ms (~220 FPS), previously not measurable. Full A/B in [ADR-0004 §Measurements](docs/ADR-0004-instanced-indirect-draw.md#measurements-post-phase-1).

**ADR-0003 (shared staging + uniform staging→SoT paradigm + split-submit) landed.** Single shared host-staging buffers replace the 4× per-FrameSlot duplication — ~144 MB saved at N=1M. The big realisation along the way: `view_proj` had to follow the same staging→SoT pattern as TRS (host writes `staging_view_proj`, the scatter primary `vkCmdCopyBuffer`s it into a stable `sot_view_proj`, mvp_build reads only that). Without that, the host's wait on the previous frame's compute had to cover mvp_build's read of `view_proj`, which serialised them and cost ~4 ms / frame at N=1M. With it, the wait fires the moment scatter+fill+copy are done (microseconds at any N), mvp_build runs in parallel with the next frame's host prep, and we end up at parity with the pre-refactor frame times at every N — still with the VRAM win. Next major work: ADR-0004 Phase 2 (GPU-driven frustum culling) to cut per-frame compute proportional to *visible* entities.
