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
| `ComponentStorage<T>` | Per-type dense store backed by `SegStorage<Mutex<T>>` with an `AtomicU32` active-bitset. Parallel update via the engine's nested/background-capable [`numa_pool`](crates/engine-core/src/util/numa_pool.rs). |
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
| `Vertex` | `#[repr(C)]` struct holding `position: Vec3`, `normal: Vec3`, `uv: Vec2`, and `tangent: Vec4` (`xyz` tangent + `w` bitangent handedness, glTF convention — the TBN basis normal mapping needs). The repr makes byte-casting for GPU upload zero-cost. |
| `Mesh` | Indexed triangle-list: `vertices: Vec<Vertex>` + `indices: Vec<u32>`. Winding is CCW (right-handed, Y-up). Provides `triangle_count()`, `aabb() -> Option<Aabb>`, and `generate_tangents()` (Lengyel accumulation from the UVs — run by every decode path whose source authored no `TANGENT`). |
| `Aabb` | Axis-aligned bounding box computed from a `Mesh`. Provides `center()`, `extent()`, and `half_extent()`. |

`mesh::primitives` contains procedural generators for common shapes.  All primitives are unit-sized (spanning `[-0.5, 0.5]`) and centred at the origin.

| Function | Description |
|----------|-------------|
| `primitives::cube()` | Unit cube, 24 vertices / 36 indices, flat per-face normals, generated tangents. |

The actual Vulkano vertex/index buffers (`GpuMesh`) will live in `engine-render`, constructed from a `&Mesh`.

### Renderer (`engine_render`)

The renderer draws indexed meshes with a full Vulkan graphics pipeline:

| Component | Details |
|-----------|--------|
| `GpuMesh` | Uploads a CPU `Mesh` to device-local vertex/index `Subbuffer`s via `Buffer::from_iter`. |
| `GpuVertex` | `#[repr(C)]` mirror of `Vertex`; derives vulkano's `BufferContents` + `Vertex` for attribute location reflection. |
| Shaders | GLSL sources live as standalone files under [`crates/engine-render/shaders/`](crates/engine-render/shaders/) (`scene.vert`, `scene.frag`, `scatter.comp`, `mvp_build.comp`, `mvp_build_pass2.comp`, `cull_pass2_args.comp`, `hiz_reduce_depth.comp`, `hiz_reduce_mip.comp`, `signal.comp`, `ui.vert`, `ui.frag`, `ui_scatter.comp`, `ui_build_args.comp`) and are compiled to SPIR-V at build time by the `vulkano-shaders` macro via `path:` (each macro registers `cargo:rerun-if-changed` for its source). Splitting them out of `src/shaders.rs` enables editor / GLSL-LSP support, scoped recompiles when iterating on a shader, and reuse by a future SPIR-V on-disk cache. Graphics: vertex shader looks up a per-instance MVP from a storage buffer (set 0, binding 0) using `gl_InstanceIndex`; fragment shader does metallic-roughness PBR (Cook-Torrance GGX + Smith visibility + Schlick Fresnel) with tangent-space normal mapping, under one hardcoded directional light plus a flat ambient term; it resolves the per-instance material through the material redirect and each of its five maps (base color, normal, metallic-roughness, occlusion, emissive) through the texture redirect. The vertex shader also unpacks a per-instance **world TRS** (`InstXform`: world position + scale in two `vec4`s, the quaternion packed as 4×f16 in their `w` lanes) written by the cull pass, and uses it to hand the fragment stage a world-space position/normal/tangent — the projection-folded MVP alone cannot, and for a TRS the normal matrix is just `R · S⁻¹`, so no matrix inverse is involved. The camera's world position (needed for every view-dependent term) rides in the shared `sot_view_proj` buffer, whose second `mat4` element carries it — a pre-recorded scene secondary rules out a push constant. Compute (`scatter_cs`): one shader, three dispatches per frame — reads a per-frame staging buffer (`vec4` per entity slot) and a per-frame `dirty` bitmask, writes the world-scoped device-local SoT buffer for that component (position / rotation / scale share the descriptor-set layout, only the bound buffers differ). Compute (`mvp_build_cs`, pass 1 of the dual-pass occlusion cull): reads the three SoT buffers, indexed via a per-camera `instance → entity` lookup, frustum- and (against last frame's Hi-Z) occlusion-tests each, multiplies survivors by a stable device-local `sot_view_proj` and writes the per-camera MVP buffer, or appends occlusion-test candidates for pass 2. Compute (`cull_pass2_args_cs`): converts pass 1's live candidate count into pass 2's `dispatch_indirect` args. Compute (`hiz_reduce_depth_cs` / `hiz_reduce_mip_cs`): max-reduce this frame's freshly-drawn depth into a full Hi-Z mip pyramid. Compute (`mvp_build_pass2_cs`): re-tests pass 1's candidates against this frame's own Hi-Z, dispatched indirectly. Compute (`signal_cs`): trivial 1×1×1 dispatch — atomically increments a host-coherent `u32` so the host can busy-poll for early-wake instead of issuing a `vkWaitSemaphores` syscall (see ADR-0003 Path C). **See [`docs/ADR-0005`](docs/ADR-0005-dual-pass-occlusion-culling.md) for the dual-pass occlusion design.** |
| Pipeline | Single `GraphicsPipeline` created once at startup with dynamic viewport, depth testing (`D32_SFLOAT`), and `PipelineRenderingCreateInfo` for dynamic rendering (no `RenderPass`/`Framebuffer`). Color attachment format is fixed at HDR `R16G16B16A16_SFLOAT` — independent of the swapchain pixel format. |
| Camera render targets & matrices | A [`RenderCamera`](crates/engine-render/src/camera.rs) owns the offscreen color image (`R16G16B16A16_SFLOAT`, `COLOR_ATTACHMENT \| TRANSFER_SRC`) and the depth image (`D32_SFLOAT`, also `SAMPLED` for the Hi-Z build). Since [ADR-0005](docs/ADR-0005-dual-pass-occlusion-culling.md) landed, the camera owns **two full, independent copies** of the compacted-output buffers (device-local MVP storage buffer, per-instance material buffer, per-instance packed world-TRS buffer, graphics descriptor set, host indirect-command template + device indirect-args) — one for pass 1's cull (draws instances visible against last frame's Hi-Z), one for pass 2's (draws instances only pass 1's occlusion sub-test deferred, confirmed against this frame's own Hi-Z) — plus two Hi-Z pyramids (`hiz_current`/`hiz_prev`, `R32_SFLOAT`, full mip chain, fixed identities mutated in place each frame rather than swapped), a `prev_view_proj` history buffer, and the candidate record list pass 1 appends to and pass 2 consumes. The swapchain image is never used as a color attachment. Each camera carries a `CameraResolution` policy (`MatchSwapchain` today; `Fixed { w, h }` and `ScaleSwapchain { num, denom }` reserved for shadow maps / half-res reflections / editor thumbnails). Invalidation now splits along two axes: **extent-dependent** (attachments, both Hi-Z pyramids, and everything that binds their views — rebuilt by `on_swapchain_resize`) and **capacity-dependent** (both passes' MVP/indirect buffers, the candidate list, the cull sets — rebuilt by `ensure_current`, geometric ≥ 2× growth), plus the pre-existing **per-world capacity** (SoT/GPURenderers/redirect/mesh_table reallocation forces `force_full`) and **per-frame-in-flight** (`FrameSlot`) axes. See ADR-0005 for the full rebuild-scope breakdown. |
| GPU transform pipeline | A [`WorldTransformGpu`](crates/engine-render/src/transform_gpu.rs) owns the **device-local SoT** ("source of truth") buffers — one per component (`vec4` per entity slot for position / rotation / scale, sized to `entity_capacity`, grown geometrically), **plus a single-mat4 `sot_view_proj`** — plus the three compute pipelines (`scatter_cs`, `mvp_build_cs`, `signal_cs`), **and (post ADR-0003 Path C) the single shared per-frame host-staging buffers (TRS triple + dirty bitmasks + view_proj), the three shared scatter descriptor sets, the shared scatter compute secondary, the host-coherent `gpu_signal` u32 buffer + its descriptor set + signal compute secondary, and a host-side `next_signal_expected` counter that gates host writes against the previous frame's GPU scatter completion via a busy-poll instead of a Vulkan timeline semaphore.** Per frame, in this order, all inside the slot's pre-recorded primary CB: (1) **scatter** ×3 (one dispatch per component) reads `staging_<comp>[i]` + `dirty_<comp>[i]` and writes `sot_<comp>[i]` iff bit `i` is set; (2) three `vkCmdFillBuffer(0)` clears re-zero the dirty bitmasks; (3) `vkCmdCopyBuffer` promotes `staging_view_proj → sot_view_proj`; (4) **`signal_cs`** atomically increments `gpu_signal[0]` — vulkano auto-sync makes this fire after every read of host-shared staging is done, so the host's busy-poll on this counter wakes the moment it's safe to overwrite staging for the next frame, even though the rest of the CB is still running; (5) the camera's dual-pass occlusion cull + render sequence runs — see [ADR-0005](docs/ADR-0005-dual-pass-occlusion-culling.md) for the full pass 1 → Hi-Z build → pass 2 → history-update breakdown; **mvp_build** (pass 1) still reads stable SoT pos/rot/scale + `sot_view_proj` the same way. **Uniform staging→SoT paradigm:** mvp_build (and any future shader) reads only stable SoT — it never touches a host-shared buffer. Host writes go into staging; a per-frame compute/transfer pass promotes staging→SoT. **Dirty-only sparse upload is live:** each frame the host first calls `host_wait_for_previous_compute()` (busy-polls `gpu_signal[0]` with `spin_loop` → `yield_now` → 100µs `sleep` fallback after ~1ms; returns immediately on the first frame because the buffer is pre-zeroed), then drains `TransformHierarchy::Dirty`'s three per-component `AtomicU32` bitmasks (atomic `swap(0, Relaxed)`) directly into the **shared** staging triple + dirty buffers in one parallel `numa_pool::parallel_for` walk (256-word tasks), and writes view_proj into the shared `staging_view_proj`; the host then submits the FrameSlot primary CB and increments `next_signal_expected` so the next frame's wait knows what value to poll for. Per-component masks mean a pure-rotation frame writes zero pos/scale data on either CPU or GPU. The SoT currently stores **local** TRS — `mvp_build_cs` composes the model matrix directly without parent-chain composition; multi-level hierarchies await a GPU-side global-composition pass. **See [`docs/ADR-0003`](docs/ADR-0003-shared-staging-with-compute-sync.md) for the shared-staging refactor, the abandoned timeline-semaphore intermediates (Paths A and B), and the final GPU-write early-wake design (Path C) that beats the previous timeline version at every measured N (+36% at N=1, +27% at N=1M static, +6% at N=1M animated) while keeping the ~144 MB VRAM saving at N=1M.** |
| Present-blit | After the scene render, the recorded CB issues `vkCmdBlitImage` to copy the camera's color image into the acquired swapchain image (1:1, `Filter::Nearest`, with format conversion HDR → sRGB). Vulkano auto-tracks barriers (`COLOR_ATTACHMENT_WRITE → TRANSFER_READ` on the offscreen color, `Undefined/PresentSrc → TransferDstOptimal` on the swapchain image) and — because swapchain images report a `final_layout_requirement` of `PresentSrc` — emits the final transition back to `PresentSrc` at end-of-CB so `vkQueuePresentKHR` is satisfied. |
| Retained-mode UI | A [`UiCore`](crates/engine-render/src/ui/mod.rs) + [`UiGpu`](crates/engine-render/src/ui/gpu.rs) pair implementing [ADR-0006](docs/ADR-0006-retained-mode-ui.md) phases 1–2. The UI is **a flat array of primitive slots** in four device-local SoT arrays — `ui_quad` (geometry, 32 B), `ui_style` (appearance, 32 B), `ui_group` (clip / offset / opacity, 32 B, ~one per panel) and `ui_order` (the `(slot, gid)` draw list) — each with its own dirty bitmask, its own `scatter_prepass.comp` run (reused verbatim from the transform pipeline) and its own `ui_scatter.comp` dispatch. The split by change frequency is the same reasoning that makes position/rotation/scale three buffers: a hover rewrites one `ui_style` record and nothing else. Host-side, `SlotArray::set` compares before it marks, so **no write path can upload a value that didn't change** — a relayout producing identical rects, or re-setting a label to the string it already holds, costs zero bytes. Drawing is **one `vkCmdDrawIndirect`** with `vertex_count = 4` and `TRIANGLE_STRIP`: there is no vertex or index buffer, `ui.vert` generates each quad's corners from `gl_VertexIndex` and clips by *shrinking* the quad (the zero-area early-out is free culling for clipped, off-screen and freed slots at once), and `ui.frag` branches on a per-instance kind — analytic rounded-box SDF with per-corner radii and a border ring, `R8` glyph coverage, or a bindless `sampler2D`. Since the draw runs *after* `signal_cs`, nothing it reads may be host-visible, so `ui_build_args.comp` promotes the host-staged primitive count into a device-local `VkDrawIndirectCommand` — which is what lets the primitive count change without re-recording a command buffer. Text is one primitive per glyph out of a **static** `R8_UNORM` atlas holding every printable ASCII character (`ui/font.rs`, a 5×9 bitmap font authored as reviewable ASCII art, uploaded once at construction); labels own power-of-two runs of contiguous slots from a bucketed free list, so retyping `100` → `101` dirties exactly one glyph. Blending is premultiplied (`ONE, ONE_MINUS_SRC_ALPHA`) into the `_SRGB` swapchain image after the present-blit, so the UI is never tonemapped and the hardware blends in linear space. Layout is **CSS flexbox and grid via [`taffy`](https://crates.io/crates/taffy)** ([`ui/tree.rs`](crates/engine-render/src/ui/tree.rs)): widgets form a node tree, taffy solves it two-pass (measure bottom-up, arrange top-down), and a placement walk pushes the solved boxes through `SlotArray::set`. Taffy was chosen over a constraint solver (Cassowary / Auto Layout) for a structural reason rather than a performance one — a constraint system is one global set of equations, so nudging any variable can ripple anywhere and "relayout this subtree" isn't expressible, which is incompatible with an architecture built end-to-end on local invalidation. Its own invalidation is the same model this ADR specifies: `mark_dirty` walks up the ancestor chain and stops at the first already-dirty node, and each node caches its `(constraint) → (size)` result so a clean subtree with an unchanged constraint short-circuits. Text leaves measure from the fixed-advance bitmap font, so no callback into the run store is needed. `ui::style` re-exports taffy's vocabulary directly — it *is* the CSS box model, and a synonym layer would only obscure it. **Ownership** follows [ADR-0008](docs/ADR-0008-ui-integration.md): the renderer ships the *system*, never a UI. `UiCore` lives in a global store reached as `engine::ui::ui()` — global for the reason `asset::global` is, that a component can reach a static but not `RenderContext` — and game and editor code each build their own trees against it. The UI tree is deliberately **not** the ECS scene tree: `TransformHierarchy::len()` sizes the transform SoT, the scatter, `GPURenderers` and the cull dispatch, so a panel modelled as entities would cost world-transform work every frame to position boxes taffy has already positioned. `run_layout` runs once per frame inside the renderer, after `Scene::update`, and early-outs on a clean tree. **Input** is polled rather than callback-driven: `set_interactive` opts a node into hit testing, `hovered` / `held` / `clicked` report state that the renderer folds once per frame *before* `Scene::update`, and `pointer_captured` is what stops a click on a button from also orbiting the camera. A component already runs every frame and holds its own state, so `if ui.clicked(btn) { self.n += 1 }` needs no closure and no app-state type threaded through the widget layer. Hit testing is a reverse-DFS over the node tree (topmost interactive node wins) on pointer events, not per frame. **Measured:** `test-game`'s overlay (F6, `ENGINE_UI_TRACE=1`) holds 263 primitives at ~11 000 FPS; a readout edit that keeps the panel's width uploads **1 dirty word** (the single glyph that differs), one that changes the widest line uploads **7** (panel background + grid strip + five swatches, all genuinely resized), and every other frame uploads nothing at all. The button's fill is re-set from its pointer state *every frame* and costs nothing except on the two frames where the state actually changes — the equality gate absorbing a per-frame write is the "a hover is 32 bytes and one workgroup" claim, exercised rather than asserted. The editor's static panel is the limiting case: **4 words at frame 0 and never again** for the rest of the session. |
| Frame sync | A custom `SwapchainRenderer` (in `engine_render::swapchain`) drives `vkAcquireNextImageKHR`, `Queue::submit_unchecked`, and `Queue::present_unchecked` directly, bypassing `vulkano-util`'s present helper and the `GpuFuture` trampolines it generates. Each frame uses one image-available semaphore (cycled from a `MAX_FRAMES_IN_FLIGHT`-sized pool) plus **per-swapchain-image** in-flight fences and render-finished semaphores. The per-image fence is what gates host-side writes to that image's reusable staging buffer (see below). Submission and presentation cost exactly **one `vkQueueSubmit2` + one `vkQueuePresentKHR`** per frame. Swapchain images are created with `TRANSFER_DST | COLOR_ATTACHMENT` usage (the latter is required for `ImageView` validation and reserved for a future fullscreen present-pass) since the renderer blits into them rather than rendering into them directly. **See [`docs/ADR-0001-custom-swapchain.md`](docs/ADR-0001-custom-swapchain.md) for the full rationale and the synchronization caveats that apply when integrating compute or other tracked submits with the render path.** |
| Reusable command buffers | One `MultipleSubmit` primary command buffer **per swapchain image** ("FrameSlot"). `FrameSlot` is minimal — just the per-image `blit_secondary` (camera color → *this* slot's swapchain image) and the composing primary CB; the staging buffers (TRS + dirty + view_proj), scatter descriptor sets, scatter secondary, signal secondary, stable `sot_view_proj`, and `gpu_signal` are all **shared** on `WorldTransformGpu`, and the dual-pass cull/render secondaries (`cull_secondary`, `hiz_build_secondary`, `cull_pass2_secondary`, `history_update_secondary`, `scene_secondary_pass1`, `scene_secondary_pass2`) are per-camera on `RenderCamera` (all `SimultaneousUse`). Each frame's `vkQueueSubmit2` carries **one batch with one CB**: the FrameSlot primary, which runs `scatter_secondary` → `spawn_scatter_secondary` → `ui.scatter_secondary` → dirty `fill_buffer` ×3 → `copy_buffer(staging_view_proj → sot_view_proj)` → `signal_secondary` → `cull_secondary` (pass 1) → `begin_rendering(Clear)` → `scene_secondary_pass1` → `end_rendering` → `hiz_build_secondary` → `cull_pass2_secondary` → `history_update_secondary` → `begin_rendering(Load)` → `scene_secondary_pass2` → `end_rendering` → `blit_secondary` → `begin_rendering(swapchain, Load)` → `ui.draw_secondary` → `end_rendering`. Vulkano auto-sync inserts every barrier (scatter→cull via SoT, fill→signal via dirty, copy→cull via sot_view_proj, depth-attachment→sampled-image transitions around the Hi-Z build, etc.). See [ADR-0005](docs/ADR-0005-dual-pass-occlusion-culling.md) for why the dual-pass sequence needs two `begin_rendering` scopes. The earlier Path A split-submit (scatter primary + FrameSlot primary in two batches with a timeline semaphore between) was abandoned because the inter-batch sync + extra CB submission cost ~30µs/frame at low N; the GPU-write `signal_cs` mid-CB recovers the early-wake behavior without the syscall. Slots are rebuilt on swapchain recreation, on camera extent change, on camera capacity growth, and on **world entity-capacity growth**. **See [`docs/ADR-0002-per-frame-cb-recording.md`](docs/ADR-0002-per-frame-cb-recording.md) for the history (per-frame recording was tried and superseded due to a ~12k→8k FPS regression).** |
| Per-frame hot path | (1) Acquire image → wait per-image fence. (2) If `hierarchy.len() > world.entity_capacity()`, grow the SoT + shared staging buffers, rebuild every camera's cull set + cull secondaries, and rebuild all FrameSlots' primary CBs (per-world axis). (3) If `draws_template.len() > camera.allocated_capacity()` (or topology length changed), grow the camera's MVP/candidate buffers geometrically and rebuild the affected FrameSlots (per-camera axis). (4) **Busy-poll `WorldTransformGpu::gpu_signal[0]`** until it reaches `next_signal_expected - 1` (`spin_loop` ×64 → `yield_now` → 100µs `sleep` after ~1ms) — first frame returns immediately because the buffer is pre-zeroed. (5) Drain `TransformHierarchy::Dirty`'s per-component atomic bitmasks (`swap(0, Relaxed)` per word) directly into the **shared** `staging_dirty_{pos,rot,scl}` + `staging_{pos,rot,scl}` via raw SoA accessors (`numa_pool::parallel_for`, no per-entity `Mutex`); write `view_proj` into the shared `staging_view_proj`. (6) Submit the slot's pre-recorded primary CB — plain submit, no extra waits/signals, one `vkQueueSubmit2` + one `vkQueuePresentKHR` per frame. The CB runs scatter (uploads dirty TRS into SoT), the dirty `fill_buffer(0)` clears, the `view_proj` `copy_buffer`, `signal_cs`, then the dual-pass occlusion cull + render sequence (see the row above), then blit. Increment `next_signal_expected` after submit so the next frame's poll knows the new target. No CB recording, no descriptor-set allocation, no buffer allocation per frame in steady state. |
| Vertex shader | `gl_InstanceIndex` (== `firstInstance + i_within_group`, where `firstInstance` is the per-mesh base offset baked into each `DrawIndexedIndirectCommand`) indexes a `readonly buffer Matrices { mat4 mvp[]; }` storage buffer that the **mvp-build compute** populated earlier in the same primary CB. Because instances are sorted by mesh on the CPU side at topology-change time, each mesh's MVP-buffer slice is contiguous and one indirect call fans out to all of that mesh's instances via HW instancing. No push constants. |
| Camera | Built-in [`OrbitController`](crates/engine-render/src/scene.rs) drives an [`engine_render::Camera`] each frame. Left-button drag orbits, right-button drag pans, scroll zooms. Pitch is clamped to avoid the gimbal flip; distance is clamped to a non-zero minimum. |
| Scene API | `Window::with_scene(Scene)` hands the window an owned root [`Scene`] (the convention is to call it `root` / `root_scene`); the renderer drives `Scene::update(dt)` once per frame on the event-loop thread immediately before the staging-buffer write. Per-frame game logic lives in `Component::update(&mut self, dt, &Transform)` implementations registered against that scene — there is no separate `on_update` callback. |

Drawables are declared by attaching a [`MeshRenderer`] component to an entity (`scene.add_component(e, MeshRenderer::new("foo.mesh"))`); the renderer derives its draw list from those components. (The old `Window::with_meshes` + `RenderInstance` table has been removed.)

#### Asset registry + component-driven rendering

The renderer is **component-driven** with **async mesh loading**: a scene declares drawables by attaching [`MeshRenderer`] components, geometry flows through the registry's GPU mega buffers, and meshes decode on a background thread — entities show the placeholder cube until their asset lands, then swap to it.

The registry is split across the GPU boundary so **mesh data is shareable** (e.g. a future physics system reads the same retained `Arc<Mesh>` for collision geometry):

| Type | Crate | Role |
|------|-------|------|
| `MeshId` | engine-core | Stable, write-once handle a future `MeshRenderer` component stores. Allocated per *unique requested path* (deduped, `u64` path hash). Indexes the redirect map. |
| `MeshSlot` | engine-core | Physical drawable slot. Slots `0`/`1` are the resident **placeholder** (cube) and **error** (tetrahedron — a deliberately distinct silhouette so failed loads are obvious) meshes. |
| `MeshBounds` | engine-core | Local-space bounding sphere per slot (GPU culling + CPU broad-phase). |
| `AssetRegistry` | engine-core | GPU-agnostic source of truth: dedup cache, `mesh_id → MeshSlot` redirect map, refcounts, and the **retained `Arc<Mesh>`** per slot. A lazily-initialized `asset::global()` (`Mutex<AssetRegistry>`, mirroring `thread_pool::global()`) lets a component constructor `request` a mesh and immediately get a `MeshId` without threading a context through the ECS. Unit-tested without a GPU. |
| `MeshTableEntry` | engine-render | Per slot, as the GPU sees it: the static `VkDrawIndexedIndirectCommand` fields (`index_count`/`first_index`/`vertex_offset`) plus the bounding sphere. std430, 32 bytes. |
| `GpuMeshStore` | engine-render | **Device-local** mirror — mega vertex/index buffers, the table, and the redirect buffer. `sync()` drains the core registry's deltas (new slots + redirect changes) and uploads them via host-staging + `vkCmdCopyBuffer`; it assigns the mega-buffer offsets (a render-side concern) as it appends. Buffers grow geometrically. |
| `MeshRenderer` | engine-render | ECS component (`HAS_UPDATE = false`) storing only a `MeshId`. `new(path)` resolves the path against `asset::global()` (so the constructor returns a handle, no path stored); `init` pushes `(transform_id, mesh_id)` onto a render-side global spawn queue the renderer drains and scatters into `GPURenderers`. |
| `GpuRenderers` | engine-render | Device-local `GPURenderers` buffer — one `mesh_id` per transform slot (indexed by `transform_id`, parallel to the SoT), sentinel `0xFFFFFFFF` for empty slots. A scatter compute (`gpu_renderers_scatter.comp`) writes drained spawns in; grows with world capacity. **This is the live instance source the cull pass reads** (Design B). |

The key decoupling: a renderer holds only a stable `MeshId`; load completion is a single redirect write (`mesh_id → slot`) — `MeshSlot::PLACEHOLDER` while loading, the real slot once resolved, `MeshSlot::ERROR` on failure — so no renderer record is ever patched and no per-renderer pending state is tracked.

**The renderer is fully GPU-driven (Design B).** There is **no CPU-sorted topology**. Each frame the pass-1 cull compute pass (`mvp_build.comp`) dispatches one invocation per transform slot and reads `GPURenderers[i] → mesh_id`, `redirect[mesh_id] → slot`, `SoT[i]` (transform), and `mesh_table[slot]` (bounds) directly. It frustum-tests the world bounding sphere (Gribb–Hartmann planes from `view_proj`, authoritative), then — for frustum-visible instances — occlusion-tests it against last frame's Hi-Z pyramid; not-occluded instances atomically claim the next slot in that drawable slot's MVP region (`base = indirect[slot].first_instance`, `local = atomicAdd(indirect[slot].instance_count, 1)`) and write the compacted MVP, while possibly-occluded ones are deferred to a second cull pass (`mvp_build_pass2.comp`) that re-tests them against this frame's own freshly-built Hi-Z. Each pass's `instance_count`s are reset every frame by a `vkCmdCopyBuffer` of a host template (counts pre-zeroed) into that pass's device args buffer, recorded just before its cull dispatch. Each pass's scene secondary then issues a **single `vkCmdDrawIndexedIndirect`** over its own `indirect[0..#slots]` against the shared mega buffers. **See [ADR-0005](docs/ADR-0005-dual-pass-occlusion-culling.md) for the full dual-pass occlusion design.**

The **only per-frame CPU work** is the `DrawPlan`: per drawable slot, the geometry (from the mesh-table mirror) + the prefix-summed `first_instance` base, where bases come from `AssetRegistry::slot_instance_totals()` (Σ refcounts per slot). That's `O(#slots)` and only runs on a topology change (spawn / load / capacity grow) — never an `O(N)` sort, and no GPU prefix sum. A spawn of an existing mesh within capacity takes the **cheap path**: the GPU scatter plus an in-place `O(#slots)` rewrite of the indirect template's bases (gated behind the per-frame compute wait, since the template is read by in-flight reset copies) — **no descriptor-set / secondary / frame-slot re-recording**. The expensive **structural rebuild** (new MVP / indirect buffers, cull set, secondaries, frame slots) happens only on **geometric capacity growth**, a new distinct mesh (`#slots` changes), or a completed load (which re-allocates a cull-bound buffer). A load itself is a one-word redirect flip whose effect the next cull picks up automatically (placeholder instances regroup onto the loaded slot).

**Async loading.** `MeshRenderer::new(path)` resolves the path against the registry (deduped); on the first request of a path it queues an `asset::request_load`, handled by a **dedicated background loader thread** (in `engine-core`, off the fork-join pool) that decodes the file (`.obj` via [`tobj`](https://crates.io/crates/tobj); paths are CWD-relative) into a CPU `Mesh` and calls `AssetRegistry::resolve` (or `fail` → error mesh, logged). Each frame `GpuMeshStore::sync()` uploads newly-resolved geometry, patches the GPU redirect buffer, and returns the per-slot totals (computed under the same registry lock, so they're consistent with the redirect the cull will read).

#### Materials & textures (metallic-roughness PBR)

Materials and textures use the **same redirect model as meshes**, so a streaming texture never forces a material re-upload and a material created before its texture decodes simply samples the placeholder.

| Type | Crate | Role |
|------|-------|------|
| `MaterialData` / `MaterialRegistry` | engine-core | The metallic-roughness parameter set: base-color factor, metallic, roughness, emissive, `normal_scale`, `occlusion_strength`, plus five optional `TextureId`s (base color, normal, metallic-roughness, occlusion, emissive). `get_or_create` content-hash-dedups, so identical materials collapse to one `MaterialId` across primitives and files. Materials resolve immediately (tiny POD — no decode phase); `update` edits in place, `duplicate` detaches for solo editing. |
| `ColorSpace` | engine-core | `Srgb` / `Linear`, decided by **usage, not by the file**: base-color and emissive maps are sRGB-encoded, normal / metallic-roughness / occlusion maps carry raw linear data. It is part of the texture dedup key, so one image referenced both ways yields two ids and two device images in two formats (`R8G8B8A8_SRGB` vs `R8G8B8A8_UNORM`). |
| `TextureRegistry` / `GpuTextureStore` | engine-core / engine-render | `TextureId → TextureSlot` redirect; slot 0 is a 1×1 white placeholder, slot 1 a magenta/black error checkerboard. Uploads are paced by a streaming time budget; redirect flips whose slot isn't resident yet stay pending. |
| `GpuMaterialStore` | engine-render | Device mirror: a 64-byte `GpuMaterial` per slot plus the `MaterialId → slot` redirect the fragment shader reads. No pacing — materials are too small to need it. |

The fragment shader walks the whole chain GPU-side: `v_material → mat_redirect → material slot → the map's raw TextureId → tex_redirect → texture slot → u_textures[slot]`. The white placeholder is the right stand-in for a still-decoding *color* map (it multiplies to the untinted factor) but not for data maps — white decodes to a 45°-tilted normal and to fully-metallic — so the normal and metallic-roughness lookups test the redirect for `PLACEHOLDER` and skip the map until its slot is resident.

**Tangents.** glTF's `TANGENT` accessor is optional and OBJ has none, so any mesh whose source didn't author tangents gets `Mesh::generate_tangents()` at decode time (per-triangle Lengyel accumulation, Gram-Schmidt-orthogonalised against the normal). glTF importing reads `normalTexture` (with `scale`), `metallicRoughnessTexture`, `occlusionTexture` (with `strength`) and `emissiveTexture`; OBJ maps `map_Kd` → base color and `map_Bump`/`bump` → normal map.

**Known follow-up:** compaction uses one global `atomicAdd` per visible instance on the slot's counter; a workgroup-local aggregation (keyed on dynamic per-workgroup slot equality) is the planned optimization.

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

The renderer draws lit cubes (warm-orange default material, metallic-roughness PBR shading) whose transforms live in an `engine_core::transform::TransformHierarchy` owned by the window's root [`Scene`]. The `test-game` defines a `Rotator` component (implementing `Component::update`) that spins each cube around its Y axis at ~45°/sec; the renderer drives `Scene::update(dt)` once per frame, which dispatches every active `update` in parallel via the engine's nested/background-capable NUMA fork-join pool ([`engine_core::util::numa_pool`](crates/engine-core/src/util/numa_pool.rs)).

Mouse controls (left-drag orbit, right-drag pan, scroll zoom) are wired through the renderer's built-in `OrbitController`.

The editor opens the test-game project by default (`--project crates/test-game`) and shows the same animated cube in its viewport, animated by an editor-side `Spinner` component until project-scene deserialisation lands. It builds its own retained-mode UI ([ADR-0008](docs/ADR-0008-ui-integration.md)) straight from `main` — a static panel naming the open project, with no per-frame update, which is why it uploads once at frame 0 and never again.

`test-game` builds a different UI from the same public API, as a `UiDemo` **component** ([`crates/test-game/src/ui_demo.rs`](crates/test-game/src/ui_demo.rs)): a rounded, bordered panel with a live readout, an ASCII specimen, a grid swatch strip, and a **click-me button with a click counter** — all laid out by taffy rather than hand-positioned. The button is an ordinary node that called `set_interactive`; there is no `Button` widget type, and the app picks its own hover/press fills from the state the engine reports. Between them the two binaries cover both regimes — event-driven and static — and neither overlay exists in the renderer, so each appears only in the binary that asked for it. **F6** toggles the game's; `ENGINE_UI_TRACE=1` prints a line only on frames where the UI uploaded anything at all.

The packager prints its intended steps without performing them.

### Stress benchmark

`test-game` accepts `--shapes N` to spawn an `N`-entity grid (and `--static-scene` to skip the per-frame `Rotator` updates). Entities cycle round-robin through cube / sphere / cylinder `MeshRenderer`s (`crates/test-game/assets/{cube,sphere,cylinder}`), exercising concurrent async mesh loads and a multi-slot `MultiDrawIndexedIndirect` once they resolve. Use `ENGINE_NUM_THREADS=1` (or its back-compat alias `RAYON_NUM_THREADS=1`) to compare single- vs multi-threaded staging writes.

#### Adaptive staging memory type (CPU ↔ GPU load balance)

The TRS staging triple can live on either side of the PCIe link, and which one is faster depends on where the frame is bottlenecked:

| `StagingMemory` | Memory type | Host writes | Scatter reads |
| --- | --- | --- | --- |
| `HostCached` | `PREFER_DEVICE \| HOST_RANDOM_ACCESS` → system RAM (GTT) | cheap, cached | pulls across PCIe, snooping the writer's caches |
| `DeviceWc` | `PREFER_DEVICE \| HOST_SEQUENTIAL_WRITE` → WC ReBAR in VRAM | stream over PCIe | local VRAM |

(`HOST_RANDOM_ACCESS` *requires* `HOST_CACHED`, which the ReBAR type is not — that required flag is what beats the `PREFER_DEVICE` beside it and pins the triple to system RAM.) Measured at N=1M with workers confined to the GPU's NUMA node: `HostCached` costs **148µs host / 320µs scatter**, `DeviceWc` **428µs / 44µs** — a near-1:1 transfer of ~280µs *between the two processors*, not a saving on either.

A frame costs `max(cpu_busy, gpu_busy)`, so `StagingBalancer` (in `lib.rs`) picks whichever mode minimises that max. It splits each side into its staging part and the rest, keeps a per-mode EMA of the staging parts, and predicts the mode it is *not* in from the last time it was in it:

```text
frame(m) = max(cpu_rest + cpu_staging[m], gpu_rest + gpu_scatter[m])
```

Only the staging terms are mode-dependent, and they are driven by the dirty-word count rather than by the camera, so they stay valid while the view changes. `cpu_rest` / `gpu_rest` are shared and refreshed every frame in whichever mode is selected, which is what lets the balancer follow a scene sliding from GPU-bound to CPU-bound.

**GPU durations are clock-normalised before they enter the model.** A GPU that is not the frame's bottleneck has idle gaps, DPM downclocks it, and every stage stretches on identical work:

| | vram (CPU-bound) | cached (GPU-bound) |
| --- | --- | --- |
| sclk / board power | 1232 MHz / 112 W | 3139 MHz / 245 W |
| GPU busy | 62% | 100% |
| `raster1` | 254µs | 116µs |

Comparing a downclocked `gpu_rest` against a full-speed `cpu_rest` concludes the GPU cannot afford the scatter — a trap, because the CPU-heavy mode manufactures the evidence that keeps it selected. `SclkMonitor` (in `gpu_telemetry.rs`) samples `freq1_input` on a background thread every 10ms and the balancer rescales each GPU sample to the reference clock; the model then works in the clock the GPU *would* run at if it were the bottleneck, which is the only case where its duration decides the frame. Reading that file costs ~19µs — 4% of a 450µs frame — so it must not be on the hot path. Without a readable clock, `auto` refuses to run rather than compare timings it knows are incomparable.

(The reference is the running maximum, seeded from `pp_dpm_sclk`. On RDNA3 that table understates reality — it tops out at 2482 MHz while `freq1_input` reports up to 3165 MHz under load — so the table alone is not a usable reference. `mclk` is pinned at 1249 MHz in every sample taken and is not a factor.)

Each side's cost is tracked as a **rolling minimum** over the last 128–256 frames rather than an average. Every input is a clean floor with spikes on top — `host_staging` reads `402.7 / 426.6 / 541.7` min/avg/max across a window, `raster1` reads `241.6 / 254.0 / 263.8` — and the spikes are asset loading, a scheduler hiccup or a DPM ramp, none of which the staging memory type can affect. The floors are stable to ~1%, which is what makes them comparable across modes, and taking a minimum means no warm-up period and no outlier rejection are needed. It does mean a *missing* sample must never reach it: a rebuild frame leaves its timestamp queries unwritten, `saturating_sub` renders that as a 0µs stage, and a minimum latches onto it permanently, so zero deltas are dropped rather than recorded.

Switching reallocates every staging slot and re-records every FrameSlot primary, hence a 5% predicted-win threshold and a 250ms minimum dwell; samples are discarded for `MAX_FRAMES_IN_FLIGHT * STAGING_SLOTS * 4` frames after a switch because the GPU timestamps read each frame come from that (image, slot) pair's previous submission. Each switch prints every term the decision rested on, so a wrong choice is distinguishable from a wrong measurement:

```text
[staging] DeviceWc 704us -> HostCached 411us | cpu_rest 298 gpu_rest 46 staging 113/406 scatter 167/34
```

Measured at N=1M, 128 threads (auto starts on `HostCached`, probes the other mode once, then holds — 3 runs per cell):

| Scene | pinned `cached` | pinned `vram` | **auto** | picked | switches |
| --- | --- | --- | --- | --- | --- |
| default (raster ~790µs, GPU-bound) | 831 FPS | 1083 FPS | **1086–1090 FPS** | `vram` | 1 |
| `ENGINE_CULL_AWAY=1` (raster 4µs, CPU-bound) | 2239 FPS | 1241 FPS | **2188–2304 FPS** | `cached` | 2 |

`ENGINE_STAGING_MODE=cached\|vram\|auto` (default `auto`) pins or frees the choice; **F6** cycles auto → cached → vram at runtime. The FPS line is tagged `staging=auto:vram` etc. so an A/B log is unambiguous. Note the ±20% run-to-run spread in both columns is `sim_update` variance, not the balancer.

**Simple work-stealing pool, initialised at startup.** `Window::run` calls `init_pinned_thread_pool` before constructing the winit event loop. `ENGINE_NUM_THREADS` (or `RAYON_NUM_THREADS`) sets the **total** participant count including the external/main caller; the pool receives `(total - 1)` worker threads. `ENGINE_NO_PIN=1` is accepted but currently a no-op (the simple pool does not pin); the flag is preserved on the CLI for a future pinning-capable scheduler. Per project rules, bad configuration panics rather than silently falling back.

The active scheduler is [`engine_core::util::my_thread_pool`](crates/engine-core/src/util/my_thread_pool.rs): a deliberately minimal work-stealing fork-join pool (~350 lines including tests). Key properties:

* **`crossbeam_deque` per-worker LIFO + shared injector.** Each worker owns a `Worker<Task>` (LIFO) and exposes a `Stealer` to peers. External callers (including the main thread) push into a shared `Injector`. Worker threads run a tight loop: own deque → injector → rotate through peer stealers → spin/yield/park-with-timeout.
* **`parallel_for` splits into one contiguous chunk per worker.** The body is captured by reference, exposed to tasks as a thin `*const ()` plus a monomorphised `call_body::<F>` function pointer (no `dyn Trait`, no `'static` bound on `F`). The dispatching thread blocks in `help_until` (work-first — it pops/steals while waiting), guaranteeing the body outlives every dereference.
* **Nested parallelism.** A worker that calls `parallel_for` inside a task pushes the sub-tasks onto its own deque; `help_until` then pops LIFO (depth-first) so children run before peers steal them. Idle workers steal across deques to load-balance imbalance.
* **Background tasks.** `spawn_background(f)` enqueues a single long-running job onto the current worker's deque (or the injector); a worker picks it up and stays in it. Remaining workers continue to service `parallel_for` dispatches.
* **Panics propagate.** Each chunk task wraps the user closure in `catch_unwind`; the *first* panic payload is stored, `pending` is still decremented, and the dispatching thread re-raises with `resume_unwind` after the dispatch drains. No silent failures, no permanent deadlocks.
* **Lifecycle is explicit.** `my_thread_pool::global::init(n)` must be called once at startup. `pool()` panics if invoked before init (no auto-default).

The older `numa_pool` (NUMA-aware, pinning, epoch-directed slots) and `thread_pool` (legacy static partitioner) source files are still in the tree under `engine-core::util` but are no longer wired into the engine's `parallel_for` callsites — they remain as references for the next iteration that wants pinning + NUMA placement.

```sh
cargo run --release -p test-game -- --cubes 100000
ENGINE_NUM_THREADS=1 cargo run --release -p test-game -- --cubes 100000
```

Current measured frame times (release build, multi-threaded staging, animated `Rotator` scene; **post ADR-0003 shared-staging refactor** — see [ADR-0003 §Measurements](docs/ADR-0003-shared-staging-with-compute-sync.md#measurements-post-path-a-landing) for the full pre-/post-refactor comparison and the throughput trade-off at very large N, plus [ADR-0004 §Measurements](docs/ADR-0004-instanced-indirect-draw.md#measurements-post-phase-1) for the original per-instance-vs-indirect-draw comparison):

| Cubes     | Frame time | Notes |
|---|---|---|
| 1         | ~0.12 ms  (~8 100 FPS) | GPU floor; single mesh, single instance. |
| 10 000    | ~0.77 ms  (~1 300 FPS) | |
| 100 000   | **~1.25 ms (~800 FPS)** | |
| 1 000 000 | **~4.0 ms (~250 FPS)** | At parity with (slightly faster than) the pre-refactor per-slot-staging baseline (4.55 ms), with the ~144 MB VRAM saving still banked. The uniform staging→SoT paradigm (host writes only staging, mvp_build reads only stable SoT) is what made this work — see [ADR-0003](docs/ADR-0003-shared-staging-with-compute-sync.md). |

The N≥1∘K baseline wins came from moving the per-component staging buffers into BAR / ReBAR memory (`MemoryTypeFilter::PREFER_DEVICE | HOST_RANDOM_ACCESS`) so the GPU's scatter compute reads them at full VRAM bandwidth instead of PCIe per cache line. The CPU staging-write loop runs in parallel via the engine's work-stealing pool (256 dirty-words / 8192 entities per task).

**ADR-0004 Phase 1 (instanced indirect draw) landed and was measured.** The scene secondary now records exactly **one `vkCmdDrawIndexedIndirect` per distinct mesh** instead of one `draw_indexed` per `RenderInstance`. Instances are sorted by `mesh_index` on the CPU at topology-change time so each mesh's MVP-buffer slice is contiguous; the indirect command's `instance_count` and `first_instance` fields then drive HW instancing for the entire group in one call. Required `multi_draw_indirect` and `draw_indirect_first_instance` device features are enabled at device creation. The vertex / compute shaders are unchanged — `gl_InstanceIndex` still indexes the same MVP buffer, just with a non-zero base from `first_instance`. Result: ~10× speedup at N=100K (~10 ms → ~1 ms), and N=1M is now interactive at ~4.5 ms (~220 FPS), previously not measurable. Full A/B in [ADR-0004 §Measurements](docs/ADR-0004-instanced-indirect-draw.md#measurements-post-phase-1).

**ADR-0003 (shared staging + uniform staging→SoT paradigm + split-submit) landed.** Single shared host-staging buffers replace the 4× per-FrameSlot duplication — ~144 MB saved at N=1M. The big realisation along the way: `view_proj` had to follow the same staging→SoT pattern as TRS (host writes `staging_view_proj`, the scatter primary `vkCmdCopyBuffer`s it into a stable `sot_view_proj`, mvp_build reads only that). Without that, the host's wait on the previous frame's compute had to cover mvp_build's read of `view_proj`, which serialised them and cost ~4 ms / frame at N=1M. With it, the wait fires the moment scatter+fill+copy are done (microseconds at any N), mvp_build runs in parallel with the next frame's host prep, and we end up at parity with the pre-refactor frame times at every N — still with the VRAM win. **GPU-driven frustum culling (ADR-0004 roadmap) and dual-pass temporal Hi-Z occlusion culling ([ADR-0005](docs/ADR-0005-dual-pass-occlusion-culling.md)) have since landed** — see the "fully GPU-driven (Design B)" section above.
