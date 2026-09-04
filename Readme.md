# Engine

A Rust game engine using Vulkan (via [vulkano](https://github.com/vulkano-rs/vulkano))
for rendering. A Cargo workspace with a strict separation between game-facing
APIs, editor-only APIs, and tooling.

This file is the map. It says what exists and where it lives; the *why* behind
each piece is an ADR or a note, linked from the row that mentions it. Public
signatures are generated — see [`docs/api/index.md`](docs/api/index.md).

## Architecture

```
crates/
├── engine-core/          # Core types and traits. Math/concurrency only — no GPU deps.
│   ├── transform/        # Hierarchical transform system (TransformHierarchy, Transform, …)
│   ├── component/        # ECS (Component, ComponentStorage, ComponentRegistry, Entity, World)
│   ├── mesh/             # CPU-side mesh data (Vertex, Mesh, Aabb) + primitive generators
│   ├── reflect.rs        # Value model behind #[derive(Export)] (ADR-0010 §3)
│   └── util/             # Internal containers (Avail, Storage, SegStorage, Container)
├── engine-derive/        # Proc macros. Just #[derive(Export)] today.
├── engine-render/        # Vulkan renderer and windowing (vulkano + winit).
├── engine-editor-api/    # Editor-only engine APIs. Not on the game's dep path.
├── engine/               # Umbrella crate. Public game-facing API surface.
├── editor/               # Editor application.
├── packager/             # CLI tool that builds and bundles a game project.
└── test-game/            # Example game using the engine.
```

## engine-core

### Transform system (`engine_core::transform`)

`TransformHierarchy` is a flat SoA store of positions, rotations and scales,
one slot per entity.

- Parent/child with dirty-flag propagation, rooted at **slot 0**
  ([ADR-0009](docs/ADR-0009-hierarchy-root-entity.md)). `parent == None` *is*
  that root, so "never written" and "parented to the root" are one value and
  the GPU walk needs no sentinel.
- Deletion takes the whole subtree. `remove_transform` is crate-private:
  `World::remove_entity` is the only public delete, because a transform-only
  one would leak the entity's components. Freed slots are never reused, so
  `len()` is a high-water mark and `active_len()` is the live count.
- One `WorldId` per *hierarchy*, not per slot — `Transform::world()` is a
  field read. The editor's camera rig and the document it edits are two
  hierarchies ([editor/document split](docs/notes/editor-document-split.md)).
- Lock-free parallel reads via `SyncUnsafeCell`; per-slot `Mutex<()>` for
  writes; `Dirty` bitsets (one `AtomicU32` per 32 slots, per component) drive
  the GPU upload.

`_Transform` is the plain-data builder; `Transform<'a>` borrows, and `.lock()`
gives a `TransformGuard` that can mutate.

### ECS and worlds (`engine_core::component`)

A **world** is a `TransformHierarchy`, a `ComponentRegistry` over it, and a
`simulating` flag ([ADR-0010](docs/ADR-0010-scene-authoring-and-play.md) §5,
[ADR-0011](docs/ADR-0011-worlds.md)). Worlds are engine-owned and refcounted;
the frame sweeps the simulating ones.

```rust
let root = engine::new_world();               // lives while a handle does
root.spawn(_Transform::default(), |mut e| {   // queued to the boundary
    e.add_component(OrbitController::new())
     .add_component(CameraComponent::new());
});
Window::new("My Game").with_world(root).run();
```

A game makes one world and never has to know they exist. The editor makes two
per document — the scene (`simulating: false`) and its own rig — so edit mode
is *a registry nobody sweeps* rather than a per-entity bit tested every frame.

- **A world owns its graph outright.** Slot indices are dense and zero-based
  per world, so a GPU buffer is sized to one world rather than to the process
  ([ADR-0011](docs/ADR-0011-worlds.md) §1).
- **A world answers only for its own** (§2). An entity id is a slot index,
  meaningless without its container; the same index asked of the wrong world
  answers `false` rather than someone else's component.
- **Structural changes queue to the frame boundary** (§3). `spawn` /
  `duplicate` / `destroy` run in `worlds::apply_pending` between frames,
  because growing a hierarchy reallocates the SoA a running sweep is reading.
  That is what keeps the sweep lock-free on `&World`.
- **Reaching another world is by handle** (§3), not by id — an id is not a way
  to keep a world alive.
- **Play mode falls out of it.** `duplicate_world(simulating)` deep-copies;
  stop-play is dropping the handle.
- **Every world reaches the GPU through buffers of its own** (steps 3–4), and
  a camera owns its worlds, its box and its matrix (steps 4–5). A world minted
  mid-run gets those buffers the frame a camera names it, so opening a
  document is not something the window has to be told about up front. See
  [render-camera](docs/notes/render-camera.md) and
  [transform-gpu](docs/notes/transform-gpu.md).

`Component::update(&mut self, dt, &Transform, &World)` is the only per-frame
game hook. `deinit` runs while the component is still there to react —
`MeshRenderer::deinit` scatters `NO_RENDERER` over its slot, so a deleted mesh
stops drawing with no new GPU code. There is no per-entity activation bit yet.

### Reflection (`engine_core::reflect` + `engine-derive`)

`#[derive(Export)]` gives a component a property list — name, type, get, set —
and a stable `TYPE_NAME`: one mechanism for the inspector, the save walk, the
file's component key and per-property deltas
([ADR-0010](docs/ADR-0010-scene-authoring-and-play.md) §3).

```rust
#[derive(Clone, Export)]
struct Rotator {
    #[export] speed: f32,
}
```

Two things are load-bearing (details in
[`docs/notes/reflection.md`](docs/notes/reflection.md)):

- **`#[export(get = f, set = g)]` routes through methods**, which is how
  `MeshRenderer::material` keeps the registry refcount and the GPU record in
  step. `Export::set` therefore takes the entity's `Transform`.
- **`PropertyInfo` carries the type without a value**, so an empty material
  slot still types its drop target and a texture dragged onto it is declined
  rather than dereferenced. That is the wrong-drop guard, implemented once
  instead of per widget.

The value set is closed — `f32 / i32 / bool / String / Vec3 / Quat / Color`,
`AssetRef`, `EntityRef` — so a non-`Exportable` field fails to compile rather
than degrading to a string.

### Mesh system (`engine_core::mesh`)

CPU-side mesh data with no GPU dependencies: `Vertex` (`#[repr(C)]`, position
/ normal / uv / tangent in the glTF convention), `Mesh` (indexed triangle
list, CCW, with `generate_tangents()` for sources that authored none) and
`Aabb`. `mesh::primitives` generates unit-sized shapes centred at the origin.
The device buffers live in
[`GpuMeshStore`](crates/engine-render/src/assets/gpu_store.rs) —
see [gpu-driven rendering](docs/notes/gpu-driven-rendering.md).

## engine-render

| Subsystem | What it is | Detail |
|---|---|---|
| Shaders & pipeline | GLSL under [`crates/engine-render/shaders/`](crates/engine-render/shaders/), compiled to SPIR-V at build time; one `GraphicsPipeline` with dynamic rendering and an HDR `R16G16B16A16_SFLOAT` colour target. | [shaders](docs/notes/shaders.md) |
| GPU-driven draw | No CPU-sorted topology. The cull compute reads `GPURenderers[i] → mesh_id → redirect → slot`, frustum- and occlusion-tests it, and compacts survivors into the MVP buffer. One `vkCmdDrawIndexedIndirect` per distinct mesh. | [gpu-driven rendering](docs/notes/gpu-driven-rendering.md), [ADR-0004](docs/ADR-0004-instanced-indirect-draw.md), [ADR-0005](docs/ADR-0005-dual-pass-occlusion-culling.md) |
| Assets | `MeshId` / `MaterialId` / `TextureId` are stable handles into a **redirect map**; load completion is a single redirect write, so no renderer record is ever patched. Placeholder while loading, error mesh on failure. | [gpu-driven rendering](docs/notes/gpu-driven-rendering.md), [texture-update](docs/notes/texture-update.md) |
| Transform upload | A `WorldTransformGpu` per world owns the device-local SoT buffers; the host writes only staging and the GPU scatters into the SoT, gated by a busy-polled `gpu_signal`. | [transform-gpu](docs/notes/transform-gpu.md), [ADR-0003](docs/ADR-0003-shared-staging-with-compute-sync.md) |
| Staging balancer | The staging triple lives in system RAM or in ReBAR VRAM; which is faster depends on which side is bottlenecked, so it is measured and switched at runtime. | [staging-balancer](docs/notes/staging-balancer.md) |
| Frame loop | Command buffers recorded once and replayed: one `vkQueueSubmit2` + one `vkQueuePresentKHR` per frame, per-image fences, per-camera and per-world secondaries. | [frame-loop](docs/notes/frame-loop.md), [ADR-0001](docs/ADR-0001-custom-swapchain.md), [ADR-0002](docs/ADR-0002-per-frame-cb-recording.md) |
| Cameras & viewports | A `RenderCamera` owns its attachments and Hi-Z pyramids; a `ui::Viewport` publishes the box that *sizes* them, so a scene renders at the size it is shown at. `MAX_CAMERAS` is 8. | [render-camera](docs/notes/render-camera.md) |
| Editor overlay | A `begin_rendering` scope per camera after both scene passes: the world grid and the TRS gizmo, both drawn indirectly out of host-written buffers. | [gizmo](docs/notes/gizmo.md) |
| Retained-mode UI | Four device-local SoT arrays of primitive slots, each with its own dirty bitmask and scatter. Every write is gated by a compare, so an idle frame uploads nothing. | [ui-core](docs/notes/ui-core.md), [ADR-0006](docs/ADR-0006-retained-mode-ui.md) |
| UI widgets | `label`, `image`, `button`, `checkbox`, `slider`, `text_field`, `radio_group`, `tabs`, `scroll_area`, `scrollbar`, `popup` / `context_menu`, `MenuBar`, a `DockSpace` of draggable panels, virtualized `RowList` / `TreeView`, typed drag-and-drop, and a semantic `Theme`. | [ui-widgets](docs/notes/ui-widgets.md), [authoring](docs/notes/ui-widget-authoring.md), [API](docs/api/engine-render/ui/index.md) |
| Input, focus, capture | One hit walk resolves click / hover / drop / scroll / focus. Text arrives as an ordered `Keystroke` queue — `Text` for what the OS resolved, `Key(Key, Mods)` for what to do. `poke shot` copies the composited swapchain image inside the frame's own submit. | [`input.rs`](crates/engine-render/src/input.rs), [`capture.rs`](crates/engine-render/src/capture.rs) |
| Debug input socket | `ENGINE_DEBUG_INPUT=1` opens a unix socket that injects input into the same `Input` fields winit writes and answers queries about the live UI. Window coordinates, so it cannot race the compositor. Movement is swept, not teleported. | [`debug_input.rs`](crates/engine-render/src/debug_input.rs), [`tools/poke`](tools/poke) |
| Threading | A work-stealing fork-join pool with nested parallelism, background tasks and propagating panics. `ENGINE_NUM_THREADS` sets the total participant count. | [thread-pool](docs/notes/thread-pool.md) |

Every process-wide registry behind a `global()` locks a `parking_lot::Mutex`
rather than a `std` one: `std::sync::Mutex` poisons, so a panic under a
component's `update` while one was held would kill the editor on the *next*
frame instead of at the fault — and it is precisely a caught panic that
[ADR-0010](docs/ADR-0010-scene-authoring-and-play.md) §7 wants to survive.

### Dependency tree

```
engine-core    ──depends on──▶  engine-derive  (#[derive(Export)])

engine-render  ──depends on──▶  engine-core  (transform + ECS)
    │
    └── vulkano, winit, GPU resources

engine  ──depends on──▶  engine-core + engine-render
```


- Games depend on `engine` only — including for `#[derive(Export)]`, which resolves its generated paths through whichever of `engine-core` / `engine` the deriving crate actually has (`proc-macro-crate`), so an implementation crate never has to appear in a game's manifest.
- The editor depends on `engine` **and** `engine-editor-api`.
- `engine` does **not** depend on `engine-editor-api`.

This is what gives the editor "privileged" access to the engine without bloating shipped game binaries. Editor-only capabilities live in a crate the game's dependency graph never touches, so the compiler enforces the boundary.

### Why crates, not cargo features?

Cargo features unify across a workspace build — if any crate in the graph enables a feature, every crate sees it enabled for that build. Putting editor-only APIs behind a feature would mean `cargo build --workspace` silently enables them for shipped games. A dedicated crate cannot leak: if the game doesn't depend on it, the symbols don't exist.

The world list (`engine_core::worlds`, [ADR-0011](docs/ADR-0011-worlds.md) §3) used to be gated this way — re-exported by `engine-editor-api` and not by `engine`. It no longer is: a facade-level export list was never the dependency-graph guarantee above, and reaching another world turned out to be game code's business too (a ghost overlay is a second world). `engine-editor-api` is now genuinely editor-only again.

## Building and running

```sh
make editor   # cargo run -p editor -- --project crates/test-game
make game     # cargo run -p test-game
make build    # cargo build --workspace
make test     # cargo test --workspace
make fmt      # cargo fmt --all
make clippy   # cargo clippy --workspace -- -D warnings
make api      # regenerate docs/api from the source
make hooks    # point git at .githooks (once per clone)
```

The packager runs standalone for CI builds:

```sh
cargo run -p packager -- --project crates/test-game --out target/dist
```

### Driving a running app

Input in, UI state out — no screenshots, no window-manager games:

```sh
ENGINE_DEBUG_INPUT=1 cargo run -p editor &
tools/poke tree                     # every visible text node + its on-screen rect
tools/poke dblclick "cube"          # aim by text, not by pixel
tools/poke rclick "cube"            # secondary button -> context menu
tools/poke wait                     # block until the queue drains
tools/poke type hull; tools/poke key Enter
tools/poke focus                    # what holds the keyboard, and its text
tools/poke shot                     # PNG of the frame -> target/shots/, path printed
tools/poke rec 26 drag a --to b     # film a gesture, one PNG per frame
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

Generated:

- [`docs/api/index.md`](docs/api/index.md) — every public signature in
  `crates/`, one digest per module, with a symbol table that routes to the
  right one. `make api`; the pre-commit hook keeps it current. Read this
  instead of the source when you only need to know what exists.
- [`tools/wrap`](tools/wrap) — rewrap a note's prose to 78 columns:
  `tools/wrap docs/notes/foo.md`. Only paragraphs a line overruns are
  touched, so editing one sentence does not reflow the file.

Decisions:

- [`docs/ADR-INDEX.md`](docs/ADR-INDEX.md) — Architecture Decision Records.
  Start here for the *why* behind structural choices.

Notes, by area:

| Area | Note |
|---|---|
| UI primitives | [ui-core](docs/notes/ui-core.md) |
| UI widgets | [ui-widgets](docs/notes/ui-widgets.md), [ui-widget-authoring](docs/notes/ui-widget-authoring.md) |
| Editor | [editor](docs/notes/editor.md), [editor-document-split](docs/notes/editor-document-split.md) |
| Rendering | [gpu-driven-rendering](docs/notes/gpu-driven-rendering.md), [shaders](docs/notes/shaders.md), [render-camera](docs/notes/render-camera.md), [gizmo](docs/notes/gizmo.md) |
| GPU data path | [transform-gpu](docs/notes/transform-gpu.md), [frame-loop](docs/notes/frame-loop.md), [staging-balancer](docs/notes/staging-balancer.md), [texture-update](docs/notes/texture-update.md) |
| Core | [reflection](docs/notes/reflection.md), [thread-pool](docs/notes/thread-pool.md) |
| Performance | [benchmarks](docs/notes/benchmarks.md), [scatter-overlap-bench](docs/notes/scatter-overlap-bench.md) |

## Status

The renderer draws lit, metallic-roughness-shaded meshes whose transforms live
in a `TransformHierarchy` owned by a `World`, swept once per frame in parallel
across the engine's thread pool. Orbit / pan / zoom are wired through the
built-in `OrbitController`.

The editor opens the test-game project and shows it in a viewport: a
`DockSpace` of documents, each its own `DockSpace` of Hierarchy / Scene /
Inspector, with a menu bar (*File > new scene* opens an empty document beside
the one in front), context menus, drag-to-reparent, a TRS gizmo and a
reflection-driven inspector. See [editor](docs/notes/editor.md).

The packager prints its intended steps without performing them.

Measured frame times and how to reproduce them are in
[benchmarks](docs/notes/benchmarks.md): ~800 FPS at 100k entities, ~250 FPS at
1M.
