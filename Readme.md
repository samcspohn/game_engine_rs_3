# Engine

A Rust game engine using Vulkan (via [vulkano](https://github.com/vulkano-rs/vulkano)) for rendering. Organized as a Cargo workspace with a strict separation between game-facing APIs, editor-only APIs, and tooling.

## Architecture

```
crates/
├── engine-core/          # Core types and traits. Math/concurrency only — no GPU deps.
│   ├── transform/        # Hierarchical transform system (TransformHierarchy, Transform, …)
│   ├── component/        # ECS (Component, ComponentStorage, ComponentRegistry, Entity, Scene)
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

## Commands

Build everything:

```sh
cargo build --workspace
```

Run the editor (opens a window, prints the editor-only hello message):

```sh
cargo run -p editor
```

Run the test game (opens a window, no editor APIs available):

```sh
cargo run -p test-game
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

Early scaffold. The renderer clears the screen each frame and exits cleanly on close. The packager prints its intended steps without performing them.
