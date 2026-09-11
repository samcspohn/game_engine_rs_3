# Project scripts as a dylib

A project's own components — `Rotator` in `crates/test-game/scripts` — are
compiled into a dynamic library the editor loads at startup. The editor is not
rebuilt when a project's script changes, and it opens a project that has no
scripts crate at all, because a level built out of engine primitives and saved
for later is a whole project.

## Why the engine crates are dylibs

The engine's state is process-wide statics: `asset::global()`, `texture`,
`material`, `worlds`, the UI store, `ACTIVE_CAMERA`, the thread pool, and
~30 others. A plugin that statically linked its own `engine-core` would get a
second set of every one of them. Nothing would fail loudly — a mesh the plugin
requested would be in a registry the renderer never reads, a world it spawned
into would never be swept.

So `engine-core`, `engine-render`, `engine-editor-api` and `engine` are
`crate-type = ["rlib", "dylib"]`, and the editor and the scripts crate are
both built with `-C prefer-dynamic`, which makes the dynamic linker resolve
both to one `libengine_core.so`. The plugin then calls `script::register`
directly — there is no context struct to thread through, because there is
nothing to thread.

LTO is the one hard conflict: rustc refuses `-C prefer-dynamic` under LTO
outright, because only `staticlib` / `bin` / `cdylib` outputs are supported
there. So the editor has its own cargo profile with `lto = false`. It gives up
the cross-crate inlining `[profile.release]` exists for, which the editor is
the one binary that does not need.

`rlib` is still first in the list and is still what a consumer picks by
default, so nothing else in the workspace changed: `make game` and every
benchmark link statically and keep the cross-crate inlining the release
profile exists for. Dynamic linking is the editor's mode, not the workspace's,
which is why `make editor` carries its own `CARGO_TARGET_DIR` — the flag is
part of cargo's fingerprint and would otherwise rebuild the world on every
switch.

## The guard

`engine_register_scripts` returns the address of the registry the *plugin*
writes to, and the loader compares it with its own. Equal means one copy;
unequal means the plugin was built statically and every registry in the engine
is doubled the same way. That is unrecoverable and undetectable at the call
site that would suffer from it, so the load is refused there and then.

A plugin built against a *different* engine revision needs no check: Rust
mangles the crate's metadata hash into every symbol, so its imports do not
resolve and `dlopen` fails.

## Generating a project

**File ▸ new project** scaffolds one with cargo rather than by writing
manifests: `cargo new` for the binary and for the scripts crate, `cargo add
--path` for the engine and for the scripts dependency. Only what cargo has no
command for is written here — `crate-type = ["rlib", "dylib"]` and the
source.

The scripts crate is then built into the editor's *own* `target/editor`, with
the same `RUSTFLAGS` and profile. That is what makes the build seconds rather
than minutes — every engine unit it links is already compiled there — and it
puts `lib<name>_scripts.so` beside the running editor, which is where `load`
looks for it.

**A generated project therefore has to be a member of the engine's own
workspace**, and one aimed anywhere else is refused rather than built. Cargo
hashes a package's path *relative to its workspace root*, so `engine-core`
built from a second root is a second unit with different symbol
disambiguators — and both write one `libengine_core.so`. The loser is
whichever binary linked the other: the editor stops starting with `undefined
symbol: engine_core::script::TYPES`. Copying the lockfile is not enough,
because the versions were never what differed.

Creating a project ends by launching the editor again on it. `project::enter`
is one `chdir` before any thread exists and a loaded script dylib can never be
unloaded, so the open project is not something a running process can swap.

## What is deliberately not here

**Unloading.** `dlclose` on a Rust dylib is not reliably sound — TLS
destructors, a registered panic hook and `parking_lot`'s statics all outlive
the call — and every live component is a value whose vtable points into the
library, as does the `&'static str` each registered type names itself by. So
`load` hands back a path and keeps the handle in a static: there is nothing
droppable to drop. It was returned by value first, and a caller that let it
fall out of scope got a menu of NUL bytes with the right lengths.

**Reload**, therefore. Compiling at runtime is here — that is what generating
a project does — but loading the result a second time is not: `register`
keeps the name already in the list, so the running code would stay running,
and every live component is a value whose vtable points into the library it
was made from. Moving that state across is the scene round-trip `save` and
`load` already do. The order is: this boundary, then a scene file, then
per-call `catch_unwind` (ADR-0010 §7), then reload.

## Registering a type

`declare_scripts!` emits two ways in, because a project's components arrive
two ways: `engine_register_scripts` for the editor's `dlopen`, and a plain
`register()` for a game binary, which links the same crate as an rlib and
never opens anything. Both are needed for a scene file to mean the same thing
in the editor and in a packaged build — a type the registry cannot name is a
component the loader drops (see [scene-file](scene-file.md)).

`ComponentType` is name → `fn(&mut World, Entity)`, keyed on the derive's
`TYPE_NAME` because `TypeId` is not stable across builds and cannot name a
type in a file or a menu (ADR-0010 §2). `ComponentType::of::<T>` requires
`Default`: a type the editor can name is a type it can construct.

`MeshRenderer`'s `Default` is the one that needed a decision: it holds a
`MeshId` refcount and had no "no mesh yet" value. `AssetRegistry::empty` mints
an id that is never handed to `request_load`, so it stays pointed at
`MeshSlot::PLACEHOLDER` — which is already what an unresolved mesh looks like,
and is deduped and refcounted like any other id. A renderer added from the
menu therefore draws the placeholder until a mesh is dropped on its
`mesh_id`.

The engine registers its own types into the same list the plugin fills, so
there is one list rather than a builtin one and a project one. It says so from
`engine::new_world` as well as from `Window::new`, and idempotently: a scene
file is read into a world, which is earlier than the window, and a type the
registry cannot name yet is one the loader silently drops.
