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

## What is deliberately not here

**Unloading.** `dlclose` on a Rust dylib is not reliably sound — TLS
destructors, a registered panic hook and `parking_lot`'s statics all outlive
the call — and every live component is a value whose vtable points into the
library. `Scripts` holds the handle for the process lifetime.

**Reload**, therefore, and runtime compilation with it. Reload means moving a
component's state to freshly compiled code, which is the `Export` save walk
(ADR-0010 §3) that is not built. Until it is, a reload could only drop the
scene, which is what restarting the editor already does. The order is: this
boundary, then a scene file, then per-call `catch_unwind` (ADR-0010 §7), then
reload — by which point it is `cargo build`, `dlopen`, and deserialise.

## Registering a type

`ComponentType` is name → `fn(&mut World, Entity)`, keyed on the derive's
`TYPE_NAME` because `TypeId` is not stable across builds and cannot name a
type in a file or a menu (ADR-0010 §2). `ComponentType::of::<T>` requires
`Default`: a type the editor can name is a type it can construct.

`MeshRenderer` is not registered. It holds a `MeshId` refcount and has no "no
mesh yet" value, so giving it a `Default` is a decision about the asset
registry rather than a missing impl.

The engine registers its own types from `Window::new` into the same list the
plugin fills, so there is one list rather than a builtin one and a project
one.
