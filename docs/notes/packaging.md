# Packaging a game

Two halves of one problem: a path in a scene file has to mean the same thing
on the machine that authored it and on the machine that runs it, and the
directory the player unzips has to contain everything the binary reaches for.

## The project root

Every path the engine stores — a `MeshId`'s origin, a `TextureId`'s, a
`SceneId`'s, and so everything `scene_file` writes — is relative, and every
loader opens it relative to the working directory. That is what the engine
always did, and it is already portable; nothing needs resolving.

What is not portable is *which* directory that is. So `project::enter` sets
it, once, before any thread exists and before any asset is requested:

- a **bundle** — a directory holding `game.json` beside the executable —
  enters its own directory, and overrides everything. It has to: the directory
  it was built from does not exist on the player's machine, and a `.desktop`
  launcher or a shell in some parent directory will not have set the working
  directory for it.
- otherwise `enter`'s argument: the editor's `--project`, or a game crate's
  own `env!("CARGO_MANIFEST_DIR")`.

The `CARGO_MANIFEST_DIR` default is why `cargo run -p test-game` now works
from any directory rather than only from the workspace root.

The alternative was to leave the working directory alone and resolve each path
against a root variable at the point of use. It reads better in isolation and
it is worse here: it puts the burden on every future loader to remember, and
forgetting is a silent read that works in development and fails only inside a
bundle. Moving the process is one call that cannot be half-done.

Its one real cost is that a path typed on the command line is relative to the
shell, not to the project — `project::pin` exists to canonicalise those before
`enter` moves what relative means, and `--glb` is the only such path today.

## The project file

`project.json`, at the project root, is what the project says about itself.
Today that is one setting:

```json
{ "name": "test-game", "startup_scene": "scenes/cube.json" }
```

`startup_scene` is the scene the game opens with — read by the game binary at
startup, by the editor's [play](play-mode.md) button, and by the packager,
which refuses to build a bundle that names no scene to open. One file, read
the same way in all three, so the editor cannot show something the player will
not get.

It ships in the bundle verbatim rather than being restated in `game.json`.
Two copies of a setting are two things that can disagree, and the manifest's
job is to describe the *build*, not the game.

A missing file is an empty one — a directory of scenes with no settings is
still a project, it just has nothing to play — and a file that will not parse
is reported and then treated as absent, because a typo in one setting should
not stop the editor opening the project it is in.

## The bundle

```sh
cargo run -p packager -- --project crates/test-game --out target/dist
```

The packager builds the project crate with `--release --target <triple>`,
reads the artifact path back out of cargo's JSON output rather than guessing a
target directory, and copies the executable plus `assets/`, `scenes/` and
`project.json` into the output. Then it writes `game.json`.

`game.json` is load-bearing twice over. Its *presence* is what makes the
directory self-locating — that is the engine's only bundle test. Its *fields*
— engine revision (with `-dirty` when the tree was not clean, which matters
more than the hash), target triple and binary name — are what a bug report
needs and what nobody remembers to record by hand.

An output directory that already holds a `game.json` is replaced wholesale,
because an asset the new build no longer references would otherwise ship
anyway. Any other non-empty directory is refused.

## What is deliberately not here

**Cooking.** No texture transcode, no vertex-layout pre-bake, no pack file.
Those are size and load-time features; none of them is why a build fails to
run on another machine. Copy first.

**The script dylib.** A shipped game does not have one. `test-game-scripts` is
`["rlib", "dylib"]` and the game binary takes the rlib under full LTO — the
`dlopen` boundary in [scripts](scripts.md) is the editor's mode, not the
workspace's. Shipping the dynamic configuration would make "the plugin must be
built by the identical rustc against the identical engine revision" a
*player-facing* failure.

**A stock player binary.** The editor can open a project that is only scenes,
and packaging one fails loudly on the missing `Cargo.toml`. The fix is a
`player` bin that links the engine plus the project's scripts rlib and boots
`project.json`'s `startup_scene` — which is what `test-game`'s own `main` now
does in four lines, so the remaining work is making it a crate the packager
can build for a project that has none. It still needs a per-project compile,
because scripts are Rust.

**Anything platform-specific.** The binary needs a Vulkan loader and an ICD,
which come from the driver. What it does not yet have is a written feature
floor (dynamic rendering, multi-draw indirect, descriptor indexing, timeline
semaphores — roughly Vulkan 1.3) or a non-panicking failure when device
creation cannot meet it. On Linux the glibc the bundle is built against is its
real floor, which is what the Steam Linux Runtime container exists to pin.
