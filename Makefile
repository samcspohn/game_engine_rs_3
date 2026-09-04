# ─────────────────────────────────────────────────────────────────────────────
# Game Engine – top-level Makefile
#
# Convenience wrappers around `cargo` commands so common workflows
# have short, memorable names.
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: editor game build test fmt clippy

# The editor loads a project's scripts as a dylib, so it and the scripts crate
# both link the engine dynamically. Its own target dir keeps that flag from
# invalidating the static, fully-LTO'd builds everything else wants.
EDITOR = CARGO_TARGET_DIR=target/editor RUSTFLAGS="-C prefer-dynamic"
EDITOR_PROFILE = --profile editor

## Open the editor with the test-game project loaded in the viewport.
editor:
	$(EDITOR) cargo build $(EDITOR_PROFILE) -p editor -p test-game-scripts
	$(EDITOR) cargo run $(EDITOR_PROFILE) -p editor -- --project crates/test-game

## Run the test game standalone (no editor overlay).
game:
	cargo run -r -p test-game

## Build the entire workspace.
build:
	cargo build --workspace

## Run all workspace tests.
test:
	cargo test --workspace

## Format all Rust source files.
fmt:
	cargo fmt --all

## Lint all crates (treat warnings as errors).
clippy:
	cargo clippy --workspace -- -D warnings

api:
	tools/apidoc

hooks:
	git config core.hooksPath .githooks
