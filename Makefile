# ─────────────────────────────────────────────────────────────────────────────
# Game Engine – top-level Makefile
#
# Convenience wrappers around `cargo` commands so common workflows
# have short, memorable names.
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: editor game build test fmt clippy

## Open the editor with the test-game project loaded in the viewport.
editor:
	cargo run -r -p editor -- --project crates/test-game

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
