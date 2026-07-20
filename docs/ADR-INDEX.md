# Architecture Decision Records

This directory captures significant, hard-to-reverse decisions about the
engine's architecture. Each ADR records the **context**, **decision**, and
**consequences** at the time it was made, so future contributors (and
future-us) can understand *why* the code looks the way it does without having
to re-derive the reasoning.

## When to write an ADR

Write one when a change:

- Locks in a structural pattern that will be expensive to undo.
- Trades off safety, performance, or ergonomics in a non-obvious way.
- Bypasses or replaces functionality from a major dependency.
- Establishes a convention that other code is expected to follow.

Routine refactors, bug fixes, and feature additions do **not** need an ADR.

## Format

- Filename: `ADR-NNNN-short-kebab-title.md` (zero-padded, monotonically
  increasing).
- Status: `Proposed` → `Accepted` → (optionally) `Superseded by ADR-NNNN` or
  `Deprecated`.
- Sections (suggested): Context, Decision, Consequences, Caveats, Revisit If.
- Keep it terse. Link to code paths rather than pasting large excerpts.

When an ADR is superseded, do **not** delete it — mark its status and add a
link to the replacement. The history is the point.

## Index

| #    | Title                                                              | Status   | Scope                                  |
|------|--------------------------------------------------------------------|----------|----------------------------------------|
| 0001 | [Custom (Unchecked) Swapchain Renderer](ADR-0001-custom-swapchain.md) | Accepted | `crates/engine-render/src/swapchain.rs` |
| 0002 | [Per-Frame Command Buffer Recording](ADR-0002-per-frame-cb-recording.md) | Accepted | `crates/engine-render/src/lib.rs`       |
| 0003 | [Shared Staging Buffers with GPU-Write Early-Wake Sync](ADR-0003-shared-staging-with-compute-sync.md) | Landed (Path C) | `crates/engine-render/{transform_gpu,lib}.rs`, `shaders/signal.comp` |
| 0004 | [Instanced / Indirect Draw for the Scene Pass](ADR-0004-instanced-indirect-draw.md) | Phase 1 landed | `crates/engine-render/{camera,lib}.rs`, `shaders/scene.vert` |

<!-- Add new rows above this line. Keep them in numeric order. -->
