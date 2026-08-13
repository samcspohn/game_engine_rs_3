//! Editor-only engine APIs.
//!
//! This crate exists so that capabilities needed only by the editor tooling
//! (asset importers, live-reload hooks, scene inspectors, …) are isolated
//! behind a separate dependency boundary.  Game binaries **must not** depend
//! on this crate — doing so would be a compile-time error rather than a
//! silent runtime cost.
//!
//! # Why a dedicated crate?
//! If editor utilities were part of `engine` or `engine-core`, every shipped
//! game binary would carry that code.  By keeping them here, the dependency
//! graph enforces the separation: `test-game` → `engine` (no editor-api),
//! while `editor` → `engine` + `engine-editor-api`.

/// The frame's world list (ADR-0011 §3).
///
/// A game only ever wants the world it is in, so `engine` does not re-export
/// this. Note the gate is the facade's export list and not the dependency
/// graph: a game adding `engine-core` directly can still reach it.
pub use engine_core::worlds::{self, count as world_count, world};

/// Print a greeting that confirms the editor-only API is reachable.
///
/// In a real engine this function would be replaced by real editor
/// bootstrapping logic (e.g. starting an asset-pipeline daemon, opening an
/// IPC channel to the editor process, etc.).
pub fn editor_only_hello() {
    println!("Hello from the editor-only engine API");
}
