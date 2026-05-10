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

/// Print a greeting that confirms the editor-only API is reachable.
///
/// In a real engine this function would be replaced by real editor
/// bootstrapping logic (e.g. starting an asset-pipeline daemon, opening an
/// IPC channel to the editor process, etc.).
pub fn editor_only_hello() {
    println!("Hello from the editor-only engine API");
}
