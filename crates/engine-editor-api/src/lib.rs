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

/// Owning a camera outright, rather than attaching a `CameraComponent` and
/// letting it drive one from an entity's pose.
///
/// This is here and not in `engine` because it is the editor's shape: a
/// camera pointed at a document world from a rig that is not part of it, fed
/// a `view_proj` by whatever the editor decides drives it (ADR-0011 §4).
pub use engine_render::{CameraHandle, MAX_CAMERAS};

/// The camera a running scene minted for itself, which is what the editor
/// shows instead of its own while that scene plays. `None` is a game with no
/// camera, and a game with no camera draws nothing.
pub use engine_render::camera_of_world;

pub mod scripts;

/// The TRS gizmo. Editor-only for the same reason as [`CameraHandle`]: it
/// is driven by whoever owns the selection, which no game has.
pub use engine_render::{gizmo, GizmoMode};

/// Print a greeting that confirms the editor-only API is reachable.
///
/// In a real engine this function would be replaced by real editor
/// bootstrapping logic (e.g. starting an asset-pipeline daemon, opening an
/// IPC channel to the editor process, etc.).
pub fn editor_only_hello() {
    println!("Hello from the editor-only engine API");
}
