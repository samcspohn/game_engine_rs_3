//! Example game binary that uses the engine's public API.
//!
//! Loads a single unit cube (from the engine's primitive library) into the
//! scene and opens a window that renders it.  This crate intentionally depends
//! only on `engine` — `engine-editor-api` is unreachable by design.

fn main() {
    engine::Window::new("Test Game")
        .with_meshes(vec![engine::mesh::primitives::cube()])
        .run();
}
