//! The engine editor binary.
//!
//! This binary has access to both the public game-facing API (`engine`) and
//! the editor-only extensions (`engine_editor_api`).  It calls
//! [`engine_editor_api::editor_only_hello`] to confirm the editor path works,
//! then opens a window titled "Editor" using the shared renderer.

fn main() {
    // Confirm the editor-only API is reachable.
    engine_editor_api::editor_only_hello();

    // Open the editor window and enter the render loop.
    engine::Window::new("Editor").run();
}
