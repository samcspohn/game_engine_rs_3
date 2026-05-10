//! Example game binary that uses the engine's public API.
//!
//! This crate intentionally depends only on `engine`.  It does **not** depend
//! on `engine-editor-api`, which means the following line would not compile:
//!
//! ```compile_fail
//! // engine_editor_api::editor_only_hello(); // compile error: not a dependency
//! ```

fn main() {
    // engine_editor_api::editor_only_hello(); // compile error: not a dependency
    engine::Window::new("Test Game").run();
}
