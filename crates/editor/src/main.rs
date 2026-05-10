//! The engine editor binary.
//!
//! Accepts a `--project <path>` argument (default: `crates/test-game`) that
//! identifies which game project to load in the viewport.  Run via:
//!
//! ```sh
//! cargo run -p editor -- --project crates/test-game
//! # or simply:
//! make editor
//! ```
//!
//! The editor has access to both the public game-facing API (`engine`) and the
//! editor-only extensions (`engine_editor_api`).

use clap::Parser;

// ─────────────────────────────────────────────────────────────────────────────
// CLI arguments
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "Game engine editor")]
struct Args {
    /// Path to the game project crate to open in the viewport.
    #[arg(long, default_value = "crates/test-game")]
    project: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    // Confirm the editor-only API is reachable.
    engine_editor_api::editor_only_hello();

    println!("Opening project: {}", args.project);
    let meshes = load_project_scene(&args.project);

    let title = format!("Editor — {}", args.project);
    engine::Window::new(&title)
        .with_meshes(meshes)
        .run();
}

// ─────────────────────────────────────────────────────────────────────────────
// Project scene loading (stub)
// ─────────────────────────────────────────────────────────────────────────────

/// Load the renderable meshes for a project.
///
/// For now every project returns the same default scene: a single unit cube.
/// Future implementation: parse a scene file from `<project>/scene.json` (or
/// similar) and deserialise the mesh + transform data from there.
fn load_project_scene(project: &str) -> Vec<engine::Mesh> {
    let _ = project; // will be used when scene serialisation is added
    vec![engine::mesh::primitives::cube()]
}
