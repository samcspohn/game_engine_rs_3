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
use engine::{
    component::Scene,
    glam::Quat,
    transform::{Transform, _Transform},
    Component, MeshRenderer, Window,
};

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

// ─── Editor-side stand-in for a project component ───────────────────────────
//
// Until project scenes are deserialised, the editor just attaches a built-in
// `Spinner` to every loaded entity so the viewport is visibly animated.

#[derive(Clone)]
struct Spinner {
    speed: f32,
}

impl Component for Spinner {
    fn update(&mut self, dt: f32, transform: &Transform) {
        transform.lock().rotate_by(Quat::from_rotation_y(self.speed * dt));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    // Confirm the editor-only API is reachable.
    engine_editor_api::editor_only_hello();

    println!("Opening project: {}", args.project);

    let root = load_project_scene(&args.project);

    let title = format!("Editor — {}", args.project);
    Window::new(&title).with_scene(root).run();
}

// ─────────────────────────────────────────────────────────────────────────────
// Project scene loading (stub)
// ─────────────────────────────────────────────────────────────────────────────

/// Load the renderable scene for a project.
///
/// For now every project returns the same default scene: a single entity with
/// a `MeshRenderer` (placeholder mesh) plus a `Spinner` that animates it.
/// Future implementation: parse a scene file from `<project>/scene.json` (or
/// similar) and deserialise entities + components from there.
fn load_project_scene(project: &str) -> Scene {
    let _ = project; // will be used when scene serialisation is added

    let mut root = Scene::new();
    let e = root.new_entity(_Transform::default());
    root.add_component(e, Spinner { speed: std::f32::consts::FRAC_PI_4 });
    root.add_component(e, MeshRenderer::new("cube.mesh"));

    root
}
