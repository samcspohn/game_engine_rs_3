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

use std::sync::Arc;

use clap::Parser;
use engine::{
    glam::Quat,
    mesh::primitives,
    transform::{TransformHierarchy, _Transform},
    Mesh, RenderInstance, Window,
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

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    // Confirm the editor-only API is reachable.
    engine_editor_api::editor_only_hello();

    println!("Opening project: {}", args.project);

    let (meshes, hierarchy, instances) = load_project_scene(&args.project);
    let hierarchy = Arc::new(hierarchy);

    // Capture the indices we need to animate — for now, spin every instance.
    let spin_indices: Vec<u32> = instances.iter().map(|i| i.transform_index).collect();

    let title = format!("Editor — {}", args.project);
    Window::new(&title)
        .with_meshes(meshes)
        .with_scene(hierarchy.clone(), instances)
        .on_update(move |h, dt| {
            let spin = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4 * dt);
            for &idx in &spin_indices {
                if let Some(t) = h.get_transform(idx) {
                    t.lock().rotate_by(spin);
                }
            }
        })
        .run();
}

// ─────────────────────────────────────────────────────────────────────────────
// Project scene loading (stub)
// ─────────────────────────────────────────────────────────────────────────────

/// Load the renderable scene for a project.
///
/// For now every project returns the same default scene: a single unit cube
/// with one transform-hierarchy entry. Future implementation: parse a scene
/// file from `<project>/scene.json` (or similar) and deserialise the mesh +
/// transform data from there.
fn load_project_scene(project: &str) -> (Vec<Mesh>, TransformHierarchy, Vec<RenderInstance>) {
    let _ = project; // will be used when scene serialisation is added

    let mut hierarchy = TransformHierarchy::new();
    let cube_idx = hierarchy
        .create_transform(_Transform::default())
        .get_idx();

    let meshes    = vec![primitives::cube()];
    let instances = vec![RenderInstance::new(0, cube_idx)];

    (meshes, hierarchy, instances)
}
