//! Example game binary that uses the engine's public API.
//!
//! Builds a single transform-hierarchy entry for a unit cube, hands the
//! hierarchy to the renderer, and animates the cube's rotation each frame.
//! Mouse controls (built into the renderer) orbit / pan / zoom the camera.
//!
//! This crate intentionally depends only on `engine` —
//! `engine-editor-api` is unreachable by design.

use std::sync::Arc;

use engine::{
    glam::Quat,
    mesh::primitives,
    transform::{TransformHierarchy, _Transform},
    RenderInstance, Window,
};

fn main() {
    // ── Build the scene graph ────────────────────────────────────────────────
    let mut hierarchy = TransformHierarchy::new();
    let cube_idx = hierarchy
        .create_transform(_Transform::default())
        .get_idx();

    let hierarchy = Arc::new(hierarchy);

    // ── Run the engine ───────────────────────────────────────────────────────
    Window::new("Test Game")
        .with_meshes(vec![primitives::cube()])
        .with_scene(hierarchy.clone(), vec![RenderInstance::new(0, cube_idx)])
        .on_update(move |h, dt| {
            // Spin the cube around its Y axis at ~45°/sec.
            let spin = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4 * dt);
            h.get_transform(cube_idx).unwrap().lock().rotate_by(spin);
        })
        .run();
}
