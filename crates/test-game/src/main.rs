//! Example game binary that uses the engine's public API.
//!
//! Demonstrates the ECS paradigm:
//!
//! ```ignore
//! let e = root.new_entity(t);
//! root.add_component(e, Rotator::new());
//! ```
//!
//! A `Rotator` component spins its entity each frame via `Component::update`.
//! The window owns the `root` scene; the renderer drives `Scene::update` once
//! per frame, which fans out to every registered component in parallel.
//!
//! This crate intentionally depends only on `engine` —
//! `engine-editor-api` is unreachable by design.

use engine::{
    component::Scene,
    glam::Quat,
    mesh::primitives,
    transform::{Transform, _Transform},
    Component, RenderInstance, Window,
};

// ─── Game-side component ────────────────────────────────────────────────────

/// Spins the entity around its local Y axis at `speed` radians per second.
#[derive(Clone)]
struct Rotator {
    speed: f32,
}

impl Rotator {
    fn new() -> Self {
        // ~45°/sec — matches the previous hard-coded test-game animation.
        Self { speed: std::f32::consts::FRAC_PI_4 }
    }
}

impl Component for Rotator {
    fn update(&mut self, dt: f32, transform: &Transform) {
        let spin = Quat::from_rotation_y(self.speed * dt);
        transform.lock().rotate_by(spin);
    }
}

// ─── Entry point ────────────────────────────────────────────────────────────

fn main() {
    // Build the root scene, spawn a single cube entity, attach the Rotator.
    let mut root = Scene::new();
    let e = root.new_entity(_Transform::default());
    root.add_component(e, Rotator::new());

    // Hand the scene to the window. The renderer will drive
    // `root.update(dt)` once per frame.
    Window::new("Test Game")
        .with_meshes(vec![primitives::cube()])
        .with_scene(root, vec![RenderInstance::new(0, e.id)])
        .run();
}
