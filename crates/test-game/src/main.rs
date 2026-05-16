//! Example game binary that uses the engine's public API.
//!
//! Demonstrates the ECS paradigm:
//!
//! ```ignore
//! let e = root.new_entity(t);
//! root.add_component(e, Rotator::new());
//! ```
//!
//! A `Rotator` component spins each entity each frame via `Component::update`.
//! The window owns the `root` scene; the renderer drives `Scene::update` once
//! per frame, which fans out to every registered component in parallel.
//!
//! ## Stress benchmark
//!
//! Pass `--cubes N` to spawn an `N`-cube grid and stress the transform /
//! scatter / mvp_build pipelines. The grid is auto-sized to be roughly
//! cubic, centred at the origin, with a fixed spacing of `1.5` units.
//!
//! ```sh
//! cargo run -p test-game --release -- --cubes 10000
//! ```
//!
//! `--cubes 1` (the default) reproduces the original single-cube scene.
//!
//! This crate intentionally depends only on `engine` —
//! `engine-editor-api` is unreachable by design.

use clap::Parser;
use engine::{
    component::Scene,
    glam::{Quat, Vec3},
    mesh::primitives,
    transform::{Transform, _Transform},
    Component, RenderInstance, Window,
};

// ─── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "Test-game / renderer stress benchmark")]
struct Args {
    /// Number of cubes to spawn in a grid. `1` keeps the legacy single-cube
    /// scene; larger values lay out a roughly cubic grid centred at origin.
    #[arg(long, default_value_t = 1)]
    cubes: usize,

    /// Skip the per-frame `Rotator` component update, so transforms stay
    /// static after creation. Useful for isolating CPU `Scene::update` cost
    /// from staging-write / GPU cost during benchmarking.
    #[arg(long, default_value_t = false)]
    static_scene: bool,
}

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

// ─── Scene construction ─────────────────────────────────────────────────────

/// Build a scene of `n` cubes laid out in a roughly cubic grid centred at
/// the origin. Each cube gets a `Rotator` unless `static_scene` is true.
/// Returns the scene plus a `Vec<RenderInstance>` pointing every instance at
/// mesh 0.
///
/// Layout: `side = ceil(n^(1/3))`, spacing = 1.5 world units. For `n = 1`
/// the cube ends up at the origin (unchanged from the old default scene).
fn build_grid_scene(n: usize, static_scene: bool) -> (Scene, Vec<RenderInstance>) {
    assert!(n >= 1, "cube count must be ≥ 1");

    let mut root = Scene::new();
    let mut instances = Vec::with_capacity(n);

    // Cube root of n, rounded up — produces the smallest grid edge that
    // still fits all `n` entities.
    let side = ((n as f64).cbrt().ceil() as usize).max(1);
    let spacing = 10_f32;
    // Centre the grid on origin so the orbit camera frames it sensibly.
    let offset = -((side as f32) - 1.0) * 0.5 * spacing;

    'outer: for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                if instances.len() >= n {
                    break 'outer;
                }
                let pos = Vec3::new(
                    offset + (x as f32) * spacing,
                    offset + (y as f32) * spacing,
                    offset + (z as f32) * spacing,
                );
                let t = _Transform {
                    position: pos,
                    rotation: Quat::IDENTITY,
                    scale:    Vec3::ONE,
                    name:     String::new(),
                    parent:   None,
                };
                let e = root.new_entity(t);
                if !static_scene {
                    root.add_component(e, Rotator::new());
                }
                instances.push(RenderInstance::new(0, e.id));
            }
        }
    }

    (root, instances)
}

// ─── Entry point ────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    println!(
        "Test-game: spawning {} cube(s){}",
        args.cubes,
        if args.static_scene { " (static, no Rotator)" } else { "" },
    );

    let (root, instances) = build_grid_scene(args.cubes, args.static_scene);

    Window::new("Test Game")
        .with_meshes(vec![primitives::cube()])
        .with_scene(root, instances)
        .run();
}
