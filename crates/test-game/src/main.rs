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
//! Pass `--shapes N` to spawn a grid of `N` entities and stress the
//! transform / scatter / mvp_build pipelines. The grid is auto-sized to be
//! roughly cubic, centred at the origin, with a fixed spacing of `10` units.
//! Entities cycle round-robin through cube / sphere / cylinder meshes, which
//! exercises multiple concurrent async mesh loads and a multi-slot
//! `MultiDrawIndexedIndirect` once they resolve.
//!
//! ```sh
//! cargo run -p test-game --release -- --shapes 10000
//! ```
//!
//! `--shapes 1` (the default) reproduces the original single-cube scene.
//!
//! This crate intentionally depends only on `engine` —
//! `engine-editor-api` is unreachable by design.

use clap::Parser;
use engine::{
    component::Scene,
    glam::{Quat, Vec3},
    transform::{_Transform, Transform},
    CameraComponent, Component, MeshRenderer, OrbitController, Window,
};

// ─── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "Test-game / renderer stress benchmark")]
struct Args {
    /// Number of shapes to spawn in a grid. `1` keeps the legacy single-cube
    /// scene; larger values lay out a roughly cubic grid centred at origin,
    /// cycling round-robin through cube / sphere / cylinder meshes.
    #[arg(long, default_value_t = 1)]
    shapes: usize,

    /// Skip the per-frame `Rotator` component update, so transforms stay
    /// static after creation. Useful for isolating CPU `Scene::update` cost
    /// from staging-write / GPU cost during benchmarking.
    #[arg(long, default_value_t = false)]
    static_scene: bool,

    /// Additionally load a `.glb` file as a subscene and spawn one instance
    /// of it at the origin. The hierarchy appears as soon as the file
    /// parses (placeholder meshes), and each primitive streams in as its
    /// background decode completes.
    #[arg(long)]
    glb: Option<String>,
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
        Self {
            speed: std::f32::consts::FRAC_PI_4,
        }
    }
}

impl Component for Rotator {
    fn update(&mut self, dt: f32, transform: &Transform) {
        let spin = Quat::from_rotation_y(self.speed * dt);
        transform.lock().rotate_by(spin);
    }
}

// ─── Scene construction ─────────────────────────────────────────────────────

/// Mesh asset paths cycled round-robin across spawned entities, so a
/// multi-shape stress run exercises several concurrent async loads and a
/// multi-slot draw plan once they resolve.
const SHAPE_PATHS: [&str; 3] = [
    "crates/test-game/assets/cube/cube.obj",
    "crates/test-game/assets/sphere/sphere.obj",
    "crates/test-game/assets/cylinder/cylinder.obj",
];

/// Build a scene of `n` shapes laid out in a roughly cubic grid centred at
/// the origin. Each entity gets a `MeshRenderer` (placeholder mesh until its
/// loader lands), cycling round-robin through `SHAPE_PATHS`, plus a
/// `Rotator` unless `static_scene` is true.
///
/// Layout: `side = ceil(n^(1/3))`, spacing = 10 world units. For `n = 1`
/// the shape ends up at the origin (unchanged from the old default scene).
fn build_grid_scene(n: usize, static_scene: bool, root: &mut Scene) {
    // assert!(n >= 1, "shape count must be ≥ 1");
    if n == 0 {
        return;
    }

    // let mut root = Scene::new();
    let mut spawned = 0usize;

    // Cube root of n, rounded up — produces the smallest grid edge that
    // still fits all `n` entities.
    let side = ((n as f64).cbrt().ceil() as usize).max(1);
    let spacing = 10_f32;
    // Centre the grid on origin so the orbit camera frames it sensibly.
    let offset = -((side as f32) - 1.0) * 0.5 * spacing;

    'outer: for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                if spawned >= n {
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
                    scale: Vec3::ONE,
                    name: String::new(),
                    parent: None,
                };
                let e = root.new_entity(t);
                if !static_scene {
                    root.add_component(e, Rotator::new());
                }
                let path = SHAPE_PATHS[spawned % SHAPE_PATHS.len()];
                root.add_component(e, MeshRenderer::new(path));
                spawned += 1;
            }
        }
    }

    // root
}

/// Spawn the scene's camera entity: an `OrbitController` (mouse-driven
/// movement, reading the global `Input` accumulator) plus a `CameraComponent`
/// (turns that entity's position/rotation into view+proj matrices) on the
/// same entity, framing the origin.
fn spawn_camera(root: &mut Scene) {
    let e = root.new_entity(_Transform::default());
    root.add_component(e, OrbitController::new());
    root.add_component(e, CameraComponent::new());
}

// ─── Entry point ────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    println!(
        "Test-game: spawning {} shape(s){}",
        args.shapes,
        if args.static_scene {
            " (static, no Rotator)"
        } else {
            ""
        },
    );

    let mut root = Scene::new();
    spawn_camera(&mut root);
    build_grid_scene(args.shapes, args.static_scene, &mut root);

    if let Some(glb) = &args.glb {
        // Fire-and-forget: the template parse is deferred until the engine
        // initialises the pool; the instance materialises via the render
        // loop's per-frame drain once the hierarchy is Ready, and its
        // meshes stream in from placeholder as decodes complete.
        let scene_id = engine::scene_asset::request_scene(glb);
        engine::scene_asset::spawn_subscene(scene_id, _Transform::default());
        println!("Requested GLB subscene: {glb}");
    }

    Window::new("Test Game").with_scene(root).run();
}
