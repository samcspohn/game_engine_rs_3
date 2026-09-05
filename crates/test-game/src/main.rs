//! Example game binary that uses the engine's public API.
//!
//! Demonstrates the ECS paradigm:
//!
//! ```ignore
//! root.spawn(t, |mut e| { e.add_component(Rotator::default()); });
//! ```
//!
//! A `Rotator` component spins each entity each frame via `Component::update`.
//! The window owns the `root` world; the renderer sweeps it once
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

mod dock_demo;
mod ui_demo;

use clap::Parser;
use engine::{
    glam::{Quat, Vec3},
    transform::{_Transform, Transform},
    CameraComponent, Component, Export, MeshRenderer, OrbitController, Window, WorldHandle,
};

use test_game_scripts::Rotator;

use dock_demo::DockDemo;
use ui_demo::UiDemo;

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
    /// static after creation. Useful for isolating CPU sweep cost
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

// ─── Scene construction ─────────────────────────────────────────────────────

/// Mesh asset paths cycled round-robin across spawned entities, so a
/// multi-shape stress run exercises several concurrent async loads and a
/// multi-slot draw plan once they resolve.
const SHAPE_PATHS: [&str; 3] = [
    "assets/cube/cube.obj",
    "assets/sphere/sphere.obj",
    "assets/cylinder/cylinder.obj",
];

/// Build a scene of `n` shapes laid out in a roughly cubic grid centred at
/// the origin. Each entity gets a `MeshRenderer` (placeholder mesh until its
/// loader lands), cycling round-robin through `SHAPE_PATHS`, plus a
/// `Rotator` unless `static_scene` is true.
///
/// Layout: `side = ceil(n^(1/3))`, spacing = 10 world units. For `n = 1`
/// the shape ends up at the origin (unchanged from the old default scene).
fn build_grid_scene(n: usize, static_scene: bool, root: &WorldHandle) {
    // assert!(n >= 1, "shape count must be ≥ 1");
    if n == 0 {
        return;
    }

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
                let path = SHAPE_PATHS[spawned % SHAPE_PATHS.len()];
                root.spawn(t, move |mut e| {
                    if !static_scene {
                        e.add_component(Rotator::default());
                    }
                    e.add_component(MeshRenderer::new(path));
                });
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
/// Load what `project.json` says the game opens with. A project that names
/// no scene, or one that will not load, leaves an empty world — and a world
/// with no camera in it draws nothing, which is the honest thing for it to
/// do.
fn load_startup(root: &WorldHandle) {
    let Some(path) = engine::project::settings().startup_scene.as_ref() else {
        eprintln!("no startup_scene in {}", engine::project::PROJECT);
        return;
    };
    // SAFETY: the window has not started, so nothing is reading this world.
    let world = unsafe { root.get_mut() };
    match engine::scene_file::load_from(path, world, engine::transform::ROOT) {
        Ok(ids) => println!("loaded {} entities from {}", ids.len(), path.display()),
        Err(e) => eprintln!("{}: {e}", path.display()),
    }
}

fn spawn_camera(root: &WorldHandle) {
    root.spawn(_Transform::default(), |mut e| {
        e.add_component(OrbitController::new())
            .add_component(CameraComponent::new());
    });
}

/// Attach the overlay (F4). Its entity carries no transform meaning — the UI
/// tree is not the scene tree (ADR-0008), so the component exists only to
/// give the demo a per-frame `update`.
fn spawn_ui(root: &WorldHandle) {
    root.spawn(_Transform::default(), |mut e| {
        e.add_component(UiDemo::new());
    });
    // After `UiDemo`, which is what calls `set_theme` — the dock's default
    // style resolves the palette when it is constructed, not later, and
    // queued spawns are built in the order they were asked for.
    root.spawn(_Transform::default(), |mut e| {
        e.add_component(DockDemo::new());
    });
}

// ─── Entry point ────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    let glb = args.glb.as_deref().map(engine::project::pin);
    // A bundle enters its own directory instead, so the same binary finds
    // its assets from the workspace and from `target/dist`.
    engine::project::enter(env!("CARGO_MANIFEST_DIR")).expect("enter project directory");
    println!(
        "Test-game: spawning {} shape(s){}",
        args.shapes,
        if args.static_scene {
            " (static, no Rotator)"
        } else {
            ""
        },
    );

    // This project's own components, which a scene file names like any
    // other. The editor gets them through the script dylib; a game binary
    // links the same crate and says so itself.
    test_game_scripts::register();

    let root = engine::new_world();
    // The project's own scene, camera and all, unless a benchmark grid was
    // asked for — which is what makes the editor's play button and this
    // binary show the same thing.
    match args.shapes {
        1 => load_startup(&root),
        n => {
            build_grid_scene(n, args.static_scene, &root);
            spawn_camera(&root);
        }
    }
    spawn_ui(&root);

    if let Some(glb) = &glb {
        // Fire-and-forget: the template parse is deferred until the engine
        // initialises the pool; the instance materialises via the render
        // loop's per-frame drain once the hierarchy is Ready, and its
        // meshes stream in from placeholder as decodes complete.
        let scene_id = engine::scene_asset::request_scene(glb);
        engine::scene_asset::spawn_subscene(scene_id, _Transform::default());
        println!("Requested GLB subscene: {}", glb.display());
    }

    Window::new("Test Game").with_world(root).run();
}
