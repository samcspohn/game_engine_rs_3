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

mod browser;
mod chrome;
mod console;
mod document;
mod hierarchy;
mod inspector;
mod new_project;
mod project;
mod settings;

use clap::Parser;
use engine::{
    Window,
    transform::_Transform,
    ui::style::{Size, Style, px},
};
use project::load_project;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Game engine editor")]
struct Args {
    /// Path to the game project crate to open in the viewport.
    #[arg(long, default_value = "crates/test-game")]
    project: String,

    /// Optional glTF/GLB to instantiate, for exercising the hierarchy panel
    /// against a real deep scene graph.
    #[arg(long)]
    glb: Option<String>,

    /// Spinning entities to spawn instead of the demo documents, split
    /// evenly across `--worlds`. See `docs/notes/scatter-overlap-bench.md`.
    #[arg(long, default_value_t = 0)]
    stress: usize,

    /// Document worlds the `--stress` entities are split across. The same
    /// entities are drawn at any value, so only the scatter changes.
    #[arg(long, default_value_t = 1)]
    worlds: usize,
}

/// Where a document's file lives: one `scenes/` directory per project, and
/// the panel's title is the file's name.
pub(crate) fn scene_path(title: &str) -> PathBuf {
    Path::new("scenes").join(format!("{title}.json"))
}

/// Take the whole of whatever holds this, and shrink with it. A flex item
/// refuses to go below its own content unless told it may, and a docked
/// panel is exactly the case where the parent decides.
pub(crate) fn fill() -> Style {
    Style {
        flex_grow: 1.0,
        flex_basis: px(0.0),
        min_size: Size {
            width: px(0.0),
            height: px(0.0),
        },
        ..Default::default()
    }
}

fn main() {
    let args = Args::parse();

    let glb = args.glb.as_deref().map(engine::project::pin);
    let project = engine::project::enter(&args.project).unwrap_or_else(|e| {
        eprintln!("cannot open project {}: {e}", args.project);
        std::process::exit(1);
    });

    // Confirm the editor-only API is reachable.
    engine_editor_api::editor_only_hello();

    println!("Opening project: {}", args.project);

    // Before `Window::new` registers the engine's own types, so what this
    // prints is exactly what crossed the dylib boundary.
    match engine_editor_api::scripts::load(&project) {
        Ok(Some(path)) => {
            let names: Vec<_> = engine::script::types().iter().map(|t| t.name).collect();
            println!("scripts: {} registered {names:?}", path.display());
        }
        Ok(None) => println!("scripts: none — {} has no scripts crate", args.project),
        Err(e) => eprintln!("scripts: {e}"),
    }

    let (documents, rig) = load_project(&args.project, args.stress, args.worlds);

    if let Some(glb) = &glb {
        let scene_id = engine::scene_asset::request_scene(glb);
        // Named, so the hierarchy panel shows the asset rather than an index.
        let name = glb.file_stem().map_or_else(
            || glb.display().to_string(),
            |s| s.to_string_lossy().into_owned(),
        );
        // `parent: None` is that world's own root. The spawn names the
        // world so the drain cannot land it in another document's.
        engine::scene_asset::spawn_subscene(
            scene_id,
            documents[0].id(),
            _Transform {
                name,
                .._Transform::default()
            },
        );
        println!("Requested GLB subscene: {}", glb.display());
    }

    let title = format!("Editor — {}", args.project);
    // The documents first — the renderer draws them — then the rig, which is
    // handed over too because a world lives only while a handle to it does.
    documents
        .into_iter()
        .fold(Window::new(&title), Window::with_world)
        .with_world(rig)
        .run();
}
