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
    ui::{
        style::{px, AlignItems, Display, FlexDirection, LengthPercentageAuto, Position, Rect, Size,
            Style, TaffyAuto, zero},
        theme, ui, NodeId, Row, RowStyle, TreeView, UiStyle,
    },
    CameraComponent, Component, MeshRenderer, OrbitController, Window,
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

    /// Optional glTF/GLB to instantiate, for exercising the hierarchy panel
    /// against a real deep scene graph.
    #[arg(long)]
    glb: Option<String>,
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

// ─── Editor chrome ──────────────────────────────────────────────────────────

/// The editor's own UI, built straight from `main` against the same public
/// API a game uses (ADR-0008) — no component and no per-frame update, since
/// nothing in it changes.
///
/// That makes it the other half of the demonstration: `test-game`'s overlay
/// shows an event-driven UI that uploads on change, this one shows a static
/// UI that uploads **once** and then costs zero dirty words for the rest of
/// the session. Docking (ADR-0006 phase 4) grows from here, in the editor,
/// rather than from inside the renderer.
fn build_editor_chrome(project: &str) {
    const PAD: f32 = 12.0;
    let t = theme();

    let mut ui = ui();
    let screen = ui.root();

    // Top-right, shrink-wrapped: `left`/`bottom` auto, so the panel sits
    // against the opposite corner from a game overlay.
    let panel = ui.node(
        screen,
        Style {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            position: Position::Absolute,
            inset: Rect {
                left: LengthPercentageAuto::AUTO,
                top: px(PAD),
                right: px(PAD),
                bottom: LengthPercentageAuto::AUTO,
            },
            padding: Rect::length(PAD),
            gap: Size {
                width: zero(),
                height: px(4.0),
            },
            align_items: Some(AlignItems::STRETCH),
            ..Default::default()
        },
    );
    ui.set_background(
        panel,
        UiStyle::fill(t.panel).border(t.outline, 1.0).radius(8.0),
    );
    ui.label(panel, 13.0, t.text, "editor");
    ui.label(panel, t.text_px, t.text_dim, project);
}

// ─── Scene hierarchy panel ──────────────────────────────────────────────────

/// The scene hierarchy, as a collapsible tree over the live
/// `TransformHierarchy` (ADR-0008 / ADR-0009).
///
/// It mirrors nothing. `TreeView` reads structure through the two closures in
/// `sync`, so what is on screen is the hierarchy itself — there is no second
/// copy to drift. Names are pulled per *visible* row, so renaming an entity
/// is not a structural event at all.
///
/// Attached as an ordinary component purely for the access path: `update` is
/// handed a `Transform`, and `Transform::hierarchy()` is how a component
/// reaches the scene graph.
#[derive(Clone)]
struct HierarchyPanel {
    view: TreeView,
    selected: Option<u64>,
    count: NodeId,
}

impl HierarchyPanel {
    fn new() -> Self {
        let t = theme();
        let mut ui = ui();
        let screen = ui.root();
        let panel = ui.node(
            screen,
            Style {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                position: Position::Absolute,
                inset: Rect {
                    left: px(12.0),
                    top: px(12.0),
                    right: LengthPercentageAuto::AUTO,
                    bottom: LengthPercentageAuto::AUTO,
                },
                padding: Rect { left: px(t.pad), right: px(t.pad), top: px(t.pad), bottom: px(t.pad) },
                gap: Size { width: zero(), height: px(4.0) },
                align_items: Some(AlignItems::STRETCH),
                ..Default::default()
            },
        );
        ui.set_background(
            panel,
            UiStyle::fill(t.panel).border(t.outline, 1.0).radius(8.0),
        );
        ui.label(panel, 13.0, t.text, "hierarchy");
        let count = ui.label(panel, 10.0, t.text_dim, "");

        let view = TreeView::new(
            &mut ui,
            panel,
            Style {
                size: Size { width: px(260.0), height: px(420.0) },
                ..Default::default()
            },
            RowStyle::default(),
            engine::transform::ROOT as u64,
        );
        ui.set_background(view.node(), UiStyle::fill(t.backdrop).radius(t.radius));

        Self { view, selected: None, count }
    }
}

impl Component for HierarchyPanel {
    fn update(&mut self, _dt: f32, transform: &Transform) {
        let h = transform.hierarchy();
        let mut ui = ui();

        // The editor patches the view itself for every edit it makes
        // (expand, collapse, drag). Subscene instantiation is the one
        // structural change it does not drive, so it arrives as an event.
        if !engine::scene_asset::drain_instantiated().is_empty() {
            self.view.invalidate();
        }

        if let Some(id) = self.view.clicked(&ui) {
            self.selected = Some(id);
        }
        let selected = self.selected;

        self.view.sync(
            &mut ui,
            |id, out| out.extend(h.children(id as u32).iter().map(|&c| c as u64)),
            |id| {
                let name = h.name(id as u32);
                Row {
                    // Most glTF nodes are unnamed; the index is what an editor
                    // can actually act on anyway.
                    text: if name.is_empty() {
                        format!("entity {id}").into()
                    } else {
                        name.into()
                    },
                    depth: 0,       // supplied by the view
                    expanded: None, // supplied by the view
                    selected: selected == Some(id),
                }
            },
        );

        let text = format!("{} entities", h.len());
        ui.set_label(self.count, &text);

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
    build_editor_chrome(&args.project);



    if let Some(glb) = &args.glb {
        let scene_id = engine::scene_asset::request_scene(glb);
        // Named, so the hierarchy panel shows the asset rather than an index.
        let name = std::path::Path::new(glb)
            .file_stem()
            .map_or_else(|| glb.clone(), |s| s.to_string_lossy().into_owned());
        engine::scene_asset::spawn_subscene(
            scene_id,
            _Transform { name, .._Transform::default() },
        );
        println!("Requested GLB subscene: {glb}");
    }

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
    let e = root.new_entity(_Transform { name: "cube".into(), .._Transform::default() });
    root.add_component(e, Spinner { speed: std::f32::consts::FRAC_PI_4 });
    root.add_component(e, MeshRenderer::new("crates/test-game/assets/cube/cube.obj"));

    // Viewport camera: the editor's own "controller" component
    // (`OrbitController`, mouse-driven via the global `Input` accumulator)
    // plus a `CameraComponent` on the same entity — the same pattern any
    // game project uses for its own player-driven camera.
    let cam = root.new_entity(_Transform { name: "editor camera".into(), .._Transform::default() });
    root.add_component(cam, OrbitController::new());
    root.add_component(cam, CameraComponent::new());
    // The panel rides on the camera rather than claiming an entity of its
    // own: `Component::update` is handed a `Transform`, and that is the only
    // reason it needs one at all. Editor chrome must not appear in the
    // scene it is displaying.
    root.add_component(cam, HierarchyPanel::new());

    root
}
