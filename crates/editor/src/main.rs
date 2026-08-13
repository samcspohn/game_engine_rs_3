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
    transform::{_Transform, Transform, ROOT},
    ui::{
        style::{percent, px, zero, Display, Size, Style},
        theme, ui, DockSpace, DockStyle, Label, NodeId, RowContent, RowStyle, ScrollbarStyle, Side,
        TextField, TextFieldStyle, TreeDrag, TreeView, UiCore, UiStyle, Viewport,
    },
    CameraComponent, Component, Export, MeshRenderer, OrbitController, Window,
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

#[derive(Clone, Export)]
struct Spinner {
    #[export]
    speed: f32,
}

impl Component for Spinner {
    fn update(&mut self, dt: f32, transform: &Transform) {
        transform
            .lock()
            .rotate_by(Quat::from_rotation_y(self.speed * dt));
    }
}

// ─── Editor chrome ──────────────────────────────────────────────────────────

/// The editor's chrome: one dock filling the window, with the viewport as a
/// panel among the others.
///
/// Built straight from `main` against the same public API a game uses
/// (ADR-0008), and arranged once — after that the *user* owns the layout,
/// which is the whole point of docking (ADR-0006 phase 4). Nothing here is
/// per-frame except [`DockSpace::update`] and the viewport's two numbers.
///
/// The Scene panel holds a [`Viewport`] — the widget that hands its own box
/// back to the renderer, so the camera's target *is* the pane. Drag the
/// divider and the scene is re-rendered at the new size rather than scaled
/// into it.
#[derive(Clone)]
struct Chrome {
    dock: DockSpace,
    view: Viewport,
    hierarchy: HierarchyPanel,
}

impl Chrome {
    /// `document` roots the hierarchy panel, so the editor's own camera —
    /// a sibling of it, not a child — is not something the tree can show.
    fn new(project: &str, document: u64, editor_entities: usize) -> Self {
        let t = theme();
        let mut ui = ui();
        let screen = ui.root();

        let mut dock = DockSpace::new(
            &mut ui,
            screen,
            Style {
                size: Size {
                    width: percent(1.0_f32),
                    height: percent(1.0_f32),
                },
                ..Default::default()
            },
            DockStyle::default(),
        );

        // Each panel is minted into whichever leaf happens to be first and
        // then moved where it belongs — `dock` is exactly what a drop does,
        // so the starting layout is built from the same call the user does.
        let viewport = dock.panel(&mut ui, "Scene");
        let hierarchy = dock.panel(&mut ui, "Hierarchy");
        dock.dock(&mut ui, hierarchy, viewport, Side::Left);
        dock.set_ratio(&mut ui, hierarchy, 0.2);
        let inspector = dock.panel(&mut ui, "Inspector");
        dock.dock(&mut ui, inspector, viewport, Side::Right);
        dock.set_ratio(&mut ui, inspector, 0.25);
        let console = dock.panel(&mut ui, "Console");
        dock.dock(&mut ui, console, viewport, Side::Bottom);
        dock.set_ratio(&mut ui, console, 0.25);
        let browser = dock.panel(&mut ui, "Browser");
        dock.dock(&mut ui, browser, console, Side::Tab);
        dock.select(&mut ui, console);

        let view = Viewport::new(&mut ui, dock.content(viewport), fill());
        placeholder(&mut ui, dock.content(inspector), "nothing selected");
        placeholder(&mut ui, dock.content(browser), "no assets indexed");
        let log = dock.content(console);
        ui.label(log, t.text_px, t.text_dim, "editor");
        ui.label(log, t.text_px, t.text_dim, &format!("opened {project}"));

        let hierarchy =
            HierarchyPanel::new(&mut ui, dock.content(hierarchy), document, editor_entities);
        Self {
            dock,
            view,
            hierarchy,
        }
    }
}

impl Component for Chrome {
    fn update(&mut self, dt: f32, transform: &Transform) {
        let mut ui = ui();
        self.dock.update(&mut ui);
        // Where the scene ended up this frame. The camera follows it, and so
        // does the question of whose pointer a drag is.
        self.view.update(&ui);
        drop(ui);
        self.hierarchy.update(dt, transform);
    }
}

/// What an unbuilt panel says for itself.
fn placeholder(ui: &mut UiCore, pane: NodeId, text: &str) {
    ui.label(pane, theme().text_px, theme().text_dim, text);
}

/// Take the whole of whatever holds this, and shrink with it. A flex item
/// refuses to go below its own content unless told it may, and a docked
/// panel is exactly the case where the parent decides.
fn fill() -> Style {
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

/// Most glTF nodes are unnamed; the index is what an editor can act on
/// anyway.
fn row_text(h: &engine::transform::TransformHierarchy, id: u64) -> String {
    let name = h.name(id as u32);
    match name.is_empty() {
        true => format!("entity {id}"),
        false => name.to_string(),
    }
}

// ─── Scene hierarchy panel ──────────────────────────────────────────────────

/// What the hierarchy puts in flight. The editor's own type, so an inspector
/// can accept *this* and decline a material or a texture without inspecting
/// either — which is the whole point of typed payloads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EntityRef(pub u64);

impl TreeDrag for EntityRef {
    fn node(&self) -> u64 {
        self.0
    }
}

/// A hierarchy row: a name, and the field that renames it in place.
///
/// Both are built once when the pool grows and exactly one is in layout, so
/// renaming is a `Display` swap rather than a `remove_node` and an allocation
/// in the middle of a gesture.
#[derive(Clone, Copy)]
struct NameRow {
    label: Label,
    field: TextField,
}

impl RowContent for NameRow {
    fn build(ui: &mut UiCore, parent: NodeId, s: &RowStyle) -> Self {
        let label = ui.label(parent, s.text_px, s.text, "");
        let t = theme();
        let field = ui.text_field(
            parent,
            "",
            TextFieldStyle {
                // Sized to sit *inside* a row: the row height is what turns a
                // scroll offset into a data index, so a field that made its
                // row taller would desynchronise the whole list.
                width: 170.0,
                text_px: s.text_px,
                padding: 1.0,
                radius: 2.0,
                fill: t.control_held,
                ..TextFieldStyle::default()
            },
        );
        let me = NameRow { label, field };
        me.set_editing(ui, false);
        me
    }

    fn set_selected(&self, ui: &mut UiCore, s: &RowStyle, selected: bool) {
        self.label
            .set_color(ui, if selected { s.text_selected } else { s.text });
    }
}

impl NameRow {
    /// Swap which half of the row is in layout. Read-modify-write, so the
    /// field keeps the size `build` gave it.
    fn set_editing(&self, ui: &mut UiCore, editing: bool) {
        for (node, shown) in [(self.label.node(), !editing), (self.field.node(), editing)] {
            let mut s = ui.node_style(node);
            s.display = match shown {
                true => Display::Flex,
                false => Display::None,
            };
            ui.set_node_style(node, s);
        }
    }
}

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
    view: TreeView<NameRow, EntityRef>,
    selected: Option<u64>,
    /// The entity being renamed, by id rather than by row — the pool
    /// recycles rows, and scrolling must not rename whatever moves in.
    editing: Option<u64>,
    count: Label,
    /// Slots the editor took before the project loaded, subtracted from the
    /// hierarchy's length so the count reports the document and not the rig.
    /// A length is not a count — it never shrinks, and it does not know the
    /// subtree. See `docs/notes/editor-document-split.md`.
    editor_entities: usize,
}

impl HierarchyPanel {
    /// Built into a dock pane, so it takes whatever box the user has dragged
    /// its panel to rather than a size of its own.
    fn new(ui: &mut UiCore, pane: NodeId, document: u64, editor_entities: usize) -> Self {
        let t = theme();
        let count = ui.label(pane, 10.0, t.text_dim, "");

        let style = RowStyle::default();
        // The tree and its gutter, side by side: a scrollbar cannot live
        // inside the area it mirrors, because anything added to a scroll area
        // scrolls with the contents.
        let gutter = ui.node(
            pane,
            Style {
                display: Display::Flex,
                gap: Size {
                    width: px(3.0),
                    height: zero(),
                },
                ..fill()
            },
        );
        let view = TreeView::new(ui, gutter, fill(), style, document);
        ui.set_background(view.node(), UiStyle::fill(t.backdrop).radius(t.radius));
        ui.scrollbar(gutter, view.node(), ScrollbarStyle::default());

        Self {
            view,
            selected: None,
            editing: None,
            count,
            editor_entities,
        }
    }

    /// Begin renaming `id`: seed the field from the model and focus it.
    ///
    /// The field is still `Display::None` here; `sync` shows it later this
    /// frame, before `run_layout` would drop focus from a node with no box.
    fn begin_rename(&mut self, ui: &mut UiCore, name: &str) -> bool {
        let Some(id) = self.editing else { return false };
        let Some(row) = self.view.row(id) else {
            // Not pooled: nothing to focus, so the edit never starts rather
            // than starting invisibly.
            self.editing = None;
            return false;
        };
        row.field.set_text(ui, name);
        row.field.focus(ui);
        true
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

        // The single click fired too and selected the row, which is what
        // should happen.
        if let Some(id) = self.view.double_clicked(&ui) {
            self.editing = Some(id);
            let name = row_text(h, id);
            self.begin_rename(&mut ui, &name);
        } else if let Some(id) = self.editing {
            // Enter commits; anything that took the keyboard away cancels —
            // clicking elsewhere, Escape, or the row scrolling out of view.
            match self.view.row(id) {
                Some(row) if row.field.submitted(&ui) => {
                    let name = row.field.text(&ui).trim().to_string();
                    if !name.is_empty() {
                        let t = h
                            .get_transform(id as u32)
                            .expect("a row being renamed names a live entity")
                            .lock();
                        h.set_name(&t, &name);
                    }
                    self.editing = None;
                }
                Some(row) if !ui.focused(row.field) => self.editing = None,
                None => self.editing = None,
                _ => {}
            }
        }

        // A drag re-parents the scene, not a copy of it, and the view is
        // patched in the same breath — the pair ADR-0008 says to keep behind
        // one function rather than re-introduce change tracking for.
        if let Some(d) = self.view.dropped() {
            let t = h
                .get_transform(d.node as u32)
                .expect("a dropped row names a live entity")
                .lock();
            if t.get_parent() == Some(d.parent as u32) {
                // Same parent: ordering only. Routing this through
                // `set_parent_at` would push a parent change that did not
                // happen down the parent stream every time a user nudged a
                // sibling.
                h.move_child(&t, d.at);
            } else {
                h.set_parent_at(&t, Some(d.parent as u32), d.at);
            }
            drop(t);
            self.view.moved(d.node, d.parent, d.at);
        }
        // The view reports what a press picked up; the editor grabs, because
        // only it knows a row here names an entity. `EntityRef` is what an
        // inspector will accept — the view never constructs one.
        if let Some(id) = self.view.picked_up(&ui) {
            self.view.grab(&mut ui, EntityRef(id), |ui, r| {
                r.label.set_text(ui, &row_text(h, id))
            });
        }

        let (selected, editing) = (self.selected, self.editing);
        self.view.sync(
            &mut ui,
            |id, out| out.extend(h.children(id as u32).iter().map(|&c| c as u64)),
            |ui, r, id| {
                // Every row every frame: the pool recycles, so a row that
                // was being renamed must be told it no longer is.
                r.set_editing(ui, editing == Some(id));
                r.label.set_text(ui, &row_text(h, id));
                r.set_selected(ui, selected == Some(id));
            },
        );

        let text = format!("{} entities", h.len() - self.editor_entities);
        self.count.set_text(&mut ui, &text);
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

    if let Some(glb) = &args.glb {
        let scene_id = engine::scene_asset::request_scene(glb);
        // Named, so the hierarchy panel shows the asset rather than an index.
        let name = std::path::Path::new(glb)
            .file_stem()
            .map_or_else(|| glb.clone(), |s| s.to_string_lossy().into_owned());
        // Pinned, not left to `None`: the instance materialises frames later,
        // and by then the scene root is whichever document has focus.
        engine::scene_asset::spawn_subscene(
            scene_id,
            _Transform {
                name,
                parent: Some(root.transform_hierarchy.scene_root()),
                .._Transform::default()
            },
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
/// The editor's own entities are created first and parented to [`ROOT`]
/// explicitly; the project is then loaded under `document`, which becomes
/// the scene root. From that point `parent: None` — what every spawn,
/// subscene instantiation and drop-to-top-level resolves to — means the
/// document, so nothing the editor is *editing* can reach the rig it is
/// editing *with*. See `docs/notes/editor-document-split.md`.
///
/// For now every project returns the same default scene: a single entity with
/// a `MeshRenderer` (placeholder mesh) plus a `Spinner` that animates it.
/// Future implementation: parse a scene file from `<project>/scene.json` (or
/// similar) and deserialise entities + components from there.
fn load_project_scene(project: &str) -> Scene {
    let mut root = Scene::new();

    // Viewport camera: the editor's own "controller" component
    // (`OrbitController`, mouse-driven via the global `Input` accumulator)
    // plus a `CameraComponent` on the same entity — the same pattern any
    // game project uses for its own player-driven camera.
    let cam = root.new_entity(_Transform {
        name: "editor camera".into(),
        parent: Some(ROOT),
        .._Transform::default()
    });
    root.add_component(cam, OrbitController::new());
    root.add_component(cam, CameraComponent::new());

    let document = root.new_entity(_Transform {
        name: "document".into(),
        parent: Some(ROOT),
        .._Transform::default()
    });
    root.transform_hierarchy.set_scene_root(document.id);

    // Everything alive at this instant is the editor's own — ROOT, the rig,
    // and the still-empty document — so it is exactly what the hierarchy
    // panel's count has to ignore. Taken rather than hardcoded, so a gizmo
    // added to the rig above needs no second edit here.
    let editor_entities = root.transform_hierarchy.len();

    // The chrome rides on the camera rather than claiming an entity of its
    // own: `Component::update` is handed a `Transform`, and that is the only
    // reason it needs one at all.
    root.add_component(
        cam,
        Chrome::new(project, document.id as u64, editor_entities),
    );

    let e = root.new_entity(_Transform {
        name: "cube".into(),
        .._Transform::default()
    });
    root.add_component(
        e,
        Spinner {
            speed: std::f32::consts::FRAC_PI_4,
        },
    );
    root.add_component(
        e,
        MeshRenderer::new("crates/test-game/assets/cube/cube.obj"),
    );

    // Last, and explicitly: a project that ships its own camera attached one
    // too, and only the mode decides which is live.
    engine::set_active_camera(cam);

    root
}
