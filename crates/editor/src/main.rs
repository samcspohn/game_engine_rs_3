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
    glam::{EulerRot, Quat, Vec3},
    transform::{_Transform, Transform, ROOT},
    ui::{
        style::{auto, percent, px, zero, AlignItems, Display, FlexDirection, Size, Style},
        theme, ui, Button, ButtonStyle, DockSpace, DockStyle, Label, MenuBar, MenuBarStyle,
        MenuStyle, NodeId, PanelId, RowContent, RowStyle, ScrollbarStyle, Scrub, Side, TextField,
        TextFieldStyle, TreeDrag, TreeView, UiCore, UiStyle, Viewport,
    },
    AssetRef, Component, Entity, Export, KeyCode, MeshRenderer, OrbitController, PropertyInfo,
    Value, ValueKind, Window, World, WorldHandle,
};
use engine_editor_api::{camera_of_world, gizmo, CameraHandle, GizmoMode};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
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

    /// Spinning entities to spawn instead of the demo documents, split
    /// evenly across `--worlds`. See `docs/notes/scatter-overlap-bench.md`.
    #[arg(long, default_value_t = 0)]
    stress: usize,

    /// Document worlds the `--stress` entities are split across. The same
    /// entities are drawn at any value, so only the scatter changes.
    #[arg(long, default_value_t = 1)]
    worlds: usize,
}

// ─── Editor chrome ──────────────────────────────────────────────────────────

/// One dock filling the window: a panel per open document, and the panes
/// that belong to the application rather than to any one of them. Arranged
/// once here; after that the layout is the user's (ADR-0006 phase 4).
#[derive(Clone)]
struct Chrome {
    dock: DockSpace,
    bar: MenuBar,
    /// The two panels the View menu brings forward. They share a leaf, so
    /// picking one is `select` and nothing else.
    console: PanelId,
    browser: PanelId,
    documents: Vec<Document>,
    /// Scenes made from the menu, so two of them never share a tab title.
    new_scenes: usize,
}

/// The bar's menus. A pick is an index into this, so what the bar shows and
/// what the match below acts on cannot drift apart.
const MENUS: [(&str, &[&str]); 3] = [
    ("File", &["new scene", "save scene", "reload scene", "quit"]),
    ("View", &["console", "browser"]),
    ("Tools", &["translate", "rotate", "scale"]),
];

impl Chrome {
    /// One entry per open scene: a title, the world it edits, and the camera
    /// that draws it. Nothing of the rig this runs in appears, so the
    /// editor's own camera is not something a tree can show.
    fn new(project: &str, documents: Vec<(String, WorldHandle, CameraHandle)>) -> Self {
        let t = theme();
        let mut ui = ui();
        let screen = ui.root();

        // A column, so the dock takes what the bar leaves rather than the
        // whole window — which would push the bar off the bottom of it.
        let shell = ui.node(
            screen,
            Style {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                size: Size {
                    width: percent(1.0_f32),
                    height: percent(1.0_f32),
                },
                ..Default::default()
            },
        );
        let bar = MenuBar::new(&mut ui, shell, &MENUS, MenuBarStyle::default());
        let mut dock = DockSpace::new(&mut ui, shell, fill(), DockStyle::default());

        // Each panel is minted into whichever leaf happens to be first and
        // then moved where it belongs — `dock` is exactly what a drop does,
        // so the starting layout is built from the same call the user does.
        let first = dock.panel(&mut ui, &documents[0].0);
        let panes: Vec<_> = documents
            .iter()
            .skip(1)
            .map(|(title, ..)| {
                let p = dock.panel(&mut ui, title);
                dock.dock(&mut ui, p, first, Side::Tab);
                p
            })
            .collect();
        let console = dock.panel(&mut ui, "Console");
        dock.dock(&mut ui, console, first, Side::Bottom);
        dock.set_ratio(&mut ui, console, 0.25);
        let browser = dock.panel(&mut ui, "Browser");
        dock.dock(&mut ui, browser, console, Side::Tab);
        dock.select(&mut ui, console);
        // Tabbing a document in opens it; the first one is the one to show.
        dock.select(&mut ui, first);

        placeholder(&mut ui, dock.content(browser), "no assets indexed");
        let log = dock.content(console);
        ui.label(log, t.text_px, t.text_dim, "editor");
        ui.label(log, t.text_px, t.text_dim, &format!("opened {project}"));

        let documents = documents
            .into_iter()
            .zip([first].into_iter().chain(panes))
            .map(|((title, world, camera), pane)| {
                let path = scene_path(&title);
                Document::new(&mut ui, &dock, pane, world, camera, path)
            })
            .collect();
        Self {
            dock,
            bar,
            console,
            browser,
            documents,
            new_scenes: 0,
        }
    }

    /// An empty document, tabbed against the one in front and opened there.
    ///
    /// A world, a camera and the rig entity that drives it — the three things
    /// `load_project` mints per document, which is the whole of what makes a
    /// document rather than a panel.
    fn new_scene(&mut self, ui: &mut UiCore, rig: &World) {
        let world = engine::new_world();
        // SAFETY: the sweep running now took its list of worlds before this
        // one existed, so nothing can be reading it.
        unsafe { world.get_mut() }.set_simulating(false);
        let camera = CameraHandle::new(world.id());
        camera.set_grid(Some(GRID));
        let controlled = camera.clone();
        rig.spawn(
            _Transform {
                name: format!("editor camera {}", self.documents.len()),
                .._Transform::default()
            },
            move |mut e| {
                e.add_component(OrbitController::for_camera(controlled));
            },
        );
        self.new_scenes += 1;
        let title = format!("untitled {}", self.new_scenes);
        let pane = self.dock.panel(ui, &title);
        self.dock.dock(ui, pane, self.front(), Side::Tab);
        self.dock.select(ui, pane);
        let path = scene_path(&title);
        self.documents
            .push(Document::new(ui, &self.dock, pane, world, camera, path));
    }

    /// Write the document in front to its file.
    fn save_front(&mut self, ui: &mut UiCore) {
        let d = self.front_document();
        let (path, world) = (d.path.clone(), d.world.clone());
        let said = match engine::scene_file::save_to(&path, &world, ROOT) {
            Ok(()) => format!("saved {}", path.display()),
            Err(e) => format!("save failed: {e}"),
        };
        self.log(ui, &said);
    }

    /// Replace the document in front with what its file says — the other
    /// half of save, and the only way to see that a save was complete.
    fn reload_front(&mut self, ui: &mut UiCore) {
        let d = self.front_document();
        let path = d.path.clone();
        // SAFETY: a document does not simulate, so no sweep is reading it,
        // and the panels that show it are updated later in this same call.
        let world = unsafe { d.world.get_mut() };
        for child in world.hierarchy().children(ROOT).to_vec() {
            world.remove_entity(Entity::new(child));
        }
        let said = match engine::scene_file::load_from(&path, world, ROOT) {
            Ok(ids) => format!("loaded {} entities from {}", ids.len(), path.display()),
            Err(e) => format!("load failed: {e}"),
        };
        // The tree is walking slots that no longer exist, and the selection
        // named one of them.
        d.hierarchy.invalidate();
        d.hierarchy.selected = None;
        self.log(ui, &said);
    }

    /// The document whose tab is open — `front` is one of theirs by
    /// construction.
    fn front_document(&mut self) -> &mut Document {
        let front = self.front();
        self.documents
            .iter_mut()
            .find(|d| d.pane == front)
            .expect("the panel in front is a document's")
    }

    fn log(&self, ui: &mut UiCore, text: &str) {
        let t = theme();
        ui.label(self.dock.content(self.console), t.text_px, t.text_dim, text);
    }

    /// The document whose tab is open. Its leaf always has one, so the
    /// fallback only covers a list `new` guarantees is not empty.
    fn front(&self) -> PanelId {
        let open = self.documents.iter().find(|d| self.dock.showing(d.pane));
        open.unwrap_or(&self.documents[0]).pane
    }
}

/// The editor's own chrome is not authored content — nothing to inspect.
impl Export for Chrome {}

impl Component for Chrome {
    fn update(&mut self, _dt: f32, _transform: &Transform, world: &World) {
        let mut ui = ui();
        match self.bar.update(&mut ui).map(|(m, i)| MENUS[m].1[i]) {
            Some("new scene") => self.new_scene(&mut ui, world),
            Some("save scene") => self.save_front(&mut ui),
            Some("reload scene") => self.reload_front(&mut ui),
            // The window's close button is `event_loop.exit()` with nothing
            // to unwind either, so this is the same exit by another door.
            Some("quit") => std::process::exit(0),
            Some("console") => self.dock.select(&mut ui, self.console),
            Some("browser") => self.dock.select(&mut ui, self.browser),
            Some("translate") => gizmo::set_mode(GizmoMode::Translate),
            Some("rotate") => gizmo::set_mode(GizmoMode::Rotate),
            Some("scale") => gizmo::set_mode(GizmoMode::Scale),
            _ => {}
        }
        self.dock.update(&mut ui);
        // Subscene instantiation is the one structural change the editor does
        // not drive, so it arrives as an event — and always in the world the
        // renderer draws subscenes into, which is the first one handed to the
        // window.
        let spawned = !engine::scene_asset::drain_instantiated().is_empty();
        let mut said = Vec::new();
        for (i, d) in self.documents.iter_mut().enumerate() {
            if spawned && i == 0 {
                d.hierarchy.invalidate();
            }
            said.extend(d.update(&mut ui));
        }
        said.iter().for_each(|s| self.log(&mut ui, s));
        // W/E/R, unless a text field is holding the keyboard — renaming an
        // entity must not also switch tool.
        if !ui.keyboard_captured() {
            for (key, mode) in [
                (KeyCode::KeyW, GizmoMode::Translate),
                (KeyCode::KeyE, GizmoMode::Rotate),
                (KeyCode::KeyR, GizmoMode::Scale),
            ] {
                if engine::input::key_pressed(key) {
                    gizmo::set_mode(mode);
                }
            }
        }
    }
}

// ─── Document ───────────────────────────────────────────────────────────────

/// One open scene, as a dock of its own nested in the document's panel.
///
/// The nesting *is* what keeps a sub-panel in its document: a dock only aims
/// a lifted panel at its own leaves, so nothing enforces the rule.
#[derive(Clone)]
struct Document {
    /// The outer dock panel this document fills — what a new scene is tabbed
    /// against, and what says which document is in front.
    pane: PanelId,
    dock: DockSpace,
    view: Viewport,
    hierarchy: HierarchyPanel,
    inspector: InspectorPanel,
    world: WorldHandle,
    /// The world play runs, while it is running: the *startup scene*, loaded
    /// fresh, not this document. Stop drops it.
    play: Option<WorldHandle>,
    /// The camera the panels edit through, kept so stop can put it back —
    /// play shows the running scene's own camera instead.
    editing: CameraHandle,
    /// Shown in the viewport's place when the scene that is playing has no
    /// camera to look through, which is what a packaged game would show.
    blind: Label,
    /// One of the two is in layout at a time — which is also which of them
    /// the next click can be.
    start: Button,
    halt: Button,
    /// The file this document saves to and reloads from.
    path: PathBuf,
}

impl Document {
    fn new(
        ui: &mut UiCore,
        outer: &DockSpace,
        pane: PanelId,
        world: WorldHandle,
        camera: CameraHandle,
        path: PathBuf,
    ) -> Self {
        let t = theme();
        let mut dock = DockSpace::new(ui, outer.content(pane), fill(), DockStyle::default());
        let scene = dock.panel(ui, "Scene");
        let tree = dock.panel(ui, "Hierarchy");
        dock.dock(ui, tree, scene, Side::Left);
        dock.set_ratio(ui, tree, 0.25);
        let props = dock.panel(ui, "Inspector");
        dock.dock(ui, props, scene, Side::Right);
        dock.set_ratio(ui, props, 0.3);

        // Above the viewport rather than in the menu bar: a document runs its
        // own copy, and two of them can be running at once.
        let bar = ui.node(
            dock.content(scene),
            Style {
                display: Display::Flex,
                align_items: Some(AlignItems::CENTER),
                ..Default::default()
            },
        );
        let style = ButtonStyle {
            text_px: 10.0,
            padding: 2.0,
            ..ButtonStyle::default()
        };
        let (start, halt) = (ui.button(bar, "play", style), ui.button(bar, "stop", style));
        show(ui, halt.node(), false);
        let view = Viewport::new(ui, dock.content(scene), fill(), camera.clone());
        let blind = ui.label(dock.content(scene), t.text_px, t.text_dim, NO_CAMERA);
        show(ui, blind.node(), false);
        let hierarchy = HierarchyPanel::new(ui, dock.content(tree), world.id().into());
        let inspector = InspectorPanel::new(ui, dock.content(props));
        Self {
            pane,
            dock,
            view,
            hierarchy,
            inspector,
            world,
            play: None,
            editing: camera,
            blind,
            start,
            halt,
            path,
        }
    }

    /// Start or stop the game.
    ///
    /// Play loads the project's **startup scene** into a world of its own and
    /// runs it, so what the viewport shows is what a packaged build would:
    /// the scene the project opens with, through whatever camera that scene
    /// carries. It is not this document — a document is a file being edited,
    /// and the game starts where the project says it starts.
    ///
    /// Stop drops the world, and the renderer retires it. Nothing is
    /// restored because nothing was touched.
    fn set_playing(&mut self, ui: &mut UiCore, play: bool) -> String {
        // A scene that will not load leaves the editor stopped rather than
        // playing nothing, so the button and the console agree.
        let (world, said) = match play.then(start_scene) {
            None => (None, "stopped".to_string()),
            Some(Ok((world, said))) => (Some(world), said),
            Some(Err(e)) => (None, e),
        };
        let playing = world.is_some();
        self.play = world;

        // The running scene's own camera, or none at all: a game with no
        // camera draws nothing, and saying so beats leaving a stale image up.
        let game = self.play.as_ref().and_then(|w| camera_of_world(w.id()));
        let blind = playing && game.is_none();
        self.view
            .set_camera(ui, game.unwrap_or_else(|| self.editing.clone()));
        show(ui, self.view.node(), !blind);
        show(ui, self.blind.node(), blind);

        // A slot index is the running world's own, so the id the tree was
        // showing names something else on the other side of the switch.
        self.hierarchy.selected = None;
        self.hierarchy.invalidate();
        show(ui, self.start.node(), !playing);
        show(ui, self.halt.node(), playing);
        said
    }

    /// What the panels edit: the running scene while there is one, the
    /// document otherwise. The file is always the document's, so a save
    /// during play cannot bake the simulation into it.
    fn shown(&self) -> &WorldHandle {
        self.play.as_ref().unwrap_or(&self.world)
    }

    /// Answers with a line for the console when there is one to say.
    fn update(&mut self, ui: &mut UiCore) -> Option<String> {
        self.dock.update(ui);
        // Where the scene ended up this frame. The camera follows it, and so
        // does the question of whose pointer a drag is.
        self.view.update(ui);
        let said = match (ui.clicked(self.start), ui.clicked(self.halt)) {
            (true, _) => Some(self.set_playing(ui, true)),
            (_, true) => Some(self.set_playing(ui, false)),
            _ => None,
        };
        let world = self.shown().clone();
        self.hierarchy.update(ui, &world);
        // After the hierarchy, so a click selects and inspects in one frame
        // rather than showing the previous selection until the next.
        self.inspector.update(ui, &world, self.hierarchy.selected);
        // The gizmo acts on this document's selection, in this document's
        // camera — another document's is a different world and a different
        // pane, and neither can reach here.
        let selected = self.hierarchy.selected.map(|id| Entity::new(id as u32));
        gizmo::set_target(self.view.camera(), world.id(), selected);
        said
    }
}

/// What the play button shows in place of the viewport.
const NO_CAMERA: &str = "no camera in the startup scene";

/// The world play runs: the project's startup scene, loaded fresh into a
/// world of its own, simulating from the first frame.
///
/// Fresh rather than kept, and the *project's* scene rather than the document
/// in front: this is the game starting, so it starts where a packaged build
/// would ([packaging](../../../docs/notes/packaging.md)).
fn start_scene() -> Result<(WorldHandle, String), String> {
    let path = engine::project::settings()
        .startup_scene
        .as_ref()
        .ok_or_else(|| {
            format!(
                "nothing to play: no startup_scene in {}",
                engine::project::PROJECT
            )
        })?;
    let world = engine::new_world();
    // SAFETY: nothing else holds this handle yet, so no sweep can be reading
    // the world it names.
    let ids = engine::scene_file::load_from(path, unsafe { world.get_mut() }, ROOT)
        .map_err(|e| format!("cannot play {}: {e}", path.display()))?;
    Ok((
        world,
        format!("playing {} ({} entities)", path.display(), ids.len()),
    ))
}

/// Where a document's file lives: one `scenes/` directory per project, and
/// the panel's title is the file's name.
fn scene_path(title: &str) -> PathBuf {
    Path::new("scenes").join(format!("{title}.json"))
}

/// Take a node out of layout, or put it back. Read-modify-write, so what is
/// hidden keeps the size it was built with.
fn show(ui: &mut UiCore, node: NodeId, shown: bool) {
    let mut s = ui.node_style(node);
    s.display = match shown {
        true => Display::Flex,
        false => Display::None,
    };
    ui.set_node_style(node, s);
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

/// A row's parent, for walking up to the root. `None` at the root itself, and
/// for a slot a destroy has freed.
fn parent_of(h: &engine::transform::TransformHierarchy, id: u64) -> Option<u64> {
    h.get_transform(id as u32)?
        .lock()
        .get_parent()
        .map(u64::from)
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

/// What the hierarchy puts in flight — the editor's own type, so an inspector
/// can decline a material without inspecting it. The world travels with the
/// id: the same slot names a different entity in every other one (§2).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EntityRef {
    pub world: u64,
    pub id: u64,
}

impl TreeDrag for EntityRef {
    fn node(&self) -> u64 {
        self.id
    }

    fn tree(&self) -> u64 {
        self.world
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
    /// Swap which half of the row is in layout.
    fn set_editing(&self, ui: &mut UiCore, editing: bool) {
        show(ui, self.label.node(), !editing);
        show(ui, self.field.node(), editing);
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
/// Driven by [`Chrome`] rather than attached as a component: what it shows is
/// the *document's* world, and a component is only ever handed its own.
#[derive(Clone)]
struct HierarchyPanel {
    view: TreeView<NameRow, EntityRef>,
    /// The world it shows, which is also what tags the rows it puts in
    /// flight: a slot index from another document is a different entity.
    world: u64,
    selected: Option<u64>,
    /// The entity being renamed, by id rather than by row — the pool
    /// recycles rows, and scrolling must not rename whatever moves in.
    editing: Option<u64>,
    count: Label,
    add: Button,
    remove: Button,
    /// Where a queued spawn leaves the entity it made, since the builder that
    /// knows the id runs at the frame boundary rather than at the click.
    spawned: Arc<AtomicU64>,
    /// A spawn or destroy is queued. It lands at the next frame boundary, so
    /// the re-walk it needs is a frame after the edit was asked for.
    queued: bool,
}

/// The row menu. The root's is `[..1]` of it, so the two cannot disagree
/// about what "new child" is called.
const ROW_MENU: [&str; 3] = ["new child", "rename", "delete"];

/// `spawned` holding no entity. Ids are `u32`, so the top of the range is free.
const NO_SPAWN: u64 = u64::MAX;

impl HierarchyPanel {
    /// Built into a dock pane, so it takes whatever box the user has dragged
    /// its panel to rather than a size of its own.
    fn new(ui: &mut UiCore, pane: NodeId, world: u64) -> Self {
        let t = theme();
        let bar = ui.node(
            pane,
            Style {
                display: Display::Flex,
                align_items: Some(AlignItems::CENTER),
                gap: Size {
                    width: px(4.0),
                    height: zero(),
                },
                ..Default::default()
            },
        );
        let button = ButtonStyle {
            text_px: 10.0,
            padding: 2.0,
            ..ButtonStyle::default()
        };
        let add = ui.button(bar, "new", button);
        let remove = ui.button(bar, "delete", button);
        // Below the bar rather than beside it: the panel is as narrow as the
        // user drags it, and a clipped count is worse than a second line.
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
        // Rooted at the document world's own `ROOT`: the rig is a separate
        // hierarchy entirely, so there is nothing left to exclude.
        let view = TreeView::new(ui, gutter, fill(), style, ROOT as u64).with_tree(world);
        ui.set_background(view.node(), UiStyle::fill(t.backdrop).radius(t.radius));
        ui.scrollbar(gutter, view.node(), ScrollbarStyle::default());

        Self {
            view,
            world,
            selected: None,
            editing: None,
            count,
            add,
            remove,
            spawned: Arc::new(AtomicU64::new(NO_SPAWN)),
            queued: false,
        }
    }

    /// The tree's structure changed behind its back — re-walk on the next
    /// sync. Every edit the panel makes itself is patched in place instead.
    fn invalidate(&mut self) {
        self.view.invalidate();
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

impl HierarchyPanel {
    /// Queue a spawn under `parent`, or under the root when there is none.
    /// Shared by the toolbar and the row menu, which ask the same thing of
    /// different rows.
    fn spawn_child(&mut self, document: &World, parent: Option<u64>) {
        let spawned = self.spawned.clone();
        document.spawn(
            _Transform {
                name: "entity".into(),
                parent: Some(parent.unwrap_or(ROOT as u64) as u32),
                .._Transform::default()
            },
            move |e| spawned.store(e.id().id as u64, Ordering::Relaxed),
        );
        self.queued = true;
    }

    /// Queue a destroy of the selection, if there is one to destroy. The root
    /// is structure rather than content, so it is never one.
    fn destroy_selected(&mut self, document: &World) {
        let Some(id) = self.selected.filter(|&id| id != ROOT as u64) else {
            return;
        };
        document.destroy(Entity::new(id as u32));
        (self.selected, self.editing, self.queued) = (None, None, true);
    }

    fn update(&mut self, ui: &mut UiCore, document: &World) {
        let h = document.hierarchy();

        // Last frame's edit landed at this frame's boundary — after the tree
        // had already walked a hierarchy that did not have it yet.
        if std::mem::take(&mut self.queued) {
            self.view.invalidate();
        }
        // A spawn's builder ran at that same boundary and left its id here.
        match self.spawned.swap(NO_SPAWN, Ordering::Relaxed) {
            NO_SPAWN => {}
            id => {
                self.selected = Some(id);
                self.view.reveal(id, |n| parent_of(h, n));
            }
        }

        if let Some(id) = self.view.clicked(ui) {
            self.selected = Some(id);
        }

        // The single click fired too and selected the row, which is what
        // should happen.
        if let Some(id) = self.view.double_clicked(ui) {
            self.editing = Some(id);
            let name = row_text(h, id);
            self.begin_rename(ui, &name);
        } else if let Some(id) = self.editing {
            // Enter commits; anything that took the keyboard away cancels —
            // clicking elsewhere, Escape, or the row scrolling out of view.
            match self.view.row(id) {
                Some(row) if row.field.submitted(ui) => {
                    let name = row.field.text(ui).trim().to_string();
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
        if let Some(id) = self.view.picked_up(ui) {
            let payload = EntityRef {
                world: self.world,
                id,
            };
            self.view
                .grab(ui, payload, |ui, r| r.label.set_text(ui, &row_text(h, id)));
        }

        // A child of the selection, so the tree is authored the way it is
        // read; dropping it on the root row is what un-parents it again.
        if ui.clicked(self.add) {
            self.spawn_child(document, self.selected);
        }
        // Delete answers to the panel the pointer is over: every document
        // holds a selection of its own, and one keystroke must not reach all
        // of them.
        let p: [f32; 2] = engine::input::cursor_position().into();
        let r = ui.node_rect(self.view.node());
        let over = (0..2).all(|k| p[k] >= r[k] && p[k] < r[k] + r[k + 2]);
        let pressed =
            over && !ui.keyboard_captured() && engine::input::key_pressed(KeyCode::Delete);
        if ui.clicked(self.remove) || pressed {
            self.destroy_selected(document);
        }

        // Right-click opens the row's menu, selecting it on the way: what a
        // menu acts on should also be what the inspector shows. The root can
        // only be spawned into, so it gets the prefix of the same list.
        if let Some(id) = self.view.right_clicked(ui) {
            self.selected = Some(id);
            let items = match id == ROOT as u64 {
                true => &ROW_MENU[..1],
                false => &ROW_MENU[..],
            };
            let about = EntityRef {
                world: self.world,
                id,
            };
            let at: [f32; 2] = engine::input::cursor_position().into();
            ui.context_menu(at, items, about, MenuStyle::default());
        }
        // Matched by label rather than by index: the two lists would
        // otherwise have to be reordered together, silently.
        let choice = ui
            .menu_choice::<EntityRef>()
            .filter(|(_, r)| r.world == self.world)
            .map(|(i, r)| (ROW_MENU[i], r.id));
        if let Some((item, id)) = choice {
            match item {
                "new child" => self.spawn_child(document, Some(id)),
                "rename" => {
                    self.editing = Some(id);
                    let name = row_text(h, id);
                    self.begin_rename(ui, &name);
                }
                "delete" => {
                    self.selected = Some(id);
                    self.destroy_selected(document);
                }
                _ => {}
            }
        }

        let (selected, editing) = (self.selected, self.editing);
        self.view.sync(
            ui,
            |id, out| out.extend(h.children(id as u32).iter().map(|&c| c as u64)),
            |ui, r, id| {
                // Every row every frame: the pool recycles, so a row that
                // was being renamed must be told it no longer is.
                r.set_editing(ui, editing == Some(id));
                r.label.set_text(ui, &row_text(h, id));
                r.set_selected(ui, selected == Some(id));
            },
        );

        // Minus the hierarchy root, which is structure rather than content.
        let text = format!("{} entities", h.active_len() - 1);
        self.count.set_text(ui, &text);
    }
}

// ─── Inspector panel ────────────────────────────────────────────────────────

/// The entity's own TRS. Not a component: the hierarchy owns it, so the
/// inspector reads and writes it directly rather than through `Export`.
const TRANSFORM: &str = "Transform";

/// Its three properties, in the order they are shown.
const TRS: [(&str, ValueKind); 3] = [
    ("position", ValueKind::Vec3),
    ("rotation", ValueKind::Quat),
    ("scale", ValueKind::Vec3),
];

/// How many fields a kind is edited through. Zero is a read-only line: a
/// text box you cannot type a `MeshId` into would be a worse lie than
/// showing the value.
fn arity(kind: ValueKind) -> usize {
    match kind {
        ValueKind::F32 | ValueKind::I32 | ValueKind::Bool | ValueKind::String => 1,
        ValueKind::Vec3 | ValueKind::Quat => 3,
        ValueKind::Color => 4,
        // Composite too: a nested value is saved and shown, but there is no
        // widget that edits one yet.
        ValueKind::Asset(_) | ValueKind::Entity => 0,
        ValueKind::Struct(_) | ValueKind::List(_) => 0,
    }
}

/// How much a field of this kind moves per pixel of sideways drag. Degrees
/// need a coarser step than metres; a kind with no numbers gets none, and
/// keeps drag-to-select. `I32` is excluded because these are all decimals.
fn step(kind: ValueKind) -> Option<f32> {
    match kind {
        ValueKind::Quat => Some(0.25),
        ValueKind::F32 | ValueKind::Vec3 => Some(0.01),
        ValueKind::Color => Some(0.005),
        _ => None,
    }
}

/// One number of a kind, as the panel writes it. Shared by the read-back and
/// by a drag, so a scrubbed value is not shown to a different precision than
/// the one it will be re-read at.
fn fmt(kind: ValueKind, x: f32) -> String {
    match kind {
        ValueKind::F32 => format!("{x:.4}"),
        ValueKind::Color => format!("{x:.2}"),
        // `-0.000` reads as a value rather than as zero, and a rotation
        // produces one constantly.
        _ => format!("{:.3}", if x == 0.0 { 0.0 } else { x }),
    }
}

/// A [`Value`] as one string per field — or, for a kind with none, the one
/// line its row shows instead.
///
/// A rotation is Euler degrees and not the quaternion's four numbers, which
/// nobody can author. That round trip is not the identity, so a field the
/// user is in keeps what they typed until they leave it — which is what the
/// read-back does with every field anyway.
fn parts(v: &Value) -> Vec<String> {
    let f = |x: f32| fmt(ValueKind::Vec3, x);
    match v {
        Value::F32(x) => vec![fmt(ValueKind::F32, *x)],
        Value::I32(x) => vec![x.to_string()],
        Value::Bool(x) => vec![x.to_string()],
        Value::String(s) => vec![s.clone()],
        Value::Vec3(v) => vec![f(v.x), f(v.y), f(v.z)],
        Value::Quat(q) => {
            let (x, y, z) = q.to_euler(EulerRot::XYZ);
            vec![f(x.to_degrees()), f(y.to_degrees()), f(z.to_degrees())]
        }
        Value::Color(c) => c.iter().map(|x| fmt(ValueKind::Color, *x)).collect(),
        // "none" and not an em-dash: the font atlas is ASCII, and a glyph it
        // does not have renders as `?`, which reads as an error.
        Value::Entity(e) => vec![e.map_or("none".into(), |e| format!("entity {}", e.id))],
        Value::Asset(a) => vec![a.map_or("none".into(), |a| match a {
            AssetRef::Mesh(id) => format!("mesh {}", id.0),
            AssetRef::Material(id) => format!("material {}", id.0),
            AssetRef::Texture(id) => format!("texture {}", id.0),
            AssetRef::Scene(id) => format!("scene {}", id.0),
        })],
        Value::Struct(f) => vec![format!("{} fields", f.len())],
        Value::List(items) => vec![format!("{} items", items.len())],
    }
}

/// A row's fields back into a [`Value`]. `None` on anything unparseable,
/// which is a field the user is still mid-way through and not an error to
/// report.
fn assemble(kind: ValueKind, fields: &[String]) -> Option<Value> {
    let n = |i: usize| -> Option<f32> { fields.get(i)?.trim().parse().ok() };
    let first = || fields.first().map(|s| s.trim());
    match kind {
        ValueKind::F32 => n(0).map(Value::F32),
        ValueKind::I32 => first()?.parse().ok().map(Value::I32),
        ValueKind::Bool => first()?.parse().ok().map(Value::Bool),
        ValueKind::String => Some(Value::String(first()?.to_string())),
        ValueKind::Vec3 => Some(Value::Vec3(Vec3::new(n(0)?, n(1)?, n(2)?))),
        ValueKind::Quat => Some(Value::Quat(Quat::from_euler(
            EulerRot::XYZ,
            n(0)?.to_radians(),
            n(1)?.to_radians(),
            n(2)?.to_radians(),
        ))),
        ValueKind::Color => Some(Value::Color([n(0)?, n(1)?, n(2)?, n(3)?])),
        ValueKind::Asset(_) | ValueKind::Entity => None,
        ValueKind::Struct(_) | ValueKind::List(_) => None,
    }
}

/// The entity's TRS as inspector values. Local and not global: it is what
/// the entity stores, and what a parent moves.
fn trs_values(t: &Transform) -> Vec<(&'static str, &'static str, Value)> {
    let g = t.lock();
    vec![
        (TRANSFORM, "position", Value::Vec3(g.get_position())),
        (TRANSFORM, "rotation", Value::Quat(g.get_rotation())),
        (TRANSFORM, "scale", Value::Vec3(g.get_scale())),
    ]
}

/// The write half. A property whose value did not survive `assemble` never
/// gets here, so there is nothing to report — only nothing to do.
fn set_trs(t: &Transform, prop: &str, v: &Value) {
    let g = t.lock();
    match (prop, v) {
        ("position", Value::Vec3(p)) => g.set_position(*p),
        ("rotation", Value::Quat(q)) => g.set_rotation(*q),
        ("scale", Value::Vec3(s)) => g.set_scale(*s),
        _ => {}
    }
}

/// One property's row.
///
/// Addressed by `(ty, prop)` and not by position: `ComponentRegistry::inspect`
/// walks a `HashMap`, and the order it hands components back in is not
/// something to bind a row to.
#[derive(Clone)]
struct PropRow {
    ty: &'static str,
    prop: &'static str,
    kind: ValueKind,
    /// One per field of the kind — three for a vector, none for a value the
    /// row shows rather than edits.
    fields: Vec<TextField>,
    /// The drag anchor beside each field. The engine reports the gesture;
    /// what a pixel is worth, and how the number is written, are the panel's.
    scrubs: Vec<Scrub>,
    text: Option<Label>,
}

/// The entity's transform, then every `#[export]`ed property of every
/// component on it (ADR-0010 §3), read through `&dyn Export` — the panel
/// names no component type, so a game's own components appear here with no
/// editor change.
///
/// Not a `Component`: [`Chrome`] owns it and calls it after the hierarchy,
/// which is also where the selection it needs lives.
#[derive(Clone)]
struct InspectorPanel {
    pane: NodeId,
    title: Label,
    /// What this panel put in the tree, so a rebuild can take it back out.
    owned: Vec<NodeId>,
    rows: Vec<PropRow>,
    /// Rebuilt on a selection change and not per frame — the set of
    /// components on an entity does not move while you look at it.
    shown: Option<u64>,
    /// Present only with a selection, so it is rebuilt with the rows.
    add: Option<Button>,
    /// What the open menu offers, in the order it offers it — the index the
    /// choice comes back as means nothing without the list that made it.
    offered: Vec<&'static str>,
    /// The component types on the shown entity, so the menu offers only what
    /// it does not already have.
    present: Vec<&'static str>,
    /// An add is queued and lands at the next frame boundary. Until the
    /// components differ from `present`, there is nothing new to draw.
    awaiting: bool,
}

/// The Add Component menu's payload. A type of its own so `menu_choice` tells
/// it from the hierarchy row menu's [`EntityRef`], which is the same two
/// numbers and would otherwise be read against the wrong list of items.
#[derive(Clone, Copy)]
struct AddTo {
    world: engine::transform::WorldId,
    id: u64,
}

impl InspectorPanel {
    fn new(ui: &mut UiCore, pane: NodeId) -> Self {
        let title = ui.label(pane, theme().text_px, theme().text_dim, "nothing selected");
        Self {
            pane,
            title,
            owned: Vec::new(),
            rows: Vec::new(),
            shown: None,
            add: None,
            offered: Vec::new(),
            present: Vec::new(),
            awaiting: false,
        }
    }

    fn rebuild(&mut self, ui: &mut UiCore, world: &World, id: Option<u64>) {
        for n in self.owned.drain(..) {
            ui.remove_node(n);
        }
        self.rows.clear();
        self.add = None;
        self.shown = id;
        let Some(id) = id else { return };

        // Collected before any widget is built: `inspect` holds each
        // component's lock for the callback, and building UI under it would
        // hold a component lock across the whole UI store's.
        let mut specs: Vec<(&'static str, &'static [PropertyInfo])> = Vec::new();
        world
            .entity(Entity::new(id as u32))
            .inspect(|e| specs.push((e.type_name(), e.properties())));

        // First, and not out of `specs`: every entity has one. The root is
        // the identity the hierarchy composes from rather than a pose, so
        // editing its transform would move nothing — it gets no section.
        if id != ROOT as u64 {
            self.section(ui, TRANSFORM, TRS.iter().copied());
        }
        for (ty, props) in &specs {
            self.section(ui, ty, props.iter().map(|p| (p.name, p.kind)));
        }

        // Last, so it sits under the sections taffy stacks above it. The root
        // gets one too: it is a pose the hierarchy composes from, not an
        // entity that cannot carry behaviour.
        let button = ui.button(
            self.pane,
            "Add Component",
            ButtonStyle {
                text_px: theme().text_px,
                padding: 3.0,
                ..ButtonStyle::default()
            },
        );
        self.owned.push(button.node());
        self.add = Some(button);
        self.present = specs.iter().map(|(ty, _)| *ty).collect();
    }

    /// One titled block of rows.
    fn section(
        &mut self,
        ui: &mut UiCore,
        ty: &'static str,
        props: impl Iterator<Item = (&'static str, ValueKind)>,
    ) {
        let t = theme();
        self.owned
            .push(ui.label(self.pane, t.text_px, t.accent, ty).node());
        for (prop, kind) in props {
            let row = ui.node(
                self.pane,
                Style {
                    display: Display::Flex,
                    size: Size {
                        width: percent(1.0_f32),
                        height: auto(),
                    },
                    gap: Size {
                        width: px(4.0),
                        height: zero(),
                    },
                    ..Default::default()
                },
            );
            ui.label(row, t.text_px, t.text_dim, prop);
            let fields: Vec<TextField> = (0..arity(kind))
                .map(|_| {
                    ui.text_field(
                        row,
                        "",
                        TextFieldStyle {
                            width: 90.0,
                            text_px: t.text_px,
                            padding: 1.0,
                            radius: 2.0,
                            fill: t.control_held,
                            ..TextFieldStyle::default()
                        },
                    )
                })
                .collect();
            // Share the row rather than each taking 90 px: three coupled
            // fields have to fit the panel one wide one did, and the panel
            // is whatever width the user has dragged it to.
            for f in &fields {
                let mut style = ui.node_style(f.node());
                style.flex_grow = 1.0;
                style.flex_basis = px(0.0);
                style.min_size.width = px(0.0);
                ui.set_node_style(f.node(), style);
            }
            let text = fields
                .is_empty()
                .then(|| ui.label(row, t.text_px, t.text, ""));
            self.owned.push(row);
            self.rows.push(PropRow {
                ty,
                prop,
                kind,
                scrubs: vec![Scrub::default(); fields.len()],
                fields,
                text,
            });
        }
    }

    fn update(&mut self, ui: &mut UiCore, world: &World, selected: Option<u64>) {
        let h = world.hierarchy();
        if selected != self.shown {
            self.rebuild(ui, world, selected);
            let title = match selected {
                Some(id) => row_text(h, id),
                None => "nothing selected".to_string(),
            };
            self.title.set_text(ui, &title);
        }
        let Some(id) = selected else { return };

        // A queued add lands a frame later, so the panel cannot rebuild on
        // the click. It rebuilds when the entity's components stop matching
        // what is drawn, which is true whichever frame the boundary ran on.
        if self.awaiting {
            let mut now: Vec<&'static str> = Vec::new();
            world
                .entity(Entity::new(id as u32))
                .inspect(|e| now.push(e.type_name()));
            if now != self.present {
                self.awaiting = false;
                self.rebuild(ui, world, selected);
            }
        }

        if self.add.is_some_and(|b| ui.clicked(b)) {
            self.offered = engine::script::types()
                .iter()
                .map(|t| t.name)
                .filter(|n| !self.present.contains(n))
                .collect();
            if !self.offered.is_empty() {
                let at: [f32; 2] = engine::input::cursor_position().into();
                let about = AddTo {
                    world: world.id(),
                    id,
                };
                ui.context_menu(at, &self.offered, about, MenuStyle::default());
            }
        }
        // The payload says which entity the menu was opened over, so a second
        // inspector's menu is not read as this one's.
        let picked = ui
            .menu_choice::<AddTo>()
            .filter(|(_, a)| a.world == world.id() && a.id == id)
            .and_then(|(i, _)| self.offered.get(i).copied());
        if let Some(name) = picked {
            if let Some(ty) = engine::script::find(name) {
                world.edit(Entity::new(id as u32), move |mut e| (ty.add)(&mut e));
                self.awaiting = true;
            }
        }

        let t = h.get_transform_unchecked(id as u32);

        // Commit first, read back second: otherwise a submit is overwritten
        // by the value it was replacing, in the same frame. A row commits
        // when any one of its fields does, and reads the rest as they
        // stand — which is the value they were showing. A drag writes its
        // field here and commits every frame it moves: the drag *is* the
        // edit, and waiting for the release would leave the viewport a frame
        // behind the number.
        let mut edits: Vec<(&'static str, &'static str, Value)> = Vec::new();
        for r in self.rows.iter_mut() {
            let mut commit = r.fields.iter().any(|f| f.submitted(ui));
            if let Some(step) = step(r.kind) {
                for (f, s) in r.fields.iter().zip(&mut r.scrubs) {
                    let at = f.text(ui).trim().parse().ok();
                    if let Some(v) = s.update(ui, *f, step, at) {
                        f.set_text(ui, &fmt(r.kind, v));
                        commit = true;
                    }
                }
            }
            if commit {
                let fields: Vec<String> = r.fields.iter().map(|f| f.text(ui).to_string()).collect();
                if let Some(v) = assemble(r.kind, &fields) {
                    edits.push((r.ty, r.prop, v));
                }
            }
        }
        for (_, prop, v) in edits.iter().filter(|e| e.0 == TRANSFORM) {
            set_trs(&t, prop, v);
        }
        if edits.iter().any(|e| e.0 != TRANSFORM) {
            world.entity(Entity::new(id as u32)).inspect(|e| {
                for (ty, prop, v) in &edits {
                    if e.type_name() == *ty {
                        e.set(prop, v.clone(), &t);
                    }
                }
            });
        }

        // Read back every frame rather than echoing what was typed: a setter
        // is free to refuse or to clamp, and this is what shows that. The
        // gizmo is the transform's other writer, so a drag moves these
        // fields too.
        let mut values = trs_values(&t);
        world.entity(Entity::new(id as u32)).inspect(|e| {
            let ty = e.type_name();
            for p in e.properties() {
                if let Some(v) = e.get(p.name) {
                    values.push((ty, p.name, v));
                }
            }
        });
        for r in &self.rows {
            let Some((_, _, v)) = values.iter().find(|(ty, p, _)| *ty == r.ty && *p == r.prop)
            else {
                continue;
            };
            let shown = parts(v);
            match &r.text {
                Some(l) => l.set_text(ui, &shown.join(", ")),
                // Never into a focused field: that deletes what the user is
                // halfway through typing. `Enter` is where that ends — it
                // says the value is finished, so it is written back in the
                // panel's own format, and left selected to type over.
                None => {
                    for (f, s) in r.fields.iter().zip(&shown) {
                        let done = f.submitted(ui);
                        if done || !ui.focused(*f) {
                            f.set_text(ui, s);
                        }
                        if done {
                            f.focus(ui);
                        }
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

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
        // `parent: None` is the document world's own root, and the drain
        // aims at that world — nothing here can reach the editor's rig.
        engine::scene_asset::spawn_subscene(
            scene_id,
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

// ─────────────────────────────────────────────────────────────────────────────
// Project scene loading (stub)
// ─────────────────────────────────────────────────────────────────────────────

/// The worlds the editor runs: one per document, then the editor's own rig.
///
/// Two hierarchies, not one graph with a boundary drawn through it — so
/// `parent: None`, what every spawn, subscene instantiation and
/// drop-to-top-level resolves to, cannot reach the rig from the document at
/// all. See `docs/notes/editor-document-split.md`.
///
/// The document does not simulate: edit mode is a registry nobody sweeps
/// (ADR-0010 §5), not a per-entity test. It still renders — a scene has to be
/// seen to be authored.
///
/// For now every project returns the same default document: a single entity
/// with a `MeshRenderer`. Behaviour is the project's to supply: a document
/// world does not simulate, and Add Component is how one is attached.
/// Future implementation: parse a scene file from `<project>/scene.json` (or
/// similar) and deserialise entities + components from there.
fn load_project(project: &str, stress: usize, worlds: usize) -> (Vec<WorldHandle>, WorldHandle) {
    let documents = match stress {
        0 => demo_documents(),
        n => stress_documents(n, worlds.max(1)),
    };
    let names: Vec<String> = match stress {
        0 => DEMO.iter().map(|(name, _)| (*name).into()).collect(),
        _ => vec!["Stress".into()],
    };

    // One camera per document, owned by the editor rather than minted by a
    // `CameraComponent` — they look at worlds the rig they are driven from is
    // not part of, and they exist before any entity does. Stress mode instead
    // composites every world through one camera, so the drawn entity count is
    // the same at any `--worlds` and only the scatter varies.
    let cameras: Vec<CameraHandle> = match stress {
        0 => documents
            .iter()
            .map(|d| CameraHandle::new(d.id()))
            .collect(),
        _ => {
            let camera = CameraHandle::new(documents[0].id());
            documents[1..]
                .iter()
                .for_each(|d| camera.draw_world(d.id()));
            vec![camera]
        }
    };
    // The hierarchy panel walks every top-level entity every frame, so under
    // stress it gets an empty world rather than a million-row document. Chrome
    // holds the handle, which is what keeps it alive.
    let shown: Vec<WorldHandle> = match stress {
        0 => documents.clone(),
        _ => vec![engine::new_world()],
    };
    cameras.iter().for_each(|c| c.set_grid(Some(GRID)));

    let rig = engine::new_world();
    for (i, camera) in cameras.iter().enumerate() {
        let camera = camera.clone();
        // The last rig entity brings the chrome up, so every panel it builds
        // has a camera to show.
        let open: Vec<_> = names
            .iter()
            .cloned()
            .zip(shown.iter().cloned())
            .zip(cameras.iter().cloned())
            .map(|((name, world), camera)| (name, world, camera))
            .collect();
        let chrome = (i + 1 == cameras.len()).then(|| (project.to_string(), open));
        rig.spawn(
            _Transform {
                name: format!("editor camera {i}"),
                .._Transform::default()
            },
            move |mut e| {
                // `for_camera`, not `new`: it feeds that camera's matrix and
                // answers only to drags inside that camera's panel.
                e.add_component(OrbitController::for_camera(camera));
                if let Some((project, open)) = chrome {
                    e.add_component(Chrome::new(&project, open));
                }
            },
        );
    }

    (documents, rig)
}

/// Cell, major every ten, and the radius it fades out over — the ground plane
/// an empty document needs to read as a place rather than a void.
const GRID: [f32; 4] = [1.0, 10.0, 120.0, 0.0];

/// The demo project's scenes, which are also the titles of their panels.
const DEMO: [(&str, &str); 2] = [
    ("cube", "assets/cube/cube.obj"),
    ("sphere", "assets/sphere/sphere.obj"),
];

/// The default project: one non-simulating document per demo mesh.
fn demo_documents() -> Vec<WorldHandle> {
    DEMO.iter()
        .map(|(name, mesh)| {
            let document = engine::new_world();
            // SAFETY: no frame has started, so nothing is reading this world.
            unsafe { document.get_mut() }.set_simulating(false);
            document.spawn(
                _Transform {
                    name: (*name).into(),
                    .._Transform::default()
                },
                move |mut e| {
                    e.add_component(MeshRenderer::new(mesh));
                },
            );
            document
        })
        .collect()
}

/// One cubic grid of `total` spinning cubes, cut into `worlds` contiguous
/// slabs — one world each. The grid is the same at any `worlds`, so the
/// scatter's *work* is fixed and only its dispatch count changes.
fn stress_documents(total: usize, worlds: usize) -> Vec<WorldHandle> {
    let side = ((total as f64).cbrt().ceil() as usize).max(1);
    let spacing = 3.0f32;
    let origin = -((side as f32) - 1.0) * 0.5 * spacing;
    let per = total.div_ceil(worlds);
    (0..worlds)
        .map(|w| {
            let world = engine::new_world();
            for k in (w * per)..((w + 1) * per).min(total) {
                let (x, y, z) = (k % side, (k / side) % side, k / (side * side));
                world.spawn(
                    _Transform {
                        position: Vec3::new(
                            origin + x as f32 * spacing,
                            origin + y as f32 * spacing,
                            origin + z as f32 * spacing,
                        ),
                        .._Transform::default()
                    },
                    |mut e| {
                        e.add_component(MeshRenderer::new("assets/cube/cube.obj"));
                    },
                );
            }
            world
        })
        .collect()
}
