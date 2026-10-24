//! One open scene: a dock of its own nested in the document's panel, and the
//! play button that runs the project's startup scene in its place.

use crate::console::Console;
use crate::fill;
use crate::hierarchy::HierarchyPanel;
use crate::inspector::InspectorPanel;
use engine::{
    Entity, WorldHandle,
    transform::ROOT,
    ui::{
        ButtonStyle, DockSpace, DockStyle, Label, NodeId, Side, TickId, UiCore, UiRef, Viewport,
        style::{AlignItems, Display, Style},
        theme,
    },
};
use engine_editor_api::{CameraHandle, camera_of_world, gizmo};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// One open scene, as a dock of its own nested in the document's panel. Kept
/// on that panel's content node (`UiCore::own`).
///
/// The nesting *is* what keeps a sub-panel in its document: a dock only aims
/// a lifted panel at its own leaves, so nothing enforces the rule.
pub(crate) struct Document {
    pub(crate) dock: UiRef<DockSpace>,
    view: Viewport,
    pub(crate) hierarchy: UiRef<HierarchyPanel>,
    inspector: UiRef<InspectorPanel>,
    pub(crate) world: WorldHandle,
    /// The world play runs, while it is running: the *startup scene*, loaded
    /// fresh, not this document. Stop drops it.
    pub(crate) play: Option<WorldHandle>,
    /// The readback clock while playing: a running scene moves values no
    /// event names.
    tick: Option<TickId>,
    /// The camera the panels edit through, kept so stop can put it back —
    /// play shows the running scene's own camera instead.
    editing: CameraHandle,
    /// The rig entity driving that camera, once the frame boundary that
    /// spawns it has run. Closing this document is what removes it.
    pub(crate) rig: Arc<OnceLock<Entity>>,
    /// Shown in the viewport's place when the scene that is playing has no
    /// camera to look through, which is what a packaged game would show.
    blind: Label,
    /// One of the two is in layout at a time.
    start: engine::ui::Button,
    halt: engine::ui::Button,
    /// The file this document saves to and reloads from.
    pub(crate) path: PathBuf,
    console: Console,
}

impl Document {
    /// Built into `content`, an outer dock panel's.
    pub(crate) fn new(
        ui: &mut UiCore,
        content: NodeId,
        world: WorldHandle,
        camera: CameraHandle,
        rig: Arc<OnceLock<Entity>>,
        path: PathBuf,
        console: Console,
    ) -> UiRef<Self> {
        let t = theme();
        let mut dock = DockSpace::new(ui, content, fill(), DockStyle::default());
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
        ui.set_visible(halt.node(), false);
        let view = Viewport::new(ui, dock.content(scene), fill(), camera.clone());
        let blind = ui.label(dock.content(scene), t.text_px, t.text_dim, NO_CAMERA);
        ui.set_visible(blind.node(), false);
        let hierarchy = HierarchyPanel::new(ui, dock.content(tree), world.id());
        let inspector = InspectorPanel::new(ui, dock.content(props));
        let dock = dock.listen(ui);
        let me = ui.own(
            content,
            Self {
                dock,
                view,
                hierarchy,
                inspector,
                world,
                play: None,
                tick: None,
                editing: camera,
                rig,
                blind,
                start,
                halt,
                path,
                console,
            },
        );
        ui.on_click(start, move |ui| {
            me.with(ui, |d, ui| d.set_playing(ui, true));
        });
        ui.on_click(halt, move |ui| {
            me.with(ui, |d, ui| d.set_playing(ui, false));
        });
        // After the hierarchy, so a click selects and inspects at once.
        hierarchy.with(ui, |h, ui| {
            h.tree.on_select(ui, move |ui, id| {
                me.with(ui, |d, ui| d.inspect(ui, id));
            })
        });
        me
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
    fn set_playing(&mut self, ui: &mut UiCore, play: bool) {
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
        ui.set_visible(self.view.node(), !blind);
        ui.set_visible(self.blind.node(), blind);

        // A slot index is the running world's own, so the id the tree was
        // showing names something else on the other side of the switch.
        let shown = self.shown().id();
        self.hierarchy.with(ui, |h, ui| h.show(ui, shown));
        self.inspect(ui, None);
        ui.set_visible(self.start.node(), !playing);
        ui.set_visible(self.halt.node(), playing);
        // 10 Hz rather than per frame: a number the user is dragging is
        // written by the drag.
        let inspector = self.inspector;
        match (playing, self.tick.take()) {
            (true, _) => {
                let tick = ui.every(self.view.node(), 10.0, move |ui, _| {
                    inspector.with(ui, |i, ui| i.read_back(ui, None));
                });
                self.tick = Some(tick);
            }
            (false, Some(t)) => ui.stop(t),
            (false, None) => {}
        }
        self.console.line(ui, &said);
    }

    /// Show `id` of the shown world in the inspector, and aim the gizmo at it.
    fn inspect(&mut self, ui: &mut UiCore, id: Option<u64>) {
        let world = self.shown().id();
        self.inspector.with(ui, |i, ui| i.show(ui, world, id));
        self.aim_gizmo(ui);
    }

    /// The gizmo acts on this document's selection, in this document's
    /// camera — so the one in front says where it is.
    pub(crate) fn aim_gizmo(&self, ui: &mut UiCore) {
        let selected = (self.hierarchy.with(ui, |h, ui| h.tree.selected(ui))).flatten();
        if let Some(camera) = self.view.camera(ui) {
            let entity = selected.map(|id| Entity::new(id as u32));
            gizmo::set_target(&camera, self.shown().id(), entity);
        }
    }

    /// Something outside the UI moved the shown entity — the gizmo.
    pub(crate) fn read_back(&self, ui: &mut UiCore) {
        self.inspector.with(ui, |i, ui| i.read_back(ui, None));
    }

    pub(crate) fn delete_key(&self, ui: &mut UiCore) {
        self.hierarchy.with(ui, |h, ui| h.delete_key(ui));
    }

    /// What the panels edit: the running scene while there is one, the
    /// document otherwise. The file is always the document's, so a save
    /// during play cannot bake the simulation into it.
    fn shown(&self) -> &WorldHandle {
        self.play.as_ref().unwrap_or(&self.world)
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
