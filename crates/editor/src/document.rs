//! One open scene: a dock of its own nested in the document's panel, and the
//! play button that runs the project's startup scene in its place.

use crate::fill;
use crate::hierarchy::HierarchyPanel;
use crate::inspector::InspectorPanel;
use engine::{
    Entity, WorldHandle,
    transform::ROOT,
    ui::{
        Button, ButtonStyle, DockSpace, DockStyle, Label, PanelId, Side, UiCore, Viewport,
        style::{AlignItems, Display, Style},
        theme,
    },
};
use engine_editor_api::{CameraHandle, camera_of_world, gizmo};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// One open scene, as a dock of its own nested in the document's panel.
///
/// The nesting *is* what keeps a sub-panel in its document: a dock only aims
/// a lifted panel at its own leaves, so nothing enforces the rule.
#[derive(Clone)]
pub(crate) struct Document {
    /// The outer dock panel this document fills — what a new scene is tabbed
    /// against, and what says which document is in front.
    pub(crate) pane: PanelId,
    pub(crate) dock: DockSpace,
    pub(crate) view: Viewport,
    pub(crate) hierarchy: HierarchyPanel,
    pub(crate) inspector: InspectorPanel,
    pub(crate) world: WorldHandle,
    /// The world play runs, while it is running: the *startup scene*, loaded
    /// fresh, not this document. Stop drops it.
    pub(crate) play: Option<WorldHandle>,
    /// The camera the panels edit through, kept so stop can put it back —
    /// play shows the running scene's own camera instead.
    pub(crate) editing: CameraHandle,
    /// The rig entity driving that camera, once the frame boundary that
    /// spawns it has run. Closing this document is what removes it.
    pub(crate) rig: Arc<OnceLock<Entity>>,
    /// Shown in the viewport's place when the scene that is playing has no
    /// camera to look through, which is what a packaged game would show.
    blind: Label,
    /// One of the two is in layout at a time — which is also which of them
    /// the next click can be.
    start: Button,
    halt: Button,
    /// The file this document saves to and reloads from.
    pub(crate) path: PathBuf,
}

impl Document {
    pub(crate) fn new(
        ui: &mut UiCore,
        outer: &DockSpace,
        pane: PanelId,
        world: WorldHandle,
        camera: CameraHandle,
        rig: Arc<OnceLock<Entity>>,
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
        ui.set_visible(halt.node(), false);
        let view = Viewport::new(ui, dock.content(scene), fill(), camera.clone());
        let blind = ui.label(dock.content(scene), t.text_px, t.text_dim, NO_CAMERA);
        ui.set_visible(blind.node(), false);
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
            rig,
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
        ui.set_visible(self.view.node(), !blind);
        ui.set_visible(self.blind.node(), blind);

        // A slot index is the running world's own, so the id the tree was
        // showing names something else on the other side of the switch.
        self.hierarchy.selected = None;
        self.hierarchy.invalidate();
        ui.set_visible(self.start.node(), !playing);
        ui.set_visible(self.halt.node(), playing);
        said
    }

    /// What the panels edit: the running scene while there is one, the
    /// document otherwise. The file is always the document's, so a save
    /// during play cannot bake the simulation into it.
    fn shown(&self) -> &WorldHandle {
        self.play.as_ref().unwrap_or(&self.world)
    }

    /// Answers with a line for the console when there is one to say.
    pub(crate) fn update(&mut self, ui: &mut UiCore) -> Option<String> {
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
