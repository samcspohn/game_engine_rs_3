//! The editor shell: the menu bar, the dock the documents open in, and
//! `Chrome::refresh`, which every handler ends in. See
//! [editor](../../../docs/notes/editor.md).

use crate::browser::FilePanel;
use crate::console::Console;
use crate::document::Document;
use crate::hierarchy::NO_SPAWN;
use crate::new_project::NewProject;
use crate::project::{GRID, Open, document_world, stem};
use crate::{fill, new_project, scene_path, settings};
use engine::inherit::SceneInstance;
use engine::{
    AssetKind, Component, Entity, Export, KeyCode, OrbitController, World, WorldHandle,
    transform::{_Transform, ROOT, Transform},
    ui::Events,
    ui::{
        DockSpace, DockStyle, MenuBar, MenuBarStyle, NodeId, PanelId, Side, This, TickId, UiCore,
        style::{Display, FlexDirection, Size, Style, percent},
        ui,
    },
};
use engine_editor_api::{CameraHandle, GizmoMode, gizmo};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, atomic::Ordering};

/// One dock filling the window: a panel per open document, and the panes
/// that belong to the application rather than to any one of them. Arranged
/// once here; after that the layout is the user's (ADR-0006 phase 4).
#[derive(Clone)]
pub(crate) struct Chrome {
    dock: DockSpace,
    bar: MenuBar,
    /// The two panels the View menu brings forward. They share a leaf, so
    /// picking one is `select` and nothing else.
    console: Console,
    browser: PanelId,
    /// The project's files, which is what the `browser` panel shows.
    files: FilePanel,
    documents: Vec<Document>,
    /// Scenes made from the menu, so two of them never share a tab title.
    new_scenes: usize,
    new_project: NewProject,
    /// What the scaffold thread leaves behind. Cargo takes seconds, and the
    /// frame that asked for it is long gone by the time it answers.
    scaffold: Arc<Mutex<Option<Result<PathBuf, String>>>>,
    /// What every handler resolves through, and the world to reach it in.
    /// Minted in `init`, which is the first place a component is told which
    /// entity it is.
    this: Option<This<Chrome>>,
    rig: engine::transform::WorldId,
    /// The readback clock, while a document is playing. See
    /// [`Chrome::follow_simulation`].
    tick: Option<TickId>,
    /// Last frame [`refresh`](Chrome::refresh) ran on.
    refreshed: Option<u64>,
}

/// What a handler has to hear about for the panels' own polls to still be
/// answering this frame's question.
///
/// `RESIZE` is not in it: `place` would then report every node that moved on
/// a relayout. The three nodes whose box is actually news say so themselves.
const WAKE: Events = Events::CLICK
    .or(Events::DOUBLE_CLICK)
    .or(Events::RIGHT_CLICK)
    .or(Events::DRAG)
    .or(Events::DROP)
    // The menu bar switches menus on hover alone, and a hover transition is
    // a crossing rather than a frame — a handful while the pointer moves,
    // against sixty a second for standing still.
    .or(Events::HOVER)
    .or(Events::SCROLL)
    .or(Events::FOCUS)
    .or(Events::CHANGE);

/// The bar's menus. A pick is an index into this, so what the bar shows and
/// what the match below acts on cannot drift apart.
const MENUS: [(&str, &[&str]); 3] = [
    (
        "File",
        &[
            "new project",
            "new scene",
            "save scene",
            "reload scene",
            "quit",
        ],
    ),
    ("View", &["console", "browser"]),
    ("Tools", &["translate", "rotate", "scale"]),
];

impl Chrome {
    /// One entry per open scene. Nothing of the rig this runs in appears, so
    /// the editor's own camera is not something a tree can show.
    pub(crate) fn new(project: &str, documents: Vec<Open>) -> Self {
        let mut ui = ui();
        let screen = ui.root();
        let saved = settings::load();

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
        let first = dock.panel(&mut ui, &documents[0].title);
        dock.set_closable(&mut ui, first);
        let panes: Vec<_> = documents
            .iter()
            .skip(1)
            .map(|d| {
                let p = dock.panel(&mut ui, &d.title);
                dock.set_closable(&mut ui, p);
                dock.dock(&mut ui, p, first, Side::Tab);
                p
            })
            .collect();
        let console = Console::new(&mut ui, &mut dock, first);
        let browser = dock.panel(&mut ui, "Browser");
        dock.dock(&mut ui, browser, console.pane, Side::Tab);
        dock.select(&mut ui, console.pane);
        // Tabbing a document in opens it; the first one is the one to show.
        dock.select(&mut ui, first);
        // The arrangement above is the default one. A remembered layout is
        // applied over it by title, so a panel this run has and the last did
        // not still lands somewhere.
        if let Some(l) = &saved.layout {
            dock.apply(&mut ui, l);
        }

        let files = FilePanel::new(&mut ui, dock.content(browser), project);
        console.line(&mut ui, "editor");
        console.line(&mut ui, &format!("opened {project}"));

        let mut documents: Vec<Document> = documents
            .into_iter()
            .zip([first].into_iter().chain(panes))
            .map(|(d, pane)| Document::new(&mut ui, &dock, pane, d.world, d.camera, d.rig, d.path))
            .collect();
        if let Some(l) = &saved.document {
            documents.iter_mut().for_each(|d| d.dock.apply(&mut ui, l));
        }
        Self {
            dock,
            bar,
            console,
            browser,
            files,
            documents,
            new_scenes: 0,
            new_project: NewProject::default(),
            scaffold: Arc::default(),
            this: None,
            rig: 0,
            tick: None,
            refreshed: None,
        }
    }

    /// Subscribe. Every editor handler ends in [`refresh`](Self::refresh):
    /// the panels' polls are unchanged, and what an event replaces is the
    /// *frame* they ran on, not the poll itself.
    fn wire(&mut self, ui: &mut UiCore, this: This<Chrome>) {
        ui.on_any(WAKE, this, |c, ui| c.refresh(ui));
        let boxes: Vec<NodeId> = (self.documents.iter())
            .flat_map(|d| [d.view.node(), d.hierarchy.view.node()])
            .chain([self.files.view.node()])
            .collect();
        boxes.into_iter().for_each(|n| self.watch_box(ui, n));
    }

    /// A node whose *box* moving is news rather than noise: a viewport sizes
    /// a camera to it, and a row pool sizes itself from the area it was
    /// measured in, so neither can wait for the next thing the user does.
    fn watch_box(&self, ui: &mut UiCore, n: NodeId) {
        let Some(this) = self.this else { return };
        ui.on_this(n, Events::RESIZE, this, |c, ui, _| c.refresh(ui));
    }

    /// An empty document, tabbed against the one in front and opened there.
    fn new_scene(&mut self, ui: &mut UiCore, rig: &World) {
        self.new_scenes += 1;
        let title = format!("untitled {}", self.new_scenes);
        let path = scene_path(&title);
        self.open_document(ui, rig, title, path, engine::new_world());
    }

    /// A new scene inheriting from `model`: one entity holding the
    /// instance, and nothing else until the user changes something.
    ///
    /// Saved under `scenes/` beside every other scene, because what it is is
    /// a scene — the model stays where it is, untouched, and is read again
    /// every time this one opens.
    fn inherit_scene(&mut self, ui: &mut UiCore, rig: &World, model: PathBuf) {
        let title = stem(&model);
        let path = scene_path(&title);
        if let Some(d) = self.documents.iter().find(|d| d.path == path) {
            self.dock.select(ui, d.pane);
            return self
                .console
                .line(ui, &format!("{} is already open", path.display()));
        }
        let world = document_world();
        let scene = engine::scene_asset::request_scene(&model);
        // SAFETY: nothing has been told this world exists yet, so no sweep
        // and no panel can be reading it.
        let w = unsafe { world.get_mut() };
        let e = w.new_entity(_Transform {
            name: title.clone(),
            .._Transform::default()
        });
        w.add_component(e, SceneInstance::new(scene));
        self.open_document(ui, rig, title, path, world);
        self.console
            .line(ui, &format!("new scene inheriting {}", model.display()));
    }

    /// Open a scene file as a document of its own, or say why it is not one.
    ///
    /// The world is loaded *before* the panels exist, so a file that is not a
    /// scene leaves no empty document behind: the handle is all that holds
    /// the world, and dropping it retires it.
    fn open_scene(&mut self, ui: &mut UiCore, rig: &World, path: PathBuf) {
        if let Some(d) = self.documents.iter().find(|d| d.path == path) {
            self.dock.select(ui, d.pane);
            return;
        }
        let world = engine::new_world();
        let loaded = match std::fs::read_to_string(&path) {
            // SAFETY: nothing has been told this world exists yet, so no
            // sweep and no panel can be reading it.
            Ok(text) => engine::scene_file::load(unsafe { world.get_mut() }, ROOT, &text),
            Err(e) => Err(e.to_string()),
        };
        let n = match loaded {
            Ok(ids) => ids.len(),
            Err(e) => return self.console.line(ui, &format!("{}: {e}", path.display())),
        };
        let title = path.file_stem().map_or_else(
            || path.display().to_string(),
            |s| s.to_string_lossy().into_owned(),
        );
        self.open_document(ui, rig, title, path.clone(), world);
        self.console
            .line(ui, &format!("opened {n} entities from {}", path.display()));
    }

    /// A world, a camera and the rig entity that drives it — the three things
    /// `load_project` mints per document, which is the whole of what makes a
    /// document rather than a panel.
    fn open_document(
        &mut self,
        ui: &mut UiCore,
        rig: &World,
        title: String,
        path: PathBuf,
        world: WorldHandle,
    ) {
        // SAFETY: the sweep running now took its list of worlds before this
        // one existed, so nothing can be reading it.
        unsafe { world.get_mut() }.set_simulating(false);
        let camera = CameraHandle::new(world.id());
        camera.set_grid(Some(GRID));
        let controlled = camera.clone();
        let entity = Arc::new(OnceLock::new());
        let mine = entity.clone();
        rig.spawn(
            _Transform {
                name: format!("editor camera {}", self.documents.len()),
                .._Transform::default()
            },
            move |mut e| {
                mine.set(e.id()).ok();
                e.add_component(OrbitController::for_camera(controlled));
            },
        );
        // A new document opens arranged like the one it was tabbed against,
        // rather than throwing the user back to the default three panes.
        let front = self.front();
        let like = (self.documents.iter())
            .find(|d| d.pane == front)
            .map(|d| d.dock.layout(ui));
        let pane = self.dock.panel(ui, &title);
        self.dock.set_closable(ui, pane);
        self.dock.dock(ui, pane, front, Side::Tab);
        self.dock.select(ui, pane);
        let mut document = Document::new(ui, &self.dock, pane, world, camera, entity, path);
        if let Some(l) = &like {
            document.dock.apply(ui, l);
        }
        for n in [document.view.node(), document.hierarchy.view.node()] {
            self.watch_box(ui, n);
        }
        self.documents.push(document);
        self.remember(ui);
    }

    /// Drop a document, and the rig entity whose controller is the last thing
    /// holding its camera slot. The last document is replaced rather than
    /// closed: the editor is a window onto a document.
    fn close_document(&mut self, ui: &mut UiCore, pane: PanelId, rig: &World) {
        let Some(at) = self.documents.iter().position(|d| d.pane == pane) else {
            return;
        };
        if self.documents.len() == 1 {
            self.new_scene(ui, rig);
        }
        let gone = self.documents.remove(at);
        self.dock.close(ui, pane);
        if let Some(&e) = gone.rig.get() {
            rig.destroy(e);
        }
        self.console
            .line(ui, &format!("closed {}", gone.path.display()));
        self.remember(ui);
    }

    /// Write the open documents and both docks back to `editor.json`, so the
    /// next run opens what this one left, arranged as it was left. A document
    /// that has never been saved is not a file to reopen, so it is left out
    /// rather than written as a path that will not load — which is also why
    /// saving one records it.
    fn remember(&self, ui: &mut UiCore) {
        let open = self
            .documents
            .iter()
            .map(|d| d.path.clone())
            .filter(|p| p.is_file())
            .collect();
        let front = self.front();
        let settings = settings::Settings {
            open,
            layout: Some(self.dock.layout(ui)),
            document: (self.documents.iter())
                .find(|d| d.pane == front)
                .map(|d| d.dock.layout(ui)),
        };
        if let Err(e) = settings::save(&settings) {
            self.console.line(ui, &format!("{}: {e}", settings::EDITOR));
        }
    }

    /// Scaffold `name` on a thread of its own: cargo takes seconds, and this
    /// is the thread that draws.
    fn scaffold_project(&mut self, ui: &mut UiCore, name: &str) {
        let dir = match name.is_empty() {
            true => Err("name a directory".to_string()),
            false => new_project::resolve(name),
        };
        let dir = match dir {
            Ok(dir) => dir,
            Err(e) => return self.console.line(ui, &format!("new project: {e}")),
        };
        self.console
            .line(ui, &format!("creating {} — cargo is on it", dir.display()));
        let done = self.scaffold.clone();
        std::thread::spawn(move || {
            let made = new_project::create(&dir).map(|()| dir);
            *done.lock().unwrap() = Some(made);
        });
    }

    /// Reopen the editor on what the thread finished, or say why there is
    /// nothing to open. Whatever cargo wrote is the reason.
    fn finish_scaffold(&mut self, ui: &mut UiCore) {
        let Some(made) = self.scaffold.lock().unwrap().take() else {
            return;
        };
        // `reopen` does not come back when it works: the project a process
        // entered is not one it can swap, so the new one gets a new process.
        if let Err(e) = made.and_then(|dir| new_project::reopen(&dir)) {
            e.lines().for_each(|l| self.console.line(ui, l));
        }
    }

    /// Write the document in front to its file.
    fn save_front(&mut self, ui: &mut UiCore) {
        let d = self.front_document();
        let (path, world) = (d.path.clone(), d.world.clone());
        let said = match engine::scene_file::save_to(&path, &world, ROOT) {
            Ok(()) => format!("saved {}", path.display()),
            Err(e) => format!("save failed: {e}"),
        };
        self.console.line(ui, &said);
        self.remember(ui);
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
        self.console.line(ui, &said);
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
    /// Where the identity every handler resolves through comes from, and the
    /// only place it can: `init` is when a component is first told which
    /// entity it is.
    fn init(&mut self, t: &Transform) {
        let this = This::of(t);
        (self.this, self.rig) = (Some(this), t.world());
        let mut ui = ui();
        self.wire(&mut ui, this);
        // No first `refresh` here: `init` runs inside the frame boundary that
        // is holding `&mut World`, and reaching the same world back through
        // the registry would alias it. The first placement walk reports the
        // panes' boxes as moved, and that is the first refresh.
    }

    /// What is left when the events are subscribed to: two queues nothing in
    /// the UI can name, and three keys the keyboard route never sees because
    /// no field has focus.
    fn update(&mut self, _dt: f32, _transform: &Transform, _world: &World) {
        let mut ui = ui();
        self.finish_scaffold(&mut ui);
        // Subscene instantiation is the one structural change the editor does
        // not drive, so it arrives as an event — and always in the world the
        // renderer draws subscenes into, which is the first one handed to the
        // window.
        if self.settling() {
            self.refresh(&mut ui);
        }
        if !engine::scene_asset::drain_instantiated().is_empty() {
            // Which document's world it landed in is not reported, and
            // rebuilding a tree that did not change costs one frame's rows.
            for d in self.documents.iter_mut() {
                d.hierarchy.invalidate();
            }
            self.refresh(&mut ui);
        }
        // Raw keys, not the keyboard route: that delivers to whatever holds
        // focus, and none of these are a field's. Delete is the hierarchy's,
        // and it answers to the panel the pointer is over.
        if !ui.keyboard_captured() {
            if engine::input::key_pressed(KeyCode::Delete) {
                self.refresh(&mut ui);
            }
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

impl Chrome {
    /// Re-read everything the panels show. Reached from the subscriptions in
    /// [`wire`](Self::wire) and nowhere else, so an idle frame runs none of
    /// it — which is the whole point of the layer above.
    fn refresh(&mut self, ui: &mut UiCore) {
        // Once a frame, however many things woke it — a click, the readback
        // clock and a queued edit landing can all be the same frame. The
        // panels read state the input fold latches for exactly one frame, so
        // a second pass would spawn the same entity again.
        if self.refreshed == Some(ui.frame()) {
            return;
        }
        self.refreshed = Some(ui.frame());
        let Some(rig) = engine::worlds::world(self.rig) else {
            return;
        };
        let world: &World = &rig;
        match self.bar.update(ui).map(|(m, i)| MENUS[m].1[i]) {
            Some("new project") => self.new_project.open(ui),
            Some("new scene") => self.new_scene(ui, world),
            Some("save scene") => self.save_front(ui),
            Some("reload scene") => self.reload_front(ui),
            // The window's close button is `event_loop.exit()` with nothing
            // to unwind either, so this is the same exit by another door.
            Some("quit") => std::process::exit(0),
            Some("console") => self.dock.select(ui, self.console.pane),
            Some("browser") => self.dock.select(ui, self.browser),
            Some("translate") => gizmo::set_mode(GizmoMode::Translate),
            Some("rotate") => gizmo::set_mode(GizmoMode::Rotate),
            Some("scale") => gizmo::set_mode(GizmoMode::Scale),
            _ => {}
        }
        if let Some(name) = self.new_project.update(ui) {
            self.scaffold_project(ui, &name);
        }
        if let Some(p) = self.dock.update(ui) {
            self.close_document(ui, p, world);
        }
        // A scene opens as a document of its own; anything else says so
        // rather than opening an empty one.
        if let Some(path) = self.files.update(ui) {
            match path.extension().is_some_and(|e| e == "json") {
                true => self.open_scene(ui, world, path),
                // A model is not a scene and never becomes one: opening it
                // makes a scene that inherits from it, which is the only way
                // to change a model without writing to the model.
                false if engine::asset_type::kind_of(&path) == Some(AssetKind::Scene) => {
                    self.inherit_scene(ui, world, path)
                }
                false => {
                    let what = match engine::asset_type::kind_of(&path) {
                        Some(k) => format!("a {} — drag it onto a property", k.name()),
                        None => "not something the editor opens".to_string(),
                    };
                    self.console
                        .line(ui, &format!("{} is {what}", path.display()))
                }
            }
        }
        let mut said = Vec::new();
        for d in self.documents.iter_mut() {
            // A document behind another tab is not drawn: its camera keeps
            // its target and its size, and stops rendering into them.
            d.editing.set_enabled(self.dock.showing(d.pane));
            said.extend(d.update(ui));
        }
        said.iter().for_each(|s| self.console.line(ui, s));
        // Once per gesture, not per frame: `changed` is set by the drop and
        // the release, and quit is a `process::exit` with nothing to hook.
        let moved =
            (self.documents.iter_mut()).fold(self.dock.changed(), |a, d| a | d.dock.changed());
        if moved {
            self.remember(ui);
        }
        self.follow_simulation(ui);
    }

    /// An edit the panels queued has landed, and the frame it landed on is
    /// not one any event names.
    ///
    /// A spawn, a destroy and an added component are all applied at the
    /// *next* frame boundary, so the click that asked for one walked a
    /// hierarchy that did not have it yet. Two booleans and an atomic per
    /// document, which is what an idle frame pays to keep this honest.
    fn settling(&self) -> bool {
        self.documents.iter().any(|d| {
            d.hierarchy.queued
                || d.hierarchy.spawned.load(Ordering::Relaxed) != NO_SPAWN
                || d.inspector.awaiting
        })
    }

    /// A running scene moves values no event names — the inspector would show
    /// the pose the last click left. So while one is playing, and only then,
    /// the panels read back on a clock.
    ///
    /// 10 Hz rather than per frame: this is for values the *simulation*
    /// moves, and a number the user is dragging is written by the drag.
    fn follow_simulation(&mut self, ui: &mut UiCore) {
        let Some(this) = self.this else { return };
        let playing = self.documents.iter().any(|d| d.play.is_some());
        match (playing, self.tick) {
            (true, None) => {
                let root = ui.root();
                self.tick = Some(ui.every(root, 10.0, this, |c, ui, _| c.refresh(ui)));
            }
            (false, Some(t)) => {
                ui.stop(t);
                self.tick = None;
            }
            _ => {}
        }
    }
}
