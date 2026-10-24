//! The editor shell: the menu bar, and the dock the documents open in. Every
//! panel drives itself; the chrome hears what changes documents. See
//! [editor](../../../docs/notes/editor.md).

use crate::browser::FilePanel;
use crate::console::Console;
use crate::document::Document;
use crate::project::{GRID, Open, document_world, stem};
use crate::{fill, new_project, scene_path, settings};
use engine::inherit::SceneInstance;
use engine::{
    AssetKind, Component, ComponentRef, Entity, Export, OrbitController, WorldHandle,
    input::{Key, Keystroke, Mods},
    transform::{_Transform, ROOT, Transform},
    ui::{
        DockSpace, DockStyle, Event, Events, Keep, MenuBar, MenuBarStyle, PanelId, Side, UiCore,
        UiRef, defer,
        style::{Display, FlexDirection, Size, Style, percent},
        ui,
    },
};
use engine_editor_api::{CameraHandle, GizmoMode, gizmo};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// One dock filling the window: a panel per open document, and the panes
/// that belong to the application rather than to any one of them. Arranged
/// once here; after that the layout is the user's (ADR-0006 phase 4).
#[derive(Clone)]
pub(crate) struct Chrome {
    dock: UiRef<DockSpace>,
    bar: MenuBar,
    /// The two panels the View menu brings forward. They share a leaf, so
    /// picking one is `select` and nothing else.
    console: Console,
    browser: PanelId,
    /// What the `browser` panel shows; built in `init`, since what it opens
    /// goes through the chrome.
    files: Option<UiRef<FilePanel>>,
    project: String,
    /// Each with the outer panel it fills, which says which is in front.
    documents: Vec<(PanelId, UiRef<Document>)>,
    /// Scenes made from the menu, so two of them never share a tab title.
    new_scenes: usize,
    /// What every handler resolves through, and the world to reach it in.
    /// Minted in `init`, which is the first place a component is told which
    /// entity it is.
    this: Option<ComponentRef<Chrome>>,
    rig: engine::transform::WorldId,
}

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
        let bar = MenuBar::new(&mut ui, shell, MenuBarStyle::default());
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
        console.line(&mut ui, "editor");
        console.line(&mut ui, &format!("opened {project}"));

        let documents: Vec<_> = documents
            .into_iter()
            .zip([first].into_iter().chain(panes))
            .map(|(d, pane)| {
                let content = dock.content(pane);
                let doc =
                    Document::new(&mut ui, content, d.world, d.camera, d.rig, d.path, console);
                if let Some(l) = &saved.document {
                    doc.with(&mut ui, |d, ui| d.dock.with(ui, |k, ui| k.apply(ui, l)));
                }
                (pane, doc)
            })
            .collect();
        let dock = dock.listen(&mut ui);
        Self {
            dock,
            bar,
            console,
            browser,
            files: None,
            project: project.to_string(),
            documents,
            new_scenes: 0,
            this: None,
            rig: 0,
        }
    }

    /// Subscribe: the menu bar's items, the dock's tabs, the browser's
    /// double-click, and the two things the root hears.
    fn wire(&mut self, ui: &mut UiCore, this: ComponentRef<Chrome>) {
        let on = move |f: fn(&mut Chrome, &mut UiCore)| {
            move |ui: &mut UiCore| {
                this.with(|c| f(c, ui));
            }
        };
        let mode = |m: GizmoMode| move |_: &mut UiCore| gizmo::set_mode(m);
        self.bar
            .menu(ui, "File")
            .item(ui, "new project", on(|c, ui| c.new_project(ui)))
            .item(ui, "new scene", on(|c, ui| c.new_scene(ui)))
            .item(ui, "save scene", on(|c, ui| c.save_front(ui)))
            .item(ui, "reload scene", on(|c, ui| c.reload_front(ui)))
            // The window's close button is `event_loop.exit()` with nothing
            // to unwind either, so this is the same exit by another door.
            .item(ui, "quit", |_| std::process::exit(0));
        self.bar
            .menu(ui, "View")
            .item(ui, "console", on(|c, ui| c.bring(ui, c.console.pane)))
            .item(ui, "browser", on(|c, ui| c.bring(ui, c.browser)));
        self.bar
            .menu(ui, "Tools")
            .item(ui, "translate", mode(GizmoMode::Translate))
            .item(ui, "rotate", mode(GizmoMode::Rotate))
            .item(ui, "scale", mode(GizmoMode::Scale));

        self.dock.with(ui, |d, _| {
            d.on_close(move |ui, p| {
                this.with(|c| c.close_document(ui, p));
            });
            d.on_change(move |ui| {
                this.with(|c| c.rearranged(ui));
            });
        });
        let content = self.dock.with(ui, |d, _| d.content(self.browser));
        if let Some(pane) = content {
            let files = FilePanel::new(ui, pane, &self.project, move |ui, path| {
                this.with(|c| c.open_path(ui, path));
            });
            self.files = Some(files);
        }
        let root = ui.root();
        ui.on(root, Events::KEY, move |ui, ev| {
            this.with(|c| c.key(ui, ev));
            Keep::Yes
        });
        ui.on_notify(root, move |ui| {
            this.with(|c| c.outside_changed(ui));
        });
        for &(_, doc) in &self.documents {
            Self::watch(ui, doc, this);
        }
    }

    /// A document's own dock is part of what `editor.json` remembers.
    fn watch(ui: &mut UiCore, doc: UiRef<Document>, this: ComponentRef<Chrome>) {
        doc.with(ui, |d, ui| {
            d.dock.with(ui, |k, _| {
                k.on_change(move |ui| {
                    this.with(|c| c.remember(ui));
                })
            })
        });
    }

    fn bring(&mut self, ui: &mut UiCore, p: PanelId) {
        self.dock.with(ui, |d, ui| d.select(ui, p));
    }

    fn new_project(&mut self, ui: &mut UiCore) {
        let Some(this) = self.this else { return };
        new_project::ask(ui, move |ui, name| {
            this.with(|c| c.scaffold_project(ui, &name));
        });
    }

    /// A tab moved, opened or a divider was released: remembered, and the
    /// gizmo follows the document in front.
    fn rearranged(&mut self, ui: &mut UiCore) {
        self.remember(ui);
        if let Some(doc) = self.front_document(ui) {
            doc.with(ui, |d, ui| d.aim_gizmo(ui));
        }
    }

    /// The root heard that something outside the UI moved: an instance
    /// materialised, or the gizmo dragged an entity.
    fn outside_changed(&mut self, ui: &mut UiCore) {
        // Which document's world an instance landed in is not reported.
        let landed = !engine::scene_asset::drain_instantiated().is_empty();
        for &(pane, doc) in &self.documents {
            let showing = self.dock.with(ui, |d, _| d.showing(pane)) == Some(true);
            doc.with(ui, |d, ui| {
                if landed {
                    d.hierarchy.with(ui, |h, ui| h.tree.invalidate(ui));
                }
                if showing {
                    d.read_back(ui);
                }
            });
        }
    }

    /// A scene opens as a document of its own; anything else says so rather
    /// than opening an empty one.
    fn open_path(&mut self, ui: &mut UiCore, path: PathBuf) {
        match path.extension().is_some_and(|e| e == "json") {
            true => self.open_scene(ui, path),
            // A model is not a scene and never becomes one: opening it makes
            // a scene that inherits from it, which is the only way to change
            // a model without writing to the model.
            false if engine::asset_type::kind_of(&path) == Some(AssetKind::Scene) => {
                self.inherit_scene(ui, path)
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

    /// An empty document, tabbed against the one in front and opened there.
    fn new_scene(&mut self, ui: &mut UiCore) {
        self.new_scenes += 1;
        let title = format!("untitled {}", self.new_scenes);
        let path = scene_path(&title);
        self.open_document(ui, title, path, engine::new_world());
    }

    /// The document open on `path`, brought to the front.
    fn bring_open(&mut self, ui: &mut UiCore, path: &PathBuf) -> bool {
        let open = (self.documents.iter())
            .find(|(_, d)| d.with(ui, |d, _| &d.path == path) == Some(true))
            .map(|&(pane, _)| pane);
        if let Some(pane) = open {
            self.bring(ui, pane);
        }
        open.is_some()
    }

    /// A new scene inheriting from `model`: one entity holding the
    /// instance, and nothing else until the user changes something.
    ///
    /// Saved under `scenes/` beside every other scene, because what it is is
    /// a scene — the model stays where it is, untouched, and is read again
    /// every time this one opens.
    fn inherit_scene(&mut self, ui: &mut UiCore, model: PathBuf) {
        let title = stem(&model);
        let path = scene_path(&title);
        if self.bring_open(ui, &path) {
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
        self.open_document(ui, title, path, world);
        self.console
            .line(ui, &format!("new scene inheriting {}", model.display()));
    }

    /// Open a scene file as a document of its own, or say why it is not one.
    ///
    /// The world is loaded *before* the panels exist, so a file that is not a
    /// scene leaves no empty document behind: the handle is all that holds
    /// the world, and dropping it retires it.
    fn open_scene(&mut self, ui: &mut UiCore, path: PathBuf) {
        if self.bring_open(ui, &path) {
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
        self.open_document(ui, title, path.clone(), world);
        self.console
            .line(ui, &format!("opened {n} entities from {}", path.display()));
    }

    /// A world, a camera and the rig entity that drives it — the three things
    /// `load_project` mints per document, which is the whole of what makes a
    /// document rather than a panel.
    fn open_document(&mut self, ui: &mut UiCore, title: String, path: PathBuf, world: WorldHandle) {
        let (Some(rig), Some(this)) = (engine::worlds::world(self.rig), self.this) else {
            return;
        };
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
        let front = self.front(ui);
        let like = self
            .front_document(ui)
            .and_then(|d| d.with(ui, |d, ui| d.dock.with(ui, |k, ui| k.layout(ui))))
            .flatten();
        let made = self.dock.with(ui, |d, ui| {
            let pane = d.panel(ui, &title);
            d.set_closable(ui, pane);
            d.dock(ui, pane, front, Side::Tab);
            d.select(ui, pane);
            (pane, d.content(pane))
        });
        let Some((pane, content)) = made else { return };
        let doc = Document::new(ui, content, world, camera, entity, path, self.console);
        if let Some(l) = &like {
            doc.with(ui, |d, ui| d.dock.with(ui, |k, ui| k.apply(ui, l)));
        }
        Self::watch(ui, doc, this);
        self.documents.push((pane, doc));
        self.remember(ui);
    }

    /// Drop a document, and the rig entity whose controller is the last thing
    /// holding its camera slot. The last document is replaced rather than
    /// closed: the editor is a window onto a document.
    fn close_document(&mut self, ui: &mut UiCore, pane: PanelId) {
        let Some(at) = self.documents.iter().position(|&(p, _)| p == pane) else {
            return;
        };
        if self.documents.len() == 1 {
            self.new_scene(ui);
        }
        let (_, gone) = self.documents.remove(at);
        let Some((rig, path)) = gone.with(ui, |d, _| (d.rig.get().copied(), d.path.clone())) else {
            return;
        };
        // The document goes with the pane's nodes, and its world with it.
        self.dock.with(ui, |d, ui| d.close(ui, pane));
        if let (Some(e), Some(w)) = (rig, engine::worlds::world(self.rig)) {
            w.destroy(e);
        }
        self.console.line(ui, &format!("closed {}", path.display()));
        self.remember(ui);
    }

    /// Write the open documents and both docks back to `editor.json`, so the
    /// next run opens what this one left, arranged as it was left. A document
    /// that has never been saved is not a file to reopen, so it is left out
    /// rather than written as a path that will not load — which is also why
    /// saving one records it.
    fn remember(&self, ui: &mut UiCore) {
        let open = (self.documents.iter())
            .filter_map(|(_, d)| d.with(ui, |d, _| d.path.clone()))
            .filter(|p| p.is_file())
            .collect();
        let document = (self.front_document(ui))
            .and_then(|d| d.with(ui, |d, ui| d.dock.with(ui, |k, ui| k.layout(ui))))
            .flatten();
        let settings = settings::Settings {
            open,
            layout: self.dock.with(ui, |d, ui| d.layout(ui)),
            document,
        };
        if let Err(e) = settings::save(&settings) {
            self.console.line(ui, &format!("{}: {e}", settings::EDITOR));
        }
    }

    /// Scaffold `name` on a thread of its own: cargo takes seconds, and this
    /// is the thread that draws. The answer comes back through `defer`.
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
        let Some(this) = self.this else { return };
        std::thread::spawn(move || {
            let made = new_project::create(&dir).map(|()| dir);
            defer(move |ui| {
                this.with(|c| c.finish_scaffold(ui, made));
            });
        });
    }

    /// Reopen the editor on what the thread finished, or say why there is
    /// nothing to open. Whatever cargo wrote is the reason.
    fn finish_scaffold(&mut self, ui: &mut UiCore, made: Result<PathBuf, String>) {
        // `reopen` does not come back when it works: the project a process
        // entered is not one it can swap, so the new one gets a new process.
        if let Err(e) = made.and_then(|dir| new_project::reopen(&dir)) {
            e.lines().for_each(|l| self.console.line(ui, l));
        }
    }

    /// Write the document in front to its file.
    fn save_front(&mut self, ui: &mut UiCore) {
        let Some(doc) = self.front_document(ui) else {
            return;
        };
        let Some((path, world)) = doc.with(ui, |d, _| (d.path.clone(), d.world.clone())) else {
            return;
        };
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
        let Some(doc) = self.front_document(ui) else {
            return;
        };
        let said = doc.with(ui, |d, ui| {
            // SAFETY: a document does not simulate, so no sweep is reading
            // it, and the panels that show it are re-walked below.
            let world = unsafe { d.world.get_mut() };
            for child in world.hierarchy().children(ROOT).to_vec() {
                world.remove_entity(Entity::new(child));
            }
            let said = match engine::scene_file::load_from(&d.path, world, ROOT) {
                Ok(ids) => format!("loaded {} entities from {}", ids.len(), d.path.display()),
                Err(e) => format!("load failed: {e}"),
            };
            // The tree is walking slots that no longer exist, and the
            // selection named one of them.
            d.hierarchy.with(ui, |h, ui| {
                h.tree.select(ui, None);
                h.tree.invalidate(ui);
            });
            said
        });
        if let Some(said) = said {
            self.console.line(ui, &said);
        }
    }

    /// The document whose tab is open.
    fn front_document(&self, ui: &mut UiCore) -> Option<UiRef<Document>> {
        let front = self.front(ui);
        (self.documents.iter())
            .find(|&&(p, _)| p == front)
            .map(|&(_, d)| d)
    }

    /// The panel of the document whose tab is open. Its leaf always has one,
    /// so the fallback only covers a list `new` guarantees is not empty.
    fn front(&self, ui: &mut UiCore) -> PanelId {
        let open = (self.documents.iter())
            .find(|&&(p, _)| self.dock.with(ui, |d, _| d.showing(p)) == Some(true));
        open.unwrap_or(&self.documents[0]).0
    }

    /// A keystroke no field took: the gizmo's W/E/R, and Delete for the
    /// hierarchy under the pointer, whose destroy wakes the panels on landing.
    fn key(&mut self, ui: &mut UiCore, ev: &Event) {
        let Event::Key(stroke) = ev else { return };
        let letter = match stroke {
            Keystroke::Text(t) => t.to_lowercase(),
            Keystroke::Key(Key::Char(c), m) if *m == Mods::NONE => c.to_string(),
            Keystroke::Key(Key::Delete, _) => {
                for &(pane, doc) in &self.documents {
                    if self.dock.with(ui, |d, _| d.showing(pane)) == Some(true) {
                        doc.with(ui, |d, ui| d.delete_key(ui));
                    }
                }
                return;
            }
            _ => return,
        };
        let mode = match letter.as_str() {
            "w" => Some(GizmoMode::Translate),
            "e" => Some(GizmoMode::Rotate),
            "r" => Some(GizmoMode::Scale),
            _ => None,
        };
        if let Some(m) = mode {
            gizmo::set_mode(m);
        }
    }
}

/// The editor's own chrome is not authored content — nothing to inspect.
impl Export for Chrome {}

impl Component for Chrome {
    const HAS_UPDATE: bool = false;

    /// Where the identity every handler resolves through comes from, and the
    /// only place it can: `init` is when a component is first told which
    /// entity it is.
    fn init(&mut self, t: &Transform) {
        let this = ComponentRef::of(t);
        (self.this, self.rig) = (Some(this), t.world());
        let mut ui = ui();
        self.wire(&mut ui, this);
    }
}
