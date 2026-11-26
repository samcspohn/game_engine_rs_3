//! The project file browser: the directory the editor entered, as a tree.

use crate::fill;
use engine::{
    AssetKind,
    ui::{
        ButtonStyle, Label, MenuStyle, NodeId, Row, RowStyle, ScrollbarStyle, Source, Tree,
        TreeDrag, UiCore, UiRef, UiStyle,
        style::{AlignItems, Display, Size, Style, px, zero},
        theme,
    },
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// What the browser puts in flight: where the file is, and what the registry
/// says it is. A row whose extension nothing claims still drags and simply
/// matches no property — the kind is what a drop target tests, never the
/// extension.
#[derive(Clone)]
pub struct AssetDrag {
    pub(crate) path: PathBuf,
    pub(crate) kind: Option<AssetKind>,
    node: u64,
}

impl TreeDrag for AssetDrag {
    fn node(&self) -> u64 {
        self.node
    }
}

/// One path under the project, as a row of the browser.
#[derive(Clone)]
struct FileNode {
    /// Relative to the directory the editor entered, which is what a scene
    /// file is named by and what the root's empty path makes true.
    path: PathBuf,
    name: String,
    dir: bool,
    kids: Vec<u64>,
}

/// Build output, version control and anything a dot hides. A browser that
/// opens on `target/` shows what the project was built into rather than the
/// project.
fn ignored(name: &str) -> bool {
    name == "target" || name.starts_with('.')
}

/// The project directory flattened into an arena a node id indexes, node 0
/// being the root.
///
/// Read in full rather than per expansion: a tree asks for a node's children
/// every time it binds a row, so a listing that hit the filesystem would
/// readdir once per visible row per bind.
struct FileTree {
    nodes: Vec<FileNode>,
}

impl FileTree {
    fn scan(root: &str) -> Self {
        let mut nodes = vec![FileNode {
            path: PathBuf::new(),
            name: root.to_string(),
            dir: true,
            kids: Vec::new(),
        }];
        let mut pending = vec![0usize];
        while let Some(i) = pending.pop() {
            // A directory that cannot be read lists as empty — the browser
            // is not the place to learn about permissions.
            let mut entries: Vec<(String, bool)> =
                std::fs::read_dir(Path::new(".").join(&nodes[i].path))
                    .into_iter()
                    .flatten()
                    .flatten()
                    .map(|e| {
                        let dir = e.file_type().is_ok_and(|t| t.is_dir());
                        (e.file_name().to_string_lossy().into_owned(), dir)
                    })
                    .filter(|(name, _)| !ignored(name))
                    .collect();
            entries.sort_by_key(|(name, dir)| (!*dir, name.to_lowercase()));
            for (name, dir) in entries {
                let path = nodes[i].path.join(&name);
                nodes.push(FileNode {
                    path,
                    name,
                    dir,
                    kids: Vec::new(),
                });
                let id = nodes.len() - 1;
                nodes[i].kids.push(id as u64);
                if dir {
                    pending.push(id);
                }
            }
        }
        Self { nodes }
    }

    fn id_of(&self, path: &Path) -> Option<u64> {
        Some(self.nodes.iter().position(|n| n.path == path)? as u64)
    }

    /// A trailing slash rather than an icon: an empty directory gets no
    /// disclosure arrow, and would otherwise read as a file.
    fn label(&self, id: u64) -> String {
        let n = &self.nodes[id as usize];
        match n.dir {
            true => format!("{}/", n.name),
            false => n.name.clone(),
        }
    }
}

/// What the tree reads: the scan, shared rather than copied, and replaced
/// whole by a rescan.
fn source(tree: &Arc<FileTree>) -> Source<Label> {
    let (a, b) = (tree.clone(), tree.clone());
    Source::new(
        move |id, out| out.extend(a.nodes[id as usize].kids.iter().copied()),
        move |ui, r: Row<'_, Label>, id| r.set_text(ui, &b.label(id)),
    )
}

/// The project's files, as a tree rooted at the directory the editor entered.
/// Kept on its gutter (`UiCore::own`).
///
/// It reports the file a double-click asked for and opens nothing itself:
/// what a document is belongs to the chrome, which is also the only thing
/// that can tell whether one is open already.
pub(crate) struct FilePanel {
    view: Tree<Label, AssetDrag>,
    tree: Arc<FileTree>,
    root: String,
    status: Label,
    me: Option<UiRef<FilePanel>>,
}

type Open = dyn Fn(&mut UiCore, PathBuf) + Send + Sync;

impl FilePanel {
    /// `open` is handed the file a double-click asks for.
    pub(crate) fn new(
        ui: &mut UiCore,
        pane: NodeId,
        project: &str,
        open: impl Fn(&mut UiCore, PathBuf) + Send + Sync + 'static,
    ) -> UiRef<Self> {
        let t = theme();
        // The crate directory's own name: `--project crates/test-game` opens
        // a browser on `test-game`, not on `crates`.
        let root = Path::new(project)
            .file_name()
            .map_or_else(|| project.to_string(), |n| n.to_string_lossy().into_owned());
        let bar = ui.node(
            pane,
            Style {
                display: Display::Flex,
                align_items: Some(AlignItems::CENTER),
                ..Default::default()
            },
        );
        let button = ButtonStyle {
            text_px: 10.0,
            padding: 2.0,
            ..ButtonStyle::default()
        };
        let refresh = ui.button(bar, "refresh", button);
        let status = ui.label(pane, 10.0, t.text_dim, "");
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
        let tree = Arc::new(FileTree::scan(&root));
        let view = Tree::new(ui, gutter, fill(), RowStyle::default(), 0, source(&tree));
        ui.set_background(view.node(), UiStyle::fill(t.backdrop).radius(t.radius));
        ui.scrollbar(gutter, view.node(), ScrollbarStyle::default());
        let me = ui.own(
            gutter,
            Self {
                view,
                tree,
                root,
                status,
                me: None,
            },
        );
        me.with(ui, |p, ui| {
            p.me = Some(me);
            p.status(ui, None);
        });

        ui.on_click(refresh, move |ui| {
            me.with(ui, |p, ui| p.rescan(ui));
        });
        let open = Arc::new(open);
        let make = open.clone();
        view.on_row_right_click(ui, move |ui, id, at| {
            me.with(ui, |p, ui| p.new_menu(ui, id, at, make.clone()));
        });
        view.on_select(ui, move |ui, id| {
            me.with(ui, |p, ui| p.status(ui, id));
        });
        view.on_row_double_click(ui, move |ui, id| {
            let tree = me.with(ui, |p, _| p.tree.clone());
            let Some(node) = tree.as_ref().map(|t| &t.nodes[id as usize]) else {
                return;
            };
            match node.dir {
                false => open(ui, node.path.clone()),
                // A directory has nothing to open, so its double click is the
                // one every explorer gives it.
                true => {
                    let want = !view.is_expanded(ui, id);
                    view.set_expanded(ui, id, want);
                }
            }
        });
        // The tree offers the row; the panel grabs, because only it knows
        // what a row names. A directory is not a handle any property can
        // hold, so it never leaves the tree.
        view.on_pick_up(ui, move |ui, id| {
            let Some(tree) = me.with(ui, |p, _| p.tree.clone()) else {
                return;
            };
            let node = &tree.nodes[id as usize];
            if !node.dir {
                let payload = AssetDrag {
                    kind: engine::asset_type::kind_of(&node.path),
                    path: node.path.clone(),
                    node: id,
                };
                let text = tree.label(id);
                view.grab(ui, payload, |ui, r| r.set_text(ui, &text));
            }
        });
        me
    }

    /// The filesystem changes without telling anyone — a save, a build, a
    /// file dropped in from outside. A new file shifts every id after it, so
    /// what was open and selected is carried over by path.
    fn rescan(&mut self, ui: &mut UiCore) {
        let (view, old) = (self.view, self.tree.clone());
        let open: Vec<u64> = (1..old.nodes.len() as u64)
            .filter(|&id| view.is_expanded(ui, id))
            .collect();
        let selected = view.selected(ui).map(|id| &old.nodes[id as usize].path);
        for &id in &open {
            view.set_expanded(ui, id, false);
        }
        self.tree = Arc::new(FileTree::scan(&self.root));
        view.show(ui, source(&self.tree));
        for id in open
            .iter()
            .filter_map(|&id| self.tree.id_of(&old.nodes[id as usize].path))
        {
            view.set_expanded(ui, id, true);
        }
        view.select(ui, selected.and_then(|p| self.tree.id_of(p)));
    }

    /// Open every folder above `path` and select it.
    fn reveal(&self, ui: &mut UiCore, path: &Path) {
        for dir in path.ancestors().skip(1).filter_map(|d| self.tree.id_of(d)) {
            self.view.set_expanded(ui, dir, true);
        }
        self.view.select(ui, self.tree.id_of(path));
    }

    /// A new resource of each registered type, in the directory under the
    /// pointer or beside the file. Made with its defaults, then opened.
    fn new_menu(&self, ui: &mut UiCore, id: u64, at: [f32; 2], open: Arc<Open>) {
        let kinds = engine::resource::kinds();
        if kinds.is_empty() {
            return;
        }
        let n = &self.tree.nodes[id as usize];
        let dir = match n.dir {
            true => n.path.clone(),
            false => n.path.parent().map(Path::to_path_buf).unwrap_or_default(),
        };
        let me = self.me;
        let mut menu = ui.context_menu(at, MenuStyle::default());
        for (name, ext) in kinds {
            let (dir, open) = (dir.clone(), open.clone());
            menu = menu.item(ui, &format!("new {name}"), move |ui| {
                let path = (1..)
                    .map(|i| match i {
                        1 => dir.join(format!("new.{ext}")),
                        i => dir.join(format!("new_{i}.{ext}")),
                    })
                    .find(|p| !p.exists())
                    .expect("an unbounded range ends at a free name");
                let made = engine::resource::create(name, &path);
                me.map(|me| {
                    me.with(ui, |p, ui| {
                        p.rescan(ui);
                        match &made {
                            Ok(_) => p.reveal(ui, &path),
                            Err(e) => p.status.set_text(ui, e),
                        }
                    })
                });
                if made.is_ok() {
                    open(ui, path);
                }
            });
        }
    }

    /// The selection's path, or how much there is. Minus the root, which is
    /// the project rather than something in it.
    fn status(&self, ui: &mut UiCore, selected: Option<u64>) {
        let text = match selected {
            Some(id) => self.tree.nodes[id as usize].path.display().to_string(),
            None => format!("{} entries", self.tree.nodes.len() - 1),
        };
        self.status.set_text(ui, &text);
    }
}
