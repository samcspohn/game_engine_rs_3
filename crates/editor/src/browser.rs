//! The project file browser: the directory the editor entered, as a tree.

use crate::fill;
use engine::{
    AssetKind,
    ui::{
        ButtonStyle, Label, NodeId, Row, RowStyle, ScrollbarStyle, Source, Tree, TreeDrag, UiCore,
        UiRef, UiStyle,
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
}

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
            },
        );
        me.with(ui, |p, ui| p.status(ui, None));

        // The filesystem changes without telling anyone — a save, a build, a
        // file dropped in from outside. Rescanning keeps the ids, so what was
        // open stays open.
        ui.on_click(refresh, move |ui| {
            me.with(ui, |p, ui| {
                p.tree = Arc::new(FileTree::scan(&p.root));
                p.view.show(ui, source(&p.tree));
            });
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
