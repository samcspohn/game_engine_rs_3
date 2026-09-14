//! The project file browser: the directory the editor entered, as a tree.

use crate::fill;
use engine::{
    AssetKind,
    ui::{
        Button, ButtonStyle, Label, NodeId, RowStyle, ScrollbarStyle, TreeDrag, TreeView, UiCore,
        UiStyle,
        style::{AlignItems, Display, Size, Style, px, zero},
        theme,
    },
};
use std::path::{Path, PathBuf};

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
/// Read in full rather than per expansion: [`TreeView::sync`] asks for a
/// node's children on every frame it draws a row, so a listing that hit the
/// filesystem would readdir once per visible row per frame.
#[derive(Clone)]
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

/// The project's files, as a tree rooted at the directory the editor entered.
///
/// It reports the file a double-click asked for and opens nothing itself:
/// what a document is belongs to the chrome, which is also the only thing
/// that can tell whether one is open already.
#[derive(Clone)]
pub(crate) struct FilePanel {
    pub(crate) view: TreeView<Label, AssetDrag>,
    tree: FileTree,
    root: String,
    selected: Option<u64>,
    status: Label,
    refresh: Button,
}

impl FilePanel {
    pub(crate) fn new(ui: &mut UiCore, pane: NodeId, project: &str) -> Self {
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
        let view: TreeView<Label, AssetDrag> =
            TreeView::new(ui, gutter, fill(), RowStyle::default(), 0);
        ui.set_background(view.node(), UiStyle::fill(t.backdrop).radius(t.radius));
        ui.scrollbar(gutter, view.node(), ScrollbarStyle::default());
        Self {
            view,
            tree: FileTree::scan(&root),
            root,
            selected: None,
            status,
            refresh,
        }
    }

    /// The file a double-click asked to open, for the one frame it asked.
    pub(crate) fn update(&mut self, ui: &mut UiCore) -> Option<PathBuf> {
        // The filesystem changes without telling anyone — a save, a build, a
        // file dropped in from outside. Rescanning keeps the ids, so what was
        // open stays open.
        if ui.clicked(self.refresh) {
            self.tree = FileTree::scan(&self.root);
            self.selected = None;
            self.view.invalidate();
        }
        if let Some(id) = self.view.clicked(ui) {
            self.selected = Some(id);
        }
        let mut opened = None;
        if let Some(id) = self.view.double_clicked(ui) {
            match self.tree.nodes[id as usize].dir {
                false => opened = Some(self.tree.nodes[id as usize].path.clone()),
                // A directory has nothing to open, so its double click is the
                // one every explorer gives it.
                true => {
                    let want = !self.view.is_expanded(id);
                    let nodes = &self.tree.nodes;
                    self.view.set_expanded(id, want, &mut |id, out| {
                        out.extend(nodes[id as usize].kids.iter().copied())
                    });
                }
            }
        }

        // The view reports what a press picked up; the panel grabs, because
        // only it knows what a row names. A directory is not a handle any
        // property can hold, so it never leaves the tree.
        if let Some(id) = self.view.picked_up(ui) {
            let node = &self.tree.nodes[id as usize];
            if !node.dir {
                let payload = AssetDrag {
                    kind: engine::asset_type::kind_of(&node.path),
                    path: node.path.clone(),
                    node: id,
                };
                let text = self.tree.label(id);
                self.view.grab(ui, payload, |ui, r| r.set_text(ui, &text));
            }
        }

        let (tree, selected) = (&self.tree, self.selected);
        self.view.sync(
            ui,
            |id, out| out.extend(tree.nodes[id as usize].kids.iter().copied()),
            |ui, r, id| {
                r.set_text(ui, &tree.label(id));
                r.set_selected(ui, selected == Some(id));
            },
        );
        // Minus the root, which is the project rather than something in it.
        let text = match selected {
            Some(id) => self.tree.nodes[id as usize].path.display().to_string(),
            None => format!("{} entries", self.tree.nodes.len() - 1),
        };
        self.status.set_text(ui, &text);
        opened
    }
}
