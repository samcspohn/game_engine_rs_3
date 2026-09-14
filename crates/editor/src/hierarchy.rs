//! The scene hierarchy panel: a collapsible tree over the live
//! `TransformHierarchy` (ADR-0008 / ADR-0009).

use crate::browser::AssetDrag;
use crate::fill;
use crate::project::stem;
use engine::inherit::SceneInstance;
use engine::{
    AssetKind, Component, Entity, Export, KeyCode, World,
    transform::{_Transform, ROOT},
    ui::{
        Button, ButtonStyle, Label, MenuStyle, NodeId, RowContent, RowStyle, ScrollbarStyle,
        TextField, TextFieldStyle, TreeDrag, TreeView, UiCore, UiStyle,
        style::{AlignItems, Display, Size, Style, px, zero},
        theme,
    },
};
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

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
pub(crate) fn row_text(h: &engine::transform::TransformHierarchy, id: u64) -> String {
    let name = h.name(id as u32);
    match name.is_empty() {
        true => format!("entity {id}"),
        false => name.to_string(),
    }
}

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
pub(crate) struct NameRow {
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
        ui.set_visible(self.label.node(), !editing);
        ui.set_visible(self.field.node(), editing);
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
/// A component on an entity of the rig world, so a handler can reach it by
/// `This<HierarchyPanel>`. What it shows is the *document's* world, by id.
#[derive(Clone)]
pub(crate) struct HierarchyPanel {
    pub(crate) view: TreeView<NameRow, EntityRef>,
    /// The world it shows, which is also what tags the rows it puts in
    /// flight: a slot index from another document is a different entity.
    world: u64,
    pub(crate) selected: Option<u64>,
    /// The entity being renamed, by id rather than by row — the pool
    /// recycles rows, and scrolling must not rename whatever moves in.
    editing: Option<u64>,
    count: Label,
    add: Button,
    remove: Button,
    /// Where a queued spawn leaves the entity it made, since the builder that
    /// knows the id runs at the frame boundary rather than at the click.
    pub(crate) spawned: Arc<AtomicU64>,
    /// A spawn or destroy is queued. It lands at the next frame boundary, so
    /// the re-walk it needs is a frame after the edit was asked for.
    pub(crate) queued: bool,
}

/// Not authored content, and nothing to sweep: the chrome drives it.
impl Export for HierarchyPanel {}
impl Component for HierarchyPanel {
    const HAS_UPDATE: bool = false;
}

/// The row menu. The root's is `[..1]` of it, so the two cannot disagree
/// about what "new child" is called.
const ROW_MENU: [&str; 3] = ["new child", "rename", "delete"];

/// `spawned` holding no entity. Ids are `u32`, so the top of the range is free.
pub(crate) const NO_SPAWN: u64 = u64::MAX;

impl HierarchyPanel {
    /// Built into a dock pane, so it takes whatever box the user has dragged
    /// its panel to rather than a size of its own.
    pub(crate) fn new(ui: &mut UiCore, pane: NodeId, world: u64) -> Self {
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
    pub(crate) fn invalidate(&mut self) {
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

    /// Queue a child that instantiates `scene`. The file is referenced, not
    /// copied in: what it says is read again every time this document opens,
    /// and the document stores only what it changes about it.
    fn instance_child(&mut self, document: &World, parent: u64, scene: PathBuf) {
        let id = engine::scene_asset::request_scene(&scene);
        let spawned = self.spawned.clone();
        document.spawn(
            _Transform {
                name: stem(&scene),
                parent: Some(parent as u32),
                .._Transform::default()
            },
            move |mut e| {
                e.add_component(SceneInstance::new(id));
                spawned.store(e.id().id as u64, Ordering::Relaxed);
            },
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

    pub(crate) fn update(&mut self, ui: &mut UiCore, document: &World) {
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

        // A scene dragged out of the browser becomes a child of the row it
        // lands on, as an instance rather than a copy. The highlight and the
        // drop ask different questions and cannot share an answer: a grab is
        // cleared on the frame it lands, which is the very frame the row it
        // landed on hears about it.
        let scene = |d: &AssetDrag| d.kind == Some(AssetKind::Scene);
        let over =
            (self.view.hovered(ui)).filter(|_| ui.dragging::<AssetDrag>().is_some_and(scene));
        let landed = (self.view.dropped_on::<AssetDrag>(ui))
            .filter(|(_, d)| scene(d))
            .map(|(id, d)| (id, d.path.clone()));
        if let Some((id, path)) = landed {
            self.instance_child(document, id, path);
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
                r.set_selected(ui, selected == Some(id) || over == Some(id));
            },
        );

        // Minus the hierarchy root, which is structure rather than content.
        let text = format!("{} entities", h.active_len() - 1);
        self.count.set_text(ui, &text);
    }
}
