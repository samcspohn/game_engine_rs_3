//! The scene hierarchy panel: a collapsible tree over the live
//! `TransformHierarchy` (ADR-0008 / ADR-0009).

use crate::browser::AssetDrag;
use crate::fill;
use crate::project::stem;
use engine::inherit::SceneInstance;
use engine::{
    AssetKind, Entity, EntityRef, WorldId,
    transform::{_Transform, ROOT},
    ui::{
        ButtonStyle, Event, Events, Keep, Label, MenuStyle, NodeId, RowContent, RowStyle,
        ScrollbarStyle, Source, TextField, TextFieldStyle, Tree, UiCore, UiRef, UiStyle, defer,
        style::{AlignItems, Display, Size, Style, px, zero},
        theme,
    },
};
use std::path::PathBuf;

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

    /// Swap which half of the row is in layout.
    fn set_editing(&self, ui: &mut UiCore, editing: bool) {
        ui.set_visible(self.label.node(), !editing);
        ui.set_visible(self.field.node(), editing);
    }
}

/// `world`'s hierarchy, read live. Holds the id, not the world: a stopped
/// play world must not be kept alive by the tree that showed it.
fn source(world: WorldId) -> Source<NameRow> {
    let children = move |id: u64, out: &mut Vec<u64>| {
        if let Some(w) = engine::worlds::world(world) {
            out.extend(w.hierarchy().children(id as u32).iter().map(|&c| c as u64));
        }
    };
    let bind = move |ui: &mut UiCore, r: engine::ui::Row<'_, NameRow>, id: u64| {
        if let Some(w) = engine::worlds::world(world) {
            r.label.set_text(ui, &row_text(w.hierarchy(), id));
        }
    };
    Source::new(children, bind).tagged(world as u64)
}

/// The scene hierarchy, as a collapsible tree over the live
/// `TransformHierarchy` (ADR-0008 / ADR-0009). Kept on its tree's gutter
/// (`UiCore::own`), so it lives and dies with the panel; what it shows is a
/// world, by id.
pub(crate) struct HierarchyPanel {
    pub(crate) tree: Tree<NameRow, EntityRef>,
    world: WorldId,
    count: Label,
    me: Option<UiRef<HierarchyPanel>>,
}

/// From the frame boundary an edit landed at: re-walk the tree and select
/// what was spawned. Deferred, since the boundary holds the world mutably.
fn landed(me: Option<UiRef<HierarchyPanel>>, spawned: Option<u64>) {
    let Some(me) = me else { return };
    defer(move |ui| {
        me.with(ui, |h, ui| {
            h.tree.invalidate(ui);
            if let Some(id) = spawned {
                h.reveal(ui, id);
                h.tree.select(ui, Some(id));
            }
            h.count(ui);
        });
    });
}

impl HierarchyPanel {
    /// Built into a dock pane, so it takes whatever box the user has dragged
    /// its panel to rather than a size of its own.
    pub(crate) fn new(ui: &mut UiCore, pane: NodeId, world: WorldId) -> UiRef<Self> {
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
        let tree = Tree::new(
            ui,
            gutter,
            fill(),
            RowStyle::default(),
            ROOT as u64,
            source(world),
        );
        ui.set_background(tree.node(), UiStyle::fill(t.backdrop).radius(t.radius));
        ui.scrollbar(gutter, tree.node(), ScrollbarStyle::default());

        let panel = Self {
            tree,
            world,
            count,
            me: None,
        };
        let me = ui.own(gutter, panel);
        me.with(ui, |h, ui| {
            h.me = Some(me);
            h.count(ui);
        });

        // A child of the selection, so the tree is authored the way it is
        // read; dropping it on the root row is what un-parents it again.
        ui.on_click(add, move |ui| {
            me.with(ui, |h, ui| {
                let parent = h.tree.selected(ui);
                h.spawn_child(parent);
            });
        });
        ui.on_click(remove, move |ui| {
            me.with(ui, |h, ui| h.destroy_selected(ui));
        });
        tree.on_row_double_click(ui, move |ui, id| {
            me.with(ui, |h, ui| h.begin_rename(ui, id));
        });
        tree.on_row_right_click(ui, move |ui, id, at| {
            me.with(ui, |h, ui| h.row_menu(ui, id, at));
        });
        tree.on_pick_up(ui, move |ui, id| {
            me.with(ui, |h, ui| h.pick_up(ui, id));
        });
        tree.on_move(ui, move |ui, d| {
            me.with(ui, |h, _| h.moved(d));
        });
        // A scene dragged out of the browser becomes a child of the row it
        // lands on, as an instance rather than a copy.
        let scene = |d: &AssetDrag| d.kind == Some(AssetKind::Scene);
        tree.on_row_drop(ui, scene, move |ui, id, d: &AssetDrag| {
            let path = d.path.clone();
            me.with(ui, |h, _| h.instance_child(id, path));
        });
        me
    }

    fn world(&self) -> Option<engine::WorldHandle> {
        engine::worlds::world(self.world)
    }

    fn count(&self, ui: &mut UiCore) {
        let Some(w) = self.world() else { return };
        // Minus the hierarchy root, which is structure rather than content.
        let text = format!("{} entities", w.hierarchy().active_len() - 1);
        self.count.set_text(ui, &text);
    }

    fn reveal(&self, ui: &mut UiCore, id: u64) {
        if let Some(w) = self.world() {
            self.tree.reveal(ui, id, |n| parent_of(w.hierarchy(), n));
        }
    }

    /// Start showing `world` — the document's, or the one playing it.
    pub(crate) fn show(&mut self, ui: &mut UiCore, world: WorldId) {
        self.world = world;
        self.tree.show(ui, source(world));
        self.count(ui);
    }

    /// Rename `id` in place: Enter commits, anything that takes the keyboard
    /// away — a click elsewhere, Escape, the row scrolling out — cancels.
    fn begin_rename(&mut self, ui: &mut UiCore, id: u64) {
        let (Some(w), Some(row)) = (self.world(), self.tree.edit(ui, Some(id))) else {
            return;
        };
        let Some(target) = EntityRef::new(&w, Entity::new(id as u32)) else {
            return;
        };
        row.field.set_text(ui, &row_text(w.hierarchy(), id));
        row.field.focus(ui);
        let tree = self.tree;
        ui.on(row.field, Events::FOCUS, move |ui, ev| {
            if let Event::Submit(text) = ev {
                let name = text.trim();
                if !name.is_empty() {
                    target.with(|w, e| {
                        let h = w.hierarchy();
                        let t = h.get_transform_unchecked(e.id).lock();
                        h.set_name(&t, name);
                    });
                }
            }
            if tree.editing(ui) == Some(id) {
                tree.edit(ui, None);
            }
            Keep::No
        });
    }

    /// The row menu. The root can only be spawned into.
    fn row_menu(&mut self, ui: &mut UiCore, id: u64, at: [f32; 2]) {
        let me = self.me;
        let with = move |ui: &mut UiCore, f: fn(&mut HierarchyPanel, &mut UiCore, u64)| {
            if let Some(me) = me {
                me.with(ui, |h, ui| f(h, ui, id));
            }
        };
        let menu = ui
            .context_menu(at, MenuStyle::default())
            .item(ui, "new child", move |ui| {
                with(ui, |h, _, id| h.spawn_child(Some(id)))
            });
        if id != ROOT as u64 {
            menu.item(ui, "rename", move |ui| {
                with(ui, |h, ui, id| h.begin_rename(ui, id))
            })
            .item(ui, "delete", move |ui| {
                with(ui, |h, ui, _| h.destroy_selected(ui))
            });
        }
    }

    /// The tree offers the row; the panel grabs, because only it knows a row
    /// here names an entity.
    fn pick_up(&mut self, ui: &mut UiCore, id: u64) {
        let Some(w) = self.world() else { return };
        let Some(payload) = EntityRef::new(&w, Entity::new(id as u32)) else {
            return;
        };
        let text = row_text(w.hierarchy(), id);
        self.tree
            .grab(ui, payload, |ui, r| r.label.set_text(ui, &text));
    }

    /// A drag re-parents the scene, not a copy of it — the tree has moved
    /// its rows already.
    fn moved(&mut self, d: engine::ui::Dropped) {
        let Some(w) = self.world() else { return };
        let h = w.hierarchy();
        let Some(t) = h.get_transform(d.node as u32) else {
            return;
        };
        let t = t.lock();
        if t.get_parent() == Some(d.parent as u32) {
            // Same parent: ordering only, which is not a parent change to push
            // down the parent stream.
            h.move_child(&t, d.at);
        } else {
            h.set_parent_at(&t, Some(d.parent as u32), d.at);
        }
    }
    /// Queue a spawn under `parent`, or under the root when there is none.
    fn spawn_child(&mut self, parent: Option<u64>) {
        let Some(w) = self.world() else { return };
        let me = self.me;
        w.spawn(
            _Transform {
                name: "entity".into(),
                parent: Some(parent.unwrap_or(ROOT as u64) as u32),
                .._Transform::default()
            },
            move |e| landed(me, Some(e.id().id as u64)),
        );
    }

    /// Queue a child that instantiates `scene`. The file is referenced, not
    /// copied in: what it says is read again every time this document opens,
    /// and the document stores only what it changes about it.
    fn instance_child(&mut self, parent: u64, scene: PathBuf) {
        let Some(w) = self.world() else { return };
        let id = engine::scene_asset::request_scene(&scene);
        let me = self.me;
        w.spawn(
            _Transform {
                name: stem(&scene),
                parent: Some(parent as u32),
                .._Transform::default()
            },
            move |mut e| {
                e.add_component(SceneInstance::new(id));
                landed(me, Some(e.id().id as u64));
            },
        );
    }

    /// Queue a destroy of the selection, if there is one to destroy. The root
    /// is structure rather than content, so it is never one.
    fn destroy_selected(&mut self, ui: &mut UiCore) {
        let Some(id) = self.tree.selected(ui).filter(|&id| id != ROOT as u64) else {
            return;
        };
        let Some(w) = self.world() else { return };
        w.destroy(Entity::new(id as u32));
        let me = self.me;
        w.then(move || landed(me, None));
        self.tree.select(ui, None);
    }

    /// The Delete key. Only the panel under the pointer answers: every
    /// document holds a selection of its own.
    pub(crate) fn delete_key(&mut self, ui: &mut UiCore) {
        let p: [f32; 2] = engine::input::cursor_position().into();
        let r = ui.node_rect(self.tree.node());
        if (0..2).all(|k| p[k] >= r[k] && p[k] < r[k] + r[k + 2]) {
            self.destroy_selected(ui);
        }
    }
}
