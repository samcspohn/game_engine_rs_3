//! Game-side UI demo — the worked example of ADR-0008's access pattern.
//!
//! This is an ordinary game component with no privileges. It reaches the UI
//! the way any game does: `engine::ui::ui()` locks the global store. None of
//! it lives in the renderer, and a game that doesn't want an overlay simply
//! doesn't attach one.
//!
//! Nothing here computes a coordinate. The panel is a flex column with
//! padding and a gap; the swatch strip is a five-track grid; both size
//! themselves from their content.
//!
//! It also makes the upload behaviour observable: the panel and the specimen
//! lines are written once and then never again, and only the readout changes
//! — at 10 Hz, not per frame. So `ENGINE_UI_TRACE=1` stays silent on the
//! large majority of frames even though the UI is on screen the whole time,
//! which is the single claim ADR-0006 phase 1 asks to be proven.
//!
//! Toggle with **F6**.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use engine::input;
use engine::stats;
use engine::transform::Transform;
use engine::ui::style::{
    evenly_sized_tracks, percent, px, zero, AlignItems, Display, FlexDirection,
    LengthPercentageAuto, Position, Rect, Size, Style, TaffyAuto,
};
use engine::ui::{
    rgb, rgba, set_theme, theme, ui, Button, ButtonStyle, Checkbox, CheckboxStyle, Label, NodeId, Slider, SliderStyle, DragNode, RowStyle, Theme, TreeView, UiStyle,
};
use engine::{Component, KeyCode};

const PAD: f32 = 12.0;
const GAP: f32 = 5.0;
const TITLE_PX: f32 = 14.0;

/// This overlay's palette: the built-in dark theme with the game's own
/// identity colour, which is the whole reason `accent` is a role rather
/// than a constant — the editor's chrome is green, this is blue, and every
/// widget below picks its colours up without knowing either.
const DEMO_THEME: Theme = Theme {
    accent: rgb(0x6C, 0xC4, 0xFF),
    outline: rgba(0x6C, 0xC4, 0xFF, 0x60),
    ..Theme::DARK
};

/// The printable ASCII the built-in font covers, split so the panel sizes
/// itself to the widest line. Written once — a permanent check that the
/// atlas is complete and correctly addressed.
const SPECIMEN: [&str; 3] = [
    " !\"#$%&'()*+,-./0123456789:;<=>?@",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`",
    "abcdefghijklmnopqrstuvwxyz{|}~",
];

const SWATCHES: [u32; 5] = [
    rgb(0xE0, 0x6C, 0x75),
    rgb(0xE5, 0xC0, 0x7B),
    rgb(0x98, 0xC3, 0x79),
    rgb(0x61, 0xAF, 0xEF),
    rgb(0xC6, 0x78, 0xDD),
];

/// Readout refresh interval. A stats line in a 10 000 FPS engine does ten
/// text relayouts per second, not ten thousand — a human reads it at 10 Hz
/// either way. This is ADR-0006's tick set in miniature, hard-coded until
/// the callback layer provides the real one.
const READOUT_HZ: Duration = Duration::from_millis(100);

/// Attach to any entity; the transform is unused. The UI store is global, so
/// this holds only the handles it needs to mutate.
#[derive(Clone)]
pub struct UiDemo {
    panel: NodeId,
    readout: Label,
    button: Button,
    counter: Label,
    tree: TreeView,
    hierarchy: Hierarchy,
    selection: Label,
    /// By id, never by row: collapsing anything above a selected row changes
    /// its index but not what is selected.
    selected: Option<u64>,
    clicks: u32,
    /// No `highlighted: bool` beside it, and no `faded: f32` beside the
    /// slider — the controls hold their own values, so there is no second
    /// copy here to keep in step.
    highlight: Checkbox,
    fade: Slider,
    fade_label: Label,
    last_readout: Instant,
    visible: bool,
}

/// Stand-in for a scene graph: `id -> children`, which is the shape
/// `TransformHierarchy` already has and the shape `TreeView` reads through a
/// closure. Drag-to-reparent edits **this**, and the view is told afterwards
/// — the view never owns the structure, so there is nothing to drift.
///
/// `Clone` only because `Component` requires it.
#[derive(Clone)]
struct Hierarchy(HashMap<u64, Vec<u64>>);

impl Hierarchy {
    /// Eight groups under the scene root, eight entities in each.
    fn demo() -> Self {
        let mut m = HashMap::from([(0, (1..=8).collect::<Vec<u64>>())]);
        for g in 1..=8u64 {
            m.insert(g, (1..=8).map(|i| g * 100 + i).collect());
        }
        Self(m)
    }

    /// `at` counts `parent`'s children once `node` has left its old parent,
    /// which is the order `Dropped` reports — so this removes first and the
    /// index needs no adjusting.
    fn reparent(&mut self, node: u64, parent: u64, at: usize) {
        for kids in self.0.values_mut() {
            kids.retain(|&k| k != node);
        }
        self.0.entry(parent).or_default().insert(at, node);
    }
}

fn name(id: u64) -> String {
    match id {
        0 => "scene".into(),
        g if g < 100 => format!("group {g}"),
        e => format!("entity {}.{}", e / 100, e % 100),
    }
}

impl Default for UiDemo {
    fn default() -> Self {
        Self::new()
    }
}

impl UiDemo {
    /// Builds the tree. `UiCore` owns no Vulkan and the global is created on
    /// first touch, so this works before the window exists — the first
    /// `run_layout` positions everything.
    pub fn new() -> Self {
        // Before any widget style is constructed — `Default` resolves the
        // palette at that moment, not later.
        set_theme(DEMO_THEME);
        let t = theme();

        let mut ui = ui();
        let screen = ui.root();

        // Absolutely positioned so the overlay doesn't participate in the
        // root's flow; `auto` width/height means it shrink-wraps its
        // content, so the panel resizes itself when a line gets longer.
        let panel = ui.node(
            screen,
            Style {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                position: Position::Absolute,
                inset: Rect {
                    left: px(PAD),
                    top: px(PAD),
                    right: LengthPercentageAuto::AUTO,
                    bottom: LengthPercentageAuto::AUTO,
                },
                padding: Rect::length(PAD),
                gap: Size {
                    width: zero(),
                    height: px(GAP),
                },
                align_items: Some(AlignItems::STRETCH),
                ..Default::default()
            },
        );
        ui.set_background(
            panel,
            UiStyle::fill(t.panel).border(t.outline, 1.0).radius(8.0),
        );

        ui.label(panel, TITLE_PX, t.text, "retained ui / ADR-0006");
        let readout = ui.label(panel, t.text_px, t.accent, "");
        for line in SPECIMEN {
            ui.label(panel, t.text_px, t.text_dim, line);
        }

        // Five equal grid tracks stretched across the panel's content box —
        // the swatches never learn their own width.
        let strip = ui.node(
            panel,
            Style {
                display: Display::Grid,
                grid_template_columns: evenly_sized_tracks(SWATCHES.len() as u16),
                gap: Size {
                    width: px(GAP),
                    height: zero(),
                },
                size: Size {
                    width: percent(1.0_f32),
                    height: px(8.0),
                },
                ..Default::default()
            },
        );
        for color in SWATCHES {
            let cell = ui.node(strip, Style::default());
            ui.set_background(cell, UiStyle::fill(color).radius(2.0));
        }

        // The button owns its own hover/press appearance — `ButtonStyle`
        // carries the three fills and the engine applies them on transition,
        // so nothing about looks appears in `update` below.
        let button = ui.button(panel, "click me", ButtonStyle::default());
        let counter = ui.label(panel, t.text_px, t.text_dim, "clicks: 0");

        // The checkbox owns this value. Clicking it is handled entirely by
        // the engine; **F5** writes the same value from outside without
        // going near the pointer, which is why `set_checked` still exists.
        let highlight = ui.checkbox(panel, "highlight readout (F5)", CheckboxStyle::default());
        highlight.set_checked(&mut ui, true);

        // Same contract over an `f32`. Seeded here rather than tracked in a
        // field: after this line the slider is the only copy.
        let fade = ui.slider(panel, SliderStyle::default());
        fade.set_value(&mut ui, 1.0);
        let fade_label = ui.label(panel, t.text_px, t.text_dim, "panel opacity 100%");

        // A virtualized hierarchy in a 108 px viewport: only enough row nodes
        // to cover it exist, and scrolling one row recycles one of them.
        // Wheel over it, click the arrows, and **drag a row onto another to
        // re-parent it** — near a row's middle to drop inside, near an edge
        // to drop beside.
        let row_style = RowStyle::default();
        let tree = TreeView::new(
            &mut ui,
            panel,
            Style {
                size: Size {
                    width: percent(1.0_f32),
                    height: px(108.0),
                },
                ..Default::default()
            },
            row_style,
            0,
        );
        ui.set_background(tree.node(), UiStyle::fill(t.backdrop).radius(t.radius));
        let selection = ui.label(panel, t.text_px, t.text_dim, "nothing selected");

        Self {
            panel,
            readout,
            button,
            counter,
            tree,
            hierarchy: Hierarchy::demo(),
            selection,
            selected: None,
            clicks: 0,
            highlight,
            fade,
            fade_label,
            last_readout: Instant::now() - READOUT_HZ,
            visible: true,
        }
    }

    /// Hide by taking the panel out of layout entirely — its boxes collapse
    /// to zero and `ui.vert`'s zero-area early-out culls them. A per-panel
    /// `ui_group` would make this one record instead of one per primitive;
    /// groups earn that when docking gives every panel its own.
    fn toggle(&mut self) {
        self.visible = !self.visible;
        let mut ui = ui();
        let mut style = ui.node_style(self.panel);
        style.display = if self.visible {
            Display::Flex
        } else {
            Display::None
        };
        ui.set_node_style(self.panel, style);
    }
}

impl Component for UiDemo {
    /// Almost always a key check and a clock read. The renderer runs
    /// `run_layout` after every component has had its turn, so this never
    /// calls it.
    fn update(&mut self, _dt: f32, _transform: &Transform) {
        if input::key_pressed(KeyCode::F6) {
            self.toggle();
        }

        // One guard for the whole body — `ui()` is a plain `Mutex`, so
        // nesting two calls in one expression would deadlock.
        let mut ui = ui();

        // Nothing here handles the click: `update_pointer` already toggled
        // the checkbox and moved the slider. **F5** is the other direction —
        // the value changing with no pointer anywhere near it.
        if input::key_pressed(KeyCode::F5) {
            let flipped = !self.highlight.checked(&ui);
            self.highlight.set_checked(&mut ui, flipped);
        }
        let t = theme();
        let highlighted = self.highlight.checked(&ui);
        self.readout
            .set_color(&mut ui, if highlighted { t.accent } else { t.text_dim });

        if ui.drag(self.fade).is_some() {
            let text = format!("panel opacity {:.0}%", self.fade.value(&ui) * 100.0);
            self.fade_label.set_text(&mut ui, &text);
        }

        if ui.clicked(self.button) {
            self.clicks += 1;
            let text = format!("clicks: {}", self.clicks);
            self.counter.set_text(&mut ui, &text);
        }

        if let Some(id) = self.tree.clicked(&ui) {
            self.selected = Some(id);
            let text = format!("selected {}", name(id));
            self.selection.set_text(&mut ui, &text);
        }

        // Re-bound every frame on purpose: the pool is viewport-sized and
        // every write goes through the equality gate, so a still list uploads
        // nothing and no dirty-flag bookkeeping is needed here. `depth` and
        // `expanded` are the view's to fill in — it knows the shape, this
        // closure only knows how a node looks.
        let (h, selected) = (&self.hierarchy, self.selected);
        // The view reports the pick-up; the game grabs, because only it knows
        // that a row here means a group or an entity.
        if let Some(id) = self.tree.picked_up(&ui) {
            self.tree
                .grab(&mut ui, DragNode(id), |ui, r| r.set_text(ui, &name(id)));
        }

        self.tree.sync(
            &mut ui,
            |id, out| out.extend(h.0.get(&id).into_iter().flatten().copied()),
            |ui, r, id| {
                r.set_text(ui, &name(id));
                r.set_selected(ui, selected == Some(id));
            },
        );

        // The hierarchy moves first and the view is told after. Ignoring the
        // drop instead is how a caller refuses a move its own rules forbid —
        // the view holds no structure of its own to have got ahead.
        if let Some(d) = self.tree.dropped() {
            self.hierarchy.reparent(d.node, d.parent, d.at);
            self.tree.moved(d.node, d.parent, d.at);
        }

        if self.last_readout.elapsed() < READOUT_HZ {
            return;
        }
        self.last_readout = Instant::now();

        let screen = stats::screen();
        let prims = ui.prim_count();
        let text = format!(
            "{:.0} fps   {prims} primitives   {}x{}",
            stats::fps(),
            screen[0] as u32,
            screen[1] as u32
        );
        self.readout.set_text(&mut ui, &text);
    }
}
