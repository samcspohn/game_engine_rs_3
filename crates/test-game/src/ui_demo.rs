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

use std::time::{Duration, Instant};

use engine::input;
use engine::stats;
use engine::transform::Transform;
use engine::ui::style::{
    evenly_sized_tracks, percent, px, zero, AlignItems, Display, FlexDirection,
    LengthPercentageAuto, Position, Rect, Size, Style, TaffyAuto,
};
use engine::ui::{rgb, rgba, ui, ButtonStyle, NodeId, Row, RowList, RowStyle, UiStyle};
use engine::{Component, KeyCode};

const PAD: f32 = 12.0;
const GAP: f32 = 5.0;
const TITLE_PX: f32 = 14.0;
const BODY_PX: f32 = 11.0;

const INK: u32 = rgb(0xE6, 0xE9, 0xEF);
const DIM: u32 = rgb(0x8A, 0x93, 0xA6);
const ACCENT: u32 = rgb(0x6C, 0xC4, 0xFF);

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
    readout: NodeId,
    button: NodeId,
    counter: NodeId,
    list: RowList,
    /// Stand-in for a scene graph flattened to `(depth, name)` — which is the
    /// shape `RowList` wants and the shape a hierarchy panel will produce.
    tree: Vec<(u16, String)>,
    selection: NodeId,
    selected: Option<usize>,
    clicks: u32,
    last_readout: Instant,
    visible: bool,
}

/// A plausible scene tree: roots every 16 rows, each with children and
/// grandchildren, so indentation is visible while scrolling.
fn fake_hierarchy(n: usize) -> Vec<(u16, String)> {
    (0..n)
        .map(|i| match i % 16 {
            0 => (0, format!("root {}", i / 16)),
            k if k % 4 == 1 => (1, format!("group {k}")),
            k => (2, format!("entity {i}.{k}")),
        })
        .collect()
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
            UiStyle::fill(rgba(0x14, 0x18, 0x22, 0xE0))
                .border(rgba(0x6C, 0xC4, 0xFF, 0x60), 1.0)
                .radius(8.0),
        );

        ui.label(panel, TITLE_PX, INK, "retained ui / ADR-0006");
        let readout = ui.label(panel, BODY_PX, ACCENT, "");
        for line in SPECIMEN {
            ui.label(panel, BODY_PX, DIM, line);
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
        let counter = ui.label(panel, BODY_PX, DIM, "clicks: 0");

        // A virtualized list of 5 000 rows in a 108 px viewport. Six row
        // nodes exist; scrolling a row recycles one of them. Wheel over it.
        let tree = fake_hierarchy(5_000);
        let list = RowList::new(
            &mut ui,
            panel,
            Style {
                size: Size {
                    width: percent(1.0_f32),
                    height: px(108.0),
                },
                ..Default::default()
            },
            RowStyle::default(),
        );
        ui.set_background(
            list.node(),
            UiStyle::fill(rgba(0x0B, 0x0E, 0x16, 0xFF)).radius(3.0),
        );
        let selection = ui.label(panel, BODY_PX, DIM, "nothing selected");

        Self {
            panel,
            readout,
            button,
            counter,
            list,
            tree,
            selection,
            selected: None,
            clicks: 0,
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

        if ui.clicked(self.button) {
            self.clicks += 1;
            let text = format!("clicks: {}", self.clicks);
            ui.set_label(self.counter, &text);
        }

        if let Some(i) = self.list.clicked(&ui) {
            self.selected = Some(i);
            let text = format!("selected {}: {}", i, self.tree[i].1);
            ui.set_label(self.selection, &text);
        }

        // Re-bound every frame on purpose: the pool is viewport-sized and
        // every write goes through the equality gate, so a still list uploads
        // nothing and no dirty-flag bookkeeping is needed here.
        let (tree, selected) = (&self.tree, self.selected);
        self.list.sync(&mut ui, tree.len(), |i| Row {
            text: tree[i].1.as_str().into(),
            depth: tree[i].0,
            selected: selected == Some(i),
        });

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
        ui.set_label(self.readout, &text);
    }
}
