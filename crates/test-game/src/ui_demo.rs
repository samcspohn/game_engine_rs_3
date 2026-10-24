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
//! Toggle with **F4**. (Not F6 — the renderer owns that one for the
//! staging-memory cycle, and two handlers on one key is one handler too
//! many.)

use std::collections::HashMap;
use std::sync::Arc;

use engine::input;
use engine::stats;
use engine::transform::Transform;
use engine::ui::style::{
    AlignItems, Display, FlexDirection, LengthPercentageAuto, Position, Rect, Size, Style,
    TaffyAuto, evenly_sized_tracks, percent, px, zero,
};
use engine::ui::{
    ButtonStyle, Checkbox, CheckboxStyle, DragNode, Label, NodeId, ProgressStyle, RadioStyle, Row,
    RowStyle, ScrollbarStyle, SliderStyle, Source, TabStyle, TextFieldStyle, Theme, Tree, UiCore,
    UiStyle, Value, rgb, rgba, set_theme, theme, ui,
};
use engine::{Component, Export, KeyCode};
use std::sync::Mutex;

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

/// Attach to any entity; the transform is unused. Everything the panel does
/// is a listener registered in `new` — this holds only what the F4 and F5
/// keys reach.
#[derive(Clone)]
pub struct UiDemo {
    panel: NodeId,
    readout: Label,
    /// No `highlighted: bool` beside it — the control holds its own value.
    highlight: Checkbox,
    visible: bool,
}

/// What the tree shows and the rename field edits, shared by the listeners
/// that read and write it.
type Shared = Arc<Mutex<(Hierarchy, HashMap<u64, String>)>>;

/// Stand-in for a scene graph: `id -> children`, which is the shape
/// `TransformHierarchy` already has and the shape `Tree` reads through a
/// closure — the tree never owns the structure, so there is nothing to drift.
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

/// A row's label: whatever it has been renamed to, else the generated name.
fn name_of(names: &HashMap<u64, String>, id: u64) -> String {
    if let Some(n) = names.get(&id) {
        return n.clone();
    }
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

        // Everything below is filed under a tab. The panes are ordinary
        // containers — build into them like any node — and the two that are
        // closed have no box at all, so a control the pointer cannot reach
        // still holds its value. `detail` below proves it: the readout keeps
        // obeying a radio group that is not on screen.
        //
        // Width pinned so switching tabs does not resize a panel that
        // otherwise shrink-wraps its widest line.
        let tabs = ui.tabs(panel, &["controls", "scene", "font"], TabStyle::default());
        let (controls, scene, font) = (tabs.pane(&ui, 0), tabs.pane(&ui, 1), tabs.pane(&ui, 2));
        for pane in [controls, scene, font] {
            // Read-modify-write, because `display` is where "closed" lives —
            // a fresh `Style` here would open all three at once.
            let mut s = ui.node_style(pane);
            s.gap = Size {
                width: zero(),
                height: px(GAP),
            };
            s.size.width = px(268.0);
            ui.set_node_style(pane, s);
        }

        for line in SPECIMEN {
            ui.label(font, t.text_px, t.text_dim, line);
        }

        // Five equal grid tracks stretched across the pane's content box —
        // the swatches never learn their own width.
        let strip = ui.node(
            font,
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

        // The button owns its own hover/press appearance; the count lives on
        // the label that shows it.
        let button = ui.button(controls, "click me", ButtonStyle::default());
        let counter = ui.label(controls, t.text_px, t.text_dim, "clicks: 0");
        let clicks = ui.own(counter, 0u32);
        ui.on_click(button, move |ui| {
            clicks.with(ui, |n, ui| {
                *n += 1;
                counter.set_text(ui, &format!("clicks: {n}"));
            });
        });

        // The checkbox owns this value. **F5** writes the same value from
        // outside without going near the pointer.
        let highlight = ui.checkbox(controls, "highlight readout (F5)", CheckboxStyle::default());
        highlight.set_checked(&mut ui, true);
        let readout_ = readout;
        ui.on_change(highlight, move |ui, v| {
            if let Value::Bool(on) = v {
                lit(ui, readout_, *on);
            }
        });

        // A slider mirrored by a progress bar, through a lock-free paint
        // write rather than a relayout.
        let fade = ui.slider(controls, SliderStyle::default());
        fade.set_value(&mut ui, 1.0);
        let fade_label = ui.label(controls, t.text_px, t.text_dim, "panel opacity 100%");
        let meter = ui.progress(controls, 1.0, ProgressStyle::default());
        ui.on_change(fade, move |ui, v| {
            if let Value::F32(v) = v {
                fade_label.set_text(ui, &format!("panel opacity {:.0}%", v * 100.0));
                meter.set(engine::ui::paint(), *v);
            }
        });

        // The first control whose value is not on the node you click: the
        // group owns it, and the readout below reads it.
        ui.label(controls, t.text_px, t.text_dim, "readout detail");
        let detail = ui.radio_group(
            controls,
            &["fps", "+ primitives", "+ resolution"],
            RadioStyle::default(),
        );
        detail.set_selected(&mut ui, 2);

        // A virtualized hierarchy in a 108 px viewport: only enough row nodes
        // to cover it exist, and scrolling one row recycles one of them.
        // Wheel over it, click the arrows, and **drag a row onto another to
        // re-parent it** — near a row's middle to drop inside, near an edge
        // to drop beside.
        let model: Shared = Arc::new(Mutex::new((Hierarchy::demo(), HashMap::new())));
        let gutter = ui.node(
            scene,
            Style {
                display: Display::Flex,
                gap: Size {
                    width: px(3.0),
                    height: zero(),
                },
                ..Default::default()
            },
        );
        let (a, b) = (model.clone(), model.clone());
        let source = Source::new(
            move |id, out| {
                out.extend(
                    a.lock()
                        .unwrap()
                        .0
                        .0
                        .get(&id)
                        .into_iter()
                        .flatten()
                        .copied(),
                )
            },
            move |ui, r: Row<'_, Label>, id| r.set_text(ui, &name_of(&b.lock().unwrap().1, id)),
        );
        let viewport = Style {
            flex_grow: 1.0,
            flex_basis: px(0.0),
            size: Size {
                width: TaffyAuto::AUTO,
                height: px(108.0),
            },
            ..Default::default()
        };
        let tree: Tree = Tree::new(&mut ui, gutter, viewport, RowStyle::default(), 0, source);
        ui.set_background(tree.node(), UiStyle::fill(t.backdrop).radius(t.radius));
        // The bar owns nothing: the tree's offset and extent are its state.
        ui.scrollbar(gutter, tree.node(), ScrollbarStyle::default());
        let selection = ui.label(scene, t.text_px, t.text_dim, "nothing selected");

        // Click a row, type, press **Enter**. Tab reaches it from anywhere,
        // Escape gives the keyboard back to the game.
        let rename = ui.text_field(scene, "", TextFieldStyle::default());
        rename.set_hint(&mut ui, "rename row (Enter)");

        let m = model.clone();
        tree.on_select(&mut ui, move |ui, id| {
            let Some(id) = id else { return };
            let name = name_of(&m.lock().unwrap().1, id);
            selection.set_text(ui, &format!("selected {name}"));
            rename.set_text(ui, &name);
        });
        let m = model.clone();
        ui.on_submit(rename, move |ui, typed| {
            let Some(id) = tree.selected(ui) else { return };
            let name = {
                let names = &mut m.lock().unwrap().1;
                match typed.trim() {
                    "" => names.remove(&id),
                    t => names.insert(id, t.to_string()),
                };
                name_of(names, id)
            };
            selection.set_text(ui, &format!("selected {name}"));
            tree.invalidate(ui);
        });
        // The tree offers the row; the game grabs, because only it knows
        // that a row here means a group or an entity.
        let m = model.clone();
        tree.on_pick_up(&mut ui, move |ui, id| {
            let label = name_of(&m.lock().unwrap().1, id);
            tree.grab(ui, DragNode(id), |ui, r| r.set_text(ui, &label));
        });
        // The tree has moved its rows; this moves the model.
        tree.on_move(&mut ui, move |_, d| {
            model.lock().unwrap().0.reparent(d.node, d.parent, d.at)
        });

        // A stats line a human reads at 10 Hz, not once a frame.
        ui.every(panel, 10.0, move |ui, _| {
            let screen = stats::screen();
            let prims = ui.prim_count();
            let fps = stats::fps();
            let text = match detail.selected(ui) {
                0 => format!("{fps:.0} fps"),
                1 => format!("{fps:.0} fps   {prims} primitives"),
                _ => format!(
                    "{fps:.0} fps   {prims} primitives   {}x{}",
                    screen[0] as u32, screen[1] as u32
                ),
            };
            readout_.set_text(ui, &text);
        });

        Self {
            panel,
            readout,
            highlight,
            visible: true,
        }
    }

    /// Hide by taking the panel out of layout entirely — its boxes collapse
    /// to zero and `ui.vert`'s zero-area early-out culls them.
    fn toggle(&mut self, ui: &mut UiCore) {
        self.visible = !self.visible;
        ui.set_visible(self.panel, self.visible);
    }
}

fn lit(ui: &mut UiCore, readout: Label, on: bool) {
    let t = theme();
    readout.set_color(ui, if on { t.accent } else { t.text_dim });
}

impl Export for UiDemo {}

impl Component for UiDemo {
    /// Two hotkeys; everything else is a listener.
    fn update(&mut self, _dt: f32, _transform: &Transform, _w: &engine::World) {
        let mut ui = ui();
        // The keyboard twin of `OrbitController`'s `pointer_captured` check:
        // what every hotkey bound to a *letter* needs.
        let typing = ui.keyboard_captured();
        if input::key_pressed(KeyCode::F4) && !typing {
            self.toggle(&mut ui);
        }
        // The value changing with no pointer anywhere near it.
        if input::key_pressed(KeyCode::F5) && !typing {
            let flipped = !self.highlight.checked(&ui);
            self.highlight.set_checked(&mut ui, flipped);
            lit(&mut ui, self.readout, flipped);
        }
    }
}
