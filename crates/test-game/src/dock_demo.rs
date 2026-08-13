//! Docking demo — drag the tabs around. **F3** toggles it.
//!
//! Three panels in a dock the user rearranges: grab a tab header and drop it
//! on another pane's edge to split it, or on its middle to join its strip;
//! drag the line between two panes to resize them. Empty a pane and its
//! split folds away, taking its line with it.
//!
//! The claim worth watching is in *inspector*: type into its field, tick its
//! checkbox, move its slider, then drag the tab somewhere else. Everything
//! is still there, because the panel was **moved** rather than rebuilt —
//! `set_parent` re-homes the live subtree and taffy does the rest.

use std::time::{Duration, Instant};

use engine::input;
use engine::stats;
use engine::transform::Transform;
use engine::ui::style::{
    px, zero, Display, FlexDirection, LengthPercentageAuto, Position, Rect, Size, Style, TaffyAuto,
};
use engine::ui::{
    theme, ui, Checkbox, CheckboxStyle, DockSpace, DockStyle, Label, NodeId, PanelId, Side, Slider,
    SliderStyle, TextFieldStyle, UiStyle,
};
use engine::{Component, Export, KeyCode};

const MARGIN: f32 = 12.0;
const PAD: f32 = 6.0;
const SIZE: [f32; 2] = [460.0, 320.0];

/// Same 10 Hz as the other overlay: a stats line a human reads does not need
/// a relayout per frame.
const READOUT_HZ: Duration = Duration::from_millis(100);

#[derive(Clone)]
pub struct DockDemo {
    frame: NodeId,
    dock: DockSpace,
    highlight: Checkbox,
    fade: Slider,
    stats: Label,
    state: Label,
    /// Whose contents get restated only while it is the open tab.
    profiler: PanelId,
    last: Instant,
    visible: bool,
}

impl Default for DockDemo {
    fn default() -> Self {
        Self::new()
    }
}

impl DockDemo {
    pub fn new() -> Self {
        let t = theme();
        let mut ui = ui();
        let screen = ui.root();

        // Top right, out of the F4 overlay's way.
        let frame = ui.node(
            screen,
            Style {
                display: Display::Flex,
                position: Position::Absolute,
                inset: Rect {
                    left: LengthPercentageAuto::AUTO,
                    top: px(MARGIN),
                    right: px(MARGIN),
                    bottom: LengthPercentageAuto::AUTO,
                },
                size: Size {
                    width: px(SIZE[0]),
                    height: px(SIZE[1]),
                },
                padding: Rect::length(PAD),
                ..Default::default()
            },
        );
        ui.set_background(frame, UiStyle::fill(t.panel).border(t.outline, 1.0).radius(8.0));

        let mut dock = DockSpace::new(&mut ui, frame, fill(), DockStyle::default());
        let (outliner, inspector, profiler) = (
            dock.panel(&mut ui, "outliner"),
            dock.panel(&mut ui, "inspector"),
            dock.panel(&mut ui, "profiler"),
        );
        // Two of the three moves a drop can make, done from code: the same
        // call, so the layout an app ships with and the one a user drags into
        // are the same kind of thing.
        dock.dock(&mut ui, inspector, outliner, Side::Right);
        dock.dock(&mut ui, profiler, inspector, Side::Bottom);

        for (id, line) in [(outliner, "drag a tab, or the line between panes"), (profiler, "")] {
            let pane = dock.content(id);
            ui.label(pane, t.text_px, t.text_dim, line);
        }
        for name in ["camera", "directional light", "cube", "floor"] {
            ui.label(dock.content(outliner), t.text_px, t.text, name);
        }

        // The controls the move has to survive. None of them is restated
        // below — each owns its value, and `set_parent` keeps the node.
        let pane = dock.content(inspector);
        // Deliberately not kept: nothing in this component ever names the
        // field again, and after a drag its text is still whatever was typed.
        ui.text_field(pane, "", TextFieldStyle::default())
            .set_hint(&mut ui, "type, then drag the tab");
        let highlight = ui.checkbox(pane, "cast shadows", CheckboxStyle::default());
        let fade = ui.slider(pane, SliderStyle::default());
        fade.set_value(&mut ui, 0.35);

        // Two short lines rather than one long one: a pane does not clip, so
        // a readout wider than its half would draw over the neighbour.
        let stats = ui.label(dock.content(profiler), t.text_px, t.accent, "");
        let state = ui.label(dock.content(profiler), t.text_px, t.text_dim, "");
        dock.select(&mut ui, outliner);

        Self {
            frame,
            dock,
            highlight,
            fade,
            stats,
            state,
            profiler,
            last: Instant::now() - READOUT_HZ,
            visible: true,
        }
    }
}

impl Export for DockDemo {}

impl Component for DockDemo {
    fn update(&mut self, _dt: f32, _transform: &Transform, _c: &engine::ComponentRegistry) {
        let mut ui = ui();
        if input::key_pressed(KeyCode::F3) && !ui.keyboard_captured() {
            self.visible = !self.visible;
            ui.set_visible(self.frame, self.visible);
        }

        // The one call the dock is owed: it folds this frame's pointer into
        // the layout — open a tab, lift a panel, land it.
        self.dock.update(&mut ui);

        // A closed tab's pane has no box, so there is nothing to read and
        // nothing to write. The dock says so; nothing here tracks tabs.
        if !self.dock.showing(self.profiler) || self.last.elapsed() < READOUT_HZ {
            return;
        }
        self.last = Instant::now();
        let text = format!("{:.0} fps  {} prims", stats::fps(), ui.prim_count());
        self.stats.set_text(&mut ui, &text);
        // Read back through the controls in the *other* pane, which is how
        // the demo shows they survived being dragged around.
        let text = format!(
            "shadows {}  fade {:.0}%",
            match self.highlight.checked(&ui) {
                true => "on",
                false => "off",
            },
            self.fade.value(&ui) * 100.0,
        );
        self.state.set_text(&mut ui, &text);
    }
}

/// Fill the frame, whatever size it is.
fn fill() -> Style {
    Style {
        display: Display::Flex,
        flex_direction: FlexDirection::Column,
        flex_grow: 1.0,
        gap: Size {
            width: zero(),
            height: zero(),
        },
        ..Default::default()
    }
}
