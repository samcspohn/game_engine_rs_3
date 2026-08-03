//! A built-in overlay that exercises the phase-1/2 primitive kinds and the
//! phase-3 layout pass.
//!
//! Nothing here computes a coordinate. The panel is a flex column with
//! padding and a gap; the swatch strip is a five-track grid; both size
//! themselves from their content. That is the point — the manual
//! `y += TITLE_PX + 6.0` accumulator this file used to carry is exactly
//! what taffy replaced.
//!
//! It also makes the upload behaviour observable: the panel and the
//! specimen lines are written once and then never again, and only the
//! readout changes — at 10 Hz, not per frame. So `ENGINE_UI_TRACE=1` stays
//! silent on the large majority of frames even though the UI is on screen
//! the whole time, which is the single claim ADR-0006 phase 1 asks to be
//! proven.
//!
//! Toggle with **F6**.

use std::time::{Duration, Instant};

use super::style::{
    evenly_sized_tracks, percent, px, zero, AlignItems, Display, FlexDirection,
    LengthPercentageAuto, Position, Rect, Size, Style, TaffyAuto,
};
use super::{rgb, rgba, NodeId, UiCore, UiStyle};

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

pub struct Demo {
    panel: NodeId,
    readout: NodeId,
    last_readout: Instant,
    visible: bool,
}

impl Demo {
    pub fn build(core: &mut UiCore) -> Self {
        let screen = core.root();

        // Absolutely positioned so the overlay doesn't participate in the
        // root's flow; `auto` width/height means it shrink-wraps its
        // content, so the panel resizes itself when a line gets longer.
        let panel = core.node(
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
        core.set_background(
            panel,
            UiStyle::fill(rgba(0x14, 0x18, 0x22, 0xE0))
                .border(rgba(0x6C, 0xC4, 0xFF, 0x60), 1.0)
                .radius(8.0),
        );

        core.label(panel, TITLE_PX, INK, "retained ui / ADR-0006");
        let readout = core.label(panel, BODY_PX, ACCENT, "");
        for line in SPECIMEN {
            core.label(panel, BODY_PX, DIM, line);
        }

        // Five equal grid tracks stretched across the panel's content box —
        // the swatches never learn their own width.
        let strip = core.node(
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
            let cell = core.node(strip, Style::default());
            core.set_background(cell, UiStyle::fill(color).radius(2.0));
        }

        Self {
            panel,
            readout,
            last_readout: Instant::now() - READOUT_HZ,
            visible: true,
        }
    }

    /// Hide by taking the panel out of layout entirely — its boxes collapse
    /// to zero and `ui.vert`'s zero-area early-out culls them. A per-panel
    /// `ui_group` would make this one record instead of one per primitive;
    /// groups earn that when docking gives every panel its own.
    pub fn toggle(&mut self, core: &mut UiCore) {
        self.visible = !self.visible;
        let mut style = core.node_style(self.panel);
        style.display = if self.visible {
            Display::Flex
        } else {
            Display::None
        };
        core.set_node_style(self.panel, style);
    }

    /// Per-frame update. Almost always a clock read and a comparison: the
    /// readout is throttled, and `run_layout` returns before doing anything
    /// when nothing marked the tree dirty.
    pub fn update(&mut self, core: &mut UiCore, screen: [f32; 2], fps: f64, prims: u32) {
        if self.last_readout.elapsed() >= READOUT_HZ {
            self.last_readout = Instant::now();
            core.set_label(
                self.readout,
                &format!(
                    "{fps:.0} fps   {prims} primitives   {}x{}",
                    screen[0] as u32, screen[1] as u32
                ),
            );
        }
        core.run_layout(screen);
    }
}
