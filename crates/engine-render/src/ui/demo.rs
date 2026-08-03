//! A built-in overlay that exercises every phase-1/2 primitive kind.
//!
//! It exists to make the upload behaviour observable rather than to be a
//! useful widget: the panel and the specimen lines are written once and then
//! never again, and only the readout line changes — at 10 Hz, not per frame.
//! So the dirty-word counter (`ENGINE_UI_TRACE=1`) reads zero on the large
//! majority of frames even though the UI is on screen the whole time, which
//! is the single claim ADR-0006 phase 1 asks to be proven.
//!
//! Toggle with **F6**.

use std::time::{Duration, Instant};

use super::{rgb, rgba, GroupId, TextId, UiCore, UiStyle};

const PAD: f32 = 12.0;
const PANEL_W: f32 = 420.0;
const PANEL_H: f32 = 102.0;
const TITLE_PX: f32 = 14.0;
const BODY_PX: f32 = 11.0;

const INK: u32 = rgb(0xE6, 0xE9, 0xEF);
const DIM: u32 = rgb(0x8A, 0x93, 0xA6);
const ACCENT: u32 = rgb(0x6C, 0xC4, 0xFF);

/// The printable ASCII the built-in font covers, split so both halves fit
/// inside the panel. Written once — a permanent proof that the atlas is
/// complete and correctly addressed.
const SPECIMEN: [&str; 3] = [
    " !\"#$%&'()*+,-./0123456789:;<=>?@",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`",
    "abcdefghijklmnopqrstuvwxyz{|}~",
];

/// Readout refresh interval. A stats line in a 1000 FPS editor does ten text
/// relayouts per second, not a thousand — a human reads it at 10 Hz either
/// way. This is ADR-0006's tick set in miniature, hard-coded until phase 3
/// provides the real one.
const READOUT_HZ: Duration = Duration::from_millis(100);

pub struct Demo {
    root: GroupId,
    readout: TextId,
    last_readout: Instant,
    visible: bool,
}

impl Demo {
    pub fn build(core: &mut UiCore, screen: [f32; 2]) -> Self {
        let root = core.group([0.0, 0.0, screen[0], screen[1]], [PAD, PAD]);

        core.rect(
            root,
            [0.0, 0.0, PANEL_W, PANEL_H],
            UiStyle::fill(rgba(0x14, 0x18, 0x22, 0xE0))
                .border(rgba(0x6C, 0xC4, 0xFF, 0x60), 1.0)
                .radius(8.0),
        );
        // Accent bar down the left edge — a second rect, so the panel proves
        // painter's order (allocation order) rather than just one quad.
        core.rect(
            root,
            [0.0, 0.0, 3.0, PANEL_H],
            UiStyle::fill(ACCENT).corners([8.0, 0.0, 0.0, 8.0]),
        );

        let mut y = PAD;
        core.text(root, [PAD, y], TITLE_PX, INK, "retained ui / ADR-0006");
        y += TITLE_PX + 6.0;
        let readout = core.text(root, [PAD, y], BODY_PX, ACCENT, "");
        y += BODY_PX + 8.0;
        for line in SPECIMEN {
            core.text(root, [PAD, y], BODY_PX, DIM, line);
            y += BODY_PX + 2.0;
        }

        Self {
            root,
            readout,
            last_readout: Instant::now() - READOUT_HZ,
            visible: true,
        }
    }

    pub fn toggle(&mut self, core: &mut UiCore) {
        self.visible = !self.visible;
        core.set_group_opacity(self.root, if self.visible { 1.0 } else { 0.0 });
    }

    /// Per-frame update. The clip re-set is unconditional and free: it only
    /// reaches staging on a frame where the window actually resized, because
    /// `SlotArray::set` compares before it marks.
    pub fn update(&mut self, core: &mut UiCore, screen: [f32; 2], fps: f64, prims: u32) {
        core.set_group_clip(self.root, [0.0, 0.0, screen[0], screen[1]]);
        if self.last_readout.elapsed() < READOUT_HZ {
            return;
        }
        self.last_readout = Instant::now();
        core.set_text(
            self.readout,
            &format!(
                "{fps:.0} fps   {prims} primitives   {}x{}",
                screen[0] as u32, screen[1] as u32
            ),
        );
    }
}
