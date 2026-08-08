//! Widgets — the first layer above nodes, styles and hit testing.
//!
//! A widget here is not a type or a trait object. It is a **constructor that
//! composes primitives already in the store** and hands back a plain
//! [`NodeId`]. `button` builds a node, gives it a state-driven background,
//! adds a centred label and opts it into hit testing; everything that then
//! works on it — `clicked`, `set_node_style`, `node_rect` — is the same API
//! that works on any node. There is no widget hierarchy to escape from.
//!
//! # Why appearance is not the app's job
//!
//! Hover/press styling used to be written out by the caller every frame:
//!
//! ```ignore
//! let fill = if ui.held(b) { HELD } else if ui.hovered(b) { HOVER } else { IDLE };
//! ui.set_background(b, UiStyle::fill(fill).radius(4.0));
//! ```
//!
//! That works — the equality gate makes the redundant writes free — but it
//! puts appearance in the update loop, where every new widget adds another
//! branch nobody can forget to write. [`StateStyle`] moves it into the
//! store: attach the three looks once, and the engine applies the right one.
//!
//! The application is **transition-driven, not per frame**.
//! `UiCore::update_pointer` already computes which node gained and lost
//! hover and press, so restyling touches at most four nodes on the frames
//! where something actually moved, and nothing at all otherwise. A thousand
//! buttons cost the same as one.

use super::{font, rgba, theme, NodeId, Theme, UiCore, UiStyle};

/// A node's three pointer looks. Attach with
/// [`UiCore::set_state_style`]; the engine picks between them.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct StateStyle {
    pub idle: UiStyle,
    pub hover: UiStyle,
    /// Shown while the pointer is held down **and** still over the node.
    /// Dragging off reverts to `idle`, which matches the click rule —
    /// releasing off the node cancels, so it should not look armed.
    pub held: UiStyle,
}

impl StateStyle {
    /// Three tints of one shape: same border and radius, different fill.
    pub fn fills(base: UiStyle, idle: u32, hover: u32, held: u32) -> Self {
        Self {
            idle: UiStyle { fill: idle, ..base },
            hover: UiStyle { fill: hover, ..base },
            held: UiStyle { fill: held, ..base },
        }
    }
}

/// How [`UiCore::button`] looks. Every field is a [`Theme`] role resolved
/// once, so `..Default::default()` still overrides any of them per widget.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ButtonStyle {
    pub idle: u32,
    pub hover: u32,
    pub held: u32,
    pub text: u32,
    pub text_px: f32,
    pub padding: f32,
    pub radius: f32,
}

impl From<Theme> for ButtonStyle {
    fn from(t: Theme) -> Self {
        Self {
            idle: t.control,
            hover: t.control_hover,
            held: t.control_held,
            text: t.text,
            text_px: t.text_px,
            padding: t.pad,
            radius: t.radius,
        }
    }
}

impl Default for ButtonStyle {
    fn default() -> Self {
        theme().into()
    }
}

/// How [`UiCore::checkbox`] looks. The row's three fills are the same
/// control roles a button uses; the square adds a fill and a border.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CheckboxStyle {
    pub idle: u32,
    pub hover: u32,
    pub held: u32,
    pub text: u32,
    /// The check mark itself. Reads against `box_fill`, not against `idle`.
    pub mark: u32,
    pub box_fill: u32,
    pub box_border: u32,
    /// Side of the square, in px.
    pub box_px: f32,
    /// Between the square and the label.
    pub gap: f32,
    pub text_px: f32,
    pub padding: f32,
    pub radius: f32,
}

impl From<Theme> for CheckboxStyle {
    fn from(t: Theme) -> Self {
        Self {
            // At rest the row is invisible, like a list row — it is the
            // square that reads as a control, not a button-shaped band.
            idle: rgba(0, 0, 0, 0),
            hover: t.control_hover,
            held: t.control_held,
            text: t.text,
            mark: t.accent,
            box_fill: t.control,
            box_border: t.outline,
            box_px: 11.0,
            gap: 6.0,
            text_px: t.text_px,
            padding: 2.0,
            radius: t.radius,
        }
    }
}

impl Default for CheckboxStyle {
    fn default() -> Self {
        theme().into()
    }
}

impl UiCore {
    /// Bind a node's background to its pointer state. Applies the current
    /// look immediately, then re-applies on every transition.
    pub fn set_state_style(&mut self, n: NodeId, style: StateStyle) {
        let idx = n.0 as usize;
        if self.state_styles.len() <= idx {
            self.state_styles.resize(idx + 1, None);
        }
        self.state_styles[idx] = Some(style);
        self.apply_state_style(n);
    }

    /// Write the look matching `n`'s current pointer state. A no-op for
    /// nodes with no [`StateStyle`], and free for nodes whose look did not
    /// change — `set_background` goes through the equality gate.
    pub(crate) fn apply_state_style(&mut self, n: NodeId) {
        let Some(Some(s)) = self.state_styles.get(n.0 as usize).copied() else {
            return;
        };
        let hovered = self.hovered(n);
        let style = if hovered && self.held(n) {
            s.held
        } else if hovered {
            s.hover
        } else {
            s.idle
        };
        self.set_background(n, style);
    }

    /// A clickable button: a state-styled box with a centred label, opted
    /// into hit testing. Poll it with [`UiCore::clicked`].
    ///
    /// Returns a plain [`NodeId`] — restyle its layout, re-parent it or read
    /// its rect with the same calls as any other node.
    pub fn button(&mut self, parent: NodeId, text: &str, style: ButtonStyle) -> NodeId {
        use super::style::{AlignItems, Display, JustifyContent, Rect, Style};

        let n = self.node(
            parent,
            Style {
                display: Display::Flex,
                justify_content: Some(JustifyContent::CENTER),
                align_items: Some(AlignItems::CENTER),
                padding: Rect::length(style.padding),
                ..Default::default()
            },
        );
        self.set_state_style(
            n,
            StateStyle::fills(
                UiStyle::fill(style.idle).radius(style.radius),
                style.idle,
                style.hover,
                style.held,
            ),
        );
        self.label(n, style.text_px, style.text, text);
        self.set_interactive(n, true);
        n
    }

    /// A checkbox: a square that shows a check mark, with a label beside it.
    /// The whole row is the control, so clicking the text toggles too.
    ///
    /// **The value is the application's**, not the widget's. Poll the click
    /// and push the result back:
    ///
    /// ```ignore
    /// if ui.clicked(self.cb) { self.wireframe = !self.wireframe; }
    /// ui.set_checked(self.cb, self.wireframe);
    /// ```
    ///
    /// The second line is unconditional on purpose — see [`set_checked`]
    /// (UiCore::set_checked).
    pub fn checkbox(&mut self, parent: NodeId, text: &str, style: CheckboxStyle) -> NodeId {
        use super::style::{px, AlignItems, Display, JustifyContent, Rect, Size, Style};

        let row = self.node(
            parent,
            Style {
                display: Display::Flex,
                align_items: Some(AlignItems::CENTER),
                gap: Size { width: px(style.gap), height: super::style::zero() },
                padding: Rect::length(style.padding),
                ..Default::default()
            },
        );
        self.set_state_style(
            row,
            StateStyle::fills(
                UiStyle::fill(style.idle).radius(style.radius),
                style.idle,
                style.hover,
                style.held,
            ),
        );

        let boxed = self.node(
            row,
            Style {
                display: Display::Flex,
                justify_content: Some(JustifyContent::CENTER),
                align_items: Some(AlignItems::CENTER),
                size: Size { width: px(style.box_px), height: px(style.box_px) },
                flex_shrink: 0.0,
                ..Default::default()
            },
        );
        self.set_background(
            boxed,
            UiStyle::fill(style.box_fill).border(style.box_border, 1.0).radius(style.radius * 0.5),
        );
        // Empty until `set_checked`: the mark is one glyph whose string is
        // either the check or nothing, so toggling dirties a single slot.
        let mark = self.label(boxed, style.text_px, style.mark, "");

        self.label(row, style.text_px, style.text, text);
        self.set_interactive(row, true);

        let idx = row.0 as usize;
        if self.checkbox_marks.len() <= idx {
            self.checkbox_marks.resize(idx + 1, None);
        }
        self.checkbox_marks[idx] = Some(mark);
        row
    }

    /// Show `checked` on a checkbox built by [`checkbox`](UiCore::checkbox).
    ///
    /// Call it **every frame, unconditionally**. That is what makes the
    /// value app-owned rather than widget-owned: the checkbox holds no
    /// truth of its own, so a value changed by a keybind, a network packet
    /// or a script shows up with no notification path — the next call
    /// simply writes what is now true. The equality gate makes the repeats
    /// free, exactly as it does for `apply_state_style`.
    ///
    /// Panics on a node that is not a checkbox, rather than silently doing
    /// nothing.
    pub fn set_checked(&mut self, n: NodeId, checked: bool) {
        let mark = self.checkbox_marks.get(n.0 as usize).copied().flatten();
        let mark = mark.unwrap_or_else(|| panic!("node {} is not a checkbox", n.0));
        let mut glyph = [0u8; 4];
        self.set_label(mark, if checked { font::CHECK.encode_utf8(&mut glyph) } else { "" });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::style::Style;

    fn checkbox(core: &mut UiCore) -> NodeId {
        let root = core.root();
        let n = core.checkbox(root, "wireframe", CheckboxStyle::default());
        core.run_layout([400.0, 400.0]);
        n
    }

    /// The value lives in the application, so a change that never went
    /// through a click still shows: `set_checked` is told what is true now
    /// and writes it, with no notification path to miss.
    #[test]
    fn the_mark_follows_a_value_changed_outside_the_widget() {
        let mut core = UiCore::new();
        let cb = checkbox(&mut core);
        let mark = core.checkbox_marks[cb.0 as usize].expect("mark recorded");

        assert_eq!(core.node_text(mark), Some(""), "starts unchecked");

        // Nothing clicked the checkbox — some other code changed the value.
        core.set_checked(cb, true);
        assert_eq!(core.node_text(mark).map(str::chars).map(|mut c| c.next()), Some(Some(font::CHECK)));

        core.set_checked(cb, false);
        assert_eq!(core.node_text(mark), Some(""));
    }

    /// `set_checked` is meant to be called every frame unconditionally, so
    /// re-asserting the current value must cost nothing.
    #[test]
    fn restating_the_same_value_uploads_nothing() {
        let mut core = UiCore::new();
        let cb = checkbox(&mut core);
        core.set_checked(cb, true);
        core.run_layout([400.0, 400.0]);

        let (mut stage, mut dirty) = (vec![0u32; 1 << 16], vec![0u32; 1 << 10]);
        let clean = (i64::MAX, -1);
        core.quad.upload(&mut stage, &mut dirty);
        core.style.upload(&mut stage, &mut dirty);

        for _ in 0..8 {
            core.set_checked(cb, true);
            core.run_layout([400.0, 400.0]);
        }
        assert_eq!(core.quad.upload(&mut stage, &mut dirty), clean);
        assert_eq!(core.style.upload(&mut stage, &mut dirty), clean);
    }

    /// The row is the control, so the label is part of the hit target — but
    /// the square must not become a second, inner one.
    #[test]
    fn clicking_the_label_hits_the_checkbox_row() {
        let mut core = UiCore::new();
        let cb = checkbox(&mut core);
        let rect = core.node_rect(cb);
        let far_right = [rect[0] + rect[2] - 1.0, rect[1] + rect[3] * 0.5];

        assert_eq!(core.hit_test(far_right), Some(cb), "label area belongs to the row");
        assert_eq!(core.hit_test([rect[0] + 1.0, rect[1] + rect[3] * 0.5]), Some(cb), "square too");
    }

    #[test]
    #[should_panic(expected = "is not a checkbox")]
    fn set_checked_on_a_plain_node_panics() {
        let mut core = UiCore::new();
        let root = core.root();
        let n = core.node(root, Style::default());
        core.set_checked(n, true);
    }
}
