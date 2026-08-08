//! Widgets — the first layer above nodes, styles and hit testing.
//!
//! A widget here is a **constructor that composes primitives already in the
//! store** and hands back a `Copy` handle. `button` builds a node, gives it
//! a state-driven background, adds a centred label and opts it into hit
//! testing; everything that then works on the result — `clicked`,
//! `set_node_style`, `node_rect`, re-parenting — is the same API that works
//! on any node, because every handle converts back into a [`NodeId`] and
//! those calls take `impl Into<NodeId>`. There is no widget hierarchy to
//! escape from and no trait object anywhere.
//!
//! # Why the handles are typed
//!
//! They are not just labels on a `u32` — they say what the node *is*, which
//! is what makes [`Checkbox::checked`] and [`Slider::value`] exist at all.
//! A `NodeId` has no value to read; a `Checkbox` does.
//!
//! Operations split accordingly. Anything true of *any* node — pointer
//! state, layout, re-parenting — stays on [`UiCore`] and accepts every
//! handle. Anything that needs the widget's own structure is a method on
//! the handle, matching [`RowList`](super::RowList) and
//! [`TreeView`](super::TreeView), which were already caller-owned types.
//! That also means `UiCore` needs no new method per widget: a *game* can
//! define its own widget type against the public node API without the
//! engine knowing.
//!
//! The handles are `Copy` indices, so a removed widget's handle would name
//! whatever moved into its slot. That is why [`NodeId`] carries a
//! generation: [`UiCore::remove_node`] bumps it, so every handle into the
//! removed subtree — typed or not — panics on next use instead.
//!
//! # Where a control's value lives
//!
//! **In the control.** [`Checkbox::checked`] and [`Slider::value`] read it;
//! [`Checkbox::set_checked`] and [`Slider::set_value`] write it; the click
//! or drag that moves it is applied by `update_pointer`, before any
//! component runs.
//!
//! The alternative — the widget storing nothing and the application
//! restating the value every frame from `clicked` / `dragged` — was tried
//! and is worse in the ordinary case. It makes *reading* a checkbox
//! impossible without keeping a parallel `bool` in sync by hand, so the
//! caller ends up owning a mirror whether they wanted one or not. Owning
//! the value here removes the mirror without removing the control: setting
//! it is still allowed, so a value loaded from disk, moved by a keybind or
//! pushed by a network packet lands the same way a click does.
//!
//! It costs one sparse table ([`Control`]) and no per-frame work. Only the
//! node the pointer clicked and the node it is dragging are ever consulted,
//! so a thousand controls fold as fast as one.
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

// ─────────────────────────────────────────────────────────────────────
// Typed handles
// ─────────────────────────────────────────────────────────────────────

/// Declare a `Copy` newtype over a `NodeId` that converts back into one,
/// so every generic node call (`clicked`, `node_rect`, `set_node_style`,
/// re-parenting) keeps working on a typed handle unchanged.
macro_rules! handle {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub struct $name(NodeId);

        impl $name {
            pub(crate) fn from_node(n: NodeId) -> Self {
                Self(n)
            }

            /// The underlying node, for APIs that take one explicitly.
            pub fn node(self) -> NodeId {
                self.0
            }
        }

        impl From<$name> for NodeId {
            fn from(h: $name) -> NodeId {
                h.0
            }
        }
    };
}

handle! {
    /// A text leaf, from [`UiCore::label`]. Owning the handle is what makes
    /// [`set_text`](Label::set_text) infallible — there is no longer a way
    /// to name a node that has no glyph run.
    Label
}

handle! {
    /// A clickable box with a centred label, from [`UiCore::button`]. Poll
    /// it with `ui.clicked(btn)` — pointer state is a property of any
    /// interactive node, not of buttons, so it stays on [`UiCore`].
    Button
}

impl Label {
    /// Retype the label. Unchanged text returns before touching anything;
    /// changed text dirties only the glyphs that differ.
    pub fn set_text(self, ui: &mut UiCore, text: &str) {
        ui.set_label(self.0, text);
    }

    pub fn set_color(self, ui: &mut UiCore, color: u32) {
        ui.set_label_color(self.0, color);
    }
}

handle! {
    /// A checkbox, from [`UiCore::checkbox`]. Owns its `bool`: read it with
    /// [`checked`](Checkbox::checked), write it with
    /// [`set_checked`](Checkbox::set_checked), and a click toggles it
    /// without the application in the loop.
    Checkbox
}

handle! {
    /// A horizontal slider, from [`UiCore::slider`]. Owns its `f32`, which
    /// a drag moves; read it with [`value`](Slider::value).
    ///
    /// Values are normalised `0.0..=1.0`. A caller with a real range scales
    /// at the two call sites, which is one multiply each and keeps the
    /// widget from carrying a range it would then have to validate.
    Slider
}

/// A control's value and the parts it redraws when that value moves.
///
/// Kept in `UiCore` rather than in the handle because `update_pointer` has
/// only a [`NodeId`] to work from: the click that toggles a checkbox and the
/// drag that moves a slider are applied there, so a caller reads a value
/// that is already current instead of reconstructing it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Control {
    Checkbox { mark: Label, checked: bool },
    Slider { fill: NodeId, thumb: NodeId, value: f32 },
}

impl Checkbox {
    /// Whether it is ticked.
    pub fn checked(self, ui: &UiCore) -> bool {
        let Control::Checkbox { checked, .. } = ui.control(self.0) else {
            unreachable!("Checkbox handle over a non-checkbox")
        };
        checked
    }

    /// Set it, as a click would. Returns before touching anything when the
    /// value already holds, so restating it costs a comparison.
    pub fn set_checked(self, ui: &mut UiCore, checked: bool) {
        let Control::Checkbox { mark, checked: was } = ui.control(self.0) else {
            unreachable!("Checkbox handle over a non-checkbox")
        };
        if was == checked {
            return;
        }
        ui.set_control(self.0, Control::Checkbox { mark, checked });
        // One glyph whose string is either the check or nothing, so a
        // toggle dirties a single slot.
        let mut glyph = [0u8; 4];
        mark.set_text(ui, if checked { font::CHECK.encode_utf8(&mut glyph) } else { "" });
    }
}

impl Slider {
    /// Its current value, `0.0..=1.0`.
    pub fn value(self, ui: &UiCore) -> f32 {
        let Control::Slider { value, .. } = ui.control(self.0) else {
            unreachable!("Slider handle over a non-slider")
        };
        value
    }

    /// Set it, as a drag would; `value` is clamped to `0.0..=1.0`. Returns
    /// before touching anything when the value already holds, so a still
    /// slider costs no relayout.
    pub fn set_value(self, ui: &mut UiCore, value: f32) {
        let Control::Slider { fill, thumb, value: was } = ui.control(self.0) else {
            unreachable!("Slider handle over a non-slider")
        };
        let v = value.clamp(0.0, 1.0);
        if was == v {
            return;
        }
        ui.set_control(self.0, Control::Slider { fill, thumb, value: v });

        let mut s = ui.node_style(fill);
        s.size.width = super::style::percent(v);
        ui.set_node_style(fill, s);

        let mut s = ui.node_style(thumb);
        s.inset.left = super::style::percent(v);
        ui.set_node_style(thumb, s);
    }
}

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

/// How [`UiCore::slider`] looks and measures.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SliderStyle {
    pub track: u32,
    pub fill: u32,
    pub thumb: u32,
    /// Track length in px. Sliders need a definite width to map a pointer
    /// position onto, so this is not left to the parent.
    pub width: f32,
    pub track_px: f32,
    pub thumb_px: f32,
}

impl From<Theme> for SliderStyle {
    fn from(t: Theme) -> Self {
        Self {
            track: t.control,
            fill: t.accent,
            thumb: t.text,
            width: 120.0,
            track_px: 4.0,
            thumb_px: 10.0,
        }
    }
}

impl Default for SliderStyle {
    fn default() -> Self {
        theme().into()
    }
}

impl UiCore {
    /// A control node's state. Infallible in practice — only `checkbox` and
    /// `slider` mint the handles that reach it, and a handle into a removed
    /// subtree is caught by `live` first.
    fn control(&self, n: NodeId) -> Control {
        self.controls
            .get(self.live(n))
            .and_then(|c| *c)
            .unwrap_or_else(|| panic!("NodeId {} is not a control", n.idx))
    }

    fn set_control(&mut self, n: NodeId, c: Control) {
        let idx = self.live(n);
        if self.controls.len() <= idx {
            self.controls.resize(idx + 1, None);
        }
        self.controls[idx] = Some(c);
    }

    /// Apply this frame's pointer to whatever control it landed on, from
    /// [`UiCore::update_pointer`].
    ///
    /// `dragging` is the pressed node captured *before* the release branch
    /// clears it, so a frame that both moves and releases still commits the
    /// position it was released at.
    ///
    /// Two lookups, whatever the tree holds: a control only changes under
    /// the pointer that is on it.
    pub(crate) fn drive_controls(&mut self, dragging: Option<NodeId>) {
        if let Some(n) = self.pointer.clicked {
            if let Some(Some(Control::Checkbox { checked, .. })) = self.controls.get(n.idx as usize)
            {
                let now = !*checked;
                Checkbox::from_node(n).set_checked(self, now);
            }
        }
        if let Some(n) = dragging {
            if let Some(Some(Control::Slider { .. })) = self.controls.get(n.idx as usize) {
                // Absolute, not incremental: the value is where the pointer
                // is along the track, so pressing anywhere jumps there and
                // the thumb cannot drift from the cursor the way accumulated
                // deltas do.
                let r = self.node_rect(n);
                let v = (self.pointer.pos[0] - r[0]) / r[2].max(1.0);
                Slider::from_node(n).set_value(self, v);
            }
        }
    }

    /// Bind a node's background to its pointer state. Applies the current
    /// look immediately, then re-applies on every transition.
    pub fn set_state_style(&mut self, n: impl Into<NodeId>, style: StateStyle) {
        let n = n.into();
        let idx = self.live(n);
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
        let Some(Some(s)) = self.state_styles.get(n.idx as usize).copied() else {
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
    pub fn button(&mut self, parent: impl Into<NodeId>, text: &str, style: ButtonStyle) -> Button {
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
        Button::from_node(n)
    }

    /// A checkbox: a square that shows a check mark, with a label beside it.
    /// The whole row is the control, so clicking the text toggles too.
    ///
    /// It owns its value — a click flips it before any component runs, so
    /// the caller only reads:
    ///
    /// ```ignore
    /// let cb = ui.checkbox(panel, "wireframe", CheckboxStyle::default());
    /// // …later, and from anywhere:
    /// if cb.checked(&ui) { /* … */ }
    /// cb.set_checked(&mut ui, from_settings_file);
    /// ```
    pub fn checkbox(&mut self, parent: impl Into<NodeId>, text: &str, style: CheckboxStyle) -> Checkbox {
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
        let mark = self.label(boxed, style.text_px, style.mark, "");

        self.label(row, style.text_px, style.text, text);
        self.set_interactive(row, true);
        self.set_control(row, Control::Checkbox { mark, checked: false });
        Checkbox::from_node(row)
    }

    /// A horizontal slider: a track with a filled portion and a thumb.
    ///
    /// Only the track is interactive, so the gesture is tracked wherever
    /// inside it the press landed — a thumb that swallowed its own hits
    /// would need the press forwarded back to the track.
    ///
    /// It owns its value, and a drag moves it. Read it with
    /// [`Slider::value`]; set it with [`Slider::set_value`].
    pub fn slider(&mut self, parent: impl Into<NodeId>, style: SliderStyle) -> Slider {
        use super::style::{percent, px, zero, LengthPercentageAuto, Position, Rect, Size,
            Style, TaffyAuto};

        let track = self.node(
            parent,
            Style {
                size: Size { width: px(style.width), height: px(style.track_px) },
                flex_shrink: 0.0,
                ..Default::default()
            },
        );
        self.set_background(
            track,
            UiStyle::fill(style.track).radius(style.track_px * 0.5),
        );

        let fill = self.node(
            track,
            Style {
                position: Position::Absolute,
                inset: Rect {
                    left: px(0.0),
                    top: px(0.0),
                    right: LengthPercentageAuto::AUTO,
                    bottom: px(0.0),
                },
                size: Size { width: percent(0.0_f32), height: TaffyAuto::AUTO },
                ..Default::default()
            },
        );
        self.set_background(fill, UiStyle::fill(style.fill).radius(style.track_px * 0.5));

        // Nudged left by half its width so it centres on the value rather
        // than starting at it — otherwise 1.0 parks it entirely outside.
        let thumb = self.node(
            track,
            Style {
                position: Position::Absolute,
                inset: Rect {
                    left: percent(0.0_f32),
                    top: px((style.track_px - style.thumb_px) * 0.5),
                    right: LengthPercentageAuto::AUTO,
                    bottom: LengthPercentageAuto::AUTO,
                },
                margin: Rect {
                    left: px(-style.thumb_px * 0.5),
                    right: zero(),
                    top: zero(),
                    bottom: zero(),
                },
                size: Size { width: px(style.thumb_px), height: px(style.thumb_px) },
                ..Default::default()
            },
        );
        self.set_state_style(
            thumb,
            StateStyle::fills(
                UiStyle::fill(style.thumb).radius(style.thumb_px * 0.5),
                style.thumb,
                style.thumb,
                style.thumb,
            ),
        );

        self.set_interactive(track, true);
        self.set_control(track, Control::Slider { fill, thumb, value: 0.0 });
        Slider::from_node(track)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::style::Style;

    fn checkbox(core: &mut UiCore) -> Checkbox {
        let root = core.root();
        let cb = core.checkbox(root, "wireframe", CheckboxStyle::default());
        core.run_layout([400.0, 400.0]);
        cb
    }

    /// The mark node, which only the store knows about.
    fn mark_of(core: &UiCore, cb: Checkbox) -> Label {
        let Control::Checkbox { mark, .. } = core.control(cb.0) else {
            unreachable!()
        };
        mark
    }

    /// The ergonomic claim: a click moves the value with nothing in the
    /// application asked to notice, and the caller reads the result.
    #[test]
    fn clicking_toggles_the_value_with_no_application_involved() {
        let mut core = UiCore::new();
        let cb = checkbox(&mut core);
        let r = core.node_rect(cb);
        let p = [r[0] + 2.0, r[1] + r[3] * 0.5];

        assert!(!cb.checked(&core), "starts unchecked");

        core.update_pointer(p, true, false, 0.0);
        assert!(!cb.checked(&core), "press alone must not toggle");
        core.update_pointer(p, false, true, 0.0);
        assert!(cb.checked(&core), "the release completes the click");

        core.update_pointer(p, true, false, 0.0);
        core.update_pointer(p, false, true, 0.0);
        assert!(!cb.checked(&core), "and back");
    }

    /// A press dragged off the control cancels, so it must not toggle
    /// either — the value follows `clicked`, not `down_on`.
    #[test]
    fn a_cancelled_click_leaves_the_value_alone() {
        let mut core = UiCore::new();
        let cb = checkbox(&mut core);
        let r = core.node_rect(cb);

        core.update_pointer([r[0] + 2.0, r[1] + r[3] * 0.5], true, false, 0.0);
        core.update_pointer([r[0] + r[2] + 200.0, r[1] + 400.0], false, false, 0.0);
        core.update_pointer([r[0] + r[2] + 200.0, r[1] + 400.0], false, true, 0.0);
        assert!(!cb.checked(&core));
    }

    /// Owning the value does not close it off: a change that never went
    /// through a click — a settings file, a keybind, a network packet —
    /// lands the same way, mark and all.
    #[test]
    fn the_mark_follows_a_value_set_from_outside() {
        let mut core = UiCore::new();
        let cb = checkbox(&mut core);
        let mark = mark_of(&core, cb).node();

        assert_eq!(core.node_text(mark), Some(""), "starts unchecked");

        cb.set_checked(&mut core, true);
        assert!(cb.checked(&core));
        assert_eq!(core.node_text(mark).and_then(|s| s.chars().next()), Some(font::CHECK));

        cb.set_checked(&mut core, false);
        assert_eq!(core.node_text(mark), Some(""));
    }

    /// Setting the value it already holds must cost nothing, so a caller
    /// pushing a value every frame stays as cheap as one that doesn't.
    #[test]
    fn restating_the_same_value_uploads_nothing() {
        let mut core = UiCore::new();
        let cb = checkbox(&mut core);
        cb.set_checked(&mut core, true);
        core.run_layout([400.0, 400.0]);

        let (mut stage, mut dirty) = (vec![0u32; 1 << 16], vec![0u32; 1 << 10]);
        let clean = (i64::MAX, -1);
        core.quad.upload(&mut stage, &mut dirty);
        core.style.upload(&mut stage, &mut dirty);

        for _ in 0..8 {
            cb.set_checked(&mut core, true);
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

        assert_eq!(core.hit_test(far_right), Some(cb.node()), "label area belongs to the row");
        assert_eq!(
            core.hit_test([rect[0] + 1.0, rect[1] + rect[3] * 0.5]),
            Some(cb.node()),
            "square too",
        );
    }

    /// A typed handle keeps working with every generic node call — that is
    /// what stops the widget layer from becoming a walled garden.
    #[test]
    fn handles_are_accepted_wherever_a_node_is() {
        let mut core = UiCore::new();
        let root = core.root();
        let btn = core.button(root, "go", ButtonStyle::default());
        core.run_layout([400.0, 400.0]);

        core.set_node_style(btn, Style::default());
        core.set_interactive(btn, true);
        assert!(!core.clicked(btn));
        assert_ne!(core.node_rect(btn), [0.0; 4]);
        // And a child can be parented into one.
        let inner = core.node(btn, Style::default());
        assert_ne!(inner, btn.node());
    }

    fn slider(core: &mut UiCore) -> Slider {
        let root = core.root();
        let sl = core.slider(root, SliderStyle { width: 100.0, ..Default::default() });
        core.run_layout([400.0, 400.0]);
        sl
    }

    /// A drag writes the value under the pointer, absolutely — pressing at
    /// the middle of the track means 0.5, no accumulation involved.
    #[test]
    fn dragging_maps_the_pointer_onto_the_track() {
        let mut core = UiCore::new();
        let sl = slider(&mut core);
        let r = core.node_rect(sl);
        let y = r[1] + r[3] * 0.5;

        assert_eq!(sl.value(&core), 0.0, "starts at zero");

        core.update_pointer([r[0] + r[2] * 0.5, y], true, false, 0.0);
        assert_eq!(sl.value(&core), 0.5, "the press alone jumps there");

        core.update_pointer([r[0] + r[2] * 0.25, y], false, false, 0.0);
        assert_eq!(sl.value(&core), 0.25, "value follows the pointer");
    }

    /// The gesture must survive the pointer leaving the track, or a slider
    /// snaps back the moment you overshoot its end.
    #[test]
    fn a_drag_continues_past_the_end_of_the_track() {
        let mut core = UiCore::new();
        let sl = slider(&mut core);
        let r = core.node_rect(sl);
        let y = r[1] + r[3] * 0.5;

        core.update_pointer([r[0] + 1.0, y], true, false, 0.0);
        core.update_pointer([r[0] + r[2] + 500.0, y], false, false, 0.0);
        assert_eq!(sl.value(&core), 1.0, "clamped, still dragging");

        // Moving and releasing on the same frame still commits: `dragging`
        // is captured before the release clears the press.
        core.update_pointer([r[0] - 500.0, y], false, true, 0.0);
        assert_eq!(sl.value(&core), 0.0);

        core.update_pointer([r[0] + r[2] * 0.5, y], false, false, 0.0);
        assert_eq!(sl.value(&core), 0.0, "released — moving no longer drags it");
    }

    /// Nothing about the pointer touches a control it is not on.
    #[test]
    fn dragging_elsewhere_leaves_a_slider_alone() {
        let mut core = UiCore::new();
        let sl = slider(&mut core);
        let root = core.root();
        let other = core.node(root, Style::default());
        core.set_interactive(other, true);
        sl.set_value(&mut core, 0.6);
        core.run_layout([400.0, 400.0]);

        let r = core.node_rect(sl);
        core.update_pointer([r[0] + r[2] + 50.0, r[1] + 200.0], true, false, 0.0);
        core.update_pointer([r[0] + r[2] * 0.1, r[1]], false, false, 0.0);
        assert_eq!(sl.value(&core), 0.6, "a drag that started elsewhere");
    }

    /// `Drag` keeps its origin, which is what lets a caller tell a drag from
    /// a click that wobbled — the rule drag-to-reparent needs.
    #[test]
    fn a_drag_measures_from_where_it_started() {
        let mut core = UiCore::new();
        let sl = slider(&mut core);
        let r = core.node_rect(sl);
        let start = [r[0] + 10.0, r[1] + r[3] * 0.5];

        core.update_pointer(start, true, false, 0.0);
        let d = core.drag(sl).expect("dragging");
        assert_eq!(d.delta(), [0.0, 0.0]);
        assert!(!d.beyond(4.0));

        core.update_pointer([start[0] + 12.0, start[1]], false, false, 0.0);
        let d = core.drag(sl).expect("still dragging");
        assert_eq!(d.origin, start, "origin is the press, not the last frame");
        assert_eq!(d.delta(), [12.0, 0.0]);
        assert!(d.beyond(4.0));
    }

    /// Same contract as the checkbox: setting the value it already holds
    /// must not relayout.
    #[test]
    fn restating_a_sliders_value_uploads_nothing() {
        let mut core = UiCore::new();
        let sl = slider(&mut core);
        sl.set_value(&mut core, 0.4);
        core.run_layout([400.0, 400.0]);

        let (mut stage, mut dirty) = (vec![0u32; 1 << 16], vec![0u32; 1 << 10]);
        let clean = (i64::MAX, -1);
        core.quad.upload(&mut stage, &mut dirty);
        core.style.upload(&mut stage, &mut dirty);

        for _ in 0..8 {
            sl.set_value(&mut core, 0.4);
            core.run_layout([400.0, 400.0]);
        }
        assert_eq!(core.quad.upload(&mut stage, &mut dirty), clean);
        assert_eq!(core.style.upload(&mut stage, &mut dirty), clean);
    }

    /// The fill has to actually move, or the two tests above would pass on a
    /// slider that renders nothing.
    #[test]
    fn the_fill_width_tracks_the_value() {
        let mut core = UiCore::new();
        let sl = slider(&mut core);
        let track = core.node_rect(sl)[2];
        let Control::Slider { fill, .. } = core.control(sl.0) else {
            unreachable!()
        };

        sl.set_value(&mut core, 0.25);
        core.run_layout([400.0, 400.0]);
        let quarter = core.node_rect(fill)[2];

        sl.set_value(&mut core, 0.75);
        core.run_layout([400.0, 400.0]);
        let three_quarters = core.node_rect(fill)[2];

        assert!((quarter - track * 0.25).abs() < 0.5, "quarter fill: {quarter}");
        assert!((three_quarters - track * 0.75).abs() < 0.5, "three-quarter fill");
    }
}
