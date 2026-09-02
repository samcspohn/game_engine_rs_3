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

use super::{font, rgba, theme, Events, NodeId, Theme, UiCore, UiStyle};

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

handle! {
    /// A single-line text field, from [`UiCore::text_field`]. Owns its
    /// `String`, which the keyboard edits; read it with
    /// [`text`](TextField::text). See [`text_field`](super::text_field).
    TextField
}

handle! {
    /// A set of mutually exclusive options, from [`UiCore::radio_group`].
    ///
    /// Owns the selected index. Unlike every other control, its value does
    /// not live on the node the pointer hits — it lives on the container, and
    /// a click on one option repaints the option it *left*. Read it with
    /// [`selected`](RadioGroup::selected).
    RadioGroup
}

handle! {
    /// A tab strip and the panes it switches between, from [`UiCore::tabs`].
    ///
    /// Same shared selection as [`RadioGroup`], spent on *layout* rather than
    /// on a fill: the unselected panes are `Display::None`, so their whole
    /// subtree collapses — nothing paints, nothing takes a hit, and a pane
    /// the caller never looks at again costs one node.
    Tabs
}

handle! {
    /// A vertical scrollbar over a [`scroll_area`](UiCore::scroll_area), from
    /// [`UiCore::scrollbar`].
    ///
    /// The one widget that owns no value at all: the thumb is a *view* of the
    /// area's offset and content extent, and both move without the bar being
    /// touched — a wheel, a window resize, a list that grew. So it is also the
    /// one thing the engine re-fits by itself, after a layout and after a
    /// scroll rather than per frame.
    Scrollbar
}

/// A control's value and the parts it redraws when that value moves.
///
/// Kept in `UiCore` rather than in the handle because `update_pointer` has
/// only a [`NodeId`] to work from: the click that toggles a checkbox and the
/// drag that moves a slider are applied there, so a caller reads a value
/// that is already current instead of reconstructing it.
///
/// The text field's state is boxed, and that is the only reason this enum is
/// no longer `Copy`. A `String` is 24 bytes and the rest of a field's state
/// another 100; inlining it would widen *every* entry of a table that is
/// `None` for almost every node, to save an indirection taken once per
/// keystroke.
///
/// Most variants sit on the node the pointer hits. [`Control::RadioGroup`] is
/// the exception and the reason the pair exists: exclusive selection has to
/// clear a *sibling*, which the hit node cannot name. The value therefore
/// lives one level up, on the container, and each option carries only a
/// back-pointer to it — so the click still resolves in the two lookups
/// `drive_controls` has always done.
///
/// [`Control::Scrollbar`] goes one step further and stores no value: the
/// scroll offset it shows already belongs to the area, and duplicating it
/// would just be a mirror to keep in sync.
#[derive(Clone, Debug)]
pub(crate) enum Control {
    Checkbox { mark: Label, checked: bool },
    Slider { fill: NodeId, thumb: NodeId, value: f32 },
    TextField(Box<super::text_field::FieldState>),
    /// On the container. Holds every option's dot node, because moving the
    /// selection repaints two of them and only one is under the pointer.
    RadioGroup { dots: Vec<NodeId>, selected: usize, on: UiStyle, off: UiStyle },
    /// On one option row: which group, and which index of it.
    Radio { group: RadioGroup, index: usize },
    /// On the container. Same shape as `RadioGroup`, but the selection is
    /// spent on hiding panes rather than on swapping a dot's fill. The style
    /// is kept whole because both looks derive from it, which is cheaper than
    /// storing the four it would take to hold them.
    Tabs { headers: Vec<NodeId>, labels: Vec<Label>, panes: Vec<NodeId>, selected: usize,
        style: TabStyle },
    /// On one tab header: which strip, and which index of it.
    Tab { tabs: Tabs, index: usize },
    /// On the track. Stores no value — the area is where it lives; these are
    /// only what a press has to reach and what a re-fit has to resize.
    Scrollbar { area: NodeId, thumb: NodeId, min_px: f32 },
}

impl Checkbox {
    /// Whether it is ticked.
    pub fn checked(self, ui: &UiCore) -> bool {
        let Control::Checkbox { checked, .. } = ui.control(self.0) else {
            unreachable!("Checkbox handle over a non-checkbox")
        };
        *checked
    }

    /// Set it, as a click would. Returns before touching anything when the
    /// value already holds, so restating it costs a comparison.
    pub fn set_checked(self, ui: &mut UiCore, checked: bool) {
        let Control::Checkbox { mark, checked: was } = ui.control(self.0) else {
            unreachable!("Checkbox handle over a non-checkbox")
        };
        let (mark, was) = (*mark, *was);
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
        *value
    }

    /// Set it, as a drag would; `value` is clamped to `0.0..=1.0`. Returns
    /// before touching anything when the value already holds, so a still
    /// slider costs no relayout.
    pub fn set_value(self, ui: &mut UiCore, value: f32) {
        let Control::Slider { fill, thumb, value: was } = ui.control(self.0) else {
            unreachable!("Slider handle over a non-slider")
        };
        let (fill, thumb, was) = (*fill, *thumb, *was);
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

impl RadioGroup {
    /// Index of the selected option. A group always has exactly one, so this
    /// is never `None`; a group built from no options reports 0.
    pub fn selected(self, ui: &UiCore) -> usize {
        let Control::RadioGroup { selected, .. } = ui.control(self.0) else {
            unreachable!("RadioGroup handle over a non-group")
        };
        *selected
    }

    /// Select by index, as a click would. An index already selected or past
    /// the end touches nothing, so restating the selection is a comparison.
    ///
    /// Two `set_background` calls and no relayout: the dot the selection left
    /// and the one it entered swap fills, and both nodes keep their size.
    pub fn set_selected(self, ui: &mut UiCore, i: usize) {
        let Control::RadioGroup { dots, selected, on, off } = ui.control(self.0) else {
            unreachable!("RadioGroup handle over a non-group")
        };
        if *selected == i || i >= dots.len() {
            return;
        }
        let (leaving, entering, on, off) = (dots[*selected], dots[i], *on, *off);
        ui.set_background(leaving, off);
        ui.set_background(entering, on);

        let Some(Some(Control::RadioGroup { selected, .. })) =
            ui.controls.get_mut(self.0.idx as usize)
        else {
            unreachable!("group control vanished between reads")
        };
        *selected = i;
    }
}

impl Tabs {
    /// Index of the open tab. A strip always has exactly one; a strip built
    /// from no labels reports 0.
    pub fn selected(self, ui: &UiCore) -> usize {
        let Control::Tabs { selected, .. } = ui.control(self.0) else {
            unreachable!("Tabs handle over a non-strip")
        };
        *selected
    }

    /// The container for tab `i`'s content — build into it like any node.
    /// Valid from the moment [`UiCore::tabs`] returns, whether or not that
    /// tab is the open one.
    pub fn pane(self, ui: &UiCore, i: usize) -> NodeId {
        let Control::Tabs { panes, .. } = ui.control(self.0) else {
            unreachable!("Tabs handle over a non-strip")
        };
        panes[i]
    }

    /// Open a tab by index, as a click would. An index already open or past
    /// the end touches nothing.
    ///
    /// Two restyled headers and two collapsed boxes: the strip is a fill swap
    /// with no layout, and only the panes relayout.
    pub fn set_selected(self, ui: &mut UiCore, i: usize) {
        let Control::Tabs { headers, labels, panes, selected, style } = ui.control(self.0) else {
            unreachable!("Tabs handle over a non-strip")
        };
        if *selected == i || i >= panes.len() {
            return;
        }
        let (was, style) = (*selected, *style);
        let (lh, eh, lp, ep) = (headers[was], headers[i], panes[was], panes[i]);
        let (ll, el) = (labels[was], labels[i]);
        ui.set_state_style(lh, style.look(false));
        ui.set_state_style(eh, style.look(true));
        ll.set_color(ui, style.text_dim);
        el.set_color(ui, style.text);
        ui.set_visible(lp, false);
        ui.set_visible(ep, true);

        let Some(Some(Control::Tabs { selected, .. })) = ui.controls.get_mut(self.0.idx as usize)
        else {
            unreachable!("tab control vanished between reads")
        };
        *selected = i;
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

/// How [`UiCore::radio_group`] looks. The row roles match a checkbox's — the
/// two differ in shape, not in palette, which is the point of naming roles.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RadioStyle {
    pub idle: u32,
    pub hover: u32,
    pub held: u32,
    pub text: u32,
    /// The selected dot. Reads against `circle_fill`, not against `idle`.
    pub mark: u32,
    pub circle_fill: u32,
    pub circle_border: u32,
    /// Diameter of the ring, in px.
    pub circle_px: f32,
    /// Diameter of the dot inside it.
    pub dot_px: f32,
    /// Between the ring and the label.
    pub gap: f32,
    /// Between options.
    pub row_gap: f32,
    pub text_px: f32,
    pub padding: f32,
    pub radius: f32,
}

impl From<Theme> for RadioStyle {
    fn from(t: Theme) -> Self {
        Self {
            idle: rgba(0, 0, 0, 0),
            hover: t.control_hover,
            held: t.control_held,
            text: t.text,
            mark: t.accent,
            circle_fill: t.control,
            circle_border: t.outline,
            circle_px: 11.0,
            dot_px: 5.0,
            gap: 6.0,
            row_gap: 2.0,
            text_px: t.text_px,
            padding: 2.0,
            radius: t.radius,
        }
    }
}

impl Default for RadioStyle {
    fn default() -> Self {
        theme().into()
    }
}

/// How [`UiCore::tabs`] looks. A header is a button that stays pressed, so
/// the roles are a button's, plus one for the tab that is open.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TabStyle {
    pub idle: u32,
    pub hover: u32,
    pub held: u32,
    /// The open tab. Matches the pane's own surface, so the two read as one
    /// shape rather than as a button above a box.
    pub selected: u32,
    /// The open tab's label. Closed ones read as `text_dim` — the strongest
    /// of the two signals, since a fill this dark is easy to miss.
    pub text: u32,
    pub text_dim: u32,
    pub text_px: f32,
    /// Between the strip and the pane.
    pub gap: f32,
    pub padding: f32,
    pub radius: f32,
}

impl TabStyle {
    /// A header's three pointer looks, open or closed. The open tab wears one
    /// look in all three: it is already where the pointer would take you, so
    /// hover has nothing left to promise.
    pub(crate) fn look(self, open: bool) -> StateStyle {
        let base = UiStyle::fill(self.idle).radius(self.radius);
        match open {
            true => StateStyle::fills(base, self.selected, self.selected, self.selected),
            false => StateStyle::fills(base, self.idle, self.hover, self.held),
        }
    }
}

impl From<Theme> for TabStyle {
    fn from(t: Theme) -> Self {
        Self {
            idle: rgba(0, 0, 0, 0),
            hover: t.control_hover,
            held: t.control_held,
            selected: t.control,
            text: t.text,
            text_dim: t.text_dim,
            text_px: t.text_px,
            gap: 4.0,
            padding: t.pad,
            radius: t.radius,
        }
    }
}

impl Default for TabStyle {
    fn default() -> Self {
        theme().into()
    }
}

/// How [`UiCore::scrollbar`] looks and measures.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ScrollbarStyle {
    /// The gutter. Always drawn, so the layout does not jump when content
    /// grows past the viewport.
    pub track: u32,
    pub thumb: u32,
    pub thumb_hover: u32,
    /// Gutter width in px. The height comes from the parent, so put the bar
    /// in a flex row beside the area it mirrors.
    pub width: f32,
    /// Shortest the thumb may get. Without it a long enough list leaves
    /// nothing to grab.
    pub min_thumb_px: f32,
    pub radius: f32,
}

impl From<Theme> for ScrollbarStyle {
    fn from(t: Theme) -> Self {
        Self {
            track: t.control,
            thumb: t.text_dim,
            thumb_hover: t.text,
            width: 8.0,
            min_thumb_px: 18.0,
            radius: 4.0,
        }
    }
}

impl Default for ScrollbarStyle {
    fn default() -> Self {
        theme().into()
    }
}

impl UiCore {
    /// A control node's state. Infallible in practice — only the control
    /// constructors mint the handles that reach it, and a handle into a
    /// removed subtree is caught by `live` first.
    ///
    /// Borrowed rather than returned by value: a text field's state is not
    /// `Copy`, and reading its string should not clone it.
    pub(crate) fn control(&self, n: NodeId) -> &Control {
        self.controls
            .get(self.live(n))
            .and_then(|c| c.as_ref())
            .unwrap_or_else(|| panic!("NodeId {} is not a control", n.idx))
    }

    pub(crate) fn set_control(&mut self, n: NodeId, c: Control) {
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
    /// position it was released at. `pressed` distinguishes the frame the
    /// gesture *began* on, which is the only frame a text field starts a
    /// fresh selection rather than extending one.
    ///
    /// Two lookups, whatever the tree holds: a control only changes under
    /// the pointer that is on it.
    pub(crate) fn drive_controls(&mut self, dragging: Option<NodeId>, pressed: bool) {
        if let Some(n) = self.pointer.clicked {
            match self.controls.get(n.idx as usize) {
                Some(Some(Control::Checkbox { checked, .. })) => {
                    let now = !*checked;
                    Checkbox::from_node(n).set_checked(self, now);
                }
                // The one control whose click lands somewhere other than the
                // node it hit: the group owns the value, so the option only
                // forwards.
                Some(Some(Control::Radio { group, index })) => {
                    let (group, index) = (*group, *index);
                    group.set_selected(self, index);
                }
                Some(Some(Control::Tab { tabs, index })) => {
                    let (tabs, index) = (*tabs, *index);
                    tabs.set_selected(self, index);
                }
                _ => {}
            }
        }
        if let Some(n) = dragging {
            match self.controls.get(n.idx as usize) {
                Some(Some(Control::Slider { .. })) => {
                    // Absolute, not incremental: the value is where the
                    // pointer is along the track, so pressing anywhere jumps
                    // there and the thumb cannot drift from the cursor the
                    // way accumulated deltas do.
                    let r = self.node_rect(n);
                    let v = (self.pointer.pos[0] - r[0]) / r[2].max(1.0);
                    Slider::from_node(n).set_value(self, v);
                }
                Some(Some(Control::TextField(_))) => {
                    // The second click of a double takes the whole value, so
                    // typing replaces it. Otherwise the same absolute rule as
                    // a slider, one dimension coarser: the press drops the
                    // caret and everything after it drags a selection out of
                    // that origin — until someone claims the gesture, which is
                    // how a number is scrubbed instead of selected.
                    let x = self.pointer.pos[0];
                    if self.click_count(n) >= 2 {
                        self.field_select_all(n);
                    } else if pressed || !self.drag_claimed(n) {
                        self.field_point(n, x, !pressed);
                    }
                }
                Some(Some(Control::Scrollbar { area, thumb, .. })) => {
                    // Absolute again, but the thumb has length, so it is its
                    // *centre* that follows the pointer — which also makes a
                    // press on bare track jump there instead of paging.
                    let (area, thumb) = (*area, *thumb);
                    let (r, th) = (self.node_rect(n), self.node_rect(thumb)[3]);
                    let max = self.max_scroll(area)[1];
                    let t = (self.pointer.pos[1] - r[1] - th * 0.5) / (r[3] - th).max(1.0);
                    let to = t.clamp(0.0, 1.0) * max;
                    self.scroll_by(area, [0.0, to - self.scroll_offset(area)[1]]);
                }
                _ => {}
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
        self.set_events(n, Events::CLICK | Events::HOVER);
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
        self.set_events(row, Events::CLICK | Events::HOVER);
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

        self.set_events(track, Events::CLICK | Events::HOVER);
        self.set_control(track, Control::Slider { fill, thumb, value: 0.0 });
        Slider::from_node(track)
    }

    /// A radio group: one column of options, exactly one of them selected.
    ///
    /// The whole set is built in one call because the options are not
    /// independent — a click on any of them clears the rest, so there is no
    /// meaningful half-built group. Selection is by index into `options`,
    /// which is what a caller matches on anyway:
    ///
    /// ```ignore
    /// let g = ui.radio_group(panel, &["linear", "srgb", "raw"], RadioStyle::default());
    /// // …later, and from anywhere:
    /// match g.selected(&ui) { 0 => linear(), 1 => srgb(), _ => raw() }
    /// g.set_selected(&mut ui, from_settings_file);
    /// ```
    ///
    /// The group is a plain node underneath, so re-styling it to a row is
    /// `set_node_style(g, …)` — nothing here assumes a column.
    pub fn radio_group(
        &mut self,
        parent: impl Into<NodeId>,
        options: &[&str],
        style: RadioStyle,
    ) -> RadioGroup {
        use super::style::{px, zero, AlignItems, Display, FlexDirection, JustifyContent, Rect,
            Size, Style};

        let group = self.node(
            parent,
            Style {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                gap: Size { width: zero(), height: px(style.row_gap) },
                ..Default::default()
            },
        );

        // The dot is a node rather than a glyph, so selecting swaps a fill
        // instead of retyping a run — `set_background` skips taffy, and the
        // dot keeps its box whether it is visible or not.
        let dot_radius = style.dot_px * 0.5;
        let on = UiStyle::fill(style.mark).radius(dot_radius);
        let off = UiStyle::fill(rgba(0, 0, 0, 0)).radius(dot_radius);

        let mut dots = Vec::with_capacity(options.len());
        for (i, text) in options.iter().enumerate() {
            let row = self.node(
                group,
                Style {
                    display: Display::Flex,
                    align_items: Some(AlignItems::CENTER),
                    gap: Size { width: px(style.gap), height: zero() },
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

            let circle = self.node(
                row,
                Style {
                    display: Display::Flex,
                    justify_content: Some(JustifyContent::CENTER),
                    align_items: Some(AlignItems::CENTER),
                    size: Size { width: px(style.circle_px), height: px(style.circle_px) },
                    flex_shrink: 0.0,
                    ..Default::default()
                },
            );
            self.set_background(
                circle,
                UiStyle::fill(style.circle_fill)
                    .border(style.circle_border, 1.0)
                    .radius(style.circle_px * 0.5),
            );

            let dot = self.node(
                circle,
                Style {
                    size: Size { width: px(style.dot_px), height: px(style.dot_px) },
                    flex_shrink: 0.0,
                    ..Default::default()
                },
            );
            self.set_background(dot, if i == 0 { on } else { off });

            self.label(row, style.text_px, style.text, text);
            self.set_events(row, Events::CLICK | Events::HOVER);
            self.set_control(row, Control::Radio { group: RadioGroup::from_node(group), index: i });
            dots.push(dot);
        }

        self.set_control(group, Control::RadioGroup { dots, selected: 0, on, off });
        RadioGroup::from_node(group)
    }

    /// A tab strip with one pane per label, the first of them open.
    ///
    /// Built in one call for the same reason a radio group is — the tabs are
    /// not independent — but it hands the panes back, because their contents
    /// are the caller's:
    ///
    /// ```ignore
    /// let t = ui.tabs(panel, &["Scene", "Assets"], TabStyle::default());
    /// let (scene, assets) = (t.pane(&ui, 0), t.pane(&ui, 1));
    /// ui.label(scene, 13.0, text, "…");   // build into them like any node
    /// ```
    ///
    /// Nothing to poll: switching is applied before any component runs, and a
    /// closed pane is collapsed rather than skipped, so the contents can stay
    /// bound and simply stop existing on screen.
    pub fn tabs(
        &mut self,
        parent: impl Into<NodeId>,
        labels: &[&str],
        style: TabStyle,
    ) -> Tabs {
        use super::style::{px, zero, Display, FlexDirection, Rect, Size, Style};

        let column = Style {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            gap: Size { width: zero(), height: px(style.gap) },
            ..Default::default()
        };
        let tabs = self.node(parent, column.clone());
        let strip = self.node(
            tabs,
            Style { display: Display::Flex, gap: Size { width: px(2.0), height: zero() },
                ..Default::default() },
        );

        let (mut headers, mut texts, mut panes) = (Vec::new(), Vec::new(), Vec::new());
        for (i, text) in labels.iter().enumerate() {
            let header = self.node(
                strip,
                Style { display: Display::Flex, padding: Rect::length(style.padding),
                    ..Default::default() },
            );
            let open = i == 0;
            self.set_state_style(header, style.look(open));
            let label =
                self.label(header, style.text_px, if open { style.text } else { style.text_dim },
                    text);
            self.set_events(header, Events::CLICK | Events::HOVER);
            self.set_control(header, Control::Tab { tabs: Tabs::from_node(tabs), index: i });

            let pane = self.node(tabs, column.clone());
            self.set_visible(pane, open);
            headers.push(header);
            texts.push(label);
            panes.push(pane);
        }

        self.set_control(tabs,
            Control::Tabs { headers, labels: texts, panes, selected: 0, style });
        Tabs::from_node(tabs)
    }

    /// A vertical scrollbar showing — and driving — `area`'s scroll.
    ///
    /// It is built under `parent` rather than inside `area`, because anything
    /// added to a scroll area scrolls with its contents. Put the two side by
    /// side and the bar is a gutter that stretches to the area's height:
    ///
    /// ```ignore
    /// let row = ui.node(panel, flex_row());
    /// let area = ui.scroll_area(row, viewport);
    /// ui.scrollbar(row, area, ScrollbarStyle::default());
    /// ```
    ///
    /// Nothing else to call: the thumb is re-fitted after every layout and
    /// every scroll, so a wheel, a resize or a list that grew all reach it.
    /// When the content fits, the thumb takes zero height and is culled.
    pub fn scrollbar(
        &mut self,
        parent: impl Into<NodeId>,
        area: impl Into<NodeId>,
        style: ScrollbarStyle,
    ) -> Scrollbar {
        use super::style::{percent, px, Size, Style};

        let track = self.node(
            parent,
            Style {
                size: Size { width: px(style.width), height: super::style::TaffyAuto::AUTO },
                flex_shrink: 0.0,
                ..Default::default()
            },
        );
        self.set_background(track, UiStyle::fill(style.track).radius(style.radius));

        // The thumb moves on the track's *group* offset, not its layout, so
        // scrolling a 10 000-row list still writes one `ui_group` record.
        // That is also what clips it to the gutter for free.
        self.open_content_group(track);

        let thumb = self.node(
            track,
            Style {
                size: Size { width: percent(1.0_f32), height: px(0.0) },
                flex_shrink: 0.0,
                ..Default::default()
            },
        );
        self.set_state_style(
            thumb,
            StateStyle::fills(
                UiStyle::fill(style.thumb).radius(style.radius),
                style.thumb,
                style.thumb_hover,
                style.thumb_hover,
            ),
        );
        // Split on purpose: the track takes the press so a click on bare
        // gutter still moves the thumb, and the thumb takes hover so only it
        // lights up. During a drag it is centred on the pointer, so it stays
        // lit without needing a `held` of its own.
        self.set_events(thumb, Events::HOVER);
        self.set_events(track, Events::CLICK);

        let area = area.into();
        self.set_control(track, Control::Scrollbar { area, thumb, min_px: style.min_thumb_px });
        let bar = Scrollbar::from_node(track);
        self.scrollbars.push(bar);
        bar
    }

    /// The area a bar mirrors, for `free_subtree`'s liveness sweep.
    pub(crate) fn bar_area(&self, bar: Scrollbar) -> NodeId {
        let Control::Scrollbar { area, .. } = self.control(bar.0) else {
            unreachable!("Scrollbar handle over a non-scrollbar")
        };
        *area
    }

    pub(crate) fn sync_scrollbars(&mut self) {
        for i in 0..self.scrollbars.len() {
            self.sync_scrollbar(self.scrollbars[i]);
        }
    }

    /// Re-fit one thumb to its area. Both writes are gated, so a bar whose
    /// area did not move costs two comparisons.
    fn sync_scrollbar(&mut self, bar: Scrollbar) {
        let Control::Scrollbar { area, thumb, min_px } = self.control(bar.0) else {
            unreachable!("Scrollbar handle over a non-scrollbar")
        };
        let (area, thumb, min_px) = (*area, *thumb, *min_px);
        let (track, view) = (self.node_rect(bar)[3], self.node_rect(area)[3]);
        let max = self.max_scroll(area)[1];

        // Thumb : track as viewport : content. Nothing to scroll gives it no
        // height at all, and a zero-area quad is culled — so the gutter stays
        // put and the indicator simply is not there.
        let h = match max > 0.0 && track > 0.0 {
            true => (track * view / (view + max)).clamp(min_px.min(track), track),
            false => 0.0,
        };
        let mut s = self.node_style(thumb);
        s.size.height = super::style::px(h);
        self.set_node_style(thumb, s);

        let t = if max > 0.0 { self.scroll_offset(area)[1] / max } else { 0.0 };
        self.set_content_offset(bar, [0.0, -t * (track - h)]);
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
        *mark
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

        core.update_pointer(p, true, false, 0.0, 0.0);
        assert!(!cb.checked(&core), "press alone must not toggle");
        core.update_pointer(p, false, true, 0.0, 0.0);
        assert!(cb.checked(&core), "the release completes the click");

        core.update_pointer(p, true, false, 0.0, 0.0);
        core.update_pointer(p, false, true, 0.0, 0.0);
        assert!(!cb.checked(&core), "and back");
    }

    /// A press dragged off the control cancels, so it must not toggle
    /// either — the value follows `clicked`, not `down_on`.
    #[test]
    fn a_cancelled_click_leaves_the_value_alone() {
        let mut core = UiCore::new();
        let cb = checkbox(&mut core);
        let r = core.node_rect(cb);

        core.update_pointer([r[0] + 2.0, r[1] + r[3] * 0.5], true, false, 0.0, 0.0);
        core.update_pointer([r[0] + r[2] + 200.0, r[1] + 400.0], false, false, 0.0, 0.0);
        core.update_pointer([r[0] + r[2] + 200.0, r[1] + 400.0], false, true, 0.0, 0.0);
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
        core.set_events(btn, Events::CLICK | Events::HOVER);
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

        core.update_pointer([r[0] + r[2] * 0.5, y], true, false, 0.0, 0.0);
        assert_eq!(sl.value(&core), 0.5, "the press alone jumps there");

        core.update_pointer([r[0] + r[2] * 0.25, y], false, false, 0.0, 0.0);
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

        core.update_pointer([r[0] + 1.0, y], true, false, 0.0, 0.0);
        core.update_pointer([r[0] + r[2] + 500.0, y], false, false, 0.0, 0.0);
        assert_eq!(sl.value(&core), 1.0, "clamped, still dragging");

        // Moving and releasing on the same frame still commits: `dragging`
        // is captured before the release clears the press.
        core.update_pointer([r[0] - 500.0, y], false, true, 0.0, 0.0);
        assert_eq!(sl.value(&core), 0.0);

        core.update_pointer([r[0] + r[2] * 0.5, y], false, false, 0.0, 0.0);
        assert_eq!(sl.value(&core), 0.0, "released — moving no longer drags it");
    }

    /// Nothing about the pointer touches a control it is not on.
    #[test]
    fn dragging_elsewhere_leaves_a_slider_alone() {
        let mut core = UiCore::new();
        let sl = slider(&mut core);
        let root = core.root();
        let other = core.node(root, Style::default());
        core.set_events(other, Events::CLICK | Events::HOVER);
        sl.set_value(&mut core, 0.6);
        core.run_layout([400.0, 400.0]);

        let r = core.node_rect(sl);
        core.update_pointer([r[0] + r[2] + 50.0, r[1] + 200.0], true, false, 0.0, 0.0);
        core.update_pointer([r[0] + r[2] * 0.1, r[1]], false, false, 0.0, 0.0);
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

        core.update_pointer(start, true, false, 0.0, 0.0);
        let d = core.drag(sl).expect("dragging");
        assert_eq!(d.delta(), [0.0, 0.0]);
        assert!(!d.beyond(4.0));

        core.update_pointer([start[0] + 12.0, start[1]], false, false, 0.0, 0.0);
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
        let fill = *fill;

        sl.set_value(&mut core, 0.25);
        core.run_layout([400.0, 400.0]);
        let quarter = core.node_rect(fill)[2];

        sl.set_value(&mut core, 0.75);
        core.run_layout([400.0, 400.0]);
        let three_quarters = core.node_rect(fill)[2];

        assert!((quarter - track * 0.25).abs() < 0.5, "quarter fill: {quarter}");
        assert!((three_quarters - track * 0.75).abs() < 0.5, "three-quarter fill");
    }

    fn radio_group(core: &mut UiCore) -> RadioGroup {
        let root = core.root();
        let g = core.radio_group(root, &["linear", "srgb", "raw"], RadioStyle::default());
        core.run_layout([400.0, 400.0]);
        g
    }

    /// The option rows, which only the store knows about — the constructor
    /// hands back the group alone. Found by asking what the pointer would hit
    /// over each dot, since the row is the only part that listens.
    fn rows_of(core: &UiCore, g: RadioGroup) -> Vec<NodeId> {
        let Control::RadioGroup { dots, .. } = core.control(g.0) else {
            unreachable!()
        };
        dots.iter()
            .map(|&d| {
                let r = core.node_rect(d);
                core.hit_test([r[0] + r[2] * 0.5, r[1] + r[3] * 0.5])
                    .expect("the row takes the hit, not the dot")
            })
            .collect()
    }

    /// Which dots are actually painted, as the screen sees it. Reading the
    /// fill rather than the control is the point: the value and the picture
    /// are two different pieces of state, and only one of them is visible.
    fn lit(core: &UiCore, g: RadioGroup) -> Vec<bool> {
        let Control::RadioGroup { dots, on, .. } = core.control(g.0) else {
            unreachable!()
        };
        dots.iter()
            .map(|&d| core.style.get(core.paint_slots(d).0.expect("a dot has a fill")).fill == on.fill)
            .collect()
    }

    fn click(core: &mut UiCore, n: NodeId) {
        let r = core.node_rect(n);
        let p = [r[0] + 2.0, r[1] + r[3] * 0.5];
        core.update_pointer(p, true, false, 0.0, 0.0);
        core.update_pointer(p, false, true, 0.0, 0.0);
    }

    /// The capability the widget exists to force: one click changes a node
    /// the pointer never touched. Every other control begins and ends on the
    /// node under the cursor.
    #[test]
    fn clicking_one_option_clears_its_siblings() {
        let mut core = UiCore::new();
        let g = radio_group(&mut core);
        let rows = rows_of(&core, g);

        assert_eq!(g.selected(&core), 0, "a group starts with its first option");
        assert_eq!(lit(&core, g), [true, false, false]);

        click(&mut core, rows[2]);
        assert_eq!(g.selected(&core), 2);
        assert_eq!(lit(&core, g), [false, false, true], "the old dot went out");

        click(&mut core, rows[1]);
        assert_eq!(g.selected(&core), 1);
        assert_eq!(lit(&core, g), [false, true, false]);
    }

    /// Unlike a checkbox, clicking the selected option is not a toggle —
    /// there is no state for a group with nothing selected to be in.
    #[test]
    fn re_clicking_the_selection_keeps_it() {
        let mut core = UiCore::new();
        let g = radio_group(&mut core);
        let rows = rows_of(&core, g);

        click(&mut core, rows[0]);
        assert_eq!(g.selected(&core), 0);
        assert_eq!(lit(&core, g), [true, false, false]);
    }

    /// A press dragged off cancels, same rule as the checkbox: the value
    /// follows `clicked`, not `down_on`.
    #[test]
    fn a_cancelled_click_leaves_the_selection_alone() {
        let mut core = UiCore::new();
        let g = radio_group(&mut core);
        let r = core.node_rect(rows_of(&core, g)[2]);
        let away = [r[0] + r[2] + 200.0, r[1] + 400.0];

        core.update_pointer([r[0] + 2.0, r[1] + r[3] * 0.5], true, false, 0.0, 0.0);
        core.update_pointer(away, false, false, 0.0, 0.0);
        core.update_pointer(away, false, true, 0.0, 0.0);
        assert_eq!(g.selected(&core), 0);
    }

    /// Owning the value does not close it off — a settings file or a keybind
    /// moves it the same way a click does, dots and all.
    #[test]
    fn the_dots_follow_a_selection_set_from_outside() {
        let mut core = UiCore::new();
        let g = radio_group(&mut core);

        g.set_selected(&mut core, 1);
        assert_eq!(g.selected(&core), 1);
        assert_eq!(lit(&core, g), [false, true, false]);

        // Past the end is ignored rather than clamped: landing on the last
        // option would be a silently wrong answer, not a safe one.
        g.set_selected(&mut core, 99);
        assert_eq!(g.selected(&core), 1);
    }

    /// The whole row is the target, so the text selects too — but the ring
    /// must not become a second, inner one.
    #[test]
    fn clicking_the_label_hits_the_option_row() {
        let mut core = UiCore::new();
        let g = radio_group(&mut core);
        let row = rows_of(&core, g)[1];
        let r = core.node_rect(row);

        assert_eq!(core.hit_test([r[0] + r[2] - 1.0, r[1] + r[3] * 0.5]), Some(row), "label area");
        assert_eq!(core.hit_test([r[0] + 1.0, r[1] + r[3] * 0.5]), Some(row), "ring too");
    }

    /// Same contract as every other control: restating the value it already
    /// holds must cost a comparison and no bytes.
    #[test]
    fn restating_the_selection_uploads_nothing() {
        let mut core = UiCore::new();
        let g = radio_group(&mut core);
        g.set_selected(&mut core, 2);
        core.run_layout([400.0, 400.0]);

        let (mut stage, mut dirty) = (vec![0u32; 1 << 16], vec![0u32; 1 << 10]);
        let clean = (i64::MAX, -1);
        core.quad.upload(&mut stage, &mut dirty);
        core.style.upload(&mut stage, &mut dirty);

        for _ in 0..8 {
            g.set_selected(&mut core, 2);
            core.run_layout([400.0, 400.0]);
        }
        assert_eq!(core.quad.upload(&mut stage, &mut dirty), clean);
        assert_eq!(core.style.upload(&mut stage, &mut dirty), clean);
    }

    /// Pins the cost of the shared-selection move. A dot built from a glyph
    /// would relayout its run, and a dot that hid by going zero-width would
    /// dirty a quad — this catches either regression, which no behavioural
    /// test above would.
    #[test]
    fn moving_the_selection_dirties_two_styles_and_no_quads() {
        let mut core = UiCore::new();
        let g = radio_group(&mut core);
        let (mut stage, mut dirty) = (vec![0u32; 1 << 16], vec![0u32; 256]);
        core.quad.upload(&mut stage, &mut dirty);
        core.style.upload(&mut stage, &mut dirty);
        dirty.fill(0);

        g.set_selected(&mut core, 2);
        core.run_layout([400.0, 400.0]);

        core.style.upload(&mut stage, &mut dirty);
        let styles: u32 = dirty.iter().map(|w| w.count_ones()).sum();
        dirty.fill(0);
        core.quad.upload(&mut stage, &mut dirty);
        let quads: u32 = dirty.iter().map(|w| w.count_ones()).sum();

        assert_eq!(styles, 2, "the dot it left and the dot it entered");
        assert_eq!(quads, 0, "no geometry moved, so taffy never ran");
    }

    /// A group with no options is a caller mistake, not a crash: `selected`
    /// answers and `set_selected` declines, rather than indexing an empty
    /// list.
    #[test]
    fn an_empty_group_is_inert() {
        let mut core = UiCore::new();
        let root = core.root();
        let g = core.radio_group(root, &[], RadioStyle::default());
        core.run_layout([400.0, 400.0]);

        assert_eq!(g.selected(&core), 0);
        g.set_selected(&mut core, 0);
        g.set_selected(&mut core, 3);
        assert_eq!(g.selected(&core), 0);
    }

    // ── Tabs ────────────────────────────────────────────────────────────

    /// Two tabs, each pane holding a label that names it — so what is on
    /// screen answers which tab is open without reading the control.
    fn tabs(core: &mut UiCore) -> Tabs {
        let root = core.root();
        let t = core.tabs(root, &["scene", "assets"], TabStyle::default());
        for (i, body) in ["scene body", "assets body"].iter().enumerate() {
            let pane = t.pane(core, i);
            core.label(pane, 13.0, 0xffff_ffff, body);
        }
        core.run_layout([400.0, 400.0]);
        t
    }

    /// Every string the screen actually shows.
    fn visible(core: &UiCore) -> Vec<String> {
        core.text_nodes().into_iter().map(|(_, s, _)| s).collect()
    }

    /// The capability: selection spent on layout. A closed pane is not
    /// skipped by the painter, it has no box — so its contents can stay bound
    /// and simply stop existing.
    #[test]
    fn a_closed_pane_shows_nothing() {
        let mut core = UiCore::new();
        let t = tabs(&mut core);

        assert!(visible(&core).contains(&"scene body".to_string()));
        assert!(!visible(&core).contains(&"assets body".to_string()));
        assert_eq!(core.node_rect(t.pane(&core, 1))[3], 0.0, "collapsed, not just unpainted");
    }

    #[test]
    fn clicking_a_tab_swaps_the_panes() {
        let mut core = UiCore::new();
        let t = tabs(&mut core);
        let Control::Tabs { headers, .. } = core.control(t.0) else {
            unreachable!()
        };
        let header = headers[1];

        click(&mut core, header);
        core.run_layout([400.0, 400.0]);

        assert_eq!(t.selected(&core), 1);
        assert!(visible(&core).contains(&"assets body".to_string()));
        assert!(!visible(&core).contains(&"scene body".to_string()));
    }

    /// Collapsing the pane collapses everything under it. Both panes sit at
    /// the same place, so a tab that has been open once would otherwise leave
    /// a live button exactly where the open tab's now is.
    #[test]
    fn a_closed_panes_contents_take_no_hits() {
        use super::super::style::{px, Size, Style};
        let mut core = UiCore::new();
        let root = core.root();
        let t = core.tabs(root, &["a", "b"], TabStyle::default());
        let box_100 =
            Style { size: Size { width: px(100.0), height: px(40.0) }, ..Default::default() };
        let buttons: Vec<NodeId> = (0..2)
            .map(|i| {
                let pane = t.pane(&core, i);
                let b = core.node(pane, box_100.clone());
                core.set_events(b, Events::CLICK);
                b
            })
            .collect();

        // Open b, so its button gets a real box, then go back to a.
        for i in [1, 0] {
            t.set_selected(&mut core, i);
            core.run_layout([400.0, 400.0]);
        }

        let r = core.node_rect(buttons[0]);
        assert_eq!(core.node_rect(buttons[1]), [0.0; 4], "b's button collapsed with its pane");
        assert_eq!(
            core.hit_test([r[0] + r[2] * 0.5, r[1] + r[3] * 0.5]),
            Some(buttons[0]),
            "the open pane's button, with nothing of b's stacked on it"
        );
    }

    /// The strip half of a switch: two header fills and the glyphs of the two
    /// labels that changed colour. Panes relayout — they have to, that is what
    /// hiding one means — but the strip itself must not move.
    #[test]
    fn switching_restyles_two_headers_and_their_glyphs() {
        let mut core = UiCore::new();
        let t = tabs(&mut core);
        let Control::Tabs { headers, .. } = core.control(t.0) else {
            unreachable!()
        };
        let (headers, before) = (headers.clone(), core.node_rect(headers[1]));

        let (mut stage, mut dirty) = (vec![0u32; 1 << 16], vec![0u32; 1 << 10]);
        core.style.upload(&mut stage, &mut dirty);

        t.set_selected(&mut core, 1);
        let (_, hi) = core.style.upload(&mut stage, &mut dirty);
        let styles: u32 = dirty[..=hi.max(0) as usize].iter().map(|w| w.count_ones()).sum();

        core.run_layout([400.0, 400.0]);
        let expected = 2 + "scene".len() + "assets".len();
        assert_eq!(styles as usize, expected, "two fills, and the glyphs that dimmed and lit");
        assert_eq!(core.node_rect(headers[1]), before, "the strip never moves");
    }

    /// Same contract as every other control: restating the selection is a
    /// comparison, so a component that re-asserts its tab every frame is free.
    #[test]
    fn restating_the_open_tab_uploads_nothing() {
        let mut core = UiCore::new();
        let t = tabs(&mut core);
        t.set_selected(&mut core, 1);
        core.run_layout([400.0, 400.0]);

        let (mut stage, mut dirty) = (vec![0u32; 1 << 16], vec![0u32; 1 << 10]);
        let clean = (i64::MAX, -1);
        core.quad.upload(&mut stage, &mut dirty);
        core.style.upload(&mut stage, &mut dirty);

        for _ in 0..8 {
            t.set_selected(&mut core, 1);
            core.run_layout([400.0, 400.0]);
        }
        assert_eq!(core.quad.upload(&mut stage, &mut dirty), clean);
        assert_eq!(core.style.upload(&mut stage, &mut dirty), clean);
    }

    /// A strip with no tabs is a caller mistake, not a crash — same as an
    /// empty radio group.
    #[test]
    fn an_empty_strip_is_inert() {
        let mut core = UiCore::new();
        let root = core.root();
        let t = core.tabs(root, &[], TabStyle::default());
        core.run_layout([400.0, 400.0]);

        assert_eq!(t.selected(&core), 0);
        t.set_selected(&mut core, 2);
        assert_eq!(t.selected(&core), 0);
    }

    // ── Scrollbar ───────────────────────────────────────────────────────

    /// A 100px viewport over `content` px of content, with a bar beside it.
    ///
    /// Laid out twice on purpose: the first pass is what tells the thumb how
    /// tall to be, and resizing it dirties taffy — so a bar settles one frame
    /// after the content it measures, and every test here starts settled.
    fn scrollbar(core: &mut UiCore, content: f32) -> (NodeId, Scrollbar) {
        use crate::ui::style::{px, Display, FlexDirection, Size, Style, TaffyAuto};

        let root = core.root();
        let row = core.node(
            root,
            Style {
                display: Display::Flex,
                size: Size { width: px(200.0), height: px(100.0) },
                ..Default::default()
            },
        );
        // A column, as `RowList` builds it: taffy only accumulates content
        // size along the main axis, so a row-direction area reports no
        // vertical overflow to scroll.
        let area = core.scroll_area(
            row,
            Style {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                size: Size { width: px(100.0), height: px(100.0) },
                flex_shrink: 0.0,
                ..Default::default()
            },
        );
        core.node(
            area,
            Style {
                size: Size { width: TaffyAuto::AUTO, height: px(content) },
                flex_shrink: 0.0,
                ..Default::default()
            },
        );
        let bar = core.scrollbar(row, area, ScrollbarStyle::default());
        core.run_layout([400.0, 400.0]);
        core.run_layout([400.0, 400.0]);
        (area, bar)
    }

    /// Where the thumb actually draws, relative to the track. Its layout box
    /// never moves — the group offset is the whole of the movement, which is
    /// the point of building it this way.
    fn thumb_top(core: &UiCore, bar: Scrollbar) -> f32 {
        -core.scroll_offset(bar)[1]
    }

    fn thumb_height(core: &UiCore, bar: Scrollbar) -> f32 {
        let Control::Scrollbar { thumb, .. } = core.control(bar.into()) else {
            unreachable!()
        };
        core.node_rect(*thumb)[3]
    }

    /// The capability: the bar is a view of state it does not own, so it has
    /// to be correct without anyone telling it the content exists.
    #[test]
    fn the_thumb_is_as_much_of_the_track_as_the_viewport_is_of_the_content() {
        let mut core = UiCore::new();
        let (_, bar) = scrollbar(&mut core, 400.0);

        assert_eq!(core.node_rect(bar)[3], 100.0, "the gutter stretches");
        assert!(
            (thumb_height(&core, bar) - 25.0).abs() < 1.0,
            "100 of 400 visible: {}",
            thumb_height(&core, bar)
        );
        assert_eq!(thumb_top(&core, bar), 0.0, "unscrolled, so parked at the top");
    }

    /// Nobody touched the bar — the wheel went to the area — and the thumb
    /// still moved. That is the whole reason `sync_scrollbars` exists.
    #[test]
    fn a_wheel_the_bar_never_saw_moves_the_thumb() {
        let mut core = UiCore::new();
        let (area, bar) = scrollbar(&mut core, 400.0);
        let h = thumb_height(&core, bar);

        core.scroll_by(area, [0.0, 150.0]);
        let half = thumb_top(&core, bar);
        assert!((half - 0.5 * (100.0 - h)).abs() < 1.0, "halfway: {half}");

        core.scroll_by(area, [0.0, f32::MAX]);
        assert!(
            (thumb_top(&core, bar) - (100.0 - h)).abs() < 1.0,
            "the end of the content is the end of the track"
        );
    }

    /// Content shorter than the viewport has no thumb rather than a full-height
    /// one, so "nothing to scroll" reads as absence — and a zero-area quad is
    /// culled, so it costs nothing to leave there.
    #[test]
    fn content_that_fits_shows_no_thumb() {
        let mut core = UiCore::new();
        let (_, bar) = scrollbar(&mut core, 40.0);
        assert_eq!(thumb_height(&core, bar), 0.0);
        assert_eq!(thumb_top(&core, bar), 0.0);
    }

    /// A long enough list would compute a sub-pixel thumb. The floor is what
    /// keeps something grabbable, and it must not push the thumb past the end.
    #[test]
    fn a_very_long_list_still_leaves_a_thumb_to_grab() {
        let mut core = UiCore::new();
        let (area, bar) = scrollbar(&mut core, 100_000.0);
        let h = thumb_height(&core, bar);
        assert_eq!(h, ScrollbarStyle::default().min_thumb_px);

        core.scroll_by(area, [0.0, f32::MAX]);
        assert!((thumb_top(&core, bar) - (100.0 - h)).abs() < 1.0, "still lands flush");
    }

    /// Dragging is absolute and centres the thumb on the pointer, so a press
    /// on bare track jumps there — the same rule the slider uses, with the
    /// thumb's length taken out of the mapping.
    #[test]
    fn dragging_the_thumb_scrolls_the_area() {
        let mut core = UiCore::new();
        let (area, bar) = scrollbar(&mut core, 400.0);
        let r = core.node_rect(bar);
        let x = r[0] + r[2] * 0.5;

        core.update_pointer([x, r[1] + r[3] * 0.5], true, false, 0.0, 0.0);
        let mid = core.scroll_offset(area)[1];
        assert!((mid - 150.0).abs() < 2.0, "halfway down the track: {mid}");

        core.update_pointer([x, r[1] + r[3]], false, false, 0.0, 0.0);
        assert_eq!(core.scroll_offset(area)[1], 300.0, "the bottom is the bottom");

        core.update_pointer([x, r[1] - 500.0], false, false, 0.0, 0.0);
        assert_eq!(core.scroll_offset(area)[1], 0.0, "dragged off the top, clamped");
    }

    /// The headline property of a scroll survives having a bar attached: the
    /// thumb rides its group's offset, so nothing relayouts and no quad moves.
    #[test]
    fn scrolling_with_a_bar_writes_two_group_records_and_no_quads() {
        let mut core = UiCore::new();
        let (area, _) = scrollbar(&mut core, 400.0);
        let (mut stage, mut dirty) = (vec![0u32; 1 << 16], vec![0u32; 256]);
        core.quad.upload(&mut stage, &mut dirty);
        core.style.upload(&mut stage, &mut dirty);
        core.group.upload(&mut stage, &mut dirty);
        dirty.fill(0);

        core.scroll_by(area, [0.0, 40.0]);
        core.run_layout([400.0, 400.0]);

        core.group.upload(&mut stage, &mut dirty);
        let groups: u32 = dirty.iter().map(|w| w.count_ones()).sum();
        dirty.fill(0);
        core.quad.upload(&mut stage, &mut dirty);
        let quads: u32 = dirty.iter().map(|w| w.count_ones()).sum();

        assert_eq!(groups, 2, "the content that moved, and the thumb that tracked it");
        assert_eq!(quads, 0, "no geometry moved, so taffy never ran");
    }

    /// The bar re-fits itself, which is exactly the shape of thing that
    /// breaks the idle-frame invariant. It settles one frame after the layout
    /// it measures and then stops — `run_layout`'s early-out never reaches the
    /// sync at all.
    #[test]
    fn a_settled_bar_uploads_nothing() {
        let mut core = UiCore::new();
        let _ = scrollbar(&mut core, 400.0);
        let (mut stage, mut dirty) = (vec![0u32; 1 << 16], vec![0u32; 256]);
        let clean = (i64::MAX, -1);
        core.quad.upload(&mut stage, &mut dirty);
        core.style.upload(&mut stage, &mut dirty);
        core.group.upload(&mut stage, &mut dirty);

        for _ in 0..8 {
            core.run_layout([400.0, 400.0]);
        }
        assert_eq!(core.quad.upload(&mut stage, &mut dirty), clean);
        assert_eq!(core.style.upload(&mut stage, &mut dirty), clean);
        assert_eq!(core.group.upload(&mut stage, &mut dirty), clean);
    }

    /// The bar names a node outside itself, which nothing else does. Tearing
    /// down the area has to take the bar with it, or the next sync reaches
    /// through a stale handle and panics.
    #[test]
    fn removing_the_area_removes_the_bar_with_it() {
        let mut core = UiCore::new();
        let (area, _) = scrollbar(&mut core, 400.0);
        assert_eq!(core.scrollbars.len(), 1);

        core.remove_node(area);
        core.run_layout([400.0, 400.0]);
        assert!(core.scrollbars.is_empty());
    }
}
