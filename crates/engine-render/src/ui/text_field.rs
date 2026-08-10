//! A single-line text field — the widget that forces keyboard input.
//!
//! # Where the value lives
//!
//! In the control, like every other control ([`widget`](super::widget)'s
//! module docs make the general argument). The difference is that a `String`
//! is not `Copy`, so [`Control::TextField`](super::widget::Control) boxes its
//! state: the enum stays small for the thousands of nodes that are *not*
//! fields, and the indirection is paid once per keystroke, which is to say
//! never.
//!
//! # The visible window
//!
//! The glyph run holds **what fits**, not what the field contains: `cols`
//! characters starting at `scroll`. Typing past the right edge advances the
//! window instead of drawing outside the box.
//!
//! The alternative — draw the whole string and clip it with a `ui_group` —
//! is how a real editor does it, and it was rejected here for a specific
//! reason rather than for effort: a group's offset is *absolute*, composed on
//! the CPU, and the one-record scroll path deliberately does not re-walk
//! nested groups. That is exactly why `scroll_area` asserts against nesting,
//! and a field is very often inside a scroll area. Reproducing that bug in a
//! second place to gain sub-character scrolling of a bitmap font with a fixed
//! advance is a bad trade. It becomes the right answer the day nested groups
//! compose, and this window then deletes cleanly.
//!
//! It also makes the cost independent of the value: a field holding ten
//! thousand characters paints the thirty you can see.
//!
//! # Why the caret does not blink
//!
//! A blink is a timer, and a timer dirties a slot twice a second forever —
//! in a UI whose entire premise is that an idle frame uploads zero bytes and
//! dispatches zero workgroups. A solid caret is not a compromise here, it is
//! the design being consistent with itself. When the tooltip's dwell clock
//! lands (the first thing that genuinely needs one), blinking is a style
//! choice on top of it rather than a hole in the invariant.

use crate::input::{Key, Keystroke, Mods};

use super::style::{
    percent, px, AlignItems, Display, LengthPercentageAuto, Position, Rect, Size, Style, TaffyAuto,
};
use super::widget::Control;
use super::{font, theme, Events, Label, NodeId, TextField, Theme, UiCore, UiStyle};

/// Caret width in px. Reserved out of the visible width so the caret at the
/// end of the window still sits inside the box.
const CARET_W: f32 = 1.0;

/// How [`UiCore::text_field`] looks and measures.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TextFieldStyle {
    pub fill: u32,
    pub border: u32,
    /// Border while focused — the focus ring, and the only feedback that
    /// keystrokes are going here.
    pub border_focus: u32,
    pub text: u32,
    /// Hint text shown while the field is empty and unfocused.
    pub hint: u32,
    pub caret: u32,
    pub selection: u32,
    /// Box width in px. Definite for the same reason a slider's is: pointer
    /// x maps onto a character index, and the visible character count is
    /// derived from it once at construction rather than re-derived per
    /// keystroke from a box that may not be solved yet.
    pub width: f32,
    pub text_px: f32,
    pub padding: f32,
    pub radius: f32,
}

impl From<Theme> for TextFieldStyle {
    fn from(t: Theme) -> Self {
        Self {
            fill: t.backdrop,
            border: t.outline,
            border_focus: t.accent,
            text: t.text,
            hint: t.text_dim,
            caret: t.text,
            selection: t.selection,
            width: 160.0,
            text_px: t.text_px,
            padding: 4.0,
            radius: t.radius,
        }
    }
}

impl Default for TextFieldStyle {
    fn default() -> Self {
        theme().into()
    }
}

/// A field's value and everything derived from it.
///
/// Positions are **character** indices, not byte offsets: every consumer —
/// the caret's x, the selection's width, the visible window — counts
/// characters, so storing bytes would mean converting at each of them
/// instead of at the two splice sites.
#[derive(Clone, Debug)]
pub(crate) struct FieldState {
    text: String,
    hint: String,
    /// Where the caret is.
    cursor: usize,
    /// The fixed end of the selection. Equal to `cursor` when there is none,
    /// which is why there is no `Option` and no separate "selecting" flag:
    /// every motion either drags the anchor along or leaves it behind.
    anchor: usize,
    /// First visible character.
    scroll: usize,
    /// How many characters fit in the box.
    cols: usize,
    /// The box the caret and selection are positioned inside. Padding-free,
    /// so an inset of `x` is `x` px from the first glyph however the field
    /// itself is padded.
    inner: NodeId,
    label: Label,
    caret: NodeId,
    sel: NodeId,
    advance: f32,
    text_px: f32,
    color: u32,
    hint_color: u32,
    idle: UiStyle,
    focused: UiStyle,
}

/// What one keystroke did, for the one-frame flags in `Keyboard`.
#[derive(Default)]
pub(crate) struct Response {
    pub(crate) changed: bool,
    pub(crate) submitted: bool,
}

impl FieldState {
    fn len(&self) -> usize {
        self.text.chars().count()
    }

    /// Byte offset of character `i`, or the end of the string.
    fn byte(&self, i: usize) -> usize {
        self.text
            .char_indices()
            .nth(i)
            .map_or(self.text.len(), |(b, _)| b)
    }

    /// The selection as an ordered range, empty when there is none.
    fn range(&self) -> (usize, usize) {
        (self.cursor.min(self.anchor), self.cursor.max(self.anchor))
    }

    /// Move the caret. `extend` is shift held: it leaves the anchor where it
    /// was, which is the entire difference between moving and selecting.
    fn set_cursor(&mut self, i: usize, extend: bool) {
        self.cursor = i.min(self.len());
        if !extend {
            self.anchor = self.cursor;
        }
    }

    fn select_all(&mut self) {
        self.anchor = 0;
        self.cursor = self.len();
    }

    /// Remove the selection, if any. Returns whether the text changed.
    fn delete_selection(&mut self) -> bool {
        let (a, b) = self.range();
        if a == b {
            return false;
        }
        let (from, to) = (self.byte(a), self.byte(b));
        self.text.replace_range(from..to, "");
        self.cursor = a;
        self.anchor = a;
        true
    }

    fn insert(&mut self, s: &str) -> bool {
        self.delete_selection();
        let at = self.byte(self.cursor);
        self.text.insert_str(at, s);
        self.cursor += s.chars().count();
        self.anchor = self.cursor;
        !s.is_empty()
    }

    /// The index one step from `i`. `word` jumps over a run of whitespace and
    /// then a run of non-whitespace, which is the behaviour every editor
    /// agrees on even where they disagree about punctuation.
    fn step(&self, i: usize, back: bool, word: bool) -> usize {
        let chars: Vec<char> = self.text.chars().collect();
        if !word {
            return if back { i.saturating_sub(1) } else { (i + 1).min(chars.len()) };
        }
        let mut i = i;
        if back {
            while i > 0 && chars[i - 1].is_whitespace() {
                i -= 1;
            }
            while i > 0 && !chars[i - 1].is_whitespace() {
                i -= 1;
            }
        } else {
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            while i < chars.len() && !chars[i].is_whitespace() {
                i += 1;
            }
        }
        i
    }

    /// Backspace. With a selection it deletes that instead — the rule that
    /// makes typing over a selection work without a special case in `insert`.
    fn backspace(&mut self, word: bool) -> bool {
        if self.delete_selection() {
            return true;
        }
        if self.cursor == 0 {
            return false;
        }
        let to = self.step(self.cursor, true, word);
        let (from, upto) = (self.byte(to), self.byte(self.cursor));
        self.text.replace_range(from..upto, "");
        self.cursor = to;
        self.anchor = to;
        true
    }

    fn delete_forward(&mut self, word: bool) -> bool {
        if self.delete_selection() {
            return true;
        }
        let to = self.step(self.cursor, false, word);
        if to == self.cursor {
            return false;
        }
        let (from, upto) = (self.byte(self.cursor), self.byte(to));
        self.text.replace_range(from..upto, "");
        true
    }

    /// Slide the window so the caret is inside it.
    fn ensure_visible(&mut self) {
        let len = self.len();
        self.scroll = self.scroll.min(len);
        if self.cursor < self.scroll {
            self.scroll = self.cursor;
        } else if self.cursor > self.scroll + self.cols {
            self.scroll = self.cursor - self.cols;
        }
        // Pull the window back when the text no longer fills it, so deleting
        // from the end never leaves a half-empty box with the value scrolled
        // off to the left. Cannot hide the caret: it only fires when the
        // whole tail fits, and the caret is in the tail.
        if len < self.scroll + self.cols {
            self.scroll = len.saturating_sub(self.cols);
        }
    }

    /// The characters currently on screen.
    fn window(&self) -> &str {
        let end = (self.scroll + self.cols).min(self.len());
        &self.text[self.byte(self.scroll)..self.byte(end)]
    }

    /// The hint, cut to what fits. Same rule as the value, for the same
    /// reason — a hint longer than the box would draw straight through the
    /// border, which is exactly what it did before this existed.
    fn hint_window(&self) -> &str {
        match self.hint.char_indices().nth(self.cols) {
            Some((b, _)) => &self.hint[..b],
            None => &self.hint,
        }
    }

    /// Apply one keystroke. Pure — nothing here touches the store, so the
    /// editing model is testable without a tree.
    fn apply(&mut self, stroke: &Keystroke) -> Response {
        let mut r = Response::default();
        match stroke {
            Keystroke::Text(s) => r.changed = self.insert(s),
            Keystroke::Key(k, m) => {
                let extend = m.has(Mods::SHIFT);
                let word = m.has(Mods::CTRL);
                match k {
                    Key::Backspace => r.changed = self.backspace(word),
                    Key::Delete => r.changed = self.delete_forward(word),
                    Key::Left => {
                        let i = self.step(self.cursor, true, word);
                        self.set_cursor(i, extend);
                    }
                    Key::Right => {
                        let i = self.step(self.cursor, false, word);
                        self.set_cursor(i, extend);
                    }
                    // One line, so vertical motion is the only thing it can
                    // mean here — and a field that ignored the arrow would
                    // read as broken.
                    Key::Home | Key::Up => self.set_cursor(0, extend),
                    Key::End | Key::Down => self.set_cursor(self.len(), extend),
                    Key::Enter => r.submitted = true,
                    // The modifier is re-checked rather than assumed: the
                    // input layer only mints `Char` with `Ctrl` held, and an
                    // invariant enforced two files away is one that changes
                    // without this noticing.
                    Key::Char('a') if m.has(Mods::CTRL) => self.select_all(),
                    _ => {}
                }
            }
        }
        r
    }

    /// Push the state into the store. Every write is gated, so re-rendering
    /// an unchanged field costs comparisons and no upload — which is why
    /// nothing here tries to work out what moved.
    fn render(&mut self, ui: &mut UiCore, node: NodeId, focused: bool) {
        self.ensure_visible();
        let (label, caret, sel) = (self.label, self.caret, self.sel);

        let hint = self.text.is_empty() && !focused;
        {
            let text = if hint { self.hint_window() } else { self.window() };
            label.set_text(ui, text);
        }
        label.set_color(ui, if hint { self.hint_color } else { self.color });

        let x = (self.cursor - self.scroll) as f32 * self.advance;
        bar(ui, caret, x, if focused { CARET_W } else { 0.0 }, self.text_px);

        // Clipped to the window, so a selection running off either end draws
        // to the edge rather than outside the box.
        //
        // With no selection the bar is parked at zero rather than left
        // trailing the caret: it is invisible either way, and following the
        // caret would dirty a slot on every arrow key to move a quad nobody
        // can see. That is one third of the cost of a keystroke.
        let (a, b) = self.range();
        let (a, b) = (a.max(self.scroll), b.min(self.scroll + self.cols));
        let (x, w) = match focused && b > a {
            true => (
                (a - self.scroll) as f32 * self.advance,
                (b - a) as f32 * self.advance,
            ),
            false => (0.0, 0.0),
        };
        bar(ui, sel, x, w, self.text_px);

        ui.set_background(node, if focused { self.focused } else { self.idle });
    }
}

/// Position and size one of the two absolutely-placed bars. A zero width is
/// how both hide: `ui.vert` culls a zero-area quad before it reads anything.
fn bar(ui: &mut UiCore, n: NodeId, x: f32, w: f32, h: f32) {
    let mut s = ui.node_style(n);
    s.inset.left = px(x);
    s.size = Size {
        width: px(w),
        height: px(h),
    };
    ui.set_node_style(n, s);
}

impl TextField {
    /// The value. Borrowed from the control, so reading it allocates nothing.
    pub fn text(self, ui: &UiCore) -> &str {
        match ui.control(self.node()) {
            Control::TextField(st) => &st.text,
            _ => unreachable!("TextField handle over a non-field"),
        }
    }

    /// Replace the value, as a paste or a load would. The caret goes to the
    /// end, which is where a caller who just supplied the text wants it.
    pub fn set_text(self, ui: &mut UiCore, text: &str) {
        let n = self.node();
        let focused = ui.focused(n);
        ui.with_field(n, |ui, st| {
            if st.text == text {
                return;
            }
            st.text.clear();
            st.text.push_str(text);
            st.cursor = st.len();
            st.anchor = st.cursor;
            st.render(ui, n, focused);
        });
    }

    /// The prompt shown while the field is empty and unfocused.
    pub fn set_hint(self, ui: &mut UiCore, hint: &str) {
        let n = self.node();
        let focused = ui.focused(n);
        ui.with_field(n, |ui, st| {
            if st.hint == hint {
                return;
            }
            st.hint.clear();
            st.hint.push_str(hint);
            st.render(ui, n, focused);
        });
    }

    /// `Enter` was pressed in this field this frame. One frame, like
    /// [`UiCore::clicked`].
    pub fn submitted(self, ui: &UiCore) -> bool {
        ui.keyboard.submitted == Some(self.node())
    }

    /// The value changed this frame — what a search box re-filters on.
    /// Covers typing and deleting, not caret motion, and not
    /// [`set_text`](Self::set_text): a change the caller made itself needs no
    /// announcement.
    pub fn changed(self, ui: &UiCore) -> bool {
        ui.keyboard.changed == Some(self.node())
    }

    /// Give it the keyboard and select its contents, as `Tab` would.
    pub fn focus(self, ui: &mut UiCore) {
        let n = self.node();
        ui.set_focus(Some(n));
        ui.field_select_all(n);
    }

    /// Caret position, in characters. Test and inspector surface — the
    /// value is what applications read.
    pub fn cursor(self, ui: &UiCore) -> usize {
        match ui.control(self.node()) {
            Control::TextField(st) => st.cursor,
            _ => unreachable!("TextField handle over a non-field"),
        }
    }
}

impl UiCore {
    /// A single-line text field.
    ///
    /// It owns its value: keystrokes are applied by `update_keyboard` before
    /// any component runs, so a caller only ever reads.
    ///
    /// ```ignore
    /// let name = ui.text_field(panel, "", TextFieldStyle::default());
    /// name.set_hint(&mut ui, "search…");
    /// // …later, and from anywhere:
    /// if name.changed(&ui) { refilter(name.text(&ui)); }
    /// ```
    pub fn text_field(
        &mut self,
        parent: impl Into<NodeId>,
        text: &str,
        style: TextFieldStyle,
    ) -> TextField {
        let scale = style.text_px / font::GLYPH_H as f32;
        let advance = font::ADVANCE as f32 * scale;
        let inner_w = (style.width - 2.0 * style.padding).max(advance);
        let cols = (((inner_w - CARET_W) / advance).floor() as usize).max(1);

        let idle = UiStyle::fill(style.fill)
            .border(style.border, 1.0)
            .radius(style.radius);
        let focused = UiStyle::fill(style.fill)
            .border(style.border_focus, 1.0)
            .radius(style.radius);

        let field = self.node(
            parent,
            Style {
                display: Display::Flex,
                align_items: Some(AlignItems::CENTER),
                size: Size {
                    width: px(style.width),
                    height: TaffyAuto::AUTO,
                },
                padding: Rect::length(style.padding),
                flex_shrink: 0.0,
                ..Default::default()
            },
        );
        self.set_background(field, idle);

        // The caret and the selection are absolutely positioned, so they need
        // a containing box with no padding of its own — otherwise every inset
        // would carry the field's padding and the two would have to agree.
        let inner = self.node(
            field,
            Style {
                size: Size {
                    width: percent(1.0_f32),
                    height: px(style.text_px),
                },
                flex_shrink: 0.0,
                ..Default::default()
            },
        );

        // Tree order is paint order: selection under the glyphs, caret over
        // them. Nothing else establishes that, and nothing else needs to.
        let sel = self.bar_node(inner, style.selection, style.text_px, 0.0);
        let label = self.label(inner, style.text_px, style.text, "");
        let caret = self.bar_node(inner, style.caret, style.text_px, 0.0);

        self.set_events(field, Events::CLICK | Events::FOCUS);

        let mut st = FieldState {
            text: text.to_string(),
            hint: String::new(),
            cursor: 0,
            anchor: 0,
            scroll: 0,
            cols,
            inner,
            label,
            caret,
            sel,
            advance,
            text_px: style.text_px,
            color: style.text,
            hint_color: style.hint,
            idle,
            focused,
        };
        st.cursor = st.len();
        st.anchor = st.cursor;
        st.render(self, field, false);
        self.set_control(field, Control::TextField(Box::new(st)));
        TextField::from_node(field)
    }

    /// One of the two absolutely-positioned bars a field draws.
    fn bar_node(&mut self, parent: NodeId, fill: u32, h: f32, radius: f32) -> NodeId {
        let n = self.node(
            parent,
            Style {
                position: Position::Absolute,
                inset: Rect {
                    left: px(0.0),
                    top: px(0.0),
                    right: LengthPercentageAuto::AUTO,
                    bottom: LengthPercentageAuto::AUTO,
                },
                size: Size {
                    width: px(0.0),
                    height: px(h),
                },
                ..Default::default()
            },
        );
        self.set_background(n, UiStyle::fill(fill).radius(radius));
        n
    }

    /// Run `f` against a node's field state, if it is a field.
    ///
    /// The state is *taken out* of the control table for the call rather than
    /// borrowed, because everything worth doing to it also needs `&mut
    /// UiCore` — writing glyphs, moving the caret node, restyling the box.
    /// Taking a `Box` is a pointer move; the alternative is cloning a string
    /// on every keystroke.
    fn with_field<R>(
        &mut self,
        n: NodeId,
        f: impl FnOnce(&mut UiCore, &mut FieldState) -> R,
    ) -> Option<R> {
        let idx = self.live(n);
        match self.controls.get_mut(idx).and_then(|c| c.take()) {
            Some(Control::TextField(mut st)) => {
                let r = f(self, &mut st);
                self.controls[idx] = Some(Control::TextField(st));
                Some(r)
            }
            other => {
                // Not a field — put back whatever was there. A `None` slot
                // round-trips as `None`.
                if let Some(slot) = self.controls.get_mut(idx) {
                    *slot = other;
                }
                None
            }
        }
    }

    /// Re-render a field. A no-op for every other kind of node, which is what
    /// lets `set_focus` call it blindly on both ends of a focus change.
    pub(crate) fn sync_field(&mut self, n: NodeId) {
        let focused = self.focused(n);
        self.with_field(n, |ui, st| st.render(ui, n, focused));
    }

    /// Route one keystroke to the focused field.
    pub(crate) fn field_keystroke(&mut self, n: NodeId, stroke: &Keystroke) {
        let Some(r) = self.with_field(n, |ui, st| {
            let r = st.apply(stroke);
            // Unconditionally, including for a keystroke the field ignored:
            // the equality gate makes an unchanged render free, and deciding
            // *here* what moved would be a second copy of that logic to keep
            // in step.
            st.render(ui, n, true);
            r
        }) else {
            return;
        };
        if r.changed {
            self.keyboard.changed = Some(n);
        }
        if r.submitted {
            self.keyboard.submitted = Some(n);
        }
    }

    /// Select everything, on arrival by `Tab` or [`TextField::focus`].
    pub(crate) fn field_select_all(&mut self, n: NodeId) {
        self.with_field(n, |ui, st| {
            st.select_all();
            st.render(ui, n, true);
        });
    }

    /// Put the caret where the pointer is. `extend` keeps the anchor, which
    /// is what turns a press-and-drag into a selection.
    pub(crate) fn field_point(&mut self, n: NodeId, x: f32, extend: bool) {
        self.with_field(n, |ui, st| {
            let r = ui.node_rect(st.inner);
            // Rounded, not truncated: clicking the right half of a character
            // means after it, which is what makes clicking at the end of a
            // word land after the last letter rather than before it.
            let rel = ((x - r[0]) / st.advance).round().max(0.0) as usize;
            let i = (st.scroll + rel.min(st.cols)).min(st.len());
            st.set_cursor(i, extend);
            st.render(ui, n, true);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::style::Style;

    const W: f32 = 100.0;

    /// A field wide enough for a known number of characters, laid out.
    fn field(core: &mut UiCore, text: &str) -> TextField {
        let root = core.root();
        let f = core.text_field(
            root,
            text,
            TextFieldStyle {
                width: W,
                padding: 4.0,
                ..TextFieldStyle::default()
            },
        );
        core.run_layout([400.0, 400.0]);
        f
    }

    fn typed(s: &str) -> Keystroke {
        Keystroke::Text(s.to_string())
    }

    fn key(k: Key) -> Keystroke {
        Keystroke::Key(k, Mods::NONE)
    }

    fn with_mods(k: Key, m: Mods) -> Keystroke {
        Keystroke::Key(k, m)
    }

    /// Nothing reaches a field that does not have focus — the whole point of
    /// a focus model, and the bug every "just read the keyboard" widget has.
    #[test]
    fn keystrokes_only_reach_the_focused_field() {
        let mut core = UiCore::new();
        let f = field(&mut core, "");
        core.update_keyboard(&[typed("a")]);
        assert_eq!(f.text(&core), "", "unfocused: the keystroke is not ours");

        f.focus(&mut core);
        core.update_keyboard(&[typed("a")]);
        assert_eq!(f.text(&core), "a");
    }

    /// Typing, deleting, and the caret that follows both.
    #[test]
    fn typing_edits_at_the_caret() {
        let mut core = UiCore::new();
        let f = field(&mut core, "");
        f.focus(&mut core);

        core.update_keyboard(&[typed("h"), typed("i"), typed("!")]);
        assert_eq!(f.text(&core), "hi!");
        assert_eq!(f.cursor(&core), 3);

        core.update_keyboard(&[key(Key::Backspace)]);
        assert_eq!(f.text(&core), "hi");

        core.update_keyboard(&[key(Key::Home), typed("o")]);
        assert_eq!(f.text(&core), "ohi", "inserted at the caret, not the end");
        assert_eq!(f.cursor(&core), 1);

        core.update_keyboard(&[key(Key::Delete)]);
        assert_eq!(f.text(&core), "oi");
    }

    /// A selection is replaced by what is typed over it, and deleted by
    /// backspace — both fall out of `delete_selection`, so both are asserted.
    #[test]
    fn selection_is_replaced_by_typing() {
        let mut core = UiCore::new();
        let f = field(&mut core, "hello");
        f.focus(&mut core); // selects all

        core.update_keyboard(&[typed("x")]);
        assert_eq!(f.text(&core), "x");

        f.set_text(&mut core, "hello");
        f.focus(&mut core);
        core.update_keyboard(&[key(Key::Backspace)]);
        assert_eq!(f.text(&core), "");
    }

    /// Shift+arrow selects; the same arrow without shift collapses it.
    #[test]
    fn shift_extends_and_a_bare_arrow_collapses() {
        let mut core = UiCore::new();
        let f = field(&mut core, "abcd");
        f.focus(&mut core);
        core.update_keyboard(&[key(Key::End)]);

        core.update_keyboard(&[
            with_mods(Key::Left, Mods::SHIFT),
            with_mods(Key::Left, Mods::SHIFT),
            typed("Z"),
        ]);
        assert_eq!(f.text(&core), "abZ", "two selected characters replaced");

        f.set_text(&mut core, "abcd");
        core.update_keyboard(&[key(Key::Left), typed("Z")]);
        assert_eq!(f.text(&core), "abcZd", "no selection: an insert, not a replace");
    }

    /// Ctrl+arrow moves by word, Ctrl+A selects everything.
    #[test]
    fn word_motion_and_select_all() {
        let mut core = UiCore::new();
        let f = field(&mut core, "one two");
        f.focus(&mut core);
        core.update_keyboard(&[key(Key::End), with_mods(Key::Left, Mods::CTRL)]);
        assert_eq!(f.cursor(&core), 4, "back over 'two' to the space");

        core.update_keyboard(&[with_mods(Key::Char('a'), Mods::CTRL), typed("!")]);
        assert_eq!(f.text(&core), "!");
    }

    /// `Enter` reports for exactly one frame, like a click.
    #[test]
    fn submitted_lasts_one_frame() {
        let mut core = UiCore::new();
        let f = field(&mut core, "");
        f.focus(&mut core);

        core.update_keyboard(&[key(Key::Enter)]);
        assert!(f.submitted(&core));
        core.update_keyboard(&[]);
        assert!(!f.submitted(&core), "an event, not a state");
    }

    /// `changed` covers what the user did and not what the caller did.
    #[test]
    fn changed_tracks_edits_only() {
        let mut core = UiCore::new();
        let f = field(&mut core, "");
        f.focus(&mut core);

        core.update_keyboard(&[typed("a")]);
        assert!(f.changed(&core));

        core.update_keyboard(&[key(Key::Left)]);
        assert!(!f.changed(&core), "caret motion is not a change");

        f.set_text(&mut core, "b");
        assert!(!f.changed(&core), "the caller already knows");
    }

    /// The window follows the caret past the right edge, and comes back.
    #[test]
    fn the_window_follows_the_caret() {
        let mut core = UiCore::new();
        let f = field(&mut core, "");
        f.focus(&mut core);

        let cols = match core.control(f.node()) {
            Control::TextField(st) => st.cols,
            _ => unreachable!(),
        };
        let long: String = (0..cols * 2).map(|i| (b'a' + (i % 26) as u8) as char).collect();
        for ch in long.chars() {
            core.update_keyboard(&[typed(&ch.to_string())]);
        }

        let (window, scroll) = match core.control(f.node()) {
            Control::TextField(st) => (st.window().to_string(), st.scroll),
            _ => unreachable!(),
        };
        assert_eq!(f.text(&core), long, "the value is whole");
        assert!(scroll > 0, "the window advanced past the box");
        assert!(window.chars().count() <= cols, "and never draws more than fits");
        assert!(long.ends_with(&window), "showing the tail the caret is in");

        core.update_keyboard(&[key(Key::Home)]);
        let scroll = match core.control(f.node()) {
            Control::TextField(st) => st.scroll,
            _ => unreachable!(),
        };
        assert_eq!(scroll, 0, "the caret pulled the window back to the start");
    }

    /// The caret node sits where the character does — the one geometric
    /// claim, and the one that silently breaks if taffy's absolute insets
    /// ever stop being relative to the box they are declared in.
    ///
    /// Within a pixel, because taffy rounds every box to whole pixels while
    /// a glyph advance (`6 * 11/9`) is fractional. That rounding is wanted —
    /// it is what keeps a 1 px caret crisp — and half a pixel of drift
    /// against the glyph beside it is not visible at any text size.
    #[test]
    fn the_caret_tracks_the_character_it_precedes() {
        let mut core = UiCore::new();
        let f = field(&mut core, "abcd");
        f.focus(&mut core);
        core.update_keyboard(&[key(Key::Home), key(Key::Right), key(Key::Right)]);
        core.run_layout([400.0, 400.0]);

        let (inner, caret, advance) = match core.control(f.node()) {
            Control::TextField(st) => (st.inner, st.caret, st.advance),
            _ => unreachable!(),
        };
        let x0 = core.node_rect(inner)[0];
        let caret_x = core.node_rect(caret)[0];
        assert!(
            (caret_x - (x0 + 2.0 * advance)).abs() <= 1.0,
            "caret at {caret_x}, expected {} give or take taffy's rounding",
            x0 + 2.0 * advance
        );
    }

    /// A hint too long for the box is cut to fit, like the value is.
    #[test]
    fn a_long_hint_is_cut_to_the_box() {
        let mut core = UiCore::new();
        let f = field(&mut core, "");
        f.set_hint(&mut core, &"h".repeat(200));
        let (hint, cols) = match core.control(f.node()) {
            Control::TextField(st) => (st.hint_window().to_string(), st.cols),
            _ => unreachable!(),
        };
        assert_eq!(hint.chars().count(), cols);
    }

    /// An unfocused field draws no caret, and a focused empty one drops its
    /// hint — both are the same zero-width / swapped-string path.
    #[test]
    fn focus_shows_the_caret_and_hides_the_hint() {
        let mut core = UiCore::new();
        let f = field(&mut core, "");
        f.set_hint(&mut core, "name");
        core.run_layout([400.0, 400.0]);

        let caret = match core.control(f.node()) {
            Control::TextField(st) => st.caret,
            _ => unreachable!(),
        };
        assert_eq!(core.node_rect(caret)[2], 0.0, "no caret without focus");
        assert_eq!(core.node_text(f.node()), None, "the field itself has no run");

        f.focus(&mut core);
        core.run_layout([400.0, 400.0]);
        assert!(core.node_rect(caret)[2] > 0.0, "focused: a caret");
    }

    /// A press moves focus, and a press elsewhere takes it away — the rule
    /// that makes clicking the scene dismiss a caret with no special case.
    #[test]
    fn pressing_moves_focus_and_pressing_away_clears_it() {
        let mut core = UiCore::new();
        let root = core.root();
        let f = core.text_field(root, "hello", TextFieldStyle::default());
        let outside = core.node(
            root,
            Style {
                position: crate::ui::style::Position::Absolute,
                inset: Rect {
                    left: px(300.0),
                    top: px(300.0),
                    right: LengthPercentageAuto::AUTO,
                    bottom: LengthPercentageAuto::AUTO,
                },
                size: Size {
                    width: px(20.0),
                    height: px(20.0),
                },
                ..Default::default()
            },
        );
        core.set_events(outside, Events::CLICK);
        core.run_layout([400.0, 400.0]);

        let r = core.node_rect(f);
        let inside = [r[0] + r[2] * 0.5, r[1] + r[3] * 0.5];
        core.update_pointer(inside, true, false, 0.0);
        assert!(core.focused(f), "the press focused it");
        assert!(core.keyboard_captured());

        core.update_pointer(inside, false, true, 0.0);
        core.update_pointer([310.0, 310.0], true, false, 0.0);
        assert!(!core.focused(f), "a press elsewhere took it away");
        assert!(!core.keyboard_captured());
    }

    /// A press inside the text places the caret there rather than at the end.
    #[test]
    fn a_press_places_the_caret() {
        let mut core = UiCore::new();
        let f = field(&mut core, "abcdef");
        let (inner, advance) = match core.control(f.node()) {
            Control::TextField(st) => (st.inner, st.advance),
            _ => unreachable!(),
        };
        let r = core.node_rect(inner);
        let p = [r[0] + 3.0 * advance, r[1] + r[3] * 0.5];
        core.update_pointer(p, true, false, 0.0);
        assert_eq!(f.cursor(&core), 3);
    }

    /// Tab walks the ring in tree order and wraps, with nothing focused to
    /// start with.
    #[test]
    fn tab_cycles_the_focus_ring() {
        let mut core = UiCore::new();
        let root = core.root();
        let a = core.text_field(root, "", TextFieldStyle::default());
        let b = core.text_field(root, "", TextFieldStyle::default());
        core.run_layout([400.0, 400.0]);

        core.update_keyboard(&[key(Key::Tab)]);
        assert!(core.focused(a), "the first stop");
        core.update_keyboard(&[key(Key::Tab)]);
        assert!(core.focused(b));
        core.update_keyboard(&[key(Key::Tab)]);
        assert!(core.focused(a), "wrapped");
        core.update_keyboard(&[with_mods(Key::Tab, Mods::SHIFT)]);
        assert!(core.focused(b), "and backwards");
    }

    /// Escape gives the keyboard back, so a game's hotkeys work again.
    #[test]
    fn escape_blurs() {
        let mut core = UiCore::new();
        let f = field(&mut core, "");
        f.focus(&mut core);
        core.update_keyboard(&[key(Key::Escape)]);
        assert!(!core.keyboard_captured());
    }

    /// Collapsing the panel a focused field lives in gives the keyboard back,
    /// so a game's hotkeys do not stay suppressed by a field nobody can see.
    #[test]
    fn hiding_a_focused_field_releases_the_keyboard() {
        let mut core = UiCore::new();
        let root = core.root();
        let panel = core.node(root, Style::default());
        let f = core.text_field(panel, "", TextFieldStyle::default());
        core.run_layout([400.0, 400.0]);
        f.focus(&mut core);
        assert!(core.keyboard_captured());

        let mut s = core.node_style(panel);
        s.display = crate::ui::style::Display::None;
        core.set_node_style(panel, s);
        core.run_layout([400.0, 400.0]);
        assert!(!core.keyboard_captured(), "no box, no keyboard");
    }

    /// What one keystroke costs, in slots. Appending a character rewrites
    /// the glyph it added and moves the caret — the rest of the value, the
    /// box, the border and the selection are all re-written with the values
    /// they already had, and the equality gate drops every one of them.
    #[test]
    fn one_keystroke_dirties_two_quads() {
        let mut core = UiCore::new();
        let f = field(&mut core, "hello");
        f.focus(&mut core);
        core.update_keyboard(&[key(Key::End)]);
        core.run_layout([400.0, 400.0]);

        let mut stage = vec![0u32; 1 << 16];
        let mut dirty = vec![0u32; 256];
        core.quad.upload(&mut stage, &mut dirty);
        dirty.fill(0);

        core.update_keyboard(&[typed("!")]);
        core.run_layout([400.0, 400.0]);
        core.quad.upload(&mut stage, &mut dirty);
        let touched: u32 = dirty.iter().map(|w| w.count_ones()).sum();
        assert_eq!(touched, 2, "one glyph and the caret");
    }

    /// The invariant the whole store rests on, restated for the field: a
    /// frame with no keystrokes uploads nothing at all.
    #[test]
    fn an_idle_frame_uploads_nothing() {
        let mut core = UiCore::new();
        let f = field(&mut core, "hello");
        f.focus(&mut core);
        core.update_keyboard(&[typed("!")]);
        core.run_layout([400.0, 400.0]);

        let mut stage = vec![0u32; 1 << 16];
        let mut dirty = vec![0u32; 256];
        core.quad.upload(&mut stage, &mut dirty);
        core.style.upload(&mut stage, &mut dirty);

        core.update_keyboard(&[]);
        core.run_layout([400.0, 400.0]);
        assert_eq!(core.quad.upload(&mut stage, &mut dirty), (i64::MAX, -1));
        assert_eq!(core.style.upload(&mut stage, &mut dirty), (i64::MAX, -1));
    }

    /// Removing a focused field must not leave the keyboard pointed at a
    /// recycled slot — the same class of bug generations exist to catch.
    #[test]
    fn removing_a_focused_field_clears_focus() {
        let mut core = UiCore::new();
        let f = field(&mut core, "");
        f.focus(&mut core);
        core.remove_node(f);
        assert!(core.focus().is_none());
        assert!(!core.keyboard_captured());
        // The next keystroke has nowhere to go, and says so by not panicking.
        core.update_keyboard(&[typed("a")]);
    }
}
