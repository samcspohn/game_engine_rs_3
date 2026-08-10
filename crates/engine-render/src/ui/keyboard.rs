//! Keyboard focus and the keystroke route.
//!
//! This is the pointer layer's twin, and the differences are the whole
//! design. The pointer asks "what is under this position?" and gets a fresh
//! answer every time it moves; the keyboard has no position, so it asks "who
//! did the user last aim at?" — a single piece of *retained* state that
//! survives until something takes it away.
//!
//! Focus therefore lives here rather than in whichever widget wants it, for
//! the same reason the in-flight drag lives in `Pointer`: there is one
//! keyboard, so there is at most one focused node, and a widget that owned
//! that fact could not know when another widget took it.
//!
//! # What takes focus, and when
//!
//! A press. Not a click — pressing and dragging off a text field still
//! focuses it, exactly as it still places a caret — and not hover, which
//! would make the keyboard follow the mouse across a panel. Pressing
//! somewhere that does not accept [`Events::FOCUS`] clears focus, including
//! a press on the 3D scene, so dismissing a caret needs no special case.
//!
//! `Tab` walks the same set in tree order. It is handled *before* the
//! focused widget sees the keystroke, so no widget can swallow it, and it
//! works with nothing focused (it takes the first stop).
//!
//! # Why there is no per-frame work
//!
//! [`UiCore::update_keyboard`] is called every frame and does nothing on a
//! frame with no keystrokes but clear two `Option`s — the same shape as
//! `clicked`, and the reason a caret that is *not* blinking was a deliberate
//! choice: a blink is a timer that dirties a slot twice a second forever,
//! for a UI whose entire premise is that an idle frame uploads zero bytes.

use crate::input::{Key, Keystroke, Mods};

use super::{NodeId, UiCore};

/// The one keyboard's state.
#[derive(Default)]
pub(crate) struct Keyboard {
    /// Who receives keystrokes. `None` means the application does.
    pub(crate) focus: Option<NodeId>,
    /// Set for exactly one frame by `Enter` on a focused field, the way the
    /// pointer's `clicked` is — a submit is an event, and storing it as a
    /// flag on the widget would leave the caller to clear it.
    pub(crate) submitted: Option<NodeId>,
    /// Set for exactly one frame by any keystroke that changed a field's
    /// text. What a search box filters on.
    pub(crate) changed: Option<NodeId>,
}

impl UiCore {
    /// The focused node, if any.
    pub fn focus(&self) -> Option<NodeId> {
        self.keyboard.focus
    }

    /// Whether `n` holds keyboard focus.
    pub fn focused(&self, n: impl Into<NodeId>) -> bool {
        let n = n.into();
        self.keyboard.focus == Some(n)
    }

    /// Move focus, or clear it with `None`.
    ///
    /// Both ends are re-synced, which is what draws and undraws a caret; for
    /// a node that is not a control this is a no-op, so focusing a plain node
    /// to swallow keystrokes is allowed and costs nothing.
    pub fn set_focus(&mut self, n: Option<NodeId>) {
        if let Some(n) = n {
            self.live(n); // a stale handle must not become the focus
        }
        if self.keyboard.focus == n {
            return;
        }
        let old = self.keyboard.focus;
        self.keyboard.focus = n;
        for node in [old, n].into_iter().flatten() {
            self.sync_field(node);
        }
    }

    /// The UI owns the keyboard — a text field has focus, so game hotkeys
    /// should sit this frame out. The keyboard twin of
    /// [`pointer_captured`](UiCore::pointer_captured), and the reason typing
    /// into a field does not also fire whatever those letters are bound to.
    pub fn keyboard_captured(&self) -> bool {
        self.keyboard.focus.is_some()
    }

    /// Drop focus if it names `idx`. Called by `free_subtree`, where the node
    /// is already half-freed and its generation no longer matches anything.
    pub(crate) fn forget_focus(&mut self, idx: usize) {
        for slot in [
            &mut self.keyboard.focus,
            &mut self.keyboard.submitted,
            &mut self.keyboard.changed,
        ] {
            if slot.is_some_and(|n| n.idx as usize == idx) {
                *slot = None;
            }
        }
    }

    /// Fold this frame's keystrokes into the focused widget. Called by the
    /// renderer right after `update_pointer`, so a press that moved focus and
    /// the typing that followed it land in the order they happened.
    pub(crate) fn update_keyboard(&mut self, strokes: &[Keystroke]) {
        // One frame each, so they clear whether or not anything was typed.
        self.keyboard.submitted = None;
        self.keyboard.changed = None;
        if strokes.is_empty() {
            return;
        }

        for stroke in strokes {
            // Focus navigation outranks the focused widget: a field that
            // handled `Tab` itself would trap the keyboard inside it.
            if let Keystroke::Key(Key::Tab, m) = stroke {
                self.cycle_focus(m.has(Mods::SHIFT));
                continue;
            }
            let Some(n) = self.keyboard.focus else {
                continue;
            };
            if let Keystroke::Key(Key::Escape, _) = stroke {
                self.set_focus(None);
                continue;
            }
            self.field_keystroke(n, stroke);
        }
    }

    /// Move to the next (or previous) stop on the Tab ring, wrapping.
    ///
    /// The ring is rebuilt per press rather than maintained: it is a walk of
    /// the tree at human frequency, and a cached list would have to be
    /// invalidated by every add, remove, re-parent and collapse — five ways
    /// to be silently wrong to save a walk nobody can measure.
    fn cycle_focus(&mut self, back: bool) {
        let ring = self.focus_order();
        if ring.is_empty() {
            return;
        }
        let next = match self.keyboard.focus.and_then(|f| ring.iter().position(|&n| n == f)) {
            Some(i) if back => (i + ring.len() - 1) % ring.len(),
            Some(i) => (i + 1) % ring.len(),
            // Nothing focused: forwards starts at the top, backwards at the
            // bottom, so both directions reach a first stop.
            None if back => ring.len() - 1,
            None => 0,
        };
        let n = ring[next];
        self.set_focus(Some(n));
        // Tabbing into a field selects its contents, so the next keystroke
        // replaces the value — the convention every form has, and the reason
        // arriving by Tab differs from arriving by click, which places a
        // caret at a position the user chose.
        self.field_select_all(n);
    }
}
