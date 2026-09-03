//! Keyboard focus and the keystroke route — the pointer layer's twin.
//!
//! The pointer asks "what is under this position?"; the keyboard has no
//! position, so it keeps one *retained* target instead. There is one
//! keyboard, so focus lives here rather than in whichever widget wants it.
//!
//! A press takes focus, a press anywhere else drops it, `Tab` walks the ring.
//! Nothing happens per frame: `update_keyboard` clears two `Option`s and
//! returns, which is also why the caret does not blink.

use crate::input::{Key, Keystroke, Mods};

use super::{NodeId, UiCore};

#[derive(Default)]
pub(crate) struct Keyboard {
    /// Who receives keystrokes. `None` means the application does.
    pub(crate) focus: Option<NodeId>,
    /// One frame each, like the pointer's `clicked`.
    pub(crate) submitted: Option<NodeId>,
    pub(crate) changed: Option<NodeId>,
}

impl UiCore {
    pub fn focus(&self) -> Option<NodeId> {
        self.keyboard.focus
    }

    pub fn focused(&self, n: impl Into<NodeId>) -> bool {
        let n = n.into();
        self.keyboard.focus == Some(n)
    }

    /// Move focus, or clear it with `None`. Both ends are re-synced, which is
    /// what draws and undraws a caret; a no-op for nodes that are not fields.
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

    /// A field has the keyboard, so game hotkeys should sit this frame out.
    /// The keyboard twin of [`pointer_captured`](UiCore::pointer_captured).
    pub fn keyboard_captured(&self) -> bool {
        self.keyboard.focus.is_some()
    }

    /// Drop focus if it names `idx`, so a recycled slot cannot inherit it.
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

    /// Fold this frame's keystrokes into the focused widget. Called right
    /// after `update_pointer`, so a press that moved focus lands first.
    pub(crate) fn update_keyboard(&mut self, strokes: &[Keystroke]) {
        self.keyboard.submitted = None;
        self.keyboard.changed = None;
        if strokes.is_empty() {
            return;
        }

        for stroke in strokes {
            // Ahead of the focused widget: a field that handled `Tab` itself
            // would trap the keyboard inside it.
            if let Keystroke::Key(Key::Tab, m) = stroke {
                self.cycle_focus(m.has(Mods::SHIFT));
                continue;
            }
            // Ahead of the focus check: a menu holds no focus, so an Escape
            // with nothing to unfocus must still take one down.
            if let Keystroke::Key(Key::Escape, _) = stroke {
                self.close_popup();
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

    /// Next (or previous) stop on the Tab ring, wrapping. The ring is rebuilt
    /// per press — a cached one would need invalidating by every add, remove,
    /// re-parent and collapse.
    fn cycle_focus(&mut self, back: bool) {
        let ring = self.focus_order();
        if ring.is_empty() {
            return;
        }
        let next = match self.keyboard.focus.and_then(|f| ring.iter().position(|&n| n == f)) {
            Some(i) if back => (i + ring.len() - 1) % ring.len(),
            Some(i) => (i + 1) % ring.len(),
            None if back => ring.len() - 1,
            None => 0,
        };
        let n = ring[next];
        self.set_focus(Some(n));
        // Arriving by Tab selects the contents; arriving by click places a
        // caret where the user aimed.
        self.field_select_all(n);
    }
}
