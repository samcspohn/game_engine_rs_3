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

use super::{rgb, NodeId, UiCore, UiStyle};

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

/// How [`UiCore::button`] looks. `Default` is the built-in dark theme —
/// enough to get a usable button with one call, and a starting point to
/// clone and edit rather than a theming system.
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

impl Default for ButtonStyle {
    fn default() -> Self {
        Self {
            idle: rgb(0x2A, 0x31, 0x42),
            hover: rgb(0x39, 0x44, 0x5C),
            held: rgb(0x1E, 0x24, 0x30),
            text: rgb(0xE6, 0xE9, 0xEF),
            text_px: 11.0,
            padding: 6.0,
            radius: 4.0,
        }
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
}
