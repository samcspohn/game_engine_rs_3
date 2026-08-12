//! The palette every widget style defaults from.
//!
//! `ButtonStyle` and `RowStyle` each used to carry their own hex literals,
//! and so did the editor's chrome and the demo overlay — four copies of
//! the same dark palette, already drifting (the same hover blue written
//! twice, two different panel alphas). A third widget with its own copy is
//! what ADR-0008 names as the trigger for this step.
//!
//! A [`Theme`] is **semantic roles, not widget looks**: `control_hover`
//! rather than `button_hover`, so a checkbox and a slider ask for the same
//! role and match by construction instead of by someone remembering the
//! number. Widget style structs stay as they are — the theme only supplies
//! what their `Default` was hardcoding, and any field is still overridable
//! per widget by editing the struct after `..Default::default()`.
//!
//! # Chosen once, at startup
//!
//! [`set_theme`] replaces the process palette. It does **not** restyle
//! widgets already built: their colours were resolved when their style
//! struct was constructed, and re-deriving them would need the store to
//! keep a per-widget record of which role each colour came from — the
//! bounded widget-state table ADR-0008 defers to steps 5–7. So set the
//! theme before building UI. Live theme editing is not supported, and
//! silently half-applying it would be worse than not having it.

use std::sync::RwLock;

use super::{rgb, rgba};

/// Semantic colour roles plus the three metrics every widget was
/// duplicating. `Copy`, so reading it is a load rather than a borrow of
/// the lock.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Theme {
    /// Deepest surface — behind scrollable content, under everything else.
    pub backdrop: u32,
    /// Panel and window fill. Usually translucent over the viewport.
    pub panel: u32,
    /// A control at rest: button fill, header strip.
    pub control: u32,
    /// Pointer over the control.
    pub control_hover: u32,
    /// Pointer held down on the control.
    pub control_held: u32,
    /// Selected row or item.
    pub selection: u32,
    /// Body text.
    pub text: u32,
    /// De-emphasised text: captions, disclosure arrows, units.
    pub text_dim: u32,
    /// Text that must win against `selection` or `control_held`.
    pub text_strong: u32,
    /// The application's identity colour — focus rings, active tabs.
    /// Editor and game deliberately differ here.
    pub accent: u32,
    /// Hairline borders. Kept separate from `accent` rather than derived,
    /// so changing one does not silently change the other.
    pub outline: u32,
    pub text_px: f32,
    pub radius: f32,
    pub pad: f32,
}

impl Theme {
    /// The built-in dark palette — what the widget defaults resolved to
    /// before there was a theme.
    pub const DARK: Theme = Theme {
        backdrop: rgba(0x0B, 0x0E, 0x16, 0xFF),
        panel: rgba(0x14, 0x18, 0x22, 0xE0),
        control: rgb(0x2A, 0x31, 0x42),
        control_hover: rgb(0x39, 0x44, 0x5C),
        control_held: rgb(0x1E, 0x24, 0x30),
        selection: rgb(0x2F, 0x5B, 0x8F),
        text: rgb(0xE6, 0xE9, 0xEF),
        text_dim: rgb(0x8A, 0x93, 0xA6),
        text_strong: rgb(0xFF, 0xFF, 0xFF),
        accent: rgb(0x98, 0xC3, 0x79),
        outline: rgba(0x98, 0xC3, 0x79, 0x60),
        text_px: 11.0,
        radius: 4.0,
        pad: 6.0,
    };
}

impl Default for Theme {
    fn default() -> Self {
        Self::DARK
    }
}

static THEME: RwLock<Theme> = RwLock::new(Theme::DARK);

/// The active palette.
pub fn theme() -> Theme {
    *THEME.read().expect("theme lock poisoned")
}

/// Replace the active palette. Call before building UI — see the module
/// docs on why this does not restyle existing widgets.
pub fn set_theme(t: Theme) {
    *THEME.write().expect("theme lock poisoned") = t;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::{ButtonStyle, RowStyle};

    /// The point of the indirection: widget looks are *derived* from roles
    /// rather than baked, and the same role reaches both widgets.
    ///
    /// Asserted against `From<Theme>` rather than by calling `set_theme`,
    /// because the palette is process-global and tests run in parallel —
    /// a test that mutated it would flake every other test's colours.
    #[test]
    fn widget_styles_derive_from_roles() {
        let hot = Theme { control_hover: rgb(1, 2, 3), text_dim: rgb(4, 5, 6), ..Theme::DARK };

        assert_eq!(ButtonStyle::from(hot).hover, hot.control_hover);
        assert_eq!(RowStyle::from(hot).hover, hot.control_hover, "one role, both widgets");
        assert_eq!(RowStyle::from(hot).arrow, hot.text_dim);
    }

    /// `Default` is the active palette, so an app that calls `set_theme`
    /// before building UI gets themed widgets with no other change.
    #[test]
    fn defaults_resolve_against_the_active_theme() {
        assert_eq!(ButtonStyle::default(), ButtonStyle::from(theme()));
        assert_eq!(RowStyle::default(), RowStyle::from(theme()));
    }
}
