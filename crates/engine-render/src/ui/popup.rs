//! Overlay lifetime, and the context menu built on it.
//!
//! There is **one** popup, for the same reason there is one [`Grab`]: it
//! belongs to the pointer rather than to whichever widget opened it. Opening
//! a second closes the first, so nothing has to arbitrate between two menus
//! and no caller can leak one by forgetting to close it.
//!
//! Three rules are the whole of the lifetime:
//!
//! - **Minted at the root**, so it paints over everything already built —
//!   the same trick the drag ghost uses, and why no `raise` is needed.
//! - **Dismissed by the next press outside it**, which is swallowed: the
//!   click that closes a menu must not also press what was behind it.
//! - **Clamped to the window** after the solve, so an item near the right
//!   edge opens inwards instead of off-screen.
//!
//! A menu's choice travels the way a drag's payload does — an opaque
//! `Box<dyn Any>` on the store, recovered by type. The caller stores no
//! handle, so there is no stale one to poll after the menu has closed.

use std::any::Any;

use super::style::{
    px, Display, FlexDirection, LengthPercentageAuto, Position, Rect, Style, TaffyAuto,
};
use super::widget::{Control, StateStyle};
use super::{theme, Events, Menu, NodeId, Popup, Theme, UiCore, UiStyle};

/// The overlay in flight.
pub(crate) struct Overlay {
    pub(crate) node: NodeId,
    /// Where the caller asked for it. Kept because the clamp is re-derived
    /// from the solved box on every layout, not stored as a corrected point.
    pub(crate) at: [f32; 2],
    /// What a menu is *about*, opaque exactly as a drag payload is.
    pub(crate) payload: Option<Box<dyn Any + Send>>,
    /// The item picked, for the one frame the application reads it in.
    pub(crate) choice: Option<usize>,
}

/// How [`UiCore::popup`] looks: a floating panel, which is a fill, a
/// hairline and a radius.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PopupStyle {
    pub fill: u32,
    pub border: u32,
    pub radius: f32,
    pub padding: f32,
}

impl From<Theme> for PopupStyle {
    fn from(t: Theme) -> Self {
        Self {
            // Opaque where a docked panel is translucent: a menu sits over
            // arbitrary content, and reading through it is not a look.
            fill: t.control,
            border: t.outline,
            radius: t.radius,
            padding: 3.0,
        }
    }
}

impl Default for PopupStyle {
    fn default() -> Self {
        theme().into()
    }
}

/// How [`UiCore::context_menu`] looks. The popup it sits in, plus what one
/// item is.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MenuStyle {
    pub popup: PopupStyle,
    pub hover: u32,
    pub text: u32,
    pub text_px: f32,
    /// Inside one item, so a menu's width is its longest label plus this.
    pub item_pad: f32,
}

impl From<Theme> for MenuStyle {
    fn from(t: Theme) -> Self {
        Self {
            popup: t.into(),
            hover: t.control_hover,
            text: t.text,
            text_px: t.text_px,
            item_pad: 5.0,
        }
    }
}

impl Default for MenuStyle {
    fn default() -> Self {
        theme().into()
    }
}

impl UiCore {
    /// Open a bare overlay at `at`, in window px, and return it to be filled.
    ///
    /// Closes whatever was already open. The node is the caller's to add to;
    /// its lifetime is not — see the module docs.
    pub fn popup(&mut self, at: [f32; 2], style: PopupStyle) -> Popup {
        self.close_popup();
        let root = self.root();
        let n = self.node(
            root,
            Style {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                padding: Rect::length(style.padding),
                ..Default::default()
            },
        );
        self.place_popup(n, at, [f32::MAX; 2]);
        self.set_background(
            n,
            UiStyle::fill(style.fill)
                .border(style.border, 1.0)
                .radius(style.radius),
        );
        // Not for a click of its own — so that the press it takes is one the
        // panel underneath does not also get. A popup occludes.
        self.set_events(n, Events::CLICK | Events::HOVER);
        self.overlay = Some(Overlay {
            node: n,
            at,
            payload: None,
            choice: None,
        });
        Popup::from_node(n)
    }

    /// Open a context menu at `at` — one clickable row per label, carrying
    /// `payload`: what the choice will be about.
    ///
    /// Read it back with [`menu_choice`](Self::menu_choice) on the frame the
    /// user picks, and nothing on any other. The menu closes itself the frame
    /// after, so there is no handle to hold and none to invalidate.
    pub fn context_menu<T: Any + Send>(
        &mut self,
        at: [f32; 2],
        items: &[&str],
        payload: T,
        style: MenuStyle,
    ) -> Menu {
        let popup = self.popup(at, style.popup);
        for (i, text) in items.iter().enumerate() {
            let item = self.node(
                popup,
                Style {
                    display: Display::Flex,
                    padding: Rect::length(style.item_pad),
                    ..Default::default()
                },
            );
            self.set_state_style(
                item,
                StateStyle::fills(
                    UiStyle::fill(0).radius(style.popup.radius - 1.0),
                    0,
                    style.hover,
                    style.hover,
                ),
            );
            self.label(item, style.text_px, style.text, text);
            self.set_events(item, Events::CLICK | Events::HOVER);
            self.set_control(item, Control::MenuItem { index: i });
        }
        self.overlay
            .as_mut()
            .expect("the popup just opened")
            .payload = Some(Box::new(payload));
        Menu::from_node(popup.node())
    }

    /// The item picked out of the open menu this frame, and what it was
    /// opened about. `None` when nothing was picked, *or* when the menu
    /// belongs to somebody else — the same refusal
    /// [`dragging`](Self::dragging) makes, for the same reason.
    pub fn menu_choice<T: Any>(&self) -> Option<(usize, &T)> {
        let o = self.overlay.as_ref()?;
        Some((o.choice?, o.payload.as_ref()?.downcast_ref()?))
    }

    /// Whether an overlay is open. A caller that opens one on right-click
    /// asks so the second click re-aims it rather than stacking.
    pub fn popup_open(&self) -> bool {
        self.overlay.is_some()
    }

    /// Close the open overlay, if any. Idempotent, and safe to call on the
    /// frame the popup was opened.
    pub fn close_popup(&mut self) {
        // Cleared before the removal, not after: `free_subtree` reaches this
        // same field, and a half-removed node must not be reachable from it.
        let Some(o) = self.overlay.take() else { return };
        self.remove_node(o.node);
    }

    /// Park the overlay at `at`, pulled back inside `limit` when its own box
    /// would hang off the edge. `limit` of infinity is "not measured yet",
    /// which is the frame it is built on.
    fn place_popup(&mut self, n: NodeId, at: [f32; 2], limit: [f32; 2]) {
        let mut s = self.node_style(n);
        s.position = Position::Absolute;
        s.inset = Rect {
            left: px(at[0].min(limit[0]).max(0.0)),
            top: px(at[1].min(limit[1]).max(0.0)),
            right: LengthPercentageAuto::AUTO,
            bottom: LengthPercentageAuto::AUTO,
        };
        self.set_node_style(n, s);
    }

    /// Re-clamp the open popup against the window, now that the solve has
    /// given it a size. `true` when that moved it, which is the caller's cue
    /// to solve again — a menu must not be visible for a frame hanging off
    /// the screen, so this is re-solved rather than deferred like a
    /// scrollbar's re-fit.
    pub(crate) fn fit_popup(&mut self, screen: [f32; 2]) -> bool {
        let Some(o) = self.overlay.as_ref() else {
            return false;
        };
        let (n, at) = (o.node, o.at);
        // Taffy's box and not `node_rect`'s: the placement walk that fills
        // `absolute` has not run yet, so that would be last frame's size.
        let size = self.solved_size(n);
        let limit = [screen[0] - size[0], screen[1] - size[1]];
        let before = self.node_style(n).inset;
        self.place_popup(n, at, limit);
        self.node_style(n).inset != before
    }

    /// Fold a press into the overlay's lifetime: `true` when it dismissed
    /// one, which also means the press goes no further.
    pub(crate) fn dismiss_popup(&mut self, pos: [f32; 2]) -> bool {
        let Some(o) = self.overlay.as_ref() else {
            return false;
        };
        let r = self.node_rect(o.node);
        if (0..2).all(|k| pos[k] >= r[k] && pos[k] < r[k] + r[k + 2]) {
            return false;
        }
        self.close_popup();
        true
    }

    /// The frame after a pick, close. One frame late on purpose, and the
    /// same deferral a drop's payload gets: the frame it stays open for is
    /// the frame the application reads the choice in.
    pub(crate) fn expire_popup(&mut self) {
        if self.overlay.as_ref().is_some_and(|o| o.choice.is_some()) {
            self.close_popup();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCREEN: [f32; 2] = [400.0, 300.0];

    /// What the hierarchy panel does: a menu about an entity.
    #[derive(Clone, Copy, PartialEq, Debug)]
    struct About(u64);

    fn menu(core: &mut UiCore, at: [f32; 2]) -> Menu {
        let m = core.context_menu(
            at,
            &["new child", "rename", "delete"],
            About(7),
            MenuStyle::default(),
        );
        core.run_layout(SCREEN);
        m
    }

    /// Centre of the `i`th item.
    fn item(core: &UiCore, m: Menu, i: usize) -> [f32; 2] {
        let r = core.node_rect(m);
        let h = (r[3] - 6.0) / 3.0;
        [r[0] + r[2] * 0.5, r[1] + 3.0 + h * (i as f32 + 0.5)]
    }

    fn click(core: &mut UiCore, p: [f32; 2]) {
        core.update_pointer(p, true, false, 0.0, 0.0);
        core.update_pointer(p, false, true, 0.0, 0.0);
    }

    /// The whole loop: open, pick, read, and the menu is gone without the
    /// caller closing it.
    #[test]
    fn a_pick_is_read_once_and_closes_the_menu() {
        let mut core = UiCore::new();
        let m = menu(&mut core, [10.0, 10.0]);

        assert_eq!(core.menu_choice::<About>(), None, "nothing picked yet");
        let pick = item(&core, m, 1);
        click(&mut core, pick);
        assert_eq!(
            core.menu_choice::<About>(),
            Some((1, &About(7))),
            "the pick, with what it is about"
        );

        // The next frame's pointer pass is what takes it down — one frame
        // late, which is the frame the application read it in.
        core.update_pointer([0.0, 0.0], false, false, 0.0, 1.0);
        assert!(!core.popup_open(), "closed itself");
        assert_eq!(core.menu_choice::<About>(), None);
    }

    /// A menu somebody else opened is not yours, exactly as a drag is not.
    #[test]
    fn a_choice_is_refused_to_the_wrong_type() {
        let mut core = UiCore::new();
        let m = menu(&mut core, [10.0, 10.0]);
        let pick = item(&core, m, 0);
        click(&mut core, pick);

        assert_eq!(core.menu_choice::<About>(), Some((0, &About(7))));
        assert_eq!(core.menu_choice::<String>(), None, "not this caller's menu");
    }

    /// The press that dismisses must not also reach what the menu covered —
    /// which for a context menu is the panel that opened it.
    #[test]
    fn an_outside_press_dismisses_and_is_swallowed() {
        let mut core = UiCore::new();
        let root = core.root();
        let button = core.button(root, "delete", crate::ui::ButtonStyle::default());
        core.run_layout(SCREEN);
        let b = core.node_rect(button);
        let on_button = [b[0] + b[2] * 0.5, b[1] + b[3] * 0.5];

        // Opened somewhere the button is not, so the dismissing press lands
        // on the button and nothing else.
        menu(&mut core, [200.0, 200.0]);
        click(&mut core, on_button);

        assert!(!core.popup_open(), "dismissed");
        assert!(
            !core.clicked(button),
            "the press that closed the menu went no further"
        );

        // …and the next one does reach it, so this suppresses one press and
        // not the button.
        click(&mut core, on_button);
        assert!(core.clicked(button));
    }

    /// Opened at the bottom-right corner, a menu opens inwards — and on the
    /// frame it is built, not the one after.
    #[test]
    fn it_is_clamped_inside_the_window_the_frame_it_opens() {
        let mut core = UiCore::new();
        let m = menu(&mut core, [SCREEN[0] - 4.0, SCREEN[1] - 4.0]);
        let r = core.node_rect(m);

        assert!(r[2] > 4.0 && r[3] > 4.0, "a menu with items has a box");
        assert!(r[0] + r[2] <= SCREEN[0] + 1.0, "right edge inside: {r:?}");
        assert!(r[1] + r[3] <= SCREEN[1] + 1.0, "bottom edge inside: {r:?}");
    }

    /// One popup, because there is one pointer to dismiss it with.
    #[test]
    fn opening_a_second_closes_the_first() {
        let mut core = UiCore::new();
        let first = core.popup([10.0, 10.0], PopupStyle::default());
        core.label(first, 11.0, 0xFFFFFFFF, "first");
        core.popup([50.0, 50.0], PopupStyle::default());
        core.run_layout(SCREEN);

        assert!(core.popup_open());
        let shown = core.text_nodes();
        assert!(
            !shown.iter().any(|(_, t, _)| t == "first"),
            "the first is gone from the tree, not merely hidden"
        );
    }

    /// The store's own bookkeeping: a popup removed by its owner leaves
    /// nothing behind to close.
    #[test]
    fn removing_the_node_forgets_the_overlay() {
        let mut core = UiCore::new();
        let p = core.popup([10.0, 10.0], PopupStyle::default());
        core.remove_node(p);

        assert!(!core.popup_open());
        core.close_popup(); // must not reach a stale handle
    }

    /// An idle frame with a menu open still uploads nothing — the clamp is a
    /// comparison once it holds, not a write per frame.
    #[test]
    fn an_open_menu_costs_nothing_per_frame() {
        let mut core = UiCore::new();
        menu(&mut core, [10.0, 10.0]);
        core.run_layout(SCREEN);

        let epoch = core.layout_epoch;
        core.run_layout(SCREEN);
        assert_eq!(core.layout_epoch, epoch, "settled: no relayout");
    }

    /// The gesture that opens one: a secondary click names its node, and the
    /// primary's `clicked` stays quiet — so the press that asks for a menu
    /// does not also select the row it asked on.
    #[test]
    fn a_right_click_names_its_node_without_clicking_it() {
        let mut core = UiCore::new();
        let root = core.root();
        let row = core.button(root, "row", crate::ui::ButtonStyle::default());
        core.run_layout(SCREEN);
        let r = core.node_rect(row);
        let p = [r[0] + r[2] * 0.5, r[1] + r[3] * 0.5];

        core.update_pointer(p, false, false, 0.0, 0.0);
        core.update_secondary(true, false);
        core.update_secondary(false, true);
        assert!(core.right_clicked(row));
        assert!(
            !core.clicked(row),
            "the other button is a different question"
        );

        core.update_pointer(p, false, false, 0.0, 1.0);
        assert!(
            !core.right_clicked(row),
            "one frame, like every pointer event here"
        );
    }

    /// A bare popup is a box the caller fills — the lifetime is what it is
    /// for, not the contents.
    #[test]
    fn a_bare_popup_takes_content_and_has_no_choice() {
        let mut core = UiCore::new();
        let p = core.popup([10.0, 10.0], PopupStyle::default());
        core.label(p, 11.0, 0xFFFFFFFF, "anything at all");
        core.run_layout(SCREEN);

        assert_eq!(core.menu_choice::<About>(), None, "not a menu");
        let r = core.node_rect(p);
        assert!(r[2] > 0.0, "sized by what the caller put in it");
    }
}
