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
//! A menu item is a closure: the pick runs it, after the menu has closed.

use super::style::{
    Display, FlexDirection, LengthPercentageAuto, Position, Rect, Style, TaffyAuto, px,
};
use super::widget::StateStyle;
use super::{Events, Menu, NodeId, Popup, Theme, UiCore, UiStyle, theme};

/// The overlay in flight.
pub(crate) struct Overlay {
    pub(crate) node: NodeId,
    /// Where the caller asked for it. Kept because the clamp is re-derived
    /// from the solved box on every layout, not stored as a corrected point.
    pub(crate) at: [f32; 2],
    /// How a [`Menu::item`] looks, when this is a menu.
    menu: Option<MenuStyle>,
    on_close: Vec<Box<dyn FnOnce(&mut UiCore) + Send>>,
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
            menu: None,
            on_close: Vec::new(),
        });
        Popup::from_node(n)
    }

    /// Open an empty context menu at `at`, for [`Menu::item`] to fill.
    pub fn context_menu(&mut self, at: [f32; 2], style: MenuStyle) -> Menu {
        let popup = self.popup(at, style.popup);
        if let Some(o) = self.overlay.as_mut() {
            o.menu = Some(style);
        }
        Menu::from_node(popup.node())
    }

    /// The open overlay's node, for a widget that has to tell its own apart.
    pub(crate) fn popup_node(&self) -> Option<NodeId> {
        self.overlay.as_ref().map(|o| o.node)
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
        o.on_close.into_iter().for_each(|f| self.later(f));
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
}

impl Menu {
    /// Add a row that runs `f` when picked. The menu closes first, so `f` may
    /// open another. Note: does nothing once this menu has closed.
    pub fn item(
        self,
        ui: &mut UiCore,
        text: &str,
        f: impl FnOnce(&mut UiCore) + Send + 'static,
    ) -> Self {
        let Some(style) = ui
            .overlay
            .as_ref()
            .filter(|o| o.node == self.node())
            .and_then(|o| o.menu)
        else {
            return self;
        };
        let item = ui.node(
            self,
            Style {
                display: Display::Flex,
                padding: Rect::length(style.item_pad),
                ..Default::default()
            },
        );
        ui.set_state_style(
            item,
            StateStyle::fills(
                UiStyle::fill(0).radius(style.popup.radius - 1.0),
                0,
                style.hover,
                style.hover,
            ),
        );
        ui.label(item, style.text_px, style.text, text);
        ui.set_events(item, Events::HOVER);
        let mut f = Some(f);
        ui.on_click(item, move |ui| {
            ui.close_popup();
            if let Some(f) = f.take() {
                f(ui);
            }
        });
        self
    }

    /// Run `f` once the menu is gone, whatever closed it.
    pub fn on_close(self, ui: &mut UiCore, f: impl FnOnce(&mut UiCore) + Send + 'static) -> Self {
        match ui.overlay.as_mut().filter(|o| o.node == self.node()) {
            Some(o) => o.on_close.push(Box::new(f)),
            None => ui.later(f),
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCREEN: [f32; 2] = [400.0, 300.0];

    type Picks = std::sync::Arc<parking_lot::Mutex<Vec<&'static str>>>;

    /// What the hierarchy panel does: three rows, each saying it ran.
    fn menu_into(core: &mut UiCore, at: [f32; 2], picks: &Picks) -> Menu {
        let mut m = core.context_menu(at, MenuStyle::default());
        for name in ["new child", "rename", "delete"] {
            let picks = picks.clone();
            m = m.item(core, name, move |_| picks.lock().push(name));
        }
        core.run_layout(SCREEN);
        m
    }

    fn menu(core: &mut UiCore, at: [f32; 2]) -> Menu {
        menu_into(core, at, &Picks::default())
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

    /// The whole loop: open, pick, and the item runs with the menu gone.
    #[test]
    fn a_pick_runs_its_item_and_closes_the_menu() {
        let mut core = UiCore::new();
        let picks = Picks::default();
        let m = menu_into(&mut core, [10.0, 10.0], &picks);

        let pick = item(&core, m, 1);
        click(&mut core, pick);
        core.dispatch(0.0);
        assert_eq!(*picks.lock(), ["rename"], "the item that was picked");
        assert!(!core.popup_open(), "and the menu closed itself");
    }

    /// Dismissed or picked, whoever opened it hears that it went.
    #[test]
    fn on_close_runs_however_the_menu_goes() {
        let mut core = UiCore::new();
        let closed = Picks::default();
        let c = closed.clone();
        menu(&mut core, [10.0, 10.0]).on_close(&mut core, move |_| c.lock().push("closed"));
        click(&mut core, [300.0, 250.0]);
        core.dispatch(0.0);
        assert_eq!(*closed.lock(), ["closed"]);
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

        let r = core.node_rect(p);
        assert!(r[2] > 0.0, "sized by what the caller put in it");
    }
}
