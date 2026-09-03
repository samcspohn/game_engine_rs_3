//! A strip of titles across the top, each opening an anchored menu.
//!
//! The capability is not the dropdown — that is `popup.rs`. It is that **an
//! open menu follows the pointer along the bar**: hovering a second title
//! switches to it with no click at all, because the press that would have
//! opened it is the one the overlay swallows to dismiss the first. So a bar
//! cannot be a row of buttons. It has to know whether the open overlay is
//! its own, and re-aim rather than re-open.
//!
//! Which is why a pick is read off the store like any other menu's, but
//! filtered by the bar's own node: a context menu somebody else opened is
//! not this bar's, and must neither be reported nor closed.
//!
//! Nested submenus are not built. `Overlay` holds one popup, and a submenu
//! is exactly the case that needs a stack of them.

use super::style::{px, AlignItems, Display, Rect, Style};
use super::widget::StateStyle;
use super::{theme, Events, MenuStyle, NodeId, Theme, UiCore, UiStyle};

/// What an open menu is about: which bar, and which of its titles.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Open {
    bar: NodeId,
    menu: usize,
}

/// How [`MenuBar`] looks: the strip, one title on it, and the menu a title
/// opens.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MenuBarStyle {
    pub fill: u32,
    /// A title under the pointer, and a title whose menu is open — the same
    /// fill, because by then the pointer is over the menu instead.
    pub hover: u32,
    pub text: u32,
    pub text_px: f32,
    /// Horizontal inside a title; half of it vertically, so the strip is a
    /// strip rather than a row of buttons.
    pub pad: f32,
    pub menu: MenuStyle,
}

impl From<Theme> for MenuBarStyle {
    fn from(t: Theme) -> Self {
        Self {
            fill: t.backdrop,
            hover: t.control_hover,
            text: t.text,
            text_px: t.text_px,
            pad: t.pad,
            menu: t.into(),
        }
    }
}

impl Default for MenuBarStyle {
    fn default() -> Self {
        theme().into()
    }
}

/// A menu bar. Caller-owned like a [`DockSpace`](super::DockSpace): build it
/// once, then call [`update`](MenuBar::update) a frame and act on what it
/// returns.
///
/// ```ignore
/// let bar = MenuBar::new(ui, shell, &[("File", &["quit"])], MenuBarStyle::default());
/// // …a frame:
/// if let Some((menu, item)) = bar.update(ui) { /* MENUS[menu].1[item] */ }
/// ```
#[derive(Clone)]
pub struct MenuBar {
    node: NodeId,
    titles: Vec<NodeId>,
    menus: Vec<Vec<String>>,
    open: Option<usize>,
    style: MenuBarStyle,
}

impl MenuBar {
    pub fn new(
        ui: &mut UiCore,
        parent: impl Into<NodeId>,
        menus: &[(&str, &[&str])],
        style: MenuBarStyle,
    ) -> Self {
        let node = ui.node(
            parent,
            Style {
                display: Display::Flex,
                align_items: Some(AlignItems::CENTER),
                ..Default::default()
            },
        );
        ui.set_background(node, UiStyle::fill(style.fill));
        Self {
            node,
            titles: menus
                .iter()
                .map(|(t, _)| title(ui, node, t, style))
                .collect(),
            menus: menus
                .iter()
                .map(|(_, items)| items.iter().map(|s| s.to_string()).collect())
                .collect(),
            open: None,
            style,
        }
    }

    /// The strip itself, for a caller that wants to lay something out
    /// against it.
    pub fn node(&self) -> NodeId {
        self.node
    }

    /// Drive the bar and report this frame's pick as `(menu, item)`, indices
    /// into what was handed to [`new`](Self::new).
    pub fn update(&mut self, ui: &mut UiCore) -> Option<(usize, usize)> {
        let pick = self.pick(ui);
        let live = self.open.filter(|_| self.owns(ui));
        let hovered = self.titles.iter().position(|&t| ui.hovered(t));
        let want = match (live, self.open) {
            // Open: a second title takes it on hover alone. The press that
            // would have opened it went into dismissing this one.
            (Some(i), _) => hovered.filter(|&h| h != i).or(live),
            // Dismissed by a press this frame: on another title that press
            // was that title's, and on the same one it was the toggle shut.
            (None, Some(i)) => hovered.filter(|&h| h != i),
            (None, None) => self.titles.iter().position(|&t| ui.clicked(t)),
        };
        if want != self.open {
            self.mark(ui, self.open, false);
            match want {
                Some(i) => self.open_menu(ui, i),
                // Only ours: a popup that expired itself is already gone,
                // and one somebody else opened is not the bar's to close.
                None if live.is_some() => ui.close_popup(),
                None => {}
            }
            self.mark(ui, want, true);
        }
        self.open = want;
        pick
    }

    fn pick(&self, ui: &UiCore) -> Option<(usize, usize)> {
        let (item, o) = ui.menu_choice::<Open>()?;
        (o.bar == self.node).then_some((o.menu, item))
    }

    fn owns(&self, ui: &UiCore) -> bool {
        ui.menu_payload::<Open>()
            .is_some_and(|o| o.bar == self.node)
    }

    fn open_menu(&self, ui: &mut UiCore, i: usize) {
        let r = ui.node_rect(self.titles[i]);
        let items: Vec<&str> = self.menus[i].iter().map(String::as_str).collect();
        let about = Open {
            bar: self.node,
            menu: i,
        };
        ui.context_menu([r[0], r[1] + r[3]], &items, about, self.style.menu);
    }

    fn mark(&self, ui: &mut UiCore, title: Option<usize>, open: bool) {
        let Some(n) = title.map(|i| self.titles[i]) else {
            return;
        };
        ui.set_state_style(n, look(self.style, open));
    }
}

fn title(ui: &mut UiCore, bar: NodeId, text: &str, style: MenuBarStyle) -> NodeId {
    let pad = style.pad;
    let n = ui.node(
        bar,
        Style {
            display: Display::Flex,
            padding: Rect {
                left: px(pad),
                right: px(pad),
                top: px(pad * 0.5),
                bottom: px(pad * 0.5),
            },
            ..Default::default()
        },
    );
    ui.label(n, style.text_px, style.text, text);
    ui.set_events(n, Events::CLICK | Events::HOVER);
    ui.set_state_style(n, look(style, false));
    n
}

fn look(style: MenuBarStyle, open: bool) -> StateStyle {
    let base = UiStyle::fill(0).radius(style.menu.popup.radius);
    let idle = if open { style.hover } else { 0 };
    StateStyle::fills(base, idle, style.hover, style.hover)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCREEN: [f32; 2] = [400.0, 300.0];
    const MENUS: [(&str, &[&str]); 2] = [
        ("File", &["new", "quit"]),
        ("Tools", &["translate", "rotate"]),
    ];

    struct Harness {
        core: UiCore,
        bar: MenuBar,
        now: f64,
    }

    impl Harness {
        fn new() -> Self {
            let mut core = UiCore::new();
            let root = core.root();
            let bar = MenuBar::new(&mut core, root, &MENUS, MenuBarStyle::default());
            let mut h = Self {
                core,
                bar,
                now: 0.0,
            };
            h.frame(AWAY, false, false);
            h
        }

        /// One frame in the renderer's order: pointer, then the widget, then
        /// the layout the widget's edits land in.
        fn frame(
            &mut self,
            pos: [f32; 2],
            pressed: bool,
            released: bool,
        ) -> Option<(usize, usize)> {
            self.now += 1.0;
            self.core
                .update_pointer(pos, pressed, released, 0.0, self.now);
            let pick = self.bar.update(&mut self.core);
            self.core.run_layout(SCREEN);
            pick
        }

        fn click(&mut self, pos: [f32; 2]) -> Option<(usize, usize)> {
            self.frame(pos, true, false);
            self.frame(pos, false, true)
        }

        fn title(&self, i: usize) -> [f32; 2] {
            centre(&self.core, self.bar.titles[i])
        }

        /// Where an item of the open menu is, by its label.
        fn item(&self, text: &str) -> [f32; 2] {
            let (n, ..) = self
                .core
                .text_nodes()
                .into_iter()
                .find(|(_, t, _)| t == text)
                .expect("no such item on screen");
            centre(&self.core, n)
        }

        /// What the open menu shows, which is everything on screen that is
        /// not a title.
        fn items(&self) -> Vec<String> {
            let titles: Vec<&str> = MENUS.iter().map(|(t, _)| *t).collect();
            self.core
                .text_nodes()
                .into_iter()
                .map(|(_, t, _)| t)
                .filter(|t| !titles.contains(&t.as_str()))
                .collect()
        }
    }

    /// Off the bar and off any menu it opens.
    const AWAY: [f32; 2] = [0.0, 200.0];

    fn centre(core: &UiCore, n: impl Into<NodeId>) -> [f32; 2] {
        let r = core.node_rect(n);
        [r[0] + r[2] * 0.5, r[1] + r[3] * 0.5]
    }

    /// The whole loop: a title opens its menu, and a pick names both halves.
    #[test]
    fn a_title_opens_its_menu_and_a_pick_names_both() {
        let mut h = Harness::new();
        assert!(h.items().is_empty(), "shut until asked");

        let tools = h.title(1);
        h.click(tools);
        assert_eq!(h.items(), ["translate", "rotate"], "the second menu");

        let rotate = h.item("rotate");
        assert_eq!(h.click(rotate), Some((1, 1)), "Tools → rotate");
        assert_eq!(h.frame(AWAY, false, false), None, "reported once");
        assert!(h.items().is_empty(), "and gone");
    }

    /// The capability: with one menu open, the next title needs no click.
    #[test]
    fn hovering_a_second_title_switches_the_open_menu() {
        let mut h = Harness::new();
        let (file, tools) = (h.title(0), h.title(1));
        h.click(file);
        assert_eq!(h.items(), ["new", "quit"]);

        h.frame(tools, false, false);
        assert_eq!(h.items(), ["translate", "rotate"], "switched on hover");

        // …and staying on the one already open leaves it alone rather than
        // rebuilding it every frame.
        h.frame(tools, false, false);
        let epoch = h.core.layout_epoch;
        h.frame(tools, false, false);
        assert_eq!(h.core.layout_epoch, epoch, "settled");
    }

    /// A press on the open title shuts it — the press the overlay swallows
    /// to dismiss is the same one the bar reads as the toggle.
    #[test]
    fn pressing_the_open_title_shuts_it() {
        let mut h = Harness::new();
        let file = h.title(0);
        h.click(file);
        h.click(file);

        assert!(h.items().is_empty(), "shut, and not reopened");
        assert!(!h.core.popup_open());
    }

    /// A press on another title switches, rather than only dismissing and
    /// making the user click twice.
    #[test]
    fn pressing_another_title_switches() {
        let mut h = Harness::new();
        let (file, tools) = (h.title(0), h.title(1));
        h.click(file);
        h.click(tools);

        assert_eq!(h.items(), ["translate", "rotate"]);
    }

    /// A context menu somebody else opened is not the bar's: it reports no
    /// pick from it, and does not take it down.
    #[test]
    fn a_menu_it_did_not_open_is_left_alone() {
        let mut h = Harness::new();
        let file = h.title(0);
        h.click(file);

        h.core
            .context_menu([200.0, 200.0], &["theirs"], 7u32, MenuStyle::default());
        assert_eq!(h.frame(AWAY, false, false), None);
        assert!(h.core.popup_open(), "still theirs");
        assert_eq!(h.items(), ["theirs"], "and the bar's is the one that went");
    }

    /// Nothing happening costs nothing.
    #[test]
    fn an_idle_bar_relayouts_nothing() {
        let mut h = Harness::new();
        h.frame(AWAY, false, false);

        let epoch = h.core.layout_epoch;
        h.frame(AWAY, false, false);
        assert_eq!(h.core.layout_epoch, epoch);
    }
}
