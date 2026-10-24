//! A strip of titles across the top, each opening an anchored menu.
//!
//! The capability is not the dropdown — that is `popup.rs`. It is that **an
//! open menu follows the pointer along the bar**: hovering a second title
//! switches to it with no click at all, because the press that would have
//! opened it is the one the overlay swallows to dismiss the first. So a bar
//! cannot be a row of buttons. It has to know whether the open overlay is
//! its own, and re-aim rather than re-open.
//!
//! Nested submenus are not built. `Overlay` holds one popup, and a submenu
//! is exactly the case that needs a stack of them.

use std::sync::Arc;

use super::style::{AlignItems, Display, Rect, Style, px};
use super::widget::StateStyle;
use super::{Events, MenuStyle, NodeId, Theme, UiCore, UiRef, UiStyle, theme};

type Item = Arc<dyn Fn(&mut UiCore) + Send + Sync>;

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

/// A menu bar: titles added with [`menu`](MenuBar::menu), items with
/// [`BarMenu::item`]. Its state lives on its node.
///
/// ```ignore
/// let bar = MenuBar::new(ui, shell, MenuBarStyle::default());
/// bar.menu(ui, "File").item(ui, "quit", |_| std::process::exit(0));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MenuBar(UiRef<Bar>);

/// One title of a [`MenuBar`], to add items to.
#[derive(Clone, Copy, Debug)]
pub struct BarMenu {
    bar: MenuBar,
    index: usize,
}

struct Bar {
    titles: Vec<NodeId>,
    menus: Vec<Vec<(String, Item)>>,
    /// Which title is open, and the menu it opened.
    open: Option<(usize, NodeId)>,
    style: MenuBarStyle,
}

impl Bar {
    fn mark(&self, ui: &mut UiCore, title: Option<usize>, open: bool) {
        if let Some(n) = title.map(|i| self.titles[i]) {
            ui.set_state_style(n, look(self.style, open));
        }
    }
}

impl MenuBar {
    pub fn new(ui: &mut UiCore, parent: impl Into<NodeId>, style: MenuBarStyle) -> Self {
        let node = ui.node(
            parent,
            Style {
                display: Display::Flex,
                align_items: Some(AlignItems::CENTER),
                ..Default::default()
            },
        );
        ui.set_background(node, UiStyle::fill(style.fill));
        let bar = Bar {
            titles: Vec::new(),
            menus: Vec::new(),
            open: None,
            style,
        };
        Self(ui.own(node, bar))
    }

    /// The strip itself, for a caller that wants to lay something out
    /// against it.
    pub fn node(self) -> NodeId {
        self.0.node()
    }

    /// Add a title to the end of the bar.
    pub fn menu(self, ui: &mut UiCore, text: &str) -> BarMenu {
        let node = self.node();
        let Some((t, index)) = self.0.with(ui, |b, ui| {
            let t = title(ui, node, text, b.style);
            b.titles.push(t);
            b.menus.push(Vec::new());
            (t, b.titles.len() - 1)
        }) else {
            return BarMenu {
                bar: self,
                index: 0,
            };
        };
        // A press with a menu open is swallowed dismissing it, so a click
        // only ever reaches a title when none of ours is open.
        ui.on_click(t, move |ui| self.open(ui, index));
        ui.on_hover(t, move |ui, entered| {
            let open = self.0.with(ui, |b, _| b.open).flatten();
            let ours = open.filter(|&(i, m)| i != index && ui.popup_node() == Some(m));
            if entered && ours.is_some() {
                self.open(ui, index);
            }
        });
        BarMenu { bar: self, index }
    }

    /// Open title `i`'s menu, closing whatever was open.
    fn open(self, ui: &mut UiCore, i: usize) {
        let Some((at, items, style)) = self.0.with(ui, |b, ui| {
            b.mark(ui, b.open.map(|(j, _)| j), false);
            b.mark(ui, Some(i), true);
            let r = ui.node_rect(b.titles[i]);
            ([r[0], r[1] + r[3]], b.menus[i].clone(), b.style.menu)
        }) else {
            return;
        };
        let mut m = ui.context_menu(at, style);
        for (text, f) in items {
            m = m.item(ui, &text, move |ui| f(ui));
        }
        m.on_close(ui, move |ui| self.closed(ui, m.node()));
        self.0.with(ui, |b, _| b.open = Some((i, m.node())));
    }

    /// `menu` went. The press that dismissed it was a title's, if it landed
    /// on another one — so that one opens rather than taking a second click.
    fn closed(self, ui: &mut UiCore, menu: NodeId) {
        let shut = self.0.with(ui, |b, ui| {
            let (i, _) = b.open.filter(|&(_, m)| m == menu)?;
            b.mark(ui, Some(i), false);
            b.open = None;
            let next = b.titles.iter().position(|&t| ui.hovered(t));
            next.filter(|&h| h != i)
        });
        if let Some(h) = shut.flatten() {
            self.open(ui, h);
        }
    }
}

impl BarMenu {
    /// Add a row that runs `f` when picked.
    pub fn item(
        self,
        ui: &mut UiCore,
        text: &str,
        f: impl Fn(&mut UiCore) + Send + Sync + 'static,
    ) -> Self {
        let item: Item = Arc::new(f);
        self.bar.0.with(ui, |b, _| {
            b.menus[self.index].push((text.to_string(), item))
        });
        self
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

    type Picks = Arc<parking_lot::Mutex<Vec<(usize, usize)>>>;

    struct Harness {
        core: UiCore,
        bar: MenuBar,
        picks: Picks,
        now: f64,
    }

    impl Harness {
        fn new() -> Self {
            let mut core = UiCore::new();
            let root = core.root();
            let bar = MenuBar::new(&mut core, root, MenuBarStyle::default());
            let picks = Picks::default();
            for (m, (title, items)) in MENUS.iter().enumerate() {
                let mut menu = bar.menu(&mut core, title);
                for (i, text) in items.iter().enumerate() {
                    let picks = picks.clone();
                    menu = menu.item(&mut core, text, move |_| picks.lock().push((m, i)));
                }
            }
            let mut h = Self {
                core,
                bar,
                picks,
                now: 0.0,
            };
            h.frame(AWAY, false, false);
            h
        }

        /// One frame in the renderer's order: pointer, dispatch, layout.
        /// Answers with what was picked on it.
        fn frame(
            &mut self,
            pos: [f32; 2],
            pressed: bool,
            released: bool,
        ) -> Option<(usize, usize)> {
            self.now += 1.0;
            self.core
                .update_pointer(pos, pressed, released, 0.0, self.now);
            self.core.dispatch(self.now);
            self.core.run_layout(SCREEN);
            self.picks.lock().pop()
        }

        fn click(&mut self, pos: [f32; 2]) -> Option<(usize, usize)> {
            self.frame(pos, true, false);
            self.frame(pos, false, true)
        }

        fn title(&mut self, i: usize) -> [f32; 2] {
            let t = self.bar.0.with(&mut self.core, |b, _| b.titles[i]).unwrap();
            centre(&self.core, t)
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

    /// A context menu somebody else opened is not the bar's to take down.
    #[test]
    fn a_menu_it_did_not_open_is_left_alone() {
        let mut h = Harness::new();
        let file = h.title(0);
        h.click(file);

        let theirs = h.core.context_menu([200.0, 200.0], MenuStyle::default());
        theirs.item(&mut h.core, "theirs", |_| {});
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
