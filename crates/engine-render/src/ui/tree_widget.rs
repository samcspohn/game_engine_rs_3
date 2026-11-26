//! [`Tree`]: a [`TreeView`] that drives itself. Its state lives on its node,
//! it re-binds on scroll and resize, and its rows call listeners with node
//! ids. See [ui-widgets](../../../../docs/notes/ui-widgets.md).

use std::any::Any;
use std::sync::Arc;

use super::list::{Row, RowContent, RowStyle};
use super::style::Style;
use super::tree_view::{DRAG_PX, DragNode, Dropped, TreeDrag, TreeView};
use super::{Event, Events, Keep, Label, NodeId, UiCore, UiRef};

type Listener<A> = Option<Arc<dyn Fn(&mut UiCore, A) + Send + Sync>>;

/// What a [`Tree`] shows: how to read a node's children, and how to fill a
/// row. `tag` is what [`TreeDrag::tree`] must say for a drag to be its own.
pub struct Source<H> {
    tag: u64,
    children: Box<dyn FnMut(u64, &mut Vec<u64>) + Send>,
    bind: Box<dyn FnMut(&mut UiCore, Row<'_, H>, u64) + Send>,
}

impl<H> Source<H> {
    pub fn new(
        children: impl FnMut(u64, &mut Vec<u64>) + Send + 'static,
        bind: impl FnMut(&mut UiCore, Row<'_, H>, u64) + Send + 'static,
    ) -> Self {
        Self {
            tag: 0,
            children: Box::new(children),
            bind: Box::new(bind),
        }
    }

    pub fn tagged(mut self, tag: u64) -> Self {
        self.tag = tag;
        self
    }
}

/// A drop of something other than the tree's own payload onto a row.
struct Accept {
    fits: Box<dyn Fn(&dyn Any) -> bool + Send>,
    land: Arc<dyn Fn(&mut UiCore, u64, &dyn Any) + Send + Sync>,
}

struct State<H: RowContent, P> {
    view: TreeView<H, P>,
    source: Source<H>,
    me: Option<Tree<H, P>>,
    selected: Option<u64>,
    editing: Option<u64>,
    /// Pooled rows with handlers on them; the pool only grows.
    wired: usize,
    on_select: Listener<Option<u64>>,
    on_click: Listener<u64>,
    on_double_click: Listener<u64>,
    on_right_click: Listener<(u64, [f32; 2])>,
    on_pick_up: Listener<u64>,
    on_move: Listener<Dropped>,
    accepts: Vec<Accept>,
}

/// Listeners run after the event that fired them, with no tree state on
/// loan — so one may call back into the tree.
fn fire<A: Send + 'static>(ui: &mut UiCore, l: &Listener<A>, a: A) {
    if let Some(f) = l.clone() {
        ui.later(move |ui| f(ui, a));
    }
}

impl<H: RowContent + Send + 'static, P: TreeDrag + Send> State<H, P> {
    fn sync(&mut self, ui: &mut UiCore) {
        let (sel, edit, style) = (self.selected, self.editing, self.view.style());
        let Source { children, bind, .. } = &mut self.source;
        self.view.sync(
            ui,
            |id, out| children(id, out),
            |ui, row, id| {
                let (node, content) = (row.node(), *row.content());
                bind(ui, row, id);
                Row::new(node, &content, style).set_selected(ui, sel == Some(id));
                content.set_editing(ui, edit == Some(id));
            },
        );
        self.track(ui);
        let pooled = self.view.pooled();
        if pooled.len() > self.wired {
            let me = self.me.expect("wired after `own`");
            for &(row, arrow) in &pooled[self.wired..] {
                me.wire(ui, row, arrow);
            }
            self.wired = pooled.len();
            // `on` made every new triangle a click target; binding again
            // takes that back from the leaves.
            self.sync(ui);
        }
    }

    fn track(&mut self, ui: &mut UiCore) {
        let accepts = &self.accepts;
        self.view.track(ui, |p| accepts.iter().any(|a| (a.fits)(p)));
    }

    fn select(&mut self, ui: &mut UiCore, id: Option<u64>) {
        if self.selected != id {
            self.selected = id;
            self.sync(ui);
            fire(ui, &self.on_select, id);
        }
    }

    fn row_event(&mut self, ui: &mut UiCore, row: NodeId, ev: &Event) {
        let Some(id) = self.view.id_at(row) else {
            return;
        };
        match ev {
            Event::Click => {
                self.select(ui, Some(id));
                fire(ui, &self.on_click, id);
            }
            Event::DoubleClick => fire(ui, &self.on_double_click, id),
            Event::RightClick => {
                self.select(ui, Some(id));
                fire(ui, &self.on_right_click, (id, ui.pointer.pos));
            }
            // The press's own row hears every frame of it; the rows under a
            // grab in flight hear it too, which is what moves the marker.
            Event::Drag(d) if ui.ghost().is_none() => {
                if ui.drag(row).is_some() && d.beyond(DRAG_PX) {
                    fire(ui, &self.on_pick_up, id);
                }
            }
            Event::Drag(_) | Event::HoverEnter | Event::HoverLeave => self.track(ui),
            Event::DroppedOn => self.landed(ui, row, id),
            _ => {}
        }
    }

    fn landed(&mut self, ui: &mut UiCore, row: NodeId, id: u64) {
        self.track(ui);
        if let Some(d) = self.view.dropped() {
            self.view.moved(d.node, d.parent, d.at);
            self.sync(ui);
            return fire(ui, &self.on_move, d);
        }
        let Some((_, payload)) = ui.pointer.drop.as_ref().filter(|(n, _)| *n == row) else {
            return;
        };
        let Some(a) = self.accepts.iter().find(|a| (a.fits)(&**payload)) else {
            return;
        };
        let land = a.land.clone();
        ui.later(move |ui| {
            let Some((n, payload)) = ui.pointer.drop.take() else {
                return;
            };
            land(ui, id, &*payload);
            ui.pointer.drop.get_or_insert((n, payload));
        });
    }
}

/// A collapsible tree whose rows say what happened to them by node id.
/// `Copy`; its state lives on [`node`](Self::node) and goes with it.
///
/// Selection is the tree's: a click or a right-click selects, and the
/// selected row is drawn so — `bind` fills only what a row says.
pub struct Tree<H: RowContent = Label, P = DragNode>(UiRef<State<H, P>>);

impl<H: RowContent, P> Clone for Tree<H, P> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<H: RowContent, P> Copy for Tree<H, P> {}

impl<H: RowContent, P> std::fmt::Debug for Tree<H, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Tree({:?})", self.0)
    }
}

impl<H: RowContent + Send + 'static, P: TreeDrag + Send> Tree<H, P> {
    /// `root` is shown as a row like any other, so there is somewhere to drop
    /// a node to un-parent it.
    pub fn new(
        ui: &mut UiCore,
        parent: NodeId,
        viewport: Style,
        style: RowStyle,
        root: u64,
        source: Source<H>,
    ) -> Self {
        let view = TreeView::new(ui, parent, viewport, style, root).with_tree(source.tag);
        let area = view.node();
        let state = State {
            view,
            source,
            me: None,
            selected: None,
            editing: None,
            wired: 0,
            on_select: None,
            on_click: None,
            on_double_click: None,
            on_right_click: None,
            on_pick_up: None,
            on_move: None,
            accepts: Vec::new(),
        };
        let me = Self(ui.own(area, state));
        me.0.with(ui, |s, _| s.me = Some(me));
        // The pool sizes itself from the area, so the first layout is also
        // the first bind.
        ui.on(area, Events::SCROLL | Events::RESIZE, move |ui, _| {
            me.0.with(ui, |s, ui| s.sync(ui));
            Keep::Yes
        });
        me
    }

    /// The scroll area, for styling it.
    pub fn node(self) -> NodeId {
        self.0.node()
    }

    fn wire(self, ui: &mut UiCore, row: NodeId, arrow: NodeId) {
        let kinds = Events::CLICK
            | Events::DOUBLE_CLICK
            | Events::RIGHT_CLICK
            | Events::DRAG
            | Events::DROP
            | Events::HOVER;
        ui.on(row, kinds, move |ui, ev| {
            self.0.with(ui, |s, ui| s.row_event(ui, row, ev));
            Keep::Yes
        });
        ui.on_click(arrow, move |ui| {
            self.0.with(ui, |s, ui| {
                let Some(id) = s.view.id_at(arrow) else {
                    return;
                };
                let open = s.view.is_expanded(id);
                let children = &mut s.source.children;
                s.view
                    .set_expanded(id, !open, &mut |id, out| children(id, out));
                s.sync(ui);
            });
        });
    }

    /// Show another model, from its root. Clears the selection.
    pub fn show(self, ui: &mut UiCore, source: Source<H>) {
        self.0.with(ui, |s, ui| {
            s.view.set_tree(source.tag);
            s.source = source;
            s.editing = None;
            s.view.invalidate();
            s.select(ui, None);
            s.sync(ui);
        });
    }

    /// The structure changed outside the tree — re-walk it now.
    pub fn invalidate(self, ui: &mut UiCore) {
        self.0.with(ui, |s, ui| {
            s.view.invalidate();
            s.sync(ui);
        });
    }

    pub fn selected(self, ui: &mut UiCore) -> Option<u64> {
        self.0.with(ui, |s, _| s.selected).flatten()
    }

    pub fn select(self, ui: &mut UiCore, id: Option<u64>) {
        self.0.with(ui, |s, ui| s.select(ui, id));
    }

    /// Expand every ancestor of `id`, so it has a row.
    pub fn reveal(self, ui: &mut UiCore, id: u64, parent_of: impl Fn(u64) -> Option<u64>) {
        self.0.with(ui, |s, ui| {
            s.view.reveal(id, parent_of);
            s.sync(ui);
        });
    }

    pub fn is_expanded(self, ui: &mut UiCore, id: u64) -> bool {
        self.0.with(ui, |s, _| s.view.is_expanded(id)) == Some(true)
    }

    pub fn set_expanded(self, ui: &mut UiCore, id: u64, want: bool) {
        self.0.with(ui, |s, ui| {
            let children = &mut s.source.children;
            s.view
                .set_expanded(id, want, &mut |id, out| children(id, out));
            s.sync(ui);
        });
    }

    /// Put `id`'s row in edit mode — [`RowContent::set_editing`] — and hand
    /// back its content to focus. `None` ends an edit, as does a row that is
    /// not pooled.
    pub fn edit(self, ui: &mut UiCore, id: Option<u64>) -> Option<H> {
        self.0
            .with(ui, |s, ui| {
                let row = id.and_then(|id| s.view.row(id)).map(|r| *r.content());
                s.editing = row.and(id);
                s.sync(ui);
                row
            })
            .flatten()
    }

    pub fn editing(self, ui: &mut UiCore) -> Option<u64> {
        self.0.with(ui, |s, _| s.editing).flatten()
    }

    /// The content of the row showing `id`, if one is.
    pub fn row(self, ui: &mut UiCore, id: u64) -> Option<H> {
        self.0
            .with(ui, |s, _| s.view.row(id).map(|r| *r.content()))
            .flatten()
    }

    /// Pick up a row: see [`TreeView::grab`]. For [`on_pick_up`](Self::on_pick_up).
    pub fn grab(self, ui: &mut UiCore, payload: P, bind: impl FnOnce(&mut UiCore, Row<'_, H>)) {
        self.0.with(ui, |s, ui| s.view.grab(ui, payload, bind));
    }

    pub fn on_select(
        self,
        ui: &mut UiCore,
        f: impl Fn(&mut UiCore, Option<u64>) + Send + Sync + 'static,
    ) {
        self.0.with(ui, |s, _| s.on_select = Some(Arc::new(f)));
    }

    pub fn on_row_click(
        self,
        ui: &mut UiCore,
        f: impl Fn(&mut UiCore, u64) + Send + Sync + 'static,
    ) {
        self.0.with(ui, |s, _| s.on_click = Some(Arc::new(f)));
    }

    pub fn on_row_double_click(
        self,
        ui: &mut UiCore,
        f: impl Fn(&mut UiCore, u64) + Send + Sync + 'static,
    ) {
        self.0
            .with(ui, |s, _| s.on_double_click = Some(Arc::new(f)));
    }

    /// With the pointer, in window px.
    pub fn on_row_right_click(
        self,
        ui: &mut UiCore,
        f: impl Fn(&mut UiCore, u64, [f32; 2]) + Send + Sync + 'static,
    ) {
        let f = move |ui: &mut UiCore, (id, at): (u64, [f32; 2])| f(ui, id, at);
        self.0.with(ui, |s, _| s.on_right_click = Some(Arc::new(f)));
    }

    /// A press on a row has travelled far enough to be a drag. Answer with
    /// [`grab`](Self::grab), or not — it is offered again while it moves.
    pub fn on_pick_up(self, ui: &mut UiCore, f: impl Fn(&mut UiCore, u64) + Send + Sync + 'static) {
        self.0.with(ui, |s, _| s.on_pick_up = Some(Arc::new(f)));
    }

    /// A row of this tree was dropped somewhere in it. The tree has moved its
    /// own rows already; `f` moves the model.
    pub fn on_move(
        self,
        ui: &mut UiCore,
        f: impl Fn(&mut UiCore, Dropped) + Send + Sync + 'static,
    ) {
        self.0.with(ui, |s, _| s.on_move = Some(Arc::new(f)));
    }

    /// A `T` that `fits` lights the row it is over, and `f` hears the row it
    /// lands on.
    pub fn on_row_drop<T: Any>(
        self,
        ui: &mut UiCore,
        fits: impl Fn(&T) -> bool + Send + 'static,
        f: impl Fn(&mut UiCore, u64, &T) + Send + Sync + 'static,
    ) {
        let accept = Accept {
            fits: Box::new(move |p| p.downcast_ref().is_some_and(&fits)),
            land: Arc::new(move |ui, id, p| {
                if let Some(t) = p.downcast_ref() {
                    f(ui, id, t);
                }
            }),
        };
        self.0.with(ui, |s, _| s.accepts.push(accept));
    }
}

#[cfg(test)]
mod tests {
    use super::super::style::{LengthPercentageAuto, Position, Rect, Size, TaffyAuto, px};
    use super::*;
    use parking_lot::Mutex;
    use std::collections::HashMap;

    /// `roots` top-level nodes, each with three children.
    fn model(roots: u64) -> HashMap<u64, Vec<u64>> {
        let mut m = HashMap::from([(0, (1..=roots).collect::<Vec<_>>())]);
        for r in 1..=roots {
            m.insert(r, (1..=3).map(|i| r * 100 + i).collect());
        }
        m
    }

    type Heard = Arc<Mutex<Vec<String>>>;

    /// A 200x100 tree of 20 px rows at the origin, reporting into `heard`.
    fn tree(core: &mut UiCore, roots: u64, heard: &Heard) -> Tree {
        let m = model(roots);
        let source = Source::new(
            move |id, out| out.extend(m.get(&id).into_iter().flatten().copied()),
            |ui, r: Row<'_, Label>, id| r.set_text(ui, &format!("n{id}")),
        );
        let root = core.root();
        let viewport = Style {
            position: Position::Absolute,
            inset: Rect {
                left: px(0.0),
                top: px(0.0),
                right: LengthPercentageAuto::AUTO,
                bottom: LengthPercentageAuto::AUTO,
            },
            size: Size {
                width: px(200.0),
                height: px(100.0),
            },
            ..Default::default()
        };
        let style = RowStyle {
            row_h: 20.0,
            ..Default::default()
        };
        let t = Tree::new(core, root, viewport, style, 0, source);
        let h = heard.clone();
        t.on_row_click(core, move |_, id| h.lock().push(format!("click {id}")));
        let h = heard.clone();
        t.on_select(core, move |_, id| h.lock().push(format!("select {id:?}")));
        let h = heard.clone();
        t.on_move(core, move |_, d| {
            h.lock()
                .push(format!("move {} {} {}", d.node, d.parent, d.at))
        });
        t.on_pick_up(core, move |ui, id| t.grab(ui, DragNode(id), |_, _| {}));
        for _ in 0..3 {
            frame(core, [300.0, 300.0], false, false);
        }
        t
    }

    fn frame(core: &mut UiCore, at: [f32; 2], pressed: bool, released: bool) {
        core.update_pointer(at, pressed, released, 0.0, 0.0);
        core.dispatch(0.0);
        core.run_layout([400.0, 400.0]);
    }

    fn shown(core: &UiCore) -> Vec<String> {
        core.text_nodes()
            .into_iter()
            .map(|(_, t, _)| t)
            .filter(|t| t.starts_with('n'))
            .collect()
    }

    #[test]
    fn a_click_selects_and_names_the_node() {
        let (mut core, heard) = (UiCore::new(), Heard::default());
        let t = tree(&mut core, 3, &heard);
        frame(&mut core, [150.0, 50.0], true, false);
        frame(&mut core, [150.0, 50.0], false, true);
        assert_eq!(*heard.lock(), ["select Some(2)", "click 2"]);
        assert_eq!(t.selected(&mut core), Some(2));
    }

    #[test]
    fn the_triangle_expands_without_selecting() {
        let (mut core, heard) = (UiCore::new(), Heard::default());
        let t = tree(&mut core, 3, &heard);
        frame(&mut core, [22.0, 50.0], true, false);
        frame(&mut core, [22.0, 50.0], false, true);
        assert!(t.is_expanded(&mut core, 2));
        assert!(
            shown(&core).contains(&"n201".to_string()),
            "{:?}",
            shown(&core)
        );
        assert!(
            heard.lock().is_empty(),
            "no click on the row: {:?}",
            heard.lock()
        );
    }

    #[test]
    fn scrolling_rebinds_the_pool() {
        let (mut core, heard) = (UiCore::new(), Heard::default());
        let t = tree(&mut core, 20, &heard);
        assert!(!shown(&core).contains(&"n12".to_string()));
        core.scroll_by(t.node(), [0.0, 200.0]);
        frame(&mut core, [300.0, 300.0], false, false);
        assert!(
            shown(&core).contains(&"n12".to_string()),
            "{:?}",
            shown(&core)
        );
    }

    #[test]
    fn a_row_dragged_onto_another_moves_under_it() {
        let (mut core, heard) = (UiCore::new(), Heard::default());
        tree(&mut core, 3, &heard);
        frame(&mut core, [150.0, 30.0], true, false);
        frame(&mut core, [150.0, 70.0], false, false);
        assert!(core.ghost().is_some(), "picked up");
        frame(&mut core, [150.0, 70.0], false, true);
        assert_eq!(heard.lock().last().map(String::as_str), Some("move 1 3 0"));
    }

    #[test]
    fn a_drop_of_another_kind_lands_on_its_row() {
        let (mut core, heard) = (UiCore::new(), Heard::default());
        let t = tree(&mut core, 3, &heard);
        let h = heard.clone();
        t.on_row_drop::<String>(
            &mut core,
            |s| s == "scene",
            move |_, id, s| h.lock().push(format!("{s} on {id}")),
        );
        frame(&mut core, [300.0, 300.0], true, false);
        core.grab("scene".to_string());
        frame(&mut core, [150.0, 50.0], false, false);
        frame(&mut core, [150.0, 50.0], false, true);
        assert_eq!(heard.lock().last().map(String::as_str), Some("scene on 2"));
    }
}
