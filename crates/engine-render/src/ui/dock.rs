//! Docking — the layout the *user* arranges.
//!
//! A [`DockSpace`] is a binary tree of cells. A leaf is a tab strip over a
//! stack of panes, one open at a time; a split is two cells side by side or
//! stacked. Dragging a tab header onto another leaf's edge splits it, onto
//! its middle joins its strip, and emptying a leaf folds its split away.
//!
//! # What it forced
//!
//! [`UiCore::set_parent`]. Every widget before this one was built where it
//! lives and stayed there, so a panel could only "move" by being rebuilt —
//! which throws away the values its controls own, the text half-typed into
//! its fields and the offset its lists are scrolled to. Docking is the case
//! where that is plainly wrong: the panel the user dragged is the *same*
//! panel. So the subtree moves, keeping every slot it had, and the only
//! thing that changes is where taffy puts it.
//!
//! # In place, not around
//!
//! Splitting a leaf does not wrap it in a new parent. The leaf's own node
//! *becomes* the split and its contents move down into a fresh child, so the
//! cell keeps its index among its siblings and nothing above it moves.
//! Collapsing is the same trick backwards: the surviving sibling's contents
//! move up into the split's node. Wrapping would have needed insert-at-index
//! re-parenting — and would have re-ordered any split whose left half was
//! the one being worked on.
//!
//! # Why the panes have no group of their own
//!
//! A [`UiGroup`](super::UiGroup) buys one record per *move*, which is what a
//! floating window wants. A docked panel does not move on its own — taffy
//! places it, and a drop is a relayout either way — so a group would only
//! buy the clip, at the cost of making every scroll area inside a panel a
//! nested one. Groups earn their offset here the day a panel floats.

use super::style::{
    px, zero, Display, FlexDirection, LengthPercentageAuto, Position, Rect, Size, Style, TaffyAuto,
};
use super::{theme, Events, Label, NodeId, StateStyle, TabStyle, Theme, UiCore, UiStyle};
use engine_core::serde::{Deserialize, Serialize};

/// How far a header must travel before a press reads as a lift rather than a
/// click on the tab.
const DRAG_PX: f32 = 4.0;

/// Shortest a pane may be dragged to. Below this its headers stop being
/// readable, and a pane you cannot read is one you cannot get back.
const MIN_PANE: f32 = 48.0;

/// The band along each edge that splits rather than joins, as a fraction of
/// the leaf. The middle — over half the box — is the tab join, because that
/// is the drop a user aims at without thinking.
const EDGE: f32 = 0.25;

/// Where a dropped panel lands relative to the one it was aimed at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Left,
    Right,
    Top,
    Bottom,
    /// Into the target's own tab strip, taking no space of its own.
    Tab,
}

impl Side {
    /// The part of the target this takes, as `[x, y, w, h]` fractions — what
    /// the aiming highlight draws.
    fn frac(self) -> [f32; 4] {
        match self {
            Side::Left => [0.0, 0.0, 0.5, 1.0],
            Side::Right => [0.5, 0.0, 0.5, 1.0],
            Side::Top => [0.0, 0.0, 1.0, 0.5],
            Side::Bottom => [0.0, 0.5, 1.0, 0.5],
            Side::Tab => [0.0, 0.0, 1.0, 1.0],
        }
    }

    fn across(self) -> bool {
        matches!(self, Side::Left | Side::Right)
    }

    /// Whether the arriving pane is the split's *first* child.
    fn leading(self) -> bool {
        matches!(self, Side::Left | Side::Top)
    }
}

/// A panel's identity. Stable for the life of the dock — it survives every
/// move, which is the point: a caller holds one of these and never learns
/// where the user has since put it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PanelId(u32);

/// A lifted panel, on the pointer. Public so a target outside the dock can
/// accept one, the same way [`DragNode`](super::DragNode) is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DragPanel(pub PanelId);

/// A dock's arrangement, by panel title — a [`PanelId`] is an index into one
/// run of the program, and a saved layout outlives the run that wrote it.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", crate = "engine_core::serde")]
pub enum Layout {
    Leaf {
        tabs: Vec<String>,
        open: usize,
    },
    Split {
        /// Side by side rather than stacked.
        across: bool,
        /// The first kid's share of the split.
        share: f32,
        kids: Box<[Layout; 2]>,
    },
}

/// A [`Layout`] matched against the panels that actually exist. A leaf that
/// matched none of its titles is dropped rather than left empty.
enum Plan {
    Leaf(Vec<PanelId>, usize),
    Split(bool, f32, Box<[Plan; 2]>),
}

impl Plan {
    /// The panel that stays put while this subtree is built around it.
    fn head(&self) -> PanelId {
        match self {
            Plan::Leaf(ids, _) => ids[0],
            Plan::Split(.., kids) => kids[0].head(),
        }
    }

    fn flatten(&self, out: &mut Vec<PanelId>) {
        match self {
            Plan::Leaf(ids, _) => out.extend(ids),
            Plan::Split(.., kids) => kids.iter().for_each(|k| k.flatten(out)),
        }
    }
}

/// Splitting and collapsing both work in place, so entry 0 is the whole dock
/// for its whole life.
const ROOT_CELL: usize = 0;

#[derive(Clone)]
struct Panel {
    title: String,
    header: NodeId,
    label: Label,
    content: NodeId,
    /// Which leaf it is currently filed under.
    cell: usize,
}

/// One box of the dock tree. Splitting and collapsing move a cell's whole
/// child run either way, so neither needs to know which kind it is holding.
#[derive(Clone)]
struct Cell {
    node: NodeId,
    parent: Option<usize>,
    kind: Kind,
}

#[derive(Clone)]
enum Kind {
    /// A strip of headers over a stack of panes, `open` of which is visible.
    Leaf {
        strip: NodeId,
        body: NodeId,
        panels: Vec<PanelId>,
        open: usize,
    },
    /// Two cells and the line between them, which is also what resizes them.
    Split { kids: [usize; 2], divider: NodeId },
}

/// How a [`DockSpace`] looks.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DockStyle {
    /// The headers. A leaf's strip *is* a tab strip, so it shares
    /// [`TabStyle`] rather than restating colours that would then drift.
    pub tab: TabStyle,
    /// Thickness of the line between the two halves of a split — and the
    /// width of the target you grab to move it.
    pub divider: f32,
    /// The line at rest, under the pointer, and while being dragged. At rest
    /// it is the hairline role, because that is what it is; under the pointer
    /// it takes the accent, because that is the promise that it moves.
    pub line: u32,
    pub line_hover: u32,
    pub line_held: u32,
    /// Behind a panel's content.
    pub surface: u32,
    /// The part of the target a drop would take, painted while aiming.
    pub zone: u32,
    pub zone_edge: u32,
    pub radius: f32,
}

impl From<Theme> for DockStyle {
    fn from(t: Theme) -> Self {
        Self {
            tab: t.into(),
            divider: 2.0,
            line: t.outline,
            line_hover: t.accent,
            line_held: t.accent,
            surface: t.backdrop,
            zone: alpha(t.accent, 0x38),
            zone_edge: t.accent,
            radius: t.radius,
        }
    }
}

impl Default for DockStyle {
    fn default() -> Self {
        theme().into()
    }
}

/// A tree of dockable panels. Caller-owned like a
/// [`TreeView`](super::TreeView): build it, add panels, and call
/// [`update`](DockSpace::update) once a frame.
///
/// ```ignore
/// let mut dock = DockSpace::new(&mut ui, root, fill(), DockStyle::default());
/// let scene = dock.panel(&mut ui, "Scene");
/// let props = dock.panel(&mut ui, "Inspector");
/// dock.dock(&mut ui, props, scene, Side::Right);
/// ui.label(dock.content(scene), 11.0, text, "…");   // an ordinary node
/// ```
#[derive(Clone)]
pub struct DockSpace {
    root: NodeId,
    /// The aiming highlight. Last child of the root, so it paints over every
    /// panel without a `raise` — nothing is ever added beside it.
    overlay: NodeId,
    style: DockStyle,
    cells: Vec<Option<Cell>>,
    panels: Vec<Panel>,
    /// What the in-flight drag is currently over, so the highlight is written
    /// on a transition rather than every frame of the gesture.
    aiming: Option<(usize, Side)>,
    /// Whether a gesture has moved something since [`DockSpace::changed`]
    /// was last asked.
    dirty: bool,
}

impl DockSpace {
    /// `style` is the outer box — give it a size or let it grow. Starts as
    /// one empty leaf filling it.
    pub fn new(ui: &mut UiCore, parent: impl Into<NodeId>, style: Style, dock: DockStyle) -> Self {
        let root = ui.node(
            parent,
            Style {
                display: Display::Flex,
                // The dock takes its size from the box it was handed, never
                // from what a panel happens to hold. Without this a single
                // long readout inside one pane widens the whole dock past
                // its parent, and every split with it.
                min_size: Size {
                    width: px(0.0),
                    height: px(0.0),
                },
                ..style
            },
        );
        let node = ui.node(root, leaf_style(dock.tab.gap));
        let kind = build_leaf(ui, node, &dock);

        let overlay = ui.node(
            root,
            Style {
                position: Position::Absolute,
                ..Default::default()
            },
        );
        ui.set_background(
            overlay,
            UiStyle::fill(dock.zone)
                .border(dock.zone_edge, 1.0)
                .radius(dock.radius),
        );
        ui.set_visible(overlay, false);

        Self {
            root,
            overlay,
            style: dock,
            cells: vec![Some(Cell {
                node,
                parent: None,
                kind,
            })],
            panels: Vec::new(),
            aiming: None,
            dirty: false,
        }
    }

    /// The outer node, for a background or a border.
    pub fn node(&self) -> NodeId {
        self.root
    }

    /// Add a panel to the first leaf and open it. Build into
    /// [`content`](Self::content) like any node; move it with
    /// [`dock`](Self::dock) or let the user drag it.
    pub fn panel(&mut self, ui: &mut UiCore, title: &str) -> PanelId {
        let cell = self.first_leaf();
        let Some(Kind::Leaf { strip, body, .. }) = self.kind(cell) else {
            unreachable!("first_leaf returned a split")
        };
        let (strip, body) = (*strip, *body);

        let header = ui.node(strip, header_style(&self.style));
        let label = ui.label(
            header,
            self.style.tab.text_px,
            self.style.tab.text_dim,
            title,
        );
        ui.set_events(header, Events::CLICK | Events::HOVER);
        let content = ui.node(body, pane_style(&self.style));

        let id = PanelId(self.panels.len() as u32);
        self.panels.push(Panel {
            title: title.into(),
            header,
            label,
            content,
            cell,
        });
        self.file(ui, id, cell);
        id
    }

    /// The container for a panel's content. Valid for the panel's whole life,
    /// whichever leaf it has been dragged into and whether or not it is the
    /// open tab.
    pub fn content(&self, p: PanelId) -> NodeId {
        self.panels[p.0 as usize].content
    }

    pub fn title(&self, p: PanelId) -> &str {
        &self.panels[p.0 as usize].title
    }

    /// Whether this panel is the open tab of its leaf. A closed one has no
    /// box at all, so a caller with expensive contents can skip them.
    pub fn showing(&self, p: PanelId) -> bool {
        let (cell, at) = self.locate(p);
        self.open_of(cell) == at
    }

    /// Bring a panel to the front of its strip, as clicking its header would.
    pub fn select(&mut self, ui: &mut UiCore, p: PanelId) {
        let (cell, at) = self.locate(p);
        self.open(ui, cell, at);
    }

    /// Give this panel's leaf `share` of the split it sits in, as dragging
    /// that split's divider would. A dock splits evenly, and a starting
    /// layout usually is not even — a viewport is not half a hierarchy.
    ///
    /// Writes the same two `flex_grow` numbers the drag does, so there is
    /// still no ratio stored anywhere but the layout. Silently does nothing
    /// for a panel in an unsplit dock: there is no boundary to place.
    pub fn set_ratio(&mut self, ui: &mut UiCore, p: PanelId, share: f32) {
        let cell = self.panels[p.0 as usize].cell;
        let Some(parent) = self.cells[cell].as_ref().and_then(|c| c.parent) else {
            return;
        };
        let Some(Kind::Split { kids, .. }) = self.kind(parent) else {
            unreachable!("a cell's parent is always a split")
        };
        let share = share.clamp(0.05, 0.95);
        for k in *kids {
            let n = self.node_of(k);
            let mut s = ui.node_style(n);
            s.flex_grow = match k == cell {
                true => share,
                false => 1.0 - share,
            };
            ui.set_node_style(n, s);
        }
    }

    /// This dock's arrangement, to write somewhere and hand back to
    /// [`apply`](Self::apply) next run.
    pub fn layout(&self, ui: &UiCore) -> Layout {
        self.layout_of(ui, ROOT_CELL)
    }

    /// Rearrange into a saved [`Layout`], with the same calls a run of drops
    /// makes. A title with no panel is skipped and a panel the layout does
    /// not name becomes an extra tab of the first leaf: a layout from an
    /// older session is a starting point, not a contract.
    pub fn apply(&mut self, ui: &mut UiCore, layout: &Layout) {
        let mut spare: Vec<PanelId> = (0..self.panels.len() as u32).map(PanelId).collect();
        let Some(plan) = self.resolve(layout, &mut spare) else {
            return;
        };
        let mut order = Vec::new();
        plan.flatten(&mut order);
        // Everything into one leaf first, so every later step is a split of
        // a cell that already holds exactly the panels it is dividing.
        for &p in order.iter().skip(1).chain(spare.iter()) {
            let home = self.panels[order[0].0 as usize].cell;
            self.move_to(ui, p, home, Side::Tab);
        }
        self.build(ui, &plan);
    }

    /// Whether a gesture has changed the layout since this was last asked —
    /// what a caller saves on. Asking clears it.
    pub fn changed(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    /// Move `panel` beside `target`, splitting the pane `target` is in — or
    /// into its strip with [`Side::Tab`]. Exactly what a drop does.
    pub fn dock(&mut self, ui: &mut UiCore, panel: PanelId, target: PanelId, side: Side) {
        let cell = self.panels[target.0 as usize].cell;
        self.move_to(ui, panel, cell, side);
    }

    /// Fold this frame's pointer into the layout: a header click opens its
    /// tab, a header drag lifts the panel and lights up where it would land,
    /// the release moves it, and a divider drag resizes the split it is in.
    ///
    /// The one call an application owes the dock. Nothing here runs per
    /// panel per frame beyond two pointer lookups — the layout only changes
    /// on the frame a gesture ends.
    pub fn update(&mut self, ui: &mut UiCore) {
        for i in 0..self.panels.len() {
            let (id, header) = (PanelId(i as u32), self.panels[i].header);

            if ui.clicked(header) {
                self.select(ui, id);
                self.dirty = true;
            }
            // The release is read before the drag, because a frame that ends
            // the gesture reports both and the move belongs where the pointer
            // actually let go.
            if let Some(d) = ui.dropped(header) {
                self.aim_at(ui, None);
                if d.beyond(DRAG_PX) {
                    if let Some((cell, side)) = self.aim(ui, d.pos) {
                        self.move_to(ui, id, cell, side);
                        self.dirty = true;
                    }
                }
                continue;
            }
            let Some(d) = ui.drag(header).filter(|d| d.beyond(DRAG_PX)) else {
                continue;
            };
            if ui.ghost().is_none() {
                self.lift(ui, id);
            }
            let aim = self.aim(ui, d.pos);
            self.aim_at(ui, aim);
        }

        // The other gesture a dock has. Kept out of the loop above because a
        // divider belongs to a split and a header to a panel, and neither
        // can be the node the other's press landed on.
        for c in 0..self.cells.len() {
            let Some(Kind::Split { kids, divider }) = self.kind(c) else {
                continue;
            };
            let (kids, divider) = (*kids, *divider);
            if let Some(d) = ui.drag(divider) {
                self.resize(ui, c, kids, d.pos);
            }
            // On the release, not every frame of the drag: a caller that
            // saves on this should write one file per gesture.
            self.dirty |= ui.dropped(divider).is_some();
        }
    }

    fn layout_of(&self, ui: &UiCore, c: usize) -> Layout {
        match self.kind(c) {
            Some(Kind::Split { kids, .. }) => {
                let grow = |k: usize| ui.node_style(self.node_of(k)).flex_grow;
                let (a, b) = (grow(kids[0]), grow(kids[1]));
                Layout::Split {
                    across: ui.node_style(self.node_of(c)).flex_direction == FlexDirection::Row,
                    share: a / (a + b).max(f32::EPSILON),
                    kids: Box::new([self.layout_of(ui, kids[0]), self.layout_of(ui, kids[1])]),
                }
            }
            Some(Kind::Leaf { panels, open, .. }) => Layout::Leaf {
                tabs: panels.iter().map(|&p| self.title(p).to_string()).collect(),
                open: *open,
            },
            None => Layout::Leaf {
                tabs: Vec::new(),
                open: 0,
            },
        }
    }

    /// Claim the panels a layout names out of `spare`, dropping the branches
    /// that claimed nothing — a split with one live half *is* that half.
    fn resolve(&self, layout: &Layout, spare: &mut Vec<PanelId>) -> Option<Plan> {
        let (across, share, kids) = match layout {
            Layout::Split {
                across,
                share,
                kids,
            } => (*across, *share, kids),
            Layout::Leaf { tabs, open } => {
                let (mut ids, mut shown) = (Vec::new(), 0);
                for (i, t) in tabs.iter().enumerate() {
                    let Some(at) = spare.iter().position(|&p| self.title(p) == t) else {
                        continue;
                    };
                    if i == *open {
                        shown = ids.len();
                    }
                    ids.push(spare.remove(at));
                }
                return (!ids.is_empty()).then_some(Plan::Leaf(ids, shown));
            }
        };
        let first = self.resolve(&kids[0], spare);
        let second = self.resolve(&kids[1], spare);
        match (first, second) {
            (Some(a), Some(b)) => Some(Plan::Split(across, share, Box::new([a, b]))),
            (a, b) => a.or(b),
        }
    }

    /// Divide the cell holding all of `plan`'s panels into the shape it
    /// describes: split the second half off, then do the same to each half.
    fn build(&mut self, ui: &mut UiCore, plan: &Plan) {
        let (across, share, kids) = match plan {
            Plan::Split(across, share, kids) => (*across, *share, kids),
            Plan::Leaf(ids, open) => return self.select(ui, ids[*open]),
        };
        let anchor = kids[0].head();
        let mut moved = Vec::new();
        kids[1].flatten(&mut moved);
        let side = match across {
            true => Side::Right,
            false => Side::Bottom,
        };
        let cell = self.panels[anchor.0 as usize].cell;
        self.move_to(ui, moved[0], cell, side);
        for &p in &moved[1..] {
            let into = self.panels[moved[0].0 as usize].cell;
            self.move_to(ui, p, into, Side::Tab);
        }
        self.set_ratio(ui, anchor, share);
        self.build(ui, &kids[0]);
        self.build(ui, &kids[1]);
    }

    /// Move a split's boundary to the pointer.
    ///
    /// Absolute, like a slider's track: the line goes where the cursor is
    /// rather than integrating a delta, so it cannot drift away from the
    /// hand. The two halves keep a `flex_basis` of zero, so their `flex_grow`
    /// *is* the proportion — there is no ratio stored anywhere to fall out of
    /// step with the layout, and a window resize keeps the split.
    fn resize(&mut self, ui: &mut UiCore, cell: usize, kids: [usize; 2], p: [f32; 2]) {
        let node = self.node_of(cell);
        let across = ui.node_style(node).flex_direction == FlexDirection::Row;
        let i = usize::from(!across);
        let r = ui.node_rect(node);
        let free = r[i + 2] - self.style.divider;
        if free <= 0.0 {
            return;
        }
        // Half a divider back, so the line's centre follows the pointer
        // rather than its leading edge.
        let min = (MIN_PANE / free).min(0.45);
        let t = ((p[i] - r[i] - self.style.divider * 0.5) / free).clamp(min, 1.0 - min);

        for (k, grow) in kids.into_iter().zip([t, 1.0 - t]) {
            let n = self.node_of(k);
            let mut s = ui.node_style(n);
            s.flex_grow = grow;
            ui.set_node_style(n, s);
        }
    }

    // ── The dock tree ───────────────────────────────────────────────────

    /// Turn a leaf into a split **in place** and return the empty leaf added
    /// beside it. The cell keeps its node, so it keeps its place among its
    /// own siblings and nothing above it is touched.
    fn split(&mut self, ui: &mut UiCore, cell: usize, side: Side) -> usize {
        let host = self.node_of(cell);
        let kept = self.children_of(cell);
        let held = self.kind(cell).expect("live cell").clone();

        // `a` inherits what the cell was, which is right whether it was a
        // leaf or already a split — but not its *share*: that belongs to the
        // cell, which is still the one its own parent is dividing space with.
        let mut style = ui.node_style(host);
        let share = std::mem::replace(&mut style.flex_grow, 1.0);
        let a = ui.node(host, style);
        for child in kept {
            ui.set_parent(child, a);
        }
        let dir = match side.across() {
            true => FlexDirection::Row,
            false => FlexDirection::Column,
        };
        ui.set_node_style(
            host,
            Style {
                flex_grow: share,
                ..cell_style(dir)
            },
        );
        let divider = self.divider(ui, host, side.across());
        let b = ui.node(host, leaf_style(self.style.tab.gap));
        let leaf = build_leaf(ui, b, &self.style);
        if side.leading() {
            // Order is the layout, so putting `b` first is two moves to the
            // end rather than an insert the node API deliberately lacks.
            ui.raise(divider);
            ui.raise(a);
        }

        let ia = self.new_cell(Cell {
            node: a,
            parent: Some(cell),
            kind: held,
        });
        let ib = self.new_cell(Cell {
            node: b,
            parent: Some(cell),
            kind: leaf,
        });
        self.rehome(ia);
        let kids = match side.leading() {
            true => [ib, ia],
            false => [ia, ib],
        };
        self.cells[cell].as_mut().expect("live cell").kind = Kind::Split { kids, divider };
        // The surface goes with the leaf, not with the node it used to be:
        // `host` is a split now and paints nothing, and `a` is where the
        // strip and the pane actually live.
        self.repaint(ui, cell);
        self.repaint(ui, ia);
        ib
    }

    /// Fold an emptied leaf away, and its split with it: the surviving
    /// sibling's contents move *up* into the split's own node, so the layout
    /// keeps its place and the two spare nodes go.
    ///
    /// The last leaf has no split to fold, and an empty dock is a valid one.
    fn collapse(&mut self, ui: &mut UiCore, cell: usize) {
        let Some(parent) = self.cells[cell].as_ref().and_then(|c| c.parent) else {
            return;
        };
        let Some(Kind::Split { kids, divider }) = self.kind(parent) else {
            unreachable!("a cell's parent is always a split")
        };
        let (kids, divider) = (*kids, *divider);
        let sib = match kids[0] == cell {
            true => kids[1],
            false => kids[0],
        };
        let (host, gone, survivor) = (self.node_of(parent), self.node_of(cell), self.node_of(sib));

        // The line goes with the split it was separating.
        ui.remove_node(divider);
        // The survivor's own share was its half of *this* split; the space
        // the host is given is still the host's argument with its parent.
        let mut style = ui.node_style(survivor);
        style.flex_grow = ui.node_style(host).flex_grow;
        ui.set_node_style(host, style);
        for child in self.children_of(sib) {
            ui.set_parent(child, host);
        }
        ui.remove_node(survivor);
        ui.remove_node(gone);

        let kind = self.cells[sib].take().expect("live sibling").kind;
        self.cells[cell] = None;
        self.cells[parent].as_mut().expect("live cell").kind = kind;
        self.rehome(parent);
        // The survivor's node went with it; the split's node is the leaf now.
        self.repaint(ui, parent);
    }

    /// Point whatever a cell holds back at the cell: its panels, or its two
    /// children. Called after a `kind` has been moved between entries.
    fn rehome(&mut self, cell: usize) {
        match self.kind(cell).expect("live cell").clone() {
            Kind::Leaf { panels, .. } => {
                for p in panels {
                    self.panels[p.0 as usize].cell = cell;
                }
            }
            Kind::Split { kids, .. } => {
                for k in kids {
                    self.cells[k].as_mut().expect("live child").parent = Some(cell);
                }
            }
        }
    }

    fn move_to(&mut self, ui: &mut UiCore, p: PanelId, target: usize, side: Side) {
        let here = self.panels[p.0 as usize].cell;
        // Dropped back where it already is: joining its own strip, or
        // splitting a pane it is the only occupant of, are both the layout it
        // already has.
        if here == target && (side == Side::Tab || self.tabs_in(here) == 1) {
            return;
        }
        let into = match side {
            Side::Tab => target,
            _ => self.split(ui, target, side),
        };
        // Re-read: splitting the target moved its own panels to a new cell,
        // and this may well be one of them.
        let from = self.panels[p.0 as usize].cell;
        self.file(ui, p, into);
        self.unfile(ui, p, from);
    }

    /// Put a panel into a leaf and open it. Both its nodes move as they are —
    /// this is the whole of "the panel survives the move".
    fn file(&mut self, ui: &mut UiCore, p: PanelId, cell: usize) {
        let Some(Kind::Leaf {
            strip,
            body,
            panels,
            ..
        }) = self.kind_mut(cell)
        else {
            unreachable!("filing a panel into a split")
        };
        let (strip, body) = (*strip, *body);
        panels.push(p);
        let at = panels.len() - 1;

        let panel = &mut self.panels[p.0 as usize];
        panel.cell = cell;
        let (header, content) = (panel.header, panel.content);
        ui.set_parent(header, strip);
        ui.set_parent(content, body);
        self.open(ui, cell, at);
    }

    /// Take a panel out of a leaf's strip. Its nodes have already moved on,
    /// so this is bookkeeping plus, if nothing is left, the collapse.
    fn unfile(&mut self, ui: &mut UiCore, p: PanelId, from: usize) {
        let Some(Kind::Leaf { panels, open, .. }) = self.kind_mut(from) else {
            return;
        };
        let at = panels.iter().position(|&q| q == p);
        panels.retain(|&q| q != p);
        // A tab that left from the left of the open one shifts it down.
        let shown = match at {
            Some(a) if a < *open => *open - 1,
            _ => *open,
        };
        match panels.is_empty() {
            true => self.collapse(ui, from),
            false => self.open(ui, from, shown),
        }
    }

    /// Show tab `i` of a leaf: the open header wears the selected look and
    /// the bright label, the rest dim, and only the open pane has a box.
    ///
    /// Restates every tab rather than the two that moved. A leaf holds a
    /// handful and every write is gated, so this uploads the same bytes as
    /// the careful version and keeps no record of what was open.
    fn open(&mut self, ui: &mut UiCore, cell: usize, i: usize) {
        let Some(Kind::Leaf { panels, open, .. }) = self.kind_mut(cell) else {
            return;
        };
        if panels.is_empty() {
            return;
        }
        *open = i.min(panels.len() - 1);
        let (open, panels) = (*open, panels.clone());

        for (k, p) in panels.into_iter().enumerate() {
            let panel = &self.panels[p.0 as usize];
            let (header, label, content) = (panel.header, panel.label, panel.content);
            ui.set_state_style(header, self.style.tab.look(k == open));
            let color = match k == open {
                true => self.style.tab.text,
                false => self.style.tab.text_dim,
            };
            label.set_color(ui, color);
            ui.set_visible(content, k == open);
        }
    }

    /// Paint a cell's box: a leaf wears the surface, a split wears nothing.
    ///
    /// The whole leaf and not just its body, because a dock that fills the
    /// window has to be opaque edge to edge — otherwise whatever is behind it
    /// shows through the gap between the strip and the pane, and behind an
    /// editor's dock is the camera.
    fn repaint(&self, ui: &mut UiCore, cell: usize) {
        let fill = match self.kind(cell) {
            Some(Kind::Leaf { .. }) => self.style.surface,
            _ => alpha(self.style.surface, 0),
        };
        let s = UiStyle::fill(fill).radius(self.style.radius);
        ui.set_background(self.node_of(cell), s);
    }

    // ── Aiming ──────────────────────────────────────────────────────────

    /// Which leaf the pointer is over, and what a drop there would do.
    fn aim(&self, ui: &UiCore, p: [f32; 2]) -> Option<(usize, Side)> {
        (0..self.cells.len())
            .filter(|&c| matches!(self.kind(c), Some(Kind::Leaf { .. })))
            .find_map(|c| zone(ui.node_rect(self.node_of(c)), p).map(|s| (c, s)))
    }

    /// Draw the aim, or clear it. Insets are relative to the dock's own box,
    /// which is why the highlight is a child of the root rather than of the
    /// leaf it covers — a leaf's box is exactly what it must not be clipped
    /// to when the drop would take half of it.
    fn aim_at(&mut self, ui: &mut UiCore, aim: Option<(usize, Side)>) {
        if self.aiming == aim {
            return;
        }
        self.aiming = aim;
        let Some((cell, side)) = aim else {
            return ui.set_visible(self.overlay, false);
        };
        let (r, base, f) = (
            ui.node_rect(self.node_of(cell)),
            ui.node_rect(self.root),
            side.frac(),
        );
        let mut s = ui.node_style(self.overlay);
        s.display = Display::Flex;
        s.inset = Rect {
            left: px(r[0] - base[0] + f[0] * r[2]),
            top: px(r[1] - base[1] + f[1] * r[3]),
            right: LengthPercentageAuto::AUTO,
            bottom: LengthPercentageAuto::AUTO,
        };
        s.size = Size {
            width: px(r[2] * f[2]),
            height: px(r[3] * f[3]),
        };
        ui.set_node_style(self.overlay, s);
    }

    /// Put the panel on the pointer. The ghost is the engine's; only what it
    /// looks like is the dock's.
    fn lift(&mut self, ui: &mut UiCore, p: PanelId) {
        let s = self.style;
        let ghost = ui.grab(DragPanel(p));
        let mut style = ui.node_style(ghost);
        style.display = Display::Flex;
        style.padding = Rect::length(s.tab.padding);
        ui.set_node_style(ghost, style);
        ui.set_background(
            ghost,
            UiStyle::fill(s.tab.selected)
                .border(s.zone_edge, 1.0)
                .radius(s.radius),
        );
        let title = self.panels[p.0 as usize].title.clone();
        ui.label(ghost, s.tab.text_px, s.tab.text, &title);
    }

    // ── Cell lookups ────────────────────────────────────────────────────

    fn kind(&self, c: usize) -> Option<&Kind> {
        self.cells.get(c)?.as_ref().map(|c| &c.kind)
    }

    fn kind_mut(&mut self, c: usize) -> Option<&mut Kind> {
        self.cells.get_mut(c)?.as_mut().map(|c| &mut c.kind)
    }

    fn node_of(&self, c: usize) -> NodeId {
        self.cells[c].as_ref().expect("live cell").node
    }

    /// A cell's child nodes in layout order — the run that splitting moves
    /// down into a new child and collapsing moves back up.
    fn children_of(&self, c: usize) -> Vec<NodeId> {
        match self.kind(c).expect("live cell") {
            Kind::Leaf { strip, body, .. } => vec![*strip, *body],
            Kind::Split { kids, divider } => {
                vec![self.node_of(kids[0]), *divider, self.node_of(kids[1])]
            }
        }
    }

    /// The line between two halves, and the thing you drag to move it. One
    /// node does both: what you can see is exactly what you can grab.
    fn divider(&self, ui: &mut UiCore, parent: NodeId, across: bool) -> NodeId {
        let s = self.style;
        let thick = px(s.divider);
        let n = ui.node(
            parent,
            Style {
                flex_shrink: 0.0,
                size: match across {
                    true => Size {
                        width: thick,
                        height: TaffyAuto::AUTO,
                    },
                    false => Size {
                        width: TaffyAuto::AUTO,
                        height: thick,
                    },
                },
                ..Default::default()
            },
        );
        let base = UiStyle::fill(s.line).radius(s.divider * 0.5);
        ui.set_state_style(
            n,
            StateStyle::fills(base, s.line, s.line_hover, s.line_held),
        );
        ui.set_events(n, Events::CLICK | Events::HOVER);
        n
    }

    fn open_of(&self, c: usize) -> usize {
        match self.kind(c) {
            Some(Kind::Leaf { open, .. }) => *open,
            _ => 0,
        }
    }

    fn tabs_in(&self, c: usize) -> usize {
        match self.kind(c) {
            Some(Kind::Leaf { panels, .. }) => panels.len(),
            _ => 0,
        }
    }

    fn first_leaf(&self) -> usize {
        (0..self.cells.len())
            .find(|&c| matches!(self.kind(c), Some(Kind::Leaf { .. })))
            .expect("a dock always holds at least one leaf")
    }

    /// Which leaf a panel is in, and where in its strip.
    fn locate(&self, p: PanelId) -> (usize, usize) {
        let cell = self.panels[p.0 as usize].cell;
        let Some(Kind::Leaf { panels, .. }) = self.kind(cell) else {
            unreachable!("a panel's cell is always a leaf")
        };
        (
            cell,
            panels
                .iter()
                .position(|&q| q == p)
                .expect("a panel is listed by its own leaf"),
        )
    }

    /// Reuse a collapsed entry if there is one — a dock dragged around all
    /// afternoon churns cells, and nothing else would ever give them back.
    fn new_cell(&mut self, c: Cell) -> usize {
        match self.cells.iter().position(|s| s.is_none()) {
            Some(i) => {
                self.cells[i] = Some(c);
                i
            }
            None => {
                self.cells.push(Some(c));
                self.cells.len() - 1
            }
        }
    }
}

/// A role's colour at a different alpha. The drop zone is a wash over the
/// pane it would take, so it has to be the accent *and* see-through — and
/// deriving it beats a second hex literal to keep in step.
const fn alpha(color: u32, a: u8) -> u32 {
    (color & 0x00FF_FFFF) | ((a as u32) << 24)
}

/// Which part of `rect` a point aims at, or `None` off the box. Nearest edge
/// wins, and only inside its band — everything else is a tab join.
fn zone(rect: [f32; 4], p: [f32; 2]) -> Option<Side> {
    let (w, h) = (rect[2], rect[3]);
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    let (x, y) = ((p[0] - rect[0]) / w, (p[1] - rect[1]) / h);
    if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
        return None;
    }
    let (d, side) = [
        (x, Side::Left),
        (1.0 - x, Side::Right),
        (y, Side::Top),
        (1.0 - y, Side::Bottom),
    ]
    .into_iter()
    .min_by(|a, b| a.0.total_cmp(&b.0))
    .expect("four edges");
    Some(match d < EDGE {
        true => side,
        false => Side::Tab,
    })
}

/// A cell fills its share of whatever holds it, whichever kind it is. The
/// basis of zero is load-bearing: it makes `flex_grow` the proportion, which
/// is what a divider drag writes and the only place a split's ratio lives.
fn cell_style(dir: FlexDirection) -> Style {
    Style {
        display: Display::Flex,
        flex_direction: dir,
        flex_grow: 1.0,
        flex_basis: px(0.0),
        // Flex items are content-sized at minimum by default, which one wide
        // label inside a panel would then enforce against the whole split.
        min_size: Size {
            width: px(0.0),
            height: px(0.0),
        },
        ..Default::default()
    }
}

/// A leaf is a cell with its strip above its body.
fn leaf_style(gap: f32) -> Style {
    Style {
        gap: Size {
            width: zero(),
            height: px(gap),
        },
        ..cell_style(FlexDirection::Column)
    }
}

/// The two nodes every leaf has, and the surface behind its panes.
fn build_leaf(ui: &mut UiCore, node: NodeId, s: &DockStyle) -> Kind {
    let strip = ui.node(
        node,
        Style {
            display: Display::Flex,
            gap: Size {
                width: px(2.0),
                height: zero(),
            },
            ..Default::default()
        },
    );
    let body = ui.node(
        node,
        Style {
            display: Display::Flex,
            flex_grow: 1.0,
            flex_basis: px(0.0),
            min_size: Size {
                width: px(0.0),
                height: px(0.0),
            },
            ..Default::default()
        },
    );
    ui.set_background(node, UiStyle::fill(s.surface).radius(s.radius));
    Kind::Leaf {
        strip,
        body,
        panels: Vec::new(),
        open: 0,
    }
}

fn header_style(s: &DockStyle) -> Style {
    Style {
        display: Display::Flex,
        padding: Rect::length(s.tab.padding),
        ..Default::default()
    }
}

/// A panel's own container: fills the leaf's body, and stacks what the
/// caller puts in it.
fn pane_style(s: &DockStyle) -> Style {
    Style {
        display: Display::Flex,
        flex_direction: FlexDirection::Column,
        flex_grow: 1.0,
        flex_basis: px(0.0),
        min_size: Size {
            width: px(0.0),
            height: px(0.0),
        },
        size: Size {
            width: TaffyAuto::AUTO,
            height: TaffyAuto::AUTO,
        },
        padding: Rect::length(s.tab.padding),
        gap: Size {
            width: zero(),
            height: px(s.tab.gap),
        },
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::style::percent;

    const W: f32 = 400.0;
    const H: f32 = 300.0;

    /// A dock filling a 400x300 screen, with two panels stacked as tabs in
    /// its one leaf. Each panel's content carries a label the tests aim at,
    /// spelled differently from the title so the two never collide.
    fn dock(core: &mut UiCore) -> (DockSpace, PanelId, PanelId) {
        let root = core.root();
        let mut d = DockSpace::new(
            core,
            root,
            Style {
                size: Size {
                    width: percent(1.0_f32),
                    height: percent(1.0_f32),
                },
                ..Default::default()
            },
            DockStyle::default(),
        );
        let (a, b) = (d.panel(core, "scene"), d.panel(core, "props"));
        for (p, body) in [(a, "in scene"), (b, "in props")] {
            core.label(d.content(p), 9.0, 0xFFFF_FFFF, body);
        }
        core.run_layout([W, H]);
        (d, a, b)
    }

    fn visible(core: &UiCore) -> Vec<String> {
        core.text_nodes().into_iter().map(|(_, t, _)| t).collect()
    }

    /// The whole box a panel's leaf owns — the split's half, headers and all,
    /// which unlike the pane is there whether or not the tab is open.
    fn leaf_rect(core: &UiCore, d: &DockSpace, p: PanelId) -> [f32; 4] {
        core.node_rect(d.node_of(d.panels[p.0 as usize].cell))
    }

    /// The line between a split's two halves.
    fn divider_of(d: &DockSpace, c: usize) -> NodeId {
        let Some(Kind::Split { divider, .. }) = d.kind(c) else {
            panic!("cell {c} is not a split")
        };
        *divider
    }

    /// Middle of the line's on-screen box — where a hand would grab it.
    fn grab_point(core: &UiCore, d: &DockSpace, c: usize) -> [f32; 2] {
        let r = core.node_rect(divider_of(d, c));
        [r[0] + r[2] * 0.5, r[1] + r[3] * 0.5]
    }

    /// Middle of the on-screen box of whichever text node says `text`.
    fn at(core: &UiCore, text: &str) -> [f32; 2] {
        let (_, _, r) = core
            .text_nodes()
            .into_iter()
            .find(|(_, t, _)| t == text)
            .unwrap_or_else(|| panic!("no visible text {text:?}"));
        [r[0] + r[2] * 0.5, r[1] + r[3] * 0.5]
    }

    /// Press, move, release — one frame each, which is all the pointer layer
    /// needs, with the dock folded in the way an application folds it.
    fn drag(core: &mut UiCore, d: &mut DockSpace, from: [f32; 2], to: [f32; 2]) {
        for (p, pressed, released) in [(from, true, false), (to, false, false), (to, false, true)] {
            core.update_pointer(p, pressed, released, 0.0, 0.0);
            d.update(core);
            core.run_layout([W, H]);
        }
    }

    /// Two documents side by side, the left one holding a dock of its own —
    /// the editor's shape, where a hierarchy and an inspector are sub-panels
    /// of the document they belong to.
    fn nested(core: &mut UiCore) -> (DockSpace, DockSpace, PanelId) {
        let root = core.root();
        let mut outer = DockSpace::new(core, root, fill_style(), DockStyle::default());
        let (left, right) = (
            outer.panel(core, "left doc"),
            outer.panel(core, "right doc"),
        );
        outer.dock(core, right, left, Side::Right);
        core.label(outer.content(right), 9.0, 0xFFFF_FFFF, "in right");

        let mut inner = DockSpace::new(
            core,
            outer.content(left),
            fill_style(),
            DockStyle::default(),
        );
        let (tree, view) = (inner.panel(core, "tree"), inner.panel(core, "view"));
        inner.dock(core, view, tree, Side::Right);
        core.run_layout([W, H]);
        (outer, inner, tree)
    }

    fn fill_style() -> Style {
        Style {
            flex_grow: 1.0,
            flex_basis: px(0.0),
            min_size: Size {
                width: px(0.0),
                height: px(0.0),
            },
            ..Default::default()
        }
    }

    /// Press, move, release with both docks folded in, the way an editor
    /// running a dock inside a dock folds them.
    fn drag_nested(
        core: &mut UiCore,
        outer: &mut DockSpace,
        inner: &mut DockSpace,
        from: [f32; 2],
        to: [f32; 2],
    ) {
        for (p, pressed, released) in [(from, true, false), (to, false, false), (to, false, true)] {
            core.update_pointer(p, pressed, released, 0.0, 0.0);
            outer.update(core);
            inner.update(core);
            core.run_layout([W, H]);
        }
    }

    /// What a dock inside a dock buys: a drop is aimed at *this* dock's
    /// leaves, and the outer document's are not among them — so nothing
    /// enforces the confinement, the nesting is it.
    #[test]
    fn a_nested_panel_cannot_be_dragged_into_another_dock() {
        let mut core = UiCore::new();
        let (mut outer, mut inner, tree) = nested(&mut core);
        let held = core.text_field(inner.content(tree), "half typed", Default::default());
        core.run_layout([W, H]);

        let (header, target) = (at(&core, "tree"), at(&core, "in right"));
        drag_nested(&mut core, &mut outer, &mut inner, header, target);

        let r = core.node_rect(inner.content(tree));
        assert!(r[2] > 0.0, "the panel is still on screen");
        assert!(
            r[0] + r[2] <= W * 0.5,
            "and still inside its own document: {r:?}"
        );
        assert_eq!(held.text(&core), "half typed", "and untouched");
    }

    /// The other half of the same fact: inside its own document it goes
    /// wherever it is dropped.
    #[test]
    fn a_nested_panel_moves_freely_within_its_own_dock() {
        let mut core = UiCore::new();
        let (mut outer, mut inner, tree) = nested(&mut core);
        let before = core.node_rect(inner.content(tree));

        let (header, target) = (at(&core, "tree"), at(&core, "view"));
        drag_nested(&mut core, &mut outer, &mut inner, header, target);

        let r = core.node_rect(inner.content(tree));
        assert_ne!(r, before, "the drop rearranged the document");
        assert!(r[0] + r[2] <= W * 0.5, "without leaving it: {r:?}");
    }

    /// The capability. The panel's content node is the same node before and
    /// after, so everything hanging off it — values, text, scroll offsets —
    /// crosses the move untouched. A rebuild could not promise this.
    #[test]
    fn a_moved_panel_is_the_same_panel() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);
        let (node, field) = (
            d.content(a),
            core.text_field(d.content(a), "half typed", Default::default()),
        );

        d.dock(&mut core, a, b, Side::Right);
        core.run_layout([W, H]);

        assert_eq!(d.content(a), node, "the pane node survives the move");
        assert_eq!(
            field.text(&core),
            "half typed",
            "and so does what was in it"
        );
        assert!(
            core.node_rect(node)[2] > 0.0,
            "and it is on screen where it landed"
        );
    }

    /// A split hands each half the same width, and the arriving panel takes
    /// the side it was aimed at.
    #[test]
    fn splitting_puts_the_two_panes_side_by_side() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);
        d.dock(&mut core, b, a, Side::Right);
        core.run_layout([W, H]);

        let (ra, rb) = (core.node_rect(d.content(a)), core.node_rect(d.content(b)));
        assert!(ra[0] < rb[0], "b was aimed right of a");
        assert!(
            (ra[2] - rb[2]).abs() <= 1.0,
            "a split is even: {ra:?} vs {rb:?}"
        );
        assert!(ra[2] < W * 0.6, "neither half still owns the whole dock");
    }

    /// Splitting downwards stacks instead, and the leading sides put the
    /// arrival first.
    #[test]
    fn every_side_lands_where_it_is_aimed() {
        for (side, first, across) in [
            (Side::Left, true, true),
            (Side::Right, false, true),
            (Side::Top, true, false),
            (Side::Bottom, false, false),
        ] {
            let mut core = UiCore::new();
            let (mut d, a, b) = dock(&mut core);
            d.dock(&mut core, b, a, side);
            core.run_layout([W, H]);

            let (ra, rb) = (core.node_rect(d.content(a)), core.node_rect(d.content(b)));
            let axis = usize::from(!across);
            let (lead, trail) = match first {
                true => (rb, ra),
                false => (ra, rb),
            };
            assert!(
                lead[axis] < trail[axis],
                "{side:?} put the panes the wrong way round: {lead:?} then {trail:?}",
            );
        }
    }

    /// The other half of the shape: a leaf that loses its last panel takes
    /// its split with it, and the survivor gets the whole box back.
    #[test]
    fn emptying_a_leaf_collapses_its_split() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);
        let whole = leaf_rect(&core, &d, a);

        d.dock(&mut core, b, a, Side::Right);
        core.run_layout([W, H]);
        assert!(leaf_rect(&core, &d, a)[2] < whole[2], "split first");

        d.dock(&mut core, b, a, Side::Tab);
        core.run_layout([W, H]);
        assert_eq!(leaf_rect(&core, &d, a), whole, "a should have it all back");
        assert_eq!(
            d.cells.iter().filter(|c| c.is_some()).count(),
            1,
            "the split and its spare leaf are gone, not just hidden",
        );
    }

    /// A closed tab's pane is collapsed, not merely covered — so it paints
    /// nothing and its contents cannot be hit.
    #[test]
    fn only_the_open_tab_has_a_box() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);

        assert!(d.showing(b) && !d.showing(a), "the newest tab opens");
        assert!(!visible(&core).contains(&"in scene".to_string()));
        let r = core.node_rect(d.content(a));
        assert_eq!(
            [r[2], r[3]],
            [0.0, 0.0],
            "a closed pane has no area to paint or hit"
        );

        d.select(&mut core, a);
        core.run_layout([W, H]);
        assert!(visible(&core).contains(&"in scene".to_string()));
        assert!(!visible(&core).contains(&"in props".to_string()));
    }

    /// The whole gesture through the pointer layer: press a header, drag it
    /// to the far edge, release. Nothing in this test names a dock method.
    #[test]
    fn dragging_a_header_to_an_edge_splits_the_dock() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);
        d.select(&mut core, a);
        core.run_layout([W, H]);

        let from = at(&core, "props");
        drag(&mut core, &mut d, from, [W - 12.0, H * 0.5]);

        let (ra, rb) = (core.node_rect(d.content(a)), core.node_rect(d.content(b)));
        assert!(
            ra[0] < rb[0] && rb[2] > 0.0,
            "props should sit right of scene"
        );
        assert!((ra[2] - rb[2]).abs() <= 1.0, "and take half the dock");
    }

    /// While the drag is in flight the highlight shows the box the drop would
    /// take — half the target for an edge, all of it for a tab join.
    #[test]
    fn the_highlight_shows_what_the_drop_would_take() {
        let mut core = UiCore::new();
        let (mut d, a, _) = dock(&mut core);
        d.select(&mut core, a);
        core.run_layout([W, H]);
        let leaf = core.node_rect(d.node_of(0));

        let from = at(&core, "props");
        core.update_pointer(from, true, false, 0.0, 0.0);
        d.update(&mut core);
        core.update_pointer([W - 12.0, H * 0.5], false, false, 0.0, 0.0);
        d.update(&mut core);
        core.run_layout([W, H]);

        let r = core.node_rect(d.overlay);
        assert_eq!(d.aiming.map(|(_, s)| s), Some(Side::Right));
        assert!(
            (r[2] - leaf[2] * 0.5).abs() <= 1.0,
            "half the leaf: {r:?} of {leaf:?}"
        );
        assert!(
            (r[0] - (leaf[0] + leaf[2] * 0.5)).abs() <= 1.0,
            "and the right half"
        );

        // Released off the dock entirely: the highlight goes and nothing moves.
        core.update_pointer([W - 12.0, H * 0.5], false, true, 0.0, 0.0);
        d.update(&mut core);
        core.run_layout([W, H]);
        assert_eq!(
            core.node_rect(d.overlay),
            [0.0; 4],
            "the aim clears on release"
        );
    }

    /// A press that never travelled is a click on the tab, not a lift.
    #[test]
    fn a_header_click_opens_its_tab_and_moves_nothing() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);
        let cells = d.cells.len();

        let p = at(&core, "scene");
        drag(&mut core, &mut d, p, p);

        assert!(d.showing(a) && !d.showing(b), "the clicked tab should open");
        assert_eq!(d.cells.len(), cells, "and no split should have appeared");
    }

    /// Dropped back where it already is, a panel is left alone rather than
    /// being torn out of its leaf and put back.
    #[test]
    fn a_panel_dropped_where_it_already_is_changes_nothing() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);
        d.dock(&mut core, b, a, Side::Right);
        core.run_layout([W, H]);
        let (ra, cells) = (core.node_rect(d.content(a)), d.cells.len());

        d.dock(&mut core, b, b, Side::Tab);
        d.dock(&mut core, b, b, Side::Left);
        core.run_layout([W, H]);

        assert_eq!(core.node_rect(d.content(a)), ra);
        assert_eq!(
            d.cells.len(),
            cells,
            "no cell was minted for a move that was not one"
        );
    }

    /// The invariant every widget owes: a dock nobody is touching costs
    /// nothing, however many times it is asked.
    #[test]
    fn an_idle_dock_uploads_nothing() {
        let mut core = UiCore::new();
        let (mut d, _, _) = dock(&mut core);
        // Parked in the open pane rather than on a header, so the baseline is
        // taken after the hover it would otherwise light up on frame one.
        core.update_pointer([W - 2.0, H - 2.0], false, false, 0.0, 0.0);
        d.update(&mut core);
        core.run_layout([W, H]);

        let (mut stage, mut dirty) = (vec![0u32; 1 << 16], vec![0u32; 1 << 10]);
        core.quad.upload(&mut stage, &mut dirty);
        core.style.upload(&mut stage, &mut dirty);

        for _ in 0..8 {
            core.update_pointer([W - 2.0, H - 2.0], false, false, 0.0, 0.0);
            d.update(&mut core);
            core.run_layout([W, H]);
        }
        let clean = (i64::MAX, -1);
        assert_eq!(
            core.quad.upload(&mut stage, &mut dirty),
            clean,
            "quads dirtied"
        );
        assert_eq!(
            core.style.upload(&mut stage, &mut dirty),
            clean,
            "styles dirtied"
        );
    }

    /// The line between two panes is also the handle that moves it, and the
    /// move is absolute: the boundary lands under the cursor rather than
    /// integrating a delta that could drift away from it.
    #[test]
    fn dragging_a_divider_resizes_both_halves() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);
        d.dock(&mut core, b, a, Side::Right);
        core.run_layout([W, H]);
        let (line, mid) = (core.node_rect(divider_of(&d, 0)), grab_point(&core, &d, 0));
        let was = leaf_rect(&core, &d, a)[2];

        drag(&mut core, &mut d, mid, [mid[0] + 80.0, mid[1]]);

        let (ra, rb) = (leaf_rect(&core, &d, a), leaf_rect(&core, &d, b));
        assert!(
            (ra[2] - (was + 80.0)).abs() <= 2.0,
            "a should follow the line: {ra:?}"
        );
        assert!(
            (ra[2] + line[2] + rb[2] - W).abs() <= 1.0,
            "and the two halves plus the line should still fill the dock",
        );
    }

    /// Dragged past the end, a divider stops rather than squashing a pane to
    /// nothing — a pane with no headers left is one nobody can get back.
    #[test]
    fn a_divider_will_not_squash_a_pane_out_of_reach() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);
        d.dock(&mut core, b, a, Side::Right);
        core.run_layout([W, H]);
        let mid = grab_point(&core, &d, 0);

        drag(&mut core, &mut d, mid, [W + 200.0, mid[1]]);

        assert!(
            leaf_rect(&core, &d, b)[2] >= MIN_PANE - 1.0,
            "b was squashed away"
        );
        assert!(
            leaf_rect(&core, &d, a)[2] > W * 0.7,
            "but a did take almost all of it"
        );
    }

    /// The line is freed with the split it separated. A stale handle panics,
    /// which is the tree's way of saying "gone" rather than "orphaned" — and
    /// an orphan here would be a 6 px bar left inside the surviving leaf.
    #[test]
    #[should_panic(expected = "stale NodeId")]
    fn collapsing_a_split_frees_its_divider() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);
        d.dock(&mut core, b, a, Side::Right);
        let line = divider_of(&d, 0);
        d.dock(&mut core, b, a, Side::Tab);
        core.node_rect(line);
    }

    /// A cell's share of its parent belongs to the *cell*, not to what it is
    /// holding at the time. Splitting one half, and folding that split back
    /// away, both leave the outer boundary exactly where the user put it.
    #[test]
    fn a_resized_split_keeps_its_share_through_a_nested_one() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);
        d.dock(&mut core, b, a, Side::Right);
        core.run_layout([W, H]);
        let mid = grab_point(&core, &d, 0);
        drag(&mut core, &mut d, mid, [mid[0] + 80.0, mid[1]]);
        let outer = grab_point(&core, &d, 0);

        let c = d.panel(&mut core, "third");
        d.dock(&mut core, c, b, Side::Bottom);
        core.run_layout([W, H]);
        assert_eq!(
            grab_point(&core, &d, 0),
            outer,
            "splitting a half moved the outer line"
        );

        d.dock(&mut core, c, b, Side::Tab);
        core.run_layout([W, H]);
        assert_eq!(
            grab_point(&core, &d, 0),
            outer,
            "and folding it away moved it back"
        );
    }

    /// A dock is sized by the box it was handed, not by what someone put in
    /// a pane. A flex item refuses to shrink below its own content unless it
    /// is told it may — so one long readout used to widen the whole dock, and
    /// every split with it, on the frame the text changed.
    #[test]
    fn a_long_line_in_one_pane_does_not_widen_the_dock() {
        let mut core = UiCore::new();
        let root = core.root();
        let frame = core.node(
            root,
            Style {
                display: Display::Flex,
                size: Size {
                    width: px(300.0),
                    height: px(200.0),
                },
                ..Default::default()
            },
        );
        let mut d = DockSpace::new(
            &mut core,
            frame,
            Style {
                flex_grow: 1.0,
                ..Default::default()
            },
            DockStyle::default(),
        );
        let (a, b) = (d.panel(&mut core, "a"), d.panel(&mut core, "b"));
        d.dock(&mut core, b, a, Side::Right);
        core.run_layout([W, H]);
        let half = leaf_rect(&core, &d, a);

        core.label(
            d.content(b),
            11.0,
            0xFFFF_FFFF,
            &"very wide readout ".repeat(6),
        );
        core.run_layout([W, H]);

        assert_eq!(
            core.node_rect(d.root)[2],
            300.0,
            "the dock kept the box it was given"
        );
        assert_eq!(
            leaf_rect(&core, &d, a),
            half,
            "so the split did not move either"
        );
    }

    /// What the leaf holding `p` paints behind everything in it.
    fn surface_of(core: &UiCore, d: &DockSpace, p: PanelId) -> u32 {
        let node = d.node_of(d.panels[p.0 as usize].cell);
        core.style
            .get(core.paint_slots(node).0.expect("a leaf has a fill"))
            .fill
    }

    /// The surface belongs to the *leaf*, not to the node it happens to be
    /// on: a split moves the strip and the pane down into a fresh child, and
    /// a collapse moves them back up into the split's own node. Miss either
    /// and a pane is a window onto whatever the dock is drawn over.
    #[test]
    fn a_leafs_surface_follows_it_through_a_split() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);
        let opaque = DockStyle::default().surface;

        d.dock(&mut core, b, a, Side::Right);
        core.run_layout([W, H]);
        assert_eq!(
            surface_of(&core, &d, a),
            opaque,
            "a moved down into the split"
        );
        assert_eq!(surface_of(&core, &d, b), opaque, "and b is the new leaf");
        let split = core
            .style
            .get(core.paint_slots(d.node_of(0)).0.unwrap())
            .fill;
        assert_eq!(split, alpha(opaque, 0), "the split itself paints nothing");

        d.dock(&mut core, b, a, Side::Tab);
        core.run_layout([W, H]);
        assert_eq!(
            surface_of(&core, &d, a),
            opaque,
            "and comes back up when it folds"
        );
    }

    /// A starting layout is rarely even. `set_ratio` writes the two numbers a
    /// divider drag writes, so the boundary lands where a drag would have put
    /// it — and stays there when the window resizes.
    #[test]
    fn a_split_can_be_given_its_proportion() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);
        d.dock(&mut core, b, a, Side::Right);
        d.set_ratio(&mut core, a, 0.25);
        core.run_layout([W, H]);

        let (ra, rb) = (leaf_rect(&core, &d, a), leaf_rect(&core, &d, b));
        assert!(
            (ra[2] - W * 0.25).abs() <= 2.0,
            "a should take a quarter: {ra:?}"
        );
        assert!((ra[2] + rb[2] - (W - 2.0)).abs() <= 1.0, "and b the rest");
    }

    /// A layout is the arrangement, not the panels: what one dock reports a
    /// fresh dock with the same titles reproduces.
    #[test]
    fn a_saved_layout_is_restored() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);
        d.dock(&mut core, b, a, Side::Bottom);
        d.set_ratio(&mut core, a, 0.75);
        core.run_layout([W, H]);
        let (saved, rect) = (d.layout(&core), leaf_rect(&core, &d, b));

        let mut core = UiCore::new();
        let (mut d, _, b) = dock(&mut core);
        d.apply(&mut core, &saved);
        core.run_layout([W, H]);
        assert_eq!(d.layout(&core), saved, "the same tree");
        assert_eq!(leaf_rect(&core, &d, b), rect, "in the same boxes");
    }

    /// A layout naming a panel that is no longer there restores the rest:
    /// the split it was half of is the half that is left.
    #[test]
    fn a_layout_survives_a_panel_that_is_gone() {
        let live = Layout::Leaf {
            tabs: vec!["scene".into(), "props".into()],
            open: 1,
        };
        let saved = Layout::Split {
            across: true,
            share: 0.3,
            kids: Box::new([
                Layout::Leaf {
                    tabs: vec!["gone".into()],
                    open: 0,
                },
                live.clone(),
            ]),
        };
        let mut core = UiCore::new();
        let (mut d, _, b) = dock(&mut core);
        d.apply(&mut core, &saved);
        core.run_layout([W, H]);

        assert_eq!(d.layout(&core), live);
        assert!(d.showing(b), "and the tab it left open is open");
    }

    /// The flag a caller saves on: a gesture sets it, building a layout does
    /// not, and the asking clears it.
    #[test]
    fn a_move_is_reported_once() {
        let mut core = UiCore::new();
        let (mut d, a, b) = dock(&mut core);
        d.dock(&mut core, b, a, Side::Right);
        core.run_layout([W, H]);
        assert!(!d.changed(), "the starting layout is not a move");

        let (from, to) = (at(&core, "props"), at(&core, "in scene"));
        drag(&mut core, &mut d, from, to);
        assert!(d.changed(), "the drop moved a panel");
        assert!(!d.changed(), "and the asking cleared it");
    }

    /// Aiming: the middle of a box — over half of it — joins the strip, and
    /// only the outer quarter of an edge splits.
    #[test]
    fn the_middle_of_a_pane_joins_its_strip() {
        let r = [0.0, 0.0, 100.0, 100.0];
        assert_eq!(zone(r, [50.0, 50.0]), Some(Side::Tab));
        assert_eq!(zone(r, [5.0, 50.0]), Some(Side::Left));
        assert_eq!(zone(r, [95.0, 50.0]), Some(Side::Right));
        assert_eq!(zone(r, [50.0, 5.0]), Some(Side::Top));
        assert_eq!(zone(r, [50.0, 95.0]), Some(Side::Bottom));
        // A corner belongs to its nearest edge, and outside is nobody's.
        assert_eq!(zone(r, [4.0, 6.0]), Some(Side::Left));
        assert_eq!(zone(r, [-1.0, 50.0]), None);
        assert_eq!(
            zone([0.0; 4], [0.0, 0.0]),
            None,
            "a collapsed leaf takes no drops"
        );
    }
}
