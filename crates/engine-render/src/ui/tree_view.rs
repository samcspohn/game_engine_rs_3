//! A collapsible tree over a [`RowList`] — the scene hierarchy panel.
//!
//! # It consumes a tree, it does not own one
//!
//! Node identity is an opaque `u64` this module never interprets, and
//! structure is read through a closure. So the view works over the transform
//! hierarchy, a folder listing, or anything else, and — more importantly —
//! there is **one source of truth**. A view that owned a mirror of the
//! hierarchy would acquire a sync obligation and a whole class of drift bugs
//! for no benefit, since `TransformHierarchy` already *is* a single-rooted
//! tree with children lists.
//!
//! What the view does own is **expansion state** and the **flattened visible
//! list**, because those are its own and nothing else knows them.
//!
//! # Sub-edits, not rebuilds
//!
//! The flat list is a DFS preorder, and a node's visible subtree is therefore
//! a **contiguous run immediately after it**. Every structural edit is a
//! splice of that run rather than a re-walk:
//!
//! | Edit | Cost |
//! |---|---|
//! | collapse | scan forward for the run, `drain` it |
//! | expand | flatten the subtree (`O(K)`), `splice` it in |
//! | reorder within a parent | one `rotate`, bounded by the drag distance |
//! | re-parent | `drain` + `splice` + a uniform depth delta over `K` |
//!
//! None of these walks the whole tree, and the moved subtree's *internal*
//! shape and expansion are untouched — only its depth shifts. Renames are not
//! structural at all: [`RowList`] pulls text per visible row.
//!
//! [`Self::invalidate`] is the escape hatch for structure that changed
//! outside the view — a spawn or destroy from game code. It is the only path
//! that re-walks, and it is the thing a hierarchy version counter would
//! automate if one is added later.

use std::any::Any;
use std::collections::HashSet;
use std::marker::PhantomData;

use super::list::{DropMark, Row, RowContent, RowList, RowStyle};
use super::style::{auto, px, AlignItems, Display, LengthPercentage, Rect, Size, Style, TaffyZero};
use super::{font, Events, Label, NodeId, UiCore, UiStyle};

/// Travel that turns a press into a drag rather than a click that wobbled.
const DRAG_PX: f32 = 4.0;

/// Fraction of a row's height at each end that reads as "between rows"
/// rather than "into this one".
const EDGE: f32 = 0.25;

/// A drag payload that names a node of a tree.
///
/// The view never *constructs* one — the app does, at the grab, because only
/// the app knows what a node is in its own world. This is only how the view
/// reads a drag back: enough to reject a subtree dropped into itself and to
/// resolve a landing into a [`Dropped`], and nothing more.
pub trait TreeDrag: Any {
    fn node(&self) -> u64;

    /// Which tree this node belongs to. Node ids are only meaningful inside
    /// one model, so a view ignores a drag tagged for another rather than
    /// resolving the id against a tree it never came from.
    fn tree(&self) -> u64 {
        0
    }
}

/// This view's contribution to a pooled row, wrapped around the caller's.
///
/// `content` exists so indentation is *this* view's: the outer node carries
/// the position the ring depends on and nothing else, so the two never fight
/// over one `Style`. `app` is whatever `H::build` made, and the view never
/// looks inside it.
#[derive(Clone, Copy)]
struct TreeRow<H> {
    content: NodeId,
    /// Disclosure triangle, a child so innermost-wins separates "toggle" from
    /// "select" for free. It accepts `CLICK` and nothing else, which leaves
    /// the row hovered — and drop-able — underneath it.
    arrow: Label,
    app: H,
}

/// The view's own row: an indent, a disclosure triangle, then the caller's
/// content. Built by [`RowList`] when the pool grows, exactly like any other.
impl<H: RowContent> RowContent for TreeRow<H> {
    fn build(ui: &mut UiCore, row: NodeId, s: &RowStyle) -> Self {
        let content = ui.node(row, content_style(s, 0));
        let arrow = ui.label(content, s.text_px, s.arrow, "");
        ui.set_node_style(
            arrow,
            Style {
                size: Size { width: px(s.indent), height: auto() },
                flex_shrink: 0.0,
                ..Default::default()
            },
        );
        let app = H::build(ui, content, s);
        TreeRow { content, arrow, app }
    }
}

/// The default payload: a bare node id, for a tree whose caller has no ref
/// type of its own yet.
///
/// It is the *protocol* rather than an internal detail — a panel that knows
/// nothing about this view can construct one and drop it on the tree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DragNode(pub u64);

impl TreeDrag for DragNode {
    fn node(&self) -> u64 {
        self.0
    }
}

/// A released drag, resolved into the move it asks for.
///
/// The view reports; the caller applies — to its own hierarchy, and to the
/// view via [`TreeView::moved`]. That split is the module's whole premise: a
/// view that re-parented on its own would be a second source of truth, and
/// the caller could not refuse a move its own rules forbid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Dropped {
    /// The dragged node.
    pub node: u64,
    pub parent: u64,
    /// Index among `parent`'s children **after `node` has left its old
    /// parent** — the order [`TreeView::moved`] applies, so a caller that
    /// removes before it inserts needs no off-by-one of its own.
    pub at: usize,
}

/// One visible row: which node, and how deep it sits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Flat {
    id: u64,
    depth: u16,
}

/// A tree rendered as a virtualized flat row list.
///
/// Selection is deliberately *not* held here: it belongs to the caller, and
/// it must key on the node id rather than a row index, because collapsing
/// anything above a selected row changes that row's index.
/// `H` is what a row contains and `P` the payload its drags carry — both
/// opaque here. The defaults are the hierarchy-panel case, so a caller that
/// wants text rows and has no ref type of its own names neither.
#[derive(Clone)]
pub struct TreeView<H: RowContent = Label, P = DragNode> {
    list: RowList<TreeRow<H>>,
    style: RowStyle,
    /// The view reads drags as `P` and never builds one.
    payload: PhantomData<fn() -> P>,
    root: u64,
    /// Expanded nodes. Default-collapsed keeps the flatten `O(visible)`:
    /// opening a million-entity scene shows its roots, not a million rows.
    expanded: HashSet<u64>,
    flat: Vec<Flat>,
    /// Set by [`Self::invalidate`]; the next [`Self::sync`] re-walks.
    dirty: bool,
    /// What [`TreeDrag::tree`] must say for a drag to be this view's.
    tree: u64,
    /// Scratch for splices, kept to reuse its capacity.
    scratch: Vec<Flat>,
    /// The drop the release produced, for one [`Self::sync`].
    drop: Option<Dropped>,
}

impl<H: RowContent, P: TreeDrag> TreeView<H, P> {
    /// `root` is shown as a row like any other; a hierarchy panel wants it
    /// visible so there is somewhere to drop a node to un-parent it.
    pub fn new(ui: &mut UiCore, parent: NodeId, viewport: Style, style: RowStyle, root: u64) -> Self {
        // The root starts expanded: default-collapsed is about *descendants*
        // — that is what keeps the flatten `O(visible)` — while a panel that
        // opens to one unexpandable row shows nothing at all.
        Self {
            list: RowList::new(ui, parent, viewport, style),
            style,
            payload: PhantomData,
            root,
            expanded: HashSet::from([root]),
            flat: Vec::new(),
            dirty: true,
            tree: 0,
            scratch: Vec::new(),
            drop: None,
        }
    }

    /// Answer only for drags carrying this [`TreeDrag::tree`]. Two views over
    /// different models share a payload type, and an id from one names
    /// something else entirely in the other.
    pub fn with_tree(mut self, tree: u64) -> Self {
        self.tree = tree;
        self
    }

    /// The scroll area, for styling it.
    pub fn node(&self) -> NodeId {
        self.list.node()
    }

    pub fn is_expanded(&self, id: u64) -> bool {
        self.expanded.contains(&id)
    }

    /// Mark the structure stale. The next [`Self::sync`] re-walks from the
    /// root, preserving expansion. Call it when something *outside* the view
    /// spawns, destroys or re-parents a node.
    pub fn invalidate(&mut self) {
        self.dirty = true;
    }

    /// Visible rows, for tests and for a caller that wants to drive the
    /// selection by row.
    pub fn visible(&self) -> impl Iterator<Item = u64> + '_ {
        self.flat.iter().map(|f| f.id)
    }

    /// Re-flatten if needed, apply any pending toggle, and bind the pool.
    ///
    /// Two closures, for the two things only the caller knows:
    ///
    /// - `children` pushes a node's children in order — the structure, read
    ///   live so there is never a second copy to drift.
    /// - `bind` fills one row. It takes the **node id** rather than a row
    ///   index, because that is what a caller thinks in and what collapsing
    ///   cannot invalidate.
    ///
    /// How a row is *made* is not here at all: it is `H`, fixed when the view
    /// was constructed, so it cannot vary between frames and does not have to
    /// be restated on every one.
    ///
    /// Each closure is called only for what is needed: `children` for expanded
    /// subtrees on a structural edit, `bind` for the handful of pooled rows.
    pub fn sync(
        &mut self,
        ui: &mut UiCore,
        mut children: impl FnMut(u64, &mut Vec<u64>),
        mut bind: impl FnMut(&mut UiCore, Row<'_, H>, u64),
    ) {
        if let Some(i) = self.toggled(ui) {
            self.toggle_row(i, &mut children);
        }
        if self.dirty {
            self.rebuild(&mut children);
        }
        // After the structure settles — the aim is resolved against the flat
        // list this frame will actually draw.
        let mark = self.update_drag(ui);

        let (flat, expanded, s) = (&self.flat, &self.expanded, self.style);
        let mut kids = Vec::new();
        self.list.sync(ui, flat.len(), |ui, row, i| {
            let f = flat[i];
            kids.clear();
            children(f.id, &mut kids);
            let open = (!kids.is_empty()).then(|| expanded.contains(&f.id));
            bind_arrow(ui, &row, &s, f.depth, open);
            // The caller sees its *own* content on the row node, so
            // `set_selected` still reaches the fill the list owns while
            // `set_text` reaches the label the caller declared.
            bind(ui, Row::new(row.node(), &row.app, s), f.id);
        });
        // Last, so the marker lands on a pool that has finished growing.
        self.list.set_drop_mark(ui, mark);
    }

    /// Pick up a row: grab `payload` and dress the ghost as a row of this
    /// view, using the same [`RowContent`] the list itself uses.
    ///
    /// The caller still owns the meaning — it builds the payload, because only
    /// it knows whether a node here is an entity, a file or a bone — but not
    /// the boilerplate: a ghost that looks like the row it came from is the
    /// only sane default, and `H` already knows how to make one.
    ///
    /// Pair it with [`picked_up`](Self::picked_up):
    ///
    /// ```ignore
    /// if let Some(id) = view.picked_up(&ui) {
    ///     view.grab(&mut ui, EntityRef(id), |ui, r| r.set_text(ui, name(id)));
    /// }
    /// ```
    pub fn grab(
        &self,
        ui: &mut UiCore,
        payload: P,
        bind: impl FnOnce(&mut UiCore, Row<'_, H>),
    ) -> NodeId
    where
        P: Send,
    {
        let s = self.style;
        let ghost = ui.grab(payload);
        // Read-modify-write: the ghost's `position` and `inset` are the
        // engine's, and a whole fresh `Style` would put it back in the flow
        // until the next frame re-placed it.
        let mut style = ui.node_style(ghost);
        style.display = Display::Flex;
        style.align_items = Some(AlignItems::CENTER);
        style.padding = Rect::length(s.pad_left + 2.0);
        ui.set_node_style(ghost, style);
        let content = H::build(ui, ghost, &s);
        bind(ui, Row::new(ghost, &content, s));
        // After `bind`, so a closure shared with `sync` — one that ends in
        // `set_selected` — cannot overwrite the ghost's own look.
        ui.set_background(
            ghost,
            UiStyle::fill(s.selected).border(s.drop, 1.0).radius(s.radius),
        );
        ghost
    }

    /// The move a released drag asked for, on the one `sync` that follows it.
    ///
    /// Apply it to the real hierarchy and then to the view with
    /// [`Self::moved`] — or ignore it, which is how a caller refuses a move
    /// its own rules forbid.
    pub fn dropped(&self) -> Option<Dropped> {
        self.drop
    }

    /// The node a press on this view has travelled far enough to be dragging,
    /// for as long as nothing is in flight yet.
    ///
    /// **The view reports; the caller grabs.** Only the caller knows what one
    /// of its nodes *is* — an entity, a file, a bone — so only it can build
    /// the payload and dress the ghost. The view reads the result back
    /// through [`TreeDrag`] and needs nothing else.
    ///
    /// Stateless, and deliberately: it stops firing because a grab exists,
    /// not because a flag was set, so a caller that declines to grab keeps
    /// being offered the row rather than losing it to a flag it never saw.
    ///
    /// Note the node is captured *here*, at the threshold. Reading it back at
    /// release would be wrong — the pooled row a press landed on is recycled
    /// by scrolling, and would report whichever index had moved into it.
    pub fn picked_up(&self, ui: &UiCore) -> Option<u64> {
        if ui.ghost().is_some() {
            return None;
        }
        let (i, d) = self.list.dragged(ui)?;
        d.beyond(DRAG_PX).then(|| self.flat[i].id)
    }

    /// Fold this frame's pointer into the drag gesture; returns what the
    /// indicator should show.
    fn update_drag(&mut self, ui: &UiCore) -> Option<DropMark> {
        // A drop is heard by whichever view owns the row under the pointer,
        // which need not be the one the drag started in — so a node dragged
        // out of another panel arrives here through exactly this path.
        self.drop = self
            .list
            .dropped_on::<P>(ui)
            .filter(|(_, p)| p.tree() == self.tree)
            .and_then(|(_, p)| self.aim(ui, p.node()))
            .map(|(_, d)| d);

        let node = ui.dragging::<P>().filter(|p| p.tree() == self.tree)?.node();
        self.aim(ui, node).map(|(m, _)| m)
    }

    /// Where the pointer is aiming: what to draw, and the move it would make.
    ///
    /// `None` when the pointer is off the list, or over the dragged subtree
    /// itself — a node cannot become its own descendant, and the view can say
    /// so from the flat list alone.
    ///
    /// `from` is `None` for a node this view cannot see: one dragged in from
    /// another panel, or one whose row is collapsed out of sight. There is
    /// then no visible run to protect, and no sibling of the target to
    /// discount. A cycle that only an invisible ancestor could cause is the
    /// caller's to reject — `set_parent_at` already panics on one.
    fn aim(&self, ui: &UiCore, node: u64) -> Option<(DropMark, Dropped)> {
        let (i, frac) = self.list.hovered_at(ui)?;
        let from = self.flat.iter().position(|f| f.id == node);
        if from.is_some_and(|f| (f..f + self.run_len(f)).contains(&i)) {
            return None;
        }

        let t = self.flat[i];
        if (EDGE..1.0 - EDGE).contains(&frac) {
            // First child rather than last: the view knows a collapsed node's
            // child count only by asking, and `at = 0` needs no closure.
            return Some((DropMark::Onto(i), Dropped { node, parent: t.id, at: 0 }));
        }

        // A sibling drop needs the target's parent, which preorder gives for
        // free: the nearest earlier row shallower than it. The root has none,
        // so it takes children but never siblings.
        let pi = self.flat[..i].iter().rposition(|f| f.depth < t.depth)?;
        let after = frac >= 1.0 - EDGE;
        let mut at = usize::from(after);
        for (k, f) in self.flat[pi + 1..i].iter().enumerate() {
            // `node` has left its old parent by the time `at` is applied, so
            // it must not be counted among the target's siblings.
            if f.depth == t.depth && Some(pi + 1 + k) != from {
                at += 1;
            }
        }
        // Below the target's whole visible run, not just its row: dropping
        // "after" an expanded parent lands past its children, and a line
        // tucked under its first child would say otherwise.
        let line = if after { i + self.run_len(i) } else { i };
        Some((DropMark::Line(line), Dropped { node, parent: self.flat[pi].id, at }))
    }

    /// Data index whose disclosure triangle was clicked this frame.
    ///
    /// Distinct from [`clicked`](Self::clicked) with no special case in the
    /// hit test: the arrow is a *child* of the row and accepts `CLICK`, so
    /// innermost-wins toggles on the triangle and selects anywhere else.
    /// Under DOM-style bubbling this would have needed `stopPropagation`.
    fn toggled(&self, ui: &UiCore) -> Option<usize> {
        (0..self.flat.len())
            .find(|&i| self.list.bound_row(i).is_some_and(|r| ui.clicked(r.arrow)))
    }

    /// Node id of the row clicked this frame — never a row index, which
    /// collapsing above it would invalidate.
    pub fn clicked(&self, ui: &UiCore) -> Option<u64> {
        self.list.clicked(ui).map(|i| self.flat[i].id)
    }

    pub fn hovered(&self, ui: &UiCore) -> Option<u64> {
        self.list.hovered(ui).map(|i| self.flat[i].id)
    }

    /// Node id right-clicked this frame. Selection is the caller's to move or
    /// leave: a menu opened on an unselected row is still about that row.
    pub fn right_clicked(&self, ui: &UiCore) -> Option<u64> {
        self.list.right_clicked(ui).map(|i| self.flat[i].id)
    }

    /// Node id double-clicked this frame. [`clicked`](Self::clicked) fires
    /// too, so selecting on one and renaming on the other compose.
    pub fn double_clicked(&self, ui: &UiCore) -> Option<u64> {
        self.list.double_clicked(ui).map(|i| self.flat[i].id)
    }

    /// The pooled row showing `id` — the same `Row<H>` `bind` is handed,
    /// reachable outside a `sync`, so a caller can reach into one row to
    /// focus what it put there. `None` when the row is not pooled, which a
    /// virtualized list must always allow for.
    pub fn row(&self, id: u64) -> Option<Row<'_, H>> {
        let i = self.flat.iter().position(|f| f.id == id)?;
        let row = self.list.bound_row(i)?;
        Some(Row::new(row.node(), &row.content().app, self.style))
    }

    /// Expand every ancestor of `id` so it becomes visible.
    pub fn reveal(&mut self, id: u64, parent_of: impl Fn(u64) -> Option<u64>) {
        let mut p = parent_of(id);
        while let Some(node) = p {
            self.expanded.insert(node);
            p = parent_of(node);
        }
        self.dirty = true;
    }

    /// Expand or collapse `id` in place. A no-op if it is not visible — an
    /// invisible row has no run in the flat list to splice.
    pub fn set_expanded(
        &mut self,
        id: u64,
        want: bool,
        children: &mut impl FnMut(u64, &mut Vec<u64>),
    ) {
        let Some(i) = self.flat.iter().position(|f| f.id == id) else {
            return;
        };
        if self.expanded.contains(&id) != want {
            self.toggle_row(i, children);
        }
    }


    /// Move a subtree: `drain` its run, shift its depths, `splice` it back.
    ///
    /// `at` is the destination index among `new_parent`'s children. This
    /// updates only the view — the caller re-parents the real hierarchy, which
    /// is where the data lives.
    ///
    /// Takes no structure closure, and that is the point: the moved run's
    /// internal shape and expansion are unchanged, so nothing has to be read
    /// back. Only its depth shifts, uniformly.
    pub fn moved(&mut self, id: u64, new_parent: u64, at: usize) {
        let (Some(from), Some(pi)) = (
            self.flat.iter().position(|f| f.id == id),
            self.flat.iter().position(|f| f.id == new_parent),
        ) else {
            self.dirty = true; // destination not visible: nothing to splice into
            return;
        };

        let run = self.run_len(from);
        self.scratch.clear();
        self.scratch.extend(self.flat.drain(from..from + run));

        // The subtree's internal shape is unchanged; only its depth shifts,
        // which is what makes a move `O(K)` instead of a re-walk.
        let pi = if pi > from { pi - run } else { pi };
        let delta = self.flat[pi].depth as i32 + 1 - self.scratch[0].depth as i32;
        for f in &mut self.scratch {
            f.depth = (f.depth as i32 + delta) as u16;
        }

        // `at`-th child of the new parent, in flat-list terms.
        let mut insert = pi + 1;
        for _ in 0..at {
            if insert >= self.flat.len() || self.flat[insert].depth <= self.flat[pi].depth {
                break;
            }
            insert += self.run_len(insert);
        }
        let moved = std::mem::take(&mut self.scratch);
        self.flat.splice(insert..insert, moved.iter().copied());
        self.scratch = moved;
    }

    /// Length of the run at `i`: the node plus every visible descendant,
    /// which are exactly the following entries deeper than it.
    fn run_len(&self, i: usize) -> usize {
        let d = self.flat[i].depth;
        1 + self.flat[i + 1..].iter().take_while(|f| f.depth > d).count()
    }

    fn toggle_row(&mut self, i: usize, children: &mut impl FnMut(u64, &mut Vec<u64>)) {
        let Flat { id, depth } = self.flat[i];
        if self.expanded.remove(&id) {
            let run = self.run_len(i);
            self.flat.drain(i + 1..i + run);
            return;
        }
        self.expanded.insert(id);
        self.scratch.clear();
        let mut sub = std::mem::take(&mut self.scratch);
        self.walk(id, depth + 1, children, &mut sub);
        self.flat.splice(i + 1..i + 1, sub.iter().copied());
        self.scratch = sub;
    }

    fn rebuild(&mut self, children: &mut impl FnMut(u64, &mut Vec<u64>)) {
        self.dirty = false;
        let mut out = std::mem::take(&mut self.flat);
        out.clear();
        out.push(Flat { id: self.root, depth: 0 });
        self.walk(self.root, 1, children, &mut out);
        self.flat = out;
    }

    /// Append `node`'s expanded descendants in preorder. Recursion depth is
    /// tree depth, not node count.
    fn walk(
        &self,
        node: u64,
        depth: u16,
        children: &mut impl FnMut(u64, &mut Vec<u64>),
        out: &mut Vec<Flat>,
    ) {
        if !self.expanded.contains(&node) {
            return;
        }
        let mut kids = Vec::new();
        children(node, &mut kids);
        for k in kids {
            out.push(Flat { id: k, depth });
            self.walk(k, depth + 1, children, out);
        }
    }
}

/// This view's own per-row work: the indent, and the disclosure triangle.
/// Everything else in the row belongs to the caller's `bind`.
fn bind_arrow<H>(ui: &mut UiCore, h: &TreeRow<H>, s: &RowStyle, depth: u16, open: Option<bool>) {
    ui.set_node_style(h.content, content_style(s, depth));

    // The arrow keeps its box on a leaf so content stays aligned down the
    // column; only its glyph goes away.
    let mut glyph = [0u8; 4];
    ui.set_label(
        h.arrow,
        match open {
            Some(true) => font::ARROW_DOWN.encode_utf8(&mut glyph),
            Some(false) => font::ARROW_RIGHT.encode_utf8(&mut glyph),
            None => "",
        },
    );
    // Clicks only, never drops: the arrow sits inside the row, and a drop
    // aimed at the row must not be caught by its triangle.
    ui.set_events(
        h.arrow,
        match open.is_some() {
            true => Events::CLICK,
            false => Events::NONE,
        },
    );
}

/// Fills the row and carries the indent, so the outer node is free to say
/// nothing but where the row sits.
fn content_style(s: &RowStyle, depth: u16) -> Style {
    Style {
        display: Display::Flex,
        align_items: Some(AlignItems::CENTER),
        size: Size {
            width: super::style::percent(1.0_f32),
            height: super::style::percent(1.0_f32),
        },
        padding: Rect {
            // `pad_left` is already on the row itself, so only the depth is
            // this node's to add.
            left: px(depth as f32 * s.indent),
            right: LengthPercentage::ZERO,
            top: LengthPercentage::ZERO,
            bottom: LengthPercentage::ZERO,
        },
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::super::style::{percent, px, LengthPercentageAuto, Position, Rect, Size, TaffyAuto};
    use super::*;
    use crate::input::{Key, Keystroke, Mods};
    use crate::ui::{TextField, TextFieldStyle};
    use std::collections::HashMap;

    /// A tree the tests own: `id -> children`, so `TreeView` reads live
    /// structure exactly as it would from the transform hierarchy.
    struct Model(HashMap<u64, Vec<u64>>);

    impl Model {
        /// `n` roots, each with `n` children, each with `n` grandchildren.
        fn pyramid(n: u64) -> Self {
            let mut m = HashMap::new();
            let roots: Vec<u64> = (1..=n).collect();
            m.insert(0, roots.clone());
            for &r in &roots {
                let kids: Vec<u64> = (1..=n).map(|i| r * 100 + i).collect();
                for &k in &kids {
                    m.insert(k, (1..=n).map(|i| k * 100 + i).collect());
                }
                m.insert(r, kids);
            }
            Self(m)
        }

        fn children(&self) -> impl FnMut(u64, &mut Vec<u64>) + '_ {
            move |id, out| out.extend(self.0.get(&id).into_iter().flatten().copied())
        }
    }

    fn view(core: &mut UiCore) -> TreeView {
        view_of(core)
    }

    fn view_of<H: RowContent, P: TreeDrag>(core: &mut UiCore) -> TreeView<H, P> {
        let root = core.root();
        let v = TreeView::new(
            core,
            root,
            Style {
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
            },
            RowStyle { row_h: 20.0, ..Default::default() },
            0,
        );
        core.run_layout([400.0, 400.0]);
        v
    }

    /// The tests use the shipped label row, which is also what the editor
    /// uses — so these exercise the same path a caller takes.
    fn bind(ui: &mut UiCore, r: Row<'_, Label>, id: u64) {
        r.set_text(ui, &format!("n{id}"));
    }

    fn ids<H: RowContent>(v: &TreeView<H>) -> Vec<u64> {
        v.visible().collect()
    }

    /// Two passes: the pool sizes itself from the *measured* viewport, so the
    /// first `sync` has no layout to measure and binds nothing.
    fn settle(core: &mut UiCore, v: &mut TreeView, m: &Model) {
        for _ in 0..2 {
            v.sync(core, m.children(), bind);
            core.run_layout([400.0, 400.0]);
        }
    }

    /// One frame: deliver a pointer event, fold it, lay out. Rows are 20 px,
    /// so `y` picks a row and where in it — which is what a drop reads.
    fn frame(core: &mut UiCore, v: &mut TreeView, m: &Model, y: f32, pressed: bool, released: bool) {
        core.update_pointer([150.0, y], pressed, released, 0.0, 0.0);
        // What a caller does: the view offers the row, the caller grabs its
        // own payload. Nothing is in flight until this runs.
        if let Some(id) = v.picked_up(core) {
            v.grab(core, DragNode(id), |ui, r| r.set_text(ui, &format!("n{id}")));
        }
        v.sync(core, m.children(), bind);
        core.run_layout([400.0, 400.0]);
    }

    /// The ghost is the engine's to position. A caller that dressed it with a
    /// whole fresh `Style` used to drop it back into the flow for one frame,
    /// which shoved every panel aside.
    #[test]
    fn picking_a_row_up_does_not_move_the_interface() {
        let mut core = UiCore::new();
        let root = core.root();
        // Full-width, like the dock the editor puts here: an extra item in
        // the flow squeezes it, which is exactly what the bug looked like.
        let sibling = core.node(
            root,
            Style {
                size: Size {
                    width: percent(1.0_f32),
                    height: px(50.0),
                },
                ..Default::default()
            },
        );
        let mut v = view(&mut core);
        let m = Model::pyramid(3);
        settle(&mut core, &mut v, &m);
        let before = core.node_rect(sibling);

        frame(&mut core, &mut v, &m, 30.0, true, false);
        frame(&mut core, &mut v, &m, 70.0, false, false);
        assert!(core.ghost().is_some(), "the row was picked up");
        assert_eq!(core.node_rect(sibling), before, "and stayed out of flow");
    }

    fn drag(core: &mut UiCore, v: &mut TreeView, m: &Model, from: f32, to: f32) {
        frame(core, v, m, from, true, false);
        frame(core, v, m, to, false, false);
        frame(core, v, m, to, false, true);
    }

    /// The headline gesture: drop a row on the *body* of another and it
    /// becomes that node's child.
    #[test]
    fn dragging_a_row_onto_another_reparents_it() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        settle(&mut core, &mut v, &m);
        assert_eq!(ids(&v), vec![0, 1, 2, 3]);

        // Row 1 (node 1) → the middle of row 3 (node 3).
        drag(&mut core, &mut v, &m, 30.0, 70.0);
        assert_eq!(v.dropped(), Some(Dropped { node: 1, parent: 3, at: 0 }));
    }

    /// You cannot aim a drop you cannot see the source of. The ghost appears
    /// only once the press becomes a drag, carries the row's own label, and
    /// goes away on release.
    #[test]
    fn a_drag_carries_a_ghost_of_the_row() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        settle(&mut core, &mut v, &m);

        frame(&mut core, &mut v, &m, 30.0, true, false);
        assert_eq!(core.ghost(), None, "a press alone is not a drag");

        frame(&mut core, &mut v, &m, 70.0, false, false);
        assert_eq!(core.dragging(), Some(&DragNode(1)), "row 1 is node 1");
        let ghost = core.ghost().expect("a drag shows what it carries");
        assert!(core.node_rect(ghost)[3] > 0.0);

        frame(&mut core, &mut v, &m, 70.0, false, true);
        assert_eq!(core.ghost(), None, "dropped, so nothing is held");
    }

    /// The disclosure arrow takes clicks and declines drops, so a drop aimed
    /// at an expandable row lands on the row even when the pointer is on its
    /// triangle. `RowList` used to name the arrow explicitly to recover this;
    /// now the arrow declares it, which is what makes a row of arbitrary
    /// caller-built widgets work the same way.
    #[test]
    fn a_drop_on_a_rows_arrow_lands_on_the_row() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        settle(&mut core, &mut v, &m);

        // Node 3 sits at depth 1, so its content starts `pad_left + indent`
        // in and the 12px arrow column runs 16..28.
        let arrow = [22.0, 70.0];
        let r = v.list.bound_row(3).expect("row 3 bound");
        let (row_node, arrow_node) = (r.node(), r.arrow);
        core.update_pointer(arrow, false, false, 0.0, 0.0);
        assert_eq!(core.hit_test(arrow), Some(arrow_node.into()), "the arrow takes clicks");
        assert!(core.hovered(row_node), "and the row is still the hovered one");

        core.grab(DragNode(1));
        core.update_pointer(arrow, false, true, 0.0, 0.0);
        v.sync(&mut core, m.children(), bind);
        assert_eq!(v.dropped(), Some(Dropped { node: 1, parent: 3, at: 0 }));
    }

    /// A real drag lasts many frames. `picked_up` must offer the row exactly
    /// once — a second offer would have the caller call `grab` while one is
    /// in flight, which panics. The guard is a live grab rather than a flag,
    /// so this is what proves it.
    #[test]
    fn a_sustained_drag_picks_up_only_once() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        settle(&mut core, &mut v, &m);

        frame(&mut core, &mut v, &m, 30.0, true, false);
        let mut offers = 0;
        for y in [50.0, 55.0, 60.0, 65.0, 70.0] {
            core.update_pointer([150.0, y], false, false, 0.0, 0.0);
            if let Some(id) = v.picked_up(&core) {
                offers += 1;
                core.grab(DragNode(id));
            }
            v.sync(&mut core, m.children(), bind);
            core.run_layout([400.0, 400.0]);
        }
        assert_eq!(offers, 1, "one gesture, one pick-up");
        assert_eq!(core.dragging(), Some(&DragNode(1)), "and it is still node 1");
    }

    /// `bind` is handed the **node id**, not the row index, and what it writes
    /// is what reaches the widget tree. Collapsing changes indices and not
    /// identities, so a caller keyed on the index would show the wrong name
    /// the moment anything above it opened.
    #[test]
    fn bind_receives_node_ids_and_its_writes_reach_the_rows() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        settle(&mut core, &mut v, &m);
        v.set_expanded(1, true, &mut m.children());
        settle(&mut core, &mut v, &m);
        assert_eq!(ids(&v), vec![0, 1, 101, 102, 103, 2, 3]);

        // Row index 2 is node 101, and index and id have diverged.
        for (i, id) in [(0u64, 0u64), (1, 1), (2, 101), (3, 102)] {
            let r = v.list.bound_row(i as usize).expect("row bound");
            assert_eq!(
                core.node_text(r.app),
                Some(format!("n{id}").as_str()),
                "row {i} should show node {id}"
            );
        }
    }

    /// The view reads drags as the caller's own type, not a type it dictates.
    /// A payload of the wrong kind is declined without being inspected —
    /// which is what lets a hierarchy ignore a material dropped on it while
    /// still accepting an entity.
    #[test]
    fn a_caller_supplies_its_own_payload_type() {
        #[derive(Clone, Copy)]
        struct EntityRef(u64);
        impl TreeDrag for EntityRef {
            fn node(&self) -> u64 {
                self.0
            }
        }
        #[derive(Clone, Copy)]
        struct MaterialRef(u64);

        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let root = core.root();
        let mut v: TreeView<Label, EntityRef> = TreeView::new(
            &mut core,
            root,
            Style {
                position: Position::Absolute,
                inset: Rect {
                    left: px(0.0),
                    top: px(0.0),
                    right: LengthPercentageAuto::AUTO,
                    bottom: LengthPercentageAuto::AUTO,
                },
                size: Size { width: px(200.0), height: px(100.0) },
                ..Default::default()
            },
            RowStyle { row_h: 20.0, ..Default::default() },
            0,
        );
        core.run_layout([400.0, 400.0]);
        for _ in 0..2 {
            v.sync(&mut core, m.children(), bind);
            core.run_layout([400.0, 400.0]);
        }

        // Something this tree does not deal in, released on row 3.
        core.update_pointer([150.0, 70.0], false, false, 0.0, 0.0);
        core.grab(MaterialRef(5));
        core.update_pointer([150.0, 70.0], false, true, 0.0, 0.0);
        v.sync(&mut core, m.children(), bind);
        assert_eq!(v.dropped(), None, "a material is not a move of the tree");

        // The caller's own ref resolves through `TreeDrag` and does.
        core.update_pointer([150.0, 30.0], false, false, 0.0, 0.0);
        core.grab(EntityRef(1));
        core.update_pointer([150.0, 70.0], false, true, 0.0, 0.0);
        v.sync(&mut core, m.children(), bind);
        assert_eq!(v.dropped(), Some(Dropped { node: 1, parent: 3, at: 0 }));
    }

    /// A payload from another model, which is what two hierarchy panels over
    /// two documents put in flight: slot 1 of one is a different entity in the
    /// next, so a view that resolved it would re-parent the wrong thing.
    #[derive(Clone, Copy)]
    struct Tagged(u64, u64);

    impl TreeDrag for Tagged {
        fn node(&self) -> u64 {
            self.1
        }

        fn tree(&self) -> u64 {
            self.0
        }
    }

    /// The tag is what confines a drag to the model it names, and nothing
    /// else about the drop changes: the same gesture lands when the tags
    /// agree.
    #[test]
    fn a_drag_tagged_for_another_tree_lands_nowhere() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v: TreeView<Label, Tagged> = view_of(&mut core).with_tree(7);
        for _ in 0..2 {
            v.sync(&mut core, m.children(), bind);
            core.run_layout([400.0, 400.0]);
        }

        for (tag, want) in [(9, None), (7, Some(Dropped { node: 1, parent: 3, at: 0 }))] {
            core.update_pointer([150.0, 70.0], false, false, 0.0, 0.0);
            core.grab(Tagged(tag, 1));
            core.update_pointer([150.0, 70.0], false, true, 0.0, 0.0);
            v.sync(&mut core, m.children(), bind);
            core.run_layout([400.0, 400.0]);
            assert_eq!(v.dropped(), want, "a drag tagged {tag} on tree 7");
        }
    }

    /// A node this view has never heard of, dropped onto one of its rows.
    /// Nothing about the gesture came from here — which is what makes the
    /// hierarchy a drop target for the rest of the editor rather than only
    /// for itself.
    #[test]
    fn a_node_dragged_in_from_outside_lands_like_any_other() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        settle(&mut core, &mut v, &m);

        core.update_pointer([150.0, 70.0], false, false, 0.0, 0.0);
        core.grab(DragNode(999));
        frame(&mut core, &mut v, &m, 70.0, false, true);
        assert_eq!(v.dropped(), Some(Dropped { node: 999, parent: 3, at: 0 }));
    }

    /// A press that never travelled is a click, not a drop — otherwise every
    /// selection in the panel would re-parent something.
    #[test]
    fn a_click_is_not_a_drop() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        settle(&mut core, &mut v, &m);

        // Down near the bottom of row 1, up 2 px later on row 2 — far enough
        // to change rows, nowhere near far enough to be a drag. Without the
        // threshold this reads as "make node 1 a sibling before node 2".
        drag(&mut core, &mut v, &m, 39.0, 41.0);
        assert_eq!(v.dropped(), None, "2 px is a wobble, not a drag");

        drag(&mut core, &mut v, &m, 30.0, 30.0);
        assert_eq!(v.dropped(), None);
        assert_eq!(v.clicked(&core), Some(1), "and it still selects");
    }

    /// Dropping near a row's edge places a *sibling*. `at` is the index after
    /// the dragged node has left its old parent, which is exactly what
    /// `moved` consumes — so the round trip is the assertion.
    #[test]
    fn dropping_on_an_edge_places_a_sibling() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        settle(&mut core, &mut v, &m);

        // Row 1 (node 1) → the top edge of row 3 (node 3).
        drag(&mut core, &mut v, &m, 30.0, 62.0);
        let d = v.dropped().expect("a drop");
        assert_eq!(d, Dropped { node: 1, parent: 0, at: 1 });

        v.moved(d.node, d.parent, d.at);
        assert_eq!(ids(&v), vec![0, 2, 1, 3], "landed before node 3, not at index 1 of [1,2,3]");
    }

    /// "After" an expanded parent means after its whole subtree, and the
    /// indicator has to say so — a line tucked under its first child reads as
    /// "first child" instead.
    #[test]
    fn dropping_after_an_expanded_row_clears_its_subtree() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        settle(&mut core, &mut v, &m);
        v.set_expanded(1, true, &mut m.children());
        settle(&mut core, &mut v, &m);
        assert_eq!(ids(&v), vec![0, 1, 101, 102, 103, 2, 3]);

        // Row 2 (node 101) → the bottom edge of row 1 (node 1).
        frame(&mut core, &mut v, &m, 50.0, true, false);
        frame(&mut core, &mut v, &m, 38.0, false, false);
        assert_eq!(
            v.aim(&core, 101).map(|(mark, _)| mark),
            Some(DropMark::Line(5)),
            "below node 1's three children, not between it and the first"
        );

        frame(&mut core, &mut v, &m, 38.0, false, true);
        assert_eq!(v.dropped(), Some(Dropped { node: 101, parent: 0, at: 1 }));
    }

    /// A node cannot become its own descendant, and the flat list is enough
    /// to know it — the run under the dragged row *is* its subtree.
    #[test]
    fn a_subtree_cannot_be_dropped_inside_itself() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        settle(&mut core, &mut v, &m);
        v.set_expanded(1, true, &mut m.children());
        settle(&mut core, &mut v, &m);

        // Row 1 (node 1) → row 3 (node 102), one of its own children.
        drag(&mut core, &mut v, &m, 30.0, 70.0);
        assert_eq!(v.dropped(), None);
    }

    /// The root takes children but has no siblings to be placed among, so an
    /// edge drop on it is refused rather than silently treated as a child.
    #[test]
    fn the_root_takes_children_but_not_siblings() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        settle(&mut core, &mut v, &m);

        drag(&mut core, &mut v, &m, 30.0, 2.0);
        assert_eq!(v.dropped(), None, "nothing can be the root's sibling");

        drag(&mut core, &mut v, &m, 30.0, 10.0);
        assert_eq!(v.dropped(), Some(Dropped { node: 1, parent: 0, at: 0 }));
    }

    /// The root opens; its descendants do not. That is what keeps the flatten
    /// proportional to what the user opened rather than to the scene, without
    /// a panel that shows nothing on load.
    #[test]
    fn opens_at_the_root_and_no_deeper() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        v.sync(&mut core, m.children(), bind);
        assert_eq!(ids(&v), vec![0, 1, 2, 3], "top level visible, nothing below it");
    }

    /// Expanding splices in exactly that node's children; collapsing removes
    /// the whole contiguous run including grandchildren.
    #[test]
    fn expand_and_collapse_splice_the_subtree_run() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        v.sync(&mut core, m.children(), bind);
        assert_eq!(ids(&v), vec![0, 1, 2, 3]);

        v.set_expanded(2, true, &mut m.children());
        assert_eq!(ids(&v), vec![0, 1, 2, 201, 202, 203, 3], "children land under their parent");

        v.set_expanded(202, true, &mut m.children());
        assert_eq!(ids(&v), vec![0, 1, 2, 201, 202, 20201, 20202, 20203, 203, 3]);

        // Collapsing 2 must take its grandchildren with it — one contiguous
        // run — and leave everything outside it untouched.
        v.set_expanded(2, false, &mut m.children());
        assert_eq!(ids(&v), vec![0, 1, 2, 3]);

        // Expansion of the inner node is remembered, so re-opening restores
        // the shape rather than the first level only.
        v.set_expanded(2, true, &mut m.children());
        assert_eq!(ids(&v), vec![0, 1, 2, 201, 202, 20201, 20202, 20203, 203, 3]);
    }

    /// Depth is what drives indentation, and a move is the one edit that
    /// changes it — uniformly, across the whole run.
    #[test]
    fn depths_track_the_tree() {
        let mut core = UiCore::new();
        let m = Model::pyramid(2);
        let mut v = view(&mut core);
        v.sync(&mut core, m.children(), bind);
        v.set_expanded(0, true, &mut m.children());
        v.set_expanded(1, true, &mut m.children());
        v.set_expanded(101, true, &mut m.children());

        let depths: Vec<u16> = v.flat.iter().map(|f| f.depth).collect();
        assert_eq!(ids(&v), vec![0, 1, 101, 10101, 10102, 102, 2]);
        assert_eq!(depths, vec![0, 1, 2, 3, 3, 2, 1]);
    }

    /// Re-parenting moves the run and shifts its depths uniformly, without
    /// disturbing the subtree's internal shape or its expansion.
    #[test]
    fn moving_a_subtree_shifts_its_depths_and_keeps_its_shape() {
        let mut core = UiCore::new();
        let m = Model::pyramid(2);
        let mut v = view(&mut core);
        v.sync(&mut core, m.children(), bind);
        v.set_expanded(0, true, &mut m.children());
        v.set_expanded(1, true, &mut m.children());
        v.set_expanded(101, true, &mut m.children());
        assert_eq!(ids(&v), vec![0, 1, 101, 10101, 10102, 102, 2]);

        // Move 101 (with its two expanded children) under 2.
        v.moved(101, 2, 0);
        assert_eq!(ids(&v), vec![0, 1, 102, 2, 101, 10101, 10102]);
        let depths: Vec<u16> = v.flat.iter().map(|f| f.depth).collect();
        assert_eq!(depths, vec![0, 1, 2, 1, 2, 3, 3], "run re-based one below its new parent");
    }

    /// Reordering within a parent is a pure permutation: same depths, same
    /// membership.
    #[test]
    fn reordering_siblings_preserves_depths() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        v.sync(&mut core, m.children(), bind);
        v.set_expanded(0, true, &mut m.children());
        assert_eq!(ids(&v), vec![0, 1, 2, 3]);

        v.moved(1, 0, 2);
        assert_eq!(ids(&v), vec![0, 2, 3, 1]);
        assert!(v.flat[1..].iter().all(|f| f.depth == 1), "siblings stay siblings");
    }

    /// The arrow is only offered where there is something to open, and it is
    /// a child node — so it wins the hit over the row without a special case.
    #[test]
    fn only_parents_get_a_disclosure_arrow() {
        let mut core = UiCore::new();
        let m = Model::pyramid(2);
        let mut v = view(&mut core);
        v.sync(&mut core, m.children(), bind);
        v.set_expanded(0, true, &mut m.children());
        v.set_expanded(1, true, &mut m.children());
        v.set_expanded(101, true, &mut m.children());
        core.run_layout([400.0, 400.0]);

        let seen: Vec<Option<bool>> = {
            let mut out = Vec::new();
            let mut kids = Vec::new();
            let mut c = m.children();
            for f in &v.flat {
                kids.clear();
                c(f.id, &mut kids);
                out.push((!kids.is_empty()).then(|| v.is_expanded(f.id)));
            }
            out
        };
        // 0, 1, 101 expanded; 10101/10102 are leaves; 102 and 2 are closed
        // parents.
        assert_eq!(
            seen,
            vec![
                Some(true),
                Some(true),
                Some(true),
                None,
                None,
                Some(false),
                Some(false)
            ]
        );
    }

    /// End-to-end: what reaches the widget tree, not what the view believes.
    /// A collapsed parent must actually carry `▸`, an expanded one `▾`, and a
    /// leaf nothing — the three states the panel is read by.
    #[test]
    fn arrow_glyphs_reach_the_row_nodes() {
        use super::super::font::{ARROW_DOWN, ARROW_RIGHT};

        let mut core = UiCore::new();
        let m = Model::pyramid(2);
        let mut v = view(&mut core);
        v.sync(&mut core, m.children(), bind);
        core.run_layout([400.0, 400.0]);
        v.sync(&mut core, m.children(), bind);
        core.run_layout([400.0, 400.0]);

        // flat: [0 (open), 1 (closed parent), 2 (closed parent)]
        assert_eq!(ids(&v), vec![0, 1, 2]);
        let arrow_of = |core: &UiCore, v: &TreeView, i: usize| {
            let arrow = v.list.bound_row(i).expect("row bound").arrow;
            core.node_text(arrow).unwrap_or("").to_string()
        };
        assert_eq!(arrow_of(&core, &v, 0), ARROW_DOWN.to_string(), "root is open");
        assert_eq!(arrow_of(&core, &v, 1), ARROW_RIGHT.to_string(), "closed parent");

        // A leaf shows nothing: open two levels so a bottom node is visible.
        v.set_expanded(1, true, &mut m.children());
        v.set_expanded(101, true, &mut m.children());
        v.sync(&mut core, m.children(), bind);
        core.run_layout([400.0, 400.0]);
        v.sync(&mut core, m.children(), bind);
        assert_eq!(ids(&v), vec![0, 1, 101, 10101, 10102, 102, 2]);
        assert_eq!(arrow_of(&core, &v, 2), ARROW_DOWN.to_string(), "101 is now open");
        assert_eq!(arrow_of(&core, &v, 3), "", "10101 is a leaf");
    }

    /// The editor's shape: a shallow root whose last child has a wide
    /// fan-out, expanded after the pool has already sized itself to the
    /// small list. Growing the pool mid-`sync` and splicing 142 rows in one
    /// go is what a real scene does the moment a GLB subscene is opened.
    #[test]
    fn expanding_a_wide_child_after_the_pool_settled() {
        let mut core = UiCore::new();
        let mut m = HashMap::new();
        m.insert(0u64, vec![1u64, 2, 3, 4]);
        m.insert(4, (100..242).collect::<Vec<u64>>());
        let m = Model(m);

        let mut v = view(&mut core);
        for _ in 0..3 {
            v.sync(&mut core, m.children(), bind);
            core.run_layout([400.0, 400.0]);
        }
        assert_eq!(ids(&v), vec![0, 1, 2, 3, 4]);

        v.set_expanded(4, true, &mut m.children());
        v.sync(&mut core, m.children(), bind);
        core.run_layout([400.0, 400.0]);
        v.sync(&mut core, m.children(), bind);
        core.run_layout([400.0, 400.0]);
        assert_eq!(v.flat.len(), 147);

        // Scroll to the bottom and back, which is what a user does next.
        core.scroll_by(v.list.node(), [0.0, f32::MAX]);
        v.sync(&mut core, m.children(), bind);
        core.run_layout([400.0, 400.0]);

        v.set_expanded(4, false, &mut m.children());
        v.sync(&mut core, m.children(), bind);
        core.run_layout([400.0, 400.0]);
        v.sync(&mut core, m.children(), bind);
        assert_eq!(ids(&v), vec![0, 1, 2, 3, 4]);
    }

    /// A row's content is now a *grandchild* of the pooled node, so hiding a
    /// parked row has to reach two levels down. A glyph's quad is sized by
    /// the font rather than by its node, so if it does not, every surplus row
    /// keeps drawing its text stacked at the top of the list.
    #[test]
    fn parked_rows_hide_their_content_through_the_nesting() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        settle(&mut core, &mut v, &m);
        v.set_expanded(1, true, &mut m.children());
        v.set_expanded(2, true, &mut m.children());
        settle(&mut core, &mut v, &m);

        v.set_expanded(1, false, &mut m.children());
        v.set_expanded(2, false, &mut m.children());
        settle(&mut core, &mut v, &m);
        assert_eq!(ids(&v), vec![0, 1, 2, 3]);

        let parked: Vec<TreeRow<Label>> =
            v.list.rows().filter(|(_, b)| b.is_none()).map(|(h, _)| *h).collect();
        assert!(!parked.is_empty(), "the pool must exceed the shrunk tree to test this");
        for h in parked {
            for t in [h.arrow, h.app] {
                let (first, count) = core.run_slots(core.text_id(t).expect("row text"));
                for slot in first..first + count {
                    let q = core.quad.get(slot).rect;
                    assert_eq!([q[2], q[3]], [0.0, 0.0], "a parked row drew a glyph");
                }
            }
        }
    }

    /// Clicking reports a node id: the whole point, since collapsing above a
    /// row changes its index but not its identity.
    #[test]
    fn clicking_reports_a_node_id_not_a_row_index() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        v.sync(&mut core, m.children(), bind);
        v.set_expanded(0, true, &mut m.children());
        v.sync(&mut core, m.children(), bind);
        core.run_layout([400.0, 400.0]);
        v.sync(&mut core, m.children(), bind);
        core.run_layout([400.0, 400.0]);

        // Third row (index 2) is node 2; click well right of the arrow.
        let p = [150.0, 50.0];
        core.update_pointer(p, true, false, 0.0, 0.0);
        core.update_pointer(p, false, true, 0.0, 0.0);
        assert_eq!(v.clicked(&core), Some(2));
    }

    // ── Rename in place ─────────────────────────────────────────────────
    //
    // The editor's hierarchy row, in miniature. Everything below tests the
    // *composition* — double click, a row that swaps which child is in
    // layout, and a field that takes the keyboard — rather than any one of
    // the three, because each already has its own tests and none of them
    // could have caught the seams.

    /// A row that can rename itself: both children built once, one displayed.
    #[derive(Clone, Copy)]
    struct NameRow {
        label: Label,
        field: TextField,
    }

    impl RowContent for NameRow {
        fn build(ui: &mut UiCore, parent: NodeId, s: &RowStyle) -> Self {
            let label = ui.label(parent, s.text_px, s.text, "");
            let field = ui.text_field(
                parent,
                "",
                TextFieldStyle {
                    width: 120.0,
                    text_px: s.text_px,
                    padding: 1.0,
                    ..Default::default()
                },
            );
            let me = NameRow { label, field };
            me.set_editing(ui, false);
            me
        }
    }

    impl NameRow {
        fn set_editing(&self, ui: &mut UiCore, editing: bool) {
            for (n, shown) in [(self.label.node(), !editing), (self.field.node(), editing)] {
                let mut s = ui.node_style(n);
                s.display = match shown {
                    true => Display::Flex,
                    false => Display::None,
                };
                ui.set_node_style(n, s);
            }
        }
    }

    /// The caller's half of the feature, exactly as the editor writes it.
    struct Panel {
        v: TreeView<NameRow>,
        editing: Option<u64>,
        names: HashMap<u64, String>,
    }

    impl Panel {
        fn new(core: &mut UiCore) -> Self {
            Self { v: view_of(core), editing: None, names: HashMap::new() }
        }

        fn name(&self, id: u64) -> String {
            self.names.get(&id).cloned().unwrap_or_else(|| format!("n{id}"))
        }

        /// One frame of the editor's `update`, in its real order.
        fn frame(&mut self, core: &mut UiCore, m: &Model, y: f32, down: bool, up: bool, now: f64) {
            core.update_pointer([150.0, y], down, up, 0.0, now);
            core.update_keyboard(&[]);
            self.settle(core, m);
        }

        /// Fold input into the edit state, re-bind, lay out.
        fn settle(&mut self, core: &mut UiCore, m: &Model) {
            if let Some(id) = self.v.double_clicked(core) {
                self.editing = Some(id);
                let name = self.name(id);
                if let Some(r) = self.v.row(id) {
                    r.field.set_text(core, &name);
                    r.field.focus(core);
                }
            } else if let Some(id) = self.editing {
                match self.v.row(id) {
                    Some(r) if r.field.submitted(core) => {
                        self.names.insert(id, r.field.text(core).to_string());
                        self.editing = None;
                    }
                    Some(r) if !core.focused(r.field) => self.editing = None,
                    None => self.editing = None,
                    _ => {}
                }
            }
            let (editing, names) = (self.editing, &self.names);
            self.v.sync(core, m.children(), |ui, r, id| {
                r.set_editing(ui, editing == Some(id));
                let name = names.get(&id).cloned().unwrap_or_else(|| format!("n{id}"));
                r.label.set_text(ui, &name);
            });
            core.run_layout([400.0, 400.0]);
        }

        /// One frame that types instead of clicking.
        ///
        /// It still calls `update_pointer` first, because that is what clears
        /// last frame's `clicked` — skipping it replays the click that began
        /// the edit and the rename restarts on every frame. The renderer
        /// calls the pair unconditionally for exactly this reason, so a test
        /// that calls only one of them is testing an order that never runs.
        fn keys(&mut self, core: &mut UiCore, m: &Model, strokes: &[Keystroke], y: f32, t: f64) {
            core.update_pointer([150.0, y], false, false, 0.0, t);
            core.update_keyboard(strokes);
            self.settle(core, m);
        }

        /// Double click row at `y`, as two complete clicks 0.1 s apart.
        fn double_click(&mut self, core: &mut UiCore, m: &Model, y: f32, t: f64) {
            self.frame(core, m, y, true, false, t);
            self.frame(core, m, y, false, true, t);
            self.frame(core, m, y, true, false, t + 0.1);
            self.frame(core, m, y, false, true, t + 0.1);
        }
    }

    fn typed(s: &str) -> Keystroke {
        Keystroke::Text(s.to_string())
    }

    /// The whole feature: a double click turns the label into a focused
    /// field, typing replaces the name, Enter commits it and the row goes
    /// back to being a label.
    #[test]
    fn double_clicking_a_row_renames_it_in_place() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut p = Panel::new(&mut core);
        p.settle(&mut core, &m);
        p.settle(&mut core, &m);
        assert_eq!(ids(&p.v), vec![0, 1, 2, 3]);

        // Row 1 is node 1: rows are 20 px, so y = 30 is its middle.
        p.double_click(&mut core, &m, 30.0, 0.0);
        assert_eq!(p.editing, Some(1), "the double click began an edit");
        let row = p.v.row(1).expect("row 1 is pooled");
        assert!(core.focused(row.field), "and the field took the keyboard");
        assert!(core.keyboard_captured());
        assert_eq!(row.field.text(&core), "n1", "seeded from the model");
        assert_eq!(
            core.node_rect(row.label.node())[3],
            0.0,
            "the label is out of layout while the field is in it"
        );
        assert!(core.node_rect(row.field.node())[3] > 0.0);

        // Focus selected the contents, so typing replaces rather than appends.
        p.keys(&mut core, &m, &[typed("hull")], 30.0, 0.5);
        assert_eq!(p.v.row(1).unwrap().field.text(&core), "hull");

        p.keys(&mut core, &m, &[Keystroke::Key(Key::Enter, Mods::NONE)], 30.0, 0.6);
        assert_eq!(p.names.get(&1).map(String::as_str), Some("hull"), "committed");
        assert_eq!(p.editing, None, "and the edit is over");

        let row = p.v.row(1).expect("row 1 is still pooled");
        assert!(core.node_rect(row.label.node())[3] > 0.0, "back to a label");
        assert_eq!(core.node_rect(row.field.node())[3], 0.0);
        assert_eq!(core.node_text(row.label.node()), Some("hull"));
    }

    /// Escape abandons the edit and the model keeps its old name — the field
    /// blurs itself, so the panel only has to notice it lost focus.
    #[test]
    fn escape_abandons_a_rename() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut p = Panel::new(&mut core);
        p.settle(&mut core, &m);
        p.settle(&mut core, &m);

        p.double_click(&mut core, &m, 30.0, 0.0);
        p.keys(&mut core, &m, &[typed("wrong")], 30.0, 0.5);

        p.keys(&mut core, &m, &[Keystroke::Key(Key::Escape, Mods::NONE)], 30.0, 0.6);
        assert_eq!(p.editing, None);
        assert!(p.names.is_empty(), "nothing was committed");
        assert_eq!(core.node_text(p.v.row(1).unwrap().label.node()), Some("n1"));
    }

    /// A single click must not start a rename — it selects. The clock is
    /// what separates them, so this is the test that fails if the double
    /// click threshold is ever read from the wrong end.
    #[test]
    fn two_slow_clicks_do_not_rename() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut p = Panel::new(&mut core);
        p.settle(&mut core, &m);
        p.settle(&mut core, &m);

        for t in [0.0, 1.0] {
            p.frame(&mut core, &m, 30.0, true, false, t);
            p.frame(&mut core, &m, 30.0, false, true, t);
        }
        assert_eq!(p.editing, None);
        assert!(!core.keyboard_captured(), "no field ever took the keyboard");
    }

    /// The pool recycles rows, so an edit that scrolls out of view has to
    /// end — otherwise the field would reappear on whatever row moved into
    /// its slot, renaming the wrong entity.
    #[test]
    fn scrolling_the_edited_row_away_ends_the_edit() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut p = Panel::new(&mut core);
        p.settle(&mut core, &m);
        p.settle(&mut core, &m);
        // Open everything so there is more content than viewport.
        let mut children = m.children();
        for id in [0, 1, 2, 3] {
            p.v.set_expanded(id, true, &mut children);
        }
        p.settle(&mut core, &m);

        p.double_click(&mut core, &m, 30.0, 0.0);
        let edited = p.editing.expect("an edit is in flight");

        core.scroll_by(p.v.node(), [0.0, 400.0]);
        p.keys(&mut core, &m, &[], 30.0, 0.5);
        p.keys(&mut core, &m, &[], 30.0, 0.6);

        assert_eq!(p.editing, None, "the edit ended with its row");
        assert!(!core.keyboard_captured(), "and gave the keyboard back");
        assert!(
            p.names.get(&edited).is_none(),
            "abandoning is not committing"
        );
        // Every pooled row shows a label, not a stray field.
        for id in p.v.visible().collect::<Vec<_>>() {
            if let Some(r) = p.v.row(id) {
                assert_eq!(
                    core.node_rect(r.field.node())[3],
                    0.0,
                    "row {id} still shows a field"
                );
            }
        }
    }
}
