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

use std::collections::HashSet;

use super::list::{DropMark, Row, RowList, RowStyle};
use super::style::Style;
use super::{NodeId, UiCore};

/// Travel that turns a press into a drag rather than a click that wobbled.
const DRAG_PX: f32 = 4.0;

/// Fraction of a row's height at each end that reads as "between rows"
/// rather than "into this one".
const EDGE: f32 = 0.25;

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
#[derive(Clone)]
pub struct TreeView {
    list: RowList,
    root: u64,
    /// Expanded nodes. Default-collapsed keeps the flatten `O(visible)`:
    /// opening a million-entity scene shows its roots, not a million rows.
    expanded: HashSet<u64>,
    flat: Vec<Flat>,
    /// Set by [`Self::invalidate`]; the next [`Self::sync`] re-walks.
    dirty: bool,
    /// Scratch for splices, kept to reuse its capacity.
    scratch: Vec<Flat>,
    /// Node an in-flight drag picked up, captured when the press passed the
    /// threshold. Read back at release it would be wrong: the pooled row the
    /// press landed on is recycled by scrolling, and would then report
    /// whichever data index moved into it.
    drag: Option<u64>,
    /// The drop the release produced, for one [`Self::sync`].
    drop: Option<Dropped>,
}

impl TreeView {
    /// `root` is shown as a row like any other; a hierarchy panel wants it
    /// visible so there is somewhere to drop a node to un-parent it.
    pub fn new(ui: &mut UiCore, parent: NodeId, viewport: Style, style: RowStyle, root: u64) -> Self {
        // The root starts expanded: default-collapsed is about *descendants*
        // — that is what keeps the flatten `O(visible)` — while a panel that
        // opens to one unexpandable row shows nothing at all.
        Self {
            list: RowList::new(ui, parent, viewport, style),
            root,
            expanded: HashSet::from([root]),
            flat: Vec::new(),
            dirty: true,
            scratch: Vec::new(),
            drag: None,
            drop: None,
        }
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
    /// `children` pushes a node's children in order; `row` supplies the
    /// appearance of one node. Both are called only for what is needed —
    /// `children` for expanded subtrees on a structural edit, `row` for the
    /// handful of pooled rows every frame.
    pub fn sync<'a>(
        &mut self,
        ui: &mut UiCore,
        mut children: impl FnMut(u64, &mut Vec<u64>),
        mut row: impl FnMut(u64) -> Row<'a>,
    ) {
        if let Some(i) = self.list.toggled(ui) {
            self.toggle_row(i, &mut children);
        }
        if self.dirty {
            self.rebuild(&mut children);
        }
        // After the structure settles — the aim is resolved against the flat
        // list this frame will actually draw.
        let mark = self.update_drag(ui);
        // Pulled through the same closure the rows use, so the ghost carries
        // whatever the caller calls that node — and stays correct across a
        // rename mid-drag.
        let ghost = self.drag.map(|id| row(id).text);

        let (flat, expanded) = (&self.flat, &self.expanded);
        let mut kids = Vec::new();
        self.list.sync(ui, flat.len(), |i| {
            let f = flat[i];
            kids.clear();
            children(f.id, &mut kids);
            Row {
                depth: f.depth,
                expanded: (!kids.is_empty()).then(|| expanded.contains(&f.id)),
                ..row(f.id)
            }
        });
        // Last, so the marker lands on a pool that has finished growing.
        self.list.set_drop_mark(ui, mark);
        self.list.set_drag_ghost(ui, ghost.as_deref());
    }

    /// The move a released drag asked for, on the one `sync` that follows it.
    ///
    /// Apply it to the real hierarchy and then to the view with
    /// [`Self::moved`] — or ignore it, which is how a caller refuses a move
    /// its own rules forbid.
    pub fn dropped(&self) -> Option<Dropped> {
        self.drop
    }

    /// Fold this frame's pointer into the drag gesture; returns what the
    /// indicator should show.
    fn update_drag(&mut self, ui: &UiCore) -> Option<DropMark> {
        self.drop = None;
        let Some((i, d)) = self.list.dragged(ui) else {
            // The press ended. `take` runs either way, so a frame that misses
            // the release — a hidden panel, a skipped `sync` — drops the
            // gesture rather than committing it at a stale position later.
            let (held, released) = (self.drag.take(), self.list.dropped(ui));
            if let (Some(id), Some(_)) = (held, released) {
                self.drop = self.aim(ui, id).map(|(_, d)| d);
            }
            return None;
        };
        if d.beyond(DRAG_PX) {
            self.drag.get_or_insert(self.flat[i].id);
        }
        self.drag.and_then(|id| self.aim(ui, id)).map(|(m, _)| m)
    }

    /// Where the pointer is aiming: what to draw, and the move it would make.
    ///
    /// `None` when the pointer is off the list, or over the dragged subtree
    /// itself — a node cannot become its own descendant, and the view can say
    /// so from the flat list alone.
    fn aim(&self, ui: &UiCore, node: u64) -> Option<(DropMark, Dropped)> {
        let (i, frac) = self.list.hovered_at(ui)?;
        let from = self.flat.iter().position(|f| f.id == node)?;
        if (from..from + self.run_len(from)).contains(&i) {
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
            if f.depth == t.depth && pi + 1 + k != from {
                at += 1;
            }
        }
        // Below the target's whole visible run, not just its row: dropping
        // "after" an expanded parent lands past its children, and a line
        // tucked under its first child would say otherwise.
        let line = if after { i + self.run_len(i) } else { i };
        Some((DropMark::Line(line), Dropped { node, parent: self.flat[pi].id, at }))
    }

    /// Node id of the row clicked this frame — never a row index, which
    /// collapsing above it would invalidate.
    pub fn clicked(&self, ui: &UiCore) -> Option<u64> {
        self.list.clicked(ui).map(|i| self.flat[i].id)
    }

    pub fn hovered(&self, ui: &UiCore) -> Option<u64> {
        self.list.hovered(ui).map(|i| self.flat[i].id)
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

#[cfg(test)]
mod tests {
    use super::super::style::{px, LengthPercentageAuto, Position, Rect, Size, TaffyAuto};
    use super::*;
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

    fn label(id: u64) -> Row<'static> {
        Row {
            text: format!("n{id}").into(),
            depth: 0,
            selected: false,
            expanded: None,
        }
    }

    fn ids(v: &TreeView) -> Vec<u64> {
        v.visible().collect()
    }

    /// Two passes: the pool sizes itself from the *measured* viewport, so the
    /// first `sync` has no layout to measure and binds nothing.
    fn settle(core: &mut UiCore, v: &mut TreeView, m: &Model) {
        for _ in 0..2 {
            v.sync(core, m.children(), label);
            core.run_layout([400.0, 400.0]);
        }
    }

    /// One frame: deliver a pointer event, fold it, lay out. Rows are 20 px,
    /// so `y` picks a row and where in it — which is what a drop reads.
    fn frame(core: &mut UiCore, v: &mut TreeView, m: &Model, y: f32, pressed: bool, released: bool) {
        core.update_pointer([150.0, y], pressed, released, 0.0);
        v.sync(core, m.children(), label);
        core.run_layout([400.0, 400.0]);
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
        let (node, label) = v.list.ghost();

        frame(&mut core, &mut v, &m, 30.0, true, false);
        assert_eq!(core.node_rect(node)[3], 0.0, "a press alone is not a drag");

        frame(&mut core, &mut v, &m, 70.0, false, false);
        assert_eq!(core.node_text(label), Some("n1"), "row 1 is node 1");
        assert!(core.node_rect(node)[3] > 0.0);

        frame(&mut core, &mut v, &m, 70.0, false, true);
        assert_eq!(core.node_rect(node)[3], 0.0, "dropped, so nothing is held");
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
        v.sync(&mut core, m.children(), label);
        assert_eq!(ids(&v), vec![0, 1, 2, 3], "top level visible, nothing below it");
    }

    /// Expanding splices in exactly that node's children; collapsing removes
    /// the whole contiguous run including grandchildren.
    #[test]
    fn expand_and_collapse_splice_the_subtree_run() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        v.sync(&mut core, m.children(), label);
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
        v.sync(&mut core, m.children(), label);
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
        v.sync(&mut core, m.children(), label);
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
        v.sync(&mut core, m.children(), label);
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
        v.sync(&mut core, m.children(), label);
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
        v.sync(&mut core, m.children(), label);
        core.run_layout([400.0, 400.0]);
        v.sync(&mut core, m.children(), label);
        core.run_layout([400.0, 400.0]);

        // flat: [0 (open), 1 (closed parent), 2 (closed parent)]
        assert_eq!(ids(&v), vec![0, 1, 2]);
        let arrow_of = |core: &UiCore, v: &TreeView, i: usize| {
            let (_, arrow, _) = v.list.bound_row(i).expect("row bound");
            core.node_text(arrow).unwrap_or("").to_string()
        };
        assert_eq!(arrow_of(&core, &v, 0), ARROW_DOWN.to_string(), "root is open");
        assert_eq!(arrow_of(&core, &v, 1), ARROW_RIGHT.to_string(), "closed parent");

        // A leaf shows nothing: open two levels so a bottom node is visible.
        v.set_expanded(1, true, &mut m.children());
        v.set_expanded(101, true, &mut m.children());
        v.sync(&mut core, m.children(), label);
        core.run_layout([400.0, 400.0]);
        v.sync(&mut core, m.children(), label);
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
            v.sync(&mut core, m.children(), label);
            core.run_layout([400.0, 400.0]);
        }
        assert_eq!(ids(&v), vec![0, 1, 2, 3, 4]);

        v.set_expanded(4, true, &mut m.children());
        v.sync(&mut core, m.children(), label);
        core.run_layout([400.0, 400.0]);
        v.sync(&mut core, m.children(), label);
        core.run_layout([400.0, 400.0]);
        assert_eq!(v.flat.len(), 147);

        // Scroll to the bottom and back, which is what a user does next.
        core.scroll_by(v.list.node(), [0.0, f32::MAX]);
        v.sync(&mut core, m.children(), label);
        core.run_layout([400.0, 400.0]);

        v.set_expanded(4, false, &mut m.children());
        v.sync(&mut core, m.children(), label);
        core.run_layout([400.0, 400.0]);
        v.sync(&mut core, m.children(), label);
        assert_eq!(ids(&v), vec![0, 1, 2, 3, 4]);
    }

    /// Clicking reports a node id: the whole point, since collapsing above a
    /// row changes its index but not its identity.
    #[test]
    fn clicking_reports_a_node_id_not_a_row_index() {
        let mut core = UiCore::new();
        let m = Model::pyramid(3);
        let mut v = view(&mut core);
        v.sync(&mut core, m.children(), label);
        v.set_expanded(0, true, &mut m.children());
        v.sync(&mut core, m.children(), label);
        core.run_layout([400.0, 400.0]);
        v.sync(&mut core, m.children(), label);
        core.run_layout([400.0, 400.0]);

        // Third row (index 2) is node 2; click well right of the arrow.
        let p = [150.0, 50.0];
        core.update_pointer(p, true, false, 0.0);
        core.update_pointer(p, false, true, 0.0);
        assert_eq!(v.clicked(&core), Some(2));
    }
}
