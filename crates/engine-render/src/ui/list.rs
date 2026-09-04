//! A virtualized row list — the scene hierarchy panel's substrate.
//!
//! A tree view is drawn as a **flat list of rows**, not a tree of nodes:
//! the caller flattens its hierarchy to `(depth, text)` and indentation is
//! left padding. That is not a shortcut, it is what makes clicking sane —
//! rows are siblings, so the innermost node accepting `Events::CLICK` is
//! already the row the pointer is on and never a parent that happens to
//! contain it.
//!
//! # Virtualization
//!
//! Only enough rows to cover the viewport exist as nodes. A single sizer
//! child gives the scroll area its full `len * row_h` content height, and
//! the rows are absolutely positioned within it.
//!
//! The pool is a **ring**: slot `k` always holds the data index congruent
//! to `k` modulo the pool size. Scrolling one row past a boundary therefore
//! rebinds *exactly one* row — it jumps from one end of the window to the
//! other — while every other row keeps both its position and its text. The
//! obvious alternative (shift all rows, rebind all of them) would make each
//! boundary crossing cost a full pool rewrite instead of a single row.
//!
//! So the two costs are separated: scrolling *within* a row is one
//! [`UiGroup`](super::UiGroup) record and nothing else, and crossing a row
//! boundary adds one row's worth of writes. Neither depends on `len`.
//!
//! # No invalidation protocol
//!
//! [`RowList::sync`] re-binds every pooled row from the caller's data on
//! every call, and that is deliberate: the pool is viewport-sized, and every
//! write it makes lands on `SlotArray::set`. Re-binding unchanged rows is a
//! few dozen comparisons and zero bytes of staging traffic, which is cheaper
//! than any dirty-flag scheme the caller would otherwise have to maintain.
//!
//! # The list owns the row; the caller owns its content
//!
//! What a row *contains* is a [`RowContent`] — a `Copy` handle type that says
//! how to build itself. It is a type parameter rather than a closure, and
//! that split is the whole ergonomic point:
//!
//! | | declared | called |
//! |---|---|---|
//! | [`RowContent::build`] | once, in the list's type | when the pool grows |
//! | `bind` closure | per [`RowList::sync`] | once per pooled row per sync |
//!
//! Construction is a property of the row *type*, so it belongs on the type;
//! only binding varies frame to frame, so only binding is a closure. A caller
//! that wants text rows names [`Label`] and passes one closure.
//!
//! It stays cheap for a reason immediate mode cannot copy: **the dynamism is
//! paid at pool size**. Seven rows are built once and rebound forever, so
//! per-frame cost is identical whatever a row contains.
//!
//! Ownership divides on one line: the list owns the row *node* — its position,
//! its pointer events, and the idle/hover/selected fill that goes with being a
//! row — and everything inside it belongs to the caller.
//!
//! Content is still *pulled*: only pooled rows are ever bound, so a caller
//! caches nothing and a rename is not a structural event — the next `sync`
//! reads the new name and `set_text` dirties exactly the glyphs that differ.

use std::any::Any;
use std::ops::Deref;

use super::style::{
    auto, percent, px, AlignItems, Display, FlexDirection, LengthPercentage, LengthPercentageAuto,
    Position, Rect, Size, Style, TaffyAuto, TaffyZero,
};
use super::tree::Drag;
use super::{rgba, theme, Events, Label, NodeId, StateStyle, Theme, UiCore, UiStyle};

/// Thickness of the between-rows drop line.
const LINE_H: f32 = 2.0;

/// Where to draw the drop indicator, in data indices.
///
/// The list draws it and knows nothing about what a drop *means* — that is
/// [`TreeView`](super::TreeView)'s to decide, because only it knows whether
/// the row under the pointer can accept a child.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DropMark {
    /// A line on the top edge of row `i`; `len` marks the very end.
    Line(usize),
    /// An outline around row `i` — the drop goes *inside* it.
    Onto(usize),
}

/// How a row looks, for both [`RowList`] and [`TreeView`](super::TreeView).
///
/// One struct rather than two, because a row is one thing: the caller sets a
/// look once and every layer that draws part of a row reads the same fields.
/// `indent` and `arrow` are the tree's; a plain list leaves them alone.
///
/// `row_h` is load-bearing: it is what converts a scroll offset into a data
/// index, so rows are a fixed height by construction. Variable heights would
/// need a prefix-sum index.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RowStyle {
    pub row_h: f32,
    /// Left padding on every row, before any tree indent.
    pub pad_left: f32,
    /// Per depth level, in a [`TreeView`](super::TreeView).
    pub indent: f32,
    pub text_px: f32,
    pub text: u32,
    pub text_selected: u32,
    /// Disclosure triangle, in a [`TreeView`](super::TreeView).
    pub arrow: u32,
    /// Rows sit on whatever the list's own background is, so at rest they
    /// draw nothing rather than a surface of their own.
    pub idle: u32,
    pub hover: u32,
    pub selected: u32,
    /// Drop indicator — the line between rows and the outline around one.
    pub drop: u32,
    pub radius: f32,
}

impl From<Theme> for RowStyle {
    fn from(t: Theme) -> Self {
        Self {
            row_h: 18.0,
            pad_left: 4.0,
            indent: 12.0,
            text_px: t.text_px,
            text: t.text,
            text_selected: t.text_strong,
            arrow: t.text_dim,
            idle: rgba(0, 0, 0, 0),
            hover: t.control_hover,
            selected: t.selection,
            drop: t.accent,
            radius: t.radius,
        }
    }
}

impl Default for RowStyle {
    fn default() -> Self {
        theme().into()
    }
}

/// What a row contains.
///
/// Implement it to put arbitrary widgets in a row — a label and a checkbox, a
/// label and a slider. `build` runs only when the pool grows, so an elaborate
/// row costs nothing per frame.
///
/// It is a trait rather than a closure because construction never varies
/// between frames: a `RowList<Label>` says what its rows are in its own type,
/// which is what lets [`RowList::sync`] take one closure instead of two — and
/// what lets a drag ghost be built by the same code that builds a row.
pub trait RowContent: Copy {
    /// Make the row's widgets inside `parent` and return a handle to them.
    fn build(ui: &mut UiCore, parent: NodeId, style: &RowStyle) -> Self;

    /// React to the row being selected. The row's own fill is handled for
    /// you; this is for content that also changes — a label brightening.
    fn set_selected(&self, ui: &mut UiCore, style: &RowStyle, selected: bool) {
        let _ = (ui, style, selected);
    }
}

/// The default row: one line of text.
impl RowContent for Label {
    fn build(ui: &mut UiCore, parent: NodeId, s: &RowStyle) -> Self {
        ui.label(parent, s.text_px, s.text, "")
    }

    fn set_selected(&self, ui: &mut UiCore, s: &RowStyle, selected: bool) {
        self.set_color(ui, if selected { s.text_selected } else { s.text });
    }
}

/// One row, handed to a `bind` closure.
///
/// Derefs to the content, so `r.set_text(ui, ..)` reaches a [`Label`] row
/// directly, while [`set_selected`](Row::set_selected) and [`node`](Row::node)
/// address the row itself. That is the split a caller actually thinks in:
/// *what this row says*, and *what this row is*.
pub struct Row<'a, H> {
    node: NodeId,
    content: &'a H,
    style: RowStyle,
}

impl<'a, H> Row<'a, H> {
    pub(super) fn new(node: NodeId, content: &'a H, style: RowStyle) -> Self {
        Self {
            node,
            content,
            style,
        }
    }

    /// The row's own node — what a click, a hover and a drop all land on.
    pub fn node(&self) -> NodeId {
        self.node
    }

    /// The content, borrowed for as long as the *list* is, not this `Row`.
    /// [`Deref`] borrows `self` instead, which is too short to re-wrap one
    /// row's content in another `Row` — what [`TreeView`](super::TreeView)
    /// does to hide its indent and arrow.
    pub fn content(&self) -> &'a H {
        self.content
    }
}

impl<H: RowContent> Row<'_, H> {
    /// Show this row as selected, or not.
    ///
    /// Selection is a *look*, not state the list keeps: the caller owns which
    /// item is selected — keyed by its own id, since collapsing a tree changes
    /// row indices — and says so here. A selected row keeps its fill through
    /// hover, because losing it under the pointer reads as a bug.
    pub fn set_selected(&self, ui: &mut UiCore, selected: bool) {
        let s = &self.style;
        let base = UiStyle::fill(s.idle).radius(s.radius);
        ui.set_state_style(
            self.node,
            match selected {
                true => StateStyle::fills(base, s.selected, s.selected, s.selected),
                false => StateStyle::fills(base, s.idle, s.hover, s.hover),
            },
        );
        self.content.set_selected(ui, s, selected);
    }
}

impl<H> Deref for Row<'_, H> {
    type Target = H;
    fn deref(&self) -> &H {
        self.content
    }
}

#[derive(Clone)]
struct PooledRow<H> {
    /// The list's own node: absolutely positioned at the data index, and the
    /// one that accepts the pointer. Handed to `build` as a parent.
    node: NodeId,
    /// Whatever `H::build` made inside it.
    content: H,
    /// Data index this row currently shows, or `None` while it is parked
    /// past the end of the data.
    bound: Option<usize>,
}

/// A scrollable list of fixed-height rows backed by a viewport-sized pool.
///
/// The caller owns it — unlike [`UiCore::button`], this widget has state the
/// tree cannot represent (which pooled node shows which data index), and
/// parking that in the store would mean the store growing a per-widget table
/// for one widget.
///
/// `H` is what a row contains, and defaults to a [`Label`].
///
/// `Clone` only because `Component` requires it; a clone refers to the same
/// nodes, exactly as the bare [`NodeId`]s a component already holds do.
#[derive(Clone)]
pub struct RowList<H: RowContent = Label> {
    area: NodeId,
    /// In-flow child whose height is the whole content extent, so taffy's
    /// `scroll_height` — and therefore the scrollbar range — covers rows that
    /// have no nodes.
    sizer: NodeId,
    rows: Vec<PooledRow<H>>,
    /// Drop indicator, drawn inside the scroll area so it scrolls with the
    /// rows it points between.
    mark: NodeId,
    style: RowStyle,
    first: usize,
}

impl<H: RowContent> RowList<H> {
    /// `viewport` supplies the box, which needs a definite height; the
    /// display and flex direction are set here.
    pub fn new(ui: &mut UiCore, parent: NodeId, viewport: Style, style: RowStyle) -> Self {
        let area = ui.scroll_area(
            parent,
            Style {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                ..viewport
            },
        );
        let sizer = ui.node(area, Style::default());
        let mark = ui.node(area, hidden());
        ui.set_background(mark, UiStyle::fill(rgba(0, 0, 0, 0)));

        Self {
            area,
            sizer,
            rows: Vec::new(),
            mark,
            style,
            first: 0,
        }
    }

    /// The scroll area, for styling it or scrolling it programmatically.
    pub fn node(&self) -> NodeId {
        self.area
    }

    pub fn style(&self) -> &RowStyle {
        &self.style
    }

    /// Point the pool at the current scroll position and re-bind it.
    ///
    /// `bind` writes data index `i` into one row, and runs once per pooled
    /// row — a handful, whatever `len` is.
    ///
    /// The pool sizes itself from the *measured* viewport, so the first frame
    /// after construction shows nothing — there is no layout yet to measure.
    pub fn sync(
        &mut self,
        ui: &mut UiCore,
        len: usize,
        mut bind: impl FnMut(&mut UiCore, Row<'_, H>, usize),
    ) {
        let s = self.style;
        ui.set_node_style(
            self.sizer,
            Style {
                size: Size {
                    width: percent(1.0_f32),
                    height: px(len as f32 * s.row_h),
                },
                flex_shrink: 0.0,
                ..Default::default()
            },
        );

        // Re-clamp: collapsing a subtree shrinks the content under a scroll
        // offset that was legal a moment ago, and nothing else would ever pull
        // it back — the viewport would just show blank. A zero delta re-runs
        // `scroll_by`'s clamp against taffy's *authoritative* content size, so
        // viewport padding and borders need no separate reasoning here. It
        // trails the shrink by one frame, since layout runs after `sync`.
        ui.scroll_by(self.area, [0.0, 0.0]);

        let want = ((ui.node_rect(self.area)[3] / s.row_h).ceil() as usize + 1).min(len);
        if self.rows.len() < want {
            while self.rows.len() < want {
                self.push_row(ui);
            }
            // Rows just landed after the marker in the child list, and paint
            // order is child order — so the marker has to climb back over the
            // opaque row fills it exists to point at.
            ui.raise(self.mark);
        }
        let pool = self.rows.len();
        if pool == 0 {
            return;
        }

        self.first = (ui.scroll_offset(self.area)[1] / s.row_h) as usize;
        for k in 0..pool {
            let i = self.first + (k + pool - self.first % pool) % pool;
            let node = self.rows[k].node;
            self.rows[k].bound = (i < len).then_some(i);
            let Some(i) = self.rows[k].bound else {
                ui.set_node_style(node, hidden());
                continue;
            };
            ui.set_node_style(node, row_style(&s, i));
            bind(ui, Row::new(node, &self.rows[k].content, s), i);
        }
    }

    /// Data index of the row clicked this frame. Rows are siblings, so this
    /// is unambiguous — there is no parent row to have swallowed the click.
    pub fn clicked(&self, ui: &UiCore) -> Option<usize> {
        self.rows
            .iter()
            .find(|r| ui.clicked(r.node))
            .and_then(|r| r.bound)
    }

    /// Data index double-clicked this frame — rename, open, drill in.
    /// [`clicked`](Self::clicked) still fires, so select-on-click and
    /// act-on-double-click need no arbitration.
    pub fn double_clicked(&self, ui: &UiCore) -> Option<usize> {
        self.rows
            .iter()
            .find(|r| ui.double_clicked(r.node))
            .and_then(|r| r.bound)
    }

    /// Data index right-clicked this frame — what opens a context menu.
    /// [`clicked`](Self::clicked) does not fire, so the row a menu is about
    /// is not also the row a menu press selected.
    pub fn right_clicked(&self, ui: &UiCore) -> Option<usize> {
        self.rows
            .iter()
            .find(|r| ui.right_clicked(r.node))
            .and_then(|r| r.bound)
    }

    /// Data index under the pointer, for a caller that wants a preview or a
    /// drop target.
    pub fn hovered(&self, ui: &UiCore) -> Option<usize> {
        self.row_at(ui).and_then(|r| r.bound)
    }

    /// Data index under the pointer and how far down that row it sits,
    /// `0.0..=1.0`.
    ///
    /// The fraction is what separates dropping *onto* a row from dropping
    /// *between* two, and only the list knows a row's box — so it reports the
    /// geometry and leaves the thresholds to whoever knows what a drop means.
    pub fn hovered_at(&self, ui: &UiCore) -> Option<(usize, f32)> {
        let r = self.row_at(ui)?;
        // `node_rect` is layout space: the scroll offset lives in the group,
        // not the box, and the pointer is in screen px.
        let top = ui.node_rect(r.node)[1] - ui.scroll_offset(self.area)[1];
        let frac = (ui.pointer.pos[1] - top) / self.style.row_h;
        Some((r.bound?, frac.clamp(0.0, 1.0)))
    }

    /// Data index a press is held on, with the gesture — `Some` for as long
    /// as the button is down, including once the pointer has left the list.
    pub fn dragged(&self, ui: &UiCore) -> Option<(usize, Drag)> {
        self.rows
            .iter()
            .find_map(|r| Some((r.bound?, ui.drag(r.node)?)))
    }

    /// Data index a drag started from, on the one frame it is released.
    ///
    /// Distinct from [`clicked`](Self::clicked), which fires only when the
    /// release lands back on the row it started from — the case a drop is
    /// defined *not* to be.
    pub fn dropped(&self, ui: &UiCore) -> Option<(usize, Drag)> {
        self.rows
            .iter()
            .find_map(|r| Some((r.bound?, ui.dropped(r.node)?)))
    }

    /// Show or hide the drop indicator.
    pub fn set_drop_mark(&mut self, ui: &mut UiCore, mark: Option<DropMark>) {
        let node = self.mark;
        let s = self.style;
        let Some(mark) = mark else {
            return ui.set_node_style(node, hidden());
        };
        let (top, h) = match mark {
            // Centred on the boundary, so it reads as *between* two rows
            // rather than as a lid on the one below.
            DropMark::Line(i) => (i as f32 * s.row_h - LINE_H * 0.5, LINE_H),
            DropMark::Onto(i) => (i as f32 * s.row_h, s.row_h),
        };
        ui.set_node_style(
            node,
            Style {
                position: Position::Absolute,
                inset: Rect {
                    left: px(0.0),
                    right: px(0.0),
                    top: px(top),
                    bottom: LengthPercentageAuto::AUTO,
                },
                size: Size {
                    width: auto(),
                    height: px(h),
                },
                ..Default::default()
            },
        );
        ui.set_background(
            node,
            match mark {
                DropMark::Line(_) => UiStyle::fill(s.drop),
                DropMark::Onto(_) => UiStyle::fill(rgba(0, 0, 0, 0))
                    .border(s.drop, 1.0)
                    .radius(s.radius),
            },
        );
    }

    /// The pooled row the pointer is on.
    ///
    /// No mention of anything *inside* a row: content takes clicks without
    /// taking hover, so the row is still the hovered node when the pointer is
    /// on a control within it.
    fn row_at(&self, ui: &UiCore) -> Option<&PooledRow<H>> {
        self.rows.iter().find(|r| ui.hovered(r.node))
    }

    /// Data index a `T` was dropped on this frame.
    ///
    /// Not restricted to drags this list started — that is the point. Whoever
    /// grabbed the payload may be another panel entirely.
    pub fn dropped_on<'a, T: Any>(&self, ui: &'a UiCore) -> Option<(usize, &'a T)> {
        self.rows
            .iter()
            .find_map(|r| Some((r.bound?, ui.dropped_on(r.node)?)))
    }

    /// Every pooled row as `(content, data index)`. The index is `None` for a
    /// row parked past the end of the data — those still exist as nodes, so a
    /// caller that walks the pool has to expect them.
    pub fn rows(&self) -> impl Iterator<Item = (&H, Option<usize>)> {
        self.rows.iter().map(|r| (&r.content, r.bound))
    }

    /// The pooled row currently showing `index`. Lets a caller reach the
    /// widgets built for one data index — and a test read back what actually
    /// reached the widget tree.
    pub fn bound_row(&self, index: usize) -> Option<Row<'_, H>> {
        self.rows
            .iter()
            .find(|r| r.bound == Some(index))
            .map(|r| Row::new(r.node, &r.content, self.style))
    }

    /// The row node is the list's: it carries the position the ring depends
    /// on, it is what accepts the pointer, and it owns the fill that says
    /// hovered or selected — so `clicked` / `hovered` / `dropped_on` all name
    /// one node whatever the caller put inside it.
    ///
    /// `DROP` lives here rather than on anything the content builds, which is
    /// what lets a control inside a row take clicks and still leave the drop
    /// to the row — see `Events`.
    fn push_row(&mut self, ui: &mut UiCore) {
        let s = self.style;
        let node = ui.node(self.area, Style::default());
        ui.set_events(node, Events::CLICK | Events::HOVER | Events::DROP);
        // Hover for free: a caller that never mentions selection still gets a
        // list that lights up under the pointer.
        let base = UiStyle::fill(s.idle).radius(s.radius);
        ui.set_state_style(node, StateStyle::fills(base, s.idle, s.hover, s.hover));
        let content = H::build(ui, node, &s);
        self.rows.push(PooledRow {
            node,
            content,
            bound: None,
        });
    }
}

/// Out of layout entirely, so the node's own primitives collapse to zero
/// area and are culled — what a parked row and a hidden drop marker both want.
fn hidden() -> Style {
    Style {
        display: Display::None,
        ..Default::default()
    }
}

/// Absolutely positioned at its data index, full width. Position is the *only*
/// thing tying a pooled node to an index, which is what lets the ring reorder
/// rows without moving anything else.
fn row_style(s: &RowStyle, index: usize) -> Style {
    Style {
        display: Display::Flex,
        align_items: Some(AlignItems::CENTER),
        position: Position::Absolute,
        inset: Rect {
            left: px(0.0),
            right: px(0.0),
            top: px(index as f32 * s.row_h),
            bottom: LengthPercentageAuto::AUTO,
        },
        size: Size {
            width: auto(),
            height: px(s.row_h),
        },
        padding: Rect {
            left: px(s.pad_left),
            right: LengthPercentage::ZERO,
            top: LengthPercentage::ZERO,
            bottom: LengthPercentage::ZERO,
        },
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WHITE: u32 = super::super::rgb(255, 255, 255);

    fn style() -> RowStyle {
        RowStyle {
            row_h: 20.0,
            text: WHITE,
            ..Default::default()
        }
    }

    /// Builds its text per call — the case a tree panel hits when it reads a
    /// name from the hierarchy rather than from a cache it maintains. The
    /// equality gate absorbs the repeats.
    fn bind(ui: &mut UiCore, r: Row<'_, Label>, i: usize) {
        r.set_text(ui, &format!("row {i}"));
    }

    /// A ghost the test builds itself — the only way now, since the list
    /// never knew what a row looked like. Shrink-wraps its label, so a
    /// non-zero box proves the content landed.
    fn grab(core: &mut UiCore, text: &str, payload: usize) -> NodeId {
        let ghost = core.grab(payload);
        core.set_background(ghost, UiStyle::fill(WHITE));
        core.label(ghost, 9.0, WHITE, text);
        ghost
    }

    fn viewport() -> Style {
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
        }
    }

    /// 10 000 rows in a 100 px viewport.
    fn list(core: &mut UiCore) -> RowList {
        let root = core.root();
        let l = RowList::new(core, root, viewport(), style());
        core.run_layout([400.0, 400.0]);
        l
    }

    /// The point of a content *type*: a row is whatever the caller declares,
    /// and the list never learns what is in it. A checkbox here takes its own
    /// click while the row still reports hover and takes the drop — which is
    /// the pairing `Events` exists for, arriving at the layer that needed it.
    ///
    /// And the cost is at *pool* size: seven checkboxes are built, not 10 000.
    #[test]
    fn a_row_can_contain_a_control_the_list_knows_nothing_about() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static BUILT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Clone, Copy)]
        struct CheckRow {
            label: Label,
            check: super::super::Checkbox,
        }

        impl RowContent for CheckRow {
            fn build(ui: &mut UiCore, parent: NodeId, s: &RowStyle) -> Self {
                BUILT.fetch_add(1, Ordering::Relaxed);
                Self {
                    check: ui.checkbox(parent, "", Default::default()),
                    label: ui.label(parent, s.text_px, s.text, ""),
                }
            }
        }

        let mut core = UiCore::new();
        let root = core.root();
        let mut l: RowList<CheckRow> = RowList::new(&mut core, root, viewport(), style());
        core.run_layout([400.0, 400.0]);

        for _ in 0..2 {
            l.sync(&mut core, 10_000, |ui, r, i| {
                r.label.set_text(ui, &format!("row {i}"))
            });
            core.run_layout([400.0, 400.0]);
        }
        assert_eq!(
            BUILT.load(Ordering::Relaxed),
            6,
            "one build per pooled row, not per data row"
        );

        // Aim at the checkbox inside row 2. It takes the click; the row is
        // still what a drop and a hover land on.
        let r = l.bound_row(2).expect("row 2 bound");
        let (row, check) = (r.node(), r.check.node());
        let c = core.node_rect(check);
        let p = [c[0] + c[2] * 0.5, c[1] + c[3] * 0.5];
        core.update_pointer(p, false, false, 0.0, 0.0);
        assert_eq!(
            core.hit_test(p),
            Some(check),
            "the checkbox takes the click"
        );
        assert!(core.hovered(row), "the row is still hovered");
        assert_eq!(l.hovered(&core), Some(2));

        core.grab(9usize);
        core.update_pointer(p, false, true, 0.0, 0.0);
        assert_eq!(
            l.dropped_on::<usize>(&core),
            Some((2, &9)),
            "the row took the drop"
        );
    }

    /// The headline property: the node count follows the *viewport*, not the
    /// data. 10 000 rows must not build 10 000 nodes.
    #[test]
    fn pool_is_viewport_sized_not_data_sized() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);

        l.sync(&mut core, 10_000, bind);
        core.run_layout([400.0, 400.0]);
        // 100px viewport / 20px rows, plus one for the partial row.
        assert_eq!(l.rows.len(), 6);
        // ...and the scroll range still covers all 10 000.
        assert_eq!(core.max_scroll(l.area)[1], 10_000.0 * 20.0 - 100.0);
    }

    /// Crossing one row boundary must rebind exactly one row — the ring's
    /// entire reason for existing.
    #[test]
    fn scrolling_one_row_rebinds_one_row() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 10_000, bind);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10_000, bind);

        let before: Vec<_> = l.rows.iter().map(|r| r.bound).collect();
        core.scroll_by(l.area, [0.0, 20.0]);
        l.sync(&mut core, 10_000, bind);
        let after: Vec<_> = l.rows.iter().map(|r| r.bound).collect();

        let moved = before.iter().zip(&after).filter(|(a, b)| a != b).count();
        assert_eq!(moved, 1, "a one-row scroll should recycle one row");
        assert_eq!(
            after[0],
            Some(6),
            "row 0 wrapped to the bottom of the window"
        );
        assert_eq!(after[1], Some(1), "everything else held");
    }

    /// A steady frame — same data, same scroll — must upload nothing at all.
    #[test]
    fn resyncing_unchanged_data_uploads_nothing() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 10_000, bind);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10_000, bind);
        core.run_layout([400.0, 400.0]);

        let (mut stage, mut dirty) = (vec![0u32; 1 << 16], vec![0u32; 1 << 10]);
        let clean = (i64::MAX, -1);
        core.quad.upload(&mut stage, &mut dirty);
        core.style.upload(&mut stage, &mut dirty);

        l.sync(&mut core, 10_000, bind);
        core.run_layout([400.0, 400.0]);
        assert_eq!(core.quad.upload(&mut stage, &mut dirty), clean);
        assert_eq!(core.style.upload(&mut stage, &mut dirty), clean);
    }

    /// Rows are siblings, so a click reports the row itself — the constraint
    /// the flat design exists to satisfy. Indentation must not change that.
    #[test]
    fn clicking_a_row_reports_its_data_index() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 10_000, bind);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10_000, bind);
        core.run_layout([400.0, 400.0]);

        // Third row down, well inside the deepest indent.
        let p = [150.0, 50.0];
        core.update_pointer(p, true, false, 0.0, 0.0);
        core.update_pointer(p, false, true, 0.0, 0.0);
        assert_eq!(l.clicked(&core), Some(2));

        // Scrolled by two rows, the same pixel is a different entity.
        core.scroll_by(l.area, [0.0, 40.0]);
        l.sync(&mut core, 10_000, bind);
        core.run_layout([400.0, 400.0]);
        core.update_pointer(p, true, false, 0.0, 0.0);
        core.update_pointer(p, false, true, 0.0, 0.0);
        assert_eq!(l.clicked(&core), Some(4));
    }

    /// Collapsing a subtree cuts the content out from under a scroll offset
    /// that was legal a moment earlier. Without a re-clamp the viewport shows
    /// blank, since nothing else ever pulls the offset back.
    #[test]
    fn shrinking_content_pulls_the_scroll_back_into_range() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 10_000, bind);
        core.run_layout([400.0, 400.0]);

        core.scroll_by(l.area, [0.0, f32::MAX]);
        assert_eq!(core.scroll_offset(l.area)[1], 10_000.0 * 20.0 - 100.0);

        // 10 rows of 20px in a 100px viewport: nothing left to scroll.
        l.sync(&mut core, 10, bind);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10, bind);
        assert_eq!(core.scroll_offset(l.area)[1], 100.0);

        l.sync(&mut core, 4, bind);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 4, bind);
        assert_eq!(
            core.scroll_offset(l.area)[1],
            0.0,
            "content shorter than the viewport"
        );
    }

    /// A row's fill goes opaque the instant it is hovered or selected, so its
    /// own content has to paint over it. Free under tree order — `place`
    /// emits a node's background, then its text, then its children — and this
    /// is the regression guard on that.
    #[test]
    fn a_rows_background_paints_under_its_own_content() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 10, bind);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10, bind);
        core.run_layout([400.0, 400.0]);

        for r in &l.rows {
            let bg = core
                .paint_slots(r.node)
                .0
                .expect("a row must own a background");
            let label = core.paint_slots(r.content).1.unwrap();
            assert!(
                core.paint_index(bg) < core.paint_index(label),
                "background paints over the label"
            );
        }
    }

    /// The list gives every row its hover fill at build, so a caller that
    /// never mentions selection still gets a list that responds to the
    /// pointer — and `set_selected` is what overrides it.
    #[test]
    fn rows_light_up_under_the_pointer_with_no_caller_help() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 10, bind);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10, bind);
        core.run_layout([400.0, 400.0]);

        let fill = |core: &UiCore, n: NodeId| {
            core.style
                .get(core.paint_slots(n).0.expect("a row has a fill"))
                .fill
        };
        let row = l.rows[2].node;
        assert_eq!(fill(&core, row), style().idle);

        core.update_pointer([150.0, 50.0], false, false, 0.0, 0.0);
        assert_eq!(fill(&core, row), style().hover, "the hovered row lit up");

        // Selected wins over hover: losing the selection under the pointer
        // reads as a bug.
        l.bound_row(2).unwrap().set_selected(&mut core, true);
        assert_eq!(fill(&core, row), style().selected);
    }

    /// Collapsing a big list leaves the pool larger than the data. Those
    /// surplus rows collapse to zero-area boxes — but a glyph's quad is sized
    /// by the *font*, not by its node, so without hiding the run explicitly
    /// every parked row's text keeps drawing, all stacked at the origin the
    /// collapsed box landed on. That is a legible pile of overlapping rows at
    /// the top of the list.
    #[test]
    fn parked_rows_draw_no_glyphs() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 200, bind);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 200, bind);
        core.run_layout([400.0, 400.0]);
        assert!(
            l.rows.len() > 4,
            "pool must exceed the shrunk data to test this"
        );

        l.sync(&mut core, 3, bind);
        core.run_layout([400.0, 400.0]);

        for r in l.rows.iter().filter(|r| r.bound.is_none()) {
            let (first, count) = core.run_slots(core.text_id(r.content).expect("row text"));
            for slot in first..first + count {
                let q = core.quad.get(slot).rect;
                assert_eq!(
                    [q[2], q[3]],
                    [0.0, 0.0],
                    "a parked row drew a glyph at {:?}",
                    [q[0], q[1]]
                );
            }
        }
    }

    /// A line sits *between* rows, an outline covers one — that difference is
    /// the whole message the indicator carries.
    #[test]
    fn the_drop_marker_lands_between_rows_or_over_one() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 100, bind);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 100, bind);
        let mark = l.mark;

        l.set_drop_mark(&mut core, Some(DropMark::Line(3)));
        core.run_layout([400.0, 400.0]);
        let r = core.node_rect(mark);
        assert_eq!([r[1], r[3]], [59.0, 2.0], "straddles the 60px boundary");

        l.set_drop_mark(&mut core, Some(DropMark::Onto(3)));
        core.run_layout([400.0, 400.0]);
        let r = core.node_rect(mark);
        assert_eq!([r[1], r[3]], [60.0, 20.0], "covers row 3");

        l.set_drop_mark(&mut core, None);
        core.run_layout([400.0, 400.0]);
        assert_eq!(core.node_rect(mark)[3], 0.0, "hidden collapses to no box");
    }

    /// The marker must paint *over* the rows it points at — a hovered row's
    /// fill is opaque. Every growth appends rows behind it in the child list,
    /// and paint order is child order, so it has to be raised each time.
    #[test]
    fn the_drop_marker_paints_over_the_rows() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 10_000, bind);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10_000, bind);
        let pool = l.rows.len();

        for h in [200.0, 300.0] {
            let mut s = core.node_style(l.area);
            s.size.height = px(h);
            core.set_node_style(l.area, s);
            core.run_layout([400.0, 400.0]);
            l.sync(&mut core, 10_000, bind);
        }
        assert!(
            l.rows.len() > pool,
            "the pool must have grown for this to bite"
        );
        l.set_drop_mark(&mut core, Some(DropMark::Onto(0)));
        core.run_layout([400.0, 400.0]);

        let mark = core.paint_index(core.paint_slots(l.mark).0.unwrap());
        for r in &l.rows {
            let row = core.paint_index(core.paint_slots(r.node).0.unwrap());
            assert!(row < mark, "row paints at {row}, over the marker at {mark}");
        }
    }

    /// The ghost is the answer to "what am I holding": it tracks the pointer
    /// wherever it goes, and it is the pointer layer's to clean up.
    #[test]
    fn the_drag_ghost_tracks_the_pointer() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 100, bind);
        core.run_layout([400.0, 400.0]);

        core.update_pointer([150.0, 50.0], true, false, 0.0, 0.0);
        let ghost = grab(&mut core, "row 7", 7);
        core.run_layout([400.0, 400.0]);
        assert!(
            core.node_rect(ghost)[2] > 0.0,
            "shrink-wrapped around its label"
        );

        // Well outside the 200x100 viewport, which is where a re-parenting
        // drag spends most of its travel — and where a ghost owned by the
        // list would have been clipped away.
        core.update_pointer([300.0, 250.0], false, false, 0.0, 0.0);
        core.run_layout([400.0, 400.0]);
        let r = core.node_rect(ghost);
        assert_eq!(
            [r[0], r[1]],
            [314.0, 260.0],
            "offset from the pointer, not on it"
        );
        assert_eq!(core.dragging::<usize>(), Some(&7));

        core.update_pointer([300.0, 250.0], false, true, 0.0, 0.0);
        assert_eq!(
            core.dragging::<usize>(),
            None,
            "the release ends the gesture"
        );
        assert_eq!(core.ghost(), None, "and takes the ghost with it");
    }

    /// The release must *free* the ghost, not merely forget the gesture — a
    /// leaked one keeps drawing at the last place the pointer was, forever.
    /// The handle going stale is the proof, since only `remove_node` bumps
    /// the generation.
    #[test]
    #[should_panic(expected = "stale NodeId")]
    fn a_released_ghost_is_freed_not_just_forgotten() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 100, bind);
        core.run_layout([400.0, 400.0]);

        core.update_pointer([150.0, 50.0], true, false, 0.0, 0.0);
        let ghost = grab(&mut core, "row 7", 7);
        core.update_pointer([150.0, 90.0], false, true, 0.0, 0.0);
        core.node_rect(ghost);
    }

    /// A ghost is minted at the grab, so it is already the last child of the
    /// root and paints over everything — with no `raise` and nothing to
    /// remember. That is the payoff for owning it per gesture rather than for
    /// the life of the list.
    #[test]
    fn the_drag_ghost_needs_no_raising() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        let root = core.root();
        let panel = core.node(root, Style::default());
        core.set_background(panel, UiStyle::fill(rgba(20, 20, 20, 255)));
        l.sync(&mut core, 100, bind);
        core.run_layout([400.0, 400.0]);

        let ghost = grab(&mut core, "row 7", 7);
        core.run_layout([400.0, 400.0]);
        let (g, p) = (
            core.paint_slots(ghost).0.unwrap(),
            core.paint_slots(panel).0.unwrap(),
        );
        assert!(core.paint_index(g) > core.paint_index(p));
    }

    /// The whole point of moving the drag into the pointer layer: a payload
    /// grabbed in one list lands in another, and neither knows the other
    /// exists. Only the list under the pointer at release hears it.
    #[test]
    fn a_drag_crosses_from_one_list_to_another() {
        let mut core = UiCore::new();
        let mut src = list(&mut core);
        let root = core.root();
        let mut dst: RowList = RowList::new(
            &mut core,
            root,
            Style {
                inset: Rect {
                    left: px(200.0),
                    ..viewport().inset
                },
                ..viewport()
            },
            style(),
        );
        for _ in 0..2 {
            src.sync(&mut core, 100, bind);
            dst.sync(&mut core, 100, bind);
            core.run_layout([400.0, 400.0]);
        }

        core.update_pointer([100.0, 30.0], true, false, 0.0, 0.0);
        grab(&mut core, "row 1", 1);
        // Over the second list's third row, and released there.
        core.update_pointer([300.0, 50.0], false, false, 0.0, 0.0);
        core.update_pointer([300.0, 50.0], false, true, 0.0, 0.0);

        assert_eq!(
            dst.dropped_on::<usize>(&core),
            Some((2, &1)),
            "row 1 landed on row 2"
        );
        assert_eq!(
            src.dropped_on::<usize>(&core),
            None,
            "the source is not the target"
        );
        // A list that does not deal in `usize` declines rather than
        // mis-reading the payload.
        assert_eq!(dst.dropped_on::<u64>(&core), None);

        // One frame, like `clicked` — and it has to clear on an *idle* frame
        // too, or a drop the caller acts on keeps landing for the rest of the
        // session.
        core.update_pointer([300.0, 50.0], false, false, 0.0, 0.0);
        assert_eq!(dst.dropped_on::<usize>(&core), None, "a drop lands once");
    }

    /// Fewer rows than the pool: the surplus must park, not draw stale text.
    #[test]
    fn surplus_rows_park_when_the_data_shrinks() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 10_000, bind);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10_000, bind);

        l.sync(&mut core, 2, bind);
        core.run_layout([400.0, 400.0]);
        assert_eq!(l.rows.iter().filter(|r| r.bound.is_some()).count(), 2);
        assert_eq!(
            core.node_rect(l.rows[3].node)[3],
            0.0,
            "parked row has no box"
        );
        assert_eq!(l.hovered(&core), None);
    }
}
