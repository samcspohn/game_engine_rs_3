//! A virtualized row list — the scene hierarchy panel's substrate.
//!
//! A tree view is drawn as a **flat list of rows**, not a tree of nodes:
//! the caller flattens its hierarchy to `(depth, text)` and indentation is
//! left padding. That is not a shortcut, it is what makes clicking sane —
//! rows are siblings, so `hit_test`'s innermost-interactive rule already
//! reports the row the pointer is on and never a parent that happens to
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
//! # Content is pulled, never pushed
//!
//! [`Row::text`] is a [`Cow`], so the closure can hand over either a borrow
//! of data the caller already holds or a string it builds on the spot. That
//! is what keeps *content* out of any invalidation scheme: a caller need not
//! cache names anywhere, because only the pooled rows are ever asked. Renaming
//! an item is therefore not a structural event at all — the next `sync` reads
//! the new name and `set_label` dirties exactly the glyphs that differ.

use std::borrow::Cow;

use super::style::{
    auto, percent, px, zero, AlignItems, Display, FlexDirection, LengthPercentage,
    LengthPercentageAuto, Position, Rect, Size, Style, TaffyAuto,
};
use super::tree::Drag;
use super::{font, rgba, theme, Label, NodeId, StateStyle, Theme, UiCore, UiStyle};

/// Thickness of the between-rows drop line.
const LINE_H: f32 = 2.0;

/// Where the drag ghost sits relative to the pointer. Down and to the right,
/// so it never covers the row being aimed at.
const GHOST_OFFSET: [f32; 2] = [14.0, 10.0];

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

/// One row's data, produced on demand by [`RowList::sync`]'s closure.
pub struct Row<'a> {
    /// Borrowed (`name.as_str().into()`) or owned (`format!(..).into()`) —
    /// both satisfy the same closure signature, so a caller that builds the
    /// string per row needs no buffer to keep it alive.
    pub text: Cow<'a, str>,
    /// Indentation level; row `n`'s content starts `n * indent` px in.
    pub depth: u16,
    pub selected: bool,
    /// `None` for a leaf — no disclosure triangle. `Some` draws one, and the
    /// row reports it through [`RowList::toggled`] when it is clicked.
    pub expanded: Option<bool>,
}

/// How a [`RowList`] looks and measures. `row_h` is load-bearing — it is
/// what converts a scroll offset into a data index, so rows are a fixed
/// height by construction.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RowStyle {
    pub row_h: f32,
    pub indent: f32,
    pub pad_left: f32,
    pub text_px: f32,
    pub text: u32,
    pub text_selected: u32,
    pub arrow: u32,
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
            indent: 12.0,
            pad_left: 4.0,
            text_px: t.text_px,
            text: t.text,
            text_selected: t.text_strong,
            arrow: t.text_dim,
            // Rows sit on whatever the list's own background is, so at rest
            // they draw nothing rather than a surface of their own.
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

#[derive(Clone)]
struct PooledRow {
    node: NodeId,
    /// Disclosure triangle, a child of `node` so the innermost-interactive
    /// hit rule separates "toggle" from "select" for free.
    arrow: Label,
    label: Label,
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
/// `Clone` only because `Component` requires it; a clone refers to the same
/// nodes, exactly as the bare [`NodeId`]s a component already holds do.
#[derive(Clone)]
pub struct RowList {
    area: NodeId,
    /// In-flow child whose height is the whole content extent, so taffy's
    /// `scroll_height` — and therefore the scrollbar range — covers rows that
    /// have no nodes.
    sizer: NodeId,
    rows: Vec<PooledRow>,
    /// Drop indicator, drawn inside the scroll area so it scrolls with the
    /// rows it points between.
    mark: NodeId,
    /// What the pointer is carrying, drawn at the **root** — a ghost that
    /// followed the pointer out of the viewport would otherwise be cut off
    /// by the scroll area's clip exactly when it matters.
    ghost: NodeId,
    ghost_label: Label,
    /// Whether the ghost is up, so it is raised once per gesture rather than
    /// re-ordered on every frame of the drag.
    ghost_up: bool,
    style: RowStyle,
    first: usize,
}

impl RowList {
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

        let root = ui.root();
        let ghost = ui.node(root, hidden());
        // A row you picked up: the selected fill it would have had, outlined
        // in the drop colour so it reads as in flight rather than dropped.
        ui.set_background(
            ghost,
            UiStyle::fill(style.selected).border(style.drop, 1.0).radius(style.radius),
        );
        let ghost_label = ui.label(ghost, style.text_px, style.text_selected, "");

        Self {
            area,
            sizer,
            rows: Vec::new(),
            mark,
            ghost,
            ghost_label,
            ghost_up: false,
            style,
            first: 0,
        }
    }

    /// The scroll area, for styling it or scrolling it programmatically.
    pub fn node(&self) -> NodeId {
        self.area
    }

    /// Point the pool at the current scroll position and re-bind it from
    /// `row`, which is called once per pooled row with a data index.
    ///
    /// The pool sizes itself from the *measured* viewport, so the first frame
    /// after construction shows nothing — there is no layout yet to measure.
    pub fn sync<'a>(&mut self, ui: &mut UiCore, len: usize, mut row: impl FnMut(usize) -> Row<'a>) {
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
            let (node, label) = (self.rows[k].node, self.rows[k].label);
            self.rows[k].bound = (i < len).then_some(i);
            let Some(i) = self.rows[k].bound else {
                ui.set_node_style(node, hidden());
                continue;
            };
            let data = row(i);
            ui.set_node_style(node, row_style(&s, i, data.depth));
            ui.set_state_style(node, states(&s, data.selected));
            ui.set_label(label, &data.text);
            ui.set_label_color(
                label,
                if data.selected { s.text_selected } else { s.text },
            );

            // The arrow keeps its box on a leaf so labels stay aligned down
            // the column; only its glyph goes away.
            let arrow = self.rows[k].arrow;
            let mut glyph = [0u8; 4];
            // Follows the label's colour: a dim arrow on an opaque selected
            // fill is nearly invisible.
            ui.set_label_color(
                arrow,
                if data.selected { s.text_selected } else { s.arrow },
            );
            ui.set_label(
                arrow,
                match data.expanded {
                    Some(true) => font::ARROW_DOWN.encode_utf8(&mut glyph),
                    Some(false) => font::ARROW_RIGHT.encode_utf8(&mut glyph),
                    None => "",
                },
            );
            ui.set_interactive(arrow, data.expanded.is_some());
        }
    }

    /// Data index whose disclosure triangle was clicked this frame.
    ///
    /// Distinct from [`clicked`](Self::clicked) with no special case in the
    /// hit test: the arrow is a *child* of the row, and `hit_test` reports the
    /// innermost interactive node, so clicking the arrow toggles and clicking
    /// anywhere else on the row selects. Under DOM-style bubbling this would
    /// have needed a `stopPropagation`.
    pub fn toggled(&self, ui: &UiCore) -> Option<usize> {
        self.rows
            .iter()
            .find(|r| ui.clicked(r.arrow))
            .and_then(|r| r.bound)
    }

    /// Data index of the row clicked this frame. Rows are siblings, so this
    /// is unambiguous — there is no parent row to have swallowed the click.
    pub fn clicked(&self, ui: &UiCore) -> Option<usize> {
        self.rows
            .iter()
            .find(|r| ui.clicked(r.node))
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
        self.rows.iter().find_map(|r| Some((r.bound?, ui.drag(r.node)?)))
    }

    /// Data index a drag started from, on the one frame it is released.
    ///
    /// Distinct from [`clicked`](Self::clicked), which fires only when the
    /// release lands back on the row it started from — the case a drop is
    /// defined *not* to be.
    pub fn dropped(&self, ui: &UiCore) -> Option<(usize, Drag)> {
        self.rows.iter().find_map(|r| Some((r.bound?, ui.dropped(r.node)?)))
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

    /// The pooled row the pointer is on, counting a hit on its disclosure
    /// arrow.
    ///
    /// The arrow is a child and wins the hit — which is exactly what keeps
    /// toggling apart from selecting — but for hovering and for aiming a drop
    /// it is still the row the pointer is over.
    fn row_at(&self, ui: &UiCore) -> Option<&PooledRow> {
        self.rows
            .iter()
            .find(|r| ui.hovered(r.node) || ui.hovered(r.arrow))
    }

    /// Show what the pointer is carrying, at the pointer, or hide it.
    ///
    /// The text is the caller's: only it knows what a dragged index *is*, and
    /// a ghost that showed a row index would say nothing.
    pub fn set_drag_ghost(&mut self, ui: &mut UiCore, text: Option<&str>) {
        let Some(text) = text else {
            if std::mem::take(&mut self.ghost_up) {
                ui.set_node_style(self.ghost, hidden());
            }
            return;
        };
        if !std::mem::replace(&mut self.ghost_up, true) {
            // Once per gesture. The ghost was built with its list, before
            // whatever panels came after it, and paint order is tree order —
            // so it has to climb over them, but only over what exists now.
            ui.raise(self.ghost);
        }
        ui.set_label(self.ghost_label, text);
        let p = ui.pointer.pos;
        ui.set_node_style(self.ghost, ghost_style(&self.style, p));
    }

    /// The drag ghost's node and label, so a test can read back what was
    /// picked up rather than what the caller meant to pick up.
    pub(crate) fn ghost(&self) -> (NodeId, Label) {
        (self.ghost, self.ghost_label)
    }

    /// The pooled row currently showing `index`, as `(row, arrow, label)`.
    /// Lets a test read back what actually reached the widget tree rather
    /// than what the caller believes it asked for.
    pub(crate) fn bound_row(&self, index: usize) -> Option<(NodeId, Label, Label)> {
        self.rows
            .iter()
            .find(|r| r.bound == Some(index))
            .map(|r| (r.node, r.arrow, r.label))
    }

    fn push_row(&mut self, ui: &mut UiCore) {
        let node = ui.node(self.area, Style::default());
        // Allocate the background *before* the glyphs. Painter's order is slot
        // order (`order[i] = (i, gid)`), so a background claimed later would
        // paint over the row's own text — invisible while the idle fill is
        // transparent, and a row that blanks itself the moment it is hovered
        // or selected.
        ui.set_background(node, UiStyle::fill(self.style.idle).radius(self.style.radius));
        let arrow = ui.label(node, self.style.text_px, self.style.arrow, "");
        ui.set_node_style(
            arrow,
            Style {
                size: Size {
                    width: px(self.style.indent),
                    height: auto(),
                },
                flex_shrink: 0.0,
                ..Default::default()
            },
        );
        let label = ui.label(node, self.style.text_px, self.style.text, "");
        ui.set_interactive(node, true);
        self.rows.push(PooledRow { node, arrow, label, bound: None });
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

/// Absolutely positioned at its data index, full width, indented by depth.
/// Position is the *only* thing tying a pooled node to an index, which is
/// what lets the ring reorder rows without moving anything else.
fn row_style(s: &RowStyle, index: usize, depth: u16) -> Style {
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
            left: px(s.pad_left + depth as f32 * s.indent),
            right: zero::<LengthPercentage>(),
            top: zero(),
            bottom: zero(),
        },
        ..Default::default()
    }
}

/// Shrink-wrapped to its label and pinned near the pointer. Absolute against
/// the root, whose group offset is zero — so layout space is screen space and
/// the pointer position goes in unconverted.
fn ghost_style(s: &RowStyle, p: [f32; 2]) -> Style {
    Style {
        display: Display::Flex,
        align_items: Some(AlignItems::CENTER),
        position: Position::Absolute,
        inset: Rect {
            left: px(p[0] + GHOST_OFFSET[0]),
            top: px(p[1] + GHOST_OFFSET[1]),
            right: LengthPercentageAuto::AUTO,
            bottom: LengthPercentageAuto::AUTO,
        },
        padding: Rect {
            left: px(s.pad_left + 2.0),
            right: px(s.pad_left + 2.0),
            top: px(2.0),
            bottom: px(2.0),
        },
        ..Default::default()
    }
}

/// A selected row keeps its fill through hover: the selection is the more
/// important signal, and losing it under the pointer reads as a bug.
fn states(s: &RowStyle, selected: bool) -> StateStyle {
    let base = UiStyle::fill(s.idle).radius(s.radius);
    match selected {
        true => StateStyle::fills(base, s.selected, s.selected, s.selected),
        false => StateStyle::fills(base, s.idle, s.hover, s.hover),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 10 000 rows in a 100 px viewport, at a depth that repeats so
    /// indentation is exercised.
    fn list(core: &mut UiCore) -> RowList {
        let root = core.root();
        let l = RowList::new(
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
            RowStyle {
                row_h: 20.0,
                ..Default::default()
            },
        );
        core.run_layout([400.0, 400.0]);
        l
    }

    /// Builds its text per call, which only compiles because `Row::text` is a
    /// `Cow` — the case a tree panel hits when it reads a name from the
    /// hierarchy rather than from a cache it maintains.
    fn data(i: usize) -> Row<'static> {
        Row {
            text: format!("row {i}").into(),
            depth: (i % 3) as u16,
            selected: false,
            expanded: None,
        }
    }

    /// The headline property: the node count follows the *viewport*, not the
    /// data. 10 000 rows must not build 10 000 nodes.
    #[test]
    fn pool_is_viewport_sized_not_data_sized() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);

        l.sync(&mut core, 10_000, data);
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
        l.sync(&mut core, 10_000, data);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10_000, data);

        let before: Vec<_> = l.rows.iter().map(|r| r.bound).collect();
        core.scroll_by(l.area, [0.0, 20.0]);
        l.sync(&mut core, 10_000, data);
        let after: Vec<_> = l.rows.iter().map(|r| r.bound).collect();

        let moved = before.iter().zip(&after).filter(|(a, b)| a != b).count();
        assert_eq!(moved, 1, "a one-row scroll should recycle one row");
        assert_eq!(after[0], Some(6), "row 0 wrapped to the bottom of the window");
        assert_eq!(after[1], Some(1), "everything else held");
    }

    /// A steady frame — same data, same scroll — must upload nothing at all.
    #[test]
    fn resyncing_unchanged_data_uploads_nothing() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 10_000, data);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10_000, data);
        core.run_layout([400.0, 400.0]);

        let (mut stage, mut dirty) = (vec![0u32; 1 << 16], vec![0u32; 1 << 10]);
        let clean = (i64::MAX, -1);
        core.quad.upload(&mut stage, &mut dirty);
        core.style.upload(&mut stage, &mut dirty);

        l.sync(&mut core, 10_000, data);
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
        l.sync(&mut core, 10_000, data);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10_000, data);
        core.run_layout([400.0, 400.0]);

        // Third row down, well inside the deepest indent.
        let p = [150.0, 50.0];
        core.update_pointer(p, true, false, 0.0);
        core.update_pointer(p, false, true, 0.0);
        assert_eq!(l.clicked(&core), Some(2));

        // Scrolled by two rows, the same pixel is a different entity.
        core.scroll_by(l.area, [0.0, 40.0]);
        l.sync(&mut core, 10_000, data);
        core.run_layout([400.0, 400.0]);
        core.update_pointer(p, true, false, 0.0);
        core.update_pointer(p, false, true, 0.0);
        assert_eq!(l.clicked(&core), Some(4));
    }

    /// Collapsing a subtree cuts the content out from under a scroll offset
    /// that was legal a moment earlier. Without a re-clamp the viewport shows
    /// blank, since nothing else ever pulls the offset back.
    #[test]
    fn shrinking_content_pulls_the_scroll_back_into_range() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 10_000, data);
        core.run_layout([400.0, 400.0]);

        core.scroll_by(l.area, [0.0, f32::MAX]);
        assert_eq!(core.scroll_offset(l.area)[1], 10_000.0 * 20.0 - 100.0);

        // 10 rows of 20px in a 100px viewport: nothing left to scroll.
        l.sync(&mut core, 10, data);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10, data);
        assert_eq!(core.scroll_offset(l.area)[1], 100.0);

        l.sync(&mut core, 4, data);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 4, data);
        assert_eq!(core.scroll_offset(l.area)[1], 0.0, "content shorter than the viewport");
    }

    /// Painter's order is slot order, so a row that claims its background
    /// after its glyphs blanks itself the instant the fill stops being
    /// transparent — which is exactly when a row is hovered or selected, and
    /// therefore invisible in any test that never paints an opaque row.
    #[test]
    fn a_rows_background_paints_under_its_own_text() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 10, data);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10, data);

        for r in &l.rows {
            let (bg, _) = core.paint_slots(r.node);
            let (_, arrow) = core.paint_slots(r.arrow);
            let (_, label) = core.paint_slots(r.label);
            let bg = bg.expect("a row must own a background");
            assert!(bg < arrow.unwrap(), "background {bg} paints over the arrow");
            assert!(bg < label.unwrap(), "background {bg} paints over the label");
        }
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
        l.sync(&mut core, 200, data);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 200, data);
        core.run_layout([400.0, 400.0]);
        assert!(l.rows.len() > 4, "pool must exceed the shrunk data to test this");

        l.sync(&mut core, 3, data);
        core.run_layout([400.0, 400.0]);

        for r in l.rows.iter().filter(|r| r.bound.is_none()) {
            for n in [r.arrow, r.label] {
                let (first, count) = core.run_slots(core.text_id(n).expect("row text"));
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
    }

    /// A line sits *between* rows, an outline covers one — that difference is
    /// the whole message the indicator carries.
    #[test]
    fn the_drop_marker_lands_between_rows_or_over_one() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 100, data);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 100, data);
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
        l.sync(&mut core, 10_000, data);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10_000, data);
        let pool = l.rows.len();

        for h in [200.0, 300.0] {
            let mut s = core.node_style(l.area);
            s.size.height = px(h);
            core.set_node_style(l.area, s);
            core.run_layout([400.0, 400.0]);
            l.sync(&mut core, 10_000, data);
        }
        assert!(l.rows.len() > pool, "the pool must have grown for this to bite");
        l.set_drop_mark(&mut core, Some(DropMark::Onto(0)));
        core.run_layout([400.0, 400.0]);

        let mark = core.paint_index(core.paint_slots(l.mark).0.unwrap());
        for r in &l.rows {
            let row = core.paint_index(core.paint_slots(r.node).0.unwrap());
            assert!(row < mark, "row paints at {row}, over the marker at {mark}");
        }
    }

    /// The ghost is the answer to "what am I holding": it tracks the pointer,
    /// and it is anchored to the root so leaving the list does not clip it.
    #[test]
    fn the_drag_ghost_tracks_the_pointer() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 100, data);
        core.run_layout([400.0, 400.0]);

        // Well outside the 200x100 viewport, which is where a re-parenting
        // drag spends most of its travel.
        core.update_pointer([300.0, 250.0], true, false, 0.0);
        l.set_drag_ghost(&mut core, Some("row 7"));
        core.run_layout([400.0, 400.0]);

        let (node, label) = l.ghost();
        let r = core.node_rect(node);
        assert_eq!([r[0], r[1]], [314.0, 260.0], "offset from the pointer, not on it");
        assert!(r[2] > 0.0 && r[3] > 0.0, "shrink-wrapped around the label");
        assert_eq!(core.node_text(label), Some("row 7"));

        l.set_drag_ghost(&mut core, None);
        core.run_layout([400.0, 400.0]);
        assert_eq!(core.node_rect(node)[3], 0.0, "released ghosts leave no box");
    }

    /// The ghost is built with its list and floats over panels created after
    /// it — which, in the editor, is every panel. Paint order is tree order,
    /// so being built first means being painted under until it is raised.
    #[test]
    fn the_drag_ghost_paints_over_later_panels() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        let root = core.root();
        let panel = core.node(root, Style::default());
        core.set_background(panel, UiStyle::fill(rgba(20, 20, 20, 255)));

        l.sync(&mut core, 100, data);
        core.run_layout([400.0, 400.0]);
        let ghost = core.paint_slots(l.ghost().0).0.unwrap();
        let over = core.paint_slots(panel).0.unwrap();
        assert!(core.paint_index(ghost) < core.paint_index(over), "starts underneath");

        l.set_drag_ghost(&mut core, Some("row 7"));
        core.run_layout([400.0, 400.0]);
        assert!(core.paint_index(ghost) > core.paint_index(over));
    }

    /// Fewer rows than the pool: the surplus must park, not draw stale text.
    #[test]
    fn surplus_rows_park_when_the_data_shrinks() {
        let mut core = UiCore::new();
        let mut l = list(&mut core);
        l.sync(&mut core, 10_000, data);
        core.run_layout([400.0, 400.0]);
        l.sync(&mut core, 10_000, data);

        l.sync(&mut core, 2, data);
        core.run_layout([400.0, 400.0]);
        assert_eq!(l.rows.iter().filter(|r| r.bound.is_some()).count(), 2);
        assert_eq!(core.node_rect(l.rows[3].node)[3], 0.0, "parked row has no box");
        assert_eq!(l.hovered(&core), None);
    }
}
