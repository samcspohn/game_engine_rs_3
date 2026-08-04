//! The widget tree and its layout pass (ADR-0006 phase 3, layout half).
//!
//! Nodes form a tree; [`taffy`] computes their boxes with CSS flexbox and
//! grid semantics; the results are pushed into the primitive slots through
//! `SlotArray::set`, so a relayout that happens to produce identical rects
//! still uploads nothing.
//!
//! # Why taffy rather than a hand-rolled solver
//!
//! Layout here runs at **event** frequency, not per frame — so the reason
//! this codebase owns its hot paths (control over the microseconds) simply
//! does not apply, while flexbox's edge cases and grid track sizing are
//! weeks of work either way.
//!
//! Its invalidation model is the one this ADR already specifies rather than
//! merely a compatible one: `TaffyTree::mark_dirty` walks up the ancestor
//! chain and **stops as soon as it finds a node already dirty**, and each
//! node caches its `(incoming constraint) → (size)` result, so a clean
//! subtree entered with an unchanged constraint short-circuits instead of
//! recomputing. That is "mark up, visit down, only into dirty subtrees",
//! implemented by someone else.
//!
//! A constraint solver (Cassowary, Apple's Auto Layout) was rejected for a
//! structural reason, not a performance one: it is one global system of
//! equations, so nudging any variable can ripple anywhere and there is no
//! such thing as relayouting one subtree. That is incompatible with an
//! architecture built end-to-end on local invalidation.
//!
//! # Coordinate spaces
//!
//! Taffy reports each node's `location` **relative to its parent**. The
//! walk in [`UiCore::run_layout`] accumulates absolute screen positions on
//! the way down and writes those into `ui_quad`, which is group-local — for
//! now every node sits in one screen-sized group, so the two coincide.
//! Scroll areas and docking (phase 4) are what make groups earn their
//! offset, and they change only this walk, not the shaders.

use taffy::{AvailableSpace, Size, TaffyTree};

use super::{font, GroupId, PrimId, TextId, UiCore, UiStyle};

/// Layout vocabulary, re-exported so callers need not name taffy directly.
///
/// This *is* the CSS box model — flex direction, grow/shrink, gap, padding,
/// alignment, grid tracks. Wrapping it in a bespoke vocabulary would only
/// obscure a spec most people already know, and would have to grow a
/// synonym for every property taffy already has.
pub mod style {
    pub use taffy::geometry::{Line, Rect, Size};
    pub use taffy::prelude::{
        auto, evenly_sized_tracks, fit_content, flex, fr, length, line, max_content, min_content,
        minmax, percent, repeat, span, zero, FromLength, TaffyAuto, TaffyZero,
    };

    /// `length()` with the numeric type pinned to `f32`. Taffy's generic
    /// version infers a bare `8.0` literal as `f64`, which it has no
    /// conversion for, so every call site would otherwise need an `_f32`
    /// suffix.
    pub fn px<T: FromLength>(v: f32) -> T {
        length(v)
    }
    pub use taffy::style::{
        AlignContent, AlignItems, AlignSelf, BoxSizing, Dimension, Display, FlexDirection,
        FlexWrap, GridAutoFlow, GridPlacement, GridTemplateComponent, JustifyContent, JustifyItems,
        JustifySelf, LengthPercentage, LengthPercentageAuto, MaxTrackSizingFunction,
        MinTrackSizingFunction, Overflow, Position, Style, TrackSizingFunction,
    };
}

use style::{px, Style};

/// A widget-tree node. Cheap and `Copy`; a stale one panics on use rather
/// than silently no-opping.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeId(u32);

struct Node {
    taffy: taffy::NodeId,
    children: Vec<NodeId>,
    /// The group every primitive this node owns belongs to. Inherited from
    /// the parent until scroll areas and docking introduce new ones.
    group: GroupId,
    /// Optional filled/bordered rect covering the node's whole box.
    background: Option<PrimId>,
    /// Optional glyph run, positioned at the node's origin.
    text: Option<TextId>,
}

/// What taffy needs to size a leaf it cannot measure itself. The built-in
/// font has a fixed advance, so a string's natural size is known the moment
/// its text is set — no callback into the run store, and therefore no
/// borrow tangle inside `compute_layout_with_measure`.
#[derive(Clone, Copy, Default)]
pub(crate) struct Measured {
    size: [f32; 2],
}

pub(crate) struct Tree {
    taffy: TaffyTree<Measured>,
    nodes: Vec<Node>,
    /// Each node's solved box, absolute in screen px — what hit testing and
    /// splitter drags read. Filled by the placement walk.
    absolute: Vec<[f32; 4]>,
    root: NodeId,
    /// Screen extent the root was last sized against; a change re-styles the
    /// root, which marks the whole tree dirty exactly once.
    screen: [f32; 2],
}

// SAFETY: taffy packs every length into `CompactLength`, whose payload is a
// tagged `*const ()`. That one raw pointer is what makes `Style`, and
// therefore `TaffyTree`, `!Send` — and it is the *only* reason, since every
// other field here is plain data.
//
// The pointer variant is `calc()`. Its sole constructor
// (`CompactLength::calc`) and its sole readers (`calc_value`, `is_calc`) are
// all `#[cfg(feature = "calc")]`, and the workspace builds taffy with
// `default-features = false` without enabling `calc` (see the dependency's
// comment in the root `Cargo.toml`). In this build the tag can therefore
// only ever hold an `f32` — never an address — so there is nothing
// thread-unsafe to move.
//
// **Enabling `taffy/calc` invalidates this.** The assertion below is what
// keeps that from being a silent regression.
unsafe impl Send for Tree {}

/// Fails to compile if `taffy/calc` is ever enabled, because `calc_value`
/// only exists under that feature — which is precisely when the `Send`
/// assertion above stops holding.
#[allow(dead_code)]
const fn assert_taffy_calc_disabled() {
    trait NoCalc {
        fn calc_value(self) -> ();
    }
    impl NoCalc for taffy::style::CompactLength {
        fn calc_value(self) -> () {}
    }
    // With `calc` on, taffy's inherent `calc_value` wins this call and the
    // `-> *const ()` return type fails to coerce to `()`.
    let _: fn(taffy::style::CompactLength) -> () = |c| c.calc_value();
}

impl Tree {
    pub(crate) fn new(group: GroupId) -> Self {
        let mut taffy = TaffyTree::new();
        let root_taffy = taffy
            .new_leaf(Style {
                size: Size {
                    width: px(0.0),
                    height: px(0.0),
                },
                ..Default::default()
            })
            .expect("taffy root");
        Self {
            taffy,
            nodes: vec![Node {
                taffy: root_taffy,
                children: Vec::new(),
                group,
                background: None,
                text: None,
            }],
            absolute: vec![[0.0; 4]],
            root: NodeId(0),
            screen: [0.0, 0.0],
        }
    }
}

impl UiCore {
    /// The screen-sized root every node descends from.
    pub fn root(&mut self) -> NodeId {
        self.tree.root
    }

    /// Add a container. Style it with the re-exported taffy vocabulary:
    /// `Style { display: Display::Flex, flex_direction: FlexDirection::Column,
    /// gap: Size { width: length(0.0), height: length(6.0) }, .. }`.
    pub fn node(&mut self, parent: NodeId, style: Style) -> NodeId {
        let group = self.tree.nodes[parent.0 as usize].group;
        let taffy_id = self.tree.taffy.new_leaf(style).expect("taffy new_leaf");
        let id = NodeId(self.tree.nodes.len() as u32);
        self.tree.nodes.push(Node {
            taffy: taffy_id,
            children: Vec::new(),
            group,
            background: None,
            text: None,
        });
        self.tree.nodes[parent.0 as usize].children.push(id);
        let parent_taffy = self.tree.nodes[parent.0 as usize].taffy;
        self.tree
            .taffy
            .add_child(parent_taffy, taffy_id)
            .expect("taffy add_child");
        id
    }

    /// Give a node a filled / bordered rect covering its whole box. Called
    /// again on the same node, it restyles in place — one dirty `ui_style`
    /// word, no layout at all, which is what makes hover cheap.
    pub fn set_background(&mut self, n: NodeId, style: UiStyle) {
        match self.tree.nodes[n.0 as usize].background {
            Some(p) => self.set_style(p, style),
            None => {
                let group = self.tree.nodes[n.0 as usize].group;
                let p = self.rect(group, [0.0; 4], style);
                self.tree.nodes[n.0 as usize].background = Some(p);
            }
        }
    }

    /// A text leaf. Its natural size is handed to taffy as a measured leaf,
    /// so it participates in flex and grid sizing like any other box.
    pub fn label(&mut self, parent: NodeId, px: f32, color: u32, text: &str) -> NodeId {
        let n = self.node(parent, Style::default());
        let group = self.tree.nodes[n.0 as usize].group;
        let t = self.text(group, [0.0, 0.0], px, color, text);
        self.tree.nodes[n.0 as usize].text = Some(t);
        self.measure_label(n, text, px);
        n
    }

    /// Retype a label. Unchanged text returns before touching anything;
    /// changed text dirties only the glyphs that differ, and re-measures
    /// only if the string's width actually moved.
    pub fn set_label(&mut self, n: NodeId, text: &str) {
        let Some(t) = self.tree.nodes[n.0 as usize].text else {
            panic!("set_label on a node with no text");
        };
        if self.text_of(t) == text {
            return;
        }
        let px = self.text_px(t);
        self.set_text(t, text);
        self.measure_label(n, text, px);
    }

    pub fn set_label_color(&mut self, n: NodeId, color: u32) {
        let Some(t) = self.tree.nodes[n.0 as usize].text else {
            panic!("set_label_color on a node with no text");
        };
        self.set_text_color(t, color);
    }

    /// The node's current layout style, for read-modify-write edits.
    pub fn node_style(&self, n: NodeId) -> Style {
        self.tree
            .taffy
            .style(self.tree.nodes[n.0 as usize].taffy)
            .expect("taffy style")
            .clone()
    }

    /// Restyle a node's box. Marks it and its ancestors dirty; the next
    /// `run_layout` recomputes that path and nothing else.
    pub fn set_node_style(&mut self, n: NodeId, style: Style) {
        let taffy_id = self.tree.nodes[n.0 as usize].taffy;
        if self.tree.taffy.style(taffy_id).expect("taffy style") == &style {
            return;
        }
        self.tree
            .taffy
            .set_style(taffy_id, style)
            .expect("taffy set_style");
    }

    /// The node's computed box, absolute in screen px. Valid after
    /// `run_layout`; this is what hit testing and splitter drags read.
    pub fn node_rect(&self, n: NodeId) -> [f32; 4] {
        self.tree.absolute[n.0 as usize]
    }

    /// Publish a label's natural size to taffy, but only when it changed —
    /// `set_node_context` unconditionally marks dirty, so an unguarded call
    /// would force a relayout on every keystroke that kept the width.
    fn measure_label(&mut self, n: NodeId, text: &str, px: f32) {
        let scale = px / font::GLYPH_H as f32;
        let size = [font::text_width(text) as f32 * scale, px];
        let taffy_id = self.tree.nodes[n.0 as usize].taffy;
        if self
            .tree
            .taffy
            .get_node_context(taffy_id)
            .is_some_and(|m| m.size == size)
        {
            return;
        }
        self.tree
            .taffy
            .set_node_context(taffy_id, Some(Measured { size }))
            .expect("taffy set_node_context");
    }

    /// Solve the tree and push the results into the primitive slots.
    ///
    /// Returns early when nothing is dirty and the screen has not moved,
    /// which is the overwhelmingly common case: hovering a button restyles
    /// one `ui_style` record and never reaches here at all.
    pub fn run_layout(&mut self, screen: [f32; 2]) {
        // The root group clips to the window. Unconditional and free — the
        // equality gate turns it into a comparison on every frame but the
        // one where the window actually resized.
        self.set_group_clip(GroupId(0), [0.0, 0.0, screen[0], screen[1]]);

        let root_taffy = self.tree.nodes[self.tree.root.0 as usize].taffy;
        if self.tree.screen != screen {
            self.tree.screen = screen;
            self.tree
                .taffy
                .set_style(
                    root_taffy,
                    Style {
                        size: Size {
                            width: px(screen[0]),
                            height: px(screen[1]),
                        },
                        ..Default::default()
                    },
                )
                .expect("taffy root resize");
        }
        if !self.tree.taffy.dirty(root_taffy).expect("taffy dirty") {
            return;
        }

        self.tree
            .taffy
            .compute_layout_with_measure(
                root_taffy,
                Size {
                    width: AvailableSpace::Definite(screen[0]),
                    height: AvailableSpace::Definite(screen[1]),
                },
                // Leaves taffy cannot measure itself: honour whatever the
                // parent already decided, fall back to the natural size.
                |known, _available, _id, ctx, _style| {
                    let natural = ctx.map(|m| m.size).unwrap_or_default();
                    Size {
                        width: known.width.unwrap_or(natural[0]),
                        height: known.height.unwrap_or(natural[1]),
                    }
                },
            )
            .expect("taffy compute_layout");

        self.tree.absolute.resize(self.tree.nodes.len(), [0.0; 4]);
        self.place(self.tree.root, [0.0, 0.0]);
    }

    /// Walk the solved tree, accumulating parent-relative positions into
    /// absolute ones and writing each node's primitives.
    ///
    /// The walk is O(tree) on frames where anything moved — but every write
    /// it makes goes through `SlotArray::set`, so the *upload* stays O(what
    /// actually changed). At editor scale that trade is fine; if the walk
    /// ever shows up, cache each node's absolute rect and skip subtrees
    /// whose origin and size both held.
    fn place(&mut self, n: NodeId, origin: [f32; 2]) {
        let idx = n.0 as usize;
        let layout = *self
            .tree
            .taffy
            .layout(self.tree.nodes[idx].taffy)
            .expect("taffy layout");
        let rect = [
            origin[0] + layout.location.x,
            origin[1] + layout.location.y,
            layout.size.width,
            layout.size.height,
        ];
        self.tree.absolute[idx] = rect;

        if let Some(p) = self.tree.nodes[idx].background {
            self.set_rect(p, rect);
        }
        if let Some(t) = self.tree.nodes[idx].text {
            self.set_text_pos(t, [rect[0], rect[1]]);
        }

        // Indexed rather than iterated: `place` needs `&mut self`, so the
        // child list cannot stay borrowed across the recursion.
        for i in 0..self.tree.nodes[idx].children.len() {
            let child = self.tree.nodes[idx].children[i];
            self.place(child, [rect[0], rect[1]]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::style::*;
    use super::*;
    use crate::ui::{rgb, UiStyle};

    const WHITE: u32 = rgb(255, 255, 255);

    /// A column that leaves children at their natural width, so a test can
    /// tell measured sizes apart.
    fn column(gap: f32, pad: f32) -> Style {
        Style {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            position: Position::Absolute,
            align_items: Some(AlignItems::START),
            padding: Rect::length(pad),
            gap: Size {
                width: zero(),
                height: px(gap),
            },
            ..Default::default()
        }
    }

    #[test]
    fn flex_column_stacks_children_and_shrink_wraps() {
        let mut core = UiCore::new();
        let root = core.root();
        let panel = core.node(root, column(6.0, 4.0));
        core.set_background(panel, UiStyle::fill(WHITE));
        let a = core.label(panel, 9.0, WHITE, "aa");
        let b = core.label(panel, 9.0, WHITE, "bbbb");
        core.run_layout([200.0, 100.0]);

        let (ra, rb, rp) = (core.node_rect(a), core.node_rect(b), core.node_rect(panel));
        assert_eq!(rb[1], ra[1] + ra[3] + 6.0, "gap not honoured");
        assert!(rb[2] > ra[2], "wider string should measure wider");
        assert_eq!(rp[2], rb[2] + 8.0, "panel should shrink-wrap widest child + padding");
        // The glyph run follows the node taffy placed it at.
        assert_eq!(core.quad.get(1).rect[0], ra[0]);
    }

    /// The property the whole design turns on, now via the layout path: a
    /// relayout that produces the same boxes must not reach staging.
    #[test]
    fn unchanged_relayout_uploads_nothing() {
        let mut core = UiCore::new();
        let root = core.root();
        let panel = core.node(root, column(6.0, 4.0));
        core.label(panel, 9.0, WHITE, "hello");
        core.run_layout([200.0, 100.0]);

        let (mut stage, mut dirty) = (vec![0u32; 4096], vec![0u32; 64]);
        core.quad.upload(&mut stage, &mut dirty);

        core.run_layout([200.0, 100.0]);
        assert_eq!(core.quad.upload(&mut stage, &mut dirty), (i64::MAX, -1));
    }

    /// Resizing the window re-solves the tree; an absolutely-positioned
    /// panel that doesn't depend on the width must still not move.
    #[test]
    fn resize_reflows_without_disturbing_independent_nodes() {
        let mut core = UiCore::new();
        let root = core.root();
        let panel = core.node(root, column(6.0, 4.0));
        let label = core.label(panel, 9.0, WHITE, "hello");
        core.run_layout([200.0, 100.0]);
        let before = core.node_rect(label);

        core.run_layout([640.0, 480.0]);
        assert_eq!(core.node_rect(label), before);
    }
}
