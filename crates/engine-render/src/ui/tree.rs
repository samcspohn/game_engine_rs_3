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
    pub use taffy::geometry::{Line, Point, Rect, Size};
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

/// Pixels scrolled per wheel line.
const WHEEL_PX: f32 = 40.0;

fn contains(r: [f32; 4], p: [f32; 2]) -> bool {
    p[0] >= r[0] && p[0] < r[2] && p[1] >= r[1] && p[1] < r[3]
}

fn intersect(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

/// `[x, y, w, h]` layout box + group offset → `[x0, y0, x1, y1]` on screen,
/// the corner form both clipping and hit testing use.
fn screen_rect(rect: [f32; 4], offset: [f32; 2]) -> [f32; 4] {
    let (x, y) = (rect[0] + offset[0], rect[1] + offset[1]);
    [x, y, x + rect[2], y + rect[3]]
}

/// A widget-tree node. Cheap and `Copy`; a stale one panics on use rather
/// than silently no-opping.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeId(pub(crate) u32);

struct Node {
    taffy: taffy::NodeId,
    children: Vec<NodeId>,
    /// The group this node's *own* primitives belong to. For a scroll area
    /// that is the **parent's** group, deliberately: the viewport frame must
    /// stay put while its contents move.
    group: GroupId,
    /// Set only on scroll areas: the group *children* inherit, whose offset
    /// carries the scroll. `None` elsewhere, so children inherit `group`.
    content_group: Option<GroupId>,
    /// Current scroll offset in px, positive = content moved up/left.
    scroll: [f32; 2],
    /// Offset this node's content group inherits from outside it, captured
    /// by the placement walk. `content_group.offset = scroll_base - scroll`,
    /// which is what lets a scroll write one record without a relayout.
    scroll_base: [f32; 2],
    /// Optional filled/bordered rect covering the node's whole box.
    background: Option<PrimId>,
    /// Optional glyph run, positioned at the node's origin.
    text: Option<TextId>,
}

impl Node {
    fn new(taffy: taffy::NodeId, group: GroupId) -> Self {
        Self {
            taffy,
            children: Vec::new(),
            group,
            content_group: None,
            scroll: [0.0; 2],
            scroll_base: [0.0; 2],
            background: None,
            text: None,
        }
    }
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
            nodes: vec![Node::new(root_taffy, group)],
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
        // A scroll area hands its children the content group, not its own —
        // that one line is what puts everything inside it under the scrolled
        // offset without any node knowing it is being scrolled.
        let p = &self.tree.nodes[parent.0 as usize];
        let group = p.content_group.unwrap_or(p.group);
        let taffy_id = self.tree.taffy.new_leaf(style).expect("taffy new_leaf");
        let id = NodeId(self.tree.nodes.len() as u32);
        self.tree.nodes.push(Node::new(taffy_id, group));
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

    /// The node's glyph run, if it has one.
    pub(crate) fn text_id(&self, n: NodeId) -> Option<TextId> {
        self.tree.nodes[n.0 as usize].text
    }

    /// The node's current label text, if it has one.
    pub(crate) fn node_text(&self, n: NodeId) -> Option<&str> {
        self.tree.nodes[n.0 as usize].text.map(|t| self.text_of(t))
    }

    /// First slot of the node's background and of its glyph run.
    ///
    /// Painter's order **is** slot order (`order[i] = (i, gid)`), so these
    /// must be ascending or a node's background covers its own text. Every
    /// widget therefore has to claim its background before any child content;
    /// exposed so that rule can be asserted instead of remembered.
    pub(crate) fn paint_slots(&self, n: NodeId) -> (Option<u32>, Option<u32>) {
        let node = &self.tree.nodes[n.0 as usize];
        (
            node.background.map(|p| p.0),
            node.text.map(|t| self.runs[t.0 as usize].first),
        )
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
        // The walk also assigns paint order, so a run the allocator moved
        // has to re-walk even when taffy has nothing to re-solve.
        let solve = self.tree.taffy.dirty(root_taffy).expect("taffy dirty");
        if !solve && !self.order_dirty() {
            return;
        }

        if solve {
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
        }

        self.tree.absolute.resize(self.tree.nodes.len(), [0.0; 4]);
        self.begin_order();
        self.place(
            self.tree.root,
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0, screen[0], screen[1]],
            false,
        );
        self.end_order();
    }

    // ── Scroll areas ────────────────────────────────────────────────────

    /// A clipped, scrollable viewport. `style` supplies the box (give it a
    /// definite height); the overflow fields are set here because they are
    /// what makes it a scroll container rather than a box that grows.
    ///
    /// The returned node is the viewport: its background, border and size are
    /// its own, and **children added to it scroll**. That split is why the
    /// frame stays put while the contents move.
    ///
    /// Vertical only for now, and **nested scroll areas panic** — the inner
    /// group's offset would have to track the outer's, which the one-record
    /// scroll path deliberately does not walk. Loud beats subtly misplaced.
    pub fn scroll_area(&mut self, parent: NodeId, style: Style) -> NodeId {
        use crate::ui::style::{Overflow, Point};

        let p = &self.tree.nodes[parent.0 as usize];
        let inherited = p.content_group.unwrap_or(p.group);
        assert_eq!(
            inherited,
            self.tree.nodes[self.tree.root.0 as usize].group,
            "nested scroll areas are not supported"
        );

        let n = self.node(
            parent,
            Style {
                overflow: Point {
                    x: Overflow::Hidden,
                    y: Overflow::Scroll,
                },
                ..style
            },
        );
        let g = self.group([0.0; 4], [0.0; 2]);
        self.tree.nodes[n.0 as usize].content_group = Some(g);
        n
    }

    /// How far this area can scroll on each axis, from taffy's content size.
    pub fn max_scroll(&self, n: NodeId) -> [f32; 2] {
        let l = self
            .tree
            .taffy
            .layout(self.tree.nodes[n.0 as usize].taffy)
            .expect("taffy layout");
        [l.scroll_width(), l.scroll_height()]
    }

    pub fn scroll_offset(&self, n: NodeId) -> [f32; 2] {
        self.tree.nodes[n.0 as usize].scroll
    }

    /// Scroll by `delta`, clamped to the content extent.
    ///
    /// **This is the payoff**: it writes one `ui_group` record and touches no
    /// quads and no layout, however many primitives are inside. A 10 000-row
    /// list scrolls for the same 32 bytes as an empty one.
    pub fn scroll_by(&mut self, n: NodeId, delta: [f32; 2]) {
        let idx = n.0 as usize;
        let g = self.tree.nodes[idx]
            .content_group
            .expect("scroll_by on a node that is not a scroll area");
        let max = self.max_scroll(n);
        let old = self.tree.nodes[idx].scroll;
        let new = [
            (old[0] + delta[0]).clamp(0.0, max[0]),
            (old[1] + delta[1]).clamp(0.0, max[1]),
        ];
        if new == old {
            return;
        }
        self.tree.nodes[idx].scroll = new;
        let base = self.tree.nodes[idx].scroll_base;
        self.set_group_offset(g, [base[0] - new[0], base[1] - new[1]]);
    }

    /// Innermost scroll area whose visible box contains `p` — independent of
    /// interactivity, because the wheel should scroll whatever is under the
    /// cursor whether or not it is clickable.
    fn scroll_target(&self, p: [f32; 2]) -> Option<NodeId> {
        self.scroll_hit(
            self.tree.root,
            p,
            [0.0, 0.0],
            [0.0, 0.0, self.tree.screen[0], self.tree.screen[1]],
        )
    }

    fn scroll_hit(
        &self,
        n: NodeId,
        p: [f32; 2],
        offset: [f32; 2],
        clip: [f32; 4],
    ) -> Option<NodeId> {
        let idx = n.0 as usize;
        let (child_offset, child_clip) = self.group_context(idx, offset, clip);
        for i in (0..self.tree.nodes[idx].children.len()).rev() {
            if let Some(h) = self.scroll_hit(self.tree.nodes[idx].children[i], p, child_offset, child_clip) {
                return Some(h);
            }
        }
        let visible = contains(screen_rect(self.tree.absolute[idx], offset), p) && contains(clip, p);
        (self.tree.nodes[idx].content_group.is_some() && visible).then_some(n)
    }

    /// The (offset, clip) a node's *children* inherit — the scroll area's own
    /// group context, or the parent's unchanged.
    fn group_context(&self, idx: usize, offset: [f32; 2], clip: [f32; 4]) -> ([f32; 2], [f32; 4]) {
        match self.tree.nodes[idx].content_group {
            Some(_) => {
                let s = self.tree.nodes[idx].scroll;
                (
                    [offset[0] - s[0], offset[1] - s[1]],
                    intersect(clip, screen_rect(self.tree.absolute[idx], offset)),
                )
            }
            None => (offset, clip),
        }
    }

    // ── Pointer input (ADR-0006 phase 3b / ADR-0008 step 2) ─────────────

    /// Opt a node into hit testing. Nodes are inert by default, so a panel's
    /// background never swallows a click meant for the world behind it.
    pub fn set_interactive(&mut self, n: NodeId, interactive: bool) {
        let p = &mut self.pointer.interactive;
        if p.len() <= n.0 as usize {
            p.resize(n.0 as usize + 1, false);
        }
        p[n.0 as usize] = interactive;
    }

    /// The topmost interactive node containing `p`, or `None`.
    ///
    /// **Reverse paint order, no pruning.** `place` writes a node before its
    /// children and children in order, so paint order is the DFS preorder and
    /// the topmost node is the *last* match — hence children reversed, before
    /// self. Deliberately not pruned on the parent's box: nothing clips
    /// per-node today (clipping is per `ui_group`), so pruning would invent a
    /// containment rule the renderer does not honour and would silently
    /// mis-hit any absolutely-positioned child that escapes its parent.
    /// Pruning becomes correct — and worth it — when scroll areas give nodes
    /// real clip rects.
    ///
    /// O(nodes), but it runs on pointer events, not per frame.
    pub fn hit_test(&self, p: [f32; 2]) -> Option<NodeId> {
        self.hit(
            self.tree.root,
            p,
            [0.0, 0.0],
            [0.0, 0.0, self.tree.screen[0], self.tree.screen[1]],
        )
    }

    /// Mirrors `place`'s walk, including its group context — so a row
    /// scrolled out of its viewport is unhittable for the same reason it is
    /// invisible, rather than by a second rule that could drift from the
    /// first.
    fn hit(&self, n: NodeId, p: [f32; 2], offset: [f32; 2], clip: [f32; 4]) -> Option<NodeId> {
        let idx = n.0 as usize;
        let (child_offset, child_clip) = self.group_context(idx, offset, clip);
        for i in (0..self.tree.nodes[idx].children.len()).rev() {
            if let Some(h) = self.hit(self.tree.nodes[idx].children[i], p, child_offset, child_clip)
            {
                return Some(h);
            }
        }
        let visible = contains(screen_rect(self.tree.absolute[idx], offset), p) && contains(clip, p);
        let interactive = self.pointer.interactive.get(idx).copied().unwrap_or(false);
        (visible && interactive).then_some(n)
    }

    /// Fold this frame's pointer into hover / press / click state. Called by
    /// the renderer before `Scene::update`, so components observe the same
    /// frame's input the `dt` they were handed belongs to.
    pub(crate) fn update_pointer(
        &mut self,
        pos: [f32; 2],
        pressed: bool,
        released: bool,
        wheel: f32,
    ) {
        // `clicked` lasts exactly one frame, so it clears even on the quiet
        // path below.
        self.pointer.clicked = None;

        // Genuinely event-driven: on a frame where the pointer neither moved
        // nor did anything, there is nothing to recompute and the tree walks
        // are skipped entirely. This is the overwhelmingly common frame.
        if pos == self.pointer.pos && !pressed && !released && wheel == 0.0 {
            return;
        }

        if wheel != 0.0 {
            if let Some(target) = self.scroll_target(pos) {
                self.scroll_by(target, [0.0, -wheel * WHEEL_PX]);
            }
        }

        let was_hovered = self.pointer.hovered;
        let was_down_on = self.pointer.down_on;

        self.pointer.pos = pos;
        let hovered = self.hit_test(pos);
        self.pointer.hovered = hovered;
        self.pointer.over_scroll = self.scroll_target(pos);

        if pressed {
            self.pointer.down_on = hovered;
        }
        if released {
            if self.pointer.down_on.is_some() && self.pointer.down_on == hovered {
                self.pointer.clicked = hovered;
            }
            self.pointer.down_on = None;
        }

        // Restyle only what moved. These four cover every node whose look can
        // have changed — the one hover left, the one it entered, and the same
        // for press — so this is O(1) per frame rather than O(widgets), and
        // nothing at all on a frame where the pointer sat still. Duplicates
        // among them need no dedup: `apply_state_style` ends in
        // `set_background`, which the equality gate makes idempotent.
        for n in [was_hovered, hovered, was_down_on, self.pointer.down_on]
            .into_iter()
            .flatten()
        {
            self.apply_state_style(n);
        }
    }

    /// Pointer is over this node.
    pub fn hovered(&self, n: NodeId) -> bool {
        self.pointer.hovered == Some(n)
    }

    /// Pointer went down on this node and has not been released. Still true
    /// while the pointer is dragged off, which is what lets a button render
    /// "armed" and still cancel.
    pub fn held(&self, n: NodeId) -> bool {
        self.pointer.down_on == Some(n)
    }

    /// A full press-and-release completed on this node this frame.
    pub fn clicked(&self, n: NodeId) -> bool {
        self.pointer.clicked == Some(n)
    }

    /// The UI owns the pointer — it is over an interactive node or a scroll
    /// area, or a press that started on one is still in flight. Camera
    /// controllers and world-picking should sit out while this is true.
    ///
    /// Scroll areas count so the wheel zooms the camera or scrolls a list,
    /// never both.
    pub fn pointer_captured(&self) -> bool {
        self.pointer.hovered.is_some()
            || self.pointer.down_on.is_some()
            || self.pointer.over_scroll.is_some()
    }

    /// Walk the solved tree, accumulating parent-relative positions into
    /// absolute ones and writing each node's primitives.
    ///
    /// The walk is O(tree) on frames where anything moved — but every write
    /// it makes goes through `SlotArray::set`, so the *upload* stays O(what
    /// actually changed). At editor scale that trade is fine; if the walk
    /// ever shows up, cache each node's absolute rect and skip subtrees
    /// whose origin and size both held.
    /// `origin` accumulates taffy's parent-relative positions. `offset` and
    /// `clip` are the enclosing group's — carried down rather than stored per
    /// node, so a scroll area costs no extra memory and the hit test can
    /// mirror this walk exactly.
    ///
    /// Quads are written in **pure layout space**, never shifted by scroll:
    /// `ui.vert` adds the group offset, so scrolling stays one group record
    /// instead of one rewrite per primitive.
    /// `hidden` propagates down from the first zero-area ancestor. Taffy
    /// zeroes a `Display::None` node but may leave its children's layouts
    /// stale, and a glyph quad is sized by the font rather than by its node,
    /// so text needs telling explicitly — collapsing a box hides a rect, not
    /// a label.
    fn place(
        &mut self,
        n: NodeId,
        origin: [f32; 2],
        offset: [f32; 2],
        clip: [f32; 4],
        hidden: bool,
    ) {
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

        let hidden = hidden || (layout.size.width == 0.0 && layout.size.height == 0.0);

        // Paint order is this walk's order: background under the node's own
        // text, both under everything the children draw.
        let group = self.tree.nodes[idx].group;
        if let Some(p) = self.tree.nodes[idx].background {
            self.set_rect(p, if hidden { [0.0; 4] } else { rect });
            self.emit_order(p.0, 1, group);
        }
        if let Some(t) = self.tree.nodes[idx].text {
            self.set_run_visible(t, !hidden);
            self.set_text_pos(t, [rect[0], rect[1]]);
            let (first, count) = self.run_slots(t);
            self.emit_order(first, count, group);
        }

        // A scroll area opens a new group for its children: clipped to its
        // own on-screen box, translated by however far it is scrolled.
        let (child_offset, child_clip) = match self.tree.nodes[idx].content_group {
            Some(g) => {
                let scroll = self.tree.nodes[idx].scroll;
                self.tree.nodes[idx].scroll_base = offset;
                let inner = intersect(clip, screen_rect(rect, offset));
                let off = [offset[0] - scroll[0], offset[1] - scroll[1]];
                self.set_group_clip(g, inner);
                self.set_group_offset(g, off);
                (off, inner)
            }
            None => (offset, clip),
        };

        // Indexed rather than iterated: `place` needs `&mut self`, so the
        // child list cannot stay borrowed across the recursion.
        for i in 0..self.tree.nodes[idx].children.len() {
            let child = self.tree.nodes[idx].children[i];
            self.place(child, [rect[0], rect[1]], child_offset, child_clip, hidden);
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

    /// Paint position of a slot: where it sits in the draw list.
    fn paint_index(core: &UiCore, slot: u32) -> usize {
        (0..core.prim_count())
            .find(|&i| core.order.get(i).0[0] == slot)
            .expect("every slot must appear in the draw list") as usize
    }

    /// The bug this array exists to prevent. The slot free list hands back
    /// **low** slots, so a run that outgrew its bucket and was recycled would,
    /// under identity ordering, be drawn *behind* opaque geometry allocated
    /// earlier — a label silently swallowed by a panel background it sits on
    /// top of. Paint order must follow the tree, not the allocator.
    #[test]
    fn a_recycled_run_still_paints_above_earlier_geometry() {
        let mut core = UiCore::new();
        let root = core.root();

        let panel = core.node(root, column(0.0, 0.0));

        // A label that will outgrow its bucket, allocated *before* the
        // background it must paint over — the editor hit this because its
        // status line predates the hierarchy list's backdrop.
        let grower = core.label(panel, 9.0, WHITE, "x");
        let freed = core.paint_slots(grower).1.unwrap();
        core.set_background(panel, UiStyle::fill(WHITE));
        let bg = core.paint_slots(panel).0.expect("panel background");
        core.set_label(grower, "long enough to need a bigger bucket");
        assert_ne!(core.paint_slots(grower).1.unwrap(), freed, "run should have moved");

        // The next label of that size recycles those low slots.
        let recycler = core.label(panel, 9.0, WHITE, "y");
        let reused = core.paint_slots(recycler).1.unwrap();
        assert_eq!(reused, freed, "expected the free list to hand back the low run");
        assert!(reused < bg, "the recycled run really is below the background's slot");

        core.run_layout([200.0, 100.0]);
        assert!(
            paint_index(&core, reused) > paint_index(&core, bg),
            "a child's glyphs must paint over its ancestor's background"
        );
    }

    /// A fixed box at a known place, so pointer tests can aim at it.
    fn box_at(x: f32, y: f32, w: f32, h: f32) -> Style {
        Style {
            position: Position::Absolute,
            inset: Rect {
                left: px(x),
                top: px(y),
                right: LengthPercentageAuto::AUTO,
                bottom: LengthPercentageAuto::AUTO,
            },
            size: Size {
                width: px(w),
                height: px(h),
            },
            ..Default::default()
        }
    }

    /// Opting in is the whole difference between a button and a decoration:
    /// an un-opted node must not swallow the pointer.
    #[test]
    fn hit_test_ignores_non_interactive_nodes() {
        let mut core = UiCore::new();
        let root = core.root();
        let plain = core.node(root, box_at(10.0, 10.0, 50.0, 20.0));
        core.set_background(plain, UiStyle::fill(WHITE));
        core.run_layout([200.0, 100.0]);

        assert_eq!(core.hit_test([20.0, 15.0]), None);
        core.set_interactive(plain, true);
        assert_eq!(core.hit_test([20.0, 15.0]), Some(plain));
        assert_eq!(core.hit_test([80.0, 15.0]), None, "outside the box");
    }

    /// Later siblings paint over earlier ones, so they must win the hit.
    #[test]
    fn hit_test_picks_the_topmost_of_overlapping_nodes() {
        let mut core = UiCore::new();
        let root = core.root();
        let under = core.node(root, box_at(0.0, 0.0, 100.0, 100.0));
        let over = core.node(root, box_at(0.0, 0.0, 50.0, 50.0));
        core.set_interactive(under, true);
        core.set_interactive(over, true);
        core.run_layout([200.0, 200.0]);

        assert_eq!(core.hit_test([25.0, 25.0]), Some(over), "last child paints on top");
        assert_eq!(core.hit_test([75.0, 75.0]), Some(under), "outside the top box");
    }

    /// Press and release must land on the same node. Dragging off cancels —
    /// and `held` stays true throughout, which is what lets a button render
    /// "armed" while the pointer is away.
    #[test]
    fn click_requires_press_and_release_on_the_same_node() {
        let mut core = UiCore::new();
        let root = core.root();
        let btn = core.node(root, box_at(0.0, 0.0, 40.0, 20.0));
        core.set_interactive(btn, true);
        core.run_layout([200.0, 100.0]);

        let inside = [10.0, 10.0];
        let outside = [100.0, 80.0];

        // Press then release inside → one click, on exactly one frame.
        core.update_pointer(inside, true, false, 0.0);
        assert!(core.held(btn) && !core.clicked(btn), "press alone is not a click");
        core.update_pointer(inside, false, true, 0.0);
        assert!(core.clicked(btn), "press+release inside should click");
        core.update_pointer(inside, false, false, 0.0);
        assert!(!core.clicked(btn), "click must last exactly one frame");

        // Press inside, drag off, release → no click.
        core.update_pointer(inside, true, false, 0.0);
        core.update_pointer(outside, false, false, 0.0);
        assert!(core.held(btn), "still armed while dragged off");
        assert!(!core.hovered(btn));
        core.update_pointer(outside, false, true, 0.0);
        assert!(!core.clicked(btn), "releasing off the node must cancel");
        assert!(!core.held(btn), "release always disarms");
    }

    /// Capture is what keeps a click off the camera. It must hold while the
    /// pointer is merely hovering *and* through a drag that left the node.
    #[test]
    fn pointer_capture_covers_hover_and_drag() {
        let mut core = UiCore::new();
        let root = core.root();
        let btn = core.node(root, box_at(0.0, 0.0, 40.0, 20.0));
        core.set_interactive(btn, true);
        core.run_layout([200.0, 100.0]);

        core.update_pointer([100.0, 80.0], false, false, 0.0);
        assert!(!core.pointer_captured(), "idle over the world");

        core.update_pointer([10.0, 10.0], false, false, 0.0);
        assert!(core.pointer_captured(), "hover captures");

        core.update_pointer([10.0, 10.0], true, false, 0.0);
        core.update_pointer([100.0, 80.0], false, false, 0.0);
        assert!(
            core.pointer_captured(),
            "a drag that started on the UI keeps the pointer even off-node"
        );

        core.update_pointer([100.0, 80.0], false, true, 0.0);
        assert!(!core.pointer_captured(), "release hands it back");
    }

    /// A viewport with more rows than fit, so scroll tests have something to
    /// move. Returns `(area, rows)`.
    fn scroller(core: &mut UiCore, rows: usize) -> (NodeId, Vec<NodeId>) {
        let root = core.root();
        let area = core.scroll_area(
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
                    width: px(100.0),
                    height: px(50.0),
                },
                flex_direction: FlexDirection::Column,
                display: Display::Flex,
                ..Default::default()
            },
        );
        // In flow, and `flex_shrink: 0` so they keep their height and
        // overflow the viewport rather than being squeezed to fit.
        let row_style = Style {
            size: Size {
                width: px(100.0),
                height: px(20.0),
            },
            flex_shrink: 0.0,
            ..Default::default()
        };
        let rows = (0..rows)
            .map(|_| {
                let r = core.node(area, row_style.clone());
                core.set_background(r, UiStyle::fill(WHITE));
                core.set_interactive(r, true);
                r
            })
            .collect();
        core.run_layout([200.0, 200.0]);
        (area, rows)
    }

    /// The headline property: scrolling moves one `ui_group` record and
    /// **touches no quads at all**, however many rows are inside.
    #[test]
    fn scrolling_writes_one_group_record_and_no_quads() {
        let mut core = UiCore::new();
        let (area, _rows) = scroller(&mut core, 20);

        let (mut stage, mut dirty) = (vec![0u32; 8192], vec![0u32; 128]);
        let clean = (i64::MAX, -1);
        core.quad.upload(&mut stage, &mut dirty);
        core.group.upload(&mut stage, &mut dirty);

        core.scroll_by(area, [0.0, 30.0]);
        assert_eq!(core.scroll_offset(area)[1], 30.0);
        assert_eq!(
            core.quad.upload(&mut stage, &mut dirty),
            clean,
            "scrolling must not rewrite a single quad"
        );
        assert_ne!(
            core.group.upload(&mut stage, &mut dirty),
            clean,
            "scrolling must move the content group"
        );
    }

    /// Scroll is clamped to the content extent taffy computed, in both
    /// directions.
    #[test]
    fn scroll_clamps_to_content() {
        let mut core = UiCore::new();
        let (area, _) = scroller(&mut core, 10); // 10 × 20px in a 50px box

        assert_eq!(core.max_scroll(area)[1], 150.0, "200px content in 50px box");
        core.scroll_by(area, [0.0, 1_000.0]);
        assert_eq!(core.scroll_offset(area)[1], 150.0, "clamped at the bottom");
        core.scroll_by(area, [0.0, -1_000.0]);
        assert_eq!(core.scroll_offset(area)[1], 0.0, "clamped at the top");
    }

    /// Hit testing follows the scroll: a row that scrolled out of the
    /// viewport must be unhittable, and the row now under the cursor must be
    /// the one that moved into place.
    #[test]
    fn hit_testing_follows_scroll_and_respects_the_clip() {
        let mut core = UiCore::new();
        let (area, rows) = scroller(&mut core, 10);

        // Rows are 20px tall in a 50px viewport at y=0.
        assert_eq!(core.hit_test([50.0, 10.0]), Some(rows[0]));
        assert_eq!(core.hit_test([50.0, 30.0]), Some(rows[1]));
        assert_eq!(core.hit_test([50.0, 70.0]), None, "below the viewport");

        core.scroll_by(area, [0.0, 20.0]);
        assert_eq!(
            core.hit_test([50.0, 10.0]),
            Some(rows[1]),
            "row 1 scrolled up into the cursor"
        );
        assert!(
            !core.hit_test([50.0, 10.0]).is_some_and(|h| h == rows[0]),
            "row 0 scrolled out of the clip"
        );
    }

    /// The wheel scrolls whatever is under the cursor, and the UI takes the
    /// pointer so the camera does not zoom at the same time.
    #[test]
    fn wheel_scrolls_the_area_under_the_cursor_and_captures() {
        let mut core = UiCore::new();
        let (area, _) = scroller(&mut core, 10);

        core.update_pointer([50.0, 25.0], false, false, -1.0);
        assert_eq!(core.scroll_offset(area)[1], WHEEL_PX, "wheel scrolled down");
        assert!(core.pointer_captured(), "a scroll area holds the pointer");

        // Off the area entirely: nothing scrolls, nothing captured.
        core.update_pointer([180.0, 180.0], false, false, -1.0);
        assert_eq!(core.scroll_offset(area)[1], WHEEL_PX, "unchanged");
        assert!(!core.pointer_captured());
    }

    /// The button owns its look: the engine restyles it on pointer
    /// transitions and, crucially, *only* on transitions — a frame where the
    /// pointer moved within the same node must not reach staging.
    #[test]
    fn state_style_applies_on_transition_and_is_free_otherwise() {
        use crate::ui::ButtonStyle;

        let mut core = UiCore::new();
        let root = core.root();
        let btn = core.button(root, "ok", ButtonStyle::default());
        core.set_node_style(btn, box_at(0.0, 0.0, 40.0, 20.0));
        core.run_layout([200.0, 100.0]);

        let (mut stage, mut dirty) = (vec![0u32; 4096], vec![0u32; 64]);
        let clean = (i64::MAX, -1);
        let inside = [10.0, 10.0];

        core.update_pointer([100.0, 80.0], false, false, 0.0);
        core.style.upload(&mut stage, &mut dirty);

        core.update_pointer(inside, false, false, 0.0);
        assert_ne!(
            core.style.upload(&mut stage, &mut dirty),
            clean,
            "entering the button must restyle it"
        );

        core.update_pointer([12.0, 12.0], false, false, 0.0);
        assert_eq!(
            core.style.upload(&mut stage, &mut dirty),
            clean,
            "moving within the same node is not a transition"
        );

        core.update_pointer(inside, true, false, 0.0);
        assert_ne!(
            core.style.upload(&mut stage, &mut dirty),
            clean,
            "pressing must restyle"
        );

        // Dragging off while held reverts to idle — releasing there cancels
        // the click, so it must not keep looking armed.
        core.update_pointer([100.0, 80.0], false, false, 0.0);
        assert_ne!(
            core.style.upload(&mut stage, &mut dirty),
            clean,
            "dragging off must restyle"
        );
        core.update_pointer([100.0, 80.0], false, false, 0.0);
        assert_eq!(
            core.style.upload(&mut stage, &mut dirty),
            clean,
            "settled off the button, nothing more to write"
        );
    }
}
