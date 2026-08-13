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

use std::any::Any;

use super::{font, Click, Grab, GroupId, Label, PrimId, TextId, UiCore, UiStyle};

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

use style::{px, Display, LengthPercentageAuto, Position, Rect, Style, TaffyAuto};

/// Pixels scrolled per wheel line.
const WHEEL_PX: f32 = 40.0;

/// Longest gap between two clicks that still reads as a double click.
const DOUBLE_CLICK_S: f64 = 0.4;

/// How far the pointer may travel between them; further means the user
/// re-aimed, whatever the clock says.
const DOUBLE_CLICK_SLOP: f32 = 5.0;

/// Where a drag ghost sits relative to the pointer. Down and to the right, so
/// it never covers whatever is being aimed at.
const GHOST_OFFSET: [f32; 2] = [14.0, 10.0];

/// Which pointer events a node accepts, as a bitmask.
///
/// One `bool` cannot answer "who takes the click?" and "who takes the drop?"
/// separately, and those genuinely differ: a checkbox inside a tree row wants
/// the click while the row wants the drop. Declaring per kind means the
/// engine never has to guess which of two nested nodes was meant — the
/// innermost node that *asked* for a kind gets it, and nesting two claimants
/// of the same kind is an author's decision rather than an ambiguity.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct Events(u8);

impl Events {
    pub const NONE: Self = Self(0);
    /// Press, click, and the origin of a drag.
    pub const CLICK: Self = Self(1);
    /// Pointer-state styling. Unlike the others this is a *set* — every
    /// accepting node on the way down is hovered, because they all are.
    pub const HOVER: Self = Self(1 << 1);
    /// Target for a released [`grab`](UiCore::grab).
    pub const DROP: Self = Self(1 << 2);
    /// Wheel target. Set by [`scroll_area`](UiCore::scroll_area) itself —
    /// a node scrolls because it has a content group, so nothing else may
    /// claim this and the two cannot disagree.
    pub const SCROLL: Self = Self(1 << 3);
    /// Takes keyboard focus when pressed, and is a stop on the Tab ring.
    /// Separate from [`CLICK`](Self::CLICK): a button is clickable but must
    /// not keep the keyboard afterwards.
    pub const FOCUS: Self = Self(1 << 4);

    pub fn has(self, bit: Self) -> bool {
        self.0 & bit.0 != 0
    }
}

impl std::ops::BitOr for Events {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Everything one hit walk resolves: the innermost node accepting each kind,
/// plus the full hover set.
#[derive(Default)]
struct Hits {
    click: Option<NodeId>,
    drop: Option<NodeId>,
    scroll: Option<NodeId>,
    focus: Option<NodeId>,
    /// Innermost first.
    hover: Vec<NodeId>,
}

fn claim(slot: &mut Option<NodeId>, m: Events, bit: Events, n: NodeId) -> bool {
    let take = m.has(bit) && slot.is_none();
    if take {
        *slot = Some(n);
    }
    take
}

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

/// A press-and-hold in progress, from [`UiCore::drag`].
///
/// Both points are absolute screen px. Keeping the *origin* rather than a
/// per-frame delta is what makes a drag threshold one comparison and lets a
/// slider map the pointer straight onto its track without integrating.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Drag {
    /// Where the press landed.
    pub origin: [f32; 2],
    /// Where the pointer is now.
    pub pos: [f32; 2],
}

impl Drag {
    /// Movement since the press.
    pub fn delta(self) -> [f32; 2] {
        [self.pos[0] - self.origin[0], self.pos[1] - self.origin[1]]
    }

    /// Whether the pointer has travelled far enough for this to read as a
    /// drag rather than a click that wobbled.
    pub fn beyond(self, threshold: f32) -> bool {
        let [dx, dy] = self.delta();
        dx * dx + dy * dy >= threshold * threshold
    }
}

/// A widget-tree node. Cheap and `Copy`; a stale one panics on use rather
/// than silently no-opping.
///
/// The `gen` half is what makes that promise real. [`UiCore::remove_node`]
/// recycles a node's slot, so an index alone would silently address whoever
/// moved in — the classic use-after-free that reads as a rendering bug three
/// screens away. Removal bumps the slot's generation, so every handle minted
/// before it mismatches and [`UiCore::live`] panics naming the slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId {
    pub(crate) idx: u32,
    pub(crate) gen: u32,
}

impl NodeId {
    /// Slot number, for printing in a log or a harness — never for
    /// addressing, which needs the generation too.
    pub fn index(self) -> u32 {
        self.idx
    }

    pub fn generation(self) -> u32 {
        self.gen
    }
}

struct Node {
    /// Bumped on free. Even while the slot is live, odd while it is not, so
    /// a handle to a freed-but-unreused slot mismatches too.
    gen: u32,
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
    fn new(gen: u32, taffy: taffy::NodeId, group: GroupId) -> Self {
        Self {
            gen,
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
    /// Slots freed by `remove_node`, reused newest-first. Recycling is the
    /// whole reason generations exist; without it they would guard nothing.
    free: Vec<u32>,
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
            nodes: vec![Node::new(0, root_taffy, group)],
            absolute: vec![[0.0; 4]],
            free: Vec::new(),
            root: NodeId { idx: 0, gen: 0 },
            screen: [0.0, 0.0],
        }
    }
}

impl UiCore {
    /// The screen-sized root every node descends from.
    pub fn root(&mut self) -> NodeId {
        self.tree.root
    }

    /// Resolve a handle to its slot, panicking if it is stale.
    ///
    /// The single enforcement point for generations. It is a panic and not
    /// an `Option` deliberately: a stale handle is a bug in the caller, and
    /// returning `None` would let it degrade into a widget that silently
    /// stops responding — the exact failure mode that is hard to trace back.
    ///
    /// **Every public method turns a caller's handle into an index through
    /// here, and nothing else does.** `n.idx as usize` is correct only for a
    /// node the engine already reached from the root — the placement, hit and
    /// scroll walks, and `free_subtree`'s recursion — or for the raw
    /// index-keyed side tables (`interactive`, `state_styles`, `controls`,
    /// the `Pointer` fields), which `free_subtree` clears precisely so an
    /// unchecked index there cannot address a recycled slot.
    #[inline]
    pub(crate) fn live(&self, n: impl Into<NodeId>) -> usize {
        let n = n.into();
        let node = self
            .tree
            .nodes
            .get(n.idx as usize)
            .unwrap_or_else(|| panic!("NodeId {} was never allocated", n.idx));
        assert_eq!(
            node.gen, n.gen,
            "stale NodeId {}: handle generation {} but slot is at {}",
            n.idx, n.gen, node.gen,
        );
        n.idx as usize
    }

    /// Remove a node and everything under it, freeing its slots for reuse.
    ///
    /// Every handle into the removed subtree becomes stale and panics on
    /// next use, rather than addressing whatever moves into the slot.
    ///
    /// Panics on the root, which is the tree itself.
    pub fn remove_node(&mut self, n: impl Into<NodeId>) {
        let n = n.into();
        assert_ne!(n, self.tree.root, "the UI root cannot be removed");
        let idx = self.live(n);

        // Detach from the parent first, so the recursive free never walks
        // back into a node it has already released.
        if let Some(p) = self.parent_of(idx) {
            self.tree.nodes[p].children.retain(|c| *c != n);
        }
        let taffy_id = self.tree.nodes[idx].taffy;
        self.tree.taffy.remove(taffy_id).expect("taffy remove");
        self.free_subtree(idx);
    }

    /// Release `idx` and its descendants: primitives back to the slot
    /// allocator, node slots onto the free list with their generation
    /// bumped odd.
    fn free_subtree(&mut self, idx: usize) {
        for c in std::mem::take(&mut self.tree.nodes[idx].children) {
            self.free_subtree(c.idx as usize);
        }
        if let Some(p) = self.tree.nodes[idx].background.take() {
            self.free(p);
        }
        if let Some(t) = self.tree.nodes[idx].text.take() {
            self.free_text(t);
        }
        self.tree.nodes[idx].gen += 1;
        self.tree.absolute[idx] = [0.0; 4];
        if let Some(s) = self.state_styles.get_mut(idx) {
            *s = None;
        }
        if let Some(c) = self.controls.get_mut(idx) {
            *c = None;
        }
        // A scrollbar is the one widget that names a node outside itself, so
        // it dies with either end — otherwise the sync would reach through a
        // handle whose generation has moved on.
        let mut bars = std::mem::take(&mut self.scrollbars);
        bars.retain(|&b| NodeId::from(b).idx as usize != idx && self.bar_area(b).idx as usize != idx);
        self.scrollbars = bars;
        // Everything keyed by raw index has to be cleared, not just the
        // generation-checked state: whoever recycles this slot would
        // otherwise inherit a hit target it never asked for.
        if let Some(e) = self.pointer.listens.get_mut(idx) {
            *e = Events::NONE;
        }
        for set in [&mut self.pointer.hover, &mut self.pointer.hover_prev] {
            set.retain(|n| n.idx as usize != idx);
        }
        for p in [
            &mut self.pointer.down_on,
            &mut self.pointer.clicked,
            &mut self.pointer.dropped,
        ] {
            if p.is_some_and(|n| n.idx as usize == idx) {
                *p = None;
            }
        }
        // A drop whose target is being torn down the same frame it landed:
        // the payload goes with it rather than being offered to whoever
        // recycles the slot.
        if self.pointer.drop.as_ref().is_some_and(|(t, _)| t.idx as usize == idx) {
            self.pointer.drop = None;
        }
        // A removed field must not leave the keyboard on a recycled slot.
        self.forget_focus(idx);
        self.tree.free.push(idx as u32);
    }

    /// Linear search for a node's parent. Nodes carry no parent link, and a
    /// removal is rare enough that adding one to every node — and keeping it
    /// correct through re-parenting — would cost more than it saves.
    fn parent_of(&self, idx: usize) -> Option<usize> {
        let me = self.tree.nodes[idx].gen;
        (0..self.tree.nodes.len()).find(|&p| {
            self.tree.nodes[p]
                .children
                .iter()
                .any(|c| c.idx as usize == idx && c.gen == me)
        })
    }

    /// Add a container. Style it with the re-exported taffy vocabulary:
    /// `Style { display: Display::Flex, flex_direction: FlexDirection::Column,
    /// gap: Size { width: length(0.0), height: length(6.0) }, .. }`.
    pub fn node(&mut self, parent: impl Into<NodeId>, style: Style) -> NodeId {
        let parent = parent.into();
        // A scroll area hands its children the content group, not its own —
        // that one line is what puts everything inside it under the scrolled
        // offset without any node knowing it is being scrolled.
        let pi = self.live(parent);
        let p = &self.tree.nodes[pi];
        let group = p.content_group.unwrap_or(p.group);
        let taffy_id = self.tree.taffy.new_leaf(style).expect("taffy new_leaf");
        // Reuse a freed slot if there is one. Its generation is odd (freed),
        // so bumping to even both marks it live and invalidates every handle
        // minted against the previous occupant.
        let id = match self.tree.free.pop() {
            Some(idx) => {
                let gen = self.tree.nodes[idx as usize].gen + 1;
                self.tree.nodes[idx as usize] = Node::new(gen, taffy_id, group);
                self.tree.absolute[idx as usize] = [0.0; 4];
                NodeId { idx, gen }
            }
            None => {
                let idx = self.tree.nodes.len() as u32;
                self.tree.nodes.push(Node::new(0, taffy_id, group));
                self.tree.absolute.push([0.0; 4]);
                NodeId { idx, gen: 0 }
            }
        };
        self.tree.nodes[pi].children.push(id);
        let parent_taffy = self.tree.nodes[pi].taffy;
        self.tree
            .taffy
            .add_child(parent_taffy, taffy_id)
            .expect("taffy add_child");
        id
    }

    /// Pick something up. Returns the **ghost**: an empty node at the
    /// pointer, for the caller to fill with whatever the thing looks like.
    ///
    /// This is the whole of drag-and-drop's source side. The pointer layer
    /// owns the parts every drag gets wrong on its own — the ghost hangs off
    /// the root so no scroll area's clip can cut it off, it tracks the pointer
    /// every frame, and it is freed on release — while the caller owns
    /// appearance and, in `payload`, meaning.
    ///
    /// The payload is any `Send + 'static` value and `UiCore` never looks
    /// inside it; a target asks [`dropped_on`](Self::dropped_on) for the type
    /// it accepts and gets `None` if this drag is not for it. That refusal is
    /// the question being answered, not a failure being swallowed: the drop
    /// simply does not happen.
    ///
    /// No [`raise`](Self::raise) is needed, and that is not luck — the ghost
    /// is minted *at the grab*, so it is already the last child of the root
    /// and paints over everything built before it, which is everything.
    ///
    /// Panics if a drag is already in flight. There is one pointer, so a
    /// second grab means a widget missed a release.
    pub fn grab<T: Any + Send>(&mut self, payload: T) -> NodeId {
        assert!(self.pointer.grab.is_none(), "a grab began while one was in flight");
        let root = self.tree.root;
        let ghost = self.node(root, Style::default());
        self.place_ghost(ghost, self.pointer.pos);
        self.pointer.grab = Some(Grab { payload: Box::new(payload), ghost });
        ghost
    }

    /// What the pointer is carrying, if it is carrying a `T`. `None` when
    /// nothing is in flight *or* the drag is somebody else's kind of thing —
    /// which is how a panel lights up only for drops it can accept.
    pub fn dragging<T: Any>(&self) -> Option<&T> {
        self.pointer.grab.as_ref()?.payload.downcast_ref()
    }

    /// The ghost of the drag in flight, for a caller that wants to restyle it
    /// mid-gesture — "move" versus "copy", or a refusal.
    pub fn ghost(&self) -> Option<NodeId> {
        self.pointer.grab.as_ref().map(|g| g.ghost)
    }

    /// A `T` that landed on `n` this frame.
    ///
    /// The target half of [`dropped`](Self::dropped), and the reason the two
    /// both exist: `dropped` tells the node a gesture *started* on that it is
    /// over, while this tells the node the gesture *ended* on what arrived.
    /// Only the node under the pointer at release hears it, so a drag can
    /// cross from one panel to another with neither knowing about the other.
    pub fn dropped_on<T: Any>(&self, n: impl Into<NodeId>) -> Option<&T> {
        let n = n.into();
        let (target, payload) = self.pointer.drop.as_ref()?;
        (*target == n).then(|| payload.downcast_ref())?
    }

    /// Park the ghost at the pointer. Position is the engine's — the caller
    /// styles everything else, so only `position` and `inset` are overwritten
    /// and the padding and size it chose survive the frame.
    fn place_ghost(&mut self, ghost: NodeId, pos: [f32; 2]) {
        let mut style = self.node_style(ghost);
        style.position = Position::Absolute;
        style.inset = Rect {
            left: px(pos[0] + GHOST_OFFSET[0]),
            top: px(pos[1] + GHOST_OFFSET[1]),
            right: LengthPercentageAuto::AUTO,
            bottom: LengthPercentageAuto::AUTO,
        };
        self.set_node_style(ghost, style);
    }

    /// Move `n` to the end of its parent's children, so it paints over its
    /// siblings.
    ///
    /// Paint order is tree order, so this is the whole of z-order: a node
    /// built before the panels it must float above — a drag ghost, a drop
    /// marker, a menu — cannot be re-ordered any other way. Taffy's child
    /// list moves with it so the two cannot disagree; for the absolutely
    /// positioned nodes that want this, the move changes no geometry.
    pub fn raise(&mut self, n: impl Into<NodeId>) {
        let n = n.into();
        let p = self.parent_of(self.live(n)).expect("the root has nothing to rise above");
        self.attach(n, p);
    }

    /// Move a live subtree under a new parent, keeping everything it owns —
    /// its primitives' slots, its controls' values, its scroll offsets.
    ///
    /// This is what a rebuild cannot stand in for, and what docking is: a
    /// panel dragged into another split is re-laid-out, not torn down and
    /// re-made, so the field halfway through being typed into survives the
    /// move. It lands last, so it also paints last among its new siblings.
    ///
    /// Panics if the two parents sit in different clip groups. A primitive's
    /// group is fixed when its slot is allocated, so crossing that boundary
    /// means rebuilding the subtree rather than moving it — the same
    /// limitation, for the same reason, as nested scroll areas.
    pub fn set_parent(&mut self, n: impl Into<NodeId>, parent: impl Into<NodeId>) {
        let (n, parent) = (n.into(), parent.into());
        let (child, pi) = (self.live(n), self.live(parent));
        let p = &self.tree.nodes[pi];
        assert_eq!(
            p.content_group.unwrap_or(p.group),
            self.tree.nodes[child].group,
            "moving a subtree across a clip group",
        );
        self.attach(n, pi);
    }

    /// Detach `n` from wherever it is and append it under `pi`. Taffy's child
    /// list moves in step, so the two can never disagree about order.
    fn attach(&mut self, n: NodeId, pi: usize) {
        let idx = self.live(n);
        let old = self.parent_of(idx).expect("the root has no parent to leave");
        self.tree.nodes[old].children.retain(|c| *c != n);
        self.tree.nodes[pi].children.push(n);
        let (from, to, child) = (
            self.tree.nodes[old].taffy,
            self.tree.nodes[pi].taffy,
            self.tree.nodes[idx].taffy,
        );
        self.tree.taffy.remove_child(from, child).expect("taffy remove_child");
        self.tree.taffy.add_child(to, child).expect("taffy add_child");
        self.order_dirty = true;
    }

    /// Give a node a filled / bordered rect covering its whole box. Called
    /// again on the same node, it restyles in place — one dirty `ui_style`
    /// word, no layout at all, which is what makes hover cheap.
    pub fn set_background(&mut self, n: impl Into<NodeId>, style: UiStyle) {
        let n = n.into();
        match self.tree.nodes[self.live(n)].background {
            Some(p) => self.set_style(p, style),
            None => {
                let group = self.tree.nodes[self.live(n)].group;
                let p = self.rect(group, [0.0; 4], style);
                let i = self.live(n);
                self.tree.nodes[i].background = Some(p);
            }
        }
    }

    /// An image leaf: a node whose background samples a texture instead of
    /// filling. `style` gives it its box — an image has no natural size here,
    /// because the store that knows the texture's dimensions is on the other
    /// side of the GPU boundary `UiCore` deliberately does not cross.
    ///
    /// It rides the background slot, so the placement walk already sizes it,
    /// hides it with its node and paints it in tree order; `set_background`
    /// on the same node turns it back into a fill.
    pub fn image(&mut self, parent: impl Into<NodeId>, tex: u32, style: Style) -> NodeId {
        let n = self.node(parent, style);
        self.set_background(n, UiStyle::image(tex));
        self.set_image_uv(n, [0.0, 0.0, 1.0, 1.0]);
        n
    }

    /// Which part of the texture an image leaf shows, in `0..1`
    /// `[u0, v0, u1, v1]`. Defaults to all of it.
    ///
    /// The same knob text already uses to pick a glyph out of the atlas —
    /// which is what an icon will want, since icons arrive packed. Clipping
    /// interpolates against the *unclipped* box, so a half-covered image
    /// still shows its correct half.
    pub fn set_image_uv(&mut self, n: impl Into<NodeId>, uv: [f32; 4]) {
        let n = n.into();
        let Some(p) = self.tree.nodes[self.live(n)].background else {
            panic!("set_image_uv on a node with no background");
        };
        let mut q = self.quad.get(p.0);
        q.uv = uv;
        self.quad.set(p.0, q);
    }

    /// A text leaf. Its natural size is handed to taffy as a measured leaf,
    /// so it participates in flex and grid sizing like any other box.
    pub fn label(&mut self, parent: impl Into<NodeId>, px: f32, color: u32, text: &str) -> Label {
        let n = self.node(parent, Style::default());
        let group = self.tree.nodes[self.live(n)].group;
        let t = self.text(group, [0.0, 0.0], px, color, text);
        let i = self.live(n);
        self.tree.nodes[i].text = Some(t);
        self.measure_label(n, text, px);
        Label::from_node(n)
    }

    /// Retype a label. Unchanged text returns before touching anything;
    /// changed text dirties only the glyphs that differ, and re-measures
    /// only if the string's width actually moved.
    pub(crate) fn set_label(&mut self, n: impl Into<NodeId>, text: &str) {
        let n = n.into();
        let Some(t) = self.tree.nodes[self.live(n)].text else {
            panic!("set_label on a node with no text");
        };
        if self.text_of(t) == text {
            return;
        }
        let px = self.text_px(t);
        self.set_text(t, text);
        self.measure_label(n, text, px);
    }

    pub(crate) fn set_label_color(&mut self, n: impl Into<NodeId>, color: u32) {
        let n = n.into();
        let Some(t) = self.tree.nodes[self.live(n)].text else {
            panic!("set_label_color on a node with no text");
        };
        self.set_text_color(t, color);
    }

    /// The node's current layout style, for read-modify-write edits.
    pub fn node_style(&self, n: impl Into<NodeId>) -> Style {
        let n = n.into();
        self.tree
            .taffy
            .style(self.tree.nodes[self.live(n)].taffy)
            .expect("taffy style")
            .clone()
    }

    /// Restyle a node's box. Marks it and its ancestors dirty; the next
    /// `run_layout` recomputes that path and nothing else.
    pub fn set_node_style(&mut self, n: impl Into<NodeId>, style: Style) {
        let n = n.into();
        let taffy_id = self.tree.nodes[self.live(n)].taffy;
        if self.tree.taffy.style(taffy_id).expect("taffy style") == &style {
            return;
        }
        self.tree
            .taffy
            .set_style(taffy_id, style)
            .expect("taffy set_style");
    }

    /// Collapse a node out of the layout, or put it back. Hiding zeroes its
    /// box *and its descendants'*, so a hidden subtree paints nothing and
    /// takes no hits — which is how tabs swap panes.
    ///
    /// Restores `Display::Flex`, the only display any widget here builds with.
    pub fn set_visible(&mut self, n: impl Into<NodeId>, visible: bool) {
        let n = n.into();
        let mut s = self.node_style(n);
        s.display = if visible { Display::Flex } else { Display::None };
        self.set_node_style(n, s);
    }

    /// The node's computed box, absolute in screen px. Valid after
    /// `run_layout`; this is what hit testing and splitter drags read.
    pub fn node_rect(&self, n: impl Into<NodeId>) -> [f32; 4] {
        let n = n.into();
        self.tree.absolute[self.live(n)]
    }

    /// The node's glyph run, if it has one.
    pub(crate) fn text_id(&self, n: impl Into<NodeId>) -> Option<TextId> {
        let n = n.into();
        self.tree.nodes[self.live(n)].text
    }

    /// The node's current label text, if it has one.
    pub(crate) fn node_text(&self, n: impl Into<NodeId>) -> Option<&str> {
        let n = n.into();
        self.tree.nodes[self.live(n)].text.map(|t| self.text_of(t))
    }

    /// First slot of the node's background and of its glyph run. Feed either
    /// to [`paint_index`](Self::paint_index) to ask what covers what.
    pub(crate) fn paint_slots(&self, n: impl Into<NodeId>) -> (Option<u32>, Option<u32>) {
        let n = n.into();
        let node = &self.tree.nodes[self.live(n)];
        (
            node.background.map(|p| p.0),
            node.text.map(|t| self.runs[t.0 as usize].first),
        )
    }

    /// Where a slot sits in the draw list. Higher paints later, so higher
    /// covers lower — the only ordering question worth asking, and one the
    /// slot number cannot answer since `place` assigns paint order from the
    /// tree rather than from the allocator.
    pub(crate) fn paint_index(&self, slot: u32) -> usize {
        (0..self.prim_count())
            .find(|&i| self.order.get(i).0[0] == slot)
            .expect("every slot must appear in the draw list") as usize
    }

    /// Publish a label's natural size to taffy, but only when it changed —
    /// `set_node_context` unconditionally marks dirty, so an unguarded call
    /// would force a relayout on every keystroke that kept the width.
    fn measure_label(&mut self, n: NodeId, text: &str, px: f32) {
        let scale = px / font::GLYPH_H as f32;
        let size = [font::text_width(text) as f32 * scale, px];
        let taffy_id = self.tree.nodes[self.live(n)].taffy;
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

        let root_taffy = self.tree.nodes[self.tree.root.idx as usize].taffy;
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
        // A node that lost its box cannot keep the keyboard — collapsing a
        // panel has to hand hotkeys back, and nothing else reports that.
        if let Some(n) = self.keyboard.focus {
            let r = self.tree.absolute[n.idx as usize];
            if r[2] == 0.0 && r[3] == 0.0 {
                self.set_focus(None);
            }
        }
        // Boxes moved, so whatever is under the pointer may have changed even
        // if the pointer did not. Reached only when something was actually
        // dirty — the early-out above is what keeps an idle frame idle.
        self.layout_epoch += 1;
        // A bar's thumb is sized by the boxes just solved, so it is fitted
        // here rather than by the caller. Resizing it dirties taffy, so a
        // changed extent settles on the next frame; an unchanged one is a
        // comparison, which is why this does not loop.
        self.sync_scrollbars();
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
    pub fn scroll_area(&mut self, parent: impl Into<NodeId>, style: Style) -> NodeId {
        let parent = parent.into();
        use crate::ui::style::{Overflow, Point};

        let p = &self.tree.nodes[self.live(parent)];
        let inherited = p.content_group.unwrap_or(p.group);
        assert_eq!(
            inherited,
            self.tree.nodes[self.tree.root.idx as usize].group,
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
        self.open_content_group(n);
        // Set here, three lines from the content group it means, so the two
        // cannot drift: a node is a wheel target exactly because it scrolls.
        // Deliberately not `HOVER` or `CLICK` — the wheel should reach a list
        // whether or not anything in it is clickable.
        self.set_events(n, Events::SCROLL);
        n
    }

    /// Give a node a group of its own, which its children inherit: clipped to
    /// its box, translated by [`set_content_offset`](Self::set_content_offset).
    /// What makes a scroll area scroll, and what moves a scrollbar's thumb.
    pub(crate) fn open_content_group(&mut self, n: impl Into<NodeId>) -> GroupId {
        let g = self.group([0.0; 4], [0.0; 2]);
        let i = self.live(n);
        self.tree.nodes[i].content_group = Some(g);
        g
    }

    /// How far this area can scroll on each axis, from taffy's content size.
    pub fn max_scroll(&self, n: impl Into<NodeId>) -> [f32; 2] {
        let n = n.into();
        let l = self
            .tree
            .taffy
            .layout(self.tree.nodes[self.live(n)].taffy)
            .expect("taffy layout");
        [l.scroll_width(), l.scroll_height()]
    }

    pub fn scroll_offset(&self, n: impl Into<NodeId>) -> [f32; 2] {
        let n = n.into();
        self.tree.nodes[self.live(n)].scroll
    }

    /// Scroll by `delta`, clamped to the content extent.
    ///
    /// **This is the payoff**: it writes one `ui_group` record and touches no
    /// quads and no layout, however many primitives are inside. A 10 000-row
    /// list scrolls for the same 32 bytes as an empty one.
    pub fn scroll_by(&mut self, n: impl Into<NodeId>, delta: [f32; 2]) {
        let n = n.into();
        let old = self.tree.nodes[self.live(n)].scroll;
        let max = self.max_scroll(n);
        self.set_content_offset(
            n,
            [
                (old[0] + delta[0]).clamp(0.0, max[0]),
                (old[1] + delta[1]).clamp(0.0, max[1]),
            ],
        );
        self.sync_scrollbars();
    }

    /// Translate a node's content group, unclamped. The one record a scroll
    /// writes, reached directly by the scrollbar thumb — whose offset is a
    /// position rather than a scroll, so it is negative and out of range.
    pub(crate) fn set_content_offset(&mut self, n: impl Into<NodeId>, offset: [f32; 2]) {
        let idx = self.live(n);
        let g = self.tree.nodes[idx]
            .content_group
            .expect("set_content_offset on a node with no content group");
        if self.tree.nodes[idx].scroll == offset {
            return;
        }
        self.tree.nodes[idx].scroll = offset;
        // Content moved under a pointer that need not have, and no relayout
        // is involved — the hit walk reads `scroll` directly through
        // `group_context`. Past the no-op guard above, so a re-clamp that
        // changes nothing costs nothing.
        self.layout_epoch += 1;
        let base = self.tree.nodes[idx].scroll_base;
        self.set_group_offset(g, [base[0] - offset[0], base[1] - offset[1]]);
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

    /// Declare which pointer events a node accepts. Nodes are inert by
    /// default, so a panel's decorative boxes never swallow a click meant for
    /// the world behind it.
    ///
    /// Opting in per *kind* is what lets a checkbox inside a tree row take the
    /// click while the row still takes the drop: they are different questions
    /// with different answers, and each node says which it is answering.
    pub fn set_events(&mut self, n: impl Into<NodeId>, events: Events) {
        let idx = self.live(n);
        let p = &mut self.pointer.listens;
        if p.len() <= idx {
            p.resize(idx + 1, Events::NONE);
        }
        p[idx] = events;
    }

    /// The innermost node accepting [`Events::CLICK`] at `p`.
    pub fn hit_test(&self, p: [f32; 2]) -> Option<NodeId> {
        let mut hits = Hits::default();
        self.hit(&mut hits, self.tree.root, p, [0.0, 0.0], self.screen_rect());
        hits.click
    }

    /// One walk, every target.
    ///
    /// **Reverse paint order, no pruning.** `place` writes a node before its
    /// children and children in order, so paint order is the DFS preorder and
    /// the topmost node is the *last* match — hence children reversed, before
    /// self. Deliberately not pruned on the parent's box: nothing clips
    /// per-node today (clipping is per `ui_group`), so pruning would invent a
    /// containment rule the renderer does not honour and would silently
    /// mis-hit any absolutely-positioned child that escapes its parent.
    /// Pruning becomes correct — and worth it — when nodes get real clip
    /// rects.
    ///
    /// Mirrors `place`'s group context too, so a row scrolled out of its
    /// viewport is unhittable for the same reason it is invisible rather than
    /// by a second rule that could drift from the first.
    ///
    /// O(nodes), but it runs on pointer events, not per frame — and it runs
    /// **once** for all four kinds. Each unclaimed kind is taken on the way
    /// out, and the DFS unwinds innermost-first, so the first claimant of a
    /// kind is the innermost node accepting it.
    ///
    /// Returns whether anything under `n` claimed a kind, which is what stops
    /// the sibling scan: a subtree that took something occludes what is
    /// painted beneath it. A node that accepts nothing blocks nothing, so
    /// decoration stays transparent to the pointer exactly as before.
    fn hit(
        &self,
        out: &mut Hits,
        n: NodeId,
        p: [f32; 2],
        offset: [f32; 2],
        clip: [f32; 4],
    ) -> bool {
        let idx = n.idx as usize;
        let (child_offset, child_clip) = self.group_context(idx, offset, clip);
        let mut claimed = false;
        for i in (0..self.tree.nodes[idx].children.len()).rev() {
            if self.hit(out, self.tree.nodes[idx].children[i], p, child_offset, child_clip) {
                claimed = true;
                break;
            }
        }
        if !contains(screen_rect(self.tree.absolute[idx], offset), p) || !contains(clip, p) {
            return claimed;
        }

        let m = self.pointer.listens.get(idx).copied().unwrap_or(Events::NONE);
        claimed |= claim(&mut out.click, m, Events::CLICK, n);
        claimed |= claim(&mut out.drop, m, Events::DROP, n);
        claimed |= claim(&mut out.scroll, m, Events::SCROLL, n);
        claimed |= claim(&mut out.focus, m, Events::FOCUS, n);
        // Hover is a *set*, not a winner: a row and the checkbox inside it
        // both light up, and the pointer is genuinely over both. Innermost
        // first, since that is the order the walk unwinds in.
        if m.has(Events::HOVER) {
            out.hover.push(n);
            claimed = true;
        }
        claimed
    }

    fn screen_rect(&self) -> [f32; 4] {
        [0.0, 0.0, self.tree.screen[0], self.tree.screen[1]]
    }

    /// Every node with visible text, as `(node, text, on-screen rect)`.
    ///
    /// The rect is where the pointer must go: group offsets applied and
    /// clipped, exactly as the hit walk resolves them. For the debug socket
    /// and for tests, so neither has to re-derive the layout.
    pub fn text_nodes(&self) -> Vec<(NodeId, String, [f32; 4])> {
        let mut out = Vec::new();
        self.collect_text(self.tree.root, [0.0, 0.0], self.screen_rect(), &mut out);
        out
    }

    fn collect_text(
        &self,
        n: NodeId,
        offset: [f32; 2],
        clip: [f32; 4],
        out: &mut Vec<(NodeId, String, [f32; 4])>,
    ) {
        let idx = n.idx as usize;
        let r = self.tree.absolute[idx];
        if n != self.tree.root && r[2] == 0.0 && r[3] == 0.0 {
            return; // collapsed: its descendants' boxes are stale
        }
        if let Some(t) = self.tree.nodes[idx].text {
            let v = intersect(screen_rect(r, offset), clip);
            if v[2] > v[0] && v[3] > v[1] {
                let text = self.text_of(t).to_string();
                if !text.is_empty() {
                    out.push((n, text, [v[0], v[1], v[2] - v[0], v[3] - v[1]]));
                }
            }
        }
        let (child_offset, child_clip) = self.group_context(idx, offset, clip);
        for i in 0..self.tree.nodes[idx].children.len() {
            self.collect_text(self.tree.nodes[idx].children[i], child_offset, child_clip, out);
        }
    }

    /// Every node accepting [`Events::FOCUS`], in tree order — the Tab ring.
    /// A collapsed subtree is skipped whole: its descendants keep stale
    /// layouts, the same reason [`place`](Self::place) propagates `hidden`.
    pub(crate) fn focus_order(&self) -> Vec<NodeId> {
        let mut out = Vec::new();
        self.collect_focusable(self.tree.root, &mut out);
        out
    }

    fn collect_focusable(&self, n: NodeId, out: &mut Vec<NodeId>) {
        let idx = n.idx as usize;
        let r = self.tree.absolute[idx];
        if n != self.tree.root && r[2] == 0.0 && r[3] == 0.0 {
            return;
        }
        if self
            .pointer
            .listens
            .get(idx)
            .is_some_and(|m| m.has(Events::FOCUS))
        {
            out.push(n);
        }
        for i in 0..self.tree.nodes[idx].children.len() {
            self.collect_focusable(self.tree.nodes[idx].children[i], out);
        }
    }

    /// Fold this frame's pointer into hover / press / click state. Called by
    /// the renderer before `World::sweep_all`, so components observe the same
    /// frame's input the `dt` they were handed belongs to.
    pub(crate) fn update_pointer(
        &mut self,
        pos: [f32; 2],
        pressed: bool,
        released: bool,
        wheel: f32,
        now: f64,
    ) {
        // All three last exactly one frame, so they clear even on the quiet
        // path below.
        self.pointer.clicked = None;
        self.pointer.dropped = None;
        self.pointer.drop = None;
        self.pointer.now = now;

        // Genuinely event-driven — but the event is not only the pointer's.
        // Asking "did the pointer move?" alone would miss a button animated
        // under a still cursor, or a panel toggled open beneath it: the
        // answer changed, the question did not. `layout_epoch` covers exactly
        // the two things that can move content, so this stays a skip on the
        // overwhelmingly common frame and never on a frame that mattered.
        let settled = self.layout_epoch == self.pointer.walked;
        if settled && pos == self.pointer.pos && !pressed && !released && wheel == 0.0 {
            return;
        }

        let was_down_on = self.pointer.down_on;
        self.pointer.pos = pos;

        // The previous set's buffer becomes this one's, and vice versa — two
        // vectors traded forever rather than an allocation per pointer event.
        let mut hits = Hits {
            hover: std::mem::take(&mut self.pointer.hover_prev),
            ..Default::default()
        };
        hits.hover.clear();
        let mut over_ui = self.hit(&mut hits, self.tree.root, pos, [0.0, 0.0], self.screen_rect());

        if wheel != 0.0 {
            if let Some(target) = hits.scroll {
                self.scroll_by(target, [0.0, -wheel * WHEEL_PX]);
                // Scrolling moved the content under a pointer that has not
                // moved, so the first walk's answers are already stale. Only
                // on wheel frames, and still one walk fewer than the two every
                // event used to cost.
                hits.hover.clear();
                hits = Hits { hover: hits.hover, ..Default::default() };
                over_ui = self.hit(&mut hits, self.tree.root, pos, [0.0, 0.0], self.screen_rect());
            }
        }

        let (hovered, drop_target, focus_target) = (hits.click, hits.drop, hits.focus);
        self.pointer.over_ui = over_ui;
        self.pointer.walked = self.layout_epoch;
        self.pointer.walks += 1;
        self.pointer.hover_prev = std::mem::replace(&mut self.pointer.hover, hits.hover);

        if pressed {
            self.pointer.down_on = hovered;
            self.pointer.press_pos = pos;
            // On the press and unconditional, so a press on the world
            // dismisses a caret. Ahead of `drive_controls`, so the field is
            // focused before the same press places its caret.
            self.set_focus(focus_target);
        }
        // Captured before the release clears it, so a frame that both moves
        // and releases still commits the position it was released at.
        let dragging = self.pointer.down_on;
        if released {
            if self.pointer.down_on.is_some() && self.pointer.down_on == hovered {
                self.pointer.clicked = hovered;
                self.record_click(hovered.expect("a click names a node"), pos);
            }
            self.pointer.dropped = self.pointer.down_on;
            self.pointer.down_on = None;
            // The gesture is over, so the ghost goes now — but the payload
            // survives one frame, which is the frame a target reads it in.
            //
            // It lands on the innermost node accepting `DROP`, which is not
            // the node that took the click: a checkbox inside a row takes
            // clicks and declines drops, so the row still catches this.
            if let Some(g) = self.pointer.grab.take() {
                self.remove_node(g.ghost);
                self.pointer.drop = drop_target.map(|h| (h, g.payload));
            }
        }
        // The ghost follows the pointer rather than the layout: it is the
        // gesture made visible, and nothing in the tree positions it.
        if let Some(ghost) = self.pointer.grab.as_ref().map(|g| g.ghost) {
            self.place_ghost(ghost, pos);
        }

        // Restyle only what moved: the hover set the pointer left, the one it
        // entered, and the press. Both sets are ancestor chains, so this is
        // O(depth) per frame rather than O(widgets), and nothing at all on a
        // frame where the pointer sat still. Overlap between them needs no
        // dedup — `apply_state_style` ends in `set_background`, which the
        // equality gate makes idempotent. Indexed, because the walk it calls
        // needs `&mut self`.
        for i in 0..self.pointer.hover_prev.len() {
            self.apply_state_style(self.pointer.hover_prev[i]);
        }
        for i in 0..self.pointer.hover.len() {
            self.apply_state_style(self.pointer.hover[i]);
        }
        for n in [was_down_on, self.pointer.down_on].into_iter().flatten() {
            self.apply_state_style(n);
        }

        // Values move here, not in the application: a component that runs
        // after this reads a checkbox or slider that is already current.
        self.drive_controls(dragging, pressed);
    }

    /// Pointer is over this node.
    ///
    /// True for *every* node accepting [`Events::HOVER`] under the pointer,
    /// not only the innermost — a tree row stays hovered while the pointer is
    /// on the checkbox inside it, which is what the row's highlight means.
    /// The set is an ancestor chain, so this is a scan of a handful.
    pub fn hovered(&self, n: impl Into<NodeId>) -> bool {
        let n = n.into();
        self.pointer.hover.contains(&n)
    }

    /// Pointer went down on this node and has not been released. Still true
    /// while the pointer is dragged off, which is what lets a button render
    /// "armed" and still cancel.
    pub fn held(&self, n: impl Into<NodeId>) -> bool {
        let n = n.into();
        self.pointer.down_on == Some(n)
    }

    /// A full press-and-release completed on this node this frame.
    pub fn clicked(&self, n: impl Into<NodeId>) -> bool {
        let n = n.into();
        self.pointer.clicked == Some(n)
    }

    /// Fold a completed click into the streak the next one is measured
    /// against. The record is always replaced; only `count` carries history.
    fn record_click(&mut self, node: NodeId, pos: [f32; 2]) {
        let now = self.pointer.now;
        let count = match self.pointer.last_click {
            Some(c)
                if c.node == node
                    && now - c.time <= DOUBLE_CLICK_S
                    && (pos[0] - c.pos[0]).abs() <= DOUBLE_CLICK_SLOP
                    && (pos[1] - c.pos[1]).abs() <= DOUBLE_CLICK_SLOP =>
            {
                c.count + 1
            }
            _ => 1,
        };
        self.pointer.last_click = Some(Click { node, time: now, pos, count });
    }

    /// Clicks in an unbroken streak, on the frame the latest one lands; `0`
    /// otherwise, so it reads like [`clicked`](Self::clicked).
    pub fn click_count(&self, n: impl Into<NodeId>) -> u32 {
        let n = n.into();
        match self.pointer.clicked == Some(n) {
            true => self.pointer.last_click.map_or(0, |c| c.count),
            false => 0,
        }
    }

    /// Two clicks on this node, close together, this frame.
    ///
    /// The single click still fires: swallowing it would mean delaying every
    /// click by the double-click interval. A triple click reports `count == 3`
    /// and is not a second double click.
    pub fn double_clicked(&self, n: impl Into<NodeId>) -> bool {
        self.click_count(n) == 2
    }

    /// The in-flight drag that started on this node, if any.
    ///
    /// `Some` for as long as the press is held, including after the pointer
    /// leaves the node — a slider must keep tracking when the cursor runs
    /// past the end of its track, and a drag-and-drop gesture is *defined*
    /// by leaving where it started.
    pub fn drag(&self, n: impl Into<NodeId>) -> Option<Drag> {
        self.held(n).then(|| Drag {
            origin: self.pointer.press_pos,
            pos: self.pointer.pos,
        })
    }

    /// The drag that started on this node and ended this frame, wherever the
    /// pointer had got to. One frame, like [`clicked`](Self::clicked).
    ///
    /// This is the drop half of drag-and-drop, and it is why `clicked` cannot
    /// stand in: a drop lands somewhere *other* than the press, which is
    /// precisely the release `clicked` refuses. A release that never moved
    /// sets both — a click is a drop of zero length, and
    /// [`Drag::beyond`] is what tells them apart.
    pub fn dropped(&self, n: impl Into<NodeId>) -> Option<Drag> {
        let n = n.into();
        (self.pointer.dropped == Some(n)).then(|| Drag {
            origin: self.pointer.press_pos,
            pos: self.pointer.pos,
        })
    }

    /// The UI owns the pointer — it is over an interactive node or a scroll
    /// area, or a press that started on one is still in flight. Camera
    /// controllers and world-picking should sit out while this is true.
    ///
    /// Scroll areas count so the wheel zooms the camera or scrolls a list,
    /// never both.
    pub fn pointer_captured(&self) -> bool {
        self.pointer.over_ui || self.pointer.down_on.is_some()
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
        let idx = n.idx as usize;
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

    /// The image leaf's whole claim: it is a *background*, so the placement
    /// walk it shares with every other node sizes it, hides it and paints it
    /// in tree order — the widget adds a kind, a texture and a uv window and
    /// nothing else. The uv window is what the editor's viewport needs: the
    /// camera renders at window size and the panel shows its own part of it.
    #[test]
    fn an_image_is_a_background_that_samples() {
        let mut core = UiCore::new();
        let root = core.root();
        let panel = core.node(root, column(0.0, 0.0));
        let img = core.image(
            panel,
            7,
            Style {
                size: Size { width: px(40.0), height: px(20.0) },
                ..Default::default()
            },
        );
        core.run_layout([200.0, 100.0]);

        let p = core.paint_slots(img).0.expect("an image has a background");
        let (style, quad) = (core.style.get(p), core.quad.get(p));
        assert_eq!(style.kind_flags, crate::ui::KIND_IMAGE);
        assert_eq!(style.tex, 7);
        assert_eq!(quad.rect, core.node_rect(img), "sized by the layout, like any background");
        assert_eq!(quad.uv, [0.0, 0.0, 1.0, 1.0], "the whole texture by default");

        core.set_image_uv(img, [0.25, 0.5, 0.75, 1.0]);
        assert_eq!(core.quad.get(p).uv, [0.25, 0.5, 0.75, 1.0]);

        // And it hides with its node: a zero-area quad is culled in the
        // vertex stage, which is how every freed or hidden primitive works.
        core.set_visible(img, false);
        core.run_layout([200.0, 100.0]);
        assert_eq!(core.quad.get(p).rect, [0.0; 4]);
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
            core.paint_index(reused) > core.paint_index(bg),
            "a child's glyphs must paint over its ancestor's background"
        );
    }

    /// An overlay is built once, before the panels it has to float over.
    /// `raise` is the only thing that can put it back on top, and it has to
    /// move taffy's child list too or the two disagree about the tree.
    #[test]
    fn raising_a_node_puts_it_over_its_later_siblings() {
        let mut core = UiCore::new();
        let root = core.root();
        // In flow, so taffy's own child order is observable: if `raise` moved
        // one list and not the other, paint order and layout would disagree
        // about which node is last.
        let stack = core.node(root, column(0.0, 0.0));
        let first = core.label(stack, 9.0, WHITE, "first");
        let second = core.label(stack, 9.0, WHITE, "second");
        core.run_layout([200.0, 100.0]);

        let (a, b) = (
            core.paint_slots(first).1.unwrap(),
            core.paint_slots(second).1.unwrap(),
        );
        assert!(core.paint_index(a) < core.paint_index(b), "built first, painted under");
        assert!(core.node_rect(first)[1] < core.node_rect(second)[1]);

        core.raise(first);
        core.run_layout([200.0, 100.0]);
        assert!(core.paint_index(a) > core.paint_index(b), "raised above its sibling");
        assert!(core.node_rect(first)[1] > core.node_rect(second)[1], "and taffy agrees");
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
        core.set_events(plain, Events::CLICK | Events::HOVER);
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
        core.set_events(under, Events::CLICK | Events::HOVER);
        core.set_events(over, Events::CLICK | Events::HOVER);
        core.run_layout([200.0, 200.0]);

        assert_eq!(core.hit_test([25.0, 25.0]), Some(over), "last child paints on top");
        assert_eq!(core.hit_test([75.0, 75.0]), Some(under), "outside the top box");
    }

    /// The scene can move under a cursor that does not. A button animated
    /// into place, or a panel toggled open beneath the pointer, has to become
    /// hovered — waiting for the mouse to be jiggled is the bug.
    #[test]
    fn content_moving_under_a_still_pointer_updates_hover() {
        let mut core = UiCore::new();
        let root = core.root();
        let panel = core.node(root, box_at(0.0, 0.0, 50.0, 50.0));
        core.set_events(panel, Events::CLICK | Events::HOVER);
        core.run_layout([200.0, 200.0]);

        let p = [100.0, 10.0];
        core.update_pointer(p, false, false, 0.0, 0.0);
        assert!(!core.hovered(panel), "starts well clear of it");

        // The pointer is not touched again from here.
        core.set_node_style(panel, box_at(80.0, 0.0, 50.0, 50.0));
        core.run_layout([200.0, 200.0]);
        core.update_pointer(p, false, false, 0.0, 0.0);
        assert!(core.hovered(panel), "it moved under the cursor");

        // And the panel-toggle case: hiding it releases the pointer without
        // a mouse
        // event of any kind.
        let mut s = core.node_style(panel);
        s.display = Display::None;
        core.set_node_style(panel, s);
        core.run_layout([200.0, 200.0]);
        core.update_pointer(p, false, false, 0.0, 0.0);
        assert!(!core.pointer_captured(), "a hidden panel captures nothing");
    }

    /// A programmatic scroll moves rows under a still pointer without any
    /// relayout at all — the walk reads `scroll` straight out of the node, so
    /// a layout-only epoch would miss it.
    #[test]
    fn scrolling_under_a_still_pointer_updates_hover() {
        let mut core = UiCore::new();
        let root = core.root();
        let area = core.scroll_area(root, box_at(0.0, 0.0, 100.0, 100.0));
        let sizer = core.node(area, box_at(0.0, 0.0, 100.0, 400.0));
        let far = core.node(area, box_at(0.0, 200.0, 100.0, 20.0));
        core.set_events(far, Events::CLICK | Events::HOVER);
        let _ = sizer;
        core.run_layout([200.0, 200.0]);

        let p = [50.0, 10.0];
        core.update_pointer(p, false, false, 0.0, 0.0);
        assert!(!core.hovered(far), "it is 200px down the content");

        core.scroll_by(area, [0.0, 200.0]);
        core.update_pointer(p, false, false, 0.0, 0.0);
        assert!(core.hovered(far), "scrolled up under the cursor");
    }

    /// The other half: an idle frame must still cost nothing. The epoch is
    /// exact rather than a "relayout maybe happened" heuristic, so a settled
    /// UI keeps skipping the walk entirely — which is the property the whole
    /// event-driven design rests on.
    #[test]
    fn a_settled_ui_still_skips_the_walk() {
        let mut core = UiCore::new();
        let root = core.root();
        let panel = core.node(root, box_at(0.0, 0.0, 50.0, 50.0));
        core.set_events(panel, Events::CLICK | Events::HOVER);
        core.run_layout([200.0, 200.0]);
        core.update_pointer([10.0, 10.0], false, false, 0.0, 0.0);

        let walks = core.pointer.walks;
        for _ in 0..100 {
            core.run_layout([200.0, 200.0]);
            core.update_pointer([10.0, 10.0], false, false, 0.0, 0.0);
        }
        assert_eq!(core.pointer.walks, walks, "100 idle frames, no walks");

        // A re-clamping scroll that moves nothing must not wake it either —
        // `RowList::sync` does exactly this on every frame.
        let area = core.scroll_area(root, box_at(0.0, 0.0, 100.0, 100.0));
        core.run_layout([200.0, 200.0]);
        core.update_pointer([10.0, 10.0], false, false, 0.0, 0.0);
        let walks = core.pointer.walks;
        for _ in 0..100 {
            core.scroll_by(area, [0.0, 0.0]);
            core.update_pointer([10.0, 10.0], false, false, 0.0, 0.0);
        }
        assert_eq!(core.pointer.walks, walks, "a zero scroll is not an event");
    }

    /// The headline of per-kind dispatch: a checkbox inside a drop zone takes
    /// the click and declines the drop, so both land where they should from
    /// one pointer position. Under a single "interactive" flag the innermost
    /// node won everything and the zone could never see the drop.
    #[test]
    fn a_child_can_take_the_click_and_leave_the_drop_to_its_zone() {
        let mut core = UiCore::new();
        let root = core.root();
        let zone = core.node(root, box_at(0.0, 0.0, 100.0, 40.0));
        let checkbox = core.node(zone, box_at(0.0, 0.0, 20.0, 20.0));
        core.set_events(zone, Events::CLICK | Events::HOVER | Events::DROP);
        core.set_events(checkbox, Events::CLICK);
        core.run_layout([200.0, 200.0]);

        // Over the checkbox: it is the innermost claimant of CLICK, the zone
        // the innermost of DROP.
        core.update_pointer([10.0, 10.0], false, false, 0.0, 0.0);
        assert_eq!(core.hit_test([10.0, 10.0]), Some(checkbox));
        assert!(core.hovered(zone), "the zone is still under the pointer");
        assert!(!core.hovered(checkbox), "and it never asked to be hovered");

        core.grab(7usize);
        core.update_pointer([10.0, 10.0], false, true, 0.0, 0.0);
        assert_eq!(core.dropped_on::<usize>(zone), Some(&7));
        assert_eq!(core.dropped_on::<usize>(checkbox), None, "it declined drops");
    }

    /// A node on top blocks the kinds it does *not* accept, too. Otherwise a
    /// drop released on a panel would fall through to a zone painted beneath
    /// it, because that zone is the innermost — and only — claimant of `DROP`.
    ///
    /// The rule is per subtree rather than per kind: whatever claimed
    /// something first occludes what is under it. A node accepting nothing
    /// still blocks nothing, so decoration stays transparent.
    #[test]
    fn a_node_on_top_blocks_kinds_it_does_not_itself_accept() {
        let mut core = UiCore::new();
        let root = core.root();
        let zone = core.node(root, box_at(0.0, 0.0, 100.0, 100.0));
        let panel = core.node(root, box_at(0.0, 0.0, 50.0, 50.0));
        core.set_events(zone, Events::DROP);
        core.set_events(panel, Events::CLICK);
        core.run_layout([200.0, 200.0]);

        core.grab(7usize);
        core.update_pointer([25.0, 25.0], false, true, 0.0, 0.0);
        assert_eq!(core.dropped_on::<usize>(zone), None, "the panel is in the way");

        // Clear of the panel, the same drop lands.
        core.grab(7usize);
        core.update_pointer([75.0, 75.0], false, false, 0.0, 0.0);
        core.update_pointer([75.0, 75.0], false, true, 0.0, 0.0);
        assert_eq!(core.dropped_on::<usize>(zone), Some(&7));
    }

    /// Hover is a set, not a winner. Both a row and a control inside it can
    /// light up, which one innermost answer cannot express — and which is why
    /// `RowList` no longer names its disclosure arrow to recover the row.
    #[test]
    fn hover_covers_every_accepting_node_under_the_pointer() {
        let mut core = UiCore::new();
        let root = core.root();
        let row = core.node(root, box_at(0.0, 0.0, 100.0, 40.0));
        let inner = core.node(row, box_at(0.0, 0.0, 20.0, 20.0));
        core.set_events(row, Events::HOVER);
        core.set_events(inner, Events::CLICK | Events::HOVER);
        core.run_layout([200.0, 200.0]);

        core.update_pointer([10.0, 10.0], false, false, 0.0, 0.0);
        assert!(core.hovered(row) && core.hovered(inner), "both, at once");

        // Off the inner box but still on the row.
        core.update_pointer([60.0, 10.0], false, false, 0.0, 0.0);
        assert!(core.hovered(row) && !core.hovered(inner));

        core.update_pointer([60.0, 90.0], false, false, 0.0, 0.0);
        assert!(!core.hovered(row) && !core.pointer_captured());
    }

    /// Two clicks close in time and space read as a double click — and the
    /// second still reports as an ordinary click, so "select on click,
    /// rename on double click" needs no arbitration between the two.
    #[test]
    fn two_quick_clicks_on_one_node_are_a_double_click() {
        let mut core = UiCore::new();
        let root = core.root();
        let row = core.node(root, box_at(0.0, 0.0, 40.0, 20.0));
        core.set_events(row, Events::CLICK);
        core.run_layout([200.0, 100.0]);

        let p = [10.0, 10.0];
        core.update_pointer(p, true, false, 0.0, 0.0);
        core.update_pointer(p, false, true, 0.0, 0.10);
        assert!(core.clicked(row));
        assert!(!core.double_clicked(row), "one click is not two");

        core.update_pointer(p, true, false, 0.0, 0.20);
        core.update_pointer(p, false, true, 0.0, 0.25);
        assert!(core.double_clicked(row));
        assert!(core.clicked(row), "the second click is still a click");

        core.update_pointer(p, false, false, 0.0, 0.30);
        assert!(!core.double_clicked(row), "one frame, like `clicked`");
    }

    /// A third click continues the streak rather than starting a second
    /// double click — otherwise a triple click would rename twice.
    #[test]
    fn a_third_click_is_not_a_second_double_click() {
        let mut core = UiCore::new();
        let root = core.root();
        let row = core.node(root, box_at(0.0, 0.0, 40.0, 20.0));
        core.set_events(row, Events::CLICK);
        core.run_layout([200.0, 100.0]);

        let p = [10.0, 10.0];
        for (i, t) in [0.0, 0.2, 0.4].into_iter().enumerate() {
            core.update_pointer(p, true, false, 0.0, t);
            core.update_pointer(p, false, true, 0.0, t + 0.05);
            assert_eq!(core.click_count(row), i as u32 + 1);
        }
        assert!(!core.double_clicked(row), "three is a triple, not a double");
    }

    /// Too slow, too far, or on something else — three ways for a second
    /// click to be a first one.
    #[test]
    fn a_double_click_needs_the_same_node_soon_and_nearby() {
        let mut core = UiCore::new();
        let root = core.root();
        let a = core.node(root, box_at(0.0, 0.0, 40.0, 20.0));
        let b = core.node(root, box_at(0.0, 40.0, 40.0, 20.0));
        core.set_events(a, Events::CLICK);
        core.set_events(b, Events::CLICK);
        core.run_layout([200.0, 100.0]);

        let click = |core: &mut UiCore, p, t: f64| {
            core.update_pointer(p, true, false, 0.0, t);
            core.update_pointer(p, false, true, 0.0, t);
        };

        // Same place, but a second apart.
        click(&mut core, [10.0, 10.0], 0.0);
        click(&mut core, [10.0, 10.0], 1.0);
        assert!(!core.double_clicked(a), "too slow");

        // Quick, but the pointer travelled across the node.
        click(&mut core, [10.0, 10.0], 2.0);
        click(&mut core, [35.0, 10.0], 2.1);
        assert!(!core.double_clicked(a), "re-aimed between clicks");

        // Quick and still, but on the neighbour.
        click(&mut core, [10.0, 10.0], 3.0);
        click(&mut core, [10.0, 50.0], 3.1);
        assert!(!core.double_clicked(b), "a different node starts over");
        assert!(!core.double_clicked(a), "and does not credit the first");
    }

    /// Press and release must land on the same node. Dragging off cancels —
    /// and `held` stays true throughout, which is what lets a button render
    /// "armed" while the pointer is away.
    #[test]
    fn click_requires_press_and_release_on_the_same_node() {
        let mut core = UiCore::new();
        let root = core.root();
        let btn = core.node(root, box_at(0.0, 0.0, 40.0, 20.0));
        core.set_events(btn, Events::CLICK | Events::HOVER);
        core.run_layout([200.0, 100.0]);

        let inside = [10.0, 10.0];
        let outside = [100.0, 80.0];

        // Press then release inside → one click, on exactly one frame.
        core.update_pointer(inside, true, false, 0.0, 0.0);
        assert!(core.held(btn) && !core.clicked(btn), "press alone is not a click");
        core.update_pointer(inside, false, true, 0.0, 0.0);
        assert!(core.clicked(btn), "press+release inside should click");
        core.update_pointer(inside, false, false, 0.0, 0.0);
        assert!(!core.clicked(btn), "click must last exactly one frame");

        // Press inside, drag off, release → no click.
        core.update_pointer(inside, true, false, 0.0, 0.0);
        core.update_pointer(outside, false, false, 0.0, 0.0);
        assert!(core.held(btn), "still armed while dragged off");
        assert!(!core.hovered(btn));
        core.update_pointer(outside, false, true, 0.0, 0.0);
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
        core.set_events(btn, Events::CLICK | Events::HOVER);
        core.run_layout([200.0, 100.0]);

        core.update_pointer([100.0, 80.0], false, false, 0.0, 0.0);
        assert!(!core.pointer_captured(), "idle over the world");

        core.update_pointer([10.0, 10.0], false, false, 0.0, 0.0);
        assert!(core.pointer_captured(), "hover captures");

        core.update_pointer([10.0, 10.0], true, false, 0.0, 0.0);
        core.update_pointer([100.0, 80.0], false, false, 0.0, 0.0);
        assert!(
            core.pointer_captured(),
            "a drag that started on the UI keeps the pointer even off-node"
        );

        core.update_pointer([100.0, 80.0], false, true, 0.0, 0.0);
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
                core.set_events(r, Events::CLICK | Events::HOVER);
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

        core.update_pointer([50.0, 25.0], false, false, -1.0, 0.0);
        assert_eq!(core.scroll_offset(area)[1], WHEEL_PX, "wheel scrolled down");
        assert!(core.pointer_captured(), "a scroll area holds the pointer");

        // Off the area entirely: nothing scrolls, nothing captured.
        core.update_pointer([180.0, 180.0], false, false, -1.0, 0.0);
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

        core.update_pointer([100.0, 80.0], false, false, 0.0, 0.0);
        core.style.upload(&mut stage, &mut dirty);

        core.update_pointer(inside, false, false, 0.0, 0.0);
        assert_ne!(
            core.style.upload(&mut stage, &mut dirty),
            clean,
            "entering the button must restyle it"
        );

        core.update_pointer([12.0, 12.0], false, false, 0.0, 0.0);
        assert_eq!(
            core.style.upload(&mut stage, &mut dirty),
            clean,
            "moving within the same node is not a transition"
        );

        core.update_pointer(inside, true, false, 0.0, 0.0);
        assert_ne!(
            core.style.upload(&mut stage, &mut dirty),
            clean,
            "pressing must restyle"
        );

        // Dragging off while held reverts to idle — releasing there cancels
        // the click, so it must not keep looking armed.
        core.update_pointer([100.0, 80.0], false, false, 0.0, 0.0);
        assert_ne!(
            core.style.upload(&mut stage, &mut dirty),
            clean,
            "dragging off must restyle"
        );
        core.update_pointer([100.0, 80.0], false, false, 0.0, 0.0);
        assert_eq!(
            core.style.upload(&mut stage, &mut dirty),
            clean,
            "settled off the button, nothing more to write"
        );
    }

    // ── Node removal and generations ────────────────────────────────

    /// The whole point: a slot is reused, and the handle that named its
    /// previous occupant is rejected rather than silently addressing the new
    /// one.
    #[test]
    fn a_recycled_slot_rejects_the_old_handle() {
        let mut core = UiCore::new();
        let root = core.root();
        let doomed = core.node(root, Style::default());
        core.remove_node(doomed);

        let fresh = core.node(root, Style::default());
        assert_eq!(fresh.idx, doomed.idx, "the slot must actually be reused");
        assert_ne!(fresh.gen, doomed.gen, "…with a new generation");
        assert_eq!(core.live(fresh), fresh.idx as usize);
    }

    #[test]
    #[should_panic(expected = "stale NodeId")]
    fn using_a_removed_node_panics() {
        let mut core = UiCore::new();
        let root = core.root();
        let n = core.node(root, Style::default());
        core.remove_node(n);
        core.node_style(n);
    }

    /// Removal is recursive, so a handle to a *descendant* is stale too —
    /// the case a parent-only free would miss.
    #[test]
    #[should_panic(expected = "stale NodeId")]
    fn using_a_removed_childs_handle_panics() {
        let mut core = UiCore::new();
        let root = core.root();
        let panel = core.node(root, Style::default());
        let inner = core.label(panel, 11.0, 0xFFFF_FFFF, "hi");
        core.remove_node(panel);
        core.node_rect(inner);
    }

    /// `set_interactive` used to write `pointer.interactive[idx]` with no
    /// generation check at all and never touch anything that had one, so a
    /// stale handle silently armed whichever node had taken the slot — the
    /// one path where staleness produced no panic anywhere.
    #[test]
    #[should_panic(expected = "stale NodeId")]
    fn arming_a_removed_node_panics() {
        let mut core = UiCore::new();
        let root = core.root();
        let n = core.node(root, Style::default());
        core.remove_node(n);
        core.node(root, Style::default()); // takes the recycled slot
        core.set_events(n, Events::CLICK | Events::HOVER);
    }

    /// The other three public entry points that resolved a caller's handle
    /// by raw index. Each reached a checked call eventually, so each panicked
    /// — but only after reading the wrong slot, and with a message blaming
    /// the wrong thing ("not a scroll area" for a handle that was merely
    /// stale).
    #[test]
    fn every_public_entry_point_rejects_a_stale_handle() {
        let stale = |f: fn(&mut UiCore, NodeId)| {
            let mut core = UiCore::new();
            let root = core.root();
            let n = core.scroll_area(root, Style::default());
            core.remove_node(n);
            core.node(root, Style::default());
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&mut core, n)))
                .expect_err("a stale handle must be rejected");
            let msg = panic
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| panic.downcast_ref::<&str>().unwrap_or(&"").to_string());
            assert!(msg.contains("stale NodeId"), "misleading panic: {msg}");
        };

        stale(|c, n| c.scroll_by(n, [0.0, 10.0]));
        stale(|c, n| {
            c.scroll_area(n, Style::default());
        });
        stale(|c, n| {
            c.set_state_style(n, crate::ui::StateStyle::fills(UiStyle::fill(WHITE), 0, 0, 0))
        });
        stale(|c, n| c.set_events(n, Events::CLICK | Events::HOVER));
    }

    #[test]
    #[should_panic(expected = "root cannot be removed")]
    fn removing_the_root_panics() {
        let mut core = UiCore::new();
        let root = core.root();
        core.remove_node(root);
    }

    /// A removed subtree must give its primitive slots back, or a UI that
    /// opens and closes a panel leaks the allocator upward forever.
    #[test]
    fn removal_returns_primitive_slots() {
        let mut core = UiCore::new();
        let root = core.root();
        let build = |core: &mut UiCore| {
            let panel = core.node(root, Style::default());
            core.set_background(panel, UiStyle::fill(0xFF00_00FF));
            core.label(panel, 11.0, 0xFFFF_FFFF, "hello");
            panel
        };

        let first = build(&mut core);
        let high_water = core.prim_count();
        core.remove_node(first);

        for _ in 0..8 {
            let p = build(&mut core);
            core.remove_node(p);
        }
        assert_eq!(core.prim_count(), high_water, "slots recycled, not re-reserved");
    }

    /// The parent must forget a removed child, or the placement walk keeps
    /// visiting a freed slot.
    #[test]
    fn a_removed_child_leaves_its_parent() {
        let mut core = UiCore::new();
        let root = core.root();
        let keep = core.node(root, Style::default());
        let drop = core.node(root, Style::default());
        core.remove_node(drop);
        core.run_layout([400.0, 400.0]);

        let ri = core.live(root);
        assert_eq!(core.tree.nodes[ri].children, vec![keep]);
    }

    /// The move docking is built on. The subtree keeps every slot it had, so
    /// a panel dragged across the screen is re-laid-out rather than remade —
    /// and nothing it owned had to be found and copied first.
    #[test]
    fn a_re_parented_subtree_keeps_its_primitives() {
        let mut core = UiCore::new();
        let root = core.root();
        let a = core.node(root, Style::default());
        let b = core.node(root, Style { padding: Rect::length(20.0_f32), ..Default::default() });
        let leaf = core.label(a, 9.0, WHITE, "moves");
        core.run_layout([400.0, 100.0]);
        let slots = core.paint_slots(leaf);

        core.set_parent(leaf, b);
        core.run_layout([400.0, 100.0]);

        assert_eq!(core.paint_slots(leaf), slots, "the glyph run should not have moved");
        assert_eq!(core.node_text(leaf), Some("moves"), "nor its text");
        assert_eq!(
            core.node_rect(leaf)[0],
            core.node_rect(b)[0] + 20.0,
            "and taffy should place it inside its new parent's padding",
        );
    }

    /// A primitive's clip group is fixed when its slot is allocated, so a
    /// move across one is a rebuild wearing a move's clothes. Loud beats a
    /// subtree that silently stops being clipped.
    #[test]
    #[should_panic(expected = "clip group")]
    fn moving_a_subtree_into_a_scroll_area_panics() {
        let mut core = UiCore::new();
        let root = core.root();
        let area = core.scroll_area(root, Style::default());
        let n = core.label(root, 9.0, WHITE, "x");
        core.set_parent(n, area);
    }
}
