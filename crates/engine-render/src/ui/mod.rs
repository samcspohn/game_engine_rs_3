//! Retained-mode UI (ADR-0006), phases 1 and 2.
//!
//! The UI is **a flat array of primitive slots** in device-local SoT
//! buffers, updated by dirty-bitmask scatter and drawn by exactly one
//! `vkCmdDrawIndirect` from a command buffer recorded once per session.
//! It is the same paradigm as the transform pipeline (ADR-0003/0004), with
//! the widget tree replacing the transform hierarchy as the thing that
//! decides *which slots changed*.
//!
//! What lives where:
//!
//! * [`UiCore`] — this file. The host-side store: four [`SlotArray`]s, the
//!   slot/run allocator, and the primitive-authoring API. Owns no Vulkan.
//! * [`UiGpu`] — `gpu.rs`. SoT buffers, double-buffered staging, the four
//!   scatter dispatches, the glyph atlas, and the draw pipeline.
//! * [`font`] — the built-in bitmap font and its static atlas.
//!
//! The whole design rests on one line, [`SlotArray::set`]'s equality gate:
//! any write path — a property setter, a relayout that happened to produce
//! identical rects, a full subtree rebuild — uploads only what genuinely
//! differs. An idle frame therefore costs zero bytes of staging traffic and
//! zero scatter workgroups: the prepasses see `word_count = 0` and the
//! scatters' indirect args come out `(0, 0, 0)`.
//!
//! Phase 3 (the callback/tick/animator layer, layout, hit testing) is not
//! implemented; the API here is the primitive layer it will sit on.
//!
//! # Who builds a UI
//!
//! Nobody, here. The renderer ships the *system*, not a UI — game and
//! editor code build their own trees through [`ui()`], the global store
//! (ADR-0008). `UiCore` owns no Vulkan, so a UI can be built before the
//! window exists; the first `run_layout` positions it.

mod dock;
mod events;
pub mod font;
mod gpu;
mod keyboard;
mod list;
mod menu_bar;
mod popup;
mod text_field;
mod theme;
mod tree;
mod tree_view;
mod viewport;
mod widget;

use std::any::Any;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::{Mutex, MutexGuard};

pub use dock::{DockSpace, DockStyle, DragPanel, Layout, PanelId, Side};
pub use events::{Event, Keep, This, TickId, Value};
pub use gpu::UiGpu;
pub use list::{DropMark, Row, RowContent, RowList, RowStyle};
pub use menu_bar::{MenuBar, MenuBarStyle};
pub use popup::{MenuStyle, PopupStyle};
pub use text_field::TextFieldStyle;
pub use theme::{Theme, set_theme, theme};
pub use tree::{DRAG_SLOP, Drag, Events, NodeId, Scrub, style};
pub use tree_view::{DragNode, Dropped, TreeDrag, TreeView};
pub use viewport::Viewport;
pub use widget::{
    Button, ButtonStyle, Checkbox, CheckboxStyle, Label, Menu, Popup, Progress, ProgressStyle,
    RadioGroup, RadioStyle, Scrollbar, ScrollbarStyle, Slider, SliderStyle, StateStyle, TabStyle,
    Tabs, TextField,
};

// ─────────────────────────────────────────────────────────────────────
// The global store
// ─────────────────────────────────────────────────────────────────────

static UI: OnceLock<Mutex<UiCore>> = OnceLock::new();
static PAINT: OnceLock<Paint> = OnceLock::new();

/// The process-wide UI store.
///
/// Global for the reason `engine_core::asset::global` is: `Component::init`
/// and `Component::update` can reach a static but not the renderer's
/// `RenderContext`. See [ui-threading](../../../../docs/notes/ui-threading.md)
/// for what a component on the parallel sweep should reach for instead.
pub fn global() -> &'static Mutex<UiCore> {
    UI.get_or_init(|| {
        let core = UiCore::new();
        PAINT.set(core.paint()).ok();
        Mutex::new(core)
    })
}

/// Lock the global UI store. A component that panicked holding this leaves
/// the store mid-edit but still lockable — `parking_lot` does not poison, so
/// the editor survives to show which component died (ADR-0010 §7).
pub fn ui() -> MutexGuard<'static, UiCore> {
    global().lock()
}

/// The global store's paint arrays, reached without its lock.
pub fn paint() -> &'static Paint {
    global();
    PAINT.get().expect("set by global()")
}

static CAPTURE: AtomicU8 = AtomicU8::new(0);
const CAPTURE_POINTER: u8 = 1;
const CAPTURE_KEYBOARD: u8 = 2;

/// [`UiCore::pointer_captured`] as of this frame's input fold, without the lock.
pub fn pointer_captured() -> bool {
    CAPTURE.load(Ordering::Relaxed) & CAPTURE_POINTER != 0
}

/// [`UiCore::keyboard_captured`] as of this frame's input fold, without the lock.
pub fn keyboard_captured() -> bool {
    CAPTURE.load(Ordering::Relaxed) & CAPTURE_KEYBOARD != 0
}

type Deferred = Box<dyn FnOnce(&mut UiCore) + Send>;

/// One queue per thread that has deferred anything, so workers never share one.
static QUEUES: Mutex<Vec<Arc<Mutex<Vec<Deferred>>>>> = Mutex::new(Vec::new());

thread_local! {
    static QUEUE: Arc<Mutex<Vec<Deferred>>> = {
        let q = Arc::<Mutex<Vec<Deferred>>>::default();
        QUEUES.lock().push(q.clone());
        q
    };
}

/// Run `f` against the global store after the sweep and before layout, on the
/// main thread. Note: `f` already holds the store, so calling `ui()` in it
/// deadlocks.
pub fn defer(f: impl FnOnce(&mut UiCore) + Send + 'static) {
    QUEUE.with(|q| q.lock().push(Box::new(f)));
}

/// Primitive kinds, matching `ui.frag`'s `KIND_*` constants. Stored in the
/// low byte of `UiStyle::kind_flags`.
pub const KIND_RECT: u32 = 0;
pub const KIND_TEXT: u32 = 1;
pub const KIND_IMAGE: u32 = 2;

/// Bindless slot holding a camera's colour target, which
/// `assets::RESERVED_SLOTS` keeps streamed textures out of. Bound in the
/// UI's copy of the array only — see `RESERVED_SLOTS` for why not the scene's.
///
/// Counted down from the top so camera 0 keeps the slot it always had.
pub const fn camera_target(slot: usize) -> u32 {
    crate::assets::MAX_TEXTURES - 1 - slot as u32
}

/// Non-premultiplied sRGB rgba8, byte order `r, g, b, a` — the packing
/// `ui.frag`'s `unpack_color` expects.
pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
}

pub const fn rgb(r: u8, g: u8, b: u8) -> u32 {
    rgba(r, g, b, 255)
}

// ─────────────────────────────────────────────────────────────────────
// Records
// ─────────────────────────────────────────────────────────────────────

/// A slot-indexed device record. `STRIDE` is its size in `u32` words, which
/// is also `ui_scatter.comp`'s push constant for this array.
pub(crate) trait Record: Copy + PartialEq + Default {
    const STRIDE: usize;
    fn write(&self, out: &mut [u32]);
    fn read(w: &[u32]) -> Self;
}

const MAX_STRIDE: usize = 8;

/// Geometry — 32 B. Changes on layout / reflow / resize.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct UiQuad {
    /// `x, y, w, h` in group-local physical px. A zero-size rect is the
    /// freed-slot encoding: `ui.vert` culls it before anything reads it.
    pub rect: [f32; 4],
    /// `u0, v0, u1, v1` — glyph cell or texture sub-rect.
    pub uv: [f32; 4],
}

impl Record for UiQuad {
    const STRIDE: usize = 8;
    fn write(&self, out: &mut [u32]) {
        for (o, v) in out.iter_mut().zip(self.rect.iter().chain(self.uv.iter())) {
            *o = v.to_bits();
        }
    }

    fn read(w: &[u32]) -> Self {
        let f = |i: usize| f32::from_bits(w[i]);
        Self {
            rect: [f(0), f(1), f(2), f(3)],
            uv: [f(4), f(5), f(6), f(7)],
        }
    }
}

/// Appearance — 32 B. Changes on hover / press / focus / theme. Split from
/// [`UiQuad`] for the same reason position/rotation/scale are three buffers:
/// the two change at completely different frequencies, and each gets its own
/// dirty mask, prepass and scatter dispatch.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct UiStyle {
    pub fill: u32,
    pub border: u32,
    /// Four `u8` corner radii in px, ascending byte order
    /// `top-left, top-right, bottom-right, bottom-left`.
    pub radius: u32,
    /// `f16 border width | f16 edge softness`.
    pub border_width: u32,
    pub kind_flags: u32,
    /// Bindless `TextureId` for [`KIND_IMAGE`], ignored otherwise.
    pub tex: u32,
    /// Fraction of the quad's width cut from its right edge. Only [`Paint`]
    /// writes it; [`UiCore::set_style`] keeps it.
    pub(crate) cut: f32,
    reserved: u32,
}

impl Record for UiStyle {
    const STRIDE: usize = 8;
    fn write(&self, out: &mut [u32]) {
        out.copy_from_slice(&[
            self.fill,
            self.border,
            self.radius,
            self.border_width,
            self.kind_flags,
            self.tex,
            self.cut.to_bits(),
            self.reserved,
        ]);
    }

    fn read(w: &[u32]) -> Self {
        Self {
            fill: w[0],
            border: w[1],
            radius: w[2],
            border_width: w[3],
            kind_flags: w[4],
            tex: w[5],
            cut: f32::from_bits(w[6]),
            reserved: w[7],
        }
    }
}

impl UiStyle {
    pub fn fill(color: u32) -> Self {
        Self {
            fill: color,
            kind_flags: KIND_RECT,
            ..Default::default()
        }
    }

    /// Sample `tex` from the bindless array instead of filling. The colour
    /// is a *tint* multiplying the texel, so opaque white leaves it alone —
    /// which is what [`UiCore::image`](UiCore::image) passes.
    pub fn image(tex: u32) -> Self {
        Self {
            fill: rgba(255, 255, 255, 255),
            kind_flags: KIND_IMAGE,
            tex,
            ..Default::default()
        }
    }

    pub fn border(mut self, color: u32, width: f32) -> Self {
        self.border = color;
        self.border_width = pack_half2(width, self.softness());
        self
    }

    /// Uniform corner radius, px. Clamped to the `u8` the record stores.
    pub fn radius(self, r: f32) -> Self {
        self.corners([r; 4])
    }

    /// Per-corner radii, px, in `top-left, top-right, bottom-right,
    /// bottom-left` order.
    pub fn corners(mut self, r: [f32; 4]) -> Self {
        self.radius = r.iter().enumerate().fold(0, |acc, (i, &v)| {
            acc | ((v.clamp(0.0, 255.0) as u32) << (8 * i))
        });
        self
    }

    /// Extra edge softness in px on top of the analytic one-pixel AA.
    pub fn softness(&self) -> f32 {
        half::f16::from_bits((self.border_width >> 16) as u16).to_f32()
    }

    pub fn soft(mut self, px: f32) -> Self {
        self.border_width = pack_half2(half::f16::from_bits(self.border_width as u16).to_f32(), px);
        self
    }
}

fn pack_half2(a: f32, b: f32) -> u32 {
    (half::f16::from_f32(a).to_bits() as u32) | ((half::f16::from_f32(b).to_bits() as u32) << 16)
}

/// Clip + transform + opacity — 32 B, roughly one per window / panel /
/// scroll area. This is the incremental-update multiplier: dragging a
/// window or scrolling a 10 000-row list dirties exactly one of these.
///
/// Groups are **flat**, not a hierarchy: a nested panel has its parent's
/// offset and clip composed in on the CPU. Group counts are in the hundreds
/// so composing them is free, and it keeps `ui.vert` to a single fetch with
/// no parent-chain walk.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct UiGroup {
    /// `x0, y0, x1, y1` in physical px, already intersected with parents.
    pub clip: [f32; 4],
    /// Translation: window position + scroll.
    pub offset: [f32; 2],
    pub opacity: f32,
    pub flags: u32,
}

impl Default for UiGroup {
    fn default() -> Self {
        Self {
            clip: [0.0; 4],
            offset: [0.0; 2],
            opacity: 1.0,
            flags: 0,
        }
    }
}

impl Record for UiGroup {
    const STRIDE: usize = 8;
    fn write(&self, out: &mut [u32]) {
        out.copy_from_slice(&[
            self.clip[0].to_bits(),
            self.clip[1].to_bits(),
            self.clip[2].to_bits(),
            self.clip[3].to_bits(),
            self.offset[0].to_bits(),
            self.offset[1].to_bits(),
            self.opacity.to_bits(),
            self.flags,
        ]);
    }

    fn read(w: &[u32]) -> Self {
        let f = |i: usize| f32::from_bits(w[i]);
        Self {
            clip: [f(0), f(1), f(2), f(3)],
            offset: [f(4), f(5)],
            opacity: f(6),
            flags: w[7],
        }
    }
}

/// One entry of the draw list: `(primitive slot, group id)`, back-to-front.
///
/// The indirection is the scene pass's `instance_to_entity` trick — it
/// decouples draw order from slot identity, so inserting a widget in the
/// middle of the z-order rewrites only this array.
///
/// **Paint order is tree order**, written by the placement walk in
/// `UiCore::place`: a node's background, then its glyph run, then its
/// children, depth-first. Identity ordering (`order[i] = (i, gid)`) was the
/// phase-1/2 placeholder and was actively wrong — the slot free list recycles
/// *low* slots, so a label whose run was recycled could be drawn **behind**
/// opaque geometry allocated earlier, which is exactly what made a tree row's
/// disclosure arrow vanish under a panel background.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub(crate) struct OrderEntry([u32; 2]);

impl Record for OrderEntry {
    const STRIDE: usize = 2;
    fn write(&self, out: &mut [u32]) {
        out.copy_from_slice(&self.0);
    }

    fn read(w: &[u32]) -> Self {
        Self([w[0], w[1]])
    }
}

// ─────────────────────────────────────────────────────────────────────
// Change detection
// ─────────────────────────────────────────────────────────────────────

/// A CPU mirror of one device array — the change *detector*, since every write
/// funnels through [`Self::set`]. Chunks never move once allocated, so writes
/// take `&self` from any thread: one writer per slot per frame.
pub(crate) struct SlotArray<T: Record> {
    chunks: Box<[OnceLock<Chunk>]>,
    _t: PhantomData<T>,
}

const CHUNK_LOG2: u32 = 12;
const CHUNK: usize = 1 << CHUNK_LOG2;
/// A million slots per array.
const MAX_CHUNKS: usize = 256;

struct Chunk {
    words: Box<[AtomicU32]>,
    dirty: Box<[AtomicU32]>,
    /// Any bit in `dirty` set, so an upload skips clean chunks unread.
    any: AtomicBool,
}

fn atomics(n: usize) -> Box<[AtomicU32]> {
    std::iter::repeat_with(|| AtomicU32::new(0))
        .take(n)
        .collect()
}

impl Chunk {
    // Note: SeqCst pairs with `upload`'s two swaps, so a bit set after the
    // reader cleared `any` is either seen now or flagged for next frame.
    fn mark(&self, j: usize) {
        self.dirty[j / 32].fetch_or(1 << (j % 32), Ordering::SeqCst);
        if !self.any.load(Ordering::SeqCst) {
            self.any.store(true, Ordering::SeqCst);
        }
    }
}

impl<T: Record> SlotArray<T> {
    fn new() -> Self {
        Self {
            chunks: (0..MAX_CHUNKS).map(|_| OnceLock::new()).collect(),
            _t: PhantomData,
        }
    }

    fn chunk(&self, i: u32) -> (&Chunk, usize) {
        let c = i as usize >> CHUNK_LOG2;
        assert!(
            c < MAX_CHUNKS,
            "UI slot {i} is past the {} slot limit",
            MAX_CHUNKS * CHUNK
        );
        let chunk = self.chunks[c].get_or_init(|| Chunk {
            words: atomics(CHUNK * T::STRIDE),
            dirty: atomics(CHUNK / 32),
            any: AtomicBool::new(false),
        });
        (chunk, i as usize & (CHUNK - 1))
    }

    pub(crate) fn set(&self, i: u32, v: T) {
        let (c, j) = self.chunk(i);
        let mut w = [0u32; MAX_STRIDE];
        v.write(&mut w[..T::STRIDE]);
        let slot = &c.words[j * T::STRIDE..(j + 1) * T::STRIDE];
        if slot
            .iter()
            .zip(&w)
            .all(|(a, b)| a.load(Ordering::Relaxed) == *b)
        {
            return; // ← the entire premise
        }
        for (a, b) in slot.iter().zip(&w) {
            a.store(*b, Ordering::Relaxed);
        }
        c.mark(j);
    }

    pub(crate) fn get(&self, i: u32) -> T {
        let (c, j) = self.chunk(i);
        let mut w = [0u32; MAX_STRIDE];
        for (o, a) in w
            .iter_mut()
            .zip(&c.words[j * T::STRIDE..(j + 1) * T::STRIDE])
        {
            *o = a.load(Ordering::Relaxed);
        }
        T::read(&w[..T::STRIDE])
    }

    /// Mark every populated slot dirty. Used after the device buffers are
    /// reallocated, when the SoT no longer holds what the mirror says.
    fn mark_all(&self, count: u32) {
        for i in 0..count {
            let (c, j) = self.chunk(i);
            c.mark(j);
        }
    }

    /// Copy this frame's dirty slots into `stage` and publish the bitmask into
    /// `dirty`. Returns `[min_word, max_word]` for the prepass bounds, with
    /// `max < 0` meaning nothing was dirty.
    ///
    /// Only dirty words are touched in either buffer — the device-side
    /// `fill_buffer(0)` in the scatter secondary is what keeps the rest of
    /// the bitmask zero between frames.
    pub(crate) fn upload(&self, stage: &mut [u32], dirty: &mut [u32]) -> (i64, i64) {
        let (mut lo, mut hi) = (i64::MAX, -1);
        for (ci, c) in self.chunks.iter().enumerate() {
            let Some(c) = c.get() else { continue };
            if !c.any.load(Ordering::SeqCst) || !c.any.swap(false, Ordering::SeqCst) {
                continue;
            }
            for (wi, d) in c.dirty.iter().enumerate() {
                let mut bits = d.swap(0, Ordering::SeqCst);
                if bits == 0 {
                    continue;
                }
                let w = ci * CHUNK / 32 + wi;
                dirty[w] = bits;
                (lo, hi) = (lo.min(w as i64), hi.max(w as i64));
                while bits != 0 {
                    let j = wi * 32 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let base = (ci * CHUNK + j) * T::STRIDE;
                    for k in 0..T::STRIDE {
                        stage[base + k] = c.words[j * T::STRIDE + k].load(Ordering::Relaxed);
                    }
                }
            }
        }
        (lo, hi)
    }
}

/// The primitive arrays, written without the store's lock from any thread.
/// Appearance only — anything that moves layout goes through [`UiCore`]. See
/// [ui-threading](../../../../docs/notes/ui-threading.md).
#[derive(Clone)]
pub struct Paint {
    pub(crate) style: Arc<SlotArray<UiStyle>>,
}

impl Paint {
    pub(crate) fn set_fill(&self, p: PrimId, color: u32) {
        let s = self.style.get(p.0);
        self.style.set(p.0, UiStyle { fill: color, ..s });
    }

    pub(crate) fn set_cut(&self, p: PrimId, cut: f32) {
        let s = self.style.get(p.0);
        self.style.set(p.0, UiStyle { cut, ..s });
    }

    pub(crate) fn cut(&self, p: PrimId) -> f32 {
        self.style.get(p.0).cut
    }
}

// ─────────────────────────────────────────────────────────────────────
// Handles
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GroupId(u32);

/// A single primitive — a rect or an image. Internally a run of length 1,
/// so it frees back into the same allocator as a text run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PrimId(pub(crate) u32);

/// A laid-out string. Glyphs are one primitive each, in the same arrays as
/// everything else, so a label is a contiguous run of slots and editing its
/// text dirties exactly the glyphs that differ.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TextId(u32);

struct TextRun {
    group: GroupId,
    first: u32,
    /// `log2` of the run's slot capacity. Rounding up to a power of two is
    /// what lets most edits reuse the run in place; only crossing a
    /// boundary reallocates.
    cap_log2: u32,
    pos: [f32; 2],
    scale: f32,
    color: u32,
    text: String,
    /// Hidden runs keep their string but write zero-area quads, so the
    /// vertex stage culls them.
    ///
    /// Collapsing a node's *box* is not enough to hide its text: a glyph's
    /// quad is sized by the font, not by the node it belongs to, so a
    /// `Display::None` node's label would keep drawing at whatever origin the
    /// collapsed box landed on. That is a pile of stacked rows at the top of
    /// a list, or a "hidden" panel whose text is still on screen.
    hidden: bool,
}

/// Largest run the bucketed free list serves: 2^15 glyphs.
const MAX_RUN_LOG2: u32 = 15;

// ─────────────────────────────────────────────────────────────────────
// UiCore
// ─────────────────────────────────────────────────────────────────────

/// The host-side UI store. Owns slot identity, the four change-detecting
/// mirrors, and the primitive-authoring API. Deliberately monomorphic and
/// Vulkan-free: phase 3's `Ui<S>` wraps this with the callback table so the
/// app-state type never goes viral through the widget layer.
pub struct UiCore {
    pub(crate) quad: SlotArray<UiQuad>,
    pub(crate) style: Arc<SlotArray<UiStyle>>,
    pub(crate) group: SlotArray<UiGroup>,
    pub(crate) order: SlotArray<OrderEntry>,

    /// Free runs bucketed by `log2(len)`; index `k` holds first-slots of
    /// free runs of `2^k` slots.
    free_runs: Vec<Vec<u32>>,
    /// `TextId`s released by `free_text`, reused by `text`. Runs are indices
    /// into `runs`, which never shrinks — recycling keeps a UI that builds
    /// and tears down panels from growing it without bound.
    free_texts: Vec<u32>,
    /// Bump watermark. Also the draw list's length — freed slots stay in
    /// `order` holding a zero-area quad rather than forcing a compaction.
    next_slot: u32,
    /// Write cursor into `order` during the placement walk, and afterwards
    /// the number of entries it emitted.
    order_cursor: u32,
    /// Slots already given a paint position this walk, so unreachable ones
    /// (free-listed, or primitives not attached to a node) can be swept into
    /// the tail — the draw covers `next_slot` instances and every one must
    /// name a slot.
    emitted: Vec<u64>,
    /// Set when the slot allocator moves a run. Paint order then has to be
    /// re-emitted even if taffy has nothing to re-solve.
    order_dirty: bool,
    next_group: u32,
    runs: Vec<TextRun>,
    /// The widget tree and its taffy-backed layout pass. See `tree.rs`.
    tree: tree::Tree,
    /// Pointer interaction state, recomputed once per frame. See
    /// `UiCore::update_pointer`.
    pointer: Pointer,
    /// Per-node pointer-state looks, indexed by `NodeId`. Sparse — only
    /// nodes that opted in. Applied on transition, never per frame.
    state_styles: Vec<Option<widget::StateStyle>>,
    /// Bumped whenever something that can change *what is under the pointer*
    /// changes: a completed placement walk, or a scroll that actually moved.
    ///
    /// Without it `update_pointer`'s early-out asks the wrong question — "did
    /// the pointer move?" rather than "can the answer have changed?" — so a
    /// button animated under a still cursor, or a panel toggled open beneath
    /// it, would never register as hovered until the mouse was jiggled.
    pub(crate) layout_epoch: u64,
    /// Per-node control state — a checkbox's bool, a slider's f32, a text
    /// field's string, and the parts each needs to redraw itself. Sparse,
    /// indexed by `NodeId` like `state_styles`, and consulted only for the
    /// node the pointer is on or the one that holds focus.
    controls: Vec<Option<widget::Control>>,
    /// Every live scrollbar. The one thing the engine re-fits on its own:
    /// a bar mirrors an area it does not own, so nothing it can hook on the
    /// node the pointer hit would notice a wheel, a resize or a list that
    /// grew. Walked after a layout and after a scroll — a handful of nodes,
    /// never per frame.
    pub(crate) scrollbars: Vec<widget::Scrollbar>,
    /// Thumb re-fits performed. Bookkeeping, like `Pointer::walks`: it is what
    /// makes "a scroll refits one bar, not all of them" a test.
    pub(crate) refits: u64,
    /// Keyboard focus and this frame's one-shot keyboard events. The
    /// pointer's opposite number: no position, one retained target. See
    /// `keyboard.rs`.
    pub(crate) keyboard: keyboard::Keyboard,
    /// The one open overlay. On the store rather than in whichever widget
    /// opened it, for the reason `Grab` is: there is one pointer to dismiss
    /// it with, so two of them could not both be right. See `popup.rs`.
    pub(crate) overlay: Option<popup::Overlay>,
    /// Per-node handlers and the queue they run from. See `events.rs`.
    pub(crate) dispatch: events::Dispatch,
    /// Frames folded, counted by `update_pointer` ahead of its own early-out.
    /// What a caller stamps against to run something once a frame however
    /// many handlers woke it — see [`UiCore::frame`].
    frames: u64,
}

/// Hover / press / click state for the one system pointer.
///
/// Recomputed before `worlds::sweep_all` so a component's `clicked()` observes
/// this frame's input, and so a camera controller can decline to orbit when
/// the UI took the press.
#[derive(Default)]
pub(crate) struct Pointer {
    /// Which pointer events each node accepts, indexed by `NodeId`. Opt-in: a
    /// node is inert until `set_events`, so a panel's decorative boxes never
    /// swallow clicks meant for the world.
    pub(crate) listens: Vec<Events>,
    pub(crate) pos: [f32; 2],
    /// Every node accepting [`Events::HOVER`] under the pointer, innermost
    /// first — an ancestor chain, not a single winner, because a row and the
    /// checkbox inside it are both genuinely hovered.
    pub(crate) hover: Vec<NodeId>,
    /// Last frame's `hover`, swapped rather than reallocated. Also what the
    /// restyle pass reads to find what the pointer *left*.
    pub(crate) hover_prev: Vec<NodeId>,
    /// The walk claimed something — the UI is under the pointer, whatever
    /// kind it accepted. Camera controllers and world-picking sit out on it,
    /// so a scroll area suppresses camera zoom without a field of its own.
    pub(crate) over_ui: bool,
    /// `UiCore::layout_epoch` as of the last hit walk. A still pointer is not
    /// enough to reuse the previous answer — the scene may have moved under
    /// it — so this is the other half of `update_pointer`'s early-out.
    pub(crate) walked: u64,
    /// Hit walks performed. Only bookkeeping, but it makes "genuinely
    /// event-driven" assertable rather than merely claimed.
    pub(crate) walks: u64,
    /// Node a press landed on, held until release. A click fires only when
    /// the release lands on that same node — standard "drag off to cancel".
    pub(crate) down_on: Option<NodeId>,
    /// Where the press that set `down_on` landed. A drag is defined against
    /// its origin rather than the previous frame, so a slider tracks the
    /// pointer exactly and a drag threshold is a single comparison instead
    /// of an accumulator the caller has to keep.
    pub(crate) press_pos: [f32; 2],
    /// Set for exactly one frame, by the release that completed a click.
    pub(crate) clicked: Option<NodeId>,
    /// The secondary button's two fields. It gets no hover set, no drag and
    /// no focus — a right click is one question ("on what?"), so folding it
    /// into the primary's state would be four more branches for nothing.
    pub(crate) right_down_on: Option<NodeId>,
    pub(crate) right_clicked: Option<NodeId>,
    /// Seconds since the app started. Passed in rather than read from a
    /// clock, so gesture tests stay deterministic instead of sleeping.
    pub(crate) now: f64,
    /// The last completed click — what the next one is measured against.
    pub(crate) last_click: Option<Click>,
    /// Node a release ended a press on, for exactly one frame — whether or
    /// not the pointer was still over it. `clicked` cannot serve a drop:
    /// a drop is *defined* by releasing somewhere other than the press, which
    /// is the one case `clicked` excludes.
    pub(crate) dropped: Option<NodeId>,
    /// The drag in flight. There is one system pointer, so there is at most
    /// one of these — which is why it lives here and not in whichever widget
    /// happened to start it.
    pub(crate) grab: Option<Grab>,
    /// Who has taken this gesture. A node whose drag is spoken for does not
    /// also read it as its own — a numeric field scrubs rather than selects.
    /// Cleared by the next press, not by the release, so the frame a drag
    /// ends on is still claimed.
    pub(crate) claimed: Option<NodeId>,
    /// Where a grab landed, for exactly one frame: the node under the pointer
    /// at release, and what it was carrying. Separate from `grab` because the
    /// payload has to outlive the gesture by one frame for a target to read
    /// it, while the ghost is gone the instant the button comes up.
    pub(crate) drop: Option<(NodeId, Box<dyn Any + Send>)>,
}

/// A completed click, kept so the next one can tell whether it continues the
/// same gesture. `count` is why this is a struct and not a timestamp.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct Click {
    pub(crate) node: NodeId,
    pub(crate) time: f64,
    pub(crate) pos: [f32; 2],
    /// 1 for a single click, 2 for a double, and up.
    pub(crate) count: u32,
}

/// A drag in flight: what is being carried, and the node drawn at the pointer
/// to show it.
///
/// The payload is opaque — `UiCore` never interprets it, exactly as
/// [`TreeView`] never interprets the `u64` it identifies nodes by. That is
/// what lets a drag cross module boundaries the renderer knows nothing about.
pub(crate) struct Grab {
    pub(crate) payload: Box<dyn Any + Send>,
    pub(crate) ghost: NodeId,
}

impl Default for UiCore {
    fn default() -> Self {
        Self::new()
    }
}

impl UiCore {
    pub fn new() -> Self {
        // Group 0 is the root: it clips to the window (set every
        // `run_layout`) and never moves. The tree's nodes inherit it until
        // scroll areas and docking introduce groups of their own.
        let group = SlotArray::new();
        group.set(0, UiGroup::default());
        Self {
            quad: SlotArray::new(),
            style: Arc::new(SlotArray::new()),
            group,
            order: SlotArray::new(),
            free_runs: (0..=MAX_RUN_LOG2).map(|_| Vec::new()).collect(),
            free_texts: Vec::new(),
            next_slot: 0,
            order_cursor: 0,
            emitted: Vec::new(),
            order_dirty: false,
            next_group: 1,
            runs: Vec::new(),
            tree: tree::Tree::new(GroupId(0)),
            layout_epoch: 0,
            pointer: Pointer::default(),
            state_styles: Vec::new(),
            controls: Vec::new(),
            scrollbars: Vec::new(),
            refits: 0,
            overlay: None,
            keyboard: keyboard::Keyboard::default(),
            dispatch: events::Dispatch::default(),
            frames: 0,
        }
    }

    /// Live primitive count — the draw list's length, promoted to the
    /// device-side `VkDrawIndirectCommand` by `ui_build_args.comp`.
    pub fn prim_count(&self) -> u32 {
        self.next_slot
    }

    pub fn group_count(&self) -> u32 {
        self.next_group
    }

    pub fn paint(&self) -> Paint {
        Paint {
            style: self.style.clone(),
        }
    }

    /// Mirror the two capture answers into the lock-free snapshot.
    pub(crate) fn publish_capture(&self) {
        let p = (self.pointer_captured() as u8) * CAPTURE_POINTER;
        let k = (self.keyboard_captured() as u8) * CAPTURE_KEYBOARD;
        CAPTURE.store(p | k, Ordering::Relaxed);
    }

    /// Everything [`defer`] queued, from every thread.
    pub(crate) fn run_deferred(&mut self) {
        let queues = QUEUES.lock().clone();
        for q in queues {
            let fs = std::mem::take(&mut *q.lock());
            fs.into_iter().for_each(|f| f(self));
        }
    }

    // ── Groups ──────────────────────────────────────────────────────

    pub fn group(&mut self, clip: [f32; 4], offset: [f32; 2]) -> GroupId {
        let id = self.next_group;
        self.next_group += 1;
        self.group.set(
            id,
            UiGroup {
                clip,
                offset,
                ..Default::default()
            },
        );
        GroupId(id)
    }

    pub fn set_group_clip(&mut self, g: GroupId, clip: [f32; 4]) {
        let mut v = self.group.get(g.0);
        v.clip = clip;
        self.group.set(g.0, v);
    }

    pub fn set_group_offset(&mut self, g: GroupId, offset: [f32; 2]) {
        let mut v = self.group.get(g.0);
        v.offset = offset;
        self.group.set(g.0, v);
    }

    pub fn set_group_opacity(&mut self, g: GroupId, opacity: f32) {
        let mut v = self.group.get(g.0);
        v.opacity = opacity;
        self.group.set(g.0, v);
    }

    // ── Primitives ──────────────────────────────────────────────────

    pub fn rect(&mut self, g: GroupId, rect: [f32; 4], style: UiStyle) -> PrimId {
        let slot = self.alloc_run(0, g);
        self.quad.set(slot, UiQuad { rect, uv: [0.0; 4] });
        self.style.set(slot, style);
        PrimId(slot)
    }

    pub fn set_rect(&mut self, p: PrimId, rect: [f32; 4]) {
        let mut q = self.quad.get(p.0);
        q.rect = rect;
        self.quad.set(p.0, q);
    }

    pub fn set_style(&mut self, p: PrimId, style: UiStyle) {
        let cut = self.style.get(p.0).cut;
        self.style.set(p.0, UiStyle { cut, ..style });
    }

    /// The canonical incremental update: a hover is one dirty word, one
    /// workgroup, 32 bytes.
    pub fn set_fill(&mut self, p: PrimId, color: u32) {
        let mut s = self.style.get(p.0);
        s.fill = color;
        self.style.set(p.0, s);
    }

    pub fn free(&mut self, p: PrimId) {
        self.free_run(p.0, 0);
    }

    // ── Text ────────────────────────────────────────────────────────

    /// Size in px of `s` rendered at `px` line height.
    pub fn text_size(&self, s: &str, px: f32) -> [f32; 2] {
        let scale = px / font::GLYPH_H as f32;
        [font::text_width(s) as f32 * scale, px]
    }

    pub fn text(&mut self, g: GroupId, pos: [f32; 2], px: f32, color: u32, s: &str) -> TextId {
        let cap_log2 = run_bucket(s.chars().count() as u32);
        let first = self.alloc_run(cap_log2, g);
        let run = TextRun {
            group: g,
            first,
            cap_log2,
            pos,
            scale: px / font::GLYPH_H as f32,
            color,
            text: String::new(),
            hidden: false,
        };
        let id = match self.free_texts.pop() {
            Some(i) => {
                self.runs[i as usize] = run;
                TextId(i)
            }
            None => {
                self.runs.push(run);
                TextId(self.runs.len() as u32 - 1)
            }
        };
        self.write_run(id, s.to_string());
        id
    }

    pub fn set_text(&mut self, t: TextId, s: &str) {
        if self.runs[t.0 as usize].text == s {
            return;
        }
        self.write_run(t, s.to_string());
    }

    /// The string a run currently holds — the retained copy the equality
    /// gate compares against.
    pub fn text_of(&self, t: TextId) -> &str {
        &self.runs[t.0 as usize].text
    }

    /// The run's line height in px.
    pub fn text_px(&self, t: TextId) -> f32 {
        self.runs[t.0 as usize].scale * font::GLYPH_H as f32
    }

    pub fn set_text_color(&mut self, t: TextId, color: u32) {
        if self.runs[t.0 as usize].color == color {
            return;
        }
        self.runs[t.0 as usize].color = color;
        let text = self.runs[t.0 as usize].text.clone();
        self.write_run(t, text);
    }

    pub fn set_text_pos(&mut self, t: TextId, pos: [f32; 2]) {
        if self.runs[t.0 as usize].pos == pos {
            return;
        }
        self.runs[t.0 as usize].pos = pos;
        let text = self.runs[t.0 as usize].text.clone();
        self.write_run(t, text);
    }

    /// Show or hide a run without disturbing its text. Idempotent.
    pub(crate) fn set_run_visible(&mut self, t: TextId, visible: bool) {
        if self.runs[t.0 as usize].hidden != visible {
            return;
        }
        self.runs[t.0 as usize].hidden = !visible;
        let text = self.runs[t.0 as usize].text.clone();
        self.write_run(t, text);
    }

    /// Lay `s` out into the run's slots. Every write goes through
    /// `SlotArray::set`, so re-typing `100` → `101` dirties one glyph, and
    /// re-writing the same string dirties nothing.
    fn write_run(&mut self, t: TextId, s: String) {
        let idx = t.0 as usize;
        let len = s.chars().count() as u32;
        let needed = run_bucket(len);
        if needed > self.runs[idx].cap_log2 {
            let (first, log2, group) = {
                let r = &self.runs[idx];
                (r.first, r.cap_log2, r.group)
            };
            self.free_run(first, log2);
            self.runs[idx].cap_log2 = needed;
            self.runs[idx].first = self.alloc_run(needed, group);
        }

        let (first, cap_log2, pos, scale, color) = {
            let r = &self.runs[idx];
            (r.first, r.cap_log2, r.pos, r.scale, r.color)
        };
        let advance = font::ADVANCE as f32 * scale;
        let size = [font::GLYPH_W as f32 * scale, font::GLYPH_H as f32 * scale];
        let style = UiStyle {
            fill: color,
            kind_flags: KIND_TEXT,
            ..Default::default()
        };
        let hidden = self.runs[idx].hidden;
        for (i, ch) in s.chars().enumerate() {
            let slot = first + i as u32;
            self.quad.set(
                slot,
                match hidden {
                    true => UiQuad::default(),
                    false => UiQuad {
                        rect: [pos[0] + i as f32 * advance, pos[1], size[0], size[1]],
                        uv: font::glyph_uv(ch),
                    },
                },
            );
            self.style.set(slot, style);
        }
        // Tail of the run: zero-area quads, culled in the vertex stage.
        for i in len..(1 << cap_log2) {
            self.quad.set(first + i, UiQuad::default());
        }
        self.runs[idx].text = s;
    }

    // ── Slot allocation ─────────────────────────────────────────────

    /// Reserve a contiguous run of `2^log2` slots and point its order
    /// entries at `g`.
    fn alloc_run(&mut self, log2: u32, g: GroupId) -> u32 {
        assert!(
            log2 <= MAX_RUN_LOG2,
            "run of 2^{log2} slots exceeds the allocator"
        );
        let n = 1u32 << log2;
        let first = match self.free_runs[log2 as usize].pop() {
            Some(f) => f,
            None => {
                let f = self.next_slot;
                self.next_slot += n;
                f
            }
        };
        let _ = g; // paint order comes from the tree walk, not allocation
        for i in 0..n {
            self.quad.set(first + i, UiQuad::default());
        }
        self.order_dirty = true;
        first
    }

    /// Release a text run's slots and recycle its [`TextId`].
    ///
    /// Only `remove_node` calls this — a label's run outlives every retype,
    /// since `write_run` reuses the bucket whenever the new string still
    /// fits.
    pub(crate) fn free_text(&mut self, t: TextId) {
        let run = &mut self.runs[t.0 as usize];
        let (first, cap_log2) = (run.first, run.cap_log2);
        run.text.clear();
        self.free_run(first, cap_log2);
        self.free_texts.push(t.0);
    }

    fn free_run(&mut self, first: u32, log2: u32) {
        for i in 0..(1u32 << log2) {
            self.quad.set(first + i, UiQuad::default());
        }
        self.free_runs[log2 as usize].push(first);
        self.order_dirty = true;
    }

    // ── Paint order ─────────────────────────────────────────────────

    /// Begin a paint-order pass. Called by `run_layout` before the walk.
    pub(crate) fn begin_order(&mut self) {
        self.order_cursor = 0;
        self.emitted.clear();
        self.emitted.resize(self.next_slot.div_ceil(64) as usize, 0);
    }

    /// Give `count` slots starting at `first` the next paint positions.
    pub(crate) fn emit_order(&mut self, first: u32, count: u32, g: GroupId) {
        for s in first..first + count {
            self.order.set(self.order_cursor, OrderEntry([s, g.0]));
            self.order_cursor += 1;
            self.emitted[(s / 64) as usize] |= 1 << (s % 64);
        }
    }

    /// Sweep every slot the walk did not reach into the tail. They are free
    /// or unattached, so they hold zero-area quads and `ui.vert` culls them —
    /// but the draw still issues `next_slot` instances, so each needs an
    /// entry. Group 0 because nothing is drawn.
    pub(crate) fn end_order(&mut self) {
        for s in 0..self.next_slot {
            if self.emitted[(s / 64) as usize] & (1 << (s % 64)) == 0 {
                self.order.set(self.order_cursor, OrderEntry([s, 0]));
                self.order_cursor += 1;
            }
        }
        debug_assert_eq!(
            self.order_cursor, self.next_slot,
            "paint order must cover every slot"
        );
        self.order_dirty = false;
    }

    pub(crate) fn order_dirty(&self) -> bool {
        self.order_dirty
    }

    /// A node's glyph run, as `(first slot, slot count)`.
    pub(crate) fn run_slots(&self, t: TextId) -> (u32, u32) {
        let r = &self.runs[t.0 as usize];
        (r.first, 1 << r.cap_log2)
    }

    /// Re-mark everything after the device buffers were reallocated — the
    /// SoT no longer holds what the mirror says it does.
    pub(crate) fn mark_all(&mut self) {
        self.quad.mark_all(self.next_slot);
        self.style.mark_all(self.next_slot);
        self.order.mark_all(self.next_slot);
        self.group.mark_all(self.next_group);
    }
}

/// Smallest power-of-two bucket holding `len` slots, floored at 8 so short
/// labels have headroom to grow a few characters in place.
fn run_bucket(len: u32) -> u32 {
    len.max(8).next_power_of_two().trailing_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole ADR rests on: writing the same value twice
    /// produces no upload the second time.
    #[test]
    fn equality_gate_suppresses_redundant_writes() {
        let mut core = UiCore::new();
        let g = core.group([0.0, 0.0, 100.0, 100.0], [0.0, 0.0]);
        let p = core.rect(g, [0.0, 0.0, 10.0, 10.0], UiStyle::fill(rgb(255, 0, 0)));

        let mut stage = vec![0u32; 4096];
        let mut dirty = vec![0u32; 64];
        core.style.upload(&mut stage, &mut dirty);
        dirty.fill(0);

        core.set_fill(p, rgb(255, 0, 0));
        assert_eq!(core.style.upload(&mut stage, &mut dirty), (i64::MAX, -1));

        core.set_fill(p, rgb(0, 255, 0));
        assert_eq!(core.style.upload(&mut stage, &mut dirty), (0, 0));
    }

    /// Retyping a label must dirty only the glyphs that differ.
    #[test]
    fn text_edit_dirties_only_changed_glyphs() {
        let mut core = UiCore::new();
        let g = core.group([0.0, 0.0, 100.0, 100.0], [0.0, 0.0]);
        let t = core.text(g, [0.0, 0.0], 9.0, rgb(255, 255, 255), "100");

        let mut stage = vec![0u32; 4096];
        let mut dirty = vec![0u32; 64];
        core.quad.upload(&mut stage, &mut dirty);
        dirty.fill(0);

        core.set_text(t, "101");
        core.quad.upload(&mut stage, &mut dirty);
        assert_eq!(dirty.iter().map(|w| w.count_ones()).sum::<u32>(), 1);
    }

    /// The health-bar case: many threads writing their own slots at once,
    /// with no lock, all reach the upload — and an unchanged write does not.
    #[test]
    fn parallel_paint_writes_all_upload() {
        let mut core = UiCore::new();
        let g = core.group([0.0, 0.0, 100.0, 100.0], [0.0, 0.0]);
        let prims: Vec<PrimId> = (0..5000)
            .map(|_| core.rect(g, [0.0, 0.0, 1.0, 1.0], UiStyle::fill(0)))
            .collect();
        let (mut stage, mut dirty) = (vec![0u32; 8 * 8192], vec![0u32; 256]);
        core.style.upload(&mut stage, &mut dirty);
        dirty.fill(0);

        let paint = core.paint();
        std::thread::scope(|s| {
            for chunk in prims.chunks(1250) {
                let paint = &paint;
                s.spawn(move || chunk.iter().for_each(|p| paint.set_cut(*p, 0.5)));
            }
        });
        core.style.upload(&mut stage, &mut dirty);
        assert_eq!(dirty.iter().map(|w| w.count_ones()).sum::<u32>(), 5000);
        assert!(
            prims
                .iter()
                .all(|p| f32::from_bits(stage[p.0 as usize * 8 + 6]) == 0.5)
        );

        dirty.fill(0);
        prims.iter().for_each(|p| paint.set_cut(*p, 0.5));
        assert_eq!(core.style.upload(&mut stage, &mut dirty), (i64::MAX, -1));
    }

    /// Restyling a fill must not refill a bar the game emptied.
    #[test]
    fn set_style_keeps_the_cut() {
        let mut core = UiCore::new();
        let root = core.root();
        let bar = core.progress(root, 0.25, ProgressStyle::default());
        let paint = core.paint();
        assert_eq!(bar.value(&paint), 0.25);

        core.set_style(bar.fill, UiStyle::fill(rgb(255, 0, 0)));
        assert_eq!(bar.value(&paint), 0.25);
    }

    /// Deferred work runs in the order one thread queued it, and only once.
    #[test]
    fn deferred_runs_in_order_once() {
        let mut core = UiCore::new();
        let root = core.root();
        let log = Arc::new(Mutex::new(Vec::new()));
        for i in 0..3 {
            let log = log.clone();
            defer(move |ui| {
                ui.label(root, 9.0, 0, "x");
                log.lock().push(i);
            });
        }
        core.run_deferred();
        core.run_deferred();
        assert_eq!(*log.lock(), [0, 1, 2]);
    }

    /// A run that outgrows its bucket reallocates; the old slots must go
    /// back on the free list and stop drawing.
    #[test]
    fn run_growth_recycles_slots() {
        let mut core = UiCore::new();
        let g = core.group([0.0, 0.0, 100.0, 100.0], [0.0, 0.0]);
        let t = core.text(g, [0.0, 0.0], 9.0, rgb(255, 255, 255), "short");
        let first = core.runs[t.0 as usize].first;

        core.set_text(t, "a string well past eight characters");
        assert_ne!(core.runs[t.0 as usize].first, first);
        assert_eq!(core.free_runs[3].last(), Some(&first));
        for i in 0..8 {
            assert_eq!(core.quad.get(first + i).rect[2], 0.0);
        }
    }
}
