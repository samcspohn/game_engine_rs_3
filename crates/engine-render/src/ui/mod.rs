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

pub mod font;
mod gpu;
mod tree;
mod widget;

use std::sync::{Mutex, MutexGuard, OnceLock};

pub use gpu::UiGpu;
pub use tree::{style, NodeId};
pub use widget::{ButtonStyle, StateStyle};

use crate::transform_gpu::dirty_word_count;

// ─────────────────────────────────────────────────────────────────────
// The global store
// ─────────────────────────────────────────────────────────────────────

static UI: OnceLock<Mutex<UiCore>> = OnceLock::new();

/// The process-wide UI store.
///
/// Global for the reason `engine_core::asset::global` is: `Component::init`
/// and `Component::update` can reach a static but not the renderer's
/// `RenderContext`. Widening the `Component` trait to carry a context would
/// touch `ComponentStorage::par_iter`'s fan-out and every component in the
/// workspace, to solve what two existing subsystems already solved this way.
///
/// `Scene::update` runs components in parallel, so this is a shared lock on
/// a parallel path. It is uncontended by construction rather than by luck:
/// UI writes happen on *events* (a value changed, a panel opened), and the
/// steady state — including a widget anchored to a moving entity, once
/// ADR-0008 lands — touches it not at all.
pub fn global() -> &'static Mutex<UiCore> {
    UI.get_or_init(|| Mutex::new(UiCore::new()))
}

/// Lock the global UI store. Panics if a previous holder panicked, which is
/// the intended behaviour: a poisoned UI is a bug to find, not to route
/// around.
pub fn ui() -> MutexGuard<'static, UiCore> {
    global().lock().expect("ui store mutex poisoned")
}

/// Primitive kinds, matching `ui.frag`'s `KIND_*` constants. Stored in the
/// low byte of `UiStyle::kind_flags`.
pub const KIND_RECT: u32 = 0;
pub const KIND_TEXT: u32 = 1;
pub const KIND_IMAGE: u32 = 2;

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
}

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
    reserved: [u32; 2],
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
            self.reserved[0],
            self.reserved[1],
        ]);
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
        self.radius = r
            .iter()
            .enumerate()
            .fold(0, |acc, (i, &v)| acc | ((v.clamp(0.0, 255.0) as u32) << (8 * i)));
        self
    }

    /// Extra edge softness in px on top of the analytic one-pixel AA.
    pub fn softness(&self) -> f32 {
        half::f16::from_bits((self.border_width >> 16) as u16).to_f32()
    }

    pub fn soft(mut self, px: f32) -> Self {
        self.border_width = pack_half2(
            half::f16::from_bits(self.border_width as u16).to_f32(),
            px,
        );
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
}

/// One entry of the draw list: `(primitive slot, group id)`, back-to-front.
///
/// The indirection is the scene pass's `instance_to_entity` trick — it
/// decouples draw order from slot identity, so inserting a widget in the
/// middle of the z-order will rewrite only this array. Phase 1/2 assign
/// `order[i] = (i, gid)`, i.e. painter's order is allocation order; the
/// array exists now so the shader is final when phase 3 decouples them.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub(crate) struct OrderEntry([u32; 2]);

impl Record for OrderEntry {
    const STRIDE: usize = 2;
    fn write(&self, out: &mut [u32]) {
        out.copy_from_slice(&self.0);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Change detection
// ─────────────────────────────────────────────────────────────────────

/// A CPU mirror of one device array plus its per-slot dirty bitmask.
///
/// The mirror is not a cache — it is the change *detector*. Every write path
/// funnels through [`Self::set`], so nothing that didn't actually change can
/// reach the staging buffer no matter how sloppily the layer above it is
/// written.
pub(crate) struct SlotArray<T: Record> {
    values: Vec<T>,
    dirty: Vec<u32>,
    /// Lowest / highest dirty *word* this frame, bounding the compaction
    /// prepass's scan. `max < 0` means "nothing dirty" and produces a
    /// zero-workgroup prepass — and transitively a zero-workgroup scatter.
    min_word: i64,
    max_word: i64,
}

impl<T: Record> SlotArray<T> {
    fn new(capacity: usize) -> Self {
        Self {
            values: vec![T::default(); capacity],
            dirty: vec![0; dirty_word_count(capacity)],
            min_word: i64::MAX,
            max_word: -1,
        }
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn set(&mut self, i: u32, v: T) {
        if self.values[i as usize] == v {
            return; // ← the entire premise
        }
        self.values[i as usize] = v;
        self.mark(i);
    }

    fn get(&self, i: u32) -> T {
        self.values[i as usize]
    }

    fn mark(&mut self, i: u32) {
        let w = (i / 32) as usize;
        self.dirty[w] |= 1 << (i % 32);
        self.min_word = self.min_word.min(w as i64);
        self.max_word = self.max_word.max(w as i64);
    }

    /// Grow the mirror, preserving contents. New slots hold `T::default()`
    /// and are left clean — nothing draws them until something writes them.
    fn grow(&mut self, capacity: usize) {
        self.values.resize(capacity, T::default());
        self.dirty.resize(dirty_word_count(capacity), 0);
    }

    /// Mark every populated slot dirty. Used after the device buffers are
    /// reallocated, when the SoT no longer holds what the mirror says.
    fn mark_all(&mut self, count: u32) {
        for i in 0..count {
            self.mark(i);
        }
    }

    /// Copy this frame's dirty slots into `stage`, publish the bitmask into
    /// `dirty`, and reset the watermarks. Returns `[min_word, max_word]` for
    /// the prepass bounds.
    ///
    /// Only dirty words are touched in either buffer — the device-side
    /// `fill_buffer(0)` in the scatter secondary is what keeps the rest of
    /// the bitmask zero between frames.
    fn upload(&mut self, stage: &mut [u32], dirty: &mut [u32]) -> (i64, i64) {
        let bounds = (self.min_word, self.max_word);
        if self.max_word >= 0 {
            for w in self.min_word as usize..=self.max_word as usize {
                let mut bits = self.dirty[w];
                if bits == 0 {
                    continue;
                }
                dirty[w] = bits;
                self.dirty[w] = 0;
                while bits != 0 {
                    let i = w * 32 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let base = i * T::STRIDE;
                    self.values[i].write(&mut stage[base..base + T::STRIDE]);
                }
            }
        }
        self.min_word = i64::MAX;
        self.max_word = -1;
        bounds
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
pub struct PrimId(u32);

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
    pub(crate) style: SlotArray<UiStyle>,
    pub(crate) group: SlotArray<UiGroup>,
    pub(crate) order: SlotArray<OrderEntry>,

    /// Free runs bucketed by `log2(len)`; index `k` holds first-slots of
    /// free runs of `2^k` slots.
    free_runs: Vec<Vec<u32>>,
    /// Bump watermark. Also the draw list's length — freed slots stay in
    /// `order` holding a zero-area quad rather than forcing a compaction.
    next_slot: u32,
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
}

/// Hover / press / click state for the one system pointer.
///
/// Recomputed before `Scene::update` so a component's `clicked()` observes
/// this frame's input, and so a camera controller can decline to orbit when
/// the UI took the press.
#[derive(Default)]
pub(crate) struct Pointer {
    /// Nodes that accept the pointer, indexed by `NodeId`. Opt-in: a node is
    /// inert until `set_interactive`, so a panel's decorative boxes never
    /// swallow clicks meant for the world.
    pub(crate) interactive: Vec<bool>,
    pub(crate) pos: [f32; 2],
    pub(crate) hovered: Option<NodeId>,
    /// Node a press landed on, held until release. A click fires only when
    /// the release lands on that same node — standard "drag off to cancel".
    pub(crate) down_on: Option<NodeId>,
    /// Set for exactly one frame, by the release that completed a click.
    pub(crate) clicked: Option<NodeId>,
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
        let mut group = SlotArray::new(16);
        group.set(0, UiGroup::default());
        Self {
            quad: SlotArray::new(256),
            style: SlotArray::new(256),
            group,
            order: SlotArray::new(256),
            free_runs: (0..=MAX_RUN_LOG2).map(|_| Vec::new()).collect(),
            next_slot: 0,
            next_group: 1,
            runs: Vec::new(),
            tree: tree::Tree::new(GroupId(0)),
            pointer: Pointer::default(),
            state_styles: Vec::new(),
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

    // ── Groups ──────────────────────────────────────────────────────

    pub fn group(&mut self, clip: [f32; 4], offset: [f32; 2]) -> GroupId {
        let id = self.next_group;
        self.next_group += 1;
        self.reserve_groups(self.next_group as usize);
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
        self.quad.set(
            slot,
            UiQuad {
                rect,
                uv: [0.0; 4],
            },
        );
        self.style.set(slot, style);
        PrimId(slot)
    }

    /// An [`KIND_IMAGE`] primitive sampling the engine's bindless array.
    /// `tint` multiplies the sampled texel; use opaque white for none.
    pub fn image(&mut self, g: GroupId, rect: [f32; 4], tex: u32, tint: u32) -> PrimId {
        let slot = self.alloc_run(0, g);
        self.quad.set(
            slot,
            UiQuad {
                rect,
                uv: [0.0, 0.0, 1.0, 1.0],
            },
        );
        self.style.set(
            slot,
            UiStyle {
                fill: tint,
                kind_flags: KIND_IMAGE,
                tex,
                ..Default::default()
            },
        );
        PrimId(slot)
    }

    pub fn set_rect(&mut self, p: PrimId, rect: [f32; 4]) {
        let mut q = self.quad.get(p.0);
        q.rect = rect;
        self.quad.set(p.0, q);
    }

    pub fn set_style(&mut self, p: PrimId, style: UiStyle) {
        self.style.set(p.0, style);
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

    pub fn text(
        &mut self,
        g: GroupId,
        pos: [f32; 2],
        px: f32,
        color: u32,
        s: &str,
    ) -> TextId {
        let cap_log2 = run_bucket(s.chars().count() as u32);
        let first = self.alloc_run(cap_log2, g);
        let id = TextId(self.runs.len() as u32);
        self.runs.push(TextRun {
            group: g,
            first,
            cap_log2,
            pos,
            scale: px / font::GLYPH_H as f32,
            color,
            text: String::new(),
        });
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
        for (i, ch) in s.chars().enumerate() {
            let slot = first + i as u32;
            self.quad.set(
                slot,
                UiQuad {
                    rect: [pos[0] + i as f32 * advance, pos[1], size[0], size[1]],
                    uv: font::glyph_uv(ch),
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
        assert!(log2 <= MAX_RUN_LOG2, "run of 2^{log2} slots exceeds the allocator");
        let n = 1u32 << log2;
        let first = match self.free_runs[log2 as usize].pop() {
            Some(f) => f,
            None => {
                let f = self.next_slot;
                self.next_slot += n;
                self.reserve_slots(self.next_slot as usize);
                f
            }
        };
        for i in 0..n {
            self.order.set(first + i, OrderEntry([first + i, g.0]));
            self.quad.set(first + i, UiQuad::default());
        }
        first
    }

    fn free_run(&mut self, first: u32, log2: u32) {
        for i in 0..(1u32 << log2) {
            self.quad.set(first + i, UiQuad::default());
        }
        self.free_runs[log2 as usize].push(first);
    }

    fn reserve_slots(&mut self, needed: usize) {
        if needed <= self.quad.len() {
            return;
        }
        let cap = needed.next_power_of_two();
        self.quad.grow(cap);
        self.style.grow(cap);
        self.order.grow(cap);
    }

    fn reserve_groups(&mut self, needed: usize) {
        if needed <= self.group.len() {
            return;
        }
        self.group.grow(needed.next_power_of_two());
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
