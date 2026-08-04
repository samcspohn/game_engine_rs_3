# ADR-0006: Retained-Mode UI on the Scatter/SoT Paradigm

**Status:** Phases 1–2 landed, phase 3 layout landed; phase 3 API + phase 4 proposed
**Date:** 2026
**Scope:** `crates/engine-render/src/ui/`, `shaders/ui.{vert,frag}`, `shaders/ui_{scatter,build_args}.comp`, two new blocks in `build_frame_slot`, `taffy` dependency
**Related:** [ADR-0003](ADR-0003-shared-staging-with-compute-sync.md) (staging → scatter → SoT), [ADR-0004](ADR-0004-instanced-indirect-draw.md) (single indirect draw), [ADR-0008](ADR-0008-ui-integration.md) (who builds a UI, and world-anchored widgets)

## Context

The engine has no UI. Every existing option (Dear ImGui, egui) is
**immediate mode**: it rebuilds the full vertex + index buffer every frame
and re-records draw commands every frame, because it has no memory of what
the UI looked like last frame. That is precisely the model the rest of this
engine has spent five ADRs deleting.

The renderer's established shape is:

1. **Stable device-local SoT arrays** indexed by slot id, never rebuilt.
2. **Host staging + a per-slot dirty bitmask**; a compute *scatter* promotes
   only dirty slots into the SoT.
3. **A word-compaction prepass** sizes the scatter's `dispatch_indirect` to
   the true dirty-*word* count, so N dirty slots cost N/32 workgroups
   regardless of capacity.
4. **Pre-recorded `MultipleSubmit` command buffers**, rebuilt only on
   structural change (capacity grow, swapchain resize).
5. **One indirect draw**, with the instance count living in a device buffer
   so the CB never needs re-recording when the draw list changes.

A retained-mode UI is the *same problem*. A widget tree is a set of
long-lived objects whose per-frame delta is almost always empty, and whose
non-empty delta is tiny and local (one button's fill color; one label's
glyphs). Points 1–5 map onto it with essentially no new machinery — the UI
becomes another consumer of the paradigm rather than an exception to it.

## Decision

Model the UI as **a flat array of primitive slots** in device-local SoT
buffers, updated by dirty-bitmask scatter, drawn by **exactly one
`draw_indirect` per frame** from a command buffer recorded once per session.

The retained widget tree lives entirely on the CPU. Its only job is to
decide *which slots changed*.

The API is **truly retained** — built once, mutated by event-driven
callbacks — not an immediate-mode authoring layer over retained storage.
See "The API layer" below for why that distinction is load-bearing here and
not merely stylistic.

### The data model

Three slot-indexed device-local arrays, plus one structural array. Each of
the three gets its own dirty bitmask, its own compaction prepass and its own
scatter dispatch — the split is by **change frequency**, exactly the reason
position/rotation/scale are three buffers and not one.

```
ui_quad[slot]   — 32 B — geometry. Changes on layout / reflow / resize.
    vec4 rect;   // x, y, w, h, in group-local physical px
    vec4 uv;     // u0, v0, u1, v1 — atlas cell or texture sub-rect

ui_style[slot]  — 32 B — appearance. Changes on hover / press / focus / theme.
    uint fill;          // rgba8, non-premultiplied sRGB
    uint border;        // rgba8
    uint radius;        // 4 × u8 per-corner radius, px
    uint border_width;  // f16 width | f16 edge softness
    uint kind_flags;    // RECT | TEXT | IMAGE, + flags
    uint tex;           // bindless TextureId, or UINT_MAX
    uvec2 _reserved;

ui_group[gid]   — 32 B — clip + transform + opacity. ~1 per window/panel/scroll area.
    vec4  clip;    // x0, y0, x1, y1, physical px, already intersected on the CPU
    vec2  offset;  // translation: window position + scroll
    float opacity;
    uint  flags;

ui_order[i] = uvec2(slot, gid)   — the draw list, back-to-front.
```

**As landed:** `ui_order` is a **fourth scattered array**, not a wholesale
`copy_buffer`. It gets its own dirty mask, prepass and scatter dispatch at
stride 2, exactly like the other three. The copy was rejected on the same
grounds the rest of the design rests on: a pre-recorded CB has to record the
copy unconditionally, so a full-capacity transfer would run on *every* frame
including idle ones — the one thing this ADR exists to avoid. Making it the
fourth consumer of machinery that already exists costs one binding in the
args builder and nothing at runtime.

Screen extent and DPI scale (the px → NDC conversion) are a **baked push
constant**, not a per-frame buffer: they change only on resize, and a resize
already forces the CB rebuild. With animation on the host there is no
per-frame clock the shaders need either, so the UI pass reads nothing that
changes between rebuilds except the four arrays above — which is what keeps
the draw side free of any promotion step.

The **instance count** is the one exception, and implementing it surfaced a
constraint the design sketch had missed. The UI draw runs at the very end of
the frame CB — long after `signal_cs` has released the host to overwrite
staging for the next frame — so *nothing the draw reads may live in
host-visible memory*, including its own `VkDrawIndirectCommand`. The count is
therefore host-staged into a one-word buffer and promoted into a device-local
draw command by `ui_build_args.comp`, which runs before `signal_cs` alongside
the dispatch-args it was already writing. This is what lets the primitive
count change without re-recording anything.

`ui_order` is the same trick as the scene pass's `instance_to_entity`: it
decouples **draw order** from **slot identity**. Slots come from a free list
and never move, so inserting a widget in the middle of the z-order rewrites
only this array, not every record after the insertion point. It is rebuilt
wholesale (a `copy_buffer` from staging) on structural frames — at 20 K
primitives that is 160 KB on the rare frame a menu opens, and zero bytes
otherwise.

Carrying `gid` in the order entry rather than in `ui_style` keeps group
re-assignment (structural) out of the hot appearance record.

### Groups are the incremental-update multiplier

`ui_group` is what makes the common bulk operations O(1) instead of
O(primitives):

| Operation | Records dirtied |
|---|---|
| Hover a button | 1 `ui_style` |
| Drag a window | 1 `ui_group` (offset) |
| Scroll a 10 000-row list | 1 `ui_group` (offset) |
| Fade a panel out | 1 `ui_group` (opacity) |
| Resize a window | 1 `ui_group` + the panel's own quads (real relayout) |
| Retype a 12-char label | 12 `ui_quad` |
| Slide a panel in over 200 ms | 1 `ui_group` per frame, for the duration |
| Nothing happened | **0** |

Groups are **flat**, not a hierarchy: nested windows/scroll areas have their
parent's offset and clip composed in on the CPU. Group counts are in the
hundreds, so composing them is free, and it keeps the vertex shader to a
single group fetch with no parent-chain walk. (`mvp_build.comp` walks a
parent chain because there are a million transforms and the CPU can't
afford to; here the opposite is true.)

Note that this makes animation cheap *regardless of who evaluates it*:
sliding a whole window is one record per frame either way. See
"Animation is CPU-driven" below for why the evaluator is the CPU.

### The vertex shader generates the quad

No vertex buffer, no index buffer, ever. `draw_indirect` with
`vertex_count = 4`, `TRIANGLE_STRIP`, `instance_count` from the device
buffer:

```glsl
uvec2 e = ui_order[gl_InstanceIndex];
UiQuad  q = ui_quad[e.x];
UiStyle s = ui_style[e.x];
UiGroup g = ui_group[e.y];

vec2 p0 = q.rect.xy + g.offset;
vec2 p1 = p0 + q.rect.zw;

// Clip in the VS by shrinking the quad, not by discarding fragments.
vec2 c0 = max(p0, g.clip.xy);
vec2 c1 = min(p1, g.clip.zw);
if (any(greaterThanEqual(c0, c1))) { gl_Position = vec4(0); return; }  // zero-area

vec2 t   = corner(gl_VertexIndex);            // (0,0) (1,0) (0,1) (1,1)
vec2 pos = mix(c0, c1, t);
v_uv     = mix(q.uv.xy, q.uv.zw, (pos - p0) / q.rect.zw);   // uv follows the clip
```

The zero-area early-out gives free per-primitive culling for three cases at
once: fully-clipped primitives, off-screen primitives, and **freed slots**
(a freed slot is just `rect.zw = 0`, so the order array does not have to be
compacted the instant a widget dies).

The fragment shader branches on `kind_flags`:

* `RECT` — analytic rounded-box SDF, `smoothstep` over one pixel for AA,
  border blended from the same distance. No MSAA required.
* `TEXT` — `coverage = texture(glyph_atlas, uv).r`, output `fill * coverage`.
* `IMAGE` — `texture(u_textures[nonuniformEXT(tex)], uv) * fill`, reusing the
  existing bindless `MAX_TEXTURES` array from `GpuTextureStore`.

### Draw order and blending

Alpha blending needs back-to-front, and a single instanced draw provides it:
Vulkan's primitive order within one draw is defined by instance index, and
blending respects primitive order. So `ui_order` *is* the painter's
algorithm — no depth buffer, no sorting on the GPU, no multi-draw.

Blend state is premultiplied (`ONE, ONE_MINUS_SRC_ALPHA`), which is the only
mode that composites text coverage and nested translucent panels correctly.
The shader premultiplies after the sRGB→linear decode of `fill`.

### Where it lands in the frame

The swapchain image is an sRGB format and the camera's HDR target is blitted
into it (that blit *is* the tonemap/encode). The UI therefore draws
**after** the blit, straight into the swapchain image, so it is never
tonemapped:

```
  … existing frame primary …
  blit_secondary            — camera color (HDR) → swapchain (sRGB)
    ↓ TRANSFER_WRITE → COLOR_ATTACHMENT_WRITE
  begin_rendering(swapchain view, LoadOp::Load, StoreOp::Store)
    ui_secondary            — ONE draw_indirect
  end_rendering
    ↓ → PresentSrc
```

Every one of those transitions is derived by vulkano's auto-sync from the
resource-usage records the secondaries carry; the UI block adds no manual
barrier.

The hardware does the linear→sRGB encode on write to an `_SRGB` attachment,
and blends in linear space, which is what we want.

Two additions at the *front* of the primary, alongside the existing scatters:

```
  world.scatter_secondary
  gpu_renderers.spawn_scatter_secondary
  ui.scatter_secondary(slot)      ← NEW: prepass ×4 → clear ×4 → build_args
                                          (dispatch args + draw args) → scatter ×4
  fill_buffer(trs dirty masks, 0)
  copy_buffer(view_proj …)
  signal_secondary                — gpu_signal++, "host staging is free"
```

Placement before `signal_cs` is load-bearing: the signal is the host's
guarantee that every read of host-visible staging has retired, and UI
staging is host-visible staging. It joins the existing double-buffered
staging slots and the same `host_wait_for_previous_compute` gate — no new
synchronization primitive.

### Where the dirty masks are cleared

Unlike the TRS masks — whose `fill_buffer(0)` sits in the FrameSlot primary —
the UI masks are cleared **inside the UI scatter secondary**, immediately
after the four prepasses consume them. Nothing outside that secondary touches
them, so there is no reason to hoist the clears into the primary, and keeping
them local means `build_frame_slot` gains exactly one `execute_commands`.

The ordering constraint on `signal_cs` is unaffected and still load-bearing:
the UI block is recorded *before* the primary's own `fill_buffer`s, so
vulkano's conservative first-use barrier for `gpu_signal` still lands where
`host_wait_for_previous_compute`'s notes say it does.

### The scatter is a generic one

`scatter.comp` hardcodes stride 3 (position/scale) vs. 2 (packed quaternion)
behind an `is_rotation` push constant, and unpacks half-floats. UI records
are already tightly packed by the CPU and need no decode, so this gets a
sibling shader, `ui_scatter.comp`, which copies `stride_words` u32s per set
bit and nothing else. The compaction prepass (`scatter_prepass.comp`) and
the args builder (`scatter_build_args.comp`) are reused **verbatim** — they
only ever look at bitmasks and counts, never at element type.

Deliberately *not* refactoring TRS onto the generic path: it works, it is
performance-critical, and the quaternion unpack means the shared version
would need a mode switch anyway. If a fourth consumer appears, revisit.

### Text: a dedicated atlas

Glyphs are **one primitive each**, in the same arrays as everything else —
so a label is a contiguous run of slots, and editing its text dirties
exactly that run.

The atlas is a **dedicated** `R8_UNORM` sampled image, *not* a slot in
`GpuTextureStore`. That is a deliberate divergence: `GpuTextureStore::sync`
returning `changed` triggers `force_full` — a descriptor-set + secondary +
frame-slot rebuild. A font has no business dragging every command buffer in
the engine through one. A dedicated image has a stable handle, so its
descriptor set is written once and never again.

**As landed, the atlas is static and complete**, and `glyph_blit.comp` does
not exist. The built-in font (`ui/font.rs`) is a 5×9 bitmap covering every
printable ASCII character, rasterised into a 112×66 image by one fence-waited
`copy_buffer_to_image` at construction. The streamed-upload machinery below
exists in this ADR because glyphs arrive continuously *as the user types new
characters*; with all 95 of them resident from frame zero, they never do. The
whole skyline-packer / `dispatch_indirect` / `fontdue` apparatus was solving
a problem the shipped font does not have, so it was not built.

When a real TTF and arbitrary code points land, the design is unchanged and
still right: the CPU rasterises into a host-visible staging buffer, appends
`{atlas_x, atlas_y, w, h, src_offset}` records, and `glyph_blit.comp`
`dispatch_indirect`s over the live record count, writing into the atlas as a
storage image — zero one-shot submits, zero stalls, zero rebuilds, which is
the whole reason not to use `copy_buffer_to_image` with baked regions there.
That variant needs the atlas in `GENERAL` layout; the static one does not.

Run capacity is rounded up (next power of two) so most edits reuse the run
in place; only crossing a capacity boundary reallocates and touches
`ui_order`.

### Host side: change detection is the storage primitive

```rust
struct SlotArray<T: Copy + PartialEq> {
    values: Vec<T>,        // CPU mirror — the change detector
    dirty:  DirtyBits,
}

impl<T: Copy + PartialEq> SlotArray<T> {
    fn set(&mut self, i: u32, v: T) {
        if self.values[i as usize] == v { return; }   // ← the entire premise
        self.values[i as usize] = v;
        self.dirty.set(i);
    }
}
```

The equality gate at the leaf is what makes everything above it cheap: any
write path — a property setter, a full subtree rebuild, a relayout that
happened to produce identical rects — uploads only what genuinely differs.
Layout gets the usual dirty-flag propagation (mark up, visit down, only into
dirty subtrees) and its output goes through `set`, so an unchanged relayout
uploads nothing.

Dirty counts are 0–50 in the overwhelming majority of frames, so the host
write loop is single-threaded and unremarkable. It does not need the thread
pool.

### Capacity

Primitive, group and order capacities grow geometrically and never shrink,
via a `UiGpu::ensure_capacity` that mirrors `WorldTransformGpu`'s: reallocate
the SoT + staging, mark everything dirty, re-record `ui_scatter_secondary`,
and force a frame-slot rebuild. Rare by construction.

## The API layer: true retained, not immediate-over-retained

An immediate-mode *authoring* API on top of this storage (re-declare the
whole tree every frame; let the equality gate absorb it) would upload
nothing on an idle frame but still cost O(tree) of CPU on every frame —
walking the declaration, re-hashing ids, re-formatting strings, re-solving
layout. That is where egui's 1–3 ms goes, and text formatting is most of it.

That number is disqualifying **in this engine specifically**: the scene pass
renders 1M entities in ~4.5 ms. A UI that burned 2 ms of CPU while
displaying an unchanged toolbar would be the most expensive thing in the
frame. So the API is **truly retained**: the tree is built once, mutated by
event-driven callbacks, and costs approximately nothing when idle
(a drained empty queue and a heap peek — hundreds of nanoseconds).

Target is **editor-class** UI: docking, panels, text editing, live
inspectors.

### `UiCore` / `Ui<S>` — the split that makes closures work

The obvious retained-mode design in Rust — `Box<dyn FnMut>` inside widgets,
capturing `&mut AppState` — does not compile, because the callback lives
inside the very structure it needs to mutate, and it cannot capture the app
state that a real click handler needs (the scene, the selection, the
project). The usual escapes are `Rc<RefCell<AppState>>` threaded everywhere
(runtime borrow panics at a distance, non-`Send` tree) or flattening
callbacks into a message enum (loses the ergonomics).

Neither is necessary. Everything with code volume in it — slot arrays,
groups, order, layout, hit testing, free lists — has nothing to do with the
app state type. Only the callback table does. So split on that line:

```rust
pub struct UiCore { /* SlotArrays, groups, order, layout, hit-test, free lists */ }

pub struct Ui<S> {
    core:      UiCore,
    callbacks: CallbackTable<S>,      // Box<dyn FnMut(&mut S, &mut UiCore)>
    queue:     Vec<(WidgetId, Trigger)>,
    ticks:     TickHeap,
    deferred:  Vec<Deferred<S>>,      // structural ops raised from callbacks
}
```

Dispatch destructures rather than re-borrowing through `self`, so the three
fields are three independent `&mut`:

```rust
pub fn dispatch(&mut self, state: &mut S) {
    let Ui { core, callbacks, queue, .. } = self;
    for (id, trigger) in queue.drain(..) {
        callbacks.get_mut(id, trigger)(state, core);
    }
    // deferred structural ops drained here, where full &mut Ui<S> is available
}
```

No `mem::take`, no take-call-restore, no `Option` holes in the callback
table, and no "the callback destroyed its own widget" resurrection case.
`UiCore` stays monomorphic, so `<S>` touches only this thin shim rather than
going viral through every widget type.

The editor owns the two as sibling fields (`struct EditorApp { state, ui }`)
so `app.ui.dispatch(&mut app.state)` is a field-disjoint borrow.

### Handles and cursors

```rust
pub struct Handle<W>(WidgetId, PhantomData<fn() -> W>);   // Copy + Send + 'static

pub trait Widget { type Mut<'a>; fn cursor(core: &mut UiCore, id: WidgetId) -> Self::Mut<'_>; }

impl UiCore {
    pub fn at<W: Widget>(&mut self, h: Handle<W>) -> W::Mut<'_> { W::cursor(self, h.0) }
}
```

A handle is a generation-checked id, `Copy` and `'static`, so closures
capture it by value with no refcount and no lifetime. A stale handle panics
rather than silently no-opping.

```rust
let hp = ui.label(panel, "100").id();

ui.button(panel, "Hit").on_click(move |app: &mut Editor, ui| {
    app.hp -= 10;
    ui.at(hp).set_text(app.hp).set_fill(RED);
});
```

`app` arrives as a parameter rather than a captured `self` — the one
concession, and it is what buys `&mut EditorState` and `&mut UiCore` live
simultaneously, which a captured `self` can never provide.

**Why a cursor and not `ui[hp]` or `hp.set_text(x)`:**

* `IndexMut` must return `&Self::Output`, but `set_text` needs the glyph-run
  allocator and dirty bits that live on `UiCore`, not on `Label`. Indexing
  cannot express it.
* A bare `hp.set_text(x)` would have to find `UiCore` through an ambient
  thread-local. That is *soundly* implementable only for methods that return
  nothing (`hp.get_mut()` returning a borrow from an ambient pointer aliases
  and is UB), and it costs three things the cursor gives for free: one
  lookup per edit *sequence* instead of per setter; batched layout
  invalidation on cursor `Drop` instead of conservative invalidation per
  call; and static enforcement of "no widget mutation during layout" — a
  callback holds `&mut UiCore`, layout and draw hand it to nobody.

Deferred deliberately. If threading `&mut UiCore` through editor helper
functions proves to be real friction, an ambient sugar layer over the same
cursor is ~40 lines and changes no data structure — see "Revisit if".

### Animation is CPU-driven

Transitions are evaluated on the CPU, by an **animator set** — the same
"maintain the index of the interesting subset, never scan" move as the tick
set and the dirty bitmask. Entries are removed when they settle, so an idle
UI holds an empty set.

Encoding animations into the GPU records instead (`from`/`to`/`t0`/`curve`
per group, evaluated in the vertex shader against a time uniform) was
considered and rejected. It saves one write per frame per animating group —
a cost that was already negligible — and gives up four things that are not:

1. **The CPU stops knowing where things are.** Hit testing runs on the host
   against `ui_order` and the group rects. If the GPU owns the animated
   offset, a panel sliding into place cannot be correctly clicked *during*
   the slide, because the host only has `offset_to`. That is a correctness
   bug, not a limitation.
2. **Layout-affecting animation becomes impossible.** An accordion
   expanding, a list row growing, a disclosure opening — all of these must
   reflow siblings, and layout is a host pass. A GPU-only size animation
   would visibly animate while the layout underneath it stayed wrong.
3. **Interruption needs the current value.** Hover a button, unhover it
   mid-fade: the new animation's `from` is wherever the old one had reached,
   and a spring additionally carries its velocity across. Only the side
   holding the evaluator can answer that, so the host would need the
   evaluator anyway — and then it is implemented twice.
4. **It covers two properties out of six.** Offset and opacity live on
   `ui_group`; color transitions live in `ui_style` and size/rect
   transitions in `ui_quad`, both of which are **per primitive**.
   Generalizing would mean adding `from`/`to` to those records — doubling
   32 B → 64 B across every primitive, so all ~20 000 static quads pay for
   animation fields they never use. The group scheme only looked cheap
   because groups number in the hundreds.

Closed-form GPU curves also rule out springs, which have no fixed duration
and integrate per frame — and springs are what modern UI motion actually
wants.

The cost of doing it on the host is not the reason to avoid it. Per active
animation per frame: one fn-pointer call, a curve evaluation, and a
`SlotArray::set`. Fifty concurrent animations is ~1 µs of CPU and two dirty
words. Both sides of the trade are noise; the decision is entirely about
capability.

```rust
struct Animation {
    target: WidgetId,
    apply:  fn(&mut UiCore, WidgetId, [f32; 4]),   // ← new kinds are new fns
    from:   [f32; 4], to: [f32; 4],
    curve:  Curve,                                  // incl. Spring { stiffness, damping, vel }
    t0:     f32, inv_dur: f32,
    on_complete: Option<CallbackId>,
}
```

`[f32; 4]` covers scalar, `vec2`, rect and color uniformly, and `apply` as a
plain fn pointer means a new animatable property is a new function rather
than a new enum variant — which is the extensibility the GPU encoding could
not offer. Completion enqueues a trigger into the normal callback queue, so
"fade out, then remove" needs no special machinery.

`ui.animate(target, to, dur)` retargets in place when an animation on the
same target is already live: current value becomes the new `from`, spring
velocity carries over.

### Ticks

Widgets that genuinely need clock-driven updates register into a due-time
heap rather than being discovered by a tree walk — the same move as the
dirty bitmask: maintain the index of the interesting subset, never scan.

```rust
ui.label(panel, "").on_tick(Hz(10), |app, ui, _dt| ui.at(fps).set_text(app.fps));
```

Entries carry an interval, so the heap is popped only for what is actually
due. A stats readout at 10 Hz in a 240 FPS editor does 10 text relayouts per
second, not 240 — the human reads it at 10 Hz either way.

The tick set should stay near-empty, and every entry in it is a question
with two better answers: an event, or the animator set. Animations in
particular are not ticks — they need interruption, retargeting, spring
state and completion callbacks, none of which a bare periodic callback
provides.

### Frame order and re-entrancy rules

```
input events → hit test → enqueue triggers → dispatch (callbacks run)
             → drain deferred structural ops → advance animations
             → run due ticks → layout dirty subtrees → write staging → submit
```

Animations advance **before** layout, because a size animation must reflow
its siblings in the same frame it moves.

1. **Callbacks run only in `dispatch`.** Never during layout or draw, so no
   callback can observe a half-solved tree. Enforced by the borrow checker:
   only dispatch hands out `&mut UiCore`.
2. **Triggers raised by a callback** land in the fresh queue and are
   processed in the same frame by looping. The loop is capped (16 rounds)
   and **panics** with the offending widget id beyond it. An event cycle is
   a bug, not something to quietly defer.
3. **Property mutation is immediate; widget creation is deferred.**
   `set_text` / `set_fill` / `set_rect` go straight through `UiCore`, which
   owns the slot free list — including growing a label's glyph run, which is
   not structural at the widget level. Only spawn / remove / rebuild queue a
   `Deferred<S>`, drained right after the loop where full `&mut Ui<S>` is
   available. This is exactly what makes the destructure in `dispatch`
   legal.
4. **Value-bearing widgets deliver their value in the trigger payload**
   (`on_change(|app, ui, v: f32| app.speed = v)`) rather than exposing a
   readback. App state stays authoritative; the widget is pure
   presentation. This removes the staleness bug class pointed the other way
   (UI as source of truth, app state mirroring it).

### Layout is taffy

The ADR originally specified layout's *invalidation* — mark up, visit down,
only into dirty subtrees, output through `set` — but never said what layout
**computes**. That gap is now closed: **CSS flexbox and grid, via
[`taffy`](https://crates.io/crates/taffy)**, wrapped by `ui/tree.rs`.

Two families were considered.

**Container models** (stacks, flex, grid) resolve in two passes. *Measure*
runs bottom-up: the parent hands each child a min/max constraint, the child
returns its desired size. *Arrange* runs top-down: the parent now knows its
own size and every child's, so it assigns final rects and distributes
leftover space among flex children. Two passes because the dependency runs
both ways — you cannot position children before sizing them, and you cannot
size a flexible child before knowing the parent's room.

**Constraint solvers** (Cassowary; Apple's Auto Layout) instead take
inequalities between edges with priorities and run simplex over them. More
expressive — you can relate widgets in unrelated subtrees — and rejected for
a structural reason rather than a performance one: **it is one global system
of equations.** Nudging any variable can ripple anywhere, so "relayout this
subtree" is not a thing the model can express. That is incompatible with an
architecture built end-to-end on local invalidation.

The container model decomposes exactly the way this design needs: a
subtree's layout depends only on the constraint handed down and its own
children, so an unchanged constraint over a clean subtree is skippable.

**Why the dependency, in a codebase that owns its hot paths.** Layout here
runs at *event* frequency, not per frame, so the usual argument — control
over the microseconds — does not apply. Meanwhile flexbox's edge cases
(min/max against flex-basis, percentage resolution under indefinite parents,
baseline alignment) and grid track sizing are weeks of work with no payoff
we would feel. Taffy costs three tiny transitive crates (`arrayvec`, `grid`,
`slotmap`) with block/float/`calc` features disabled.

Its invalidation model is not merely compatible, it is the same one:
`mark_dirty` walks up the ancestor chain and **stops as soon as it finds a
node already dirty**, and every node caches its `(constraint) → (size)`
result, so a clean subtree entered with an unchanged constraint
short-circuits. Its output then goes through `SlotArray::set`, so a
relayout that recomputes identical rects still uploads zero bytes — taffy
never needs to know the equality gate exists.

Text measurement is trivial and needs no callback into the run store: the
built-in font has a fixed advance, so a string's natural size is known the
moment its text is set and is published to taffy as node context.

`ui::style` re-exports taffy's vocabulary rather than wrapping it. It *is*
the CSS box model, which is what "flexbox and grids" means; a bespoke
synonym layer would only obscure a spec most people already know.

**Not yet done.** The placement walk that converts taffy's parent-relative
positions into absolute rects is O(tree) on any frame where anything moved.
The *upload* stays O(what changed) because every write goes through `set`,
which is the property that matters — but if the walk itself ever shows up,
cache each node's absolute rect and skip subtrees whose origin and size both
held.

### Docking (phase 4)

Docking is not a layout algorithm; it is **state the layout algorithm
consumes**, and the design hinges on one choice: make adjacency
**structural, not geometric**. Store the dock layout as a binary split tree,
each node carrying one axis and one ratio. A splitter then *is* a node, and
the only two things it can affect are that node's two children — so
"adjacent panels shrink and grow" stops being something to compute and
becomes true by construction. Dragging writes one number and dirties one
subtree.

Which makes a dock split a flex row/column whose parameters are mutable user
state rather than authored constants. No second layout system.

Two things this will need that are not built:

* **Per-child `Fixed(px)` / `Fill(weight)`,** not a bare ratio. A pure ratio
  scales a 250 px properties sidebar when the window resizes, which is
  wrong; the viewport should absorb it.
* **Intrinsic minimums as a query.** A drag must not shrink a panel below
  what its contents need, so the drag handler has to ask layout for the
  minimum size of the subtrees on both sides before clamping. Taffy computes
  and caches these already.

The payoff on this architecture: give each dock panel its own `ui_group` and
a panel that merely **translates** — a fixed-width sidebar when the window
widens, or any fixed-size subtree pushed sideways — is **one record and zero
quad writes**, however many widgets it contains. That is the multiplier in
the table above landing in the case editors actually hit.

### Subtree rebuild: the escape hatch for dynamic content

The real cost of a truly retained tree is **staleness** — the widget holds a
copy, and if nobody calls the setter the UI silently lies. For editor-class
UI the dynamic content is inspectors, scene hierarchies, asset browsers and
consoles, all of which change on *events* (selection changed, entity
spawned), not per frame. So:

```rust
ui.rebuild(inspector, |b| build_inspector(b, &editor.selection));
```

is a `Deferred<S>` carrying the builder — one event, one closure, one
subtree re-derived from state. That gives immediate-mode ergonomics locally,
at event frequency, instead of globally at frame frequency.

**Keyed reconciliation matters here for a non-obvious reason.** A naive
rebuild that frees all children and reallocates defeats the equality gate:
fresh slots hold defaults, so every field reads as changed and the whole
panel uploads. Matching new widgets to existing slots by stable key is what
preserves "only changed data uploads" *across* rebuilds — re-selecting an
entity of the same type then uploads only the fields that actually differ.
Ship rebuild-as-realloc first (an event frame can afford it) and add keying
when panel rebuilds show up in the dirty-word counters; the two mechanisms
are designed to compose, so nothing here forecloses it.

### Hit testing needs no tree walk

`ui_order` is already maintained in painter's order, so hit testing is a
**reverse linear scan of it** — first hit wins. Reject whole panels up front
against the group clip rects (hundreds of them), then scan only the
survivors. It runs on pointer-move and click events, not per frame.

## Consequences

### Wins

* **The steady-state UI costs zero bytes and zero dispatch groups.** An idle
  frame runs one `draw_indirect`; the three prepasses see `word_count = 0`
  and the scatter's indirect args come out `(0,0,0)`.
* **A hover is 32 bytes and one workgroup.** No relayout, no vertex-buffer
  rebuild, no CB re-record.
* **The UI command buffer is recorded once per session** — only a capacity
  grow or a swapchain resize touches it.
* **No new synchronization.** UI staging rides the existing dual staging
  slots and the `gpu_signal` early-wake gate.
* **The idle frame costs ~nothing on the CPU either** — an empty
  `Vec::drain` and a heap peek, versus O(tree) for any immediate-mode
  authoring API.
* **Animating a whole window is one record per frame**, not one per
  primitive — `ui_group` gives that regardless of which side evaluates the
  curve.
* **The GPU's role stays "draw what the arrays say".** No simulation, no
  state evolution, no clock — so the host always knows where every widget
  is, which is what hit testing and layout require.
* **The editor viewport becomes a UI primitive.** Register the camera's
  color image in the bindless array and the viewport is an `IMAGE` quad —
  docking, resizing and overlaying it need no renderer special-casing.
* **Render-on-demand becomes possible.** With a global "anything dirty"
  flag (UI, scene, camera) the editor can skip the submit and not present.
  Immediate mode forecloses this outright: determining that nothing changed
  requires running the UI. Not in scope here — the engine currently
  free-runs on mailbox — but true retained is the precondition and this
  design does not block it.

### Costs

* A retained tree is more code than an immediate-mode one, and the widget
  layer has to maintain slot ownership (runs, free lists, order rebuilds)
  that immediate mode gets for free by rebuilding everything.
* **Staleness is the standing bug class.** A widget holds a copy of what it
  displays; miss a setter and the UI lies. Rules 3–4 above and the subtree
  rebuild hatch contain it, they do not eliminate it.
* One `Box<dyn FnMut>` allocation per interactive widget at build time
  (~500 for an editor). Once, never per frame.
* Four new shaders and a fourth staging consumer to keep in lockstep with
  the staging-slot / `signal_cs` ordering rules, which the ADR-0003 docs
  already flag as fragile.
* The UI keeps its **own** copy of the bindless texture descriptors rather
  than sharing the scene's set, so a texture arrival rebinds twice. That is
  1024 descriptors and no memory; sharing would have coupled the UI pipeline
  layout to the scene's, which is the more expensive kind of coupling.
* Draw order is allocation order until phase 3 populates `ui_order`
  independently. The array and the shader are already final; only the host
  side assigns `order[i] = (i, gid)`.
* Full-screen UI overdraw is paid every frame regardless of dirtiness. The
  design optimizes *upload*, not raster. Acceptable: UI quads are trivially
  shaded, and the alternative (damage regions + a cached UI target) only
  pays off when the 3D scene is also static.
* `taffy` is a new dependency and brings CSS semantics with it. Accepted:
  layout is not a hot path here, and the alternative is reimplementing
  flexbox and grid track sizing.
* The built-in bitmap font rasterizes ASCII only; real typography needs a
  TTF rasterizer and, for complex scripts, `swash`/`cosmic-text` — at which
  point the streamed glyph atlas above stops being hypothetical.

### Caveats

* `ui_scatter_secondary` **must** be recorded before `signal_secondary`.
  See `host_wait_for_previous_compute`'s notes on what actually orders that
  dispatch.
* Sampling the glyph atlas from `GENERAL` layout is legal but not the
  optimal layout; if it ever measures, add the two transitions.
* The bindless `IMAGE` path inherits `GpuTextureStore`'s `force_full`
  rebuild on texture arrival. Fine for static UI imagery; do not route
  anything streaming through it.

## Implementation plan

**Phase 1 — the pipeline. ✅ Landed.** `ui_quad` / `ui_style` / `ui_order`
+ generic scatter + one `draw_indirect`.

**Phase 2 — groups and text. ✅ Landed.** `ui_group` (clip / offset /
opacity), the glyph atlas, `TEXT` and `IMAGE` kinds, power-of-two run
allocation with a bucketed free list.

**Measured.** The built-in overlay (`ui/demo.rs`, F6, `ENGINE_UI_TRACE=1`)
holds 234 primitives — a bordered rounded panel, an accent bar, and five
strings including the full printable-ASCII specimen. At 1280×720 the engine
runs ~11 900 FPS with it on screen, and the trace prints **1–2 dirty words
roughly once every 900 frames** — the 10 Hz readout line, and nothing else,
ever. Frames on which the UI uploads a single byte are ~0.1% of frames. That
is phase 1's acceptance test, passed at the number the ADR predicted rather
than merely in the right direction.

**Phase 3a — the tree and layout. ✅ Landed.** `ui/tree.rs`: a node tree,
taffy-backed flexbox/grid layout, measured text leaves, and the placement
walk that pushes solved rects through `SlotArray::set`. The demo overlay is
now laid out rather than hand-positioned — a shrink-wrapping flex column
plus a five-track grid strip, with no coordinate computed by hand.

**Measured.** 238 primitives, ~12 800 FPS. The dirty-word trace shows the
reflow billing correctly: a readout edit that keeps the panel's width costs
**1 word** (the one glyph that differs), while one that changes the widest
line costs **7** — the panel background, the grid strip, and its five
swatches, all genuinely resized. Every other frame is zero.

**Phase 3b — the API.** `Ui<S>` (handles, cursors, callback + tick +
animator tables, deferred structural ops, frame order) and hit testing,
which runs before `Scene::update` so game components can ask whether the UI
captured the pointer. Subtree rebuild lands here; keyed reconciliation does
not.

**Phase 4 — editor integration.** Camera color image into the bindless
array; viewport-as-primitive; docking (see above).

**Ownership and world-space attachment** are split out into
[ADR-0008](ADR-0008-ui-integration.md), which supersedes this plan's
implicit assumption that the renderer owns a UI: `UiCore` becomes reachable
from game and editor code, `ui/demo.rs` leaves `engine-render`, and
world-anchored widgets get their position written GPU-side into a
`ui_group` offset. That reorders the work — access, then phase 3b input,
then anchoring, then docking last, so docking is written in `crates/editor`
against a public API rather than inside the renderer and moved afterwards.

## Revisit if

* Primitive counts reach the point where a full `ui_order` rebuild on
  structural frames shows up — then `ui_order` gets its own dirty mask, or
  a GPU-side compaction.
* Panel rebuilds show up in the dirty-word counters — then add keyed
  reconciliation, which is what preserves the equality gate across a
  rebuild.
* Threading `&mut UiCore` through editor helper functions proves to be real
  friction — then add ambient `hp.set_text(x)` sugar over the same cursor.
  Rules: no handle method may return a borrow; the scope is installed only
  during build / dispatch / tick and panics elsewhere; `with_ui` holds a
  re-entrancy guard so aliasing is a loud crash, not UB. Precedent exists in
  `input::global()`, but that hands out shared reads — mutable ambient
  access is a different risk class, which is why it is not the base design.
* A UI ever animates *thousands* of primitives simultaneously (a waveform,
  a visualizer, a particle-ish effect) — that is the one regime where
  per-primitive GPU-side curves would beat the host animator set. It is not
  editor UI, and it should be a separate primitive kind rather than a
  change to this one.
* Overdraw dominates — then a cached UI render target with damage regions.
* A fourth dirty-scatter consumer appears — then `ui_scatter.comp` and
  `scatter.comp` should converge into one parameterized pipeline.
