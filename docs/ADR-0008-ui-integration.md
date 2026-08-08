# ADR-0008: UI Integration — Access, Ownership, and World-Anchored Widgets

**Status:** Steps 1–2 (access, input) landed; steps 3–4 proposed
**Date:** 2026
**Scope:** `crates/engine-render/src/ui/{mod,anchor}.rs`, `shaders/ui_anchor.comp`, `crates/engine/src/lib.rs` (`engine::ui`), `crates/test-game/`, `crates/editor/`
**Related:** [ADR-0006](ADR-0006-retained-mode-ui.md) (the UI system itself), [ADR-0007](ADR-0007-global-transform-pass.md) (supplies world TRS by slot)

## Context

ADR-0006 built the UI system. It did not answer **who builds a UI**, and the
code reflects that: `ui/demo.rs` lives in `engine-render` and
`RenderContext` owns a `ui_demo` field, so the demo overlay ships inside the
renderer and appears in the editor and in every game alike.

That is not merely misplaced — it is the only place the demo *can* live
today. `UiCore` is a private field of `RenderContext`, and the component
hook games extend the engine through is:

```rust
fn update(&mut self, _dt: f32, _transform: &Transform) {}
```

No context, no renderer, no UI. There is currently **no path from game code
to the UI at all**.

Two questions follow, and they are separable:

1. **How does game/editor code reach the UI?**
2. **How does a widget attach to a world entity** — a health bar above an
   enemy, a name tag, an interaction prompt — and stay attached as that
   entity moves?

Question 2 is the one with an architecturally interesting answer, and the
one that decides whether question 1's answer is expensive.

## Decision

### Access: a global, matching existing precedent

`UiCore` moves out of `RenderContext` behind a global mutex, reached as
`engine::ui::ui()`. The renderer locks it once during the harvest.

The codebase has already answered this twice — `asset::global()`,
`material::global()` — and `components.rs` states the reason directly:

> Global (like `engine_core::asset::global`) because `Component::init` can
> reach a static but not the renderer's `RenderContext`.

The alternative, widening `Component` to carry a context, touches
`ComponentStorage::par_iter`'s fan-out and every component in the workspace,
to solve a problem two existing subsystems already solved the other way.

**On `Send`.** Putting `UiCore` in a static requires it to be `Send`, and it
is not: taffy packs every length into a `CompactLength` whose payload is a
tagged `*const ()`, which makes `Style` — and therefore `TaffyTree` —
`!Send`. That one raw pointer is the only obstruction; everything else in
the store is plain data.

The pointer variant is `calc()`, and its sole constructor
(`CompactLength::calc`) and sole readers (`calc_value`, `is_calc`) are all
`#[cfg(feature = "calc")]`. The workspace builds taffy with
`default-features = false` and does not enable `calc`, so in this build the
tag can only ever hold an `f32` — never an address. `ui/tree.rs` therefore
carries a narrow `unsafe impl Send for Tree`, scoped to the one type that
needs it, with a compile-time assertion that fails the build if
`taffy/calc` is ever switched on. Enabling that feature is the single thing
that would invalidate the reasoning, so it is checked rather than trusted.

**On contention.** `Scene::update` runs components in parallel across the
pool, so a single mutex looks like a hazard. It is not, and the reason is
structural rather than optimistic: **the design below removes per-frame UI
writes entirely.** A health bar's *position* is resolved on the GPU; its
*value* changes on a damage event. Frames on which any component touches the
UI are rare, so the lock is uncontended by construction — the same property
that makes the whole system cost zero dirty words at rest.

### UI nodes are not ECS nodes

Godot models UI as scene nodes (`Control` extends `Node`). Rejected here,
for a concrete cost reason rather than a taste one.

`TransformHierarchy::len()` sizes the world-transform SoT, its dirty
bitmask, the scatter dispatch, `GPURenderers`, the cull dispatch, and — per
ADR-0007 — the global composition pass. **Every entity costs GPU work every
frame whether or not it draws anything.** Putting a 238-primitive panel in
the scene tree means 238 world transforms getting scattered, composed and
frustum-culled forever, to position boxes that a 2D layout engine has
already positioned.

It would also mean two trees to keep synchronized, since taffy already owns
the UI tree with the dirty-propagation semantics ADR-0006 chose it for.

So the trees stay separate. What Godot gets right — *UI is authored as a
tree of styled nodes* — is already what `ui::node` / `ui::label` provide.
**The link between a widget and an entity is a component, not shared node
identity.**

### World-anchored widgets: the group offset *is* the anchor

`ui_group` already carries an offset, and `ui.vert` already applies it to
every quad in the group:

```glsl
vec2 p0 = q.rect.xy + g.xf.xy;
```

So a world-anchored widget needs no new primitive kind and no new vertex
path. It needs **its own `ui_group`**, and a compute pass that writes that
group's offset from the entity's projected world position:

```
ui_anchor.comp:
    world  = sot_global_pos[transform_id] + world_offset      // ADR-0007
    clip   = view_proj * vec4(world, 1.0)
    ui_group[gid].offset  = ndc_to_px(clip.xyz / clip.w) + pivot_px
    ui_group[gid].opacity = distance_fade(clip.w)
    ui_group[gid].clip    = behind_camera ? degenerate : screen
```

Everything it reads already exists in the frame: `sot_global_pos` from
ADR-0007, and `sot_view_proj`, promoted into a device buffer by the
`copy_buffer` baked into every `FrameSlot` primary.

What this buys, and why it is the whole point:

* **The enemy moves → zero host writes, zero dirty words, zero relayout.**
  The widget's *local* layout is unchanged; only its group offset moves, on
  the GPU, from data the frame already computed for other reasons.
* **The entire widget subtree rides along on one record** — bar background,
  fill, border, text. That is precisely the multiplier ADR-0006 built groups
  for, landing in a case the editor-chrome examples never reached.
* **Off-screen and behind-camera cost nothing new.** Writing a degenerate
  clip drives `ui.vert`'s existing zero-area early-out. No new branch, no
  host-side visibility test.
* **Distance fade is `g.xf.z`** — one line, same pass.
* **The host writes only when the value changes**, which is a damage event,
  not a frame.

#### The anchor table is a fifth scattered array

Anchors are slot-indexed records maintained by the equality gate like
everything else, riding the generic `ui_scatter` path:

```
ui_anchor[i]  — 32 B
    uint  transform_id;   // index into the global TRS buffers
    uint  group_id;       // the ui_group this pass writes
    vec3  world_offset;   // e.g. +2.2 world units up from the entity origin
    vec2  pivot_px;       // −pivot × laid-out size, written by the host at layout time
```

`pivot_px` is folded on the host rather than passed as a normalized pivot,
so the pass never needs the widget's size. It changes only when the widget
relayouts, so the equality gate suppresses it like any other field.

`ui_build_args.comp` gains a fifth output: the anchor count → `(ceil(n/64),
1, 1)`, promoted device-side exactly like the draw's instance count already
is. No new mechanism.

#### The component is data-only

```rust
root.add_component(enemy, UiAnchor::new(bar_root)
    .offset(Vec3::Y * 2.2)
    .pivot([0.5, 1.0]));      // widget's bottom-centre sits at the anchor
```

`HAS_UPDATE = false`; `init` pushes its record, exactly as `MeshRenderer`
does. The symmetry is deliberate and worth stating: **`MeshRenderer` puts a
mesh in the world, `UiAnchor` puts a widget over an entity, and both are
pure data resolved GPU-side with no per-frame component hook.**

Game logic then owns only the value:

```rust
impl Component for Health {
    fn update(&mut self, _dt: f32, _t: &Transform) {
        if self.dirty {
            engine::ui::ui().set_node_style(self.fill, /* width: percent(hp / max) */);
            self.dirty = false;
        }
    }
}
```

#### Ordering in the command buffer

`ui_anchor.comp` runs in **its own secondary, immediately before the UI
draw**, not alongside `ui_scatter`.

The reason is a dependency the current placement cannot satisfy: the UI
scatter secondary executes early, inside the scatter block, while the anchor
pass needs both the global composition pass (ADR-0007) and the `view_proj`
promotion, which come later. Placing it at the end also puts it after
`ui_scatter`, which matters:

> **The scatter rewrites the full 8-word group record whenever *any* field
> of it dirties**, which would stomp a GPU-written offset. Anchor running
> later in the same CB re-establishes it before the draw, so there is no
> visible glitch — but the ordering is load-bearing and belongs in a
> comment next to both.

One new barrier: `COMPUTE_SHADER_WRITE → VERTEX_SHADER_READ` on `ui_group`.

### Widgets are constructors, not types

A widget here is **a function that composes primitives already in the store
and returns a plain `NodeId`**. `button` builds a node, attaches a
state-driven background, adds a centred label and opts it into hit testing;
everything that then works on it — `clicked`, `set_node_style`, `node_rect`,
re-parenting — is the same API that works on any node.

No `Widget` trait, no base class, no per-frame widget loop. The alternative
— a trait object with an `update` — would reintroduce O(widgets) per frame
in an architecture whose entire premise is that idle costs nothing.

**Appearance belongs to the widget, not the update loop.** Hover/press
styling started life in app code:

```rust
let fill = if ui.held(b) { HELD } else if ui.hovered(b) { HOVER } else { IDLE };
ui.set_background(b, UiStyle::fill(fill).radius(4.0));
```

That is correct and, thanks to the equality gate, free — but it puts
appearance in the update loop, where every widget adds a branch a caller can
forget. `StateStyle` moves it into the store: attach three looks once, and
the engine picks.

The application is **transition-driven**. `update_pointer` already computes
which node hover left, which it entered, and the same for press — so
restyling touches **at most four nodes**, only on frames where the pointer
actually crossed a boundary, and nothing at all otherwise. A thousand
buttons cost the same as one. The four need no dedup, because
`set_background` ends at the equality gate.

Note the styling rule deliberately differs from the `held` query: `held`
stays true when the pointer is dragged off (so a button can be asked whether
it is armed), but the *look* reverts to idle, because releasing off the node
cancels the click and it should not look otherwise.

### The path to a widget library

Sequenced by what each widget **forces into the engine**, which is the only
ordering that matters — the drawing is never the hard part:

| Step | Widget | What it forces |
|---|---|---|
| ✅ | `button` | `StateStyle`; appearance as a function of interaction state |
| ✅ | `scroll_area` | **per-node clipping and offset** — its own `ui_group` |
| ✅ | `RowList` | **virtualization** — node count follows the viewport, not the data; the first widget the caller owns |
| ✅ | `TreeView` | **splice-based sub-edits** — collapse/expand/move patch a preorder run instead of re-walking; opaque `u64` identity |
| ✅ | *theme* | **semantic colour roles**; widget styles derive rather than hardcode |
| ✅ | `checkbox` | the engine-owned vs. app-owned question — decided **app-owned** |
| 6 | `slider` | **drag**: pointer delta and press-origin while captured |
| 7 | `text_field` | **keyboard focus + character events** — a genuinely new input axis (winit text/IME), caret, selection |
| 8 | docking | built on the scroll area's group machinery |

Steps 5–7 grow `UiCore` by a bounded widget-state table; only 7 adds new
engine capability. Docking is last because it needs the group work scroll
areas introduced, and because building it before step 1 of this ADR would
have meant writing editor chrome inside the renderer and moving it
afterwards.

### Scroll areas: the group indirection paying off

ADR-0006 predicted that scroll areas "change only this walk, not the
shaders." That held exactly — `ui.vert` and `ui.frag` are untouched.

The model:

* **Quads stay in pure layout space.** The placement walk never applies
  scroll; `ui.vert` already adds `g.xf.xy` to every quad in the group. So
  `scroll_by` writes **one `ui_group` record and zero quads**, however many
  rows are inside. A 10 000-row list scrolls for the same 32 bytes as an
  empty one — and it needs no relayout at all, so taffy is not even entered.
* **Clipping is the group's `clip` rect**, intersected with the enclosing
  one. Content scrolled out shrinks to zero area in the VS and is culled by
  the early-out that already existed for off-screen and freed slots.
* **A scroll area's own primitives stay in the *parent's* group**, while its
  children inherit a new `content_group`. That one split is what keeps the
  viewport frame still while its contents move, without a second node.
* **`hit_test` mirrors the placement walk**, carrying the same `(offset,
  clip)` pair. A row scrolled out of view is unhittable for the *same*
  reason it is invisible, rather than by a second rule that could drift.
* **The wheel routes to the innermost scroll area under the cursor**,
  independent of interactivity, and `pointer_captured` counts scroll areas —
  so the wheel scrolls a list or zooms the camera, never both.

`update_pointer` also became genuinely event-driven here: it early-outs when
the pointer neither moved nor did anything, so the tree walks it performs are
skipped on the overwhelming majority of frames. Previously it walked every
frame, which the phase-3b notes claimed it did not.

**Not built:** scrollbars (visual only), horizontal scrolling, and **nested
scroll areas, which panic**. The inner group's offset would have to track the
outer's, which the one-record scroll path deliberately does not walk; a loud
failure beats a subtly misplaced panel. Nesting becomes worth building when
a docked panel needs a scrollable sub-region.

**Measured.** Rows clip per-pixel at the boundary — a partially visible row
is cut mid-glyph — and fully hidden rows cost nothing.

### The flat virtualized row list

The scene hierarchy panel is a **flat list of rows**, not a tree of nodes:
the caller flattens its hierarchy to `(depth, text)` and indentation is left
padding. That is not a drawing shortcut, it is what satisfies the clicking
constraint — rows are *siblings*, so `hit_test`'s innermost-interactive rule
already reports the row the pointer is on and can never report a parent that
happens to contain it. No bubbling rule, no `clicked_within`, nothing to get
wrong.

Only enough rows to cover the viewport exist as nodes. One in-flow sizer
child gives the scroll area its full `len * row_h` content height — so the
scroll range covers rows that have no nodes at all — and the pooled rows are
absolutely positioned within it.

**The pool is a ring.** Slot `k` always holds the data index congruent to
`k` modulo the pool size, so scrolling one row past a boundary rebinds
*exactly one* row: it jumps from one end of the window to the other while
every other row keeps both its position and its text. The obvious
alternative — shift every row down and rebind all of them — turns each
boundary crossing into a full pool rewrite. There is a test asserting
exactly one row moves.

So the two costs stay separated, and neither depends on `len`: scrolling
*within* a row is one `ui_group` record and nothing else; crossing a row
boundary adds one row's worth of writes.

**No invalidation protocol.** `RowList::sync` re-binds every pooled row from
the caller's data on every call. The pool is viewport-sized and every write
lands on `SlotArray::set`, so a still list costs a few dozen comparisons and
zero bytes — cheaper than any dirty-flag scheme the caller would otherwise
have to maintain, and impossible to get out of sync.

`RowList` is the first widget the **caller owns** rather than a constructor
returning a `NodeId`, because it has state the tree cannot represent: which
pooled node currently shows which data index. Parking that in `UiCore` would
mean the store growing a per-widget table for one widget.

**Measured.** The demo is a 5 000-row hierarchy in a 108 px viewport: seven
row nodes, three indent levels, **407 primitives** — where the fixed 16-row
scroll area it replaced cost 392. Row height is load-bearing (it is what
converts a scroll offset into a data index), so rows are fixed-height by
construction; variable heights would need a prefix-sum index.

### `TreeView`: a collapsible tree that owns no tree

The scene hierarchy panel is a `TreeView` over a `RowList`. Node identity is
an **opaque `u64`** the view never interprets, and structure is read through a
closure, so it works over the transform hierarchy, a folder listing, or
anything else.

**It consumes a tree; it does not own one.** A view holding a mirror of the
hierarchy would acquire a sync obligation and a drift-bug class for no gain —
`TransformHierarchy` already *is* a single-rooted tree with children lists
(ADR-0009), and the flatten reading it live makes "stale mirror"
unrepresentable. What the view does own is **expansion state** and the
**flattened visible list**, because nothing else knows them.

#### Sub-edits, not rebuilds

The flat list is a DFS preorder, so a node's visible subtree is a **contiguous
run immediately after it**. Every structural edit is a splice of that run:

| Edit | Cost |
|---|---|
| collapse | scan forward for the run, `drain` it |
| expand | flatten that subtree only (`O(K)`), `splice` it in |
| reorder within a parent | `rotate`, bounded by the drag distance |
| re-parent | `drain` + `splice` + a uniform depth delta over `K` |
| rename | **not structural at all** — text is pulled per visible row |

None of these walks the whole tree. `moved` takes *no structure closure*,
which is the sharpest statement of why it is cheap: the run's internal shape
and expansion are unchanged, so nothing has to be read back — only its depth
shifts, uniformly.

**Default-collapsed** is what keeps the flatten `O(visible)` rather than
`O(scene)`: opening a 40 000-entity scene shows one row.

`invalidate()` is the one path that re-walks, for structure that changed
*outside* the view.

**Who calls it, and why not a version counter.** The tempting automation is a
structural version on `TransformHierarchy` that the view polls. That is
rejected: the hierarchy's job is TRS and parent links, and it should not also
be in the business of telling observers what changed. Entity management belongs
to the editor, so the editor calls `set_parent` *and* `moved` — the view is
patched deliberately, never as a byproduct of a hierarchy mutation. The cost is
that the two can diverge if a caller does one without the other; the mitigation
is to keep both behind a single editor-side function rather than to reintroduce
tracking.

That leaves exactly one structural change the editor does not drive: a subscene
instantiating when its template resolves, inside the render loop. So it is
published as an event — `scene_asset::drain_instantiated()` hands back the new
instance roots, and the panel invalidates on a non-empty drain. An event from
the module that owns instance lifecycle, rather than a poll against the module
that owns transforms.

This generalises to the remote-debugging case, where the game runs as a
separate process (forced, not merely prudent: `scene_asset`, `ui()`, the input
accumulator and the hierarchy are all process globals that an in-process game
would share with the editor's own instance). There `Scene` — which owns entity
lifecycle — emits `Spawned` / `Despawned` / `Reparented` behind a build flag,
and because the consumer is out-of-process, emission and transport both compile
out of a shipped release build. The transform hierarchy's existing dirty
bitmasks and parent stream stay what they are: a GPU upload channel with a
single destructive drain, unsuitable as a second observer's event source and
far too wide to put on a wire.

### Theme: roles, not looks

`ButtonStyle` and `RowStyle` each carried their own hex literals, and so did
the editor's chrome and the demo overlay — four copies of one dark palette,
already drifting: the same hover blue written twice, two different panel
alphas for the same kind of surface. `Theme`
([`ui/theme.rs`](../crates/engine-render/src/ui/theme.rs)) is the fix, and it
is deliberately **semantic roles rather than widget looks**: `control_hover`,
not `button_hover`. A checkbox and a slider ask for the same role and match by
construction, instead of by whoever writes them remembering the number.

Widget style structs are unchanged. Each gains `From<Theme>`, and its `Default`
is `theme().into()` — so `..Default::default()` still overrides any individual
field per widget, and no call site had to change. Only the three metrics that
were genuinely duplicated (`text_px`, `radius`, `pad`) joined the colours;
`row_h` and `indent` stayed on `RowStyle`, being list geometry rather than
palette.

**The palette is process-global and chosen at startup.** `set_theme` does not
restyle widgets already built: their colours were resolved when their style
struct was constructed, and re-deriving them would require the store to record
which role each colour came from — the per-widget table steps 5–7 introduce. A
half-applied theme swap would look like a bug and read as correct, so it is
simply not offered. This is also why the tests assert against `From<Theme>`
rather than by calling `set_theme`: the palette is global and tests run in
parallel, so a test that mutated it would flake every other test's colours.

`accent` is the role that justifies the indirection existing at all rather than
being a rename of the old constants: the editor's chrome is green and
`test-game`'s overlay is blue, and each sets its own without any widget below
knowing which.

### Checkbox: the value is the application's

This step existed to decide who owns widget state, and the answer is **the
application**. `UiCore::checkbox` returns a plain `NodeId` and stores no bool.
The caller writes:

```rust
if ui.clicked(self.cb) { self.wireframe = !self.wireframe; }
ui.set_checked(self.cb, self.wireframe);
```

The second line is **unconditional**, and that is the whole design. An
engine-owned `ui.checked(h)` would be a second copy of the truth, and the
moment a keybind, a network packet or a script changed the real value the two
would disagree with no notification path to reconcile them — the same "stale
mirror" failure `TreeView` avoids by reading structure through a closure. Here
the widget is told what is true every frame, so a value changed anywhere shows
up on the next one by construction. The equality gate makes the repeats free,
exactly as it already does for `apply_state_style`, and
`restating_the_same_value_uploads_nothing` pins that.

**What the store *does* keep** is one `Vec<Option<NodeId>>` mapping a checkbox
row to its mark label — the bounded widget-state table this ADR predicted, but
holding *structure* rather than value. It answers "which descendant is the
mark", which the tree cannot be asked, and nothing else. `set_checked` on a
node that is not a checkbox panics rather than doing nothing quietly.

The mark is a single glyph (`✓`, an eighth atlas entry), so toggling dirties
one slot. The **row** is the control rather than the square, so clicking the
label toggles too and the innermost-interactive rule still reports one node —
making the square a second interactive node would have made the label and the
box disagree about what was clicked.

#### The disclosure arrow is free

The triangle is a **child node of the row**, marked interactive. `hit_test`
returns the innermost interactive node, so clicking the arrow toggles and
clicking anywhere else selects — no bubbling rule, no `stopPropagation`, no
hit-test special case. This is exactly the case that would have needed a
carve-out under DOM-style event bubbling, and it is why exclusive innermost
hit testing was chosen.

Two glyphs (`▸` `▾`) were added to the bitmap font in a seventh atlas row —
`>` / `v` read as text rather than as a control. The atlas had exactly one
free cell, so `EXTRA` now appends non-ASCII glyphs past the contiguous block.

**Selection is the caller's**, keyed on the node id and never a row index:
collapsing anything above a selected row changes its index but not its
identity.

#### Verified

The editor's `HierarchyPanel` is an ordinary component — its access path to
the scene graph is `Transform::hierarchy()`, since `Component::update` is
handed a `Transform` and nothing else. Against
`OopsWholePlane1.glb`'s **40 271 entities** it opens to a single collapsed
root row with a working disclosure triangle. `TransformGuard::get_children`
was tightened from `&mut Vec<u32>` to `&[u32]` at the same time: child-list
mutation must stay inside the hierarchy, or no invalidation signal it ever
grows can be trusted.

`ui::widget` lives in `engine-render` today because it is small. When it
grows it becomes its own crate depending on `engine` — never
`engine-editor-api`, or the editor boundary leaks into every game that wants
a button.

### Three consequences that are not free

1. **`run_layout` moves into the renderer's per-frame block.** It is called
   from `Demo::update` today. It early-outs on a clean tree, so calling it
   unconditionally is correct and nearly free — but it must move, or a game
   that never calls it gets a UI that never lays out.
2. **Anchored widgets are additional layout roots.** They size to their own
   content, not the window, so they cannot hang off the screen root.
   `TaffyTree` supports multiple roots; `run_layout` becomes "screen root at
   window size, plus each anchored root shrink-wrapped".
3. **Input must land with this.** A game UI that cannot be clicked is not an
   interface, so hit testing is a co-requisite, not a follow-up (landed —
   see step 2). **World-anchored widgets stay non-interactive**, because
   their positions exist only on the GPU and testing them would require a
   readback — an acceptable limit, since health bars and name tags are not
   click targets. Note this also means `hit_test`'s node-tree walk keeps
   working unchanged when anchoring lands: it reads `Tree::absolute`, which
   anchored widgets never populate in screen space.

### Where the code lands

| Location | Contents |
|---|---|
| `engine-render/src/ui/` | store, GPU pipeline, taffy tree, font, `UiAnchor`, `ui_anchor.comp` — the *system* |
| `engine::ui` | re-exports: `NodeId`, `PrimId`, `style::*`, `UiStyle`, `rgb`/`rgba`, `ui()` |
| `crates/test-game/src/ui_demo.rs` | today's `demo.rs`, as a game component — the worked example |
| `crates/editor/` | editor chrome, built against the same public API |

The editor and the game each build their own trees against one shared
system. The separation stays enforced by the dependency graph rather than by
convention: `test-game` still cannot see `engine-editor-api`.

## Consequences

### Wins

* **The renderer stops shipping a UI.** `RenderContext` loses `ui_demo`
  entirely, and no game inherits an overlay it did not ask for.
* **A moving entity's widget costs nothing on the host.** No projection, no
  write, no relayout, no lock — for any number of anchored widgets.
* **No new primitive kind, no new vertex path, no new sync primitive.**
  Anchoring reuses groups, the generic scatter, the args-promotion trick and
  the existing zero-area cull.
* **One access pattern for both consumers**, so editor panels and game HUD
  exercise the same code and the same dirty-word counters.

### Costs

* **A global mutex on `UiCore`.** Justified by measurement rather than
  assumption only once anchoring lands; before that, a game that positions
  widgets from `Component::update` will contend, and will deserve to.
* **`UiAnchor` couples the UI to ADR-0007.** Screen-space UI does not, so
  the access work can and should land first.
* **Anchored widget draw order is arbitrary among themselves.** `ui_order`
  is host-maintained in painter's order and cannot know GPU-computed depths,
  so two overlapping bars resolve in slot order rather than by distance.
  Acceptable — overlapping world-space widgets are rare and the failure is
  cosmetic — but it is a real limitation, not an oversight.
* **`engine::stats::fps()` needs exposing** for the relocated demo, which
  currently reads a value only the renderer has.

### Caveats

* The anchor pass writes fields the host also owns. The host must never
  write `offset` for an anchored group after creation; nothing enforces
  this, so it is a comment and a test rather than a type.
* Non-uniform DPI / swapchain resize changes `pivot_px`'s meaning, since it
  is baked in physical pixels at layout time. `on_resize` already triggers a
  relayout, which recomputes it — but only because the widget relayouts, so
  a widget whose layout is resize-invariant must still be re-pivoted.

## Implementation plan

**Step 1 — access. ✅ Landed.** Global `UiCore` (`ui::ui()`); `engine::ui`
and `engine::stats` re-exports; `run_layout` moved into the renderer's
per-frame block; `ui/demo.rs` moved to `crates/test-game/src/ui_demo.rs` as
a `UiDemo` component; `RenderContext` lost both `ui_core` and `ui_demo`;
the editor builds a static panel of its own from `main`.

**Measured.** The two binaries now cover both regimes and neither sees the
other's UI. `test-game`: 238 primitives at ~11 300 FPS, 1 dirty word when
the readout keeps the panel's width and 7 when it changes it, zero on every
other frame — unchanged from before the move, which is the point. The
editor's static panel is the limiting case: **4 words at frame 0 and never
again** for the rest of the session.

`stats::fps()` is smoothed (EMA) rather than instantaneous `1/dt`, which at
12 000 FPS swings by thousands between frames; `stats::dt()` stays raw.

**Step 2 — input. ✅ Landed (polled, not callback-driven).** `set_interactive`
opts a node into hit testing; `update_pointer` folds cursor + button edges
into hover / held / clicked state once per frame, before `Scene::update`;
`hovered` / `held` / `clicked` / `pointer_captured` are the query API.
`OrbitController` consults `pointer_captured` so a click on a button neither
orbits the camera nor, over a panel, zooms it.

**Two deliberate divergences from ADR-0006 phase 3b:**

* **Polling, not a callback table.** A component already runs every frame
  and already holds its own state, so `if ui.clicked(btn) { self.n += 1 }`
  needs no closure, no `Ui<S>`, and no app-state type going viral through
  the widget layer. The callback design remains right for editor-class UI,
  where there is no component to poll from — but it is not needed to make
  buttons work, and building it first would have been speculative.
* **Hit testing walks the node tree, not `ui_order`.** ADR-0006 specified a
  reverse linear scan of the draw list; that yields a *slot*, and a slot has
  no path back to a node, so the widget API would need a slot→node map it
  otherwise never wants. Reverse-DFS over the node tree returns node
  identity directly, is the same O(), and runs on pointer events rather than
  per frame.

Hit testing is deliberately **not pruned** on parent boxes: clipping today
is per `ui_group`, not per node, so pruning would invent a containment rule
the renderer does not honour and silently mis-hit absolutely-positioned
children. Pruning becomes correct when scroll areas give nodes real clip
rects.

**Measured.** `test-game` gained a button and a click counter: 263
primitives, ~11 000 FPS. The button's fill is re-set *every frame* from its
pointer state and costs nothing on all but the two frames where the state
actually changes — the equality gate absorbing a per-frame write is
ADR-0006's "a hover is 32 bytes and one workgroup" claim, exercised rather
than asserted. Verified end to end by clicking: hit test → `clicked()` →
counter relabel → upload.

**Step 3 — anchoring.** Depends on ADR-0007 stage 1. `ui_anchor` array,
`ui_anchor.comp`, `UiAnchor` component, the fifth `ui_build_args` output,
multiple layout roots.

**Step 4 — docking** (ADR-0006 phase 4). Now writable in `crates/editor`
against a public API, which is the point of doing it last.

## Revisit if

* **Anchored widgets need to be interactive** — then hit testing needs
  screen rects the host can see, which means either a readback of the
  anchored group offsets (one frame stale, probably fine for a tooltip) or
  duplicating the projection on the host for the few widgets that need it.
  Do not make it the default for all of them.
* **A game wants thousands of anchored widgets** (an RTS unit-bar layer) —
  then the per-widget group stops being the right granularity and the whole
  layer wants one group plus a per-primitive anchor, which is a different
  and larger change.
* **The `UiCore` mutex shows up in a profile** — then make `SlotArray::set`
  lock-free for value writes. Distinct indices already touch disjoint
  values; only the dirty bitmask and the min/max watermarks are shared, and
  both are atomic-OR / atomic-min-max shaped. Structural operations
  (allocate, free, relayout) stay behind the lock.
* **The editor and games want to share widget code** — then a widget library
  crate, which must depend on `engine` and not on `engine-editor-api` or the
  separation stops being enforceable.
