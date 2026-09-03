# Retained-mode UI core

The primitive layer under every widget: what a UI node *is* on the GPU,
and why. Widgets are in [ui-widgets.md](ui-widgets.md); building one is
[ui-widget-authoring.md](ui-widget-authoring.md); the signatures are in
[the API digest](../api/engine-render/ui/index.md).

## The primitive arrays

A [`UiCore`](../../crates/engine-render/src/ui/mod.rs) +
[`UiGpu`](../../crates/engine-render/src/ui/gpu.rs) pair implementing
[ADR-0006](../ADR-0006-retained-mode-ui.md) phases 1–2. The UI is **a flat
array of primitive slots** in four device-local SoT arrays — `ui_quad`
(geometry, 32 B), `ui_style` (appearance, 32 B), `ui_group` (clip / offset /
opacity, 32 B, ~one per panel) and `ui_order` (the `(slot, gid)` draw list,
written in **tree paint order** by the placement walk — background, then the
node's own glyphs, then its children; identity ordering was wrong because the
slot free list recycles low slots, so a recycled run could paint behind opaque
geometry allocated earlier) — each with its own dirty bitmask, its own
`scatter_prepass.comp` run (reused verbatim from the transform pipeline) and
its own `ui_scatter.comp` dispatch. The split by change frequency is the same
reasoning that makes position/rotation/scale three buffers: a hover rewrites
one `ui_style` record and nothing else. Host-side, `SlotArray::set` compares
before it marks, so **no write path can upload a value that didn't change** —
a relayout producing identical rects, or re-setting a label to the string it
already holds, costs zero bytes. Drawing is **one `vkCmdDrawIndirect`** with
`vertex_count = 4` and `TRIANGLE_STRIP`: there is no vertex or index buffer,
`ui.vert` generates each quad's corners from `gl_VertexIndex` and clips by
*shrinking* the quad (the zero-area early-out is free culling for clipped,
off-screen and freed slots at once — though **collapsing a box does not hide a
label**, since a glyph quad is sized by the font rather than by its node, so
the placement walk hides a hidden node's text run explicitly and propagates
that to descendants), and `ui.frag` branches on a per-instance kind — analytic
rounded-box SDF with per-corner radii and a border ring, `R8` glyph coverage,
or a bindless `sampler2D`. Since the draw runs *after* `signal_cs`, nothing it
reads may be host-visible, so `ui_build_args.comp` promotes the host-staged
primitive count into a device-local `VkDrawIndirectCommand` — which is what
lets the primitive count change without re-recording a command buffer. Text is
one primitive per glyph out of a **static** `R8_UNORM` atlas holding every
printable ASCII character (`ui/font.rs`, a 5×9 bitmap font authored as
reviewable ASCII art, uploaded once at construction); labels own power-of-two
runs of contiguous slots from a bucketed free list, so retyping `100` → `101`
dirties exactly one glyph. Blending is premultiplied (`ONE,
ONE_MINUS_SRC_ALPHA`) into the `_SRGB` swapchain image after the present-blit,
so the UI is never tonemapped and the hardware blends in linear space. Layout
is

## Layout

**CSS flexbox and grid via [`taffy`](https://crates.io/crates/taffy)**
([`ui/tree.rs`](../../crates/engine-render/src/ui/tree.rs)): widgets form a
node tree, taffy solves it two-pass (measure bottom-up, arrange top-down), and
a placement walk pushes the solved boxes through `SlotArray::set`. Taffy was
chosen over a constraint solver (Cassowary / Auto Layout) for a structural
reason rather than a performance one — a constraint system is one global set
of equations, so nudging any variable can ripple anywhere and "relayout this
subtree" isn't expressible, which is incompatible with an architecture built
end-to-end on local invalidation. Its own invalidation is the same model this
ADR specifies: `mark_dirty` walks up the ancestor chain and stops at the first
already-dirty node, and each node caches its `(constraint) → (size)` result so
a clean subtree with an unchanged constraint short-circuits. Text leaves
measure from the fixed-advance bitmap font, so no callback into the run store
is needed. `ui::style` re-exports taffy's vocabulary directly — it *is* the
CSS box model, and a synonym layer would only obscure it.

## Ownership

**Ownership** follows [ADR-0008](../ADR-0008-ui-integration.md): the renderer
ships the *system*, never a UI. `UiCore` lives in a global store reached as
`engine::ui::ui()` — global for the reason `asset::global` is, that a
component can reach a static but not `RenderContext` — and game and editor
code each build their own trees against it. The UI tree is deliberately
**not** the ECS scene tree: `TransformHierarchy::len()` sizes the transform
SoT, the scatter, `GPURenderers` and the cull dispatch, so a panel modelled as
entities would cost world-transform work every frame to position boxes taffy
has already positioned. `run_layout` runs once per frame inside the renderer,
after `Scene::update`, and early-outs on a clean tree.

## Input and the hit walk

**Input** is polled rather than callback-driven: `set_events` opts a node into
hit testing, `hovered` / `held` / `clicked` report state that the renderer
folds once per frame *before* `Scene::update`, and `pointer_captured` is what
stops a click on a button from also orbiting the camera. A component already
runs every frame and holds its own state, so `if ui.clicked(btn) { self.n += 1
}` needs no closure and no app-state type threaded through the widget layer.
Hit testing is a reverse-DFS over the node tree on **events**, not per frame —
where the event is not only the pointer's, since a button animated under a
still cursor or a panel toggled open beneath it changes what is hovered
without the mouse moving at all (`test-game` shipped exactly that bug: F4
hides its overlay, and toggling it under the cursor left `pointer_captured`
stale so a click orbited the camera through the panel). A `layout_epoch`
bumped at the only two places content can move — a placement walk that
actually ran, and a `scroll_by` past its own no-op guard, since scrolling
shifts rows under the pointer with no relayout at all — is what the early-out
tests alongside the pointer. It is exact rather than heuristic on purpose: a
test asserts a settled UI performs **zero** walks over 100 frames and that the
re-clamping `scroll_by(area, [0,0])` `RowList::sync` issues every frame is not
an event, which is why `Pointer` keeps a `walks` counter — it makes "genuinely
event-driven" assertable rather than claimed. The walk is and it is **per
event kind**, because one `interactive` flag answers one question while there
is more than one: a checkbox inside a tree row should take the *click* and the
row should take the *drop*, which innermost-wins alone cannot express. Nodes
declare an `Events` mask (`CLICK | HOVER | DROP | SCROLL`), and **one walk
resolves every target**: each kind is claimed on the way out, and the DFS
unwinds innermost-first, so the first claimant of a kind is the innermost node
accepting it. That deleted a second near-identical walk (`scroll_hit`, which
existed only because the wheel wants a different target than the click —
`SCROLL` is now set by `scroll_area` itself, beside the content group it
means) and a hard-coded special case in `RowList` that named the disclosure
arrow to recover its row.

## Hover

**Hover is a set, not a winner**: click and drop have one correct target, but
a row tints *and* the control inside it tints, and the pointer is genuinely
over both — so `hover` is the ancestor chain of accepting nodes and the
restyle pass diffs last frame's chain against this one. Occlusion is per
subtree rather than per kind: once anything under a sibling claims something
the scan stops, or a drop released on a panel would fall through to a zone
painted beneath it; a node accepting *nothing* claims nothing and so blocks
nothing, which keeps decoration transparent to the pointer. The walk is still
deliberately unpruned by the parent's box, since a child can be drawn outside
its parent and pruning would invent a containment rule the renderer does not
honour.

## Widgets

**Widgets are constructors returning typed handles**: `ui.button(parent,
"click me", ButtonStyle::default())` composes a node, a `StateStyle`, a label
and `set_interactive`, and hands back a `Copy` `Button` — and every other node
API still works on it, because each handle converts back into a `NodeId` and
those calls take `impl Into<NodeId>`. There is still no `Widget` trait and no
per-frame widget loop, which would reintroduce the O(widgets) cost the design
exists to avoid; a newtype of two `u32`s reintroduces neither. The handles are
typed for a reason beyond labelling: **a handle says what the node is**, which
is what makes `Checkbox::checked` and `Slider::value` expressible at all — a
`NodeId` has no value to read, a `Checkbox` does, and without the type the
caller is pushed back into keeping its own copy. It also retires a runtime
panic: `Label` makes "this node has no glyph run" unrepresentable, and
`UiCore::set_label` dropped to `pub(crate)`. Operations split on a clean line
— anything true of *any* node (pointer state, layout, re-parenting) stays on
`UiCore` and accepts every handle, while anything needing the widget's own
structure is a method on the handle, matching `RowList` and `TreeView`, which
were already caller-owned types. So `UiCore` needs no new method per widget: a
**game** can define its own widget type against the public node API without
patching the engine. What typing does *not* fix is staleness — the handles are
`Copy` indices, so a removed widget's handle would name whatever moved into
its slot. **`NodeId` therefore carries a generation** beside its index:
[`remove_node`](../../crates/engine-render/src/ui/tree.rs) walks the subtree,
returns each node's background rect and glyph run to the slot allocator,
pushes the node slots onto a free list that `node()` pops from, and bumps each
freed slot's generation — so every handle minted before the removal mismatches
and `live()`, the single point every public accessor resolves through, panics
naming the slot and both generations. It panics rather than returning `None`
because a stale handle is a caller bug, and an `Option` would let it degrade
into a widget that silently stops responding, the failure mode hardest to
trace back. Generations and recycling landed together deliberately: without
reuse the tags would guard nothing, and `removal_returns_primitive_slots`
builds and tears down the same panel eight times asserting the primitive
high-water mark never moves. **Appearance comes from a
[`Theme`](../../crates/engine-render/src/ui/theme.rs) of semantic roles** —
`control_hover`, not `button_hover` — so a checkbox and a slider match a
button by construction rather than by whoever writes them remembering the same
hex. It replaced four drifting copies of one palette (two widget styles, the
editor's chrome, the demo overlay: the same hover blue written twice, two
different alphas for the same kind of surface). Widget style structs are
unchanged and gain `From<Theme>`, with `Default` as `theme().into()`, so
`..Default::default()` still overrides any single field and no call site
moved; only the three genuinely duplicated metrics (`text_px`, `radius`,
`pad`) joined the colours, while `row_h` and `indent` stayed list geometry.
The palette is process-global and chosen **at startup**: `set_theme`
deliberately does not restyle widgets already built, since re-deriving them
needs the store to record which role each colour came from, and a half-applied
swap would look like a bug while reading as correct. `accent` is what shows
the roles are semantic rather than renamed constants — the editor's chrome is
green, `test-game`'s overlay blue, neither known to any widget below.

## Controls

**Controls own their value.** `cb.checked(&ui)` reads a checkbox,
`sl.value(&ui)` reads a slider, and the click or drag that moves either is
applied by `update_pointer` — before any component runs, on the same ordering
guarantee that already made `clicked()` observable the frame it happened. This
**reverses** an earlier app-owned decision, in which the widget stored nothing
and the caller restated the value every frame. The reasoning behind that
decision was staleness — an engine-owned copy could disagree with the
application's, the failure `TreeView` avoids by reading structure through a
closure — and it holds; what it missed is which direction the mirror appears
in. Storing nothing does not remove the copy, it *relocates it into the
caller*: with no way to read a checkbox, every user keeps a parallel `bool` in
step by hand, which is the same mirror written once per call site instead of
once in the engine. Owning the value removes it without closing the control
off, because **setting is still allowed** — a value loaded from disk, moved by
a keybind or pushed by a network packet lands the same way a click does, and
`set_checked` / `set_value` return early when the value already holds. The
rule this leaves is narrower than "app-owned" and easier to apply: *a control
owns the value it is a control for; nobody owns a mirror of it* — a view over
someone else's tree, like `TreeView`, is not a control over a value. The store
keeps one sparse `Vec<Option<Control>>` indexed by `NodeId` like
`state_styles`, holding the value and the parts that redraw it; it lives there
rather than in the handle because `update_pointer` works from a `NodeId`
alone, and it is not per-frame work — only the node that was clicked and the
node being dragged are ever consulted, so a thousand controls fold as fast as
one. A checkbox toggles on `clicked` rather than `down_on`, so a press dragged
off cancels the toggle exactly as it cancels the click. The mark is a single
glyph (`✓`, an eighth atlas entry) so a toggle dirties one slot, and the
**row** is the control rather than the square, so clicking the label toggles
too without a second interactive node that could disagree about what was
clicked. A `StateStyle` binds a node's background to its pointer state, and
the engine applies it **on transition**: `update_pointer` already knows which
node hover left and entered, so restyling touches at most four nodes on frames
where the pointer crossed a boundary and nothing otherwise — a thousand
buttons cost the same as one.

## Drag

**Drag is press-origin, not per-frame delta.** `Pointer` carries `press_pos` —
where the press that set `down_on` landed — and `ui.drag(n)` returns
`Some(Drag { origin, pos })` for as long as that press is held, *including
after the pointer leaves the node*. That inheritance from `held` is what lets
a slider keep tracking when the cursor overshoots its track. Keeping the
origin also makes a drag threshold one comparison (`Drag::beyond(px)`) instead
of an accumulator the caller maintains. The other half is **`ui.dropped(n)`**,
which reports the drag that started on a node and ended this frame *wherever
the pointer had got to*; `clicked` cannot stand in for it, since a click fires
only when the release lands back on the node it started from — precisely the
release a drop is not. Until it existed, nothing outside `update_pointer`
could observe a gesture ending anywhere but where it began (the slider only
worked because the fold runs *inside* `update_pointer`), so drag-and-drop was
unreachable from application code. A release that never travelled sets both,
which is right: a click is a drop of zero length, and `beyond` is what
separates them. **`Slider`** maps the pointer *absolutely* onto its track
rather than accumulating deltas, so pressing anywhere jumps there and the
thumb cannot drift from the cursor over a long drag; only the track is
interactive, since a thumb swallowing its own hits would need the press
forwarded back under the innermost-hit rule. Its `f32` is owned by the control
on the same contract as the checkbox, and the drag is applied from the press
captured *before* the release branch clears it, so a frame that both moves and
releases still commits the position it was released at rather than dropping
the last movement. `UiCore::drag` stays public and un-narrowed — the slider
consumes it internally, but the gesture belongs to any node, which is the
point of having built it as a pointer-layer primitive rather than a slider
feature. Values are normalised `0.0..=1.0`; a caller with a real range scales
at the two call sites rather than making the widget carry a range to validate.

## Scroll areas and clip groups

**Scroll areas** are where `ui_group` earns its offset: quads stay in pure
layout space and the group's offset supplies the scroll, so `scroll_by` writes
**one 32-byte group record and zero quads** however many rows are inside, with
no relayout at all. Clipping is the group's clip rect, and content scrolled
out of view shrinks to zero area in `ui.vert` and is culled by the early-out
that already existed for off-screen slots — neither shader changed. Hit
testing carries the same `(offset, clip)` pair as the placement walk, so a row
scrolled out of view is unhittable for the same reason it is invisible rather
than by a second rule that could drift. Nested scroll areas panic rather than
mis-render; scrollbars and horizontal scrolling are not built.
**`RowList<H>`** ([`ui/list.rs`](../../crates/engine-render/src/ui/list.rs))
is the virtualized list the scene hierarchy panel is built from, and it is
deliberately *flat*: the caller flattens its tree to `(depth, text)` and
indentation is left padding, so rows are siblings and the
innermost-interactive hit rule already reports the row clicked and never a
parent containing it — no bubbling rule to get wrong. Only enough rows to
cover the viewport exist as nodes; one in-flow sizer child gives the scroll
area the full `len × row_h` content height so the scroll range covers rows
with no nodes, and the pooled rows are absolutely positioned within it. The
pool is a **ring** — slot `k` holds the data index congruent to `k` mod the
pool size — so scrolling one row past a boundary rebinds *exactly one* row,
which jumps from one end of the window to the other while every other row
keeps its position and its text. Scrolling *within* a row remains one group
record; neither cost depends on the row count. `sync` re-binds every pooled
row every call and needs no invalidation protocol: the pool is viewport-sized
and the equality gate absorbs the repeats, so a still list uploads nothing. It
was the first widget the *caller* owns, because which pooled node shows which
data index is state the tree cannot represent — a shape typed handles later
made the rule rather than the exception.

## Rows and virtualization

**A row is whatever the caller builds.** `sync` takes two closures rather than
one — `build` runs once per pooled row *ever* and returns the caller's own
handle `H`, `bind` writes a data index into it once per sync — which is the
recycler pattern, and it is cheaper in a retained tree than in immediate mode
rather than merely possible: **the dynamism is paid at pool size**, so seven
rows are built once and rebound forever and per-frame cost is identical
whatever a row contains, where an immediate-mode list rebuilds every visible
row's widget tree every frame (a test asserts `build` runs 6 times for 10 000
rows). Ownership divides on one line: `RowList` owns the outer node — its
absolute position at `i × row_h`, the only thing tying a pooled node to an
index and so the only thing the ring depends on, plus its `Events` mask so
`clicked` / `hovered` / `dropped_on` all name one node whatever is inside —
and everything within belongs to the caller, **including selection**, which
stops being a field the list has to know about and becomes a `StateStyle` the
caller sets. Style split accordingly: `ListStyle` is the pitch and the drop
mark, while `RowStyle` (indent, text colours, the arrow) moved to `TreeView`,
whose rows it describes. `TreeView` now *contributes a layer* rather than
configuring one — its `build` puts a `content` node in the row and hangs the
arrow and label off it, so indentation lives there and the two never fight
over one `Style`; the extra node costs **zero primitives**, since a node with
no background and no text owns no slots. This is where the event masks pay
off: a checkbox the caller puts in a row takes its own click while the row
still reports hover and takes the drop, with no list-side knowledge of what is
in it — under one `interactive` flag that row was unbuildable. **`TreeView`**
([`ui/tree_view.rs`](../../crates/engine-render/src/ui/tree_view.rs)) layers
collapsing on top, and it **owns no tree**: node identity is an opaque `u64`
it never interprets and structure is read through a closure, so there is one
source of truth and "stale mirror" is unrepresentable. What it does own is
expansion state and the flattened visible list. Because the flatten is a DFS
preorder, a node's visible subtree is a *contiguous run immediately after it*,
so every structural edit is a splice rather than a re-walk: collapse drains
the run, expand flattens `O(K)` and splices it in, a re-parent drains +
splices + shifts depths uniformly, and a sibling reorder is a `rotate` bounded
by the drag distance. `moved` takes no structure closure at all — the run's
shape and expansion are unchanged, only its depth shifts — which is the
sharpest statement of why it is cheap. Renames are not structural: text is
pulled per visible row. Default-collapsed keeps the flatten `O(visible)`
rather than `O(scene)`. The disclosure triangle is a **child node of the
row**, so the innermost-interactive hit rule separates "toggle" from "select"
with no bubbling rule and no special case — two glyphs (`▸` `▾`) were added to
the bitmap font's seventh atlas row for it. Selection stays with the caller
and is keyed on node id, never a row index, since collapsing above a row
changes its index but not its identity. **`TreeView<H, P>` takes the caller's
row *and* the caller's payload.** `sync`'s `build` gets two nodes — `content`,
where widgets go after the arrow and inside the indent, and `row`, which a
selection fill belongs on — while `bind` takes the **node id**, never a row
index, the same argument that made `clicked` return a `u64`. The old `Row`
struct is gone entirely: `text` is the caller's to write, `selected` became a
`StateStyle` it sets, and `depth`/`expanded` were always the view's own
answers being routed through the caller for no reason.

## Picking a row up

**The view reports the pick-up; the caller grabs** — `picked_up()` offers the
node a press has dragged far enough, and the caller builds its own payload
(`ui.grab(EntityRef(id))`) and dresses the ghost, because only it knows a row
here means an entity. That report is *stateless*: it stops offering because a
grab exists rather than because a flag was set, so a caller that declines
keeps being offered the row. Reading a drag back needs exactly one thing, so
that is all the trait asks — `trait TreeDrag { fn node(&self) -> u64 }` — and
`TreeView<H, P = DragNode>` resolves `dragging::<P>()` / `dropped_on::<P>()`
through it, enough to refuse a subtree dropped into itself and to turn a
landing into a `Dropped`, nothing more. The view never *constructs* a payload,
which is why there is no inverse: a `MaterialRef` released on the tree is
declined without being inspected, while the editor's own `EntityRef` resolves
and moves the entity. `LabelRow` ships as the one instance of `H` a hierarchy
panel wants, and is what the view's own tests use, so they exercise the
caller's path rather than a private one.

## Re-parenting by drag

**Drag-to-reparent** is what finally calls `moved`, and it needed exactly one
new primitive — `dropped`, above — rather than anything new about dragging.
The middle half of a row means "into it" and the outer quarters mean "beside
it": `RowList` reports geometry (`hovered_at` gives a data index and a
`0.0..=1.0` fraction) and knows nothing about what a drop means, while
`TreeView` picks the thresholds because only it knows the row is a tree node
that can take children. Both structural questions are answered by the flat
list alone, with no trip through the caller's closure — a sibling drop's
parent is the nearest earlier row shallower than the target (the list is a
preorder), and refusing to drop a subtree into itself is a range check against
the run immediately after the dragged row. `Dropped { node, parent, at }`
*reports* a move rather than performing one, so a caller whose own rules
forbid it simply ignores it; `at` counts the destination's children **after
the dragged node has left its old parent**, matching what `moved` already did,
so a caller that removes before it inserts needs no off-by-one of its own. Two
things the gesture forced that were not obvious: the dragged node is captured
at **press** and not read back at release, because the pooled row a press
landed on is recycled by scrolling and would then name whichever data index
moved into it — the same identity argument that made `clicked` return a `u64`,
arriving a second time; and an overlay has to be re-established above what it
floats over, which is where **`UiCore::raise`** comes from. Since `ui_order`
is written by the placement walk, paint order *is* child order, so "on top"
means "last child" and one operation says it: `raise` moves a node to the end
of its parent's children, taffy's child list included so layout and paint
cannot disagree about which node is last. Every pool growth appends rows
behind the drop marker, and a hovered row's fill is opaque, so the marker
climbs back each time. It is not a z-index — there is no second ordering
concept to keep in sync, only the tree.

## Drop payloads

**Drag-and-drop belongs to the pointer, not to a widget.** A ghost was built
inside `RowList` first, and that was a symptom rather than a design: there was
no drag *session* anywhere for it to belong to, so it landed in the nearest
type that knew a drag was happening — while the node itself always hung off
the **root**, never off the list. There is one system pointer, so there is at
most one thing in flight, which makes the session `Pointer` state beside
`down_on` and `clicked`: **`ui.grab(payload)`** puts something in flight and
returns an empty **ghost** node at the pointer, **`ui.dragging::<T>()`**
reports what is being carried, and **`ui.dropped_on::<T>(n)`** is the target
half that was missing entirely — `dropped(n)` says "the gesture I started is
over", while `dropped_on(n)` says "something landed on me", heard only by the
node under the pointer at release, so a drag crosses from one panel to another
with neither knowing the other exists. The split is ownership: the pointer
layer owns what every drag would otherwise re-derive and get wrong — the ghost
hangs off the root so no scroll area's clip can cut it off, it tracks the
pointer every frame, and it is *freed* on release — while the caller owns
appearance (fill the returned node with a label, a thumbnail, a swatch) and,
in the payload, meaning. The payload is `Any + Send` and `UiCore` never looks
inside it, exactly as `TreeView` never interprets the `u64` it identifies
nodes by; a target asks for the type it accepts and gets `None` otherwise,
which is the question being answered rather than a failure swallowed — the
drop simply does not happen. Two consequences fell out: allocating per gesture
**deletes the z-order problem**, since a ghost minted at the grab is already
the last child of the root and paints over everything built before it (no
`raise`, nothing to remember), and it made idle free — two lists used to hold
two permanent ghost nodes and two 8-slot glyph runs for a gesture at most one
could be making, which is the 486 → 477 primitives the overlay gave back.
`TreeView` lost its `drag` field to the session, and its `aim` now treats "the
dragged node is not in my flat list" as *nothing to protect* rather than a
refusal — so a `DragNode(id)` constructed by any panel drops into the
hierarchy on the same path a row dragged within it does. The editor's
`HierarchyPanel` is an ordinary component whose access path to the scene graph
is `Transform::hierarchy()`; against a 40 271-entity GLB it opens to one
collapsed root row, and a drag there re-parents the **live hierarchy**. That
is what `set_parent` could not express — it appends, so the position a drop
names had nowhere to go — and it is why
[ADR-0009](../ADR-0009-hierarchy-root-entity.md) grew **`set_parent_at(t,
parent, at)`** and **`move_child(t, at)`**: two functions rather than one
extra argument, because a reorder *within* a parent changes no parent link,
and the GPU composition walk runs child → parent and never reads sibling
order. Routing a reorder through the re-parent path would push a parent change
that did not happen down the parent stream every time a user nudged a sibling;
the panel picks between them on one comparison against the node's current
parent.

## Invalidation

**Invalidation is driven, not detected**: there is deliberately no structural
version counter on `TransformHierarchy`, whose job is TRS and parent links
rather than reporting what changed — the editor owns entity management, so it
calls `set_parent_at` / `move_child` *and* `moved`, and the view is patched on
purpose instead of as a byproduct. The one structural change the editor does
not drive is a subscene materialising when its template resolves, so that is
published as an event instead: `scene_asset::drain_instantiated()` returns the
new instance roots and the panel invalidates on a non-empty drain — from the
module that owns instance lifecycle, rather than a poll against the module
that owns transforms.

## Cost

**Measured:** `test-game`'s overlay (F4, `ENGINE_UI_TRACE=1`) holds 477
primitives at ~11 700 FPS — including a **collapsible 73-node hierarchy**,
default-collapsed to nine visible rows over seven pooled row nodes, which a
drag re-parents. Frame 0 uploads **46 dirty words**; the whole rest of the
session uploads **1–2**, ten times a second — the glyphs of the readout that
actually differ — and roughly 840 frames in a row between those uploads upload
nothing at all. A readout edit that changed the panel's widest line would
upload more, since the panel background, the grid strip and its five swatches
are all genuinely resized by it. The button's fill is re-set from its pointer
state *every frame* and costs nothing except on the two frames where the state
actually changes — the equality gate absorbing a per-frame write is the "a
hover is 32 bytes and one workgroup" claim, exercised rather than asserted.
The editor's static panel is the limiting case: **4 words at frame 0 and never
again** for the rest of the session.
