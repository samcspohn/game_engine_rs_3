# UI event dispatch

Handlers registered per node and run once, from the renderer, on the frames
something happened. The half of [ADR-0006](../ADR-0006-retained-mode-ui.md)
that did not land with the rest of the retained store.

## Why it had to exist

`UiCore` is genuinely retained — `update_pointer`, `run_layout` and every
`SlotArray::upload` early-out on an idle frame, and a profile confirms it.
The cost was the layer *above* it. Every panel re-bound every pooled row on
every frame, re-formatted every inspector value, and let the equality gate
throw the result away, because
[ui-widget-authoring](ui-widget-authoring.md) invariant 1 said to. **The gate
is an upload guard, not a compute guard.** It saves bytes on the bus; it does
not save the `format!`, the `Style` clone or the hierarchy read that produced
the value it compares.

So the UI's CPU cost grew with the number of elements on screen, which is the
one thing the retained design was supposed to make impossible.

## Identity, not pointers

The objection that kept callbacks out was that they need stable references
into component storage, and there are none: `ComponentStorage<T>` is a `Vec`
that moves, and a `WorldHandle` is an `Arc` that would keep a stopped play
world alive past the drop that retires it.

A handler does not need a pointer. It needs to be able to *find* the thing
again:

```rust
pub struct ComponentRef<T> { entity: EntityRef, .. }   // world, slot, generation
```

Three integers, resolved through `worlds::world` and `World::get_component`
at the moment the event fires — which is also the moment "is it still there?"
is a question worth asking. The generation is what makes the answer honest
now that slots are reused: a handler whose entity has gone, even if another
has taken its slot, returns `Keep::No` and unregisters itself. The family of
these handles is in [refs](refs.md).

## The ordering rule

```
update_pointer → update_secondary → update_keyboard → dispatch → sweep_all → run_layout
```

`dispatch` sits between the input fold and the component sweep, and that
position is load-bearing at both ends:

- **After input**, so a handler sees the same frame a polling component
  would. Nothing has to be deferred or replayed.
- **Before the sweep**, so no component's `update` is running. Dispatch takes
  the UI mutex and then a component mutex; `OrbitController` takes them the
  other way round from inside the sweep. The two orders never overlap because
  the two phases never overlap, and `dispatch` asserts it is not re-entered.

`Resized` is the one event queued from `run_layout`, so it arrives on the
*next* frame's dispatch. That is the same one-frame trail `sync_scrollbars`
already had, and for the same reason: a box's new size is not known until
after the solve that produced it.

## Where each event comes from

Every emission is a one-line push at a place that had already computed the
answer — none of them is a new walk.

| Event | Emitted by |
|---|---|
| `Click`, `DoubleClick` | `update_pointer`, where `clicked` is set |
| `RightClick` | `update_secondary` |
| `HoverEnter` / `HoverLeave` | the hover-chain diff, beside `apply_state_style` |
| `Drag`, `Dropped` | the press / release branches beside `down_on` |
| `DroppedOn` | the release that resolved a grab onto a target |
| `Scroll` | `scroll_by`, past its no-op guard |
| `Changed` | `drive_controls`, and `field_keystroke` for text |
| `Submit`, `FocusLost` | `field_keystroke` and `set_focus` |
| `Resized` | `place`, when the absolute rect it writes differs |
| `Tick` | `every`, from the due list at the top of `dispatch` |
| `Key` | `update_keyboard`, for a keystroke no field has focus for — sent to the root |
| `Notify` | `UiCore::notify`, by whoever changed something outside the UI |

While a grab is in flight, `Drag` also reaches every node under the pointer.
A drop target is not the node the press started on, so it would otherwise
hear nothing until the release — and a drop indicator has to track the
pointer *within* a row to say "into this one" rather than "between these
two".

`DroppedOn` carries no payload. The grab stays in the pointer for the frame,
so the handler reads it with `dropped_on::<T>` and a target that cannot hold
that type simply declines; a handler that took the box by value would make
the first one to look the only one allowed to.

## Registration is opting in

`on` ORs its `kinds` into the node's `Events` mask, so a click handler is
what makes the node a click target — there is no second call to forget.
`Events` gained `DRAG`, `DOUBLE_CLICK`, `RIGHT_CLICK`, `CHANGE`, `RESIZE` and
`TICK` for this. The first three are found by the `CLICK` walk, so asking for
one implies it; the last three are not pointer kinds at all and the hit walk
ignores them.

Splitting drag and double click out of `CLICK` is not tidiness. A handler
registered for "clicked" would otherwise fire four times for one
press-drag-release on the same node.

## Two grains

| | |
|---|---|
| `on` / `on_this` / `every` | one node, one reaction |
| `on_any` | once per dispatch in which anything of `kinds` happened |

`on_any` exists for a caller whose reaction is "re-read everything" rather
than "this node was clicked" — the editor chrome, which owns panels across
four widgets whose internals it does not name. Once per *dispatch* and not
once per event: a click that is also a drop is one thing happening, and a
caller that re-read twice for it would be paying for a distinction it does
not use.

That is what makes the editor conversion small. `Chrome::refresh` is the old
per-frame body verbatim, and every panel's polls — `clicked`, `hovered_at`,
`dropped_on`, `menu_choice` — still answer for the current frame, because
dispatch runs where the poll used to. What an event replaced is the *frame*,
not the poll.

**A coarse subscriber has to be idempotent within a frame.** Everything the
input fold latches — `clicked`, `dropped`, a submitted field, a menu choice —
lasts exactly one frame, so a reader that runs twice in one applies the same
click twice — and a dispatch can run more than one round, as a handler's own
emits are delivered. The rule is a stamp against `UiCore::frame()`, which is
counted ahead of `update_pointer`'s early-out — an idle frame is still a
frame, and a counter that froze on one would suppress every refresh after
it.

## Changes from outside the UI

A spawn, a destroy and an added component are applied at the **next** frame
boundary, so the click that asked for one walked a hierarchy that did not have
it yet. The edit says when it has landed instead of being polled for: a
spawn's builder and a `World::then` queued behind a destroy both run at that
boundary, and `defer` a `ui.notify(node)`. The boundary holds the document
world mutably, which is why the panel is not touched there and then.
`notify` queues a `Notify` for the next dispatch, so it reaches handlers and
`on_any` subscribers the way a click does. The same path carries the
scaffold thread's answer and the renderer's report that a subscene has
materialised (posted to the root).

Keystrokes no field has focus for are a `Key` event on the root, which is
where W/E/R and Delete are heard. So the editor's chrome has no `update` at
all, and an idle frame runs none of its code.

Three nodes are watched individually instead: the viewport and the two tree
areas subscribe to `RESIZE`, because a camera sizes itself to its box and a
row pool sizes itself from the area it was measured in, and neither can wait
for the next thing the user does. `RESIZE` is deliberately not in the
chrome's `on_any` mask — `place` would then report every node that moved on
a relayout.

## Dispatch itself

- The queue is drained a round at a time; handlers registered during a round
  run on the next event, not within their own. Sixteen rounds and a queue
  that is still filling is a storm, and it panics rather than dropping
  frames.
- A node's table is taken with `mem::take` for the call, which is what hands
  the handler `&mut UiCore` without a self-borrow. A `Some(vec![])` is left
  in its place, not `None`, so a handler that emits to its own node still
  finds a listener there.
- Generations are checked on both sides of the call: a handler may free the
  node whose event it is running, and the table must not be restored to a
  slot somebody else now owns.
- `free_subtree` clears handlers, tickers and queued events for the slot. This
  is step 6 of adding a widget in
  [ui-widget-authoring](ui-widget-authoring.md) and the reason it is stated
  there.
- `UiCore::dispatched()` counts handler calls, beside `Pointer::walks`. It
  is what makes "an idle frame runs nothing" a test rather than a claim.

## Ticks

`every(node, hz, this, f)` is for a value only the simulation can move, which
no event names — the inspector's readback while a document is playing. It is
a scanned list rather than a heap because the editor has at most one, and an
empty list costs an `is_empty`. A tick that returns `Keep::No`, or whose node
is freed, is dropped.

Anything on a timer breaks the zero-cost idle frame by construction, which is
why this is a named subscription and not a frame hook: what pays for it has
to say so.
