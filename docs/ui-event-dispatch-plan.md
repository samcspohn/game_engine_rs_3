# UI event dispatch: implementation plan

**Status: implemented.** [ui-events](notes/ui-events.md) owns the result and
is the document to read; this one is kept for the reasoning that led there
and for what the build decided differently:

- **`sync` was not deleted.** Phase 2 assumed `RowList` and `TreeView` could
  store `bind` and re-run it from their own handlers. They cannot: the list
  is owned by a panel owned by `Chrome`, so a handler that reached it would
  have to lock `Chrome` — and it is already locked, by the handler. So the
  lists kept `sync`, and what changed is who calls it and when. Since
  lifted for the hierarchy: the panel is a component of its own, reached by
  `This<HierarchyPanel>` ([editor](notes/editor.md)).
- **`on_any` instead of per-node editor handlers.** With `sync` intact, the
  editor's reaction to any event is "re-read everything", which is one
  subscription rather than thirty. `Chrome::refresh` is the old `update`
  body verbatim, so no panel's logic changed and no wake can be missed.
- **`Event::DroppedOn` carries no payload**, which settles the open question
  below: the grab stays in the pointer for the frame, so the handler reads
  it with `dropped_on::<T>` and a target that cannot hold that type simply
  declines.
- **`Events` gained `DRAG`, `DOUBLE_CLICK` and `RIGHT_CLICK`** as kinds of
  their own. Routing them all through `CLICK` fired a click handler four
  times for one press-drag-release.
- **Ticks are a scanned list, not a heap.** The editor has at most one.

**Goal:** make an idle editor frame cost the UI nothing on the CPU, by
finishing the "true retained, callback driven" half of
[ADR-0006](ADR-0006-retained-mode-ui.md) that never landed.

**Why now.** Profiled 2026-09-13 on the editor, one document, no input:
UI + taffy self time is ~20% of the main thread (~27 µs of a 135 µs frame).
`UiCore` itself early-outs correctly; the cost is the layer above it, which
re-binds every visible row, re-formats every inspector value and refits every
scrollbar every frame, then lets the equality gate throw the result away. The
gate is an upload guard, not a compute guard, and
[ui-widget-authoring](notes/ui-widget-authoring.md) invariant 1 currently
tells authors to lean on it that way.

**Shape.** Handlers are registered per node and run once, in a `dispatch` on
the main thread, before `worlds::sweep_all`. A handler that needs a component
captures `(WorldHandle, Entity)` and resolves it through
`World::get_component::<T>` when it fires. No pointers, no statics; a dead
entity or freed node unregisters the handler, on the same rule that makes a
stale `NodeId` a caller bug.

The polled API (`clicked`, `hovered`, `drag`, `dropped_on`) stays public for
game components that run every frame anyway. Nothing in `test-game` changes.

## Phase 0: two fixes that stand alone

Land these first, on their own commit. They are correct under either model and
remove the two structural costs that do not depend on the design.

1. **`scroll_by` refits only its own area's scrollbars.** Today `tree.rs:1087`
   ends in `sync_scrollbars`, which walks the global `UiCore::scrollbars`
   registry (`mod.rs:520`). `RowList::sync` calls `scroll_by(area, [0,0])`
   every frame, so D lists × B bars per frame. Key the registry by area, or
   filter on it, and keep the full walk only in `run_layout`. Test: two lists,
   scroll one, assert the other bar's refit count is zero.
2. **Cheap setters for the per-row restyles.** `bind_arrow`
   (`tree_view.rs:622`) and `show` (`editor/main.rs:759`) go through
   `node_style` clone + full taffy `Style` compare, ~300 bytes each way. Add
   `set_padding_left` and use the existing `set_visible`, both touching one
   field. After phase 3 these run only on rebind, so measure before deciding
   whether the setters are still worth keeping.

## Phase 1: the dispatcher in `engine-render`

New file `ui/events.rs`. `UiCore` gains three fields: `handlers`
(`Vec<Option<Handlers>>`, indexed by node slot like `controls`), `queue`
(`Vec<(NodeId, Event)>`), `pending` (registrations made during a round).

```rust
pub enum Event {
    Click, RightClick, DoubleClick,
    HoverEnter, HoverLeave,
    Drag(Drag), PickedUp(Drag), Dropped(Drag),
    DroppedOn(Box<dyn Any + Send>),
    Scroll(f32),
    Changed(Value),          // Bool | F32 | Index
    Submit(String), FocusLost,
    Resized(Rect),           // node's absolute box moved (placement walk)
    Tick(f32),
}
pub enum Keep { Yes, No }
pub struct This<T: Component> { world: WorldHandle, entity: Entity, .. }

impl UiCore {
    pub fn on(&mut self, n: impl Into<NodeId>, kinds: Events,
              f: impl FnMut(&mut UiCore, &Event) -> Keep + Send + 'static);
    pub fn on_this<T: Component>(&mut self, n, kinds, this: This<T>,
              f: impl FnMut(&mut T, &mut UiCore, &Event) + Send + 'static);
    pub fn every<T: Component>(&mut self, n, hz: f32, this: This<T>,
              f: impl FnMut(&mut T, &mut UiCore, f32) + Send + 'static) -> TickId;
    pub fn stop(&mut self, t: TickId);
    pub(crate) fn dispatch(&mut self, now: f64);
}
```

- `on` ORs `kinds` into the node's `Events` mask, so registering a click
  handler is also what makes the node hittable.
- `on_this` is one wrapper over `on`: look the component up, lock it, call,
  return `Keep::No` when the lookup fails.
- `dispatch` takes the table with `mem::take` for one round, drains `queue`,
  restores the table, merges `pending`, and loops while the queue refilled.
  Cap 16 rounds, panic naming the node. The take is what hands the handler
  `&mut UiCore` without a self-borrow.
- Ticks are a due-time min-heap (`BinaryHeap<Reverse<(due, TickId)>>`);
  `dispatch` pops what is due and pushes `Tick(dt)` for it.
- `free_subtree` (`tree.rs:450`) clears the node's handlers and its ticks.
  This is invariant 5 in the authoring note; add it to the list there.
- A `dispatched: u64` counter beside `Pointer::walks` (`mod.rs:560`), so
  "zero handler calls on an idle frame" is assertable, not claimed.

**Emission points**, each a one-line push guarded by "node has handlers":

| Event | Where it is already computed |
|---|---|
| `Click`, `RightClick`, `DoubleClick` | `update_pointer`, `tree.rs:1409` and the secondary path at `tree.rs:1463` |
| `HoverEnter` / `HoverLeave` | the hover-chain diff, `tree.rs:1492` |
| `Drag`, `PickedUp`, `Dropped`, `DroppedOn` | the press / release branches beside `down_on` and `drop` |
| `Scroll` | `scroll_by` past its no-op guard, `tree.rs:1117` |
| `Changed` | `drive_controls`, `widget.rs:774` |
| `Submit`, `FocusLost` | `update_keyboard` / `sync_field` in `keyboard.rs` |
| `Resized` | `place`, `tree.rs:1629`, when the absolute rect it writes differs |

**Frame order** in `lib.rs` becomes `update_pointer` → `update_secondary` →
`update_keyboard` → `dispatch` → `sweep_all` → `run_layout`. `dispatch` sits
right after line 1714. The ordering rule goes in a comment at both ends:
handlers run here and never from a component's `update`, so a handler locking
its component while the UI mutex is held cannot deadlock.

**Tests** (in `events.rs`, no GPU, same harness as `widget.rs`):

- click on a node with a handler dispatches exactly once; a node without one
  enqueues nothing
- `HoverEnter` and `HoverLeave` arrive as a pair when the pointer crosses
- `remove_node` drops the handlers and a later click on the recycled slot does
  not fire the old one
- `on_this` against a destroyed entity fires once, returns `Keep::No`, and is
  gone on the next event
- a handler that registers a handler during dispatch sees it on the next event
- a handler that re-enqueues itself forever panics at round 16
- 100 idle frames: `dispatched == 0`, `walks == 0`, upload returns
  `(i64::MAX, -1)`

## Phase 2: lists and trees keep their bind

`RowList` and `TreeView` store `children` and `bind` as boxed closures taken
at construction, and `sync` is deleted.

```rust
impl<H, P> TreeView<H, P> {
    pub fn new(ui, parent, viewport, style, root,
               children: impl FnMut(u64, &mut Vec<u64>) + Send + 'static,
               bind: impl FnMut(&mut UiCore, Row<'_, H>, u64) + Send + 'static) -> Self;
    pub fn on_row(&mut self, ui, kinds: Events,
                  f: impl FnMut(&mut UiCore, u64, &Event) -> Keep + Send + 'static);
    pub fn invalidate(&mut self, ui);                 // was: flag + next sync
    pub fn moved(&mut self, ui, id, parent, at);      // unchanged
    pub fn rebind(&mut self, ui, id);                 // one bound row
}
```

- The view registers a `Scroll` handler on its area. On scroll it computes
  the new `first`, and binds only rows whose data index changed, which is one
  row per boundary crossed. The pool still sizes itself from the measured
  viewport, so it also registers `Resized` on the area and grows the pool
  there instead of on the first two `sync` calls.
- `on_row` puts one handler on every pooled row node that maps the row to its
  bound id and forwards. Rows recycle, so the id is read at fire time, which
  is the identity argument `clicked` already made.
- `invalidate` and `moved` re-run the flatten and rebind the pool
  immediately, instead of setting `dirty` for the next `sync`.
- Drop targeting (`hovered_at`, the drop mark) moves into the view's own
  `Drag` handler on the area, so `set_drop_mark` is called on drag frames
  only.
- Tests to keep: `build` runs pool-size times for 10 000 rows; scrolling one
  row past a boundary rebinds exactly one row. New: an idle list issues zero
  `bind` calls over 100 frames, which is the one the old contract could not
  pass.

`DockSpace::update` (`dock.rs:437`) and `MenuBar::update` (`menu_bar.rs:120`)
convert the same way: header and divider polls become `Drag` / `Dropped` /
`Click` handlers registered when the panel or split is built. `Viewport` drops
its per-frame `set_rect` for a `Resized` handler. `Scrub` becomes a `Drag`
handler on the field.

## Phase 3: the editor

`Chrome` is the only editor component, so every editor handler resolves
through `This::<Chrome>` and finds its document by world id:

```rust
impl Chrome {
    fn doc(&mut self, world: WorldId) -> &mut Document { .. }
}
```

Mint the `This<Chrome>` in `Chrome::init` from the transform's entity and
world, and thread it into each panel's `new`. Panels are plain structs owned
by `Chrome`, as now.

Order, one panel per commit, each verified with `poke` before the next:

1. **HierarchyPanel.** Click / double-click / drop / right-click / add /
   remove become handlers at construction; `select`, `begin_rename` and the
   count label write at the moment they change. `drain_instantiated` stays
   in `Chrome::update` for now, calling `invalidate` on a non-empty drain.
2. **InspectorPanel.** Rebuild on selection change as today. `Submit` and
   `Drag` handlers per field replace the three per-frame row loops. Live
   readback is `ui.every(pane, 10.0, ..)` registered when the document starts
   simulating and stopped when it halts; in edit mode a value changes only
   through a gizmo drag, which publishes at its own drag end.
3. **FilePanel.** Same as the hierarchy; the status line writes on
   selection.
4. **Document and Chrome.** Play buttons, the View menu and the New Project
   dialog become handlers. What is left in `Chrome::update` is the
   `drain_instantiated` poll and the three keybinds, both of which can become
   subscriptions later.

Acceptance for the phase: the idle-editor profile's UI + taffy self share
drops from ~20% of the main thread to noise, and `sim_update` in the FPS line
drops from ~14 µs to ~1 µs. Keep the perf recipe from the
`ui-per-frame-cost-diagnosis` memory beside the numbers in
[benchmarks](notes/benchmarks.md).

## Phase 4: docs

- New note `docs/notes/ui-events.md` owning dispatch: the identity argument,
  the ordering rule, the emission table above, and the tick heap. Link it
  from the Retained-mode UI row in `Readme.md` and from
  [ui-core](notes/ui-core.md) "Input and the hit walk", which shrinks to the
  hit walk itself.
- [ui-widget-authoring](notes/ui-widget-authoring.md): rewrite invariant 1
  to say the gate guards *upload* and that a widget rebinds on events, with
  the new `TreeView` as the worked example; add "clears its handlers" to
  invariant 5; add a "handler" row to the cost table and a "register, do not
  poll" trap.
- [ui-widgets](notes/ui-widgets.md): `RowList` / `TreeView` constructor and
  `on_row`.
- [editor](notes/editor.md): `Chrome` as the identity every handler resolves
  through.
- ADR-0006 status line: API layer landed; note that `Ui<S>` became a
  per-node table plus `This<T>`, and why (there is no single app-state
  type; components are the state).

## Risks and open questions

- **Lock order.** The only new pair is UI mutex → component mutex, taken in
  `dispatch`. `OrbitController` takes the UI mutex from inside `sweep_all`
  while holding its own component lock, which is the reverse order, but the
  two never run at the same time. The comment at the dispatch call site is
  the guard; a debug assertion that `dispatch` is not re-entered is cheap
  and worth adding.
- **`Event::DroppedOn` payload ownership.** Today `dropped_on::<T>` hands out
  a reference and the drop is consumed at frame end. A handler gets the box
  by value; a target that declines should be able to let it fall through to
  the next claimant. Simplest rule: the first handler that returns `Keep`
  without downcasting successfully does not consume it. Decide in phase 1.
- **Context menus.** `menu_choice::<T>()` is polled by whoever opened the
  menu. Making menu items ordinary nodes with `Click` handlers carrying the
  payload removes the poll; it is phase 3 step 4 work and can slip.
- **Play-mode readback rate.** 10 Hz is the ADR's number. If a scrubbed
  value visibly lags under drag, the drag handler already writes on each
  drag frame; the tick is only for values moved by simulation.
- **`HAS_UPDATE` on `Chrome`.** Leaving it `true` with a two-line body costs
  one storage sweep per frame. Flip it once `drain_instantiated` is a
  subscription; do not chase it before then.
