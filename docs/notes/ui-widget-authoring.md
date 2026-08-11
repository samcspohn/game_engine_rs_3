# Building a UI widget

Field notes from implementing `text_field`, focus, and double click. Read
[ui-widgets.md](ui-widgets.md) for *what* to build; this is *how*, and what
bites.

## The five invariants everything rests on

1. **Every write is gated.** All paths funnel through `SlotArray::set`, which
   compares before it marks dirty. Re-writing an unchanged value costs a
   comparison and zero bytes. **So do not build dirty tracking** — re-bind
   everything every frame and let the gate sort it out. `RowList::sync` is the
   worked example.
2. **Paint order is tree order.** The placement walk emits a node's background,
   then its own glyphs, then its children. Slot numbers mean nothing (the free
   list recycles low slots). z-order is `raise()`, which moves a node to the end
   of its parent's children.
3. **Layout runs on events, not frames.** `run_layout` returns immediately
   unless taffy is dirty or the slot allocator moved a run. `set_node_style`
   dirties taffy; `set_background` does not — which is why hover is nearly free
   and a caret move is not (but still fine at editor scale).
4. **A zero-area quad is culled** by `ui.vert`. That is how freed slots, hidden
   runs, and invisible widget parts all work: give it zero width.
5. **Groups are flat**, composed on the CPU. Nested scroll areas `assert!`.
   Anything wanting its own clip inherits that limitation — see the character
   window in `text_field.rs` for the workaround and why it was chosen.

## Adding a widget

1. **Typed handle** — `handle! { MyWidget }` in `widget.rs`, next to the others.
   Methods that need the widget's *structure* go on the handle; anything true of
   any node (`clicked`, `node_rect`, re-parenting) stays on `UiCore` and takes
   `impl Into<NodeId>`.
2. **Style struct** — `MyStyle` with `impl From<Theme>` and
   `impl Default { theme().into() }`. Never hardcode a colour; every existing
   widget derives from a role.
3. **Constructor** — `UiCore::my_widget(parent, …, style) -> MyWidget`. Compose
   nodes, call `set_background` / `set_state_style`, opt into events last.
4. **Owns its value?** Add a `Control` variant. `update_pointer` /
   `update_keyboard` apply input *before* any component runs, so callers only
   ever read. Non-`Copy` state gets boxed (see `Control::TextField`) and is
   reached with the take-and-return `with_field` pattern — everything worth
   doing to it also needs `&mut UiCore`.
   **The value does not have to sit on the node the pointer hits.** If a click
   has to change a *sibling*, put the value on the container and give each
   child a back-pointer to it — `Control::Radio` is two fields and forwards to
   `Control::RadioGroup`. `drive_controls` still does one table lookup on the
   clicked node, so shared state costs nothing per frame.
   **A selection can be spent on layout rather than on a fill.** `Control::Tabs`
   is `Control::RadioGroup` with the payoff changed: instead of swapping two
   dots, it collapses one pane and opens another with `set_visible`. That is a
   relayout, and it should be — hiding a subtree is a layout change — but the
   strip half stays a fill swap and does not move.
   **A widget can move a subtree instead of rebuilding it.** `set_parent`
   re-homes a live node and everything under it — same slots, same control
   values, same scroll offsets — so "the panel the user dragged" stays the
   same panel. It panics across a clip group, because a primitive's group is
   fixed when its slot is allocated. `DockSpace` is the worked example.
   **A widget that shows someone else's value stores none of its own.** A
   scrollbar reads the area's offset and extent; duplicating either would be a
   mirror to keep in sync. What it needs instead is a refresh, and the answer
   is *not* a per-frame one — register it (`UiCore::scrollbars`) and re-fit it
   where the truth can move, which for a scroll is `scroll_by` and `run_layout`
   and nowhere else.
5. **Clear it on removal.** `free_subtree` clears every index-keyed side table.
   A new per-node table that is not cleared there hands a recycled slot state it
   never asked for.

## Cost model

| Operation | Cost |
|---|---|
| restyle a background | one `ui_style` record, no layout |
| retype a label | only the glyphs that differ |
| scroll a list | one `ui_group` record, whatever the length (two with a scrollbar — the thumb moves the same way) |
| recolour a label | one `ui_style` record per glyph, no layout |
| switch a tab | two header fills, both labels' glyphs, and a relayout of the two panes |
| drag a split's divider | two `flex_grow` numbers and one relayout; no quad is written by the drag itself |
| move a panel between docks | two `set_parent` calls and one relayout — no primitive is written, because none of them changed |
| `set_node_style` | taffy relayout of that path next frame |
| idle frame | **zero** bytes, zero workgroups |

The last row is a hard invariant, not an aspiration. Anything that dirties a
slot on a timer breaks it — that is why the caret does not blink, and why a
tooltip's dwell clock is a real design question rather than a detail.

## Testing

`UiCore` owns no Vulkan, so widgets test fully without a GPU.

```rust
let mut core = UiCore::new();
let w = core.my_widget(root, style);
core.run_layout([400.0, 400.0]);
core.update_pointer(p, true, false, 0.0, 0.0);   // pos, pressed, released, wheel, now
```

- **Prove the zero-upload claim.** `SlotArray::upload` returns
  `(i64::MAX, -1)` when nothing is dirty. Assert it after an idle frame.
- **Pin the per-interaction cost.** Count set bits in the dirty mask —
  `one_keystroke_dirties_two_quads` in `text_field.rs` caught a real bug that
  no behavioural test would have.
- **Virtualized rows need two `sync` passes** before they exist: the pool sizes
  itself from the *measured* viewport, so the first pass has no layout to read.
- **Time is a parameter.** `update_pointer` takes `now` so gesture tests stay
  deterministic instead of sleeping.

## Traps

- **`update_pointer` must run every frame, before `update_keyboard`** — it is
  what clears `clicked`. A test that calls only its own settle helper replays
  last frame's click forever; the rename tests failed exactly this way.
- **Pooled rows recycle.** Per-row state must key on the data id, never the row.
  `TreeView::row(id)` reaches into a bound row from outside `sync`, and returns
  `None` when it is not pooled — which is a real case, not a formality.
- **A row's height is load-bearing**: `row_h` converts scroll offset into a data
  index. A widget that makes its row taller desynchronises the whole list.
- **Absolutely-positioned children need a padding-free container**, or every
  inset silently carries the parent's padding.
- **Moving a node costs a relayout — unless it has a group.**
  `open_content_group` gives any node the scroll area's machinery, and
  `set_content_offset` then translates its children for one `ui_group` record.
  That is how the scrollbar thumb moves without dirtying taffy; it also clips
  the children to the node's box for free.
- **Hidden lives in the style.** `set_visible` is a read-modify-write of
  `display`, so a caller that restyles a hidden node with a fresh `Style`
  reopens it. Panes, collapsed rows and the demo's F4 panel all share this.
- **`flex_grow` over a zero basis is a proportion you can write to.** It is
  how a dock resizes without keeping a ratio of its own — the layout is the
  only copy. The corollary is that a *share* belongs to the node, so anything
  restyling a node wholesale (`split`, `collapse`) has to carry `flex_grow`
  across by hand or silently reset the user's drag.
- **A flex item will not shrink below its own content unless told it may.**
  `min_size: 0` is what makes a split even; without it one long readout in
  one pane widens the whole dock past the box it was handed. Every container
  whose size comes from its *parent* rather than its children needs it.
- **Anything added to a scroll area scrolls.** A decoration that must stay put
  — a scrollbar, a header — goes beside the area, not inside it.
- **Taffy accumulates content size along the main axis only.** A scroll area
  left in the default row direction reports no vertical overflow, so
  `max_scroll` is 0 and a scrollbar quietly shows nothing. `RowList` sets
  `FlexDirection::Column`; a hand-built area has to as well.
- **Taffy rounds to whole pixels.** Geometry assertions need ±1 tolerance; a
  fractional glyph advance will not line up exactly.
- **A `Display::None` node's descendants keep stale layouts.** Skip such
  subtrees whole when walking (`focus_order`, `text_nodes`), and remember a node
  with no box cannot hold focus — `run_layout` drops it.
- **Park invisible things at a fixed position.** An empty selection rect that
  followed the caret dirtied a slot per arrow key to move a quad nobody could
  see.
- **Window text you do not clip.** The field's hint drew straight through its
  border until it was cut to `cols` like the value.

## Verifying in the real app

Use `tools/poke` (see the `poke` skill), never ydotool/kdotool/spectacle:

```sh
ENGINE_DEBUG_INPUT=1 cargo run -p editor &
tools/poke tree                    # every visible text node + on-screen rect
tools/poke dblclick "cube"         # aim by text; poke resolves the layout
tools/poke wait && tools/poke focus
tools/poke shot                    # PNG of the frame, path printed
tools/poke rec 26 drag a --to b    # film the gesture, one PNG per frame
```

Assert with `focus` / `find`; use `shot` to check appearance, not to read back
state.

Movement is swept, one step per frame, so a drag crosses every row between its
endpoints — which is what exercises hover transitions, the drag threshold and
the drop mark. A white dot marks the pointer: if it sits on one row while a
different row is highlighted, the hit walk has an offset bug.

## Where things live

| | |
|---|---|
| `ui/mod.rs` | slot arrays, records, the change detector, `UiCore` |
| `ui/tree.rs` | nodes, taffy layout, the placement walk, pointer input, hit walk |
| `ui/widget.rs` | typed handles, `Control`, `StateStyle`, button/checkbox/slider |
| `ui/text_field.rs` | the field, and the editing model as pure functions |
| `ui/keyboard.rs` | focus and keystroke routing |
| `ui/list.rs` | `RowList`, `RowContent`, `Row` — virtualization |
| `ui/tree_view.rs` | `TreeView` — splice-based expand/collapse over `RowList` |
| `ui/dock.rs` | `DockSpace` — the cell tree, splitting, collapsing, aiming |
| `ui/gpu.rs` | the four scatters, the atlas, the single indirect draw |
