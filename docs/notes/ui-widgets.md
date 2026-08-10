# UI widgets

Every widget is built to force one capability into the core. The capability
is the point; the widget is the thing that proves it works.

Building one? Read [ui-widget-authoring.md](ui-widget-authoring.md) first — the
invariants, the cost model, and the traps.

## Built

| Widget | Handle | Steps to build one | Capability it forced |
|---|---|---|---|
| `node` | `NodeId` | `ui.node(parent, style)` | the taffy tree; generational ids so a stale handle panics |
| `label` | `Label` | `ui.label(parent, px, color, text)` | glyph runs; text painted in tree order |
| `button` | `Button` | `ui.button(parent, text, style)` → `ui.clicked(b)` | `StateStyle` — look is a function of pointer state |
| `checkbox` | `Checkbox` | `ui.checkbox(parent, text, style)` → `cb.checked(&ui)` | **control storage** — the widget owns its value, click flips it before any app code runs |
| `slider` | `Slider` | `ui.slider(parent, style)` → `s.value(&ui)` | **drag** — press origin held while captured, so the gesture survives leaving the node |
| `text_field` | `TextField` | `ui.text_field(parent, text, style)` → `f.text(&ui)`, `f.submitted(&ui)` | **character events** — the OS resolves *what to insert* (`Keystroke::Text`, layout and dead keys already applied), a named `Key` + `Mods` says *what to do*; the value is a `String` in the control and the glyph run holds only the window that fits |
| `radio_group` | `RadioGroup` | `ui.radio_group(parent, &["a", "b"], style)` → `g.selected(&ui)` | **selection shared across siblings** — the first control whose value is not on the node the pointer hits. Clearing a sibling is something the hit node cannot name, so the index lives on the container and each option holds only a back-pointer; the click still resolves in the two lookups `drive_controls` always did |
| `scroll_area` | `NodeId` | `ui.scroll_area(parent, style)` | **per-node clip + offset** (its own `ui_group`); `Events::SCROLL` |
| `scrollbar` | `Scrollbar` | `ui.scrollbar(parent, area, style)` | **a widget that mirrors state it does not own** — the thumb follows an offset and a content extent that move without the bar being touched (a wheel, a resize, a list that grew), so nothing hung off the node the pointer hit would ever notice. The engine re-fits every bar after a layout and after a scroll; the thumb rides its own group's offset, so scrolling still writes group records and no quads |
| `RowList<H = Label>` | — | `RowList::new(..)` then `sync(len, bind)` | **virtualization** — node count follows the viewport, not the data |
| `TreeView<H = Label, P>` | — | `TreeView::new(..)` then `sync(children, bind)` | **splice-based sub-edits** — expand/collapse/move patch a preorder run instead of re-walking |
| `RowContent` | — | `impl RowContent for MyRow` | **construction as a type, not a closure** — a row is whatever `H` is, so `sync` binds and never builds; `Label` is the default |
| `Row<'_, H>` | — | handed to `bind` | derefs to the content (`r.set_text`), owns the row (`r.set_selected`) |
| drag session | `NodeId` (ghost) | `ui.grab(payload)` → `ui.dragging::<T>()` → `ui.dropped_on::<T>(n)` | **typed payloads on the pointer** — grab anything, drop anywhere; the ghost is minted per gesture so z-order is just tree order |
| `DropMark` | — | `list.set_drop_mark(ui, mark)` | the insertion line; `raise` for z-order against later-appended rows |
| event masks | — | `ui.set_events(n, CLICK \| HOVER \| DROP \| SCROLL \| FOCUS)` | **hit testing per kind** — one walk, five targets; hover is a set, not a winner; nearest sibling occludes |
| focus | — | `Events::FOCUS`, `ui.set_focus(..)`, `ui.keyboard_captured()` | **one retained target, no position** — a press takes it and a press anywhere else drops it; `Tab` walks the ring in tree order ahead of the focused widget, so nothing can trap it |
| theme | — | `ui::theme()`, `set_theme` | semantic colour roles; widget styles derive rather than hardcode |

## Next

Something concrete is waiting on each of these.

| Widget | Capability it would force | Blocked on |
|---|---|---|
| docking | drag-to-split panels, tab strips | reuses the scroll area's group machinery |
| drop-target highlight | `dragging::<T>().is_some() && hovered(n)` → a style | wants a second panel (inspector) to drop onto |
| context menu / popup | **overlay lifetime** — dismiss on outside click, anchored to a node | nothing; `raise` covers z-order |
| splitter | live resize writing back into sibling styles | nothing |

## Backlog

Not blocking anything, but needed for feature completeness. Grouped by the
capability they share — most of a group lands with the first one built.

**Keyboard** — focus, keystrokes and the editing model are built. Each of
these is now a widget rather than a capability.

| Widget | Notes |
|---|---|
| number field | text entry *and* horizontal drag on one node |
| text area | multiline: wrapping, vertical caret movement — the one that still needs a capability (a run per line, or a run that wraps) |
| search / filter field | `changed()` plus a predicate over the model |
| list & tree multi-select | shift/ctrl ranges. `Mods` exists per *keystroke*; the pointer layer still carries none, so a shift-click cannot tell itself apart from a click |
| tree keyboard nav | arrows to move and expand; take `Events::FOCUS` on the viewport |

**Exclusive selection** — the shared shape is built; each of these is now
assembly over `radio_group` rather than a capability.

| Widget | Notes |
|---|---|
| segmented control | the group restyled to a row, options styled as buttons — nothing here assumes a column |
| tabs | selection swaps which child renders — `Display::None` on the rest |
| dropdown / combo box | selection *plus* an anchored popup |

**Overlay** — all want the popup lifetime the context menu establishes.

| Widget | Notes |
|---|---|
| menu bar / nested menus | submenu chains, hover-to-open |
| tooltip | **dwell timing** — the first thing to need a clock in the pointer layer |
| modal / dialog | input capture: blocks the hit walk beneath it |
| toast / notification | timed self-removal |

**Composition only** — no new core capability, just assembly.

| Widget | Notes |
|---|---|
| toggle switch | a checkbox that looks different |
| progress bar | a slider with no input |
| toolbar | a flex row of icon buttons, once the icon leaf exists |
| accordion | collapsible section headers |

**New paint or layout work**

| Widget | Notes |
|---|---|
| icon / image leaf | `ui.image` paints into a *group*; a node-attached prim is what's missing. It rides the `background` slot, so the placement walk already sizes and hides it — what's new is the constructor and a measured leaf for the texture's natural size, exactly the shape of `label`. Icon button and toolbar fall out of it |
| colour picker | 2D gesture on a saturation/value square, hue strip, HSV↔RGB — the first widget needing a gradient fill |
| table / columns | resizable and sortable headers over `RowList`; column widths shared across rows |
| plot / graph | for the profiler — wants line primitives, which the quad pipeline has none of |
| viewport widget | a render target painted into the UI, for the editor's scene view |

## Known gaps in what exists

- Dragging onto a collapsed node should expand it (spring-loaded).
- No auto-scroll when a drag reaches the viewport edge.
- The drop line is not indented to the target's depth.
- Nested scroll areas `assert!` rather than compose. This is also why a text
  field draws a character window instead of clipping with a group of its own;
  that window deletes cleanly the day groups nest.
- No keyboard navigation in `TreeView`.
- No clipboard. `Ctrl+C/V/X` reach the field as `Key::Char` and do nothing —
  a system clipboard is the first thing the UI would take a dependency for.
- IME: composed text arrives only where the platform delivers it through
  `KeyEvent::text`. `set_ime_allowed` is never called and preedit is not
  drawn, so CJK input shows no underlined composition.
- The caret does not blink, deliberately — a blink is a timer that dirties a
  slot forever in a UI whose premise is that an idle frame uploads nothing.
- Shift-click does not extend a selection: same missing pointer modifiers as
  multi-select above.
- A scrollbar is vertical only, matching `scroll_area`, and the wheel over the
  gutter does nothing — `Events::SCROLL` names the node that scrolls, and the
  track is not it. It also sits *beside* the area rather than over it: an
  overlay bar wants a fade, and a fade is a timer that dirties a slot forever.
- A radio group is pointer-only. It takes no `Events::FOCUS`, so `Tab` walks
  past it and arrows do not move the selection — the convention is that a
  group is one tab stop and arrows move within it, which needs focus on the
  container and a keystroke route that reaches a control other than a field.
- A field's visible window is character-quantized, so a long value scrolls a
  character at a time rather than a pixel at a time.
- Hit walk is a full DFS every time the layout epoch moves — fine at
  ~40 nodes, wants incremental `place` then a BVH before it grows.
