# UI widgets

Every widget is built to force one capability into the core. The capability
is the point; the widget is the thing that proves it works.

## Built

| Widget | Handle | Steps to build one | Capability it forced |
|---|---|---|---|
| `node` | `NodeId` | `ui.node(parent, style)` | the taffy tree; generational ids so a stale handle panics |
| `label` | `Label` | `ui.label(parent, px, color, text)` | glyph runs; text painted in tree order |
| `button` | `Button` | `ui.button(parent, text, style)` → `ui.clicked(b)` | `StateStyle` — look is a function of pointer state |
| `checkbox` | `Checkbox` | `ui.checkbox(parent, text, style)` → `cb.checked(&ui)` | **control storage** — the widget owns its value, click flips it before any app code runs |
| `slider` | `Slider` | `ui.slider(parent, style)` → `s.value(&ui)` | **drag** — press origin held while captured, so the gesture survives leaving the node |
| `scroll_area` | `NodeId` | `ui.scroll_area(parent, style)` | **per-node clip + offset** (its own `ui_group`); `Events::SCROLL` |
| `RowList<H = Label>` | — | `RowList::new(..)` then `sync(len, bind)` | **virtualization** — node count follows the viewport, not the data |
| `TreeView<H = Label, P>` | — | `TreeView::new(..)` then `sync(children, bind)` | **splice-based sub-edits** — expand/collapse/move patch a preorder run instead of re-walking |
| `RowContent` | — | `impl RowContent for MyRow` | **construction as a type, not a closure** — a row is whatever `H` is, so `sync` binds and never builds; `Label` is the default |
| `Row<'_, H>` | — | handed to `bind` | derefs to the content (`r.set_text`), owns the row (`r.set_selected`) |
| drag session | `NodeId` (ghost) | `ui.grab(payload)` → `ui.dragging::<T>()` → `ui.dropped_on::<T>(n)` | **typed payloads on the pointer** — grab anything, drop anywhere; the ghost is minted per gesture so z-order is just tree order |
| `DropMark` | — | `list.set_drop_mark(ui, mark)` | the insertion line; `raise` for z-order against later-appended rows |
| event masks | — | `ui.set_events(n, CLICK \| HOVER \| DROP \| SCROLL)` | **hit testing per kind** — one walk, four targets; hover is a set, not a winner; nearest sibling occludes |
| theme | — | `ui::theme()`, `set_theme` | semantic colour roles; widget styles derive rather than hardcode |

## Next

Something concrete is waiting on each of these.

| Widget | Capability it would force | Blocked on |
|---|---|---|
| `text_field` | **keyboard focus + character events** — winit text/IME, caret, selection. The only genuinely new input axis left | nothing |
| docking | drag-to-split panels, tab strips | reuses the scroll area's group machinery |
| drop-target highlight | `dragging::<T>().is_some() && hovered(n)` → a style | wants a second panel (inspector) to drop onto |
| scrollbar | a visible thumb — scrolling is wheel-only today, with no indication that content extends past the viewport | nothing |
| context menu / popup | **overlay lifetime** — dismiss on outside click, anchored to a node | nothing; `raise` covers z-order |
| splitter | live resize writing back into sibling styles | nothing |

## Backlog

Not blocking anything, but needed for feature completeness. Grouped by the
capability they share — most of a group lands with the first one built.

**Keyboard** — all of these arrive with `text_field`.

| Widget | Notes |
|---|---|
| number field | text entry *and* horizontal drag on one node |
| text area | multiline: wrapping, vertical caret movement |
| search / filter field | text field plus a predicate over the model |
| list & tree multi-select | shift/ctrl ranges — needs modifier state in the pointer layer |
| tree keyboard nav | arrows to move and expand; focus ring |

**Exclusive selection** — one value shared across sibling nodes, which no
control does yet. Today `Control` is per-node.

| Widget | Notes |
|---|---|
| radio group | the base case |
| segmented control | radio group with button styling |
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
| icon button | `ui.image` already exists |
| toolbar | a flex row of icon buttons |
| accordion | collapsible section headers |

**New paint or layout work**

| Widget | Notes |
|---|---|
| colour picker | 2D gesture on a saturation/value square, hue strip, HSV↔RGB — the first widget needing a gradient fill |
| table / columns | resizable and sortable headers over `RowList`; column widths shared across rows |
| plot / graph | for the profiler — wants line primitives, which the quad pipeline has none of |
| viewport widget | a render target painted into the UI, for the editor's scene view |

## Known gaps in what exists

- Dragging onto a collapsed node should expand it (spring-loaded).
- No auto-scroll when a drag reaches the viewport edge.
- The drop line is not indented to the target's depth.
- Nested scroll areas `assert!` rather than compose.
- No keyboard navigation in `TreeView`.
- Hit walk is a full DFS every time the layout epoch moves — fine at
  ~40 nodes, wants incremental `place` then a BVH before it grows.
