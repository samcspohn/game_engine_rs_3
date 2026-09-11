# UI widgets

Every widget is built to force one capability into the core. The capability
is the point; the widget is the thing that proves it works.

Building one? Read [ui-widget-authoring.md](ui-widget-authoring.md) first —
the invariants, the cost model, and the traps. For the signatures alone, see
[the API digest](../api/engine-render/ui/index.md) (`make api`).

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
| `tabs` | `Tabs` | `ui.tabs(parent, &["a", "b"], style)` → `t.pane(&ui, i)`, `t.selected(&ui)` | **selection spent on layout** — the same shared value as a radio group, but a closed pane is `Display::None`, so its whole subtree collapses: nothing paints, nothing takes a hit, and the contents stay bound. The panes come back as plain nodes, so building into one is `ui.label(pane, …)` and nothing else |
| `image` | `NodeId` | `ui.image(parent, tex, style)` → `ui.set_image_uv(n, uv)` | **a node-attached texture** — the primitive kind existed and nothing could reach it, because a background was always a fill. An image *is* a background, so the placement walk already sizes it, hides it with its node and paints it in tree order; what it adds is the `tex` and a uv window, the same knob text uses to pick a glyph out of the atlas |
| `Viewport` | — | `Viewport::new(ui, pane, style)`, then `v.update(&ui)` a frame | **a widget that sizes something outside the UI** — every other widget takes the box taffy hands it and draws inside it; this one hands the box *back*, and the renderer re-allocates the camera's attachments to match (`CameraResolution::Fixed`). So the scene is rendered *at* the size it is shown at: no scaling, no skewed projection, one texel per pixel, and the aspect is the pane's because the hardware viewport covers the whole target. Its box is two answers at once — how big to make the camera, and whose pointer a drag belongs to (`scene::in_viewport`). A camera this size is no longer the swapchain's shape, so the present-blit *cannot* composite it: the frame drops the blit and the UI pass clears instead of loading. That switch is the honest statement of who paints the swapchain — the camera for a game, the widget for an editor |
| `DockSpace` | `PanelId` | `DockSpace::new(..)`, `d.panel(ui, "name")` → `d.content(p)`, `d.set_ratio`, then `d.update(ui)` a frame; `d.layout(ui)` / `d.apply(ui, &saved)` to keep the arrangement between runs, on `d.changed()` | **a live subtree that moves** — every widget before this was built where it lives, so a panel could only "move" by being rebuilt, throwing away the values its controls own and the text half-typed into its fields. `UiCore::set_parent` re-homes the subtree with every slot it had, so the panel the user dragged is the *same* panel. Splitting converts a leaf into a split **in place**, which is what keeps insert-at-index re-parenting out of the API. A saved `Layout` names its panels by **title**, not by `PanelId`: an id is an index into one run of the program, and it is the layout that outlives the run. A split's line is also its **splitter**: `flex_basis: 0` makes each half's `flex_grow` its proportion, so a drag writes two numbers and there is no ratio stored anywhere to fall out of step with the layout. **Nests**: a dock built into another dock's pane aims a lifted panel at its own leaves only, so its panels cannot leave it — which is how the editor keeps a document's hierarchy and inspector inside that document |
| `scroll_area` | `NodeId` | `ui.scroll_area(parent, style)` | **per-node clip + offset** (its own `ui_group`); `Events::SCROLL` |
| `popup` | `Popup` | `ui.popup(at, style)` → `ui.close_popup()` | **overlay lifetime** — a node minted at the root so it paints over everything, dismissed by the next press outside it, and that press *swallowed*: the click that closes a menu must not also press what the menu covered. Clamped to the window between the solve and the placement walk, so a menu opened at the screen edge opens inwards on the frame it opens rather than the frame after |
| `context_menu` | `Menu` | `ui.context_menu(at, &["a", "b"], payload, style)` → `ui.menu_choice::<T>()` | **a widget that outlives no handle** — the pick and what it is *about* come back off the store, typed, exactly as a drag's payload does. There is one popup for the same reason there is one `Grab`, so the caller stores nothing and has nothing stale to poll: the menu closes itself the frame after the pick, which is the frame the choice was read in |
| `MenuBar` | — | `MenuBar::new(ui, parent, &[("File", &["quit"])], style)` → `bar.update(ui)` | **an overlay that changes what it is anchored to** — a bar cannot be a row of buttons, because once one menu is open the press that would open the next is the one the overlay swallows to dismiss the first. So a second title takes the menu on *hover alone*, and the bar re-aims rather than re-opens. Which means it has to tell its own overlay from anybody else's: it asks by the payload it opened with, so a context menu in front of it is neither reported as a pick nor closed |
| secondary button | — | `ui.right_clicked(n)`, `list.right_clicked(&ui)`, `view.right_clicked(&ui)` | **a second button with no gesture** — a right click takes no focus, starts no drag and drives no control, so it is folded in after `update_pointer` rather than as four more parameters on it |
| `scrollbar` | `Scrollbar` | `ui.scrollbar(parent, area, style)` | **a widget that mirrors state it does not own** — the thumb follows an offset and a content extent that move without the bar being touched (a wheel, a resize, a list that grew), so nothing hung off the node the pointer hit would ever notice. The engine re-fits every bar after a layout and after a scroll; the thumb rides its own group's offset, so scrolling still writes group records and no quads |
| `RowList<H = Label>` | — | `RowList::new(..)` then `sync(len, bind)` | **virtualization** — node count follows the viewport, not the data |
| `TreeView<H = Label, P>` | — | `TreeView::new(..)`, optionally `.with_tree(id)`, then `sync(children, bind)` | **splice-based sub-edits** — expand/collapse/move patch a preorder run instead of re-walking. `TreeDrag::tree` tags a payload with the model it names, and a view answers only for its own: two trees over different models share a payload type, and an id from one means something else in the other |
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
| drop-target highlight | `dragging::<T>().is_some() && hovered(n)` → a style | nothing; the dock's aiming overlay is the same question answered geometrically |

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
| dropdown / combo box | selection *plus* an anchored popup |

**Overlay** — the popup lifetime is built (`ui.popup`), and so is the bar
that anchors one (`MenuBar`); each of these is now what to put in one,
except where noted.

| Widget | Notes |
|---|---|
| nested submenus | a submenu is open *while its parent is* — the one thing here that needs `Overlay` to hold a **stack** rather than the single popup it does |
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
| icon button / toolbar | the image leaf is built; what these still want is a **measured** one, sizing itself from the texture's natural dimensions the way `label` does from its string. `UiCore` cannot ask — the store that knows is on the GPU side of a boundary it does not cross — so it needs the size pushed in, and a font-free `ui.icon` on top |
| colour picker | 2D gesture on a saturation/value square, hue strip, HSV↔RGB — the first widget needing a gradient fill |
| table / columns | resizable and sortable headers over `RowList`; column widths shared across rows |
| plot / graph | for the profiler — wants line primitives, which the quad pipeline has none of |
| a *second* camera in a panel | built for one: `ui::CAMERA_TARGET` is a single reserved slot and `scene::set_viewport` a single rect. Two viewports want a `RenderCamera` per panel, each with its own target slot and its own framing — the plumbing is the same shape, the global is what has to go |

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
- Tabs are pointer-only and the strip never scrolls: enough of them and the
  headers wrap or overflow the panel. There is also no close button and no
  reorder — a tab is a fixed set named once, and `DockSpace` is the version
  whose set the user edits.
- A split's proportion can be named (`set_ratio`, which writes the same two
  `flex_grow` numbers a drag does) and a whole arrangement saved and put back
  (`layout` / `apply`), but there is still no double-click-to-even.
- **A dock pane does not clip.** Panes have no `ui_group` of their own, on
  purpose: a group buys one record per *move*, which a docked panel never
  makes, and it would turn every scroll area inside a panel into a nested
  one. So a line wider than its half draws over the neighbour. The dock's own
  box is safe — its root pins `min_size` to zero — but the contents are not.
- **One viewport per process.** `Viewport` publishes to a static, because
  `UiCore` owns no Vulkan and cannot hold a camera. A second one needs a
  `RenderCamera` per widget and a target slot each, not a second static.
- **Resizing a viewport re-allocates.** Every frame of a divider drag
  re-creates the colour, depth and both Hi-Z images, their descriptor sets,
  the extent-shaped secondaries and every frame slot — measured at ~0.8 ms
  per frame over a 0.35 ms baseline, which is a gesture-time cost and not a
  steady-state one. Hysteresis (resize on settle, stretch while dragging) is
  the fix if it ever matters; it did not, so it is not there.
- A dock panel cannot be closed or dragged out into a window of its own, and
  a leaf's strip does not reorder: a drop onto a strip appends.
- A radio group is pointer-only. It takes no `Events::FOCUS`, so `Tab` walks
  past it and arrows do not move the selection — the convention is that a
  group is one tab stop and arrows move within it, which needs focus on the
  container and a keystroke route that reaches a control other than a field.
- A field's visible window is character-quantized, so a long value scrolls a
  character at a time rather than a pixel at a time.
- Hit walk is a full DFS every time the layout epoch moves — fine at
  ~40 nodes, wants incremental `place` then a BVH before it grows.
