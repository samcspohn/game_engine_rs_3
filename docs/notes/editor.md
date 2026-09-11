
  1. #[export] derive + value model — new crates/engine-derive proc-macro
     crate, an Export trait in engine-core (TYPE_NAME + &'static
  [Property] with get/set fn pointers), and the value enum.

  The ADR is right that MeshRenderer is where a naïve model breaks, and it's
worse than it says: both fields are private
  (components.rs:39-44), mesh_id has no setter at all, and set_material (:100)
takes a &Transform and does retain/release refcounting. So the
  derive needs method-routed accessors (#[export(get = …, set = …)]), not
field access — decide that before writing the macro.

  2. enabled bitset — enabled + active_in_hierarchy Vec<AtomicU32> next to
     active/has_children, grown in the same block at
  transform/mod.rs:589-594. CPU side is one AND into the word load at
component/mod.rs:251 (par_iter already takes &TransformHierarchy). GPU
  side pushes [transform_id, NO_RENDERER, MATERIAL_INHERIT] through the
existing queue drained at lib.rs:1912 — re-enabling needs to reach
  the live MeshRenderer to get the real ids back.

  3. Play root as a sibling — the one place the ADR understates the work.
     Scene::instantiate (component/mod.rs:613) deep-clones another
  Scene, walking 0..len. You need a subtree clone within one Scene, and
clone_from_other borrows source and destination storage from the same
  self.components map. That's a real refactor, not a call.

  4. Node keys + delta — NodeKey on TemplateNode (scene_asset.rs:109),
     populated in build_template from node.index() / prim.index() (:388,
  :409) instead of the nodes.len() push order. Then SubsceneInstance, the save
walk, and threading a delta through spawn_subscene since drain
  lands frames later.

  5. Per-call catch_unwind (component/mod.rs:266) — clears the entity's
     enabled bit, so it depends on step 2; the set_hook ring buffer feeds
  the Console pane, which is a placeholder today (main.rs:126).

  6. Inheritance — base: Option<AssetRef> and a recursive call, by which point
     it's small.

## The editor as it stands

The editor opens the test-game project by default (`--project
crates/test-game`) and shows a cube in its viewport. Nothing animates it: a
document world is created with `simulating: false`, so edit mode is a registry
nobody sweeps ([ADR-0011](../ADR-0011-worlds.md)) and no `update` runs at all.
The editor-side `Spinner` that used to stand in for a project component is
gone — behaviour is the project's to define, and [scripts](scripts.md) is how
it arrives. Its UI ([ADR-0008](../ADR-0008-ui-integration.md)) is **a
`DockSpace` filling the window with one panel per open document**, and **each
document is a `DockSpace` of its own**
([`crates/editor/src/main.rs`](../../crates/editor/src/main.rs)): inside a
document, *Hierarchy* / *Scene* / *Inspector* are sub-panels the user arranges
freely; outside it, the documents open **tabbed into one leaf** — *cube* in
front, *sphere* behind it — with *Console* — still a placeholder — and the
*Browser* tabbed underneath. Editing is **per document** — a hierarchy shows
one world, an inspector shows that hierarchy's selection, and the gizmo aims
that document's camera at it — so each document keeps its own selection rather
than one panel switching between them, and dragging a document's tab out puts
the two scenes side by side without anything else changing. **The nesting is
what confines them**: a `DockSpace` only aims a lifted panel at its own
leaves, so a *Hierarchy* dragged over the next document lands nowhere and
stays put, while every arrangement inside its own document — left, right,
tabbed onto the view — is still the user's to make. Nothing tests for "is this
the same document"; there is no rule to keep in step, because the only dock
that can accept the drop is the one that owns the panel. The *Hierarchy* is
the same `TreeView` over the live `TransformHierarchy` (click to select,
double-click to rename, drag to re-parent), and its rows now carry a payload
**tagged with the world they came from**: a row dragged into another
document's tree is refused rather than resolved, because an entity id is a
slot index into one world and the same number names something else in every
other ([ADR-0011](../ADR-0011-worlds.md) §2) — without the tag the second
document would silently re-parent whatever sits at that index, or panic when
it holds fewer entities. `TreeDrag::tree` is that tag and defaults to zero, so
a panel with one tree never mentions it. Above the tree sit its two structural
edits: **new** spawns an empty entity as a child of the selection — the root
when there is none — then selects it and expands whatever it landed under, and
**delete** destroys the selection and everything below it, refusing the root
because that is structure rather than content. The `Delete` key does the same,
but only for the panel the pointer is over: every document holds a selection
of its own, and one keystroke must not reach all of them. Both edits queue on
the world and land at the next frame boundary
([ADR-0011](../ADR-0011-worlds.md) §3), which is a frame *after* the click —
so the tree re-walks then rather than on the frame that asked, and the spawn's
builder leaves the id it was handed where the panel can pick the new entity up
and select it. The row count reads `TransformHierarchy::active_len()` rather
than `len()`: a removed slot is freed but nothing reuses it, so `len()` is a
high-water mark a delete would leave unchanged. Right-clicking a row opens a
**context menu** on the same three operations — *new child*, *rename*,
*delete*, and only *new child* on the root — built on the UI's new overlay
lifetime
([`crates/engine-render/src/ui/popup.rs`](../../crates/engine-render/src/ui/popup.rs)):
a node minted at the root so it paints over the panels rather than being
clipped to the one that opened it, dismissed by the next press outside it, and
that press **swallowed**, so the click that closes a menu cannot also press
what the menu was covering. There is one popup at a time for the same reason
there is one drag, and the choice comes back off the store *typed* the way a
drag payload does — `menu_choice::<EntityRef>()` — so two documents' menus
tell themselves apart by the world in the payload, the panel keeps no handle,
and there is nothing stale to poll after the menu closes itself the frame
after the pick. The secondary mouse button is folded in after `update_pointer`
rather than added to it: a right click takes no focus, starts no drag and
drives no control, so all it needs is the node it went down and up on. Across
the top of the window sits a **menu bar** on that same overlay
([`crates/engine-render/src/ui/menu_bar.rs`](../../crates/engine-render/src/ui/menu_bar.rs))
— *File* opens a new scene, saves or reloads the one in front, or quits,
*View* brings the *Console* or *Browser* tab forward, *Tools* switches the
gizmo the way W/E/R does. A document carries the file it round-trips through
(`<project>/scenes/<title>.json`), and *reload scene* is what makes a save
checkable: it clears the document and reads the file back, so what returns is
exactly what was written (see [scene-file](scene-file.md)). A title opens its
menu anchored under it, and once one is open **hovering another title switches
to it with no click at all**, because the press that would have opened it is
the press the overlay swallows to dismiss the first. That is what keeps a bar
from being a row of buttons: it has to know whether the open overlay is *its
own*, which it asks by the payload it opened with — so a row's context menu
opened in front of it is neither reported as a pick nor closed out from under
the panel that owns it. Nested submenus are not built; `Overlay` holds one
popup, and a submenu is the case that needs a stack of them.

The **Browser** is that same `TreeView` over the project directory rather than
over a world — the view's own header names a folder listing as the other thing
it consumes, and it costs a `children` closure to find out. The listing is
**read in full and cached**, because `sync` asks a visible row for its
children on every frame it draws one, to know whether to give it an arrow: a
tree that hit the filesystem there would `readdir` once per visible row per
frame. So the project is walked once into an arena a node id indexes,
`target/` and dot-names pruned — a browser that opens on build output shows
what the project was built into rather than the project — and **refresh**
re-walks it, which is what covers a save, a `cargo build` or a file dropped in
from outside, none of which the editor is told about. A node's path is
relative to the directory the editor entered, so what a row names is already
what a scene file is named by. Double-clicking a directory toggles it, and
double-clicking a file **reports it to the chrome** rather than acting: what a
document is belongs there, and only the chrome can see the file is open
already — in which case the double click selects that tab instead of opening a
second one. A scene opens as a document of its own; anything else says so in
the console. The load happens **before the panels exist** — the world is made,
the file is read into it, and only then does it get a camera, a rig entity and
a pane — so `project.json`, which is JSON and is not a scene, leaves a console
line and no empty document behind, because the handle was the only thing
holding that world. That ordering is also why *new scene* and *open scene* are
one function taking a world rather than two that drift.

The starting arrangement is built from the same `dock` call a drop makes, plus
`set_ratio` to say what the split proportions are, and after that the user
owns the layout: every panel can be dragged to another edge, joined to another
strip, or resized, and the hierarchy keeps its scroll offset and a half-typed
rename across the move because `set_parent` re-homes the subtree instead of
rebuilding it — including when the whole document panel moves, which carries
its inner dock with it. The *Inspector* shows the selected entity's
**transform first** — position, rotation and scale, each as three editable
fields, rotation in Euler degrees because nobody authors a quaternion — and
then every `#[export]`ed property of every component on it. Editing is per
*kind* rather than per property, so a `Vec3` on a game's own component gets
the same three coupled fields the transform does, and the fields share their
row's width rather than each claiming 90 px. A numeric field is also a
**scrubber**: drag it sideways and the value follows the cursor — 0.01 per
pixel for a length, 0.25° for a rotation — measured from the press rather than
integrated, and applied every frame it moves so the viewport never lags the
number. That split is deliberate: the UI owns the gesture and the panel owns
its meaning. `ui::Scrub` works on *any* node and holds only the anchor
`ui.drag` cannot supply, while the step and the number format stay in the
editor. Nothing marks a field as draggable — there is one pointer and so one
gesture, so `Scrub` calls `ui.claim_drag(n)` when it takes it and a field
whose drag is spoken for stops selecting, which means "draggable" and "does
not select" are the same fact rather than two that can drift. A field nobody
claims still selects text with no configuration at all, and a **double click
selects the whole value**, so typing replaces it. **Enter** commits and
reformats in place — `3` becomes `3.000` without waiting for blur, which is
also where a setter that clamps or refuses shows its answer — and leaves the
value focused and selected, so the next number is typed straight over it. The
transform is the one section not read through `Export`: the hierarchy owns it,
not the registry, so the panel reads and writes it directly. The root gets no
section — it is the identity the hierarchy composes from rather than a pose.
Under the last section sits **Add Component**, which is where a project's own
components become reachable rather than merely visible: the menu lists every
type in `engine_core::script`'s registry — the engine's own, and whatever the
project's script dylib registered — minus the ones the entity already carries,
so the list shrinks as an entity fills out. The panel names no type and holds
no table of its own; it asks the registry, which is the same list a scene file
will name components by. Its payload is an `AddTo` rather than the `EntityRef`
the hierarchy's row menu carries, because `menu_choice` discriminates by
payload *type* and two menus holding the same two numbers would otherwise read
each other's pick against the wrong list of items. The add queues on the world
like every other structural edit and lands at the next frame boundary
([ADR-0011](../ADR-0011-worlds.md) §3) — `World::edit` is that queue's
already-exists case, and it drops the edit when the entity was destroyed in
the meantime, because a delete queued alongside it wins. So the panel cannot
rebuild on the click: it rebuilds when the entity's component list stops
matching what is drawn, which is true whichever frame the boundary ran on and
needs no guess about how late the change is. The root gets the button even
though it gets no transform section — it is a pose the hierarchy composes
from, not an entity that cannot carry behaviour. Selecting an entity puts a
**TRS gizmo** on it in the scene view — **W** move, **E** turn, **R** scale,
drag an axis, a plane or a ring — over a **world grid** that fades out with
distance and is occluded by whatever is in front of it
([`docs/notes/gizmo.md`](../notes/gizmo.md)). The gizmo runs in the renderer
between the UI's pointer update and the sweep rather than as a component,
because it and `OrbitController` answer to the same press and a component
would race it. Above each *Scene* sits its **play toggle**, which is a
per-document control because play is a per-document thing: it deep-copies that
document into a world of its own and aims the panels and the camera at the
copy, so two documents can be running at once and the document itself is never
touched — see [play-mode](play-mode.md). Each document's *Scene* holds a
`ui::Viewport`, which is where the camera lives now: the panel's box *is* the
camera's target, so dragging a divider re-renders the scene at the new size
rather than rescaling it, and the whole window-sized render plus its
present-blit are gone. A document tabbed behind another publishes a zero box,
which the camera reads as "present but not showing" and holds its size
through. A dock filling the window also has to cover every pixel, which is why
a leaf paints its surface across its whole box rather than only behind its
panes: the gap between a strip and its pane was a hard-edged strip of raw
camera.

## test-game's UI

`test-game` builds a different UI from the same public API, as a `UiDemo`
**component**
([`crates/test-game/src/ui_demo.rs`](../../crates/test-game/src/ui_demo.rs)):
a rounded, bordered panel with a live readout above a **three-tab strip**
filing everything else: a **click-me button with a click counter**, a
**checkbox** and a **slider** that own their own values and a **radio group**
whose selected index the container owns under *controls*; a **virtualized
collapsible hierarchy with a scrollbar beside it** you can wheel through,
expand, click to select, and **drag a row onto another to re-parent** under
*scene* — the bar's thumb tracks all three without the component telling it
anything; and the ASCII specimen and grid swatch strip under *font*. A closed
tab's pane has no box at all, yet the readout keeps obeying the radio group
inside it — all laid out by taffy rather than hand-positioned. Nothing in the
component computes a coordinate, picks a hover fill or handles a control's
click: taffy places the boxes, `StateStyle` supplies the three looks on
transition, and `update_pointer` moves the checkbox and slider values. A drop
is the one place the component does work, and it is two lines — re-parent its
own `HashMap`, then tell the view — because the view reports the move rather
than performing it and holds no structure of its own to have got ahead. Beside
it, a `DockDemo`
([`crates/test-game/src/dock_demo.rs`](../../crates/test-game/src/dock_demo.rs),
**F3**) is three panels in a `DockSpace`: drag a tab header onto another
pane's edge to split it or onto its middle to join its strip, drag the line
between two panes to resize them, and empty a pane to fold its split away. The
claim it exists to show is in *inspector* — type into its field, move its
slider, then drag the tab somewhere else and everything is still there,
because `set_parent` moved the subtree instead of rebuilding it. Between them
the binaries cover both regimes — event-driven and static — and none of these
overlays exists in the renderer, so each appears only in the binary that asked
for it. **F4** toggles the game's, and **F5** flips the checkbox's value with
no pointer anywhere near it — the mark follows, which is why a control owning
its value still has a setter; nothing in the component handles the click at
all; `ENGINE_UI_TRACE=1` prints a line only on frames where the UI uploaded
anything at all.
