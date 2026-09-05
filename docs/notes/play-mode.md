# Play mode

**Play runs the game, not the document.** It loads the project's startup
scene into a world of its own, sets it running, and shows it through whatever
camera that scene carries — which is what a packaged build does, from the same
file, in the same order. Stop drops the world.

That is the whole design goal: the button is not a preview of the panel it
sits in, it is the product. A document is a file being edited; the game starts
where [the project](packaging.md#the-project-file) says it starts.

The control is a button above each document's viewport rather than one in the
menu bar — it is per document only in the sense that any document can press
it, and two of them running at once is two copies of the game.

## A game with no camera draws nothing

The viewport binds to the running scene's own camera (`camera::of_world`,
which is the `CameraComponent` binding list read by world). While that camera
is up, the editor's own is not driving anything: no grid, no orbit rig, no
gizmo geometry the game did not ask for. Stop puts the document's camera back.

A scene with no `CameraComponent` gets no camera, and the panel says so
instead of leaving the last frame on screen. A stale image would be the one
answer that is wrong in both directions — it is not what the editor was
showing and not what the game will.

## Edit-time construction, play-time `init`

A scene that carries a camera can be *opened* as well as played, and
`CameraComponent::init` mints a real camera with real attachments. Doing that
for a document would put a second view on screen and take a camera slot
nothing samples — which is [ADR-0010](../ADR-0010-scene-authoring-and-play.md)
§5's rule, finally with a case that needs it.

So `Component::INIT_IN_EDIT` says whether a type's `init` runs in a world that
is not simulating. It defaults to `true` and the rule is stated per type,
because the blanket version is wrong in the other direction: `MeshRenderer`
has to publish its GPU record in a document or the viewport goes black.

## Camera slots come back

Loading a scene with a camera mints a camera. Doing it every time the button
is pressed, against a hard `MAX_CAMERAS` of 8 and a registry that never reused
a slot, gave a session six plays before it asserted.

So `CAMERAS` holds `Weak`s and hands a dead slot to the next camera made,
exactly as `worlds` does. Two things had to stop keeping dead cameras alive:

* **The bound-camera list**, which held a strong handle per
  `CameraComponent`. A world dropped whole never reaches `deinit`, so those
  entries were immortal. It holds `Weak`s now and drops what no longer
  upgrades, which is the only sweep it needs.
* **The renderer**, which holds one per camera it has built a device half
  for — and is therefore the *only* thing that can tell a camera has no owner
  left, since its own handle is what makes the slot look occupied.
  `CameraHandle::is_orphan` is that test; `camera::retire` hands the slot
  back. The device half stays until something takes the slot, so there is
  nothing to tear down and `camera_targets` stays indexed by slot.

An idle camera also comes out of the frame primaries — it would otherwise
draw a whole scene into a target nothing samples, every frame, forever.

## A scene names its components, so they have to be registered first

A scene file names components by `TYPE_NAME`, and one read before the registry
is filled loses every component in it — quietly enough that the first symptom
is a black window. Two paths fill it, and both now happen early enough:

* `engine::new_world` registers the engine's own types. It wraps
  `engine_core::new_world` for exactly this reason: making a world is the
  first thing a game does, and it is earlier than `Window::new`, which is
  where the registration used to live alone.
* `declare_scripts!` emits a plain `register()` beside the `dlopen` entry
  point, for a game binary that links the same crate as an rlib and never
  opens anything. Without it a packaged build drops the project's own
  components and the editor does not, which is the one difference that would
  make "play is the product" a lie.

## Retiring a world

"Stop-play is dropping the handle" was true of the editor and false of the
process: `WorldRender` holds a `WorldHandle` for every world the renderer
draws, and nothing ever dropped one. A stopped game would have stayed alive,
kept its GPU buffers, and — being a simulating world in `worlds::live()` —
gone on running invisibly, forever, once per play.

So the renderer retires a world nobody outside it names:
`WorldHandle::is_orphan` is `strong_count == 1`, and the retain runs at the
top of the frame **before** `worlds::live()`, which would otherwise be holding
a handle of its own. The window's own worlds are held by `RenderApp` for the
whole run, so `DRAWN` is never a candidate and the index stays put.

Retiring has to force the same rebuild adopting does, and for a sharper reason
than symmetry: a `WorldId` is handed to the next world made, so a camera's
`draw.world != src.id` test reads a fresh world in a reused slot as the one it
already drew, and keeps its descriptor sets pointed at buffers that went with
the old one. The second play in a session drew nothing at all until retirement
fed `worlds_changed`.

## The staging phase

`GpuRenderers` keeps a `write_slot` that has to stay in step with the frame
loop's: the host writes `spawn_staging[write_slot]` and the FrameSlot's
pre-recorded scatter reads its own. It started at zero. That is correct for
every world the window is handed at startup, because the frame loop starts at
zero too — and wrong for every world adopted afterwards, which joins at
whatever phase the loop has reached and then writes every renderer record it
ever has into a slot no frame reads. The renderer is simply never there.

It is a mid-run adoption bug rather than a play-mode one, and it was invisible
because nothing adopted worlds routinely before this. `WorldRender::new` now
takes the phase to join at.

## What is deliberately not here

**Playing the document.** The button used to deep-copy the document world and
run that, which is what `World::duplicate_world` existed for; both are gone.
It answered a different question — "what does this scene do" rather than "what
does the game do" — and answering the first badly is worse than not answering
it. Editing the startup scene and pressing play is the whole of it today.

**Pause and step.** `World::set_simulating` is the mechanism and a pause
button would be one line, but the frame it steps is a frame of *everything* —
there is no per-world `dt` to hold.

**Input arbitration.** The game and the editor's `OrbitController` read the
same `Input`, so a drag in the viewport reaches both. Deciding who gets a
press is the [gizmo](gizmo.md)'s problem too, and worth solving once for all
three rather than by adding a play-mode special case.

**Surviving a panic in game code.** Per-call `catch_unwind`
([ADR-0010](../ADR-0010-scene-authoring-and-play.md) §7) is what makes stop a
safety net instead of a hope; until it is built, a component that panics takes
the editor with it.

One frame flashes on the toggle: the world set and the camera set both
changed, so the camera and the FrameSlot primaries are rebuilt, which is the
same one-frame cost a divider drag already pays.
