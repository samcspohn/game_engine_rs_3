# The TRS gizmo and the world grid

Two pieces of editor chrome that live in the renderer because they need a
depth buffer and a camera matrix, and neither exists anywhere else.

- [`crates/engine-render/src/overlay.rs`](../../crates/engine-render/src/overlay.rs) — the render path both use.
- [`crates/engine-render/src/gizmo.rs`](../../crates/engine-render/src/gizmo.rs) — hit testing, the drag maths, the triangles.
- Shaders: `grid.vert` / `grid.frag`, `overlay.vert` / `overlay.frag`.

## The overlay pass

One `begin_rendering` scope per camera, after both scene passes and before
the present blit, loading the colour and depth those passes left. Two
pipelines run inside it:

| | vertex input | depth | why |
|---|---|---|---|
| grid | none — a fullscreen triangle from `gl_VertexIndex` | tests `Less`, writes none | the scene occludes the ground, the gizmo is not occluded by the ground |
| gizmo | `OverlayVertex` (world position + linear RGBA) | none at all | a handle buried in the mesh it moves is a handle you cannot grab |

It is its own scope rather than a secondary appended to pass 2's, because
pass 2 is skipped entirely when occlusion culling is off (F8). Pass 2's
depth `store_op` changed from `DontCare` to `Store` to feed it — free on a
desktop GPU, where the depth attachment is a real image either way.

The primary command buffer is recorded once and replayed, so nothing here
may change a *command*, only the bytes a command reads:

- geometry goes into a host-visible vertex buffer of fixed capacity
  (8192 vertices; three rotate rings, the worst case, is ~2300),
- both draws are `vkCmdDrawIndirect` with host-written counts, so an empty
  overlay is two zero-instance draws rather than a re-record,
- the camera block (`view_proj`, its inverse, the eye, the grid
  parameters) is a UBO rather than a push constant, for the same reason,
- all three are double-buffered by staging slot, like every other
  host-visible buffer in the frame, and written under the same compute
  wait.

Only the viewport is baked into the secondary, so a camera resize is the
one thing that re-records it.

## The grid

`grid.frag` intersects the ray through each pixel with `y = 0` itself and
writes `gl_FragDepth` from the hit, so the grid is unbounded with no
geometry to bound and still sorts correctly against the scene. Line
coverage is `1 - min(|fract(uv - ½) - ½| / fwidth(uv), 1)` at two cell
sizes (minor, and `major` times it), each faded out by
`1 - max(fwidth(uv))` — without that term a grid seen edge-on turns into
solid fill instead of dissolving. The two world axes through the origin
take the gizmo's own red and blue.

`CameraHandle::set_grid(Some([cell, major, fade_radius, _]))` turns it on;
a game's camera leaves it `None` and draws nothing.

## The gizmo

### Who runs it

`gizmo::update()` runs in the renderer between the UI's pointer update and
`worlds::sweep_all` — not as a component. `OrbitController` and the gizmo
answer to the same press, and a component running in the sweep would race
it: whichever ran second would find the other had already decided. Running
before the sweep means `gizmo::captures_pointer()` is settled by the time
any component looks at it.

The editor only says *what* to aim at:

```rust
gizmo::set_target(camera, world_id, Some(entity));  // per camera slot
gizmo::set_mode(GizmoMode::Rotate);                 // W / E / R in the editor
```

`OrbitController` defers on the *drag* only. The wheel still zooms with the
cursor over a handle, which is where it sits for most of an edit.

### Frames

Translate and rotate handles are world-axis-aligned; scale handles are the
entity's own axes, because a scale has no meaning in any other frame.

### Picking

Screen space for anything long and thin — the cursor's distance in pixels
to an axis's projected segment, or to a ring sampled as a 32-chord
polyline — because a world-space threshold is a different size on screen at
every distance. Plane handles are the exception: ray-plane, then a bounds
test in the plane's own two axes, which is exact and simpler than
projecting a quad. Planes are tested first; they sit between two axes and
would otherwise never win.

### Dragging

Every anchor is taken at the press (`origin`, `axes`, the parent basis, the
local TRS, and the handle's own parameter) and the transform is written as
an absolute function of them. A drag that fed its own output back would
accumulate drift from every rounding in the chain.

Per handle, the parameter is:

- **axis** — the point on the axis closest to the cursor ray. Translation
  moves by the difference; scale multiplies by `1 + Δ/size`.
- **plane** — the ray's intersection with it; translation moves by the
  difference.
- **ring** — the angle of that intersection around the ring.
- **uniform** — the cursor's distance from the centre on the plane facing
  the ray, so it behaves the same from any angle.

### Getting the answer back into a local transform

A transform stores local TRS, and `get_global_position` composes a parent
chain as `v ↦ p + (r · v) · s` per level — scale *after* rotation, which is
not the same as any single `Mat4` you might be tempted to build.
`parent_basis` composes that same map over the three basis vectors, giving
a `Mat3` whose inverse turns a world-space delta into a local one. A world
turn becomes a local one by conjugating with the parent's rotation, since
`set_rotation` writes the value that composes in the parent's frame.

With no parent both are the identity, which is the case the editor is in
today — but a glTF import is a deep chain of rotated, scaled nodes, and
this is what stops the handle and the object parting ways there.

## The inspector's transform section

The gizmo's other half. `InspectorPanel` shows the entity's TRS above every
reflected component, and the two write the same three values — drag a handle
and the fields move, type into a field and the gizmo follows.

It is the one section not read through `Export`, because the hierarchy owns
a transform and the component registry does not: `trs_values` / `set_trs`
read and write it directly, keyed on the `"Transform"` type name the rows
carry. Everything else about the row is shared, so the rest of the panel
does not know the difference.

Rows are built per `ValueKind` rather than per property: `arity` says how
many fields a kind takes (three for a `Vec3` or a `Quat`, four for a
`Color`, none for an asset or entity reference), `parts` fills them and
`assemble` reads them back. That is what makes a game component's own
`Vec3` editable too, with no editor change. A rotation is Euler degrees;
the round trip is not the identity (180° comes back as −180°), which is
survivable only because the read-back already leaves a focused field alone.

The fields `flex_grow` into whatever width the panel has instead of taking
a fixed 90 px each — three of them have to fit the column one used to.

### Dragging a number

Drag a numeric field sideways and the value follows the cursor. The split is
the point: the UI owns the *gesture*, the panel owns what it means.

`Scrub` ([`ui/tree.rs`](../../crates/engine-render/src/ui/tree.rs)) sits beside
`Drag` and works on any node, not just a field:

```rust
if let Some(v) = self.scrub.update(ui, node, step, current) {
    // `current` is read at the press and ignored after
}
```

It holds one thing — the anchor — because that is the only part a caller
cannot get from `ui.drag(n)`. The value is `from + dx * step`, absolute for
the same reason the gizmo's drags are: a gesture that integrates its own
output drifts away from the cursor, and the test asserts it by handing
`update` a *different* `current` after the press.

The step, the format and the commit are the inspector's. Rotation moves
0.25° per pixel and a length 0.01, because a degree is a smaller thing than a
metre; `fmt` writes the number, and it is the same `fmt` the read-back uses,
so a dragged value is never shown at a precision it will not be re-read at.
The panel commits every frame the drag moves, unlike a keystroke, which waits
for `Enter`: a drag *is* the edit, and waiting for the release would leave the
viewport a frame behind the number.

Nothing declares that a field is draggable, because nothing needs to. There
is one pointer and therefore one gesture, so `Scrub` calls `ui.claim_drag(n)`
when it takes it, and `drive_controls` skips the selection for a node whose
drag is spoken for. A field is draggable exactly when something is reading
its drag — the two cannot drift apart, because they are the same fact.

The frame ordering is what makes that safe. `drive_controls` runs inside
`update_pointer`, before any component, so a claim always lands *after* the
frame's selection would have. It works anyway because of which frame does
what: the press frame only places a caret (`extend` is false), and the claim
goes in at the end of it, so the second frame — the first that could select —
already sees it. That is also why `Scrub` claims as soon as it sees a drag
rather than when it arms past the slop: claiming on arm would leak a visible
selection through the first few pixels of a slow drag.

The claim is cleared by the next press rather than by the release, since the
release frame runs `drive_controls` before the component too.

A field with no anchor — text that is not a number, or a value half-typed —
claims nothing, so its drag stays an ordinary selection.

The second click of a double click takes the whole value instead of placing a
caret, so typing over a number replaces it.

### What `Enter` means

The read-back leaves a focused field alone so it cannot delete what someone
is halfway through typing. `Enter` is where that ends: the value is finished,
so it is committed and then written back in the panel's format — `3` becomes
`3.000` on the spot rather than at blur, and a setter that clamps or refuses
shows its answer while the field is still in front of you.

Focus stays, and the value is left selected, so a second number can be typed
straight over it. `Tab` already means "the next field" and `Escape` "give the
keyboard back"; `Enter` moving focus would duplicate one and reformatting
without focus would duplicate the other.

## Driving it from `tools/poke`

`poke key <letter>` now injects the physical key beside the keystroke
(pressed one frame, released the next), so tool shortcuts are drivable and
not just text fields. Handles are aimed at by coordinate:

```sh
tools/poke click "cube"          # select it in the hierarchy
tools/poke key e                 # rotate mode
tools/poke drag 297,240 --to 250,290
```
