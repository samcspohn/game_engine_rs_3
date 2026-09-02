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

## Driving it from `tools/poke`

`poke key <letter>` now injects the physical key beside the keystroke
(pressed one frame, released the next), so tool shortcuts are drivable and
not just text fields. Handles are aimed at by coordinate:

```sh
tools/poke click "cube"          # select it in the hierarchy
tools/poke key e                 # rotate mode
tools/poke drag 297,240 --to 250,290
```
