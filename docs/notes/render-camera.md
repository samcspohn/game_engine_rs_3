# Cameras, render targets and the viewport

A camera owns its attachments; a UI panel owns their size. How those two
meet, and who is allowed to write a camera's matrix.

## Render targets and matrices

A [`RenderCamera`](../../crates/engine-render/src/camera.rs) owns the
offscreen color image (`R16G16B16A16_SFLOAT`, `COLOR_ATTACHMENT \|
TRANSFER_SRC`) and the depth image (`D32_SFLOAT`, also `SAMPLED` for the Hi-Z
build). Since [ADR-0011](../ADR-0011-worlds.md) step 5, the per-world half of
that lives in a `WorldDraw` — one per world the camera composites — while the
attachments, Hi-Z pyramids and camera block stay camera-level. Since
[ADR-0005](../ADR-0005-dual-pass-occlusion-culling.md) landed, each
`WorldDraw` owns **two full, independent copies** of the compacted-output
buffers (device-local MVP storage buffer, per-instance material buffer,
per-instance packed world-TRS buffer, graphics descriptor set, host
indirect-command template + device indirect-args) — one for pass 1's cull
(draws instances visible against last frame's Hi-Z), one for pass 2's (draws
instances only pass 1's occlusion sub-test deferred, confirmed against this
frame's own Hi-Z) — plus a candidate record list pass 1 appends to and pass 2
consumes. Camera-level: two Hi-Z pyramids (`hiz_current`/`hiz_prev`,
`R32_SFLOAT`, full mip chain, fixed identities mutated in place each frame
rather than swapped), the camera block (`view_proj` + eye, promoted from host
staging every frame), a `prev_view_proj` history buffer, and the cull-test
`view_proj` the debug frustum-lock freezes. The swapchain image is never used
as a color attachment. Each camera carries a `CameraResolution` policy
(`MatchSwapchain` today; `Fixed { w, h }` and `ScaleSwapchain { num, denom }`
reserved for shadow maps / half-res reflections / editor thumbnails).
Invalidation now splits along two axes: **extent-dependent** (attachments,
both Hi-Z pyramids, and everything that binds their views — rebuilt by
`on_swapchain_resize`) and **capacity-dependent** (both passes' MVP/indirect
buffers, the candidate list, the cull sets — rebuilt by `ensure_current`,
geometric ≥ 2× growth), plus the pre-existing **per-world capacity**
(SoT/GPURenderers/redirect/mesh_table reallocation forces `force_full`) and
**per-frame-in-flight** (`FrameSlot`) axes. See ADR-0005 for the full
rebuild-scope breakdown.

## Controller

Built-in [`OrbitController`](../../crates/engine-render/src/scene.rs) moves
its entity's transform each frame. Left-button drag orbits, right-button drag
pans, scroll zooms; pitch is clamped to avoid the gimbal flip and distance to
a non-zero minimum. Gestures are bounded by the camera's own box, so a drag in
one panel leaves the camera beside it alone. Attaching a `CameraComponent`
mints a camera on the world the entity was spawned in and a post-frame pass
feeds it that entity's pose; `OrbitController::for_camera` skips the component
and drives a camera the app owns. `camera.draw_world(id)` adds a world to what
that camera composites — gizmos over a document, sharing one depth buffer.

Naming a world is also what makes it *drawn*: a world minted after `run` —
the editor's new-scene document, a game's overlay — gets its SoT and its
`GPURenderers` the frame a camera first names it, which forces the same frame
slot + camera rebuild a capacity grow does. `Window::with_world` is only what
keeps a world alive plus the one a camera naming nothing falls back to, so a
document made from a menu draws its own contents rather than the first
world's.

## Camera in a panel

A [`ui::Viewport`](../../crates/engine-render/src/ui/viewport.rs) is a node
that shows a camera **and sizes one**. It publishes its own box every frame;
the renderer adopts it as `CameraResolution::Fixed`, re-allocating the colour
/ depth / Hi-Z attachments, their descriptor sets, the extent-shaped
secondaries and the frame slots — the same rebuild a window resize does, asked
for by the layout instead of by the compositor. So the scene is rendered *at*
the size it is shown at: the hardware viewport covers the whole target, the
projection is a plain `view_proj(aspect)` with no skew, and the panel samples
it one texel per pixel. The target is bound as a texture at a reserved slot in
the UI's *own* copy of the bindless array (`assets::RESERVED_SLOTS`); the
scene pipeline's copy must not have it, or the camera's own attachment would
be a sampled image inside the render pass that writes it. Colour needs no
conversion — the target is `R16G16B16A16_SFLOAT` and the UI writes linear
premultiplied into an `_SRGB` swapchain, exactly what the blit's format
conversion did. **A camera this size cannot be blitted**: it is not the
swapchain's shape, so `build_frame_slot` drops the present-blit and the UI
pass clears instead of loading. That is the whole compositing decision, and it
is one predicate — the camera paints the swapchain for a game, the widget does
for an editor. `OrbitController` gates on the same published box (latched at
the press, so a drag that leaves the panel keeps orbiting) alongside
`pointer_captured`, which is why dragging in the console does not spin the
scene. Cost: a resize frame measured ~0.8 ms over a 0.35 ms baseline, i.e. a
divider drag, not a steady state.

## Editor overlay

One `begin_rendering` scope per camera after both scene passes
([`overlay.rs`](../../crates/engine-render/src/overlay.rs)), loading the
colour and depth they left. Two pipelines: the **world grid**, a fullscreen
triangle whose fragment stage intersects `y = 0` per pixel and writes its own
`gl_FragDepth` (unbounded with no geometry to bound, still occluded by the
scene), and the **TRS gizmo**, host-built world-space triangles with no depth
test at all. Both draw indirectly out of host-written,
staging-slot-double-buffered buffers, because the primary CB is recorded once
and replayed — only a camera resize re-records the secondary. A game that sets
no grid and no gizmo target pays two zero-instance draws. See
[`docs/notes/gizmo.md`](../notes/gizmo.md).


- [ADR-0011 worlds](../ADR-0011-worlds.md)
- [gizmo](gizmo.md)
