# ADR-0011 — Worlds: a hierarchy and a registry per scene

**Status:** Accepted; build order steps 1–4 built. §3 was revised after step
2 — see the note at the end of it. §4's *shared staging arena* and §5's
*several worlds into one viewport* are the two pieces still open.
**Related:** [ADR-0009](ADR-0009-hierarchy-root-entity.md) (`ROOT` and
`parent: None`, both of which this simplifies),
[ADR-0010](ADR-0010-scene-authoring-and-play.md) (§4 documents-as-subtrees, §5
edit vs. play, §6 the activation bitset this supersedes),
[ADR-0005](ADR-0005-dual-pass-occlusion-culling.md) (the per-camera state that
makes a second viewport expensive),
[`docs/notes/editor-document-split.md`](notes/editor-document-split.md).

## Context

The editor edits one document at a time because that is all the engine can
represent. Two viewports onto two scenes is the case that breaks, and it
breaks in four places at once: `ACTIVE_CAMERA` is one global
(`scene.rs:167`), `VIEWPORT` is one global (`scene.rs:48`), there is one
`RenderCamera` (`lib.rs:1166`), and `mvp_build.comp` culls the entire
`GPURenderers` buffer with no notion of which scene a slot belongs to
(`mvp_build.comp:356`).

Two attempts have already been made at the "which scene is this entity in"
half of the problem, and both were per-entity filters:

1. **An `enabled` bitset** (ADR-0010 §6) ANDed into `par_iter`'s word load,
   with `NO_RENDERER` scattered on toggle. Built, then removed.
2. **A `WorldId` per slot** with a `ComponentRegistry` per world, so an edited
   scene is a registry nobody sweeps rather than entities nobody dispatches.
   Built; the registry half survives, the per-slot half is what this ADR
   removes.

The bitset failed on a structural point worth keeping: visibility is a
property of the **(camera, entity) pair**, and a per-entity bit has one value
per frame. It cannot say "visible to viewport A, not to viewport B" about the
same entity at the same time. Neither can any refinement of it.

The registry split was right and is kept. It just stopped one level short.

## Decision

### 1. A world owns a hierarchy *and* a registry

```rust
struct World { hierarchy: TransformHierarchy, registry: ComponentRegistry, simulating: bool }
```

Worlds are engine-owned and refcounted — §3.

The argument that settles it is buffer sizing, not tidiness. With one shared
hierarchy, a world's slots are interleaved with every other world's across the
whole index range, so **every per-camera GPU buffer must be sized to the
global slot count no matter how it filters** —
`WorldTransformGpu::ensure_capacity` takes one number, and the cull dispatches
`ceil(total / 64)` workgroups. Two 1M-entity documents means each camera walks
2M slots to draw 1M. Per-world hierarchies make each slot space dense and
zero-based: buffers and dispatches are proportional to the world. Filtering is
`O(cameras × total)`; disjoint buffers are `O(total)`, once.

This is the same principle that moved registries per world — do not visit what
you are not drawing — applied where the cost is larger, because the cull
dispatch spans the slot range and the component sweep only spans what exists.

Three things fall out that were not the motivation:

* **Stop-play stops leaking.** `avail` is push-only and `create_transform`
  never pops it (see the document-split note), so in a shared hierarchy every
  play→stop cycle burns its entity count *permanently* and the SoT, sized off
  `hierarchy.len()`, ratchets up for the session. A world that owns its
  hierarchy drops the whole allocation.
* **`scene_root` disappears.** It exists only because documents share a graph
  with the editor's rig. If a document *is* a hierarchy then `parent: None`
  means its own `ROOT` — what a game already gets — and the `spawn_subscene`
  drain-time hazard stops being expressible.
* Document entities stop paying the extra level in the GPU parent walk.

### 2. `Entity` is an index. The world comes from the accessor

Not `Entity { world, id }`. An entity id is a slot index and is meaningless
without its container, the same way a `usize` is meaningless without its
`Vec`. The codebase already commits to this one level down: `Transform<'a>` is
`{ hierarchy: &'a TransformHierarchy, idx: u32 }` — an index paired with its
container *at the point of use*, never a self-describing handle.

```rust
world.entity(e).get_component(|c: &mut Spinner| c.speed = 2.0);
```

A closure rather than a returned `&Mutex<T>`: the lock stays inside the world,
the borrow does not escape, and it matches `inspect`, which is already
closure-shaped for the same reason.

Two consequences of not tagging:

* **`Value::Entity` gets stricter, not looser.** A tagged entity written to a
  scene file encodes a reference to a world that may not exist on load — the
  same class of hazard as resolving `parent: None` at drain time. A bare index
  can only mean "in this component's own world", which is the only
  cross-reference a document should be able to express.
* **It sorts correctly against generations.** ADR-0010's third caveat says
  slot recycling will need generation tags. A generation is intrinsic to the
  slot and belongs in the handle; a world is the container and does not. The
  rule discriminates between them rather than collapsing both into a fat id.

### 3. `update` is handed its own world. Worlds are engine-owned

```rust
fn update(&mut self, dt: f32, transform: &Transform, world: &World)
```

Its own world, because that is the one a component is *in*. Reaching another
is ordinary, not an escape hatch: hold its `WorldHandle`.

```rust
let overlay = engine::new_world();
for mesh in transform.get_children() {
    overlay.duplicate(world.entity(Entity::new(*mesh)), |mut e| {
        e.get_component::<MeshRenderer>(|m| m.set_material(ghost));
    });
}
```

`engine::new_world()` registers a world and hands back a handle; the world
lives exactly as long as a handle to it does. A ghost-overlay component owns
its overlay world by holding the handle, and dropping the component ends the
world — no registration to undo, no id to invalidate. The frame sweeps
`worlds::live()`, so a world made mid-frame joins the next one on its own.

**Structural changes queue. `&mut World` is the frame boundary.**

Growing a hierarchy reallocates the SoA that a running sweep is reading, so
`spawn`, `duplicate` and `destroy` record the request and take a builder
callback, which `worlds::apply_pending` runs between frames once the slot
exists:

```rust
world.spawn(t, |mut e| { e.add_component(Spinner::new()); });
```

This is what keeps `positions_raw()` a contiguous slice, `create_transform`
plainly `&mut self`, and the whole frame path free of a lock — the sweep only
ever needs `&World`, because components mutate through their own `Mutex<T>`
and the hierarchy through per-slot ones. The renderer's `drain_ready_spawns`
already had this shape; it is now the shape of every structural change.

`WorldHandle::get_mut` is the boundary's door and is `unsafe`: it aliases
every `&World` a sweep hands out, and is sound exactly where no sweep runs.

#### Superseded: the frame-scoped global, gated to the editor

This section first said the opposite — that reaching across worlds was the
editor's privilege, exposed only through `engine-editor-api`, and that the
list was an `AtomicPtr` published for the duration of the sweep so it could be
read without a lock. Both halves are gone:

* **The gate.** A component compositing a ghost overlay out of a second world
  is game code, not tooling. Making that the editor's privilege would have
  meant a game reaching for `engine-core` to get it — a facade-level boundary
  the ADR already admitted was not enforceable.
* **The frame-scoped publication.** It existed to answer "which worlds is the
  frame running", from a `&[World]` the app owned. Engine ownership answers
  that without publishing anything, and refcounting answers "for how long"
  better than a frame ever could. The re-entrancy hazard it was shaped around
  goes with it: the registry lock is taken to *snapshot* handles, never held
  across a sweep, so chrome calling `worlds::world(id)` from inside one is an
  ordinary uncontended acquire.

What survives unchanged is the signature: `update` takes `&World`, its own,
and anything wider is something the component went and got.

### 4. Per-world SoT, **shared** staging

The SoT is what the cull indexes, so it splits per world — that is the point
of §1. Staging does not. The host→device path is the tuned one: SDMA-bound
around 4.35 GB/s, an adaptive HOST_CACHED-vs-WC balancer, NUMA-confined
workers. Splitting it fragments the transfer into small ones with worse SDMA
efficiency, and gives N independent balancers each making policy against a
shared queue.

One staging arena with per-world regions keeps transfers contiguous and the
policy singular while the SoT stays disjoint. The TRS scatter gains an outer
loop over worlds; the dirty harvest already costs nothing for a world whose
transforms did not move, which is every non-simulating document.

Asset stores (`gpu_mesh_store`, texture, material) stay **global**. A mesh
used by two documents is one upload.

### 5. A viewport owns attachments; worlds contribute draws

"Rendering a world is disjoint from rendering another" holds for *inputs*, not
outputs. The editor's scene view wants a document world and a
gizmo/grid/selection world composited into one image against one depth buffer,
correctly interleaved — the normal case, not an exception.

So the viewport owns the colour + depth attachments and the camera, and
rendering it is a sequence of per-world draw passes into them. N sets of
inputs, one output. Renderers are never filtered; a world's renderers are
simply in a different buffer, and a viewport draws the worlds it lists.

`ACTIVE_CAMERA` and `VIEWPORT` are gone; a `CameraHandle` carries its own
box, so a controller asks its own camera whether a point is over it. The
remaining step is a camera holding a *list* of worlds rather than one, at
which point the render loop nests camera-outer / world-inner with `Clear`
then `Load`.

### 6. Crossing worlds is copy-and-delete, not a move

Dragging an object from document A into document B would otherwise mean: build
transforms in B, copy TRS, re-link parents, destroy in A, migrate components,
and invalidate every outstanding handle into the subtree. Framing it as
cut-and-paste — a new identity in B — makes the problem go away, and is what a
user means by it anyway.

This deletes `set_world`, `drain_world_moves` and the type-erased `move_slot`
migration, which exist only to compensate for the shared hierarchy.
`Scene::instantiate` goes back toward hierarchy-to-hierarchy, which is what it
was before the registry split.

## Consequences

### Wins

* Two viewports onto two scenes becomes representable, with no per-entity
  filter anywhere in the CPU sweep or the cull kernel.
* GPU buffers and cull dispatches size to the world, not to the process.
* Stop-play frees its allocation instead of leaking slots for the session.
* `scene_root` and its drain-time hazard are deleted, not documented.
* ADR-0010 §8's process boundary gets a real seam: a world already owns
  everything a separate process would.

### Costs

* Every API taking `(scene, entity)` becomes `(world, entity)`. Mechanical,
  and touches almost everything.
* Anything holding an entity across time must hold its world beside it —
  `HierarchyPanel::selected`, `SubsceneInstance::nodes`, the inspector's
  `shown`. Each already knows which document it points at, so this is
  discipline rather than work, but it is a rule a tagged handle would have
  enforced for free. **This is the one place §2 is a real trade, not a free
  win.**
* `EntityRef`, the editor's drag payload, *does* carry `{ world, id }` — a UI
  payload crosses contexts by nature, so a cross-world drop is rejected on
  arrival rather than silently landing on whatever slot shares that index.
  The engine type stays bare; the editor's does not.
* N `WorldTransformGpu` SoTs, each with scatter pipelines and pre-recorded
  secondaries.
* The world registry is a process-global, so a test that makes a world shares
  it with every other test in the binary. Handles rather than ids keep that
  invisible: a test sweeps the worlds it holds, never `live()`.
* **Settled: `Scene` dissolved into `World`** *(built)*. `Scene` is gone and a
  world is the top-level object. It is reached by handle after all —
  `create_transform` still needs `&mut`, but §3's queue means only the frame
  boundary ever asks for it, so the handle costs the hierarchy neither a lock
  nor interior mutability.
* **Building an entity is two steps, not one.** `new_entity` then
  `add_component` becomes `spawn(t, |e| …)`, because the entity does not exist
  until the boundary. Reads better for a whole entity; more awkward when
  something outside the callback wants the id, which now has to be carried out
  of it.
* Many small worlds means many small `par_iter` dispatches, each with pool
  overhead. Fine at 2–3; a reason not to make worlds cheap enough to sprinkle.

### What this supersedes

ADR-0010 §6 is already marked superseded by the registry split. This finishes
that: the `WorldId`-per-slot plumbing added there is transitional and is
removed here. Roughly two thirds of that change (registry per world,
`simulating`, `Scene::update` skipping non-simulating worlds) carries over
unchanged.

Per-entity activation — hiding one object, ADR-0010 §6's original subject —
remains unbuilt and is now clearly a separate feature rather than a
half-measure toward this one.

## Caveats

* A second **viewport** is expensive regardless: ADR-0005's per-camera hi-Z
  pyramid and last-frame visibility duplicate per viewport. Worlds make that
  cost proportional to what is in view; they do not remove it.
* `Entity` remains a bare `u32` while slot recycling is off (ADR-0009's
  caveat). When recycling lands it gains a generation — *not* a world.
* Two cameras onto the **same** world still fight: `view_proj` lives in the
  world's SoT, and the first camera pointed at a world wins that slot. The
  camera now owns its matrix on the host side, so what is left is moving
  `sot_view_proj` off the world onto the camera — a small change made at the
  point there is a reason to. `cull_view_proj` is already camera-owned and is
  the template.
* A camera driven directly (the editor's, via `OrbitController::for_camera`)
  builds its matrix in the sweep, one frame before the renderer publishes the
  aspect of a target it just resized. A divider drag therefore renders one
  frame at the previous aspect. Cameras driven by a `CameraComponent` have no
  such skew — the post-frame pass runs after the resolution sync.
* The draw plan is still global — per-mesh instance totals across every world
  — so each camera's MVP and indirect buffers are sized to the process rather
  than to the world it draws. Correct, and over-allocated by the ratio between
  the two. Per-world plans are the same change as per-world SoT, one level up.
* GPU per-stage timestamps are written by camera 0 only, so the q2..q6
  readout means "what the first camera cost", not the frame's total raster.

## Build order

Introduce the seam, then move the wall:

1. ~~**`World` as an accessor**~~ *(built)* over the current shared hierarchy —
   `world.entity(e).…` at every call site, `update` taking `&World`, storage
   unchanged. The `WorldId`-per-slot plumbing is what makes this transitional
   state work. The global and its `engine-editor-api` export land here too,
   since the editor's inspector is what needs them.

   `World<'a>` is `{ scene, id }` and `EntityView<'a>` is `{ world, id }`, so
   the lifetime is elided in `fn update(&mut self, …, world: &World)` and step
   2 — which makes `World` the owned type — changes no `Component` impl.
   `worlds::publish` is an `AtomicPtr` set for the duration of the sweep, so
   `world(id)` is `None` outside a frame rather than stale. *(Both the global's
   shape and its editor-only export were revised after step 2 — see §3.)*
2. ~~**Hierarchy per world.**~~ *(built)* Deleted that plumbing, `scene_root`,
   `set_world` / `drain_world_moves` / `move_slot`, and `Scene` itself.
   `TransformHierarchy::new(id)` carries one `WorldId` for the whole graph, so
   `Transform::world()` is a field read. The sweep took `&[World]` and
   published it; §3's revision made that `worlds::sweep_all(&[WorldHandle])`
   over the engine's own registry.

   The renderer is not split yet, so **only one world is drawn**: one SoT, one
   `GPURenderers` buffer. `Window::with_world` draws the first world it is
   given and keeps the rest alive; spawn records are world-tagged and another
   world's are dropped at the ingest rather than landing on whatever slot
   shares the index; `ACTIVE_CAMERA` is `(WorldId, Entity)` — the editor's
   camera is in the rig's world and looks at the document's. Step 3 removes
   the restriction.
3. ~~**Per-world SoT**~~ *(built)*, outer loop in the TRS scatter. Each world
   owns a `WorldTransformGpu` and a `GpuRenderers`; the FrameSlot primary
   records one scatter block per world. What they share is
   `TransformGpuShared` — the six compute pipelines, the staging allocator,
   and `gpu_signal`, which is the *frame's* gate: one `signal_cs` after every
   world's scatter, so the host still wakes once.

   **The staging arena is not shared yet.** Each world has its own staging
   slots rather than a region of one arena, so the SDMA transfer is one per
   world instead of one contiguous upload, and the balancer switches all of
   them together or none. Right at 2–3 worlds, wrong at 20.

   Spawn records are no longer world-filtered-and-dropped: `drain_spawns`
   groups the queue by world and each world scatters its own.
4. ~~**Per-camera box, attachments and matrix**~~ *(built)*. A `CameraHandle`
   is the whole surface: the world it draws, its `view_proj`, its projection,
   and the box the panel showing it published. `RenderCamera` holds the same
   `Arc`, so the device half and whoever drives the camera are one object.
   Two writers, never both on one camera: `CameraComponent` mints a camera
   bound to the world it was spawned in and a post-frame pass feeds it that
   entity's settled pose; `OrbitController::for_camera` writes the matrix
   itself, which is safe only because that state is the controller's alone.
   Nothing writes a camera from inside the sweep on data other components
   share — component order there is nondeterministic, so a matrix built
   mid-sweep would race the transform writes the scatter is about to upload.
   `ui::Viewport` writes its box onto the camera; `MAX_CAMERAS` bindless
   slots are reserved. `engine` exports `CameraComponent` and nothing else;
   owning a camera outright is `engine-editor-api`'s `CameraHandle`.
5. **Multiple worlds per viewport** — the gizmo-over-document composite.

Steps 1–2 are the bulk and are CPU-only. The editor now shows two documents
side by side; its own gizmos are what need 5.

## Revisit if

* Worlds get cheap enough that code starts creating them per-object, at which
  point the per-world dispatch overhead in §4 stops being noise.
* Two viewports onto one world turns out to be the common case, which would
  argue for splitting camera from world more sharply than §5 does.
* Cross-world entity references acquire a real use case, which would reopen
  §2 — though a stable name, not a slot index, is the likelier answer.
* ~~A second set of worlds is ever wanted in one process~~ — this happened
  first, and §3's revision is the answer: worlds are independent values held
  by handle, so a headless simulation beside the editor is several worlds, not
  a second list of them.
* Worlds start being made per frame rather than per component, at which point
  registry-slot reuse (a dropped world's id goes to the next one made) needs a
  generation the way entity slots will.
