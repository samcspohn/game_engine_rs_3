# ADR-0011 — Worlds: a hierarchy and a registry per scene

**Status:** Proposed.
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

The list of them is a process-global, gated to the editor — §3.

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

### 3. `update` is handed its own world. The list of worlds is a global

```rust
fn update(&mut self, dt: f32, transform: &Transform, world: &World)
```

Nothing wider. A game only ever wants the world it is in, and a signature that
carries an escape hatch teaches every reader that reaching across is a normal
thing to do.

The editor does need to reach across — `Chrome` lives in the editor's world
and inspects a document in another — but that is an **ambient capability**,
not a parameter. It goes where the engine's other ambient capabilities
already are: a process-global beside `asset::global()`, `material::global()`,
`ui()` and the renderer's spawn queue. The editor's chrome already reaches
`ui()` from inside its own `update`; reaching the world list is the same move.

Two constraints on that global, both load-bearing:

**No outer lock during a frame.** `Chrome::update` runs *inside* the sweep
over the worlds. If the sweep holds a lock on the list and chrome takes it to
reach a document, that is a re-entrant acquire on a `parking_lot::Mutex`,
which does not recurse — the editor deadlocks against the loop running it.
The list is therefore stable for the duration of a frame and read without a
lock, exactly as `TransformHierarchy` is: `Component::update` needs only
`&World` (the registry sweeps through `&self`; components mutate through their
own `Mutex<T>`), so nothing in the frame wants `&mut`. Creating, dropping or
re-ordering worlds is a structural change and queues, drained between frames —
the same shape as `spawn_subscene` and the renderer's spawn queue.

**Exposed only through `engine-editor-api`.** A game reaching into another
world is not a use case; it is the editor's privilege, and the workspace
already has a crate for exactly that. `engine` does not re-export it.

Note what that gate is and is not. The worlds live in `engine-core`, which
every game depends on, so this is a **facade-level** boundary enforced by
`engine`'s re-export list — not the dependency-graph guarantee the Readme
claims for `engine_editor_api` ("if the game doesn't depend on it, the symbols
don't exist"). A game that adds `engine-core` directly can still reach it.
That is worth having and worth not overstating; a Cargo feature would be
worse, for the reason the Readme already gives.

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

`ACTIVE_CAMERA` and `VIEWPORT` become per-widget state, and `in_viewport(p)`
becomes "which viewport is this point in" so a controller can ask about its
own.

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
* A process-global world list means the running app has exactly one set of
  worlds. `World` and `TransformHierarchy` stay ordinary owned types a test
  can construct directly, so only the tests that exercise *cross-world* lookup
  have to serialise on it — the pattern `thread_pool::lock_for_test` and the
  renderer's spawn-queue tests already use.
* **Open:** what remains of `Scene`. If the worlds are global it is either
  gone — `Window::with_scene` becomes "which worlds to run" — or a thin handle
  over the global. Worth settling before step 1, because it decides what
  `main` in a game looks like.
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
* Nothing here addresses two viewports onto the **same** world, which is a
  second camera over one set of inputs and should stay that simple.

## Build order

Introduce the seam, then move the wall:

1. **`World` as an accessor** over the current shared hierarchy —
   `world.entity(e).…` at every call site, `update` taking `&World`, storage
   unchanged. The `WorldId`-per-slot plumbing is what makes this transitional
   state work. The global and its `engine-editor-api` export land here too,
   since the editor's inspector is what needs them.
2. **Hierarchy per world.** Deletes that plumbing, `scene_root`, and the
   migration path.
3. **Per-world SoT + shared staging arena**, outer loop in the TRS scatter.
4. **Per-viewport camera, box and attachments**; `in_viewport` returns which.
5. **Multiple worlds per viewport** — the gizmo-over-document composite.

Steps 1–2 are the bulk and are CPU-only. A viewport showing one world works
after 4; the editor's own gizmos are what need 5.

## Revisit if

* Worlds get cheap enough that code starts creating them per-object, at which
  point the per-world dispatch overhead in §4 stops being noise.
* Two viewports onto one world turns out to be the common case, which would
  argue for splitting camera from world more sharply than §5 does.
* Cross-world entity references acquire a real use case, which would reopen
  §2 — though a stable name, not a slot index, is the likelier answer.
* A second set of worlds is ever wanted in one process (a headless simulation
  beside the editor, a test harness running two projects), which is the one
  thing §3's global forecloses and `Ctx` would not have.
