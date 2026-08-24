# Do per-world scatter dispatches overlap?

**Answer: no, and the cost of not overlapping is ~11 µs per world, flat.**
ADR-0011 §4 keeps staging per world; this measures what that costs on the GPU
side, separately from the SDMA fragmentation §4 is mostly about.

## The question

`build_frame_slot` records one scatter secondary per world, back to back:

```rust
for wr in worlds {
    builder.execute_commands(wr.transforms.scatter_secondary(staging_slot).clone())
}
```

Worlds share no buffer, so vulkano's auto-sync emits no barrier between them —
`find_buffer_conflict` keys on the `Buffer` object and misses, and the
conservative first-use `ALL_COMMANDS` barrier (`auto/builder.rs:920`) only
lands when `pending_barrier` is flushed, which is at the `fill_buffer` *after*
the whole block. So the recorded stream really is N adjacent dispatch chains
with nothing separating them, and RADV emits no partial flush without one.

The hardware can therefore run them concurrently. The question is whether it
does, and whether that recovers the per-dispatch cost.

## The harness

`editor --stress N --worlds W` spawns `N` spinning cubes as one cubic grid cut
into `W` contiguous slabs, one world each, **all composited through a single
camera** — so the drawn entity count, the draw plan and the raster load are
identical at every `W` and only the scatter's dispatch count changes. Every
entity is dirty every frame (the `Spinner` rotates it), so the work is real.

The metric is the `trs by slot` line (GPU timestamps q8→q9), which brackets
exactly the per-world scatter secondaries. `dirty words by slot` is printed
per world and confirms the fairness: ×W it is ~31.2K in every run.

```sh
cargo build -p editor --release
ENGINE_NUM_THREADS=128 ./target/release/editor --stress 1000000 --worlds 8
```

## Results — RX 7900 XTX, 1M entities held constant

| worlds | entities/world | trs µs | scatter block µs | mvp1 µs | FPS |
|-------:|---------------:|-------:|-----------------:|--------:|----:|
|      1 |      1 000 000 |  322.8 |            343.4 |    43.7 | 2086 |
|      2 |        500 000 |  336.0 |            358.8 |    52.8 | 1550 |
|      4 |        250 000 |  362.0 |            387.1 |    67.1 | 1419 |
|      8 |        125 000 |  438.6 |            476.8 |   103.8 | 1089 |
|     16 |         62 500 |  580.2 |            641.1 |   178.5 |  686 |

Monotonically worse: +257 µs for the same work, ~17 µs per extra world.

## Isolating fixed cost from work

| worlds | total | entities/world | trs µs |
|-------:|------:|---------------:|-------:|
|      1 |    16 000 |     16 000 |   31.1 |
|     16 |    16 000 |      1 000 |  196.8 |
|     16 |   250 000 |     15 625 |  284.8 |
|     16 | 1 000 000 |     62 500 |  580.2 |

The first two rows do **identical** work — 16 000 entities, ~500 dirty words —
and differ by 166 µs, ~11 µs per extra world. That floor is independent of how
much work each world does; the last three rows are that floor plus work.

## Why — the barriers are inside the secondary, not between them

`record_scatter_secondary` is four dependent stages: `fill_buffer` → word
compaction prepass (×3) → `scatter_build_args` → TRS scatter (×3) + parent
scatter. Each arrow is a genuine RAW on the same buffer, so vulkano inserts a
barrier, and on AMD a barrier is a CS partial flush that drains **everything**
in flight, not only the buffers it names.

```
w1: [fill] ▮ [prepass ×3] ▮ [build_args] ▮ [scatter ×3][parent]
w2: [fill] ▮ [prepass ×3] ▮ [build_args] ▮ [scatter ×3][parent]     ▮ = full drain
```

`w2`'s prepass may start under `w1`'s tail, but `w2`'s own first drain then
waits for all of it. Three drains per world, 48 at `W=16`, and at 62 500
entities per world there is nothing left to hide them behind. `mvp1` — the
cull, a per-world secondary of the same shape — grows on the same curve
(43.7 → 178.5 µs), which is the second, independent confirmation.

So the dispatches are free to overlap and partly do; what does not overlap is
the barrier between each world's own stages.

## The fix: record stage-major, not world-major

Share the barriers instead of repeating them. `TransformGpuShared::record_scatter_secondary`
now records **one** secondary over every world, in stage order:

```
[all worlds' count resets] ▮ [all prepasses] ▮ [all build-args] ▮ [all scatters][all parents]
```

Three drains regardless of `W`, and same-stage dispatches from different
worlds — genuinely independent — sit adjacent with nothing between them.
Vulkano places the barriers correctly on its own: it accumulates each
collision into `pending_barrier` and flushes the whole set at
`first_unflushed`, which is the index of the first command in the stage, so
the flush before the first build-args orders *every* prepass ahead of *every*
build-args.

Hoisting the `fill_buffer(count, 0)` resets ahead of the prepasses is part of
the same idea and matters on its own: inline, each reset collides with its own
prepass and buys a barrier per component.

## Results — same harness, same machine

`trs` µs, 1M entities held constant:

| worlds | world-major | stage-major | Δ |
|-------:|------------:|------------:|-----:|
|      1 |       322.8 |       308.3 |  −14.5 |
|      2 |       336.0 |       305.3 |  −30.7 |
|      4 |       362.0 |       309.3 |  −52.7 |
|      8 |       438.6 |       320.1 | −118.6 |
|     16 |       580.2 |       336.3 | −243.8 |

**Per-world slope: ~17 µs → ~1.9 µs.** At 16 worlds the scatter is 42% faster
and FPS goes 686 → 715.

The overhead-dominated corners, where it is starkest:

| worlds | total | world-major | stage-major | ratio |
|-------:|------:|------------:|------------:|------:|
|      1 |    16 000 |  31.1 |  11.6 | 2.7× |
|     16 |    16 000 | 196.8 |  34.5 | 5.7× |
|     16 |   250 000 | 284.8 | 111.9 | 2.5× |

The single-world row is the reset hoist alone — two barriers removed with no
worlds involved at all.

What is left is real dispatch-launch cost: 16 worlds at 16 000 entities is
34.5 µs against one world's 11.6, so ~1.5 µs per extra world over its 8
dispatches, ~0.19 µs each. Overlap does happen now; there is simply a floor.

## Caveat

The scatter secondary is no longer per world, so it cannot be rebuilt by one
world in isolation — `build_all_frame_slots` re-records it, because the events
that invalidate a frame slot (a capacity grow, a staging re-home) are exactly
the ones that invalidate it.

## The same shape, again: the cull

With the scatter fixed, `mvp1` / `mvp2` were the wall — 283 / 138 µs at 16
worlds against 44 / 12 at one. Identical cause. Pass 1's cull was one
secondary per (camera, world):

```
w1: [copy template→args][fill candidate_count] ▮ [cull dispatch] ▮ [pass-2 args]
w2: [copy template→args][fill candidate_count] ▮ [cull dispatch] ▮ [pass-2 args]
```

Two drains per world per pass. `record_cull_secondary` and
`record_cull_pass2_secondary` now take `&[WorldDraw]` and record stage-major,
so both secondaries moved off `WorldDraw` onto `RenderCamera` — which is where
they belonged anyway: the camera is what owns the composite.

| worlds | mvp1 before | after | mvp2 before | after | gpu total before | after |
|-------:|------------:|------:|------------:|------:|-----------------:|------:|
|      1 |        43.7 |  43.7 |        11.6 |  11.6 |            426.4 | 424.1 |
|      2 |        53.5 |  45.6 |        15.6 |  11.8 |            440.0 | 427.3 |
|      4 |        79.5 |  59.4 |        38.6 |  21.3 |            517.8 | 509.5 |
|      8 |       150.9 |  76.3 |        64.8 |  29.2 |            677.0 | 601.7 |
|     16 |       283.1 |  88.5 |       137.6 |  33.1 |            981.3 | 680.1 |

**Per-world slope: mvp1 16.0 → 3.0 µs, mvp2 8.4 → 1.4 µs.** GPU total at 16
worlds is down 31%.

## Where the wall is now: the CPU

FPS barely moved (715 → 705 at 16 worlds) because the frame stopped being
GPU-bound. Per-frame µs at 16 worlds:

| worlds | ms/frame | gpu total | sim_update | host_staging |
|-------:|---------:|----------:|-----------:|-------------:|
|      1 |      555 |       424 |        302 |          121 |
|      4 |      692 |       510 |        339 |          204 |
|     16 |     1418 |       680 |        660 |          553 |
|        |     *(µs)* |         |            |              |

1418 µs of frame against 680 µs of GPU. `sim_update` and `host_staging` both
roughly double from 4 to 16 worlds on identical total work — the many-small-
`par_iter`-dispatches cost ADR-0011's Costs section predicted ("fine at 2–3;
a reason not to make worlds cheap enough to sprinkle"). That is the next
thing to attack, and it is a CPU-side problem, not a barrier one.

The remaining GPU-side per-world growth is `raster1`/`raster2` (11 → 55 and
2 → 45 µs). Those secondaries execute inside one `begin_rendering` scope where
barriers are illegal, so no drains are involved — it is per-draw-call bind
cost. **See the correction below: in this harness that is *all* it is, because
the scene draws 3–4 instances.**

## Per-world draw plans

Not a timing fix — an allocation one, and the reason it belongs here is that
it was the blocker on merging the raster draws. Every `WorldDraw` sized its
MVP and indirect buffers to `plan.total_renderers`, which was the whole
process, so 16 worlds each allocated room for all 1M instances to hold 62 500.

The tally could not come from the asset registry: a refcount is per `MeshId`
and has no world. `GpuRenderers` now keeps its own, folded from the spawn
stream it already receives per world, against a CPU mirror of each slot's mesh
word — the only thing that knows which id a record replaces. `WorldSource`
carries its world's plan, and the camera drops the global one entirely
(`drawCount` stays camera-level, since every world's plan spans the same
slots).

Process-wide totals left `GpuMeshStore::sync`'s return with the same change,
which also drops a per-frame `Vec<u32>` clone taken under the registry lock.

`self vram`, 1M entities:

| worlds | before | after |
|-------:|-------:|------:|
|      1 | 1680 MB | 1680 MB |
|      4 | 2716 MB | 1692 MB |
|     16 | 7370 MB | **1738 MB** |

Flat in world count, which is the whole point — and 1 world is unchanged,
because with one world the per-world plan *is* the global one. GPU timings
are identical (mvp1 88.5 → 88.4 µs at 16 worlds); nothing about the dispatch
count changed.

## The raster numbers above measure nothing being drawn

**Correction, and it invalidates every `raster1`/`raster2` figure in this
note.** A readback of the indirect commands' `instance_count` says the harness
draws **3–4 instances**, at every entity count and every world count:

| scene | instances drawn (pass 1 / pass 2) |
|---|---|
| `--stress 1000000 --worlds 1` | 4 / 0 |
| `--stress 1000000 --worlds 16` | 3 / 0 |
| `--stress 8000 --worlds 16` | 4 / 0 |
| `--stress 27 --worlds 3` *(control)* | 21 / 0 |
| default editor, 1 cube per document *(control)* | 1 / 0 per camera |

The controls match what is on screen, so the readings are sound. The cause is
the harness, not the engine: `stress_documents` centres the grid on the origin
and the orbit camera starts inside it, so frustum and occlusion culling
correctly reject essentially everything. **The cull is working; there is just
nothing to draw.**

So `raster1` and `raster2` here are almost pure per-secondary cost — binding a
pipeline, two descriptor sets, vertex/index buffers and a viewport, then
issuing a `multiDrawIndexedIndirect` whose every `instance_count` is zero.
That is ~2.9 µs per world per pass, which is what produced `raster1` 11 → 55
and `raster2` 2 → 45 µs across 1 → 16 worlds, and it is worth knowing on its
own: an empty per-world scene secondary is not free.

`pass2` draws zero in every scene measured — with a stationary camera, pass
1's test against the reprojected Hi-Z already resolves everything, so the
dual-pass second stage is pure overhead here.

### What this does and does not invalidate

* `scatter`, `mvp1`, `mvp2` are **unaffected** — they are dispatched over
  entity/candidate counts, not over what survives culling, so those figures
  and the barrier findings behind them stand.
* Every `raster1` / `raster2` figure, and the `gpu total` that includes them,
  measures draw-call overhead on an empty scene. Treat them as a bind-cost
  microbenchmark, not as rendering.
* **Merged raster draws were reverted.** They did remove real overhead — 16
  empty draw calls become one — but only at world counts nothing asks for, in
  a benchmark that renders nothing. See the ADR-0011 §5 discussion: the
  gizmo case that motivated many-worlds-per-camera wants an overlay pass, not
  a world.

### Fixing the harness

To measure raster, the camera has to be outside the grid. Until that is done,
do not quote a raster number from `--stress`.

## Empty-draw compaction

Built after the finding above: if `raster` is mostly the cost of issuing
draws, stop issuing the ones that draw nothing. `draw_compact.comp` appends
the slots the cull left with instances into a compacted list plus a
GPU-written count, and the raster switched to
`vkCmdDrawIndexedIndirectCount`.

**It does not pay yet, and the number that says so was worth measuring
first.** `drawCount` — one command per uploaded mesh slot — is 2–5 in every
scene the engine can currently build:

| scene | drawCount |
|---|---:|
| `editor --stress` | 2–3 |
| default editor | 2–4 |
| `test-game --shapes` (all three meshes) | 5 |

Walking three indirect structs is nanoseconds; the ~2.9 µs per world per
pass measured above is pipeline/descriptor/vertex/viewport binds and
`vkCmdExecuteCommands`, which compaction does not touch. At 4 worlds / 1M
entities the GPU frame went 509.6 → 516.3 µs — a ~7 µs regression from the
extra dispatch, its two resets and their barriers.

Kept as groundwork: the win scales with mesh-slot count, and nothing here
generates hundreds of slots. Revisit with a scene that does.

### The bug it shipped with, and the process failure around it

The first version dispatched over `slot_capacity` rather than the live
`slot_count`. `write_indirect_template` writes only `commands.len()` and left
the tail undefined; `slot_capacity` grows geometrically, so the compaction
read garbage `instance_count`s, copied them in as real commands, and
`vkCmdDrawIndexedIndirectCount` issued garbage `index_count`/`first_index`
against the mega index buffer. **That hangs the GPU and takes the desktop
with it.**

Four layers now: the live count bounds the dispatch; the template tail is
zeroed so a stray read draws nothing; the shader clamps its write index; and
`compact_args` is cleared each frame so a racy read cannot see stale device
memory.

Two process lessons, both of which cost real time:

* A `cargo build --workspace` builds the editor in **debug**. The runs that
  segfaulted were executing a stale `target/release/editor`, so a fixed
  source was A/B'd against an unfixed binary and the wrong component got
  the credit. Rebuild the profile you are about to run.
* **lavapipe is the first stop for anything touching indirect draws.**
  `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` — it supports
  `drawIndirectCount`, and a bad command segfaults a process instead of the
  session. It is where the counters above were read back.
