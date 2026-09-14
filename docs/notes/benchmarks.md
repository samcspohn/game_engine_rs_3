# Stress benchmarks and measured frame times

How to reproduce the numbers, and what the numbers are.

## Running one

`test-game` accepts `--shapes N` to spawn an `N`-entity grid (and
`--static-scene` to skip the per-frame `Rotator` updates). `--glb <path>`
loads a glTF scene template and spawns one instance at the origin —
placeholder meshes appear immediately and each primitive streams in as its
background decode completes. Entities cycle round-robin through cube / sphere
/ cylinder `MeshRenderer`s (`crates/test-game/assets/{cube,sphere,cylinder}`),
exercising concurrent async mesh loads and a multi-slot
`MultiDrawIndexedIndirect` once they resolve. Use `ENGINE_NUM_THREADS=1` (or
its back-compat alias `RAYON_NUM_THREADS=1`) to compare single- vs
multi-threaded staging writes.

`editor` accepts `--stress N --worlds W`, which spawns `N` spinning cubes as
one grid cut into `W` worlds, all composited through a single camera — the
drawn set is identical at every `W`, so only the number of per-world scatter
dispatches changes. It is what measured the world-major → stage-major scatter
change
([`docs/notes/scatter-overlap-bench.md`](../notes/scatter-overlap-bench.md)):
at 16 worlds and 1M entities the scatter went 580 → 336 µs, and the per-world
cost 17 → 1.9 µs.

```sh
cargo run --release -p test-game -- --shapes 100000
ENGINE_NUM_THREADS=1 cargo run --release -p test-game -- --shapes 100000
```

## Editor UI, idle

What an editor frame costs with one document open and nothing being touched.
This is the number [ui-events](ui-events.md) exists for: the UI was
re-binding every visible row and re-formatting every inspector value on every
frame, and the equality gate only stopped the result reaching the GPU.

| | before | after |
|---|---|---|
| `sim_update` (FPS line, avg) | ~14 µs | **1.2 µs** |
| UI + taffy self time | ~8% of samples, ~20% of the main thread | **0.4% of samples** |
| frame | 0.135 ms (~7 400 FPS) | 0.132 ms (~7 580 FPS) |

The frame barely moved because it is bound by `acquire` (~29 µs) and
`host_staging` (~14 µs); what went away was CPU the editor was spending to
produce values nothing uploaded. What is left of the UI on the main thread is
`UiGpu::write_staging`, `run_layout` and one taffy entry, none above 0.21%.

To reproduce, build the editor the way `make editor` does (`-C
prefer-dynamic`, `CARGO_TARGET_DIR=target/editor`) and run the **binary**
rather than `cargo run`, so `perf` sees it:

```sh
R=$PWD
STD=$(dirname $(find ~/.rustup -name 'libstd-*.so' | head -1))
export LD_LIBRARY_PATH=$R/target/editor/editor:$R/target/editor/editor/deps:$STD
cd crates/test-game
perf record -F 999 -o /tmp/ui.data $R/target/editor/editor/editor --project .
perf report -i /tmp/ui.data --stdio --no-children | grep -E "::ui::|taffy"
```

`libstd-*.so` has to be found by hash — the `prefer-dynamic` build links the
nightly toolchain's, which is not the one `rustc --print sysroot` names.
Three counters make the claim testable rather than visual:
`UiCore::dispatched()` (handler calls), `Pointer::walks` (hit walks) and
`refits` (scrollbar re-fits), each asserted to stay flat over 100 idle
frames.

## Frame times

Current measured frame times (release build, multi-threaded staging, animated
`Rotator` scene; **post ADR-0003 shared-staging refactor** — see [ADR-0003
§Measurements](../ADR-0003-shared-staging-with-compute-sync.md#measurements-post-path-a-landing)
for the full pre-/post-refactor comparison and the throughput trade-off at
very large N, plus [ADR-0004
§Measurements](../ADR-0004-instanced-indirect-draw.md#measurements-post-phase-1)
for the original per-instance-vs-indirect-draw comparison):

| Cubes     | Frame time | Notes |
|---|---|---|
| 1         | ~0.12 ms  (~8 100 FPS) | GPU floor; single mesh, single instance. |
| 10 000    | ~0.77 ms  (~1 300 FPS) | |
| 100 000   | **~1.25 ms (~800 FPS)** | |
| 1 000 000 | **~4.0 ms (~250 FPS)** | At parity with (slightly faster than) the pre-refactor per-slot-staging baseline (4.55 ms), with the ~144 MB VRAM saving still banked. The uniform staging→SoT paradigm (host writes only staging, mvp_build reads only stable SoT) is what made this work — see [ADR-0003](../ADR-0003-shared-staging-with-compute-sync.md). |

The N≥1∘K baseline wins came from moving the per-component staging buffers
into BAR / ReBAR memory (`MemoryTypeFilter::PREFER_DEVICE |
HOST_RANDOM_ACCESS`) so the GPU's scatter compute reads them at full VRAM
bandwidth instead of PCIe per cache line. The CPU staging-write loop runs in
parallel via the engine's work-stealing pool (256 dirty-words / 8192 entities
per task).

**ADR-0004 Phase 1 (instanced indirect draw) landed and was measured.** The
scene secondary now records exactly **one `vkCmdDrawIndexedIndirect` per
distinct mesh** instead of one `draw_indexed` per `RenderInstance`. Instances
are sorted by `mesh_index` on the CPU at topology-change time so each mesh's
MVP-buffer slice is contiguous; the indirect command's `instance_count` and
`first_instance` fields then drive HW instancing for the entire group in one
call. Required `multi_draw_indirect`, `draw_indirect_first_instance` and
`draw_indirect_count` device features are enabled at device creation. The
vertex / compute shaders are unchanged — `gl_InstanceIndex` still indexes the
same MVP buffer, just with a non-zero base from `first_instance`. Result: ~10×
speedup at N=100K (~10 ms → ~1 ms), and N=1M is now interactive at ~4.5 ms
(~220 FPS), previously not measurable. Full A/B in [ADR-0004
§Measurements](../ADR-0004-instanced-indirect-draw.md#measurements-post-phase-1).

**ADR-0003 (shared staging + uniform staging→SoT paradigm + split-submit)
landed.** Single shared host-staging buffers replace the 4× per-FrameSlot
duplication — ~144 MB saved at N=1M. The big realisation along the way:
`view_proj` had to follow the same staging→SoT pattern as TRS (host writes
staging, the primary `vkCmdCopyBuffer`s it into a stable device buffer,
mvp_build reads only that — that buffer has since moved from the world onto
the camera, ADR-0011 step 5). Without that, the host's wait on the previous
frame's compute had to cover mvp_build's read of `view_proj`, which serialised
them and cost ~4 ms / frame at N=1M. With it, the wait fires the moment
scatter+fill+copy are done (microseconds at any N), mvp_build runs in parallel
with the next frame's host prep, and we end up at parity with the pre-refactor
frame times at every N — still with the VRAM win. **GPU-driven frustum culling
(ADR-0004 roadmap) and dual-pass temporal Hi-Z occlusion culling
([ADR-0005](../ADR-0005-dual-pass-occlusion-culling.md)) have since landed** —
see the "fully GPU-driven (Design B)" section above.


- [staging balancer](staging-balancer.md)
- [scatter overlap bench](scatter-overlap-bench.md)
