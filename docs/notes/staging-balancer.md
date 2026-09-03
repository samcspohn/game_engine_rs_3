# Adaptive staging memory type

The TRS staging triple can live in system RAM or in ReBAR VRAM, and the
faster choice depends on which side of the frame is the bottleneck.
`StagingBalancer` measures both and picks per scene.

The TRS staging triple can live on either side of the PCIe link, and which one
is faster depends on where the frame is bottlenecked:

| `StagingMemory` | Memory type | Host writes | Scatter reads |
| --- | --- | --- | --- |
| `HostCached` | `PREFER_DEVICE \| HOST_RANDOM_ACCESS` → system RAM (GTT) | cheap, cached | pulls across PCIe, snooping the writer's caches |
| `DeviceWc` | `PREFER_DEVICE \| HOST_SEQUENTIAL_WRITE` → WC ReBAR in VRAM | stream over PCIe | local VRAM |

(`HOST_RANDOM_ACCESS` *requires* `HOST_CACHED`, which the ReBAR type is not —
that required flag is what beats the `PREFER_DEVICE` beside it and pins the
triple to system RAM.) Measured at N=1M with workers confined to the GPU's
NUMA node: `HostCached` costs **148µs host / 320µs scatter**, `DeviceWc`
**428µs / 44µs** — a near-1:1 transfer of ~280µs *between the two processors*,
not a saving on either.

A frame costs `max(cpu_busy, gpu_busy)`, so `StagingBalancer` (in `lib.rs`)
picks whichever mode minimises that max. It splits each side into its staging
part and the rest, keeps a per-mode EMA of the staging parts, and predicts the
mode it is *not* in from the last time it was in it:

```text
frame(m) = max(cpu_rest + cpu_staging[m], gpu_rest + gpu_scatter[m])
```

Only the staging terms are mode-dependent, and they are driven by the
dirty-word count rather than by the camera, so they stay valid while the view
changes. `cpu_rest` / `gpu_rest` are shared and refreshed every frame in
whichever mode is selected, which is what lets the balancer follow a scene
sliding from GPU-bound to CPU-bound.

**GPU durations are clock-normalised before they enter the model.** A GPU that
is not the frame's bottleneck has idle gaps, DPM downclocks it, and every
stage stretches on identical work:

| | vram (CPU-bound) | cached (GPU-bound) |
| --- | --- | --- |
| sclk / board power | 1232 MHz / 112 W | 3139 MHz / 245 W |
| GPU busy | 62% | 100% |
| `raster1` | 254µs | 116µs |

Comparing a downclocked `gpu_rest` against a full-speed `cpu_rest` concludes
the GPU cannot afford the scatter — a trap, because the CPU-heavy mode
manufactures the evidence that keeps it selected. `SclkMonitor` (in
`gpu_telemetry.rs`) samples `freq1_input` on a background thread every 10ms
and the balancer rescales each GPU sample to the reference clock; the model
then works in the clock the GPU *would* run at if it were the bottleneck,
which is the only case where its duration decides the frame. Reading that file
costs ~19µs — 4% of a 450µs frame — so it must not be on the hot path. Without
a readable clock, `auto` refuses to run rather than compare timings it knows
are incomparable.

(The reference is the running maximum, seeded from `pp_dpm_sclk`. On RDNA3
that table understates reality — it tops out at 2482 MHz while `freq1_input`
reports up to 3165 MHz under load — so the table alone is not a usable
reference. `mclk` is pinned at 1249 MHz in every sample taken and is not a
factor.)

Each side's cost is tracked as a **rolling minimum** over the last 128–256
frames rather than an average. Every input is a clean floor with spikes on top
— `host_staging` reads `402.7 / 426.6 / 541.7` min/avg/max across a window,
`raster1` reads `241.6 / 254.0 / 263.8` — and the spikes are asset loading, a
scheduler hiccup or a DPM ramp, none of which the staging memory type can
affect. The floors are stable to ~1%, which is what makes them comparable
across modes, and taking a minimum means no warm-up period and no outlier
rejection are needed. It does mean a *missing* sample must never reach it: a
rebuild frame leaves its timestamp queries unwritten, `saturating_sub` renders
that as a 0µs stage, and a minimum latches onto it permanently, so zero deltas
are dropped rather than recorded.

Switching reallocates every staging slot and re-records every FrameSlot
primary, hence a 5% predicted-win threshold and a 250ms minimum dwell; samples
are discarded for `MAX_FRAMES_IN_FLIGHT * STAGING_SLOTS * 4` frames after a
switch because the GPU timestamps read each frame come from that (image, slot)
pair's previous submission. Each switch prints every term the decision rested
on, so a wrong choice is distinguishable from a wrong measurement:

```text
[staging] DeviceWc 704us -> HostCached 411us | cpu_rest 298 gpu_rest 46 staging 113/406 scatter 167/34
```

Measured at N=1M, 128 threads (auto starts on `HostCached`, probes the other
mode once, then holds — 3 runs per cell):

| Scene | pinned `cached` | pinned `vram` | **auto** | picked | switches |
| --- | --- | --- | --- | --- | --- |
| default (raster ~790µs, GPU-bound) | 831 FPS | 1083 FPS | **1086–1090 FPS** | `vram` | 1 |
| `ENGINE_CULL_AWAY=1` (raster 4µs, CPU-bound) | 2239 FPS | 1241 FPS | **2188–2304 FPS** | `cached` | 2 |

`ENGINE_STAGING_MODE=cached\|vram\|auto` (default `auto`) pins or frees the
choice; **F6** cycles auto → cached → vram at runtime. The FPS line is tagged
`staging=auto:vram` etc. so an A/B log is unambiguous. Note the ±20%
run-to-run spread in both columns is `sim_update` variance, not the balancer.


- [transform-gpu](transform-gpu.md)
- [benchmarks](benchmarks.md)
