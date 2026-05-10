# ADR-0002: Per-Frame Command Buffer Recording

**Status:** Superseded — see "Update" section below.
**Date:** 2025
**Scope:** `crates/engine-render/src/lib.rs`
**Related:** [ADR-0001](ADR-0001-custom-swapchain.md)

> ## Update (superseded)
>
> The per-frame `OneTimeSubmit` recording path described below was retired
> after a measured FPS drop in the simple-cube scene
> (~12k FPS → ~8k FPS). The renderer now maintains **one reusable
> `MultipleSubmit` primary command buffer per swapchain image**, plus a
> per-image staging buffer + device-local matrix buffer + descriptor set
> ("FrameSlot"). The CB pre-records `copy_buffer(staging → device)` followed
> by all draws (one `draw_indexed(…, first_instance = i)` per
> `RenderInstance`). Each frame the host only:
>
> 1. Computes per-instance MVPs into a reusable scratch `Vec`.
> 2. Memcpy's them into the slot's host-mapped staging buffer.
> 3. Submits the pre-recorded CB.
>
> Slots are rebuilt only when the swapchain is recreated (resize / present-mode
> change) or when scene topology changes — the latter is *not yet exposed* via
> the public API; instances are baked at `Window::with_scene` time. The
> per-image `in_flight` fence (now per-image rather than per-frame-in-flight)
> guarantees the CPU never writes to a staging buffer the GPU is still
> reading from.
>
> This is also the scaffolding for the future GPU-driven path: a single
> `draw_indexed_indirect_count` will replace the per-instance draw loop, and
> the matrix mega-buffer will be one big device-local SSBO populated by
> compute (or by a much larger staging buffer transferred per frame).
>
> The original "per-frame recording" rationale below is preserved for
> historical context.
>
> ---

## Context

The first iteration of the renderer pre-recorded one
`SimultaneousUse` primary command buffer per swapchain image at startup (and
on resize). The MVP push constant was baked into the recorded commands. The
hot path was an `Arc::clone` + submit, with no per-frame Vulkan recording at
all — the source of much of the 11k+ FPS we measured (see ADR-0001).

That design assumes the rendered scene is *static between resizes*. As soon
as we want:

- A camera the user can move (orbit / pan / zoom), or
- Per-instance model matrices read from a `TransformHierarchy`, or
- Animated transforms,

…the baked-in MVP becomes a hard wall. The push constant must change every
frame, so the command buffer must be re-recorded every frame.

## Decision

Drop the cached-command-buffer path. Each frame:

1. Acquire the swapchain image (`SwapchainRenderer::acquire`).
2. Resolve every `RenderInstance`'s world matrix from the
   `TransformHierarchy` (interior mutability — no `&mut` required).
3. Build a fresh `PrimaryAutoCommandBuffer` with
   `CommandBufferUsage::OneTimeSubmit`.
4. For each instance: push the per-instance MVP and `draw_indexed`.
5. Submit + present via the unchecked swapchain path (still ADR-0001).

`OneTimeSubmit` is the right hint here: we never re-submit the same CB, so
vulkano can release the CB's resource locks as soon as the submission is
acknowledged by the host-side tracker. (And recall from ADR-0001 that the
render submit itself is `submit_unchecked` and bypasses tracking — but the
allocator's per-CB bookkeeping still benefits from the correct usage hint.)

## Consequences

**Positive:**

- Per-instance model matrices are free to change every frame.
- Camera state lives entirely on the CPU; no GPU resources need updating.
- Adding/removing draws is just a list mutation — no CB invalidation logic.
- Sets the stage for per-frame UI overlays, debug draws, picking, etc.

**Negative (the tradeoff):**

- One `AutoCommandBufferBuilder` allocation + record + build per frame.
- Vulkano's `StandardCommandBufferAllocator` pools the underlying Vulkan
  command-pool memory, but there's still per-frame work in the host wrapper:
  resource access tracking, pipeline-barrier inference, the build step's
  validation pass, etc.
- Expected FPS impact on the test machine: the prior steady-state of
  ~11.5k–11.7k FPS will drop. We have not yet re-measured; rough estimate
  is into the high-thousands. Still ample for development.

## What We Did NOT Do (and why)

We considered several approaches that would preserve some caching:

1. **Per-frame UBO + cached CB.** Move the MVP to a uniform buffer indexed by
   `frame_in_flight` slot; bind a descriptor set in the cached CB; host
   writes the UBO before submit. *Rejected for now:* introduces the
   descriptor-set allocator, requires per-`(image_index, frame_in_flight)`
   CB variants because the descriptor-set binding bakes into the CB, and
   adds layout machinery we don't yet need. Worth revisiting once we have
   many instances or a proper material system.
2. **Storage buffer of model matrices + `gl_InstanceIndex` push constant.**
   The natural design for hundreds of instances. *Premature* with a single
   cube. Will pair naturally with GPU-driven indirect draws later.
3. **Re-record only when transforms change.** Requires a dirty-flag bridge
   between the renderer and the hierarchy (the hierarchy already has
   `Dirty`!). Plausible future optimisation, but the per-frame cost is small
   enough that "always re-record" is a fine starting point.

## Caveats

This decision interacts cleanly with ADR-0001 — the `OneTimeSubmit` CB lives
entirely on the renderer's owned hot path; the unchecked submit is fine with
it because the CB does not outlive the frame's `in_flight` fence.

The `on_update` callback runs on the event-loop thread synchronously before
recording. Any expensive game logic there blocks rendering. If/when a real
update loop appears, it should run on its own thread and the renderer should
just snapshot transforms at frame start.

## Revisit If

- A profiler shows command-buffer recording is the dominant frame cost.
- We want to draw thousands of instances (→ go to option 2 above).
- We add a material/descriptor-set system (→ option 1 becomes natural).
- The single-threaded `on_update` hook becomes a bottleneck.
