# GPU-driven rendering and the asset registry

Drawables are components; the draw list is derived on the GPU. The redirect
model that lets an asset stream in without patching a single renderer.

The renderer is **component-driven** with **async mesh loading**: a scene
declares drawables by attaching [`MeshRenderer`] components, geometry flows
through the registry's GPU mega buffers, and meshes decode on a background
thread — entities show the placeholder cube until their asset lands, then swap
to it.

The registry is split across the GPU boundary so **mesh data is shareable**
(e.g. a future physics system reads the same retained `Arc<Mesh>` for
collision geometry):

| Type | Crate | Role |
|------|-------|------|
| `MeshId` | engine-core | Stable, write-once handle a future `MeshRenderer` component stores. Allocated per *unique requested path* (deduped, `u64` path hash). Indexes the redirect map. |
| `MeshSlot` | engine-core | Physical drawable slot. Slots `0`/`1` are the resident **placeholder** (cube) and **error** (tetrahedron — a deliberately distinct silhouette so failed loads are obvious) meshes. |
| `MeshBounds` | engine-core | Local-space bounding sphere per slot (GPU culling + CPU broad-phase). |
| `AssetRegistry` | engine-core | GPU-agnostic source of truth: dedup cache, `mesh_id → MeshSlot` redirect map, refcounts, and the **retained `Arc<Mesh>`** per slot. A lazily-initialized `asset::global()` (`Mutex<AssetRegistry>`, mirroring `thread_pool::global()`) lets a component constructor `request` a mesh and immediately get a `MeshId` without threading a context through the ECS. Unit-tested without a GPU. |
| `MeshTableEntry` | engine-render | Per slot, as the GPU sees it: the static `VkDrawIndexedIndirectCommand` fields (`index_count`/`first_index`/`vertex_offset`) plus the bounding sphere. std430, 32 bytes. |
| `GpuMeshStore` | engine-render | **Device-local** mirror — mega vertex/index buffers, the table, and the redirect buffer. `sync()` drains the core registry's deltas (new slots + redirect changes) and uploads them via host-staging + `vkCmdCopyBuffer`; it assigns the mega-buffer offsets (a render-side concern) as it appends. Buffers grow geometrically. |
| `MeshRenderer` | engine-render | ECS component (`HAS_UPDATE = false`) storing only a `MeshId`. `new(path)` resolves the path against `asset::global()` (so the constructor returns a handle, no path stored); `init` pushes `(transform_id, mesh_id)` onto a render-side global spawn queue the renderer drains and scatters into `GPURenderers`. |
| `GpuRenderers` | engine-render | Device-local `GPURenderers` buffer — one `mesh_id` per transform slot (indexed by `transform_id`, parallel to the SoT), sentinel `0xFFFFFFFF` for empty slots. A scatter compute (`gpu_renderers_scatter.comp`) writes drained spawns in; grows with world capacity. **This is the live instance source the cull pass reads** (Design B). |

Every process-wide registry behind a `global()` — meshes, materials, textures,
scene templates, the UI store, the active camera, the renderer spawn queue —
locks a `parking_lot::Mutex` rather than a `std` one. `std::sync::Mutex`
poisons, so a panic anywhere under a component's `update` while one was held
would kill the editor on the *next* frame instead of at the fault, and it is
precisely a caught panic that
[ADR-0010](../ADR-0010-scene-authoring-and-play.md) §7 wants to survive.

The key decoupling: a renderer holds only a stable `MeshId`; load completion
is a single redirect write (`mesh_id → slot`) — `MeshSlot::PLACEHOLDER` while
loading, the real slot once resolved, `MeshSlot::ERROR` on failure — so no
renderer record is ever patched and no per-renderer pending state is tracked.

**The renderer is fully GPU-driven (Design B).** There is **no CPU-sorted
topology**. Each frame the pass-1 cull compute pass (`mvp_build.comp`)
dispatches one invocation per transform slot and reads `GPURenderers[i] →
mesh_id`, `redirect[mesh_id] → slot`, `SoT[i]` (transform), and
`mesh_table[slot]` (bounds) directly. It frustum-tests the world bounding
sphere (Gribb–Hartmann planes from `view_proj`, authoritative), then — for
frustum-visible instances — occlusion-tests it against last frame's Hi-Z
pyramid; not-occluded instances atomically claim the next slot in that
drawable slot's MVP region (`base = indirect[slot].first_instance`, `local =
atomicAdd(indirect[slot].instance_count, 1)`) and write the compacted MVP,
while possibly-occluded ones are deferred to a second cull pass
(`mvp_build_pass2.comp`) that re-tests them against this frame's own
freshly-built Hi-Z. Each pass's `instance_count`s are reset every frame by a
`vkCmdCopyBuffer` of a host template (counts pre-zeroed) into that pass's
device args buffer, recorded just before its cull dispatch. A small compaction
dispatch (`draw_compact.comp`, appended to the same cull secondary) then drops
the slots the cull left empty — one invocation per slot, survivors appended by
atomic into a compacted list plus a GPU-written `drawCount` — and each pass's
scene secondary issues a **single `vkCmdDrawIndexedIndirectCount`** over that
list against the shared mega buffers, so the raster never walks a slot with no
instances. Compaction only reads the *live* command range: past it the
template is zeroed, because a garbage `index_count` reaching an indirect draw
hangs the GPU rather than drawing nothing. **See
[ADR-0005](../ADR-0005-dual-pass-occlusion-culling.md) for the full dual-pass
occlusion design.**

The **only per-frame CPU work** is the `DrawPlan`: per drawable slot, the
geometry (from the mesh-table mirror) + the prefix-summed `first_instance`
base, where bases come from `AssetRegistry::slot_instance_totals()` (Σ
refcounts per slot). That's `O(#slots)` and only runs on a topology change
(spawn / load / capacity grow) — never an `O(N)` sort, and no GPU prefix sum.
A spawn of an existing mesh within capacity takes the **cheap path**: the GPU
scatter plus an in-place `O(#slots)` rewrite of the indirect template's bases
(gated behind the per-frame compute wait, since the template is read by
in-flight reset copies) — **no descriptor-set / secondary / frame-slot
re-recording**. The expensive **structural rebuild** (new MVP / indirect
buffers, cull set, secondaries, frame slots) happens only on **geometric
capacity growth**, a new distinct mesh (`#slots` changes), or a completed load
(which re-allocates a cull-bound buffer). A load itself is a one-word redirect
flip whose effect the next cull picks up automatically (placeholder instances
regroup onto the loaded slot).

**Async loading.** `MeshRenderer::new(path)` resolves the path against the
registry (deduped); on the first request of a path it queues an
`asset::request_load`, handled by a **dedicated background loader thread** (in
`engine-core`, off the fork-join pool) that decodes the file (`.obj` via
[`tobj`](https://crates.io/crates/tobj); paths are CWD-relative) into a CPU
`Mesh` and calls `AssetRegistry::resolve` (or `fail` → error mesh, logged).
Each frame `GpuMeshStore::sync()` uploads newly-resolved geometry, patches the
GPU redirect buffer, and returns the per-slot totals (computed under the same
registry lock, so they're consistent with the redirect the cull will read).

## Materials and textures

Materials and textures use the **same redirect model as meshes**, so a
streaming texture never forces a material re-upload and a material created
before its texture decodes simply samples the placeholder.

| Type | Crate | Role |
|------|-------|------|
| `MaterialData` / `MaterialRegistry` | engine-core | The metallic-roughness parameter set: base-color factor, metallic, roughness, emissive, `normal_scale`, `occlusion_strength`, plus five optional `TextureId`s (base color, normal, metallic-roughness, occlusion, emissive). `get_or_create` content-hash-dedups, so identical materials collapse to one `MaterialId` across primitives and files. Materials resolve immediately (tiny POD — no decode phase); `update` edits in place, `duplicate` detaches for solo editing. |
| `ColorSpace` | engine-core | `Srgb` / `Linear`, decided by **usage, not by the file**: base-color and emissive maps are sRGB-encoded, normal / metallic-roughness / occlusion maps carry raw linear data. It is part of the texture dedup key, so one image referenced both ways yields two ids and two device images in two formats (`R8G8B8A8_SRGB` vs `R8G8B8A8_UNORM`). |
| `TextureRegistry` / `GpuTextureStore` | engine-core / engine-render | `TextureId → TextureSlot` redirect; slot 0 is a 1×1 white placeholder, slot 1 a magenta/black error checkerboard. Uploads are paced by a streaming time budget; redirect flips whose slot isn't resident yet stay pending. |
| `GpuMaterialStore` | engine-render | Device mirror: a 64-byte `GpuMaterial` per slot plus the `MaterialId → slot` redirect the fragment shader reads. No pacing — materials are too small to need it. |

The fragment shader walks the whole chain GPU-side: `v_material → mat_redirect
→ material slot → the map's raw TextureId → tex_redirect → texture slot →
u_textures[slot]`. The white placeholder is the right stand-in for a
still-decoding *color* map (it multiplies to the untinted factor) but not for
data maps — white decodes to a 45°-tilted normal and to fully-metallic — so
the normal and metallic-roughness lookups test the redirect for `PLACEHOLDER`
and skip the map until its slot is resident.

**Tangents.** glTF's `TANGENT` accessor is optional and OBJ has none, so any
mesh whose source didn't author tangents gets `Mesh::generate_tangents()` at
decode time (per-triangle Lengyel accumulation, Gram-Schmidt-orthogonalised
against the normal). glTF importing reads `normalTexture` (with `scale`),
`metallicRoughnessTexture`, `occlusionTexture` (with `strength`) and
`emissiveTexture`; OBJ maps `map_Kd` → base color and `map_Bump`/`bump` →
normal map.

**Known follow-up:** compaction uses one global `atomicAdd` per visible
instance on the slot's counter; a workgroup-local aggregation (keyed on
dynamic per-workgroup slot equality) is the planned optimization.


- [ADR-0004 instanced indirect draw](../ADR-0004-instanced-indirect-draw.md)
- [ADR-0005 dual-pass occlusion
  culling](../ADR-0005-dual-pass-occlusion-culling.md)
- [texture-update](texture-update.md)
