//! Device-local GPU mirror of [`engine_core::asset`].
//!
//! [`GpuMeshStore`] owns the mega vertex/index buffers, the per-slot
//! [`MeshTableEntry`] table, and the redirect buffer the cull kernel reads.
//! [`GpuMeshStore::sync`] drains the core registry's deltas — newly resolved
//! slots and redirect changes — and uploads them via host-staging + copy.
//!
//! # Offsets are a render concern
//!
//! The core registry is GPU-agnostic and stores no buffer offsets. This store
//! assigns each mesh's `vertex_offset` / `first_index` as it appends geometry
//! into *its* mega buffers, and builds the [`MeshTableEntry`] from those
//! offsets plus the core registry's bounds. Indices stay 0-based per mesh;
//! `vertex_offset` rebases them at draw time.
//!
//! # Synchronization (first slice)
//!
//! Uploads use **one-shot fence-waited submits** — correct and self-contained
//! but blocking. New geometry always lands in a previously unused mega-buffer
//! region (append-only), so the copy never races a draw. Frame-loop
//! integration will move these onto a between-frames transfer CB, with live
//! redirect flips gated to the per-frame safe window.

use std::sync::Arc;

use engine_core::asset::{self, MeshBounds, MeshSlot};
use engine_core::mesh::Mesh;
use engine_core::texture::TextureId;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo,
    },
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::GpuFuture,
};

use super::MeshTableEntry;
use crate::gpu_mesh::GpuVertex;

// Initial capacities. All grow geometrically (≥ 2×) on demand.
const INITIAL_VERTEX_CAP: u32 = 1 << 12;
const INITIAL_INDEX_CAP: u32 = 1 << 13;
const INITIAL_TABLE_CAP: u32 = 8;
const INITIAL_REDIRECT_CAP: u32 = 64;

// ── Streaming time budget ───────────────────────────────────────────────
// Uploads are paced by **time**, not fixed counts. Without pacing, a
// decode burst snowballs: a frame uploads everything resolved since the
// last frame (fence-waited), the longer frame lets more decodes land, the
// next batch is bigger — geometric growth until the whole backlog lands in
// one or two giant frames.
//
// The contract: while streaming, the frame should stay above
// [`STREAM_MIN_FPS`]. This store's sync may spend [`UPLOAD_FRAME_SHARE`]
// of that frame budget (the rest is left for texture uploads, the
// draw-plan/frame-slot rebuild, and the frame itself). Since upload cost
// isn't known until after the fence wait, the caps are **adaptive**: each
// sync that uploads measures its wall time and rescales the slot/byte caps
// toward the budget (damped, step-clamped), converging on whatever
// throughput the platform sustains. Redirect flips whose slot hasn't been
// uploaded yet are deferred, so a renderer keeps drawing its placeholder
// until its geometry is resident.

/// Frame-rate floor the streaming pacer aims to hold.
const STREAM_MIN_FPS: f64 = 30.0;
/// Fraction of the `1 / STREAM_MIN_FPS` frame budget mesh uploads may
/// spend per sync.
const UPLOAD_FRAME_SHARE: f64 = 0.5;
/// Starting caps; the controller adapts from here.
const INITIAL_UPLOAD_SLOTS_CAP: usize = 1024;
const INITIAL_UPLOAD_BYTES_CAP: usize = 8 << 20;
/// Absolute bounds on the adaptive caps (floor keeps progress under
/// pathological stalls; ceiling bounds staging memory).
const UPLOAD_SLOTS_CAP_RANGE: (usize, usize) = (64, 1 << 16);
const UPLOAD_BYTES_CAP_RANGE: (usize, usize) = (1 << 20, 512 << 20);

/// Sentinel in the per-slot texture buffer: this drawable slot has no
/// base-color texture (fragment shader falls back to the flat base color).
pub const NO_TEXTURE: u32 = u32::MAX;

/// GPU-resident mirror of the core mesh registry.
pub struct GpuMeshStore {
    // ── Mega geometry buffers (device-local, append-only) ───────────────
    mega_vertex: Subbuffer<[GpuVertex]>,
    mega_index: Subbuffer<[u32]>,
    vertex_cap: u32,
    index_cap: u32,
    /// Append cursors — the render side's mega-buffer offset bookkeeping.
    vertex_used: u32,
    index_used: u32,

    // ── Per-slot table + redirect map (device-local mirrors) ────────────
    table_buf: Subbuffer<[MeshTableEntry]>,
    table_cap: u32,
    /// `mesh_id → slot` as raw `u32`s. Zero-filled so unresolved ids read as
    /// [`MeshSlot::PLACEHOLDER`] (slot 0).
    redirect_buf: Subbuffer<[u32]>,
    redirect_cap: u32,
    /// Per drawable slot: the raw [`TextureId`] of its base-color texture,
    /// or [`NO_TEXTURE`]. Read by the fragment shader (via `gl_DrawID` ==
    /// slot), resolved through the texture store's own redirect buffer.
    /// Sized/grown in lock-step with `table_buf`.
    slot_texture_buf: Subbuffer<[u32]>,

    /// CPU mirror of the per-slot table, indexed by [`MeshSlot`]. Lets the
    /// camera build indirect-draw commands (mega-buffer offsets + index
    /// counts) without reading back the device-local table buffer.
    cpu_table: Vec<MeshTableEntry>,

    /// Number of core registry slots already uploaded (sync watermark).
    /// Lags the registry while a streaming burst is budget-paced.
    synced_slots: u32,

    /// Redirect flips drained from the registry whose target slot hasn't
    /// been uploaded yet (budget pacing). Re-examined every sync; applied
    /// once the watermark passes their slot.
    pending_redirects: Vec<(engine_core::asset::MeshId, MeshSlot)>,
    /// CPU mirror of the **GPU** redirect buffer (`mesh_id → slot` as the
    /// cull currently sees it — i.e. with pending flips still at their old
    /// value). Per-slot instance totals are computed against this, keeping
    /// the draw plan's `first_instance` regions consistent with what the
    /// cull will actually write.
    cpu_redirect: Vec<u32>,

    /// Adaptive per-sync upload caps (see the streaming-time-budget notes
    /// on the constants above). Rescaled after every uploading sync from
    /// its measured wall time vs the time budget.
    upload_slots_cap: usize,
    upload_bytes_cap: usize,

    // ── Allocators / queue for uploads ──────────────────────────────────
    memory_allocator: Arc<StandardMemoryAllocator>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
}

impl GpuMeshStore {
    /// Allocate empty device buffers (redirect zero-filled). The placeholder /
    /// error meshes resident in the core registry are uploaded on the first
    /// [`sync`](Self::sync), like any other slots.
    pub fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        cb_allocator: Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
    ) -> Self {
        let mega_vertex = alloc_device::<GpuVertex>(
            &memory_allocator,
            INITIAL_VERTEX_CAP,
            BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
        );
        let mega_index = alloc_device::<u32>(
            &memory_allocator,
            INITIAL_INDEX_CAP,
            BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
        );
        let table_buf = alloc_device::<MeshTableEntry>(
            &memory_allocator,
            INITIAL_TABLE_CAP,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
        );
        let redirect_buf = alloc_device::<u32>(
            &memory_allocator,
            INITIAL_REDIRECT_CAP,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
        );
        let slot_texture_buf = alloc_device::<u32>(
            &memory_allocator,
            INITIAL_TABLE_CAP,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
        );

        let store = Self {
            mega_vertex,
            mega_index,
            vertex_cap: INITIAL_VERTEX_CAP,
            index_cap: INITIAL_INDEX_CAP,
            vertex_used: 0,
            index_used: 0,
            table_buf,
            table_cap: INITIAL_TABLE_CAP,
            redirect_buf,
            redirect_cap: INITIAL_REDIRECT_CAP,
            slot_texture_buf,
            cpu_table: Vec::new(),
            synced_slots: 0,
            pending_redirects: Vec::new(),
            cpu_redirect: Vec::new(),
            upload_slots_cap: INITIAL_UPLOAD_SLOTS_CAP,
            upload_bytes_cap: INITIAL_UPLOAD_BYTES_CAP,
            memory_allocator,
            cb_allocator,
            queue,
        };

        // Zero the redirect buffer so unresolved ids default to PLACEHOLDER,
        // and sentinel-fill the per-slot texture buffer (untextured).
        let mut builder = store.primary_builder();
        builder
            .fill_buffer(store.redirect_buf.clone(), MeshSlot::PLACEHOLDER.0)
            .expect("zero-fill redirect buffer");
        builder
            .fill_buffer(store.slot_texture_buf.clone(), NO_TEXTURE)
            .expect("sentinel-fill slot texture buffer");
        store.submit_and_wait(builder.build().expect("build init CB"));

        store
    }

    /// Drain the core registry's deltas — newly resolved slots and redirect
    /// changes — upload them **within the per-frame streaming time budget**
    /// (the adaptive caps targeting [`UPLOAD_FRAME_SHARE`] of a
    /// [`STREAM_MIN_FPS`] frame), and return `(changed, slot_totals)`.
    /// Over-budget slots and redirect flips whose slot isn't uploaded yet
    /// carry over to subsequent frames, so a decode burst streams over many
    /// short frames instead of snowballing into a few giant ones.
    ///
    /// `changed` is `true` if anything uploaded, flipped, or grew (so the
    /// caller rebuilds the draw plan / camera). `slot_totals` is the
    /// per-slot instance count computed against this store's
    /// **GPU-visible** redirect mirror — not the registry's redirect, which
    /// runs ahead of the GPU during budget pacing — keeping the cull's
    /// `first_instance` regions consistent with what it will write. It's
    /// returned every frame (even when nothing uploaded) because a spawn
    /// shifts the totals without changing the redirect.
    ///
    /// Briefly locks the global registry to clone out the budgeted
    /// `Arc<Mesh>`es + bounds, the redirect updates, and the refcount
    /// snapshot, then releases it before doing GPU work so a background
    /// `resolve` is never blocked.
    pub fn sync(&mut self) -> (bool, Vec<u32>) {
        let t0 = std::time::Instant::now();
        let from = self.synced_slots;
        // Budget-limited drain: take at most the current adaptive slot/byte
        // caps (always ≥ 1 so a single over-budget mesh still lands). The
        // remainder stays in the registry for the following frames.
        let (new_slots, redirect_updates, mesh_id_count, refcounts): (
            Vec<(Arc<Mesh>, MeshBounds, Option<TextureId>)>,
            Vec<(engine_core::asset::MeshId, MeshSlot)>,
            u32,
            Vec<u32>,
        ) = {
            let mut reg = asset::global()
                .lock()
                .expect("asset registry mutex poisoned");
            let slot_count = reg.slot_count();
            let mut new = Vec::new();
            let mut bytes = 0usize;
            let mut s = from;
            while s < slot_count
                && new.len() < self.upload_slots_cap
                && (bytes < self.upload_bytes_cap || new.is_empty())
            {
                let (mesh, bounds) = reg.slot(MeshSlot(s));
                bytes += mesh.vertices.len() * std::mem::size_of::<GpuVertex>()
                    + mesh.indices.len() * std::mem::size_of::<u32>();
                new.push((mesh, bounds, reg.slot_texture(MeshSlot(s))));
                s += 1;
            }
            (
                new,
                reg.take_redirect_updates(),
                reg.mesh_id_count(),
                reg.refcounts(),
            )
        };
        // Drained flips join the pending queue; they apply only once their
        // slot is uploaded (below) so placeholders never dangle into
        // not-yet-resident geometry.
        self.pending_redirects.extend(redirect_updates);
        if self.cpu_redirect.len() < mesh_id_count as usize {
            // New ids read as PLACEHOLDER until flipped, mirroring the
            // zero-filled GPU redirect buffer.
            self.cpu_redirect
                .resize(mesh_id_count as usize, MeshSlot::PLACEHOLDER.0);
        }

        let needs_redirect_grow = mesh_id_count > self.redirect_cap;
        if new_slots.is_empty() && self.pending_redirects.is_empty() && !needs_redirect_grow {
            return (false, self.slot_totals(&refcounts));
        }

        // Assign mega-buffer offsets for the new slots (render-side concern).
        let mut placements = Vec::with_capacity(new_slots.len());
        let mut v_cursor = self.vertex_used;
        let mut i_cursor = self.index_used;
        for (mesh, bounds, _texture) in &new_slots {
            let vcount = mesh.vertices.len() as u32;
            let icount = mesh.indices.len() as u32;
            placements.push(SlotPlacement {
                vertex_offset: v_cursor,
                first_index: i_cursor,
                vcount,
                icount,
                bounds: *bounds,
            });
            v_cursor += vcount;
            i_cursor += icount;
        }

        // Grow backing buffers as needed (each grow submits its own CB).
        let mut grew = false;
        grew |= self.ensure_vertex_cap(v_cursor);
        grew |= self.ensure_index_cap(i_cursor);
        grew |= self.ensure_table_cap(from + new_slots.len() as u32);
        grew |= self.ensure_redirect_cap(mesh_id_count);

        // Record every upload into a single CB.
        let mut builder = self.primary_builder();

        // Geometry + the contiguous block of new table entries (+ each new
        // slot's base-color TextureId for the fragment lookup).
        let mut table_entries = Vec::with_capacity(new_slots.len());
        let mut texture_entries = Vec::with_capacity(new_slots.len());
        for ((mesh, _bounds, texture), p) in new_slots.iter().zip(placements.iter()) {
            if p.vcount > 0 {
                let verts = to_gpu_verts(mesh);
                self.record_copy(&mut builder, &verts, &self.mega_vertex, p.vertex_offset);
            }
            if p.icount > 0 {
                self.record_copy(&mut builder, &mesh.indices, &self.mega_index, p.first_index);
            }
            table_entries.push(MeshTableEntry {
                index_count: p.icount,
                first_index: p.first_index,
                vertex_offset: p.vertex_offset as i32,
                _pad0: 0,
                bounds_center: p.bounds.center,
                bounds_radius: p.bounds.radius,
            });
            texture_entries.push(texture.map(|t| t.0).unwrap_or(NO_TEXTURE));
        }
        self.record_copy(&mut builder, &table_entries, &self.table_buf, from);
        self.record_copy(&mut builder, &texture_entries, &self.slot_texture_buf, from);

        // Redirect flips (scattered single-word writes) — only those whose
        // target slot is uploaded as of this sync; the rest stay pending.
        // The geometry/table copies above are in the same submission, so a
        // flip and its slot data land atomically for the next cull.
        let new_synced = from + new_slots.len() as u32;
        let mut applied_any = false;
        let mut still_pending = Vec::new();
        for (id, slot) in std::mem::take(&mut self.pending_redirects) {
            if slot.0 < new_synced {
                let word = [slot.0];
                self.record_copy(&mut builder, &word, &self.redirect_buf, id.0);
                self.cpu_redirect[id.0 as usize] = slot.0;
                applied_any = true;
            } else {
                still_pending.push((id, slot));
            }
        }
        self.pending_redirects = still_pending;

        self.submit_and_wait(builder.build().expect("build sync CB"));

        self.cpu_table.extend_from_slice(&table_entries);
        self.vertex_used = v_cursor;
        self.index_used = i_cursor;
        self.synced_slots = new_synced;
        self.adapt_upload_caps(new_slots.len(), t0.elapsed().as_secs_f64());

        (
            grew || applied_any || !new_slots.is_empty(),
            self.slot_totals(&refcounts),
        )
    }

    /// Rescale the adaptive upload caps from a sync's measured wall time:
    /// aim for ~80% of the time budget (headroom for jitter), with the
    /// step clamped to [½×, 2×] per sync so one outlier (e.g. a grow's
    /// extra fence wait) can't collapse or explode the rate. Syncs that
    /// uploaded nothing carry no timing signal and leave the caps alone.
    fn adapt_upload_caps(&mut self, uploaded_slots: usize, elapsed_secs: f64) {
        if uploaded_slots == 0 {
            return;
        }
        let budget = UPLOAD_FRAME_SHARE / STREAM_MIN_FPS;
        let f = (0.8 * budget / elapsed_secs.max(1e-6)).clamp(0.5, 2.0);
        self.upload_slots_cap = ((self.upload_slots_cap as f64 * f) as usize)
            .clamp(UPLOAD_SLOTS_CAP_RANGE.0, UPLOAD_SLOTS_CAP_RANGE.1);
        self.upload_bytes_cap = ((self.upload_bytes_cap as f64 * f) as usize)
            .clamp(UPLOAD_BYTES_CAP_RANGE.0, UPLOAD_BYTES_CAP_RANGE.1);
    }

    /// Per-slot instance totals against the **GPU-visible** redirect (the
    /// `cpu_redirect` mirror): for each mesh id, its refcount accrues to
    /// the slot the cull will actually resolve it to this frame. Length is
    /// the uploaded slot count, matching `cpu_table` / the draw plan.
    fn slot_totals(&self, refcounts: &[u32]) -> Vec<u32> {
        let mut totals = vec![0u32; self.synced_slots as usize];
        for (id, &rc) in refcounts.iter().enumerate() {
            let slot = self
                .cpu_redirect
                .get(id)
                .copied()
                .unwrap_or(MeshSlot::PLACEHOLDER.0);
            totals[slot as usize] += rc;
        }
        totals
    }

    // ── Accessors for the cull / draw pipeline ──────────────────────────

    pub fn redirect_buffer(&self) -> &Subbuffer<[u32]> {
        &self.redirect_buf
    }
    pub fn mesh_table_buffer(&self) -> &Subbuffer<[MeshTableEntry]> {
        &self.table_buf
    }
    /// Per drawable slot: raw base-color [`TextureId`] or [`NO_TEXTURE`].
    /// Bound in graphics set 1 (fragment texture lookup by `gl_DrawID`).
    pub fn slot_texture_buffer(&self) -> &Subbuffer<[u32]> {
        &self.slot_texture_buf
    }
    pub fn mega_vertex_buffer(&self) -> &Subbuffer<[GpuVertex]> {
        &self.mega_vertex
    }
    pub fn mega_index_buffer(&self) -> &Subbuffer<[u32]> {
        &self.mega_index
    }
    /// Number of drawable slots uploaded so far (→ indirect-command sizing).
    pub fn slot_count(&self) -> u32 {
        self.synced_slots
    }

    /// CPU-side geometry for a drawable slot — the mega-buffer offsets and
    /// index count the camera bakes into a `DrawIndexedIndirectCommand`.
    /// `None` if the slot hasn't been synced yet.
    pub fn slot_geometry(&self, slot: u32) -> Option<MeshTableEntry> {
        self.cpu_table.get(slot as usize).copied()
    }

    // ── Internal: capacity growth ───────────────────────────────────────

    fn ensure_vertex_cap(&mut self, needed: u32) -> bool {
        if needed <= self.vertex_cap {
            return false;
        }
        let new_cap = self.vertex_cap.saturating_mul(2).max(needed);
        let old = self.mega_vertex.clone();
        self.mega_vertex = self.grow_buffer(
            &old,
            self.vertex_used as u64,
            new_cap,
            BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
        );
        self.vertex_cap = new_cap;
        true
    }

    fn ensure_index_cap(&mut self, needed: u32) -> bool {
        if needed <= self.index_cap {
            return false;
        }
        let new_cap = self.index_cap.saturating_mul(2).max(needed);
        let old = self.mega_index.clone();
        self.mega_index = self.grow_buffer(
            &old,
            self.index_used as u64,
            new_cap,
            BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
        );
        self.index_cap = new_cap;
        true
    }

    fn ensure_table_cap(&mut self, needed: u32) -> bool {
        if needed <= self.table_cap {
            return false;
        }
        let new_cap = self.table_cap.saturating_mul(2).max(needed);
        let old = self.table_buf.clone();
        self.table_buf = self.grow_buffer(
            &old,
            self.synced_slots as u64,
            new_cap,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
        );
        // The per-slot texture buffer grows in lock-step (same indexing).
        // Only slots < synced_slots are ever read, so the un-copied tail
        // needs no sentinel fill — sync writes each new slot's entry.
        let old_tex = self.slot_texture_buf.clone();
        self.slot_texture_buf = self.grow_buffer(
            &old_tex,
            self.synced_slots as u64,
            new_cap,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
        );
        self.table_cap = new_cap;
        true
    }

    fn ensure_redirect_cap(&mut self, needed: u32) -> bool {
        if needed <= self.redirect_cap {
            return false;
        }
        let new_cap = self.redirect_cap.saturating_mul(2).max(needed);
        // Zero-fill the grown tail so not-yet-resolved ids keep reading as
        // PLACEHOLDER (slot 0).
        let old = self.redirect_buf.clone();
        let old_used = self.redirect_cap as u64;
        let new = alloc_device::<u32>(
            &self.memory_allocator,
            new_cap,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
        );
        let mut builder = self.primary_builder();
        builder
            .fill_buffer(new.clone(), MeshSlot::PLACEHOLDER.0)
            .expect("zero-fill grown redirect buffer");
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                old.slice(0..old_used),
                new.clone().slice(0..old_used),
            ))
            .expect("copy old redirect into grown buffer");
        self.submit_and_wait(builder.build().expect("build redirect grow CB"));
        self.redirect_buf = new;
        self.redirect_cap = new_cap;
        true
    }

    /// Allocate a larger device buffer and copy the first `old_used` elements
    /// of `old` into it. The tail beyond `old_used` is written by the caller
    /// (geometry / table) or never read (unused slots).
    fn grow_buffer<T: BufferContents>(
        &self,
        old: &Subbuffer<[T]>,
        old_used: u64,
        new_cap: u32,
        usage: BufferUsage,
    ) -> Subbuffer<[T]> {
        let new = alloc_device::<T>(&self.memory_allocator, new_cap, usage);
        if old_used > 0 {
            let mut builder = self.primary_builder();
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    old.clone().slice(0..old_used),
                    new.clone().slice(0..old_used),
                ))
                .expect("grow copy");
            self.submit_and_wait(builder.build().expect("build grow CB"));
        }
        new
    }

    // ── Internal: upload plumbing ───────────────────────────────────────

    /// Stage `data` into a host-visible buffer and record a copy into
    /// `dst[dst_offset..]`. Empty slices are a no-op (Vulkan rejects
    /// zero-sized buffers). Offsets are in **elements**.
    fn record_copy<T: BufferContents + Clone>(
        &self,
        builder: &mut PrimaryBuilder,
        data: &[T],
        dst: &Subbuffer<[T]>,
        dst_offset: u32,
    ) {
        if data.is_empty() {
            return;
        }
        let staging = self.stage(data);
        let len = staging.len();
        let off = dst_offset as u64;
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                staging,
                dst.clone().slice(off..off + len),
            ))
            .expect("record staging→device copy");
    }

    fn stage<T: BufferContents + Clone>(&self, data: &[T]) -> Subbuffer<[T]> {
        Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.iter().cloned(),
        )
        .expect("create staging buffer")
    }

    fn primary_builder(&self) -> PrimaryBuilder {
        AutoCommandBufferBuilder::primary(
            self.cb_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("create one-shot primary command buffer")
    }

    fn submit_and_wait<C>(&self, cb: Arc<C>)
    where
        C: vulkano::command_buffer::PrimaryCommandBufferAbstract + 'static,
    {
        vulkano::sync::now(self.queue.device().clone())
            .then_execute(self.queue.clone(), cb)
            .expect("submit one-shot upload")
            .then_signal_fence_and_flush()
            .expect("flush one-shot upload")
            .wait(None)
            .expect("await one-shot upload");
    }
}

/// Per-slot mega-buffer placement computed during a sync.
struct SlotPlacement {
    vertex_offset: u32,
    first_index: u32,
    vcount: u32,
    icount: u32,
    bounds: MeshBounds,
}

/// Builder type for the store's one-shot upload command buffers.
type PrimaryBuilder = AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>;

fn alloc_device<T: BufferContents>(
    allocator: &Arc<StandardMemoryAllocator>,
    count: u32,
    usage: BufferUsage,
) -> Subbuffer<[T]> {
    Buffer::new_slice::<T>(
        allocator.clone(),
        BufferCreateInfo {
            usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        count as u64,
    )
    .expect("allocate device-local buffer")
}

fn to_gpu_verts(mesh: &Mesh) -> Vec<GpuVertex> {
    mesh.vertices.iter().copied().map(GpuVertex::from).collect()
}
