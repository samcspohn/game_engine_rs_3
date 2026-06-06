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

    /// Number of core registry slots already uploaded (sync watermark).
    synced_slots: u32,

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
            BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
        );
        let mega_index = alloc_device::<u32>(
            &memory_allocator,
            INITIAL_INDEX_CAP,
            BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
        );
        let table_buf = alloc_device::<MeshTableEntry>(
            &memory_allocator,
            INITIAL_TABLE_CAP,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
        );
        let redirect_buf = alloc_device::<u32>(
            &memory_allocator,
            INITIAL_REDIRECT_CAP,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
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
            synced_slots: 0,
            memory_allocator,
            cb_allocator,
            queue,
        };

        // Zero the redirect buffer so unresolved ids default to PLACEHOLDER.
        let mut builder = store.primary_builder();
        builder
            .fill_buffer(store.redirect_buf.clone(), MeshSlot::PLACEHOLDER.0)
            .expect("zero-fill redirect buffer");
        store.submit_and_wait(builder.build().expect("build init CB"));

        store
    }

    /// Drain the core registry's deltas — newly resolved slots and redirect
    /// changes — and upload them. Returns whether a backing buffer grew (a
    /// future caller will use this to rebuild dependent command buffers).
    ///
    /// Briefly locks the global registry to clone out the new `Arc<Mesh>`es +
    /// bounds and the redirect updates, then releases it before doing GPU work
    /// so a background `resolve` is never blocked on a copy.
    pub fn sync(&mut self) -> bool {
        let from = self.synced_slots;
        let (new_slots, redirect_updates, mesh_id_count): (
            Vec<(Arc<Mesh>, MeshBounds)>,
            Vec<(engine_core::asset::MeshId, MeshSlot)>,
            u32,
        ) = {
            let mut reg = asset::global()
                .lock()
                .expect("asset registry mutex poisoned");
            let slot_count = reg.slot_count();
            let new = (from..slot_count).map(|s| reg.slot(MeshSlot(s))).collect();
            let updates = reg.take_redirect_updates();
            let mid = reg.mesh_id_count();
            (new, updates, mid)
        };

        let needs_redirect_grow = mesh_id_count > self.redirect_cap;
        if new_slots.is_empty() && redirect_updates.is_empty() && !needs_redirect_grow {
            return false;
        }

        // Assign mega-buffer offsets for the new slots (render-side concern).
        let mut placements = Vec::with_capacity(new_slots.len());
        let mut v_cursor = self.vertex_used;
        let mut i_cursor = self.index_used;
        for (mesh, bounds) in &new_slots {
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

        // Geometry + the contiguous block of new table entries.
        let mut table_entries = Vec::with_capacity(new_slots.len());
        for ((mesh, _bounds), p) in new_slots.iter().zip(placements.iter()) {
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
        }
        self.record_copy(&mut builder, &table_entries, &self.table_buf, from);

        // Redirect flips (scattered single-word writes).
        for (id, slot) in &redirect_updates {
            let word = [slot.0];
            self.record_copy(&mut builder, &word, &self.redirect_buf, id.0);
        }

        self.submit_and_wait(builder.build().expect("build sync CB"));

        self.vertex_used = v_cursor;
        self.index_used = i_cursor;
        self.synced_slots = from + new_slots.len() as u32;

        grew
    }

    // ── Accessors for the cull / draw pipeline ──────────────────────────

    pub fn redirect_buffer(&self) -> &Subbuffer<[u32]> {
        &self.redirect_buf
    }
    pub fn mesh_table_buffer(&self) -> &Subbuffer<[MeshTableEntry]> {
        &self.table_buf
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
            BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
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
            BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
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
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
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
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
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
