//! Device-side mesh asset registry: owns the mega vertex/index buffers, the
//! drawable-slot table, and the redirect map as **device-local** buffers, and
//! keeps them in sync with a [`MeshCatalog`] via staging→copy uploads.
//!
//! # Why device-local + staging
//!
//! These buffers are read by the per-frame cull / draw pipeline at full VRAM
//! bandwidth, so they live in `PREFER_DEVICE` memory. Host data reaches them
//! by being written into a host-visible staging buffer and `vkCmdCopyBuffer`d
//! into the destination region. New geometry always lands in a previously
//! unused region of the mega buffers (append-only), so the copy never races a
//! draw reading existing geometry.
//!
//! # Synchronization (first slice)
//!
//! Uploads here currently use **one-shot fence-waited submits** — correct and
//! self-contained, but blocking. Asset loads complete off the hot path (once
//! per *unique* asset, deduped), so this is acceptable to start. Frame-loop
//! integration will replace these with copies recorded into a between-frames
//! transfer command buffer (growth regions are safe any time; a redirect flip
//! that repoints a *live* `MeshId` must land in the per-frame safe window,
//! exactly like the transform staging→SoT promotion).

use std::path::Path;
use std::sync::Arc;

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

use super::{bounding_sphere, hash_path, MeshCatalog, MeshId, MeshSlot, MeshTableEntry};
use crate::gpu_mesh::GpuVertex;

// Initial capacities. All grow geometrically (≥ 2×) on demand.
const INITIAL_VERTEX_CAP: u32 = 1 << 12;
const INITIAL_INDEX_CAP: u32 = 1 << 13;
const INITIAL_TABLE_CAP: u32 = 8;
const INITIAL_REDIRECT_CAP: u32 = 64;

/// GPU-resident mesh registry. Wraps a [`MeshCatalog`] (CPU source of truth)
/// and mirrors its state into device-local buffers.
pub struct MeshRegistry {
    catalog: MeshCatalog,

    // ── Mega geometry buffers (device-local, append-only) ───────────────
    mega_vertex: Subbuffer<[GpuVertex]>,
    mega_index: Subbuffer<[u32]>,
    vertex_cap: u32,
    index_cap: u32,

    // ── Per-slot table + redirect map (device-local mirrors) ────────────
    table_buf: Subbuffer<[MeshTableEntry]>,
    table_cap: u32,
    /// `mesh_id → slot` as raw `u32`s. Allocated zero-filled so any
    /// not-yet-resolved id reads as [`MeshSlot::PLACEHOLDER`] (slot 0).
    redirect_buf: Subbuffer<[u32]>,
    redirect_cap: u32,

    // ── Allocators / queue for uploads ──────────────────────────────────
    memory_allocator: Arc<StandardMemoryAllocator>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
}

impl MeshRegistry {
    /// Build the registry with the placeholder (slot 0) and error (slot 1)
    /// meshes uploaded and resident. Their geometry occupies the head of the
    /// mega buffers; the redirect buffer is zero-filled (every future id
    /// defaults to the placeholder until resolved).
    pub fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        cb_allocator: Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
        placeholder: &Mesh,
        error: &Mesh,
    ) -> Self {
        let ph_verts = to_gpu_verts(placeholder);
        let er_verts = to_gpu_verts(error);
        let (ph_c, ph_r) = bounding_sphere(placeholder);
        let (er_c, er_r) = bounding_sphere(error);

        let need_v = ph_verts.len() as u32 + er_verts.len() as u32;
        let need_i = placeholder.indices.len() as u32 + error.indices.len() as u32;
        let vertex_cap = INITIAL_VERTEX_CAP.max(need_v.next_power_of_two().max(1));
        let index_cap = INITIAL_INDEX_CAP.max(need_i.next_power_of_two().max(1));
        let table_cap = INITIAL_TABLE_CAP;
        let redirect_cap = INITIAL_REDIRECT_CAP;

        let mega_vertex = alloc_device::<GpuVertex>(
            &memory_allocator,
            vertex_cap,
            BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
        );
        let mega_index = alloc_device::<u32>(
            &memory_allocator,
            index_cap,
            BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
        );
        let table_buf = alloc_device::<MeshTableEntry>(
            &memory_allocator,
            table_cap,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
        );
        let redirect_buf = alloc_device::<u32>(
            &memory_allocator,
            redirect_cap,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
        );

        let mut catalog = MeshCatalog::new();
        let ph_p = catalog.alloc_slot(
            ph_verts.len() as u32,
            placeholder.indices.len() as u32,
            ph_c,
            ph_r,
        );
        let er_p = catalog.alloc_slot(
            er_verts.len() as u32,
            error.indices.len() as u32,
            er_c,
            er_r,
        );
        assert_eq!(
            ph_p.slot,
            MeshSlot::PLACEHOLDER,
            "placeholder must be slot 0"
        );
        assert_eq!(er_p.slot, MeshSlot::ERROR, "error must be slot 1");

        let reg = Self {
            catalog,
            mega_vertex,
            mega_index,
            vertex_cap,
            index_cap,
            table_buf,
            table_cap,
            redirect_buf,
            redirect_cap,
            memory_allocator,
            cb_allocator,
            queue,
        };

        // Single init CB: zero the redirect buffer, upload reserved geometry
        // and the two table entries.
        let table_entries = [reg.catalog.table()[0], reg.catalog.table()[1]];
        let mut builder = reg.primary_builder();
        builder
            .fill_buffer(reg.redirect_buf.clone(), MeshSlot::PLACEHOLDER.0)
            .expect("zero-fill redirect buffer");
        reg.record_copy(
            &mut builder,
            &ph_verts,
            &reg.mega_vertex,
            ph_p.vertex_offset,
        );
        reg.record_copy(
            &mut builder,
            &er_verts,
            &reg.mega_vertex,
            er_p.vertex_offset,
        );
        reg.record_copy(
            &mut builder,
            &placeholder.indices,
            &reg.mega_index,
            ph_p.first_index,
        );
        reg.record_copy(
            &mut builder,
            &error.indices,
            &reg.mega_index,
            er_p.first_index,
        );
        reg.record_copy(&mut builder, &table_entries, &reg.table_buf, 0);
        reg.submit_and_wait(builder.build().expect("build registry init CB"));

        reg
    }

    /// Deduped request for `path`. Returns its stable [`MeshId`] and whether
    /// the caller must kick an async load (`true` only on the first request of
    /// a path). A freshly allocated id already resolves to the placeholder via
    /// the zero-filled redirect buffer, so no GPU write happens here unless
    /// the redirect buffer has to grow.
    pub fn request(&mut self, path: &Path) -> (MeshId, bool) {
        let hash = hash_path(path);
        let old_ids = self.catalog.mesh_id_count();
        let (id, needs_load) = self.catalog.request(hash);
        if needs_load {
            self.ensure_redirect_cap(self.catalog.mesh_id_count(), old_ids);
        }
        (id, needs_load)
    }

    /// A load finished: append the geometry to the mega buffers, record the
    /// new drawable slot's table entry, and flip `redirect[id]` to it.
    /// Returns whether a backing buffer grew (a future caller will use this to
    /// rebuild dependent command buffers).
    pub fn resolve(&mut self, id: MeshId, mesh: &Mesh) -> bool {
        let gpu_verts = to_gpu_verts(mesh);
        let vcount = gpu_verts.len() as u32;
        let icount = mesh.indices.len() as u32;
        let (center, radius) = bounding_sphere(mesh);

        // Capture the pre-resolve cursors so growth copies preserve exactly
        // the geometry/table already on the GPU.
        let old_v = self.catalog.vertex_used();
        let old_i = self.catalog.index_used();
        let old_slots = self.catalog.slot_count();

        let placement = self.catalog.resolve(id, vcount, icount, center, radius);

        let grew_v = self.ensure_vertex_cap(self.catalog.vertex_used(), old_v);
        let grew_i = self.ensure_index_cap(self.catalog.index_used(), old_i);
        let grew_t = self.ensure_table_cap(self.catalog.slot_count(), old_slots);

        let table_entry = [self.catalog.table()[placement.slot.0 as usize]];
        let redirect_word = [placement.slot.0];

        let mut builder = self.primary_builder();
        if vcount > 0 {
            self.record_copy(
                &mut builder,
                &gpu_verts,
                &self.mega_vertex,
                placement.vertex_offset,
            );
        }
        if icount > 0 {
            self.record_copy(
                &mut builder,
                &mesh.indices,
                &self.mega_index,
                placement.first_index,
            );
        }
        self.record_copy(
            &mut builder,
            &table_entry,
            &self.table_buf,
            placement.slot.0,
        );
        self.record_copy(&mut builder, &redirect_word, &self.redirect_buf, id.0);
        self.submit_and_wait(builder.build().expect("build resolve CB"));

        grew_v || grew_i || grew_t
    }

    /// A load failed: flip `redirect[id]` to the error slot. No geometry is
    /// uploaded (the error mesh is already resident at slot 1).
    pub fn fail(&mut self, id: MeshId) {
        self.catalog.fail(id);
        let redirect_word = [MeshSlot::ERROR.0];
        let mut builder = self.primary_builder();
        self.record_copy(&mut builder, &redirect_word, &self.redirect_buf, id.0);
        self.submit_and_wait(builder.build().expect("build fail CB"));
    }

    /// Drop one reference to a `MeshId`.
    pub fn release(&mut self, id: MeshId) {
        self.catalog.release(id);
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
    /// Number of drawable slots (→ indirect-command array sizing).
    pub fn slot_count(&self) -> u32 {
        self.catalog.slot_count()
    }
    /// Number of allocated `MeshId`s (→ redirect buffer sizing).
    pub fn mesh_id_count(&self) -> u32 {
        self.catalog.mesh_id_count()
    }
    /// Current drawable slot a `MeshId` resolves to.
    pub fn redirect_of(&self, id: MeshId) -> MeshSlot {
        self.catalog.redirect_of(id)
    }

    // ── Internal: capacity growth ───────────────────────────────────────

    fn ensure_vertex_cap(&mut self, needed: u32, old_used: u32) -> bool {
        if needed <= self.vertex_cap {
            return false;
        }
        let new_cap = self.vertex_cap.saturating_mul(2).max(needed);
        let old = self.mega_vertex.clone();
        self.mega_vertex = self.grow_buffer(
            &old,
            old_used as u64,
            new_cap,
            BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
        );
        self.vertex_cap = new_cap;
        true
    }

    fn ensure_index_cap(&mut self, needed: u32, old_used: u32) -> bool {
        if needed <= self.index_cap {
            return false;
        }
        let new_cap = self.index_cap.saturating_mul(2).max(needed);
        let old = self.mega_index.clone();
        self.mega_index = self.grow_buffer(
            &old,
            old_used as u64,
            new_cap,
            BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
        );
        self.index_cap = new_cap;
        true
    }

    fn ensure_table_cap(&mut self, needed: u32, old_used: u32) -> bool {
        if needed <= self.table_cap {
            return false;
        }
        let new_cap = self.table_cap.saturating_mul(2).max(needed);
        let old = self.table_buf.clone();
        self.table_buf = self.grow_buffer(
            &old,
            old_used as u64,
            new_cap,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
        );
        self.table_cap = new_cap;
        true
    }

    fn ensure_redirect_cap(&mut self, needed: u32, old_used: u32) -> bool {
        if needed <= self.redirect_cap {
            return false;
        }
        let new_cap = self.redirect_cap.saturating_mul(2).max(needed);
        // The redirect buffer must zero-fill its grown tail so not-yet-
        // resolved ids keep reading as PLACEHOLDER (slot 0). `grow_buffer`
        // leaves the tail undefined, so use a dedicated zero-filling grow.
        let old = self.redirect_buf.clone();
        let new = alloc_device::<u32>(
            &self.memory_allocator,
            new_cap,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
        );
        let mut builder = self.primary_builder();
        builder
            .fill_buffer(new.clone(), MeshSlot::PLACEHOLDER.0)
            .expect("zero-fill grown redirect buffer");
        if old_used > 0 {
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    old.slice(0..old_used as u64),
                    new.clone().slice(0..old_used as u64),
                ))
                .expect("copy old redirect into grown buffer");
        }
        self.submit_and_wait(builder.build().expect("build redirect grow CB"));
        self.redirect_buf = new;
        self.redirect_cap = new_cap;
        true
    }

    /// Allocate a larger device buffer and copy the first `old_used` elements
    /// of `old` into it. The tail beyond `old_used` is left undefined — every
    /// caller either writes it immediately (geometry / table) or never reads
    /// it (unused slots).
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

    /// Stage `data` into a host-visible buffer and record a copy of it into
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

/// Builder type for the registry's one-shot upload command buffers.
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
