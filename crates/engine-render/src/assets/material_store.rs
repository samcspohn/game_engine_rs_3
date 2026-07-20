//! Device-local GPU mirror of [`engine_core::material`].
//!
//! [`GpuMaterialStore`] owns the material SSBO (one 48-byte [`GpuMaterial`]
//! per slot) and the redirect buffer (`MaterialId → material slot`) the
//! fragment shader reads. [`GpuMaterialStore::sync`] drains the core
//! registry's deltas — newly created slots, in-place edits
//! (`MaterialRegistry::update`), and redirect entries — and uploads them via
//! host-staging + copy, mirroring [`super::GpuTextureStore`].
//!
//! Unlike meshes and textures there is **no streaming time budget**:
//! materials are tiny POD (even 10k of them is < 0.5 MB), so every sync
//! uploads everything pending. The deferred-redirect watermark is kept
//! purely for ordering correctness — a redirect entry flips only in the same
//! submission that makes its slot's data resident, so an id never resolves
//! to garbage; until then it reads 0 (the engine default material).
//!
//! Texture references inside a material are raw [`TextureId`] words — the
//! fragment shader double-indirects through the texture store's redirect —
//! so a streaming texture never forces a material re-upload.

use engine_core::material::{self, MaterialData, MaterialSlot};
use std::sync::Arc;
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

/// Sentinel for "material has no base-color texture" — matches `NO_TEXTURE`
/// in `shaders/scene.frag`.
pub const NO_TEXTURE: u32 = u32::MAX;

const INITIAL_MATERIAL_CAP: u32 = 64;
const INITIAL_REDIRECT_CAP: u32 = 64;

/// std430 mirror of [`MaterialData`] — **must match** the `Material` struct
/// in `shaders/scene.frag` (48 bytes).
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
pub struct GpuMaterial {
    base_color: [f32; 4],
    emissive: [f32; 3],
    roughness: f32,
    metallic: f32,
    /// Raw [`engine_core::TextureId`] or [`NO_TEXTURE`].
    base_color_tex: u32,
    _pad: [u32; 2],
}

impl From<MaterialData> for GpuMaterial {
    fn from(d: MaterialData) -> Self {
        Self {
            base_color: d.base_color,
            emissive: d.emissive,
            roughness: d.roughness,
            metallic: d.metallic,
            base_color_tex: d.base_color_tex.map_or(NO_TEXTURE, |t| t.0),
            _pad: [0; 2],
        }
    }
}

/// GPU-resident mirror of the core material registry.
pub struct GpuMaterialStore {
    /// One [`GpuMaterial`] per slot (index == [`MaterialSlot`]).
    materials_buf: Subbuffer<[GpuMaterial]>,
    materials_cap: u32,
    /// `material_id → slot` as raw `u32`s. Zero-filled so ids whose slot
    /// isn't resident yet read as [`MaterialSlot::DEFAULT`] (slot 0).
    redirect_buf: Subbuffer<[u32]>,
    redirect_cap: u32,
    /// Number of core registry slots already uploaded (sync watermark).
    synced_slots: u32,
    /// Redirect flips awaiting their slot's upload (ordering only —
    /// normally applied in the very same sync that uploads the slot).
    pending_redirects: Vec<(material::MaterialId, MaterialSlot)>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
}

impl GpuMaterialStore {
    /// Allocate the material + redirect buffers (zero-filled). The default
    /// material resident in the core registry uploads on the first
    /// [`sync`](Self::sync) like any other slot.
    pub fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        cb_allocator: Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
    ) -> Self {
        let materials_buf = alloc_materials(&memory_allocator, INITIAL_MATERIAL_CAP);
        let redirect_buf = alloc_u32(&memory_allocator, INITIAL_REDIRECT_CAP);
        let store = Self {
            materials_buf,
            materials_cap: INITIAL_MATERIAL_CAP,
            redirect_buf,
            redirect_cap: INITIAL_REDIRECT_CAP,
            synced_slots: 0,
            pending_redirects: Vec::new(),
            memory_allocator,
            cb_allocator,
            queue,
        };
        store.zero_fill(store.redirect_buf.clone());
        store
    }

    /// Drain the core registry's deltas and upload them all (materials are
    /// too small to need pacing). Returns `true` if anything changed — the
    /// caller must rebuild the descriptor set + scene secondary, which the
    /// existing `force_full` path does.
    pub fn sync(&mut self) -> bool {
        let from = self.synced_slots;
        let (new_slots, dirty_data, redirect_updates, id_count): (
            Vec<MaterialData>,
            Vec<(MaterialSlot, MaterialData)>,
            Vec<(material::MaterialId, MaterialSlot)>,
            u32,
        ) = {
            let mut reg = material::global()
                .lock()
                .expect("material registry mutex poisoned");
            let new = (from..reg.slot_count())
                .map(|s| reg.slot(MaterialSlot(s)))
                .collect();
            let dirty = reg
                .take_dirty_slots()
                .into_iter()
                // Slots at/above the watermark upload below with current
                // data anyway; only resident slots need a patch.
                .filter(|s| s.0 < from)
                .map(|s| (s, reg.slot(s)))
                .collect();
            (
                new,
                dirty,
                reg.take_redirect_updates(),
                reg.material_id_count(),
            )
        };
        self.pending_redirects.extend(redirect_updates);

        let needs_redirect_grow = id_count > self.redirect_cap;
        let new_synced = from + new_slots.len() as u32;
        let needs_material_grow = new_synced > self.materials_cap;
        if new_slots.is_empty()
            && dirty_data.is_empty()
            && self.pending_redirects.is_empty()
            && !needs_redirect_grow
            && !needs_material_grow
        {
            return false;
        }

        if needs_redirect_grow {
            self.grow_redirect(id_count);
        }
        if needs_material_grow {
            self.grow_materials(new_synced);
        }

        // One CB for every upload + patch this sync.
        let mut builder = AutoCommandBufferBuilder::primary(
            self.cb_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("create material upload CB");

        if !new_slots.is_empty() {
            let staging = staging_from(&self.memory_allocator, new_slots.iter().copied());
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    staging,
                    self.materials_buf
                        .clone()
                        .slice(from as u64..new_synced as u64),
                ))
                .expect("record material slot upload");
        }
        for (slot, data) in &dirty_data {
            let staging = staging_from(&self.memory_allocator, [*data].into_iter());
            let off = slot.0 as u64;
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    staging,
                    self.materials_buf.clone().slice(off..off + 1),
                ))
                .expect("record material edit patch");
        }

        // Apply the flips whose slot is resident as of this sync (with the
        // upload-everything policy that is normally all of them; the guard
        // is ordering insurance, mirroring the mesh/texture stores).
        let mut still_pending = Vec::new();
        let mut applied_any = false;
        for (id, slot) in std::mem::take(&mut self.pending_redirects) {
            if slot.0 >= new_synced {
                still_pending.push((id, slot));
                continue;
            }
            let word = Buffer::from_iter(
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
                [slot.0],
            )
            .expect("create material redirect staging word");
            let off = id.0 as u64;
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    word,
                    self.redirect_buf.clone().slice(off..off + 1),
                ))
                .expect("record material redirect patch");
            applied_any = true;
        }
        self.pending_redirects = still_pending;

        if !new_slots.is_empty() || !dirty_data.is_empty() || applied_any {
            self.submit_and_wait(builder.build().expect("build material sync CB"));
        }
        self.synced_slots = new_synced;
        true
    }

    /// The redirect buffer (`MaterialId → slot`) for graphics set 1.
    pub fn redirect_buffer(&self) -> &Subbuffer<[u32]> {
        &self.redirect_buf
    }

    /// The material SSBO for graphics set 1.
    pub fn materials_buffer(&self) -> &Subbuffer<[GpuMaterial]> {
        &self.materials_buf
    }

    // ── Internal ────────────────────────────────────────────────────────

    /// Grow the redirect buffer geometrically: zero-fill the new (so fresh
    /// ids read the default material) and copy the old entries over.
    fn grow_redirect(&mut self, needed: u32) {
        let new_cap = self.redirect_cap.saturating_mul(2).max(needed);
        let new = alloc_u32(&self.memory_allocator, new_cap);
        let mut builder = AutoCommandBufferBuilder::primary(
            self.cb_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("create material redirect grow CB");
        builder
            .fill_buffer(new.clone(), MaterialSlot::DEFAULT.0)
            .expect("zero-fill grown material redirect");
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.redirect_buf.clone(),
                new.clone().slice(0..self.redirect_cap as u64),
            ))
            .expect("copy old material redirect");
        self.submit_and_wait(builder.build().expect("build material redirect grow CB"));
        self.redirect_buf = new;
        self.redirect_cap = new_cap;
    }

    /// Grow the material SSBO geometrically, copying resident slots over.
    fn grow_materials(&mut self, needed: u32) {
        let new_cap = self.materials_cap.saturating_mul(2).max(needed);
        let new = alloc_materials(&self.memory_allocator, new_cap);
        if self.synced_slots > 0 {
            let mut builder = AutoCommandBufferBuilder::primary(
                self.cb_allocator.clone(),
                self.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .expect("create material grow CB");
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    self.materials_buf.clone().slice(0..self.synced_slots as u64),
                    new.clone().slice(0..self.synced_slots as u64),
                ))
                .expect("copy old materials");
            self.submit_and_wait(builder.build().expect("build material grow CB"));
        }
        self.materials_buf = new;
        self.materials_cap = new_cap;
    }

    fn zero_fill(&self, buf: Subbuffer<[u32]>) {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.cb_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("create material zero-fill CB");
        builder
            .fill_buffer(buf, MaterialSlot::DEFAULT.0)
            .expect("zero-fill material redirect");
        self.submit_and_wait(builder.build().expect("build material zero-fill CB"));
    }

    fn submit_and_wait<C>(&self, cb: Arc<C>)
    where
        C: vulkano::command_buffer::PrimaryCommandBufferAbstract + 'static,
    {
        vulkano::sync::now(self.queue.device().clone())
            .then_execute(self.queue.clone(), cb)
            .expect("submit material upload")
            .then_signal_fence_and_flush()
            .expect("flush material upload")
            .wait(None)
            .expect("await material upload");
    }
}

fn alloc_materials(
    allocator: &Arc<StandardMemoryAllocator>,
    count: u32,
) -> Subbuffer<[GpuMaterial]> {
    Buffer::new_slice::<GpuMaterial>(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_DST
                | BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        count.max(1) as u64,
    )
    .expect("allocate material buffer")
}

fn alloc_u32(allocator: &Arc<StandardMemoryAllocator>, count: u32) -> Subbuffer<[u32]> {
    Buffer::new_slice::<u32>(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_DST
                | BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        count as u64,
    )
    .expect("allocate material redirect buffer")
}

fn staging_from(
    allocator: &Arc<StandardMemoryAllocator>,
    items: impl ExactSizeIterator<Item = MaterialData>,
) -> Subbuffer<[GpuMaterial]> {
    Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        items.map(GpuMaterial::from),
    )
    .expect("create material staging buffer")
}
