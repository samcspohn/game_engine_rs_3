//! Device-local GPU mirror of [`engine_core::texture`].
//!
//! [`GpuTextureStore`] owns one sampled image per resolved texture slot, a
//! single shared linear sampler, and the redirect buffer (`TextureId →
//! texture slot`) the fragment shader reads. [`GpuTextureStore::sync`]
//! drains the core registry's deltas — newly decoded slots and redirect
//! changes — and uploads them via host-staging + copy, exactly mirroring
//! [`super::GpuMeshStore`].
//!
//! # Descriptor model
//!
//! The graphics pipeline's set 1 binds the whole store as a **fixed-size
//! array** of [`MAX_TEXTURES`] combined image samplers (see
//! `shaders/scene.frag`); unused elements are bound to the placeholder
//! view. Indexing is dynamically uniform per draw, so only the
//! `shader_sampled_image_array_dynamic_indexing` feature is required — no
//! descriptor-indexing extension. A texture arrival flips `sync` to
//! `changed`, which rides the existing `mesh_changed → force_full` rebuild
//! (descriptor set + scene secondary + frame slots) — rare by construction.
//!
//! # Synchronization
//!
//! Uploads are **one-shot fence-waited submits** (rare path — once per
//! decoded texture), matching the mesh store's first-slice model. Images
//! are write-once: a slot's pixels never change after upload, so no
//! in-frame hazard exists once the set is rebuilt.

use std::sync::Arc;

use engine_core::texture::{self, TextureData, TextureSlot};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, CopyBufferToImageInfo,
    },
    device::Queue,
    format::Format,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::GpuFuture,
};

/// Size of the fragment shader's `sampler2D` array — **must match** the
/// `u_textures[…]` declaration in `shaders/scene.frag`. Exceeding it is a
/// loud panic (no silent eviction); bump both together when needed.
pub const MAX_TEXTURES: u32 = 1024;

const INITIAL_REDIRECT_CAP: u32 = 64;

/// GPU-resident mirror of the core texture registry.
pub struct GpuTextureStore {
    /// One view per uploaded texture slot (index == [`TextureSlot`]).
    views: Vec<Arc<ImageView>>,
    /// Shared trilinear-free (no mips yet) repeat sampler.
    sampler: Arc<Sampler>,
    /// `texture_id → slot` as raw `u32`s. Zero-filled so unresolved ids read
    /// as [`TextureSlot::PLACEHOLDER`] (slot 0).
    redirect_buf: Subbuffer<[u32]>,
    redirect_cap: u32,
    /// Number of core registry slots already uploaded (sync watermark).
    synced_slots: u32,

    memory_allocator: Arc<StandardMemoryAllocator>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
}

impl GpuTextureStore {
    /// Allocate the redirect buffer (zero-filled). The placeholder / error
    /// textures resident in the core registry upload on the first
    /// [`sync`](Self::sync), like any other slots.
    pub fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        cb_allocator: Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
    ) -> Self {
        let sampler = Sampler::new(
            queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .expect("create texture sampler");

        let redirect_buf = alloc_redirect(&memory_allocator, INITIAL_REDIRECT_CAP);
        let store = Self {
            views: Vec::new(),
            sampler,
            redirect_buf,
            redirect_cap: INITIAL_REDIRECT_CAP,
            synced_slots: 0,
            memory_allocator,
            cb_allocator,
            queue,
        };
        store.zero_fill(store.redirect_buf.clone());
        store
    }

    /// Drain the core registry's deltas — newly decoded slots and redirect
    /// changes — and upload them. Returns `true` if anything changed (the
    /// caller must rebuild the texture descriptor set + scene secondary,
    /// which the existing `force_full` path does).
    pub fn sync(&mut self) -> bool {
        let from = self.synced_slots;
        let (new_slots, redirect_updates, id_count): (
            Vec<Arc<TextureData>>,
            Vec<(texture::TextureId, TextureSlot)>,
            u32,
        ) = {
            let mut reg = texture::global()
                .lock()
                .expect("texture registry mutex poisoned");
            let new = (from..reg.slot_count()).map(|s| reg.slot(TextureSlot(s))).collect();
            (new, reg.take_redirect_updates(), reg.texture_id_count())
        };

        let needs_redirect_grow = id_count > self.redirect_cap;
        if new_slots.is_empty() && redirect_updates.is_empty() && !needs_redirect_grow {
            return false;
        }
        assert!(
            from as usize + new_slots.len() <= MAX_TEXTURES as usize,
            "texture slot count exceeds MAX_TEXTURES ({MAX_TEXTURES}) — \
             bump the constant and the scene.frag array size together"
        );

        if needs_redirect_grow {
            self.grow_redirect(id_count);
        }

        // One CB for every image upload + redirect patch this sync.
        let mut builder = AutoCommandBufferBuilder::primary(
            self.cb_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("create texture upload CB");

        for data in &new_slots {
            let image = Image::new(
                self.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    // Base-color maps are authored in sRGB; the view decodes
                    // to linear for the shader.
                    format: Format::R8G8B8A8_SRGB,
                    extent: [data.width, data.height, 1],
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .expect("allocate texture image");
            let staging = Buffer::from_iter(
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
                data.rgba8.iter().copied(),
            )
            .expect("create texture staging buffer");
            builder
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, image.clone()))
                .expect("record texture upload");
            self.views
                .push(ImageView::new_default(image).expect("create texture view"));
        }

        for (id, slot) in &redirect_updates {
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
            .expect("create redirect staging word");
            let off = id.0 as u64;
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    word,
                    self.redirect_buf.clone().slice(off..off + 1),
                ))
                .expect("record redirect patch");
        }

        self.submit_and_wait(builder.build().expect("build texture sync CB"));
        self.synced_slots = from + new_slots.len() as u32;
        true
    }

    /// The redirect buffer (`TextureId → slot`) for graphics set 1.
    pub fn redirect_buffer(&self) -> &Subbuffer<[u32]> {
        &self.redirect_buf
    }

    /// Exactly [`MAX_TEXTURES`] `(view, sampler)` elements for the
    /// fragment shader's fixed-size array: one per uploaded slot, the tail
    /// padded with the placeholder (slot 0) view.
    pub fn descriptor_array(&self) -> Vec<(Arc<ImageView>, Arc<Sampler>)> {
        assert!(
            !self.views.is_empty(),
            "descriptor_array before first sync — the placeholder must be uploaded first"
        );
        let placeholder = self.views[0].clone();
        (0..MAX_TEXTURES as usize)
            .map(|i| {
                (
                    self.views.get(i).cloned().unwrap_or_else(|| placeholder.clone()),
                    self.sampler.clone(),
                )
            })
            .collect()
    }

    // ── Internal ────────────────────────────────────────────────────────

    /// Grow the redirect buffer geometrically: zero-fill the new (so fresh
    /// ids read PLACEHOLDER) and copy the old ids over.
    fn grow_redirect(&mut self, needed: u32) {
        let new_cap = self.redirect_cap.saturating_mul(2).max(needed);
        let new = alloc_redirect(&self.memory_allocator, new_cap);
        let mut builder = AutoCommandBufferBuilder::primary(
            self.cb_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("create redirect grow CB");
        builder
            .fill_buffer(new.clone(), TextureSlot::PLACEHOLDER.0)
            .expect("zero-fill grown redirect");
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.redirect_buf.clone(),
                new.clone().slice(0..self.redirect_cap as u64),
            ))
            .expect("copy old redirect");
        self.submit_and_wait(builder.build().expect("build redirect grow CB"));
        self.redirect_buf = new;
        self.redirect_cap = new_cap;
    }

    fn zero_fill(&self, buf: Subbuffer<[u32]>) {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.cb_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("create zero-fill CB");
        builder
            .fill_buffer(buf, TextureSlot::PLACEHOLDER.0)
            .expect("zero-fill redirect");
        self.submit_and_wait(builder.build().expect("build zero-fill CB"));
    }

    fn submit_and_wait<C>(&self, cb: Arc<C>)
    where
        C: vulkano::command_buffer::PrimaryCommandBufferAbstract + 'static,
    {
        vulkano::sync::now(self.queue.device().clone())
            .then_execute(self.queue.clone(), cb)
            .expect("submit texture upload")
            .then_signal_fence_and_flush()
            .expect("flush texture upload")
            .wait(None)
            .expect("await texture upload");
    }
}

fn alloc_redirect(
    allocator: &Arc<StandardMemoryAllocator>,
    count: u32,
) -> Subbuffer<[u32]> {
    Buffer::new_slice::<u32>(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        count as u64,
    )
    .expect("allocate texture redirect buffer")
}
