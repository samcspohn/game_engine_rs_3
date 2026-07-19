//! Texture asset registry — the GPU-agnostic source of truth.
//!
//! Deliberately the **same redirect model as [`crate::asset`]** (meshes): a
//! consumer holds a stable, write-once [`TextureId`]; the registry maps it
//! through a redirect table to a [`TextureSlot`]:
//!
//! * while the asset decodes, the id resolves to [`TextureSlot::PLACEHOLDER`]
//!   (a 1×1 white pixel — textured surfaces show their base shading);
//! * once resolved, to the real slot;
//! * on failure, to [`TextureSlot::ERROR`] (a loud magenta/black
//!   checkerboard — per the project's no-silent-fallback rule).
//!
//! Load completion is a single redirect write; no consumer record is ever
//! patched. `engine-render`'s `GpuTextureStore` mirrors resolved slots into
//! device images and the redirect table into a device buffer the fragment
//! shader reads.
//!
//! # Sources
//!
//! * **Files** ([`request_load`]): PNG / JPEG referenced by OBJ `.mtl`
//!   `map_Kd` entries or `.gltf` image URIs.
//! * **Embedded bytes** ([`resolve_bytes_task`]-style tasks spawned by
//!   [`crate::scene_asset`]): GLB bufferView images and `data:` URIs.
//!
//! All decodes run as pool background tasks (same deferral rules as mesh
//! loads — see [`crate::asset::spawn_when_pool_ready`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

// ─────────────────────────────────────────────────────────────────────────────
// Identifiers
// ─────────────────────────────────────────────────────────────────────────────

/// Stable, write-once handle held by a consumer (a mesh slot's base-color
/// reference). Allocated per unique requested path (deduped) and indexes the
/// redirect map.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TextureId(pub u32);

/// Physical texture slot — indexes the retained pixel data (and, on the GPU
/// side, the descriptor array of sampled images). Slots `0` and `1` are
/// permanently reserved.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TextureSlot(pub u32);

impl TextureSlot {
    /// Resolved-to by any `TextureId` whose asset is still decoding.
    pub const PLACEHOLDER: TextureSlot = TextureSlot(0);
    /// Resolved-to by any `TextureId` whose decode failed.
    pub const ERROR: TextureSlot = TextureSlot(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// TextureData
// ─────────────────────────────────────────────────────────────────────────────

/// Decoded CPU pixels: always tightly-packed RGBA8, row-major, top-left
/// origin. `rgba8.len() == width * height * 4`.
#[derive(Clone, Debug)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

impl TextureData {
    /// A `width × height` fill of one RGBA color.
    pub fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Self {
        Self {
            width,
            height,
            rgba8: rgba.repeat((width * height) as usize),
        }
    }
}

/// Default placeholder texture: a single white pixel, so a still-loading
/// base-color map multiplies to the surface's untinted shading.
pub fn placeholder_texture() -> TextureData {
    TextureData::solid(1, 1, [0xFF, 0xFF, 0xFF, 0xFF])
}

/// Default error texture: an 8×8 magenta/black checkerboard — deliberately
/// impossible to miss, mirroring the error mesh's distinct silhouette.
pub fn error_texture() -> TextureData {
    const N: u32 = 8;
    let mut rgba8 = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        for x in 0..N {
            let magenta = (x + y) % 2 == 0;
            rgba8.extend_from_slice(if magenta {
                &[0xFF, 0x00, 0xFF, 0xFF]
            } else {
                &[0x00, 0x00, 0x00, 0xFF]
            });
        }
    }
    TextureData {
        width: N,
        height: N,
        rgba8,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TextureRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-agnostic texture registry. See the module docs for the redirect model.
pub struct TextureRegistry {
    /// Dedup cache: path hash → already-allocated `TextureId`.
    by_hash: HashMap<u64, TextureId>,
    /// `texture_id → slot`. New ids default to [`TextureSlot::PLACEHOLDER`];
    /// `resolve`/`fail` repoint them.
    redirect: Vec<TextureSlot>,
    /// Reference count per `TextureId` (reclamation deferred, like meshes).
    refcount: Vec<u32>,
    /// Retained pixels per slot.
    slots: Vec<Arc<TextureData>>,
    /// `TextureId`s whose redirect entry changed since the last
    /// [`take_redirect_updates`](Self::take_redirect_updates) drain.
    dirty_redirect: Vec<TextureId>,
}

impl TextureRegistry {
    /// Build a registry with `placeholder` at slot 0 and `error` at slot 1.
    pub fn new(placeholder: Arc<TextureData>, error: Arc<TextureData>) -> Self {
        let mut reg = Self {
            by_hash: HashMap::new(),
            redirect: Vec::new(),
            refcount: Vec::new(),
            slots: Vec::new(),
            dirty_redirect: Vec::new(),
        };
        let ph = reg.alloc_slot(placeholder);
        let er = reg.alloc_slot(error);
        assert_eq!(ph, TextureSlot::PLACEHOLDER, "placeholder must be slot 0");
        assert_eq!(er, TextureSlot::ERROR, "error must be slot 1");
        reg
    }

    /// Build with the engine's default placeholder / error textures.
    pub fn with_defaults() -> Self {
        Self::new(Arc::new(placeholder_texture()), Arc::new(error_texture()))
    }

    fn alloc_slot(&mut self, data: Arc<TextureData>) -> TextureSlot {
        let slot = TextureSlot(self.slots.len() as u32);
        self.slots.push(data);
        slot
    }

    /// Deduped request for `path` (real file or virtual sub-asset path such
    /// as `scene.glb#image0`). Cache miss → fresh placeholder-pointing id +
    /// `needs_load = true`; hit → refcount bump.
    pub fn request(&mut self, path: &Path) -> (TextureId, bool) {
        let hash = hash_path(path);
        if let Some(&id) = self.by_hash.get(&hash) {
            self.refcount[id.0 as usize] += 1;
            return (id, false);
        }
        let id = TextureId(self.redirect.len() as u32);
        self.redirect.push(TextureSlot::PLACEHOLDER);
        self.refcount.push(1);
        self.by_hash.insert(hash, id);
        (id, true)
    }

    /// A decode finished: retain the pixels in a fresh slot and flip
    /// `redirect[id]` to it.
    pub fn resolve(&mut self, id: TextureId, data: Arc<TextureData>) -> TextureSlot {
        let slot = self.alloc_slot(data);
        self.redirect[id.0 as usize] = slot;
        self.dirty_redirect.push(id);
        slot
    }

    /// A decode failed: point `redirect[id]` at the error slot.
    pub fn fail(&mut self, id: TextureId) {
        self.redirect[id.0 as usize] = TextureSlot::ERROR;
        self.dirty_redirect.push(id);
    }

    /// Add one reference to an already-allocated id (handle duplicated
    /// without a `request`).
    pub fn retain(&mut self, id: TextureId) {
        self.refcount[id.0 as usize] += 1;
    }

    /// Drop one reference. Slot reclamation on zero is deferred.
    pub fn release(&mut self, id: TextureId) {
        let rc = &mut self.refcount[id.0 as usize];
        debug_assert!(*rc > 0, "release of TextureId({}) with zero refcount", id.0);
        *rc = rc.saturating_sub(1);
    }

    // ── Reads (GPU mirror) ──────────────────────────────────────────────

    /// Current slot an id resolves to.
    pub fn redirect_of(&self, id: TextureId) -> TextureSlot {
        self.redirect[id.0 as usize]
    }

    /// Retained pixels for a slot (clones the `Arc`).
    pub fn slot(&self, slot: TextureSlot) -> Arc<TextureData> {
        self.slots[slot.0 as usize].clone()
    }

    /// Number of texture slots.
    pub fn slot_count(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Number of allocated ids (→ redirect buffer sizing).
    pub fn texture_id_count(&self) -> u32 {
        self.redirect.len() as u32
    }

    /// Reference count for an id.
    pub fn refcount_of(&self, id: TextureId) -> u32 {
        self.refcount[id.0 as usize]
    }

    /// Drain the `(TextureId, TextureSlot)` redirect changes accumulated
    /// since the last call. Consumed by the GPU mirror.
    pub fn take_redirect_updates(&mut self) -> Vec<(TextureId, TextureSlot)> {
        self.dirty_redirect
            .drain(..)
            .map(|id| (id, self.redirect[id.0 as usize]))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global instance
// ─────────────────────────────────────────────────────────────────────────────

static REGISTRY: OnceLock<Mutex<TextureRegistry>> = OnceLock::new();

/// The process-wide texture registry, lazily initialized with the default
/// placeholder/error textures on first access.
pub fn global() -> &'static Mutex<TextureRegistry> {
    REGISTRY.get_or_init(|| Mutex::new(TextureRegistry::with_defaults()))
}

// ─────────────────────────────────────────────────────────────────────────────
// Decode + async loaders
// ─────────────────────────────────────────────────────────────────────────────

/// Decode PNG/JPEG `bytes` into RGBA8 pixels. Shared by the file loader and
/// the embedded (GLB bufferView / `data:` URI) decode tasks.
pub fn decode_texture_bytes(bytes: &[u8]) -> Result<TextureData, String> {
    let img = image::load_from_memory(bytes).map_err(|e| format!("image decode error: {e}"))?;
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(TextureData {
        width,
        height,
        rgba8: rgba.into_raw(),
    })
}

/// Queue an asynchronous load+decode of the image file at `path` for
/// `texture_id`. Call once per id — when [`TextureRegistry::request`]
/// reports a new path. Same pool/deferral semantics as mesh loads.
pub fn request_load(texture_id: TextureId, path: impl Into<PathBuf>) {
    let path: PathBuf = path.into();
    crate::asset::spawn_when_pool_ready(move || {
        let decoded = std::fs::read(&path)
            .map_err(|e| format!("read error: {e}"))
            .and_then(|bytes| decode_texture_bytes(&bytes));
        finish(texture_id, decoded, &path.display().to_string());
    });
}

/// Queue an asynchronous decode of already-loaded `bytes` (an embedded GLB
/// image or a `data:` URI payload) for `texture_id`. `origin` names the
/// source in failure diagnostics.
pub fn request_decode_bytes(texture_id: TextureId, bytes: Arc<[u8]>, origin: String) {
    crate::asset::spawn_when_pool_ready(move || {
        finish(texture_id, decode_texture_bytes(&bytes), &origin);
    });
}

/// Resolve or fail `texture_id` from a decode result, loudly on failure.
fn finish(texture_id: TextureId, decoded: Result<TextureData, String>, origin: &str) {
    match decoded {
        Ok(data) => {
            global()
                .lock()
                .expect("texture registry mutex poisoned")
                .resolve(texture_id, Arc::new(data));
        }
        Err(e) => {
            eprintln!("texture load failed for {origin}: {e}");
            global()
                .lock()
                .expect("texture registry mutex poisoned")
                .fail(texture_id);
        }
    }
}

/// Hash a path to the `u64` dedup-cache key (same scheme as meshes).
fn hash_path(path: &Path) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> TextureRegistry {
        TextureRegistry::with_defaults()
    }

    #[test]
    fn reserved_slots_take_zero_and_one() {
        let reg = fresh();
        assert_eq!(reg.slot_count(), 2);
        assert_eq!(reg.texture_id_count(), 0);
        // Placeholder is a single white pixel; error is loud magenta.
        assert_eq!(reg.slot(TextureSlot::PLACEHOLDER).rgba8, vec![0xFF; 4]);
        assert_eq!(&reg.slot(TextureSlot::ERROR).rgba8[0..4], &[0xFF, 0, 0xFF, 0xFF]);
    }

    #[test]
    fn request_dedups_resolve_flips_redirect() {
        let mut reg = fresh();
        let (a, load_a) = reg.request(Path::new("a.png"));
        assert!(load_a);
        assert_eq!(reg.redirect_of(a), TextureSlot::PLACEHOLDER);
        let (a2, load_a2) = reg.request(Path::new("a.png"));
        assert_eq!(a, a2);
        assert!(!load_a2);
        assert_eq!(reg.refcount_of(a), 2);

        let slot = reg.resolve(a, Arc::new(TextureData::solid(2, 2, [1, 2, 3, 4])));
        assert_eq!(slot, TextureSlot(2), "first real texture lands in slot 2");
        assert_eq!(reg.redirect_of(a), slot);
        assert_eq!(reg.take_redirect_updates(), vec![(a, slot)]);
        assert!(reg.take_redirect_updates().is_empty());
    }

    #[test]
    fn fail_points_redirect_at_error_slot() {
        let mut reg = fresh();
        let (a, _) = reg.request(Path::new("missing.png"));
        reg.fail(a);
        assert_eq!(reg.redirect_of(a), TextureSlot::ERROR);
        assert_eq!(reg.slot_count(), 2, "failing allocates no new slot");
    }

    #[test]
    fn decode_png_bytes_round_trips() {
        // Encode a 2×1 RGBA PNG via the image crate, then decode it back.
        let pixels: [u8; 8] = [255, 0, 0, 255, 0, 255, 0, 255];
        let mut png: Vec<u8> = Vec::new();
        image::write_buffer_with_format(
            &mut std::io::Cursor::new(&mut png),
            &pixels,
            2,
            1,
            image::ExtendedColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .expect("encode test png");
        let data = decode_texture_bytes(&png).expect("decode");
        assert_eq!((data.width, data.height), (2, 1));
        assert_eq!(data.rgba8, pixels);
    }

    #[test]
    fn garbage_bytes_fail_loudly() {
        assert!(decode_texture_bytes(&[0xDE, 0xAD, 0xBE, 0xEF]).is_err());
    }
}
