//! Material asset registry — the GPU-agnostic source of truth.
//!
//! Same redirect model as [`crate::asset`] (meshes) and [`crate::texture`],
//! with one big simplification: materials are ~48 bytes of POD built straight
//! from glTF JSON / MTL text, so **there is no decode phase** — a
//! [`MaterialId`] resolves to its [`MaterialSlot`] the moment it's created.
//! No placeholder state, no background task, no failure mode of its own.
//! (The GPU mirror may still *display* the default material briefly: its
//! redirect buffer flips to a slot only after that slot's data is resident —
//! see `engine-render`'s `GpuMaterialStore`.)
//!
//! # Sharing, mutation, duplication
//!
//! * [`get_or_create`](MaterialRegistry::get_or_create) content-hash-dedups —
//!   importers calling it per primitive automatically share identical
//!   materials.
//! * [`update`](MaterialRegistry::update) edits a material **in place**: every
//!   renderer referencing the id sees the change (that's the point of
//!   sharing). Updating evicts the id from the dedup cache — its content no
//!   longer matches the hash it was interned under.
//! * [`duplicate`](MaterialRegistry::duplicate) is the "edit just this one
//!   object" path: clone the data under a fresh id, re-point the renderer,
//!   then `update` the clone.
//!
//! # Texture references
//!
//! Materials reference textures by [`TextureId`], never by texture slot. The
//! GPU material struct stores the raw id and the fragment shader resolves it
//! through the texture redirect buffer — so a streaming texture never forces
//! a material re-upload, and a material created before its texture decodes
//! samples the white placeholder until the redirect lands.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::texture::TextureId;

// ─────────────────────────────────────────────────────────────────────────────
// Identifiers
// ─────────────────────────────────────────────────────────────────────────────

/// Stable, write-once handle held by a consumer (a mesh slot's authored
/// material, or a renderer's explicit override). Indexes the redirect map.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MaterialId(pub u32);

impl MaterialId {
    /// The engine default material (warm orange, rough, untextured) — what
    /// untextured geometry has always drawn as. Permanently id 0 / slot 0.
    pub const DEFAULT: MaterialId = MaterialId(0);
}

/// Physical material slot — indexes the retained [`MaterialData`] (and, on
/// the GPU side, the material SSBO). Slot 0 is permanently the default
/// material.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MaterialSlot(pub u32);

impl MaterialSlot {
    /// Slot of [`MaterialId::DEFAULT`]; also what the GPU redirect buffer
    /// resolves to for any id whose slot hasn't been uploaded yet.
    pub const DEFAULT: MaterialSlot = MaterialSlot(0);
}

// ─────────────────────────────────────────────────────────────────────────────
// MaterialData
// ─────────────────────────────────────────────────────────────────────────────

/// CPU material description. Plain values plus optional texture references;
/// factors multiply their texture (glTF semantics), so a texture-less
/// material is just its factors and an untextured surface tints by
/// `base_color`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MaterialData {
    /// RGBA base-color factor; multiplies `base_color_tex` when present.
    pub base_color: [f32; 4],
    /// Metallic factor in `[0, 1]`.
    pub metallic: f32,
    /// Perceptual roughness factor in `[0, 1]`.
    pub roughness: f32,
    /// RGB emissive factor (added after lighting).
    pub emissive: [f32; 3],
    /// Base-color (albedo) texture, if any.
    pub base_color_tex: Option<TextureId>,
}

impl Default for MaterialData {
    /// The engine default: the warm orange untextured look the renderer has
    /// always used, fully rough and non-metallic.
    fn default() -> Self {
        Self {
            base_color: [0.85, 0.55, 0.20, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            base_color_tex: None,
        }
    }
}

impl MaterialData {
    /// Content hash over exact bit patterns — the dedup-cache key. Two
    /// materials share an id iff every field is bit-identical.
    fn content_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.base_color.map(f32::to_bits).hash(&mut h);
        self.metallic.to_bits().hash(&mut h);
        self.roughness.to_bits().hash(&mut h);
        self.emissive.map(f32::to_bits).hash(&mut h);
        self.base_color_tex.map(|t| t.0).hash(&mut h);
        h.finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MaterialRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-agnostic material registry. See the module docs for the model.
pub struct MaterialRegistry {
    /// Dedup cache: content hash → interned id. Ids leave the cache when
    /// [`update`](Self::update)d (their content no longer matches).
    by_hash: HashMap<u64, MaterialId>,
    /// The hash each id is currently interned under (`None` once updated or
    /// for never-interned ids from [`create`](Self::create) /
    /// [`duplicate`](Self::duplicate)).
    interned_hash: Vec<Option<u64>>,
    /// `material_id → slot`. Fixed at creation — materials resolve
    /// immediately; there is no placeholder state to flip out of.
    redirect: Vec<MaterialSlot>,
    /// Reference count per id (reclamation deferred, like meshes/textures).
    refcount: Vec<u32>,
    /// Retained data per slot; mutated in place by [`update`](Self::update).
    slots: Vec<MaterialData>,
    /// Newly-created ids since the last
    /// [`take_redirect_updates`](Self::take_redirect_updates) drain — the GPU
    /// mirror flips its redirect entry once the slot's data is resident.
    dirty_redirect: Vec<MaterialId>,
    /// Slots whose data changed via [`update`](Self::update) since the last
    /// [`take_dirty_slots`](Self::take_dirty_slots) drain.
    dirty_slots: Vec<MaterialSlot>,
}

impl MaterialRegistry {
    /// Build a registry holding only the default material (id 0 / slot 0).
    pub fn new() -> Self {
        let mut reg = Self {
            by_hash: HashMap::new(),
            interned_hash: Vec::new(),
            redirect: Vec::new(),
            refcount: Vec::new(),
            slots: Vec::new(),
            dirty_redirect: Vec::new(),
            dirty_slots: Vec::new(),
        };
        let (id, _) = reg.get_or_create(MaterialData::default());
        assert_eq!(id, MaterialId::DEFAULT, "default material must be id 0");
        assert_eq!(reg.redirect[0], MaterialSlot::DEFAULT);
        // Slot 0 is always resident on the GPU (the redirect buffer's fill
        // value) — no flip to defer.
        reg.dirty_redirect.clear();
        reg
    }

    /// Intern `data`: returns the existing id when bit-identical content is
    /// already registered (refcount bump), otherwise a fresh id. The importer
    /// entry point — sharing across primitives/files is automatic.
    pub fn get_or_create(&mut self, data: MaterialData) -> (MaterialId, bool) {
        let hash = data.content_hash();
        if let Some(&id) = self.by_hash.get(&hash) {
            self.refcount[id.0 as usize] += 1;
            return (id, false);
        }
        let id = self.alloc(data);
        self.by_hash.insert(hash, id);
        self.interned_hash[id.0 as usize] = Some(hash);
        (id, true)
    }

    /// Always-fresh id for `data`, never interned in the dedup cache — for
    /// user-managed materials that expect to be edited.
    pub fn create(&mut self, data: MaterialData) -> MaterialId {
        self.alloc(data)
    }

    /// Clone `id`'s current data under a fresh, never-interned id — the
    /// "detach this object's material so I can edit it alone" path.
    pub fn duplicate(&mut self, id: MaterialId) -> MaterialId {
        let data = self.slots[self.redirect[id.0 as usize].0 as usize];
        self.alloc(data)
    }

    /// Replace `id`'s data in place. Every renderer referencing the id (or a
    /// mesh slot authored with it) sees the change. Evicts the id from the
    /// dedup cache. Texture refcounts are not adjusted (reclamation is
    /// deferred engine-wide).
    pub fn update(&mut self, id: MaterialId, data: MaterialData) {
        let slot = self.redirect[id.0 as usize];
        self.slots[slot.0 as usize] = data;
        self.dirty_slots.push(slot);
        if let Some(hash) = self.interned_hash[id.0 as usize].take() {
            self.by_hash.remove(&hash);
        }
    }

    fn alloc(&mut self, data: MaterialData) -> MaterialId {
        let slot = MaterialSlot(self.slots.len() as u32);
        self.slots.push(data);
        let id = MaterialId(self.redirect.len() as u32);
        self.redirect.push(slot);
        self.refcount.push(1);
        self.interned_hash.push(None);
        self.dirty_redirect.push(id);
        id
    }

    /// Add one reference to an already-allocated id.
    pub fn retain(&mut self, id: MaterialId) {
        self.refcount[id.0 as usize] += 1;
    }

    /// Drop one reference. Slot reclamation on zero is deferred.
    pub fn release(&mut self, id: MaterialId) {
        let rc = &mut self.refcount[id.0 as usize];
        debug_assert!(*rc > 0, "release of MaterialId({}) with zero refcount", id.0);
        *rc = rc.saturating_sub(1);
    }

    // ── Reads (GPU mirror) ──────────────────────────────────────────────

    /// The slot `id` resolves to (fixed at creation).
    pub fn slot_of(&self, id: MaterialId) -> MaterialSlot {
        self.redirect[id.0 as usize]
    }

    /// Current data for a slot.
    pub fn slot(&self, slot: MaterialSlot) -> MaterialData {
        self.slots[slot.0 as usize]
    }

    /// Number of material slots.
    pub fn slot_count(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Number of allocated ids (→ redirect buffer sizing).
    pub fn material_id_count(&self) -> u32 {
        self.redirect.len() as u32
    }

    /// Reference count for an id.
    pub fn refcount_of(&self, id: MaterialId) -> u32 {
        self.refcount[id.0 as usize]
    }

    /// Drain the `(MaterialId, MaterialSlot)` pairs created since the last
    /// call. The GPU mirror applies each flip only once the slot's data is
    /// resident (until then the id draws as the default material).
    pub fn take_redirect_updates(&mut self) -> Vec<(MaterialId, MaterialSlot)> {
        self.dirty_redirect
            .drain(..)
            .map(|id| (id, self.redirect[id.0 as usize]))
            .collect()
    }

    /// Drain the slots mutated via [`update`](Self::update) since the last
    /// call. The GPU mirror re-uploads any that were already resident (a
    /// not-yet-uploaded slot's pending initial upload reads current data
    /// anyway).
    pub fn take_dirty_slots(&mut self) -> Vec<MaterialSlot> {
        std::mem::take(&mut self.dirty_slots)
    }
}

impl Default for MaterialRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global instance
// ─────────────────────────────────────────────────────────────────────────────

static REGISTRY: OnceLock<Mutex<MaterialRegistry>> = OnceLock::new();

/// The process-wide material registry, lazily initialized with the default
/// material on first access.
pub fn global() -> &'static Mutex<MaterialRegistry> {
    REGISTRY.get_or_init(|| Mutex::new(MaterialRegistry::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn red() -> MaterialData {
        MaterialData {
            base_color: [1.0, 0.0, 0.0, 1.0],
            ..MaterialData::default()
        }
    }

    #[test]
    fn default_material_is_id_and_slot_zero() {
        let reg = MaterialRegistry::new();
        assert_eq!(reg.slot_of(MaterialId::DEFAULT), MaterialSlot::DEFAULT);
        assert_eq!(reg.slot_count(), 1);
        assert_eq!(reg.slot(MaterialSlot::DEFAULT), MaterialData::default());
    }

    #[test]
    fn get_or_create_dedups_by_content() {
        let mut reg = MaterialRegistry::new();
        let (a, new_a) = reg.get_or_create(red());
        assert!(new_a);
        let (b, new_b) = reg.get_or_create(red());
        assert_eq!(a, b);
        assert!(!new_b);
        assert_eq!(reg.refcount_of(a), 2);
        // Redirect updates carry the newly created id exactly once.
        assert_eq!(reg.take_redirect_updates(), vec![(a, reg.slot_of(a))]);
        assert!(reg.take_redirect_updates().is_empty());
    }

    #[test]
    fn default_data_dedups_to_default_id() {
        let mut reg = MaterialRegistry::new();
        let (id, new) = reg.get_or_create(MaterialData::default());
        assert_eq!(id, MaterialId::DEFAULT);
        assert!(!new);
    }

    #[test]
    fn update_mutates_in_place_and_leaves_dedup_cache() {
        let mut reg = MaterialRegistry::new();
        let (a, _) = reg.get_or_create(red());
        let slot = reg.slot_of(a);
        let mut edited = red();
        edited.roughness = 0.25;
        reg.update(a, edited);
        assert_eq!(reg.slot(slot), edited);
        assert_eq!(reg.take_dirty_slots(), vec![slot]);
        // The edited id no longer answers for its old content: interning the
        // original red gets a fresh id.
        let (b, new_b) = reg.get_or_create(red());
        assert_ne!(a, b);
        assert!(new_b);
    }

    #[test]
    fn duplicate_detaches_for_solo_editing() {
        let mut reg = MaterialRegistry::new();
        let (a, _) = reg.get_or_create(red());
        let dup = reg.duplicate(a);
        assert_ne!(a, dup);
        assert_ne!(reg.slot_of(a), reg.slot_of(dup));
        assert_eq!(reg.slot(reg.slot_of(dup)), red());
        // Editing the duplicate leaves the original untouched.
        let mut edited = red();
        edited.metallic = 1.0;
        reg.update(dup, edited);
        assert_eq!(reg.slot(reg.slot_of(a)), red());
        // The duplicate was never interned — dedup still finds the original.
        let (c, _) = reg.get_or_create(red());
        assert_eq!(c, a);
    }

    #[test]
    fn create_never_interns() {
        let mut reg = MaterialRegistry::new();
        let a = reg.create(red());
        let (b, new_b) = reg.get_or_create(red());
        assert_ne!(a, b);
        assert!(new_b, "create()d materials must not answer dedup lookups");
    }
}
