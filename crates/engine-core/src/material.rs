//! Material asset registry — the GPU-agnostic source of truth.
//!
//! Same redirect model as [`crate::asset`] (meshes) and [`crate::texture`],
//! with one big simplification: materials are ~64 bytes of POD built straight
//! from glTF JSON / MTL text, so **there is no decode phase** — a
//! [`MaterialId`] resolves to its [`MaterialSlot`] the moment it's created.
//! No placeholder state, no background task, no failure mode of its own.
//! (The GPU mirror may still *display* the default material briefly: its
//! redirect buffer flips to a slot only after that slot's data is resident —
//! see `engine-render`'s `GpuMaterialStore`.)
//!
//! A material is also a [`Resource`]: a `.mat` file holds one, and editing
//! it updates its slot in place, so every renderer on it follows.
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

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use parking_lot::Mutex;

use crate::reflect::{AssetKind, AssetRef, Export};
use crate::registry::Registry;
use crate::resource::{self, Res, ResId, Resource};
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

/// CPU material description — the metallic-roughness PBR parameter set.
/// Plain values plus optional texture references; factors multiply their
/// texture (glTF semantics), so a texture-less material is just its factors
/// and an untextured surface tints by `base_color`.
///
/// Texture color spaces follow glTF: `base_color_tex` / `emissive_tex` are
/// sRGB-encoded, the rest ([`normal_tex`](Self::normal_tex),
/// [`metallic_roughness_tex`](Self::metallic_roughness_tex),
/// [`occlusion_tex`](Self::occlusion_tex)) carry raw linear data. Importers
/// request each with the matching [`crate::texture::ColorSpace`].
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, Export)]
/// Absent means the default, so a file carries only what was authored and an
/// older one still reads once a field is added.
#[serde(default)]
pub struct MaterialData {
    /// RGBA base-color factor; multiplies `base_color_tex` when present.
    #[export]
    pub base_color: [f32; 4],
    /// Metallic factor in `[0, 1]`; multiplies the B channel of
    /// `metallic_roughness_tex`.
    #[export]
    pub metallic: f32,
    /// Perceptual roughness factor in `[0, 1]`; multiplies the G channel of
    /// `metallic_roughness_tex`.
    #[export]
    pub roughness: f32,
    /// RGB emissive factor (added after lighting); multiplies `emissive_tex`.
    #[export]
    pub emissive: [f32; 3],
    /// Scales the tangent-space XY of `normal_tex` (glTF `normalTexture.scale`).
    #[export]
    pub normal_scale: f32,
    /// Blend toward unoccluded for `occlusion_tex` (glTF `occlusionTexture.strength`).
    #[export]
    pub occlusion_strength: f32,
    /// Base-color (albedo) texture, if any.
    #[export]
    pub base_color_tex: Option<TextureId>,
    /// Tangent-space normal map, if any.
    #[export]
    pub normal_tex: Option<TextureId>,
    /// Packed metallic-roughness map (G = roughness, B = metallic), if any.
    #[export]
    pub metallic_roughness_tex: Option<TextureId>,
    /// Ambient-occlusion map (R channel), if any.
    #[export]
    pub occlusion_tex: Option<TextureId>,
    /// Emissive map, if any.
    #[export]
    pub emissive_tex: Option<TextureId>,
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
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            base_color_tex: None,
            normal_tex: None,
            metallic_roughness_tex: None,
            occlusion_tex: None,
            emissive_tex: None,
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
        self.normal_scale.to_bits().hash(&mut h);
        self.occlusion_strength.to_bits().hash(&mut h);
        for tex in [
            self.base_color_tex,
            self.normal_tex,
            self.metallic_roughness_tex,
            self.occlusion_tex,
            self.emissive_tex,
        ] {
            tex.map(|t| t.0).hash(&mut h);
        }
        h.finish()
    }
}

impl Resource for MaterialData {
    const NAME: &'static str = "Material";
    const EXTENSION: &'static str = "mat";
    const KIND: AssetKind = AssetKind::Material;

    fn reference(id: ResId) -> AssetRef {
        AssetRef::Material(of_file(id))
    }

    fn edited(id: ResId) {
        if let Some((path, data)) = file(id) {
            global().lock().refresh_file(&path, data);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MaterialRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// What an id answers for: a file, or content an importer interned.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    Content(u64),
    File(PathBuf),
}

/// GPU-agnostic material registry. See the module docs for the model.
pub struct MaterialRegistry {
    /// Materials resolve the moment they exist, so an id is never pointed
    /// anywhere but its own slot.
    reg: Registry<MaterialId, MaterialSlot, Key, MaterialData>,
    /// Slots whose data changed since the last
    /// [`take_dirty_slots`](Self::take_dirty_slots) drain.
    dirty_slots: Vec<MaterialSlot>,
}

impl MaterialRegistry {
    /// Build a registry holding only the default material (id 0 / slot 0).
    pub fn new() -> Self {
        let mut reg = Self {
            reg: Registry::new(),
            dirty_slots: Vec::new(),
        };
        let (id, _) = reg.get_or_create(MaterialData::default());
        assert_eq!(id, MaterialId::DEFAULT, "default material must be id 0");
        assert_eq!(reg.slot_of(id), MaterialSlot::DEFAULT);
        // Slot 0 is always resident on the GPU (the redirect buffer's fill
        // value) — no flip to defer.
        reg.reg.take_redirect_updates();
        reg
    }

    /// Intern `data`: returns the existing id when bit-identical content is
    /// already registered (refcount bump), otherwise a fresh id. The importer
    /// entry point — sharing across primitives/files is automatic.
    pub fn get_or_create(&mut self, data: MaterialData) -> (MaterialId, bool) {
        let key = Key::Content(data.content_hash());
        self.keyed(key, data)
    }

    /// The id `key` answers for, made from `data` if there is none.
    fn keyed(&mut self, key: Key, data: MaterialData) -> (MaterialId, bool) {
        if let Some(id) = self.reg.find(&key) {
            self.reg.retain(id);
            return (id, false);
        }
        (self.made(Some(key), data), true)
    }

    /// A new id on a new slot. Pointed rather than inserted at it, so the GPU
    /// mirror flips the id only once the slot's data is resident.
    fn made(&mut self, key: Option<Key>, data: MaterialData) -> MaterialId {
        let slot = self.reg.alloc(data);
        let id = self.reg.insert(key, MaterialSlot::DEFAULT);
        self.reg.point(id, slot);
        id
    }

    /// Always-fresh id for `data`, never interned in the dedup cache — for
    /// user-managed materials that expect to be edited.
    pub fn create(&mut self, data: MaterialData) -> MaterialId {
        self.made(None, data)
    }

    /// Clone `id`'s current data under a fresh, never-interned id — the
    /// "detach this object's material so I can edit it alone" path.
    pub fn duplicate(&mut self, id: MaterialId) -> MaterialId {
        let data = *self.reg.slot(self.slot_of(id));
        self.create(data)
    }

    /// Replace `id`'s data in place. Every renderer referencing the id (or a
    /// mesh slot authored with it) sees the change. Evicts an interned id
    /// from the dedup cache, since its content no longer matches. Texture
    /// refcounts are not adjusted (reclamation is deferred engine-wide).
    pub fn update(&mut self, id: MaterialId, data: MaterialData) {
        self.write(id, data);
        if matches!(self.reg.key_of(id), Some(Key::Content(_))) {
            self.reg.unkey(id);
        }
    }

    fn write(&mut self, id: MaterialId, data: MaterialData) {
        let slot = self.slot_of(id);
        *self.reg.slot_mut(slot) = data;
        self.dirty_slots.push(slot);
    }

    /// The file `id` was read from, if it was.
    pub fn path_of(&self, id: MaterialId) -> Option<&Path> {
        match self.reg.key_of(id)? {
            Key::File(path) => Some(path),
            Key::Content(_) => None,
        }
    }

    /// The material file at `path` holds `data` now.
    fn refresh_file(&mut self, path: &Path, data: MaterialData) {
        if let Some(id) = self.reg.find(&Key::File(path.to_path_buf())) {
            self.write(id, data);
        }
    }

    /// Add one reference to an already-allocated id.
    pub fn retain(&mut self, id: MaterialId) {
        self.reg.retain(id);
    }

    /// Drop one reference. Slot reclamation on zero is deferred.
    pub fn release(&mut self, id: MaterialId) {
        self.reg.release(id);
    }

    // ── Reads (GPU mirror) ──────────────────────────────────────────────

    /// The slot `id` resolves to (fixed at creation).
    pub fn slot_of(&self, id: MaterialId) -> MaterialSlot {
        self.reg.redirect_of(id)
    }

    /// Current data for a slot.
    pub fn slot(&self, slot: MaterialSlot) -> MaterialData {
        *self.reg.slot(slot)
    }

    /// Number of material slots.
    pub fn slot_count(&self) -> u32 {
        self.reg.slot_count()
    }

    /// Number of allocated ids (→ redirect buffer sizing).
    pub fn material_id_count(&self) -> u32 {
        self.reg.id_count()
    }

    /// Reference count for an id.
    pub fn refcount_of(&self, id: MaterialId) -> u32 {
        self.reg.refcount_of(id)
    }

    /// Drain the `(MaterialId, MaterialSlot)` pairs created since the last
    /// call. The GPU mirror applies each flip only once the slot's data is
    /// resident (until then the id draws as the default material).
    pub fn take_redirect_updates(&mut self) -> Vec<(MaterialId, MaterialSlot)> {
        self.reg.take_redirect_updates()
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

/// The material file at `path`, as an id a renderer can take — with the
/// reference it will hold.
pub fn request_path(path: &Path) -> MaterialId {
    of_file(Res::<MaterialData>::request(path).id())
}

/// The id resource `id` feeds, with one more reference.
fn of_file(id: ResId) -> MaterialId {
    let (path, data) = file(id).expect("only a material resource feeds a material id");
    global().lock().keyed(Key::File(path), data).0
}

/// Read before the registry is locked: a resource is read under its own.
fn file(id: ResId) -> Option<(PathBuf, MaterialData)> {
    let data = Res::<MaterialData>::checked(id)?.with(|d| *d);
    Some((resource::path_of(id)?, data))
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

    /// A material file is one id however often it is asked for, and an edit
    /// to the file lands in that id's slot for the GPU mirror to re-upload.
    #[test]
    fn a_material_file_feeds_its_slot() {
        let path = std::env::temp_dir().join(format!("engine_mat_{}.mat", std::process::id()));
        let id = request_path(&path);
        assert_eq!(request_path(&path), id);
        assert_eq!(global().lock().path_of(id), Some(path.as_path()));
        global().lock().take_dirty_slots();

        let res = Res::<MaterialData>::request(&path).id();
        resource::edit(res, |c, t| c.set("roughness", crate::Value::F32(0.25), t));
        let slot = global().lock().slot_of(id);
        assert_eq!(global().lock().slot(slot).roughness, 0.25);
        assert!(global().lock().take_dirty_slots().contains(&slot));
        assert_eq!(
            crate::asset_type::reference(&path),
            Some(AssetRef::Material(id)),
            "a .mat file drops as the material it feeds"
        );
    }
}
