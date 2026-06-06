//! Pure-CPU mesh bookkeeping — the source of truth the GPU mirrors track.
//!
//! [`MeshCatalog`] owns no GPU resources, so it is fully unit-testable
//! without a Vulkan device. [`super::MeshRegistry`] wraps it and replays its
//! mutations into device-local buffers.

use std::collections::HashMap;

use super::{MeshId, MeshSlot, MeshTableEntry};

/// Where a mesh's geometry was placed in the mega vertex/index buffers, plus
/// the drawable slot that now describes it. Returned by [`MeshCatalog::resolve`]
/// (and [`MeshCatalog::alloc_slot`]) so the registry knows exactly which
/// device-buffer regions to upload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    /// The drawable slot allocated for this mesh.
    pub slot: MeshSlot,
    /// First vertex of this mesh in the mega vertex buffer
    /// (== `DrawIndexedIndirectCommand::vertex_offset`).
    pub vertex_offset: u32,
    /// First index of this mesh in the mega index buffer
    /// (== `DrawIndexedIndirectCommand::first_index`).
    pub first_index: u32,
    /// Number of indices (== `DrawIndexedIndirectCommand::index_count`).
    pub index_count: u32,
}

/// CPU mirror of the asset registry's state.
pub struct MeshCatalog {
    /// Dedup cache: path hash → the `MeshId` already allocated for it.
    by_hash: HashMap<u64, MeshId>,
    /// `mesh_id → drawable slot`. New ids default to
    /// [`MeshSlot::PLACEHOLDER`]; `resolve`/`fail` repoint them.
    redirect: Vec<MeshSlot>,
    /// Reference count per `MeshId` (how many renderers requested this path).
    refcount: Vec<u32>,
    /// One entry per drawable slot. Indexed by [`MeshSlot`].
    table: Vec<MeshTableEntry>,
    /// Running append cursor into the mega vertex buffer.
    vertex_used: u32,
    /// Running append cursor into the mega index buffer.
    index_used: u32,
}

impl MeshCatalog {
    /// Empty catalog — no reserved slots yet. The registry calls
    /// [`alloc_slot`](Self::alloc_slot) twice at construction to install the
    /// placeholder (slot 0) and error (slot 1) meshes.
    pub fn new() -> Self {
        Self {
            by_hash: HashMap::new(),
            redirect: Vec::new(),
            refcount: Vec::new(),
            table: Vec::new(),
            vertex_used: 0,
            index_used: 0,
        }
    }

    /// Allocate the next drawable slot for `(vertex_count, index_count)` of
    /// geometry, advancing the mega-buffer cursors and recording the slot's
    /// [`MeshTableEntry`]. Returns where the geometry must be uploaded.
    ///
    /// This is the shared primitive behind both reserved-slot installation
    /// (placeholder / error) and [`resolve`](Self::resolve).
    pub fn alloc_slot(
        &mut self,
        vertex_count: u32,
        index_count: u32,
        bounds_center: [f32; 3],
        bounds_radius: f32,
    ) -> Placement {
        let slot = MeshSlot(self.table.len() as u32);
        let vertex_offset = self.vertex_used;
        let first_index = self.index_used;
        self.vertex_used += vertex_count;
        self.index_used += index_count;
        self.table.push(MeshTableEntry {
            index_count,
            first_index,
            vertex_offset: vertex_offset as i32,
            _pad0: 0,
            bounds_center,
            bounds_radius,
        });
        Placement {
            slot,
            vertex_offset,
            first_index,
            index_count,
        }
    }

    /// Deduped request keyed by a path hash. On a cache miss, allocate a new
    /// `MeshId` pointing at [`MeshSlot::PLACEHOLDER`] and return
    /// `needs_load = true`. On a hit, bump the refcount and return the
    /// existing id with `needs_load = false`.
    pub fn request(&mut self, hash: u64) -> (MeshId, bool) {
        if let Some(&id) = self.by_hash.get(&hash) {
            self.refcount[id.0 as usize] += 1;
            return (id, false);
        }
        let id = MeshId(self.redirect.len() as u32);
        self.redirect.push(MeshSlot::PLACEHOLDER);
        self.refcount.push(1);
        self.by_hash.insert(hash, id);
        (id, true)
    }

    /// A load finished: allocate a drawable slot for the geometry and repoint
    /// `redirect[id]` at it. Returns the geometry placement so the registry
    /// can upload it.
    pub fn resolve(
        &mut self,
        id: MeshId,
        vertex_count: u32,
        index_count: u32,
        bounds_center: [f32; 3],
        bounds_radius: f32,
    ) -> Placement {
        let placement = self.alloc_slot(vertex_count, index_count, bounds_center, bounds_radius);
        self.redirect[id.0 as usize] = placement.slot;
        placement
    }

    /// A load failed: point `redirect[id]` at the error slot.
    pub fn fail(&mut self, id: MeshId) {
        self.redirect[id.0 as usize] = MeshSlot::ERROR;
    }

    /// Drop one reference. Slot reclamation on zero is deferred (the mega
    /// buffers are append-only for now), so this only adjusts bookkeeping.
    pub fn release(&mut self, id: MeshId) {
        let rc = &mut self.refcount[id.0 as usize];
        debug_assert!(*rc > 0, "release of MeshId({}) with zero refcount", id.0);
        *rc = rc.saturating_sub(1);
    }

    // ── Read accessors ──────────────────────────────────────────────────

    /// Current drawable slot a `MeshId` resolves to.
    pub fn redirect_of(&self, id: MeshId) -> MeshSlot {
        self.redirect[id.0 as usize]
    }

    /// The full redirect mirror (`mesh_id → slot`).
    pub fn redirect(&self) -> &[MeshSlot] {
        &self.redirect
    }

    /// The full drawable-slot table.
    pub fn table(&self) -> &[MeshTableEntry] {
        &self.table
    }

    /// Number of drawable slots (→ indirect-command array sizing).
    pub fn slot_count(&self) -> u32 {
        self.table.len() as u32
    }

    /// Number of allocated `MeshId`s (→ redirect buffer sizing).
    pub fn mesh_id_count(&self) -> u32 {
        self.redirect.len() as u32
    }

    /// Vertices appended to the mega vertex buffer so far.
    pub fn vertex_used(&self) -> u32 {
        self.vertex_used
    }

    /// Indices appended to the mega index buffer so far.
    pub fn index_used(&self) -> u32 {
        self.index_used
    }

    /// Reference count for a `MeshId`.
    pub fn refcount_of(&self, id: MeshId) -> u32 {
        self.refcount[id.0 as usize]
    }
}

impl Default for MeshCatalog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_slots_take_zero_and_one() {
        let mut c = MeshCatalog::new();
        let ph = c.alloc_slot(10, 30, [0.0; 3], 1.0);
        assert_eq!(ph.slot, MeshSlot::PLACEHOLDER);
        assert_eq!(ph.vertex_offset, 0);
        assert_eq!(ph.first_index, 0);

        let er = c.alloc_slot(4, 12, [0.0; 3], 0.5);
        assert_eq!(er.slot, MeshSlot::ERROR);
        assert_eq!(er.vertex_offset, 10);
        assert_eq!(er.first_index, 30);
    }

    #[test]
    fn request_dedups_and_refcounts() {
        let mut c = MeshCatalog::new();
        c.alloc_slot(10, 30, [0.0; 3], 1.0); // placeholder
        c.alloc_slot(4, 12, [0.0; 3], 0.5); // error

        let (a, load_a) = c.request(0xAAAA);
        assert!(load_a, "first request of a path must trigger a load");
        assert_eq!(c.redirect_of(a), MeshSlot::PLACEHOLDER);

        let (a2, load_a2) = c.request(0xAAAA);
        assert_eq!(a, a2, "same path must dedup to the same MeshId");
        assert!(!load_a2, "repeat request must not trigger a second load");
        assert_eq!(c.refcount_of(a), 2);

        let (b, load_b) = c.request(0xBBBB);
        assert!(load_b);
        assert_ne!(a, b, "distinct paths must get distinct MeshIds");
    }

    #[test]
    fn resolve_flips_redirect_and_places_geometry() {
        let mut c = MeshCatalog::new();
        c.alloc_slot(10, 30, [0.0; 3], 1.0); // placeholder
        c.alloc_slot(4, 12, [0.0; 3], 0.5); // error
        let (a, _) = c.request(0xAAAA);

        // Pending → placeholder.
        assert_eq!(c.redirect_of(a), MeshSlot::PLACEHOLDER);

        let p = c.resolve(a, 8, 24, [1.0, 0.0, 0.0], 2.0);
        assert_eq!(p.slot, MeshSlot(2), "first real mesh lands in slot 2");
        assert_eq!(
            p.vertex_offset, 14,
            "cursor after placeholder(10) + error(4)"
        );
        assert_eq!(
            p.first_index, 42,
            "cursor after placeholder(30) + error(12)"
        );

        // Redirect now points at the real slot — renderers untouched.
        assert_eq!(c.redirect_of(a), MeshSlot(2));

        // Table entry recorded for the real slot.
        let entry = c.table()[2];
        assert_eq!(entry.index_count, 24);
        assert_eq!(entry.vertex_offset, 14);
        assert_eq!(entry.first_index, 42);
        assert_eq!(entry.bounds_radius, 2.0);

        assert_eq!(c.slot_count(), 3);
        assert_eq!(c.mesh_id_count(), 1);
        assert_eq!(c.vertex_used(), 22); // 10 + 4 + 8
        assert_eq!(c.index_used(), 66); // 30 + 12 + 24
    }

    #[test]
    fn fail_points_redirect_at_error_slot() {
        let mut c = MeshCatalog::new();
        c.alloc_slot(10, 30, [0.0; 3], 1.0);
        c.alloc_slot(4, 12, [0.0; 3], 0.5);
        let (a, _) = c.request(0xCAFE);

        c.fail(a);
        assert_eq!(c.redirect_of(a), MeshSlot::ERROR);
        // Failing must not allocate a new drawable slot.
        assert_eq!(c.slot_count(), 2);
    }
}
