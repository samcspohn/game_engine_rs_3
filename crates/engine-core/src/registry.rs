//! The bookkeeping every asset store shares: a key deduplicated to a stable
//! id, the id redirected to a slot of data, a refcount per id, and the
//! redirect changes a GPU mirror drains. Each store keeps its own slot data
//! and its own idea of a key — a path, a path and color space, content.

use std::collections::HashMap;
use std::hash::Hash;

/// A `u32` newtype a registry indexes by.
pub trait Index: Copy + Eq {
    fn from_index(i: u32) -> Self;
    fn index(self) -> u32;
}

macro_rules! index {
    ($($t:ty),*) => {$(
        impl Index for $t {
            fn from_index(i: u32) -> Self {
                Self(i)
            }
            fn index(self) -> u32 {
                self.0
            }
        }
    )*};
}

index!(
    crate::asset::MeshId,
    crate::asset::MeshSlot,
    crate::texture::TextureId,
    crate::texture::TextureSlot,
    crate::material::MaterialId,
    crate::material::MaterialSlot
);

/// Ids `I` keyed by `K`, redirected to slots `S` holding `D`. Slots are
/// append-only and reclamation is deferred, so a refcount reaching zero frees
/// nothing yet.
pub struct Registry<I, S, K, D> {
    by_key: HashMap<K, I>,
    keys: Vec<Option<K>>,
    redirect: Vec<S>,
    refcount: Vec<u32>,
    slots: Vec<D>,
    /// Ids whose redirect changed since the last drain.
    dirty: Vec<I>,
}

impl<I: Index, S: Index, K: Eq + Hash + Clone, D> Registry<I, S, K, D> {
    pub fn new() -> Self {
        Self {
            by_key: HashMap::new(),
            keys: Vec::new(),
            redirect: Vec::new(),
            refcount: Vec::new(),
            slots: Vec::new(),
            dirty: Vec::new(),
        }
    }

    /// The id `key` names, with one more reference. A first request makes it,
    /// pointing at `fresh`, and says so with `true`.
    pub fn request(&mut self, key: K, fresh: S) -> (I, bool) {
        if let Some(id) = self.find(&key) {
            self.retain(id);
            return (id, false);
        }
        (self.insert(Some(key), fresh), true)
    }

    /// A new id with one reference, pointing at `slot`. `None` is an id no
    /// request can find. Not a redirect change: a mirror fills new ids with
    /// what `slot` is, until [`point`](Self::point) says otherwise.
    pub fn insert(&mut self, key: Option<K>, slot: S) -> I {
        let id = I::from_index(self.redirect.len() as u32);
        if let Some(k) = &key {
            self.by_key.insert(k.clone(), id);
        }
        self.keys.push(key);
        self.redirect.push(slot);
        self.refcount.push(1);
        id
    }

    pub fn find(&self, key: &K) -> Option<I> {
        self.by_key.get(key).copied()
    }

    pub fn key_of(&self, id: I) -> Option<&K> {
        self.keys.get(id.index() as usize)?.as_ref()
    }

    /// Stop `id` answering for its key, so the next request makes a new one.
    pub fn unkey(&mut self, id: I) {
        if let Some(k) = self.keys[id.index() as usize].take() {
            self.by_key.remove(&k);
        }
    }

    /// A new slot. Points no id at it — [`point`](Self::point) does.
    pub fn alloc(&mut self, data: D) -> S {
        self.slots.push(data);
        S::from_index(self.slots.len() as u32 - 1)
    }

    pub fn point(&mut self, id: I, slot: S) {
        self.redirect[id.index() as usize] = slot;
        self.dirty.push(id);
    }

    pub fn redirect_of(&self, id: I) -> S {
        self.redirect[id.index() as usize]
    }

    pub fn slot(&self, slot: S) -> &D {
        &self.slots[slot.index() as usize]
    }

    pub fn slot_mut(&mut self, slot: S) -> &mut D {
        &mut self.slots[slot.index() as usize]
    }

    pub fn slot_count(&self) -> u32 {
        self.slots.len() as u32
    }

    pub fn id_count(&self) -> u32 {
        self.redirect.len() as u32
    }

    pub fn retain(&mut self, id: I) {
        self.refcount[id.index() as usize] += 1;
    }

    pub fn release(&mut self, id: I) {
        let rc = &mut self.refcount[id.index() as usize];
        debug_assert!(*rc > 0, "release of id {} with zero refcount", id.index());
        *rc = rc.saturating_sub(1);
    }

    pub fn refcount_of(&self, id: I) -> u32 {
        self.refcount[id.index() as usize]
    }

    /// The `(id, slot)` redirects changed since the last call.
    pub fn take_redirect_updates(&mut self) -> Vec<(I, S)> {
        let dirty = std::mem::take(&mut self.dirty);
        dirty
            .into_iter()
            .map(|id| (id, self.redirect_of(id)))
            .collect()
    }
}

impl<I: Index, S: Index, K: Eq + Hash + Clone, D> Default for Registry<I, S, K, D> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{MeshId, MeshSlot};

    type Reg = Registry<MeshId, MeshSlot, &'static str, u8>;

    #[test]
    fn a_key_is_one_id_and_counts_its_requests() {
        let mut r = Reg::new();
        let zero = r.alloc(0);
        let (a, made) = r.request("a", zero);
        assert!(made);
        assert_eq!(r.request("a", zero), (a, false));
        assert_eq!(r.refcount_of(a), 2);
        assert_eq!(r.key_of(a), Some(&"a"));
    }

    #[test]
    fn a_redirect_drains_once() {
        let mut r = Reg::new();
        let (zero, one) = (r.alloc(0), r.alloc(1));
        let (a, _) = r.request("a", zero);
        assert!(r.take_redirect_updates().is_empty(), "a fresh id is its fill");
        r.point(a, one);
        assert_eq!(r.take_redirect_updates(), [(a, one)]);
        assert!(r.take_redirect_updates().is_empty());
        assert_eq!(*r.slot(r.redirect_of(a)), 1);
    }

    #[test]
    fn an_unkeyed_id_no_longer_answers() {
        let mut r = Reg::new();
        let zero = r.alloc(0);
        let (a, _) = r.request("a", zero);
        r.unkey(a);
        let (b, made) = r.request("a", zero);
        assert!(made && a != b);
    }
}
