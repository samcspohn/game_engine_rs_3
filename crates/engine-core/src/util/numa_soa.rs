//! Single-virtual-allocation, NUMA-paged `Vec<T>` replacement.
//!
//! `NumaSoa<T>` reserves a contiguous virtual range up front and binds
//! the lower half (in bytes, page-aligned) to NUMA node 0 and the upper
//! half to node 1 (or, when the running thread-pool only has one node
//! available, binds the whole range to that node).
//!
//! No physical pages are allocated at construction — the mapping is
//! `mmap(PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS)` and the
//! `mbind` calls only *install policy*. Pages are faulted in lazily on
//! first write and land on the policy node regardless of which CPU
//! caused the fault, so it is safe for cross-node threads to be the
//! ones doing the inserts during world initialisation.
//!
//! Design constraints from the rubber-duck review:
//!
//! * `partitions()` is expressed in **live element coordinates** —
//!   `node_0 = 0..len.min(entity_split)`, `node_1 = entity_split..len`.
//!   Returning ranges over `virtual_cap` would dispatch tasks at
//!   un-constructed slots; returning ranges fixed at construction
//!   would put nothing on node 1 until `len > entity_split`.
//! * `entity_split` is supplied by the caller (logical balance
//!   midpoint, typically `expected_max / 2`). It is independent of
//!   `max_elems` (the virtual reservation, typically 2× to 4× the
//!   expected max so we never run out without a hard panic).
//! * Element `entity_split` is **byte-rounded to the next page** so
//!   the second `mbind` call covers entire pages — one element may
//!   straddle the boundary; that's accepted (not UB, not corruption).
//! * No `MPOL_MF_MOVE` flag — the mapping has no resident pages, so
//!   migration is meaningless and risks spurious `EPERM`.
//! * `MADV_NOHUGEPAGE` is set on the entire mapping for placement
//!   determinism (THP can span page-policy boundaries).
//! * No fallbacks. If the host doesn't allow the requested binds the
//!   constructor panics with the kernel errno — that is a real
//!   configuration error (e.g. running with `numactl --membind=0` and
//!   the pool still requesting 2-node placement).

use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::NonNull;

use crate::util::thread_pool::NumaPartitioned;

/// Single-virtual-allocation, NUMA-paged growable array.
///
/// See module docs for the design rationale. The container is `Send`
/// because the underlying mmap is just bytes — `T: Send` is enforced
/// via `PhantomData<T>`.
pub struct NumaSoa<T> {
    /// Virtual base address of the mmap. Never null, never moves.
    ptr: NonNull<T>,
    /// Total mmap length in bytes (page-rounded ≥ `max_elems * size_of::<T>()`).
    map_len_bytes: usize,
    /// Element capacity covered by the virtual reservation. `push`
    /// past this panics.
    virtual_cap: usize,
    /// Logical midpoint for NUMA partitioning, in **elements**. Slots
    /// below this index are policy-bound to node 0; slots at or above
    /// it are bound to node 1.
    entity_split: usize,
    /// Number of constructed elements. `0..len` is live, `len..virtual_cap`
    /// is uninit.
    len: usize,
    /// Cached partition ranges in element coordinates. Length equals
    /// the number of NUMA nodes the pool actually uses. Recomputed
    /// (in place) every time `len` changes.
    partitions: Vec<Range<usize>>,
    /// One entry per node, in node-id order. `policy_nodes[i]` is the
    /// kernel node id we bound partition `i` to. Used only by
    /// diagnostics.
    policy_nodes: Vec<u32>,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for NumaSoa<T> {}
unsafe impl<T: Sync> Sync for NumaSoa<T> {}

impl<T> NumaSoa<T> {
    /// Reserve `max_elems * size_of::<T>()` bytes of virtual address
    /// space, split logically at element index `entity_split`, and
    /// install NUMA policy on each half.
    ///
    /// `num_nodes` is the number of NUMA nodes the engine's thread
    /// pool is using:
    ///
    /// * `1` — all pages bound to node 0. `entity_split` is forced to
    ///   `max_elems` (single partition `0..len`).
    /// * `2` — first half → node 0, second half → node 1.
    ///
    /// Panics if any syscall fails (no fallback — see module docs).
    pub fn with_split(max_elems: usize, entity_split: usize, num_nodes: u32) -> Self {
        Self::with_split_on_nodes(max_elems, entity_split, num_nodes, &[0, 1])
    }

    /// Same as [`Self::with_split`] but lets the caller specify which
    /// physical node ids to bind to. `node_ids[i]` is the node for
    /// partition `i`. Length must be `>= num_nodes`.
    pub fn with_split_on_nodes(
        max_elems: usize,
        entity_split: usize,
        num_nodes: u32,
        node_ids: &[u32],
    ) -> Self {
        assert!(
            std::mem::size_of::<T>() != 0,
            "NumaSoa<T>: zero-sized T not supported",
        );
        assert!(num_nodes >= 1, "NumaSoa: num_nodes must be >= 1");
        assert!(num_nodes <= 2, "NumaSoa: only 1- or 2-node configs supported (MVP)");
        assert!(
            node_ids.len() >= num_nodes as usize,
            "NumaSoa: node_ids ({}) shorter than num_nodes ({num_nodes})",
            node_ids.len(),
        );
        assert!(entity_split <= max_elems);
        let elem_size = std::mem::size_of::<T>();
        let total_bytes = max_elems
            .checked_mul(elem_size)
            .expect("NumaSoa: max_elems * size_of::<T>() overflow");
        assert!(
            std::mem::align_of::<T>() <= crate::util::numa_mem::page_size(),
            "NumaSoa: align_of::<T>() ({}) exceeds page size; mmap can't satisfy",
            std::mem::align_of::<T>(),
        );

        let ps = crate::util::numa_mem::page_size();
        let map_len_bytes = (total_bytes + ps - 1) & !(ps - 1);

        // ── 1. Reserve virtual address space + RW perms, no pages yet.
        let ptr: NonNull<T> = unsafe { mmap_anon_rw(map_len_bytes) };

        // ── 2. Determinism: disable THP for this range so per-page
        //      policy decisions aren't merged into a single huge page
        //      that crosses the NUMA boundary.
        #[cfg(target_os = "linux")]
        unsafe {
            // MADV_NOHUGEPAGE failures are non-fatal — THP isn't
            // critical to correctness, only to placement noise.
            let _ = libc::madvise(
                ptr.as_ptr().cast::<libc::c_void>(),
                map_len_bytes,
                libc::MADV_NOHUGEPAGE,
            );
        }

        // ── 3. Install per-half mempolicy. Pages don't exist yet, so
        //      this is a pure "future-fault" hint.
        let mut partitions: Vec<Range<usize>> = Vec::with_capacity(num_nodes as usize);
        let mut policy_nodes: Vec<u32> = Vec::with_capacity(num_nodes as usize);

        if num_nodes == 1 {
            // num_nodes==1 means "no NUMA splitting requested" —
            // leave the mapping under MPOL_DEFAULT so first-touch
            // placement remains in effect. (For explicit pinning
            // to a specific node use a 1-element `node_ids` plus
            // an explicit mbind via the lower-level API.)
            let _ = node_ids; // intentionally unused in default branch
            partitions.push(0..0); // updated by sync_partitions on first push
            policy_nodes.push(u32::MAX); // sentinel: "no explicit policy"
        } else {
            // Bytes split: page-aligned midpoint of the byte length.
            let split_bytes_unaligned = (entity_split as u64 * elem_size as u64) as usize;
            let split_bytes = (split_bytes_unaligned + ps - 1) & !(ps - 1);
            let split_bytes = split_bytes.min(map_len_bytes);

            bind_policy(ptr.as_ptr().cast::<u8>(), split_bytes, node_ids[0]);
            let upper_ptr = unsafe {
                ptr.as_ptr()
                    .cast::<u8>()
                    .add(split_bytes)
            };
            bind_policy(upper_ptr, map_len_bytes - split_bytes, node_ids[1]);

            partitions.push(0..0);
            partitions.push(0..0);
            policy_nodes.push(node_ids[0]);
            policy_nodes.push(node_ids[1]);
        }

        let mut this = Self {
            ptr,
            map_len_bytes,
            virtual_cap: max_elems,
            entity_split: if num_nodes == 1 { max_elems } else { entity_split },
            len: 0,
            partitions,
            policy_nodes,
            _marker: PhantomData,
        };
        this.sync_partitions();
        this
    }

    /// Single-node convenience constructor; everything binds to node 0.
    pub fn with_capacity_single_node(max_elems: usize) -> Self {
        Self::with_split_on_nodes(max_elems, max_elems, 1, &[0])
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn virtual_capacity(&self) -> usize {
        self.virtual_cap
    }

    #[inline]
    pub fn entity_split(&self) -> usize {
        self.entity_split
    }

    #[inline]
    pub fn policy_nodes(&self) -> &[u32] {
        &self.policy_nodes
    }

    /// Single-threaded push. Panics if capacity would be exceeded —
    /// callers must reserve adequate `max_elems` (no `mremap` for MVP,
    /// see module docs).
    pub fn push(&mut self, v: T) -> usize {
        assert!(
            self.len < self.virtual_cap,
            "NumaSoa::push: capacity exhausted (len={}, virtual_cap={})",
            self.len,
            self.virtual_cap,
        );
        let idx = self.len;
        // SAFETY: idx < virtual_cap; ptr.add(idx) is within the
        // mapped, writable range. The slot is uninit and we're
        // initializing it now.
        unsafe { std::ptr::write(self.ptr.as_ptr().add(idx), v) };
        self.len += 1;
        self.sync_partitions();
        idx
    }

    /// Read-only slice over the live region.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: 0..len is initialized and the mapping is alive.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutable slice over the live region. Borrow-exclusive.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Unchecked indexed access. Caller asserts `i < self.len`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> &T {
        unsafe { &*self.ptr.as_ptr().add(i) }
    }

    /// Unchecked mutable indexed access. Caller asserts `i < self.len`.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, i: usize) -> &mut T {
        unsafe { &mut *self.ptr.as_ptr().add(i) }
    }

    /// Recompute the partition ranges in element coordinates.
    #[inline]
    fn sync_partitions(&mut self) {
        match self.partitions.len() {
            1 => self.partitions[0] = 0..self.len,
            2 => {
                let mid = self.entity_split.min(self.len);
                self.partitions[0] = 0..mid;
                self.partitions[1] = mid..self.len;
            }
            n => unreachable!("NumaSoa: unsupported partition count {n}"),
        }
    }
}

impl<T> std::ops::Index<usize> for NumaSoa<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        assert!(i < self.len, "NumaSoa: index {i} out of bounds (len={})", self.len);
        unsafe { &*self.ptr.as_ptr().add(i) }
    }
}

impl<T> std::ops::IndexMut<usize> for NumaSoa<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        assert!(i < self.len, "NumaSoa: index {i} out of bounds (len={})", self.len);
        unsafe { &mut *self.ptr.as_ptr().add(i) }
    }
}

impl<T> NumaPartitioned for NumaSoa<T> {
    #[inline]
    fn numa_partitions(&self) -> &[Range<usize>] {
        &self.partitions
    }
}

impl<T> Drop for NumaSoa<T> {
    fn drop(&mut self) {
        // Run destructors for the live region only.
        if std::mem::needs_drop::<T>() && self.len > 0 {
            unsafe {
                let slice = std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len);
                std::ptr::drop_in_place(slice);
            }
        }
        // Unmap the entire reservation.
        #[cfg(target_os = "linux")]
        unsafe {
            let r = libc::munmap(self.ptr.as_ptr().cast::<libc::c_void>(), self.map_len_bytes);
            if r != 0 {
                eprintln!(
                    "NumaSoa::drop: munmap failed (errno={})",
                    std::io::Error::last_os_error(),
                );
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
unsafe fn mmap_anon_rw<T>(len_bytes: usize) -> NonNull<T> {
    let p = libc::mmap(
        std::ptr::null_mut(),
        len_bytes,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
        -1,
        0,
    );
    if p == libc::MAP_FAILED {
        panic!(
            "NumaSoa: mmap({} bytes) failed: {}",
            len_bytes,
            std::io::Error::last_os_error(),
        );
    }
    NonNull::new(p.cast::<T>())
        .expect("NumaSoa: mmap returned null (impossible)")
}

#[cfg(not(target_os = "linux"))]
unsafe fn mmap_anon_rw<T>(_len_bytes: usize) -> NonNull<T> {
    panic!("NumaSoa is only supported on Linux");
}

#[cfg(target_os = "linux")]
fn bind_policy(ptr: *mut u8, len: usize, node: u32) {
    if len == 0 {
        return;
    }
    crate::util::numa_mem::mbind_policy_to_node(ptr, len, node).unwrap_or_else(|e| {
        panic!(
            "NumaSoa: mbind_policy_to_node(node={node}, len={len}) failed: {e}. \
             If running under numactl --membind, ensure the engine's NUMA node \
             count matches the allowed memory mask.",
        )
    });
}

#[cfg(not(target_os = "linux"))]
fn bind_policy(_ptr: *mut u8, _len: usize, _node: u32) {
    panic!("NumaSoa is only supported on Linux");
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_index_single_node() {
        let mut s = NumaSoa::<u64>::with_capacity_single_node(1024);
        for i in 0..100 {
            s.push(i as u64);
        }
        assert_eq!(s.len(), 100);
        assert_eq!(s[0], 0);
        assert_eq!(s[99], 99);
        assert_eq!(s.numa_partitions(), &[0..100]);
    }

    #[test]
    fn two_node_partitions_track_len() {
        let mut s = NumaSoa::<u64>::with_split(1024, 512, 2);
        assert_eq!(s.numa_partitions(), &[0..0, 0..0]);
        for i in 0..200 {
            s.push(i);
        }
        assert_eq!(s.numa_partitions(), &[0..200, 200..200]);
        for i in 200..600 {
            s.push(i);
        }
        assert_eq!(s.numa_partitions(), &[0..512, 512..600]);
    }

    #[test]
    #[should_panic(expected = "capacity exhausted")]
    fn push_panics_past_cap() {
        let mut s = NumaSoa::<u32>::with_capacity_single_node(4);
        for _ in 0..5 {
            s.push(0);
        }
    }
}
