//! NUMA memory binding primitives.
//!
//! Thin wrappers over Linux's `mbind(2)` and `move_pages(2)` syscalls.
//! These are the tools we use to pin individual virtual ranges (or
//! per-page slices of one range) to a chosen NUMA node, and to verify
//! after the fact that pages actually landed where we asked.
//!
//! Per project rules: **no fallbacks**. A failed `mbind` is fatal at
//! the call site (callers `expect`/`panic`). We intentionally do not
//! degrade silently to "pages might land anywhere" because the entire
//! reason for invoking these calls is to control where the page lands.
//!
//! On non-Linux targets every function returns an `Err`. Callers must
//! be ready to gate this with `cfg(target_os = "linux")` if they want
//! to support other platforms.

#[cfg(target_os = "linux")]
use std::io;
#[cfg(target_os = "linux")]
use std::os::raw::{c_int, c_long, c_void};

#[cfg(not(target_os = "linux"))]
use std::io;

/// Memory policy: bind strictly to the supplied node set; if any page
/// can't be allocated on those nodes the allocation fails (with
/// `MPOL_MF_STRICT` we also error on existing pages already off-node
/// that can't be moved).
#[cfg(target_os = "linux")]
const MPOL_BIND: c_int = 2;
/// Try to move any pages already faulted into the range that aren't on
/// the target node. Without this flag `mbind` only sets the policy
/// for future faults — existing pages stay put.
#[cfg(target_os = "linux")]
const MPOL_MF_MOVE: u32 = 1 << 1;
/// Fail if any existing page can't be moved to the target node.
/// Combined with `MPOL_MF_MOVE` this turns a "best-effort" bind into a
/// "must land on node X" bind, which is exactly what we want for the
/// staging buffer.
#[cfg(target_os = "linux")]
const MPOL_MF_STRICT: u32 = 1 << 0;

/// System page size. Cached because `sysconf(_SC_PAGESIZE)` takes a
/// VDSO call on every invocation — cheap individually but called on
/// every staging realloc.
#[inline]
pub fn page_size() -> usize {
    #[cfg(target_os = "linux")]
    {
        use std::sync::OnceLock;
        static PS: OnceLock<usize> = OnceLock::new();
        *PS.get_or_init(|| {
            let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            assert!(v > 0, "sysconf(_SC_PAGESIZE) returned {v}");
            v as usize
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        4096
    }
}

/// Align `addr` UP to the next page boundary; align `addr + len` DOWN
/// to a page boundary. Returns `(aligned_addr, aligned_len)`. If the
/// supplied range doesn't span a full page after alignment, returns
/// `(addr, 0)`.
#[inline]
fn page_clamp(addr: *mut u8, len: usize) -> (*mut u8, usize) {
    let ps    = page_size();
    let base  = addr as usize;
    let end   = base.saturating_add(len);
    let aligned_base = (base + ps - 1) & !(ps - 1);
    let aligned_end  = end & !(ps - 1);
    if aligned_end <= aligned_base {
        (addr, 0)
    } else {
        (aligned_base as *mut u8, aligned_end - aligned_base)
    }
}

/// Bind every full page intersecting `[addr, addr+len)` to NUMA node
/// `node`. Already-resident pages that aren't on `node` are migrated;
/// future faults must land on `node`.
///
/// `len` is in bytes. The range is silently page-clamped — partial-
/// page tails at either end aren't covered (a partial page is owned
/// by whichever caller allocated the rest of it).
///
/// Returns `Ok(0)` if the range is empty after clamping (e.g. a small
/// buffer that doesn't span a page). Returns `Ok(n_pages)` otherwise,
/// the number of pages the kernel was asked to bind.
///
/// Errors are converted from the underlying `errno`. The most common
/// failure is `EPERM` (kernel forbids `MPOL_MF_MOVE` for unprivileged
/// processes when migrating pages cross-node) — usually meaning
/// `vm.swappiness` / cgroups need adjusting, or the calling user
/// doesn't have `CAP_SYS_NICE`. We surface the raw errno rather than
/// swallowing it.
#[cfg(target_os = "linux")]
pub fn mbind_to_node(addr: *mut u8, len: usize, node: u32) -> io::Result<usize> {
    let (aligned, aligned_len) = page_clamp(addr, len);
    if aligned_len == 0 {
        return Ok(0);
    }
    assert!(node < 64, "mbind_to_node: node {node} >= 64 not supported (single-u64 nodemask)");

    // nodemask: bit `node` set in a 64-bit word.
    // maxnode is the **number of bits to inspect**, INCLUDING the unused
    // high bits past the highest node id (kernel inspects `maxnode` bits
    // and complains if maxnode == 0). 64 covers nodes 0..=63.
    let nodemask: u64 = 1u64 << node;
    let maxnode: c_long = 64;

    // SAFETY: SYS_mbind is documented to take exactly these args; the
    // nodemask pointer is read-only and lives for the duration of the
    // call (stack local).
    let ret = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            aligned as *mut c_void,
            aligned_len as c_long,
            MPOL_BIND as c_long,
            &nodemask as *const u64 as *const c_void,
            maxnode,
            (MPOL_MF_MOVE | MPOL_MF_STRICT) as c_long,
        )
    };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(aligned_len / page_size())
}

/// Query the resident NUMA node of every page intersecting
/// `[addr, addr+len)`. Returns one entry per page, in order. A
/// negative value at index `i` is the negated errno for that page
/// (e.g. `-libc::ENOENT` if the page isn't resident yet).
#[cfg(target_os = "linux")]
pub fn page_residency(addr: *const u8, len: usize) -> io::Result<Vec<i32>> {
    let (aligned, aligned_len) = page_clamp(addr as *mut u8, len);
    let ps     = page_size();
    let n_pages = aligned_len / ps;
    if n_pages == 0 {
        return Ok(Vec::new());
    }

    let pages: Vec<*mut c_void> = (0..n_pages)
        .map(|i| unsafe { aligned.add(i * ps) as *mut c_void })
        .collect();
    let mut status: Vec<c_int> = vec![-1; n_pages];

    // SAFETY: SYS_move_pages with NULL `nodes` is a pure query: it
    // doesn't migrate anything, just fills `status` with each page's
    // current node id (or a negative errno).
    let ret = unsafe {
        libc::syscall(
            libc::SYS_move_pages,
            0 as c_long, // self
            n_pages as c_long,
            pages.as_ptr() as *const *const c_void,
            std::ptr::null::<c_int>(),
            status.as_mut_ptr(),
            0 as c_long,
        )
    };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(status.into_iter().map(|s| s as i32).collect())
}

/// Verify every resident page in `[addr, addr+len)` is on `expected_node`.
/// Pages that aren't yet faulted (status `-ENOENT`) are skipped without
/// complaint. Returns `(checked, mismatched)`: number of pages whose
/// node we read, and number that were on the wrong node.
#[cfg(target_os = "linux")]
pub fn verify_residency_single_node(
    addr: *const u8,
    len: usize,
    expected_node: u32,
) -> io::Result<(usize, usize)> {
    let status = page_residency(addr, len)?;
    let mut checked = 0usize;
    let mut wrong   = 0usize;
    for s in status {
        if s < 0 {
            // -ENOENT = not yet resident; ignore.
            continue;
        }
        checked += 1;
        if s as u32 != expected_node {
            wrong += 1;
        }
    }
    Ok((checked, wrong))
}

/// Memory policy: no special policy (system default — first-touch on
/// whichever node the faulting CPU happens to be on).
#[cfg(target_os = "linux")]
const MPOL_DEFAULT: c_int = 0;

/// Set the memory policy of a virtual range to `MPOL_BIND { node }`
/// **without** migrating existing pages.
///
/// Use this immediately after `mmap` on an anonymous, unfaulted range:
/// no physical pages exist yet, and we want future page faults to land
/// on `node`. Unlike [`mbind_to_node`] (which uses
/// `MPOL_MF_MOVE | MPOL_MF_STRICT` and is intended for already-resident
/// pages), this passes `flags = 0` — pure policy installation, no
/// migration attempt, no `EPERM` failure modes from
/// `CAP_SYS_NICE`-gated migration.
///
/// `len` is in bytes; the range is page-clamped silently.
#[cfg(target_os = "linux")]
pub fn mbind_policy_to_node(addr: *mut u8, len: usize, node: u32) -> io::Result<usize> {
    let (aligned, aligned_len) = page_clamp(addr, len);
    if aligned_len == 0 {
        return Ok(0);
    }
    assert!(node < 64, "mbind_policy_to_node: node {node} >= 64 not supported");
    let nodemask: u64 = 1u64 << node;
    let maxnode: c_long = 64;
    // SAFETY: same contract as mbind_to_node; only difference is flags=0.
    let ret = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            aligned as *mut c_void,
            aligned_len as c_long,
            MPOL_BIND as c_long,
            &nodemask as *const u64 as *const c_void,
            maxnode,
            0 as c_long,
        )
    };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(aligned_len / page_size())
}

/// RAII guard that binds the **calling thread's** future allocations
/// to a single NUMA node, restoring `MPOL_DEFAULT` on drop.
///
/// Unlike [`mbind_to_node`] (which operates on already-faulted pages
/// of a known virtual range), this controls where *new* page faults
/// land. It's the mechanism we need to influence GPU-driver-internal
/// mappings — drivers `mmap` host-visible DMA buffers from inside
/// vulkan/ioctl calls, and `mbind` after the fact is ineffective on
/// those pages (the kernel doesn't track them as anonymous user pages,
/// `move_pages` reports `-ENOENT`).
///
/// Scope this guard **as tightly as possible** around the allocator
/// call that produces the buffer you want pinned. Any other allocation
/// the thread performs while the guard is alive will also land on
/// `node`, which is usually not what you want.
///
/// On non-Linux targets this is a no-op (the constructor returns
/// `Ok(MempolicyGuard {})` without doing anything).
#[cfg(target_os = "linux")]
pub struct MempolicyGuard {
    /// True if `set_mempolicy(BIND)` actually succeeded — only then
    /// do we need to restore on drop.
    active: bool,
}

#[cfg(target_os = "linux")]
impl MempolicyGuard {
    /// Set the calling thread's allocation policy to
    /// `MPOL_BIND { node }`. Errors propagate from the syscall.
    pub fn bind_to_node(node: u32) -> io::Result<Self> {
        assert!(node < 64, "MempolicyGuard: node {node} >= 64 not supported");
        let nodemask: u64 = 1u64 << node;
        // SAFETY: SYS_set_mempolicy takes (mode, *nodemask, maxnode).
        // nodemask points to a stack local that outlives the call.
        let ret = unsafe {
            libc::syscall(
                libc::SYS_set_mempolicy,
                MPOL_BIND as c_long,
                &nodemask as *const u64 as *const c_void,
                64 as c_long,
            )
        };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { active: true })
    }
}

#[cfg(target_os = "linux")]
impl Drop for MempolicyGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        // Restore default. nodemask is empty (NULL pointer + maxnode 0
        // is the documented way to clear the bound mask for
        // MPOL_DEFAULT).
        // SAFETY: passing a null pointer with mode = MPOL_DEFAULT is
        // the documented "clear" form of set_mempolicy.
        let ret = unsafe {
            libc::syscall(
                libc::SYS_set_mempolicy,
                MPOL_DEFAULT as c_long,
                std::ptr::null::<c_void>(),
                0 as c_long,
            )
        };
        if ret != 0 {
            // Don't panic in drop. Log loudly; this leaks the bind
            // into surrounding code, which is a real bug but not
            // recoverable from inside drop.
            eprintln!(
                "[numa-staging] MempolicyGuard drop: failed to restore \
                 MPOL_DEFAULT (errno={})",
                io::Error::last_os_error(),
            );
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Non-Linux stubs.
// ────────────────────────────────────────────────────────────────────────

#[cfg(not(target_os = "linux"))]
pub struct MempolicyGuard;

#[cfg(not(target_os = "linux"))]
impl MempolicyGuard {
    pub fn bind_to_node(_node: u32) -> io::Result<Self> {
        Err(io::Error::other("MempolicyGuard not supported on this OS"))
    }
}

#[cfg(not(target_os = "linux"))]
pub fn mbind_to_node(_addr: *mut u8, _len: usize, _node: u32) -> io::Result<usize> {
    Err(io::Error::other("NUMA mbind not supported on this OS"))
}

#[cfg(not(target_os = "linux"))]
pub fn verify_residency_single_node(
    _addr: *const u8,
    _len: usize,
    _expected_node: u32,
) -> io::Result<(usize, usize)> {
    Err(io::Error::other("NUMA move_pages query not supported on this OS"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_size_is_power_of_two() {
        let ps = page_size();
        assert!(ps.is_power_of_two(), "page_size = {ps} not power of two");
        assert!(ps >= 4096, "page_size = {ps} unexpectedly small");
    }

    #[test]
    fn page_clamp_handles_partial_pages() {
        let ps = page_size();
        // Range entirely inside one page: nothing to bind.
        let base = (16 * ps) as *mut u8;
        let (_a, l) = page_clamp(unsafe { base.add(7) }, 32);
        assert_eq!(l, 0, "sub-page range should clamp to zero");

        // Range spanning exactly one page after alignment.
        let (a, l) = page_clamp(base, 2 * ps);
        assert_eq!(a as usize, base as usize);
        assert_eq!(l, 2 * ps);

        // Misaligned start, generous length: aligned start advances,
        // aligned end backs off.
        let (a, l) = page_clamp(unsafe { base.add(7) }, 3 * ps);
        assert_eq!(a as usize, base as usize + ps);
        assert_eq!(l, 2 * ps);
    }
}
