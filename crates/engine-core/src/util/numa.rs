//! NUMA topology probe.
//!
//! Reads `/sys/devices/system/node/` on Linux to enumerate nodes and the
//! logical CPUs each contains. Used by the thread pool and (in later
//! phases) by per-page memory binding for the SoA backing storage.
//!
//! Per project rules: no fallbacks. If `/sys` lookups fail, the engine
//! panics — we want to know about it explicitly rather than silently
//! degrade to "one giant node" and lose the NUMA-aware dispatch.
//!
//! For non-Linux developer machines (e.g. WSL without /sys/node
//! visible), [`NumaTopology::single_node`] synthesises a 1-node topology
//! containing the supplied CPU set so the rest of the engine can still
//! work — only the renderer's pool init reaches for `single_node` as an
//! explicit fallback when `/sys/devices/system/node/` doesn't exist.

use std::fs;
use std::io;

/// One NUMA node: its id and the list of logical CPU indices it
/// contains. CPU indices match `core_affinity::CoreId::id`.
#[derive(Debug, Clone)]
pub struct NumaNode {
    pub id:   u32,
    pub cpus: Vec<usize>,
}

/// Snapshot of the machine's NUMA topology, in node-id order.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    nodes: Vec<NumaNode>,
}

impl NumaTopology {
    /// Probe `/sys/devices/system/node/`. Returns the list of online
    /// nodes and their CPU sets. Fails (returns `Err`) if `/sys` is
    /// unavailable or any node's `cpulist` is missing/malformed.
    pub fn detect() -> io::Result<Self> {
        let online_path = "/sys/devices/system/node/online";
        let online = fs::read_to_string(online_path)?;
        let node_ids = parse_cpulist(online.trim()).map_err(io::Error::other)?;
        if node_ids.is_empty() {
            return Err(io::Error::other(format!(
                "{online_path} reported no online nodes",
            )));
        }

        let mut nodes = Vec::with_capacity(node_ids.len());
        for id in node_ids {
            let path = format!("/sys/devices/system/node/node{id}/cpulist");
            let s = fs::read_to_string(&path)?;
            let cpus = parse_cpulist(s.trim()).map_err(io::Error::other)?;
            nodes.push(NumaNode { id: id as u32, cpus });
        }
        Ok(Self { nodes })
    }

    /// Synthesize a 1-node topology containing the given CPU ids.
    /// Use as a deliberate fallback when `/sys/devices/system/node/`
    /// is not available (e.g. some container / WSL setups).
    pub fn single_node(cpus: Vec<usize>) -> Self {
        Self { nodes: vec![NumaNode { id: 0, cpus }] }
    }

    pub fn nodes(&self) -> &[NumaNode] { &self.nodes }
    pub fn num_nodes(&self) -> usize  { self.nodes.len() }

    /// Return the node id containing `cpu`, or `None` if no node owns
    /// it (shouldn't happen on a well-formed system).
    pub fn node_of_cpu(&self, cpu: usize) -> Option<u32> {
        for n in &self.nodes {
            if n.cpus.contains(&cpu) {
                return Some(n.id);
            }
        }
        None
    }

    /// CPUs belonging to `node`, or `None` if the node isn't present.
    pub fn cpus_of_node(&self, node: u32) -> Option<&[usize]> {
        self.nodes
            .iter()
            .find(|n| n.id == node)
            .map(|n| n.cpus.as_slice())
    }
}

/// NUMA node the discrete GPU is attached to, from
/// `/sys/class/drm/card*/device/numa_node`. `None` when there is no DRM
/// card, or the kernel reports `-1` (no affinity — single-socket boxes and
/// most consumer hardware).
///
/// This matters far more than it looks. The scatter compute pulls dirty
/// transforms straight out of *host-cached* staging, so every read has to
/// snoop whichever socket's caches hold the freshly-written lines. With
/// staging writers spread over both sockets of a 2P box, the GPU pays a
/// remote-node fetch (distance 32 vs 10 here) for most of the 8 MB: the
/// scatter measures ~673µs unbound versus ~320µs with the writers confined
/// to the GPU's node. Binding the *pages* (`mbind`/`set_mempolicy`) does
/// nothing for this — the cost is cache residency, not page residency.
#[cfg(target_os = "linux")]
pub fn gpu_numa_node() -> Option<u32> {
    let mut best = None;
    for entry in fs::read_dir("/sys/class/drm").ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name();
        let name = name.to_str()?;
        // `card0`, but not the `card0-DP-1` connector subdirectories.
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let raw = match fs::read_to_string(entry.path().join("device/numa_node")) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if let Ok(node) = raw.trim().parse::<i32>() {
            if node >= 0 {
                best = Some(node as u32);
                break;
            }
        }
    }
    best
}

#[cfg(not(target_os = "linux"))]
pub fn gpu_numa_node() -> Option<u32> {
    None
}

/// Restrict the **calling thread** to `cpus` via `sched_setaffinity`.
///
/// Call this on the main thread before spawning the worker pool: threads
/// inherit the creating thread's affinity mask, so one call covers every
/// worker. Equivalent to launching under `numactl --cpunodebind=<node>`.
#[cfg(target_os = "linux")]
pub fn restrict_affinity_to(cpus: &[usize]) -> io::Result<()> {
    if cpus.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "restrict_affinity_to called with an empty CPU set",
        ));
    }
    // SAFETY: `set` is zero-initialised then populated only with CPU
    // indices below `CPU_SETSIZE` (checked below), and `sched_setaffinity`
    // is passed the matching size. pid 0 = calling thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cpus {
            if c >= libc::CPU_SETSIZE as usize {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("cpu {c} exceeds CPU_SETSIZE"),
                ));
            }
            libc::CPU_SET(c, &mut set);
        }
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn restrict_affinity_to(_cpus: &[usize]) -> io::Result<()> {
    Ok(())
}

/// The calling thread's current CPU affinity mask, as a CPU index list.
///
/// Pair with [`restrict_affinity_to`] to bind a *pool* rather than the
/// process: narrow the mask, spawn the pool (workers inherit the creating
/// thread's mask), then restore. That gives node-local workers without
/// confining anything else.
#[cfg(target_os = "linux")]
pub fn current_affinity() -> io::Result<Vec<usize>> {
    // SAFETY: `set` is zero-initialised and passed with its own size;
    // pid 0 = calling thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok((0..libc::CPU_SETSIZE as usize)
            .filter(|&c| libc::CPU_ISSET(c, &set))
            .collect())
    }
}

#[cfg(not(target_os = "linux"))]
pub fn current_affinity() -> io::Result<Vec<usize>> {
    Ok(Vec::new())
}

/// Parse a Linux-style cpulist string: `"0-3,8,12-15"` → `[0,1,2,3,8,12,13,14,15]`.
/// Used for both `/sys/.../online` (where the values are node ids) and
/// `/sys/.../nodeN/cpulist` (where they are CPU ids).
fn parse_cpulist(s: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let lo: usize = a.parse().map_err(|_| format!("bad cpulist range start: {a:?}"))?;
            let hi: usize = b.parse().map_err(|_| format!("bad cpulist range end:   {b:?}"))?;
            if hi < lo {
                return Err(format!("inverted cpulist range: {lo}-{hi}"));
            }
            for v in lo..=hi {
                out.push(v);
            }
        } else {
            out.push(part.parse().map_err(|_| format!("bad cpulist value: {part:?}"))?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpulist_parses_ranges_and_singletons() {
        assert_eq!(parse_cpulist("0-3").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpulist("0").unwrap(), vec![0]);
        assert_eq!(
            parse_cpulist("0-3,8,12-15").unwrap(),
            vec![0, 1, 2, 3, 8, 12, 13, 14, 15],
        );
        assert_eq!(parse_cpulist("").unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn single_node_topology_round_trip() {
        let t = NumaTopology::single_node(vec![0, 1, 2, 3]);
        assert_eq!(t.num_nodes(), 1);
        assert_eq!(t.node_of_cpu(2), Some(0));
        assert_eq!(t.node_of_cpu(99), None);
    }
}
