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
