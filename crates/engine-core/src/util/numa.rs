//! NUMA topology detection.
//!
//! On systems where some CPU cores have no local DRAM controller — the
//! AMD Threadripper 2990WX's compute dies are the canonical example,
//! with 2 of 4 NUMA nodes having zero local memory — workers pinned to
//! those cores pay 1.5-2× the latency on memory-bound workloads
//! because every byte routes over Infinity Fabric.
//!
//! This module identifies which CPU cores have local DRAM so the
//! engine can build a separate "staging" thread pool that avoids the
//! slow remote-memory cores for the host-staging walk.
//!
//! Platform support:
//! * Linux: parses `/sys/devices/system/node/node*/meminfo` + `cpulist`
//! * Windows / macOS / other: stub returning `None` (engine falls back
//!   to using all cores for staging, identical to compute)

/// CPU IDs (matching `core_affinity::CoreId.id`) that belong to NUMA
/// nodes with local DRAM. Returns:
///
/// * `Some(cpus)` — the listed CPUs sit on memory-die NUMA nodes (have
///   local DRAM). The engine uses these for memory-bound workloads.
/// * `None` — no exploitable NUMA asymmetry: either the topology can't
///   be probed (unsupported OS, sysfs missing), there is only one NUMA
///   node, or every node has its own DRAM. Callers should treat this as
///   "use all cores" and skip the separate staging pool.
///
/// The CPU IDs in the returned vector are sorted ascending and deduped.
pub fn local_dram_cpus() -> Option<Vec<usize>> {
    imp::local_dram_cpus()
}

#[cfg(target_os = "linux")]
mod imp {
    use std::fs;

    pub fn local_dram_cpus() -> Option<Vec<usize>> {
        let entries = fs::read_dir("/sys/devices/system/node/").ok()?;

        let mut node_ids: Vec<u32> = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix("node") {
                if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                    if let Ok(id) = rest.parse::<u32>() {
                        node_ids.push(id);
                    }
                }
            }
        }

        if node_ids.len() < 2 {
            return None;
        }

        node_ids.sort_unstable();

        let mut local_cpus: Vec<usize> = Vec::new();
        let mut all_have_memory = true;

        for id in &node_ids {
            let meminfo_path = format!("/sys/devices/system/node/node{}/meminfo", id);
            let meminfo = fs::read_to_string(&meminfo_path).ok()?;
            let mem_kb = parse_mem_total_kb(&meminfo)?;

            if mem_kb == 0 {
                all_have_memory = false;
                continue;
            }

            let cpulist_path = format!("/sys/devices/system/node/node{}/cpulist", id);
            let cpulist = fs::read_to_string(&cpulist_path).ok()?;
            local_cpus.extend(parse_cpulist(cpulist.trim()));
        }

        if all_have_memory {
            return None;
        }

        if local_cpus.is_empty() {
            return None;
        }

        local_cpus.sort_unstable();
        local_cpus.dedup();
        Some(local_cpus)
    }

    fn parse_mem_total_kb(meminfo: &str) -> Option<u64> {
        for line in meminfo.lines() {
            if let Some(idx) = line.find("MemTotal:") {
                let rest = &line[idx + "MemTotal:".len()..];
                let mut tokens = rest.split_whitespace();
                let num = tokens.next()?;
                return num.parse::<u64>().ok();
            }
        }
        None
    }

    fn parse_cpulist(s: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some((lo, hi)) = part.split_once('-') {
                let lo: usize = lo.trim().parse().expect("invalid cpulist range start");
                let hi: usize = hi.trim().parse().expect("invalid cpulist range end");
                for cpu in lo..=hi {
                    out.push(cpu);
                }
            } else {
                let cpu: usize = part.parse().expect("invalid cpulist cpu id");
                out.push(cpu);
            }
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parse_mem_total_handles_typical_line() {
            let s = "Node 0 MemTotal:       16777216 kB\nNode 0 MemFree:          0 kB\n";
            assert_eq!(parse_mem_total_kb(s), Some(16777216));
        }

        #[test]
        fn parse_mem_total_handles_zero() {
            let s = "Node 1 MemTotal:       0 kB\n";
            assert_eq!(parse_mem_total_kb(s), Some(0));
        }

        #[test]
        fn parse_cpulist_single_range() {
            assert_eq!(parse_cpulist("0-7"), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        }

        #[test]
        fn parse_cpulist_multi_range() {
            assert_eq!(
                parse_cpulist("0-7,32-39"),
                vec![0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39]
            );
        }

        #[test]
        fn parse_cpulist_single_cpu() {
            assert_eq!(parse_cpulist("5"), vec![5]);
        }

        #[test]
        fn parse_cpulist_mixed() {
            assert_eq!(parse_cpulist("0,2-4,7"), vec![0, 2, 3, 4, 7]);
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    pub fn local_dram_cpus() -> Option<Vec<usize>> {
        // TODO: Windows impl via GetLogicalProcessorInformationEx +
        // GetNumaAvailableMemoryNodeEx. Until then, return None so the
        // engine falls back to using all cores for staging.
        None
    }
}
