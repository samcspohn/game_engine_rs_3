//! Best-effort `amdgpu` GPU telemetry, sampled once per FPS print window.
//!
//! Motivation: at 1M cubes the engine settles into one of two stable
//! attractors per launch — a "good" GPU-bound run (~1050 FPS, ~25% CPU,
//! ~98% GPU, cheap `host_staging`) and a "bad" run (~920 FPS, ~45% CPU,
//! ~60% GPU, `host_staging` ballooning ~23×). The staging triple lives in
//! `DEVICE_LOCAL | HOST_VISIBLE` VRAM (ReBAR), so the host's write-combined
//! stores are serviced by the GPU memory controller across the PCIe BAR.
//! Whether those posted stores drain without back-pressure depends on GPU
//! state we can't see from CPU timers alone: memory-controller activity,
//! actual sclk/mclk, PCIe link state, board power, throttling, and how full
//! the visible-VRAM (BAR) aperture is.
//!
//! This module reads the stable text sysfs files amdgpu exposes (no binary
//! `gpu_metrics` parsing, which is version-fragile) once per ~1 s print, so
//! it costs nothing on the hot path. The sampled line is printed directly
//! under each `FPS:` line so good vs. bad runs self-diagnose.
//!
//! Per project rules there is **no silent fallback**: [`GpuTelemetry::discover`]
//! returns `None` only when there is genuinely no `amdgpu` DRM node (non-AMD
//! or non-Linux), and the caller prints whether telemetry is active. Once a
//! card is found, a field that unexpectedly fails to read is surfaced inline
//! as an explicit error token rather than being hidden.

#[cfg(target_os = "linux")]
use std::path::{Path, PathBuf};

/// Handle to a discovered `amdgpu` DRM device's sysfs tree.
#[cfg(target_os = "linux")]
pub struct GpuTelemetry {
    /// `/sys/class/drm/cardN/device`
    dev: PathBuf,
    /// `/sys/class/drm/cardN/device/hwmon/hwmonM` (clocks / power / temps).
    hwmon: PathBuf,
    /// Human label, e.g. `card2 (0x1002:0x744c)`.
    label: String,
}

#[cfg(target_os = "linux")]
impl GpuTelemetry {
    /// Locate the first AMD (`vendor == 0x1002`) DRM card that exposes
    /// `gpu_busy_percent` and an `hwmon` subdirectory. Returns `None` when
    /// no such card exists; the caller is responsible for printing that
    /// telemetry is disabled so this is never a silent no-op.
    pub fn discover() -> Option<Self> {
        let drm = Path::new("/sys/class/drm");
        let mut entries: Vec<PathBuf> = std::fs::read_dir(drm)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                // Match `cardN` but not the `cardN-DP-1` connector nodes.
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("card") && !n.contains('-'))
                    .unwrap_or(false)
            })
            .collect();
        entries.sort();

        for card in entries {
            let dev = card.join("device");
            let vendor = std::fs::read_to_string(dev.join("vendor")).ok();
            let is_amd = vendor.as_deref().map(str::trim) == Some("0x1002");
            if !is_amd || !dev.join("gpu_busy_percent").exists() {
                continue;
            }
            let hwmon = match find_hwmon(&dev) {
                Some(h) => h,
                None => continue,
            };
            let device_id = std::fs::read_to_string(dev.join("device"))
                .ok()
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "?".to_string());
            let card_name = card
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("card?")
                .to_string();
            return Some(Self {
                label: format!("{card_name} (0x1002:{device_id})"),
                dev,
                hwmon,
            });
        }
        None
    }

    /// Short identifier printed in the one-time status line.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Read the current state and format it as a single compact line.
    /// Instantaneous (one read per ~1 s window); the run sits in one
    /// attractor for its whole duration, so a single sample is
    /// representative while staying off the hot path.
    pub fn sample_line(&self) -> String {
        let gpu_busy = read_u64(&self.dev.join("gpu_busy_percent"));
        let mem_busy = read_u64(&self.dev.join("mem_busy_percent"));
        let sclk_hz = read_u64(&self.hwmon.join("freq1_input"));
        let mclk_hz = read_u64(&self.hwmon.join("freq2_input"));
        let ppt_uw = read_u64(&self.hwmon.join("power1_average"));
        let t_edge = read_u64(&self.hwmon.join("temp1_input"));
        let t_junc = read_u64(&self.hwmon.join("temp2_input"));
        let t_mem = read_u64(&self.hwmon.join("temp3_input"));
        let pcie = read_starred(&self.dev.join("pp_dpm_pcie"));
        let vis_used = read_u64(&self.dev.join("mem_info_vis_vram_used"));
        let vis_total = read_u64(&self.dev.join("mem_info_vis_vram_total"));
        // Where *this* process's GPU buffers physically live. The staging
        // triple is requested `DEVICE_LOCAL | HOST_VISIBLE`, but amdgpu/TTM
        // is free to back that host-visible "VRAM" allocation with GTT
        // (system RAM) under VRAM pressure — and that residency, not the
        // Vulkan memory-type index, decides whether the host's staging
        // stores are cached DRAM writes (fast) or write-combined PCIe BAR
        // writes (slow). This is the field that distinguishes the bimodal
        // "good" (GTT) vs "bad" (VRAM) launches.
        let (own_vram_kib, own_gtt_kib) = read_self_drm_residency();

        format!(
            "  gpu: busy {} mem {} | sclk {} mclk {} | pcie {} | PPT {} | temp e/j/m {}/{}/{} | visVRAM {} | self vram/gtt {}",
            fmt_pct(gpu_busy),
            fmt_pct(mem_busy),
            fmt_mhz(sclk_hz),
            fmt_mhz(mclk_hz),
            pcie.as_deref().unwrap_or("ERR"),
            fmt_watts(ppt_uw),
            fmt_celsius(t_edge),
            fmt_celsius(t_junc),
            fmt_celsius(t_mem),
            fmt_vram(vis_used, vis_total),
            fmt_residency(own_vram_kib, own_gtt_kib),
        )
    }
}

/// Sum this process's `drm-memory-vram` / `drm-memory-gtt` (in KiB) across
/// its open amdgpu render-node fds. Returns `(vram_kib, gtt_kib)`; either is
/// `None` if no amdgpu DRM fd with those keys is open. Used to reveal
/// whether the staging triple is physically resident in VRAM (slow host
/// write-combined BAR stores) or GTT/system RAM (fast cached stores).
#[cfg(target_os = "linux")]
fn read_self_drm_residency() -> (Option<u64>, Option<u64>) {
    let fd_dir = match std::fs::read_dir("/proc/self/fd") {
        Ok(d) => d,
        Err(_) => return (None, None),
    };
    let mut vram = 0u64;
    let mut gtt = 0u64;
    let mut found = false;
    for entry in fd_dir.filter_map(|e| e.ok()) {
        // Only DRM render nodes carry the amdgpu memory accounting.
        let target = std::fs::read_link(entry.path());
        let is_render = target
            .as_ref()
            .ok()
            .and_then(|p| p.to_str())
            .map(|s| s.starts_with("/dev/dri/renderD"))
            .unwrap_or(false);
        if !is_render {
            continue;
        }
        let fdnum = match entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<u64>().ok())
        {
            Some(n) => n,
            None => continue,
        };
        let info = match std::fs::read_to_string(format!("/proc/self/fdinfo/{fdnum}")) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut saw_keys = false;
        for line in info.lines() {
            if let Some(kib) = parse_kib_line(line, "drm-memory-vram:") {
                vram += kib;
                saw_keys = true;
            } else if let Some(kib) = parse_kib_line(line, "drm-memory-gtt:") {
                gtt += kib;
                saw_keys = true;
            }
        }
        found |= saw_keys;
    }
    if found {
        (Some(vram), Some(gtt))
    } else {
        (None, None)
    }
}

/// Parse a `"<key>\t<n> KiB"` fdinfo line, returning `<n>` if it matches.
#[cfg(target_os = "linux")]
fn parse_kib_line(line: &str, key: &str) -> Option<u64> {
    let rest = line.trim().strip_prefix(key)?;
    rest.trim().split_whitespace().next()?.parse::<u64>().ok()
}

#[cfg(target_os = "linux")]
fn fmt_residency(vram_kib: Option<u64>, gtt_kib: Option<u64>) -> String {
    match (vram_kib, gtt_kib) {
        (Some(v), Some(g)) => format!("{}/{}MB", v / 1024, g / 1024),
        _ => "ERR".into(),
    }
}

/// First `hwmon/hwmonN` subdirectory under `dev`, if any.
#[cfg(target_os = "linux")]
fn find_hwmon(dev: &Path) -> Option<PathBuf> {
    let mut hwmons: Vec<PathBuf> = std::fs::read_dir(dev.join("hwmon"))
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    hwmons.sort();
    hwmons.into_iter().next()
}

/// Read a file containing a single unsigned integer. `None` on any error
/// (missing file / unparseable) so the formatter can surface it explicitly.
#[cfg(target_os = "linux")]
fn read_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Return the inner text of the DPM line marked current (`*`), e.g. for
/// `pp_dpm_pcie` line "2: 16.0GT/s, x16 623Mhz *" yields "16.0GT/s, x16".
#[cfg(target_os = "linux")]
fn read_starred(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let line = content.lines().find(|l| l.trim_end().ends_with('*'))?;
    // Strip the "N: " index prefix and the trailing clock + "*".
    let after_colon = line.split_once(':').map(|(_, r)| r).unwrap_or(line);
    let body = after_colon.trim_end_matches('*').trim();
    // Drop the trailing "<n>Mhz" token amdgpu appends to the pcie line.
    let trimmed = body
        .rsplit_once(' ')
        .filter(|(_, last)| last.to_ascii_lowercase().ends_with("mhz"))
        .map(|(head, _)| head.trim())
        .unwrap_or(body);
    Some(trimmed.to_string())
}

#[cfg(target_os = "linux")]
fn fmt_pct(v: Option<u64>) -> String {
    v.map(|p| format!("{p:>3}%"))
        .unwrap_or_else(|| "ERR".into())
}

#[cfg(target_os = "linux")]
fn fmt_mhz(hz: Option<u64>) -> String {
    hz.map(|h| format!("{:>4}MHz", h / 1_000_000))
        .unwrap_or_else(|| "ERR".into())
}

#[cfg(target_os = "linux")]
fn fmt_watts(uw: Option<u64>) -> String {
    uw.map(|w| format!("{:>3}W", w / 1_000_000))
        .unwrap_or_else(|| "ERR".into())
}

#[cfg(target_os = "linux")]
fn fmt_celsius(milli_c: Option<u64>) -> String {
    milli_c
        .map(|m| format!("{}C", m / 1000))
        .unwrap_or_else(|| "ERR".into())
}

#[cfg(target_os = "linux")]
fn fmt_vram(used: Option<u64>, total: Option<u64>) -> String {
    match (used, total) {
        (Some(u), Some(t)) => format!("{}/{}MB", u / (1024 * 1024), t / (1024 * 1024)),
        _ => "ERR".into(),
    }
}

// ── Non-Linux stub: no DRM sysfs, so telemetry is genuinely unavailable. ──
#[cfg(not(target_os = "linux"))]
pub struct GpuTelemetry;

#[cfg(not(target_os = "linux"))]
impl GpuTelemetry {
    pub fn discover() -> Option<Self> {
        None
    }
    pub fn label(&self) -> &str {
        "unavailable"
    }
    pub fn sample_line(&self) -> String {
        String::new()
    }
}
