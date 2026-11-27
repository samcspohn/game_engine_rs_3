//! Vulkano-based renderer and windowing for the game engine.
//!
//! Public surface is [`Window`]. A typical setup:
//!
//! ```no_run
//! use engine_render::{Window, MeshRenderer};
//! use engine_core::transform::_Transform;
//! use engine_core::component::{Component, World};
//! use engine_core::reflect::Export;
//!
//! #[derive(Clone)]
//! struct Spinner;
//! impl Export for Spinner {}
//! impl Component for Spinner {
//!     fn update(&mut self, dt: f32, t: &engine_core::transform::Transform, _w: &World) {
//!         use glam::Quat;
//!         t.lock().rotate_by(Quat::from_rotation_y(dt));
//!     }
//! }
//!
//! let root = engine_core::new_world();
//! root.spawn(_Transform::default(), |mut e| {
//!     e.add_component(Spinner)
//!      .add_component(MeshRenderer::new("cube.mesh"));
//! });
//!
//! Window::new("My Game")
//!     .with_world(root)
//!     .run();
//! ```
//!
//! The renderer maintains **one reusable primary command buffer per swapchain
//! image**. Each per-image "frame slot" owns:
//!
//! * A host-mapped staging buffer of MVP matrices (`HOST_SEQUENTIAL_WRITE`).
//! * A device-local matrix buffer bound as a storage buffer to set 0.
//! * **Offscreen** color (`R16G16B16A16_SFLOAT`) + depth (`D32_SFLOAT`)
//!   attachments — the camera's render targets, never the swapchain image.
//! * A pre-recorded command buffer that copies staging → device, renders the
//!   scene into the offscreen color+depth, and finally `vkCmdBlitImage`s the
//!   offscreen color into the swapchain image. Vulkano auto-tracks the final
//!   layout transition to `PresentSrcKHR` on swapchain-owned images.
//!
//! Decoupling the camera's color target from the swapchain image is step 1 of
//! the multi-camera / post-processing roadmap (`todo.txt`): once the camera
//! owns its attachments, multiple cameras, mirrors, picture-in-picture, and
//! HDR → sRGB tonemapping all become "another pass before the present-blit."
//!
//! On the hot path the renderer (a) computes per-instance MVPs into the
//! staging buffer and (b) submits the pre-recorded CB. Slots are rebuilt only
//! when the swapchain or scene topology changes. This is the scaffolding for
//! a future GPU-driven indirect path with millions of objects — the staging
//! → device pattern is the same; only the draw call collapses to a single
//! `draw_indexed_indirect_count`.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use engine_core::worlds::{self, WorldHandle};
use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::{
        StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
    },
    device::DeviceFeatures,
    memory::allocator::StandardMemoryAllocator,
    query::{QueryPool, QueryResultFlags},
    swapchain::{PresentMode, SurfaceInfo},
};
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{WindowAttributes, WindowId},
};

use engine_core::util::parallel;

pub mod assets;
mod camera;
mod capture;
pub mod components;
mod debug_input;
mod frame;
mod frame_slot;
pub mod gizmo;
mod gpu;
mod gpu_mesh;
mod gpu_renderers;
mod gpu_telemetry;
pub mod input;
mod overlay;
mod scene;
mod shaders;
pub mod stats;
mod step;
mod swapchain;
mod transform_gpu;
pub mod ui;

use frame::RenderContext;
use gpu::{Gpu, Pipelines};
use step::Rebuild;
use swapchain::SwapchainRenderer;
use transform_gpu::StagingMemory;

pub use camera::{
    CameraHandle, CameraResolution, MAX_CAMERAS, camera_count, of_world as camera_of_world,
};
pub use components::MeshRenderer;
pub use gizmo::GizmoMode;
pub use input::{Input, KeyCode, MouseButton};
pub use scene::register_builtin_components;
pub use scene::{CameraComponent, OrbitController};

// ─────────────────────────────────────────────────────────────────────────────
// Pinned static thread pool (engine-core fork-join scheduler)
// ─────────────────────────────────────────────────────────────────────────────

/// Initialise the engine's **global** static thread pool with one worker
/// per logical core (minus one for the main thread), each pinned to its
/// assigned core via `core_affinity`. The **main thread is deliberately
/// left unpinned**: it spins on the dispatch barrier, and a pinned
/// spinning thread heats its core into throttling while forbidding the
/// OS from migrating it to a cooler one. Leaving main free to migrate
/// recovers turbo headroom on low-N / single-core-bound workloads.
///
/// Worker pinning eliminates per-frame jitter from the scheduler
/// bouncing the hot dirty-harvest / scatter-staging workers between
/// cores and lets the L1/L2 caches retain the SoT staging pages across
/// frames. Set `ENGINE_NO_PIN=1` to disable worker pinning entirely
/// (e.g. on laptops, shared CI boxes, or for thermal experiments).
///
/// `ENGINE_NUM_THREADS` (alias: `RAYON_NUM_THREADS` for back-compat with
/// existing benchmark scripts) sets the **total** participant count
/// (workers + main). Parse strictly — no fallback on a bad value.
///
/// Per project rules: **no fallbacks**. If the OS refuses to enumerate
/// cores, or any pin fails, we panic.
/// Worker count for the dedicated TRS-staging pool. Override with
/// `ENGINE_STAGING_THREADS`.
const STAGING_POOL_THREADS: usize = 16;

/// Dedicated pool for the TRS staging drain, with its workers confined to
/// the GPU's NUMA node.
///
/// The scatter compute pulls dirty transforms out of host-**cached**
/// staging, so every read snoops whichever socket's caches hold the lines
/// the staging workers just wrote. Writers spread across both sockets of a
/// 2P box make most of that 8 MB a remote fetch — 673µs versus 320µs for
/// the scatter, measured on this machine. Confining the *whole process* to
/// the GPU's node fixes it but costs every other subsystem half the
/// machine, so instead only the staging drain gets node-local workers.
///
/// Pinning trick: worker threads inherit the creating thread's affinity
/// mask, so this narrows the calling thread's mask, builds the pool, and
/// restores the original mask. No pool-implementation changes needed.
///
/// `None` on single-socket machines and wherever the kernel reports no GPU
/// affinity — callers fall back to the global pool.
fn staging_pool() -> Option<&'static parallel::Pool> {
    use engine_core::util::numa::{self, NumaTopology};
    use std::sync::OnceLock;

    static POOL: OnceLock<Option<parallel::Pool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let node = numa::gpu_numa_node()?;
        let topo = NumaTopology::detect().ok()?;
        if topo.num_nodes() <= 1 {
            return None;
        }
        let cpus = topo.cpus_of_node(node)?.to_vec();
        let n_threads = match std::env::var("ENGINE_STAGING_THREADS") {
            Ok(s) => s
                .parse::<usize>()
                .expect("ENGINE_STAGING_THREADS must parse as a positive integer"),
            Err(_) => STAGING_POOL_THREADS,
        };
        assert!(n_threads > 0, "staging pool needs at least one thread");

        // Narrow → spawn → restore. Failing to restore would silently
        // confine the main thread (and everything it later spawns) to the
        // GPU's node, which is exactly what this is trying to avoid, so
        // both affinity calls are fatal on error.
        let saved = numa::current_affinity().expect("sched_getaffinity failed");
        numa::restrict_affinity_to(&cpus)
            .unwrap_or_else(|e| panic!("sched_setaffinity to GPU node {node} failed: {e}"));
        let pool = parallel::Pool::new(parallel::BackendKind::from_env(), n_threads);
        numa::restrict_affinity_to(&saved)
            .unwrap_or_else(|e| panic!("failed to restore main-thread affinity: {e}"));

        println!(
            "[numa] TRS staging pool: {n_threads} thread(s) confined to the GPU's node {node} \
             ({} cpus); main pool unbound",
            cpus.len(),
        );
        Some(pool)
    })
    .as_ref()
}

fn init_pinned_thread_pool() {
    use engine_core::util::numa::{self, NumaTopology};

    // Whether to skip all CPU affinity pinning.
    // Set ENGINE_NO_PIN=1 (or =true) to disable; default is pinned.
    let no_pin = std::env::var("ENGINE_NO_PIN")
        .ok()
        .map(|v| match v.as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => panic!("ENGINE_NO_PIN must be 0/1/true/false, got {v:?}"),
        })
        .unwrap_or(false);

    // ── Confine every thread to the GPU's NUMA node ─────────────────────
    //
    // Done *first*, so the cpuset-filtered topology and default worker
    // count below simply observe the narrowed mask — exactly as they
    // already do under `numactl --cpunodebind`.
    //
    // The scatter compute pulls dirty transforms out of host-**cached**
    // staging, so each read snoops whichever socket's caches hold the
    // lines the staging workers just wrote. Letting those workers spread
    // across both sockets of a 2P box makes most of that 8 MB a remote
    // fetch: measured on this machine (2×128 CPU, GPU on node 0,
    // `--shapes 1000000`), the scatter runs 673µs unbound versus 320µs
    // confined, i.e. 11.9 GB/s versus 25 GB/s, for 637 → 813 FPS. It also
    // removes a large bimodal spread — which fraction of workers happened
    // to land on the far node varied run to run, so the same build would
    // measure anywhere from 320µs to 700µs.
    //
    // Binding the staging *pages* instead (`ENGINE_STAGING_NUMA_NODE`,
    // `mbind`) does nothing here: the cost is cache residency, not page
    // residency.
    //
    // Defaults **on**: this is the best-measured configuration (803 FPS
    // versus 622 unbound on `--shapes 1000000`). Confining the process does
    // cost every other subsystem half the machine, but the alternative —
    // binding only a separate staging pool — measured worse, because two
    // pools evict each other (see `ENGINE_STAGING_POOL` in the frame-loop note). One pool,
    // node-local, wins.
    //
    // No-op on single-socket machines (one node owns every CPU) and
    // wherever the kernel reports no GPU affinity.
    let gpu_node_affinity = std::env::var("ENGINE_GPU_NODE_AFFINITY")
        .ok()
        .map(|v| match v.as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => panic!("ENGINE_GPU_NODE_AFFINITY must be 0/1/true/false, got {v:?}"),
        })
        .unwrap_or(true);
    if gpu_node_affinity {
        if let (Some(node), Ok(topo)) = (numa::gpu_numa_node(), NumaTopology::detect()) {
            if topo.num_nodes() > 1 {
                let cpus = topo
                    .cpus_of_node(node)
                    .unwrap_or_else(|| panic!("GPU reports NUMA node {node}, absent from topology"))
                    .to_vec();
                numa::restrict_affinity_to(&cpus).unwrap_or_else(|e| {
                    panic!(
                        "sched_setaffinity to GPU node {node} ({} cpus) failed: {e}",
                        cpus.len()
                    )
                });
                println!(
                    "[numa] confined to GPU's node {node} ({} cpus) — \
                     keeps the scatter's host-cached reads socket-local",
                    cpus.len(),
                );
            }
        }
    }

    // Build a cpuset-filtered NUMA topology so that callers (including
    // future schedulers that pin) never try to pin to a CPU outside our
    // allowed set (matters under numactl --cpunodebind, cgroups, or taskset).
    // The current `my_thread_pool` is simple and does not pin, but we still
    // honour the topology computation so the worker count derived below
    // reflects the cpuset-filtered core count.
    let core_ids = core_affinity::get_core_ids()
        .expect("core_affinity::get_core_ids() returned None — cannot enumerate logical cores");
    assert!(
        !core_ids.is_empty(),
        "core_affinity returned an empty core list"
    );
    let available: std::collections::HashSet<usize> = core_ids.iter().map(|c| c.id).collect();

    let raw = NumaTopology::detect()
        .unwrap_or_else(|_| NumaTopology::single_node(core_ids.iter().map(|c| c.id).collect()));

    let topology: Vec<Vec<usize>> = raw
        .nodes()
        .iter()
        .map(|n| {
            n.cpus
                .iter()
                .copied()
                .filter(|c| available.contains(c))
                .collect::<Vec<usize>>()
        })
        .filter(|cpus| !cpus.is_empty())
        .collect();
    assert!(
        !topology.is_empty(),
        "no NUMA node has any CPU in the current cpuset",
    );

    // ENGINE_NUM_THREADS / RAYON_NUM_THREADS: total participant count
    // (workers + main thread). We subtract one for the main thread so the
    // user-specified value matches the overall thread budget. With no env
    // var, default to one worker per CPU in the cpuset-filtered topology.
    let n_workers =
        match std::env::var("ENGINE_NUM_THREADS").or_else(|_| std::env::var("RAYON_NUM_THREADS")) {
            Ok(s) => {
                let total = s.parse::<usize>().expect(
                    "ENGINE_NUM_THREADS / RAYON_NUM_THREADS must parse as a positive integer",
                );
                assert!(total > 0, "engine pool participant count must be > 0");
                total.saturating_sub(1).max(1)
            }
            Err(_) => topology.iter().map(|n| n.len()).sum::<usize>().max(1),
        };

    // Backend is selected via ENGINE_POOL_BACKEND (mypool | rayon | orx);
    // defaults to the in-tree work-stealing pool.
    let backend = parallel::BackendKind::from_env();
    let ok = parallel::global::init(backend, n_workers);
    assert!(ok, "parallel global pool already initialized");
    let _ = no_pin; // simple pool doesn't pin; flag preserved for future use.

    let n = parallel::global::num_threads();
    println!(
        "engine pool: {n} thread(s) on {backend:?} backend{}",
        if no_pin { " [pinning disabled]" } else { "" },
    );

    // Scene construction runs before this init, so any MeshRenderer built
    // there deferred its asset load; hand those to the pool now.
    engine_core::asset::flush_pending_loads();
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Triple-buffer depth: CPU can record frame N+1/N+2 while GPU renders N.
const MAX_FRAMES_IN_FLIGHT: usize = 4;

/// Number of host-staging slots (double buffering). The host writes slot
/// `k` while the GPU still reads slot `k ^ 1`, which is what lets
/// `TransformGpuShared::host_wait` gate on frame `N-2` instead of `N-1`.
pub(crate) const STAGING_SLOTS: usize = 2;

/// Sample the system clock only every N frames (must be a power of two).
const FRAMES_PER_FPS_SAMPLE: u32 = 128;

/// A raw pointer handed to parallel tasks that each touch a disjoint range.
pub(crate) struct SyncMut<T>(pub *mut T);
unsafe impl<T> Send for SyncMut<T> {}
unsafe impl<T> Sync for SyncMut<T> {}
impl<T> SyncMut<T> {
    #[inline(always)]
    pub fn get(&self) -> *mut T {
        self.0
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// An OS window backed by a Vulkan swapchain.
///
/// Holds a [`WorldHandle`] for each world it keeps alive. Worlds themselves
/// are engine-owned ([`worlds::new_world`]); the renderer sweeps every live
/// one each frame, which fans out to every registered [`Component::update`].
pub struct Window {
    title: String,
    worlds: Vec<WorldHandle>,
}

impl Window {
    /// Create a window descriptor with the given title.
    pub fn new(title: &str) -> Self {
        scene::register_builtin_components();
        Window {
            title: title.to_owned(),
            worlds: Vec::new(),
        }
    }

    /// Keep `world` alive for as long as the window runs. The first one given
    /// is what a camera naming no world of its own draws.
    ///
    /// Not the set that gets drawn: a world is drawn when a camera names it,
    /// including one minted mid-run. Every *live* world is swept too, whether
    /// the window was handed it or not — a component that made its own keeps
    /// it running by holding the handle.
    pub fn with_world(mut self, world: WorldHandle) -> Self {
        self.worlds.push(world);
        self
    }

    /// Open the OS window, initialise Vulkan, and block on the event loop.
    pub fn run(self) {
        init_pinned_thread_pool();
        // Before the event loop, so a harness can connect while the window is
        // still coming up rather than racing the first frame.
        debug_input::start();
        let event_loop = EventLoop::new().expect("Failed to create winit EventLoop");
        let mut app = RenderApp::new(self.title, self.worlds);
        event_loop
            .run_app(&mut app)
            .expect("Event loop exited with an error");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FPS tracker
// ─────────────────────────────────────────────────────────────────────────────

// ── Frame stats: FPS + per-phase timings ──────────────────────────────

/// Cumulative `(min, max, sum_ns, count)` for a single phase across the
/// FPS sample window. Avg is `sum_ns / count`.
#[derive(Default, Clone, Copy)]
struct PhaseAcc {
    min_ns: u64,
    max_ns: u64,
    sum_ns: u128,
    count: u64,
}

impl PhaseAcc {
    fn record(&mut self, ns: u64) {
        if self.count == 0 {
            self.min_ns = ns;
            self.max_ns = ns;
        } else {
            if ns < self.min_ns {
                self.min_ns = ns;
            }
            if ns > self.max_ns {
                self.max_ns = ns;
            }
        }
        self.sum_ns += ns as u128;
        self.count += 1;
    }

    /// Format as "min/avg/max µs" with one decimal place. Returns "—" if
    /// no samples were recorded in this window (happens for the very first
    /// FPS line if a phase didn't fire on every frame in the window).
    fn fmt_us(&self) -> String {
        if self.count == 0 {
            return "—".to_string();
        }
        let min = self.min_ns as f64 / 1000.0;
        let max = self.max_ns as f64 / 1000.0;
        let avg = (self.sum_ns as f64 / self.count as f64) / 1000.0;
        format!("{:>6.1}/{:>6.1}/{:>6.1}", min, avg, max)
    }

    /// Same min/avg/max shape for accumulators holding something other
    /// than nanoseconds (e.g. word counts), divided by `scale`.
    fn fmt_scaled(&self, scale: f64) -> String {
        if self.count == 0 {
            return "—".to_string();
        }
        let min = self.min_ns as f64 / scale;
        let max = self.max_ns as f64 / scale;
        let avg = (self.sum_ns as f64 / self.count as f64) / scale;
        format!("{:>7.0}/{:>7.0}/{:>7.0}", min, avg, max)
    }
}

/// Frame-time + per-phase telemetry, printed once per FPS sample window.
///
/// Each phase is recorded by calling the corresponding `record_*(ns)` from
/// the per-frame loop. The window is the same as `FpsTracker`'s
/// (`FRAMES_PER_FPS_SAMPLE` frames AND ≥ 1 second of wall time), so the
/// per-phase numbers line up 1:1 with the FPS line above them.
struct FrameStats {
    last_print: Instant,
    frame_count: u32,
    acquire: PhaseAcc,
    host_wait_compute: PhaseAcc,
    host_staging: PhaseAcc,
    staging_locks: PhaseAcc,
    staging_setup: PhaseAcc,
    staging_parallel: PhaseAcc,
    staging_parents: PhaseAcc,
    staging_steps: PhaseAcc,
    sim_update: PhaseAcc,
    /// Per-GPU-stage times from the in-CB timestamp queries (see
    /// `frame_slot::ts` for the stage layout): `[seam, scatter, mvp1,
    /// raster1, hiz, mvp2, raster2, blit]`. `seam` is not this frame's
    /// work — it is the previous submissions draining.
    gpu_stages: [PhaseAcc; 8],
    /// q8 → q7: this frame's own GPU execution time. Excludes the seam, so
    /// `seam + total` is the full q0→q7 span and `total` is the sum of the
    /// seven real stages.
    gpu_total: PhaseAcc,
    /// Scatter-block time (q8 → q1) split by staging slot. Diagnostic for
    /// whether the scatter's wide spread is per-slot or per-frame.
    scatter_by_slot: [PhaseAcc; STAGING_SLOTS],
    /// The scatter block subdivided: `[trs, spawn, clears+VP, signal]` (see
    /// `frame_slot::ts`'s q9..q11), split by staging slot so the first-used
    /// slot's penalty can be attributed to one of the four.
    scatter_parts_by_slot: [[PhaseAcc; 4]; STAGING_SLOTS],
    /// Dirty-word span per frame, attributed to the staging slot that
    /// received it. Companion to `scatter_by_slot`.
    prepass_words_by_slot: [PhaseAcc; STAGING_SLOTS],
    /// Best-effort AMD GPU telemetry, sampled once per print window. `None`
    /// when no `amdgpu` DRM node is present (non-AMD / non-Linux).
    gpu: Option<gpu_telemetry::GpuTelemetry>,
}

impl FrameStats {
    fn new() -> Self {
        let gpu = gpu_telemetry::GpuTelemetry::discover();
        match &gpu {
            Some(g) => println!("[gpu-telemetry] monitoring {}", g.label()),
            None => println!("[gpu-telemetry] disabled: no amdgpu DRM card found"),
        }
        Self {
            last_print: Instant::now(),
            frame_count: 0,
            acquire: PhaseAcc::default(),
            host_wait_compute: PhaseAcc::default(),
            host_staging: PhaseAcc::default(),
            staging_locks: PhaseAcc::default(),
            staging_setup: PhaseAcc::default(),
            staging_parallel: PhaseAcc::default(),
            staging_parents: PhaseAcc::default(),
            staging_steps: PhaseAcc::default(),
            sim_update: PhaseAcc::default(),
            gpu_stages: [PhaseAcc::default(); 8],
            gpu_total: PhaseAcc::default(),
            scatter_by_slot: [PhaseAcc::default(); STAGING_SLOTS],
            scatter_parts_by_slot: [[PhaseAcc::default(); 4]; STAGING_SLOTS],
            prepass_words_by_slot: [PhaseAcc::default(); STAGING_SLOTS],
            gpu,
        }
    }

    /// Background shader-clock sampler for [`StagingBalancer`], on the same
    /// card this prints telemetry for. `None` when there is no amdgpu card.
    fn spawn_sclk_monitor(&self) -> Option<gpu_telemetry::SclkMonitor> {
        self.gpu.as_ref().map(|g| g.spawn_sclk_monitor())
    }

    fn record_acquire(&mut self, ns: u64) {
        self.acquire.record(ns);
    }
    fn record_host_wait_compute(&mut self, ns: u64) {
        self.host_wait_compute.record(ns);
    }
    fn record_host_staging(&mut self, ns: u64) {
        self.host_staging.record(ns);
    }
    fn record_stage(&mut self, t: &transform_gpu::StageTimes, steps_ns: u64) {
        self.staging_locks.record(t.locks);
        self.staging_setup.record(t.setup);
        self.staging_parallel.record(t.parallel);
        self.staging_parents.record(t.parents);
        self.staging_steps.record(steps_ns);
    }
    fn record_sim_update(&mut self, ns: u64) {
        self.sim_update.record(ns);
    }
    /// Reads this primary's timestamps from its previous run, which `acquire`
    /// already waited for. A newly rebuilt slot has none, so it's skipped.
    fn read_gpu(
        &mut self,
        pool: &QueryPool,
        slot: usize,
        period_ns: f64,
        balancer: &mut StagingBalancer,
        trace: bool,
    ) {
        use frame_slot::ts::*;
        let mut t = [0u64; COUNT as usize];
        if !matches!(
            pool.get_results(0..COUNT, &mut t, QueryResultFlags::empty()),
            Ok(true)
        ) {
            return;
        }
        let d = |a: u32, b: u32| {
            (t[b as usize].saturating_sub(t[a as usize]) as f64 * period_ns) as u64
        };
        let upload = d(SEAM, UPLOAD);
        self.record_gpu_timestamps(&[
            d(START, SEAM),
            upload,
            d(UPLOAD, CULL1),
            d(CULL1, RASTER1),
            d(RASTER1, HIZ),
            d(HIZ, CULL2),
            d(CULL2, RASTER2),
            d(RASTER2, END),
            d(SEAM, END),
        ]);
        balancer.record_gpu(d(SEAM, END), upload);
        // Per slot, because a gap between slots comes from memory placement.
        self.scatter_by_slot[slot].record(upload);
        let parts = [
            d(SEAM, TRS),
            d(TRS, STEPS),
            d(STEPS, CLEARS),
            d(CLEARS, UPLOAD),
        ];
        for (acc, ns) in self.scatter_parts_by_slot[slot].iter_mut().zip(parts) {
            acc.record(ns);
        }
        if trace {
            println!(
                "[trace] slot{slot} scatter_us {:.1}",
                upload as f64 / 1000.0
            );
        }
    }
    /// Record one frame's GPU per-stage times. `deltas_ns[0..8]` are the
    /// seam + seven stage deltas, `deltas_ns[8]` the q0→q7 total — already
    /// converted from ticks to nanoseconds by the caller.
    fn record_gpu_timestamps(&mut self, deltas_ns: &[u64; 9]) {
        for (acc, &ns) in self.gpu_stages.iter_mut().zip(&deltas_ns[..8]) {
            acc.record(ns);
        }
        self.gpu_total.record(deltas_ns[8]);
    }

    /// `wait_mode` is tagged onto the sample line so an A/B log of the F7
    /// host-sync experiment is unambiguous about which mode produced it.
    fn tick(&mut self, wait_mode: &str, staging_mode: &str) {
        self.frame_count += 1;
        if self.frame_count & (FRAMES_PER_FPS_SAMPLE - 1) == 0 {
            let elapsed = self.last_print.elapsed();
            if elapsed.as_secs() >= 1 {
                let fps = self.frame_count as f64 / elapsed.as_secs_f64();
                println!(
                    "FPS: {:.0}  ({:.3} ms/frame)  [wait={} staging={}] | us min/avg/max  acquire {} | host_wait_compute {} | host_staging {} [locks {} | setup {} | parallel {} | parents {} | steps {}] | sim_update {}",
                    fps,
                    1000.0 / fps,
                    wait_mode,
                    staging_mode,
                    self.acquire.fmt_us(),
                    self.host_wait_compute.fmt_us(),
                    self.host_staging.fmt_us(),
                    self.staging_locks.fmt_us(),
                    self.staging_setup.fmt_us(),
                    self.staging_parallel.fmt_us(),
                    self.staging_parents.fmt_us(),
                    self.staging_steps.fmt_us(),
                    self.sim_update.fmt_us(),
                );
                println!(
                    "  gpu us min/avg/max  seam {} | scatter {} | mvp1 {} | raster1 {} | hiz {} | mvp2 {} | raster2 {} | blit {} | total {}",
                    self.gpu_stages[0].fmt_us(),
                    self.gpu_stages[1].fmt_us(),
                    self.gpu_stages[2].fmt_us(),
                    self.gpu_stages[3].fmt_us(),
                    self.gpu_stages[4].fmt_us(),
                    self.gpu_stages[5].fmt_us(),
                    self.gpu_stages[6].fmt_us(),
                    self.gpu_stages[7].fmt_us(),
                    self.gpu_total.fmt_us(),
                );
                let per_slot = |accs: &[PhaseAcc; STAGING_SLOTS], f: fn(&PhaseAcc) -> String| {
                    accs.iter()
                        .enumerate()
                        .map(|(i, a)| format!("slot{i} {}", f(a)))
                        .collect::<Vec<_>>()
                        .join(" | ")
                };
                println!(
                    "  scatter by staging slot  {}",
                    per_slot(&self.scatter_by_slot, PhaseAcc::fmt_us),
                );
                println!(
                    "  dirty words by slot      {}",
                    per_slot(&self.prepass_words_by_slot, |a| a.fmt_scaled(1.0)),
                );
                for (part, label) in ["trs", "spawn", "clears+vp", "signal"]
                    .iter()
                    .enumerate()
                    .map(|(i, l)| (i, l))
                {
                    let accs: [PhaseAcc; STAGING_SLOTS] =
                        std::array::from_fn(|s| self.scatter_parts_by_slot[s][part]);
                    println!(
                        "  {label:<10} by slot     {}",
                        per_slot(&accs, PhaseAcc::fmt_us)
                    );
                }
                if let Some(gpu) = &self.gpu {
                    println!("{}", gpu.sample_line());
                }
                self.frame_count = 0;
                self.last_print = Instant::now();
                self.acquire = PhaseAcc::default();
                self.host_wait_compute = PhaseAcc::default();
                self.host_staging = PhaseAcc::default();
                self.staging_locks = PhaseAcc::default();
                self.staging_setup = PhaseAcc::default();
                self.staging_parallel = PhaseAcc::default();
                self.staging_parents = PhaseAcc::default();
                self.staging_steps = PhaseAcc::default();
                self.sim_update = PhaseAcc::default();
                self.gpu_stages = [PhaseAcc::default(); 8];
                self.gpu_total = PhaseAcc::default();
                self.scatter_by_slot = [PhaseAcc::default(); STAGING_SLOTS];
                self.scatter_parts_by_slot = [[PhaseAcc::default(); 4]; STAGING_SLOTS];
                self.prepass_words_by_slot = [PhaseAcc::default(); STAGING_SLOTS];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Staging memory-type balancer
// ─────────────────────────────────────────────────────────────────────────────

/// Rolling minimum over the last `WINDOW`..`2 * WINDOW` samples, `None` until
/// the first window closes.
///
/// The balancer wants each side's *steady* cost, and every signal it feeds on
/// is a clean floor with spikes on top: across a window `host_staging` reads
/// `402.7 / 426.6 / 541.7` min/avg/max, `raster1` reads `241.6 / 254.0 /
/// 263.8`. The floors are stable to ~1% and the spikes are asset loading,
/// a scheduler hiccup or a DPM ramp — none of which the choice of staging
/// memory can do anything about. A mean chases them; a minimum ignores them,
/// which is also why it needs no warm-up period and no outlier rejection.
#[derive(Clone, Copy)]
struct RollingMin {
    /// Last closed window's minimum; `INFINITY` until one closes.
    closed: f64,
    open: f64,
    n: u32,
}

impl Default for RollingMin {
    fn default() -> Self {
        Self {
            closed: f64::INFINITY,
            open: f64::INFINITY,
            n: 0,
        }
    }
}

impl RollingMin {
    /// Samples per window. Long enough to step over a hitch, short enough to
    /// follow a camera pan — ~60ms at the frame rates this runs at.
    const WINDOW: u32 = 128;

    fn push(&mut self, v: f64) {
        self.open = self.open.min(v);
        self.n += 1;
        if self.n == Self::WINDOW {
            *self = Self {
                closed: self.open,
                open: f64::INFINITY,
                n: 0,
            };
        }
    }
    fn get(self) -> Option<f64> {
        self.closed.is_finite().then(|| self.closed.min(self.open))
    }
}

/// Runtime CPU↔GPU balancer for the TRS staging memory type.
///
/// [`StagingMemory`] moves the transform upload's cost between the two
/// processors — ~280µs of it at 1M shapes, near 1:1. A frame costs
/// `max(cpu_busy, gpu_busy)`, so the right mode is whichever minimises that
/// max, and which one that is flips with the scene's CPU:GPU ratio.
///
/// Split each side into its staging part and the rest, and predict the mode
/// we are *not* in from the last time we were in it:
///
/// ```text
/// frame(m) = max(cpu_rest + cpu_staging[m], gpu_rest + gpu_scatter[m])
/// ```
///
/// Only the staging terms are mode-dependent, and they are driven by the
/// dirty-word count rather than by the camera, so they stay valid while the
/// view changes. `cpu_rest` / `gpu_rest` are shared and refreshed every frame
/// whichever mode is selected — which is what lets the balancer follow a
/// scene sliding from GPU-bound to CPU-bound. Splitting `gpu_rest` per mode
/// instead breaks exactly that case: the idle mode's raster estimate freezes
/// at the load it had when it was last selected.
///
/// **GPU durations must be clock-normalised before they enter this model.**
/// A GPU that is not the bottleneck has idle gaps, DPM downclocks it, and
/// every stage stretches on identical work — measured across one switch,
/// sclk 1232 → 3139MHz and `raster1` 254 → 116µs. Comparing a downclocked
/// `gpu_rest` against a full-speed `cpu_rest` says the GPU cannot afford the
/// scatter, which is a trap: the CPU-heavy mode manufactures the evidence
/// that keeps it selected. [`SclkMonitor`] supplies the correction, and the
/// model then works in the clock the GPU would run at *if* it were the
/// bottleneck — the only case in which its duration decides the frame.
///
/// A switch costs a full staging + FrameSlot rebuild, hence
/// [`SWITCH_MARGIN`](Self::SWITCH_MARGIN) and [`MIN_DWELL`](Self::MIN_DWELL).
struct StagingBalancer {
    /// `false` pins the mode (`ENGINE_STAGING_MODE=cached|vram`, or F6).
    auto: bool,
    mode: StagingMemory,
    /// `host_staging` / GPU scatter-block minima, indexed by `mode as usize`.
    cpu_staging: [RollingMin; 2],
    gpu_scatter: [RollingMin; 2],
    /// Everything else on each side — mode-independent.
    cpu_rest: RollingMin,
    gpu_rest: RollingMin,
    /// Live shader clock, for normalising the GPU samples. `None` when the
    /// platform exposes no clock, in which case the balancer is disabled
    /// rather than run on timings it cannot compare.
    sclk: Option<gpu_telemetry::SclkMonitor>,
    frames_in_mode: u32,
    last_switch: Instant,
    pending: Option<StagingMemory>,
}

impl StagingBalancer {
    /// Samples discarded after a switch. The GPU timestamps read each frame
    /// come from that (image, slot) pair's *previous* submission — up to
    /// `MAX_FRAMES_IN_FLIGHT * STAGING_SLOTS` frames back — so they describe
    /// the old mode for a while after the buffers change.
    const SETTLE_FRAMES: u32 = (MAX_FRAMES_IN_FLIGHT * STAGING_SLOTS) as u32 * 4;
    /// Minimum time between switches; a switch reallocates every staging
    /// slot and re-records every FrameSlot primary.
    const MIN_DWELL: Duration = Duration::from_millis(250);
    /// Predicted frame-time win required to pay for that.
    const SWITCH_MARGIN: f64 = 0.05;

    fn new(sclk: Option<gpu_telemetry::SclkMonitor>) -> Self {
        let (mut auto, mode) = match std::env::var("ENGINE_STAGING_MODE").as_deref() {
            Ok("cached") => (false, StagingMemory::HostCached),
            Ok("vram") => (false, StagingMemory::DeviceWc),
            Ok("auto") | Err(_) => (true, StagingMemory::HostCached),
            Ok(other) => panic!("ENGINE_STAGING_MODE must be cached|vram|auto, got {other:?}"),
        };
        if auto && sclk.is_none() {
            auto = false;
            println!(
                "[staging] no shader-clock telemetry — GPU timings are not comparable across modes, auto disabled"
            );
        }
        println!(
            "[staging] {} mode, starting on {}",
            if auto { "auto" } else { "pinned" },
            mode.label(),
        );
        Self {
            auto,
            mode,
            cpu_staging: [RollingMin::default(); 2],
            gpu_scatter: [RollingMin::default(); 2],
            cpu_rest: RollingMin::default(),
            gpu_rest: RollingMin::default(),
            sclk,
            frames_in_mode: 0,
            last_switch: Instant::now(),
            pending: None,
        }
    }

    fn settled(&self) -> bool {
        self.auto && self.frames_in_mode >= Self::SETTLE_FRAMES
    }

    /// One frame's GPU timings, from the in-CB timestamp queries, rescaled to
    /// what they would have cost at the reference clock. Without that the two
    /// modes' GPU numbers are measured on what is effectively different
    /// hardware — see the type docs.
    fn record_gpu(&mut self, total_ns: u64, scatter_ns: u64) {
        // A zero delta means the frame never wrote those queries — a rebuild
        // frame, or the first submission into a fresh pool. That is missing
        // data, not free work, and a minimum would latch onto it for good.
        if total_ns == 0 || scatter_ns == 0 {
            return;
        }
        let Some(scale) = self.sclk.as_ref().and_then(|s| s.normalise()) else {
            return;
        };
        if !self.settled() {
            return;
        }
        self.gpu_scatter[self.mode as usize].push(scatter_ns as f64 * scale);
        self.gpu_rest
            .push(total_ns.saturating_sub(scatter_ns) as f64 * scale);
    }

    /// One frame's host timings: total CPU work (frame period minus the
    /// blocking waits on the GPU) and the staging drain's share of it.
    /// Advances the frame counter and re-evaluates the mode.
    fn record_cpu(&mut self, busy_ns: u64, staging_ns: u64) {
        self.frames_in_mode = self.frames_in_mode.saturating_add(1);
        if !self.settled() {
            return;
        }
        self.cpu_staging[self.mode as usize].push(staging_ns as f64);
        self.cpu_rest
            .push(busy_ns.saturating_sub(staging_ns) as f64);
        self.evaluate();
    }

    fn evaluate(&mut self) {
        if self.pending.is_some() || self.last_switch.elapsed() < Self::MIN_DWELL {
            return;
        }
        let (Some(cpu_rest), Some(gpu_rest)) = (self.cpu_rest.get(), self.gpu_rest.get()) else {
            return;
        };
        let predict = |m: StagingMemory| -> Option<f64> {
            let cpu = cpu_rest + self.cpu_staging[m as usize].get()?;
            let gpu = gpu_rest + self.gpu_scatter[m as usize].get()?;
            Some(cpu.max(gpu))
        };
        let other = self.mode.other();
        let Some(current) = predict(self.mode) else {
            return;
        };
        // Never-measured mode: probe it. One dwell of a possibly-worse mode
        // buys the only numbers that can rule it out.
        let win = match predict(other) {
            None => true,
            Some(o) => o < current * (1.0 - Self::SWITCH_MARGIN),
        };
        if win {
            // Every term the decision rested on, in µs. Only fires on a
            // switch, and without it a wrong choice is indistinguishable from
            // a wrong measurement.
            let us = |v: Option<f64>| match v {
                Some(v) => format!("{:.0}", v / 1000.0),
                None => "-".to_string(),
            };
            println!(
                "[staging] {:?} {}us -> {:?} {}us | cpu_rest {} gpu_rest {} staging {}/{} scatter {}/{}",
                self.mode,
                us(Some(current)),
                other,
                us(predict(other)),
                us(Some(cpu_rest)),
                us(self.gpu_rest.get()),
                us(self.cpu_staging[0].get()),
                us(self.cpu_staging[1].get()),
                us(self.gpu_scatter[0].get()),
                us(self.gpu_scatter[1].get()),
            );
            self.pending = Some(other);
        }
    }

    /// Mode the renderer should switch to this frame, if any. Taking it
    /// commits the switch: the settle window and dwell timer restart.
    fn take_pending(&mut self) -> Option<StagingMemory> {
        let mode = self.pending.take()?;
        self.mode = mode;
        self.frames_in_mode = 0;
        self.last_switch = Instant::now();
        Some(mode)
    }

    /// F6 cycles auto → cached → vram → auto.
    fn cycle(&mut self) {
        let (auto, mode) = match (self.auto, self.mode) {
            (true, _) => (false, StagingMemory::HostCached),
            (false, StagingMemory::HostCached) => (false, StagingMemory::DeviceWc),
            (false, StagingMemory::DeviceWc) => (true, self.mode),
        };
        self.auto = auto;
        if mode != self.mode {
            self.pending = Some(mode);
        }
        println!(
            "[staging] {} mode, {}",
            if auto { "auto" } else { "pinned" },
            mode.label(),
        );
    }

    /// Short tag for the FPS line.
    fn tag(&self) -> &'static str {
        match (self.auto, self.mode) {
            (true, StagingMemory::HostCached) => "auto:cached",
            (true, StagingMemory::DeviceWc) => "auto:vram",
            (false, StagingMemory::HostCached) => "cached",
            (false, StagingMemory::DeviceWc) => "vram",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RenderApp  (internal event-loop handler)
// ─────────────────────────────────────────────────────────────────────────────

/// All state that lives for the entire event-loop lifetime.
struct RenderApp {
    title: String,
    context: VulkanoContext,
    gpu: Gpu,
    swapchain_renderer: Option<SwapchainRenderer>,
    rcx: Option<RenderContext>,
    fps: FrameStats,
    /// Picks which side of the PCIe link pays for the transform upload.
    staging_balancer: StagingBalancer,
    /// The worlds this window keeps alive. A camera that lists none of them
    /// draws the first (see [`Window::with_world`]).
    worlds: Vec<WorldHandle>,
    last_frame_time: Option<Instant>,
    /// Absolute time for the UI pointer, since a double click spans frames.
    started: Instant,
    total_frames: u64,
    timestamp_period_ns: f64,
    /// F7 or `ENGINE_WAIT_MODE=frame` waits for the whole previous frame instead
    /// of its mid-frame signal. It doesn't change what's recorded.
    wait_on_frame: bool,
    knobs: Knobs,
}

/// Benchmark and trace switches, read once. What each measures is in
/// [frame-loop](../../docs/notes/frame-loop.md#knobs).
struct Knobs {
    scatter_trace: bool,
    ui_trace: bool,
    use_staging_pool: bool,
    cull_away: bool,
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1" || v == "true")
}

impl RenderApp {
    fn new(title: String, worlds: Vec<WorldHandle>) -> Self {
        let context = VulkanoContext::new(VulkanoConfig {
            device_features: DeviceFeatures {
                dynamic_rendering: true,
                // ADR-0004 Phase 1 (instanced indirect draw):
                // * `multi_draw_indirect` lets a single `vkCmdDrawIndexedIndirect`
                //   read more than one `DrawIndexedIndirectCommand` from the
                //   indirect buffer (we call it once per mesh group with
                //   drawCount = 1 today, but enable for future-proofing /
                //   multi-mesh scenes that batch into a single call).
                // * `draw_indirect_first_instance` lets per-draw structs set a
                //   non-zero `first_instance`, which is what makes
                //   `gl_InstanceIndex` index correctly into the per-camera MVP
                //   buffer when the same vkCmdDrawIndexedIndirect emits
                //   `instance_count` GPU-side instances per mesh.
                multi_draw_indirect: true,
                draw_indirect_first_instance: true,
                // `vkCmdDrawIndexedIndirectCount` — the compaction pass
                // (`shaders/draw_compact.comp`) writes `drawCount` on the
                // GPU, so the raster skips slots the cull left empty
                // without the host knowing how many survived.
                draw_indirect_count: true,
                // Material/texture pipeline:
                // * `shader_sampled_image_array_non_uniform_indexing` — the
                //   fragment shader indexes its fixed-size `sampler2D` array
                //   by the per-**instance** material's texture, which is NOT
                //   dynamically uniform within a draw (two instances of one
                //   mesh may carry different materials); the shader marks
                //   the index `nonuniformEXT`. Core in Vulkan 1.2's
                //   descriptor-indexing feature block.
                shader_sampled_image_array_non_uniform_indexing: true,
                // ADR-0003 (shared staging + timeline-semaphore sync):
                // We use a Vulkan timeline semaphore signaled at
                // `COMPUTE_SHADER` stage end of every submission to gate
                // host writes to the shared staging triple. Promoted to
                // core in Vulkan 1.2; still must be opted into via the
                // device features struct on devices that report 1.2+.
                timeline_semaphore: true,
                ..Default::default()
            },
            ..Default::default()
        });

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            StandardCommandBufferAllocatorCreateInfo {
                primary_buffer_count: 32,
                // Two secondaries per FrameSlot (scene + blit); allocate enough
                // headroom for several swapchain images per pool reset.
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
            context.device().clone(),
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            context.device().clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        let gpu = Gpu {
            queue: context.graphics_queue().clone(),
            memory: memory_allocator,
            cbs: command_buffer_allocator,
            sets: descriptor_set_allocator,
            pipelines: Pipelines::new(context.device()),
        };
        let fps = FrameStats::new();
        let staging_balancer = StagingBalancer::new(fps.spawn_sclk_monitor());
        let timestamp_period_ns = context
            .device()
            .physical_device()
            .properties()
            .timestamp_period as f64;

        RenderApp {
            title,
            context,
            gpu,
            swapchain_renderer: None,
            rcx: None,
            fps,
            staging_balancer,
            worlds,
            last_frame_time: None,
            started: Instant::now(),
            total_frames: 0,
            timestamp_period_ns,
            wait_on_frame: std::env::var("ENGINE_WAIT_MODE")
                .is_ok_and(|v| v.eq_ignore_ascii_case("frame")),
            knobs: Knobs {
                scatter_trace: env_flag("ENGINE_SCATTER_TRACE"),
                ui_trace: env_flag("ENGINE_UI_TRACE"),
                use_staging_pool: env_flag("ENGINE_STAGING_POOL"),
                cull_away: env_flag("ENGINE_CULL_AWAY"),
            },
        }
    }
}

impl ApplicationHandler for RenderApp {
    /// Called once at startup (and again on Android resume cycles).
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Drop stale renderer on mobile resume.
        self.swapchain_renderer = None;

        // ── Pick a present mode ahead of swapchain creation ─────────────────
        let probe_window = event_loop
            .create_window(WindowAttributes::default().with_title(self.title.clone()))
            .expect("Failed to create window");
        let probe_window = Arc::new(probe_window);
        let probe_surface = vulkano::swapchain::Surface::from_window(
            self.context.instance().clone(),
            probe_window.clone(),
        )
        .expect("Surface::from_window failed");
        let supported = self
            .context
            .device()
            .physical_device()
            .surface_present_modes(probe_surface.as_ref(), SurfaceInfo::default())
            .expect("Failed to query surface present modes");
        let chosen = if supported.contains(&PresentMode::Mailbox) {
            PresentMode::Mailbox
        } else if supported.contains(&PresentMode::Immediate) {
            PresentMode::Immediate
        } else {
            PresentMode::Fifo
        };
        println!("Present mode: {chosen:?}  (supported: {supported:?})");

        drop(probe_surface);
        drop(probe_window);

        // `ENGINE_WINDOW_SIZE=WxH` forces the initial size. Benchmark knob:
        // shrinking the viewport makes raster's pixel work negligible while
        // leaving the scene in view and the transform/cull work untouched,
        // which separates "raster is expensive" from "the scene is visible".
        let mut attrs = WindowAttributes::default().with_title(self.title.clone());
        if let Ok(spec) = std::env::var("ENGINE_WINDOW_SIZE") {
            let (w, h) = spec
                .split_once(['x', 'X'])
                .and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)))
                .expect("ENGINE_WINDOW_SIZE must look like 1920x1080");
            attrs = attrs.with_inner_size(winit::dpi::PhysicalSize::<u32>::new(w, h));
        }
        let real_window = event_loop
            .create_window(attrs)
            .expect("Failed to create window");

        let swapchain_renderer = SwapchainRenderer::new(
            self.context.instance().clone(),
            self.context.device().clone(),
            self.gpu.queue.clone(),
            real_window,
            chosen,
            MAX_FRAMES_IN_FLIGHT,
        );

        self.rcx = Some(RenderContext::new(
            &self.gpu,
            swapchain_renderer.image_views().to_vec(),
            swapchain_renderer.swapchain_format(),
            &self.worlds,
            self.staging_balancer.mode,
        ));
        self.swapchain_renderer = Some(swapchain_renderer);
        self.last_frame_time = Some(Instant::now());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Even before the renderer is ready, because components read input.
        input::global_mut().feed_window_event(&event);

        let Some(renderer) = self.swapchain_renderer.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => renderer.resize(),
            _ => {}
        }
    }

    /// One frame, run continuously (`ControlFlow::Poll`), one call per phase.
    /// The frame-loop note says what each phase may touch.
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        let (Some(renderer), Some(rcx)) = (self.swapchain_renderer.as_mut(), self.rcx.as_mut())
        else {
            return;
        };
        let gpu = &self.gpu;
        let frame_start = Instant::now();
        // Clamped, so a stall such as a window drag isn't one huge step.
        let dt = self
            .last_frame_time
            .map_or(0.0, |t| (frame_start - t).as_secs_f32())
            .min(0.1);
        self.last_frame_time = Some(frame_start);
        stats::publish(dt, rcx.extent());

        // ── Input, before anything reads it ─────────────────────────────────
        dispatch_input(self.started.elapsed().as_secs_f64());
        gizmo::update();

        // ── Boundary, the only point where `&mut World` is safe ─────────────
        let mut rebuild = rcx.retire_worlds();
        let live = worlds::live();
        // SAFETY: no sweep is running and nothing holds a `&World` from one.
        unsafe { worlds::apply_pending(&live) };
        components::forget_dead_worlds(&live);
        notify_instantiations();

        // ── Simulate components, then each world's systems ──────────────────
        let sim = Instant::now();
        worlds::sweep_all(&live, dt);
        self.fps.record_sim_update(sim.elapsed().as_nanos() as u64);
        rebuild = rebuild.max(rcx.adopt_worlds(gpu, self.staging_balancer.mode));

        // ── Acquire ─────────────────────────────────────────────────────────
        let acquire = Instant::now();
        let Some(frame) = rcx.acquire(renderer, gpu) else {
            return; // out of date or minimised
        };
        let acquire_ns = acquire.elapsed().as_nanos() as u64;
        self.fps.record_acquire(acquire_ns);
        let (image, slot) = (
            frame.image_index as usize,
            rcx.transform_shared.write_slot(),
        );
        self.fps.read_gpu(
            rcx.timestamp_pool(image, slot),
            slot,
            self.timestamp_period_ns,
            &mut self.staging_balancer,
            self.knobs.scatter_trace,
        );

        // ── Prepare buffers and re-record what that invalidated ─────────────
        rcx.layout_ui();
        rebuild = rebuild.max(rcx.prepare(gpu, self.staging_balancer.take_pending()));
        rebuild = rcx.sync_cameras(gpu, rebuild);
        rcx.drive_cameras();
        rebuild = rebuild.max(debug_keys(
            rcx,
            gpu,
            &mut self.staging_balancer,
            &mut self.wait_on_frame,
            self.knobs.cull_away,
        ));
        if rebuild >= Rebuild::FrameSlots {
            rcx.rebuild_frame_slots(gpu);
        }
        input::global_mut().end_frame();

        // ── Host wait, then stage this slot ─────────────────────────────────
        let wait = Instant::now();
        match self.wait_on_frame {
            true => renderer.wait_previous_frame(),
            false => rcx.transform_shared.host_wait(),
        }
        let host_wait_ns = wait.elapsed().as_nanos() as u64;
        self.fps.record_host_wait_compute(host_wait_ns);
        let staging = Instant::now();
        let pool = self.knobs.use_staging_pool.then(staging_pool).flatten();
        rcx.stage(slot, pool, &mut self.fps);
        let host_staging_ns = staging.elapsed().as_nanos() as u64;
        self.fps.record_host_staging(host_staging_ns);
        if self.knobs.ui_trace && rcx.ui_gpu.last_dirty_words() != 0 {
            println!(
                "[ui] frame {} dirty_words={} prims={}",
                self.total_frames,
                rcx.ui_gpu.last_dirty_words(),
                ui::ui().prim_count(),
            );
        }
        if let Some(wr) = rcx.worlds.first() {
            let (slot, words) = wr.transforms.last_prepass_span_words();
            self.fps.prepass_words_by_slot[slot].record(words as u64);
        }

        // ── Submit ──────────────────────────────────────────────────────────
        // A `poke shot` capture is recorded into this submission, after the primary.
        let shot = capture::take(
            rcx.swapchain_image_views[image].image(),
            &gpu.memory,
            &gpu.cbs,
            gpu.queue.queue_family_index(),
        );
        let after = shot.as_ref().map(|c| c.cb.clone());
        renderer.submit_and_present(
            frame,
            None,
            rcx.frame_cb(image, slot),
            Vec::new(),
            Vec::new(),
            after,
        );
        if let Some(shot) = shot {
            renderer.wait_previous_frame();
            shot.finish();
        }
        rcx.transform_shared.end_frame();

        // ── Stats ───────────────────────────────────────────────────────────
        // CPU busy time is the frame minus the two waits on the GPU.
        let busy = (frame_start.elapsed().as_nanos() as u64)
            .saturating_sub(acquire_ns)
            .saturating_sub(host_wait_ns);
        self.staging_balancer.record_cpu(busy, host_staging_ns);
        let wait_mode = if self.wait_on_frame {
            "frame"
        } else {
            "compute"
        };
        self.fps.tick(wait_mode, self.staging_balancer.tag());
        self.total_frames += 1;
        // By frame 120 every staging page has been faulted in.
        if self.total_frames == 120 {
            if let Some(wr) = rcx.worlds.first() {
                wr.transforms.report_staging_residency();
            }
        }
    }
}

/// Pointer, right button, keys, then handlers, in that order, each reading
/// what the one before settled. All of it runs before any component's `update`.
fn dispatch_input(now: f64) {
    // Apply `ENGINE_DEBUG_INPUT`'s injected input before anything reads input.
    debug_input::pump(input::global_mut());
    let inp = input::global();
    let c = inp.cursor_position();
    let mut ui = ui::ui();
    ui.update_pointer(
        [c.x, c.y],
        inp.mouse_pressed(MouseButton::Left),
        inp.mouse_released(MouseButton::Left),
        inp.scroll_delta(),
        now,
    );
    ui.update_secondary(
        inp.mouse_pressed(MouseButton::Right),
        inp.mouse_released(MouseButton::Right),
    );
    // A click that moves focus is applied before the keys typed after it.
    ui.update_keyboard(inp.keystrokes());
    // Handlers run only here, so locks are always taken UI first, then
    // component, never the reverse.
    ui.dispatch(now);
    ui.publish_capture();
}

/// Tells the UI root when a scene file was instantiated, since nothing in
/// the UI asked for it.
fn notify_instantiations() {
    static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let built = engine_core::scene_asset::instantiations();
    if SEEN.swap(built, std::sync::atomic::Ordering::Relaxed) != built {
        ui::defer(|ui| {
            let root = ui.root();
            ui.notify(root);
        });
    }
}

/// F6 staging memory, F7 host wait, F8 occlusion culling, F9 cull lock.
fn debug_keys(
    rcx: &mut RenderContext,
    gpu: &Gpu,
    balancer: &mut StagingBalancer,
    wait_on_frame: &mut bool,
    cull_away: bool,
) -> Rebuild {
    if input::key_pressed(KeyCode::F6) {
        balancer.cycle();
    }
    if input::key_pressed(KeyCode::F7) {
        *wait_on_frame = !*wait_on_frame;
        println!(
            "[wait mode] {}",
            match *wait_on_frame {
                true => "previous FRAME (in_flight fence — uncontended staging)",
                false => "previous COMPUTE (mid-CB early wake — pipelined staging)",
            }
        );
    }
    if cull_away && !rcx.cull_locked() {
        rcx.toggle_cull_lock(true);
        println!("[cull-away] frustum locked off-scene; pass 1 should draw nothing");
    }
    if input::key_pressed(KeyCode::F9) {
        rcx.toggle_cull_lock(false);
    }
    match input::key_pressed(KeyCode::F8) {
        true => rcx.toggle_occlusion(gpu),
        false => Rebuild::None,
    }
}
