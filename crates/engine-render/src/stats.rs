//! Per-frame stats published for game and editor code.
//!
//! The renderer knows the frame time and the swapchain extent; a component
//! reached through [`Component::update`](engine_core::Component::update)
//! knows neither — it receives `(dt, &Transform)` and nothing else. Same
//! global-static pattern as `asset::global` and `input`, for the same
//! reason: a component can reach a static but not `RenderContext`.

use std::sync::atomic::{AtomicU32, Ordering};

static DT: AtomicU32 = AtomicU32::new(0);
static FPS: AtomicU32 = AtomicU32::new(0);
static SCREEN_W: AtomicU32 = AtomicU32::new(0);
static SCREEN_H: AtomicU32 = AtomicU32::new(0);

/// EMA weight for the published frame rate. At 12 000 FPS an instantaneous
/// `1/dt` swings by thousands between consecutive frames, which is unusable
/// in a readout; the smoothed value is what a human can actually read.
/// [`dt`] stays raw for anything that needs the real measurement.
const FPS_SMOOTHING: f32 = 0.05;

/// Publish this frame's measurements. Called once per frame by the renderer,
/// before `Scene::update`, so a component's [`dt()`] agrees with the `dt` it
/// was handed.
pub(crate) fn publish(dt: f32, extent: [u32; 2]) {
    DT.store(dt.to_bits(), Ordering::Relaxed);

    let instant = 1.0 / dt.max(1e-6);
    let prev = f32::from_bits(FPS.load(Ordering::Relaxed));
    let smoothed = if prev > 0.0 {
        prev + (instant - prev) * FPS_SMOOTHING
    } else {
        instant
    };
    FPS.store(smoothed.to_bits(), Ordering::Relaxed);

    SCREEN_W.store(extent[0], Ordering::Relaxed);
    SCREEN_H.store(extent[1], Ordering::Relaxed);
}

/// Last frame's duration in seconds, clamped to 100 ms across stalls.
pub fn dt() -> f32 {
    f32::from_bits(DT.load(Ordering::Relaxed))
}

/// Smoothed frame rate. See [`FPS_SMOOTHING`] for why this is not `1/dt()`.
pub fn fps() -> f32 {
    f32::from_bits(FPS.load(Ordering::Relaxed))
}

/// Swapchain extent in physical pixels — the UI's coordinate space.
pub fn screen() -> [f32; 2] {
    [
        SCREEN_W.load(Ordering::Relaxed) as f32,
        SCREEN_H.load(Ordering::Relaxed) as f32,
    ]
}
