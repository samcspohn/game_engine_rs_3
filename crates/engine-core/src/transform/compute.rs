//! CPU-side transform compute helpers.
//!
//! GPU-accelerated transform propagation (`TransformCompute`) lives in
//! `engine-render` and depends on Vulkano.  This module contains only the
//! lightweight types that *every* crate can depend on without pulling in a
//! graphics driver: [`PerfCounter`], [`PerfCounterDisplay`], and
//! [`StaticPerfCounters`].

use std::{collections::HashMap, fmt::Debug};

// ---------------------------------------------------------------------------
// PerfCounter
// ---------------------------------------------------------------------------

/// A simple wall-clock performance counter.
///
/// Call [`start`](PerfCounter::start) before the work and
/// [`stop`](PerfCounter::stop) after it.  The counter accumulates samples so
/// you can compute a running average with
/// [`display_and_reset`](PerfCounter::display_and_reset).
pub struct PerfCounter {
    start_time: Option<std::time::Instant>,
    sum: f32,
    count: u32,
}

impl PerfCounter {
    pub fn new() -> Self {
        Self {
            start_time: None,
            sum: 0.0,
            count: 0,
        }
    }

    #[inline]
    pub fn start(&mut self) {
        self.start_time = Some(std::time::Instant::now());
    }

    #[inline]
    pub fn stop(&mut self) {
        if let Some(start) = self.start_time.take() {
            self.sum += start.elapsed().as_secs_f32();
            self.count += 1;
        }
    }

    fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }

    /// Returns the average time in milliseconds over all accumulated samples
    /// and resets the counter.
    pub fn display_and_reset(&mut self) -> PerfCounterDisplay {
        let avg_ms = if self.count > 0 {
            (self.sum / self.count as f32) * 1000.0
        } else {
            0.0
        };
        self.reset();
        PerfCounterDisplay(avg_ms)
    }
}

impl Default for PerfCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for PerfCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let avg_ms = if self.count > 0 {
            (self.sum / self.count as f32) * 1000.0
        } else {
            0.0
        };
        write!(f, "{:.2} ms", avg_ms)
    }
}

// ---------------------------------------------------------------------------
// PerfCounterDisplay
// ---------------------------------------------------------------------------

/// Snapshot of a [`PerfCounter`]'s average, ready to print.
pub struct PerfCounterDisplay(f32);

impl Debug for PerfCounterDisplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.2} ms", self.0)
    }
}

// ---------------------------------------------------------------------------
// StaticPerfCounters
// ---------------------------------------------------------------------------

/// A collection of [`PerfCounter`]s keyed by `&'static str`.
///
/// Using static string keys avoids per-frame heap allocations after the first
/// lookup for each key.
pub struct StaticPerfCounters {
    counters: HashMap<&'static str, PerfCounter>,
}

impl StaticPerfCounters {
    pub fn new() -> Self {
        Self {
            counters: HashMap::with_capacity(32),
        }
    }

    /// Return (or insert) the counter for `key`.
    #[inline]
    pub fn get(&mut self, key: &'static str) -> &mut PerfCounter {
        self.counters.entry(key).or_insert_with(PerfCounter::new)
    }

    /// Start timing for `key`.
    #[inline]
    pub fn start(&mut self, key: &'static str) {
        self.get(key).start();
    }

    /// Stop timing for `key`.
    #[inline]
    pub fn stop(&mut self, key: &'static str) {
        if let Some(counter) = self.counters.get_mut(key) {
            counter.stop();
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&&'static str, &PerfCounter)> {
        self.counters.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&&'static str, &mut PerfCounter)> {
        self.counters.iter_mut()
    }
}

impl Default for StaticPerfCounters {
    fn default() -> Self {
        Self::new()
    }
}
