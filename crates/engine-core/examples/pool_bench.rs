//! Micro-benchmark for the `util::parallel` backends.
//!
//!     cargo run -p engine-core --example pool_bench --release [num_threads]
//!
//! Times the two regimes that matter to the engine:
//!   * dispatch overhead — many back-to-back `parallel_for`s with a
//!     near-empty body (the per-frame sim/staging pattern);
//!   * throughput — a large summation where the body dominates.

use engine_core::util::parallel::{BackendKind, Pool};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct SyncPtr<T>(*mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("num_threads must be an integer"))
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
        });
    println!("threads: {threads}\n");

    for kind in [
        BackendKind::MyPool,
        BackendKind::Rayon,
        BackendKind::RayonBroadcast,
        BackendKind::Orx,
    ] {
        let pool = Pool::new(kind, threads);

        // Warm-up: fault in stacks, spin up lazy threads.
        for _ in 0..100 {
            pool.parallel_for(0..1024, |r| {
                std::hint::black_box(r.len());
            });
        }

        // ── Dispatch overhead: tiny body, many dispatches ────────────
        const DISPATCHES: usize = 2_000;
        const SMALL_N: usize = 10_000;
        let sink = AtomicUsize::new(0);
        let t0 = Instant::now();
        for _ in 0..DISPATCHES {
            pool.parallel_for(0..SMALL_N, |r| {
                let mut acc = 0usize;
                for i in r {
                    acc = acc.wrapping_add(std::hint::black_box(i));
                }
                sink.fetch_add(acc, Ordering::Relaxed);
            });
        }
        let dispatch = t0.elapsed();

        // ── Throughput: heavy body ───────────────────────────────────
        const BIG_N: usize = 8_000_000;
        const REPS: usize = 20;
        let t0 = Instant::now();
        for _ in 0..REPS {
            pool.parallel_for(0..BIG_N, |r| {
                let mut acc = 0u64;
                for i in r {
                    // A few flops per item so memory isn't the only cost.
                    let x = i as u64;
                    acc = acc.wrapping_add(x.wrapping_mul(x) ^ (x >> 3));
                }
                sink.fetch_add(acc as usize, Ordering::Relaxed);
            });
        }
        let throughput = t0.elapsed();

        // ── Affinity: per-frame sweep over persistent 64-byte items ──
        // Mimics the sim update: the same array is mutated in place
        // every dispatch. A backend with a stable chunk→thread mapping
        // keeps each core's slice warm in its private cache across
        // dispatches; a dynamic scheduler reshuffles ownership and pays
        // cross-core (and cross-NUMA-node) traffic instead.
        const AFF_N: usize = 1_000_000;
        const AFF_REPS: usize = 500;
        let mut items = vec![[1.0f32; 16]; AFF_N]; // 64 B per item
        let ptr = SyncPtr(items.as_mut_ptr());
        // Warm pass so first-touch page placement is settled.
        pool.parallel_for(0..AFF_N, |r| {
            let _ = &ptr;
            for i in r {
                let item = unsafe { &mut *ptr.0.add(i) };
                for v in item.iter_mut() {
                    *v = *v * 1.0001 + 0.5;
                }
            }
        });
        let t0 = Instant::now();
        for _ in 0..AFF_REPS {
            pool.parallel_for(0..AFF_N, |r| {
                let _ = &ptr;
                for i in r {
                    // SAFETY: sub-ranges within one dispatch are disjoint
                    // and the dispatch blocks until every chunk is done.
                    let item = unsafe { &mut *ptr.0.add(i) };
                    for v in item.iter_mut() {
                        *v = *v * 1.0001 + 0.5;
                    }
                }
            });
        }
        let affinity = t0.elapsed();
        std::hint::black_box(&items);

        println!(
            "{kind:?}: dispatch {:>8.2} us/call ({DISPATCHES} x {SMALL_N} items) | \
             throughput {:>7.2} ms/pass ({REPS} x {BIG_N} items) | \
             affinity {:>8.2} us/pass ({AFF_REPS} x {AFF_N} x 64B)",
            dispatch.as_secs_f64() * 1e6 / DISPATCHES as f64,
            throughput.as_secs_f64() * 1e3 / REPS as f64,
            affinity.as_secs_f64() * 1e6 / AFF_REPS as f64,
        );
        std::hint::black_box(sink.load(Ordering::Relaxed));
    }
}
