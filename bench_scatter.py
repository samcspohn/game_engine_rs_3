#!/usr/bin/env python3
"""
bench_scatter.py  –  benchmark and compare any two (or more) git branches

Workflow
--------
1. For each branch, check it out and build the `editor` binary into a
   branch-specific target directory (so cargo never clobbers cross-branch
   incremental artefacts).  The finished binary is copied to bench_binaries/.

2. Restore the original git branch.

3. Run each binary for a configurable duration while parsing FPS lines from
   stdout.  A configurable warmup window is discarded at the start.

4. Print a side-by-side statistics table with absolute and percentage deltas.

Usage
-----
    # Compare any two branches
    python3 bench_scatter.py transform-gpu transform-gpu-super-kernel

    # Compare three or more branches
    python3 bench_scatter.py main feature-a feature-b

    # Shorter run, skip recompiling
    python3 bench_scatter.py --duration 30 --skip-build branch-a branch-b

    # Pool 3 runs per binary for more statistical confidence
    python3 bench_scatter.py --runs 3 branch-a branch-b
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent
BENCH_DIR = REPO_ROOT / "bench_binaries"
FPS_RE = re.compile(r"FPS:\s+(\d+)")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def branch_to_label(branch: str) -> str:
    """Convert an arbitrary git branch name into a filesystem-safe label.

    Replaces every run of characters that are not alphanumeric or underscores
    with a single underscore and strips leading/trailing underscores.

    Examples
    --------
    'transform-gpu'              → 'transform_gpu'
    'feature/my-cool-thing'      → 'feature_my_cool_thing'
    'refs/heads/main'            → 'refs_heads_main'
    """
    label = re.sub(r"[^A-Za-z0-9]+", "_", branch).strip("_")
    return label or "branch"


# ─────────────────────────────────────────────────────────────────────────────
# Git helpers
# ─────────────────────────────────────────────────────────────────────────────


def current_branch() -> str:
    return (
        subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
        )
        .decode()
        .strip()
    )


def checkout(branch: str) -> None:
    print(f"  git checkout {branch}")
    subprocess.check_call(
        ["git", "checkout", branch],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def short_sha(branch: str) -> str:
    return (
        subprocess.check_output(
            ["git", "rev-parse", "--short", branch],
            cwd=REPO_ROOT,
        )
        .decode()
        .strip()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────


def build_editor(label: str) -> Path:
    """Build the editor binary for the currently checked-out branch.

    Each label gets its own --target-dir so incremental artefacts for both
    branches can coexist without cargo invalidating the other's cache.
    The finished binary is copied to bench_binaries/editor-<label>.
    """
    target_dir = REPO_ROOT / "target" / f"bench-{label}"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"  cargo build --release -p editor  →  target/bench-{label}/")
    result = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--package",
            "editor",
            "--target-dir",
            str(target_dir),
        ],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        sys.exit(
            f"\nERROR: cargo build failed for label '{label}' (exit {result.returncode})"
        )

    src = target_dir / "release" / "editor"
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    dest = BENCH_DIR / f"editor-{label}"
    shutil.copy2(src, dest)
    dest.chmod(0o755)
    print(f"  copied  →  {dest.relative_to(REPO_ROOT)}")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


def collect_fps(binary: Path, duration: float, warmup: float) -> list[int]:
    """Launch the binary, discard `warmup` seconds of FPS output, then collect
    samples for `duration` seconds, then terminate.

    The binary is run from REPO_ROOT so that '--project crates/test-game'
    resolves correctly.
    """
    cmd = [str(binary), "--project", "crates/test-game"]
    print(
        f"  {binary.name}  ({warmup:.0f}s warmup + {duration:.0f}s collection)  …",
        flush=True,
    )

    samples: list[int] = []
    start_time: Optional[float] = None
    collection_done = threading.Event()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
        text=True,
        bufsize=1,
    )

    def reader() -> None:
        nonlocal start_time
        assert proc.stdout is not None
        for line in proc.stdout:
            m = FPS_RE.search(line)
            if not m:
                continue
            now = time.monotonic()
            if start_time is None:
                start_time = now  # clock starts on first FPS line
            elapsed = now - start_time
            if elapsed < warmup:
                continue  # discard warmup window
            if elapsed > warmup + duration:
                collection_done.set()
                break  # enough data; stop reading
            samples.append(int(m.group(1)))
        collection_done.set()  # ensure event is always set

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    # Wait until collection is complete or the process exits early.
    collection_done.wait(timeout=warmup + duration + 30.0)

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    t.join(timeout=3)

    print(f"  collected {len(samples)} samples")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────


def compute_stats(samples: list[int]) -> dict:
    if not samples:
        return {
            k: float("nan")
            for k in ("n", "mean", "median", "stdev", "min", "max", "p1", "p99")
        }
    s = sorted(samples)
    n = len(s)
    return {
        "n": n,
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "stdev": statistics.stdev(s) if n > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
        "p1": s[max(0, int(n * 0.01) - 1)],
        "p99": s[min(n - 1, int(n * 0.99))],
    }


def merge_runs(all_runs: list[list[int]]) -> list[int]:
    """Flatten multiple runs into one sample list."""
    merged: list[int] = []
    for r in all_runs:
        merged.extend(r)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

METRICS = [
    ("n", "samples", "{:>20.0f}", False),
    ("mean", "mean", "{:>20.1f}", True),
    ("median", "median", "{:>20.1f}", True),
    ("stdev", "stdev", "{:>20.1f}", True),
    ("min", "min", "{:>20.0f}", True),
    ("max", "max", "{:>20.0f}", True),
    ("p1", "p1  (1%)", "{:>20.0f}", True),
    ("p99", "p99 (99%)", "{:>20.0f}", True),
]


def print_report(
    labels: list[str],
    results: dict[str, list[int]],
    shas: dict[str, str],
) -> None:
    data = {lbl: compute_stats(results[lbl]) for lbl in labels}

    COL = 22
    W_LBL = 12
    has_two = len(labels) == 2

    # Header labels show "label (sha)"
    hdrs = [f"{lbl}\n({shas.get(lbl, '?')})" for lbl in labels]

    # Top line widths
    total_w = W_LBL + COL * len(labels) + (COL + 9 if has_two else 0) + 2
    sep = "─" * total_w

    print()
    print("┌" + sep + "┐")
    print("│" + " SCATTER BENCHMARK  (FPS – higher is better) ".center(total_w) + "│")
    print("├" + sep + "┤")

    # Column header row
    hrow = f"│ {'metric':<{W_LBL - 1}}"
    hrow += "".join(f"{lbl:>{COL}}" for lbl in labels)
    if has_two:
        hrow += f"{'Δ (fused-single)':>{COL}}  {'Δ%':>7}"
    hrow += " │"
    print(hrow)
    print("├" + sep + "┤")

    for key, display, fmt, show_delta in METRICS:
        row = f"│ {display:<{W_LBL - 1}}"
        vals = []
        for lbl in labels:
            v = data[lbl][key]
            row += fmt.format(v)
            vals.append(v)
        if has_two and show_delta:
            delta = vals[1] - vals[0]
            delta_pc = 100.0 * delta / vals[0] if vals[0] else float("nan")
            row += f"{delta:>{COL}.1f}  {delta_pc:>+7.2f}%"
        elif has_two:
            row += " " * (COL + 9)
        row += " │"
        print(row)

    print("└" + sep + "┘")
    print()

    if has_two:
        a_mean = data[labels[0]]["mean"]
        b_mean = data[labels[1]]["mean"]
        if a_mean != a_mean or b_mean != b_mean:  # NaN check
            print("  (insufficient data for winner determination)")
        else:
            delta_pc = 100.0 * (b_mean - a_mean) / a_mean
            winner = labels[1] if b_mean > a_mean else labels[0]
            print(f"  Winner (mean FPS): {winner}  ({delta_pc:+.2f}%)")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CSV dump (for later analysis / plotting)
# ─────────────────────────────────────────────────────────────────────────────


def dump_csv(results: dict[str, list[int]], path: Path) -> None:
    max_len = max((len(v) for v in results.values()), default=0)
    labels = list(results.keys())
    with path.open("w") as f:
        f.write(",".join(labels) + "\n")
        for i in range(max_len):
            row = []
            for lbl in labels:
                s = results[lbl]
                row.append(str(s[i]) if i < len(s) else "")
            f.write(",".join(row) + "\n")
    print(f"  raw samples saved to {path.relative_to(REPO_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark single-scatter vs fused-scatter editor builds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="seconds of FPS data to collect per run (default 60)",
    )
    ap.add_argument(
        "--warmup",
        type=float,
        default=5.0,
        help="warmup seconds to discard at the start of each run (default 5)",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=1,
        help="number of timed runs per binary; samples are pooled (default 1)",
    )
    ap.add_argument(
        "--skip-build",
        action="store_true",
        help="skip compilation and use existing binaries in bench_binaries/",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=BENCH_DIR / "results.csv",
        help="path to write raw FPS samples as CSV",
    )
    ap.add_argument(
        "branches",
        nargs="+",
        metavar="BRANCH",
        help="two or more git branch names to benchmark (e.g. main feature-x)",
    )
    args = ap.parse_args()

    if len(args.branches) < 2:
        ap.error("provide at least two branch names to compare")

    original_branch = current_branch()
    print(f"Current branch : {original_branch}")
    print(f"Repo root      : {REPO_ROOT}\n")

    binaries: dict[str, Path] = {}
    shas: dict[str, str] = {}

    # Build the branch → label mapping from positional args.
    # Duplicate branch names are allowed (e.g. bench the same branch twice)
    # but their labels are disambiguated with a numeric suffix.
    seen_labels: dict[str, int] = {}
    branches: dict[str, str] = {}  # label → branch
    for branch in args.branches:
        base = branch_to_label(branch)
        count = seen_labels.get(base, 0)
        seen_labels[base] = count + 1
        label = base if count == 0 else f"{base}_{count}"
        branches[label] = branch

    # ── Build ─────────────────────────────────────────────────────────────────
    if not args.skip_build:
        for label, branch in branches.items():
            print(f"[BUILD] {label}  ←  {branch}")
            shas[label] = short_sha(branch)
            checkout(branch)
            binaries[label] = build_editor(label)
            print()

        print(f"[GIT] restoring original branch: {original_branch}")
        checkout(original_branch)
        print()
    else:
        print("[BUILD] skipped — using existing binaries in bench_binaries/\n")
        for label, branch in branches.items():
            p = BENCH_DIR / f"editor-{label}"
            if not p.exists():
                sys.exit(
                    f"ERROR: {p} not found.\n"
                    f"       Run without --skip-build to compile first."
                )
            binaries[label] = p
            # Try to resolve SHA even without checkout.
            try:
                shas[label] = short_sha(branch)
            except subprocess.CalledProcessError:
                shas[label] = "unknown"

    # ── Run ───────────────────────────────────────────────────────────────────
    results: dict[str, list[int]] = {lbl: [] for lbl in branches}

    for run_idx in range(args.runs):
        if args.runs > 1:
            print(f"══ Run {run_idx + 1}/{args.runs} ══")
        for label, binary in binaries.items():
            print(f"[RUN] {label}  (run {run_idx + 1})")
            samples = collect_fps(binary, args.duration, args.warmup)
            if not samples:
                print(
                    f"  WARNING: no FPS samples collected — check that the window "
                    f"opens and the engine prints 'FPS: ...' lines to stdout."
                )
            results[label].extend(samples)
            print()

    # ── Report ────────────────────────────────────────────────────────────────
    labels = list(branches.keys())
    print_report(labels, results, shas)

    # Dump CSV for offline analysis / plotting.
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    dump_csv(results, args.csv)


if __name__ == "__main__":
    main()
