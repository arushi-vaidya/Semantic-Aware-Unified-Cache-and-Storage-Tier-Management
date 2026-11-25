"""DRAM -> SSD -> HDD tiering benchmark (uses SAGE survival curves cached by bench_cache.py)."""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bench_cache import CACHE_DIR, OUT_DIR, SAGE_HMULT, TEST_TRACES  # noqa: E402
from src.sim.policies import S4FIFO_GRID  # noqa: E402
from src.sim.tiered import run_cascade, run_sage_tiered, tune_s4fifo_cascade  # noqa: E402
from src.sim.traces import load_trace  # noqa: E402

POLICIES = ["LRU-cascade", "ARC-cascade", "S4-FIFO-cascade", "SR-S3-FIFO-cascade", "SAGE-L1+LRU-SSD", "SAGE-Unified"]
SIZE_PAIRS = [(0.005, 0.025), (0.01, 0.05), (0.02, 0.10)]


def job(args):
    name, n, f1, f2, policy = args
    trace = load_trace(name, n)
    c1, c2 = max(16, int(f1 * trace.n_objects)), max(16, int(f2 * trace.n_objects))
    ev = int(0.2 * trace.n)
    if policy.startswith("SAGE"):
        logS = np.load(CACHE_DIR / f"{name}_full.npy")
        r = run_sage_tiered(trace, logS, c1, c2, ev, h_mult=SAGE_HMULT, unified=policy == "SAGE-Unified")
    elif policy == "S4-FIFO-cascade":
        k = tune_s4fifo_cascade(trace, c1, c2)
        r = run_cascade(trace, "S4-FIFO", c1, c2, ev, **S4FIFO_GRID[k])
        r["s4_config"] = k
    else:
        r = run_cascade(trace, policy.rsplit("-", 1)[0], c1, c2, ev)
    return {"trace": name, "dram_frac": f1, "ssd_frac": f2, "policy": policy, **r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="*", default=TEST_TRACES)
    ap.add_argument("--requests", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    jobs = [(t, args.requests, f1, f2, p) for t in args.traces for f1, f2 in SIZE_PAIRS for p in POLICIES]
    jobs.sort(key=lambda j: j[4] != "S4-FIFO-cascade")
    out = OUT_DIR / "tier_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            rows.append(r)
            print(f"{r['trace']:14s} {r['policy']:16s} lat={r['mean_latency_ms']:.3f}ms dram={r['dram_hit']:.3f} "
                  f"ssd={r['ssd_hit']:.3f} writes/kreq={r['ssd_writes_per_kreq']:.1f}", flush=True)
    with out.open("w", newline="") as fh:
        fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["trace"], r["dram_frac"], r["policy"])))


if __name__ == "__main__":
    main()
