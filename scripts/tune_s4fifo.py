"""Realistic S4-FIFO: pick its queue configuration using only the warm-up prefix (first 20%).

Each of the 40 configurations is simulated on requests [0, 20%) (miss ratio measured on [10%, 20%));
the winner per (trace, size) is the configuration S4-FIFO would deploy. No test-window data is used.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_cache import OUT_DIR, SIZES, TEST_TRACES  # noqa: E402
from src.sim.policies import S4FIFO, S4FIFO_GRID  # noqa: E402
from src.sim.traces import load_trace  # noqa: E402

_T = {}


def job(a):
    name, frac, k = a
    if name not in _T:
        _T.clear()
        _T[name] = load_trace(name, 2_000_000)
    tr = _T[name]
    cap = max(16, int(frac * tr.n_objects))
    end = int(0.2 * tr.n)
    start = int(0.1 * tr.n)
    p = S4FIFO(cap, **S4FIFO_GRID[k])
    keys = tr.keys[:end].tolist()
    hits = 0
    for i, key in enumerate(keys):
        h = p.access(key, i)
        if i >= start:
            hits += h
    return name, frac, k, 1 - hits / (end - start)


def main():
    jobs = [(t, f, k) for t in TEST_TRACES for f in SIZES for k in range(len(S4FIFO_GRID))]
    rows = []
    with mp.get_context("fork").Pool(7) as pool:
        rows = pool.map(job, jobs, chunksize=40)
    best = {}
    for name, frac, k, mr in rows:
        if (name, frac) not in best or mr < best[(name, frac)][1]:
            best[(name, frac)] = (k, mr)
    with (OUT_DIR / "s4fifo_prefix_choice.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trace", "size_frac", "variant", "prefix_miss_ratio"])
        for (name, frac), (k, mr) in sorted(best.items()):
            w.writerow([name, frac, k, mr])
    print(len(best), "choices written")


if __name__ == "__main__":
    main()
