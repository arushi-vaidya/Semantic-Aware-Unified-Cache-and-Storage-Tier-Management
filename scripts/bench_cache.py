"""Trace-driven benchmark: SAGE vs. ARC, LeCaR, LRB (+ LRU, S3-FIFO, Belady bound).

Stage 1 (``--stage model``) trains SAGE's hazard models for each trace once
(models are independent of cache size) and caches per-request survival curves.
Stage 2 (``--stage sim``) runs every (trace, cache size, policy) simulation in a
process pool and appends rows to ``results/benchmark/raw_results.csv``.

Protocol: first N requests of each trace; first 20% is warm-up for every policy;
miss ratio is measured on the remaining 80%. Cache sizes are fractions of the
distinct objects in the N-request prefix. SAGE models are only ever trained on
requests before the window they are applied to.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sim.features import FEATURE_NAMES, OBJECT_ONLY, build_features  # noqa: E402
from src.sim.policies import S4FIFO_GRID  # noqa: E402
from src.sim.runner import run_python_policy, run_sage  # noqa: E402
from src.sim.sage import predict_survival  # noqa: E402
from src.sim.traces import load_trace  # noqa: E402

DEV_TRACES = ["msr_proj_0", "fiu_webresearch"]
TEST_TRACES = [
    "msr_hm_0", "msr_prn_0", "msr_prxy_0", "msr_usr_2", "msr_web_2",
    "fiu_casa", "fiu_ikki", "fiu_online", "fiu_webmail", "fiu_webusers", "w31",
]
SIZES = [0.005, 0.01, 0.05, 0.10]
LRB_WINDOWS = [4, 16, 64]  # memory window as a multiple of cache size; best-in-hindsight is reported
CACHE_DIR = ROOT / "data" / "sage_cache"
OUT_DIR = ROOT / "results" / "benchmark"
FIELDS = ["trace", "size_frac", "capacity", "policy", "variant", "miss_ratio", "hits", "requests",
          "seconds", "inserts", "bypasses"]

# SAGE variants: name -> (feature set, simulate_sage kwargs). Hyper-parameters fixed on DEV_TRACES.
SAGE_HMULT = 0.5
SAGE_VARIANTS = {
    "SAGE": ("full", dict(admission=True, conditional=True)),
    "SAGE-noAdmit": ("full", dict(admission=False, conditional=True)),
    "SAGE-noCond": ("full", dict(admission=True, conditional=False)),
    "SAGE-objOnly": ("obj", dict(admission=True, conditional=True)),
    # post-hoc robustness variant: LeCaR-style regret hedge between the SAGE expert and sampled LFU
    "SAGE-H": ("full", dict(admission=True, conditional=True, hedge=True)),
}

_TRACES: dict = {}


def get_trace(name: str, n: int):
    if name not in _TRACES:
        _TRACES.clear()
        _TRACES[name] = load_trace(name, n)
    return _TRACES[name]


def model_stage(name: str, n: int, threads: int, tags=("full", "obj")) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    trace = load_trace(name, n)
    feats = build_features(trace)
    for tag, cols in (("full", None), ("obj", OBJECT_ONLY)):
        if tag not in tags:
            continue
        path = CACHE_DIR / f"{name}_{tag}.npy"
        if path.exists():
            continue
        t0 = time.time()
        logS, report = predict_survival(feats, trace.next_index, feature_cols=cols, num_threads=threads,
                                        feature_names=list(FEATURE_NAMES))
        np.save(path, logS)
        meta = {
            "trace": name, "features": tag, "requests": trace.n, "objects": trace.n_objects,
            "regions": trace.n_regions, "retrain_points": report.retrain_points, "train_rows": report.train_rows,
            "train_seconds": report.train_seconds, "predict_seconds": report.predict_seconds,
            "wall_seconds": time.time() - t0, "feature_importance": report.feature_importance,
        }
        (CACHE_DIR / f"{name}_{tag}.json").write_text(json.dumps(meta, indent=2))
        print(f"[model] {name} {tag} done in {time.time() - t0:.0f}s", flush=True)


def sim_job(job: tuple) -> dict:
    name, n, frac, policy, variant = job
    trace = get_trace(name, n)
    capacity = max(16, int(frac * trace.n_objects))
    eval_start = int(0.2 * trace.n)
    if policy.startswith("SAGE"):
        tag, kwargs = SAGE_VARIANTS[policy]
        h_mult = float(variant) if variant else SAGE_HMULT
        logS = np.load(CACHE_DIR / f"{name}_{tag}.npy", mmap_mode="r")
        result = run_sage(trace, np.ascontiguousarray(logS), capacity, eval_start, h_mult=h_mult, **kwargs)
    elif policy == "S4-FIFO":
        result = run_python_policy(trace, "S4-FIFO", capacity, eval_start, **S4FIFO_GRID[int(variant)])
    elif policy == "LRB":
        result = run_python_policy(trace, "LRB", capacity, eval_start, memory_window=int(variant) * capacity)
    else:
        result = run_python_policy(trace, policy, capacity, eval_start)
    return {"trace": name, "size_frac": frac, "capacity": capacity, "policy": policy, "variant": variant,
            **{k: result.get(k, "") for k in ("miss_ratio", "hits", "requests", "seconds", "inserts", "bypasses")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["model", "sim"], required=True)
    ap.add_argument("--traces", nargs="*", default=TEST_TRACES)
    ap.add_argument("--policies", nargs="*", default=["LRU", "ARC", "LeCaR", "S3-FIFO", "Belady", "LRB", *SAGE_VARIANTS])
    ap.add_argument("--sizes", nargs="*", type=float, default=SIZES)
    ap.add_argument("--requests", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tags", nargs="*", default=["full", "obj"])
    ap.add_argument("--hmult", nargs="*", default=[""], help="SAGE h_mult values to sweep (dev tuning)")
    ap.add_argument("--out", default=str(OUT_DIR / "raw_results.csv"))
    args = ap.parse_args()

    if args.stage == "model":
        for name in args.traces:
            model_stage(name, args.requests, args.threads, tuple(args.tags))
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {(r["trace"], float(r["size_frac"]), r["policy"], r["variant"]) for r in csv.DictReader(fh)}
    jobs = []
    for name in args.traces:
        for frac in args.sizes:
            for policy in args.policies:
                if policy == "LRB":
                    variants = [str(w) for w in LRB_WINDOWS]
                elif policy == "S4-FIFO":
                    variants = [str(k) for k in range(len(S4FIFO_GRID))]
                else:
                    variants = args.hmult if policy.startswith("SAGE") else [""]
                for v in variants:
                    if (name, frac, policy, v) not in done:
                        jobs.append((name, args.requests, frac, policy, v))
    jobs.sort(key=lambda j: (j[3] not in ("LRB", "S4-FIFO"), j[0]))  # slowest first, grouped by trace
    print(f"{len(jobs)} simulations", flush=True)
    new_file = not out.exists()
    with out.open("a", newline="") as fh, mp.get_context("fork").Pool(args.workers, maxtasksperchild=8) as pool:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for row in pool.imap_unordered(sim_job, jobs):
            writer.writerow(row)
            fh.flush()
            print(f"{row['trace']:16s} {row['size_frac']:<6} {row['policy']:13s} {row['variant']:>4} "
                  f"miss={row['miss_ratio']:.4f} ({row['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
