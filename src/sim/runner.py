"""Glue for running one (trace, policy, cache size) simulation."""
from __future__ import annotations

import time

import numpy as np

from src.sim.lrb import LRB
from src.sim.policies import BASELINES
from src.sim.sage import simulate_sage


def run_python_policy(trace, policy_name: str, capacity: int, eval_start: int, flags=None, **kwargs) -> dict:
    if policy_name == "LRB":
        policy = LRB(capacity, n_objects=trace.n_objects, **kwargs)
    else:
        policy = BASELINES[policy_name](capacity, next_index=trace.next_index, **kwargs)
    access = policy.access
    keys = trace.keys.tolist()
    started = time.time()
    hits = 0
    if flags is not None:
        for i, k in enumerate(keys):
            if access(k, i):
                flags[i] = 1
        hits = int(flags[eval_start:].sum())
    else:
        for i in range(eval_start):
            access(keys[i], i)
        for i in range(eval_start, len(keys)):
            hits += access(keys[i], i)
    total = len(keys) - eval_start
    return {"hits": int(hits), "requests": total, "miss_ratio": 1 - hits / total, "seconds": time.time() - started}


def run_sage(trace, logS: np.ndarray, capacity: int, eval_start: int, h_mult: float = 1.0,
             admission: bool = True, conditional: bool = True, sample_k: int = 64, seed: int = 42, hedge: bool = False,
             mrl_mode: bool = False, mrl_mult: float = 8.0, flags=None) -> dict:
    started = time.time()
    hits, inserts, bypasses = simulate_sage(
        trace.keys, logS, trace.n_objects, capacity, eval_start, h_mult, admission, conditional, sample_k, seed, hedge, 0.45, mrl_mode, mrl_mult, flags
    )
    total = trace.n - eval_start
    return {
        "hits": int(hits), "requests": total, "miss_ratio": 1 - hits / total, "seconds": time.time() - started,
        "inserts": int(inserts), "bypasses": int(bypasses),
    }
