"""Two-level exclusive storage hierarchy: DRAM cache (L1) -> SSD tier (L2) -> HDD backing store.

Baselines (classic tiering): an L1 replacement policy (LRU or ARC) whose victims
are always demoted into an LRU-managed SSD tier; SSD hits are promoted to DRAM.
This is the cascade used in Phase 1 of this project.

SAGE-Unified: one predicted survival curve per request drives *every* placement
decision. An object's value for tier T is P(reuse within H_T | idle e), where
H_T is tier T's turnover time. DRAM victims are demoted only if they beat the
SSD victim; misses that are not worth DRAM can still be placed directly on SSD;
everything else bypasses to HDD. Fewer, better-chosen SSD writes.
"""
from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
from numba import njit

from src.sim.policies import ARC, LRU, S4FIFO, S4FIFO_GRID, SRS3FIFO
from src.sim.sage import _logS_at

LAT_DRAM_MS = 0.0001
LAT_SSD_MS = 0.1
LAT_HDD_MS = 5.0


def _summary(h1, h2, total, ssd_writes, seconds):
    miss = total - h1 - h2
    return {
        "dram_hit": h1 / total, "ssd_hit": h2 / total, "hdd_miss": miss / total,
        "mean_latency_ms": (h1 * LAT_DRAM_MS + h2 * LAT_SSD_MS + miss * LAT_HDD_MS) / total,
        "ssd_writes": int(ssd_writes), "ssd_writes_per_kreq": 1000 * ssd_writes / total, "seconds": seconds,
    }


def run_cascade(trace, l1_policy: str, c1: int, c2: int, eval_start: int, end: int | None = None,
                flags=None, **policy_kwargs) -> dict:
    """Classic cascade: L1 policy in DRAM, every L1 victim demoted into an LRU SSD tier."""
    l2: OrderedDict[int, None] = OrderedDict()
    state = {"writes": 0, "counting": False}

    def demote(victim: int) -> None:
        if len(l2) >= c2:
            l2.popitem(last=False)
        l2[victim] = None
        if state["counting"]:
            state["writes"] += 1

    cls = {"LRU": LRU, "ARC": ARC, "S4-FIFO": S4FIFO, "SR-S3-FIFO": SRS3FIFO}[l1_policy]
    l1 = cls(c1, **policy_kwargs)
    l1.on_evict = demote
    if l1_policy == "LRU":
        in_l1 = lambda k: k in l1.d  # noqa: E731
    elif l1_policy == "ARC":
        in_l1 = lambda k: k in l1.t1 or k in l1.t2  # noqa: E731
    else:
        in_l1 = lambda k: k in l1.freq  # noqa: E731
    keys = trace.keys[:end].tolist() if end else trace.keys.tolist()
    h1 = h2 = 0
    started = time.time()
    for i, k in enumerate(keys):
        if i == eval_start:
            state["counting"] = True
        if in_l1(k):
            l1.access(k, i)
            h1 += i >= eval_start
            if flags is not None:
                flags[i] = 1
        elif k in l2:
            del l2[k]
            l1.access(k, i)
            h2 += i >= eval_start
            if flags is not None:
                flags[i] = 2
        else:
            l1.access(k, i)
    return _summary(h1, h2, len(keys) - eval_start, state["writes"], time.time() - started)


def tune_s4fifo_cascade(trace, c1: int, c2: int) -> int:
    """Pick the S4-FIFO configuration with the lowest mean latency on the warm-up prefix only."""
    lo, hi = int(0.1 * trace.n), int(0.2 * trace.n)
    scores = [run_cascade(trace, "S4-FIFO", c1, c2, lo, end=hi, **cfg)["mean_latency_ms"] for cfg in S4FIFO_GRID]
    return int(np.argmin(scores))


@njit(cache=True)
def _victim(slots, used, last_t, row, logS, i, H, k):
    best = -1
    best_score = 1e18
    best_age = -1
    m = k if k < used else used
    for s in range(m):
        c = slots[np.random.randint(0, used)] if k < used else slots[s]
        e = float(i - last_t[c])
        score = 1.0 - np.exp(_logS_at(logS, row[c], e + H) - _logS_at(logS, row[c], e))
        age = i - last_t[c]
        if score < best_score or (score == best_score and age > best_age):
            best, best_score, best_age = c, score, age
    return best, best_score


@njit(cache=True)
def _remove(slots, pos, used, o):
    p = pos[o]
    last = slots[used - 1]
    slots[p] = last
    pos[last] = p
    pos[o] = -1
    return used - 1


@njit(cache=True)
def simulate_sage_tiered(keys, logS, n_objects, c1, c2, eval_start, h_mult, unified, sample_k, seed, flags=None):
    np.random.seed(seed)
    n = keys.shape[0]
    tier = np.zeros(n_objects, dtype=np.int8)
    pos = np.full(n_objects, -1, dtype=np.int64)
    s1 = np.empty(c1, dtype=np.int64)
    s2 = np.empty(c2, dtype=np.int64)
    u1 = 0
    u2 = 0
    last_t = np.zeros(n_objects, dtype=np.int64)
    row = np.zeros(n_objects, dtype=np.int64)
    r1 = 1.0
    r2 = 1.0
    h1 = 0
    h2 = 0
    writes = 0
    for i in range(n):
        o = keys[i]
        counting = i >= eval_start
        H1 = h_mult * c1 / max(r1, 1e-3)
        H2 = h_mult * c2 / max(r2, 1e-3)
        t = tier[o]
        last_t_prev = last_t[o]
        last_t[o] = i
        row[o] = i
        if t == 1:
            if counting:
                h1 += 1
            if flags is not None:
                flags[i] = 1
            r1 *= 0.999
            r2 *= 0.999
            continue
        if t == 2:
            if counting:
                h2 += 1
            if flags is not None:
                flags[i] = 2
            u2 = _remove(s2, pos, u2, o)
            tier[o] = 0
        # o needs a home: try DRAM
        placed = False
        v = -1
        if u1 < c1:
            pos[o] = u1
            s1[u1] = o
            u1 += 1
            tier[o] = 1
            placed = True
        else:
            v, vs = _victim(s1, u1, last_t, row, logS, i, H1, sample_k)
            new_score = 1.0 - np.exp(_logS_at(logS, i, H1))
            if (not unified) or t == 2 or new_score >= vs:
                p = pos[v]
                pos[v] = -1
                tier[v] = 0
                s1[p] = o
                pos[o] = p
                tier[o] = 1
                placed = True
            else:
                v = o  # o itself is the DRAM reject; consider it for SSD
        r1 = r1 * 0.999 + (0.001 if placed else 0.0)
        # candidate for SSD: the DRAM victim (or o if it was rejected)
        wrote = False
        if v >= 0:
            if u2 < c2:
                pos[v] = u2
                s2[u2] = v
                u2 += 1
                tier[v] = 2
                wrote = True
            else:
                w, ws = _victim(s2, u2, last_t, row, logS, i, H2, sample_k)
                e = float(i - last_t[v])
                vs2 = 1.0 - np.exp(_logS_at(logS, row[v], e + H2) - _logS_at(logS, row[v], e))
                if (not unified) or vs2 >= ws:
                    if not unified:
                        # LRU SSD tier for the non-unified ablation: evict the oldest sampled object
                        w = -1
                        oldest = -1
                        for s in range(sample_k if sample_k < u2 else u2):
                            c = s2[np.random.randint(0, u2)]
                            if i - last_t[c] > oldest:
                                oldest = i - last_t[c]
                                w = c
                    p = pos[w]
                    pos[w] = -1
                    tier[w] = 0
                    s2[p] = v
                    pos[v] = p
                    tier[v] = 2
                    wrote = True
        if wrote and counting:
            writes += 1
        r2 = r2 * 0.999 + (0.001 if wrote else 0.0)
    total = n - eval_start
    return h1, h2, total, writes


def run_sage_tiered(trace, logS, c1, c2, eval_start, h_mult=0.5, unified=True, sample_k=64, seed=42, flags=None) -> dict:
    started = time.time()
    h1, h2, total, writes = simulate_sage_tiered(trace.keys, logS, trace.n_objects, c1, c2, eval_start,
                                                 h_mult, unified, sample_k, seed, flags)
    return _summary(h1, h2, total, writes, time.time() - started)
