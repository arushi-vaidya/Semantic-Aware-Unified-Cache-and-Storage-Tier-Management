"""Cache-state-independent access features for every request of a trace.

All features at request ``i`` are computed from requests ``< i`` plus the
identity of request ``i`` itself, so they are causal and do not depend on what a
particular cache happens to hold. That property lets SAGE score a whole window of
requests in one batched model call instead of one call per eviction.

Feature groups
--------------
* object  : last 8 inter-arrival gaps, prior access count, 6 exponentially
            decayed counters (EDCs), request size.
* region  : the object's 1 MiB extent ("semantic/spatial context"): gaps, count,
            EDCs, fraction of first-touch blocks (scan indicator), footprint.
* global  : trace-wide first-touch rate (scan phase), spatial jump from the
            previous request.
"""
from __future__ import annotations

import numpy as np
from numba import njit

N_GAPS = 8
OBJ_HALF_LIVES = (2.0**6, 2.0**8, 2.0**10, 2.0**12, 2.0**14, 2.0**16)
REG_HALF_LIVES = (2.0**8, 2.0**12, 2.0**16)
NEW_HALF_LIVES = (2.0**10, 2.0**14)
GLOBAL_HALF_LIVES = (2.0**12, 2.0**16)

FEATURE_NAMES = (
    [f"obj_gap{k}" for k in range(N_GAPS)]
    + ["obj_count"]
    + [f"obj_edc_{int(np.log2(h))}" for h in OBJ_HALF_LIVES]
    + ["log_size"]
    + ["reg_gap0", "reg_gap1", "reg_count"]
    + [f"reg_edc_{int(np.log2(h))}" for h in REG_HALF_LIVES]
    + [f"reg_new_frac_{int(np.log2(h))}" for h in NEW_HALF_LIVES]
    + ["reg_footprint", "is_first"]
    + [f"glob_new_rate_{int(np.log2(h))}" for h in GLOBAL_HALF_LIVES]
    + ["same_region_as_prev", "log_jump"]
)
OBJECT_ONLY = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("obj_") or name in ("log_size", "is_first")]


@njit(cache=True)
def _build(keys, regions, raw_ids, sizes, n_objects, n_regions, obj_hl, reg_hl, new_hl, glob_hl, out):
    n = keys.shape[0]
    nan = np.nan
    obj_prev = np.full((n_objects, 8), -1, dtype=np.int64)
    obj_count = np.zeros(n_objects, dtype=np.int64)
    obj_edc = np.zeros((n_objects, obj_hl.shape[0]), dtype=np.float64)
    reg_prev = np.full((n_regions, 2), -1, dtype=np.int64)
    reg_count = np.zeros(n_regions, dtype=np.int64)
    reg_edc = np.zeros((n_regions, reg_hl.shape[0]), dtype=np.float64)
    reg_new = np.zeros((n_regions, new_hl.shape[0]), dtype=np.float64)
    reg_all = np.zeros((n_regions, new_hl.shape[0]), dtype=np.float64)
    reg_foot = np.zeros(n_regions, dtype=np.int64)
    g_new = np.zeros(glob_hl.shape[0], dtype=np.float64)
    g_all = np.zeros(glob_hl.shape[0], dtype=np.float64)
    n_obj_hl = obj_hl.shape[0]
    for i in range(n):
        o = keys[i]
        r = regions[i]
        c = 0
        # ---- object gaps: gap0 = i - prev1, gapk = prev_k - prev_{k+1}
        p1 = obj_prev[o, 0]
        if p1 >= 0:
            out[i, 0] = np.log2(1.0 + i - p1)
        else:
            out[i, 0] = nan
        for k in range(1, 8):
            a = obj_prev[o, k - 1]
            b = obj_prev[o, k]
            if b >= 0:
                out[i, k] = np.log2(1.0 + a - b)
            else:
                out[i, k] = nan
        c = 8
        out[i, c] = np.log2(1.0 + obj_count[o]); c += 1
        first = obj_count[o] == 0
        # ---- object EDCs (decayed to i, including the current access)
        for h in range(n_obj_hl):
            if p1 >= 0:
                v = 1.0 + obj_edc[o, h] * 2.0 ** (-(i - p1) / obj_hl[h])
            else:
                v = 1.0
            obj_edc[o, h] = v
            out[i, c] = v; c += 1
        out[i, c] = np.log2(1.0 + sizes[i]); c += 1
        # ---- region features
        q1 = reg_prev[r, 0]
        q2 = reg_prev[r, 1]
        out[i, c] = np.log2(1.0 + i - q1) if q1 >= 0 else nan; c += 1
        out[i, c] = np.log2(1.0 + q1 - q2) if q2 >= 0 else nan; c += 1
        out[i, c] = np.log2(1.0 + reg_count[r]); c += 1
        for h in range(reg_hl.shape[0]):
            if q1 >= 0:
                v = 1.0 + reg_edc[r, h] * 2.0 ** (-(i - q1) / reg_hl[h])
            else:
                v = 1.0
            reg_edc[r, h] = v
            out[i, c] = v; c += 1
        for h in range(new_hl.shape[0]):
            decay = 2.0 ** (-(i - q1) / new_hl[h]) if q1 >= 0 else 0.0
            reg_all[r, h] = reg_all[r, h] * decay
            reg_new[r, h] = reg_new[r, h] * decay
            # fraction of first-touch blocks among *prior* region accesses
            out[i, c] = reg_new[r, h] / reg_all[r, h] if reg_all[r, h] > 0 else nan; c += 1
            reg_all[r, h] += 1.0
            if first:
                reg_new[r, h] += 1.0
        if first:
            reg_foot[r] += 1
        out[i, c] = np.log2(1.0 + reg_foot[r]); c += 1
        out[i, c] = 1.0 if first else 0.0; c += 1
        # ---- global first-touch rate over prior requests
        for h in range(glob_hl.shape[0]):
            out[i, c] = g_new[h] / g_all[h] if g_all[h] > 0 else nan; c += 1
            d = 2.0 ** (-1.0 / glob_hl[h])
            g_all[h] = g_all[h] * d + 1.0
            g_new[h] = g_new[h] * d + (1.0 if first else 0.0)
        if i > 0:
            out[i, c] = 1.0 if regions[i - 1] == r else 0.0; c += 1
            a = raw_ids[i]
            b = raw_ids[i - 1]
            jump = a - b if a >= b else b - a
            out[i, c] = np.log2(1.0 + jump); c += 1
        else:
            out[i, c] = nan; c += 1
            out[i, c] = nan; c += 1
        # ---- update state
        for k in range(7, 0, -1):
            obj_prev[o, k] = obj_prev[o, k - 1]
        obj_prev[o, 0] = i
        obj_count[o] += 1
        reg_prev[r, 1] = q1
        reg_prev[r, 0] = i
        reg_count[r] += 1


def build_features(trace) -> np.ndarray:
    out = np.empty((trace.n, len(FEATURE_NAMES)), dtype=np.float32)
    _build(
        trace.keys,
        trace.regions,
        trace.raw_ids.astype(np.float64),
        trace.sizes,
        trace.n_objects,
        trace.n_regions,
        np.array(OBJ_HALF_LIVES),
        np.array(REG_HALF_LIVES),
        np.array(NEW_HALF_LIVES),
        np.array(GLOBAL_HALF_LIVES),
        out,
    )
    return out
