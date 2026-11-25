"""SAGE: Survival-Aware, Group-context Eviction and admission.

Method
------
1. Causal, cache-independent features for every request (``features.py``):
   object history + spatial-region ("semantic group") context + global scan rate.
2. A discrete-time *hazard* model (LightGBM, binary objective, person-period
   expansion) predicts, at access time, the full distribution of the time until
   the object's next request over 12 log-spaced bins (16 ... 1.5M requests).
   Training labels are right-censored at the training cut-off, so no future
   information leaks into the model. The model is retrained periodically on
   the recent past and applied to the next window in a single batched call.
3. Eviction: K=64 random cached objects are sampled. For each one, the
   *conditional* reuse probability given that it has already been idle for e
   requests is  P(reuse in (e, e+H] | no reuse by e) = 1 - S(e+H)/S(e),
   where S is the predicted survival curve and H is the cache turnover time
   (C / insertion rate, scaled by ``h_mult``). The lowest-probability object is
   evicted. Conditioning on e makes mispredicted "overdue" objects age out.
4. Admission: on a miss, the incoming object's P(reuse within H) is compared
   with the victim's; if lower, the object bypasses the cache (no pollution by
   scans, no extra writes to the cache device).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
from numba import njit

LOG2_E0 = 4.0
LOG2_STEP = 1.5
N_BINS = 12
EDGES = 2.0 ** (LOG2_E0 + LOG2_STEP * np.arange(N_BINS))

HAZARD_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.08,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}


@dataclass
class SageModelReport:
    retrain_points: list[int] = field(default_factory=list)
    train_rows: list[int] = field(default_factory=list)
    train_seconds: float = 0.0
    predict_seconds: float = 0.0
    feature_importance: dict[str, float] = field(default_factory=dict)
    holdout: list[dict] = field(default_factory=list)


def hazard_rows(next_dist: np.ndarray, censor: np.ndarray):
    """Person-period expansion. Returns (sample_idx, bin_idx, label, weight)."""
    idx, bins, labels, weights = [], [], [], []
    lo = 0.0
    for k in range(N_BINS):
        hi = EDGES[k]
        at_risk = (next_dist > lo) & (censor > lo)
        event = at_risk & (next_dist <= hi) & (next_dist <= censor)
        survived = at_risk & ~event & (censor >= hi)
        partial = at_risk & ~event & ~survived
        for mask, lab, w in ((event, 1, None), (survived, 0, None), (partial, 0, (censor - lo) / (hi - lo))):
            sel = np.nonzero(mask)[0]
            idx.append(sel)
            bins.append(np.full(len(sel), k, dtype=np.float32))
            labels.append(np.full(len(sel), lab, dtype=np.float32))
            weights.append(np.ones(len(sel), dtype=np.float32) if w is None else w[sel].astype(np.float32))
        lo = hi
    return np.concatenate(idx), np.concatenate(bins), np.concatenate(labels), np.concatenate(weights)


def kaplan_meier_logS(next_dist: np.ndarray, censor: np.ndarray) -> np.ndarray:
    _, bins, labels, weights = hazard_rows(next_dist, censor)
    logS = np.zeros(N_BINS)
    acc = 0.0
    for k in range(N_BINS):
        m = bins == k
        h = float((labels[m] * weights[m]).sum() / max(weights[m].sum(), 1e-9)) if m.any() else 0.5
        acc += np.log(np.clip(1 - h, 1e-6, 1))
        logS[k] = acc
    return logS


def _expand_predict(model: lgb.Booster, F: np.ndarray, num_threads: int, chunk: int = 50_000) -> np.ndarray:
    n = F.shape[0]
    out = np.empty((n, N_BINS), dtype=np.float32)
    for s in range(0, n, chunk):
        block = F[s:s + chunk]
        m = block.shape[0]
        X = np.empty((m * N_BINS, block.shape[1] + 1), dtype=np.float32)
        X[:, :-1] = np.repeat(block, N_BINS, axis=0)
        X[:, -1] = np.tile(np.arange(N_BINS, dtype=np.float32), m)
        h = model.predict(X, num_threads=num_threads).reshape(m, N_BINS)
        out[s:s + m] = np.cumsum(np.log(np.clip(1.0 - h, 1e-6, 1.0)), axis=1)
    return out


def predict_survival(
    F: np.ndarray,
    next_index: np.ndarray,
    feature_cols: list[int] | None = None,
    n_retrain: int = 9,
    train_window: int = 1_000_000,
    train_samples: int = 120_000,
    num_rounds: int = 150,
    num_threads: int = 2,
    seed: int = 42,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, SageModelReport]:
    """Return per-request log-survival at EDGES, using only models trained on the past."""
    n = F.shape[0]
    if feature_cols is not None:
        F = F[:, feature_cols]
        if feature_names is not None:
            feature_names = [feature_names[c] for c in feature_cols]
    rng = np.random.default_rng(seed)
    req = np.arange(n, dtype=np.int64)
    next_dist = (next_index - req).astype(np.float64)
    next_dist[next_index >= n] = np.inf
    logS = np.empty((n, N_BINS), dtype=np.float32)
    report = SageModelReport()
    points = [int(n * (k + 1) / (n_retrain + 1)) for k in range(n_retrain)]
    # warm-up prior before the first model: Kaplan-Meier marginal from the warm-up prefix
    T0 = points[0]
    logS[:T0] = kaplan_meier_logS(next_dist[:T0], (T0 - req[:T0]).astype(np.float64))[None, :]
    gain = np.zeros(F.shape[1] + 1)
    for j, T in enumerate(points):
        end = points[j + 1] if j + 1 < len(points) else n
        lo = max(0, T - train_window)
        sample = np.sort(rng.choice(np.arange(lo, T), size=min(train_samples, T - lo), replace=False))
        censor = (T - sample).astype(np.float64)
        si, bins, labels, weights = hazard_rows(next_dist[sample], censor)
        X = np.empty((len(si), F.shape[1] + 1), dtype=np.float32)
        X[:, :-1] = F[sample[si]]
        X[:, -1] = bins
        t0 = time.time()
        params = dict(HAZARD_PARAMS, num_threads=num_threads)
        model = lgb.train(params, lgb.Dataset(X, labels, weight=weights, free_raw_data=True), num_boost_round=num_rounds)
        report.train_seconds += time.time() - t0
        report.retrain_points.append(T)
        report.train_rows.append(len(si))
        gain += model.feature_importance("gain")
        t0 = time.time()
        logS[T:end] = _expand_predict(model, F[T:end], num_threads)
        report.predict_seconds += time.time() - t0
    names = (feature_names or [f"f{c}" for c in range(F.shape[1])]) + ["horizon_bin"]
    total = gain.sum() or 1.0
    report.feature_importance = {nm: float(g / total) for nm, g in sorted(zip(names, gain), key=lambda x: -x[1])}
    return logS, report


# ---------------------------------------------------------------- simulation
@njit(cache=True, inline="always")
def _logS_at(logS, row, t):
    if t <= 0.0:
        return 0.0
    e0 = 2.0 ** LOG2_E0
    if t <= e0:
        return logS[row, 0] * t / e0
    x = (np.log2(t) - LOG2_E0) / LOG2_STEP
    k = int(x)
    if k >= N_BINS - 1:
        slope = logS[row, N_BINS - 1] - logS[row, N_BINS - 2]
        return logS[row, N_BINS - 1] + slope * (x - (N_BINS - 1))
    f = x - k
    return logS[row, k] + (logS[row, k + 1] - logS[row, k]) * f


@njit(cache=True, inline="always")
def _mrl(logS, row, e, horizon):
    """Truncated mean residual life: E[min(D - e, horizon) | D > e], integrated on a log grid."""
    base = _logS_at(logS, row, e)
    total = 0.0
    prev_t = 0.0
    prev_s = 1.0
    t = 8.0
    while prev_t < horizon:
        if t > horizon:
            t = horizon
        cur = np.exp(_logS_at(logS, row, e + t) - base)
        total += (t - prev_t) * 0.5 * (cur + prev_s)
        prev_t = t
        prev_s = cur
        t *= 2.0
    return total


@njit(cache=True)
def _ghost_add(ring, ring_t, head, owner, gtime, expert, o, i):
    cap = ring.shape[0]
    h = head[expert]
    old = ring[expert, h]
    if old >= 0 and owner[old] == expert + 1 and gtime[old] == ring_t[expert, h]:
        owner[old] = 0
    ring[expert, h] = o
    ring_t[expert, h] = i
    owner[o] = expert + 1
    gtime[o] = i
    head[expert] = (h + 1) % cap


@njit(cache=True)
def simulate_sage(keys, logS, n_objects, capacity, eval_start, h_mult, admission, conditional, sample_k, seed,
                  hedge=False, learning_rate=0.45, mrl_mode=False, mrl_mult=8.0, flags=None):
    """SAGE eviction/admission. With ``hedge`` the victim is chosen by a LeCaR-style regret-weighted
    choice between the SAGE expert and a sampled-LFU expert (robustness to model error)."""
    np.random.seed(seed)
    n = keys.shape[0]
    slots = np.empty(capacity, dtype=np.int64)
    pos = np.full(n_objects, -1, dtype=np.int64)
    last_t = np.zeros(n_objects, dtype=np.int64)
    row = np.zeros(n_objects, dtype=np.int64)
    cnt = np.zeros(n_objects, dtype=np.int64)
    owner = np.zeros(n_objects, dtype=np.int8)
    gtime = np.zeros(n_objects, dtype=np.int64)
    ring = np.full((2, capacity), -1, dtype=np.int64)
    ring_t = np.zeros((2, capacity), dtype=np.int64)
    head = np.zeros(2, dtype=np.int64)
    discount = 0.005 ** (1.0 / capacity)
    w_sage = 0.5
    used = 0
    hits = 0
    ins_rate = 1.0
    inserts = 0
    bypasses = 0
    for i in range(n):
        o = keys[i]
        cnt[o] += 1
        if pos[o] >= 0:
            if i >= eval_start:
                hits += 1
            if flags is not None:
                flags[i] = 1
            last_t[o] = i
            row[o] = i
            ins_rate *= 0.999
            continue
        if hedge and owner[o] > 0:
            r = discount ** (i - gtime[o])
            ws = w_sage
            wl = 1.0 - w_sage
            if owner[o] == 1:
                ws *= np.exp(-learning_rate * r)
            else:
                wl *= np.exp(-learning_rate * r)
            w_sage = ws / (ws + wl)
            if w_sage > 0.99:
                w_sage = 0.99
            elif w_sage < 0.01:
                w_sage = 0.01
            owner[o] = 0
        H = h_mult * capacity / max(ins_rate, 1e-3)
        if used < capacity:
            pos[o] = used
            slots[used] = o
            used += 1
            last_t[o] = i
            row[o] = i
            ins_rate = ins_rate * 0.999 + 0.001
            inserts += 1
            continue
        # sampled candidates: SAGE expert (min conditional reuse prob) and LFU expert (min count, oldest)
        best = -1
        best_score = 1e18
        best_age = -1
        lfu = -1
        lfu_cnt = 1 << 62
        lfu_age = -1
        m = sample_k if sample_k < capacity else capacity
        for s in range(m):
            if sample_k < capacity:
                c = slots[np.random.randint(0, capacity)]
            else:
                c = slots[s]
            e = float(i - last_t[c]) if conditional else 0.0
            if mrl_mode:
                score = -_mrl(logS, row[c], e, mrl_mult * H)
            else:
                a = _logS_at(logS, row[c], e)
                b = _logS_at(logS, row[c], e + H)
                score = 1.0 - np.exp(b - a)
            age = i - last_t[c]
            if score < best_score or (score == best_score and age > best_age):
                best_score = score
                best = c
                best_age = age
            if cnt[c] < lfu_cnt or (cnt[c] == lfu_cnt and age > lfu_age):
                lfu_cnt = cnt[c]
                lfu = c
                lfu_age = age
        use_sage = True
        if hedge:
            use_sage = np.random.random() < w_sage
        victim = best if use_sage else lfu
        if admission and use_sage:
            if mrl_mode:
                new_score = -_mrl(logS, i, 0.0, mrl_mult * H)
            else:
                new_score = 1.0 - np.exp(_logS_at(logS, i, H))
            if new_score < best_score:
                bypasses += 1
                ins_rate *= 0.999
                if hedge:
                    _ghost_add(ring, ring_t, head, owner, gtime, 0, o, i)
                continue
        if hedge and best != lfu:
            _ghost_add(ring, ring_t, head, owner, gtime, 0 if use_sage else 1, victim, i)
        p = pos[victim]
        pos[victim] = -1
        slots[p] = o
        pos[o] = p
        last_t[o] = i
        row[o] = i
        ins_rate = ins_rate * 0.999 + 0.001
        inserts += 1
    return hits, inserts, bypasses
