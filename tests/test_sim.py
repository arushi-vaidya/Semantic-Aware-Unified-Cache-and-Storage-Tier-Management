import numpy as np
import pytest

from src.sim.features import FEATURE_NAMES, build_features
from src.sim.policies import ARC, LRU, Belady, LeCaR, S3FIFO
from src.sim.sage import EDGES, N_BINS, hazard_rows, predict_survival, simulate_sage
from src.sim.tiered import run_cascade, run_sage_tiered
from src.sim.traces import Trace, compute_next_index


def make_trace(keys, seed=0):
    keys = np.asarray(keys, dtype=np.int32)
    n = len(keys)
    return Trace(
        name="msr_synthetic", keys=keys, raw_ids=(keys.astype(np.uint64) * np.uint64(4096)),
        regions=(keys // 256).astype(np.int32), sizes=np.full(n, 4096, dtype=np.uint32),
        timestamps=np.arange(n, dtype=np.uint32), next_index=compute_next_index(keys),
        n_objects=int(keys.max()) + 1, n_regions=int(keys.max()) // 256 + 1,
    )


def zipf_trace(n=30000, objects=3000, seed=1):
    rng = np.random.default_rng(seed)
    hot = rng.zipf(1.2, size=n) % objects
    scan = np.arange(n) % objects + objects
    mix = np.where(rng.random(n) < 0.7, hot, scan)
    return make_trace(mix)


def misses(policy, keys):
    return sum(not policy.access(int(k), i) for i, k in enumerate(keys))


def test_next_index():
    nxt = compute_next_index(np.array([1, 2, 1, 3, 2]))
    assert nxt.tolist() == [2, 4, 5, 5, 5]


def test_lru_basic():
    assert misses(LRU(2), [1, 2, 1, 3, 2]) == 4  # 2 evicted by 3


@pytest.mark.parametrize("cls", [LRU, ARC, LeCaR, S3FIFO])
def test_belady_is_lower_bound(cls):
    tr = zipf_trace()
    keys = tr.keys
    opt = misses(Belady(200, next_index=tr.next_index), keys)
    assert opt <= misses(cls(200, next_index=tr.next_index), keys)


@pytest.mark.parametrize("cls", [ARC, LeCaR, S3FIFO])
def test_capacity_respected(cls):
    p = cls(50)
    for i, k in enumerate(zipf_trace(5000).keys):
        p.access(int(k), i)
    size = {"ARC": lambda: len(p.t1) + len(p.t2), "LeCaR": lambda: len(p.freq),
            "S3-FIFO": lambda: len(p.small) + len(p.main)}[p.name]()
    assert size <= 50


def test_features_are_causal():
    tr = zipf_trace(4000)
    f1 = build_features(tr)
    keys = tr.keys.copy()
    keys[3000:] = (keys[3000:] + 17) % tr.n_objects  # change the future only
    f2 = build_features(make_trace(keys))
    assert f1.shape[1] == len(FEATURE_NAMES)
    np.testing.assert_array_equal(np.nan_to_num(f1[:3000], nan=-1), np.nan_to_num(f2[:3000], nan=-1))


def test_hazard_rows_censoring():
    # reused after 10 requests (event in bin 0); never reused but censored at 100
    idx, bins, labels, weights = hazard_rows(np.array([10.0, np.inf]), np.array([1e9, 100.0]))
    first = idx == 0
    assert bins[first].tolist() == [0] and labels[first].tolist() == [1]
    second = idx == 1
    # survives bins whose upper edge <= 100, then a partially observed bin with weight < 1
    full = int((EDGES <= 100).sum())
    assert labels[second].sum() == 0 and len(bins[second]) == full + 1
    assert 0 < weights[second][-1] < 1


def test_sage_beats_lru_on_scan_mix_and_never_leaks():
    tr = zipf_trace(60000, 4000)
    F = build_features(tr)
    logS, rep = predict_survival(F, tr.next_index, n_retrain=3, train_samples=20000, num_rounds=30, num_threads=1)
    assert logS.shape == (tr.n, N_BINS)
    assert np.all(np.diff(logS, axis=1) <= 1e-6)  # survival is non-increasing
    assert all(p < tr.n for p in rep.retrain_points)
    ev = tr.n // 5
    hits, _, _ = simulate_sage(tr.keys, logS, tr.n_objects, 200, ev, 0.5, True, True, 64, 0)
    lru = LRU(200)
    lru_hits = sum(lru.access(int(k), i) and i >= ev for i, k in enumerate(tr.keys))
    assert hits > lru_hits
    t = run_sage_tiered(tr, logS, 100, 400, ev)
    c = run_cascade(tr, "LRU", 100, 400, ev)
    assert abs(t["dram_hit"] + t["ssd_hit"] + t["hdd_miss"] - 1) < 1e-9
    assert t["mean_latency_ms"] <= c["mean_latency_ms"]
