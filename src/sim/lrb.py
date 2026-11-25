"""LRB: Learning Relaxed Belady (Song et al., NSDI 2020), unit-size re-implementation.

Faithful to the paper's design and published defaults:
* features at eviction time: age (delta0), up to 31 past inter-arrival deltas,
  10 exponentially decayed counters with windows 2^(9+i);
* online training data: at every request a random object inside the memory
  window is sampled; its label is the distance to its next request, or 2*W if it
  is not requested again within the memory window W;
* GBM regression on log1p(distance): 32 iterations, 32 leaves, lr 0.1,
  bagging/feature fraction 0.8, retrained every 131072 labelled samples;
* eviction: objects whose age exceeds W are evicted first (relaxed Belady
  boundary); otherwise 64 random cached objects are scored and the one with the
  largest predicted next-access distance is evicted. LRU before the first model.
"""
from __future__ import annotations

import random
from collections import OrderedDict

import lightgbm as lgb
import numpy as np

N_DELTAS = 31
N_EDC = 10
EDC_WINDOWS = np.array([2.0 ** (9 + i) for i in range(N_EDC)])
N_FEATURES = 1 + N_DELTAS + N_EDC
GBM_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.1,
    "num_leaves": 32,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "num_threads": 1,
    "verbose": -1,
    "seed": 42,
}


class LRB:
    name = "LRB"

    def __init__(self, capacity: int, n_objects: int, memory_window: int, batch_size: int = 131072,
                 sample_size: int = 64, num_iterations: int = 32, seed: int = 42, **_):
        self.c = capacity
        self.W = int(memory_window)
        self.batch = batch_size
        self.k = sample_size
        self.iters = num_iterations
        self.rng = random.Random(seed)
        self.last = np.full(n_objects, -1, dtype=np.int64)
        self.deltas = np.full((n_objects, N_DELTAS), np.nan, dtype=np.float32)
        self.edc = np.zeros((n_objects, N_EDC), dtype=np.float64)
        self.edc_t = np.zeros(n_objects, dtype=np.int64)
        # cache
        self.lru: OrderedDict[int, None] = OrderedDict()
        self.slots: list[int] = []
        self.pos = np.full(n_objects, -1, dtype=np.int64)
        # memory-window membership for training-sample selection
        self.tracked: list[int] = []
        self.tpos = np.full(n_objects, -1, dtype=np.int64)
        # training buffer
        cap = 2 * batch_size + 16
        self.X = np.empty((cap, N_FEATURES), dtype=np.float32)
        self.y = np.full(cap, np.nan, dtype=np.float64)
        self.st = np.zeros(cap, dtype=np.int64)
        self.n_buf = 0
        self.n_labelled = 0
        self.pending = np.full(n_objects, -1, dtype=np.int64)
        self.model: lgb.Booster | None = None
        self.n_trainings = 0

    # ---------------------------------------------------------------- features
    def _features(self, objs: np.ndarray, t: int, out: np.ndarray) -> None:
        last = self.last[objs]
        out[:, 0] = t - last
        out[:, 1:1 + N_DELTAS] = self.deltas[objs]
        decay = np.power(2.0, -(t - self.edc_t[objs])[:, None] / EDC_WINDOWS[None, :])
        out[:, 1 + N_DELTAS:] = self.edc[objs] * decay

    # ---------------------------------------------------------------- training
    def _sample_training(self, t: int) -> None:
        tracked = self.tracked
        if not tracked:
            return
        o = tracked[self.rng.randrange(len(tracked))]
        if t - self.last[o] > self.W:
            self._untrack(o)
            return
        if self.pending[o] >= 0 or self.n_buf >= self.X.shape[0]:
            return
        row = self.n_buf
        self._features(np.array([o]), t, self.X[row:row + 1])
        self.st[row] = t
        self.y[row] = np.nan
        self.pending[o] = row
        self.n_buf += 1

    def _untrack(self, o: int) -> None:
        p = self.tpos[o]
        if p < 0:
            return
        last = self.tracked.pop()
        if last != o:
            self.tracked[p] = last
            self.tpos[last] = p
        self.tpos[o] = -1

    def _maybe_train(self, t: int) -> None:
        n = self.n_buf
        y = self.y[:n]
        expired = np.isnan(y) & (t - self.st[:n] > self.W)
        if self.n_labelled + int(expired.sum()) < self.batch and n < self.X.shape[0]:
            return
        y = y.copy()
        y[expired] = 2 * self.W
        ok = ~np.isnan(y)
        if ok.sum() > 100:
            data = lgb.Dataset(self.X[:n][ok], np.log1p(y[ok]), free_raw_data=True)
            self.model = lgb.train(GBM_PARAMS, data, num_boost_round=self.iters)
            self.n_trainings += 1
        self.pending[:] = -1
        self.n_buf = 0
        self.n_labelled = 0

    # ---------------------------------------------------------------- cache
    def _evict(self, t: int) -> None:
        oldest = next(iter(self.lru))
        if self.model is None or t - self.last[oldest] > self.W:
            victim = oldest
        else:
            slots = self.slots
            cand = np.fromiter((slots[self.rng.randrange(len(slots))] for _ in range(self.k)), dtype=np.int64, count=self.k)
            X = np.empty((self.k, N_FEATURES), dtype=np.float32)
            self._features(cand, t, X)
            pred = self.model.predict(X, num_threads=1)
            victim = int(cand[int(np.argmax(pred))])
        del self.lru[victim]
        p = self.pos[victim]
        last = self.slots.pop()
        if last != victim:
            self.slots[p] = last
            self.pos[last] = p
        self.pos[victim] = -1

    def access(self, key: int, i: int) -> bool:
        # label a pending training sample of this object
        row = self.pending[key]
        if row >= 0:
            self.y[row] = i - self.st[row]
            self.n_labelled += 1
            self.pending[key] = -1
        hit = self.pos[key] >= 0
        if hit:
            self.lru.move_to_end(key)
        else:
            if len(self.slots) >= self.c:
                self._evict(i)
            self.pos[key] = len(self.slots)
            self.slots.append(key)
            self.lru[key] = None
        # update per-object metadata
        last = self.last[key]
        if last >= 0:
            d = self.deltas[key]
            d[1:] = d[:-1]
            d[0] = i - last
            self.edc[key] = 1.0 + self.edc[key] * np.power(2.0, -(i - self.edc_t[key]) / EDC_WINDOWS)
        else:
            self.edc[key] = 1.0
        self.edc_t[key] = i
        self.last[key] = i
        if self.tpos[key] < 0:
            self.tpos[key] = len(self.tracked)
            self.tracked.append(key)
        self._sample_training(i)
        if (i & 1023) == 0:
            self._maybe_train(i)
        return hit
