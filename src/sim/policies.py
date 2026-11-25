"""Reference cache-replacement policies for trace-driven simulation (unit-size objects).

Every policy exposes ``access(key, i) -> bool`` (True on hit), where ``i`` is the
request index. Implementations follow the published algorithms:

* LRU
* ARC      - Megiddo & Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache", FAST 2003.
* LeCaR    - Vietri et al., "Driving Cache Replacement with ML-based LeCaR", HotStorage 2018
             (lambda=0.45, discount 0.005**(1/N), per-policy history of size N, as in the
             authors' public code).
* S3-FIFO  - Yang et al., "FIFO Queues are All You Need for Cache Eviction", SOSP 2023.
* Belady   - offline optimum (MIN with bypass), used only as an upper bound.
"""
from __future__ import annotations

import heapq
import math
import random
from collections import OrderedDict

import numpy as np


class LRU:
    name = "LRU"

    def __init__(self, capacity: int, on_evict=None, **_):
        self.c = capacity
        self.d: OrderedDict[int, None] = OrderedDict()
        self.on_evict = on_evict

    def access(self, key: int, i: int) -> bool:
        d = self.d
        if key in d:
            d.move_to_end(key)
            return True
        if len(d) >= self.c:
            victim, _ = d.popitem(last=False)
            if self.on_evict is not None:
                self.on_evict(victim)
        d[key] = None
        return False


class ARC:
    name = "ARC"

    def __init__(self, capacity: int, on_evict=None, **_):
        self.c = capacity
        self.on_evict = on_evict
        self.p = 0.0
        self.t1: OrderedDict[int, None] = OrderedDict()
        self.t2: OrderedDict[int, None] = OrderedDict()
        self.b1: OrderedDict[int, None] = OrderedDict()
        self.b2: OrderedDict[int, None] = OrderedDict()

    def _replace(self, in_b2: bool) -> None:
        t1, t2 = self.t1, self.t2
        if t1 and (len(t1) > self.p or (in_b2 and len(t1) == self.p)):
            y, _ = t1.popitem(last=False)
            self.b1[y] = None
        else:
            y, _ = t2.popitem(last=False)
            self.b2[y] = None
        if self.on_evict is not None:
            self.on_evict(y)

    def access(self, key: int, i: int) -> bool:
        t1, t2, b1, b2, c = self.t1, self.t2, self.b1, self.b2, self.c
        if key in t1:
            del t1[key]
            t2[key] = None
            return True
        if key in t2:
            t2.move_to_end(key)
            return True
        if key in b1:
            self.p = min(c, self.p + max(len(b2) / len(b1), 1.0))
            self._replace(False)
            del b1[key]
            t2[key] = None
            return False
        if key in b2:
            self.p = max(0.0, self.p - max(len(b1) / len(b2), 1.0))
            self._replace(True)
            del b2[key]
            t2[key] = None
            return False
        l1 = len(t1) + len(b1)
        total = l1 + len(t2) + len(b2)
        if l1 == c:
            if len(t1) < c:
                b1.popitem(last=False)
                self._replace(False)
            else:
                y, _ = t1.popitem(last=False)
                if self.on_evict is not None:
                    self.on_evict(y)
        elif total >= c:
            if total >= 2 * c:
                b2.popitem(last=False)
            self._replace(False)
        t1[key] = None
        return False


class LeCaR:
    name = "LeCaR"

    def __init__(self, capacity: int, learning_rate: float = 0.45, seed: int = 42, **_):
        self.c = capacity
        self.lr = learning_rate
        self.discount = 0.005 ** (1.0 / capacity)
        self.w = [0.5, 0.5]  # [LRU, LFU]
        self.lru: OrderedDict[int, None] = OrderedDict()
        self.freq: dict[int, int] = {}
        self.last: dict[int, int] = {}
        self.heap: list[tuple[int, int, int]] = []
        self.hist = (OrderedDict(), OrderedDict())  # key -> (freq, evicted_time)
        self.rng = random.Random(seed)

    def _adjust(self, policy: int, reward: float) -> None:
        w = self.w
        w[policy] *= math.exp(self.lr * reward)
        s = w[0] + w[1]
        w[0] /= s
        w[1] /= s
        if w[0] >= 0.99:
            self.w = [0.99, 0.01]
        elif w[1] >= 0.99:
            self.w = [0.01, 0.99]

    def _lfu_min(self) -> int:
        heap, freq, last = self.heap, self.freq, self.last
        while True:
            f, t, k = heap[0]
            if k in freq and freq[k] == f and last[k] == t:
                return k
            heapq.heappop(heap)

    def _touch(self, key: int, i: int) -> None:
        self.last[key] = i
        heapq.heappush(self.heap, (self.freq[key], i, key))
        if len(self.heap) > 8 * self.c:
            self.heap = [(self.freq[k], self.last[k], k) for k in self.freq]
            heapq.heapify(self.heap)

    def access(self, key: int, i: int) -> bool:
        if key in self.freq:
            self.freq[key] += 1
            self.lru.move_to_end(key)
            self._touch(key, i)
            return True
        freq = 1
        for policy in (0, 1):
            hist = self.hist[policy]
            if key in hist:
                old_freq, evicted_at = hist.pop(key)
                freq = old_freq + 1
                # the policy that evicted this key is penalised
                self._adjust(policy, -(self.discount ** (i - evicted_at)))
                break
        if len(self.freq) >= self.c:
            self._evict(i)
        self.freq[key] = freq
        self.lru[key] = None
        self._touch(key, i)
        return False

    def _evict(self, i: int) -> None:
        lru_victim = next(iter(self.lru))
        lfu_victim = self._lfu_min()
        if lru_victim == lfu_victim:
            victim, policy = lru_victim, -1
        else:
            policy = 0 if self.rng.random() < self.w[0] else 1
            victim = lru_victim if policy == 0 else lfu_victim
        f = self.freq.pop(victim)
        del self.lru[victim]
        del self.last[victim]
        if policy >= 0:
            hist = self.hist[policy]
            if len(hist) >= self.c:
                hist.popitem(last=False)
            hist[victim] = (f, i)


class S3FIFO:
    name = "S3-FIFO"

    def __init__(self, capacity: int, small_ratio: float = 0.1, **_):
        self.c = capacity
        self.small_cap = max(1, int(round(capacity * small_ratio)))
        self.main_cap = capacity - self.small_cap
        self.small: OrderedDict[int, None] = OrderedDict()  # oldest first
        self.main: OrderedDict[int, None] = OrderedDict()
        self.ghost: OrderedDict[int, None] = OrderedDict()
        self.freq: dict[int, int] = {}

    def _evict_main(self) -> None:
        main, freq = self.main, self.freq
        while main:
            k, _ = main.popitem(last=False)
            f = freq[k]
            if f > 0:
                freq[k] = f - 1
                main[k] = None
            else:
                del freq[k]
                if getattr(self, "on_evict", None) is not None:
                    self.on_evict(k)
                return

    def _evict_small(self) -> None:
        small, freq = self.small, self.freq
        while small:
            k, _ = small.popitem(last=False)
            if freq[k] > 1:
                if len(self.main) >= self.main_cap:
                    self._evict_main()
                self.main[k] = None
                freq[k] = 0
            else:
                del freq[k]
                if getattr(self, "on_evict", None) is not None:
                    self.on_evict(k)
                self.ghost[k] = None
                if len(self.ghost) > self.main_cap:
                    self.ghost.popitem(last=False)
                return

    def access(self, key: int, i: int) -> bool:
        freq = self.freq
        if key in freq:
            if freq[key] < 3:
                freq[key] += 1
            return True
        while len(self.small) + len(self.main) >= self.c:
            if len(self.small) >= self.small_cap or not self.main:
                self._evict_small()
            else:
                self._evict_main()
        if key in self.ghost:
            del self.ghost[key]
            if len(self.main) >= self.main_cap:
                self._evict_main()
            self.main[key] = None
        else:
            self.small[key] = None
        freq[key] = 0
        return False


class Belady:
    """Offline MIN with bypass: never admits an object whose next use is farther than every cached one."""

    name = "Belady"

    def __init__(self, capacity: int, next_index: np.ndarray, **_):
        self.c = capacity
        self.nxt = next_index
        self.cached: dict[int, int] = {}
        self.heap: list[tuple[int, int]] = []

    def access(self, key: int, i: int) -> bool:
        cached, heap = self.cached, self.heap
        nxt = int(self.nxt[i])
        hit = key in cached
        if not hit and len(cached) >= self.c:
            while True:
                neg, k = heap[0]
                if cached.get(k) == -neg:
                    break
                heapq.heappop(heap)
            if -neg <= nxt:  # incoming object is reused later than every cached one: bypass
                return False
            heapq.heappop(heap)
            del cached[k]
        cached[key] = nxt
        heapq.heappush(heap, (-nxt, key))
        if len(heap) > 8 * self.c:
            self.heap = [(-v, k) for k, v in cached.items()]
            heapq.heapify(self.heap)
        return hit


BASELINES = {"LRU": LRU, "ARC": ARC, "LeCaR": LeCaR, "S3-FIFO": S3FIFO, "Belady": Belady}


class S4FIFO:
    """Parameterised S3-FIFO family used by S4-FIFO (Xia et al., "Learning-Augmented Heuristics:
    Simple, yet Smart, Robust and Interpretable Cache Eviction", arXiv 2608.27975, OSDI'26).

    S4-FIFO's learned selector picks one configuration per (trace, cache size) from a discrete
    grid: small-queue fraction rho_s, ghost size (multiple of main queue), skip ratio kappa
    (hits in the newest kappa fraction of S do not count), promotion threshold tau_s.
    Our benchmark reports the best grid configuration in hindsight (an upper bound on the selector).
    """

    name = "S4-FIFO"

    def __init__(self, capacity: int, rho_s: float = 0.1, ghost_mult: float = 1.0, kappa: float = 0.0,
                 tau_s: int = 2, **_):
        self.c = capacity
        self.small_cap = max(1, int(round(capacity * rho_s)))
        self.main_cap = max(1, capacity - self.small_cap)
        self.ghost_cap = max(1, int(ghost_mult * self.main_cap))
        self.kappa_n = kappa * self.small_cap
        self.tau = tau_s
        self.small: OrderedDict[int, int] = OrderedDict()  # key -> insertion sequence
        self.main: OrderedDict[int, None] = OrderedDict()
        self.ghost: OrderedDict[int, None] = OrderedDict()
        self.freq: dict[int, int] = {}
        self.seq = 0

    def _evict_main(self) -> None:
        main, freq = self.main, self.freq
        while main:
            k, _ = main.popitem(last=False)
            f = freq[k]
            if f > 0:
                freq[k] = f - 1
                main[k] = None
            else:
                del freq[k]
                if getattr(self, "on_evict", None) is not None:
                    self.on_evict(k)
                return

    def _evict_small(self) -> None:
        small, freq = self.small, self.freq
        while small:
            k, _ = small.popitem(last=False)
            if freq[k] >= self.tau:
                if len(self.main) >= self.main_cap:
                    self._evict_main()
                self.main[k] = None
                freq[k] = 0
            else:
                del freq[k]
                if getattr(self, "on_evict", None) is not None:
                    self.on_evict(k)
                self.ghost[k] = None
                if len(self.ghost) > self.ghost_cap:
                    self.ghost.popitem(last=False)
                return

    def access(self, key: int, i: int) -> bool:
        freq = self.freq
        if key in freq:
            s = self.small.get(key)
            if s is not None and self.seq - s < self.kappa_n:
                return True  # probationary skip region
            if freq[key] < 3:
                freq[key] += 1
            return True
        while len(self.small) + len(self.main) >= self.c:
            if len(self.small) >= self.small_cap or not self.main:
                self._evict_small()
            else:
                self._evict_main()
        if key in self.ghost:
            del self.ghost[key]
            if len(self.main) >= self.main_cap:
                self._evict_main()
            self.main[key] = None
        else:
            self.seq += 1
            self.small[key] = self.seq
        freq[key] = 0
        return False


# Configuration grid searched for S4-FIFO (subset of the paper's 168-point grid).
S4FIFO_GRID = [
    dict(rho_s=r, ghost_mult=g, kappa=k, tau_s=t)
    for r in (0.05, 0.10, 0.20, 0.50, 0.90) for g in (1.0, 3.0) for k in (0.0, 0.25) for t in (1, 2)
]


class SRS3FIFO:
    """SR-S3-FIFO: S3-FIFO + SR-Gadgets (Zheng & Yang, "SR-Gadgets: Make Scan-Resistant Caching
    Practical", arXiv 2609.30468, 2026): ProbBypass (warm-up misses go to main) and RecencyGuard
    (watermark = last-access time of the oldest freq-0 object in main; small-queue victims create a
    ghost entry only if accessed after the watermark; ghost hits go to main only if newer than it)."""

    name = "SR-S3-FIFO"

    def __init__(self, capacity: int, small_ratio: float = 0.1, **_):
        self.c = capacity
        self.small_cap = max(1, int(round(capacity * small_ratio)))
        self.main_cap = capacity - self.small_cap
        self.small: OrderedDict[int, None] = OrderedDict()
        self.main: OrderedDict[int, None] = OrderedDict()
        self.ghost: OrderedDict[int, int] = OrderedDict()  # key -> last access vtime
        self.freq: dict[int, int] = {}
        self.vtime: dict[int, int] = {}
        self.warm = True

    def _watermark(self) -> int:
        freq, vt = self.freq, self.vtime
        for n, k in enumerate(self.main):
            if freq[k] == 0:
                return vt[k]
            if n >= 256:
                return vt[k]
        return -1

    def _evict_main(self) -> None:
        main, freq = self.main, self.freq
        while main:
            k, _ = main.popitem(last=False)
            f = freq[k]
            if f > 0:
                freq[k] = f - 1
                main[k] = None
            else:
                del freq[k]
                if getattr(self, "on_evict", None) is not None:
                    self.on_evict(k)
                del self.vtime[k]
                return

    def _evict_small(self) -> None:
        small, freq = self.small, self.freq
        while small:
            k, _ = small.popitem(last=False)
            if freq[k] > 1:
                if len(self.main) >= self.main_cap:
                    self._evict_main()
                self.main[k] = None
                freq[k] = 0
            else:
                v = self.vtime.pop(k)
                del freq[k]
                if getattr(self, "on_evict", None) is not None:
                    self.on_evict(k)
                if v > self._watermark():
                    self.ghost[k] = v
                    if len(self.ghost) > self.main_cap:
                        self.ghost.popitem(last=False)
                return

    def access(self, key: int, i: int) -> bool:
        freq = self.freq
        if key in freq:
            if freq[key] < 3:
                freq[key] += 1
            self.vtime[key] = i
            return True
        if len(self.small) + len(self.main) >= self.c:
            self.warm = False
        while len(self.small) + len(self.main) >= self.c:
            if len(self.small) >= self.small_cap or not self.main:
                self._evict_small()
            else:
                self._evict_main()
        to_main = self.warm and len(self.main) < self.main_cap
        if key in self.ghost:
            v = self.ghost.pop(key)
            to_main = v > self._watermark()
        if to_main:
            if len(self.main) >= self.main_cap:
                self._evict_main()
            self.main[key] = None
        else:
            self.small[key] = None
        freq[key] = 0
        self.vtime[key] = i
        return False


class SIEVE:
    """SIEVE (Zhang et al., NSDI 2024): FIFO queue + visited bit + a hand sweeping from tail to head."""

    name = "SIEVE"

    def __init__(self, capacity: int, **_):
        self.c = capacity
        self.visited: dict[int, bool] = {}
        self.prev: dict[int, int | None] = {}  # towards head (newer)
        self.next: dict[int, int | None] = {}  # towards tail (older)
        self.head = None
        self.tail = None
        self.hand = None

    def _remove(self, k: int) -> None:
        p, n = self.prev.pop(k), self.next.pop(k)
        if p is None:
            self.head = n
        else:
            self.next[p] = n
        if n is None:
            self.tail = p
        else:
            self.prev[n] = p
        del self.visited[k]

    def access(self, key: int, i: int) -> bool:
        if key in self.visited:
            self.visited[key] = True
            return True
        if len(self.visited) >= self.c:
            obj = self.hand if self.hand is not None else self.tail
            while self.visited[obj]:
                self.visited[obj] = False
                obj = self.prev[obj] if self.prev[obj] is not None else self.tail
            self.hand = self.prev[obj]
            self._remove(obj)
        self.visited[key] = False
        self.prev[key] = None
        self.next[key] = self.head
        if self.head is not None:
            self.prev[self.head] = key
        self.head = key
        if self.tail is None:
            self.tail = key
        return False


BASELINES.update({"S4-FIFO": S4FIFO, "SR-S3-FIFO": SRS3FIFO, "SIEVE": SIEVE})
