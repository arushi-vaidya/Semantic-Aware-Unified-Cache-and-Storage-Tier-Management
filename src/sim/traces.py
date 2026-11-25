"""Loading of real block-I/O traces in libCacheSim ``oracleGeneral`` format.

Each record is ``<u4 timestamp, <u8 obj_id, <u4 obj_size, <i8 next_access_vtime``
(24 bytes). Block traces use the byte (or sector) offset as ``obj_id``; we treat
every distinct offset as one unit-sized cache object (page-granularity caching,
the convention used by ARC, LeCaR and CACHEUS).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ORACLE_DTYPE = np.dtype([("ts", "<u4"), ("id", "<u8"), ("size", "<u4"), ("nxt", "<i8")])
TRACE_DIR = Path(__file__).resolve().parents[2] / "data" / "traces" / "bin"

# Spatial region granularity (shift applied to obj_id) per trace family.
# MSR and FIU ids are byte offsets -> 1 MiB extents; CloudPhysics ids are
# 512-byte sectors -> 2048 sectors = 1 MiB extents.
REGION_SHIFT = {"msr": 20, "fiu": 20, "w": 11}

TRACE_SOURCES = {
    "msr": "MSR Cambridge enterprise block traces (Narayanan et al., FAST'08), SNIA IOTTA",
    "fiu": "FIU SRCMap/IODedup block traces (Koller & Rangaswami, FAST'10), SNIA IOTTA",
    "w": "CloudPhysics VMware block traces (Waldspurger et al., FAST'15)",
}


@dataclass
class Trace:
    name: str
    keys: np.ndarray        # int32 dense object ids, 0..n_objects-1
    raw_ids: np.ndarray     # uint64 original offsets
    regions: np.ndarray     # int32 dense spatial-region ids
    sizes: np.ndarray       # uint32 request sizes in bytes
    timestamps: np.ndarray  # uint32 seconds
    next_index: np.ndarray  # int64 index of next request to same key, or n
    n_objects: int
    n_regions: int

    @property
    def n(self) -> int:
        return len(self.keys)


def family(name: str) -> str:
    for prefix in ("msr", "fiu"):
        if name.startswith(prefix):
            return prefix
    return "w"


def _dense(values: np.ndarray) -> tuple[np.ndarray, int]:
    uniques, inverse = np.unique(values, return_inverse=True)
    return inverse.astype(np.int32), len(uniques)


def compute_next_index(keys: np.ndarray) -> np.ndarray:
    n = len(keys)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    nxt = np.full(n, n, dtype=np.int64)
    same = sorted_keys[1:] == sorted_keys[:-1]
    nxt[order[:-1][same]] = order[1:][same]
    return nxt


def load_trace(name: str, max_requests: int | None = None, trace_dir: Path = TRACE_DIR) -> Trace:
    path = Path(trace_dir) / f"{name}.bin"
    raw = np.memmap(path, dtype=ORACLE_DTYPE, mode="r")
    if max_requests is not None:
        raw = raw[:max_requests]
    ids = np.asarray(raw["id"], dtype=np.uint64)
    keys, n_objects = _dense(ids)
    regions, n_regions = _dense(ids >> np.uint64(REGION_SHIFT[family(name)]))
    return Trace(
        name=name,
        keys=keys,
        raw_ids=ids,
        regions=regions,
        sizes=np.asarray(raw["size"], dtype=np.uint32),
        timestamps=np.asarray(raw["ts"], dtype=np.uint32),
        next_index=compute_next_index(keys),
        n_objects=n_objects,
        n_regions=n_regions,
    )
