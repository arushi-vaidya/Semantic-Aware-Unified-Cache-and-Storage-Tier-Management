from __future__ import annotations

import json
import statistics
from pathlib import Path

from src.cache.lru_cache import LRUCache
from src.metadata.metadata_store import MetadataStore
from src.storage.storage_manager import StorageManager
from src.storage.tier import Tier


class MetricsCollector:
    def __init__(self, cache: LRUCache, storage: StorageManager, metadata: MetadataStore):
        self.cache = cache
        self.storage = storage
        self.metadata = metadata

    def collect(self) -> dict:
        rows = self.metadata.connection.execute("SELECT latency_ms FROM access_log WHERE success = 1 ORDER BY id").fetchall()
        latencies = [float(row[0]) for row in rows]
        request_counts = self.metadata.connection.execute("SELECT tier, COUNT(*) FROM access_log WHERE operation = 'read' AND success = 1 GROUP BY tier").fetchall()
        hit_count = sum(int(row[1]) for row in request_counts if row[0] == Tier.CACHE)
        miss_count = sum(int(row[1]) for row in request_counts if row[0] != Tier.CACHE)
        total = hit_count + miss_count
        def percentile(value: float) -> float:
            if not latencies:
                return 0.0
            ordered = sorted(latencies)
            index = max(0, min(len(ordered) - 1, int((value * len(ordered) + 0.999999) - 1)))
            return ordered[index]
        migrations = self.metadata.connection.execute("SELECT COUNT(*), COALESCE(SUM(bytes_moved), 0) FROM migration_log WHERE success = 1").fetchone()
        result = {
            "cache_hits": hit_count,
            "cache_misses": miss_count,
            "total_cache_requests": total,
            "cache_hit_ratio": hit_count / total if total else 0.0,
            "cache_miss_ratio": miss_count / total if total else 0.0,
            "average_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
            "p95_latency_ms": percentile(0.95),
            "p99_latency_ms": percentile(0.99),
            "migration_count": int(migrations[0]),
            "migration_volume_bytes": int(migrations[1]),
            "storage": {},
        }
        capacities = {Tier.CACHE: self.cache.capacity_bytes, Tier.SSD: self.cache.ssd_capacity_bytes, Tier.HDD: self.cache.hdd_capacity_bytes, Tier.ARCHIVE: None}
        for tier in Tier:
            used = sum(self.storage.path(object_id, tier).stat().st_size for object_id in self.storage.list_objects(tier))
            capacity = capacities[tier]
            result["storage"][tier.value] = {"used_bytes": used, "capacity_bytes": capacity, "utilization_percentage": (used / capacity * 100 if capacity else None)}
        try:
            import psutil
            result["memory_rss_bytes"] = psutil.Process().memory_info().rss
        except ImportError:
            result["memory_rss_bytes"] = None
        return result

    def save(self, path: str | Path) -> dict:
        metrics = self.collect()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics