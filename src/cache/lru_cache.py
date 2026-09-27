from __future__ import annotations

from collections import OrderedDict
import time

from src.metadata.metadata_store import MetadataStore
from src.storage.storage_manager import StorageManager
from src.storage.tier import Tier


class LRUCache:
    def __init__(self, capacity_bytes: int, storage: StorageManager, metadata: MetadataStore, ssd_capacity_bytes: int, hdd_capacity_bytes: int):
        if capacity_bytes <= 0 or ssd_capacity_bytes <= 0 or hdd_capacity_bytes <= 0:
            raise ValueError("Tier capacities must be positive")
        self.capacity_bytes = capacity_bytes
        self.storage = storage
        self.metadata = metadata
        self.ssd_capacity_bytes = ssd_capacity_bytes
        self.hdd_capacity_bytes = hdd_capacity_bytes
        self._items: OrderedDict[str, int] = OrderedDict()
        self.hits = 0
        self.misses = 0
        for object_id in storage.list_objects(Tier.CACHE):
            self._items[object_id] = storage.path(object_id, Tier.CACHE).stat().st_size

    def _used(self, tier: Tier) -> int:
        return sum(self.storage.path(object_id, tier).stat().st_size for object_id in self.storage.list_objects(tier))

    def _demote_overflow(self, tier: Tier, limit: int, destination: Tier) -> None:
        while self._used(tier) > limit:
            candidates = [self.metadata.get_object(object_id) for object_id in self.storage.list_objects(tier)]
            if not candidates:
                raise RuntimeError(f"Unable to enforce {tier} capacity")
            victim = min(candidates, key=lambda item: item.last_accessed_at)
            self.storage.move(victim.object_id, tier, destination)

    def _evict(self) -> None:
        while sum(self._items.values()) > self.capacity_bytes:
            object_id, size = self._items.popitem(last=False)
            self.storage.move(object_id, Tier.CACHE, Tier.SSD)
            self._demote_overflow(Tier.SSD, self.ssd_capacity_bytes, Tier.HDD)
            self._demote_overflow(Tier.HDD, self.hdd_capacity_bytes, Tier.ARCHIVE)

    def put(self, object_id: str, data: bytes) -> None:
        started = time.monotonic()
        try:
            if self.storage.exists(object_id):
                if not self.contains(object_id):
                    source = self.storage.current_tier(object_id)
                    self.storage.move(object_id, source, Tier.CACHE)
                self.storage.write_existing(object_id, data)
            else:
                self.storage.put(object_id, data)
            self._items[object_id] = len(data)
            self._items.move_to_end(object_id)
            self._evict()
            self.metadata.update_access(object_id, "write", Tier.CACHE, (time.monotonic() - started) * 1000, True)
        except Exception as exc:
            if self.metadata.connection.execute("SELECT 1 FROM objects WHERE object_id = ?", (object_id,)).fetchone():
                self.metadata.update_access(object_id, "write", Tier.CACHE, (time.monotonic() - started) * 1000, False, str(exc))
            raise

    def get(self, object_id: str) -> bytes:
        started = time.monotonic()
        hit = object_id in self._items and self.storage.path(object_id, Tier.CACHE).is_file()
        source_tier = Tier.CACHE if hit else self.storage.current_tier(object_id)
        try:
            data = self.storage.get(object_id, source_tier)
            if hit:
                self.hits += 1
                self._items.move_to_end(object_id)
            else:
                self.misses += 1
                self.storage.move(object_id, source_tier, Tier.CACHE)
                self._items[object_id] = len(data)
                self._items.move_to_end(object_id)
                self._evict()
            self.metadata.update_access(object_id, "read", source_tier, (time.monotonic() - started) * 1000, True)
            return data
        except Exception as exc:
            if self.metadata.connection.execute("SELECT 1 FROM objects WHERE object_id = ?", (object_id,)).fetchone():
                self.metadata.update_access(object_id, "read", source_tier, (time.monotonic() - started) * 1000, False, str(exc))
            raise

    def contains(self, object_id: str) -> bool:
        return object_id in self._items and self.storage.path(object_id, Tier.CACHE).is_file()

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses