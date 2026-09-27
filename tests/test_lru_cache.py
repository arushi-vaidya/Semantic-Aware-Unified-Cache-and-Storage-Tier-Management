from src.cache.lru_cache import LRUCache
from src.metadata.metadata_store import MetadataStore
from src.storage.storage_manager import StorageManager
from src.storage.tier import Tier


def make_system(tmp_path, cache=5, ssd=100, hdd=100):
    metadata = MetadataStore(tmp_path / "metadata.db")
    paths = {tier.value.lower(): tmp_path / tier.value.lower() for tier in Tier}
    storage = StorageManager(paths, metadata)
    return LRUCache(cache, storage, metadata, ssd, hdd), storage, metadata


def test_lru_hit_miss_and_eviction(tmp_path):
    cache, storage, metadata = make_system(tmp_path)
    cache.put("a", b"1234")
    cache.put("b", b"5678")
    assert storage.current_tier("a") == Tier.SSD
    assert cache.get("a") == b"1234"
    assert cache.hits == 0 and cache.misses == 1
    assert storage.current_tier("a") == Tier.CACHE
    cache.get("a")
    assert cache.hits == 1
    metadata.close()


def test_capacity_is_enforced(tmp_path):
    cache, storage, metadata = make_system(tmp_path, cache=4)
    cache.put("a", b"1234")
    cache.put("b", b"5678")
    assert storage.list_objects(Tier.CACHE) == ["b"]
    assert storage.list_objects(Tier.SSD) == ["a"]
    metadata.close()