from src.metadata.metadata_store import MetadataStore
from src.storage.storage_manager import StorageManager
from src.storage.tier import Tier


def test_write_read_delete_and_tier_detection(tmp_path):
    metadata = MetadataStore(tmp_path / "db.sqlite")
    storage = StorageManager({tier.value.lower(): tmp_path / tier.value.lower() for tier in Tier}, metadata)
    storage.put("item", b"payload")
    assert storage.exists("item")
    assert storage.get("item") == b"payload"
    assert storage.current_tier("item") == Tier.CACHE
    metadata.update_access("item", "read", "CACHE", 0.1, True)
    storage.delete("item")
    assert not storage.exists("item")
    metadata.close()