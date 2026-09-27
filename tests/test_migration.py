import pytest

from src.metadata.metadata_store import MetadataStore
from src.storage.storage_manager import StorageManager
from src.storage.storage_manager import StorageError
from src.storage.tier import Tier


def test_migration_updates_metadata_and_preserves_data(tmp_path):
    metadata = MetadataStore(tmp_path / "db.sqlite")
    storage = StorageManager({tier.value.lower(): tmp_path / tier.value.lower() for tier in Tier}, metadata)
    storage.put("item", b"important data")
    storage.move("item", Tier.CACHE, Tier.SSD)
    assert storage.get("item") == b"important data"
    assert storage.current_tier("item") == Tier.SSD
    record = metadata.get_object("item")
    assert record.previous_tier == Tier.CACHE and record.migration_count == 1
    assert metadata.connection.execute("SELECT success FROM migration_log").fetchone()[0] == 1
    metadata.close()


def test_missing_source_is_reported(tmp_path):
    metadata = MetadataStore(tmp_path / "db.sqlite")
    storage = StorageManager({tier.value.lower(): tmp_path / tier.value.lower() for tier in Tier}, metadata)
    with pytest.raises(KeyError):
        storage.current_tier("missing")
    metadata.close()