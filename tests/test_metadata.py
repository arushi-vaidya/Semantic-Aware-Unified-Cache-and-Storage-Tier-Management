from src.metadata.metadata_store import MetadataStore


def test_object_and_access_metadata(tmp_path):
    metadata = MetadataStore(tmp_path / "db.sqlite")
    metadata.create_object("item", "/tmp/item", 3, "CACHE")
    metadata.update_access("item", "read", "CACHE", 1.25, True)
    record = metadata.get_object("item")
    assert record.access_count == 1 and record.read_count == 1 and record.write_count == 1
    assert metadata.connection.execute("SELECT COUNT(*) FROM access_log").fetchone()[0] == 1
    metadata.close()