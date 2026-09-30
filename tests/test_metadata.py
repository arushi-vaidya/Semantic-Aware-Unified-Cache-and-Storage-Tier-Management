from src.metadata.metadata_store import MetadataStore


def test_object_and_access_metadata(tmp_path):
    metadata = MetadataStore(tmp_path / "db.sqlite")
    metadata.create_object("item", "/tmp/item", 3, "CACHE")
    metadata.update_access("item", "read", "CACHE", 1.25, True)
    record = metadata.get_object("item")
    assert record.access_count == 1 and record.read_count == 1 and record.write_count == 1
    assert metadata.connection.execute("SELECT COUNT(*) FROM access_log").fetchone()[0] == 1
    snapshot = metadata.connection.execute("SELECT object_size_bytes, current_tier FROM access_log").fetchone()
    assert tuple(snapshot) == (3, "CACHE")
    metadata.close()


def test_phase1_database_is_upgraded_without_losing_access_logs(tmp_path):
    import sqlite3

    database = tmp_path / "old.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE objects (object_id TEXT PRIMARY KEY, file_path TEXT NOT NULL, size_bytes INTEGER NOT NULL, "
        "created_at TEXT NOT NULL, last_accessed_at TEXT NOT NULL, access_count INTEGER NOT NULL, read_count INTEGER NOT NULL, "
        "write_count INTEGER NOT NULL, current_tier TEXT NOT NULL, previous_tier TEXT, migration_count INTEGER NOT NULL); "
        "CREATE TABLE access_log (id INTEGER PRIMARY KEY AUTOINCREMENT, object_id TEXT NOT NULL, operation TEXT NOT NULL, "
        "timestamp TEXT NOT NULL, tier TEXT NOT NULL, latency_ms REAL NOT NULL, success INTEGER NOT NULL, error TEXT); "
        "CREATE TABLE migration_log (id INTEGER PRIMARY KEY AUTOINCREMENT, object_id TEXT NOT NULL, source_tier TEXT NOT NULL, "
        "destination_tier TEXT NOT NULL, timestamp TEXT NOT NULL, bytes_moved INTEGER NOT NULL, duration_ms REAL NOT NULL, "
        "success INTEGER NOT NULL, error TEXT); "
        "INSERT INTO objects VALUES ('old', '/tmp/old', 4, '2026-01-01', '2026-01-01', 0, 0, 0, 'CACHE', NULL, 0);"
    )
    connection.close()

    metadata = MetadataStore(database)
    assert metadata.connection.execute("SELECT COUNT(*) FROM access_log").fetchone()[0] == 0
    metadata.update_access("old", "read", "CACHE", 0.1, True)
    snapshot = metadata.connection.execute("SELECT object_size_bytes, current_tier FROM access_log").fetchone()
    assert tuple(snapshot) == (4, "CACHE")
    metadata.close()