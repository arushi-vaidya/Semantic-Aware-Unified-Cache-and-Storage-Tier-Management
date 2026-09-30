from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ObjectMetadata:
    object_id: str
    file_path: str
    size_bytes: int
    created_at: str
    last_accessed_at: str
    access_count: int
    read_count: int
    write_count: int
    current_tier: str
    previous_tier: str | None
    migration_count: int


class MetadataStore:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS objects (
                object_id TEXT PRIMARY KEY, file_path TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL, last_accessed_at TEXT NOT NULL, access_count INTEGER NOT NULL,
                read_count INTEGER NOT NULL, write_count INTEGER NOT NULL, current_tier TEXT NOT NULL,
                previous_tier TEXT, migration_count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, object_id TEXT NOT NULL, operation TEXT NOT NULL,
                timestamp TEXT NOT NULL, tier TEXT NOT NULL, latency_ms REAL NOT NULL, success INTEGER NOT NULL,
                error TEXT, object_size_bytes INTEGER, current_tier TEXT,
                FOREIGN KEY(object_id) REFERENCES objects(object_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS migration_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, object_id TEXT NOT NULL, source_tier TEXT NOT NULL,
                destination_tier TEXT NOT NULL, timestamp TEXT NOT NULL, bytes_moved INTEGER NOT NULL,
                duration_ms REAL NOT NULL, success INTEGER NOT NULL, error TEXT,
                FOREIGN KEY(object_id) REFERENCES objects(object_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_access_object ON access_log(object_id);
            CREATE INDEX IF NOT EXISTS idx_access_time ON access_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_migration_object ON migration_log(object_id);
            """
        )
        access_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(access_log)")}
        if "object_size_bytes" not in access_columns:
            self.connection.execute("ALTER TABLE access_log ADD COLUMN object_size_bytes INTEGER")
        if "current_tier" not in access_columns:
            self.connection.execute("ALTER TABLE access_log ADD COLUMN current_tier TEXT")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS semantic_importance ("
            "object_id TEXT PRIMARY KEY, score REAL NOT NULL CHECK(score >= 0 AND score <= 1), "
            "category TEXT, updated_at TEXT NOT NULL, "
            "FOREIGN KEY(object_id) REFERENCES objects(object_id) ON DELETE CASCADE)"
        )
        self.connection.commit()

    def create_object(self, object_id: str, file_path: str, size_bytes: int, tier: str) -> ObjectMetadata:
        now = utc_now()
        self.connection.execute(
            "INSERT INTO objects VALUES (?, ?, ?, ?, ?, 0, 0, 1, ?, NULL, 0)",
            (object_id, file_path, size_bytes, now, now, tier),
        )
        self.connection.commit()
        return self.get_object(object_id)

    def get_object(self, object_id: str) -> ObjectMetadata:
        row = self.connection.execute("SELECT * FROM objects WHERE object_id = ?", (object_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown object: {object_id}")
        return ObjectMetadata(**dict(row))

    def update_access(self, object_id: str, operation: str, tier: str, latency_ms: float, success: bool, error: str | None = None) -> None:
        now = utc_now()
        object_state = self.connection.execute(
            "SELECT size_bytes, current_tier FROM objects WHERE object_id = ?", (object_id,)
        ).fetchone()
        if object_state is None:
            raise KeyError(f"Unknown object: {object_id}")
        self.connection.execute(
            "UPDATE objects SET last_accessed_at = ?, access_count = access_count + 1, read_count = read_count + ?, write_count = write_count + ? WHERE object_id = ?",
            (now, int(operation == "read" and success), int(operation == "write" and success), object_id),
        )
        self.connection.execute(
            "INSERT INTO access_log(object_id, operation, timestamp, tier, latency_ms, success, error, object_size_bytes, current_tier) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (object_id, operation, now, tier, latency_ms, int(success), error, object_state["size_bytes"], object_state["current_tier"]),
        )
        self.connection.commit()

    def set_semantic_importance(self, object_id: str, score: float, category: str | None = None) -> None:
        if not 0.0 <= score <= 1.0:
            raise ValueError("semantic importance score must be between 0 and 1")
        self.get_object(object_id)
        self.connection.execute(
            "INSERT INTO semantic_importance(object_id, score, category, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(object_id) DO UPDATE SET score = excluded.score, category = excluded.category, updated_at = excluded.updated_at",
            (object_id, score, category, utc_now()),
        )
        self.connection.commit()

    def get_semantic_importance(self, object_id: str) -> tuple[float, str | None] | None:
        row = self.connection.execute(
            "SELECT score, category FROM semantic_importance WHERE object_id = ?", (object_id,)
        ).fetchone()
        return None if row is None else (float(row["score"]), row["category"])

    def update_tier(self, object_id: str, tier: str) -> None:
        current = self.get_object(object_id).current_tier
        self.connection.execute("UPDATE objects SET previous_tier = ?, current_tier = ?, migration_count = migration_count + 1 WHERE object_id = ?", (current, tier, object_id))
        self.connection.commit()

    def update_size(self, object_id: str, size_bytes: int) -> None:
        self.connection.execute("UPDATE objects SET size_bytes = ? WHERE object_id = ?", (size_bytes, object_id))
        self.connection.commit()

    def log_migration(self, object_id: str, source: str, destination: str, bytes_moved: int, duration_ms: float, success: bool, error: str | None = None) -> None:
        self.connection.execute(
            "INSERT INTO migration_log(object_id, source_tier, destination_tier, timestamp, bytes_moved, duration_ms, success, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (object_id, source, destination, utc_now(), bytes_moved, duration_ms, int(success), error),
        )
        self.connection.commit()

    def all_objects(self) -> list[ObjectMetadata]:
        return [ObjectMetadata(**dict(row)) for row in self.connection.execute("SELECT * FROM objects ORDER BY object_id")]

    def close(self) -> None:
        self.connection.close()