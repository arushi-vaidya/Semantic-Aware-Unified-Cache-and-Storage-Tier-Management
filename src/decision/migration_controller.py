from __future__ import annotations

import threading

from src.metadata.metadata_store import MetadataStore
from src.storage.storage_manager import StorageManager
from src.storage.tier import Tier


class MigrationController:
    def __init__(
        self,
        storage: StorageManager,
        metadata: MetadataStore,
        min_residency_operations: int = 0,
        cooldown_operations: int = 0,
    ):
        if min_residency_operations < 0 or cooldown_operations < 0:
            raise ValueError("migration residency and cooldown must be non-negative")
        self.storage = storage
        self.metadata = metadata
        self.min_residency_operations = min_residency_operations
        self.cooldown_operations = cooldown_operations
        self._in_progress: set[str] = set()
        self._lock = threading.Lock()

    def _state(self, object_id: str, operation_index: int) -> tuple[int, int]:
        row = self.metadata.connection.execute(
            "SELECT tier_since_operation, last_migration_operation FROM placement_state WHERE object_id = ?",
            (object_id,),
        ).fetchone()
        if row is not None:
            return int(row[0]), int(row[1])

        first_access = self.metadata.connection.execute(
            "SELECT MIN(id) FROM access_log WHERE object_id = ?", (object_id,)
        ).fetchone()[0]
        latest_migration = self.metadata.connection.execute(
            "SELECT timestamp FROM migration_log WHERE object_id = ? AND success = 1 ORDER BY id DESC LIMIT 1",
            (object_id,),
        ).fetchone()
        if latest_migration is None:
            tier_since = max(0, int(first_access or operation_index) - 1)
            last_migration = -10**18
        else:
            migration_access = self.metadata.connection.execute(
                "SELECT COALESCE(MAX(id), 0) FROM access_log WHERE timestamp <= ?",
                (latest_migration[0],),
            ).fetchone()[0]
            tier_since = int(migration_access)
            last_migration = int(migration_access)
        self.metadata.connection.execute(
            "INSERT OR IGNORE INTO placement_state(object_id, tier_since_operation, last_migration_operation) VALUES (?, ?, ?)",
            (object_id, tier_since, last_migration),
        )
        self.metadata.connection.commit()
        return tier_since, last_migration

    def blocked_reason(self, object_id: str, operation_index: int) -> str | None:
        tier_since, last_migration = self._state(object_id, operation_index)
        if operation_index - tier_since < self.min_residency_operations:
            return "minimum residency period has not elapsed"
        if operation_index - last_migration < self.cooldown_operations:
            return "migration cooldown is active"
        return None

    def register_placement(self, object_id: str, operation_index: int) -> None:
        self.metadata.connection.execute(
            "INSERT INTO placement_state(object_id, tier_since_operation, last_migration_operation) VALUES (?, ?, ?) "
            "ON CONFLICT(object_id) DO NOTHING",
            (object_id, operation_index, -10**18),
        )
        self.metadata.connection.commit()

    def migrate(
        self,
        object_id: str,
        source: Tier,
        destination: Tier,
        operation_index: int,
        force: bool = False,
    ) -> None:
        if source == destination:
            raise ValueError("source and destination tiers must differ")
        blocked = None if force else self.blocked_reason(object_id, operation_index)
        if blocked:
            raise RuntimeError(f"migration blocked for {object_id}: {blocked}")
        with self._lock:
            if object_id in self._in_progress:
                raise RuntimeError(f"migration already in progress for {object_id}")
            self._in_progress.add(object_id)
        try:
            if self.storage.current_tier(object_id) != source:
                raise RuntimeError(f"metadata source tier changed for {object_id}")
            if not self.storage.path(object_id, source).is_file():
                raise FileNotFoundError(f"source object is missing for {object_id}")
            if self.storage.path(object_id, destination).exists():
                raise FileExistsError(f"destination object already exists for {object_id}")
            self.storage.move(object_id, source, destination)
            self.metadata.connection.execute(
                "INSERT INTO placement_state(object_id, tier_since_operation, last_migration_operation) VALUES (?, ?, ?) "
                "ON CONFLICT(object_id) DO UPDATE SET tier_since_operation = excluded.tier_since_operation, "
                "last_migration_operation = excluded.last_migration_operation",
                (object_id, operation_index, operation_index),
            )
            self.metadata.connection.commit()
        finally:
            with self._lock:
                self._in_progress.discard(object_id)