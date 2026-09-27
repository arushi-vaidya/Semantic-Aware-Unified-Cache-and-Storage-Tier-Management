from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from src.metadata.metadata_store import MetadataStore
from src.storage.tier import Tier


class StorageError(RuntimeError):
    pass


class StorageManager:
    def __init__(self, tier_paths: dict[str, str | Path], metadata: MetadataStore):
        self.paths = {Tier(str(name).upper()): Path(path) for name, path in tier_paths.items()}
        if set(self.paths) != set(Tier):
            raise ValueError("tier_paths must define cache, ssd, hdd, and archive")
        self.metadata = metadata
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_id(object_id: str) -> None:
        if not object_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", object_id):
            raise ValueError("object_id must contain only letters, numbers, _, -, and .")

    def path(self, object_id: str, tier: Tier) -> Path:
        self._validate_id(object_id)
        return self.paths[tier] / object_id

    def put(self, object_id: str, data: bytes) -> None:
        self._validate_id(object_id)
        if self.exists(object_id):
            raise FileExistsError(f"Object already exists: {object_id}")
        destination = self.path(object_id, Tier.CACHE)
        fd, temporary = tempfile.mkstemp(prefix=f".{object_id}.", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            self.metadata.create_object(object_id, str(destination), len(data), Tier.CACHE)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise

    def get(self, object_id: str, tier: Tier | None = None) -> bytes:
        if tier is None:
            tier = self.current_tier(object_id)
        file_path = self.path(object_id, tier)
        if not file_path.is_file():
            raise FileNotFoundError(f"Object {object_id} is not present in {tier}")
        return file_path.read_bytes()

    def write_existing(self, object_id: str, data: bytes) -> None:
        tier = self.current_tier(object_id)
        destination = self.path(object_id, tier)
        fd, temporary = tempfile.mkstemp(prefix=f".{object_id}.", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            self.metadata.update_size(object_id, len(data))
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise

    def exists(self, object_id: str) -> bool:
        self._validate_id(object_id)
        return any(self.path(object_id, tier).is_file() for tier in Tier)

    def delete(self, object_id: str) -> None:
        tier = self.current_tier(object_id)
        self.path(object_id, tier).unlink()
        self.metadata.connection.execute("DELETE FROM objects WHERE object_id = ?", (object_id,))
        self.metadata.connection.commit()

    def current_tier(self, object_id: str) -> Tier:
        return Tier(self.metadata.get_object(object_id).current_tier)

    def list_objects(self, tier: Tier) -> list[str]:
        return sorted(path.name for path in self.paths[tier].iterdir() if path.is_file() and not path.name.startswith("."))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def move(self, object_id: str, source_tier: Tier, destination_tier: Tier) -> None:
        if source_tier == destination_tier:
            raise ValueError("Source and destination tiers must differ")
        source = self.path(object_id, source_tier)
        destination = self.path(object_id, destination_tier)
        if not source.is_file():
            raise FileNotFoundError(f"Source does not exist: {source}")
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        started = time.monotonic()
        bytes_moved = source.stat().st_size
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix=f".{object_id}.", dir=destination.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                with source.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, destination)
            if destination.stat().st_size != bytes_moved or self._sha256(source) != self._sha256(destination):
                raise StorageError("Migration integrity verification failed")
            source.unlink()
            self.metadata.update_tier(object_id, destination_tier)
            self.metadata.log_migration(object_id, source_tier, destination_tier, bytes_moved, (time.monotonic() - started) * 1000, True)
        except Exception as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            self.metadata.log_migration(object_id, source_tier, destination_tier, bytes_moved, (time.monotonic() - started) * 1000, False, str(exc))
            raise