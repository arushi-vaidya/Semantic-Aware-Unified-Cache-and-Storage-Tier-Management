from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.prediction.feature_extractor import (
    CATEGORICAL_FEATURES,
    DEFAULT_RECENT_WINDOWS,
    IDENTIFIER_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    extract_features,
)


def chronological_split(
    dataset: pd.DataFrame,
    prediction_window: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> dict[str, pd.DataFrame]:
    if prediction_window <= 0 or not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("invalid prediction window or split fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")
    positions = sorted(dataset["operation_index"].unique())
    if not positions:
        return {name: dataset.iloc[0:0].copy() for name in ("train", "validation", "test")}
    train_boundary = positions[min(math.floor(len(positions) * train_fraction), len(positions) - 1)]
    validation_boundary = positions[
        min(math.floor(len(positions) * (train_fraction + validation_fraction)), len(positions) - 1)
    ]
    operation = dataset["operation_index"]
    train = dataset[operation < train_boundary - prediction_window]
    validation = dataset[(operation >= train_boundary) & (operation < validation_boundary - prediction_window)]
    test = dataset[operation >= validation_boundary]
    return {"train": train.copy(), "validation": validation.copy(), "test": test.copy()}


def build_dataset(
    database_path: str | Path,
    output_dir: str | Path,
    prediction_window: int = 20,
    recent_windows: Sequence[int] = DEFAULT_RECENT_WINDOWS,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    cold_start_recency: float = 1_000_000.0,
    workload: str = "unspecified",
    workload_seed: int | None = None,
) -> pd.DataFrame:
    if prediction_window <= 0:
        raise ValueError("prediction_window must be positive")
    if prediction_window not in recent_windows:
        recent_windows = (*recent_windows, prediction_window)
    database = Path(database_path)
    if not database.is_file():
        raise FileNotFoundError(f"SQLite access log database not found: {database}")
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(access_log)")}
        if not {"id", "object_id", "operation", "timestamp", "success"}.issubset(columns):
            raise ValueError("database does not contain a Phase 1 access_log table")
        records = [dict(row) for row in connection.execute("SELECT * FROM access_log ORDER BY id")]
        migration_records = [dict(row) for row in connection.execute(
            "SELECT object_id, destination_tier, timestamp FROM migration_log WHERE success = 1 ORDER BY timestamp, id"
        )]
    finally:
        connection.close()
    if len(records) <= prediction_window:
        raise ValueError("not enough access-log operations for the configured prediction window")

    for index, record in enumerate(records):
        record["operation_index"] = index
    histories: dict[str, list[dict]] = {}
    snapshots: dict[str, tuple[int | None, str | None]] = {}
    migration_index = 0
    rows: list[dict] = []
    for prediction_position in range(1, len(records) - prediction_window + 1):
        previous = records[prediction_position - 1]
        object_id = previous["object_id"]
        prediction_time = datetime.fromisoformat(previous["timestamp"])
        while migration_index < len(migration_records):
            migration = migration_records[migration_index]
            migration_time = datetime.fromisoformat(migration["timestamp"])
            if migration_time > prediction_time:
                break
            size, _ = snapshots.get(migration["object_id"], (None, None))
            snapshots[migration["object_id"]] = (size, migration["destination_tier"])
            migration_index += 1
        if previous.get("object_size_bytes") is not None or previous.get("current_tier") is not None:
            old_size, old_tier = snapshots.get(object_id, (None, None))
            snapshots[object_id] = (
                previous.get("object_size_bytes") if previous.get("object_size_bytes") is not None else old_size,
                previous.get("current_tier") if previous.get("current_tier") is not None else old_tier,
            )
        if previous["success"]:
            histories.setdefault(object_id, []).append(previous)

        future = records[prediction_position:prediction_position + prediction_window]
        accessed = {event["object_id"] for event in future if event["success"]}
        for known_object_id, history in histories.items():
            size, tier = snapshots.get(known_object_id, (None, None))
            features = extract_features(
                history,
                prediction_position,
                previous["timestamp"],
                size,
                tier,
                recent_windows,
                cold_start_recency,
            )
            rows.append({
                "timestamp": previous["timestamp"],
                "operation_index": prediction_position,
                "object_id": known_object_id,
                **features,
                TARGET_COLUMN: int(known_object_id in accessed),
            })

    dataset = pd.DataFrame(rows)
    ordered_columns = [
        *IDENTIFIER_COLUMNS,
        *NUMERIC_FEATURES,
        *(f"accesses_in_last_{window}_operations" for window in recent_windows if window not in DEFAULT_RECENT_WINDOWS),
        *CATEGORICAL_FEATURES,
        TARGET_COLUMN,
    ]
    dataset = dataset.loc[:, list(dict.fromkeys(ordered_columns))]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output / "features.csv", index=False)
    splits = chronological_split(dataset, prediction_window, train_fraction, validation_fraction)
    for name, frame in splits.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    metadata = {
        "source_database": str(database),
        "workload": workload,
        "workload_seed": workload_seed,
        "prediction_window": prediction_window,
        "recent_windows": list(recent_windows),
        "cold_start_recency": cold_start_recency,
        "split_fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": 1.0 - train_fraction - validation_fraction,
        },
        "split_policy": "chronological by operation index; prediction windows crossing train/validation boundaries are purged",
        "feature_columns": [*NUMERIC_FEATURES, *(f"accesses_in_last_{window}_operations" for window in recent_windows if window not in DEFAULT_RECENT_WINDOWS), *CATEGORICAL_FEATURES],
        "identifier_columns_excluded_from_model": IDENTIFIER_COLUMNS,
        "samples": len(dataset),
        "class_distribution": {str(int(key)): int(value) for key, value in dataset[TARGET_COLUMN].value_counts().sort_index().items()},
        "split_samples": {name: len(frame) for name, frame in splits.items()},
        "historical_snapshot_note": "event-time size/tier snapshots are null for access rows written before the Phase 2 schema extension",
    }
    (output / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return dataset