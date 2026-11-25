from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.metadata.metadata_store import MetadataStore
from src.prediction.feature_extractor import DEFAULT_RECENT_WINDOWS, extract_features


@lru_cache(maxsize=8)
def _load_bundle(model_path: str) -> dict:
    return joblib.load(model_path)


def predict_object(model_path: str | Path, metadata: MetadataStore, object_id: str) -> float:
    return predict_objects(model_path, metadata, [object_id])[object_id]


def predict_objects(model_path: str | Path, metadata: MetadataStore, object_ids: Iterable[str]) -> dict[str, float]:
    requested = list(dict.fromkeys(object_ids))
    if not requested:
        return {}
    bundle = _load_bundle(str(Path(model_path).resolve()))
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    rows = [dict(row) for row in metadata.connection.execute("SELECT * FROM access_log ORDER BY id")]
    for index, row in enumerate(rows):
        row["operation_index"] = index
    requested_set = set(requested)
    histories: dict[str, list[dict]] = {object_id: [] for object_id in requested}
    for row in rows:
        if row["object_id"] in requested_set and row["success"] and row["operation"] in {"read", "write"}:
            histories[row["object_id"]].append(row)
    records = {record.object_id: record for record in metadata.all_objects() if record.object_id in requested_set}
    missing = requested_set - records.keys()
    if missing:
        raise KeyError(f"Unknown objects for prediction: {', '.join(sorted(missing))}")
    windows = set(DEFAULT_RECENT_WINDOWS)
    for name in feature_columns:
        match = re.fullmatch(r"accesses_in_last_(\d+)_operations", name)
        if match:
            windows.add(int(match.group(1)))
    prediction_timestamp = datetime.now(timezone.utc).isoformat()
    feature_rows = [
        extract_features(
            histories[object_id], len(rows), prediction_timestamp,
            records[object_id].size_bytes, records[object_id].current_tier, sorted(windows),
        )
        for object_id in requested
    ]
    probabilities = model.predict_proba(pd.DataFrame(feature_rows)[feature_columns])
    class_index = list(model.classes_).index(1)
    return {object_id: float(probabilities[index][class_index]) for index, object_id in enumerate(requested)}