from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.metadata.metadata_store import MetadataStore
from src.prediction.feature_extractor import DEFAULT_RECENT_WINDOWS, extract_features


def predict_object(model_path: str | Path, metadata: MetadataStore, object_id: str) -> float:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    record = metadata.get_object(object_id)
    rows = [dict(row) for row in metadata.connection.execute("SELECT * FROM access_log ORDER BY id")]
    for index, row in enumerate(rows):
        row["operation_index"] = index
    history = [
        row for row in rows
        if row["object_id"] == object_id and row["success"] and row["operation"] in {"read", "write"}
    ]
    current_size = record.size_bytes
    current_tier = record.current_tier
    windows = set(DEFAULT_RECENT_WINDOWS)
    for name in feature_columns:
        match = re.fullmatch(r"accesses_in_last_(\d+)_operations", name)
        if match:
            windows.add(int(match.group(1)))
    feature_row = extract_features(
        history,
        len(rows),
        datetime.now(timezone.utc).isoformat(),
        current_size if current_size is not None else record.size_bytes,
        current_tier or record.current_tier,
        sorted(windows),
    )
    probabilities = model.predict_proba(pd.DataFrame([feature_row])[feature_columns])[0]
    class_index = list(model.classes_).index(1)
    return float(probabilities[class_index])