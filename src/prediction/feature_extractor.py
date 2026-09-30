from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from statistics import fmean, median


DEFAULT_RECENT_WINDOWS = (10, 50, 100)
NUMERIC_FEATURES = [
    "operations_since_last_access",
    "seconds_since_last_access",
    "total_access_count",
    *(f"accesses_in_last_{window}_operations" for window in DEFAULT_RECENT_WINDOWS),
    "read_count",
    "write_count",
    "read_write_ratio",
    "mean_inter_access_time",
    "median_inter_access_time",
    "inter_access_missing",
    "object_size_bytes",
    "object_size_missing",
]
CATEGORICAL_FEATURES = ["current_tier"]
IDENTIFIER_COLUMNS = ["timestamp", "operation_index", "object_id"]
TARGET_COLUMN = "target"


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def extract_features(
    history: Sequence[Mapping],
    prediction_position: int,
    prediction_timestamp: str,
    object_size_bytes: int | None,
    current_tier: str | None,
    recent_windows: Sequence[int] = DEFAULT_RECENT_WINDOWS,
    cold_start_recency: float = 1_000_000.0,
) -> dict[str, int | float | str]:
    """Build features only from successful object events before prediction_position."""
    if prediction_position < 0 or cold_start_recency <= 0:
        raise ValueError("prediction_position must be non-negative and cold_start_recency positive")
    if any(window <= 0 for window in recent_windows):
        raise ValueError("recent windows must be positive")

    events = list(history)
    reads = sum(event["operation"] == "read" for event in events)
    writes = sum(event["operation"] == "write" for event in events)
    last_event = events[-1] if events else None
    operations_since = prediction_position - int(last_event["operation_index"]) if last_event else cold_start_recency

    intervals: list[float] = []
    parsed_times = [_parse_timestamp(event["timestamp"]) for event in events]
    for earlier, later in zip(parsed_times, parsed_times[1:]):
        if earlier is not None and later is not None:
            intervals.append(max(0.0, (later - earlier).total_seconds()))
    prediction_time = _parse_timestamp(prediction_timestamp)
    last_time = parsed_times[-1] if parsed_times else None
    seconds_since = (
        max(0.0, (prediction_time - last_time).total_seconds())
        if prediction_time is not None and last_time is not None
        else 0.0
    )

    features: dict[str, int | float | str] = {
        "operations_since_last_access": float(operations_since),
        "seconds_since_last_access": seconds_since,
        "total_access_count": len(events),
        "read_count": reads,
        "write_count": writes,
        "read_write_ratio": reads / max(1, writes),
        "mean_inter_access_time": fmean(intervals) if intervals else 0.0,
        "median_inter_access_time": median(intervals) if intervals else 0.0,
        "inter_access_missing": int(len(intervals) < 1),
        "object_size_bytes": int(object_size_bytes) if object_size_bytes is not None else 0,
        "object_size_missing": int(object_size_bytes is None),
        "current_tier": current_tier or "UNKNOWN",
    }
    for window in recent_windows:
        features[f"accesses_in_last_{window}_operations"] = sum(
            int(event["operation_index"]) >= prediction_position - window for event in events
        )
    return features