from src.prediction.feature_extractor import extract_features


def test_features_use_only_prior_accesses_and_calculate_recency():
    history = [
        {"operation": "write", "operation_index": 2, "timestamp": "2026-01-01T00:00:00+00:00"},
        {"operation": "read", "operation_index": 6, "timestamp": "2026-01-01T00:00:04+00:00"},
    ]
    features = extract_features(history, 10, "2026-01-01T00:00:08+00:00", 128, "SSD")
    assert features["total_access_count"] == 2
    assert features["operations_since_last_access"] == 4
    assert features["seconds_since_last_access"] == 4
    assert features["read_count"] == 1 and features["write_count"] == 1
    assert features["read_write_ratio"] == 1
    assert features["object_size_bytes"] == 128
    assert features["current_tier"] == "SSD"
    assert features["accesses_in_last_10_operations"] == 2


def test_cold_start_features_are_explicit():
    features = extract_features([], 5, "2026-01-01T00:00:00+00:00", None, None, cold_start_recency=999)
    assert features["total_access_count"] == 0
    assert features["operations_since_last_access"] == 999
    assert features["inter_access_missing"] == 1
    assert features["object_size_missing"] == 1
    assert features["current_tier"] == "UNKNOWN"