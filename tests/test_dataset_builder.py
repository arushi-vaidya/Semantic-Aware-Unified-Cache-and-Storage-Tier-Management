import random

import pandas as pd

from src.metadata.metadata_store import MetadataStore
from src.prediction.dataset_builder import build_dataset, chronological_split


def test_targets_come_from_future_events_and_splits_are_chronological(tmp_path):
    metadata = MetadataStore(tmp_path / "access.db")
    for object_id in ("a", "b", "c"):
        metadata.create_object(object_id, str(tmp_path / object_id), 8, "CACHE")
    operations = ["a", "b", "c", "a", "b", "c"] * 20
    for object_id in operations:
        metadata.update_access(object_id, "read", "CACHE", 0.1, True)
    metadata.close()

    output = tmp_path / "prediction"
    dataset = build_dataset(tmp_path / "access.db", output, prediction_window=3)
    row = dataset[(dataset["object_id"] == "a") & (dataset["operation_index"] == 1)].iloc[0]
    assert row["total_access_count"] == 1
    assert row["target"] == 1
    assert dataset["operation_index"].min() < dataset["operation_index"].max()
    for split_name in ("train", "validation", "test"):
        assert (output / f"{split_name}.csv").is_file()
    assert dataset["object_id"].notna().all()


def test_chronological_split_purges_windows_at_boundaries():
    dataset = pd.DataFrame({"operation_index": list(range(100)), "target": [index % 2 for index in range(100)]})
    splits = chronological_split(dataset, prediction_window=10)
    assert splits["train"]["operation_index"].max() < 60
    assert splits["validation"]["operation_index"].min() >= 70
    assert splits["validation"]["operation_index"].max() < 75
    assert splits["test"]["operation_index"].min() >= 85