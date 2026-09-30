import random

import pytest

from src.metadata.metadata_store import MetadataStore
from src.prediction.dataset_builder import build_dataset
from src.prediction.evaluator import evaluate_models
from src.prediction.trainer import train_model
from src.semantic.importance import importance_for_category, store_importance


def test_metrics_plots_and_semantic_metadata_persist(tmp_path):
    database = tmp_path / "access.db"
    metadata = MetadataStore(database)
    object_ids = [f"item_{index}" for index in range(10)]
    for object_id in object_ids:
        metadata.create_object(object_id, str(tmp_path / object_id), 16, "CACHE")
        metadata.update_access(object_id, "write", "CACHE", 0.1, True)
    generator = random.Random(23)
    for _ in range(450):
        object_id = generator.choice(object_ids[:2] if generator.random() < 0.65 else object_ids)
        metadata.update_access(object_id, "read", "CACHE", 0.1, True)
    store_importance(metadata, "item_0", importance_for_category("transaction"), "transaction")
    assert metadata.get_semantic_importance("item_0") == (1.0, "transaction")
    with pytest.raises(ValueError):
        store_importance(metadata, "item_0", 1.1, "invalid")
    metadata.close()

    prediction_dir = tmp_path / "prediction"
    build_dataset(database, prediction_dir, prediction_window=10)
    train_model(prediction_dir, "logistic")
    train_model(prediction_dir, "random-forest")
    report = evaluate_models(prediction_dir)
    assert report["class_distribution"]["test_samples"] > 0
    assert report["metrics"]["Always-Negative"]["confusion_matrix"]
    assert report["feature_importance"]
    for filename in ("class_distribution.png", "confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png", "feature_importance.png", "model_comparison.png"):
        assert (prediction_dir / "plots" / filename).is_file()