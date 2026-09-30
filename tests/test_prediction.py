import random

from src.metadata.metadata_store import MetadataStore
from src.prediction.dataset_builder import build_dataset
from src.prediction.predictor import predict_object
from src.prediction.trainer import train_model


def test_logistic_and_random_forest_train_save_reload_and_predict(tmp_path):
    metadata = MetadataStore(tmp_path / "access.db")
    object_ids = [f"item_{index}" for index in range(10)]
    for object_id in object_ids:
        metadata.create_object(object_id, str(tmp_path / object_id), 16, "CACHE")
        metadata.update_access(object_id, "write", "CACHE", 0.1, True)
    generator = random.Random(17)
    for _ in range(500):
        object_id = generator.choice(object_ids[:2] if generator.random() < 0.65 else object_ids)
        metadata.update_access(object_id, "read", "CACHE", 0.1, True)
    metadata.close()

    prediction_dir = tmp_path / "prediction"
    build_dataset(tmp_path / "access.db", prediction_dir, prediction_window=10)
    logistic_path = train_model(prediction_dir, "logistic")
    forest_path = train_model(prediction_dir, "random-forest")
    metadata = MetadataStore(tmp_path / "access.db")
    try:
        probability = predict_object(logistic_path, metadata, "item_0")
        metadata.create_object("cold_item", str(tmp_path / "cold_item"), 32, "SSD")
        cold_start_probability = predict_object(logistic_path, metadata, "cold_item")
    finally:
        metadata.close()
    assert 0 <= probability <= 1
    assert 0 <= cold_start_probability <= 1
    assert forest_path.is_file()
    assert logistic_path.is_file()