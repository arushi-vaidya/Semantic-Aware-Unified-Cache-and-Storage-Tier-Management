from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.prediction.dataset_builder import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.prediction.feature_extractor import IDENTIFIER_COLUMNS, TARGET_COLUMN
from src.prediction.models import build_model


def train_model(
    dataset_dir: str | Path,
    model_type: str,
    models_dir: str | Path | None = None,
) -> Path:
    dataset_path = Path(dataset_dir)
    if dataset_path.is_file():
        dataset_path = dataset_path.parent
    output_dir = Path(models_dir) if models_dir is not None else dataset_path / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(dataset_path / "train.csv")
    dataset_metadata = json.loads((dataset_path / "dataset_metadata.json").read_text(encoding="utf-8"))
    feature_columns = dataset_metadata["feature_columns"]
    numeric_features = [name for name in feature_columns if name in NUMERIC_FEATURES or name.startswith("accesses_in_last_")]
    categorical_features = [name for name in feature_columns if name in CATEGORICAL_FEATURES]
    if not numeric_features or not categorical_features:
        raise ValueError("training dataset is missing required numeric or tier features")
    if len(train) == 0 or train[TARGET_COLUMN].nunique() < 2:
        raise ValueError("training split must contain both target classes")
    model, hyperparameters = build_model(model_type, numeric_features, categorical_features)
    model.fit(train[feature_columns], train[TARGET_COLUMN].astype(int))

    normalized_type = "random_forest" if model_type in {"random-forest", "random_forest"} else "logistic_regression"
    model_path = output_dir / f"{normalized_type}.joblib"
    joblib.dump({"model": model, "model_type": normalized_type, "feature_columns": feature_columns}, model_path)
    schema = {
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "identifier_columns_excluded_from_model": IDENTIFIER_COLUMNS,
        "target_column": TARGET_COLUMN,
        "categorical_encoding": "one-hot with unknown categories ignored",
    }
    (output_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    experiment_path = dataset_path / "experiment_metadata.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8")) if experiment_path.exists() else {}
    trained_models = experiment.get("trained_models", {})
    trained_models[normalized_type] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": normalized_type,
        "model_hyperparameters": hyperparameters,
        "training_class_distribution": {str(int(key)): int(value) for key, value in train[TARGET_COLUMN].value_counts().sort_index().items()},
        "model_path": str(model_path),
    }
    experiment.update({
        "timestamp": experiment.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "workload": dataset_metadata.get("workload", "unspecified"),
        "workload_seed": dataset_metadata.get("workload_seed"),
        "prediction_window": dataset_metadata["prediction_window"],
        "feature_list": feature_columns,
        "split": dataset_metadata["split_fractions"],
        "split_policy": dataset_metadata["split_policy"],
        "model_type": normalized_type,
        "dataset_size": dataset_metadata["samples"],
        "split_samples": dataset_metadata["split_samples"],
        "class_distribution": dataset_metadata["class_distribution"],
        "model_path": str(model_path),
        "trained_models": trained_models,
    })
    experiment_path.write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    return model_path