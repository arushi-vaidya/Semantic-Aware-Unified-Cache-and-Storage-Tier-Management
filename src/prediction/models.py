from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_model(model_type: str, numeric_features: list[str], categorical_features: list[str]) -> tuple[Pipeline, dict]:
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )
    if model_type == "logistic":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        parameters = {"max_iter": 1000, "class_weight": "balanced", "random_state": 42}
    elif model_type in {"random-forest", "random_forest"}:
        estimator = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        parameters = {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": 42,
        }
    else:
        raise ValueError(f"unsupported model type: {model_type}")
    return Pipeline([("preprocess", preprocess), ("classifier", estimator)]), parameters