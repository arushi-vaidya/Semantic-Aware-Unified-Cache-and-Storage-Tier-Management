from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from src.prediction.feature_extractor import TARGET_COLUMN


def _metrics(target: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    roc_auc = roc_auc_score(target, probabilities) if len(np.unique(target)) == 2 else None
    pr_auc = average_precision_score(target, probabilities) if len(np.unique(target)) == 2 else None
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "confusion_matrix": confusion_matrix(target, predictions, labels=[0, 1]).tolist(),
    }


def _baseline_probabilities(test: pd.DataFrame, prediction_window: int) -> dict[str, np.ndarray]:
    frequency_column = f"accesses_in_last_{prediction_window}_operations"
    if frequency_column not in test.columns:
        frequency_column = "accesses_in_last_50_operations"
    frequency = np.clip(test[frequency_column].to_numpy(dtype=float) / prediction_window, 0.0, 1.0)
    recency = test["operations_since_last_access"].to_numpy(dtype=float)
    recency_score = np.exp(-recency / prediction_window)
    heuristic = np.clip(0.5 * frequency + 0.5 * recency_score, 0.0, 1.0)
    return {
        "Always-Negative": np.zeros(len(test), dtype=float),
        "Frequency Baseline": frequency,
        "Recency/Frequency Heuristic": heuristic,
    }


def _plot_curves(target: np.ndarray, probabilities: dict[str, np.ndarray], plots_dir: Path) -> None:
    if len(np.unique(target)) != 2:
        for filename, title in (("roc_curve.png", "ROC curve"), ("precision_recall_curve.png", "Precision-recall curve")):
            figure, axis = plt.subplots()
            axis.text(0.5, 0.5, "Undefined: test set contains one class", ha="center", va="center")
            axis.set_title(title)
            axis.set_axis_off()
            figure.savefig(plots_dir / filename, bbox_inches="tight")
            plt.close(figure)
        return
    figure, axis = plt.subplots()
    for name, scores in probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(target, scores)
        axis.plot(false_positive_rate, true_positive_rate, label=name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axis.legend()
    figure.savefig(plots_dir / "roc_curve.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots()
    for name, scores in probabilities.items():
        precision, recall, _ = precision_recall_curve(target, scores)
        axis.plot(recall, precision, label=name)
    axis.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axis.legend()
    figure.savefig(plots_dir / "precision_recall_curve.png", bbox_inches="tight")
    plt.close(figure)


def evaluate_models(
    dataset_dir: str | Path,
    models_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    dataset_path = Path(dataset_dir)
    if dataset_path.is_file():
        dataset_path = dataset_path.parent
    model_path = Path(models_dir) if models_dir is not None else dataset_path / "models"
    output = Path(output_dir) if output_dir is not None else dataset_path
    plots_dir = output / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(dataset_path / "test.csv")
    if test.empty:
        raise ValueError("test split is empty after chronological window purging")
    target = test[TARGET_COLUMN].astype(int).to_numpy()
    dataset_metadata = json.loads((dataset_path / "dataset_metadata.json").read_text(encoding="utf-8"))
    prediction_window = int(dataset_metadata["prediction_window"])
    probabilities = _baseline_probabilities(test, prediction_window)
    bundles = {}
    for path in sorted(model_path.glob("*.joblib")):
        bundle = joblib.load(path)
        name = "Logistic Regression" if bundle["model_type"] == "logistic_regression" else "Random Forest"
        bundles[name] = bundle
        model = bundle["model"]
        model_probabilities = model.predict_proba(test[bundle["feature_columns"]])
        positive_class = list(model.classes_).index(1)
        probabilities[name] = model_probabilities[:, positive_class]
    if not bundles:
        raise FileNotFoundError(f"no saved models found in {model_path}")

    metrics = {name: _metrics(target, scores) for name, scores in probabilities.items()}
    comparison = pd.DataFrame.from_dict(metrics, orient="index").drop(columns=["confusion_matrix"])
    comparison.index.name = "model"
    comparison.to_csv(output / "model_comparison.csv")
    prediction_frame = test[["timestamp", "operation_index", "object_id", TARGET_COLUMN]].copy()
    for name, scores in probabilities.items():
        prediction_frame[f"{name.lower().replace(' ', '_')}_probability"] = scores
    prediction_frame.to_csv(output / "test_predictions.csv", index=False)

    class_counts = {str(int(key)): int(value) for key, value in pd.Series(target).value_counts().sort_index().items()}
    positive = class_counts.get("1", 0)
    negative = class_counts.get("0", 0)
    class_distribution = {
        "test_samples": len(target),
        "positive_count": positive,
        "negative_count": negative,
        "positive_percentage": positive * 100 / len(target),
        "negative_percentage": negative * 100 / len(target),
    }
    report = {
        "class_distribution": class_distribution,
        "metrics": metrics,
        "feature_importance": None,
    }

    random_forest = bundles.get("Random Forest")
    if random_forest is not None:
        fitted = random_forest["model"]
        transformed_names = fitted.named_steps["preprocess"].get_feature_names_out()
        importances = fitted.named_steps["classifier"].feature_importances_
        importance_frame = pd.DataFrame({"feature": transformed_names, "importance": importances}).sort_values("importance", ascending=False)
        importance_frame.to_csv(output / "feature_importance.csv", index=False)
        report["feature_importance"] = importance_frame.to_dict(orient="records")
        figure, axis = plt.subplots(figsize=(9, max(3, min(10, len(importance_frame) * 0.3))))
        top = importance_frame.head(20).sort_values("importance")
        axis.barh(top["feature"], top["importance"], color="#247a74")
        axis.set(xlabel="Gini importance", title="Random Forest feature importance")
        figure.savefig(plots_dir / "feature_importance.png", bbox_inches="tight")
        plt.close(figure)

    distribution_figure, distribution_axis = plt.subplots()
    distribution_axis.bar(["No access", "Access"], [negative, positive], color=["#bd7255", "#247a74"])
    distribution_axis.set(ylabel="Test samples", title="Test-set class distribution")
    distribution_figure.savefig(plots_dir / "class_distribution.png", bbox_inches="tight")
    plt.close(distribution_figure)

    selected_name = "Random Forest" if "Random Forest" in metrics else next(iter(metrics))
    matrix = np.asarray(metrics[selected_name]["confusion_matrix"])
    figure, axis = plt.subplots()
    image = axis.imshow(matrix, cmap="Greens")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No access", "Access"], yticklabels=["No access", "Access"], xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix: {selected_name}")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.savefig(plots_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(figure)
    _plot_curves(target, probabilities, plots_dir)

    plotted_metrics = ["f1", "roc_auc"]
    figure, axes = plt.subplots(1, len(plotted_metrics), figsize=(10, 4))
    for axis, metric_name in zip(axes, plotted_metrics):
        names = list(metrics)
        values = [metrics[name][metric_name] if metrics[name][metric_name] is not None else np.nan for name in names]
        axis.barh(names, values, color="#397d9a")
        axis.set(title=f"Model comparison: {metric_name}", xlim=(0, 1))
    figure.tight_layout()
    figure.savefig(plots_dir / "model_comparison.png", bbox_inches="tight")
    plt.close(figure)

    available_auc = [(name, values["roc_auc"]) for name, values in metrics.items() if values["roc_auc"] is not None]
    if available_auc:
        best_model = max(available_auc, key=lambda item: (item[1], metrics[item[0]]["f1"]))[0]
    else:
        best_model = max(metrics, key=lambda name: metrics[name]["f1"])
    report["best_measured_model"] = best_model
    (output / "evaluation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    experiment_path = dataset_path / "experiment_metadata.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8")) if experiment_path.exists() else {}
    experiment["evaluation_metrics"] = metrics
    experiment["test_class_distribution"] = class_distribution
    experiment["best_measured_model"] = best_model
    experiment_path.write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    return report