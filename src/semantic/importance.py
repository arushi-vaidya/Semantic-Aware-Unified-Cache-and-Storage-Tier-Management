from __future__ import annotations

from src.metadata.metadata_store import MetadataStore


DEFAULT_CATEGORY_SCORES = {
    "transaction": 1.0,
    "active_user_data": 0.9,
    "configuration": 0.8,
    "normal_log": 0.3,
    "temporary_data": 0.1,
}


def importance_for_category(category: str, scores: dict[str, float] | None = None) -> float:
    configured = DEFAULT_CATEGORY_SCORES if scores is None else scores
    if category not in configured:
        raise KeyError(f"No semantic importance configured for category: {category}")
    score = float(configured[category])
    if not 0.0 <= score <= 1.0:
        raise ValueError("semantic importance score must be between 0 and 1")
    return score


def store_importance(
    metadata: MetadataStore,
    object_id: str,
    score: float,
    category: str | None = None,
) -> None:
    metadata.set_semantic_importance(object_id, score, category)