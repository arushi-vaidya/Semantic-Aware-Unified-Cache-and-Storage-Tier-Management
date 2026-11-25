from __future__ import annotations

from dataclasses import dataclass

from src.storage.tier import Tier


@dataclass(frozen=True)
class DecisionWeights:
    access: float
    semantic: float
    latency: float
    migration: float
    resource: float

    def __post_init__(self) -> None:
        if any(value < 0 for value in (self.access, self.semantic, self.latency, self.migration, self.resource)):
            raise ValueError("decision weights must be non-negative")


@dataclass(frozen=True)
class TierProperties:
    latency_weight: float
    capacity_bytes: int | None
    relative_cost: float = 0.0
    migration_cost: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.latency_weight <= 1:
            raise ValueError("latency_weight must be between 0 and 1")
        if self.capacity_bytes is not None and self.capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative or None")
        if not 0 <= self.relative_cost <= 1 or not 0 <= self.migration_cost <= 1:
            raise ValueError("relative and migration costs must be normalized between 0 and 1")


@dataclass(frozen=True)
class ScoreBreakdown:
    access: float
    semantic: float
    latency: float
    migration: float
    resource: float
    total: float


def score_tier(
    access_probability: float,
    semantic_importance: float,
    object_size_bytes: int,
    current_tier: Tier,
    candidate_tier: Tier,
    source_properties: TierProperties,
    candidate_properties: TierProperties,
    projected_utilization: float,
    weights: DecisionWeights,
    migration_reference_bytes: int = 1_000_000,
) -> ScoreBreakdown:
    if not 0 <= access_probability <= 1:
        raise ValueError("access_probability must be between 0 and 1")
    if not 0 <= semantic_importance <= 1:
        raise ValueError("semantic_importance must be between 0 and 1")
    if object_size_bytes < 0 or migration_reference_bytes <= 0:
        raise ValueError("object size must be non-negative and migration reference positive")
    if not 0 <= projected_utilization <= 1:
        raise ValueError("projected_utilization must be between 0 and 1")

    latency_benefit = candidate_properties.latency_weight
    access = weights.access * access_probability * latency_benefit
    semantic = weights.semantic * semantic_importance * latency_benefit
    latency = weights.latency * latency_benefit
    if current_tier == candidate_tier:
        migration_penalty = 0.0
    else:
        size_factor = min(1.0, object_size_bytes / migration_reference_bytes)
        route_cost = (source_properties.migration_cost + candidate_properties.migration_cost) / 2
        migration_penalty = weights.migration * size_factor * route_cost
    resource_cost = weights.resource * (candidate_properties.relative_cost + projected_utilization) / 2
    total = access + semantic + latency - migration_penalty - resource_cost
    return ScoreBreakdown(access, semantic, latency, -migration_penalty, -resource_cost, total)