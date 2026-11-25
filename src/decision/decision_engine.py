from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from src.decision.constraints import CapacitySnapshot, capacity_snapshot
from src.decision.migration_controller import MigrationController
from src.decision.scoring import DecisionWeights, ScoreBreakdown, TierProperties, score_tier
from src.metadata.metadata_store import MetadataStore
from src.prediction.predictor import predict_object
from src.storage.storage_manager import StorageManager
from src.storage.tier import Tier


@dataclass(frozen=True)
class PlacementDecision:
    object_id: str
    current_tier: Tier
    selected_tier: Tier
    access_probability: float
    semantic_importance: float
    utilities: dict[Tier, float | None]
    contributions: dict[Tier, ScoreBreakdown | None]
    decision_margin: float
    migration_required: bool
    reason: str
    decision_latency_ms: float


class DecisionEngine:
    def __init__(
        self,
        storage: StorageManager,
        metadata: MetadataStore,
        model_path: str | Path,
        weights: DecisionWeights,
        tier_properties: dict[Tier, TierProperties],
        migration_controller: MigrationController,
        hysteresis_threshold: float = 0.05,
        minimum_decision_margin: float = 0.05,
        migration_reference_bytes: int = 1_000_000,
    ):
        if set(tier_properties) != set(Tier):
            raise ValueError("properties must define CACHE, SSD, HDD, and ARCHIVE")
        if hysteresis_threshold < 0 or minimum_decision_margin < 0:
            raise ValueError("decision margins must be non-negative")
        if migration_reference_bytes <= 0:
            raise ValueError("migration_reference_bytes must be positive")
        self.storage = storage
        self.metadata = metadata
        self.model_path = Path(model_path)
        self.weights = weights
        self.tier_properties = tier_properties
        self.migration_controller = migration_controller
        self.hysteresis_threshold = hysteresis_threshold
        self.minimum_decision_margin = minimum_decision_margin
        self.migration_reference_bytes = migration_reference_bytes
        self.prediction_time_ms = 0.0
        self.decision_time_ms = 0.0
        self.migration_time_ms = 0.0
        self.decision_count = 0
        self.decision_latencies_ms: list[float] = []
        self.migration_failures = 0
        self.prediction_failures = 0
        self.no_feasible_tier = 0
        self.last_decisions: dict[str, PlacementDecision] = {}

    @staticmethod
    def _utility_reason(tier: Tier, breakdown: ScoreBreakdown) -> str:
        positive = {
            "predicted access": breakdown.access,
            "semantic importance": breakdown.semantic,
            "latency benefit": breakdown.latency,
        }
        driver = max(positive, key=positive.get)
        penalties = []
        if breakdown.migration < 0:
            penalties.append("migration cost")
        if breakdown.resource < 0:
            penalties.append("resource cost")
        penalty_detail = f" after accounting for {' and '.join(penalties)}" if penalties else ""
        return f"{driver} is the largest positive utility contribution for {tier.value}{penalty_detail}"

    def evaluate(
        self,
        object_id: str,
        operation_index: int,
        predicted_probability: float | None = None,
        prediction_latency_ms: float | None = None,
        snapshot: CapacitySnapshot | None = None,
    ) -> PlacementDecision:
        started = time.monotonic()
        prediction_was_batched = predicted_probability is not None
        record = self.metadata.get_object(object_id)
        current = Tier(record.current_tier)
        prediction_started = time.monotonic()
        if predicted_probability is None:
            try:
                probability = float(predict_object(self.model_path, self.metadata, object_id))
            except Exception:
                self.prediction_failures += 1
                raise
            finally:
                prediction_elapsed_ms = (time.monotonic() - prediction_started) * 1000
                self.prediction_time_ms += prediction_elapsed_ms
        else:
            probability = float(predicted_probability)
            prediction_elapsed_ms = float(prediction_latency_ms or 0.0)
            self.prediction_time_ms += prediction_elapsed_ms
        if not 0 <= probability <= 1:
            self.prediction_failures += 1
            raise ValueError(f"Phase 2 returned an invalid probability for {object_id}: {probability}")
        semantic_record = self.metadata.get_semantic_importance(object_id)
        semantic = semantic_record[0] if semantic_record is not None else 0.0
        snapshot = snapshot or capacity_snapshot(self.storage, self.tier_properties)
        feasible = [
            tier for tier in Tier
            if snapshot.feasible(tier, record.size_bytes, current)
        ]
        if not feasible:
            self.no_feasible_tier += 1
            raise RuntimeError(f"No feasible tier can hold {object_id} ({record.size_bytes} bytes)")

        score_started = time.monotonic()
        utilities: dict[Tier, float | None] = {tier: None for tier in Tier}
        breakdowns: dict[Tier, ScoreBreakdown | None] = {tier: None for tier in Tier}
        for candidate in feasible:
            properties = self.tier_properties[candidate]
            breakdown = score_tier(
                probability,
                semantic,
                record.size_bytes,
                current,
                candidate,
                self.tier_properties[current],
                properties,
                snapshot.projected_utilization(candidate, record.size_bytes, current),
                self.weights,
                self.migration_reference_bytes,
            )
            utilities[candidate] = breakdown.total
            breakdowns[candidate] = breakdown
        current_utility = utilities[current]
        if current_utility is None:
            properties = self.tier_properties[current]
            current_utility = score_tier(
                probability,
                semantic,
                record.size_bytes,
                current,
                current,
                properties,
                properties,
                1.0,
                self.weights,
                self.migration_reference_bytes,
            ).total
        ordered = sorted(feasible, key=lambda tier: (utilities[tier], tier == current), reverse=True)
        best = ordered[0]
        margin = float(utilities[best] - current_utility)
        selected = best
        reason = self._utility_reason(best, breakdowns[best])
        required_margin = max(self.hysteresis_threshold, self.minimum_decision_margin)
        if best != current and current in feasible and margin <= required_margin:
            selected = current
            reason = "current tier retained: utility margin is below hysteresis/confidence requirement"
        elif best != current and current_utility is None:
            reason = "current tier exceeds configured capacity; selected a feasible tier"

        if selected != current:
            blocked = None if current_utility is None else self.migration_controller.blocked_reason(object_id, operation_index)
            if blocked:
                selected = current
                reason = f"current tier retained: {blocked}"

        migration_required = selected != current
        decision_latency = (time.monotonic() - started) * 1000
        if prediction_was_batched:
            decision_latency += prediction_elapsed_ms
        self.decision_time_ms += max(0.0, decision_latency - prediction_elapsed_ms)
        decision = PlacementDecision(
            object_id, current, selected, probability, semantic, utilities, breakdowns,
            margin, migration_required, reason, decision_latency,
        )
        self.metadata.log_decision(
            object_id, current.value, selected.value, probability, semantic,
            {tier.value: value for tier, value in utilities.items()},
            {tier.value: None if value is None else value.__dict__ for tier, value in breakdowns.items()},
            migration_required, reason, margin, operation_index,
        )
        self.last_decisions[object_id] = decision
        self.decision_count += 1
        self.decision_latencies_ms.append(decision_latency)
        if migration_required:
            migration_started = time.monotonic()
            try:
                self.migration_controller.migrate(
                    object_id, current, selected, operation_index, force=current not in feasible
                )
                snapshot.used_bytes[current] -= record.size_bytes
                snapshot.used_bytes[selected] += record.size_bytes
            except Exception:
                self.migration_failures += 1
                raise
            finally:
                self.migration_time_ms += (time.monotonic() - migration_started) * 1000
        return decision

    def explain_decision(self, object_id: str) -> dict:
        decision = self.last_decisions.get(object_id)
        if decision is None:
            history = self.metadata.decision_history(object_id, 1)
            if not history:
                raise KeyError(f"No recorded decision for {object_id}")
            row = history[0]
            return {
                "object_id": object_id,
                "current_tier": row["current_tier"],
                "selected_tier": row["selected_tier"],
                "access_probability": row["access_probability"],
                "semantic_importance": row["semantic_importance"],
                "utilities": row["utilities"],
                "contributions": row["contributions"],
                "decision_margin": row["decision_margin"],
                "reason": row["migration_reason"],
            }
        return {
            "object_id": object_id,
            "current_tier": decision.current_tier.value,
            "selected_tier": decision.selected_tier.value,
            "access_probability": decision.access_probability,
            "semantic_importance": decision.semantic_importance,
            "utilities": {tier.value: score for tier, score in decision.utilities.items()},
            "contributions": {
                tier.value: None if value is None else value.__dict__
                for tier, value in decision.contributions.items()
            },
            "decision_margin": decision.decision_margin,
            "reason": decision.reason,
        }