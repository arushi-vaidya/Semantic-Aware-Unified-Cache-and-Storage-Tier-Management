from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from collections.abc import Sequence

from src.cache.lru_cache import LRUCache
from src.config import load_config
from src.decision.constraints import capacity_snapshot
from src.decision.decision_engine import DecisionEngine
from src.decision.migration_controller import MigrationController
from src.decision.scoring import DecisionWeights, TierProperties
from src.decision.tiering_policy import IndependentTieringPolicy
from src.metadata.metadata_store import MetadataStore
from src.prediction.predictor import predict_objects
from src.storage.storage_manager import StorageManager
from src.storage.tier import Tier
from src.workload.generator import Operation, WorkloadGenerator


def tier_properties_from_config(config: dict) -> dict[Tier, TierProperties]:
    decision = config.get("decision", {})
    configured = decision.get("tiers", {})
    capacities = {
        Tier.CACHE: config["cache"]["capacity_bytes"],
        Tier.SSD: config["policy"]["ssd_capacity_bytes"],
        Tier.HDD: config["policy"]["hdd_capacity_bytes"],
        Tier.ARCHIVE: config.get("policy", {}).get("archive_capacity_bytes"),
    }
    defaults = {
        Tier.CACHE: (1.0, 0.4),
        Tier.SSD: (0.75, 0.3),
        Tier.HDD: (0.4, 0.15),
        Tier.ARCHIVE: (0.1, 0.05),
    }
    result = {}
    for tier in Tier:
        values = configured.get(tier.value.lower(), {})
        result[tier] = TierProperties(
            latency_weight=float(values.get("latency_weight", defaults[tier][0])),
            capacity_bytes=values.get("capacity_bytes", capacities[tier]),
            relative_cost=float(values.get("relative_cost", defaults[tier][1])),
            migration_cost=float(values.get("migration_cost", 1.0)),
        )
    return result


class AdaptiveWorkloadRunner:
    def __init__(
        self,
        config: dict,
        storage: StorageManager,
        metadata: MetadataStore,
        model_path: str | Path,
        output_dir: str | Path,
        decision_interval: int | None = None,
        semantic_importance: dict[str, tuple[float, str | None]] | None = None,
    ):
        decision_config = config.get("decision", {})
        self.policy_name = decision_config.get("policy", "unified")
        if self.policy_name not in {"unified", "independent-tiering", "lru"}:
            raise ValueError(f"Unsupported decision policy: {self.policy_name}")
        if not decision_config.get("enabled", True):
            raise ValueError("decision.enabled must be true for adaptive execution")
        self.config = config
        self.storage = storage
        self.metadata = metadata
        self.model_path = Path(model_path)
        self.semantic_importance = semantic_importance or {}
        self.output_dir = Path(output_dir)
        self.interval = int(decision_interval or decision_config.get("interval_operations", 20))
        if self.interval <= 0:
            raise ValueError("decision interval must be positive")
        weights = DecisionWeights(**decision_config.get("weights", {
            "access": 0.35, "semantic": 0.25, "latency": 0.2, "migration": 0.1, "resource": 0.1,
        }))
        properties = tier_properties_from_config(config)
        controller = MigrationController(
            storage,
            metadata,
            int(decision_config.get("min_residency_operations", 40)),
            int(decision_config.get("migration_cooldown_operations", 40)),
        )
        self.controller = controller
        self.tiering_policy = None
        self.cache = None
        if self.policy_name == "unified":
            self.engine = DecisionEngine(
                storage,
                metadata,
                self.model_path,
                weights,
                properties,
                controller,
                float(decision_config.get("hysteresis_threshold", 0.05)),
                float(decision_config.get("minimum_decision_margin", 0.05)),
                int(decision_config.get("migration_reference_bytes", 1_000_000)),
            )
        elif self.policy_name == "independent-tiering":
            self.engine = None
            baseline = decision_config.get("independent_tiering", {})
            self.tiering_policy = IndependentTieringPolicy(
                int(baseline.get("hot_access_count", 10)),
                int(baseline.get("hot_age_operations", 20)),
                int(baseline.get("warm_access_count", 3)),
                int(baseline.get("archive_age_operations", 500)),
            )
        else:
            self.engine = None
            self.cache = LRUCache(
                int(config["cache"]["capacity_bytes"]), storage, metadata,
                int(config["policy"]["ssd_capacity_bytes"]),
                int(config["policy"]["hdd_capacity_bytes"]),
            )
        self.properties = properties
        self.latencies: list[float] = []
        self.payload_random = random.Random()
        self.cache_hits = 0
        self.cache_misses = 0
        self.placement_history: list[dict] = []
        self.decision_count = 0
        self.decision_latencies_ms: list[float] = []
        self.decision_time_ms = 0.0
        self.migration_time_ms = 0.0
        self.operations_executed: list[Operation] = []

    def _initial_tier(self, size_bytes: int) -> Tier:
        snapshot = capacity_snapshot(self.storage, self.properties)
        for tier in Tier:
            capacity = snapshot.capacity_bytes[tier]
            if capacity is None or snapshot.used_bytes[tier] + size_bytes <= capacity:
                return tier
        raise RuntimeError(f"No tier has capacity for a new {size_bytes}-byte object")

    def _execute(self, operation) -> None:
        started = time.monotonic()
        if self.cache is not None:
            if operation.operation == "write":
                self.cache.put(operation.object_id, self.payload_random.randbytes(operation.size_bytes))
            else:
                cache_hit = self.cache.contains(operation.object_id)
                self.cache.get(operation.object_id)
                self.cache_hits += int(cache_hit)
                self.cache_misses += int(not cache_hit)
            latency_ms = (time.monotonic() - started) * 1000
            row = self.metadata.connection.execute("SELECT id FROM access_log ORDER BY id DESC LIMIT 1").fetchone()
            self.metadata.connection.execute("UPDATE access_log SET latency_ms = ? WHERE id = ?", (latency_ms, row[0]))
            self.metadata.connection.commit()
            self.latencies.append(latency_ms)
            return
        if operation.operation == "write":
            data = self.payload_random.randbytes(operation.size_bytes)
            if self.storage.exists(operation.object_id):
                tier = self.storage.current_tier(operation.object_id)
                self.storage.write_existing(operation.object_id, data)
            else:
                tier = self._initial_tier(operation.size_bytes)
                self.storage.put_at_tier(operation.object_id, data, tier)
                importance = self.semantic_importance.get(operation.object_id)
                if importance is not None:
                    self.metadata.set_semantic_importance(operation.object_id, importance[0], importance[1])
            self.metadata.update_access(operation.object_id, "write", tier.value, 0.0, True)
        else:
            tier = self.storage.current_tier(operation.object_id)
            try:
                self.storage.get(operation.object_id, tier)
                self.cache_hits += int(tier == Tier.CACHE)
                self.cache_misses += int(tier != Tier.CACHE)
                self.metadata.update_access(operation.object_id, "read", tier.value, 0.0, True)
            except Exception as exc:
                self.metadata.update_access(operation.object_id, "read", tier.value, 0.0, False, str(exc))
                raise
        latency_ms = (time.monotonic() - started) * 1000
        row = self.metadata.connection.execute(
            "SELECT id FROM access_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.metadata.connection.execute("UPDATE access_log SET latency_ms = ? WHERE id = ?", (latency_ms, row[0]))
        self.metadata.connection.commit()
        self.latencies.append(latency_ms)

    def _evaluate_all(self) -> None:
        operation_index = int(self.metadata.connection.execute(
            "SELECT COALESCE(MAX(id), 0) FROM access_log"
        ).fetchone()[0])
        records = self.metadata.all_objects()
        snapshot = capacity_snapshot(self.storage, self.properties) if records and self.cache is None else None
        probabilities: dict[str, float] = {}
        average_prediction_latency = 0.0
        if self.engine is not None and records:
            prediction_started = time.monotonic()
            try:
                probabilities = predict_objects(
                    self.model_path, self.metadata, [record.object_id for record in records]
                )
            except Exception:
                self.engine.prediction_failures += 1
                raise
            average_prediction_latency = (time.monotonic() - prediction_started) * 1000 / len(records)
        for record in records:
            if self.engine is not None:
                decision = self.engine.evaluate(
                    record.object_id,
                    operation_index,
                    probabilities[record.object_id],
                    average_prediction_latency,
                    snapshot,
                )
                self.decision_count += 1
                self.decision_latencies_ms.append(decision.decision_latency_ms)
                continue
            if self.cache is not None:
                started = time.monotonic()
                current = Tier(record.current_tier)
                self.metadata.log_decision(
                    record.object_id, current.value, current.value, None, None,
                    {tier.value: None for tier in Tier}, {tier.value: None for tier in Tier},
                    False, "LRU baseline placement observed", 0.0, operation_index,
                )
                latency_ms = (time.monotonic() - started) * 1000
                self.decision_count += 1
                self.decision_latencies_ms.append(latency_ms)
                self.decision_time_ms += latency_ms
                continue
            started = time.monotonic()
            feasible = [tier for tier in Tier if snapshot.feasible(tier, record.size_bytes, Tier(record.current_tier))]
            last_access = self.metadata.connection.execute(
                "SELECT MAX(id) FROM access_log WHERE object_id = ? AND success = 1", (record.object_id,)
            ).fetchone()[0]
            window = int(self.config.get("decision", {}).get("independent_tiering", {}).get("recent_window_operations", 50))
            recent = int(self.metadata.connection.execute(
                "SELECT COUNT(*) FROM access_log WHERE object_id = ? AND success = 1 AND id > ?",
                (record.object_id, max(0, operation_index - window)),
            ).fetchone()[0])
            age = operation_index - int(last_access or 0)
            choice = self.tiering_policy.choose(record.access_count, recent, age, feasible)
            current = Tier(record.current_tier)
            selected = choice.tier
            reason = choice.reason
            margin = 0.0
            if selected != current and current in feasible:
                blocked = self.controller.blocked_reason(record.object_id, operation_index)
                if blocked:
                    selected = current
                    reason = f"current tier retained: {blocked}"
            latency_ms = (time.monotonic() - started) * 1000
            self.metadata.log_decision(
                record.object_id, current.value, selected.value, None, None,
                {tier.value: None for tier in Tier}, {tier.value: None for tier in Tier},
                selected != current, reason, margin, operation_index,
            )
            self.decision_count += 1
            self.decision_latencies_ms.append(latency_ms)
            self.decision_time_ms += latency_ms
            if selected != current:
                migration_started = time.monotonic()
                self.controller.migrate(
                    record.object_id, current, selected, operation_index, force=current not in feasible
                )
                snapshot.used_bytes[current] -= record.size_bytes
                snapshot.used_bytes[selected] += record.size_bytes
                self.migration_time_ms += (time.monotonic() - migration_started) * 1000
        self.placement_history.append({
            "operation_index": operation_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "distribution": self._placement_distribution(),
        })

    def _placement_distribution(self) -> dict[str, dict[str, int]]:
        result = {}
        for tier in Tier:
            object_ids = self.storage.list_objects(tier)
            result[tier.value] = {
                "objects": len(object_ids),
                "bytes": sum(self.storage.path(object_id, tier).stat().st_size for object_id in object_ids),
            }
        return result

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[max(0, min(len(ordered) - 1, int((percentile * len(ordered) + 0.999999) - 1)))]

    def _metrics(self, operation_count: int, elapsed_seconds: float) -> dict:
        migrations = self.metadata.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(bytes_moved), 0) FROM migration_log WHERE success = 1"
        ).fetchone()
        migration_diagnostics = self.metadata.connection.execute(
            "SELECT COALESCE(SUM(duration_ms), 0), "
            "SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN success = 0 AND error LIKE '%integrity%' THEN 1 ELSE 0 END) "
            "FROM migration_log"
        ).fetchone()
        used = {
            tier: sum(self.storage.path(object_id, tier).stat().st_size for object_id in self.storage.list_objects(tier))
            for tier in Tier
        }
        violations = {
            tier.value: used[tier] - self.properties[tier].capacity_bytes
            for tier in Tier
            if self.properties[tier].capacity_bytes is not None
            and used[tier] > self.properties[tier].capacity_bytes
        }
        migrations_by_object = self.metadata.connection.execute(
            "SELECT object_id, timestamp FROM migration_log WHERE success = 1 ORDER BY id"
        ).fetchall()
        recent: dict[str, int] = {}
        thrashing = 0
        for row in migrations_by_object:
            index = int(self.metadata.connection.execute(
                "SELECT COALESCE(MAX(id), 0) FROM access_log WHERE timestamp <= ?", (row["timestamp"],)
            ).fetchone()[0])
            if row["object_id"] in recent and index - recent[row["object_id"]] < int(
                self.config.get("decision", {}).get("migration_cooldown_operations", 40)
            ):
                thrashing += 1
            recent[row["object_id"]] = index
        total_reads = self.cache_hits + self.cache_misses
        return {
            "operations": operation_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": self.cache_hits / total_reads if total_reads else 0.0,
            "average_latency_ms": statistics.fmean(self.latencies) if self.latencies else 0.0,
            "p95_latency_ms": self._percentile(self.latencies, 0.95),
            "p99_latency_ms": self._percentile(self.latencies, 0.99),
            "migration_count": int(migrations[0]),
            "migrations_per_operation": int(migrations[0]) / operation_count if operation_count else 0.0,
            "migration_volume_bytes": int(migrations[1]),
            "decision_count": self.decision_count,
            "average_decision_latency_ms": (
                statistics.fmean(self.decision_latencies_ms) if self.decision_latencies_ms else 0.0
            ),
            "time_spent_predicting_ms": self.engine.prediction_time_ms if self.engine is not None else 0.0,
            "time_spent_deciding_ms": self.engine.decision_time_ms if self.engine is not None else self.decision_time_ms,
            "time_spent_migrating_ms": float(migration_diagnostics[0] or 0.0),
            "decision_overhead_percentage": (
                ((self.engine.prediction_time_ms + self.engine.decision_time_ms) if self.engine is not None
                 else self.decision_time_ms) / (elapsed_seconds * 1000) * 100
                if elapsed_seconds else 0.0
            ),
            "placement_distribution": self._placement_distribution(),
            "capacity_violations": violations,
            "thrashing_count": thrashing,
            "prediction_failures": self.engine.prediction_failures if self.engine is not None else 0,
            "decision_failures": self.engine.no_feasible_tier if self.engine is not None else 0,
            "migration_failures": self.engine.migration_failures if self.engine is not None else 0,
            "recorded_migration_failures": int(migration_diagnostics[1] or 0),
            "integrity_failures": int(migration_diagnostics[2] or 0),
        }

    def _write_artifacts(self, metrics: dict, experiment: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        decisions = self.metadata.connection.execute("SELECT * FROM decision_log ORDER BY id").fetchall()
        decision_path = self.output_dir / "decisions.csv"
        fields = [
            "timestamp", "object_id", "current_tier", "selected_tier", "access_probability",
            "semantic_importance", *(f"utility_{tier.value.lower()}" for tier in Tier),
            "migration_required", "migration_reason", "decision_margin", "operation_index",
        ]
        with decision_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in decisions:
                utilities = json.loads(row["utilities_json"])
                writer.writerow({
                    **{field: row[field] for field in fields if field in row.keys()},
                    **{f"utility_{tier.value.lower()}": utilities.get(tier.value) for tier in Tier},
                })
        (self.output_dir / "workload.json").write_text(
            json.dumps([operation.__dict__ for operation in self.operations_executed], indent=2),
            encoding="utf-8",
        )
        with (self.output_dir / "migrations.csv").open("w", newline="", encoding="utf-8") as handle:
            rows = self.metadata.connection.execute("SELECT * FROM migration_log ORDER BY id").fetchall()
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
                "id", "object_id", "source_tier", "destination_tier", "timestamp", "bytes_moved", "duration_ms", "success", "error"
            ])
            writer.writeheader()
            writer.writerows(dict(row) for row in rows)
        with (self.output_dir / "placement_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["operation_index", "timestamp", "tier", "objects", "bytes"])
            writer.writeheader()
            for sample in self.placement_history:
                for tier, distribution in sample["distribution"].items():
                    writer.writerow({"operation_index": sample["operation_index"], "timestamp": sample["timestamp"],
                                     "tier": tier, **distribution})
        (self.output_dir / "decision_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (self.output_dir / "experiment_metadata.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")

    def run(
        self,
        workload_type: str,
        operation_count: int,
        seed: int,
        object_count: int = 20,
        object_size: int = 1024,
        read_ratio: float = 0.8,
        trace_operations: Sequence[Operation] | None = None,
    ) -> dict:
        if self.policy_name == "unified" and not self.model_path.is_file():
            raise FileNotFoundError(f"trained Phase 2 model does not exist: {self.model_path}")
        operations = list(trace_operations) if trace_operations is not None else WorkloadGenerator(seed).generate(
            workload_type, operation_count, object_count, read_ratio, object_size
        )
        operation_count = len(operations)
        self.payload_random.seed(seed)
        if any(operation.operation not in {"read", "write"} for operation in operations):
            raise ValueError("trace operations must be read or write")
        self.operations_executed = list(operations)
        run_started = time.monotonic()
        for index, operation in enumerate(operations, 1):
            self._execute(operation)
            if index % self.interval == 0:
                self._evaluate_all()
        if operation_count and operation_count % self.interval:
            self._evaluate_all()
        elapsed = time.monotonic() - run_started
        model_digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest() if self.model_path.is_file() else None
        experiment = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workload": workload_type,
            "seed": seed,
            "operations": operation_count,
            "objects": object_count,
            "object_size_bytes": object_size,
            "read_ratio": read_ratio,
            "decision_interval_operations": self.interval,
            "policy": self.policy_name,
            "model_path": str(self.model_path.resolve()) if self.model_path.is_file() else None,
            "model_sha256": model_digest,
            "semantic_metadata_source": self.config.get("_semantic_metadata_source"),
            "trace_source": self.config.get("_trace_source"),
            "elapsed_seconds": elapsed,
            "decision_config": self.config.get("decision", {}),
            "tier_properties": {
                tier.value: {
                    "latency_weight": properties.latency_weight,
                    "capacity_bytes": properties.capacity_bytes,
                    "relative_cost": properties.relative_cost,
                    "migration_cost": properties.migration_cost,
                }
                for tier, properties in self.properties.items()
            },
        }
        metrics = self._metrics(operation_count, elapsed)
        self._write_artifacts(metrics, experiment)
        return {"experiment": experiment, "metrics": metrics, "output_dir": str(self.output_dir)}


def run_isolated_adaptive_workload(
    config_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    workload_type: str,
    operation_count: int,
    seed: int,
    object_count: int = 20,
    object_size: int = 1024,
    read_ratio: float = 0.8,
    decision_interval: int | None = None,
    trace_path: str | Path | None = None,
) -> dict:
    config = load_config(config_path)
    source_database = Path(config["metadata"]["database"])
    semantic_importance: dict[str, tuple[float, str | None]] = {}
    if source_database.is_file():
        connection = sqlite3.connect(f"file:{source_database.resolve()}?mode=ro", uri=True)
        try:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'semantic_importance'"
            ).fetchone()
            if table:
                semantic_importance = {
                    row[0]: (float(row[1]), row[2])
                    for row in connection.execute("SELECT object_id, score, category FROM semantic_importance")
                }
        finally:
            connection.close()
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    tier_paths = {tier.value.lower(): root / "tiers" / tier.value.lower() for tier in Tier}
    metadata = MetadataStore(root / "metadata.sqlite")
    storage = StorageManager(tier_paths, metadata)
    config["_semantic_metadata_source"] = str(source_database.resolve()) if source_database.is_file() else None
    runner = AdaptiveWorkloadRunner(
        config, storage, metadata, model_path, root, decision_interval, semantic_importance
    )
    try:
        trace_operations = None
        if trace_path is not None:
            config["_trace_source"] = str(Path(trace_path).resolve())
            raw_operations = json.loads(Path(trace_path).read_text(encoding="utf-8"))
            trace_operations = [
                Operation(str(item["operation"]), str(item["object_id"]), int(item.get("size_bytes", 0)))
                for item in raw_operations
            ]
        result = runner.run(
            workload_type, operation_count, seed, object_count, object_size, read_ratio, trace_operations
        )
    finally:
        metadata.close()
    return result