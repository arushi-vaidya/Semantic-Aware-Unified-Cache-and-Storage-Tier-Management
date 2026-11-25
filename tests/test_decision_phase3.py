from __future__ import annotations

import random
import hashlib

import pytest

from src.decision.constraints import capacity_snapshot, feasible_tiers
from src.decision.decision_engine import DecisionEngine
from src.decision.migration_controller import MigrationController
from src.decision.placement import AdaptiveWorkloadRunner
from src.decision.scoring import DecisionWeights, TierProperties
from src.decision.tiering_policy import IndependentTieringPolicy
from src.metadata.metadata_store import MetadataStore
from src.prediction.dataset_builder import build_dataset
from src.prediction.trainer import train_model
from src.storage.storage_manager import StorageManager
from src.storage.tier import Tier


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase3-model")
    metadata = MetadataStore(root / "training.sqlite")
    object_ids = [f"train_{index}" for index in range(8)]
    for object_id in object_ids:
        metadata.create_object(object_id, str(root / object_id), 32, Tier.CACHE.value)
        metadata.update_access(object_id, "write", Tier.CACHE.value, 0.1, True)
    rng = random.Random(219)
    for _ in range(600):
        object_id = rng.choice(object_ids[:2] if rng.random() < 0.65 else object_ids)
        metadata.update_access(object_id, "read", Tier.CACHE.value, 0.1, True)
    metadata.close()
    dataset_dir = root / "dataset"
    build_dataset(root / "training.sqlite", dataset_dir, prediction_window=10)
    return train_model(dataset_dir, "logistic")


def make_storage(root, capacities=(128, 256, 512, None)):
    metadata = MetadataStore(root / "metadata.sqlite")
    paths = {tier.value.lower(): root / tier.value.lower() for tier in Tier}
    storage = StorageManager(paths, metadata)
    properties = {
        tier: TierProperties(latency, capacity, cost, 1.0)
        for tier, latency, capacity, cost in zip(
            Tier, (1.0, 0.7, 0.4, 0.1), capacities, (0.4, 0.3, 0.15, 0.05)
        )
    }
    return metadata, storage, properties


def test_capacity_rejects_full_tier_and_reports_feasible_candidates(tmp_path):
    metadata, storage, properties = make_storage(tmp_path, (10, 20, 30, None))
    storage.put_at_tier("existing", b"1234567890", Tier.CACHE)
    snapshot = capacity_snapshot(storage, properties)

    assert not snapshot.feasible(Tier.CACHE, 1, Tier.SSD)
    assert feasible_tiers(snapshot, 15, Tier.SSD) == [Tier.SSD, Tier.HDD, Tier.ARCHIVE]
    metadata.close()


def test_engine_uses_phase2_prediction_logs_and_migrates_real_file(trained_model, tmp_path):
    metadata, storage, properties = make_storage(tmp_path)
    storage.put_at_tier("target", b"actual payload", Tier.SSD)
    metadata.set_semantic_importance("target", 0.85, "critical")
    controller = MigrationController(storage, metadata)
    engine = DecisionEngine(
        storage, metadata, trained_model,
        DecisionWeights(access=0, semantic=0, latency=1, migration=0, resource=0),
        properties, controller, hysteresis_threshold=0, minimum_decision_margin=0,
    )

    decision = engine.evaluate("target", 10)

    assert decision.access_probability is not None
    assert 0 <= decision.access_probability <= 1
    assert decision.semantic_importance == pytest.approx(0.85)
    assert set(decision.utilities) == set(Tier)
    assert decision.selected_tier == Tier.CACHE
    assert storage.current_tier("target") == Tier.CACHE
    assert storage.get("target") == b"actual payload"
    assert storage._sha256(storage.path("target", Tier.CACHE)) == hashlib.sha256(b"actual payload").hexdigest()
    assert metadata.connection.execute("SELECT COUNT(*) FROM decision_log").fetchone()[0] == 1
    assert metadata.connection.execute("SELECT success FROM migration_log").fetchone()[0] == 1
    assert metadata.get_object("target").file_path == str(storage.path("target", Tier.CACHE))
    assert engine.explain_decision("target")["selected_tier"] == "CACHE"
    metadata.close()


def test_migration_controller_enforces_residency_and_cooldown(trained_model, tmp_path):
    metadata, storage, _ = make_storage(tmp_path)
    storage.put_at_tier("target", b"payload", Tier.CACHE)
    controller = MigrationController(storage, metadata, min_residency_operations=5, cooldown_operations=10)
    controller.register_placement("target", 0)

    assert controller.blocked_reason("target", 2) == "minimum residency period has not elapsed"
    controller.migrate("target", Tier.CACHE, Tier.SSD, 5)
    assert controller.blocked_reason("target", 10) == "migration cooldown is active"
    with pytest.raises(RuntimeError, match="migration blocked"):
        controller.migrate("target", Tier.SSD, Tier.CACHE, 10)
    assert controller.blocked_reason("target", 15) is None
    assert storage.current_tier("target") == Tier.SSD
    assert storage.get("target") == b"payload"
    metadata.close()


def test_isolated_online_workload_writes_measured_artifacts(trained_model, tmp_path):
    config = {
        "cache": {"capacity_bytes": 128},
        "tiers": {tier.value.lower(): str(tmp_path / tier.value.lower()) for tier in Tier},
        "metadata": {"database": str(tmp_path / "unused.sqlite")},
        "policy": {"ssd_capacity_bytes": 256, "hdd_capacity_bytes": 512},
        "decision": {
            "enabled": True,
            "policy": "unified",
            "interval_operations": 5,
            "weights": {"access": 0.35, "semantic": 0.25, "latency": 0.2, "migration": 0.1, "resource": 0.1},
            "hysteresis_threshold": 0,
            "minimum_decision_margin": 0,
            "min_residency_operations": 0,
            "migration_cooldown_operations": 0,
            "tiers": {},
        },
    }
    root = tmp_path / "run"
    metadata = MetadataStore(root / "metadata.sqlite")
    paths = {tier.value.lower(): root / "tiers" / tier.value.lower() for tier in Tier}
    storage = StorageManager(paths, metadata)
    runner = AdaptiveWorkloadRunner(config, storage, metadata, trained_model, root, decision_interval=5)

    result = runner.run("hot", 40, 42, object_count=4, object_size=16)

    assert result["metrics"]["operations"] == 40
    assert result["metrics"]["decision_count"] > 0
    assert result["metrics"]["prediction_failures"] == 0
    assert result["metrics"]["capacity_violations"] == {}
    assert (root / "decisions.csv").is_file()
    assert (root / "migrations.csv").is_file()
    assert (root / "placement_history.csv").is_file()
    assert sum(item["objects"] for item in result["metrics"]["placement_distribution"].values()) == 4
    metadata.close()


def test_independent_tiering_uses_frequency_and_age_without_prediction(tmp_path):
    policy = IndependentTieringPolicy(hot_access_count=3, hot_age_operations=5)
    assert policy.choose(10, 3, 2, list(Tier)).tier == Tier.CACHE
    assert policy.choose(1, 0, 600, list(Tier)).tier == Tier.ARCHIVE

    config = {
        "cache": {"capacity_bytes": 128},
        "tiers": {tier.value.lower(): str(tmp_path / tier.value.lower()) for tier in Tier},
        "metadata": {"database": str(tmp_path / "unused.sqlite")},
        "policy": {"ssd_capacity_bytes": 256, "hdd_capacity_bytes": 512},
        "decision": {
            "enabled": True,
            "policy": "independent-tiering",
            "interval_operations": 5,
            "min_residency_operations": 0,
            "migration_cooldown_operations": 0,
            "independent_tiering": {"hot_access_count": 3, "hot_age_operations": 5},
            "tiers": {},
        },
    }
    root = tmp_path / "independent-run"
    metadata = MetadataStore(root / "metadata.sqlite")
    paths = {tier.value.lower(): root / "tiers" / tier.value.lower() for tier in Tier}
    storage = StorageManager(paths, metadata)
    runner = AdaptiveWorkloadRunner(config, storage, metadata, root / "no-model.joblib", root)

    result = runner.run("hot", 20, 7, object_count=4, object_size=16)

    assert result["experiment"]["policy"] == "independent-tiering"
    assert result["metrics"]["time_spent_predicting_ms"] == 0
    assert result["metrics"]["decision_count"] > 0
    assert metadata.connection.execute(
        "SELECT COUNT(*) FROM decision_log WHERE access_probability IS NOT NULL OR semantic_importance IS NOT NULL"
    ).fetchone()[0] == 0
    metadata.close()


def test_lru_baseline_is_selectable_in_isolated_runner(tmp_path):
    config = {
        "cache": {"capacity_bytes": 32},
        "tiers": {tier.value.lower(): str(tmp_path / tier.value.lower()) for tier in Tier},
        "metadata": {"database": str(tmp_path / "unused.sqlite")},
        "policy": {"ssd_capacity_bytes": 64, "hdd_capacity_bytes": 128},
        "decision": {"enabled": True, "policy": "lru", "interval_operations": 5, "tiers": {}},
    }
    root = tmp_path / "lru-run"
    metadata = MetadataStore(root / "metadata.sqlite")
    paths = {tier.value.lower(): root / "tiers" / tier.value.lower() for tier in Tier}
    storage = StorageManager(paths, metadata)
    runner = AdaptiveWorkloadRunner(config, storage, metadata, root / "no-model.joblib", root)

    result = runner.run("hot", 20, 11, object_count=4, object_size=16)

    assert result["experiment"]["policy"] == "lru"
    assert result["metrics"]["decision_count"] > 0
    assert metadata.connection.execute(
        "SELECT COUNT(*) FROM decision_log WHERE access_probability IS NOT NULL"
    ).fetchone()[0] == 0
    assert result["metrics"]["cache_hits"] + result["metrics"]["cache_misses"] > 0
    metadata.close()