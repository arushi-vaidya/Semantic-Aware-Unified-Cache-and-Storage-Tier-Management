from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from src.cache.lru_cache import LRUCache
from src.config import load_config
from src.metadata.metadata_store import MetadataStore
from src.metrics.collector import MetricsCollector
from src.storage.storage_manager import StorageManager
from src.storage.tier import Tier
from src.workload.generator import WorkloadGenerator


def build_system(config_path: str = "config/config.yaml") -> tuple[LRUCache, StorageManager, MetadataStore]:
    config = load_config(config_path)
    metadata = MetadataStore(config["metadata"]["database"])
    storage = StorageManager(config["tiers"], metadata)
    cache = LRUCache(config["cache"]["capacity_bytes"], storage, metadata, config["policy"]["ssd_capacity_bytes"], config["policy"]["hdd_capacity_bytes"])
    return cache, storage, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 real storage/cache baseline")
    parser.add_argument("--config", default="config/config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup")
    put = sub.add_parser("put"); put.add_argument("--id", required=True); put.add_argument("--size", type=int, required=True)
    get = sub.add_parser("get"); get.add_argument("--id", required=True)
    sub.add_parser("status")
    workload = sub.add_parser("run-workload"); workload.add_argument("--type", choices=["sequential", "hot", "uniform", "mixed"], required=True); workload.add_argument("--operations", type=int, required=True); workload.add_argument("--seed", type=int); workload.add_argument("--objects", type=int, default=20); workload.add_argument("--size", type=int, default=1024); workload.add_argument("--read-ratio", type=float, default=0.8)
    sub.add_parser("metrics")
    migrate = sub.add_parser("migrate"); migrate.add_argument("--id", required=True); migrate.add_argument("--from", dest="source", required=True); migrate.add_argument("--to", dest="destination", required=True)
    sub.add_parser("reset")
    dataset = sub.add_parser("build-prediction-dataset")
    dataset.add_argument("--output", default="results/prediction")
    dataset.add_argument("--prediction-window", type=int, default=20)
    dataset.add_argument("--recent-windows", default="10,50,100")
    dataset.add_argument("--train-fraction", type=float, default=0.7)
    dataset.add_argument("--validation-fraction", type=float, default=0.15)
    dataset.add_argument("--workload", default="unspecified")
    dataset.add_argument("--seed", type=int)
    train = sub.add_parser("train-access-model")
    train.add_argument("--input", default="results/prediction")
    train.add_argument("--model", choices=["logistic", "random-forest"], default="logistic")
    train.add_argument("--models-dir")
    evaluate = sub.add_parser("evaluate-access-model")
    evaluate.add_argument("--input", default="results/prediction")
    evaluate.add_argument("--models-dir")
    evaluate.add_argument("--output")
    predict = sub.add_parser("predict-access")
    predict.add_argument("--object-id", required=True)
    predict.add_argument("--model", default="results/prediction/models/logistic_regression.joblib")
    semantic = sub.add_parser("set-semantic-importance")
    semantic.add_argument("--object-id", required=True)
    semantic.add_argument("--score", type=float, required=True)
    semantic.add_argument("--category")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "reset":
        for path in config["tiers"].values(): shutil.rmtree(path, ignore_errors=True)
        database = Path(config["metadata"]["database"]); database.unlink(missing_ok=True)
        print("Reset complete")
        return
    if args.command == "setup":
        _, _, metadata = build_system(args.config); metadata.close(); print("Storage directories and SQLite metadata initialized"); return
    if args.command == "build-prediction-dataset":
        from src.prediction.dataset_builder import build_dataset

        windows = tuple(int(value) for value in args.recent_windows.split(",") if value.strip())
        dataset = build_dataset(
            config["metadata"]["database"], args.output, args.prediction_window, windows,
            args.train_fraction, args.validation_fraction, workload=args.workload, workload_seed=args.seed,
        )
        print(json.dumps({"output": args.output, "samples": len(dataset)}, indent=2)); return
    if args.command == "train-access-model":
        from src.prediction.trainer import train_model

        model_path = train_model(args.input, args.model, args.models_dir)
        print(json.dumps({"model": str(model_path)}, indent=2)); return
    if args.command == "evaluate-access-model":
        from src.prediction.evaluator import evaluate_models

        report = evaluate_models(args.input, args.models_dir, args.output)
        print(json.dumps(report, indent=2)); return
    if args.command in {"predict-access", "set-semantic-importance"}:
        metadata = MetadataStore(config["metadata"]["database"])
        try:
            if args.command == "predict-access":
                from src.prediction.predictor import predict_object

                probability = predict_object(args.model, metadata, args.object_id)
                print(json.dumps({"object_id": args.object_id, "access_probability": probability}, indent=2))
            else:
                from src.semantic.importance import store_importance

                store_importance(metadata, args.object_id, args.score, args.category)
                print(json.dumps({"object_id": args.object_id, "semantic_importance": args.score, "category": args.category}, indent=2))
        finally:
            metadata.close()
        return
    cache, storage, metadata = build_system(args.config)
    try:
        if args.command == "put":
            if args.size < 0: raise ValueError("size must be non-negative")
            cache.put(args.id, os.urandom(args.size)); print(f"Stored {args.id} ({args.size} bytes)")
        elif args.command == "get":
            data = cache.get(args.id); print(f"Read {args.id} ({len(data)} bytes)")
        elif args.command == "status":
            print(json.dumps({tier.value: storage.list_objects(tier) for tier in Tier}, indent=2))
        elif args.command == "migrate":
            storage.move(args.id, Tier(args.source.upper()), Tier(args.destination.upper())); print(f"Migrated {args.id}")
        elif args.command == "run-workload":
            generator = WorkloadGenerator(args.seed)
            operations = generator.generate(args.type, args.operations, args.objects, args.read_ratio, args.size)
            for operation in operations:
                if operation.operation == "write": cache.put(operation.object_id, os.urandom(operation.size_bytes))
                else: cache.get(operation.object_id)
            print(json.dumps(MetricsCollector(cache, storage, metadata).collect(), indent=2))
        elif args.command == "metrics":
            print(json.dumps(MetricsCollector(cache, storage, metadata).collect(), indent=2))
    finally:
        metadata.close()


if __name__ == "__main__":
    main()