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
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "reset":
        for path in config["tiers"].values(): shutil.rmtree(path, ignore_errors=True)
        database = Path(config["metadata"]["database"]); database.unlink(missing_ok=True)
        print("Reset complete")
        return
    if args.command == "setup":
        _, _, metadata = build_system(args.config); metadata.close(); print("Storage directories and SQLite metadata initialized"); return
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