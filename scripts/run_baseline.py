from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import os
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import build_system
from src.config import load_config
from src.metrics.collector import MetricsCollector
from src.workload.generator import WorkloadGenerator
from src.storage.tier import Tier

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="hot", choices=["sequential", "hot", "uniform", "mixed"])
parser.add_argument("--operations", type=int, default=100)
parser.add_argument("--objects", type=int, default=20)
parser.add_argument("--size", type=int, default=1024)
parser.add_argument("--read-ratio", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--config", default="config/config.yaml")
args = parser.parse_args()
experiment = Path("results/raw") / datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
experiment.mkdir(parents=True, exist_ok=False)
config = load_config(args.config)
for path in config["tiers"].values():
    shutil.rmtree(path, ignore_errors=True)
Path(config["metadata"]["database"]).unlink(missing_ok=True)
cache, storage, metadata = build_system(args.config)
try:
    operations = WorkloadGenerator(args.seed).generate(args.type, args.operations, args.objects, args.read_ratio, args.size)
    executed = []
    for operation in operations:
        if operation.operation == "write": cache.put(operation.object_id, os.urandom(operation.size_bytes))
        else: cache.get(operation.object_id)
        executed.append(operation.__dict__)
    metrics = MetricsCollector(cache, storage, metadata).collect()
    (experiment / "workload.json").write_text(json.dumps(executed, indent=2), encoding="utf-8")
    (experiment / "experiment.json").write_text(json.dumps({"type": args.type, "operations": args.operations, "objects": args.objects, "size_bytes": args.size, "read_ratio": args.read_ratio, "seed": args.seed}, indent=2), encoding="utf-8")
    (experiment / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
finally:
    metadata.close()
print(json.dumps({"experiment": str(experiment), "metrics": metrics}, indent=2))