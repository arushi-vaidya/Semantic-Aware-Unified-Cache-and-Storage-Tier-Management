from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


parser = argparse.ArgumentParser(description="Compare measured baseline experiment metrics")
parser.add_argument("experiments", nargs="+", help="Experiment directories containing metrics.json")
parser.add_argument("--output", default="results/baseline_report.md")
args = parser.parse_args()

rows = []
for experiment_path in args.experiments:
    directory = Path(experiment_path)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    workload = json.loads((directory / "workload.json").read_text(encoding="utf-8"))
    experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    rows.append({
        "workload": experiment["type"],
        "experiment": str(directory),
        "operations": len(workload),
        "objects": len({operation["object_id"] for operation in workload}),
        "object_size_bytes": experiment["size_bytes"],
        "cache_hit_ratio": metrics["cache_hit_ratio"],
        "cache_hits": metrics["cache_hits"],
        "cache_misses": metrics["cache_misses"],
        "average_latency_ms": metrics["average_latency_ms"],
        "p95_latency_ms": metrics["p95_latency_ms"],
        "p99_latency_ms": metrics["p99_latency_ms"],
        "migration_count": metrics["migration_count"],
        "migration_volume_bytes": metrics["migration_volume_bytes"],
    })

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
csv_path = output.with_suffix(".csv")
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

columns = ["workload", "operations", "objects", "object_size_bytes", "cache_hit_ratio", "average_latency_ms", "p95_latency_ms", "p99_latency_ms", "migration_count", "migration_volume_bytes"]
lines = ["# Phase 1 Baseline Comparison", "", "Metrics are read from the recorded SQLite-derived experiment JSON files; no values are synthesized.", "", "| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
for row in rows:
    lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
lines.extend(["", f"CSV data: `{csv_path}`"])
output.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(output)
print(csv_path)