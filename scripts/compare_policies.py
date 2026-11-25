from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabulate measured metrics from completed policy runs")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/decision/policy_metrics.csv"))
    args = parser.parse_args()
    records = []
    for run_dir in args.runs:
        metadata = json.loads((run_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "decision_metrics.json").read_text(encoding="utf-8"))
        records.append({
            "run_dir": str(run_dir),
            "policy": metadata["policy"],
            "workload": metadata["workload"],
            "seed": metadata["seed"],
            "operations": metrics["operations"],
            "cache_hit_ratio": metrics["cache_hit_ratio"],
            "average_latency_ms": metrics["average_latency_ms"],
            "p95_latency_ms": metrics["p95_latency_ms"],
            "p99_latency_ms": metrics["p99_latency_ms"],
            "migration_count": metrics["migration_count"],
            "migration_volume_bytes": metrics["migration_volume_bytes"],
            "decision_count": metrics["decision_count"],
            "decision_overhead_percentage": metrics["decision_overhead_percentage"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"output": str(args.output), "runs": records}, indent=2))


if __name__ == "__main__":
    main()