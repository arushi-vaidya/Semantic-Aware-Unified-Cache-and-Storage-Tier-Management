from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.tier import Tier


def evaluate_run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    connection = sqlite3.connect(f"file:{(run_dir / 'metadata.sqlite').resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT object_id, current_tier, file_path, size_bytes FROM objects").fetchall()
    finally:
        connection.close()
    distribution = {}
    errors = []
    seen = set()
    for tier in Tier:
        tier_path = run_dir / "tiers" / tier.value.lower()
        files = {path.name: path for path in tier_path.iterdir() if path.is_file() and not path.name.startswith(".")}
        distribution[tier.value] = {
            "objects": len(files),
            "bytes": sum(path.stat().st_size for path in files.values()),
        }
        for object_id in files:
            if object_id in seen:
                errors.append(f"duplicate physical object: {object_id}")
            seen.add(object_id)
    for row in rows:
        expected = run_dir / "tiers" / row["current_tier"].lower() / row["object_id"]
        if not expected.is_file():
            errors.append(f"metadata points to missing physical object: {row['object_id']}")
        elif expected.stat().st_size != row["size_bytes"]:
            errors.append(f"metadata size mismatch: {row['object_id']}")
        elif Path(row["file_path"]).resolve() != expected.resolve():
            errors.append(f"metadata path mismatch: {row['object_id']}")
    extras = seen - {row["object_id"] for row in rows}
    errors.extend(f"untracked physical object: {object_id}" for object_id in sorted(extras))
    result = {"run_dir": str(run_dir), "physical_distribution": distribution, "object_count": len(rows),
              "metadata_physical_consistent": not errors, "consistency_errors": errors}
    metrics_path = run_dir / "decision_metrics.json"
    if metrics_path.is_file():
        result["recorded_metrics_distribution"] = json.loads(metrics_path.read_text(encoding="utf-8"))["placement_distribution"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit actual placement against run-local SQLite metadata")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = evaluate_run(args.run_dir)
    print(json.dumps(result, indent=2))
    if not result["metadata_physical_consistent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()