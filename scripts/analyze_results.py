import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("metrics_json")
parser.add_argument("--output-dir", default="results/plots")
args = parser.parse_args()
metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
summary = pd.DataFrame({"metric": ["cache_hit_ratio", "average_latency_ms", "p95_latency_ms", "migration_volume_bytes"], "value": [metrics["cache_hit_ratio"], metrics["average_latency_ms"], metrics["p95_latency_ms"], metrics["migration_volume_bytes"]]})
processed = output.parent / "processed"; processed.mkdir(parents=True, exist_ok=True); summary.to_csv(processed / "summary.csv", index=False)
for _, row in summary.iterrows():
    figure, axis = plt.subplots(); axis.bar([row["metric"]], [row["value"]]); axis.set_ylabel("measured value"); figure.tight_layout(); figure.savefig(output / f"{row['metric']}.png"); plt.close(figure)
tiers = pd.DataFrame([{"tier": tier, **values} for tier, values in metrics["storage"].items()])
figure, axis = plt.subplots(); axis.bar(tiers["tier"], tiers["used_bytes"]); axis.set_ylabel("used bytes"); figure.tight_layout(); figure.savefig(output / "storage_utilization.png"); plt.close(figure)