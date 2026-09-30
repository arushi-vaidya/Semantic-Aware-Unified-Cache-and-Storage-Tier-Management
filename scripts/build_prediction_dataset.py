from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.prediction.dataset_builder import build_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/config.yaml")
parser.add_argument("--output", default="results/prediction")
parser.add_argument("--prediction-window", type=int, default=20)
parser.add_argument("--recent-windows", default="10,50,100")
parser.add_argument("--train-fraction", type=float, default=0.7)
parser.add_argument("--validation-fraction", type=float, default=0.15)
parser.add_argument("--workload", default="unspecified")
parser.add_argument("--seed", type=int)
args = parser.parse_args()
config = load_config(args.config)
windows = tuple(int(value) for value in args.recent_windows.split(",") if value.strip())
dataset = build_dataset(
    config["metadata"]["database"], args.output, args.prediction_window, windows,
    args.train_fraction, args.validation_fraction, workload=args.workload, workload_seed=args.seed,
)
print(f"Wrote {len(dataset)} prediction samples to {args.output}")