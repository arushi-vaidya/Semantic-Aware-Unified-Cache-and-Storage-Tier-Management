from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prediction.evaluator import evaluate_models


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="results/prediction")
parser.add_argument("--models-dir")
parser.add_argument("--output")
args = parser.parse_args()
print(evaluate_models(args.input, args.models_dir, args.output))