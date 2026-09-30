from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prediction.trainer import train_model


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="results/prediction")
parser.add_argument("--model", choices=["logistic", "random-forest"], default="logistic")
parser.add_argument("--models-dir")
args = parser.parse_args()
print(train_model(args.input, args.model, args.models_dir))