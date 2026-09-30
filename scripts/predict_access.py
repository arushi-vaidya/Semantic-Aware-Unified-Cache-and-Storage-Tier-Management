from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.metadata.metadata_store import MetadataStore
from src.prediction.predictor import predict_object


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/config.yaml")
parser.add_argument("--object-id", required=True)
parser.add_argument("--model", default="results/prediction/models/logistic_regression.joblib")
args = parser.parse_args()
config = load_config(args.config)
metadata = MetadataStore(config["metadata"]["database"])
try:
    probability = predict_object(args.model, metadata, args.object_id)
    print(json.dumps({"object_id": args.object_id, "access_probability": probability}, indent=2))
finally:
    metadata.close()