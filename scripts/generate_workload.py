import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.workload.generator import WorkloadGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--type", choices=["sequential", "hot", "uniform", "mixed"], required=True)
parser.add_argument("--operations", type=int, required=True)
parser.add_argument("--seed", type=int)
parser.add_argument("--output", required=True)
args = parser.parse_args()
operations = WorkloadGenerator(args.seed).generate(args.type, args.operations)
with open(args.output, "w", encoding="utf-8") as handle:
    json.dump([operation.__dict__ for operation in operations], handle, indent=2)