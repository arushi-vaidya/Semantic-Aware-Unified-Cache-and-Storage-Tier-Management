# Semantic-Aware Unified Cache and Storage Tier Management

## Phase 1 baseline

This repository implements only the executable Phase 1 baseline: a real LRU cache backed by persistent filesystem tiers and SQLite metadata. New data enters `CACHE`; deterministic capacity enforcement migrates it through `SSD`, `HDD`, and `ARCHIVE`.

In Phase 1, physical storage tiers are represented by separate configurable filesystem paths. The system performs real file I/O and migration between these paths. Physical SSD/HDD benchmarking can be configured later when separate devices are available.

No ML, semantic classification, access prediction, or unified optimization engine is included.

## Installation

Python 3.11+ is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Configuration

`config/config.yaml` defines cache capacity, four tier paths, the SQLite database path, SSD/HDD capacity limits, and the default workload seed. Relative paths are resolved from the repository root. Archive capacity is limited by the filesystem rather than an artificial configured limit.

## Running

```bash
python -m src.cli setup
python -m src.cli put --id object_001 --size 1024
python -m src.cli get --id object_001
python -m src.cli status
python -m src.cli run-workload --type hot --operations 100 --seed 42
python -m src.cli metrics
python -m src.cli migrate --id object_001 --from SSD --to HDD
python -m src.cli reset
```

All files, accesses, migrations, checksums, and latencies are real local operations. Object IDs are validated to prevent path traversal. Migrations copy to a temporary destination, fsync and verify size plus SHA-256, then remove the source and update metadata.

## Experiments and analysis

```bash
python scripts/run_baseline.py --type hot --operations 1000 --seed 42
python scripts/analyze_results.py results/raw/<experiment>/metrics.json
```

Experiments use timestamped directories and save the executed workload and measured metrics. Analysis uses pandas and matplotlib to create CSV summaries and plots under `results/processed` and `results/plots`.

For a larger cache-locality comparison, the runner accepts `--objects`, `--size`, and `--read-ratio`:

```bash
python3.11 scripts/run_baseline.py --type sequential --operations 5000 --objects 200 --size 65536 --seed 42
python3.11 scripts/run_baseline.py --type uniform --operations 5000 --objects 200 --size 65536 --seed 42
python3.11 scripts/run_baseline.py --type hot --operations 5000 --objects 200 --size 65536 --seed 42
python3.11 scripts/report_baselines.py results/raw/<sequential-dir> results/raw/<uniform-dir> results/raw/<hot-dir>
```

The report command writes `results/baseline_report.md` and `results/baseline_report.csv` from the recorded experiment artifacts.

## Metrics

The collector calculates actual cache hit/miss ratios, mean/P95/P99 successful access latency, successful migration count and bytes, per-tier used bytes and configured-capacity utilization, and process RSS when `psutil` is installed. Empty samples are reported as zero or null, never fabricated.

## Tests and reproducibility

```bash
pytest -q
```

Tests use temporary directories. Workloads use `random.Random(seed)`, so the same generator configuration and seed produce the same operation sequence. Random payload bytes are generated at execution time and are stored on disk.

## Limitations and Phase boundary

The local implementation uses configurable filesystem paths, not guaranteed physically distinct media. Separate SSD/HDD devices are required for physical-device benchmarking. Phase 1 deliberately stops before ML, embeddings, semantic importance scoring, prediction, reinforcement learning, optimization, and an adaptive unified decision engine.