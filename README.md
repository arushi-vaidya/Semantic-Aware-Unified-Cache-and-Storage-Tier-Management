# Semantic-Aware Unified Cache and Storage Tier Management

## Phase 1 baseline and Phase 2 prediction

The Phase 1 baseline is a real LRU cache backed by persistent filesystem tiers and SQLite metadata. New data enters `CACHE`; deterministic capacity enforcement migrates it through `SSD`, `HDD`, and `ARCHIVE`. Phase 2 adds access characterization and prediction as a separate analysis pipeline; cache policy and tier placement remain unchanged.

In Phase 1, physical storage tiers are represented by separate configurable filesystem paths. The system performs real file I/O and migration between these paths. Physical SSD/HDD benchmarking can be configured later when separate devices are available.

Phase 2 predicts access probability only. It does not make storage decisions or automatically migrate data. Semantic importance is supplied as deterministic metadata; semantic classification and the Phase 3 decision engine are not included.

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

## Phase 2: Access prediction

Run a Phase 1 workload first so SQLite contains real `access_log` events. Build a dataset, fit either supported model, evaluate on the chronological test split, and request a current probability:

```bash
python -m src.cli run-workload --type hot --operations 1000 --objects 200 --seed 42
python -m src.cli build-prediction-dataset --prediction-window 20 --workload hot --seed 42
python -m src.cli train-access-model --model logistic
python -m src.cli train-access-model --model random-forest
python -m src.cli evaluate-access-model
python -m src.cli predict-access --object-id object_001 --model results/prediction/models/random_forest.joblib
```

The target at operation position `t` is 1 when that object has at least one successful access in the next `N` `access_log` operations (`N=20` by default), otherwise 0. Feature history ends before position `t`. Rows without a complete future window are omitted. Chronological 70/15/15 train/validation/test partitions are used by default; samples within one prediction window of a partition boundary are purged so their labels cannot cross into the next partition. This avoids the temporal leakage and optimistic metrics that random splitting can introduce.

Features are operation recency, timestamp-based seconds recency, total and recent successful access counts (default windows 10/50/100 and the prediction window), read/write counts and ratio, mean/median inter-access seconds, event-time object size, and event-time storage tier. The model excludes timestamp, operation index, and object ID. Tier is one-hot encoded rather than assigned an arbitrary numeric rank. Inter-arrival statistics are zero with an explicit missing indicator until there are at least two prior accesses. Cold-start recency uses a configurable large sentinel (default 1,000,000 operations); unknown historical size/tier remain explicit missing/unknown values.

`access_log.id` provides the operation order. Phase 2 adds nullable size/tier snapshots to new log events and replays successful, timestamped migrations when reconstructing historical tier state. Existing Phase 1 databases are upgraded additively when opened; old access rows are not backfilled from current mutable object metadata, because that would leak future state. Their missing historical size/tier values remain marked unknown.

The baseline predictors are always-negative, recent-frequency (`recent_count / prediction_window`, clipped to [0,1]), and a recency/frequency heuristic. Logistic Regression and Random Forest use class weighting and deterministic seeds. Test reports include accuracy, precision, recall, F1, ROC-AUC, average precision (PR-AUC), log loss, Brier score, confusion matrix, class distribution, and Random Forest feature importance. ROC-AUC/PR-AUC are reported as undefined when the test set contains only one class. The report's best-model label selects highest ROC-AUC, breaking ties by F1. Actual datasets, splits, model files, schema, prediction output, metrics, experiment metadata, and plots are written under `results/prediction/`.

The standalone equivalents are `scripts/build_prediction_dataset.py`, `scripts/train_access_model.py`, `scripts/evaluate_access_model.py`, and `scripts/predict_access.py`. The CLI accepts split fractions and recent windows for reproducible experiments. The workload name and seed are experiment metadata supplied when building a dataset; they are not inferred from SQLite.

Semantic importance is a separate externally supplied score in [0,1], persisted in SQLite and not used as a model feature or placement input:

```bash
python -m src.cli set-semantic-importance --object-id object_001 --score 1.0 --category transaction
```

The category defaults in `src/semantic/importance.py` are configurable by the caller. There is no LLM, embedding, or automatic semantic classification in Phase 2.

## Tests and reproducibility

```bash
pytest -q
```

Tests use temporary directories. Workloads use `random.Random(seed)`, so the same generator configuration and seed produce the same operation sequence. Random payload bytes are generated at execution time and are stored on disk.

## Limitations and Phase boundary

The local implementation uses configurable filesystem paths, not guaranteed physically distinct media. Separate SSD/HDD devices are required for physical-device benchmarking. Prediction quality depends on workload size and diversity; a small or single-class chronological split cannot support all metrics or model fitting. The prediction output is not evidence of improved storage performance and does not affect cache contents or tier placement. Using access probabilities or semantic importance to make placement decisions is reserved for Phase 3.