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

## Phase 3: Unified placement

Phase 3 adds an online decision engine while leaving the Phase 1 LRU cache and Phase 2 prediction interfaces intact. The unified runner uses `StorageManager` and its filesystem tiers, the existing SQLite access/semantic metadata, `predict_object` from Phase 2, and the existing copy, fsync, size, and SHA-256 verification in `StorageManager.move`. Adaptive runs use isolated directories under `results/decision`; they do not reset or overwrite the configured Phase 1 data.

At each configured interval, the runner predicts each object's probability of access using the selected, trained Phase 2 model, reads semantic importance from the persistent metadata, snapshots actual filesystem use, scores all feasible tiers, persists the decision, then performs any eligible migration. Reads and writes are actual filesystem operations and are recorded in the run-local SQLite database. An unassigned semantic score is represented as 0.0; no score is inferred or fabricated. The isolated runner copies a score only when the source SQLite database already contains one for the same object ID.

For object $x$ and candidate tier $T$, the implemented utility is:

$$
U(T\mid x)=\alpha P_x L_T+\beta S_x L_T+\gamma L_T-\delta M_{x,T}-\epsilon C_{x,T}
$$

Here $P_x$ is the Phase 2 `predict_proba` probability, $S_x$ is the separately persisted semantic score, and $L_T$ is the configured normalized latency benefit (not a hardware latency measurement). For a move, $M_{x,T}=\min(1,\mathrm{size}_x/\mathrm{migration\_reference\_bytes})\times(\mathrm{source\_migration\_cost}+\mathrm{destination\_migration\_cost})/2$; it is zero when the candidate is the current tier. Resource cost is $C_{x,T}=(\mathrm{relative\_cost}_T+\mathrm{projected\_utilization}_T)/2$. The five weights, normalized tier properties, and migration reference size are configurable experimental parameters, not tuned or claimed optimal values.

Capacity is derived from actual object files and the configured per-tier byte limits. A candidate that cannot fit is rejected; an object in an already-overfull tier can be moved out even if ordinary hysteresis would retain it. Before applying a move, the migration controller verifies metadata and source/destination paths and prevents duplicate in-flight moves. `StorageManager.move` copies to a temporary file, fsyncs, verifies byte count and SHA-256, and updates metadata only after the source is safely verified. Failures are logged and raised.

Unified policy migrations are gated by both a score hysteresis threshold and a minimum decision margin. Minimum residency and migration cooldown are operation-count limits persisted per object in SQLite. The configurable interval avoids reevaluating on every access. Decisions record probabilities, semantic values, all tier utilities and score contributions, feasibility, the selected tier, margin, and reason. `explain-decision` returns the latest persisted explanation. The deterministic `independent-tiering` policy is selectable through `decision.policy`; it uses only recent access frequency and age thresholds and records prediction/semantic values as null. `decision.policy: lru` runs the existing Phase 1 LRU cache in the isolated experiment harness. ARC is not present in this repository and is not added here.

Initial parameters are in `config/config.yaml`. Tier latency weights and relative costs are normalized policy inputs, not measured device characteristics. The values should be tuned only through a future controlled evaluation; Phase 3 makes no performance-improvement claim.

Train Phase 2 from real access logs before running the unified policy:

```bash
python3.11 -m src.cli build-prediction-dataset --workload unspecified
python3.11 -m src.cli train-access-model --model logistic
```

Run one isolated adaptive trace (the same command also works for `uniform` and `hot`):

```bash
python3.11 -m src.cli run-adaptive-workload --type sequential --operations 5000 --objects 200 --size 65536 --seed 42
python3.11 -m src.cli run-adaptive-workload --type uniform --operations 5000 --objects 200 --size 65536 --seed 42
python3.11 -m src.cli run-adaptive-workload --type hot --operations 5000 --objects 200 --size 65536 --seed 42
```

Each run writes `decisions.csv`, `migrations.csv`, `placement_history.csv`, `workload.json`, `decision_metrics.json`, and `experiment_metadata.json`, alongside its isolated `metadata.sqlite` and tier directories. Outputs are measured from that run's real state. The metadata includes workload seed, selected policy, model path and SHA-256 when applicable, effective parameters, and timing. Set `decision.policy` to `independent-tiering` or `lru` in a copied config to run either baseline without a model. `scripts/evaluate_placement.py <run-dir>` audits actual files against SQLite. `scripts/compare_policies.py <run-dir> ...` tabulates measured metrics without interpreting them as evidence of improvement.

To replay a saved Phase 1 operation trace instead of generating a new one, pass `--trace path/to/workload.json`; the replayed trace is copied into the run output as `workload.json`. Inspect that isolated run with `python3.11 -m src.cli explain-decision --run-dir results/decision/<run> --object-id object_0001`, `decision-history --run-dir ...`, and `placement-status --run-dir ...`.

Current limitations: the tiers are filesystem paths and may not correspond to physically distinct devices; latency-benefit and relative-cost values are configurable assumptions; semantic scores must be assigned externally and the current legacy database may not contain any; archive capacity is unlimited unless configured; decisions are evaluated at intervals, not after every operation. Phase 4 remains responsible for controlled baseline comparisons, repeated trials, statistical analysis, and any performance claims. Phase 3 supplies reproducible traces, decisions, migrations, placement snapshots, access latency/cache counts, prediction/decision/migration timing, and capacity/thrashing diagnostics for that work.

## Tests and reproducibility

```bash
pytest -q
```

Tests use temporary directories. Workloads use `random.Random(seed)`, so the same generator configuration and seed produce the same operation sequence. Random payload bytes are generated at execution time and are stored on disk.

## Limitations and Phase boundary

The local implementation uses configurable filesystem paths, not guaranteed physically distinct media. Separate SSD/HDD devices are required for physical-device benchmarking. Prediction quality depends on workload size and diversity; a small or single-class chronological split cannot support all metrics or model fitting. Phase 3 uses predictions and semantic metadata for placement, but those inputs and the utility weights require controlled evaluation before drawing performance conclusions.
## Phase 4: SAGE — learned eviction and unified tiering on real traces

`src/sim/` is a fast trace-driven simulator (unit-size blocks) used for the controlled evaluation the
earlier phases deferred. It replays real block-I/O traces against published policies and SAGE
(Survival-Aware, Group-context Eviction). Full numbers: `results/benchmark/REPORT.md`.

* Data: 13 public traces in libCacheSim `oracleGeneral` format from
  `https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/` (MSR Cambridge, FIU,
  CloudPhysics). Dev (hyper-parameter tuning only): `msr_proj_0`, `fiu_webresearch`. Test: 11 others.
* Protocol: first 2M requests per trace, 20% warm-up, miss ratio on the remaining 80%; cache sizes
  0.5/1/5/10% of distinct blocks; SAGE models are trained only on requests before the window they serve.
* Baselines: S4-FIFO (OSDI'26), SR-Gadgets/SR-S3-FIFO (2026), SIEVE (NSDI'24), S3-FIFO (SOSP'23),
  LRB (NSDI'20, partial), LeCaR (HotStorage'18), ARC (FAST'03), LRU, Belady (bound).

```bash
python3.11 -m pip install -r requirements.txt
# download + decompress traces into data/traces/bin/<name>.bin (zstd -d)
python3.11 scripts/bench_cache.py --stage model --threads 7                # SAGE hazard models (once per trace)
python3.11 scripts/bench_cache.py --stage sim --workers 7                  # all policies x sizes
python3.11 scripts/bench_cache.py --stage sim --policies S4-FIFO SR-S3-FIFO SIEVE --out results/benchmark/raw_results_2026.csv
python3.11 scripts/tune_s4fifo.py                                          # S4-FIFO config chosen from warm-up prefix
python3.11 scripts/bench_tiers.py                                          # DRAM -> SSD -> HDD experiment
python3.11 scripts/make_report.py                                          # tables, Wilcoxon tests, plots
```
