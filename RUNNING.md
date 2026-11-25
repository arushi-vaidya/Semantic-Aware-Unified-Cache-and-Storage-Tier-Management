# How to run everything

All commands run from the project folder:

```bash
cd "/Users/arushivaidya/Desktop/College/sem 7/ISM/EL"
```

## 0. One-time setup

```bash
python3.11 -m pip install -r requirements.txt
```

`zstd` is needed to unpack the traces (`brew install zstd`). `tectonic` is needed only to build the paper PDF (`brew install tectonic`).

## 1. Just open the UI (data and models are already on this laptop)

```bash
python3.11 ui/server.py
```

Then open **http://localhost:8050**. Press Ctrl+C in the terminal to stop it.

| Page | What it shows |
|---|---|
| Overview | Headline numbers, miss ratio vs cache size, SSD writes |
| Live simulator | Pick a trace, cache size and policies, then press **Run**. It replays the real trace through each policy (about 5–30 s) and shows hit ratio over time, a leaderboard, and latency/SSD writes in 3-tier mode |
| Cache benchmark | Full results table, per-trace curves, a SAGE-vs-baseline heatmap, ablations |
| DRAM→SSD→HDD | Tiering latency, SSD writes and per-trace wins at 3 size pairs |
| Model & method | Pipeline, feature importance, parameters, the scoring rule |
| Data & papers | The traces, the base papers and how SAGE compares |

## 2. Reproduce everything from scratch

Run these in order. The times are for an 8-core, 8 GB laptop; close heavy apps so it doesn't swap.

```bash
bash scripts/download_traces.sh                                    # 1. download traces (~430 MB)
python3.11 scripts/bench_cache.py --stage model --threads 7        # 2. train SAGE models (~45 min)
python3.11 scripts/bench_cache.py --stage sim --workers 7 \
  --policies LRU ARC LeCaR S3-FIFO Belady SAGE SAGE-noAdmit SAGE-noCond SAGE-objOnly SAGE-H   # 3. single-cache runs (~25 min)
python3.11 scripts/bench_cache.py --stage sim --workers 7 \
  --policies S4-FIFO SR-S3-FIFO SIEVE --out results/benchmark/raw_results_2026.csv           # 4. 2026 baselines (~25 min)
python3.11 scripts/tune_s4fifo.py                                  # 5. realistic S4-FIFO config choice (~5 min)
python3.11 scripts/bench_tiers.py --workers 7                      # 6. DRAM->SSD->HDD experiment (~15 min)
python3.11 scripts/make_report.py                                  # 7. results/benchmark/REPORT.md + plots
```

Optional: LRB is slow (about 3 h on this laptop):

```bash
python3.11 scripts/bench_cache.py --stage sim --policies LRB --workers 7
```

## 3. Build the paper

```bash
cd paper && python3.11 gen_assets.py && tectonic sage_ieee.tex && open sage_ieee.pdf
```

`gen_assets.py` regenerates every number, table and figure from the result CSVs, so the PDF always matches the data.

## 4. Tests

```bash
python3.11 -m pytest -q
```

## 5. Where things are

| Path | Contents |
|---|---|
| `src/sim/` | Simulator, SAGE model, all baseline policies, tiering |
| `scripts/` | Benchmarks, S4-FIFO tuning, report, trace download |
| `ui/` | Dashboard server (`server.py`) and frontend (`static/`) |
| `paper/` | IEEE paper (`sage_ieee.tex`, `refs.bib`, `generated/`) |
| `results/benchmark/` | Raw CSVs, `REPORT.md`, plots |
| `data/traces/bin/` | Decompressed traces (not in git) |
| `data/sage_cache/` | Trained SAGE predictions per trace (not in git) |
