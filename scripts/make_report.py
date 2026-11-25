"""Aggregate benchmark CSVs into tables, statistics and plots (results/benchmark/)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_cache import CACHE_DIR, OUT_DIR, TEST_TRACES  # noqa: E402

BASES = ["S4-FIFO", "SR-S3-FIFO"]  # 2026 base papers (S4-FIFO = realistic prefix-tuned selector)
ORACLE = "S4-FIFO-oracle"
OTHERS = ["SIEVE", "S3-FIFO", "ARC", "LeCaR", "LRU"]
ABLATIONS = ["SAGE-H", "SAGE-noAdmit", "SAGE-noCond", "SAGE-objOnly"]


def load() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in sorted(OUT_DIR.glob("raw_results*.csv"))])
    d = d.drop_duplicates(["trace", "size_frac", "policy", "variant"], keep="last")
    d = d[d.trace.isin(TEST_TRACES)]
    d["variant"] = d["variant"].fillna("").astype(str)
    tuned = []
    best_w = {}
    choice = pd.read_csv(OUT_DIR / "s4fifo_prefix_choice.csv")
    s4 = d[d.policy == "S4-FIFO"].copy()
    s4["variant_i"] = s4.variant.astype(float).astype(int)
    real = choice.rename(columns={"variant": "variant_i"})[["trace", "size_frac", "variant_i"]].merge(s4, on=["trace", "size_frac", "variant_i"])
    best_w["S4-prefix"] = real.set_index(["trace", "size_frac"]).variant_i
    tuned.append(real.drop(columns="variant_i"))
    d = d.assign(policy=d.policy.where(d.policy != "S4-FIFO", ORACLE))
    for pol in (ORACLE, "LRB"):
        g = d[d.policy == pol]
        if pol == "LRB":  # only configs where every memory window finished
            g = g[g.groupby(["trace", "size_frac"]).variant.transform("nunique") == 3]
        if len(g):
            b = g.loc[g.groupby(["trace", "size_frac"]).miss_ratio.idxmin()]
            best_w[pol] = b.set_index(["trace", "size_frac"]).variant
            tuned.append(b)
    d = pd.concat([d[~d.policy.isin([ORACLE, "LRB"])], *tuned])
    table = d.pivot_table(index=["trace", "size_frac"], columns="policy", values="miss_ratio")
    return table, best_w


def reduction(table, base, ours="SAGE"):
    return (table[base] - table[ours]) / table[base] * 100


def main() -> None:
    table, best_w = load()
    lines = ["# SAGE benchmark report", "",
             f"Test traces: {len(table.index.get_level_values(0).unique())}; cache sizes: "
             f"{sorted(table.index.get_level_values(1).unique())} of distinct objects; "
             "miss ratio measured on the last 80% of the first 2M requests.", "",
             "S4-FIFO picks one of 40 queue configurations per (trace, size) using only the warm-up prefix (a realistic stand-in for its pre-trained selector). S4-FIFO-oracle is the best configuration in hindsight (an upper bound no deployable selector reaches).", ""]
    # ---- headline
    lines += ["## Headline: miss-ratio reduction of SAGE vs each baseline", "",
              "| baseline | mean reduction % | median % | min % | max % | wins / configs | Wilcoxon p |",
              "|---|---|---|---|---|---|---|"]
    summary = {}
    for base in BASES + [ORACLE] + OTHERS + (["LRB"] if "LRB" in table else []):
        r = reduction(table, base).dropna()
        if len(r) < 6:
            continue
        p = wilcoxon(table.loc[r.index, base], table.loc[r.index, "SAGE"], alternative="greater").pvalue
        summary[base] = r
        lines.append(f"| {base}{' (partial: ' + str(len(r)) + ' configs)' if base == 'LRB' else ''} | {r.mean():.2f} | {r.median():.2f} | {r.min():.2f} | {r.max():.2f} | "
                     f"{int((r > 0).sum())} / {len(r)} | {p:.2e} |")
    lines += ["", "SAGE-H (regret-hedged variant, added after inspecting large-cache results):", "",
              "| baseline | mean reduction % | wins / configs |", "|---|---|---|"]
    for base in BASES + [ORACLE] + OTHERS:
        r = reduction(table, base, "SAGE-H").dropna()
        lines.append(f"| {base} | {r.mean():.2f} | {int((r > 0).sum())} / {len(r)} |")
    lines += ["", "### Wins by cache size (SAGE / SAGE-H vs each 2026 baseline)", "", "| baseline | 0.5% | 1% | 5% | 10% |", "|---|---|---|---|---|"]
    for base in BASES + [ORACLE]:
        cells = []
        for sz in sorted(table.index.get_level_values(1).unique()):
            g = table.xs(sz, level=1)
            cells.append(f"{int((g['SAGE'] < g[base]).sum())} / {int((g['SAGE-H'] < g[base]).sum())} of {len(g)}")
        lines.append(f"| {base} | " + " | ".join(cells) + " |")
    # Belady gap closure vs best base paper
    best_base = table[BASES].min(axis=1)
    gap = ((best_base - table["SAGE"]) / (best_base - table["Belady"]).clip(lower=1e-9) * 100)
    lines += ["", f"SAGE vs the *best* of the 2026 base papers per config (oracle choice of baseline): "
              f"mean miss-ratio reduction {((best_base - table['SAGE']) / best_base * 100).mean():.2f}%, "
              f"wins {int((table['SAGE'] < best_base).sum())}/{len(table)}; "
              f"closes a mean {gap.median():.1f}% (median) of the remaining gap to Belady.", ""]
    # ---- per size
    lines += ["## Mean miss ratio by cache size", "", "| size | " + " | ".join(["Belady", "SAGE", *BASES, *OTHERS]) + " |",
              "|---" * (2 + len(BASES) + len(OTHERS)) + "|---|"]
    for size, g in table.groupby(level=1):
        lines.append(f"| {size*100:g}% | " + " | ".join(f"{g[c].mean():.4f}" for c in ["Belady", "SAGE", *BASES, *OTHERS]) + " |")
    # ---- full table
    cols = ["Belady", "SAGE", "SAGE-H", *BASES, ORACLE, *OTHERS, *ABLATIONS[1:]]
    cols = [c for c in cols if c in table.columns]
    lines += ["", "## Full results (miss ratio)", "", "| trace | size | " + " | ".join(cols) + " | S4 cfg (prefix/oracle) |",
              "|---" * (len(cols) + 3) + "|"]
    for (trace, size), row in table.iterrows():
        best = min(row[c] for c in ["SAGE", *BASES, *OTHERS])
        cells = [(f"**{row[c]:.4f}**" if c != "Belady" and abs(row[c] - best) < 1e-12 else f"{row[c]:.4f}") for c in cols]
        lines.append(f"| {trace} | {size*100:g}% | " + " | ".join(cells) + f" | {best_w['S4-prefix'].get((trace, size), '')} / {best_w[ORACLE].get((trace, size), '')} |")
    # ---- ablations
    lines += ["", "## Ablations (mean miss ratio; increase vs full SAGE)", "", "| variant | mean MR | Δ vs SAGE (pp) | worse in |", "|---|---|---|---|"]
    for a in ["SAGE", *ABLATIONS]:
        if a in table:
            delta = (table[a] - table["SAGE"]) * 100
            lines.append(f"| {a} | {table[a].mean():.4f} | {delta.mean():+.3f} | {int((delta > 0).sum())}/{len(delta)} |")
    # ---- tiers
    tier_path = OUT_DIR / "tier_results.csv"
    if tier_path.exists():
        t = pd.read_csv(tier_path)
        t = t[t.trace.isin(TEST_TRACES)]
        lines += ["", "## DRAM(1%) → SSD(5%) → HDD tiering", "",
                  "Latency model: DRAM 0.1 µs, SSD 100 µs, HDD 5 ms per request.", "",
                  "| policy | mean latency ms | DRAM hit | SSD hit | HDD miss | SSD writes / 1k req |", "|---|---|---|---|---|---|"]
        agg = t.groupby("policy")[["mean_latency_ms", "dram_hit", "ssd_hit", "hdd_miss", "ssd_writes_per_kreq"]].mean()
        for pol, r in agg.iterrows():
            lines.append(f"| {pol} | {r.mean_latency_ms:.3f} | {r.dram_hit:.4f} | {r.ssd_hit:.4f} | {r.hdd_miss:.4f} | {r.ssd_writes_per_kreq:.1f} |")
        piv = t.pivot_table(index="trace", columns="policy", values=["mean_latency_ms", "ssd_writes_per_kreq"])
        lat = piv["mean_latency_ms"]
        wr = piv["ssd_writes_per_kreq"]
        lines += ["", f"SAGE-Unified vs ARC-cascade: latency {((lat['ARC-cascade'] - lat['SAGE-Unified']) / lat['ARC-cascade'] * 100).mean():.2f}% lower "
                  f"(wins {int((lat['SAGE-Unified'] < lat['ARC-cascade']).sum())}/{len(lat)}), SSD writes "
                  f"{((wr['ARC-cascade'] - wr['SAGE-Unified']) / wr['ARC-cascade'] * 100).mean():.1f}% fewer.", "",
                  "| trace | " + " | ".join(f"{p} lat" for p in lat.columns) + " | " + " | ".join(f"{p} wr/k" for p in wr.columns) + " |",
                  "|---" * (1 + 2 * len(lat.columns)) + "|"]
        for tr in lat.index:
            lines.append(f"| {tr} | " + " | ".join(f"{v:.3f}" for v in lat.loc[tr]) + " | " + " | ".join(f"{v:.0f}" for v in wr.loc[tr]) + " |")
    # ---- model stats + feature importance
    imp = {}
    secs = []
    for tr in TEST_TRACES:
        p = CACHE_DIR / f"{tr}_full.json"
        if p.exists():
            m = json.loads(p.read_text())
            secs.append((tr, m["train_seconds"], m["predict_seconds"], m["requests"], np.mean(m["train_rows"])))
            for k, v in m["feature_importance"].items():
                imp.setdefault(k, []).append(v)
    if imp:
        mean_imp = sorted(((k, np.mean(v)) for k, v in imp.items()), key=lambda x: -x[1])
        lines += ["", "## Learned model: mean gain-importance across test traces (top 15)", "", "| feature | share of gain |", "|---|---|"]
        lines += [f"| {k} | {v:.3f} |" for k, v in mean_imp[:15]]
        lines += ["", "| trace | train s (9 models) | batch predict s | µs per request (predict) | mean training rows |", "|---|---|---|---|---|"]
        lines += [f"| {t} | {a:.0f} | {b:.0f} | {b / n * 1e6:.1f} | {int(r):,} |" for t, a, b, n, r in secs]
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")
    table.to_csv(OUT_DIR / "miss_ratio_table.csv")

    # ---- plots
    fig, ax = plt.subplots(figsize=(7, 3.6))
    sizes = sorted(table.index.get_level_values(1).unique())
    width = 0.8 / len(BASES + OTHERS)
    for j, base in enumerate(BASES + OTHERS):
        vals = [summary[base].xs(s, level=1).mean() for s in sizes]
        ax.bar(np.arange(len(sizes)) + j * width, vals, width, label=base)
    ax.set_xticks(np.arange(len(sizes)) + width * (len(BASES + OTHERS) - 1) / 2)
    ax.set_xticklabels([f"{s*100:g}%" for s in sizes])
    ax.set_xlabel("cache size (fraction of distinct objects)")
    ax.set_ylabel("SAGE miss-ratio reduction (%)")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reduction_by_size.png", dpi=160)

    traces = sorted(table.index.get_level_values(0).unique())
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.5))
    for ax, tr in zip(axes.flat, traces):
        g = table.xs(tr, level=0)
        for c, st in [("Belady", "k:"), ("SAGE", "r-o"), ("SAGE-H", "r--"), ("S4-FIFO", "b-s"), ("SR-S3-FIFO", "g-^"), ("ARC", "m-d"), ("LRU", "c--")]:
            ax.plot([s * 100 for s in g.index], g[c], st, label=c, ms=4)
        ax.set_xscale("log")
        ax.set_title(tr, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(traces):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.supxlabel("cache size (% of distinct objects, log)")
    fig.supylabel("miss ratio")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "miss_ratio_curves.png", dpi=150)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
