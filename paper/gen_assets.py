"""Generate every number, table and figure used in paper/sage_ieee.tex from results/benchmark/*.csv."""
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
from scripts.bench_cache import OUT_DIR, TEST_TRACES  # noqa: E402

GEN = ROOT / "paper" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})

LABEL = {"SAGE": r"\textbf{SAGE}", "SAGE-H": "SAGE-H", "S4-FIFO": "S4-FIFO~\\cite{s4fifo}",
         "S4-FIFO-oracle": "S4-FIFO (oracle)", "SR-S3-FIFO": "SR-S3-FIFO~\\cite{srgadgets}",
         "SIEVE": "SIEVE~\\cite{sieve}", "S3-FIFO": "S3-FIFO~\\cite{s3fifo}", "ARC": "ARC~\\cite{arc}",
         "LeCaR": "LeCaR~\\cite{lecar}", "LRU": "LRU", "Belady": "Belady (bound)~\\cite{belady}",
         "LRB": "LRB~\\cite{lrb}"}


def miss_table() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in sorted(OUT_DIR.glob("raw_results*.csv"))])
    d = d[d.trace.isin(TEST_TRACES)]
    d["variant"] = d["variant"].fillna("").astype(str)
    d = d.drop_duplicates(["trace", "size_frac", "policy", "variant"], keep="last")
    s4 = d[d.policy == "S4-FIFO"].copy()
    s4["vi"] = s4.variant.astype(float).astype(int)
    choice = pd.read_csv(OUT_DIR / "s4fifo_prefix_choice.csv").rename(columns={"variant": "vi"})
    real = choice[["trace", "size_frac", "vi"]].merge(s4, on=["trace", "size_frac", "vi"]).drop(columns="vi")
    oracle = s4.loc[s4.groupby(["trace", "size_frac"]).miss_ratio.idxmin()].assign(policy="S4-FIFO-oracle")
    lrb = d[d.policy == "LRB"]
    lrb = lrb[lrb.groupby(["trace", "size_frac"]).variant.transform("nunique") == 3]
    lrb = lrb.loc[lrb.groupby(["trace", "size_frac"]).miss_ratio.idxmin()]
    rest = d[~d.policy.isin(["S4-FIFO", "LRB"])]
    return pd.concat([rest, real, oracle, lrb]).pivot_table(index=["trace", "size_frac"], columns="policy", values="miss_ratio")


def tex_table(path: Path, header: list[str], rows: list[list[str]], colspec: str) -> None:
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    macros: dict[str, str] = {}
    t = miss_table()
    sizes = sorted(t.index.get_level_values(1).unique())

    # ---- Table: dataset statistics
    from src.sim.traces import load_trace
    rows = []
    for name in TEST_TRACES:
        tr = load_trace(name, 2_000_000)
        fam = {"msr": "MSR", "fiu": "FIU"}.get(name[:3], "CloudPhysics")
        reuse = 1 - tr.n_objects / tr.n
        rows.append([name.replace("_", r"\_"), fam, f"{tr.n/1e6:.1f}M", f"{tr.n_objects/1e3:,.0f}K", f"{reuse:.2f}"])
    tex_table(GEN / "tab_traces.tex", ["Trace", "Source", "Requests", "Blocks", "Reuse"], rows, "llrrr")

    # ---- Table: mean miss ratio by cache size
    cols = ["Belady", "SAGE", "SAGE-H", "S4-FIFO", "S4-FIFO-oracle", "SR-S3-FIFO", "SIEVE", "S3-FIFO", "ARC", "LeCaR", "LRU"]
    rows = []
    for c in cols:
        vals = [t.xs(s, level=1)[c].mean() for s in sizes]
        best = [min(t.xs(s, level=1)[k].mean() for k in cols if k not in ("Belady", "S4-FIFO-oracle")) for s in sizes]
        cells = [(f"\\textbf{{{v:.3f}}}" if abs(v - b) < 1e-12 else f"{v:.3f}") for v, b in zip(vals, best)]
        rows.append([LABEL[c]] + cells)
    tex_table(GEN / "tab_miss.tex", ["Policy"] + [f"{s*100:g}\\%" for s in sizes], rows, "lrrrr")

    # ---- Table: SAGE vs each baseline
    rows = []
    for base in ["S4-FIFO", "SR-S3-FIFO", "S4-FIFO-oracle", "SIEVE", "S3-FIFO", "ARC", "LeCaR", "LRU"]:
        r = ((t[base] - t["SAGE"]) / t[base] * 100).dropna()
        small = r[r.index.get_level_values(1) <= 0.01]
        p = wilcoxon(t.loc[r.index, base], t.loc[r.index, "SAGE"], alternative="greater").pvalue
        rows.append([LABEL[base], f"{r.mean():.2f}", f"{small.mean():.2f}", f"{int((r > 0).sum())}/{len(r)}",
                     f"{int((small > 0).sum())}/{len(small)}", f"{p:.1e}".replace("e-0", "e-")])
        key = base.replace("-", "").replace("3", "Three").replace("4", "Four")
        macros[f"red{key}"] = f"{r.mean():.1f}"
        macros[f"redSmall{key}"] = f"{small.mean():.1f}"
        macros[f"win{key}"] = f"{int((r > 0).sum())}"
        macros[f"winSmall{key}"] = f"{int((small > 0).sum())}"
    tex_table(GEN / "tab_vs.tex", ["Baseline", "Red.\\%", "Red.\\% ($\\le$1\\%)", "Wins", "Wins ($\\le$1\\%)", "$p$"], rows, "lrrrrr")
    macros["nConfigs"] = str(len(t))

    # ---- Table: ablations
    rows = []
    for a, desc in [("SAGE", "full model"), ("SAGE-noCond", "no idle-time conditioning"), ("SAGE-noAdmit", "no admission control"),
                    ("SAGE-objOnly", "object features only"), ("SAGE-H", "+ regret hedge (post hoc)")]:
        dlt = (t[a] - t["SAGE"]) * 100
        rows.append([a, desc, f"{t[a].mean():.4f}", f"{dlt.mean():+.2f}", f"{int((dlt > 1e-12).sum())}/{len(dlt)}"])
    tex_table(GEN / "tab_ablation.tex", ["Variant", "Change", "Mean MR", "$\\Delta$ (pp)", "Worse"], rows, "llrrr")
    macros["ablNoCond"] = f"{((t['SAGE-noCond'] - t['SAGE']) * 100).mean():.1f}"
    macros["ablNoAdmit"] = f"{((t['SAGE-noAdmit'] - t['SAGE']) * 100).mean():.2f}"

    # ---- Figure: mean miss ratio vs size
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for c, st in [("Belady", "k:"), ("SAGE", "r-o"), ("S4-FIFO", "b-s"), ("SR-S3-FIFO", "g-^"), ("ARC", "m-d"), ("LRU", "c--")]:
        ax.plot([s * 100 for s in sizes], [t.xs(s, level=1)[c].mean() for s in sizes], st, label=c, ms=3, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([s * 100 for s in sizes])
    ax.set_xticklabels([f"{s*100:g}%" for s in sizes])
    ax.set_xlabel("cache size (% of distinct blocks)")
    ax.set_ylabel("mean miss ratio")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(GEN / "fig_miss.pdf")

    # ---- Tiering
    tr = pd.read_csv(OUT_DIR / "tier_results.csv")
    tr = tr[tr.trace.isin(TEST_TRACES)]
    pols = ["LRU-cascade", "ARC-cascade", "S4-FIFO-cascade", "SR-S3-FIFO-cascade", "SAGE-L1+LRU-SSD", "SAGE-Unified"]
    pols = [p for p in pols if p in set(tr.policy)]
    pairs = sorted({(a, b) for a, b in zip(tr.dram_frac, tr.ssd_frac)})
    rows = []
    for p in pols:
        cells = []
        for a, b in pairs:
            g = tr[(tr.dram_frac == a) & (tr.policy == p)]
            cells += [f"{g.mean_latency_ms.mean():.3f}", f"{g.ssd_writes_per_kreq.mean():.0f}"]
        rows.append([p.replace("SAGE-Unified", r"\textbf{SAGE-Unified}")] + cells)
    hdr = ["Policy"] + [x for a, b in pairs for x in (f"Lat. {a*100:g}/{b*100:g}", "Wr/k")]
    tex_table(GEN / "tab_tiers.tex", hdr, rows, "l" + "rr" * len(pairs))
    best_base = [p for p in pols if p.endswith("cascade")]
    lat_red, wr_red, wins = [], [], 0
    per = []
    for (a, b) in pairs:
        g = tr[tr.dram_frac == a].pivot_table(index="trace", columns="policy", values=["mean_latency_ms", "ssd_writes_per_kreq"])
        lat = g["mean_latency_ms"]
        wr = g["ssd_writes_per_kreq"]
        bestlat = lat[best_base].min(axis=1)
        bestcol = lat[best_base].idxmin(axis=1)
        bestwr = pd.Series([wr.loc[i, c] for i, c in bestcol.items()], index=bestcol.index)
        lr = (bestlat - lat["SAGE-Unified"]) / bestlat * 100
        wrr = (bestwr - wr["SAGE-Unified"]) / bestwr * 100
        lat_red.append(lr)
        wr_red.append(wrr)
        wins += int((lr > 0).sum())
        per.append((a, b, lr, wrr))
        for base in ["S4-FIFO-cascade", "SR-S3-FIFO-cascade", "ARC-cascade"]:
            if base in lat:
                key = base.split("-cascade")[0].replace("-", "").replace("3", "Three").replace("4", "Four")
                macros[f"tierLat{key}{ {5: 'Half', 10: 'One', 20: 'Two'}[int(round(a*1000))] }"] = f"{((lat[base] - lat['SAGE-Unified']) / lat[base] * 100).mean():.1f}"
                macros[f"tierWr{key}{ {5: 'Half', 10: 'One', 20: 'Two'}[int(round(a*1000))] }"] = f"{((wr[base] - wr['SAGE-Unified']) / wr[base] * 100).mean():.0f}"
    allr = pd.concat(lat_red)
    allw = pd.concat(wr_red)
    macros["tierLatRedBest"] = f"{allr.mean():.1f}"
    macros["tierLatRedMax"] = f"{allr.max():.1f}"
    macros["tierWrRedBest"] = f"{allw.mean():.0f}"
    macros["tierWins"] = str(wins)
    macros["tierN"] = str(len(allr))
    s4 = tr[tr.policy == "S4-FIFO-cascade"]
    p_lat = wilcoxon(pd.concat([tr[(tr.dram_frac == a)].pivot_table(index="trace", columns="policy", values="mean_latency_ms")[best_base].min(axis=1) for a, _ in pairs]),
                     pd.concat([tr[(tr.dram_frac == a)].pivot_table(index="trace", columns="policy", values="mean_latency_ms")["SAGE-Unified"] for a, _ in pairs]),
                     alternative="greater").pvalue
    m, ex = f"{p_lat:.1e}".split("e")
    macros["tierP"] = f"{m}\\times 10^{{{int(ex)}}}"

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    x = np.arange(len(TEST_TRACES))
    mid = [p for p in per if abs(p[0] - 0.01) < 1e-9][0]
    for ax, series, lab in [(axes[0], mid[2], "latency reduction (%)"), (axes[1], mid[3], "SSD-write reduction (%)")]:
        vals = series.reindex(TEST_TRACES).values
        ax.bar(x, vals, color=["#c0392b" if v >= 0 else "#7f8c8d" for v in vals], width=0.7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(TEST_TRACES, rotation=55, ha="right", fontsize=6)
        ax.set_ylabel(lab)
    fig.suptitle("SAGE-Unified vs. the best cascade baseline per trace (DRAM 1%, SSD 5%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(GEN / "fig_tiers.pdf")

    # ---- model statistics
    imp: dict[str, list[float]] = {}
    for name in TEST_TRACES:
        m = json.loads((ROOT / "data" / "sage_cache" / f"{name}_full.json").read_text())
        for k, v in m["feature_importance"].items():
            imp.setdefault(k, []).append(v)
    top = sorted(((k, np.mean(v)) for k, v in imp.items()), key=lambda kv: -kv[1])[:8]
    tex_table(GEN / "tab_importance.tex", ["Feature", "Gain share"], [[k.replace("_", r"\_"), f"{v:.3f}"] for k, v in top], "lr")

    (GEN / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print(json.dumps(macros, indent=1))


if __name__ == "__main__":
    main()
