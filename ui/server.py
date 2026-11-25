"""SAGE dashboard + live simulator.

Run from the repository root:
    python3.11 ui/server.py            # then open http://localhost:8050

Endpoints
---------
GET  /api/meta            traces, policies, cache-size presets
GET  /api/results         every benchmark number (from results/benchmark/*.csv)
POST /api/simulate        start a live simulation job -> {"job": id}
GET  /api/job/<id>        job status; each policy's result appears as soon as it finishes
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bench_cache import CACHE_DIR, DEV_TRACES, OUT_DIR, SAGE_HMULT, TEST_TRACES  # noqa: E402
from src.sim.policies import S4FIFO_GRID  # noqa: E402
from src.sim.runner import run_python_policy, run_sage  # noqa: E402
from src.sim.tiered import LAT_DRAM_MS, LAT_HDD_MS, LAT_SSD_MS, run_cascade, run_sage_tiered  # noqa: E402
from src.sim.traces import TRACE_DIR, load_trace  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
PORT = 8050
N_WINDOWS = 80
CACHE_POLICIES = ["SAGE", "SAGE-H", "S4-FIFO", "SR-S3-FIFO", "SIEVE", "S3-FIFO", "ARC", "LeCaR", "LRU", "Belady"]
TIER_POLICIES = ["SAGE-Unified", "S4-FIFO-cascade", "SR-S3-FIFO-cascade", "ARC-cascade", "LRU-cascade"]
TRACE_INFO = {
    "msr": "MSR Cambridge enterprise server",
    "fiu": "FIU university server",
    "w": "CloudPhysics cloud VM",
}

# ----------------------------------------------------------------------------- simulation workers
_TRACE = {}


def _trace(name: str, n: int):
    key = (name, n)
    if key not in _TRACE:
        _TRACE.clear()
        _TRACE[key] = load_trace(name, n)
    return _TRACE[key]


def _s4_config(name: str, frac: float) -> int:
    """S4-FIFO configuration chosen on the warm-up prefix (nearest benchmark size)."""
    path = OUT_DIR / "s4fifo_prefix_choice.csv"
    if path.exists():
        ch = pd.read_csv(path)
        ch = ch[ch.trace == name]
        if len(ch):
            row = ch.iloc[(ch.size_frac - frac).abs().argsort().iloc[0]]
            return int(row.variant)
    return 3  # rho_s=0.1, ghost 1x, kappa 0, tau 2 == S3-FIFO default


def _windows(flags: np.ndarray, start: int, tiered: bool) -> dict:
    seg = flags[start:]
    edges = np.linspace(0, len(seg), N_WINDOWS + 1).astype(int)
    out = {"x": [], "hit": [], "dram": [], "ssd": []}
    for a, b in zip(edges[:-1], edges[1:]):
        w = seg[a:b]
        out["x"].append(int(start + b))
        if tiered:
            out["dram"].append(float((w == 1).mean()))
            out["ssd"].append(float((w == 2).mean()))
            out["hit"].append(float((w > 0).mean()))
        else:
            out["hit"].append(float(w.mean()))
    return out


def simulate_one(args: tuple) -> dict:
    name, n, frac, policy, mode = args
    started = time.time()
    trace = _trace(name, n)
    ev = int(0.2 * trace.n)
    flags = np.zeros(trace.n, dtype=np.int8)
    c1 = max(16, int(frac * trace.n_objects))
    res: dict = {"policy": policy}
    logS_path = CACHE_DIR / f"{name}_full.npy"
    if policy.startswith("SAGE") and not logS_path.exists():
        return {"policy": policy, "error": "no trained SAGE model for this trace (run bench_cache.py --stage model)"}
    if mode == "cache":
        if policy.startswith("SAGE"):
            logS = np.ascontiguousarray(np.load(logS_path, mmap_mode="r")[: trace.n])
            r = run_sage(trace, logS, c1, ev, h_mult=SAGE_HMULT, hedge=policy == "SAGE-H", flags=flags)
        elif policy == "S4-FIFO":
            k = _s4_config(name, frac)
            r = run_python_policy(trace, "S4-FIFO", c1, ev, flags=flags, **S4FIFO_GRID[k])
            res["note"] = f"config {k}: {S4FIFO_GRID[k]}"
        else:
            r = run_python_policy(trace, policy, c1, ev, flags=flags)
        res.update(miss_ratio=r["miss_ratio"], hit_ratio=1 - r["miss_ratio"], bypasses=r.get("bypasses"))
        res["windows"] = _windows(flags, ev, False)
    else:
        c2 = 5 * c1
        if policy == "SAGE-Unified":
            logS = np.ascontiguousarray(np.load(logS_path, mmap_mode="r")[: trace.n])
            r = run_sage_tiered(trace, logS, c1, c2, ev, h_mult=SAGE_HMULT, flags=flags)
        elif policy == "S4-FIFO-cascade":
            k = _s4_config(name, frac)
            r = run_cascade(trace, "S4-FIFO", c1, c2, ev, flags=flags, **S4FIFO_GRID[k])
            res["note"] = f"config {k}"
        else:
            r = run_cascade(trace, policy.rsplit("-", 1)[0], c1, c2, ev, flags=flags)
        res.update({k: r[k] for k in ("dram_hit", "ssd_hit", "hdd_miss", "mean_latency_ms", "ssd_writes_per_kreq")})
        res["windows"] = _windows(flags, ev, True)
    res.update(seconds=time.time() - started, capacity=c1, requests=trace.n, objects=trace.n_objects)
    return res


# ----------------------------------------------------------------------------- static results
def build_results() -> dict:
    out: dict = {}
    try:
        sys.path.insert(0, str(ROOT / "paper"))
        from gen_assets import miss_table  # type: ignore
        t = miss_table()
        out["miss"] = [{"trace": tr, "size": float(sz), **{c: (None if pd.isna(v) else float(v)) for c, v in row.items()}}
                       for (tr, sz), row in t.iterrows()]
    except Exception as exc:  # results not generated yet
        out["miss_error"] = str(exc)
    tp = OUT_DIR / "tier_results.csv"
    if tp.exists():
        tr = pd.read_csv(tp)
        tr = tr[tr.trace.isin(TEST_TRACES)]
        out["tiers"] = tr[["trace", "dram_frac", "ssd_frac", "policy", "mean_latency_ms", "dram_hit", "ssd_hit",
                           "hdd_miss", "ssd_writes_per_kreq"]].to_dict("records")
    imp: dict[str, list[float]] = {}
    for name in TEST_TRACES:
        p = CACHE_DIR / f"{name}_full.json"
        if p.exists():
            for k, v in json.loads(p.read_text())["feature_importance"].items():
                imp.setdefault(k, []).append(v)
    out["importance"] = sorted(([k, float(np.mean(v))] for k, v in imp.items()), key=lambda kv: -kv[1])[:14]
    out["latency_model"] = {"dram_ms": LAT_DRAM_MS, "ssd_ms": LAT_SSD_MS, "hdd_ms": LAT_HDD_MS}
    return out


def build_meta() -> dict:
    traces = []
    for f in sorted(TRACE_DIR.glob("*.bin")):
        name = f.stem
        fam = "msr" if name.startswith("msr") else "fiu" if name.startswith("fiu") else "w"
        traces.append({
            "name": name, "family": TRACE_INFO[fam], "requests_total": f.stat().st_size // 24,
            "role": "dev" if name in DEV_TRACES else "test" if name in TEST_TRACES else "extra",
            "sage_model": (CACHE_DIR / f"{name}_full.npy").exists(),
        })
    return {"traces": traces, "cache_policies": CACHE_POLICIES, "tier_policies": TIER_POLICIES,
            "sizes": [0.005, 0.01, 0.02, 0.05, 0.10]}


# ----------------------------------------------------------------------------- HTTP
JOBS: dict[str, dict] = {}
POOL = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC), **kw)

    def log_message(self, fmt, *args):  # quieter console
        if "/api/job/" not in str(args[0] if args else ""):
            sys.stderr.write("[ui] " + fmt % args + "\n")

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/meta":
            return self._json(META)
        if self.path == "/api/results":
            return self._json(RESULTS)
        if self.path.startswith("/api/job/"):
            job = JOBS.get(self.path.rsplit("/", 1)[-1])
            return self._json(job or {"error": "unknown job"}, 200 if job else 404)
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/simulate":
            return self._json({"error": "not found"}, 404)
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        names = {t["name"] for t in META["traces"]}
        name = req.get("trace")
        mode = req.get("mode", "cache")
        allowed = CACHE_POLICIES if mode == "cache" else TIER_POLICIES
        policies = [p for p in req.get("policies", []) if p in allowed]
        try:
            frac = float(req.get("size", 0.01))
            n = int(req.get("requests", 500_000))
        except (TypeError, ValueError):
            return self._json({"error": "bad size/requests"}, 400)
        if name not in names or not policies or not (0.0005 <= frac <= 0.5) or not (50_000 <= n <= 2_000_000):
            return self._json({"error": "invalid request"}, 400)
        jid = uuid.uuid4().hex[:10]
        job = {"id": jid, "trace": name, "mode": mode, "size": frac, "requests": n, "policies": policies,
               "results": {}, "done": False, "started": time.time()}
        JOBS[jid] = job

        def finished(res):
            job["results"][res["policy"]] = res
            if len(job["results"]) == len(policies):
                job["done"] = True
                job["elapsed"] = time.time() - job["started"]

        for p in policies:
            POOL.apply_async(simulate_one, ((name, n, frac, p, mode),), callback=finished,
                             error_callback=lambda e, p=p: finished({"policy": p, "error": str(e)}))
        return self._json({"job": jid})


def main() -> None:
    global POOL, META, RESULTS
    POOL = mp.get_context("fork").Pool(max(2, (mp.cpu_count() or 4) - 2))  # fork before any server threads
    print("[ui] loading benchmark results ...", flush=True)
    META = build_meta()
    RESULTS = build_results()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[ui] SAGE dashboard running at http://localhost:{PORT}  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        POOL.terminate()


META: dict = {}
RESULTS: dict = {}

if __name__ == "__main__":
    main()
