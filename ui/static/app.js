/* SAGE dashboard: dependency-free SVG charts + live simulator client. */
const COLORS = {
  "SAGE": "#ff7a1a", "SAGE-H": "#ffb36b", "S4-FIFO": "#60a5fa", "S4-FIFO-oracle": "#93c5fd", "SR-S3-FIFO": "#34d399",
  "SIEVE": "#c084fc", "S3-FIFO": "#2dd4bf", "ARC": "#facc15", "LeCaR": "#f472b6", "LRU": "#94a3b8", "Belady": "#e5e7eb", "LRB": "#a3e635",
  "SAGE-Unified": "#ff7a1a", "S4-FIFO-cascade": "#60a5fa", "SR-S3-FIFO-cascade": "#34d399", "ARC-cascade": "#facc15",
  "LRU-cascade": "#94a3b8", "SAGE-L1+LRU-SSD": "#ffb36b",
};
const DASH = { "Belady": "5 4", "SAGE-H": "4 3", "S4-FIFO-oracle": "4 3" };
const isSage = p => p.startsWith("SAGE");
const $ = s => document.querySelector(s);
const el = (tag, attrs = {}, html = "") => { const e = document.createElement(tag); Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v)); e.innerHTML = html; return e; };
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const mean = a => a.reduce((s, x) => s + x, 0) / Math.max(1, a.length);
const SVGNS = "http://www.w3.org/2000/svg";
const svg = (tag, attrs = {}) => { const e = document.createElementNS(SVGNS, tag); Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v)); return e; };
const tip = $("#tip");
function showTip(ev, html) { tip.innerHTML = html; tip.style.opacity = 1; const x = Math.min(ev.clientX + 14, innerWidth - 220); tip.style.left = x + "px"; tip.style.top = (ev.clientY + 14) + "px"; }
function hideTip() { tip.style.opacity = 0; }

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || mag * 10;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(10)); return out;
}

/* ---------------------------------------------------------------- line chart */
function lineChart(host, { series, xlabel = "", ylabel = "", xlog = false, xfmt = v => v, yfmt = v => v.toFixed(2), xticks, animate = false, height = 260, ypad = 0.08 }) {
  host.innerHTML = "";
  const W = Math.max(320, host.clientWidth || 600), H = height, m = { l: 54, r: 14, t: 10, b: 38 };
  const xs = series.flatMap(s => s.x), ys = series.flatMap(s => s.y).filter(v => v != null);
  if (!xs.length) { host.innerHTML = '<div class="empty">No data</div>'; return; }
  const tx = v => xlog ? Math.log10(v) : v;
  const x0 = Math.min(...xs.map(tx)), x1 = Math.max(...xs.map(tx));
  let y0 = Math.min(...ys), y1 = Math.max(...ys); const pad = (y1 - y0) * ypad || 0.01; y0 -= pad; y1 += pad;
  const X = v => m.l + (tx(v) - x0) / (x1 - x0 || 1) * (W - m.l - m.r), Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, height: H });
  const g = svg("g", { class: "axis" }); s.appendChild(g);
  niceTicks(y0, y1, 5).forEach(v => { g.appendChild(svg("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "gridl" }));
    const t = svg("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }); t.textContent = yfmt(v); g.appendChild(t); });
  (xticks || niceTicks(Math.min(...xs), Math.max(...xs), 6)).forEach(v => { const t = svg("text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }); t.textContent = xfmt(v); g.appendChild(t); });
  const xl = svg("text", { x: (W + m.l) / 2, y: H - 4, "text-anchor": "middle", class: "lbltxt" }); xl.textContent = xlabel; s.appendChild(xl);
  const yl = svg("text", { x: 12, y: (H - m.b) / 2, transform: `rotate(-90 12 ${(H - m.b) / 2})`, "text-anchor": "middle", class: "lbltxt" }); yl.textContent = ylabel; s.appendChild(yl);
  series.forEach(se => {
    const d = se.x.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(se.y[i]).toFixed(1)}`).join("");
    const p = svg("path", { d, fill: "none", stroke: se.color || COLORS[se.name] || "#fff", "stroke-width": isSage(se.name) ? 2.8 : 1.6,
      "stroke-dasharray": DASH[se.name] || "", "stroke-linejoin": "round", opacity: isSage(se.name) || series.length < 4 ? 1 : 0.85 });
    s.appendChild(p);
    if (se.dots) se.x.forEach((x, i) => s.appendChild(svg("circle", { cx: X(x), cy: Y(se.y[i]), r: isSage(se.name) ? 3.5 : 2.5, fill: se.color || COLORS[se.name] })));
    if (animate && !DASH[se.name]) { const L = p.getTotalLength(); p.style.strokeDasharray = L; p.style.strokeDashoffset = L;
      p.getBoundingClientRect(); p.style.transition = "stroke-dashoffset 2.2s ease-out"; p.style.strokeDashoffset = 0; }
  });
  const cross = svg("line", { y1: m.t, y2: H - m.b, stroke: "#4b5a7a", "stroke-dasharray": "3 3", opacity: 0 }); s.appendChild(cross);
  const ov = svg("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }); s.appendChild(ov);
  ov.addEventListener("mousemove", ev => {
    const r = s.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
    const ref = series[0]; let bi = 0, bd = 1e9; ref.x.forEach((x, i) => { const d = Math.abs(X(x) - px); if (d < bd) { bd = d; bi = i; } });
    cross.setAttribute("x1", X(ref.x[bi])); cross.setAttribute("x2", X(ref.x[bi])); cross.setAttribute("opacity", 1);
    const rows = series.map(se => [se.name, se.y[bi]]).filter(r => r[1] != null).sort((a, b) => b[1] - a[1]);
    showTip(ev, `<b>${xfmt(ref.x[bi])}</b>` + rows.map(([n, v]) => `<div class="r"><span style="color:${COLORS[n] || "#fff"}">${n}</span><span>${yfmt(v)}</span></div>`).join(""));
  });
  ov.addEventListener("mouseleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
  host.appendChild(s);
  const lg = el("div", { class: "legend" }); series.forEach(se => lg.appendChild(el("span", {}, `<i style="background:${se.color || COLORS[se.name]}"></i>${se.name}`))); host.appendChild(lg);
}

/* ---------------------------------------------------------------- bar chart (horizontal) */
function hbars(host, rows, { fmt = v => v.toFixed(3), lowerBetter = true, unit = "" } = {}) {
  host.innerHTML = "";
  const max = Math.max(...rows.map(r => r[1])), min = Math.min(...rows.map(r => r[1]));
  const sorted = [...rows].sort((a, b) => lowerBetter ? a[1] - b[1] : b[1] - a[1]);
  sorted.forEach(([name, v, extra], i) => {
    const row = el("div", { class: "board-row" + (isSage(name) ? " me" : "") });
    const w = lowerBetter ? 15 + 85 * (v - min * 0.9) / (max - min * 0.9 || 1) : 8 + 92 * v / (max || 1);
    row.innerHTML = `<span class="rk">${i + 1}</span><span style="color:${COLORS[name] || "#fff"};font-weight:${isSage(name) ? 750 : 500}">${name}</span>
      <div class="bar"><div style="width:0;background:${COLORS[name] || "#888"}"></div></div><span class="val">${fmt(v)}${unit}</span>`;
    if (extra) row.title = extra;
    host.appendChild(row);
    requestAnimationFrame(() => setTimeout(() => row.querySelector(".bar>div").style.width = Math.max(3, w) + "%", 30 + i * 40));
  });
}

/* ---------------------------------------------------------------- grouped vertical bars */
function vbars(host, { cats, series, yfmt = v => v.toFixed(1), height = 260, ylabel = "", zero = true }) {
  host.innerHTML = "";
  const W = Math.max(320, host.clientWidth || 600), H = height, m = { l: 50, r: 10, t: 10, b: 54 };
  const vals = series.flatMap(s => s.values); let y0 = zero ? Math.min(0, ...vals) : Math.min(...vals) * 0.97, y1 = Math.max(...vals) * 1.08;
  if (y1 <= y0) y1 = y0 + 1;
  const Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, height: H }); const g = svg("g", { class: "axis" }); s.appendChild(g);
  niceTicks(y0, y1, 5).forEach(v => { g.appendChild(svg("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "gridl" }));
    const t = svg("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }); t.textContent = yfmt(v); g.appendChild(t); });
  const cw = (W - m.l - m.r) / cats.length, bw = Math.min(28, cw * 0.8 / series.length);
  cats.forEach((c, ci) => {
    const t = svg("text", { x: m.l + cw * (ci + 0.5), y: H - m.b + 14, "text-anchor": cats.length > 6 ? "end" : "middle",
      transform: cats.length > 6 ? `rotate(-35 ${m.l + cw * (ci + 0.5)} ${H - m.b + 14})` : "" }); t.textContent = c; g.appendChild(t);
    series.forEach((se, si) => {
      const v = se.values[ci]; if (v == null) return;
      const x = m.l + cw * ci + (cw - bw * series.length) / 2 + si * bw, y = Math.min(Y(v), Y(0)), h = Math.abs(Y(v) - Y(Math.max(0, y0)));
      const col = se.colorFn ? se.colorFn(v) : (se.color || COLORS[se.name]);
      const r = svg("rect", { x: x + 1, y: Y(Math.max(0, y0)), width: bw - 2, height: 0, rx: 3, fill: col });
      r.style.transition = "all .7s cubic-bezier(.2,.8,.2,1)"; s.appendChild(r);
      requestAnimationFrame(() => setTimeout(() => { r.setAttribute("y", y); r.setAttribute("height", Math.max(1, h)); }, 20 + ci * 25));
      r.addEventListener("mousemove", ev => showTip(ev, `<b>${c}</b><div class="r"><span>${se.name}</span><span>${yfmt(v)}</span></div>`));
      r.addEventListener("mouseleave", hideTip);
    });
  });
  if (ylabel) { const yl = svg("text", { x: 12, y: (H - m.b) / 2, transform: `rotate(-90 12 ${(H - m.b) / 2})`, "text-anchor": "middle", class: "lbltxt" }); yl.textContent = ylabel; s.appendChild(yl); }
  host.appendChild(s);
  if (series.length > 1) { const lg = el("div", { class: "legend" }); series.forEach(se => lg.appendChild(el("span", {}, `<i style="background:${se.color || COLORS[se.name]}"></i>${se.name}`))); host.appendChild(lg); }
}

/* ---------------------------------------------------------------- state & nav */
let META = null, R = null;
function go(page) {
  document.querySelectorAll(".page").forEach(p => p.classList.toggle("active", p.id === "page-" + page));
  document.querySelectorAll("nav a").forEach(a => a.classList.toggle("active", a.dataset.page === page));
  if (location.hash !== "#" + page) history.replaceState(null, "", "#" + page);
  scrollTo(0, 0);
  setTimeout(() => RENDER[page] && RENDER[page](), 10);
}
addEventListener("hashchange", () => META && go(location.hash.slice(1) || "overview"));
document.querySelectorAll("nav a, [data-goto]").forEach(a => a.addEventListener("click", e => { e.preventDefault(); go(a.dataset.page || a.dataset.goto); }));

/* ---------------------------------------------------------------- derived numbers */
function missBySize() {
  const sizes = [...new Set(R.miss.map(r => r.size))].sort((a, b) => a - b);
  return { sizes, mean: p => sizes.map(s => mean(R.miss.filter(r => r.size === s && r[p] != null).map(r => r[p]))) };
}
function reduction(base, ours = "SAGE", filt = () => true) {
  const rows = R.miss.filter(r => r[base] != null && r[ours] != null && filt(r));
  const red = rows.map(r => (r[base] - r[ours]) / r[base]);
  return { mean: mean(red), wins: red.filter(v => v > 0).length, n: red.length };
}
function tierPairs() { return [...new Set(R.tiers.map(t => t.dram_frac))].sort((a, b) => a - b).map(d => [d, R.tiers.find(t => t.dram_frac === d).ssd_frac]); }
function tierStats(d) {
  const rows = R.tiers.filter(t => t.dram_frac === d), traces = [...new Set(rows.map(r => r.trace))];
  const bases = ["LRU-cascade", "ARC-cascade", "S4-FIFO-cascade", "SR-S3-FIFO-cascade"];
  const per = traces.map(tr => {
    const g = Object.fromEntries(rows.filter(r => r.trace === tr).map(r => [r.policy, r]));
    const best = bases.filter(b => g[b]).reduce((a, b) => g[a].mean_latency_ms <= g[b].mean_latency_ms ? a : b);
    return { trace: tr, best, lat: (g[best].mean_latency_ms - g["SAGE-Unified"].mean_latency_ms) / g[best].mean_latency_ms,
      wr: (g[best].ssd_writes_per_kreq - g["SAGE-Unified"].ssd_writes_per_kreq) / g[best].ssd_writes_per_kreq };
  });
  return { rows, per };
}

/* ---------------------------------------------------------------- pages */
const RENDER = {};
RENDER.overview = () => {
  if (!R.miss) return;
  const pairs = tierPairs(); const all = pairs.flatMap(([d]) => tierStats(d).per);
  const small = reduction("S4-FIFO", "SAGE", r => r.size <= 0.01);
  const sr = reduction("SR-S3-FIFO", "SAGE", r => r.size <= 0.01);
  const kp = [
    [`${all.filter(p => p.lat > 0).length}<small>/${all.length}</small>`, "3-tier settings where SAGE has the lowest latency", "vs best of S4-FIFO, SR-S3-FIFO, ARC, LRU cascades"],
    [`−${(mean(all.map(p => p.wr)) * 100).toFixed(0)}<small>%</small>`, "SSD writes vs best baseline", "longer flash lifetime"],
    [`−${(small.mean * 100).toFixed(1)}<small>%</small>`, "miss ratio vs S4-FIFO (OSDI'26), small caches", `wins ${small.wins}/${small.n} at 0.5–1%`],
    [`−${(sr.mean * 100).toFixed(1)}<small>%</small>`, "miss ratio vs SR-Gadgets (2026), small caches", `wins ${sr.wins}/${sr.n} at 0.5–1%`],
  ];
  $("#kpis").innerHTML = kp.map(([v, l, t]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div><div class="t">${t}</div></div>`).join("");
  const { sizes, mean: mm } = missBySize();
  lineChart($("#ov-miss"), { series: ["Belady", "SAGE", "S4-FIFO", "SR-S3-FIFO", "ARC", "LRU"].map(p => ({ name: p, x: sizes.map(s => s * 100), y: mm(p), dots: true })),
    xlog: true, xticks: sizes.map(s => s * 100), xfmt: v => v + "%", xlabel: "cache size (% of distinct blocks)", ylabel: "miss ratio", yfmt: v => v.toFixed(2) });
  const pol = ["SAGE-Unified", "S4-FIFO-cascade", "SR-S3-FIFO-cascade", "ARC-cascade", "LRU-cascade"];
  const rows = R.tiers.filter(t => t.dram_frac === 0.01);
  hbars($("#ov-writes"), pol.map(p => [p, mean(rows.filter(r => r.policy === p).map(r => r.ssd_writes_per_kreq))]), { fmt: v => v.toFixed(0) });
  heroArt();
};

function heroArt() {
  const host = $("#hero-art"); if (host.dataset.done) return; host.dataset.done = 1;
  const s = svg("svg", { viewBox: "0 0 360 230" });
  const tiers = [["DRAM", 40, "#ff7a1a", "0.1 µs"], ["SSD", 105, "#60a5fa", "100 µs"], ["HDD", 170, "#64748b", "5 ms"]];
  tiers.forEach(([n, y, c, l], i) => {
    s.appendChild(svg("rect", { x: 150, y, width: 190 - i * 0, height: 46, rx: 10, fill: c + "22", stroke: c, "stroke-width": 1.5 }));
    const t = svg("text", { x: 166, y: y + 28, fill: "#e6ebf5", "font-size": 14, "font-weight": 700 }); t.textContent = n; s.appendChild(t);
    const t2 = svg("text", { x: 326, y: y + 28, fill: "#8b97ad", "font-size": 11, "text-anchor": "end" }); t2.textContent = l; s.appendChild(t2);
  });
  s.appendChild(svg("rect", { x: 10, y: 92, width: 100, height: 56, rx: 12, fill: "#ff7a1a", opacity: .95 }));
  const t = svg("text", { x: 60, y: 117, fill: "#fff", "font-size": 13, "font-weight": 800, "text-anchor": "middle" }); t.textContent = "SAGE"; s.appendChild(t);
  const t3 = svg("text", { x: 60, y: 134, fill: "#fff", "font-size": 9.5, "text-anchor": "middle" }); t3.textContent = "survival S(t)"; s.appendChild(t3);
  [63, 128, 193].forEach(y => s.appendChild(svg("path", { d: `M110,120 C130,120 130,${y} 150,${y}`, stroke: "#ffb36b", fill: "none", "stroke-width": 1.5, "stroke-dasharray": "4 4" })));
  for (let k = 0; k < 9; k++) { const c = svg("circle", { r: 4, fill: ["#ff7a1a", "#60a5fa", "#64748b"][k % 3] });
    const am = svg("animateMotion", { dur: `${2.4 + (k % 3) * 0.4}s`, repeatCount: "indefinite", begin: `${k * 0.35}s`, path: `M110,120 C130,120 130,${[63, 128, 193][k % 3]} 150,${[63, 128, 193][k % 3]} L${250 + (k % 4) * 18},${[63, 128, 193][k % 3]}` });
    c.appendChild(am); s.appendChild(c); }
  host.appendChild(s);
}

RENDER.benchmark = () => {
  if (!R.miss) return;
  const { sizes, mean: mm } = missBySize();
  const cols = ["Belady", "SAGE", "SAGE-H", "S4-FIFO", "S4-FIFO-oracle", "SR-S3-FIFO", "SIEVE", "S3-FIFO", "ARC", "LeCaR", "LRU"];
  const means = Object.fromEntries(cols.map(c => [c, mm(c)]));
  const deploy = cols.filter(c => !["Belady", "S4-FIFO-oracle"].includes(c));
  const best = sizes.map((_, i) => Math.min(...deploy.map(c => means[c][i])));
  $("#bm-table").innerHTML = `<table><tr><th>Policy</th>${sizes.map(s => `<th>${s * 100}%</th>`).join("")}<th>vs SAGE (wins)</th></tr>` +
    cols.map(c => { const r = c === "SAGE" || c === "Belady" ? null : reduction(c);
      return `<tr class="${c === "SAGE" ? "me" : ""}"><td><span style="color:${COLORS[c]}">●</span> ${c}${c === "Belady" ? ' <span class="badge">bound</span>' : c === "S4-FIFO-oracle" ? ' <span class="badge">hindsight</span>' : ""}</td>` +
        means[c].map((v, i) => `<td class="${deploy.includes(c) && Math.abs(v - best[i]) < 1e-12 ? "best" : ""}">${v.toFixed(4)}</td>`).join("") +
        `<td>${r ? `${(r.mean * 100).toFixed(2)}% · ${r.wins}/${r.n}` : ""}</td></tr>`; }).join("") + "</table>";
  const traces = [...new Set(R.miss.map(r => r.trace))];
  const ts = $("#bm-trace"); if (!ts.options.length) { traces.forEach(t => ts.add(new Option(t, t))); ts.onchange = RENDER.benchmark; }
  const rows = R.miss.filter(r => r.trace === ts.value).sort((a, b) => a.size - b.size);
  lineChart($("#bm-curve"), { series: ["Belady", "SAGE", "S4-FIFO", "SR-S3-FIFO", "ARC", "LeCaR", "LRU"].map(p => ({ name: p, x: rows.map(r => r.size * 100), y: rows.map(r => r[p]), dots: true })),
    xlog: true, xticks: sizes.map(s => s * 100), xfmt: v => v + "%", xlabel: "cache size", ylabel: "miss ratio", yfmt: v => v.toFixed(2) });
  const bs = $("#bm-base"); if (!bs.options.length) { ["S4-FIFO", "SR-S3-FIFO", "S4-FIFO-oracle", "SIEVE", "S3-FIFO", "ARC", "LeCaR", "LRU"].forEach(b => bs.add(new Option(b, b))); bs.onchange = RENDER.benchmark; }
  const base = bs.value;
  $("#bm-heat").innerHTML = `<table class="heat"><tr><th>trace</th>${sizes.map(s => `<th>${s * 100}%</th>`).join("")}</tr>` + traces.map(tr =>
    `<tr><td style="text-align:left">${tr}</td>` + sizes.map(s => { const r = R.miss.find(x => x.trace === tr && x.size === s); const v = (r[base] - r.SAGE) / r[base];
      const a = Math.min(1, Math.abs(v) / 0.12); const c = v >= 0 ? `rgba(52,211,153,${0.15 + 0.6 * a})` : `rgba(248,113,113,${0.15 + 0.6 * a})`;
      return `<td style="background:${c}" title="${base} ${r[base].toFixed(4)} vs SAGE ${r.SAGE.toFixed(4)}">${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}</td>`; }).join("") + "</tr>").join("") + "</table>";
  const abl = [["SAGE", "full model"], ["SAGE-noCond", "no idle-time conditioning"], ["SAGE-noAdmit", "no admission control"], ["SAGE-objOnly", "no region/global features"], ["SAGE-H", "+ regret hedge"]];
  const base0 = mean(R.miss.map(r => r.SAGE));
  vbars($("#bm-abl"), { cats: abl.map(a => a[1]), series: [{ name: "Δ miss ratio vs SAGE (percentage points)", values: abl.map(([a]) => (mean(R.miss.map(r => r[a])) - base0) * 100),
    colorFn: v => v > 0.05 ? "#f87171" : v < -0.05 ? "#34d399" : "#8b97ad" }], yfmt: v => (v >= 0 ? "+" : "") + v.toFixed(2), height: 230, zero: true, ylabel: "pp worse →" });
};

let tierPair = null;
RENDER.tiering = () => {
  if (!R.tiers) return;
  const pairs = tierPairs(); if (tierPair == null) tierPair = 0.01;
  const seg = $("#tier-pair"); seg.innerHTML = pairs.map(([d, s]) => `<button data-v="${d}" class="${d === tierPair ? "on" : ""}">DRAM ${d * 100}% · SSD ${s * 100}%</button>`).join("");
  seg.querySelectorAll("button").forEach(b => b.onclick = () => { tierPair = +b.dataset.v; RENDER.tiering(); });
  const { rows, per } = tierStats(tierPair);
  const pol = ["SAGE-Unified", "S4-FIFO-cascade", "SR-S3-FIFO-cascade", "ARC-cascade", "LRU-cascade", "SAGE-L1+LRU-SSD"];
  const avg = (p, k) => mean(rows.filter(r => r.policy === p).map(r => r[k]));
  hbars($("#tier-lat"), pol.map(p => [p, avg(p, "mean_latency_ms")]), { fmt: v => v.toFixed(3), unit: " ms" });
  hbars($("#tier-wr"), pol.map(p => [p, avg(p, "ssd_writes_per_kreq")]), { fmt: v => v.toFixed(0) });
  $("#tier-sum").innerHTML = `SAGE-Unified faster in <b style="color:#34d399">${per.filter(p => p.lat > 0).length}/${per.length}</b> traces · avg latency −${(mean(per.map(p => p.lat)) * 100).toFixed(1)}% · SSD writes −${(mean(per.map(p => p.wr)) * 100).toFixed(0)}%`;
  vbars($("#tier-per"), { cats: per.map(p => p.trace), series: [
    { name: "latency reduction %", values: per.map(p => p.lat * 100), colorFn: v => v >= 0 ? "#ff7a1a" : "#64748b" },
    { name: "SSD-write reduction %", values: per.map(p => p.wr * 100), color: "#60a5fa" }], yfmt: v => v.toFixed(0) + "%", height: 280 });
  const sp = ["SAGE-Unified", "S4-FIFO-cascade", "SR-S3-FIFO-cascade", "ARC-cascade", "LRU-cascade"];
  vbars($("#tier-stack"), { cats: sp, series: [
    { name: "DRAM hit", values: sp.map(p => avg(p, "dram_hit") * 100), color: "#ff7a1a" },
    { name: "SSD hit", values: sp.map(p => avg(p, "ssd_hit") * 100), color: "#60a5fa" },
    { name: "HDD (miss)", values: sp.map(p => avg(p, "hdd_miss") * 100), color: "#64748b" }], yfmt: v => v.toFixed(0) + "%", height: 240 });
};

RENDER.model = () => {
  const host = $("#pipeline"); if (!host.innerHTML) {
    host.innerHTML = `<div class="steps">
      <div class="step"><b>Request x<sub>i</sub></b><p>A block read/write from the real trace.</p></div>
      <div class="step"><b>Causal features (30)</b><p>8 past gaps, access count, 6 decayed counters, size, 1 MiB region stats, global first-touch rate. Computed from the past only, independent of cache state.</p></div>
      <div class="step"><b>Hazard GBDT</b><p>One LightGBM classifier with the time bin as input → hazard h<sub>k</sub> for 12 bins (16 … 1.5M requests). Labels right-censored at the training cut-off.</p></div>
      <div class="step"><b>Survival curve → placement</b><p>S(t)=∏(1−h<sub>k</sub>). Conditional score per tier drives eviction, admission, demotion, bypass.</p></div></div>`;
  }
  if (R.importance) hbars($("#imp"), R.importance.slice(0, 12), { fmt: v => (v * 100).toFixed(1) + "%", lowerBetter: false });
  $("#params").innerHTML = [["Model", "LightGBM binary (discrete-time hazard)"], ["Rounds / leaves / lr", "150 / 63 / 0.08"], ["Min leaf, L2", "200, 1.0"],
    ["Time bins", "12, edges 2^(4+1.5k) requests"], ["Training", "120K samples from last 1M requests"], ["Retraining", "9× per trace, past data only"],
    ["Eviction sample", "64 blocks"], ["Horizon multiplier h", "0.5 (tuned on 2 dev traces)"], ["Warm-up / measured", "20% / 80% of 2M requests"]]
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
};

RENDER.data = () => {
  $("#traces").innerHTML = `<table><tr><th>Trace</th><th>Source</th><th>Role</th><th>Total requests</th><th>SAGE model</th></tr>` +
    META.traces.map(t => `<tr><td>${t.name}</td><td>${t.family}</td><td><span class="badge ${t.role}">${t.role}</span></td><td>${(t.requests_total / 1e6).toFixed(1)}M</td><td>${t.sage_model ? "✓ trained" : "—"}</td></tr>`).join("") + "</table>";
  const s4 = R.miss ? reduction("S4-FIFO") : null, s4s = R.miss ? reduction("S4-FIFO", "SAGE", r => r.size <= 0.01) : null;
  const sr = R.miss ? reduction("SR-S3-FIFO") : null, srs = R.miss ? reduction("SR-S3-FIFO", "SAGE", r => r.size <= 0.01) : null;
  const papers = [
    ["S4-FIFO", "OSDI 2026 (arXiv 2608.27975)", "Xia, Nixon, Marthen, Bhandari, Yang. A learned model picks one of many S3-FIFO-style queue configurations per workload.",
      s4 && `Single cache: SAGE ${(s4.mean * 100).toFixed(1)}% lower miss ratio overall (${s4.wins}/${s4.n} wins), <b>${(s4s.mean * 100).toFixed(1)}% lower at 0.5–1%</b> (${s4s.wins}/${s4s.n}). Tiering: SAGE-Unified faster with far fewer SSD writes.`],
    ["SR-Gadgets", "2026 (arXiv 2609.30468)", "Zheng & Yang. Adds LIRS-style scan resistance (ProbBypass, RecencyGuard) to S3-FIFO, ARC, SIEVE…",
      sr && `SR-S3-FIFO: SAGE ${(sr.mean * 100).toFixed(1)}% lower overall (${sr.wins}/${sr.n}), <b>${(srs.mean * 100).toFixed(1)}% lower at 0.5–1%</b> (${srs.wins}/${srs.n}).`],
    ["Earlier baselines", "ARC (FAST'03) · LeCaR (HotStorage'18) · S3-FIFO (SOSP'23) · SIEVE (NSDI'24)", "Classic adaptive, regret-learning and FIFO-family policies.",
      R.miss && `SAGE beats ARC ${reduction("ARC").wins}/44, LeCaR ${reduction("LeCaR").wins}/44, S3-FIFO ${reduction("S3-FIFO").wins}/44, SIEVE ${reduction("SIEVE").wins}/44 configurations.`],
    ["Honest limits", "where SAGE loses", "At 5–10% cache size on several FIU traces, S4-FIFO's large ghost queue beats SAGE as a single cache. Region features did not help in the ablation.",
      "Reported openly in the paper's limitations section."],
  ];
  $("#papers").innerHTML = papers.map(([t, v, d, vs]) => `<div class="card paper"><h3>${t}<span class="venue">${v}</span></h3><p>${d}</p>${vs ? `<div class="vs">${vs}</div>` : ""}</div>`).join("");
};

/* ---------------------------------------------------------------- simulator */
const SIM = { view: "cum", mode: "cache", size: 0.01, req: 2000000, sel: new Set(["SAGE", "S4-FIFO", "SR-S3-FIFO", "ARC", "Belady"]),
  selTier: new Set(["SAGE-Unified", "S4-FIFO-cascade", "SR-S3-FIFO-cascade", "ARC-cascade", "LRU-cascade"]), job: null, last: null };
function simInit() {
  const ts = $("#sim-trace");
  META.traces.forEach(t => { const o = new Option(`${t.name}  ·  ${t.role}${t.sage_model ? "" : " (no SAGE model)"}`, t.name); ts.add(o); });
  ts.value = (META.traces.find(t => t.name === "msr_hm_0") || META.traces[0]).name;
  const desc = () => { const t = META.traces.find(x => x.name === ts.value); $("#trace-desc").textContent = `${t.family} · ${(t.requests_total / 1e6).toFixed(1)}M requests in file`; };
  ts.onchange = desc; desc();
  const sizes = $("#sim-size"); META.sizes.forEach(s => { const b = el("button", { "data-v": s }, (s * 100) + "%"); if (s === SIM.size) b.classList.add("on"); sizes.appendChild(b); });
  sizes.onclick = e => { if (e.target.dataset.v) { SIM.size = +e.target.dataset.v; sizes.querySelectorAll("button").forEach(b => b.classList.toggle("on", b === e.target)); } };
  $("#sim-mode").onclick = e => { if (!e.target.dataset.v) return; SIM.mode = e.target.dataset.v; $("#sim-mode").querySelectorAll("button").forEach(b => b.classList.toggle("on", b === e.target)); drawPols();
    $("#size-label").textContent = SIM.mode === "cache" ? "Cache size" : "DRAM size (SSD = 5× DRAM)"; };
  $("#sim-req").onclick = e => { if (!e.target.dataset.v) return; SIM.req = +e.target.dataset.v; $("#sim-req").querySelectorAll("button").forEach(b => b.classList.toggle("on", b === e.target)); };
  drawPols();
  $("#sim-run").onclick = runSim;
  $("#sim-replay").onclick = () => SIM.last && drawSim(SIM.last, true);
  $("#sim-view").onclick = e => { if (!e.target.dataset.v) return; SIM.view = e.target.dataset.v;
    $("#sim-view").querySelectorAll("button").forEach(b => b.classList.toggle("on", b === e.target)); SIM.last && drawSim(SIM.last, true); };
}
function drawPols() {
  const list = SIM.mode === "cache" ? META.cache_policies : META.tier_policies, sel = SIM.mode === "cache" ? SIM.sel : SIM.selTier;
  const host = $("#sim-pols"); host.innerHTML = "";
  list.forEach(p => { const l = el("label", { class: "pol" + (sel.has(p) ? " on" : "") }, `<span class="dot" style="background:${COLORS[p]}"></span>${p}`);
    l.onclick = () => { sel.has(p) ? sel.delete(p) : sel.add(p); l.classList.toggle("on"); }; host.appendChild(l); });
}
async function runSim() {
  const sel = [...(SIM.mode === "cache" ? SIM.sel : SIM.selTier)]; if (!sel.length) return;
  const body = { trace: $("#sim-trace").value, mode: SIM.mode, size: SIM.size, requests: SIM.req, policies: sel };
  $("#sim-run").disabled = true; $("#sim-replay").disabled = true;
  $("#sim-chart").innerHTML = '<div class="empty"><div style="text-align:center"><div class="spin" style="margin:0 auto 10px;width:22px;height:22px"></div>Replaying ' + (SIM.req / 1e6) + 'M real requests…</div></div>';
  $("#sim-board").innerHTML = ""; $("#sim-extra-title").textContent = "Progress";
  const r = await fetch("/api/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(r => r.json());
  if (r.error) { $("#sim-status").textContent = r.error; $("#sim-run").disabled = false; return; }
  SIM.job = r.job; const t0 = Date.now();
  const poll = async () => {
    const j = await fetch("/api/job/" + SIM.job).then(r => r.json());
    $("#sim-status").textContent = `${Object.keys(j.results).length}/${j.policies.length} done · ${((Date.now() - t0) / 1000).toFixed(0)} s`;
    $("#sim-extra").innerHTML = j.policies.map(p => { const x = j.results[p];
      return `<div class="prog">${x ? (x.error ? "⚠️" : "✅") : '<div class="spin"></div>'}<span style="color:${COLORS[p]}">${p}</span><span class="st">${x ? (x.error ? "error" : x.seconds.toFixed(1) + " s") : "running"}</span></div>`; }).join("");
    if (Object.keys(j.results).length) drawSim(j, false);
    if (!j.done) setTimeout(poll, 700); else { SIM.last = j; drawSim(j, true); $("#sim-run").disabled = false; $("#sim-replay").disabled = false; }
  };
  poll();
}
function drawSim(j, animate) {
  const ok = Object.values(j.results).filter(x => !x.error);
  if (!ok.length) { $("#sim-chart").innerHTML = `<div class="empty">${Object.values(j.results)[0].error}</div>`; return; }
  const tier = j.mode === "tier";
  $("#sim-chart-title").textContent = tier ? "Fast-tier hit ratio over time (DRAM + SSD)" : "Hit ratio over time";
  $("#sim-chart-sub").textContent = `${j.trace} · ${tier ? `DRAM ${j.size * 100}% + SSD ${j.size * 500}%` : `cache ${j.size * 100}%`} of ${ok[0].objects.toLocaleString()} distinct blocks · ${(ok[0].requests / 1e6).toFixed(1)}M requests`;
  const cum = a => { let t = 0; return a.map((v, i) => (t += v) / (i + 1)); };
  const yv = x => SIM.view === "cum" ? cum(x.windows.hit) : x.windows.hit;
  $("#sim-chart-sub").textContent += SIM.view === "cum" ? " · cumulative hit ratio since warm-up" : " · hit ratio per window";
  lineChart($("#sim-chart"), { series: ok.map(x => ({ name: x.policy, x: x.windows.x, y: yv(x) })), xlabel: "request number", ylabel: "hit ratio",
    xfmt: v => (v / 1e6).toFixed(2) + "M", yfmt: v => pct(v, 0), animate, height: 340 });
  if (!tier) {
    $("#sim-board-title").textContent = "Leaderboard · miss ratio (lower = better)";
    hbars($("#sim-board"), ok.map(x => [x.policy, x.miss_ratio, x.note]), { fmt: v => pct(v, 2) });
    const sage = ok.find(x => x.policy === "SAGE"), others = ok.filter(x => !isSage(x.policy) && x.policy !== "Belady");
    if (sage && others.length && j.done) { const best = others.reduce((a, b) => a.miss_ratio < b.miss_ratio ? a : b), d = (best.miss_ratio - sage.miss_ratio) / best.miss_ratio;
      $("#sim-board").appendChild(el("div", { class: "callout" + (d < 0 ? " bad" : "") }, d >= 0 ? `SAGE has a <b>${(d * 100).toFixed(1)}% lower</b> miss ratio than the best baseline here (${best.policy}).`
        : `${best.policy} beats SAGE by ${(-d * 100).toFixed(1)}% on this setting (expected for some large caches; see limitations).`)); }
  } else {
    $("#sim-board-title").textContent = "Mean access latency (lower = better)";
    hbars($("#sim-board"), ok.map(x => [x.policy, x.mean_latency_ms]), { fmt: v => v.toFixed(3), unit: " ms" });
    if (j.done) { $("#sim-extra-title").textContent = "SSD writes per 1K requests";
      const h = el("div"); $("#sim-extra").innerHTML = ""; $("#sim-extra").appendChild(h); hbars(h, ok.map(x => [x.policy, x.ssd_writes_per_kreq]), { fmt: v => v.toFixed(0) });
      const sage = ok.find(x => x.policy === "SAGE-Unified"), others = ok.filter(x => x.policy !== "SAGE-Unified");
      if (sage && others.length) { const best = others.reduce((a, b) => a.mean_latency_ms < b.mean_latency_ms ? a : b);
        const dl = (best.mean_latency_ms - sage.mean_latency_ms) / best.mean_latency_ms, dw = (best.ssd_writes_per_kreq - sage.ssd_writes_per_kreq) / best.ssd_writes_per_kreq;
        $("#sim-extra").appendChild(el("div", { class: "callout" + (dl < 0 ? " bad" : "") }, `vs best baseline (${best.policy}): latency ${dl >= 0 ? "−" : "+"}${Math.abs(dl * 100).toFixed(1)}%, SSD writes ${dw >= 0 ? "−" : "+"}${Math.abs(dw * 100).toFixed(0)}%.`)); } }
  }
}

/* ---------------------------------------------------------------- boot */
(async () => {
  [META, R] = await Promise.all([fetch("/api/meta").then(r => r.json()), fetch("/api/results").then(r => r.json())]);
  simInit();
  go((location.hash || "#overview").slice(1));
  let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { const p = document.querySelector(".page.active").id.slice(5); RENDER[p] && RENDER[p](); if (p === "simulator" && SIM.last) drawSim(SIM.last, false); }, 200); });
})();
