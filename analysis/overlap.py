#!/usr/bin/env python3
"""Phase 0, tasks 0.6-0.7 — aggregate mask statistics and render the H1 verdict.

Reads the committed summary stats produced by `dump_masks.py measure` and emits the
tables in `analysis/report.md`. Deterministic: same summary in, same bytes out.

    uv run analysis/overlap.py --check            # self-tests, exit 0 on success
    uv run analysis/overlap.py --summary S.json   # print tables
    uv run analysis/overlap.py --summary S.json --report analysis/report.md
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np

K_MAX = 4
CLASS_ORDER = ("attn-qkv", "attn-o", "deltanet-in", "deltanet-out",
               "ffn-up-gate", "ffn-down")
COARSE_ORDER = ("attn-QKVO", "ffn-up-gate", "ffn-down", "deltanet-proj")

# H1, verbatim from docs/03-experiment-plan.md
H1_PASS_50 = 0.30      # union sparsity at k=4, per-token target 50%
H1_PASS_40 = 0.20      # union sparsity at k=4, per-token target 40%
H1_KILL_40 = 0.20      # below this at k=4 / 40% -> C1 dead


# --------------------------------------------------------------------------
def weights(sites):
    """Bytes saved by skipping one input column = dim x Σ(consumer output dims)."""
    return np.array([s["dim"] * s["out_sum"] for s in sites], np.float64)


def wmean(v, w, mask=None):
    if mask is not None:
        v, w = v[mask], w[mask]
    return float((v * w).sum() / max(w.sum(), 1e-9))


def pooled(summary, group, target, key):
    d = summary["groups"][group][f"{target:.2f}"]
    return np.array(d[key], np.float64)


def class_table(summary, group, target, attr, order):
    sites = summary["sites"]
    w = weights(sites)
    rows = []
    for c in order:
        m = np.array([s[attr] == c for s in sites])
        if not m.any():
            continue
        row = {"class": c, "n_sites": int(m.sum()),
               "dim": int(sites[int(np.argmax(m))]["dim"])}
        row["per_token"] = wmean(pooled(summary, group, target,
                                        "per_token_sparsity"), w, m)
        for k in range(2, K_MAX + 1):
            row[f"k{k}"] = wmean(pooled(summary, group, target,
                                        f"union_sparsity_k{k}"), w, m)
        row["jaccard"] = wmean(pooled(summary, group, target, "jaccard_adj"), w, m)
        rows.append(row)
    return rows


def model_wide(summary, group, target):
    sites = summary["sites"]
    w = weights(sites)
    out = {"per_token": wmean(pooled(summary, group, target,
                                     "per_token_sparsity"), w),
           "jaccard": wmean(pooled(summary, group, target, "jaccard_adj"), w)}
    for k in range(2, K_MAX + 1):
        out[f"k{k}"] = wmean(pooled(summary, group, target,
                                    f"union_sparsity_k{k}"), w)
    return out


def per_trace_stats(summary, group, target, key):
    """Between-trace mean / sd / sem of a model-wide statistic."""
    v = [t["agg"][f"{target:.2f}"]["model_wide"][key]
         for t in summary["traces"] if t["kind"] == group]
    if not v:
        return None
    a = np.array(v, np.float64)
    n = len(a)
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    return {"n": n, "mean": float(a.mean()), "sd": sd,
            "sem": sd / math.sqrt(n) if n else 0.0,
            "min": float(a.min()), "max": float(a.max())}


# --------------------------------------------------------------------------
def md_table(headers, rows, fmt=None):
    fmt = fmt or {}
    cells = [[str(h) for h in headers]]
    for r in rows:
        cells.append([fmt.get(h, str)(r[i]) if not callable(r[i]) else r[i]
                      for i, h in enumerate(headers)])
    widths = [max(len(c[i]) for c in cells) for i in range(len(headers))]
    out = ["| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells[0])) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in cells[1:]:
        out.append("| " + " | ".join(c.ljust(widths[i])
                                     for i, c in enumerate(row)) + " |")
    return "\n".join(out)


def pct(x):
    return f"{100 * x:.1f}%"


def render(summary):
    L = []
    targets = summary["targets"]
    sites = summary["sites"]
    A = lambda g, t: model_wide(summary, g, t)

    # ---- verdict block (<=10 lines, top of the report) ---------------------
    mw50 = A("agentic", 0.50)
    mw40 = A("agentic", 0.40)
    s50 = per_trace_stats(summary, "agentic", 0.50, "union_sparsity_k4")
    s40 = per_trace_stats(summary, "agentic", 0.40, "union_sparsity_k4")
    p50 = mw50["k4"] >= H1_PASS_50
    p40 = mw40["k4"] >= H1_PASS_40
    kill = mw40["k4"] < H1_KILL_40
    # "with margin" = the gate sits outside a 2-SEM band around the estimate
    margin50 = (mw50["k4"] - H1_PASS_50) / s50["sem"] if s50 and s50["sem"] > 0 else float("inf")
    margin40 = (mw40["k4"] - H1_PASS_40) / s40["sem"] if s40 and s40["sem"] > 0 else float("inf")
    if kill:
        verdict = "**H1 FALSIFIED — kill criterion met. C1 is dead.**"
    elif p50 and p40 and min(margin50, margin40) >= 2.0:
        verdict = "**H1 HOLDS with margin.**"
    elif p50 and p40:
        verdict = "**H1 holds but WITHIN NOISE — stop and request review.**"
    else:
        verdict = "**H1 FAILS (above the kill floor) — stop and request review.**"

    L += [
        "## Phase 0 gate — H1",
        "",
        verdict,
        "",
        f"- union sparsity, k=4, per-token target 50%: **{pct(mw50['k4'])}** "
        f"(gate ≥30%; realized per-token {pct(mw50['per_token'])}; "
        f"{s50['n']} traces, sd {pct(s50['sd'])}, {margin50:+.1f} SEM from gate)",
        f"- union sparsity, k=4, per-token target 40%: **{pct(mw40['k4'])}** "
        f"(gate ≥20%; kill <20%; realized per-token {pct(mw40['per_token'])}; "
        f"sd {pct(s40['sd'])}, {margin40:+.1f} SEM from gate)",
        f"- adjacent-token Jaccard (target 50%): {mw50['jaccard']:.3f} agentic vs "
        f"{A('control', 0.50)['jaccard']:.3f} WikiText control",
        f"- next action: "
        + ("write up the negative result and archive." if kill else
           ("proceed to Phase 1 (calibration + quality)." if verdict.startswith("**H1 HOLDS")
            else "stop; report is complete; awaiting review.")),
        "",
        "All figures are **byte-weighted** model-wide means over the language trunk "
        "(a skipped input column saves `Σ consumer output dims` weight bytes); "
        "`lm_head` and the vision tower are excluded.",
        "",
    ]

    # ---- model-wide ---------------------------------------------------------
    L += ["## Model-wide union sparsity", ""]
    rows = []
    for g in ("agentic", "control"):
        for t in targets:
            mw = A(g, t)
            st = per_trace_stats(summary, g, t, "union_sparsity_k4")
            rows.append([g, f"{t:.0%}", pct(mw["per_token"]), pct(mw["k2"]),
                         pct(mw["k3"]), pct(mw["k4"]),
                         f"±{pct(st['sem'])}" if st else "—",
                         f"{mw['jaccard']:.3f}"])
    L += [md_table(["corpus", "target", "per-token", "k=2", "k=3", "k=4",
                    "k=4 sem", "Jaccard"], rows), ""]
    L += ["`target` is the calibration quantile; `per-token` is the sparsity actually "
          "realized on the measurement split under those **fixed** thresholds.", ""]

    # ---- per class ----------------------------------------------------------
    for t in targets:
        L += [f"## Per matrix class — agentic traces, per-token target {t:.0%}", ""]
        rows = [[r["class"], r["dim"], r["n_sites"], pct(r["per_token"]),
                 pct(r["k2"]), pct(r["k3"]), pct(r["k4"]), f"{r['jaccard']:.3f}"]
                for r in class_table(summary, "agentic", t, "cls", CLASS_ORDER)]
        L += [md_table(["class", "in-dim", "sites", "per-token", "k=2", "k=3",
                        "k=4", "Jaccard"], rows), ""]

    # ---- coarse (docs/03 four-class view) -----------------------------------
    L += ["## docs/03 four-class view — agentic, per-token target 50%", ""]
    rows = [[r["class"], r["n_sites"], pct(r["per_token"]), pct(r["k2"]),
             pct(r["k3"]), pct(r["k4"]), f"{r['jaccard']:.3f}"]
            for r in class_table(summary, "agentic", 0.50, "coarse", COARSE_ORDER)]
    L += [md_table(["class", "sites", "per-token", "k=2", "k=3", "k=4", "Jaccard"],
                   rows), ""]

    # ---- agentic vs control -------------------------------------------------
    L += ["## Agentic traces vs. WikiText control (k=4)", ""]
    rows = []
    for t in targets:
        a, c = A("agentic", t), A("control", t)
        rows.append([f"{t:.0%}", pct(a["per_token"]), pct(a["k4"]),
                     pct(c["per_token"]), pct(c["k4"]),
                     f"{100 * (a['k4'] - c['k4']):+.1f} pp"])
    L += [md_table(["target", "agentic per-token", "agentic k=4",
                    "control per-token", "control k=4", "Δ k=4"], rows), ""]

    # ---- depth profile ------------------------------------------------------
    L += ["## Union sparsity by depth (k=4, target 50%, agentic)", ""]
    w = weights(sites)
    v = pooled(summary, "agentic", 0.50, "union_sparsity_k4")
    rows = []
    nl = max(s["layer"] for s in sites) + 1
    for lo in range(0, nl, 8):
        m = np.array([lo <= s["layer"] < lo + 8 for s in sites])
        ml = np.array([lo <= s["layer"] < lo + 8 and s["cls"].startswith("deltanet")
                       for s in sites])
        mf = np.array([lo <= s["layer"] < lo + 8 and s["cls"].startswith("ffn")
                       for s in sites])
        rows.append([f"{lo}-{min(lo + 7, nl - 1)}", pct(wmean(v, w, m)),
                     pct(wmean(v, w, ml)) if ml.any() else "—",
                     pct(wmean(v, w, mf)) if mf.any() else "—"])
    L += [md_table(["layers", "all", "deltanet", "ffn"], rows), ""]

    # ---- per-trace spread ---------------------------------------------------
    L += ["## Per-trace spread (k=4, target 50%, model-wide)", ""]
    rows = []
    for tr in summary["traces"]:
        a = tr["agg"]["0.50"]["model_wide"]
        rows.append([tr["id"], tr["kind"], tr["tokens"],
                     ",".join(tr["langs"])[:24], pct(a["per_token_sparsity"]),
                     pct(a["union_sparsity_k4"]), f"{a['jaccard_adj']:.3f}"])
    L += [md_table(["trace", "kind", "tokens", "langs", "per-token", "k=4",
                    "Jaccard"], rows), ""]

    return "\n".join(L), {"verdict": verdict, "kill": kill,
                          "mw50_k4": mw50["k4"], "mw40_k4": mw40["k4"],
                          "margin50": margin50, "margin40": margin40}


# --------------------------------------------------------------------------
def synth_summary(seed=0, n_sites=8, n_traces=3):
    """Deterministic fake summary for --check."""
    rng = np.random.default_rng(seed)
    sites = []
    for i in range(n_sites):
        cls = CLASS_ORDER[i % len(CLASS_ORDER)]
        coarse = {"attn-qkv": "attn-QKVO", "attn-o": "attn-QKVO",
                  "deltanet-in": "deltanet-proj", "deltanet-out": "deltanet-proj",
                  "ffn-up-gate": "ffn-up-gate", "ffn-down": "ffn-down"}[cls]
        sites.append({"idx": i, "layer": i, "cls": cls, "coarse": coarse,
                      "dim": 128 * (i + 1), "out_sum": 256,
                      "key": f"L{i:02d}.{cls}"})

    def dump():
        d = {"tokens": [64] * n_sites,
             "per_token_sparsity": list(rng.uniform(0.45, 0.55, n_sites)),
             "jaccard_adj": list(rng.uniform(0.5, 0.8, n_sites)),
             "jaccard_pairs": [63] * n_sites}
        for k in range(2, K_MAX + 1):
            d[f"union_sparsity_k{k}"] = list(rng.uniform(0.30, 0.42, n_sites))
            d[f"union_windows_k{k}"] = [64 - k + 1] * n_sites
        return d

    groups = {g: {f"{t:.2f}": dump() for t in (0.25, 0.40, 0.50)}
              for g in ("agentic", "control")}
    traces = []
    for i in range(n_traces):
        for kind in ("agentic", "control"):
            traces.append({
                "id": f"{kind}-{i:02d}", "kind": kind, "project": "synth",
                "langs": ["ts"], "tokens": 64,
                "agg": {f"{t:.2f}": {"model_wide": {
                    "per_token_sparsity": 0.5 + 0.01 * i,
                    "jaccard_adj": 0.6,
                    **{f"union_sparsity_k{k}": 0.35 + 0.005 * i
                       for k in range(2, K_MAX + 1)}}}
                    for t in (0.25, 0.40, 0.50)}})
    return {"kind": "phase0-summary", "model": "synthetic",
            "targets": [0.25, 0.40, 0.50], "k_max": K_MAX, "env": {},
            "sites": sites, "traces": traces, "groups": groups}


def check():
    ok = True

    def t(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}")

    s = synth_summary()
    md1, v1 = render(s)
    md2, v2 = render(json.loads(json.dumps(s)))
    t("render is deterministic", md1 == md2)
    t("verdict is deterministic", v1 == v2)

    w = weights(s["sites"])
    t("byte weights = dim x out_sum",
      np.allclose(w, [x["dim"] * x["out_sum"] for x in s["sites"]]))

    # weighted mean must sit inside the value range
    v = pooled(s, "agentic", 0.50, "union_sparsity_k4")
    t("weighted mean within range", v.min() - 1e-12 <= wmean(v, w) <= v.max() + 1e-12)

    # union sparsity can never exceed per-token sparsity, and must fall with k
    s2 = synth_summary(seed=7)
    for g in ("agentic", "control"):
        for tg in (0.25, 0.40, 0.50):
            d = s2["groups"][g][f"{tg:.2f}"]
            for k in range(2, K_MAX + 1):
                d[f"union_sparsity_k{k}"] = list(
                    np.array(d["per_token_sparsity"]) * (0.9 ** (k - 1)))
    md, _ = render(s2)
    mw = model_wide(s2, "agentic", 0.50)
    t("k=2 >= k=3 >= k=4 monotone", mw["k2"] >= mw["k3"] >= mw["k4"])
    t("union <= per-token", mw["k4"] <= mw["per_token"] + 1e-12)

    # gate logic at the boundaries
    def verdict_for(k4_50, k4_40, sd=0.0):
        ss = synth_summary(seed=3)
        for g in ("agentic", "control"):
            for tg, val in ((0.50, k4_50), (0.40, k4_40)):
                d = ss["groups"][g][f"{tg:.2f}"]
                d["union_sparsity_k4"] = [val] * len(ss["sites"])
                d["per_token_sparsity"] = [tg] * len(ss["sites"])
        seen = {}
        for tr in ss["traces"]:
            i = seen[tr["kind"]] = seen.get(tr["kind"], -1) + 1
            for tg, val in ((0.50, k4_50), (0.40, k4_40)):
                tr["agg"][f"{tg:.2f}"]["model_wide"]["union_sparsity_k4"] = \
                    val + sd * (1 if i % 2 else -1)
        return render(ss)[1]

    t("kill below 20% at 40%", verdict_for(0.35, 0.15)["kill"] is True)
    t("no kill at 25% / 40%", verdict_for(0.35, 0.25)["kill"] is False)
    t("pass with margin", verdict_for(0.40, 0.30)["verdict"].startswith(
        "**H1 HOLDS"))
    t("within-noise flagged", "WITHIN NOISE" in verdict_for(
        0.3005, 0.2005, sd=0.05)["verdict"])
    t("fail above kill floor flagged", "H1 FAILS" in verdict_for(
        0.25, 0.22)["verdict"])

    # md_table alignment
    tb = md_table(["a", "bb"], [["1", "2"], ["333", "4"]])
    t("md_table shape", tb.count("\n") == 3 and tb.startswith("| a  "))

    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--summary", default="analysis/summary/phase0.json")
    ap.add_argument("--report", default=None,
                    help="write the tables into this markdown file, between "
                         "<!--BEGIN TABLES--> and <!--END TABLES-->")
    a = ap.parse_args()
    if a.check:
        sys.exit(check())

    summary = json.loads(pathlib.Path(a.summary).read_text())
    md, v = render(summary)
    if a.report:
        p = pathlib.Path(a.report)
        B, E = "<!--BEGIN TABLES-->", "<!--END TABLES-->"
        old = p.read_text() if p.exists() else f"# Phase 0 report\n\n{B}\n{E}\n"
        head, _, rest = old.partition(B)
        _, _, tail = rest.partition(E)
        p.write_text(f"{head}{B}\n{md}\n{E}{tail}")
        print(f"wrote tables into {a.report}")
    else:
        print(md)
