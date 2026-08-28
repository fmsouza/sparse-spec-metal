#!/usr/bin/env python3
"""Phase 0, tasks 0.3-0.5 — instrumented teacher-forced decode over Qwen3.8-27B.

What it does
------------
Taps the *input activation* of every weight matrix in the language trunk and, at
TEAL-style magnitude thresholds, accumulates streaming statistics about which input
columns are skippable — per token, and for the union over k adjacent tokens.

Key definitions
---------------
site        A distinct input vector. Matrices consuming the same vector share one
            mask, which is what a sparse kernel actually exploits. 256 sites here
            (see SITE_CLASSES); lm_head and the vision tower are excluded.
threshold   Fixed per site, calibrated offline as the empirical quantile of |x| over
            a held-out calibration split, then held constant. Realized per-token
            sparsity therefore varies -- the honest setting, and the one a kernel
            faces.
union       A column is skippable for a k-token window iff it is inactive for ALL k
            tokens.  union_sparsity = 1 - |union of active sets| / D.
decode mode Prefix is prefilled in one pass with the tap OFF; target tokens are then
            fed one at a time (S=1) with the tap ON. No sampling.

Masks are never stored -- a rolling window of k_max tokens updates the accumulators.

Subcommands
-----------
  calibrate   build per-site thresholds from the calib split -> thresholds JSON
  measure     run the measurement split with fixed thresholds -> summary JSON
  selftest    tiny shape/logic checks that need no model
"""

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np

K_MAX = 4
TARGETS = (0.25, 0.40, 0.50)

# histogram grid for quantile estimation, in log2(|x|)
HIST_LO, HIST_HI, HIST_BINS = -24.0, 12.0, 4096

SITE_CLASSES = ("attn-qkv", "attn-o", "deltanet-in", "deltanet-out",
                "ffn-up-gate", "ffn-down")

# docs/03 reports four coarse classes; map the six sites onto them too
COARSE = {
    "attn-qkv": "attn-QKVO", "attn-o": "attn-QKVO",
    "deltanet-in": "deltanet-proj", "deltanet-out": "deltanet-proj",
    "ffn-up-gate": "ffn-up-gate", "ffn-down": "ffn-down",
}


# --------------------------------------------------------------------------
# environment capture (results/README.md requires this on every run)
# --------------------------------------------------------------------------
def env_info():
    def sh(*c):
        try:
            return subprocess.run(c, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return "?"
    import mlx.core as mx
    import mlx_lm
    return {
        "macos": f"{sh('sw_vers','-productVersion')} ({sh('sw_vers','-buildVersion')})",
        "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "ram_bytes": int(sh("sysctl", "-n", "hw.memsize") or 0),
        "iogpu_wired_limit_mb": sh("sysctl", "-n", "iogpu.wired_limit_mb"),
        "power": "AC" if "AC Power" in sh("pmset", "-g", "ps") else "battery",
        "powermode_ac": next(
            (l.split()[-1] for l in sh("pmset", "-g", "custom").splitlines()
             if "powermode" in l), "?"),
        "python": platform.python_version(),
        "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__,
        "numpy": np.__version__,
    }


# --------------------------------------------------------------------------
# site table + tap
# --------------------------------------------------------------------------
def out_dim(mod):
    """Output dim of an nn.Linear or nn.QuantizedLinear."""
    for attr in ("output_dims",):
        if hasattr(mod, attr):
            return int(getattr(mod, attr))
    return int(mod.weight.shape[0])


def in_dim(mod):
    for attr in ("input_dims",):
        if hasattr(mod, attr):
            return int(getattr(mod, attr))
    if hasattr(mod, "scales"):          # quantized: (out, in/group)
        g = int(getattr(mod, "group_size", 64))
        return int(mod.scales.shape[1]) * g
    return int(mod.weight.shape[1])


def build_sites(model):
    """-> (sites, primary_of_id) where sites[i] describes one distinct input vector.

    `primary` is the single module whose call we tap; `consumers` is every matrix
    reading that same vector (their output dims give the byte weight of a skipped
    column).
    """
    sites, primary_of = [], {}

    def add(layer_idx, cls, primary, consumers):
        d = in_dim(primary)
        for c in consumers:
            assert in_dim(c) == d, (cls, layer_idx, in_dim(c), d)
        sites.append({
            "idx": len(sites), "layer": layer_idx, "cls": cls,
            "coarse": COARSE[cls], "dim": d,
            "out_sum": int(sum(out_dim(c) for c in consumers)),
            "key": f"L{layer_idx:02d}.{cls}",
        })
        primary_of[id(primary)] = len(sites) - 1

    for li, layer in enumerate(model.layers):
        if layer.is_linear:
            dn = layer.linear_attn
            add(li, "deltanet-in", dn.in_proj_qkv,
                [dn.in_proj_qkv, dn.in_proj_z, dn.in_proj_b, dn.in_proj_a])
            add(li, "deltanet-out", dn.out_proj, [dn.out_proj])
        else:
            at = layer.self_attn
            add(li, "attn-qkv", at.q_proj, [at.q_proj, at.k_proj, at.v_proj])
            add(li, "attn-o", at.o_proj, [at.o_proj])
        m = layer.mlp
        add(li, "ffn-up-gate", m.gate_proj, [m.gate_proj, m.up_proj])
        add(li, "ffn-down", m.down_proj, [m.down_proj])
    return sites, primary_of


class Tap:
    """Class-level monkeypatch of Linear/QuantizedLinear __call__.

    Records nothing unless `enabled`; identifies the site by object identity, so the
    module tree is untouched (no wrapper modules, no parameter reshaping).
    """

    def __init__(self, primary_of, n_sites):
        import mlx.nn as nn
        self.primary_of = primary_of
        self.buf = [None] * n_sites
        self.enabled = False
        self._orig = {}
        for cls in (nn.Linear, nn.QuantizedLinear):
            self._orig[cls] = cls.__call__
            self._install(cls)

    def _install(self, cls):
        orig, tap = self._orig[cls], self

        def patched(mod, x, *a, **kw):
            if tap.enabled:
                i = tap.primary_of.get(id(mod))
                if i is not None:
                    tap.buf[i] = x
            return orig(mod, x, *a, **kw)

        cls.__call__ = patched

    def restore(self):
        for cls, fn in self._orig.items():
            cls.__call__ = fn


def flat_abs(tap, mx):
    """Concatenate |input| of every site into one float32 vector on the host."""
    parts = []
    for i, x in enumerate(tap.buf):
        assert x is not None, f"site {i} not tapped"
        parts.append(mx.abs(x.astype(mx.float32)).reshape(-1))
        tap.buf[i] = None
    return np.asarray(mx.concatenate(parts, axis=0))


# --------------------------------------------------------------------------
# model + trace plumbing
# --------------------------------------------------------------------------
def load(model_id):
    from mlx_lm.utils import load as mlx_load
    model, tokenizer = mlx_load(model_id)
    trunk = model.language_model.model     # skips lm_head: logits are not needed
    return model, trunk, tokenizer


def read_traces(dirpath, kinds):
    out = []
    for p in sorted(pathlib.Path(dirpath).glob("*.json")):
        if p.name == "MANIFEST.json":
            continue
        r = json.loads(p.read_text())
        if r["kind"] in kinds:
            out.append(r)
    return out


def run_trace(model, trunk, tok, tap, rec, on_token, max_target=None):
    """Prefill the prefix (tap off), then teacher-force target tokens (tap on)."""
    import mlx.core as mx
    pre = tok.encode(rec["prefix"])
    tgt = tok.encode(rec["target"], add_special_tokens=False)
    if max_target:
        tgt = tgt[:max_target]
    cache = model.make_cache()

    tap.enabled = False
    step = 2048
    for s in range(0, len(pre), step):
        trunk(mx.array([pre[s:s + step]]), cache=cache)
        mx.eval([c.state for c in cache])

    tap.enabled = True
    for t in tgt:
        trunk(mx.array([[t]]), cache=cache)
        on_token(flat_abs(tap, mx))
    tap.enabled = False
    return len(tgt)


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------
def cmd_calibrate(args):
    import mlx.core as mx
    model, trunk, tok = load(args.model)
    sites, primary_of = build_sites(model)
    tap = Tap(primary_of, len(sites))

    offs = np.cumsum([0] + [s["dim"] for s in sites]).astype(np.int64)
    total = int(offs[-1])
    site_base = np.repeat(np.arange(len(sites), dtype=np.int64) * HIST_BINS,
                          [s["dim"] for s in sites])
    hist = np.zeros(len(sites) * HIST_BINS, dtype=np.int64)
    scale = HIST_BINS / (HIST_HI - HIST_LO)
    tiny = np.float32(2.0 ** HIST_LO)

    n_tok = [0]

    def on_token(fa):
        b = np.log2(np.maximum(fa, tiny))
        b = ((b - HIST_LO) * scale).astype(np.int64)
        np.clip(b, 0, HIST_BINS - 1, out=b)
        hist.__iadd__(np.bincount(b + site_base, minlength=hist.size))
        n_tok[0] += 1

    traces = read_traces(args.traces, {"calib"})
    if args.limit_traces:
        traces = traces[:args.limit_traces]
    t0 = time.time()
    for i, rec in enumerate(traces):
        n = run_trace(model, trunk, tok, tap, rec, on_token, args.max_target)
        print(f"  [{i+1}/{len(traces)}] {rec['id']:>16}  {n:4d} tok  "
              f"({time.time()-t0:6.1f}s, {n_tok[0]} total)", flush=True)
    tap.restore()

    # per-site quantiles from the histogram CDF, log-linear interpolation in-bin
    H = hist.reshape(len(sites), HIST_BINS)
    thr = {}
    for si, s in enumerate(sites):
        h = H[si].astype(np.float64)
        c = np.cumsum(h)
        tot = c[-1]
        row = {}
        for q in TARGETS:
            want = q * tot
            j = int(np.searchsorted(c, want, side="left"))
            j = min(j, HIST_BINS - 1)
            below = c[j - 1] if j > 0 else 0.0
            frac = (want - below) / h[j] if h[j] > 0 else 0.0
            edge = HIST_LO + (j + frac) / scale
            row[f"{q:.2f}"] = float(2.0 ** edge)
        thr[s["key"]] = row

    out = {
        "kind": "phase0-thresholds",
        "model": args.model,
        "method": ("fixed per-site empirical quantile of |x| over the held-out "
                   "calibration split; 4096-bin log2 histogram, log-linear in-bin "
                   "interpolation"),
        "targets": list(TARGETS),
        "calib_traces": [t["id"] for t in traces],
        "calib_tokens": n_tok[0],
        "env": env_info(),
        "sites": sites,
        "total_mask_bits": total,
        "thresholds": thr,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}  ({n_tok[0]} calibration tokens, {len(sites)} sites)")


# --------------------------------------------------------------------------
# measure
# --------------------------------------------------------------------------
class Acc:
    """Streaming per-site accumulators for one threshold level."""

    def __init__(self, n_sites, offs, thr_vec):
        self.offs = offs
        self.thr = thr_vec
        self.n = np.zeros(n_sites, np.int64)          # tokens seen
        self.act = np.zeros(n_sites, np.int64)        # Σ per-token active
        self.uni = {k: np.zeros(n_sites, np.int64) for k in range(2, K_MAX + 1)}
        self.uni_n = {k: np.zeros(n_sites, np.int64) for k in range(2, K_MAX + 1)}
        self.jac_num = np.zeros(n_sites, np.float64)  # Σ per-pair Jaccard
        self.jac_n = np.zeros(n_sites, np.int64)
        self.win = []                                 # rolling masks, newest last
        self.prev_act = None

    def push(self, fa):
        m = fa >= self.thr
        red = lambda a: np.add.reduceat(a, self.offs, dtype=np.int64)
        a = red(m)
        self.n += 1
        self.act += a

        self.win.append(m)
        if len(self.win) > K_MAX:
            self.win.pop(0)

        u = m
        for k in range(2, K_MAX + 1):
            if len(self.win) < k:
                break
            u = u | self.win[-k]
            uk = red(u)
            self.uni[k] += uk
            self.uni_n[k] += 1
            if k == 2:
                # |A∩B| = |A| + |B| - |A∪B|;  Jaccard = |A∩B| / |A∪B|
                inter = self.prev_act + a - uk
                with np.errstate(divide="ignore", invalid="ignore"):
                    j = np.where(uk > 0, inter / np.maximum(uk, 1), 1.0)
                self.jac_num += j
                self.jac_n += 1
        self.prev_act = a

    def reset_window(self):
        self.win = []
        self.prev_act = None

    def dump(self, sites):
        dims = np.array([s["dim"] for s in sites], np.float64)
        d = {
            "tokens": self.n.tolist(),
            "per_token_sparsity": (1.0 - self.act / np.maximum(self.n, 1) / dims).tolist(),
            "jaccard_adj": (self.jac_num / np.maximum(self.jac_n, 1)).tolist(),
            "jaccard_pairs": self.jac_n.tolist(),
        }
        for k in range(2, K_MAX + 1):
            d[f"union_sparsity_k{k}"] = (
                1.0 - self.uni[k] / np.maximum(self.uni_n[k], 1) / dims).tolist()
            d[f"union_windows_k{k}"] = self.uni_n[k].tolist()
        return d


def aggregate(dump, sites):
    """Reduce a per-site dump to model-wide (byte-weighted) + per-class scalars."""
    w = np.array([s["dim"] * s["out_sum"] for s in sites], np.float64)
    n = np.array(dump["tokens"], np.float64)
    w = w * (n > 0)
    keys = ["per_token_sparsity", "jaccard_adj"] + \
           [f"union_sparsity_k{k}" for k in range(2, K_MAX + 1)]
    out = {"model_wide": {}, "by_class": {}, "by_coarse": {}}
    for key in keys:
        v = np.array(dump[key], np.float64)
        out["model_wide"][key] = float((v * w).sum() / max(w.sum(), 1e-9))
        for field, attr in (("by_class", "cls"), ("by_coarse", "coarse")):
            for c in sorted({s[attr] for s in sites}):
                m = np.array([s[attr] == c for s in sites])
                out[field].setdefault(c, {})[key] = float(
                    (v[m] * w[m]).sum() / max(w[m].sum(), 1e-9))
    out["tokens"] = int(n.max()) if len(n) else 0
    return out


def cmd_measure(args):
    import mlx.core as mx
    cal = json.loads(pathlib.Path(args.thresholds).read_text())
    model, trunk, tok = load(args.model)
    sites, primary_of = build_sites(model)
    assert [s["key"] for s in sites] == [s["key"] for s in cal["sites"]], \
        "site table does not match the calibration file"
    tap = Tap(primary_of, len(sites))

    offs = np.cumsum([0] + [s["dim"] for s in sites]).astype(np.int64)[:-1]
    dims = [s["dim"] for s in sites]

    groups = {}      # trace-kind -> {target -> Acc}
    for kind in ("agentic", "control"):
        groups[kind] = {}
        for q in TARGETS:
            tv = np.concatenate([
                np.full(d, cal["thresholds"][s["key"]][f"{q:.2f}"], np.float32)
                for s, d in zip(sites, dims)])
            groups[kind][q] = Acc(len(sites), offs, tv)

    traces = read_traces(args.traces, {"agentic", "control"})
    if args.limit_traces:
        traces = traces[:args.limit_traces]
    per_trace, t0 = [], time.time()
    for i, rec in enumerate(traces):
        accs = groups[rec["kind"]]
        for a in accs.values():
            a.reset_window()          # windows never span a trace boundary
        # a second, per-trace set of accumulators gives between-trace error bars,
        # which is what separates "PASS with margin" from "within noise"
        solo = {q: Acc(len(sites), offs, accs[q].thr) for q in TARGETS}

        def on_token(fa, accs=accs, solo=solo):
            for a in accs.values():
                a.push(fa)
            for a in solo.values():
                a.push(fa)

        n = run_trace(model, trunk, tok, tap, rec, on_token, args.max_target)
        per_trace.append({"id": rec["id"], "kind": rec["kind"],
                          "project": rec["project"], "langs": rec["langs"],
                          "tokens": n,
                          "agg": {f"{q:.2f}": aggregate(a.dump(sites), sites)
                                  for q, a in solo.items()}})
        el = time.time() - t0
        print(f"  [{i+1}/{len(traces)}] {rec['id']:>14} {rec['kind']:>8} "
              f"{n:4d} tok  {el:6.1f}s", flush=True)
    tap.restore()

    out = {
        "kind": "phase0-summary",
        "model": args.model,
        "thresholds_file": os.path.basename(args.thresholds),
        "threshold_method": cal["method"],
        "targets": list(TARGETS),
        "k_max": K_MAX,
        "env": env_info(),
        "sites": sites,
        "traces": per_trace,
        "wall_seconds": round(time.time() - t0, 1),
        "groups": {kind: {f"{q:.2f}": acc.dump(sites) for q, acc in g.items()}
                   for kind, g in groups.items()},
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}")


# --------------------------------------------------------------------------
# selftest — logic checks with no model
# --------------------------------------------------------------------------
def cmd_selftest(_):
    sites = [{"dim": 6, "key": "a"}, {"dim": 4, "key": "b"}]
    offs = np.array([0, 6], np.int64)
    thr = np.full(10, 0.5, np.float32)
    acc = Acc(2, offs, thr)
    # site a: token0 active {0,1}, token1 active {1,2}  -> union {0,1,2}
    # site b: both tokens active {0}                    -> union {0}
    t0 = np.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 0], np.float32)
    t1 = np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 0], np.float32)
    acc.push(t0)
    acc.push(t1)
    d = acc.dump(sites)
    assert np.allclose(d["per_token_sparsity"], [1 - 2 / 6, 1 - 1 / 4]), d
    assert np.allclose(d["union_sparsity_k2"], [1 - 3 / 6, 1 - 1 / 4]), d
    assert np.allclose(d["jaccard_adj"], [1 / 3, 1.0]), d
    assert d["union_windows_k2"] == [1, 1] and d["union_windows_k3"] == [0, 0]
    # union sparsity must never exceed per-token sparsity
    assert d["union_sparsity_k2"][0] <= d["per_token_sparsity"][0] + 1e-12
    print("dump_masks selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = dict(model="mlx-community/Qwen3.8-27B-4bit", traces="analysis/traces")

    c = sub.add_parser("calibrate")
    c.add_argument("--model", default=common["model"])
    c.add_argument("--traces", default=common["traces"])
    c.add_argument("--out", default="analysis/thresholds/phase0.json")
    c.add_argument("--max-target", type=int, default=0)
    c.add_argument("--limit-traces", type=int, default=0)
    c.set_defaults(fn=cmd_calibrate)

    m = sub.add_parser("measure")
    m.add_argument("--model", default=common["model"])
    m.add_argument("--traces", default=common["traces"])
    m.add_argument("--thresholds", default="analysis/thresholds/phase0.json")
    m.add_argument("--out", default="analysis/summary/phase0.json")
    m.add_argument("--max-target", type=int, default=0)
    m.add_argument("--limit-traces", type=int, default=0)
    m.set_defaults(fn=cmd_measure)

    s = sub.add_parser("selftest")
    s.set_defaults(fn=cmd_selftest)

    a = ap.parse_args()
    a.fn(a)
