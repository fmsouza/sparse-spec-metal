# analysis — Phase 0: mask-overlap measurement (COMPLETE — H1 falsified)

Measured per-token TEAL mask sparsity and k-token **union sparsity** on Qwen3.8-27B
over real agentic-coding traces. **H1 is falsified and the `docs/03` kill criterion is
met**: union sparsity at k=4 / per-token 40% is 5.1% against a <20% floor.

Read [`report.md`](report.md) — verdict, tables, findings, method, and limitations.

## Contents

| | |
|---|---|
| `build_traces.py` | assembles agentic-coding traces from local Claude Code session logs + a WikiText-2 control; disjoint calibration split |
| `dump_masks.py` | instrumented teacher-forced decode; `calibrate` → thresholds, `measure` → summary stats, `--oracle` sensitivity mode, `selftest` |
| `overlap.py` | aggregation, tables, H1 verdict; `--check` runs 18 self-tests and exits 0 |
| `thresholds/phase0.json` | per-site fixed thresholds at {25, 40, 50}% |
| `summary/phase0.json` | committed summary statistics (the report regenerates from these alone) |
| `summary/phase0-oracle.json` | per-token oracle sensitivity run |
| `report.md` | Phase 0 findings + H1 gate decision |
| `traces/` | **gitignored** — input transcripts contain private source code |

```bash
uv run analysis/overlap.py --check                                   # 18 checks, exit 0
uv run analysis/overlap.py --summary analysis/summary/phase0.json \
                           --report analysis/report.md               # regenerate tables
```
