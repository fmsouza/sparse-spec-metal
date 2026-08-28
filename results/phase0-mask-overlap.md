# phase0 — mask-overlap measurement (H1 gate)

**Verdict: H1 FALSIFIED. Kill criterion met. C1 is dead. Phases 1–3 do not start.**

| Gate (`docs/03`) | Threshold | Measured | Result |
|---|---|---|---|
| H1, k=4, per-token 50% | ≥ 30% union sparsity | **10.8%** | fail, −95.8 SEM |
| H1, k=4, per-token 40% | ≥ 20% union sparsity | **5.1%** | fail, −154.6 SEM |
| **Kill**, k=4, per-token 40% | < 20% → C1 dead | **5.1%** | **kill criterion met** |

Full analysis: [`analysis/report.md`](../analysis/report.md).
Next action: negative result written up; **awaiting review** before archiving.

## Config

| | |
|---|---|
| Model | `mlx-community/Qwen3.8-27B-4bit`, revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` (16.08 GB, 14.98 GiB) |
| Engine | MLX text path only (`mlx_lm.models.qwen3_5`); vision tower and `lm_head` excluded |
| Mask sites | 256 distinct input vectors; 2,162,688 mask bits/token; 24.35 G params tapped |
| Thresholds | fixed per-site empirical quantile of `\|x\|` at {25, 40, 50}%, from a 4096-bin `log2\|x\|` histogram over a **held-out** calibration split (4,646 tokens: 6 agentic turns + 2 WikiText slices) |
| Forward pass | teacher-forced, **decode mode** (S=1 per step), no sampling; prefix prefilled with the tap off |
| Measurement corpus | 20 agentic-coding traces (11,171 decode tokens; 5 projects, 10 languages) + 4 WikiText-2-raw-v1 control slices (3,072 tokens) |
| Sensitivity run | 8 agentic traces re-run with an exact per-token per-site quantile (`--oracle`) |
| Seeds | trace selection seed 1337; decoding is greedy/teacher-forced, so no sampling seed applies |

## Environment

| | |
|---|---|
| Machine | Apple M3 Max, 14-core (10P/4E), 36 GB unified (`hw.memsize` 38,654,705,664) |
| macOS | 26.6.2 (build 25G83) |
| `iogpu.wired_limit_mb` | **0 (system default, ≈75% of RAM)** — raising it needs `sudo`; default recorded and used per the operating rules. Peak MLX memory 15.6 GB, so the limit was not binding. |
| Power state | **AC power, `powermode 2` (High Power)**, battery charged |
| Python / MLX | 3.13.1 / `mlx` 0.32.2, `mlx-lm` 0.31.3, `mlx-vlm` 0.6.17, `numpy` 2.5.2 |
| Wall clock | calibration 417.5 s; measurement 1,394.0 s (14,243 decode tokens); oracle 852.6 s |

## Runs

**One run, not three — deliberately.** `docs/03`'s "3 runs, report median" targets
*timing* cells. Phase 0 emits deterministic statistics: two full re-runs of the same
configuration were compared byte-for-byte and are **identical**
(SHA-256 of the summary with `env`/`wall_seconds` stripped: `f92744a0fd132d0d…`,
both runs). Three runs would report the same number three times. Phase 2/3 timing
runs get the full 3-run median treatment. Flagged as a deviation in
`analysis/report.md`.

## Headline numbers (byte-weighted, model-wide, agentic corpus)

| target | per-token realized | k=2 | k=3 | k=4 | adjacent Jaccard |
|---|---|---|---|---|---|
| 25% | 25.5% | 7.9% | 2.7% | 1.1% | 0.616 |
| 40% | 40.8% | 19.2% | 9.6% | 5.1% | 0.462 |
| 50% | 50.9% | 29.1% | 17.4% | 10.8% | 0.380 |

Per-trace spread at k=4 / 50%: 9.3–12.2% across all 20 traces (sd 0.9 pp). WikiText
control: 8.2% at k=4 / 50%.

Union sparsity is only **1.59×** what statistically independent per-token masks would
give (`s^k` = 6.8%). `docs/02` assumed 30–40% retention under a k=4 union; measured
retention is 21% of the per-token figure.

## What the gate implies for H3

Decode is bandwidth-bound, so a verification pass reading fraction `f` of the trunk
costs `1/f`. At k=4 the union leaves `f` = 0.892 → adding sparsity to MTP is capped at
**1.12×**, against H3's ≥1.25× gate — before index-collection overhead, kernel
inefficiency, or quality loss. H3 is unreachable via C1 as designed. (At k=2 the
ceiling is 1.41×; see `analysis/report.md` §5 for why that is not a rescue.)

## Side results worth keeping

- **Dense 4-bit decode on this machine: 17.9 tok/s** (24 steps, warm cache, tap off;
  17.5 tok/s with instrumentation). Independently confirms `docs/01`'s 14–18 tok/s
  estimate — that row is no longer an estimate.
- **The MLX 4-bit conversion has no MTP head.** `model.safetensors.index.json`
  contains no `mtp.*` tensors and `mlx_lm`'s `sanitize()` filters them. Any MTP
  baseline must come from llama.cpp + a GGUF that ships the head. Recorded in
  `PLAN.md`; no impact on Phase 0.
- **Peak MLX memory 15.6 GB** for 4-bit weights + 256-site instrumentation.

## Artifacts

| Path | Committed? |
|---|---|
| `analysis/build_traces.py`, `analysis/dump_masks.py`, `analysis/overlap.py` | yes |
| `analysis/thresholds/phase0.json` | yes |
| `analysis/summary/phase0.json`, `analysis/summary/phase0-oracle.json` | yes |
| `analysis/report.md` | yes |
| `analysis/traces/*.json` | **no** — gitignored; contains the operator's private source code |

Verification: `uv run analysis/overlap.py --check` exits 0 (18 checks); the report
tables regenerate deterministically from the committed summary statistics alone, with
no model and no traces.
