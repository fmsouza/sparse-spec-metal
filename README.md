# sparse-spec-metal

**Sparsity-aware speculative (MTP) decoding for dense LLMs on Apple Silicon.**

Status: **Phase 0 complete — H1 falsified. C1 is dead as designed.**

> On Qwen3.8-27B, the union of TEAL activation masks over k=4 adjacent tokens retains
> **10.8%** skippable columns at 50% per-token sparsity (**5.1%** at 40%), against a
> `<20% at k=4/40% → C1 dead` kill criterion. Adjacent-token mask overlap is only
> **1.59×** what independent masks would give, so union sparsity collapses nearly
> geometrically in k. Adding union-mask sparsity to an MTP pass is capped at
> **1.12×** at k=4 — below H3's 1.25× gate, before any kernel is written.
>
> Full write-up: [`analysis/report.md`](analysis/report.md) ·
> gate record: [`results/phase0-mask-overlap.md`](results/phase0-mask-overlap.md) ·
> task map: [`PLAN.md`](PLAN.md)
>
> Phases 1–3 did not start. Awaiting review before archiving.

## The hypothesis that was tested

The two best accelerators for `Qwen3.8-27B` on Apple Silicon are mutually exclusive today:

1. **MTP self-speculative decoding** — the model ships a multi-token-prediction draft head; llama.cpp (`b10419+`, `--spec-type draft-mtp`) and MLX-based servers use it for ~2–3× decode speedups on code.
2. **Quantized activation sparsity** — SpQt (arXiv 2511.04477) demonstrated TEAL-style activation sparsity on Q4_K weights with custom Metal kernels ("zigzag" layout), up to 1.55× at 50% sparsity — **but its sparse path only runs at batch size 1 and falls back to dense kernels whenever token count > 1.**

MTP *verification* processes 2–4 draft tokens per step, i.e. exactly the case SpQt punts on. This repo set out to make the two compose: **union-mask sparse verification kernels** for skinny GEMM (k=2–4) on Metal, plus **DeltaNet-aware sparsity calibration** for Qwen3.8's hybrid architecture (48 of 64 layers are Gated DeltaNet linear attention — never studied under activation sparsity).

The whole design rests on one empirical claim — that adjacent tokens activate similar
neurons, so a k-token union mask stays sparse. **Phase 0 measured that claim directly
and it is false on this model.** No kernel was written; nothing downstream of it is
worth building as specified.

## What was found

| | |
|---|---|
| Adjacent-token mask Jaccard (50% per-token sparsity) | **0.380** — active sets share barely a third of their members |
| Union sparsity, k = 2 / 3 / 4 | **29.1% / 17.4% / 10.8%** — collapses nearly geometrically |
| vs. statistically independent masks | only **1.59×** better; `docs/02`'s 30–40% assumption implies 4–6× |
| Agentic code vs. WikiText prose | code overlaps **less**, not more (1.59× vs 1.79× excess) |
| Per-token sparsity needed to clear H1 | **~66%** — past the point SpQt reports quality degradation |
| Ceiling on MTP × union-sparse at k=4 | **1.12×** vs H3's ≥1.25× gate, before any kernel overhead |

Robustness: consistent across all 20 traces (9.3–12.2%), all 10 languages, and all
depth bands (9.4–12.8%). An exact per-token oracle threshold — not implementable in a
kernel — is slightly *worse* (9.7%), so the calibration shortcut is not the cause.

Batch-1 sparse decode is untouched by this: per-token sparsity is 50.9% at the 50%
target, a 2.04× bandwidth ceiling. The negative result is specifically about
*composing* that with multi-token verification.

## Target

| | |
|---|---|
| Model | Qwen3.8-27B (dense, 27.78B, MLX 4-bit — 16.08 GB) |
| Hardware | M3 Max, 36 GB unified memory (~300 GB/s), macOS 26.6.2 |
| Baseline (dense 4-bit) | **17.9 tok/s decode — measured**, confirming `docs/01`'s 14–18 tok/s estimate |
| MTP-only | ~2–2.5× on code (reported; not re-measured — Phase 3 never started) |
| ~~Goal (MTP × sparse verify)~~ | ~~45–55 tok/s~~ — **not reachable via C1**; capped at 1.12× over MTP at k=4 |

## Repo map

- `docs/01-context.md` — hardware + model facts, bandwidth arithmetic, ecosystem state
- `docs/02-design.md` — proposed components C1–C3, non-goals
- `docs/03-experiment-plan.md` — hypotheses, phases, metrics, kill criteria
- `docs/04-references.md` — papers and artifacts
- `PLAN.md` — task map, estimates, and phase status
- `analysis/` — **Phase 0 (done)**: instrumentation, mask-overlap measurement, and the H1 verdict
- `results/` — one entry per run; `phase0-mask-overlap.md` is the gate record
- `calibration/` — Phase 1, **not started** (blocked by the Phase 0 gate)
- `kernels/` — Phase 2, **not started** (empty)
- `bench/` — Phase 2, **not started** (empty)
- `integration/` — Phase 3, **not started** (empty)

## Reproducing

```bash
uv sync
uv run analysis/overlap.py --check     # 18 self-tests, exits 0, no model needed
uv run analysis/overlap.py --summary analysis/summary/phase0.json   # print the tables
```

The report regenerates deterministically from the committed summary statistics alone.
To redo the measurement end to end (~40 min on an M3 Max, one 16 GB download):

```bash
uv run python analysis/build_traces.py && uv run python analysis/dump_masks.py calibrate && uv run python analysis/dump_masks.py measure
```

Input traces are assembled from local Claude Code session logs and are **gitignored** —
they contain private source code. Only aggregate statistics are committed.

## Ground rules

- Phase 0 was a **pure measurement study** designed to falsify the core hypothesis cheaply before any kernel work. It did exactly that, in ~4 h of wall clock and one 16 GB download — against an estimated 70–100 h for Phases 1–3.
- Every phase has kill criteria (see `docs/03-experiment-plan.md`). H1 died; the negative result is written up in `analysis/report.md`, including what was *not* tested and what a revival would need (a mask rule with far higher adjacent-token overlap — a different research question, not a tuning exercise).
- Two incidental findings are recorded in `PLAN.md` and `results/`: dense 4-bit decode measures 17.9 tok/s on this machine, and the MLX 4-bit conversion **drops the MTP head**, so any MTP baseline must come from llama.cpp + GGUF.
