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

## Thesis

The two best accelerators for `Qwen3.8-27B` on Apple Silicon are mutually exclusive today:

1. **MTP self-speculative decoding** — the model ships a multi-token-prediction draft head; llama.cpp (`b10419+`, `--spec-type draft-mtp`) and MLX-based servers use it for ~2–3× decode speedups on code.
2. **Quantized activation sparsity** — SpQt (arXiv 2511.04477) demonstrated TEAL-style activation sparsity on Q4_K weights with custom Metal kernels ("zigzag" layout), up to 1.55× at 50% sparsity — **but its sparse path only runs at batch size 1 and falls back to dense kernels whenever token count > 1.**

MTP *verification* processes 2–4 draft tokens per step, i.e. exactly the case SpQt punts on. This repo investigates making the two compose: **union-mask sparse verification kernels** for skinny GEMM (k=2–4) on Metal, plus **DeltaNet-aware sparsity calibration** for Qwen3.8's hybrid architecture (48 of 64 layers are Gated DeltaNet linear attention — never studied under activation sparsity).

## Target

| | |
|---|---|
| Model | Qwen3.8-27B (dense, 27.78B, Q4_K / MLX 4-bit) |
| Hardware | M3 Max, 36 GB unified memory (~300 GB/s) |
| Baseline (dense Q4_K, est.) | ~14–18 tok/s decode |
| MTP-only (reported/claimed) | ~2–2.5× on code |
| **Goal (MTP × sparse verify)** | **45–55 tok/s, stretch 60+ on edit-heavy agentic output, ≤~1% quality loss** |

## Repo map

- `docs/01-context.md` — hardware + model facts, bandwidth arithmetic, ecosystem state
- `docs/02-design.md` — proposed components C1–C3, non-goals
- `docs/03-experiment-plan.md` — hypotheses, phases, metrics, kill criteria
- `docs/04-references.md` — papers and artifacts
- `PLAN.md` — task map, estimates, and phase status
- `analysis/` — **Phase 0 (done)**: activation-mask overlap measurement + the H1 verdict
- `calibration/` — per-layer threshold computation (TEAL-derived)
- `kernels/` — Metal shaders (Phase 2, empty for now)
- `bench/` — micro-benchmark harness (Phase 2)
- `integration/` — llama.cpp fork wiring notes (Phase 3)
- `results/` — experiment logs and summaries

## Ground rules

- Phase 0 was a **pure measurement study** designed to falsify the core hypothesis cheaply before any kernel work. It did exactly that, in ~4 h of wall clock and one 16 GB download.
- Every phase has kill criteria (see `docs/03-experiment-plan.md`). H1 died; the negative result is written up in `analysis/report.md`.
