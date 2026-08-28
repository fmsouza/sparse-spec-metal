# sparse-spec-metal

**Sparsity-aware speculative (MTP) decoding for dense LLMs on Apple Silicon.**

Status: **investigation / pre-implementation**. No code yet — see `docs/` for design and the experiment plan.

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
- `analysis/` — Phase 0: activation-mask overlap measurement (no kernels)
- `calibration/` — per-layer threshold computation (TEAL-derived)
- `kernels/` — Metal shaders (Phase 2, empty for now)
- `bench/` — micro-benchmark harness (Phase 2)
- `integration/` — llama.cpp fork wiring notes (Phase 3)
- `results/` — experiment logs and summaries

## Ground rules

- Phase 0 is a **pure measurement study** designed to falsify the core hypothesis cheaply before any kernel work.
- Every phase has kill criteria (see `docs/03-experiment-plan.md`). If H1 dies, the repo archives cleanly with a negative result written up.
