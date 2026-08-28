# 03 — Experiment plan

Design principle: **cheapest falsification first.** Phase 0 requires no kernels and can kill the project in a weekend.

## Hypotheses

- **H1 (overlap).** On real agentic-coding traces with Qwen3.8-27B, the union of TEAL activation masks across k = 4 adjacent tokens retains **≥ 30% skippable columns** at a per-token sparsity of 50% (≥ 20% at per-token 40%).
- **H2 (quality).** ≤ 40% model-wide sparsity — with DeltaNet projections capped at 25% — costs **≤ ~1%** on coding evals and shows bounded DeltaNet state drift at 32K context.
- **H3 (end-to-end).** MTP × union-sparse verification delivers **≥ 1.25×** over MTP-only, landing **45–55 tok/s** decode on the M3 Max 36 GB (stretch: 60+ on edit-heavy output).

## Phase 0 — Mask-overlap measurement (no kernels) → `analysis/`

1. Instrument a forward pass (mlx-lm or llama.cpp `--dump-activations`-style patch) to log per-token, per-matrix binary masks at calibrated thresholds {25, 40, 50}% on:
   - a WikiText slice (control),
   - ~20 real agentic-coding transcripts (file edits, tool output, diffs — the actual workload).
2. Compute, per matrix class (attn-QKVO / FFN-up-gate / FFN-down / DeltaNet-proj):
   - per-token sparsity,
   - union sparsity for k ∈ {2, 3, 4} adjacent tokens,
   - mask Jaccard overlap between adjacent tokens.
3. **Gate:** H1 holds → Phase 1. H1 fails → write up negative result, archive.

## Phase 1 — Calibration + quality → `calibration/`

1. Port TEAL threshold calibration to Qwen3.8-27B (incl. DeltaNet projections as a separate matrix class; block-wise greedy budget).
2. Evals at model-wide sparsity {0, 25, 40, 50}%, per-token masking (exact), on:
   - perplexity: WikiText + a held-out code corpus,
   - HumanEval+ (or EvalPlus subset) and a small fixed set of private agentic tasks,
   - DeltaNet state drift vs. position up to 32K.
3. **Gate:** find the largest sparsity with ≤ ~1% degradation. If < 25%, C1 is not worth building → stop or restrict to attn+FFN-only budget (C2 fallback) and re-gate.

## Phase 2 — Kernel micro-benchmarks → `kernels/`, `bench/`

1. Reproduce SpQt-class sparse GEMV on M3 Max (sanity: ≥ 1.4× at 50% on Qwen3.8 GEMV shapes: hidden 5120, FFN dims, vocab 248,320 head excluded).
2. Implement union-mask skinny GEMM (k = 2–4); benchmark vs. dense Q4_K Metal GEMM at union sparsities measured in Phase 0.
3. **Gate:** ≥ 1.3× kernel-level at the Phase-0-measured union sparsity, else revisit layout/load-balancing hyperparameters before integration.

## Phase 3 — End-to-end integration → `integration/`

1. Wire kernels into a llama.cpp fork (≥ b10419) alongside `--spec-type draft-mtp`; dense prefill; sparse path active only for decode + verification.
2. Measure the full grid on fixed prompts (short chat, long-context 32K, agentic edit loop):

| Config | tok/s | ms/tok | E[accepted] | union sparsity | ppl | HumanEval+ | RSS |
|---|---|---|---|---|---|---|---|
| dense Q4_K | | | — | — | | | |
| MTP only | | | | — | | | |
| sparse only (batch-1) | | | — | | | | |
| **MTP × sparse (ours)** | | | | | | | |
| (ref) REAP-25B-A3B MoE | | | — | — | | | |

3. **Gate:** H3. Report acceptance-length interaction: does union-masking change draft acceptance rates? (It shouldn't much — verification logits shift slightly — but measure it.)

## Controls & hygiene

- Fixed seeds, greedy decoding for speed runs; temperature runs only for quality evals.
- Same GGUF/quant artifacts across configs; macOS version, `iogpu.wired_limit_mb`, and power state (plugged, High Power) recorded in every `results/` entry.
- 3 runs per cell, report median; thermals: 2-min cooldown between runs.

## Kill criteria (explicit)

- Phase 0: union sparsity < 20% at k = 4 / per-token 40% → **C1 dead.**
- Phase 1: no sparsity level ≥ 25% survives the quality bar even attn+FFN-only → **project dead.**
- Phase 2: kernel < 1.15× at measured union sparsity after tuning → integration not worth it; publish kernels + negative result.
- Any phase: DeltaNet drift unbounded with position at all tested sparsities → C2 fallback permanently (attn+FFN only).
