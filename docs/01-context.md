# 01 — Context: hardware, model, and the arithmetic

Facts gathered 2026-08-28. Items marked *(anecdotal)* are community reports, not measured here — Phase 0/1 re-verifies anything load-bearing.

## Hardware target

- **M3 Max, 36 GB unified memory** — the binned variant (14-core CPU / 30-core GPU), **~300 GB/s** memory bandwidth.
- Default GPU wired limit is ~75% of RAM; raise for headroom:
  `sudo sysctl iogpu.wired_limit_mb=30720`
- Single-stream decode on dense models is memory-bandwidth-bound: `tok/s ≈ bandwidth / bytes-read-per-token`.

## Model: Qwen3.8-27B (released 2026-08-13/14, Apache 2.0)

- Dense, **27.78B total params** including a vision encoder (~0.78B; strippable for text-only serving).
- **64 layers: 16 full attention + 48 Gated DeltaNet linear attention** (`full_attention_interval: 4`), hidden 5120, vocab 248,320.
- **262K native context** (YaRN-extensible to 1M; static YaRN hurts short-prompt quality — not used here).
- **MTP speculative-decoding head included in the released weights.**
- KV cache: only the 16 full-attention layers hold KV → **~64 KB/token** (¼ of a conventional 64-layer dense model). 8K ctx ≈ 0.5 GB, 32K ≈ 2 GB, 128K ≈ 8 GB.
- Weights: 55.6 GB BF16; **~15.0 GiB** at 4-bit (mlx-community/Qwen3.8-27B-4bit, mlx-vlm 0.6.8); Ollama build ~18 GB.

## Quantization sensitivity *(anecdotal, must re-verify)*

- Multiple community reports: quality "falls off a cliff" **below Q4** on this model, more than typical for 27B-class.
- Aggressive **KV-cache quantization reportedly broke at least one private eval**. Default: keep KV at bf16/fp16.
- Consequence for this project: sparsity budgets must be validated on-model; do not assume Llama-calibrated tolerances transfer.

## Baseline speed arithmetic (M3 Max 36 GB)

| Configuration | Bytes/token (approx) | Ceiling (300 GB/s) | Realistic |
|---|---|---|---|
| Dense Q4_K / MLX 4-bit | ~15 GB | ~20 tok/s | ~14–18 tok/s |
| + MTP self-spec (code) | amortized over accepted tokens | — | ~30–42 tok/s (2–2.5×) |
| + union-sparse verification (this project) | ~0.60–0.70× bytes per verify pass | — | **~45–55 tok/s target** |

Reference points: ~8 tok/s reported on M4 MacBook Air (~120 GB/s) for this model; sibling Qwen3.6-27B estimated ~30 tok/s on M5 Max. Both consistent with bandwidth scaling.

Optional comparison row for the eval table: Qwen3-Coder-REAP-25B-A3B (MoE, ~3B active) on the same machine — the sparse-by-architecture alternative this project tries to approximate on a dense model.

## Ecosystem state (what already exists)

- **llama.cpp `b10419+`**: MTP decoding via `--spec-type draft-mtp`; bartowski imatrix GGUFs ship the MTP head at Q4_0.
- **MTPLX** (github.com/youssofal/MTPLX): MLX server with native MTP self-drafting, claims ~3× on Apple Silicon; curated Qwen3.8-27B dynamic quants.
- **SpQt** (arXiv 2511.04477): zigzag-layout Q4_K + sparse Metal kernels on llama.cpp b5711 (~8K SLOC). Reported: up to 1.55× end-to-end decode at 50% sparsity; sparse GEMV kernel 1.51–1.78×; perplexity degradation marginal ≤50% sparsity, noticeable at 65%. **Sparse path is batch-1 only; dispatches dense kernels when token count > 1. Prefill stays dense (low aggregate sparsity across tokens; sparse prefill degrades quality).**
- **TEAL** (ICLR 2025, arXiv 2408.14690): training-free magnitude thresholding of hidden states; 40–50% model-wide sparsity on Llama-2/3 and Mistral 7B–70B with minimal degradation; 1.53–1.8× single-batch decode; composes with weight quantization.

## The gap

No published work (as of 2026-08-28) combines activation sparsity with speculative/MTP verification, and none applies activation sparsity to Gated DeltaNet layers. Both are required to stack the two speedups on this model. That is the subject of `02-design.md`.
