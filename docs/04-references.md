# 04 — References

## Core

- **TEAL** — *Training-Free Activation Sparsity in Large Language Models.* Liu, Ponnusamy, Cai, Guo, Kim, Athiwaratkun. ICLR 2025. arXiv:2408.14690. 40–50% model-wide magnitude sparsity, 1.53–1.8× single-batch decode, composes with weight quantization.
- **SpQt** — *Enabling Dynamic Sparsity in Quantized LLM Inference.* Wang, Shu, Lin (UVA / Zoom). arXiv:2511.04477. Zigzag quantization layout + load-balanced Metal kernels on llama.cpp Q4_K; up to 1.55× e2e decode at 50% sparsity on Apple Silicon; sparse path batch-1 only, dense for token count > 1.

## Supporting

- **DejaVu** — *Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time.* Liu et al., ICML 2023. Predictor-based contextual sparsity (ReLU-era).
- **CATS** — training-free thresholding of SwiGLU FFNs (~25% model-wide); predecessor to TEAL.
- **LLM in a Flash** — Alizadeh et al. (Apple), 2024. arXiv:2312.11514. Windowed neuron reuse across adjacent tokens; the overlap premise behind C1's union masks and C3's residency scheme.
- **PowerInfer** — Song et al., SOSP 2024. Hot/cold neuron placement on consumer GPUs.
- **Polar Sparsity** — Shrestha et al., arXiv:2505.14884. Batched sparse GEMM kernels (CUDA); relevant to skinny-GEMM design, up to 5.5× kernel-level.
- **LaRoSA** — *Layerwise Rotated Sparse Activation.* arXiv:2507.01299. Rotation-based activation sparsity + fused Top-K GEMV kernel; alternative to magnitude thresholding if TEAL-style calibration underperforms on this model.
- **R-Sparse** — Zhang et al., ICLR 2025. Rank-aware activation sparsity.

## Model & tooling artifacts

- Qwen/Qwen3.8-27B (Hugging Face, Apache 2.0, released 2026-08-13/14) — dense 27.78B, 16 full-attn + 48 Gated DeltaNet layers, MTP head in released weights, 262K ctx.
- mlx-community/Qwen3.8-27B-4bit — ~15.0 GiB, mlx-vlm 0.6.8.
- bartowski Qwen3.8-27B imatrix GGUFs — include MTP head at Q4_0.
- llama.cpp ≥ b10419 — `--spec-type draft-mtp`.
- MTPLX — github.com/youssofal/MTPLX — MLX serving with native MTP self-drafting (~3× claimed); useful as an MTP-only baseline.
- vLLM recipe for Qwen3.8-27B — recipes.vllm.ai/Qwen/Qwen3.8-27B (server-side reference config, fp8 KV — note we keep KV bf16 locally).

## Verification status

arXiv IDs above were confirmed against search results on 2026-08-28 except CATS (cited by title; add ID when pulled). Community-reported numbers (MTPLX 3×, quality cliff below Q4, KV-quant eval breakage) are anecdotal until re-measured in Phase 1.
