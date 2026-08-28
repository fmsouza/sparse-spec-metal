# analysis — Phase 0: mask-overlap measurement

Purpose: measure per-token TEAL mask sparsity and k-token **union sparsity** on Qwen3.8-27B, on real agentic-coding traces, before any kernel work. This phase can falsify H1 (see `docs/03-experiment-plan.md`).

Planned contents (not yet implemented):
- `dump_masks.py` — instrumented forward pass (mlx-lm) logging per-token, per-matrix binary masks at thresholds {25,40,50}%
- `overlap.py` — per-token sparsity, union sparsity for k∈{2,3,4}, adjacent-token Jaccard, split by matrix class (attn / ffn-up-gate / ffn-down / deltanet-proj)
- `traces/` (gitignored) — input transcripts
- `report.md` — Phase 0 findings + H1 gate decision
