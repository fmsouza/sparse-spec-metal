# calibration — Phase 1: thresholds + quality gates

Purpose: TEAL-style per-layer, per-matrix threshold calibration for Qwen3.8-27B, with DeltaNet projections as a separate matrix class, plus the quality evals and the DeltaNet state-drift metric.

Planned contents (not yet implemented):
- `calibrate.py` — activation histograms → thresholds for target sparsities (block-wise greedy budget)
- `drift.py` — ‖state_sparse − state_dense‖ vs. position up to 32K on DeltaNet layers
- `evals/` — perplexity (WikiText + code corpus), HumanEval+ runner config, fixed private agentic task set
- `thresholds/` — committed JSON threshold sets per sparsity level
