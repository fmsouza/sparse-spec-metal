# 02 — Design

Three components, ordered by expected value. C1 is the core contribution; C2 is the open research question; C3 is optional.

## C1 — Union-mask sparse verification kernels (skinny GEMM, k = 2–4)

**Problem.** SpQt's sparse Metal kernels handle GEMV (batch 1) and fall back to dense for token count > 1. MTP verification is a batch of k = 2–4 draft tokens — exactly the dense-fallback case. So today: MTP × sparsity don't compose.

**Approach.**
1. Compute per-token TEAL masks for the k draft tokens (threshold per matrix group, calibrated offline).
2. Take the **union of active column indices** across the k tokens → one index list.
3. Run one load-balanced sparse pass over the zigzag-laid-out Q4_K weights for all k tokens (skinny GEMM instead of GEMV). Reuse SpQt's two-stage index collection (Blelloch prefix-sum + atomic offsets) and its load-balancing scheme (partition the *active index array*, not static superblock ranges, across simdgroups).

**Why union sparsity should survive.** Activation patterns of adjacent tokens in the same context overlap heavily — the premise LLM-in-a-Flash exploits with windowed neuron caching. Per-token 50% sparsity is expected to degrade to ~30–40% skippable columns under a k=4 union. **This number is the whole bet — Phase 0 measures it before any kernel is written.**

**Why the speedups multiply.** Speculation amortizes one weight pass over `E[accepted]` tokens; sparsity shrinks the bytes of that one pass. Independent effects on the same bandwidth budget:
`tok/s ≈ BW / bytes_per_pass × E[accepted per pass]`.

**Design notes / risks.**
- Union masks mean per-token results include columns that token would have pruned — output differs slightly from per-token TEAL. Quantify quality delta (Phase 1) vs. per-token masking with k separate GEMVs (fallback design, less efficient but exact).
- Zigzag layout hyperparameters (threadgroups × simdgroups per superblock row) were tuned for GEMV; skinny GEMM changes arithmetic intensity — retune on M3 Max.
- Drafting passes themselves (MTP head on trunk hidden state) remain dense; they are cheap relative to verification.

## C2 — DeltaNet-aware sparsity calibration

**Problem.** TEAL/SpQt calibrated standard transformer blocks (QKV + SwiGLU FFN). 48/64 layers here are Gated DeltaNet: projections feed a **recurrent state updated via the delta rule**. Thresholding error in a recurrent state may **accumulate over the sequence** rather than wash out per-token as in softmax attention. Unstudied.

**Approach.**
- Calibrate per-layer, per-matrix thresholds as in TEAL (block-wise greedy budget allocation), but add a **sequence-level error metric**: DeltaNet state drift (‖state_sparse − state_dense‖ over position) and long-sequence perplexity, not just per-token activation error.
- Expected budget shape: full-attention layers + FFNs at 40–50%; DeltaNet projections at 0–25% until drift data says otherwise.
- Fallback if drift is bad at any useful sparsity: restrict sparsity to the 16 full-attention layers + all FFNs, re-derive the model-wide budget, re-check e2e math.

## C3 — Optional: hot/cold superblock residency

Column-activation statistics are heavily skewed. mmap the weights; keep hot superblocks resident (~10–11 GB instead of 15 GB), let cold ones fault from SSD (LLM-in-a-Flash-style windowed reuse). **Riskier** — SSD faults stall decode; only worth it if memory pressure, not speed, becomes the binding constraint. Not in the critical path.

## Non-goals (phase 1 of the project)

- No training / fine-tuning / ReLUfication.
- No sparse prefill (SpQt: aggregate sparsity across prompt tokens is low; sparse prefill degrades quality).
- No sub-4-bit weights (model reportedly quality-cliffs below Q4).
- No KV-cache quantization (anecdotal eval breakage; KV is small here anyway — 64 KB/token).
- Batch size 1 serving only (single-user agentic coding), beam search out of scope.
