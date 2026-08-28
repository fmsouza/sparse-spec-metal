# kernels — Phase 2: Metal shaders

Empty by design until Phase 0/1 gates pass.

Planned contents:
- zigzag Q4_K weight layout converter (offline tool)
- sparse index-collection kernel (Blelloch prefix-sum + atomic offsets, per SpQt)
- sparse GEMV (batch-1 parity with SpQt on M3 Max)
- **union-mask skinny GEMM (k=2–4)** — the new piece; load-balanced over the active-index array
- hyperparameter sweep configs (threadgroups × simdgroups per superblock row)
