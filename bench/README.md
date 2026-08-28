# bench — Phase 2: kernel micro-benchmarks

Purpose: isolate kernel-level speedup at the union sparsities measured in Phase 0, on Qwen3.8-27B GEMV/GEMM shapes (hidden 5120, FFN dims, k∈{1..4}), vs. dense Q4_K Metal baseline.

Gate: ≥1.3× at measured union sparsity before integration. Record: macOS version, iogpu.wired_limit_mb, power state; 3 runs, median.
