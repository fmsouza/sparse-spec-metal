# integration — Phase 3: llama.cpp fork wiring

Purpose: notes and patches for wiring the sparse verification path into a llama.cpp fork (≥ b10419) alongside `--spec-type draft-mtp`.

Constraints: dense prefill; sparse kernels active only for decode + MTP verification; KV stays bf16; end-to-end grid per `docs/03-experiment-plan.md`.
