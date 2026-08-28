# PLAN — sparse-spec-metal investigation

Derived from `docs/03-experiment-plan.md`. This file maps the phase plan to concrete
tasks, estimates, and artifacts. Estimates are wall-clock on the target machine
(M3 Max 36 GB), assuming a single operator working sequentially.

**Status legend:** ☐ todo · ◐ in progress · ☑ done · ✗ killed by gate

---

## Environment (verified 2026-08-28, before Phase 0)

| Fact | Doc claim | Measured | Verdict |
|---|---|---|---|
| Chip | M3 Max | Apple M3 Max, 14-core (10P/4E) | ✓ |
| RAM | 36 GB | 36 GB (`hw.memsize` 38.65e9) | ✓ |
| macOS | — | 26.6.2 (25G83) | recorded |
| `iogpu.wired_limit_mb` | raise to 30720 | **0 (system default, ≈75% ≈ 27 GB)** | no sudo → default used, recorded |
| Power | plugged, High Power | AC power, `powermode 2` (High Power) on AC | ✓ |
| Free disk | — | 274 GiB | ✓ (16 GB download fits) |
| `mlx` / `mlx-lm` / `mlx-vlm` | mlx-vlm 0.6.8 | 0.32.2 / 0.31.3 / 0.6.17 | newer, pinned in `uv.lock` |
| `mlx-community/Qwen3.8-27B-4bit` | ~15.0 GiB | **16.08 GB** on the Hub (14.98 GiB) | ✓ |

Model config (`Qwen/Qwen3.8-27B/config.json`) — all doc claims confirmed:
64 layers, `layer_types` = 48 `linear_attention` + 16 `full_attention`
(`full_attention_interval` 4), hidden 5120, **intermediate 17408**, vocab 248320,
`mtp_num_hidden_layers: 1`, 262144 ctx, head_dim 256, 24 q-heads / 4 kv-heads,
DeltaNet 16 k-heads × 128 / 48 v-heads × 128, conv kernel 4, vision tower 27 blocks.

**New risk found (not in docs):** the MLX 4-bit conversion **drops the MTP head** —
`model.safetensors.index.json` contains no `mtp.*` tensors, and `mlx_lm`'s
`sanitize()` explicitly filters `"mtp." not in k`. Consequence: MLX cannot serve as
the MTP baseline. Phase 3 must use llama.cpp + a GGUF that ships the MTP head
(bartowski imatrix, MTP at Q4_0) as `docs/01` already assumes. **No impact on
Phase 0**, which needs only trunk activations. Flagged here; carried into the Phase 3
plan.

---

## Phase 0 — Mask-overlap measurement (gate: H1)  ·  ☑ DONE — **GATE FAILED**

**H1 falsified; kill criterion met.** Union sparsity at k=4 / per-token 40% is 5.1%
against a <20% kill floor. See [`analysis/report.md`](analysis/report.md) and
[`results/phase0-mask-overlap.md`](results/phase0-mask-overlap.md). Phases 1–3 below
do not start. Actual cost: ~4 h wall clock.

| # | Task | Artifact | Est. |
|---|---|---|---|
| 0.1 ☑ | uv env; pull `mlx-community/Qwen3.8-27B-4bit` (16 GB) | `pyproject.toml`, `uv.lock` | 0.5 h ☑ |
| 0.2 ☑ | Assemble ~20 agentic-coding traces from local Claude Code session logs (multi-file TS/Rust/Python edits, diffs, tool output); WikiText-2 control slice | `analysis/traces/` (gitignored), `analysis/build_traces.py` | 1.0 h |
| 0.3 ☑ | Instrumented decode-mode forward pass: tap every `nn.Linear`/`nn.QuantizedLinear` input, streaming mask statistics | `analysis/dump_masks.py` | 2.0 h |
| 0.4 ☑ | Calibration pass → per-site fixed thresholds at per-token targets {25,40,50}% | `analysis/thresholds/*.json` | 0.5 h |
| 0.5 ☑ | Measurement pass over all traces + control | `results/raw/*.json` (gitignored), `analysis/summary/*.json` (committed) | 1.5 h |
| 0.6 ☑ | Aggregation + tables: per-class per-token sparsity, union sparsity k∈{2,3,4}, adjacent Jaccard, byte-weighted model-wide | `analysis/overlap.py` | 1.5 h |
| 0.7 ☑ | `--check` self-test exits 0; tables regenerate deterministically from committed summary stats | `analysis/overlap.py --check` | 0.5 h |
| 0.8 ☑ | Write-up + explicit H1 verdict | `analysis/report.md`, `results/phase0-*.md` | 1.0 h |

### Method (decisions, stated up front)

- **Mask site = distinct input vector**, not distinct matrix. Matrices consuming the
  same input share one mask, which is what a kernel would exploit. Six site classes
  (finer than docs/03's four; the four-class view is also reported):

  | Site class | Input dim | Consumers (Σ output dim) | Layers |
  |---|---|---|---|
  | `attn-qkv` | 5120 | q(12288)+k(1024)+v(1024) = 14336 | 16 |
  | `attn-o` | 6144 | o(5120) | 16 |
  | `deltanet-in` | 5120 | qkv(10240)+z(6144)+b(48)+a(48) = 16480 | 48 |
  | `deltanet-out` | 6144 | out(5120) | 48 |
  | `ffn-up-gate` | 5120 | gate(17408)+up(17408) = 34816 | 64 |
  | `ffn-down` | 17408 | down(5120) | 64 |

  256 sites total, 2,162,688 mask bits per token. `lm_head` and the vision tower are
  excluded (docs/02 non-goals; vocab head excluded per docs/03 Phase 2).
- **Thresholds are fixed offline**, TEAL-style: per-site empirical quantile of |x|
  over a held-out calibration slice, then held constant during measurement. Realized
  per-token sparsity therefore varies around the target — this is the honest setting
  and the one a kernel faces. A per-token oracle quantile is also computed as a
  sensitivity upper bound. (docs/03 permits the empirical-quantile shortcut for
  Phase 0; this is what was used.)
- **Teacher-forced decode mode**: each trace is fed one token at a time (S=1) through
  the KV/SSM cache, no sampling. This matches decode-time activation statistics,
  which differ from chunked prefill.
- **Aggregation is byte-weighted** for the model-wide number: skipping an input
  column saves `Σ output_dim` weight bytes, so classes are weighted by
  `input_dim × Σ output_dim`. Unweighted per-class numbers are reported too.
- **Streaming statistics** — masks are never stored. A rolling window of k=4 tokens
  updates the accumulators; only summary stats are committed.

### Known assumption to state in the report

MTP verification's k tokens are *draft* tokens; a rejected draft's activations are
off-distribution relative to teacher forcing. Teacher-forced adjacent tokens are a
good proxy only when acceptance is high. Phase 3 re-measures union sparsity on the
real draft stream; Phase 0 reports the teacher-forced bound and flags the gap.

### Gate — RESULT

| | threshold | measured | |
|---|---|---|---|
| H1, k=4, per-token 50% | ≥30% | **10.8%** | fail (−95.8 SEM) |
| H1, k=4, per-token 40% | ≥20% | **5.1%** | fail (−154.6 SEM) |
| Kill, k=4, per-token 40% | <20% → C1 dead | **5.1%** | **met** |

Robustness: an exact per-token oracle quantile (not kernel-implementable) gives
9.7% at k=4 / 50% — slightly *worse*, so the calibration shortcut is not the cause.
Measured union sparsity is only 1.59× what independent masks would give.

**Next action: negative result written up; awaiting review before archiving.**

---

## Phase 1 — Calibration + quality (gate: H2)  ·  ✗ NOT STARTED

Blocked: the Phase 0 gate failed. Kept for the record.

| # | Task | Artifact | Est. |
|---|---|---|---|
| 1.1 | TEAL block-wise greedy budget allocation over the 6 site classes, DeltaNet as its own class | `calibration/calibrate.py`, `calibration/thresholds/*.json` | 3 h |
| 1.2 | Sparse-forward harness (per-token exact masking, no union) for eval | `calibration/sparse_forward.py` | 2 h |
| 1.3 | Perplexity: WikiText-2 + held-out code corpus at model-wide {0,25,40,50}% | `calibration/evals/ppl.py` | 2 h |
| 1.4 | EvalPlus HumanEval+ at a pinned release, greedy, fixed seed | `calibration/evals/` | 3 h |
| 1.5 | DeltaNet state drift ‖state_sparse − state_dense‖ vs. position to 32K | `calibration/drift.py` | 3 h |
| 1.6 | Report + gate | `calibration/report.md`, `results/phase1-*.md` | 2 h |

**Gate:** largest sparsity with ≤ ~1% degradation. If < 25% even under the
attn+FFN-only C2 fallback → **project dead**. If DeltaNet drift grows unbounded with
position at every tested sparsity → C2 fallback permanent (attn+FFN only), re-derive
the model-wide budget, re-check the e2e arithmetic before Phase 2.

---

## Phase 2 — Kernel micro-benchmarks (gate: ≥1.3×)  ·  ✗ NOT STARTED

Kernels are **reimplemented from the SpQt paper design** — assume no public code.

| # | Task | Artifact | Est. |
|---|---|---|---|
| 2.1 | Offline zigzag Q4_K layout converter | `kernels/zigzag_convert.py` | 4 h |
| 2.2 | Index collection: Blelloch prefix-sum + atomic offsets | `kernels/index_collect.metal` | 6 h |
| 2.3 | Sparse GEMV (batch-1 SpQt parity, target ≥1.4× at 50%) | `kernels/sparse_gemv.metal` | 10 h |
| 2.4 | **Union-mask skinny GEMM, k=2–4** (the contribution) | `kernels/union_gemm.metal` | 12 h |
| 2.5 | Bench harness vs. dense Q4_K Metal, Qwen3.8 shapes, 3 runs median, 2-min cooldown | `bench/` | 4 h |
| 2.6 | Hyperparameter sweep (threadgroups × simdgroups per superblock row) retuned for skinny GEMM | `bench/sweep/` | 6 h |
| 2.7 | Report + gate | `results/phase2-*.md` | 2 h |

**Gate:** ≥1.3× kernel-level at the Phase-0-measured union sparsity. <1.15× after
tuning → integration not worth it; publish kernels + negative result.

---

## Phase 3 — End-to-end integration (gate: H3)  ·  ✗ NOT STARTED

| # | Task | Artifact | Est. |
|---|---|---|---|
| 3.1 | llama.cpp ≥ b10419 as **sibling checkout** (never vendored); pin the build | `integration/README.md`, `integration/*.patch` | 3 h |
| 3.2 | GGUF with MTP head (bartowski imatrix); verify `--spec-type draft-mtp` baseline | `results/phase3-mtp-baseline.md` | 4 h |
| 3.3 | Wire sparse path: decode + MTP verification only, dense prefill, KV bf16 | `integration/*.patch` | 15 h |
| 3.4 | Full grid on fixed prompts (short chat, 32K long-context, agentic edit loop) | `results/phase3-grid.md` | 8 h |
| 3.5 | Acceptance-length interaction: does union masking move E[accepted]? | same | 4 h |
| 3.6 | Final report | `results/phase3-*.md`, root `README.md` update | 3 h |

**Gate:** H3 — ≥1.25× over MTP-only, landing 45–55 tok/s.

---

## Cross-phase hygiene (binding)

- Every measurement run writes a `results/` entry per `results/README.md`: config;
  env (macOS build, engine versions, `iogpu.wired_limit_mb`, power state); 3 runs,
  median. Benchmarks: plugged in, High Power, ≥2-min cooldown between runs.
- Fixed seeds; greedy decoding for speed runs; identical quant artifacts across all
  configs; EvalPlus pinned to a fixed release.
- One model resident at a time; stop inference before compiles.
- Never commit weights, traces, or raw dumps (`.gitignore` covers them).
- Commit per milestone: `"phase0: <what>"`.
- Ask before: sudo, downloads beyond ~40 GB total, paid APIs, force-pushes.

## Download budget

| Item | Size | Running total |
|---|---|---|
| `mlx-community/Qwen3.8-27B-4bit` | 16.1 GB | 16.1 GB |
| WikiText-2-raw | ~0.02 GB | 16.1 GB |
| WikiText-2-raw (`datasets`) | 0.02 GB | 16.1 GB |
| ~~(Phase 3) bartowski GGUF + MTP~~ | ~~~17 GB~~ | not pulled — phase blocked |
| ~~(Phase 1) EvalPlus + code corpus~~ | ~~~0.5 GB~~ | not pulled — phase blocked |

**Actually downloaded: 16.1 GB**, well inside the ~40 GB budget.
