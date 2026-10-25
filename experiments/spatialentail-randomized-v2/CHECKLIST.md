# SpatialEntail FYP execution checklist

This is the operational source of truth for the remaining SpatialEntail work.
Do not add a new experiment to the required path unless it replaces an existing
item or directly answers an FYP research question.

Status markers:

- `[x]` complete and evidenced
- `[ ]` required and incomplete
- `[~]` running or awaiting artifacts
- `[-]` optional if time remains
- `[>]` deferred future work

Priority levels:

- **P0:** required to complete the FYP; may run in parallel when dependencies permit
- **P1:** valuable extension only after the P0 path is secure
- **P2:** deferred future work; must not delay submission

## Completion definition

The randomized-v2 study is complete when:

1. all six adapters finish from the pinned base model;
2. the untuned model and six adapters are evaluated on the frozen 3,600-row
   benchmark;
3. results are summarized by contract, partition, query, G, L, and X;
4. the primary SINGLE and MULTI comparisons are written into the report; and
5. at least the frozen SpatialMap-TQA-Corr external-transfer evaluation is
   reported for the central Natural models.

## A. P0 — Frozen data and benchmark — complete

- [x] Five-rule randomized structural calibration completed.
- [x] Wrong-G lift defect fixed and regression-tested.
- [x] Count COMPOSITION-OOD minima corrected.
- [x] Empirical-operational Count capacity caps recorded.
- [x] Frozen benchmark candidate generated: 3,600/3,600 unique structures.
- [x] Benchmark audit passed with exact G/L/X and transition contracts.
- [x] Benchmark source SHA-256 frozen:
  `523341d95ccc9f7baa1198c5dc451ed4eea76d50979f4c9aa7d1336efab9a97d`.
- [x] Nested training/development candidate generated: 15,400 unique bases.
- [x] Benchmark overlap is zero; train/development overlap is zero.
- [x] Pinned Qwen tokenizer audit passed; all rows fit `sequence_len: 4096`.
- [x] Lean Axolotl exports and six job configs prepared.

Evidence:

- `benchmark-candidate-audit.json`
- `training-candidate-audit.json`
- `prepared-training-candidate/token-audit.json`
- `prepared-sft/jobs.json`

## B. P0 — Six central SFT jobs — awaiting training receipts

All jobs start independently from pinned `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

All six adapter evaluation outputs were imported and validated on 2026-10-25.
The evaluation archives do not contain the training completion markers,
adapter hashes, W&B receipts, or training logs, so the training evidence below
remains awaiting artifacts rather than being marked complete.

| Adapter | Seed | Status | Slurm ID | Completion evidence |
|---|---:|---|---|---|
| SINGLE Natural | 42 | [~] | fill in | `models/single-natural-seed-42/COMPLETED` |
| SINGLE Natural | 43 | [~] | fill in | `models/single-natural-seed-43/COMPLETED` |
| SINGLE answer-only | 42 | [~] | fill in | `models/single-answer-only-seed-42/COMPLETED` |
| SINGLE answer-only | 43 | [~] | fill in | `models/single-answer-only-seed-43/COMPLETED` |
| MULTI Natural | 42 | [~] | fill in | `models/multi-natural-seed-42/COMPLETED` |
| MULTI Natural | 43 | [~] | fill in | `models/multi-natural-seed-43/COMPLETED` |

For each completed job, record:

- final checkpoint path and adapter hash;
- W&B run URL/ID;
- elapsed time and peak GPU memory;
- final train/evaluation loss;
- whether the job resumed from a checkpoint; and
- confirmation that the `COMPLETED` marker and adapter files exist.

## C. P0 — Frozen-benchmark evaluation

Models:

- [x] Untuned Qwen3.5-4B, native-thinking and two-stage.
- [x] SINGLE Natural seed 42.
- [x] SINGLE Natural seed 43.
- [x] SINGLE answer-only seed 42.
- [x] SINGLE answer-only seed 43.
- [x] MULTI Natural seed 42.
- [x] MULTI Natural seed 43.

Evidence imported on 2026-10-25:

- `prepared-eval/jobs.json` and `prepared-eval/results/<model>/COMPLETED`.
- Seven full runs, each with 48 tasks and 3,600 unique labeled examples.
- [Evaluation summary](evaluation-summary/SUMMARY.md) and
  [machine-readable breakdowns](evaluation-summary/summary.json).
- Saved per-example metrics agree with `results.json`; all three imported
  batches share the frozen benchmark hash and the same labeled examples.
- This validates the evaluation artifacts, not training provenance or an
  independent regeneration of the benchmark.

Evaluation checks:

- [x] Native stage-1 extraction recorded.
- [x] Stage-2 constrained-choice score recorded as the primary metric.
- [x] Exact-set scoring supports one through four selected options.
- [x] Overall and macro partition scores: MATCHED, LENGTH-OOD, NOISE-OOD,
  COMPOSITION-OOD.
- [x] Direction, Selection, and Count breakdown.
- [x] G1--G4 breakdown.
- [x] L and X curves (tabulated and plotted by contract and training seed).
- [x] Held-out transition breakdown.
- [x] Seed 42/43 results reported separately plus mean and range.
- [x] Sample outputs and diagnostic stage-outcome categories retained.
  Causal reasoning-error classification has not been performed.

Primary comparisons:

- [x] SINGLE Natural versus SINGLE answer-only.
- [x] MULTI Natural versus untuned.
- [x] MULTI Natural versus SINGLE Natural transfer on ALL-POSSIBLE.
- [x] SINGLE retention after MULTI training.

These comparisons are summarized locally; dissertation integration remains
open in section E. SINGLE is the 900-row G1 subset, not the full mixed-contract
benchmark. MULTI Natural averages 99.08% overall and 98.85% on ALL-POSSIBLE
across the two seeds. SINGLE Natural averages 99.89% on SINGLE but 0.30% on
ALL-POSSIBLE; its approximately 25% overall score must not be described as
failure on its trained contract.

Research follow-up on 2026-10-25:

- [x] Replay all 61,200 saved responses through the original scorer.
- [x] Compare every saved prompt payload with hash-bound benchmark messages.
- [x] Audit candidate train/development overlap and held-out proof transitions.
- [x] Analyze answer cardinality, missing/extra options, and paired checkpoint
  differences with fixed-cell, checkpoint-conditional bootstrap intervals.
- [x] Export four research figures as PNG and PDF.
- [x] Document extraction sensitivity, token-budget proxies, and stage-2
  regressions without modifying the primary scores.
- [Research findings and limitations](research-analysis/REPORT.md).
  Candidate/preparation provenance is checked; actual cluster training
  consumption, finish reasons, and final checkpoint hashes remain unverified.

## D. P0 — SpatialMap-TQA-Corr external transfer

- [x] Prepare prompt-matched evaluation configs for the central Natural models,
  both answer-only seeds, and untuned baseline.
- [ ] Evaluate untuned, SINGLE Natural 42/43, and MULTI Natural 42/43 on the
  frozen SpatialMap-TQA-Corr release.
- [ ] Report native contract metrics; do not merge them with SpatialEntail.
- [ ] Record whether MULTI training improves, preserves, or degrades external
  SINGLE-style spatial reasoning.

`prepared-corr-eval/` contains seven hash-bound jobs, not submitted. It uses
the current 1,500-row SINGLE correction (667 determinate, 833 underdetermined),
not the legacy 1,329-row task. The corrected data were regenerated and checked
against the documented counts; GPU runtime remains untested.
See the [cluster handoff](research-analysis/RUNBOOK.md).

## E. P0 — Report closure

- [ ] Record six training-job receipts and hashes.
- [ ] Add randomized-v2 headline result tables.
- [ ] Add stage-1/stage-2 comparison.
- [ ] Add partition, query, G, L, and X analyses.
- [ ] Add external-transfer results.
- [ ] Discuss seed sensitivity without claiming a seed-level confidence
  interval from two runs.
- [ ] State that certificates are offline audit artifacts, not SFT targets.
- [ ] State that Natural traces are controlled solver-rendered explanations,
  not human-authored prose.
- [ ] Document calibration overlap, empirical-operational caps, and the bounded
  propositional-logic limitation.
- [ ] Compile and visually review the final PDF.

## F. P1 — Optional extensions if time remains

- [x] Qwen3.5-4B thinking-off and fresh thinking-on runs imported and
  summarized separately from the P0 untuned run.
- [x] Recommended <=9B comparator panel evaluated: Qwen3.5-9B thinking off/on,
  Gemma-4-E4B-it (or explicitly labelled W4A16 QAT), Gemma-3-4B-IT
  instruction-only, Ministral-3-8B-Instruct, DeepSeek-R1-Distill-Qwen-7B,
  Ministral-3-8B-Reasoning, and Phi-4-mini-instruct.
- General and reasoning evidence: `prepared-general-eval/jobs.json`,
  `prepared-reasoning-eval/jobs.json`, and their full-run `results/`
  directories. All ten runs pass the 48-task/3,600-example summary checks.
  The reasoning runs' low native stage-1 extraction accuracy (1.28%--4.19%)
  requires response-budget/extraction inspection before attributing differences
  to model reasoning ability. P0 and fresh reasoning settings are not identical.
- [x] Reasoning batch generator prepared: fresh Qwen3.5-4B/9B thinking-on,
  Ministral-3-8B-Reasoning, and DeepSeek-R1-Distill-Qwen-7B, with H200
  smoke/full chains and recorded fixed-seed sampling.
- [-] Fixed five-model spatial-specialist panel: SpatialASM-Qwen3.5-9B,
  SenseNova-SI-1.2-InternVL3-8B, SpaceQwen3-VL-2B-Thinking,
  RoboBrain2.5-8B-NV, and Cosmos-Reason2-8B. Require a text-only preflight
  before each full run; do not introduce dummy images.
  No spatial-specialist result archive was supplied in this import.
- [x] Spatial-specialist batch generator prepared with separate one-question
  preflight, 48-cell smoke, and full H200 chains. All prompts are text-only;
  SenseNova uses the sole remote-code loader while the other four use vLLM.
- [-] Paired VISIBLE-POSSIBLE transfer diagnostic: 180 bases, 720 VISIBLE views,
  and 180 ALL controls.
- [-] Special-response diagnostic: direct L0, ambiguous SINGLE, and Selection
  no-match.
- [-] Paired strict-cycle consistency diagnostic.
- [-] Additional external datasets: StepGame, Text2Space, SpartQA-Human, ReSQ.
- [-] Detailed option-position and paired-contract consistency analysis.

## G. P2 — Deferred future work

- [>] Models above the agreed approximately 9B class are outside the FYP
  experiment scope; do not prepare or run a larger-model sweep.
- [>] General capacitated Z3 allocation and capacity-provenance system.
- [>] Chunked resumable generation and atomic publication pipeline.
- [>] Exhaustive proof of minimum-Count canonical capacity.
- [>] VISIBLE-specific SFT.
- [>] Compact symbolic token-efficiency arm.
- [>] GRPO or verifier-reward training.
- [>] More seeds, larger models, or publication-scale benchmark expansion.

## Scope rule

P0 work may overlap when dependencies permit: evaluation preparation and the
untuned benchmark run can proceed while SFT jobs train. Do not begin P1 until
the six adapters, frozen-benchmark evaluation path, Corr transfer path, and
report templates are secure. P2 is explicitly outside the current FYP
execution path.
