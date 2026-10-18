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

## Completion definition

The randomized-v2 study is complete when:

1. all six adapters finish from the pinned base model;
2. the untuned model and six adapters are evaluated on the frozen 3,600-row
   benchmark;
3. results are summarized by contract, partition, query, G, L, and X;
4. the primary SINGLE and MULTI comparisons are written into the report; and
5. at least the frozen SpatialMap-TQA-Corr external-transfer evaluation is
   reported for the central Natural models.

## A. Frozen data and benchmark

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

## B. Six central SFT jobs

All jobs start independently from pinned `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

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

## C. Required frozen-benchmark evaluation

Models:

- [ ] Untuned Qwen3.5-4B.
- [ ] SINGLE Natural seed 42.
- [ ] SINGLE Natural seed 43.
- [ ] SINGLE answer-only seed 42.
- [ ] SINGLE answer-only seed 43.
- [ ] MULTI Natural seed 42.
- [ ] MULTI Natural seed 43.

Evaluation checks:

- [ ] Native stage-1 extraction recorded.
- [ ] Stage-2 constrained-choice score recorded as the primary metric.
- [ ] Exact-set scoring supports one through four selected options.
- [ ] Overall and macro partition scores: MATCHED, LENGTH-OOD, NOISE-OOD,
  COMPOSITION-OOD.
- [ ] Direction, Selection, and Count breakdown.
- [ ] G1--G4 breakdown.
- [ ] L and X curves.
- [ ] Held-out transition breakdown.
- [ ] Seed 42/43 results reported separately plus mean and range.
- [ ] Sample outputs and failure categories retained.

Primary comparisons:

- [ ] SINGLE Natural versus SINGLE answer-only.
- [ ] MULTI Natural versus untuned.
- [ ] MULTI Natural versus SINGLE Natural transfer on ALL-POSSIBLE.
- [ ] SINGLE retention after MULTI training.

## D. Required external transfer

- [ ] Prepare prompt-matched evaluation configs for the central Natural models.
- [ ] Evaluate untuned, SINGLE Natural 42/43, and MULTI Natural 42/43 on the
  frozen SpatialMap-TQA-Corr release.
- [ ] Report native contract metrics; do not merge them with SpatialEntail.
- [ ] Record whether MULTI training improves, preserves, or degrades external
  SINGLE-style spatial reasoning.

## E. Required report closure

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

## F. Optional if time remains

- [-] Paired VISIBLE-POSSIBLE transfer diagnostic: 180 bases, 720 VISIBLE views,
  and 180 ALL controls.
- [-] Special-response diagnostic: direct L0, ambiguous SINGLE, and Selection
  no-match.
- [-] Paired strict-cycle consistency diagnostic.
- [-] Additional external datasets: StepGame, Text2Space, SpartQA-Human, ReSQ.
- [-] Detailed option-position and paired-contract consistency analysis.

## G. Deferred future work

- [>] General capacitated Z3 allocation and capacity-provenance system.
- [>] Chunked resumable generation and atomic publication pipeline.
- [>] Exhaustive proof of minimum-Count canonical capacity.
- [>] VISIBLE-specific SFT.
- [>] Compact symbolic token-efficiency arm.
- [>] GRPO or verifier-reward training.
- [>] More seeds, larger models, or publication-scale benchmark expansion.

## Scope rule

Complete sections B--E before starting an item in F. Section G is explicitly
outside the current FYP execution path.
