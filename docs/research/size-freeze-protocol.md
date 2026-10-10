# SpatialEntail data-size freeze protocol

Status: predeclared decision procedure, 2026-10-11. No final training size is
selected because no SFT learning curve has been run.

## Distinct quantities

- **Feasibility pilot size:** examples used to test generation, certificates,
  language quality, and context admission. It does not select training size.
- **Maximum candidate pool:** the largest formally admitted, quality-screened
  nested pool available for calibration.
- **Selected training size:** the smallest nested subset satisfying the
  development-set saturation rule.
- **Evaluation size:** held-out examples chosen for reporting precision, not
  included in any training-size total.

## Freeze sequence

1. Freeze the supported benchmark-cell registry. A cell is a declared
   `query contract × support/rule structure × semantic status × evidence burden`
   combination, not merely a depth or template label.
2. Run a feasibility pilot with 50 accepted examples per ready cell. Record
   rejection reasons, acceptance rate, structural duplicates, evidence use,
   semantic interventions, and Natural/Symbolic lengths.
3. Run the source-blinded two-model quality screen on the pilot. Split verdicts
   are held out; unanimous rejections reach the selective human appeal queue.
   Revise failing generators before scaling.
4. Set minimum coverage constraints across reporting cells, query families,
   reasoning structures, semantic strata, directions, and answer positions.
   Use the smallest integer allocation satisfying all constraints. Do not give
   a cell more examples merely because its path depth is larger.
5. Generate an oversupplied maximum pool, apply 100% formal checks and exact
   context admission, run the full quality screen, and then construct nested
   subsets by deterministic stratified prefixes.
6. Run checked-Symbolic calibration with seed 42 on both model sizes. Use the
   development set only; the final test remains sealed.

## Nested sizes

Let `b_c` be the minimum training allocation for frozen reporting cell `c` and
let `N_min = sum_c b_c`. Candidate nested sizes are complete multiples of this
coverage block, provisionally `N_min`, `2N_min`, and `4N_min`, subject to the
approved compute ceiling. Report both base-problem counts and supervised target
tokens. Do not force totals to 4K/8K/17K if those values split cells or violate
coverage.

## Saturation rule

For each adjacent pair of sizes and for both Qwen3.5-2B and Qwen3.5-4B, compute
on the development set:

- macro strict accuracy over frozen reporting cells;
- full Symbolic validity;
- DIR, SEL, and CNT macro accuracy;
- RR, LR, and MTR lens summaries; and
- held-out composition accuracy where available.

Select the smaller size only when:

1. the larger size improves macro accuracy by less than 2 percentage points;
2. the larger size improves full Symbolic validity by less than 2 points;
3. neither model has a protected query/lens regression beyond the predeclared
   material-regression tolerance; and
4. paired item-level uncertainty does not leave a practically important gain
   unresolved.

The material-regression and inconclusive-interval thresholds must be fixed
before calibration. If no adjacent comparison saturates, select the largest
tested size and report that saturation was not observed within budget; do not
call it optimal.

## Role of quality judges

The `DeepSeek-V4-Pro` and `GPT-5.5` quality screen determines whether sampled
prompts and checked Natural traces are acceptable for use. It does not choose
the number of training examples. The selective human appeal checks only
unanimous model rejections for false-positive rejection and therefore cannot
estimate population-level human acceptability.

## Current evidence

The current feasibility registry has thirteen ready representative cells and no
blocked cells. The seed-1911 v3 pilot generated 50 checked Natural rows per
cell, 650 total. The 65-item quality packet sampled five per cell. Both model
judges approved all 65 items, leaving no split verdicts or human appeals. These
receipts establish pilot generation and automated presentation quality for the
representative cells only; they do not determine `N_min`, the maximum pool, or
the selected training size.

See the tracked [v3 pilot audit](../audits/capability-pilot-v3.md) for artifact
hashes and [`human-review-workflow.md`](human-review-workflow.md)
for routing and appeal semantics.
