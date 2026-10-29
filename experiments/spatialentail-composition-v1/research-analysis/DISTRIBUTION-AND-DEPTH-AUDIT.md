# Frozen Dataset: Answer Mix and Proof Depth

## Verdict

The frozen data has verified labels and reproducible artifacts, but its answer
mix and depth coverage are weak for a controlled spatial-composition study.
Recommend revising sampling before training. This is a diagnostic recommendation,
not authorization to regenerate, change the study, or launch jobs.

The earlier freeze established semantic/evidence checks, all 33 semantic cells
in every split, and token admission. It did not establish adequate depth
coverage or protection against contract-only answer shortcuts. Those limits
should have been surfaced more clearly before presenting preparation as complete.

## Scope and Counting

Inspected `prepared-training-v2`, generated from `study-candidate-v2`.
The v2 seal and current source hashes verify. No frozen artifact was changed.
This report is outside the sealed package.

- 200 premise sets, each with Direction, Selection and Count questions:
  600 underlying questions.
- Three contracts and one or two menus per question: 2,652 examples per style.
- Natural and Symbolic share the same labels and splits. Counts below are
  per style, not doubled for the two renderings.
- Depth counts deduplicate contract/menu copies by `base_id`.

## E/F Frequency

| Contract | Examples | E | F | E or F |
|---|---:|---:|---:|---:|
| Unique | 884 | 620 | 130 | 750 / 884 = 84.8% |
| Listed | 884 | 69 | 169 | 238 / 884 = 26.9% |
| Exhaustive | 884 | 69 | 481 | 550 / 884 = 62.2% |
| Total | 2,652 | 758 | 780 | 1,538 / 2,652 = 58.0% |

The combined rate is 58.3% in training, 57.7% in dev, and 57.5% in test.
This is not merely a test split fluctuation.

E/F answers are valid parts of the task, not automatically incorrect or trivial.
However, their concentration matters:

- Unique has only 134 substantive A-D answers in the full dataset.
- Unique training has 49 substantive Direction answers, 9 substantive
  Selection answers, and 29 substantive Count answers.
- Predicting the correct escape does not necessarily demonstrate a complete
  spatial answer. The pivot explicitly limits what can be inferred from F.
- Reference explanations are supervised too; imbalance does not prove that a
  trained model will exploit a shortcut. It makes answer-only aggregate
  performance less informative.

### A Premise-Blind Baseline

Select the most frequent complete answer separately for each contract using
training labels only:

| Contract | Fixed Prediction |
|---|---|
| Unique | E |
| Listed | ABCD |
| Exhaustive | F |

This fixed rule achieves:

| Split | Correct | Accuracy |
|---|---:|---:|
| Train | 776 / 1,596 | 48.6% |
| Dev | 271 / 522 | 51.9% |
| Test | 274 / 534 | 51.3% |

It ignores all premise content. It is a calculated answer-selection baseline,
not a model run, and does not assess explanation correctness. The three
contracts have equal example counts, so overall accuracy here also equals
macro-contract accuracy.

The 58% combined E/F share is NOT an always-one-letter accuracy: always F
scores 29.4% over all examples. E and F cannot be returned together.

### Why It Happens

The current source sampler chooses query targets and relations without semantic
quotas. Of the 200 underlying questions of each type:

- Direction: 101 underdetermined, 76 unique, 23 inconsistent.
- Count: 130 underdetermined, 47 unique, 23 inconsistent.
- Selection: 126 underdetermined, 11 unique, 25 no-match, 15 multiple-certain,
  23 inconsistent.

Menu construction further increases escapes. Each feasible full menu receives
one deliberately incomplete counterpart; all contracts are retained for each
menu. Exhaustive has:

- 284 incomplete menus introduced by that deliberate omission policy.
- 172 incomplete menus where the full possibility set exceeds four slots.
- All 456 incomplete menus return F.
- Another 25 F answers come from empty Selection sets; 69 E answers are
  contradictions.

Unique E is mostly underdetermination (510 examples), plus inconsistency (69),
no-match (25), and multiple-certain-match (16). Unique F adds 130 cases whose
unique value was deliberately omitted. Listed F contains 144 no-listed-possible
cases and 25 empty sets.

These counts follow the current contract semantics. The concern is sampling
adequacy, not a detected label bug.

## Actual Recorded Depth

The active pivot defines chain length as the number of premise links in a
supporting axis path, recording x and y separately. It explicitly says this
need not be the minimum reasoning complexity.

Current generation populates `supporting_axis_path_lengths` only for consistent,
uniquely determined Direction questions with an available axis proof. There are
76 such questions. In this release, x and y lengths agree in every one of them.

| Recorded Links Per Axis | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| 1 | 35 | 9 | 11 | 55 |
| 2 | 12 | 1 | 2 | 15 |
| 3 | 1 | 3 | 1 | 5 |
| 4 | 1 | 0 | 0 | 1 |
| 5-8 | 0 | 0 | 0 | 0 |
| Measured Direction Questions | 49 | 13 | 14 | 76 |

Consequences:

- 72.4% of measured Direction questions have one-link support paths.
- Only six underlying questions have three or four links.
- The only four-link question is in training. Its six contract/menu views are
  not six independent four-link problems.
- Neither dev nor test contains a recorded four-link Direction question.
- The other 524 underlying questions have no recorded chain-length value:
  Selection/Count, nonunique Direction, and inconsistent Direction. Missing
  is not zero, and does not prove their certificates contain no longer paths.
- Therefore "all proofs have depth 1-4" is too broad. The defensible statement
  is "recorded supporting paths for 76 unique Direction questions range 1-4."
- These measurements do not establish a balanced depth-controlled benchmark,
  or a universal final depth for Selection/Count.

The targeted chain recipe varies its length as `2 + sample % 3`. Random
questions are not selected to meet depth quotas. Family splitting is stratified
by source recipe, not measured depth, which explains why semantic-cell coverage
can pass while four-link evaluation coverage is absent.

## The Remembered 1-4 / 5-8 Plan

That plan exists in the historical workflow:

- Train/dev depth must not exceed four.
- The depth benchmark includes matched depth-3/4 anchors and extrapolative
  depth-5 through depth-8 cases.
- The historical release YAML contains corresponding question-type strata.

However, `spatial/v2/study/README.md` explicitly labels this workflow historical,
not the current composition execution plan. The current pivot states that
depth extrapolation is not a required research question. It still requires
varied chain lengths and freezing the final ranges, but does not mandate the
old 5-8 benchmark.

Thus the remembered bands are real, but they were not carried into the current
sealed dataset. The 5-8 objects used in random generation is an object-count
range, not a depth-5-8 guarantee.

## Recommended Decision Before Training

1. Preserve this release as the reproducible baseline; do not overwrite it.
2. Set semantic-case and menu proportions deliberately at the underlying
   question/story level. Retain paired contracts and Natural/Symbolic views.
   Do not independently downsample answer letters or falsify E/F labels.
3. Increase uniquely answerable questions, especially Selection, and reduce
   the mandatory one-for-one omitted-menu expansion. Keep genuinely large
   possibility sets and empty sets with their correct escapes.
4. Set minimum counts for each intended supporting-path length in every split.
   For an in-range study, 1-4 can remain the range, but each level needs useful
   representation. Distinguish recorded support length from minimum depth.
5. Define appropriate membership-path descriptors for applicable Selection/
   Count proofs; do not assign one arbitrary depth to ambiguity or conflicts.
6. Decide explicitly whether to restore depth-5-8 extrapolation. That would
   extend the current study scope, not merely fix an omitted required test.
7. Before a replacement freeze, review the full answer/depth cross-tables,
   no-premise baseline, unique base counts, and token/evidence admission.

No universal E/F target percentage follows from the task definition. The
objective is adequate substantive and compositional coverage, not cosmetic
uniformity. All changes above remain proposals; no new dataset was generated.

## Evidence

- [Current pivot scope](../pivot_20261026.md#direction).
- [Current chain-length definition](../pivot_20261026.md#dataset-descriptors).
- [Current sampling requirements](../pivot_20261026.md#minimum-dataset).
- [Frozen source and menu policy](../GENERATION-FREEZE.md#fixed-attempt-budget).
- [Current path measurement code](../../../spatial/v2/composition_pilot.py#L378-L397).
- [Historical workflow status](../../../spatial/v2/study/README.md).
- [Historical depth constraints](../../../spatial/v2/study/build.py#L182-L193).
- [Historical release strata](../../../spatial/v2/study/release-candidate.yaml).

Inspected release hashes:

```text
manifest:   e01aaed5063863758de15d3af196bfdf46ceb9cf94bb34af6467cf59b4dd4cea
references: 09de9e05d46759192bead645538e457c0147bef891b521d07399aeec825fde4f
seal:       13606cf879e8e6187c42a3491c7dd6a1f1baa5c7abca478453b62eebc51ca343
```

Counts were computed from the sealed `references.jsonl` and saved candidate
`bases[*].checks.supporting_axis_path_lengths`, deduplicating by `base_id`.
The majority baseline is selected from training frequencies and then applied
unchanged to dev/test. No model was run; no new semantic/evidence audit was
claimed.
