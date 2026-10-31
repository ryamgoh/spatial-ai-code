# Unique Pre-Training Review

Verdict: **correctness/allocation verified, but hold the main SFT launch pending
a shortcut-risk decision and reference cleanup**. This review changes no data,
labels, menus, quotas, splits or generation authorization.

## Findings

### 1. Scene Size Predicts E Without Spatial Reasoning

A single-split decision tree was trained using only object and premise counts.
It predicts E versus non-E, not the full A-F answer. Five fixed,
family-and-story-held-out folds were drawn solely from the 6,000 training
questions, separately by query. No dev, test, extrapolation or external examples
were used to train, tune or score these probes.

| Query | Size-only accuracy | Size-only balanced accuracy | Always non-E accuracy |
|---|---:|---:|---:|
| Direction | 94.20% | 93.86% | 70.00% |
| Which | 79.30% | 76.98% | 70.00% |
| Count | 94.70% | 94.79% | 70.00% |

For Direction and Count, all five folds learned the same rule:

> At most 13 premises: predict E. At least 14 premises: predict non-E.

The constant non-E baseline has 50% balanced accuracy. A second prespecified
depth-three tree using size, aggregate graph-degree and direction-frequency
counts reached 95.90% Direction, 80.85% Which and 95.90% Count accuracy.
Neither model receives object names, source labels, IDs, proof-depth metadata,
answer menus, reference explanations, semantic metadata or solver outputs.

This demonstrates an available shortcut, not evidence that Qwen has learned it.
It does not mean a model can choose the correct A-D/F letter with 94% accuracy.
Family/story grouping is not a claim of complete structural independence.
There are no confidence intervals or external-transfer results in this probe.

The mechanism is consistent with the measured source-answer table: all 1,400
controlled Direction/Count examples are A-D/F, and all 600 noncontrolled examples
are E. The controlled scenes tend to be larger. Naming, shuffling the menu or
hiding the source label cannot remove a premise-count shortcut.

This correlation is also constrained by the design: 1,400 controlled, sole-
entailed Direction/Count questions fill all 1,400 desired A-D/F slots. The
remaining 600 source slots must then all be E. Simply generating more candidates
under unchanged recipes and quotas does not remove that source-label relation.
Reducing its visible footprint requires broader scene-size/structure overlap;
changing source/answer independence itself would require a design decision.

### 2. Exact-Value Explanations Contain Redundant Rejection Lists

All 1,200 A-D Direction examples and 1,200 A-D Count examples per style include
explicit alternative-value rejection sections after the exact answer proof.
Across each style this contributes:

- 8,400 Direction negative-claim lines, seven per example.
- 21,201 Count negative-claim lines.

For consistent premises, a proof of one exact direction or exact total in every
arrangement already excludes the other exact values. The additional witnesses
and per-value rejection lists are sound but unnecessary supervision.
The existing F renderer already avoids them for Direction/Count, so the two
menu outcomes currently teach unnecessarily different explanation lengths.

Recommended reference-only change: retain the full exact-value proof, then a
short mutual-exclusivity statement and the option mapping. Preserve negative
evidence in the audit payload, even when not repeated in the student reference.
For Count, retain membership/exclusion reasoning needed to establish the exact
total; do not replace it with counting only individually entailed Which members.

In the inspected no-entailment Count example, two valid arrangements give totals
1 and 0, followed by a separate rejection of every domain count. Two distinct
totals already establish that no exact total is invariant. This can be a later
conciseness improvement, but must preserve checked arrangements and not conflate
Which's non-entailment obligations with mutually exclusive exact values.

### 3. Internal Terminology Remains In Both Styles

Full-corpus phrase counts, per style:

| Phrase | Examples |
|---|---:|
| "Unique requires exactly one" | 300 |
| "every contract requires E" | 375 |
| "No minimality claim is made" | 375 |
| "The complete entailed-answer set" | 3,600 |

These are not label errors, but the first three belong to implementation/audit
discussion more than student-facing reasoning. Natural conclusions should state
the actual result, for example:

- "Both Gym and High School must be northeast of Bank, so there is no single
  entailed answer. Answer: E."
- "These premises contradict each other, so no valid arrangement satisfies them.
  Under the task rules, the answer is E."
- "Library is the only object guaranteed to be north of Supermarket. It is not
  among A-D, so the answer is F."

These are proposed wording examples, not replacements already applied.
Symbolic conclusions can be compact while expressing the same obligations.
The Which F sample correctly contains a positive membership proof and
counterexamples for every alternative. Do not remove those uniqueness checks
in pursuit of shorter prose.

## Scope And Evidence

Read the renderer's obligation selection and inspected representative Natural
and Symbolic references for Direction A-D, Count A-D, Which multiple-entailment
E, Which F, and Count no-entailment E. Phrase counts and redundant-section counts
cover the full training artifact. This is not a manual review of all 12,000
style-specific explanations or a new independent proof of all spatial labels.
Existing full evidence and artifact audits remain authoritative for correctness.

Reproducer: `review_unique_pretraining.py`.
Detailed results, fold rules and sample references:
`UNIQUE-PRETRAINING-REVIEW-v1.json`.
Scikit-learn version: 1.9.1. Grouped-fold and model seed: 31031.
No GPU model training was performed; only the small diagnostic trees were fit.
Two feature tests pass: label/internal-metadata changes do not affect the input
features, and object renaming/premise permutation do not affect them.

Candidate manifest SHA-256:
`d3b93725dbafb0e920a3490fe5885d8378e8bdad9cb87f193656a7476e5fb42a`.
All candidate payload hashes were checked before and after the diagnostic.

## Recommended Next Decision

1. Approve a versioned reference-only cleanup: simplify exact-value conclusions
   and remove internal terminology, with unchanged premises, labels, menus,
   positive proofs and required Which uniqueness evidence. Re-render and
   re-audit both styles rather than editing the verified artifact in place.
2. Decide how to handle the measured shortcut before main training. A bounded
   diagnostic of size/structure overlap is preferable to another blind supply
   batch. It should determine whether existing candidates can improve overlap
   under the retained depth/answer constraints. Do not claim this is feasible
   or regenerate/rebalance without approval.
3. Alternatively, explicitly accept the limitation and use initial training as
   a pilot, with careful E/non-E and transfer reporting. Correct labels alone
   cannot establish that the model learned entailment rather than the shortcut.

No new distribution targets are imposed by this report. The completed allocation
goal remains completed as specified; scientific fitness for a main experiment
is a separate gate. This report does not authorize sealing or training.
