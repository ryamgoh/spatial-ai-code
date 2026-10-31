# Existing-Pool Length Check

Verdict: **reselection from the existing pool cannot remove the measured
Direction/Count premise-count shortcut while preserving the current constraints**.
No data, quota, prompt, reference, source assignment or evaluation set was changed.
No new generation ran.

## Result

The fixed rule is: at most 13 premises predicts E; at least 14 predicts non-E.
Its classification task is E versus non-E, not choosing the correct A-F letter.

| Query | Current rule accuracy | Accuracy cannot fall below, under any allowed reselection |
|---|---:|---:|
| Direction | 94.20% | 88.55% |
| Count | 94.70% | 89.00% |

These are optimistic lower bounds on shortcut accuracy, not achieved alternative
selections. Family coverage, E-subtype coupling, negative-evidence coverage and
menu availability may prevent reaching even these lower values. The non-E
majority baseline is 70%.

At most, reselection could reduce this fixed rule's accuracy by 5.65 percentage
points for Direction and 5.70 for Count. A newly fitted classifier could find
another threshold or feature, so defeating this one rule would not establish
absence of structural shortcuts in any event.

## Why

The entire replayed pool contains 15,450 independently checked training
candidates. Random-world and random-atoms scenes have only 3-9 premises.
Controlled scenes span 6-32 premises but are predominantly long.

For Direction and Count, all controlled candidates have a sole entailed value.
The required 1,400 controlled questions therefore consume every one of the
1,400 A-D/F training slots. The remaining 600 noncontrolled slots must be E.
Short random-source A-D/F questions exist, but cannot be selected under these
simultaneous constraints without exceeding the target number of non-E answers.

### Counter-Shortcut Supply

To make the fixed rule wrong, a selected question must be either short non-E or
long E. Count each underlying question once, regardless of its menu variants.

| Query | Maximum short controlled non-E under depth quotas | Maximum long noncontrolled E | Maximum rule errors out of 2,000 |
|---|---:|---:|---:|
| Direction | 151 | 78 | 229 |
| Count | 147 | 73 | 220 |

Direction short support is capped at 112 L1 + 33 L3 + 6 L4 questions.
Count short support is capped at 112 L1 + 30 L3 + 4 L4 + 1 L5 questions;
the diagnostic also respects each controlled depth/count-value quota.
Other depths have no eligible short examples in this pool.

Only targeted sources supply long E cases: 78 Direction and 73 Count.
Selecting all of them is an optimistic assumption, not a joint feasibility
claim. Random-world and random-atoms sources supply no long E examples.

Thus the error upper bounds are 151 + 78 = 229 and 147 + 73 = 220.
Shortcut accuracy is at least 1 - errors / 2,000, producing the stated bounds.
A full selection optimizer is unnecessary to prove this obstruction.

## Which

Which is different: controlled candidates include both E and non-E.
The pool has 1,679 long controlled E candidates and 108 short controlled non-E
candidates, plus 196 short noncontrolled non-E candidates and 82 long targeted
E candidates. These are raw supply counts, not simultaneous selection capacity.
No constrained optimum or shortcut-free Which selection was established.

Optimizing Which alone would not resolve the proven Direction/Count issue, so
this check does not replace any part of the selected dataset.

## Concrete Trade-Off

Keep the task semantics and 1,200/600/200 answer targets. The existing pool is
not sufficient to solve the size issue while keeping every generation and
allocation choice fixed.

A next proposal could broaden the size/structure of semantic E scenes so that
long scenes also include E, while retaining their true non-depth or verified
decision-depth status. Context must be generated coherently and independently
checked; extra sentences cannot earn fictitious proof-depth credit. Adding
objects can change all-scene Which/Count domains, so neither arbitrary padding
nor copied labels is acceptable.

That is a generator/presentation change requiring approval, versioned artifacts,
reverification and a repeated shortcut diagnostic, not an approved action here.
Alternatives include changing the controlled/depth allocation or explicitly
accepting the shortcut as a pilot limitation. No numerical repair budget or
success guarantee is established by this report.

Reference-only prose cleanup is independent of this issue: it can remove
redundant explanations but cannot change a shortcut in the user question.
No reference cleanup was performed during this bounded pool check.

## Evidence

Reproducer: `check_unique_length_supply.py`.
Machine-readable census and per-depth/count-value bounds:
`UNIQUE-LENGTH-SUPPLY-CHECK-v1.json`.

The diagnostic hash-checks the replayed full pool and selected artifact, and
derives the exact depth/count-value/source quotas from the already validated
selection. It uses no dev/test/external examples. The threshold is the one
identified in the preceding training-only review, not a newly tuned threshold.

This is a necessary-bound analysis, not a new model evaluation or a completed
rebalancing. The existing allocation and correctness audits remain valid; the
main SFT launch recommendation remains on hold pending a quality decision.
