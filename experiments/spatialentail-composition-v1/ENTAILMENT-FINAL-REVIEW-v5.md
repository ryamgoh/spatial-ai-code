# Entailment Dataset Final Review

## Verdict

The named entailment-only candidate is generated, independently audited and
targetedly reviewed. It is suitable for the user's sealing decision, with the
scope and style limitations below. This is not a seal, a training-ready package,
or authorization to launch training.

Current artifacts:

- Synthetic: `entailment-candidate-v5/`.
- External evaluation: `spatialmap-entailment-all-five-v3/`.
- Synthetic audit: `SYNTHETIC-INDEPENDENT-AUDIT-v5.json`.
- External audit: `EXTERNAL-INDEPENDENT-AUDIT-v3.json`.
- Distribution: `ENTAILMENT-CANDIDATE-REVIEW-v5.md`.
- Paired examples: `ENTAILMENT-REFERENCE-SAMPLES-v5.md`.

## Reportable Method

We generate atomic spatial problems with controlled composition depths and
varied scene structures and irrelevant context. We compute the complete set
of individually entailed answers, then apply Unique, Listed and Exhaustive
answer contracts to fixed menus. Shared verified evidence is rendered as
matched Natural and Symbolic explanations. Distinct place names are sampled
from a fixed 31-name vocabulary using one consistent mapping per story.
Related constructions remain within one split, and fixed sampling and coverage
requirements determine the retained dataset.

## Verified Dataset

| Split | Bases | Direction / Selection / Count | Views per style |
|---|---:|---|---:|
| Train | 6,000 | 2,000 / 2,000 / 2,000 | 21,222 |
| Dev | 1,200 | 400 / 400 / 400 | 4,344 |
| In-range test | 2,400 | 800 / 800 / 800 | 8,622 |
| Extrapolation test | 1,200 | 400 / 400 / 400 | 4,335 |
| Total | 10,800 | 3,600 / 3,600 / 3,600 | 38,523 |

The synthetic candidate contains 6,245 retained stories and 2,314 families.
All configured source, depth, Count-value, family, direction and coverage gates
pass, with zero unresolved verification failures. The agreed base sizes and
70/15/10/5 in-range source mixture are unchanged.

Full independent synthetic replay covers 8,794 positive claims, 62,614
counterexamples, every view's label and all 77,046 input-only message rows.
The latest external replay covers 1,500 bases from 500 stories, 5,070 views
per style, 1,023 positive claims and 8,477 counterexamples. It checks the
original-name legends and all 10,140 input rows. Corr SINGLE is unchanged;
external data is not training or checkpoint-selection data.

Every synthetic input was scanned: no numbered `ObjectN` identifiers are
model-visible. All saved token admission records pass. Maximum Stage-2
reserved context is 5,016 tokens, below the retained 8,192 limit.

| Explanation style | Mean target tokens | Maximum target tokens |
|---|---:|---:|
| Natural | 334.94 | 1,002 |
| Symbolic | 348.04 | 1,019 |

These are reference lengths, not measured model-generated token costs.
Symbolic is not assumed to be shorter or more accurate.

## Targeted Explanation Review

Read all 22 paired samples spanning depths 1-10, all query types, all contracts,
empty entailment, inconsistency, omissions, multiple entailed answers, position
variants and sole-entailment-plus-contingent Selection.

Also inspected the corrected named Unique-unlisted Selection example
`production-b656-s25-which-m1-unique-entailed` and the corrected external
example `spatialmap.tqa.2001.1-m1-unique-entailed`. Both now prove the unlisted
candidate and refute entailment of every other eligible candidate.

Inspected the actual longest Natural reference
`production-b2830-s5-count-m0-unique-entailed` and longest Symbolic reference
`production-b846-s7-count-m0-unique-entailed`. Names, deductions and conclusions
remain coherent. Count zero is substantive, not an empty entailed-answer set.
No numeric coordinate assignment is printed in the reviewed supervision.

No new correctness issue was found in this targeted review. Natural remains
template-based instructional prose; mixed-axis paths can be long, and negative
claims are sometimes repetitive. Symbolic is readable notation with explanatory
text, not a purely formal calculus. This review is not a human-subject study or
a population-wide readability estimate.

## Scope and Safeguards

Positive synthetic explanations support axis paths and fixed-membership Count
proofs. The original admitted pool excluded one Count candidate because a
negative membership lacked a supported axis exclusion. It was not assigned a
guessed label or moved to another split. General solver Count invariance with
changing member identities passes a separate diagnostic; that non-atomic
fixture is explicitly rejected by atomic dataset admission.

Position augmentation affects 74 existing bases and adds 1,158 views per style.
It does not add independent bases or families. Report correlated exposure and
use base-level or family-clustered uncertainty, rather than treating every
view as an independent observation.

Pre-generation evidence includes the 268/268 training-source support check
(36.76 seconds) and the six-query development timing receipt (3.36 seconds for
entailment admission versus 19.90 for the historical possibility task).
The tasks differ, so this is not a production-wide or server speedup claim.

Internal and external evaluation use the same versioned entailment contracts.
Score Stage 1 and Stage 2 separately; only Listed directly exposes complete
displayed-option inclusion/exclusion. E/F alone does not classify every option.
Select model checkpoints using synthetic Stage-1 dev accuracy, never Corr.

## Audit Binding

Synthetic manifest SHA-256:
`f326b91aeec015401f28390a453886a0ee8a195efd8fc2b4dd934a7bbbb0c452`.

External manifest SHA-256:
`74f4bace92d881797f11c5b07faffa103d5b9dedb68c13e6cdd2aa564e012486`.

This separate report records completion of targeted manual review after the
creation-time pending flags. Candidate manifests remain unchanged so their
independent audit bindings remain valid. All historical candidates, receipts
and possibility-based packages are preserved.

Next gate: separate user approval to seal this named candidate. Training
packaging and GPU preflight follow that approval; no model training has run.
