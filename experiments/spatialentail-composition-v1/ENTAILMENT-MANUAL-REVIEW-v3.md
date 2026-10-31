# Paired Explanation Review

## Verdict

The v3 candidate passes automated label, evidence and coverage checks, but is
not approved for sealing. Manual inspection found a missing printed uniqueness
justification in a specific Selection escape case. Repair those references
without changing questions, labels, splits, menus or saved evidence.

## Reviewed Scope

Read the 22 deterministic paired samples in
`ENTAILMENT-REFERENCE-SAMPLES-v3.md`, plus:

- The actual longest Natural reference:
  `production-b2892-s5-count-m0-unique-entailed` (1,139 tokens).
- The actual longest Symbolic reference:
  `production-b2713-s4-count-m0-unique-entailed` (1,002 tokens).
- All three omission contracts for
  `production-b2916-s6-which-m1`: Unique E, Listed ABD, Exhaustive F.
- Unique unlisted Selection:
  `production-b656-s25-which-m1-unique-entailed`.

This is targeted review, not a random sample or a measured population-wide
readability score. Automated evidence checks cover the full dataset separately.

## Finding Requiring Repair

Unique Selection requires exactly one individually entailed eligible candidate,
even if that candidate is absent from A-D. The unlisted-reference branch prints
its positive proof and concludes "The sole entailed answer is unlisted", but
omits the counterexamples that establish non-entailment of the other eligible
candidates. Proving one candidate alone does not prove uniqueness for Selection.

Affected scope: 429 internal views (214 train, 48 dev, 97 in-range, 70
extrapolation) and 45 external views. These labels remain correct according to
the full independent audit; the missing part is printed explanation support.
The corrected renderer must retain full-domain evidence for this branch.
Direction and Count do not need the same repair: two distinct exact values
cannot both hold in one valid arrangement.

## Readability Findings

The two styles express matching deductions and distinguish universal proofs
from counterexample arrangements. Count zero is treated as a substantive answer.
The sampled Selection omission contrast follows the intended contracts.
No numeric coordinate assignments are printed in the reviewed explanations.

Natural is readable, deterministic instructional prose, not free-form
human-authored reasoning. Mixed-axis paths can produce long semicolon-heavy
sentences. Direction and fixed Count explanations repeat one negative sentence
per alternative after already proving an exact value; these sentences are
correct but unnecessarily repetitive. Symbolic is a compact hybrid of notation
and explanatory text, not a purely symbolic calculus.

These style limitations should not be concealed or described as a human
readability study. They do not warrant new scenes or altered labels. No broad
style rewrite is authorized by this review; address the proof-support gap first.

## Required Follow-Up

Regenerate only affected paired references, recheck their token budgets, verify
byte-identical model-input exports, rerun the independent candidate audit, and
inspect corrected examples. Preserve v3 as a historical unsealed candidate.
Obtain separate user approval before sealing or model training.
