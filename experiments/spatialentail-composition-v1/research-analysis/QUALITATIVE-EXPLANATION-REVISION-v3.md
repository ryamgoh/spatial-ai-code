# Qualitative Explanation Revision

## Presentation Rule

Numerical coordinates belong in audit evidence, not in Natural or Symbolic
supervision. This clarification is now recorded in `pivot_20261026.md`.
The new renderer version is `readable-qualitative-evidence-v3`.

For Direction ambiguity, show relevant horizontal/vertical orders and their
different outcomes. Include common premise anchors where available, such as O
when both X and Y are east of O. Do not assert that an illustrative ordering is
entailed by the premises.

For Selection and Count, express candidates' spatial relationships to the query
reference, grouping candidates with the same relation in Natural. Symbolic gives
the matching predicates. State the resulting qualifying set and, for Count,
its joint size. These describe jointly possible alternatives, not independently
possible memberships combined without verification.

Numerical completions remain in full evidence and in the outline's
`audit_witnesses`. Supervised text is rendered only from qualitative units.
Equivalent projected alternatives are described once per answer and get
consistent Case labels. Candidate explanations refer explicitly to those labels,
not to an ambiguously restarted "map 1/map 2".

## Example

Premises: X is east of O; Y is east of O. Query: X relative to Y, Unique.

Natural:

```text
Case 1: One possibility consistent with all premises is the following.
From left to right, the order is O, Y and X.
O, X and Y share the same height. So X is east of Y.

Case 2: One possibility consistent with all premises is the following.
From left to right, the order is O, X and Y.
O, X and Y share the same height. So X is west of Y.

These possibilities give different outcomes, so the answer is not uniquely
determined. Select E.
Answer: E
```

Symbolic:

```text
Case 1 (possible together under all premises):
x(O) < x(Y) < x(X). y(O) = y(X) = y(Y). => E(X,Y).

Case 2 (possible together under all premises):
x(O) < x(X) < x(Y). y(O) = y(X) = y(Y). => W(X,Y).

These possibilities give different outcomes, so the answer is not uniquely
determined. Select E.
Answer: E
```

Only line wrapping differs from the saved preview.

## Checks

- Full evidence is verified before projection.
- The independent text audit parses actual spatial sentences, predicates,
  qualitative orders, membership/count outcomes, and case references.
- Every printed alternative must match a complete audit witness satisfying
  every original premise and the distinct-point rule.
- Printed orders must establish the stated Direction outcome; candidate
  relations must establish the stated qualifying set and joint Count.
- Cross-style evidence must agree. Missing cases, duplicate case definitions,
  wrong case references, and leaked numerical coordinate assignments are rejected.
- Audit tests include matching wrong content introduced into both styles, so
  cross-style agreement alone cannot make a bad explanation pass.
- The existing coordinate-path and answer-integrity checks remain in use.
- The singular/plural Count exclusion sentence is corrected.

607 regression tests passed, including the qualitative audit tests; lint passed.
The eleven-pair preview passed separate printed-text auditing: 28 premise paths,
nine qualitative scenarios per style, and zero numerical-map descriptions.

## Full-Pilot Result

All 282 existing bases and 987 paired views passed the same-input replay,
with no unsupported, verification-failed, or token-rejected cases. Questions,
menus, contracts, answers, semantic reasons, splits, and depths remain unchanged.
Compared with the v2 readable replay, 444 pairs changed and 543 are byte-identical.

The independent printed-text audit checked all 987 pairs: 3,804 premise paths
and 727 qualitative scenarios per style, with zero numerical-map descriptions.
All 810 selected witness occurrences remain in audit metadata. Deduplication
affected 34 views; no duplicate qualitative case definition remains within an
answer. Changing audit-only metadata was also confirmed not to alter token IDs
or assistant loss labels in the existing training tokenizer.

| Target tokens | Natural | Symbolic |
|---|---:|---:|
| Median | 282 | 233 |
| Mean | 289.8 | 244.9 |
| Maximum | 868 | 744 |
| Same depth-ten Direction | 236 | 262 |

The previously longest Selection example now uses eight qualitative cases,
rather than fourteen numerical witness listings: 681 Natural / 696 Symbolic
tokens, down from 1514 / 1182. Both new maximum-length examples are Count
explanations. Their checked pair-exclusion arguments remain longer than a human
might choose to write; this revision removes numerical witness presentation,
not every opportunity to streamline mathematical prose.

The [25 saved pairs](QUALITATIVE-REPLAY-EXAMPLES-v3.md) include the previous
inspection sample and both new maxima. Direct inspection covered all eleven
fixture-preview pairs, the previously longest Selection answer, a depth-nine
Count answer, and both new maxima. All-pair automated verification is distinct
from this bounded manual reading.

- [Measurements](QUALITATIVE-REPLAY-MEASUREMENTS-v3.json)
- [Independent text-audit receipt](QUALITATIVE-TEXT-AUDIT-v3.json)
- Replay artifacts: `../depth-feasibility-qualitative-v3/`

## Scope

This revises how existing verified evidence is communicated. It does not change
spatial semantics, question generation, gold answers, depth definitions, or
evidence selection policies. No distances or unit steps are introduced.
It does not fix the inherited upstream indirect-coincidence certificate limit
or broaden the supported joint Count fragment.

Natural remains deterministic mathematical prose, not unrestricted human
paraphrasing. Long definite proofs can still have mechanical coordinate-chain
wording. Human difficulty and fluency have not been experimentally measured.

Historical v1/v2 artifacts remain unchanged. The revised preview is
[QUALITATIVE-EXPLANATION-PREVIEW-v3.md](QUALITATIVE-EXPLANATION-PREVIEW-v3.md);
its JSON includes the complete audit evidence.
No new stories, production dataset, seal, or training launch is authorized here.
