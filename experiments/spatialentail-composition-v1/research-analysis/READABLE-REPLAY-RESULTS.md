# Readable Replay Results

## Verdict

The readable renderer now covers every query/contract view in the existing
282-base pilot. All 987 paired views passed evidence, answer, token-admission,
and protected-input checks. No new stories were sampled.

This is **pilot coverage achieved, not production sign-off**. Inspection found
remaining repeated witness maps in contingent Selection and minor prose
roughness. The inherited certificate fragment also remains narrower than the
solver's complete atomic spatial semantics. Resolve or explicitly scope these
limitations before production generation; do not seal or train yet.

## What Changed

Added ambiguous Direction explanations for Listed and Exhaustive contracts:
whole-map witnesses for included values, and premise-supported coordinate
comparisons excluding the required remaining values. One coordinate derivation
is reused when it excludes multiple directions.

Temporary-assumption conflict rendering is available where a direct opposing
axis path is unavailable. It is tested for indirect forced coincidence, but
the upstream shared certificate builder can still reject such inputs before
the renderer receives them; see the boundary below.

Selection, fixed/variable/joint Count, no-match and multiple-certain-match
Selection, zero Count, both contradiction types, and E/F escape reasons were
exercised across the existing pilot and fixture contracts.

## Preservation

- 282/282 bases accepted; zero unsupported, verification-failed, or token-rejected bases.
- 987 views per style: 333 Direction, 315 Selection, 339 Count.
- All 987 reference pairs changed from the previous compact/formal renderer.
- All questions, menus, gold answers, semantic/escape reasons, splits, depth
  measurements, graph metadata, and original audit certificates stayed identical.
- Rebuilt evidence is stored separately under `readable_evidence`.
- Source-pilot, source-code, and output hashes were verified.
- The original eleven-pair preview was not rewritten.
- Both historical training archives still match their receipt SHA-256 values.
- No training launch, seal, production generation, commit, or push occurred.

The output is `../depth-feasibility-readable-v2/`, with `manifest.json`,
`attempts.jsonl`, and `summary.json`. This directory is a feasibility artifact,
not a training release.

## Token Measurements

Pinned Qwen3.5-4B tokenizer revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; unchanged 4096-token response allowance
and 8192-token context. These are target-token lengths including the answer,
not training speed measurements. Each of the 987 contract/menu views is weighted
once; these are not 987 independent stories.

| Style | Before mean | After mean | Before median | After median | After p95 | After max |
|---|---:|---:|---:|---:|---:|---:|
| Natural | 950.0 | 330.2 | 861 | 294 | 687 | 1514 |
| Symbolic | 833.0 | 267.7 | 767 | 239 | 552 | 1182 |

Before means the compact Direction pilot, which still used formal transcripts
elsewhere. Total target tokens fell from 937,689 to 325,947 for Natural, and
from 822,201 to 264,230 for Symbolic.

| Query family | Natural max | Symbolic max |
|---|---:|---:|
| Direction | 405 | 299 |
| Selection | 1514 | 1182 |
| Count | 916 | 734 |

The same saved depth-ten Direction remains **236 Natural / 262 Symbolic**.
Across all 66 depth-nine views, the respective maxima are 761 / 675; across
all 66 depth-ten views, 781 / 627. These per-style maxima need not occur on the
same row. Depth is supported premise-path length, not paragraph count or a
measurement of human solving difficulty.

Symbolic is shorter on 897 pairs, Natural on 86, with four ties. Shortness is
therefore an observed tendency in this pilot, not a guarantee of notation.

## Verification

- 532 composition/contract/depth/evidence/replay regression tests passed.
- 69 separate printed-text audit tests passed, including matching errors
  deliberately introduced into both styles.
- Lint passed for the new and modified implementation/test files.
- The independent printed-text audit passed all 987 pairs: **3,804 paths and
  810 complete witness maps per style**.
- The auditor parses printed coordinate links and citations, checks derived
  endpoints against chain order, checks every printed map's premises and
  distinctness, evaluates its Direction/Selection/joint Count outcome, and
  compares the printed evidence across styles.

The text auditor does not import the renderer or its direction-sign helper.
It is still not a theorem prover for all conclusion prose, nor a fluency
evaluation. Full certificate verification, template integrity, and targeted
wording tests remain complementary checks.

## Inspection Findings

Inspected 23 distinct saved pairs, including both global output maxima, long
depth-nine/ten examples by family, ambiguous Direction under all contracts,
joint Count, zero Count, Selection semantic cases, contradictions, and escapes.
The full paired texts are in [the examples](READABLE-REPLAY-EXAMPLES.md).

1. **Repeated witnesses remain in 34 views.** The longest example,
   `random-atoms-4-which-m0-single`, prints 14 witness occurrences but only eight
   distinct coordinate maps. It repeats a common counterexample while establishing
   contingency for different candidates. This is valid evidence, but inefficient
   prose. Reuse named example maps in both styles and make candidate-to-map
   references explicit; preserve all selected evidence.
2. **Long mixed-axis paths are still mechanical.** The 405-token depth-ten
   Direction example preserves each comparison, but its semicolon-heavy Natural
   paragraphs are less fluent than the regular northward chain. Grouping adjacent
   equalities could improve flow without deleting meaningful links.
3. **Minor grammar remains.** The joint Count example
   `random-world-8-count-m1-visible-possible` says "Counts 7 are therefore
   impossible." It should use singular "Count 7 is therefore impossible."
4. **No incorrect displayed coordinate link, witness outcome, or gold answer
   was found in these checks.** That does not establish arbitrary-input coverage
   or human-level naturalness.

## Inherited Boundary

The evidence builder still rejects some consistent atomic inputs whose
impossible direction can only be shown by forcing other named objects to
coincide. For example:

```text
X is North of O.
Y is North of O.
X is East of T.
Query: Y relative to T.
```

Y cannot be exactly East of T: that would give X and Y equal x and y coordinates.
The readable conflict helper can express and verify this argument, but
`composition_evidence.build_evidence` currently fails upstream with
`UnsupportedEvidence: ... missing a model for non-refutable candidate East`.
This is a rejection, not a wrong label. It did not occur in the 282 saved bases.
Extending this shared certificate support is a separate correctness/coverage
task before claiming general atomic-input support.

Count evidence remains limited to at most eight eligible candidates and
exclusions covered by fixed memberships and pair incompatibilities. The pilot
uses at most seven. Higher-order correlations outside this fragment are not
silently accepted.

## Before Production

First address the witness reuse/prose findings and the certificate boundary
(or explicitly approve that boundary as an exclusion). Then specify and approve
structural diversity, depth, and feasible gold quotas before production sampling.
Explanation improvements do not repair sampling balance: the existing pilot
still has 325/987 E/F views, including 156/329 Unique views. Those counts are
unchanged by this replay.

Supporting files:

- [Protocol](READABLE-REPLAY-PROTOCOL.md)
- [Measurements and inspected IDs](READABLE-REPLAY-MEASUREMENTS.json)
- [Printed-text audit receipt](READABLE-REPLAY-TEXT-AUDIT.json)
- [Paired examples](READABLE-REPLAY-EXAMPLES.md)
