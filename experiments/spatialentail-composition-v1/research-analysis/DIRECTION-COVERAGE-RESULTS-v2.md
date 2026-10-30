# All-Direction Selection Coverage Results

## Verdict

The targeted direction gap is closed for the three controlled unique-match
Selection patterns. All 240 fixed cases pass full admission, with substantive
printed-depth support under Unique, Listed and Exhaustive. This is a development
coverage check, not a production dataset or a new balancing result.

| Query direction | Accepted bases | Patterns | Depths |
|---|---:|---|---|
| North | 30 | All three | 1-10 |
| Northeast | 30 | All three | 1-10 |
| East | 30 | All three | 1-10 |
| Southeast | 30 | All three | 1-10 |
| South | 30 | All three | 1-10 |
| Southwest | 30 | All three | 1-10 |
| West | 30 | All three | 1-10 |
| Northwest | 30 | All three | 1-10 |

There is one case per pattern/direction/depth intersection, not multiple
independent families per intersection.

## Implementation

The generator accepts an explicit direction. Diagonal cases start with actual
Northeast constraints on both axes; only quarter-turn rotations are applied.
No 45-degree relabeling of cardinal constraints is used.

The original default generator and 60-story schedule are unchanged, verified
against a pre-edit SHA-256 fingerprint. All directions retain the existing depth
roles: deep match proof, deep single-rival exclusion, or deep shared exclusions.

## Menu Correction

Version 1 admitted 214 of 240 cases under the full coverage requirement.
All 240 passed spatial, evidence, explanation and token admission, but 26
rival-chain cases lacked the requested printed depth in their substantive
Listed view: the deep rival was absent from that menu. The shorter Listed
explanation was correct and must not be counted at the longer depth.

Version 2 preserves version 1 and rechecks those 26 cases using the
[depth-visible menu rule](DIRECTION-COVERAGE-PROTOCOL-v2.md). It includes the
deep rival in the substantive menu, replacing a shallow nonmatching option.

Independent comparison confirmed:

- All 240 scenes, semantic results, depth measurements and gold answers match
  version 1 exactly.
- The 214 previously accepted JSON records are identical.
- Exactly 26 substantive menus changed.
- Match-omitting views remain unchanged, including their explanations.
- Every case now has a substantive requested-depth view under each contract.
- Final source hashes and both runs' output hashes match their manifests.

This is not a weaker depth criterion or replacement sampling.

## Views and Lengths

Each case has one match-including and one match-omitting menu, each with all
three contracts: 480 bundles and 1,440 paired views. The deliberate menu check
therefore includes equal numbers of substantive and F views. That 50% F share
is not a proposed production distribution.

Pinned Qwen3.5-4B target tokens:

| Style | Mean | Maximum |
|---|---:|---:|
| Natural | 249.5 | 427 |
| Symbolic | 238.5 | 444 |

All views pass the existing context and response caps.

## Evidence

Final composition and pivot suite: **1,086 passed, 1 skipped**. Ruff passed for
the changed Python files. The regression includes the unchanged historical
schedule fingerprint and the actual previously failing Southwest Listed-menu
case before and after correction.

- [Six saved Northeast pairs](../direction-coverage-v2/examples.md), covering
  all three patterns at depths 3 and 10.
- [Final manifest](../direction-coverage-v2/manifest.json) and
  [complete coverage matrix and summary](../direction-coverage-v2/summary.json).
- [Preserved first-run summary](../direction-coverage-v1/summary.json).
- [Original fixed-check protocol](DIRECTION-COVERAGE-PROTOCOL-v1.md).

The coding assistant inspected all six saved pairs, plus freshly rendered
Southwest, Southeast and Northwest shared-exclusion examples. This is not an
independent human fluency evaluation.

## Limits

The 240 cases are variants of three construction families. Rotations and depth
variants are deliberately kept in the same families; they are not independent
training or benchmark families. Depths 9-10 here are development checks, not a
held-out benchmark. No random graphs, extra distractor policy, ambiguous cases,
Direction/Count direction quotas or production mixture were added by this work.

Nothing was committed, sealed, or trained as part of this check.
