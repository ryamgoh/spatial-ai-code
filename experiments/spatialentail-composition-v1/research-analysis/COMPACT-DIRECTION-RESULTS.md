# Compact Direction: Same-Input Replay

## Verdict

The repeated Direction exclusions were unnecessary. The new renderer resolves
all 35 token rejections in this finite pilot without increasing the token caps.
This does **not** establish that the dataset is ready for production.
The structural-diversity and semantic-role/gold-coverage concerns in the
[first verdict](DEPTH-FEASIBILITY-VERDICT.md) still apply.

No new problems were sampled. No production set, benchmark release, seal,
training run or commit was created. The existing v2 seal and both archived
package hashes still verify.

## What Changed

The separately versioned `composition-entailed-direction-reuse-v1` renderer
keeps the full positive Direction derivation once. It partitions the other
seven exact directions into groups contradicted by the proved x or y comparison.
The result is a complete singleton possibility set, then the ordinary
contract/menu mapping.

This is an entailment-specific optimization, not general proof deduplication.
Ambiguous Direction, contradictions, Selection, Count, and Exhaustive F's
unlisted witness retain their previous rendering. Original renderer source
and frozen packages remain untouched. Future generation must explicitly use
the new renderer; the old generator's default was not silently changed.

## Measured Results

Same pinned Qwen tokenizer and 4,096-token response / 8,192-token context limits:

| Metric | Original | Compact |
| --- | ---: | ---: |
| Attempted query bases | 282 | 282 |
| Accepted bases | 247 | 282 |
| Token-rejected bases | 35 | 0 |
| Accepted views per style | 861 | 987 |
| Longest Natural target text | 9,062 | 2,589 |
| Longest Symbolic target text | 7,920 | 2,281 |

The maximum supervised response including the ending is now 2,591 tokens.
Maximum full training sequence is 3,246 tokens; maximum reserved two-stage
context is 4,868 tokens. Reference lengths are not measurements of future
model-generated lengths or GPU feasibility.

There are 259 changed reference pairs and 728 byte-identical pairs. All 282
serialized problems, saved depth measurements, and every row's non-reference
fields are unchanged, including IDs, roles, prompts, menus and gold answers.
All six Direction bases at each depth 1-10 now pass token admission.

| Depth | Maximum Natural Direction Target | Maximum Symbolic Direction Target |
| --- | ---: | ---: |
| 1 | 139 | 127 |
| 5 | 981 | 857 |
| 6 | 1,062 | 928 |
| 8 | 1,334 | 1,164 |
| 9 | 1,631 | 1,419 |
| 10 | 1,674 | 1,460 |

Accepted views by provisional role: train 330, dev 246, in-range test 279,
extrapolation 132. These are still pilot allocations, not final quotas.
E/F now account for 325/987 views (32.9%). No gold changed: the accepted
mixture changed because the previously rejected Direction bases were restored.
This percentage alone does not validate the mixture.

## Inspected Depth-Ten Trace

Both complete new traces were inspected for
`depth-10-sample-4-direction-m0-all-possible`, the previously longest Natural
example. The question asks where Object16 is relative to Object14.
Its Natural target fell from 9,062 to **1,671 tokens**; Symbolic is 1,457.
Including the supervised ending, these are 1,673 and 1,459 respectively.

The full ten-link x and y derivations remain. The end of the Symbolic trace is:

```text
D-X-9: [D-X-8, P11-X1] => x(Object16) = x(Object14)
...
D-Y-9: [D-Y-8, P4-Y1] => y(Object14) < y(Object16)
Q-DIR: [D-X-9, D-Y-9] => North(Object16,Object14)
The proved x comparison excludes Northeast, East, Southeast, Southwest, West, Northwest.
The proved y comparison excludes South.
The complete possible set is {North}. It is displayed at A.
Answer: A
```

The ellipsis above abbreviates this report excerpt only; no proof steps were
truncated from the actual saved training reference. Each long axis derivation
occurs once. The x equality excludes every nonzero-x direction, and the y
inequality excludes South, establishing completeness over the eight directions.
Natural renders the same checked proof steps in coordinate prose and uses the
same exclusion conclusions.

## Verification

- 46 new tests passed: all eight directions, all contracts, present/omitted
  menus, existing fixtures, independent sign-table exclusion checks,
  one copy of each long derivation, evidence immutability, corrupted semantics
  rejection, source-record immutability and whole-pair token rejection.
- 361 existing composition, depth, difficulty and training tests passed.
- Ruff checks passed for the new files.
- Full evidence for eligible unique Direction cases was rebuilt and verified
  before rendering; possible/certain sets and gold mappings were compared.
- Other evidence was retained from the original verified pilot rather than
  regenerated. Full certificates, including explicit exclusions, remain in
  audit records.
- Every resulting Natural/Symbolic pair passed factual alignment and answer
  parsing; all styles were remeasured with matching tokenizer-file hashes.
- Original pilot and replay-source hashes were checked before and after
  replay. Existing package archives were separately checked.

These checks are not a new complete release audit or an independent human
review of all 987 pairs.

## Artifacts and Reproduction

The pre-run [protocol](COMPACT-DIRECTION-PROTOCOL.md) describes the bounded
same-input experiment. Full records and measurements are in
[`depth-feasibility-compact-v1`](../depth-feasibility-compact-v1/).

Run `python -m spatial.v2.composition_rerender --source
experiments/spatialentail-composition-v1/depth-feasibility-v1 --output
<new-directory> --tokenizer <pinned-Qwen-tokenizer-directory>` with the
versions recorded in the manifest. Output overwrite is refused.

SHA-256 at completion:

```text
manifest.json  65aa24dffdff51dd40b7248d82539544600193e7b4220255ee363afbdc440867
attempts.jsonl 762051e617559c6676cc8ea6d1bcaae0aa6d6405f0d237f68bca3a2c092c4c34
summary.json  5b08b9f6ff8ab93bcb1ac2b5301d06975fa64b5a1ef50415bb380d35ccf489f4
```

Next, address structural diversity and feasible gold coverage in a separately
bounded generation pilot. Present its observed mixture and proposed final
quotas before production generation or sealing.
