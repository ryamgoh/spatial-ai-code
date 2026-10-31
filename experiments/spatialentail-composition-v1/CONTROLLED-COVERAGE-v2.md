# Controlled Coverage Check

Development verification only. No training or evaluation replacement.

- Fixed attempted questions: 968; statuses: {'accepted': 968}.
- Count depth/direction/value cells passed: 400/400.
- Original artifact hashes unchanged: True.
- Maximum reference tokens: {'natural': 2527, 'symbolic': 2796}.
- Observed (objects, premises): [(6, 6), (9, 9), (12, 12), (15, 18), (16, 19), (17, 20), (19, 22), (20, 23), (21, 24), (23, 26)].
- All admitted examples have independent spatial evidence replay and matched style references.
- Each question retains one selected menu. F is a sole unlisted entailment, not an E case.
- Positive requested depths are measured, not inferred from sentence count.
- No-entailed and inconsistent E rows receive no positive-depth label.

## Coverage

| Query | Case | Status | Questions |
|---|---|---|---|
| count | inconsistent | accepted | 36 |
| count | listed | accepted | 216 |
| count | no-entailed | accepted | 36 |
| count | unlisted | accepted | 216 |
| direction | inconsistent | accepted | 36 |
| direction | listed | accepted | 56 |
| direction | no-entailed | accepted | 36 |
| direction | unlisted | accepted | 56 |
| which | inconsistent | accepted | 36 |
| which | listed | accepted | 56 |
| which | multiple | accepted | 96 |
| which | no-entailed | accepted | 36 |
| which | unlisted | accepted | 56 |

## Scope and Limits

- Compact scenes have 6 or 9 premises across all applicable answer cases and all eight directions.
- Deep positive coverage crosses L1-L10 and eight directions; Count also crosses values 0-4.
- Each deep positive cell uses one of A-D or F. Both menu roles are separately checked in compact scenes; this is not both roles at every deep cell.
- Multiple-Which cardinalities cycle through 2-4; not every cardinality is crossed with every depth.
- E coverage adds disconnected-component ambiguity and variable strict cycles alongside the original fork and three-edge cycle. It is not exhaustive semantic coverage.
- Deep E examples are checked at every construction span with two direction/recipe variants, not every depth-direction-case product.
- Size matching is a construction check, not proof of absence of all structural shortcuts. No classifier is fitted in this coverage run.
- Seeds are variant groups, not verified canonical families. No canonical deduplication, evaluation-overlap screening, or final 6,000-question selection has been performed.
- The old source quotas and dataset files are untouched. These development examples must not enter training.
- Existing explanation prose is reused; the prior prose-cleanup task is not claimed complete.

Full receipt: `CONTROLLED-COVERAGE-v2.json`

Evidence spool: `/var/folders/g4/ccx77z2d60sgsjrt9ck_ppwr0000gn/T/spatial-controlled-coverage-53x_vbxx/records.jsonl`
