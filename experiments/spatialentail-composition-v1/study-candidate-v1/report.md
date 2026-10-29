# Composition Generator Pilot

Pilot only. Not a training release. Split assignment precedes evidence acceptance.

## Counts

```json
{
  "attempted_stories": 200,
  "accepted_stories": 200,
  "canonical_story_groups": 192,
  "attempted_bases": 600,
  "accepted_bases": 600,
  "views": 2652,
  "families": 188,
  "attempt_status": {
    "accepted": 600
  },
  "splits": {
    "train": 1584,
    "test": 555,
    "dev": 513
  },
  "answer_frequencies": {
    "E": 758,
    "F": 780,
    "ABCD": 268,
    "BCD": 65,
    "A": 104,
    "B": 159,
    "AC": 30,
    "C": 137,
    "CD": 45,
    "ABD": 34,
    "BD": 33,
    "D": 118,
    "BC": 37,
    "AB": 24,
    "ABC": 38,
    "ACD": 14,
    "AD": 8
  },
  "reference_crosschecks": {
    "not-run: limited to Direction with at most six objects": 498,
    "passed: shared domain types; independent reference search": 102
  },
  "coverage_frequencies": {
    "partial": 1368,
    "full": 1284
  }
}
```

## Coverage

| Query / Contract / Case | Views |
|---|---:|
| count / all-possible / inconsistent | 23 |
| count / all-possible / underdetermined | 213 |
| count / all-possible / unique | 94 |
| count / single / inconsistent | 23 |
| count / single / underdetermined | 213 |
| count / single / unique | 94 |
| count / visible-possible / inconsistent | 23 |
| count / visible-possible / underdetermined | 213 |
| count / visible-possible / unique | 94 |
| direction / all-possible / inconsistent | 23 |
| direction / all-possible / underdetermined | 124 |
| direction / all-possible / unique | 152 |
| direction / single / inconsistent | 23 |
| direction / single / underdetermined | 124 |
| direction / single / unique | 152 |
| direction / visible-possible / inconsistent | 23 |
| direction / visible-possible / underdetermined | 124 |
| direction / visible-possible / unique | 152 |
| which / all-possible / inconsistent | 23 |
| which / all-possible / multiple-certain-match | 16 |
| which / all-possible / no-match | 25 |
| which / all-possible / underdetermined | 173 |
| which / all-possible / unique | 18 |
| which / single / inconsistent | 23 |
| which / single / multiple-certain-match | 16 |
| which / single / no-match | 25 |
| which / single / underdetermined | 173 |
| which / single / unique | 18 |
| which / visible-possible / inconsistent | 23 |
| which / visible-possible / multiple-certain-match | 16 |
| which / visible-possible / no-match | 25 |
| which / visible-possible / underdetermined | 173 |
| which / visible-possible / unique | 18 |

## Split Coverage Gaps

- train: missing none.
- dev: missing which / all-possible / unique, which / single / unique, which / visible-possible / unique.
- test: missing which / all-possible / unique, which / single / unique, which / visible-possible / unique.

## Rejections

None in this bounded run. This does not establish support for arbitrary atomic stories.

## Remaining Gates

- Manual audit must be recorded before any release; recipe is not a frozen evidence taxonomy.
- Reference crosschecks cover only Direction with <=6 objects and share domain/sign definitions; Selection/Count and larger cases are not crosschecked.
- Final evidence-category matrix, sampling, tokenizer measurement and runtime decoding are not frozen.
- Templates are bounded constructions, not representative sampling of all atomic maps.
- No training release or automatic dataset expansion is authorized.

Manual audit status: pending. See audit-samples.md for selected paired examples.
