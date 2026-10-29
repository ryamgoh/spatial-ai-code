# Composition Generator Pilot

Pilot only. Not a training release. Split assignment precedes evidence acceptance.

## Counts

```json
{
  "attempted_stories": 50,
  "accepted_stories": 50,
  "canonical_story_groups": 45,
  "attempted_bases": 150,
  "accepted_bases": 150,
  "views": 678,
  "families": 41,
  "attempt_status": {
    "accepted": 150
  },
  "splits": {
    "train": 411,
    "test": 144,
    "dev": 123
  },
  "answer_frequencies": {
    "C": 46,
    "F": 244,
    "E": 168,
    "A": 48,
    "D": 57,
    "B": 54,
    "BC": 4,
    "AD": 2,
    "AC": 10,
    "AB": 4,
    "ABCD": 31,
    "CD": 4,
    "BD": 6
  },
  "reference_crosschecks": {
    "passed: shared domain types; independent reference search": 16,
    "not-run: limited to Direction with at most six objects": 134
  },
  "coverage_frequencies": {
    "full": 393,
    "partial": 285
  }
}
```

## Coverage

| Query / Contract / Case | Views |
|---|---:|
| count / all-possible / inconsistent | 10 |
| count / all-possible / underdetermined | 20 |
| count / all-possible / unique | 60 |
| count / single / inconsistent | 10 |
| count / single / underdetermined | 20 |
| count / single / unique | 60 |
| count / visible-possible / inconsistent | 10 |
| count / visible-possible / underdetermined | 20 |
| count / visible-possible / unique | 60 |
| direction / all-possible / inconsistent | 10 |
| direction / all-possible / underdetermined | 20 |
| direction / all-possible / unique | 50 |
| direction / single / inconsistent | 10 |
| direction / single / underdetermined | 20 |
| direction / single / unique | 50 |
| direction / visible-possible / inconsistent | 10 |
| direction / visible-possible / underdetermined | 20 |
| direction / visible-possible / unique | 50 |
| which / all-possible / inconsistent | 10 |
| which / all-possible / multiple-certain-match | 10 |
| which / all-possible / no-match | 15 |
| which / all-possible / underdetermined | 13 |
| which / all-possible / unique | 8 |
| which / single / inconsistent | 10 |
| which / single / multiple-certain-match | 10 |
| which / single / no-match | 15 |
| which / single / underdetermined | 13 |
| which / single / unique | 8 |
| which / visible-possible / inconsistent | 10 |
| which / visible-possible / multiple-certain-match | 10 |
| which / visible-possible / no-match | 15 |
| which / visible-possible / underdetermined | 13 |
| which / visible-possible / unique | 8 |

## Split Coverage Gaps

- train: missing none.
- dev: missing none.
- test: missing none.

## Rejections

None in this bounded run. This does not establish support for arbitrary atomic stories.

## Remaining Gates

- Manual audit must be recorded before any release; recipe is not a frozen evidence taxonomy.
- Reference crosschecks cover only Direction with <=6 objects and share domain/sign definitions; Selection/Count and larger cases are not crosschecked.
- Final evidence-category matrix, sampling, tokenizer measurement and runtime decoding are not frozen.
- Templates are bounded constructions, not representative sampling of all atomic maps.
- No training release or automatic dataset expansion is authorized.

Manual audit status: pending. See audit-samples.md for selected paired examples.
