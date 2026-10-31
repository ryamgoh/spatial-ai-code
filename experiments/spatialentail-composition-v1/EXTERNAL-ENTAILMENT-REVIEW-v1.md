# External Entailment Candidate Review

## Artifact

Candidate: `spatialmap-entailment-all-five-v1/`.
It contains 1,500 source questions from 500 stories, with 5,070 contract/menu
views per style. All views have reference explanations. Selection uses the
user-approved five nonreference objects; original names are retained in the
alias legend. The historical Corr SINGLE files are unchanged.

This is an unsealed candidate, not a model-evaluation result.

## Inspected Examples

| View ID | Finding |
|---|---|
| `spatialmap.tqa.2000.0-m0-unique-entailed` | The cited premise establishes both strict axis comparisons for Northeast. Answer A is consistent with the displayed menu. |
| `spatialmap.tqa.2003.0-m0-unique-entailed` | Checked North/Northeast arrangements falsify different exact-direction claims. No individual direction is entailed; Unique E is justified. |
| `spatialmap.tqa.2001.1-m1-exhaustive-entailed` | The unlisted candidate is proved Southwest by inversion of the cited Northeast premise. Exhaustive F follows from missing entailed coverage, not mere possibility. |
| `spatialmap.tqa.2000.2-m0-unique-entailed` | Every eligible object is above the reference, excluding Southeast membership. Exact Count zero is entailed and maps to D, not an empty-set escape. |
| `spatialmap.tqa.2001.2-m0-unique-entailed` | Valid joint arrangements give counts one and zero. Counterexamples establish that no exact count is entailed; Unique E is appropriate. |

Both reference styles were inspected for these examples. The selected
qualitative relationships and conclusions agree; no numeric witness coordinate
assignments are printed in the explanations. No semantic or gold error was
found in these samples. This is a representative inspection, not a manual
review of all 5,070 views.

## Wording Findings

- Count claims currently use plural wording for one candidate:
  "exactly 1 eligible candidates qualify." Change this to
  "exactly 1 eligible candidate qualifies" before sealing.
- For an already proved unique Direction or invariant Count, repeated
  counterexample prose for mutually exclusive alternative values is redundant.
  A matched, checked domain-exclusion statement could express the same
  obligation more naturally. Keep full verification evidence in audit metadata.

These findings do not change the inspected labels. They remain reference
quality findings to address before a final release. Any wording revision must
be versioned, applied consistently to both styles, and rechecked against the
pinned tokenizer and original evidence. Do not silently mix revised text into
an existing frozen generation run.

## Remaining Gates

The full synthetic candidate is being generated separately. It still needs
retained split/source/depth/family coverage verification, distribution review,
and paired reference inspection. Neither this external review nor the saved
configuration authorizes a dataset seal or model training.
