# Plain All-Scene Transition

## Subsequent Presentation Update

Newly rendered scenes now place all premise sentences in one paragraph, with
the final question and A-F options below it. External scenes preserve their
original `is in the map` introduction and complete source-name spelling,
including leading `The`; only whitespace is normalized. All 1,500 original
questions were round-trip checked under all three contracts. The line-layout
sample and v1 diagnostic counts below document the earlier rendering, not the
latest presentation. Old artifacts have not been overwritten.

## Decision

Implemented the approved plain presentation and all-nonreference scene scope as
`plain-all-scene-entailment-v1`. The three entailment answer contracts are
unchanged. This is a verified protocol implementation and bounded diagnostic,
not a replacement dataset release.

User prompts now contain spatial premises, a contract-specific question, and
A-F options. Shared assumptions and answer formatting are in the system message.
There are no visible candidate lists or technical query/interface headers.
External prompts and references can show original source names without an alias
legend; reversible aliases remain in audit metadata.

Selection and Count include every named scene object except the reference,
including intermediate and context objects. Options do not define eligibility.
Old v5 artifacts and their scoped meanings remain untouched.

Validation: all 456 entailment regression tests pass; changed-code Ruff checks
pass. This includes all eight directions across the three query types and
contracts, old/new parser separation, saved evidence replay, original external
name display, system-message scope, and corrupted-label/metadata rejection.
The previously interrupted multi-answer fixture incorrectly relied on shuffled
candidate order; its assertion was corrected without changing generation policy.

## Bounded Evidence

The machine-readable evidence is [ALL-SCENE-TRANSITION-AUDIT-v1.json](ALL-SCENE-TRANSITION-AUDIT-v1.json).
It binds preserved inputs by SHA-256 and records individual results.

Sampling used the first retained train/extrapolation record per query,
requested depth, and source, plus the first eight external bases per query.
This is deliberately small and deterministic, not a prevalence estimate or a
quota feasibility study. Related source questions are not independent stories.

| Check | Result |
|---|---|
| Internal bases reanalyzed | 39: 13 each Direction, Selection, Count |
| Direction entailed sets changed | 0/13 |
| Selection entailed sets changed | 6/13 |
| Count entailed sets changed | 10/13 |
| Old-menu view labels changed | 46 |
| All claims and evidence independently replayed | 39/39 internal; 24/24 external |
| Old measured depth preserved by the current depth checker | 16/39 |
| Sampled token budgets passed | All checked internal and external views |
| Largest internal explanation | 2,306 target tokens |
| External bases checked | 24: eight per query |
| External label changes | 0 |

Eight of the twelve consistent Count bases no longer have a guaranteed exact
total. A thirteenth Count base is inconsistent and returns E by convention.
One retained Count changes from 4 to 5; the old 0-4 construction targets therefore
cannot be reused as all-scene gold values.

Selection entailed-set sizes include 8, 10, and 11. These are not G8-G11 outputs:
there are only four substantive options. Listed selects entailed displayed
objects; Exhaustive returns F when the complete entailed set cannot fit.

The depth checker often lacks an axis exclusion for a newly included contingent
membership. This does not invalidate the solver label or negative counterexample;
it means the old controlled depth admission has not been established for that
expanded query. In a depth-9 Selection example, the complete set retains
axis-support depth 9, but Unique's cheapest two-membership certificate has depth
2. Full-set depth and contract-decision depth must remain distinct.

Six existing development-source questions also passed the actual original
context-removal checks. Retained v5 records do not preserve those premise indices,
so that result does not establish context irrelevance for all 39 retained bases.
A regression fixture explicitly rejects a formerly irrelevant scoped context
statement that changes all-scene answers.

## What A Prompt Looks Like

The following all-scene example is covered by the protocol regression tests.
Common assumptions are supplied by the system message, not repeated here.

```text
Consider a map with multiple objects:
Clinic is to the North of Park.
Library is to the North of Clinic.
Museum is to the North of Clinic.
Bank is to the South of Park.
Cafe is to the South of Park.
Hotel is to the South of Park.
Which listed objects are guaranteed to be north of Park? Select all that apply, or F if none are guaranteed.
A. Library
B. Museum
C. Bank
D. Cafe
E. Cannot be determined
F. None of the above
```

The full entailed set is **Clinic, Library, Museum**. Clinic remains an answer
even though it supports the other two premises and is absent from the menu.

- Unique: E, because more than one scene object is entailed.
- Listed: AB, because both displayed Library and Museum are entailed.
- Exhaustive: F, because Clinic is missing from the options.
- Count of north-of-Park objects: exactly 3, not the previous scoped count of 2.

Each contract uses its own final question. Unique asks for one guaranteed object;
Exhaustive asks for the complete set and explicitly handles missing options.

## Before Replacement Generation

The frozen scoped production configuration remains unchanged. New all-scene
production planning is explicitly blocked until construction and admission are
reviewed for the new scope.

The next engineering work is to make controlled scenes satisfy their intended
all-scene Count targets, retain suitable irrelevant context, and assign depth
according to the answer obligation of each contract. Then test the fixed reserve
against the existing approved quotas. Do not solve those problems by restoring
hidden eligible subsets or silently changing labels.

No 25% multi-answer allocation was applied. No replacement dataset, seal, or
training run was started. Full quota feasibility, full manual prose review, and
full external regeneration remain outside this bounded check.
