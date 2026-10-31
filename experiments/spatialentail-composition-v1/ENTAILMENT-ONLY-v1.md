# SpatialEntail Entailment-Only Amendment

## Active Unique-Only Scope

The user has approved dropping Listed and Exhaustive from the active study.
[UNIQUE-ONLY-v1.md](UNIQUE-ONLY-v1.md) now governs contract scope, coverage
selection, and training/evaluation. Direction, Which, and Count all use
`unique-entailed`; the spatial semantics, all-scene presentation, evidence,
base targets, and depth/family safeguards below remain applicable.
Mixed-contract comparisons, Listed/Exhaustive requirements, and the exact
900 multi-Selection diagnostic target below are historical, not active gates.
The preserved 10,800-base projection passed its Unique-only checks, yielding
12,578 examples per style, including 6,942 training examples. That is a historical
selection, not the latest training target. The approved 2026-10-31 goal is
6,000 final examples per style from distinct questions, one selected menu each:
1,200 A-D / 600 E / 200 F per query type, with explicit E-subtype targets and
uniqueness proof obligations in
[UNIQUE-ONLY-v1.md](UNIQUE-ONLY-v1.md#approved-training-allocation-goal).
The [full-pool bounds check](UNIQUE-TRAINING-ALLOCATION-FEASIBILITY-v1.md) proves
source-supply and inherited primary-A-D depth-rule conflicts. The user subsequently
approved [one bounded repair](UNIQUE-TRAINING-REPAIR-PROPOSAL-v1.md): selected
explanation depth (including F) at the same minimum and up to 2,048 supplementary
query slots, without automatic refill. Evaluation sets stay unchanged.
Nothing is sealed or launched for training.

Current outcome: a separately approved second 1,600-slot batch cleared the
remaining supply bounds. Evidence replay, exact one-menu selection and export
audit have now passed for the
[6,000-question training candidate](unique-training-candidate-v1/README.md).
The requested answer/subtype targets and retained source/depth/family safeguards
are achieved, with unchanged evaluation selections. Historical counts and
pending migration statements below are provenance, not the current export status.
Manual prose review and separate release approval remain.

## Authority and Status

This amendment implements the user's approved research direction: select
individually entailed answers, not answers that are merely possible. It
supersedes the possibility-based contracts and evidence requirements in the
historical portion of `pivot_20261026.md`.

Interface version: `spatialentail-entailment-v1`.
Contract identifiers: `unique-entailed`, `listed-entailed`,
`exhaustive-entailed`. Old `single`, `visible-possible`, and `all-possible`
artifacts retain their original meanings. Do not rename or resume them as
entailment-only data.

The scoped v5 unsealed candidate and its full audits are available. See
[ENTAILMENT-FINAL-REVIEW-v5.md](ENTAILMENT-FINAL-REVIEW-v5.md) for the
current evidence and targeted review. This document is not a dataset seal or
authorization for model training.

## Approved Plain All-Scene Protocol

The user approved this scope and presentation amendment after reviewing the
visible candidate lists. New protocol: `plain-all-scene-entailment-v1`;
scope policy: `all-nonreference-scene-objects-v1`. The entailment interface and
three answer contracts remain unchanged.

Visible user prompts contain plain atomic spatial sentences, followed by the
contract-specific question and A-F options. They contain no `Eligible:`,
`Objects:`, `Query:`, `Interface:`, or `Contract:` headers. Shared exact-direction,
distinct-point, open-world assumptions, all-scene scope, inconsistency convention,
and answer formatting belong in the system message. Every scene object must
appear in at least one premise. Internal prompts use ordinary place names;
external prompts use original source names without a visible alias legend.
Reversible external aliases remain audit-only.

The approved layout is the opening line, one paragraph of scene sentences, a
blank line, then the contract-specific question and separate A-F option lines.
External adaptation preserves the original introductory
`<Object> is in the map.` sentence, all spatial sentences, their order, and
original object spelling (including leading `The`). Only whitespace is
normalized to form the paragraph. The introductory declaration adds no direction
constraint or coordinate origin. Synthetic scenes use their existing spatial
sentences in the same paragraph layout, without inventing an external source
declaration. Previously saved line-layout prompts remain parseable.

Selection and Count range over **every named scene object except the reference**,
independently of the displayed menu. Supporting and context objects are not
excluded. Direction still ranges over all eight exact directions.

This is a semantic change, not just a shorter template. Recompute full domains,
entailments, counterexamples, menus, labels, explanations, depth descriptors,
context irrelevance, and token budgets. Selection can entail more than four
objects; Exhaustive then returns F for incomplete menu coverage. Count ranges
from zero to the number of all nonreference objects, not just the previous
candidate subset. Count is still a joint invariant total, not the number of
entailed memberships.

Preserve v5 and earlier candidates with their original scoped semantics. Their
acceptance does not establish all-scene quota feasibility. The old controlled
Count 0-4 allocation and substantive-depth admission need reassessment before a
replacement run. The proposed 25% multi-answer allocation has not been approved
or applied. Bounded verification comes before replacement generation; neither
sealing nor training is authorized by this amendment.

The bounded transition results and a plain prompt example are recorded in
[ALL-SCENE-TRANSITION-REVIEW-v1.md](ALL-SCENE-TRANSITION-REVIEW-v1.md).
All-scene source adaptation is implemented, but production planning remains
blocked pending construction and depth-admission review.

The proposed gold-diversity check is documented separately in
[ALL-SCENE-BALANCE-FEASIBILITY-v1.md](ALL-SCENE-BALANCE-FEASIBILITY-v1.md).
Its development-only constructions are not wired into the production schedule.
Constructive depth/cardinality support does not approve a 25% allocation or
establish fixed-reserve, source-mixture, split-isolation, or family-floor supply.

Fixed-slot all-scene recipe repair is a separate diagnostic, documented in
[REPAIRED-FIXED-RESERVE-v2.md](REPAIRED-FIXED-RESERVE-v2.md).
It preserves existing seeds and source lineage rather than refilling the
reserve. Renaming, premise ordering, and option positioning do not create new
independent base/family supply. The old requirement for all extrapolation
Unique Selection answers to have substantive A-D output conflicts with a
multi-entailment allocation there: those Unique answers correctly return E.
The user approved including deep E decisions in all four splits, extending the
initial depth-9/10 approval to train, dev, and in-range test.
For the new all-scene policy, the Unique Selection primary-depth minima
(1,400 train, 280 dev, 560 in-range, and 400 extrapolation bases) use
requested-depth **decision support**, not substantive A-D output.
Singleton answers retain substantive support; a
multiple-entailment E answer must have its cheapest two-member certificate
at the requested depth. All numeric base/source/Count and family minima stay
unchanged. This revision is implemented in a diagnostic policy copy; the old
frozen config and scoped artifacts have not been altered. It does not approve
a final 25% allocation, sealing, or training.

The combined diagnostic is documented in
[COMBINED-ALL-SCENE-FEASIBILITY-v1.md](COMBINED-ALL-SCENE-FEASIBILITY-v1.md).
It tests one exact overall target of 900 multi-entailment Selection bases out
of 3,600 Selection bases, including E/F cases in the denominator. It does not
introduce per-split/depth multi-answer quotas or approve a G2:G3:G4 allocation.
Mandatory noncontrolled source supply is checked before retention; a source
shortage cannot be filled with controlled bases or renamed views. This is
diagnostic authorization only, not a production freeze.

The first completed source audit passes every raw source quota, but the old
train/dev/in-range Unique substantive-depth minima impose an optimistic
overall multi-Selection capacity bound of 598, below the proposed 900.
This disproved the combined target under the previous requirements without
requiring full retention MILP. The user has now approved replacing that
A-D-only condition with decision-depth support for the in-range minima too.
The same numeric minima remain. The revised combined check is recorded in
[COMBINED-ALL-SCENE-FEASIBILITY-v2.md](COMBINED-ALL-SCENE-FEASIBILITY-v2.md).
The prior controlled Count supply repair remains valid.

The user subsequently requested a fresh revised run with no saved-result
reuse. The v2 continuation was stopped. The fresh v3 check uses the same
latest all-scene recipe and approved decision-depth rules, redoing both source
and controlled admission from zero. See
[COMBINED-ALL-SCENE-FEASIBILITY-v3.md](COMBINED-ALL-SCENE-FEASIBILITY-v3.md).
This does not add seeds, change numeric requirements, or authorize a dataset
seal or training.

The fresh v3 audit completed: all 21,758 controlled queries passed; 25,344
distinct verified candidates remain overall. One necessary supply requirement
fails: dev depth-2 Count total 4 needs 7 distinct bases and has 6. No other
individual quota or coverage/family supply shortage was reported. The retention
MILP was therefore not run in v3. That receipt remains an unchanged record of
the failed necessary supply check.

The user authorized fixing this shortage. The cause was shallow canonical-scene
collisions between fixed dev Count and training multi-Selection slots.
The v4 recipe varies only depth-two context, uniformly across splits, without
changing seeds, numeric requirements, core support, or family identities.
All 2,061 affected eligible queries passed fresh admission and independent
replay; other depths remained exactly unchanged. Combined retention then
passed: 10,800 bases at the approved split sizes, exactly 900 multi-entailment
Selection bases, and every quota and coverage/family requirement satisfied
together. The missing dev group selected its required seven bases from 20
verified candidates. See
[COMBINED-ALL-SCENE-FEASIBILITY-v4.md](COMBINED-ALL-SCENE-FEASIBILITY-v4.md).
This clears diagnostic feasibility, not final production approval. Selected IDs
are recorded; no replacement training package was exported, sealed, or trained.

## Research Question

Build a solver-verified atomic spatial entailment dataset with controlled
composition depth, matched Natural/Symbolic explanations, and the Unique
entailed-answer contract across Direction, Which, and Count.
Evaluate whether the model distinguishes entailed from
non-entailed claims. Compare answer accuracy and generated-token cost under
matched Natural and Symbolic supervision without assuming either style wins.

Retain the exact eight-direction, distinct-point, open-world spatial semantics:
North means equal x and greater y; Northeast means greater x and greater y.
No implicit unit distances, finite grid, boundary, or equal-length steps.
Inputs remain flat conjunctions of atomic spatial statements. Logical formulas
used internally for verification do not restore Boolean-premise curricula.

## Entailment and Non-Entailment

For consistent premises P and query claim Q:

- Q is entailed if every valid arrangement satisfying P also satisfies Q.
  Verify by UNSAT of `P AND NOT Q`.
- Q is not entailed if at least one valid arrangement satisfies P and falsifies
  Q. Verify by SAT of `P AND NOT Q`, retaining a checked counterexample.
- A non-entailed claim may be contingent or impossible. Do not describe every
  non-entailed claim as false in every arrangement.
- Failure of forward chaining establishes neither entailment nor non-entailment
  unless completeness has been proved for that input class.
- Check whole-scene consistency first. Inconsistent premises return E under
  every contract by dataset convention; do not use vacuous entailment to select
  every option. UNKNOWN, timeout, or verification failure is an error, not E/F.

Possibility enumeration is not a model task or a gold-answer target. SAT checks
and counterexamples remain internal evidence for non-entailment. Optional
diagnostic classification of negatives as contingent or impossible must not
change which options are selected.

## Query Types

Query type specifies the claim; contract specifies the response.

| Query type | Claim for one domain value | Full domain |
|---|---|---|
| Direction | A has direction d relative to B. | All eight exact directions. |
| Selection/Which | Named scene object c has the queried relation to B. | All named scene objects, excluding B. |
| Count | Exactly k nonreference scene objects have the queried relation to B. | Integers 0 through the number of nonreference scene objects. |

Direction and Count have one outcome in each valid arrangement. Each therefore
has at most one individually entailed answer value. Count 2 is one answer,
not two Selection answers. Count 0 is a substantive value.

Selection may have zero, one, or several individually entailed candidates.
Unique Selection now means exactly one entailed candidate, not exactly one
qualifying object in every arrangement. If X always qualifies and Y qualifies
only in some arrangements, X is the sole entailed candidate and Unique selects
X when displayed. This intentionally replaces the old requirement that all
other candidates be impossible.

Count must be verified jointly over all nonreference scene objects. It is not the
number of entailed Selection memberships: candidate identities may vary while
the total count remains invariant.

The positive synthetic explanation calculus currently supports fixed-membership
Count derivations. The general solver's changing-membership invariant Count
diagnostic is tested separately using an existing non-atomic fixture, which
the atomic dataset admission rejects. Do not describe that diagnostic as a
Boolean training curriculum or claim complete Count proof coverage.

## Contracts and Escapes

Let T be the complete set of individually entailed domain answers and M the
four substantive menu values. Menus never redefine the eligible domain.

| Contract | Consistent-premise rule |
|---|---|
| Unique | If `|T| = 1`, select its displayed letter, or F if unlisted. If `|T| = 0` or `|T| > 1`, select E. |
| Listed | Select all letters corresponding to `T intersect M`. Select F if that intersection is empty. |
| Exhaustive | Select all letters for T if T is nonempty and `T subset M`. Otherwise select F. |

All contracts return E for inconsistent premises. E/F must stand alone.
Substantive letters are sorted uppercase without separators. Stage 1 ends
with exactly one `Answer: <letters>` line; Stage 2 returns letters only.

| Contract | E label | F label |
|---|---|---|
| Unique | Cannot be determined | None of the above |
| Listed | Cannot be determined | None of the above |
| Exhaustive | Cannot be determined | No complete entailed answer can be formed from A-D. |

These empty-set conventions preserve the earlier interface pattern:
Unique empty T -> E; Listed/Exhaustive empty T -> F. Retain distinct audit reasons
for inconsistency, no entailed answer, multiple entailed answers, a sole unlisted
answer, no listed entailments, and incomplete entailed coverage.

E for multiple entailed Selection answers is a failure of the request for one
answer, not uncertainty about those memberships. Empty T does not establish
that no candidate can qualify.

## Examples and Contract Overlap

- A relative to B can be North or Northeast, with neither individually forced:
  T is empty. Unique -> E; Listed -> F; Exhaustive -> F.
- X always qualifies, Y is contingent, and every other candidate is non-entailed:
  T is `{X}`. All contracts select X when displayed.
- X and Y always qualify: T is `{X, Y}`. Unique -> E; Listed selects whichever
  of X/Y are displayed; Exhaustive selects both only when both are displayed,
  otherwise F.
- Exactly two candidates qualify in every arrangement: Count T is `{2}`.
  Unique Count selects the letter for 2, or F if 2 is unlisted.

Listed and Exhaustive are identical for Direction and Count for every menu,
not only complete menus: their entailed sets have cardinality zero or one.
Selection supplies nontrivial Listed/Exhaustive differences. Report this
structural overlap; do not fabricate diversity or claim three independent
reasoning tasks for every query type.

Entailment and possibility are not disjoint categories: every entailed claim
is possible on consistent premises. The study's labels distinguish entailed
from non-entailed, not "entailed versus possible."

## Evidence and Explanations

For each admitted consistent synthetic base, retain full-domain semantic
checks. Every entailed claim needs a sound independently checked derivation
in the declared supported proof calculus. Every non-entailed claim needs a
full counterexample that independently passes every premise, distinctness,
and falsity of that claim.

Coordinate assignments are audit metadata, not supervision prose. Positive
entailment proofs do not require an additional positive coordinate witness.
Counterexamples need valid completions, not all-pairs canonicalization.
Extracting and rank-normalizing a solver model is allowed because it preserves
all coordinate equalities and strict orders.

For reproducible atomic-scene audit records, construct coordinate ranks from
deterministically ordered equality groups and strict-order graphs on each axis.
If that arrangement does not falsify a non-entailed claim, use incremental
unbounded SAT checks to fix the required query relation in a deterministic
direction order, then construct the graph arrangement again. Count fixes one
feasible alternative count and all eligible-candidate query relations jointly.
Context object pairs do not require separate SMT completion.

Entailment uses unbounded negated-claim checks; non-entailment uses independently
validated counterexamples. No rank bound is used to establish a semantic label.
Ranks represent coordinate orders, not unit distances or a finite-grid problem
assumption. Unknown solver results or failed graph completion reject the example.

Natural and Symbolic render the same checked evidence units, with identical
scope, conclusions, menus, and final answers. Natural uses ordinary spatial
language; Symbolic uses fixed predicates and axis comparisons. A counterexample
is explicitly an arrangement falsifying a claim, never a universal deduction.
Relevant qualitative relationships are backed by a checked full completion.

Premise sentences in the question are untouchable: do not prepend proof labels,
append axis annotations, or rewrite them for either explanation style. The
contract-specific final question and menu do not change the shared scene.

If an explanation cites premise keys, define them inside that explanation first.
The `self-contained-premise-keys-v1` reference renderer restates each cited
spatial premise once, verbatim, as `P1: <original sentence>`, using its one-based
position among the spatial statements in the question. The nonrelational
`is in the map` introduction has no proof index. It preserves the original order rather than renumbering a
selected subset. Natural and Symbolic share this factual key, then explain in
their respective styles. Uncited premises remain in the question and are still
enforced by the solver; they need not be repeated as reasoning steps.
Counterexample-only explanations without premise citations need no key.
Old saved reference versions remain unchanged.

Listed explanations cover the displayed claims. Exhaustive substantive answers
cover the full domain. An Exhaustive missing-coverage escape needs a checked
entailment proof for an unlisted answer, not a witness merely making it true.
Unique singleton justification covers the other domain claims; multiple
entailed answers can be demonstrated by two positive proofs. Empty-set
conclusions need non-entailment evidence for the relevant domain.

Include both contingent and impossible negative options. Report their coverage
separately using checked diagnostic metadata, without making possibility
enumeration a supervised output. Unsupported proofs, unverified qualitative
scenarios, unknown results, and token overflows are rejection reasons.

## Sampling and Evaluation

Retain the approved retained base targets: train 6,000; dev 1,200; in-range test
2,400; extrapolation 1,200. Equal query allocation; 70/15/10/5 in-range
controlled/random-world/random-atoms/targeted mixture; controlled-only
extrapolation. Retain depths 1-8 in-range and 9-10 extrapolation, the approved
depth weights, family floors, proof-diversity settings,
all-eight-direction presence, and existing verified irrelevant context.
Count 0-4 weights describe the approved scoped construction targets; their
all-scene feasibility and the depth/context checks must be reassessed under
the amendment above before a new execution configuration can be frozen.

Matched omission menus replace an entailed answer with a non-entailed valid
domain value. Retain the seeded 20% eligibility probability and the control.
Do not require the replacement to be impossible.

Recompute depth dependence against the entailed-answer set. Removing a premise
that changes only positive possibilities, but leaves T unchanged, cannot alone
establish entailment-answer dependence. Null-depth cases remain separate.

The approved entailment-only coverage changes are:

- No nonzero Listed/Exhaustive difference gate for Direction or Count, whose
  outputs always coincide under these definitions.
- Selection Listed/Exhaustive differences: at least 10 train / 3 dev /
  5 in-range bases from at least 10 / 3 / 5 distinct families, respectively.
  The user approved replacing 120 / 24 / 48 after a training-reserve check found
  only 23 qualifying bases. These are minima, not caps or statistical-power claims.
- Keep the applicable Unique/Listed and Unique/Exhaustive difference minima.
- Require contingent and impossible negatives at 30 train / 6 dev /
  12 in-range bases per query, using the approved proposed minima.
- Include at least one sole-entailed-with-contingent Selection case in each
  in-range split.

These approvals do not certify that the completed retained dataset satisfies
the gates. Do not reuse the old allocation as an entailment-only execution plan
or silently relax other requirements.

### Construction Reuse and Variants

The user permits reusing verified reasoning constructions; each training
example need not introduce a novel proof pattern. Vary object names, premise
order, menu positions and supported scene instances. Deliberately include
multiple-entailed Selection cases with missing-option menus.

Pure renamings and equivalent reorderings are additional views of the same
underlying question, not new independent bases or families. Keep the approved
base targets above unchanged and report additional training views separately.
Match Natural/Symbolic examples, labels and exposure for every such variant.

Synthetic model-visible text uses the existing 30-place vocabulary from
`spatial/v2/generation.py`, plus Courthouse. The extra name covers the 153 retained
bases with 31 objects without changing their scenes. Sample distinct names
without replacement using a deterministic story-level mapping. Reuse that
mapping across Direction, Selection, Count, menu variants, both explanation
styles, positive proofs and negative counterexamples. Preserve multiword names
such as City Hall and Train Station rather than converting them to numbered IDs.

Keep the original base, story and family IDs and split assignments. Store the
reversible naming map in audit metadata, never in model-input messages.
Re-render and remeasure all visible text, independently verify the renamed
claims, and retain the same labels, reasoning depths, quotas and token limits.
External SpatialMap source names and its approved reversible legend are a
separate protocol and remain unchanged by this synthetic naming revision.

Assign the split to the related family before creating variants. Never put a
renamed training construction into dev or test. Equivalent variants within a
held-out family must not be counted as independent evaluation observations or
used to satisfy distinct-family floors.

### Saved-Pool Menu-Position Revision

The first entailment-only execution was interrupted after preserving 20,087
verified admissions. Its fixed in-range menus could not display the required
`ABC` gold for Listed or Exhaustive, even though the fixed reserve contained
three-entailed-member Selection questions. This was a menu-position coverage
failure, not a missing spatial entailment construction.

Version `entailed-menu-position-views-v1` reuses those admissions. For Selection
with two or three entailed members, enumerate slot permutations and retain one
per distinct displayed entailed-letter mask: six positions for two members,
four for three. Apply each permutation to both primary and omission menus.
Other questions retain their original position only. Regenerate matched
Natural/Symbolic references, including the Count singular-wording correction,
and check every new label and token budget.

These are correlated views with unchanged base and family IDs. Report their
extra exposure separately; do not inflate independent evaluation sample sizes
or distinct-family counts. Keep all 10,800 base targets, source/depth quotas,
coverage minima and split assignments unchanged. Bind the new assembly to the
original configuration and immutable admission checkpoints, and separately
record its current software and tokenizer hashes. Preserve the interrupted
run unchanged; assembly is a new unsealed candidate, not an old-run resume.

The saved prefix also lacks a dev base with exactly three entailed Selection
members. If it cannot satisfy the unchanged gates, continue through unused
slots of the original fixed reserve in its original order, with the same seeds
and split assignments. Verify original task identities and development
exclusions, checkpoint new admissions separately, and stop at a
coverage-passing retention check. This is completion of the existing reserve,
not a refill or a new sampling plan.

### Coverage-Protected Retention

Use the fixed ordered reserve without retrying, refilling or moving families.
Verified admissions remain eligible for retention even when an earlier quota
prefix has filled. Select complete base questions, including every matched
contract/menu view and both explanation styles, subject to exact split,
source, depth and controlled Count quotas plus the approved coverage and
distinct-family floors.

The retention MILP selects already verified questions; it does not compute
spatial labels or replace Z3. Candidate order supplies a fixed rank preference.
Try selection when the ordered prefix can first supply every requirement and
stop once a feasible retained subset passes independent quota/coverage checks.
Record selected IDs separately from raw checkpoints. If the reserve is exhausted
without satisfying coverage, export only a clearly marked diagnostic subset
and report shortfalls; do not treat it as an accepted release.

Retain family-held-out internal evaluation and separately versioned external
SpatialMap-TQA-Corr views under these same contracts. Keep Corr SINGLE and
all historical possibility-based artifacts unchanged. Recompute new labels;
do not copy historical oracle letters. Corr is never training or checkpoint
selection data.

The user approved all five nonreference scene objects as the eligible domain
for external Selection. This is an explicit expansion of the historical
four-option candidate scope, not a change to Corr SINGLE. Preserve original
entity names through a verified reversible alias legend in the new views.

Retain the two matched SFT arms, two style-matched untuned evaluations, pinned
model/tokenizer, and separate Stage-1/Stage-2 evaluation. Report answer accuracy,
entailed-option inclusion and non-entailed-option exclusion where directly
observable, semantic escape reasons, depth, negative categories, and token cost.
An E/F response alone does not identify all non-entailed options.

## Migration and Release Gates

1. Version contracts, semantic records, evidence, prompts, configuration, and
   checkpoints so possibility-based output cannot be resumed or mislabelled.
2. Test truth tables, no-entailed cases, sole-entailment-plus-contingent Selection,
   multiple entailed memberships, invariant Count including changing identities,
   inconsistency, menu omission, permutations, and non-entailment evidence.
3. Check independent proof and counterexample replay, error rejection,
   transformations, explanation matching, token limits, and split isolation.
4. Measure development-fixture runtime and solver work before bulk generation.
   A faster design is not a measured speedup.
5. Freeze the revised supported-scope and coverage matrix; generate an unsealed
   candidate; report distribution, unsupported cases, and manual audit evidence.
6. Obtain separate approval before sealing or model training. Preserve old
   partial runs and receipts; never mix their labels with the new candidate.
