# SpatialEntail: Evidence Examples and Dataset Quality Checks

## Historical Contract Examples

The possibility-based answer obligations and golds in this note are historical.
They are not the active entailment-only supervision specification. Use
[ENTAILMENT-ONLY-v1.md](../ENTAILMENT-ONLY-v1.md) for current contracts and
[ENTAILMENT-FINAL-REVIEW-v5.md](../ENTAILMENT-FINAL-REVIEW-v5.md) for the
reviewed named candidate. The underlying spatial examples remain useful
diagnostics, but their old menu answers must not be copied into new data.

Supporting note for [spatialentail-composition-v1](../README.md).
The [main pivot](../pivot_20261026.md) is the source of truth for research
decisions; this note provides worked evidence examples and supporting checks.

Working specification, 2026-10-26. Supports findings 1 and 2 of the pivot review.
These are manually authored, solver-checked examples, not a generated training
release or evidence that arbitrary model-written explanations can be verified.

## Evidence Contract

Each base has one structured evidence record with premise IDs, semantic answer,
candidate domain, and checked claims/witnesses/conflicts. Render that record
twice: Natural sentences and Symbolic notation. Both expose the same facts,
deductions, witness assignments, and conclusion. Match the displayed evidence,
not just the final answer. No distance assumptions or Boolean input formulas.

Full possible sets and all candidate-exclusion solver checks remain in audit
metadata. Supervision contains enough checked evidence to justify the requested
contract answer, not a transcript of every solver call:

| Answer obligation | Required supervised evidence | Additional audit-only evidence |
|---|---|---|
| Unique Direction | Axis derivation establishing the exact direction. | Full eight-direction analysis and exclusion results. |
| Unique Selection | One always-qualifying candidate and exclusion of all others. | Membership possibility/necessity checks for every candidate. |
| Unique Count | Proof that the exact count is invariant. | Feasibility checks for every count from 0 to candidate count. |
| Underdetermined Unique | Different valid direction/count outcomes, or Selection membership evidence ruling out one invariant singleton. | Complete semantic analysis; use explicit no-match/multiple-match evidence where appropriate instead of fabricated ambiguity witnesses. |
| Listed set | A witness or guaranteed-membership proof for each included value; exclusion evidence for every other listed value. | Full-domain possibilities, including unlisted values. |
| Exhaustive set | Inclusion and exclusion evidence over the full declared domain. | Full feasibility receipt, not just the displayed options. |
| Unique F | Proof of the unique answer and its absence from the menu. | Canonical unique value and option mapping. |
| Exhaustive F, missing values | At least one valid witness for an unlisted value. | Full possible set; generating that full set is not required in the final answer or explanation. |
| Empty Selection / no listed possibilities | Exclusion evidence for all relevant candidates. | Full-domain analysis distinguishing emptiness from missing coverage. |
| Inconsistent E | Actual premise IDs forming a conflicting subset and a short contradiction derivation. | UNSAT check of that subset with named spatial axioms; no minimum-size claim. |

If a solver label is available but a faithful explanation is not, mark the
example unsupported for every synthetic split: training, development, and test.
Investigate explanation-generation failures and report exclusions and affected
categories. Preserve story/family split assignments; do not move rejected
training examples into test. Do not replace evidence with "the solver says so,"
invent an axis proof, or silently drop it from reported coverage. Reference
explanations and gold answers remain hidden from development/test model inputs.

External SpatialMap answer-accuracy evaluation may omit rendered reference
explanations if questions, domains, menus, and labels are verified. Retain
semantic audit evidence and report explanation availability. Neither a correct
answer nor an available reference explanation verifies the model's explanation.

## Conventions

All objects are distinct points. Exact North means equal x and greater y;
exact East means greater x and equal y. Coordinates below are witness
assignments illustrating order, not distances supplied in the question.

Object names X/Y/Z/U/V/W/T/O/Q are not menu letters. Symbolic `N(U,O)` means
U is exactly north of O; `!N(U,O)` means that query claim is false, not a
negated input premise. `possible={...}` lists canonical values, not simultaneous
memberships. `members(map)` lists objects qualifying together in that map.

Every prompt uses the pivot's fixed E/F labels. Answer strings below are in
**Unique / Listed / Exhaustive** order. Natural and Symbolic blocks are matched
evidence bodies. Append the appropriate *same* contract conclusion and
`Answer: <letters>` to both, using the menu table.

Premise summaries below are audit notation, not final prompts. Render input
stories as separate atomic pairwise sentences in SpatialMap-TQA style; never
merge several subjects into one premise. The new examples below show that form.

## D1: Entailed Direction and Excluded Candidates

Premises: p1 X is northeast of Y; p2 Z is south of Y.
Query: direction of X relative to Z.

Natural:
```text
p1 gives X a greater x-coordinate and greater y-coordinate than Y.
p2 gives Z the same x-coordinate as Y and a smaller y-coordinate.
Thus X's x-coordinate is greater than Z's, and its y-coordinate is greater
than Y's, which is greater than Z's. X is northeast of Z in every valid map.
The other seven exact directions are excluded by these coordinate signs.
```

Symbolic:
```text
p1: xX>xY; yX>yY
p2: xZ=xY; yZ<yY
xX>xY=xZ => xX>xZ
yX>yY>yZ => yX>yZ
NE(X,Z); possible={Northeast}; all other exact directions excluded
```

Menus: `[West, Northeast, South, Northwest]` gives `B / B / B`.
`[North, East, South, West]` gives `F / F / F`.
For the second menu, explain that the unique Northeast value is missing;
Listed has no possible displayed value, and Exhaustive lacks the complete set.

## D2: Two Possibilities, Complete and Partial Menus

Premises: p1 X is east of O; p2 Y is east of O.
Query: direction of X relative to Y.

Natural:
```text
X and Y share O's y-coordinate. Distinctness prevents equal x-coordinates,
so only East and West remain. With O at (0,0), X at (1,0), and Y at (2,0),
both premises hold and X is west of Y. Swapping X and Y gives another valid
map where X is east of Y. Both answers are possible; neither is guaranteed.
```

Symbolic:
```text
p1,p2: yX=yO=yY; xX>xO; xY>xO
distinct(X,Y) => xX!=xY
w1={O:(0,0), X:(1,0), Y:(2,0)} => W(X,Y)
w2={O:(0,0), X:(2,0), Y:(1,0)} => E(X,Y)
possible={East,West}; no unique direction
```

Menu `[North, East, South, Northeast]`: `E / B / F`.
Menu `[West, East, South, Northeast]`: `E / AB / AB`.
For partial-menu Exhaustive F, the West witness alone suffices to establish
missing coverage; retain both witnesses in audit metadata. If shortened for
that view, shorten both renderings identically.

## S1: Unique Selection with Full Candidate Exclusion

Eligible objects U,V,W,T; reference O; queried relation North.
Premises: p1 U north of Q; p2 Q north of O; p3 V east of O;
p4 W south of O; p5 T west of O. Menu `[U,V,W,T]`: `A / A / A`.

Natural:
```text
U, Q, and O have equal x-coordinates, and U's y-coordinate exceeds Q's,
which exceeds O's. U is always north of O. V and T have different
x-coordinates from O; W has a smaller y-coordinate. None can be north of O.
U is therefore the only qualifying object in every valid map.
```

Symbolic:
```text
p1,p2: xU=xQ=xO; yU>yQ>yO => N(U,O)
p3: xV>xO => !N(V,O)
p4: yW<yO => !N(W,O)
p5: xT<xO => !N(T,O)
possible=certain={U}; unique=U
```

## S2: Empty Selection Is Not Uncertainty

Eligible U,V,W,T; query North of O.
Premises: p1 U east of O; p2 V west of O; p3 W south of O;
p4 T southeast of O. Menu `[U,V,W,T]`: `E / F / F`.

Natural:
```text
U and T have greater x-coordinates than O; V has a smaller x-coordinate.
W has a smaller y-coordinate. Each candidate is therefore excluded from
being exactly north of O. No eligible object can qualify.
```

Symbolic:
```text
p1,p4: xU>xO; xT>xO => !N(U,O), !N(T,O)
p2: xV<xO => !N(V,O)
p3: yW<yO => !N(W,O)
possible=certain={}; no qualifying object
```

Unique uses E because no unique object exists. Listed/Exhaustive use F for
the empty set. This is not evidence of uncertain memberships.

## S3: Multiple Certain Selection Matches

Eligible U,V,W,T; query North of O.
Premises: p1 U north of O; p2 V north of U; p3 W east of O;
p4 T south of O. Menu `[U,V,W,T]`: `E / AB / AB`.

Natural:
```text
U is north of O. V shares U's and O's x-coordinate, and V's y-coordinate
exceeds U's, which exceeds O's, so V is also north of O. W's x-coordinate
differs from O's, and T's y-coordinate is smaller. Exactly U and V always
qualify, so there is no single qualifying object.
```

Symbolic:
```text
p1: N(U,O)
p1,p2: xV=xU=xO; yV>yU>yO => N(V,O)
p3: xW>xO => !N(W,O)
p4: yT<yO => !N(T,O)
possible=certain={U,V}; no unique object
```

## S4: One Possible Candidate Is Not One Guaranteed Candidate

Eligible U,V,W,T; query North of O.
Premises: p1 U northeast of V; p2 O northeast of V;
p3 W east of O; p4 T south of O. Menu `[U,V,W,T]`: `E / A / A`.

Natural:
```text
V's y-coordinate is below O's; W's x-coordinate differs from O's;
T's y-coordinate is below O's. None qualifies. Set O=(0,0), V=(-2,-2),
W=(2,0), T=(0,-2). U=(0,1) satisfies the premises and is north of O.
U=(1,1) also satisfies them but is not north of O. Only U is possible,
but it is not guaranteed, so Unique has no invariant sole answer.
```

Symbolic:
```text
p2: yV<yO => !N(V,O)
p3: xW>xO => !N(W,O)
p4: yT<yO => !N(T,O)
base={O:(0,0), V:(-2,-2), W:(2,0), T:(0,-2)}
w1=base+{U:(0,1)} => members(w1)={U}
w2=base+{U:(1,1)} => members(w2)={}
possible={U}; certain={}; no unique object
```

## S5: Five Qualifying Objects, Four Menu Slots

Eligible U,V,W,T,Y; query North of O. Premise IDs p1-p5 follow sentence order:

```text
Consider a map with multiple objects:
U is to the North of O.
V is to the North of O.
W is to the North of O.
T is to the North of O.
Y is to the North of O.
```

Natural:
```text
Each of p1-p5 states that its subject shares O's x-coordinate and has a
greater y-coordinate. All five eligible objects therefore always qualify.
A valid map is O=(0,0), U=(0,1), V=(0,2), W=(0,3), T=(0,4), Y=(0,5).
There is no unique qualifying object. The menu lists U,V,W,T but omits Y.
```

Symbolic:
```text
p1: xU=xO; yU>yO => N(U,O)
p2: xV=xO; yV>yO => N(V,O)
p3: xW=xO; yW>yO => N(W,O)
p4: xT=xO; yT>yO => N(T,O)
p5: xY=xO; yY>yO => N(Y,O)
w={O:(0,0),U:(0,1),V:(0,2),W:(0,3),T:(0,4),Y:(0,5)}
possible=certain={U,V,W,T,Y}; no unique object
menu={U,V,W,T}; Y not in menu
```

Menu `[U,V,W,T]`: `E / ABCD / F`. Exhaustive F means
"No complete answer can be formed from A-D." Do not drop Y from the answer
domain or limit the qualifying set to four. No full-coverage four-value menu
exists for this case.

## S6: Unique Selection Answer Missing from the Menu

Eligible U,V,W,T,Y; query North of O. Premise IDs p1-p5 follow sentence order:

```text
Consider a map with multiple objects:
U is to the East of O.
V is to the West of O.
W is to the South of O.
T is to the Southeast of O.
Y is to the North of O.
```

Natural:
```text
U and T have greater x-coordinates than O; V has a smaller x-coordinate.
W has a smaller y-coordinate. These four objects cannot be exactly north of O.
Y shares O's x-coordinate and has a greater y-coordinate, so Y is the only
qualifying object in every valid map. A valid map is O=(0,0), U=(1,0),
V=(-1,0), W=(0,-1), T=(1,-1), Y=(0,1).
```

Symbolic:
```text
p1,p4: xU>xO; xT>xO => !N(U,O), !N(T,O)
p2: xV<xO => !N(V,O)
p3: yW<yO => !N(W,O)
p5: xY=xO; yY>yO => N(Y,O)
w={O:(0,0),U:(1,0),V:(-1,0),W:(0,-1),T:(1,-1),Y:(0,1)}
possible=certain={Y}; unique=Y
```

Menu `[U,V,W,T]`: `F / F / F`; Y is uniquely guaranteed but unlisted.
Paired menu `[U,V,W,Y]`: `D / D / D`. Only the menu changes, not the
candidate set or premises. Append the corresponding menu conclusion in both
styles.

## C1: Joint Count Feasibility

Eligible U,V,W,T; query count North of O.
Premises: p1 U east of V; p2 W south of O; p3 T south of W.
Menu `[0,1,2,3]`: `E / AB / AB`. The full count domain is 0 through 4.

Natural:
```text
W and T are below O and cannot qualify. U's x-coordinate is greater than
V's, so they cannot both have O's x-coordinate. At most one can be north
of O. Set O=(0,0), W=(0,-1), T=(0,-2). With U=(0,1), V=(-1,1), exactly
U qualifies. With U=(1,1), V=(0,1), exactly V qualifies. With U=(2,1),
V=(1,1), neither qualifies. These maps satisfy all premises. The possible
counts are exactly zero and one, not two, three, or four.
```

Symbolic:
```text
p2,p3: yT<yW<yO => !N(W,O), !N(T,O)
p1: xU>xV => not(xU=xO=xV) => count<=1
base={O:(0,0), W:(0,-1), T:(0,-2)}
w1=base+{U:(0,1), V:(-1,1)} => members(w1)={U}; count=1
w2=base+{U:(1,1), V:(0,1)} => members(w2)={V}; count=1
w3=base+{U:(2,1), V:(1,1)} => members(w3)={}; count=0
possible_counts={0,1}; counts 2,3,4 excluded
```

This specifically rejects summing independently possible memberships.

Paired partial menu `[0,2,3,4]`: `E / A / F`. The full domain remains 0
through 4 and the possible counts remain `{0,1}`. Listed selects zero;
Exhaustive uses F because one is missing. For that Exhaustive view, either
count-one witness above demonstrates missing coverage; shorten both styles
identically if selecting only that evidence.

## C2 and C3: Invariant Count, Including Zero

C2 uses S3's premises and candidate domain, but asks Count.
Menu `[0,1,2,3]`: `C / C / C`.
Paired menu `[0,1,3,4]`: `F / F / F`, because the invariant count two is
unlisted. Reuse the membership proof in both styles and explain the omission.

Natural: use S3's membership derivation, then:
```text
Exactly U and V qualify in every valid map. The invariant count is two.
```
Symbolic: use the same S3 deductions, then:
```text
members={U,V} in every valid map => possible_counts={2}
```

C3 uses S2's premises and candidate domain, but asks Count.
Menu `[0,1,2,3]`: `A / A / A`.

Natural: use S2's exclusions, then:
```text
No eligible object qualifies in any valid map. The invariant count is zero.
```
Symbolic: use the same S2 exclusions, then:
```text
members={} in every valid map => possible_counts={0}
```

These renderers reuse the complete preceding evidence, not just the final
sentence. Zero is a substantive answer, unlike an empty Selection set.

## I1 and I2: Global Contradiction and Consistent-Cycle Control

I1 premises: p1 X east of Y; p2 Y east of Z; p3 Z east of X;
p4 U north of V. Query direction U relative to V.
Menu `[North,East,South,West]`: `E / E / E`.

Natural:
```text
p1 and p2 imply X's x-coordinate is greater than Z's. But p3 requires
Z's x-coordinate to be greater than X's. Premises p1,p2,p3 cannot all
hold. The entire map is inconsistent, despite the directly stated query fact.
```
Symbolic:
```text
p1,p2: xX>xY>xZ => xX>xZ
p3: xZ>xX
xX>xZ>xX => contradiction; conflict={p1,p2,p3}
```

I2 replaces only p3 with X east of Z. Same query/menu: `A / A / A`.

Natural:
```text
p1 and p2 imply X is east of Z, agreeing with p3. A valid map is
Z=(0,0), Y=(1,0), X=(2,0), V=(0,2), U=(0,3).
p4 guarantees U is north of V, excluding every other exact direction.
```
Symbolic:
```text
p1,p2 => E(X,Z), matching p3
w={Z:(0,0),Y:(1,0),X:(2,0),V:(0,2),U:(0,3)}
p4: xU=xV; yU>yV => possible={North}
```

The undirected premise graph has a triangle in both examples. Cycle presence
is not itself inconsistency; check the constraints and the whole map.

## I3: Contradiction from Forced Coincidence

Premises: p1 X north of O; p2 Y north of O; p3 X east of T;
p4 Y east of T. Query direction X relative to O.
Menu `[North,East,South,West]`: `E / E / E`.

Natural:
```text
p1 and p2 force X and Y to share O's x-coordinate. p3 and p4 force them
to share T's y-coordinate. X and Y would occupy the same point, violating
the distinct-point assumption. Premises p1,p2,p3,p4 conflict with that axiom.
```
Symbolic:
```text
p1,p2: xX=xO=xY
p3,p4: yX=yT=yY
(xX=xY) AND (yX=yY) contradicts distinct(X,Y)
conflict={p1,p2,p3,p4, distinct-point axiom}
```

Strict-cycle detection alone would miss this conflict.

## Executable Checks and Limits

Source: `spatial/v2/tests/test_pivot_evidence.py`.

```bash
uv run --offline --python 3.12 --no-project --with pytest --with z3-solver \
  python -m pytest spatial/v2/tests/test_pivot_evidence.py -q
```

The fixtures check all fourteen semantic cases, 19 menu views across three
contracts and all A-D permutations, explicit and solver-produced witnesses,
conflicting subsets, renaming/order invariance, atomic-story render/parse
round trips, D1 claim entailment/refutation, and small reference-backend
agreement for D1/D2. The menu mapper is a test specification, not production
six-option grading. Natural/Symbolic prose has been manually checked against
these fixtures; no arbitrary-prose validity checker is claimed.

Witness-enabled checks require a witness for every possible value, then
independently validate each map and its query outcome. Negative tests remove
one or all witnesses for Direction, Selection, and Count and require rejection.
Menu checks include domain validity, five qualifying Selection objects,
unlisted unique Selection/Count answers, and partially listed Count sets.

Implementation update: the [expanded evidence path](../README.md#expanded-evidence-path)
now constructs checked evidence and matched renderings for all fourteen fixtures
and all 57 contract/menu views. It reuses shared Direction/Selection certificates,
adds replayable strict-cycle/forced-coincidence conflicts, and supports Count
through fixed memberships and verified pair incompatibilities. Count support is
bounded to eight eligible objects and rejects uncovered higher-order exclusions.
The separately versioned interface includes nine prompt templates, full-prompt
round trips, A-F mapping, and strict answer parsers.
The prose in this note remains hand-authored; it is not the generated renderer.
The generated pack uses fixed fixture stories, not random generation, and does
not establish dataset release readiness. Broader evidence feasibility, manual
coverage audits, final format/sampling freeze, release gates, runtime structured
decoding, and model runs remain pending.

Validation on 2026-10-27: all 75 evidence tests passed. Combined with
`spatial/v2/tests/test_solver_correctness.py`, 740 tests passed using Python 3.12
and the installed `z3-solver` package version 5.3.2.0. Ruff checks passed.
These results validate the fixture suite, not the unimplemented release gates.

## Dataset Quality Checks

These are release requirements, not a statement that a release has passed.

1. **Scope and evidence coverage:** freeze the supported family/contract/case
   matrix and evidence schema before bulk generation. Each admitted evidence
   category needs a checked Natural/Symbolic pair. Report unsupported cells
   and every rejected attempt by reason; never count a solver timeout as E/F.
   Every released synthetic training/development/test example needs both
   verified reference renderings. The external SpatialMap exception above is
   restricted to answer-accuracy evaluation.
2. **Semantic fixtures:** all known-answer checks must pass, including this pack,
   equality substitution, unknown ordering, zero count versus empty Selection,
   inconsistent whole maps, and menu omission without altered candidate domains.
   Add error/timeout injection tests before admitting the production generator.
3. **Every released row:** require successful full-domain solver analysis,
   complete witness coverage for possible values and independent direct
   evaluation of every supplied witness against all premises
   and distinctness, conflict-subset rechecking, evidence-step validation, and
   exact menu/answer mapping. A single failed check quarantines the row.
4. **Text round trips:** parse the released story and structured query/domain
   back to the intended problem, and validate contract instructions separately
   against their frozen templates. Require exact normalized agreement. Test
   the final six-option prompt, not only the legacy story parser.
5. **Metamorphic checks:** for every released base, check one deterministic
   object-renaming and premise-order variant plus an A-D permutation per view.
   Require preserved canonical semantics and correctly remapped answer letters.
   Check paired full/partial menus against the same full possible set.
6. **Independent cross-check:** run small supported cases through the existing
   reference backend or an independently implemented bounded order enumeration.
   Report its scope, disagreement count, and any shared implementation. A bounded
   search is not a general UNSAT proof without a completeness argument.
7. **Manual audit:** before training, inspect at least one example from every
   populated family/contract/semantic-case cell and at least five distinct bases
   per evidence category (overlap allowed). Inspect both renderings, query domain,
   menu, proof/witness content, and escape meaning. Any confirmed semantic,
   rendering, or label error blocks release pending a fix and re-audit.
8. **Reproducible receipt:** save source/config/solver versions, seeds, file
   hashes, unique-base and rendered-row counts, test results, audit sample IDs,
   coverage, and rejection reasons. Assign splits by story/family, keeping all
   related queries, menus, renderings, and matched mutations together; validate
   disjointness. Keep audit evidence outside model inputs unless explicitly
   selected as matched training supervision. Never expose reference explanations
   or gold answers in development/test inputs.

Passing Z3 checks alone is insufficient. Release requires all applicable gates
and zero unresolved correctness failures; unsupported verification coverage
must be reported rather than presented as a passed check.
