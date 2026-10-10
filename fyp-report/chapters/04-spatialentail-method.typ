#pagebreak(weak: true)
#import "../figures/solver-lifecycle.typ": system-lifecycle, solver-lifecycle

= SpatialEntail Method

== Framework Overview

SpatialEntail uses one structured boundary, `SpatialProblem`, to separate text
and dataset conventions from semantic reasoning. A released benchmark row and a
synthetic generation policy reach the same solver contract. Presentation,
grading, certificates, and storage occur only after semantic analysis.

#system-lifecycle <spatialentail-system-lifecycle>

The active pipeline is:

```text
generation policy
→ visible premises and typed query
→ all-model semantic analysis
→ checked answer certificate
→ Natural or Symbolic supervision projection
→ exact context admission
→ frozen train/development/test artifacts
```

No generated row reads its gold answer from a privileged map. A world-first
route may propose a problem from coordinates, but the proposed answer is
discarded and the visible premises are re-solved before acceptance.

== Semantic Solver and Answer Policy

The solver receives objects, one finite premise formula, and a Direction,
Selection, or Count query. @spatialentail-solver-lifecycle summarizes the decision
path.

#solver-lifecycle <spatialentail-solver-lifecycle>

The exact backend translates formulas to integer constraints and checks every
candidate against the complete premise formula. A separate exhaustive backend
supports bounded differential tests. Models are canonicalized only after a
satisfying assignment has been found; canonical coordinates improve artifact
determinism but do not define the semantics.

The answer-policy layer maps query analyses into `SINGLE`, `ALL_POSSIBLE`, or
`VISIBLE_POSSIBLE` decisions before a menu encoder assigns option letters.
Ordinary and special options participate in one seeded shuffle, preventing a
fixed _Cannot be determined_ position from becoming a label shortcut.

== Checked Evidence

SpatialEntail distinguishes deductive and constructive evidence:

#figure(
  table(
    columns: (auto, 1fr),
    inset: (x: 7pt, y: 5pt),
    table.header([*Evidence*], [*Obligation*]),
    [Proof], [Replay rule applications and declared dependencies to establish an entailed relation.],
    [Refutation], [Assume a candidate in a scoped branch and derive an explicit contradiction.],
    [Consistency witness], [Exhibit one world satisfying the premises so a positive proof is not vacuous.],
    [Possibility witness], [Exhibit one premise-satisfying world in which a candidate holds.],
    [Countermodel], [Exhibit a premise-satisfying world in which a proposed necessity fails.],
    [Answer certificate], [Cover every declared Direction candidate, Selection entity, or Count value.],
  ),
  caption: [Complementary proof, refutation, and model evidence.],
) <spatialentail-evidence-types>

Proof steps name their inputs, rule, conclusion, premise index, and branch.
Replay rejects unknown or later dependencies, cross-branch leakage, arbitrary
refutation assumptions, dead steps, and unsupported rule applications. Model
certificates independently evaluate every premise and the expected claim value.

Direction certificates distinguish unique entailment, contingency, and
impossibility. Selection certificates classify every candidate as entailed,
contingent, or impossible. Count certificates preserve joint dependencies:
fixed memberships are proved once, while only residual compatible assignments
are enumerated. Count evidence can still grow exponentially in genuinely
contingent candidates, so exact context admission remains necessary.

== Generation Provenance

The generator records construction order explicitly.

#figure(
  table(
    columns: (auto, 1fr),
    inset: (x: 7pt, y: 5pt),
    table.header([*Route*], [*Construction*]),
    [World-first], [Sample a map, derive visible premises, discard the source answer, and re-solve the text.],
    [Proof-template-first], [Instantiate a controlled path or Boolean obligation, solve it, then construct and replay the actual certificate.],
    [Premise-first], [Sample visible premises and a query without selecting an answer, then solve, classify, and certify the discovered semantics.],
  ),
  caption: [Implemented SpatialEntail generation routes. Certificate-first remains a proposed stronger design, not an implementation label.],
) <spatialentail-provenance>

Every accepted candidate is solved, checked against its requested semantic and
difficulty cell, certificate-built, rendered to text, parsed back, and solved
again. Unsupported proof construction, inconsistent premises, menu disagreement,
or round-trip drift causes rejection.

Canonical source signatures ignore entity names, premise order, candidate order,
and commutative operand order while retaining directions and implication order.
Exact canonicalization is bounded; a conservative fallback may merge distinct
structures. Signature separation is a leakage defence, not proof of
compositional generalisation.

== Task Taxonomy

SpatialEntail applies one entailment-and-possibility semantics to three query
families. Direction Identification (`DIR`) classifies exact directions, Entity
Selection (`SEL`) classifies candidate memberships, and Cardinality
Determination (`CNT`) classifies jointly compatible counts. `SEL` is the generic
name for SpatialMap's _Which_ questions. Entailment is not a fourth query family.

#figure(
  table(
    columns: (auto, 1fr, 1.35fr),
    inset: (x: 6pt, y: 5pt),
    table.header([*Query*], [*Candidate domain*], [*`SINGLE` contract*]),
    [`DIR`], [Declared exact compass directions], [Exactly one possible direction.],
    [`SEL`], [Declared candidate entities], [Exactly one entailed entity and no other possible entity.],
    [`CNT`], [Integers from zero through the candidate count], [Exactly one possible count, preserving joint memberships.],
  ),
  caption: [Three query projections governed by the same all-model semantics.],
) <query-family-contracts>

Selection is a singular contract. If two entities are both certainly in the
requested relation, neither membership is unknown, but `SINGLE` cannot select
one. `ALL_POSSIBLE` returns the union of individually possible entities rather
than possible complete membership sets. Unique selection, multiple certain
matches, contingent membership, and no match are therefore separate result
strata.

Three capability lenses organize the supported reasoning structures:

#figure(
  table(
    columns: (auto, 1fr, 1.7fr),
    inset: (x: 6pt, y: 5pt),
    table.header([*Code*], [*Capability lens*], [*Scope*]),
    [`RR`], [Relational Reasoning], [Relation access, inversion, decomposition, transitivity, and axis recomposition.],
    [`LR`], [Logical Reasoning], [Negation, conjunction, disjunction, implication, equivalence, and branch scope.],
    [`MTR`], [Model-Theoretic Reasoning], [Necessity, possibility, impossibility, and query-level invariance across satisfying worlds.],
  ),
  caption: [Capability lenses. They are not chronological stages or universal hardness tiers.],
) <capability-lenses>

The experimental source of truth is not one flat task list. A cell is the
predeclared combination

$ "query contract" times "support/rule structure" times "semantic status" times "evidence burden". $

Compact structure codes include atomic access (`ARC`), transitive composition
(`TRC`), independent-axis composition (`AXC`), modus ponens (`MP`), equivalence
elimination (`IFF`), negation (`NEG`), disjunctive elimination (`DJE`), and
case-split reasoning (`CSR`). Semantic/evidence codes include unique or
invariant resolution (`UNI`), alternative answers (`ALT`), multiple certain
selections (`MUL`), no match (`NOM`), candidate exclusion (`EXC`), and joint
Count reasoning (`JCR`). For example, `DIR-TRC-UNI` and `CNT-CSR-JCR-UNI`
identify interpretable cells without claiming that all code combinations are
valid.

Depth, formula nesting, branch count, candidate and possibility-domain sizes,
independent-axis support, distractors, token length, and rejection rate remain
separate measured fields. Mixed logical-spatial composition is a holdout regime:
every primitive appears during training while selected rule compounds or
formula trees are withheld. Syntax exposure, replay-verified rule use, and a
semantic change under a controlled intervention are reported separately; none
alone proves that one rule is unavoidable across every derivation.

The maximum `4K`/`8K`/`17K` nested sizes remain candidate calibration budgets,
but cell quotas, matched-test totals, and quality-review strata must be frozen
together after a capability-coverage and generation-yield pilot. No final 17K
corpus or cell allocation is claimed in this draft.

== Supervision Views

Natural traces narrate checked premise, dependency, and branch identifiers.
Symbolic traces serialize typed reasoning and menu-decision records as NDJSON.
Training model evidence uses qualitative X/Y rank groups; coordinate assignments
remain audit artifacts.

The planned arms are:

- answer-only;
- checked Natural;
- checked Symbolic with rank witness;
- corrupted Symbolic, in which one typed assertion is invalid while syntax,
  decision, and answer remain intact; and
- Symbolic local proof, which omits the full rank witness while retaining the
  query-directed proof/refutation.

The local-proof versus rank-witness comparison is restricted to unique/entailed
cells. Ambiguous cases retain witnesses because contrasting valid worlds are the
evidence for ambiguity. A valid external trace does not establish that the LLM
used the same steps internally.

== Validation and Admission

All variants of one base problem share its prompt, menu, gold domain, and split.
Structural clusters and explicit holdout cells are separated across training,
development, and test. Exact tokenizer admission checks the complete rendered
training sequence and the evaluation prompt plus generation reserve. If any
required arm fails, the entire paired base is rejected and a diagnostic artifact
records the reason. No trace is truncated to force admission.
