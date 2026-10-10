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
Which, or Count query. @spatialentail-solver-lifecycle summarizes the decision
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
    [Answer certificate], [Cover every declared Direction candidate, Which entity, or Count value.],
  ),
  caption: [Complementary proof, refutation, and model evidence.],
) <spatialentail-evidence-types>

Proof steps name their inputs, rule, conclusion, premise index, and branch.
Replay rejects unknown or later dependencies, cross-branch leakage, arbitrary
refutation assumptions, dead steps, and unsupported rule applications. Model
certificates independently evaluate every premise and the expected claim value.

Direction certificates distinguish unique entailment, contingency, and
impossibility. Which certificates classify every candidate as entailed,
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

== Four-Tier Curriculum

Difficulty is multi-dimensional. Spatial depth, propositional depth, branch
count, ambiguity size, candidate count, independent-axis support, distractors,
target length, and rejection rate remain separate manifest fields. The tiers
organize training coverage; they do not replace those measurements.

=== Tier 1: Foundations, depths 1–2

#figure(
  table(
    columns: (1fr, auto),
    table.header([*Task bucket*], [*17K quota*]),
    [Cardinal Direction, depths 1–2], [600],
    [Diagonal Direction, depths 1–2], [600],
    [Which membership, depths 1–2], [600],
    [Fixed Count, depths 1–2], [600],
    [*Tier total*], [*2,400*],
  ),
  caption: [Foundation-task quotas in the full nested pool.],
) <tier-one-curriculum>

=== Tier 2: Composition, depths 3–4

#figure(
  table(
    columns: (1fr, auto),
    table.header([*Task bucket*], [*17K quota*]),
    [Direction depth 3], [1,000],
    [Direction depth 4], [1,000],
    [Independent or unequal axes, depths 2–4], [1,000],
    [Which membership chains, depths 3–4], [1,000],
    [Derived unique Count, depths 3–4], [1,000],
    [*Tier total*], [*5,000*],
  ),
  caption: [Compositional-task quotas.],
) <tier-two-curriculum>

=== Tier 3: Uncertainty and Exclusion

#figure(
  table(
    columns: (1fr, auto),
    table.header([*Task bucket*], [*17K quota*]),
    [Direction ambiguity size 2], [1,000],
    [Direction ambiguity size 3], [1,000],
    [Which contingent/no-match mixture], [1,000],
    [Correlated Count ambiguity], [1,000],
    [*Tier total*], [*4,000*],
  ),
  caption: [Uncertainty-task quotas. These cells are described by semantic possibility and exclusion burden rather than fabricated positive-proof depth.],
) <tier-three-curriculum>

=== Tier 4: Logic and Branching

#figure(
  table(
    columns: (1fr, auto),
    table.header([*Task bucket*], [*17K quota*]),
    [Implication and IFF], [1,400],
    [Double negation and disjunctive syllogism], [1,400],
    [Case split], [1,400],
    [Nested case split], [1,400],
    [*Tier total*], [*5,600*],
  ),
  caption: [Logical-task quotas. Query families are balanced internally where the supported construction fragment permits.],
) <tier-four-curriculum>

The complete pool contains 17,000 unique base problems. Deterministic 4K and 8K
subsets preserve the tier/task proportions. The unequal allocation is a
predeclared curriculum choice: foundational tasks receive fewer examples,
while composition, uncertainty, and branching receive more.

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
