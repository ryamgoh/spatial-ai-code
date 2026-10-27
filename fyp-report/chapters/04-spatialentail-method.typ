#pagebreak(weak: true)
= SpatialEntail Method

== Construction and Verification

The proposed pipeline constructs atomic spatial stories, assigns their split,
and generates typed queries and matched contract views. The existing solver
and training infrastructure are reused. The new release pipeline, explanation
renderers, and six-option interface require implementation and validation;
the specification alone is not evidence of completion.

Two components have complementary roles. An axis graph constructs transparent
deductions, while Z3 checks consistency, entailment, feasibility, and Count
correlations against the full spatial axioms.

For each axis, known-equal coordinates form groups and strict comparisons
form directed edges. Reachability propagates order; equality substitution
allows paths through equal coordinates. Strict cycles, including a strict
edge within an equality group, are contradictory. Every displayed deduction
retains premise provenance.

Failure to find a graph derivation is not proof of underdetermination.
Distinct-point constraints couple the axes, and joint Count reasoning cannot
be reduced to independent reachability. All admitted evidence must be checked
against the complete problem.

== Evidence and Matched Explanations

One structured evidence record supplies both explanation styles. Natural uses
ordinary sentences describing coordinate order; Symbolic uses a fixed
predicate-and-axis grammar. Both expose the same selected deductions,
witnesses, exclusions, and conclusion. Replayable audit records are separate
from these supervision targets.

For a definite direction, the explanation interprets premises, composes
axis constraints, states the resulting direction, and encodes the menu answer.
For example, A northeast of B and C south of B imply greater X and Y for A
relative to C, hence A northeast of C. No distance or equal-step assumption
is needed.

Evidence depends on the contract:

- Unique answers require a definite direction, complete singleton-membership
  justification, or invariant-count justification.
- Unique E explains the actual failure of uniqueness or inconsistency.
  Known-empty Selection and multiple certain matches must not be described
  using fabricated ambiguity witnesses.
- Listed answers justify every included value and exclude every other listed
  value.
- Ordinary Exhaustive answers cover inclusion and exclusion over the entire
  answer domain. An unlisted-value witness suffices to justify Exhaustive F
  for missing coverage.
- Empty-set escapes justify the relevant exclusions.
- Inconsistent E cites a verified conflicting premise subset and any required
  spatial axioms. The subset is not claimed minimal.

Full semantic answers and verification receipts remain in audit metadata.
Every released synthetic train, development, and test example requires
verified matched reference explanations. Reference answers and explanations
are excluded from development/test model inputs.

== Difficulty and Split Controls

Chain length counts premise links along a supporting axis path, recorded
separately for X and Y when needed. A supporting path need not be shortest.
Verified distractors are a checked set removable individually and jointly
without changing consistency or the complete semantic answer.

Semantic case is a category, not a difficulty ranking. Possible-answer count,
returned-answer count, and menu coverage are reported separately.

Split assignment occurs once per story family before constructing questions
and views. Questions about one story share a story identifier. Renamed,
reordered, and deliberately matched consistent/inconsistent variants remain
in the same family and split. Merely assigning different question identifiers
does not establish independence.

== Release Gates

Before bulk generation, freeze the supported family/contract/semantic-case
matrix and evidence schema. Each admitted evidence category needs a checked
Natural/Symbolic pair.

Per-row checks require conclusive full-domain analysis, independent direct
witness evaluation, conflicting-subset rechecks, evidence-step validation,
and correct option mapping. Rendered premises, query, candidates, contract
instructions, and the full prompt must round-trip to the intended problem.
Failures and unknown solver outcomes are quarantined, not relabelled.

Metamorphic checks cover deterministic renaming and premise order, A-D
permutations, and matched menu coverage. Small supported cases are compared
against a reference backend or justified order enumeration. Shared code and
cross-check coverage must be disclosed.

A manual audit inspects at least one item per populated
family/contract/semantic-case cell and five distinct bases per evidence
category, with overlap allowed, checking both renderings. Any confirmed
semantic, rendering, or label error blocks release until fixed and re-audited.

The release receipt records versions, hashes, seeds, coverage, test outcomes,
audit sample IDs, split checks, counts, and rejection reasons. Unsupported
evidence categories are reported explicitly; unsupported samples must not be
silently moved to another split.
