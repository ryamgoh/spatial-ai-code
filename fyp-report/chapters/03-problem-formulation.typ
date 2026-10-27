#pagebreak(weak: true)
= Problem Formulation

== Spatial Assumptions

A problem consists of distinct named points, atomic pairwise premises, and a
Direction, Selection, or Count query. All points share fixed global X and Y
axes. Two objects may share one coordinate but may not occupy the same point.
There is no implicit grid spacing, map boundary, or distance constraint.
Unstated relations remain unknown rather than false.

Each premise is rendered as one sentence, for example, "A is to the Northeast
of B." Northeast means greater X and greater Y; exact North means equal X and
greater Y. Northeast is not the conjunction of exact North and exact East.
Selection and Count declare eligible objects separately from the displayed
menu, excluding the reference object.

== All-Valid-Map Semantics

Let $P$ denote the conjunction of the atomic premises and the declared spatial
axioms. A valid map satisfies all of $P$. Whole-map consistency is checked
first, including conflicts unrelated to the query.

A claim $Q$ is possible when $P and Q$ is satisfiable. On consistent premises,
it is entailed when $P and not Q$ is unsatisfiable. These solver checks do not
introduce logical-rule premises into the visible dataset. A timeout or unknown
result establishes neither inconsistency nor a semantic answer.

A witness establishes possibility. A counterexample disproves entailment but
does not alone prove ambiguity: the claim might be impossible. Underdetermination
requires valid maps yielding different answers.

Direction possibilities range over all eight exact compass directions.
Selection possibilities are individually feasible eligible objects, which
need not qualify simultaneously. Count possibilities require joint feasibility
in a single valid map; summing independently possible memberships is invalid.
Zero is an ordinary Count answer.

== Answer Contracts

Let $S$ be the complete possible-answer set and $M$ the substantive values in
the menu. Every new prompt has four distinct substantive options at A-D and
fixed escapes at E and F.

#figure(
  table(
    columns: (1fr, 1.7fr),
    table.header([*Contract*], [*Consistent-premise answer*]),
    [Unique Answer], [One invariant value; Selection additionally requires that
      candidate to qualify in every valid map and every other candidate never
      to qualify.],
    [Listed Possibilities], [All displayed values in the intersection of S and M.],
    [Exhaustive Possibilities], [The entire nonempty S when A-D can represent it;
      otherwise the escape F.],
  ),
  caption: [Matched contracts over the same underlying spatial problem.],
)

Under every contract, inconsistent premises return E. Under Unique, a
uniquely guaranteed listed answer returns its letter and an unlisted unique
answer returns F. Failure of uniqueness returns E, including Selection
no-match, multiple certain matches, and uncertain membership. These reasons
remain distinct in metadata even when their final letter is identical.

Under Listed, an empty displayed possibility set returns F, not E. Under
Exhaustive, F denotes a missing possible value or an empty complete set.
Multiple possibilities alone never justify E under Listed or Exhaustive.
A Selection domain may contain more than four qualifying objects; the menu
does not restrict the semantic domain.

E is labelled "Cannot be determined" in all contracts. F is labelled "None of
the above" for Unique and Listed, and "No complete answer can be formed from
A-D." for Exhaustive. These labels are fixed independently of the gold answer.

Final answers contain only letters: one letter under Unique; a nonempty A-D
subset for ordinary Listed/Exhaustive answers; or E alone or F alone. Escape
letters cannot be combined with other letters. Exact set serialization and
parser rules must be frozen before training.

== External Evaluation Boundary

The existing SpatialMap-TQA-Corr SINGLE release and its protocol remain
unchanged. New contract views will be versioned separately under these
declared semantics, retaining source story IDs, question IDs, query families,
targets, and references. The original authors' semantics must not be inferred
from our adaptation.

Each eligible source question receives matched contract views, explicit
candidate domains, and rebuilt six-option menus. Full-coverage controls are
paired with partial-coverage variants where feasible. More than four possible
values necessarily prevents full coverage.
