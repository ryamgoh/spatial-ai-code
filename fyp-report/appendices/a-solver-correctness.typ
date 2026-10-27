#pagebreak(weak: true)
#heading(level: 1, numbering: none)[Appendix A: Spatial Correctness Argument]

This argument concerns the declared qualitative spatial encoding, not formal
verification of the implementation, renderer, or solver executable.

#heading(level: 2, numbering: none)[Exact Directions and Distinct Points]

For two distinct objects a and b, an exact direction is the sign pair of
their X and Y coordinate differences. The pair (0, 0) is excluded.

#figure(
  table(
    columns: (1fr, auto, auto),
    table.header([*Direction*], [*X comparison*], [*Y comparison*]),
    [North], [$x_a = x_b$], [$y_a > y_b$],
    [Northeast], [$x_a > x_b$], [$y_a > y_b$],
    [East], [$x_a > x_b$], [$y_a = y_b$],
    [Southeast], [$x_a > x_b$], [$y_a < y_b$],
    [South], [$x_a = x_b$], [$y_a < y_b$],
    [Southwest], [$x_a < x_b$], [$y_a < y_b$],
    [West], [$x_a < x_b$], [$y_a = y_b$],
    [Northwest], [$x_a < x_b$], [$y_a > y_b$],
  ),
  caption: [Exact compass directions, which partition the relative positions
    of distinct points.],
)

The solver enforces distinctness using
$x_a != x_b or y_a != y_b$ for every pair of distinct objects.
Sharing one coordinate remains allowed.

#heading(level: 2, numbering: none)[Finite Order Representation]

Every finite real-coordinate map has an equivalent integer-rank
representation. Replace each distinct X value by its sorted rank and repeat
independently for Y. This preserves every strict comparison and equality.
Distinct points cannot become coincident unless both coordinates were
already equal. Conversely, integer coordinates satisfying distinctness
define a valid map.

Thus ranks can represent qualitative order without imposing distances,
equal-length steps, or a new semantic boundary. For n objects, n ranks
on each axis suffice in principle. A smaller computational bound must not
be used to declare a possible configuration impossible.

#heading(level: 2, numbering: none)[Semantic Query Checks]

Let P include all atomic premises and spatial axioms. First check whether
P is satisfiable. Only a conclusive unsatisfiable result establishes
inconsistency.

On consistent premises, a claim Q is possible precisely when P together
with Q is satisfiable, and entailed precisely when P together with its
negation is unsatisfiable. The rank correspondence preserves these claims.
Internal conjunction, negation, and disjunction remain necessary for these
checks; they are not visible curriculum operators.

Direction checks all eight directions. Selection checks possibility and
necessity for every eligible candidate. A Unique Selection answer requires
one necessary member and no other possible member.

For Count, the solver evaluates the sum of membership indicators in one
shared model and checks each integer from zero through the candidate count.
This preserves correlations. Separate witnesses for individual candidates
cannot establish that those candidates qualify simultaneously.

#heading(level: 2, numbering: none)[Axis Derivations]

Decomposition follows the direction table. Equality substitution preserves
comparisons, and strict-order transitivity preserves order along a directed
path. Recombination requires both axis comparisons for the same ordered pair.
Premise IDs and path dependencies permit local checking of these steps.

A strict cycle is contradictory, but an arbitrary graph cycle need not be.
Two objects both east of a third do not acquire an ordering relative to each
other. Failure of axis closure is not evidence of ambiguity or impossibility,
especially when distinct-point constraints couple the axes.

#heading(level: 2, numbering: none)[Witnesses and Trust Boundary]

Witness validation checks exact object coverage, distinct points, each atomic
premise by direct coordinate comparison, and the requested query outcome.
Differing valid query outcomes establish underdetermination. Conflicting
subsets require a fresh consistency check under the same spatial axioms;
minimality is not claimed.

Full-domain solver decisions, evidence-step validation, direct witness
evaluation, small-case cross-checks, parser round trips, and metamorphic
tests provide complementary checks. Shared definitions and incomplete
coverage must be reported. Unknown results, timeouts, or unsupported
explanations block admission rather than becoming gold labels.
