#pagebreak(weak: true)
#import "../figures/solver-correctness.typ": semantic-correspondence, assurance-case

#heading(level: 1, numbering: none)[Appendix A — Solver Correctness Argument]

This appendix states the correctness argument for the structured solver. The
claim concerns the encoding relative to the declared finite qualitative
ontology. It does not formally verify the Python implementation, the natural-
language adapter, or Z3 itself.

#heading(level: 2, numbering: none)[Definitions]

Let $L$ be a finite, non-empty set of locations. A spatial interpretation is a
function $p: L arrow.r RR^2$ such that distinct locations do not share both
coordinates. For $a, b in L$, let

$
  sigma(a,b) = ("sgn"(p_x(a)-p_x(b)), "sgn"(p_y(a)-p_y(b))).
$

The no-co-location condition excludes $(0,0)$. The remaining eight sign pairs
are identified with North, Northeast, East, Southeast, South, Southwest, West,
and Northwest. A relation atom $R_D(a,b)$ holds exactly when
$sigma(a,b)$ belongs to the sign-pair set denoted by $D$.

The eight exact relations are therefore the following partition.

#figure(
  table(
    columns: (1fr, auto, auto, auto),
    table.header([*Direction*], [*X sign*], [*Y sign*], [*Comparison*]),
    [North], [$0$], [$+1$], [$x_a = x_b, y_a > y_b$],
    [Northeast], [$+1$], [$+1$], [$x_a > x_b, y_a > y_b$],
    [East], [$+1$], [$0$], [$x_a > x_b, y_a = y_b$],
    [Southeast], [$+1$], [$-1$], [$x_a > x_b, y_a < y_b$],
    [South], [$0$], [$-1$], [$x_a = x_b, y_a < y_b$],
    [Southwest], [$-1$], [$-1$], [$x_a < x_b, y_a < y_b$],
    [West], [$-1$], [$0$], [$x_a < x_b, y_a = y_b$],
    [Northwest], [$-1$], [$+1$], [$x_a < x_b, y_a > y_b$],
  ),
  caption: [Exact compass relations as pairs of axis comparisons for subject
    $a$ relative to reference $b$.],
) <solver-direction-table>

The encoding introduces integer variables $x_a$ and $y_a$ for each location.
Let $"NC"_L$ be the conjunction

$
  "NC"_L = and.big_(a,b in L, a != b) (x_a != x_b or y_a != y_b),
$

and let $E(phi)$ be the integer translation of formula $phi$. The complete
encoding is $"Enc"(L,phi) = "NC"_L and E(phi)$.

A spatial world $W$ and integer assignment $M$ *correspond* when, for every
pair of locations, their X-coordinate differences have the same sign in $W$
and $M$, and likewise for Y. Equivalently, they agree on every `<`, `=`, and
`>` comparison on both axes. An integer assignment satisfying $"NC"_L$ directly
defines a spatial world $W_M$ by setting $p_M(a) = (x_a,y_a)$.

#semantic-correspondence <solver-semantic-correspondence>

The propositional layer uses the standard syntax, model semantics, and
definition of logical entailment presented by Russell and Norvig
@russellNorvig2020aima[Sections 7.3--7.6]. The spatial atoms and their
interpretation are specific to this work.

#heading(level: 2, numbering: none)[Immediate Direction Properties]

Two useful facts follow directly from @solver-direction-table. First,
trichotomy gives one sign from ${-1,0,1}$ on each axis, and no-co-location
removes $(0,0)$. The eight remaining pairs are mutually exclusive and
exhaustive, so exactly one exact direction holds between distinct locations.
Second, reversing an ordered pair negates both signs, giving the opposite
direction:

$
  R_D(a,b) <=> R_("opposite"(D))(b,a).
$

#heading(level: 2, numbering: none)[Lemma 1: Finite Representation Equivalence]

Every finite spatial world has an order-equivalent integer representation using
at most $abs(L)$ ranks on each axis. Conversely, every integer assignment
satisfying $"NC"_L$ denotes a valid spatial world.

_Proof._ For the first direction, sort the distinct X values and replace the
$i$-th value by integer rank $i$; repeat independently for Y. Ranking preserves
and reflects `<`, `=`, and `>`, so it preserves every direction. Two locations
cannot acquire the same rank on both axes unless they were originally
co-located. For the reverse direction, interpret each integer pair $(x_a,y_a)$
as the coordinates of $a$. The constraint $"NC"_L$ prevents co-location, and
integer trichotomy assigns exactly one row of @solver-direction-table to every
ordered pair.

#heading(level: 2, numbering: none)[Lemma 2: Truth Preservation]

For every supported formula $phi$ and corresponding spatial world $W$ and
integer model $M$,

$
  W |= phi <=> M |= E(phi).
$

_Proof._ For an exact spatial atom, $E$ uses the two comparisons in
@solver-direction-table, and correspondence preserves and reflects both. A
coarse atom is a finite disjunction of exact atoms. Structural induction then
extends the equivalence through `NOT`, `AND`, `OR`, implication, and equivalence
because the source formula and its encoding use the same Boolean truth
conditions.

#heading(level: 2, numbering: none)[Theorem 1: Encoding Correctness]

For every supported formula $phi$ over finite location set $L$,

$
  phi " has a spatial model" <=> "Enc"(L,phi) " has an integer model".
$

_Proof._ From left to right, Lemma 1 rank-compresses a spatial model into an
integer assignment satisfying $"NC"_L$, and Lemma 2 preserves the truth of
$phi$. From right to left, Lemma 1 interprets a satisfying integer assignment as
a spatial world, and Lemma 2 transfers the truth of $E(phi)$ back to $phi$.
Thus the encoding introduces no invalid spatial models and omits no valid ones.

#heading(level: 2, numbering: none)[Theorem 2: Query Correctness]

Assume that premises $P$ are consistent. The solver's Direction, Which, and
Count possibility results, and its entailment checks, agree exactly with the
corresponding model-theoretic definitions.

*Direction.* For target $a$, reference $b$, and candidate direction $D$, the
solver includes $D$ exactly when $P and R_D(a,b)$ has a spatial model. By
Theorem 1, this is equivalent to satisfiability of its complete encoding. The
solver checks each declared candidate direction; it supports all eight exact
directions, while the SpatialEval adapter exposes only four ordinal options.

*Which.* Candidate $a$ is possible exactly when $P$ together with the requested
membership relation for $a$ is satisfiable. It is entailed exactly when $P$
together with the negation of that relation is unsatisfiable. Theorem 1 makes
both checks exact with respect to the spatial semantics.

*Count.* Let $cal(M)(P)$ be the set of spatial worlds satisfying $P$, and define

$
  "count"_W(C,D,b) = abs({a in C : W |= R_D(a,b)}).
$

For declared candidate set $C$, the returned count set is

$
  {"count"_W(C,D,b) : W in cal(M)(P)}.
$

The encoding uses one sum of all candidate-membership predicates within the
same model. Checking every integer from zero to $abs(C)$ therefore preserves
dependencies between candidates; it does not add together entities that are
possible only in different worlds.

In particular, for any supported claim $A$,

$
  "Possible"(P,A) <=> "SAT"("Enc"(L,P and A)),
$

and

$
  "Entailed"(P,A) <=> "UNSAT"("Enc"(L,P and not A)).
$

The solver checks $"Enc"(L,P)$ first and reports inconsistent premises
separately rather than using the classical convention that they entail every
formula. Answer modes such as `SINGLE` are deterministic policies over these
exact possibility and entailment results; they are not part of the spatial
encoding theorem.

#heading(level: 2, numbering: none)[Implementation Proposition: Propagation Safety]

Let $"Req"(P)$ contain only positive relation atoms that occur as required
conjuncts of $P$: an atom is required by itself, required atoms are collected
through `AND`, and no atom is extracted through `NOT`, `OR`, implication, or
equivalence. Every model of $P$ therefore satisfies every atom in $"Req"(P)$.

The path-consistency prepass over $"Req"(P)$ cannot remove a direction realised
by a model of $P$. A realised relation from $a$ to $c$ must occur in the
sign-pair composition of the realised relations from $a$ to $b$ and $b$ to $c$.
The implementation generates this composition from the three possible signs on
each axis, including every geometrically realisable result. Repeated
intersection is therefore safe. The solver still asserts the propagated
constraints, $"NC"_L$, and the full Boolean translation $E(P)$; propagation is
only a pruning optimisation and is not relied upon for completeness.

#heading(level: 2, numbering: none)[Implementation Evidence and Trust Boundary]

#assurance-case <solver-assurance-case>

The implementation is tested separately from the mathematical argument. The
current evidence includes bounded coordinate-model enumeration, reference--Z3
differential cases, targeted Boolean and query tests, returned-witness
revalidation, parser round trips, and fail-closed error handling. These tests
support conformance on covered cases but do not formally verify the Python
implementation.

The argument does not cover unrestricted English, omitted parser semantics,
distance, adjacency, betweenness, navigation, quantification, or
three-dimensional geometry. Satisfying coordinate witnesses certify existence;
entailment additionally relies on an unsatisfiability result from the trusted
solver.
