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

#heading(level: 2, numbering: none)[Proof Certificate Calculus]

The SMT encoding decides model-theoretic truth, but a training trace requires a
separate derivation object. This work therefore uses a typed proof certificate.
Each step has a unique identifier, a rule name, an ordered list of dependencies,
and one conclusion. Premise steps additionally identify their source premise.
Derived steps may occur in the global scope or inside one explicitly named case
branch. Natural narrates these checked step identifiers, dependencies and scopes;
Symbolic serializes the typed records. Only the latter has a model-output parser.

Let $Gamma$ denote the visible premise formulas of one `SpatialProblem`. The
judgement $Gamma tack.r phi$ means that formula $phi$ has a checked derivation
from $Gamma$. The propositional meaning of the connectives and the general
notions of sound inference and proof follow Russell and Norvig
@russellNorvig2020aima[Sections 7.4--7.5]. The particular certificate schema and
spatial rules below are defined by this work.

#heading(level: 3, numbering: none)[Boolean rules]

The checker currently admits the following propositional rule schemas.

#figure(
  table(
    columns: (auto, 1fr),
    inset: (x: 6pt, y: 5pt),
    table.header([*Rule*], [*Checked schema*]),
    [Premise], [$phi in Gamma$ permits $Gamma tack.r phi$.],
    [`AND` elimination], [$Gamma tack.r phi and psi$ permits either conjunct.],
    [`AND` introduction], [$Gamma tack.r phi$ and $Gamma tack.r psi$ permit $Gamma tack.r phi and psi$.],
    [Modus ponens], [$Gamma tack.r phi$ and $Gamma tack.r phi arrow.r psi$ permit $Gamma tack.r psi$.],
    [Disjunctive syllogism], [$Gamma tack.r phi or psi$ and $Gamma tack.r not psi$ permit $Gamma tack.r phi$.],
    [`IFF` elimination], [$Gamma tack.r phi <=> psi$ together with either side permits the other.],
    [Double negation], [$Gamma tack.r not not phi$ permits $Gamma tack.r phi$.],
    [Contradiction], [$Gamma tack.r phi$ and $Gamma tack.r not phi$ close the current branch.],
  ),
  caption: [Core non-branching rules in the proof certificate calculus. The
    implementation generalises conjunction and disjunction to finite operand
    lists.],
) <certificate-boolean-rules>

These rules are truth preserving under the ordinary propositional truth
conditions. For example, no model can make $phi$, $phi arrow.r psi$, and
$not psi$ true simultaneously, which establishes Modus Ponens. Likewise, a
model satisfying $phi or psi$ and $not psi$ must satisfy $phi$, which establishes
disjunctive syllogism. The remaining schemas follow immediately from the truth
conditions of conjunction, equivalence, and negation.

#heading(level: 3, numbering: none)[Branches and case analysis]

An assumption step may select one operand of a cited disjunction and must carry
a branch identifier. A step within that branch may depend only on global steps
or earlier steps with the same identifier. It cannot depend on another branch.
If a branch derives both $phi$ and $not phi$, it records a contradiction. Since
the branch has no satisfying model, an explosion step may derive the common case
conclusion inside that closed branch.

For a finite disjunction $or.big_i phi_i$, case split applies the schema

$
  frac(
    Gamma tack.r or.big_i phi_i quad
    (Gamma, phi_i tack.r psi) " for every " i,
    Gamma tack.r psi,
  ).
$

The checker requires one distinct scoped result for every disjunct, verifies
that every branch concludes the same $psi$, and only then returns $psi$ to the
global scope. These conditions prevent assumptions or intermediate results from
leaking between cases. A positive Direction certificate cannot contain a
refutation-assumption step, and its designated conclusion must be global.
Consequently, a branch-local result cannot be presented directly as the answer
or laundered through case analysis using a refutation-only assumption.

#heading(level: 3, numbering: none)[Spatial rules]

An exact direction atom $R_D(a,b)$ decomposes into the X and Y comparisons in
@solver-direction-table. Reversing an axis fact reverses `<` and `>`, while `=`
is symmetric. Axis transitivity admits exactly the deterministic compositions

$
  "=" circle "<" = "<",
  quad
  "<" circle "=" = "<",
  quad
  "<" circle "<" = "<",
  quad
  "=" circle "=" = "=",
$

and their `>` analogues. Opposing strict signs are not accepted as a derived
comparison because their composition is not uniquely determined. Finally, one
checked X fact and one checked Y fact about the same ordered pair may be
recomposed into $R_D(a,b)$ exactly when their sign pair is the row assigned to
$D$ in @solver-direction-table. The checker also permits the equivalent reversed
orientation and negates its comparison sign before recomposition. When an exact
relation atom has already been derived by the Boolean calculus, direction
introduction may lift that atom directly into the query-level `DirectionClaim`;
this avoids a redundant decompose--recompose round trip.

#heading(level: 2, numbering: none)[Theorem 3: Certificate Soundness]

Let $C$ be a Direction proof certificate accepted by the replay checker for
premises $Gamma$ and query pair $(a,b)$. If its final step concludes
$R_D(a,b)$, then

$
  Gamma |= R_D(a,b).
$

_Proof._ Proceed by induction over the checked step order. A premise step is in
$Gamma$ by its validated index. Each Boolean step preserves truth by the schemas
in @certificate-boolean-rules. A scoped assumption is used only within its
branch. If all branches of a cited disjunction derive the same conclusion, every
model satisfying that disjunction satisfies the conclusion in its corresponding
case; a contradictory branch has no model and is therefore vacuous. Branch
isolation prevents any assumption from escaping except through this checked case
rule.

For spatial steps, direction decomposition follows from the definition of
$R_D$; inversion follows from reversing an ordered comparison; and the admitted
axis compositions follow from transitivity of `<` and substitution through `=`.
Direction recomposition is sound because the eight non-zero sign pairs form the
partition in @solver-direction-table. Thus every accepted step is true in every
spatial world satisfying its available premises and branch assumptions. The
global final step is therefore true in every world satisfying $Gamma$.
$square$

The theorem does not require $Gamma$ to be consistent: under classical
semantics an inconsistent premise set has no models. Operationally, the pipeline
checks consistency separately and does not emit an ordinary answer for an
inconsistent problem. Unique Direction training includes a checked model witness
as well as the positive proof, so the emitted evidence establishes non-vacuous
entailment. A model without a proof establishes only possibility.

#heading(level: 3, numbering: none)[Scope of the claim]

The certificate calculus is claimed to be sound, not complete. The automatic
builder saturates conjunction elimination and introduction, modus ponens,
biconditional elimination, double negation, and disjunctive syllogism. It can
also construct finite nested case splits while preventing repeated splits of the
same disjunction on one branch, close contradictory branches by explosion, feed
Boolean-derived atoms into axis transitivity, and refute formulas through
explicit or axis contradictions. It does not claim complete search for every
classically equivalent reformulation. Absence of a certificate is
therefore not evidence of non-entailment; the SMT oracle remains the complete
semantic decision procedure within the declared encoding.

The replay checker does not call Z3. It shares the formula types and direction
sign table with the solver, then independently validates local dependencies and
rule applications. This reduces correlated implementation risk but is not a
formally verified trusted kernel. Theorem 3 establishes the intended calculus;
tests establish only that the Python implementation conforms on covered cases.
The adversarial scope tests include branch-local positive conclusions,
refutation assumptions inside positive proofs, and attempted case-split
discharge of refutation assumptions.

#heading(level: 3, numbering: none)[Corollary 3.1: Refutation Soundness]

Let $C_D$ be an accepted refutation certificate whose scoped assumption is
$R_D(a,b)$ and whose final step is an axis contradiction. Then

$
  Gamma and R_D(a,b) |= bot,
$

so $Gamma |= not R_D(a,b)$. The result follows from Theorem 3's step argument:
the premise-derived axis fact is true in every model of $Gamma$, the assumed
candidate supplies a different exact comparison on the same axis pair, and
trichotomy prevents both comparisons from holding in one world. The checker
requires exactly one refutation-assumption step, requires it to be the declared
$R_D(a,b)$ assumption, and requires that assumption to open a root scope rather
than reuse a selected disjunct's branch. This prevents an unrelated or
case-local assumption from manufacturing the contradiction.

#heading(level: 2, numbering: none)[Constructive Model Certificates]

A proof certificate establishes that a claim follows from the premises. A model
certificate establishes a different fact: that one concrete spatial world
exists. It contains one integer coordinate pair for every problem object, a
formula $A$, and an expected truth value. The checker requires unique object
names, exact object coverage, integer coordinates, no co-location, satisfaction
of the complete premise formula $P$, and the declared truth value of $A$.

#heading(level: 2, numbering: none)[Proposition 4: Model-Certificate Correctness]

If the model checker accepts coordinates $M$ with expected value `true`, then
$M$ denotes a spatial world satisfying $P and A$. If it accepts expected value
`false`, then $M$ denotes a spatial world satisfying $P and not A$.

_Proof._ Exact object coverage and the no-co-location check make the coordinate
assignment a valid spatial interpretation. Relation atoms are evaluated by the
sign pair of the two assigned coordinates. The checker then evaluates `NOT`,
`AND`, `OR`, implication, and equivalence recursively using their ordinary truth
conditions. Acceptance therefore directly witnesses the corresponding
conjunction. $square$

A contingency certificate contains both an accepted supporting model for
$P and A$ and an accepted countermodel for $P and not A$, using the same problem
and claim. It proves that $A$ is possible but not entailed. One or two models do
not prove that an answer set is complete; impossible candidates still require a
closed proof or an independent unsatisfiability decision.

#heading(level: 2, numbering: none)[Complete Direction Answer Sets]

A Direction answer-set certificate enumerates the query's declared candidate
directions in canonical compass order. Each candidate carries exactly one of:

- for the unique entailed direction, an accepted supporting model and an
  accepted positive proof;
- for a contingent direction, an accepted model satisfying $P and R_D(a,b)$;
  or
- an accepted refutation certificate showing that $P and R_D(a,b)$ has no
  spatial model.

Missing candidates, duplicate candidates, evidence for an undeclared direction,
or evidence with the wrong status invalidate the certificate. A singleton
possibility set is accepted only when its candidate carries positive proof
evidence; an ambiguous set cannot mark any exact direction as entailed.

#heading(level: 2, numbering: none)[Proposition 5: Answer-Set Completeness]

If a Direction answer-set certificate is accepted, its entailed, contingent,
and impossible classifications match the model-theoretic status of every
declared direction.

_Proof._ Every witness-backed candidate is possible by Proposition 4. An
entailed candidate additionally has an accepted positive proof by Theorem 3.
Every refutation-backed candidate is impossible by Corollary 3.1. If multiple
exact directions are possible, their mutual exclusivity makes each contingent;
if exactly one is possible, exhaustive exclusion of all alternatives plus its
positive proof makes it the unique entailed direction. $square$

For a unique answer, both training views include the positive derivation and a
consistency witness. Mutual exclusivity rules out the remaining exact directions.
The full certificate's individual refutations are checked before rendering.
Natural narrates the proof dependencies; Symbolic serializes replayable records.
The ambiguous case retains explicit evidence for every candidate.

This is completeness relative to the query's declared candidate set. A dataset
adapter that exposes only four ordinal directions is making a narrower contract
than the generic eight-direction solver; the certificate does not silently add
directions outside that contract.

#heading(level: 2, numbering: none)[Complete Which Membership Sets]

For a Which query with reference object $r$, direction set $S$, and declared
candidate $c$, define the membership claim

$
  M_c = or_(D in S) R_D(c,r).
$

The certificate assigns exactly one of three statuses to every declared
candidate, in query order:

- *entailed*: an accepted model establishes consistency, together with either
  an accepted exact-direction proof whose conclusion lies in $S$, or accepted
  refutations excluding every direction outside $S$;
- *contingent*: one accepted model satisfies $P and M_c$, while another
  satisfies $P and not M_c$; or
- *impossible*: an accepted countermodel establishes consistency, and accepted
  refutations exclude every direction in $S$.

The refutations use a Direction-query projection over the same objects and
premise formula. The Which checker verifies that projection before replaying
each refutation; it does not accept evidence from a different problem.

#heading(level: 2, numbering: none)[Proposition 6: Which Classification Correctness]

If a Which answer-set certificate is accepted, its possible, entailed,
contingent, and impossible entity sets equal the corresponding model-theoretic
classes among the query's declared candidates.

_Proof._ The eight exact directions partition every valid relative position.
For an entailed candidate, an accepted exact-direction proof within $S$
directly establishes $P |= M_c$; alternatively, refuting every direction
outside $S$ establishes the same result. Its accepted witness establishes
consistency. For an impossible candidate, refuting every direction in $S$
establishes $P |= not M_c$; its accepted countermodel establishes consistency.
The two accepted models in a contingency certificate establish both
$P and M_c$ and $P and not M_c$. Exact candidate coverage therefore yields the
stated partition. $square$

This result is deliberately per-candidate. It does not determine a Count answer
by counting individually entailed or possible memberships: two contingent
memberships may be correlated so that exactly one is true in every world.
Count therefore requires certificates for whole joint assignments or count
values rather than a reduction to independent Which statuses.

#heading(level: 2, numbering: none)[Complete Correlated Count Domains]

Let a Count query have ordered candidates $c_1, ..., c_n$ and membership claims
$M_1, ..., M_n$. For a subset $T$ of candidate indices, define its complete
membership assignment as

$
  A_T = (and_(i in T) M_i) and (and_(j in.not T) not M_j).
$

The exact-count formula for $k$ is the disjunction of all assignments whose
member set has size $k$:

$
  C_k = or_(T subset.eq {1, ..., n}, |T| = k) A_T.
$

A Count answer-set certificate covers every $k$ from zero through $n$ exactly
once. A possible count carries an accepted model satisfying $P and C_k$.
Fixed membership certificates are stored once: Proposition 6 establishes each
fixed candidate as entailed or impossible. Assignments contradicting a fixed
classification need no repeated exclusion entry. Each remaining impossible-count
assignment compatible with all fixed classifications requires a checked
refutation. The checker verifies exact coverage of those compatible assignments,
as well as each fixed classification, problem, and claim.

#heading(level: 2, numbering: none)[Proposition 7: Count-Domain Completeness]

If a Count answer-set certificate is accepted, the values marked by model
certificates are exactly the possible counts among $0, ..., n$.

_Proof._ Each model-backed count is possible by Proposition 4. For an
impossible count $k$, every model of $P and C_k$ would induce some assignment
$A_T$ with $|T| = k$. If that assignment contradicts a fixed membership, it is
impossible by Proposition 6. Otherwise it is among the compatible assignments
whose checked refutations establish $P and A_T |= bot$. No model of
$P and C_k$ therefore exists. Exact coverage of all count values leaves precisely
the model-backed values as possible. $square$

This construction preserves correlations. For example, if $M_A equiv not M_B$,
both individual memberships are contingent, but assignments with neither or
both members are refutable and the only possible count is one. No independent
summation of Which statuses can establish that result.

The evidence size remains potentially exponential in the number of contingent
candidates, although fixed classifications no longer appear in every assignment.
Compression therefore does not guarantee that a trace fits a model context.
The automatic builder cannot synthesize every correlated Boolean refutation;
missing evidence is a construction limitation rather than proof of possibility.

Generated proofs retain steps supporting the designated conclusion. The local
checker establishes validity of supplied rule applications; relevance is a
separate construction property. Neither property establishes that a model used
the emitted steps internally. Only Symbolic model output has a replay parser;
Natural prose is not covered by these output-validation claims.

#heading(level: 2, numbering: none)[Implementation Evidence and Trust Boundary]

#assurance-case <solver-assurance-case>

The implementation is tested separately from the mathematical argument. The
current evidence includes bounded coordinate-model enumeration, reference--Z3
differential cases, targeted Boolean and query tests, returned-witness
revalidation, parser round trips, fail-closed error handling, proof-certificate
mutation tests, branch-isolation tests, and corrupted-model rejection tests.
These tests support conformance on covered cases but do not formally verify the
Python implementation.

The argument does not cover unrestricted English, omitted parser semantics,
distance, adjacency, betweenness, navigation, quantification, or
three-dimensional geometry. Satisfying coordinate witnesses certify existence;
entailment additionally relies on an unsatisfiability result from the trusted
solver.
