#pagebreak(weak: true)

= How the Spatial Solver Works

== What the Solver Decides

The solver answers a question from the stated spatial information alone. It
does not guess the generator's original coordinates or choose one convenient
map. Instead, it considers every map that satisfies the premises and determines
which answers remain possible across those maps.

The solver handles three query families:

#figure(
  table(
    columns: (auto, 1fr),
    table.header([*Query*], [*Question answered*]),
    [Direction], [Where is one location relative to another?],
    [Which], [Which candidate locations lie in a requested direction?],
    [Count], [How many candidate locations lie in a requested direction?],
  ),
  caption: [The three query families supported by the solver.],
) <solver-query-families>

The solver itself receives structured locations, premises, and a query. Dataset
wording, answer menus, oracle labels, and grading rules are handled by adapters
outside the solver. This keeps the reasoning engine independent of SpatialEval
or any future generated dataset.

== Directions as Two Comparisons

A compass direction describes one horizontal relationship and one vertical
relationship. For a subject A relative to a reference B:

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    table.header([*Direction*], [*Horizontal fact*], [*Vertical fact*]),
    [Northeast], [`A is right of B`], [`A is above B`],
    [Northwest], [`A is left of B`], [`A is above B`],
    [Southeast], [`A is right of B`], [`A is below B`],
    [Southwest], [`A is left of B`], [`A is below B`],
  ),
  caption: [The four ordinal directions used by SpatialEval, decomposed into
    axis comparisons. The complete eight-direction ontology appears in
    Appendix A.],
) <solver-axis-examples>

The solver records these comparisons as simple constraints such as
`x_A > x_B`, `x_A = x_B`, or `y_A < y_B`. Only relative order matters. Moving
the whole map, stretching its distances, or replacing its coordinates with
small integer ranks does not change any compass relationship.

This decomposition also explains ambiguity in the released SpatialEval data.
The premises of item `spatialmap.tqa.2003.0` determine that Recycle Center is
above Nightingale Novelties, but do not determine whether it is to the left or
right. The two remaining menu candidates are therefore Northwest and Northeast;
the solver must not arbitrarily select one of them.

== Possible, Entailed, and Undetermined

The solver uses an open-world interpretation: an unstated relationship remains
unknown rather than becoming false. It distinguishes three cases.

#figure(
  table(
    columns: (auto, 1fr),
    table.header([*Status*], [*Meaning*]),
    [Possible], [At least one map satisfying the premises gives this answer.],
    [Entailed], [Every map satisfying the premises gives this answer.],
    [Possible but not guaranteed], [Some valid maps give this answer and others do not.],
  ),
  caption: [Answer status under the solver's open-world semantics.],
) <solver-answer-status>

These meanings follow the standard definitions of possibility and entailment
in propositional logic @russellNorvig2020aima[Sections 7.3--7.6]. In practical
terms, the solver searches for a valid map supporting a candidate answer and
then searches for a counterexample in which that answer is false. Finding a
supporting map proves possibility; proving that no counterexample exists proves
entailment.

== SAT and UNSAT on a Released SpatialEval Item

The distinction can be demonstrated directly on `spatialmap.tqa.2003.0`, whose
nine premises and full coordinate witnesses appear in
@spatialeval-counterexample-premises and
@spatialeval-counterexample-coordinates. Let $P$ denote the conjunction of
those nine premises, $R$ Recycle Center, and $N$ Nightingale Novelties. The
released question offers exactly four candidates: Southeast, Southwest,
Northeast, and Northwest.

For each candidate $D$, the solver checks whether $P and R_D(R,N)$ is
satisfiable. `SAT` means that at least one assignment of coordinates satisfies
all nine premises and the candidate simultaneously. `UNSAT` means that no such
assignment exists.

#figure(
  table(
    columns: (auto, 1fr, auto, 1.35fr),
    table.header([*Option*], [*Formula checked*], [*Result*], [*Consequence*]),
    [A. Southeast], [$P and "SE"(R,N)$], [UNSAT], [Impossible],
    [B. Southwest], [$P and "SW"(R,N)$], [UNSAT], [Impossible],
    [C. Northeast], [$P and "NE"(R,N)$], [SAT], [Possible, not entailed],
    [D. Northwest], [$P and "NW"(R,N)$], [SAT], [Possible, not entailed],
  ),
  caption: [Exact candidate checks for the original SpatialEval item
    `spatialmap.tqa.2003.0`. The published oracle selects only C.],
) <solver-spatialeval-sat-checks>

The Northeast check is `SAT` because the solver finds a complete model with
$R=(1,5)$ and $N=(0,0)$. The Northwest check is independently `SAT` because it
finds another complete model with $R=(0,5)$ and $N=(1,0)$. In the first model,
$1>0$ and $5>0$, so R is northeast of N. In the second, $0<1$ and $5>0$, so R
is northwest of N. The remaining coordinates in
@spatialeval-counterexample-coordinates verify all nine premises in both
models. These are existence certificates, not reconstructions of the hidden
generator map.

The two southern candidates are `UNSAT` for a concrete reason. Let $S$ denote
Sally's Salon. Two released premises state that N is southwest of S and S is
southeast of R. Their vertical components give

$
  y_N < y_S quad "and" quad y_S < y_R,
$

and hence $y_N < y_R$. Both Southeast$(R,N)$ and Southwest$(R,N)$ instead
require $y_R < y_N$. Adding either candidate to $P$ therefore demands
$y_R > y_N$ and $y_R < y_N$ simultaneously, so no coordinate model exists.

Possibility is not the same as entailment. To test whether Northeast is forced,
the solver searches for a counterexample by checking $P and not "NE"(R,N)$.
That formula is `SAT`: the Northwest witness is a model of it. Conversely,
$P and not "NW"(R,N)$ is `SAT` because the Northeast witness is a model of it.
Thus each answer has a supporting model and each has a counterexample. Neither
is uniquely entailed by the released text.

Both the Z3 engine and the independent exhaustive reference engine return the
same possibility set, ${"Northeast", "Northwest"}$. The original oracle is not
used in any satisfiability check; it is compared with this result only after
solving. Under `SINGLE`, the row is therefore underdetermined. Under
`ALL_POSSIBLE`, the semantic answer is the pair of options ${C,D}$.

Under the single-answer style (`SINGLE`), exactly one answer must be entailed.
If several answers remain possible, the result is _Cannot be determined_. The
complete-set style (`ALL_POSSIBLE`) requests every possible answer. The
visible-options style (`VISIBLE_POSSIBLE`) reports only possible answers shown
in a particular menu and is used as a diagnostic. These policies operate above
the spatial solver; the menu never determines what is spatially possible.

#pagebreak(weak: true)

== From a Problem to an Answer

The solving process has five steps:

+ *Receive a structured problem.* An adapter supplies the locations, premises,
  and query without exposing dataset-specific text or answer letters to the
  solver.
+ *Split spatial relations by axis.* Each direction becomes one X comparison
  and one Y comparison.
+ *Run a fast first pass.* The solver narrows direction possibilities using
  straightforward consequences of facts that must be true.
+ *Check the remaining possibilities exactly.* The full problem is translated
  into integer constraints. An SMT solver tests the premises together with each
  candidate answer. A separate exhaustive backend is retained for small
  differential checks.
+ *Return answers and evidence.* The solver reports possible and entailed
  answers and can return a coordinate witness for each possible answer. These
  coordinates certify that a map exists; they are not training rationales.

The input language can combine spatial statements using `NOT`, `AND`, `OR`,
implication, and equivalence. The SpatialEval adapter uses only the smaller
subset that appears in its released text: positive conjunctions of ordinal
direction statements.

== Why the Result Is Trustworthy

The correctness argument rests on three ideas:

+ *No spatial arrangements are lost.* Any finite map can be replaced by integer
  coordinate ranks while preserving all left/right, above/below, and equality
  relationships.
+ *The translation does not change meaning.* Each compass direction becomes
  exactly its horizontal and vertical conditions, and the ordinary Boolean
  connectives retain their standard meaning.
+ *Every candidate is checked against the whole problem.* A supporting map
  establishes possibility, while the absence of any counterexample establishes
  entailment.

Appendix A states this argument formally using two lemmas and two theorems. It
also proves that the propagation step is a safe optimisation. That mathematical
argument concerns the encoding. The Python implementation is checked
separately through small-world enumeration, comparisons with an independent
exhaustive backend, witness revalidation, targeted Boolean and query tests,
parser round trips, and checks that invalid inputs stop rather than produce an
answer. Passing those tests increases confidence that the implementation
follows the method, but it does not formally verify the Python program or the
underlying SMT implementation.

== Scope

The solver is complete only within its declared language and spatial ontology.
It does not model distance, adjacency, betweenness, nearest-object relations,
unrestricted quantification, navigation, or three-dimensional geometry.
Unsupported syntax, inconsistent premises, timeouts, and unparsed text are
reported explicitly rather than silently guessed.
