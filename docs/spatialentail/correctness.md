# Solver Correctness Argument

Status: mathematical argument and executable validation, 21 October 2026.
This is not a proof-assistant verification of the Python implementation or Z3.
It establishes what must hold for the declared qualitative semantics, maps
those obligations to the implementation, and records bounded independent
checks. It does not establish that SpatialEval intended exactly these semantics.

## 1. The Claim We Are Proving

Let `O` be a finite set of `n >= 2` named objects. A world `W` assigns each
object a point `(x, y)` in the plane. Distinct objects occupy distinct points;
they may share one coordinate. Only the signs of coordinate comparisons matter.
There are no metric distances, fixed unit steps, quantifiers, or extra unnamed
objects in the query domain.

An exact direction is a pair of signs:

| Direction | X Comparison | Y Comparison |
|---|---|---|
| North | `=` | `>` |
| Northeast | `>` | `>` |
| East | `>` | `=` |
| Southeast | `>` | `<` |
| South | `=` | `<` |
| Southwest | `<` | `<` |
| West | `<` | `=` |
| Northwest | `<` | `>` |

An atomic constraint `R(A, B, D)` is true when the exact direction of `A`
relative to `B` belongs to the nonempty set `D`. Coarse directions are such
sets, not new inference rules. Formulas are finite combinations of these atoms
with `NOT`, `AND`, `OR`, `IMPLIES`, and `IFF`, interpreted classically.

The desired solver guarantee, conditional on a successful exact SMT decision,
is:

```text
possible(P, S) iff some allowed world satisfies P and S
entailed(P, S) iff P has a model and every allowed P-world satisfies S
```

The extra consistency requirement is the application's convention. Classical
logic entails every formula from inconsistent premises; our interface instead
identifies inconsistency separately.

This separates three obligations:

1. **Semantic fidelity:** these definitions represent the task we intend.
2. **Decision correctness:** the encoding and search preserve these definitions.
3. **Implementation assurance:** the code implements that encoding and search.

Only the second admits the mathematical proof below without an additional
argument about natural-language interpretation. Tests support the third; they
do not universally prove it.

## 2. Lemma: Finite Rank Models Are Complete

Take any allowed real-coordinate world with `n` objects. On each axis, list the
distinct used coordinates in increasing order and replace each coordinate by
its index in that list. Equal coordinates receive the same index.

For every pair of objects, this replacement preserves all three comparisons:
`<`, `=`, and `>`. Therefore it preserves exact directions, arbitrary allowed
direction sets, and distinct positions. Each axis uses at most `n` ranks, all
in `0..n-1`.

By induction on formula structure, every Boolean formula has the same truth
value before and after ranking. Query-region membership is also preserved.
Consequently, the sum of membership indicators, and hence every queried count,
is preserved.

**Conclusion:** every supported satisfiable problem or counterexample has an
integer realization on an `n x n` rank grid. Conversely, every valid rank-grid
assignment is an allowed world. Exhaustively enumerating distinct placements
on that grid therefore covers every qualitative possibility, not merely an
arbitrary bounded coordinate sample.

This lemma covers all order types for a fixed object set. It does not make
testing finitely many formulas a proof for every formula. It also does not
apply after introducing constraints on distances, angles, adjacency, or fixed
coordinate bounds.

## 3. Theorem: The Formula Encoding Preserves Truth

Define `E` recursively. For an exact direction, use its two comparisons from
the table. For a set-valued atom, take the disjunction over its allowed exact
directions:

```text
E(R(A, B, D)) = OR over d in D of Exact_d(A, B)
E(NOT F)     = NOT E(F)
E(AND Fs)    = AND over F in Fs of E(F)
E(OR Fs)     = OR over F in Fs of E(F)
E(F -> G)    = E(F) -> E(G)
E(F IFF G)   = E(F) == E(G)
```

Let `B` be the conjunction of pairwise position-distinctness constraints:

```text
B = AND over distinct unordered A,B of
    (xA != xB OR yA != yB)
```

**Atomic case:** by the direction definition, a world satisfies an exact atom
if and only if its coordinates satisfy the encoded comparisons. Disjunction
gives the same equivalence for any allowed direction set.

**Inductive cases:** assume the equivalence for each immediate subformula.
The Boolean truth tables for negation, conjunction, disjunction, implication,
and equivalence preserve it for the enclosing formula.

Thus, for every allowed integer world `W` and supported formula `F`:

```text
W satisfies F iff W's coordinate assignment satisfies E(F)
```

Combining this theorem with the rank lemma gives both directions of
satisfiability preservation between allowed real worlds and the integer
encoding `B AND E(F)`.

Negation does not require a growing list of spatial inference rules:

```text
NOT Northeast(A,B)
==
NOT (xA > xB AND yA > yB)
==
xA <= xB OR yA <= yB
```

It is not Southwest. Retaining the disjunction preserves the correlation
between the two axes.

Implementation: `_Z3Engine._atom`, `_formula`, and `_build_solver` in
[`solver.py`](../../spatial/v2/solver.py).

## 4. Lemma: Preprocessing Does Not Remove Models

The production solver adds path-consistency constraints before calling Z3.
Even a correct SMT encoding could produce a false entailment if these added
constraints removed a genuine counterexample. Their soundness is therefore a
separate obligation.

### Required Atoms

`_required_atoms` extracts an atom directly, descends through conjunctions,
and extracts nothing through other connectives. Every satisfying world must
satisfy every extracted atom. This follows immediately by induction on those
two extraction cases. In particular, an implication's consequent or one
disjunct is not assumed unconditionally.

### Composition

For one axis, compare `A` with `B` and `B` with `C`:

- equality followed by a comparison preserves that comparison;
- a comparison followed by equality preserves that comparison;
- two comparisons with the same strict sign preserve that sign;
- opposite strict signs permit all three final signs.

These cases follow from equality substitution and strict-order transitivity.
X and Y can be considered independently for three point objects. Their sign
products describe the possible final directions. `(0,0)` is excluded because
the endpoints are distinct.

The implementation unions these products over allowed direction pairs.
If the true direction of each input pair is in its current domain, then the
true endpoint direction is in the composed domain.

### Propagation Invariant

Initially every valid premise-world's directions belong to all current pair
domains. Intersecting with a required atom preserves that property. Inversion
preserves it by swapping comparison signs. Intersecting a pair domain with a
composition preserves it by the composition lemma.

Induction over the sequence of updates proves that no valid premise-world
is discarded. An empty domain therefore establishes inconsistency. Adding
the surviving domain constraints to `E(P)` preserves exactly its models.

There are finitely many pair domains, and each effective update strictly
removes directions, so the loop terminates. Path consistency is used only as
sound pruning; nonempty final domains alone are not accepted as a global
satisfiability proof.

Implementation: `_compose_signs`, `_compose_directions`, `_required_atoms`,
and `_propagate` in [`solver.py`](../../spatial/v2/solver.py).

## 5. Theorem: Possibility and Entailment Decisions

By truth preservation and sound preprocessing:

```text
S is possible under P
iff SAT(B AND E(P) AND E(S))

S follows from consistent P
iff SAT(B AND E(P)) and
    UNSAT(B AND E(P) AND NOT E(S))
```

For entailment, suppose there were a counterexample world satisfying `P`
and falsifying `S`. Ranking and encoding would make it a satisfying assignment
of the second formula, contradicting UNSAT. Conversely, any satisfying
assignment of that formula denotes such a counterexample world.

`assess` checks the claim and its negation. Their satisfiability disjunction
also checks consistency, since every allowed world satisfies either the claim
or its negation.

Z3 `unknown` or a timeout is not UNSAT. `_check` raises on `unknown`;
public analyses carry an error, and grading must check the error before
interpreting the accompanying Boolean fields.

### Reference Search

The reference engine enumerates allowed exact directions for each relevant
unordered object pair. Every real model determines one of those assignments,
so this enumeration cannot omit a model's direction choices. For a general
Boolean formula it evaluates the formula after the mentioned pairs have
been assigned; unmentioned pairs need not be enumerated explicitly.

To realize a selected assignment, it merges equality classes separately on
X and Y and builds strict-order graphs between classes. A strict edge inside
an equality class or a strict cycle is impossible in any numerical order.
If neither occurs, a topological ordering assigns distinct integer ranks to
the classes and realizes every selected comparison. Distinct objects forced
into the same classes on both axes are rejected; otherwise those class ranks
also realize the position-distinctness constraint.

This gives an exact feasibility test for each selected conjunction. Applying
it to partial assignments safely prunes only assignments with no extension.
The full finite search is therefore sound and complete for the supported
formula language, assuming the code implements these operations correctly.
Reference Count search explicitly enumerates joint membership assignments
for each total, preserving the same count semantics as the SMT sum.

### Witness Construction

Incremental query checks use `push`/`pop`, so assumptions for one candidate
do not become premises for the next. Canonical witness construction extends
the current satisfiable constraints one pair at a time with an exact direction
whose addition is checked satisfiable. Position distinctness ensures some
direction exists. This preserves the active query assumption and premises.
Ranking the resulting model preserves them by the rank lemma.

Independent witness validation tests this property without trusting the
canonicalization algorithm. A checked witness still establishes only its own
claim's possibility, not the exclusion of alternative worlds.

### Query Projections

- **Direction:** test satisfiability separately for each candidate exact
  direction. With all eight candidates, mutual exclusion and exhaustiveness
  make a singleton answer equivalent to an invariant exact direction.
- **Which:** test each candidate's region predicate and its negation. A
  possible entity need not be entailed; an empty qualifying region remains
  possible unless ruled out.
- **Count:** use `SUM(If(region(candidate), 1, 0))` and test each integer
  `0..m`, where `m` is the number of query candidates. In each world that sum
  is exactly the count. Joint SMT checks retain correlations; counting the
  individually possible entities would not.

**Restricted-direction caveat:** if the candidate set omits possible
directions, a singleton returned set is only a singleton projection. It does
not prove full-world invariance. For example, a premise allowing NE or NW
with a candidate set containing only NE returns NE, but `assess(NE)` is
not entailed. Either cover the full domain, independently prove that it is
covered, or explicitly condition the premises on an intended menu guarantee.
Merely restricting which queries are checked is not conditioning the world.

Full-domain answering and menu-conditioned answering are different contracts.
See [SpatialMap contract alignment](../archive/research/spatialmap-contract-alignment.md).

## 6. Soundness of Checked Proof Rules

A proof step has a context: all premises plus the assumptions of its enclosing
branches. The invariant is that every world satisfying that context satisfies
the step's conclusion. A contradiction represents a context with no model.

| Rule | Soundness Reason |
|---|---|
| `premise` | The conclusion is an indexed premise, true in every context-world. |
| `assumption` | The chosen disjunct is added to a new child context; it is not asserted globally. |
| `refutation-assumption` | The tested claim is added to a fresh root refutation context. |
| `and-elimination` | Every true conjunction has each conjunct true. |
| `and-introduction` | Truth of every supplied conjunct gives their conjunction. |
| `modus-ponens` | `F` and `F -> G` jointly imply `G` by the truth table. |
| `disjunctive-syllogism` | A true disjunction and a falsified disjunct leave a true remaining disjunction. |
| `iff-elimination` | Equivalent formulas have equal truth values. |
| `double-negation` | Classical `NOT NOT F` has the truth value of `F`. |
| `contradiction` | No context-world satisfies both `F` and `NOT F`. |
| `explosion` | An empty context-world set satisfies any conclusion vacuously. |
| `case-split` | Every parent world satisfies some source disjunct; every corresponding child proves the same conclusion. |
| `direction-decomposition` | A singleton direction entails both defining comparisons. |
| `axis-inversion` | Swapping endpoints reverses `<`/`>` and preserves `=`. |
| `axis-transitivity` | Equality substitution and same-sign strict transitivity preserve the checked comparison. |
| `axis-contradiction` | Two unequal comparison signs on the same oriented axis pair cannot both hold. |
| `direction-introduction` | A singleton spatial atom is the corresponding exact-direction claim. |
| `direction-recomposition` | The checked X/Y comparisons are precisely the claimed exact direction. |
| `spatial-atom-recomposition` | The same comparisons establish the corresponding singleton atom. |

Induction over the ordered proof steps establishes the invariant, provided
the checker enforces each listed rule and context-access restriction. Branch
results cannot enter global scope directly. A case split must cover every
disjunct and return a common conclusion in the parent context.

A positive proof must have a global conclusion and contain no refutation
assumption. A refutation must introduce exactly one root claim assumption and
derive a contradiction in that scope; its conclusion is that `P AND claim`
has no model. To prove `S` by refutation, refute `NOT S`.

Implementation: `_check_step`, `_replay_steps`, `check_direction_proof`,
and `_check_refutation` in [`proofs.py`](../../spatial/v2/proofs.py).
This is a rule-soundness argument, not a mechanized proof of all checker paths.

### Answer-Set Certificates

Coordinate certificates independently evaluate the premises and claim in a
distinct-position integer map. They establish possibility or a counterexample,
never universal entailment on their own.

Which certificates require appropriate proof/refutation/model evidence for
every declared candidate. Count certificates cover every value `0..m`.
For an impossible count, they refute every full membership assignment that
could realize it, after excluding assignments incompatible with separately
checked fixed memberships. Any real model of that count would realize one
remaining assignment, contradicting its refutation. This preserves correlated
membership and explains why evidence can grow exponentially.

These arguments establish validity of checked evidence, not completeness of
automatic evidence construction. The current constructor cannot refute
`NOT NE(A,C)` from `North(A,B)` and `East(B,C)`, although the solver entails
NE and the positive proof constructor can derive it. The validation suite
records this explicit fail-closed boundary.

## 7. Complexity and Trust Boundary

For positive axis comparisons, consistency is an order-graph problem and a
single reachability query takes `O(V+E)`. Full transitive closure or the naive
path-consistency loop need not take linear time.

The Boolean-rich fragment can express arbitrary propositional SAT: represent
each propositional variable by NE on a separate object pair. Either truth
value can be realized independently by placing that pair in its own coordinate
region. Translate the propositional formula using the same connectives.
Thus general search is at least SAT-hard; the rank lemma provides a finite
complete decision procedure but not a polynomial-time one.

Proof checking need not perform that search. It follows an explicit derivation,
although the current Python checker reconstructs scope/dependency information
and is not claimed to run in strictly linear time. A short local-rule proof
is cheap relative to enumerating every possible world, but proof and Count
certificate sizes themselves may be exponential.

The remaining trusted software includes the parser, AST validation, translation,
preprocessing, query projection, proof/model checkers, Python, and the SMT
implementation. Solver-returned UNSAT results are not automatically accompanied
by independently replayed low-level Z3 proofs. Accepted high-level certificates
provide a different assurance route for the supported construction fragment.

Neither tests nor an UNSAT core constitute a formal proof of the implementation.
Further assurance could mechanize the translation and rule lemmas in a proof
assistant or check a complete exported refutation with a verified checker.
That additional work has not been performed here.

## 8. Executable Validation

[`test_solver_correctness.py`](../../spatial/v2/tests/test_solver_correctness.py)
uses a separate coordinate direction function and Boolean evaluator. It does
not use production direction-sign tables, composition, preprocessing, model
evaluation, or the reference backend to calculate expected answers.

It checks:

- all 255 nonempty atomic direction sets and complements against the raw SMT
  encoding on all 12 distinct two-object placements on a 2x2 grid;
- all 64 exact-direction composition pairs against all 504 distinct
  three-object placements on a 3x3 grid;
- both backends against that grid for 40 fixed/seeded Boolean formulas and
  six query configurations, including coarse regions and restricted candidates;
- arbitrary compound claim assessment, consistency, witnesses, and compact ranks;
- required-atom extraction and propagation on the formula corpus and 100
  seeded four-constraint relation networks;
- accepted/rejected axis-transitivity rules, Boolean inference rules,
  independently evaluated proof conclusions, refutations, and a
  missing-construction fail-closed case;
- controlled-text semantic round trips and the model-certificate evaluator;
- injected SMT `unknown` outcomes and the restricted-singleton caveat.

These are exhaustive worlds for the tested object counts, not exhaustive
formulas, arbitrary object counts, all proof trees, or all parser inputs.
The existing mutation/scope/certificate suites add coverage; sharing production
helpers limits the independence of those older suites.

Run the new suite:

```bash
uv --system-certs run --python 3.12 --no-project \
  --with pytest --with z3-solver \
  python -m pytest spatial/v2/tests/test_solver_correctness.py -q
```

Run the broader correctness regression:

```bash
uv --system-certs run --python 3.12 --no-project \
  --with pytest --with z3-solver --with typer \
  python -m pytest \
  spatial/v2/tests/test_solver_correctness.py \
  spatial/v2/tests/test_solver.py \
  spatial/v2/tests/test_proofs.py \
  spatial/v2/tests/test_model_certificates.py \
  spatial/v2/tests/test_count_certificates.py \
  spatial/v2/tests/test_which_certificates.py \
  spatial/v2/tests/test_answer_certificates.py \
  spatial/v2/tests/test_grading.py \
  spatial/v2/tests/test_audit_spatialeval.py -q
```

### Executed Results

On 21 October 2026, with Python 3.12 and the installed `z3-solver`:

| Check | Result |
|---|---|
| New independent suite | 665 passed in 5.92 seconds; no skips |
| Broader correctness regression above | 849 passed in 11.05 seconds; no skips |
| Ruff `E4,E7,E9,F,I` on the new test module | Passed |
| Git whitespace check | Passed |
| Saved SpatialMap-TQA witness revalidation | 2,600 maps checked across all 1,500 original rows |

No solver/oracle decision mismatch or invalid saved witness occurred in these
checks. The new suite contains expected-error checks for unsupported automatic
refutation construction and injected `unknown`; their passing results do not
claim that those operations successfully construct a proof.

The one-off witness revalidation used `validate_world`, `actual_direction`,
and `members` from the new test module, not the production direction evaluator.
It reparsed the original prompts, independently checked every premise and the
realized direction/object/count for each saved witness, checked row-ID coverage
and prompt equality, and verified the original input SHA-256:

```text
53f4cff9d3baef1a1495fbf028f2a5ec80b85de2afb2ed7031f721f271982766
```

Input: `data/spatialeval_org.jsonl`. The prior audit artifacts were at
`/tmp/spatialmap-research-audit-20261021/`; temporary JUnit receipts are
`/tmp/spatialentail-correctness-new-20261021.xml` and
`/tmp/spatialentail-correctness-regression-20261021.xml`. These temporary
receipts are not committed deliverables.

This additional pass validates the saved *possibility witnesses*. It does not
independently prove that no other worlds or answers exist, validate the
natural-language interpretation by itself, or complete the all-family
menu-conditioned evaluation. A passing suite remains implementation evidence,
not formal verification.

## 9. What This Allows Us To Say About SpatialEval

The proven semantic argument is conditional on the declared coordinate/sign
interpretation. SpatialMap-TQA's positive diagonal premises are a sublanguage,
so Boolean negation need not be introduced into their *premises* to audit them.
Negating a candidate conclusion is a method of testing entailment, not a claim
that the benchmark itself supplies negated descriptions.

Before describing a released item as underdetermined, verify complete parsing,
direction meanings, object scope, and the role of its menu. If two independently
checked maps satisfy all those constraints and give different listed answers,
they establish underdetermination under that interpretation without needing to
trust an UNSAT verdict. The gold may still be correct for the intended scene.

The result is not "SpatialEval has adopted our rule system" or "its private
generator is wrong." It is a conditional answerability finding, kept separate
from solver correctness and from the unfinished all-family menu-conditioned
evaluation protocol.
