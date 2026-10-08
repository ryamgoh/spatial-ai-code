# Proof-first completion contract

The SpatialEntail proof stack is complete when every requirement below passes.
This contract defines 100% for the project; it does not claim unrestricted
first-order theorem proving.

## Supported language

- Finite named objects.
- Eight exact compass directions and the declared coarse direction sets.
- Ground atoms of the form `X is DIR of Y`.
- Explicit `NOT`, `AND`, `OR`, `IF ... THEN`, and `IFF` composition.
- Direction, Which, and correlated Count queries over declared candidates.
- No quantifiers, metric distance, adjacency, betweenness, or free-form spatial
  paraphrase interpretation.

## Semantic decision

- The reference and Z3 engines implement the same `SpatialProblem` contract.
- Possibility is satisfiability of premises and claim.
- Entailment is unsatisfiability of premises and claim negation.
- Models and countermodels are revalidated independently of Z3.

## Checked evidence

- Direction: positive proof plus witness when entailed, model when contingent,
  refutation when impossible.
- Which: entailed, contingent, and impossible evidence for every candidate.
- Count: model or exhaustive joint-assignment exclusion for every count.
- Natural and Symbolic traces are deterministic renderings of one accepted
  certificate.
- Unsupported proof construction fails closed; there is no legacy renderer
  fallback.

## Automatic construction fragment

The constructor automatically handles:

- conjunction elimination and goal-directed conjunction introduction;
- modus ponens;
- biconditional elimination;
- double-negation elimination;
- disjunctive syllogism;
- finite nested case splits without re-splitting the same disjunction on one
  branch;
- contradiction and explosion;
- direction decomposition, inversion, axis transitivity and recomposition;
- explicit Boolean and spatial formula refutation;
- correlated Count assignment refutation.

The constructor does not silently add classical rewrites such as contraposition
or De Morgan conversion when those steps are absent from the checked calculus.
Such formulas remain semantically decidable by the solver but outside automatic
proof-search completeness until the corresponding typed rules are added.

## Curriculum and validation gates

- Atomic and six Boolean proof shapes are selectable in policies, workload CLI,
  and matrices.
- Every Boolean shape works for Direction, Which, and Count.
- Direction shapes cover all eight target directions.
- Boolean curricula construct proof obligations before requesting any coordinate
  witness.
- Rendered prompts parse back into the original structured formulas.
- Full spatial tests, V2 lint/format, report compilation, and a real multi-shape
  workload smoke test pass.

Meeting these requirements is 100% completion within the declared fragment.
Any later addition expands the research scope rather than completing missing
infrastructure.
