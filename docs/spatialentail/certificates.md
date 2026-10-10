# Certificate-stack implementation and acceptance contract

This document defines acceptance requirements for the supported certificate
fragment. Test coverage and constructor support are implementation evidence;
they do not establish unrestricted proof-search completeness, SFT readiness,
learning benefit, human readability, or internal reasoning faithfulness.

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
- Positive proofs reject refutation assumptions and require a global conclusion.
  Refutations admit exactly one declared claim assumption, opened at a root
  scope; additional or case-local refutation assumptions are invalid.
- Which: entailed, contingent, and impossible evidence for every candidate.
- Count: model evidence for possible counts; fixed membership classifications
  stored once, plus exhaustive exclusions for assignments compatible with those
  classifications for impossible counts. Correlations remain explicit and
  evidence can still grow exponentially in contingent candidates.
- Natural narrates checked step IDs, inputs, branch scopes and query evidence;
  Symbolic encodes those objects in typed records. Corpus-wide evidence mapping
  and length must be audited before a notation-only claim. Symbolic model output
  is parsed and replayed; arbitrary
  Natural model output has no process checker.
- Unique Direction training includes a consistency witness alongside its proof.
- Generated proofs are pruned to relevant steps; proof validity and relevance
  remain distinct acceptance properties.
- Training model evidence uses qualitative ranks; every audit query kind
  exposes coordinate-bearing witnesses.
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
- direct exact-direction introduction, direction decomposition, inversion, axis
  transitivity, and recomposition;
- explicit Boolean and spatial formula refutation;
- correlated Count assignment refutation.

The constructor does not silently add classical rewrites such as contraposition
or De Morgan conversion when those steps are absent from the checked calculus.
Such formulas remain semantically decidable by the solver but outside automatic
proof-search completeness until the corresponding typed rules are added.

## Required curriculum and validation gates

The following are gates to verify, not a declaration that the current branch
has passed them. The remediation ledger records executed checks and remaining
limits separately.

- Verify selection of atomic and six Boolean proof shapes through policies,
  workload CLI, and matrices, including fail-closed unsupported combinations.
- Exercise each claimed Boolean/query-family combination and all claimed target
  directions; do not infer coverage from enum availability.
- Check that proof-template curricula choose their obligations before witness
  construction, while premise-first proposals do not consult coordinates.
- Require rendered prompts to parse back into the original structured formulas.
- Run spatial tests, scoped lint/format checks, report compilation, and actual
  multi-shape workload smoke checks with the selected tokenizer. Token admission
  must pass for the exact configuration being proposed for training.

The current verification record is maintained in
[the remediation audit](../audits/critique-remediation.md).
Meeting these requirements validates the tested implementation boundaries.
Premise-first proposals are now a distinct generation route, not a synonym for
certificate-first construction. Training, structural generalisation, readability,
and causal explanation claims still require empirical studies.
