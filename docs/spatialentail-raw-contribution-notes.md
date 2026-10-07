# Raw thoughts: what this project is trying to do

Status: deliberately raw working notes, not contribution claims ready for the
report or abstract.

Primary-source annotations:
[`What each source supports and does not support`](spatialentail-annotated-sources.md)

## The central discomfort

The project started from a simple discomfort: spatial-reasoning benchmarks can
look formally generated and therefore trustworthy while still asking the model
to recover information that was never shown to it.

A generator knows a complete world. It renders some facts from that world into
text. It then reads an answer from the complete world and calls that answer the
label. But truth in the generator's chosen world is not the same as entailment
from the visible text. If another world satisfies every visible premise and
gives another answer, the question is not a valid single-answer question under
the stated information.

That is not merely annotation noise. It changes what capability the benchmark
measures. A model may be marked wrong for refusing to guess hidden construction
state, while another model may be rewarded for exploiting generator regularity
or selecting a plausible completion.

The core idea is therefore:

> Spatial reasoning should be evaluated and supervised against the complete
> semantics of the information given to the model, not against one privileged
> world that the model never saw.

Everything else in the project follows from taking that sentence seriously.

## Part I: use formal semantics to interrogate an existing benchmark

SpatialEval is not only a leaderboard target. It is a case study in benchmark
validity.

For each released Spatial-Map TQA problem, we treat the text as a set of
constraints and ask:

```text
Which answers are possible in at least one world satisfying the text?
Which answers hold in every world satisfying the text?
Is the published answer uniquely entailed, merely possible, contradicted, or
attached to an inconsistent or unparseable problem?
```

This produces a sharper distinction than ordinary exact-match evaluation:

```text
truth in one sampled map
!=
entailment from the released description
```

The intended contribution is not “we found mistakes in a dataset.” The deeper
contribution is a reproducible way to state an answer contract, audit a
generated benchmark against that contract, retain witnesses and countermodels,
and publish a corrected evaluation view without overwriting the original
artifact.

The audit should remain modest about authorial intent. It shows what follows
under our declared open-world semantics. It does not prove that the SpatialEval
authors intended exactly the same semantics.

## Part II: do not reproduce the same problem in our own benchmark

Once we criticise latent-world labels, we cannot quietly generate SpatialEntail
from complete maps and call the hidden-map relation the answer.

SpatialEntail is supposed to explore a different data contract:

- the visible premises are the authoritative problem;
- possible and entailed answers are computed over all satisfying worlds;
- ambiguity is a semantic outcome, not a generation failure;
- every gold answer has formal evidence;
- every reasoning trace is derived from a structured certificate rather than
  written after seeing the answer; and
- the final emitted text is reparsed and re-solved so that surface rendering
  cannot silently change the problem.

The benchmark is not just “more spatial questions.” It is an experiment in how
the provenance of a reasoning problem affects label validity, proof fidelity,
and generalisation.

## Three generation provenances

### World-first

```text
sample complete world
-> render some premises
-> read answer from the same world
```

This is useful for generating coherent scenes, but its hidden answer proves only
possibility. If used as the gold label without recomputing all models, it
recreates the exact validity problem we are studying.

Coordinates are not inherently bad. A coordinate model returned after solving
is a valid witness that an answer is possible. What is problematic is treating
the coordinates as privileged truth when the model receives only a partial
description.

### Proof-first

```text
sample a typed derivation
-> instantiate spatial and Boolean atoms
-> derive the visible premises, query, conclusion, and certificate
-> replay the certificate
-> independently solve the emitted problem
```

Proof-first generation gives us controlled, checkable supervision. We can vary
proof depth, branching, rule composition, independent X/Y support, negation,
distractors, and ambiguity deliberately.

Its danger is template learning. A model may learn the fingerprints of our proof
constructor rather than the underlying reasoning. A generated proof may also be
valid but not minimal because some accidental shorter route exists.

### Premise-first

```text
sample visible premises and query without choosing an answer
-> solve every candidate
-> discover the semantic class
-> search for proofs and countermodels
-> bucket the item by what was actually found
```

Premise-first construction is less answer-conditioned and can produce
unanticipated logical interactions. It is therefore attractive for evaluation
and structural generalisation. Its costs are rejection rate, class imbalance,
uncontrolled difficulty, and hard proof extraction.

The current bet is a hybrid:

```text
proof-first for controlled trace supervision
premise-first for the main generalisation test
premise-first training only when a concise proof can be extracted and replayed
world-first answers never accepted as gold without semantic recomputation
```

## The solver is necessary, but it is not the whole contribution

Z3 gives an exact semantic oracle for the supported ontology:

```text
Possible(P, A) iff SAT(P AND A)
Entailed(P, A) iff UNSAT(P AND NOT A)
```

This is essential, but “we used Z3” is not a sufficient research contribution.
Z3 can tell us that a formula is satisfiable or unsatisfiable. It does not
automatically give the kind of domain-level derivation we want to place in an
LLM training trace.

The proposed trust structure is deliberately redundant:

```text
SpatialProblem
  |-- SMT oracle: decides complete model-theoretic semantics
  |-- proof engine: constructs a typed derivation or branch certificate
  |-- proof checker: replays every local rule application
  |-- witness checker: validates constructive models and countermodels
  `-- text round trip: reparses and re-solves the rendered prompt
```

The proof engine and SMT oracle may share the ontology and formula AST, but they
should not share their core reasoning implementation if their agreement is used
as evidence. Otherwise one encoding bug can validate itself twice.

## What a proof-carrying spatial example should mean

A proof-carrying example should contain more than a correct final answer and a
plausible explanation. Each step should identify:

- the premises or earlier steps it depends on;
- the rule being applied;
- the exact conclusion produced;
- any branch assumption;
- whether the branch remains satisfiable or closes by contradiction; and
- the final answer semantics supported by the complete proof object.

For example:

```text
P1: NE(A,B) OR NW(A,B)
P2: NOT NW(A,B)
S1: NE(A,B)                   [disjunctive syllogism, P1, P2]
Answer: Northeast
```

Or, for a spatial contradiction:

```text
P1 gives y_N < y_S
P2 gives y_S < y_R
S1 gives y_N < y_R           [strict transitivity, P1, P2]
Assume y_R < y_N
S1 and the assumption form a strict cycle
Therefore this candidate is impossible
```

Natural CoT and Symbolic CoS should be two views of this same object. They should
not be generated independently, because independently authored traces can
silently disagree while ending at the same answer.

## The full SpatialEntail language is intentionally richer than SpatialEval

SpatialEval's released map premises use a narrow conjunction of ordinal
directions. SpatialEntail is meant to cover all eight directions and finite
propositional combinations over spatial atoms:

```text
NOT
AND
OR
implication
equivalence
```

This changes the proof problem. Two axis-order graphs are sufficient for the
narrow strict-conjunction fragment, but not for arbitrary Boolean structure.
The proof system needs a Boolean layer for branching and a spatial layer for
direction domains, axis equality and order, composition, witnesses, and
contradictions.

The project should resist expanding into unrestricted natural-language theorem
proving. The formal language is finite and declared. The scientific value comes
from controlling and studying the interaction of spatial and propositional
reasoning, not from pretending to solve every form of logic expressed in
English.

## What we want to learn about LLMs

The benchmark and solver are infrastructure for empirical questions:

1. Does correcting answer semantics materially change conclusions about LLM
   spatial capability?
2. Does proof-backed supervision improve more than answer-only exposure?
3. Do gains survive premise-first problems and held-out proof compositions, or
   only matched generation templates?
4. Is a compact Symbolic trace more useful than a Natural trace when both encode
   exactly the same derivation?
5. Are Delta traces better than Full state serialization because they preserve
   the inferential step without repeatedly copying the entire world state?
6. Can models distinguish possible, entailed, impossible, contingent, and
   inconsistent conclusions?
7. Does solver-verifiable RL add anything after strong proof-backed SFT, or does
   it mainly optimise answer formatting and generator-specific shortcuts?

The important comparison is not simply which model gets the highest score. It
is which supervision and representation produce structural generalisation under
a valid answer contract.

## What may actually be novel

None of the individual ingredients should be claimed as new:

- text-only spatial reasoning benchmarks already exist;
- solver-derived spatial paths already exist;
- Natural and Symbolic spatial reasoning traces already exist;
- proof-first logical data already exist;
- proof-carrying reasoning already exists;
- SMT-backed and neuro-symbolic LLM reasoning already exist; and
- pure-text spatial SFT and GRPO already exist.

The plausible contribution is the integration:

> proof-carrying qualitative spatial entailment under partial information, with
> proof-first supervision, premise-first generalisation evaluation, independent
> local proof replay and global SMT semantics, and paired Natural/Symbolic views
> of one derivation.

That remains a “to our knowledge” claim and must be rechecked before submission.
The honest framing is an underexplored intersection of existing research lines,
not the invention of a new field from nothing.

## A possible contribution stack

### 1. Semantic contribution

A precise open-world contract for possible, entailed, contingent, impossible,
and inconsistent spatial answers over a declared eight-direction ontology and
finite propositional language.

### 2. Evaluation contribution

A solver-backed audit of SpatialEval Spatial-Map TQA that separates latent-world
truth from textual entailment, preserves the original artifact, and publishes a
distinct corrected evaluation view with witnesses and countermodels.

### 3. Data-methodology contribution

An explicit account of world-first, proof-first, and premise-first generation,
with construction provenance stored in the manifest and structural splits that
hold out proof compositions rather than merely random seeds or entity names.

### 4. Verification contribution

A typed spatial proof certificate, small replay checker, separately implemented
SMT oracle, parser round trip, and differential tests across reasoning backends.

### 5. Representation contribution

Matched Natural and Symbolic renderings of the same certificate, crossed with
Final-only, Delta, and Full state schedules without changing the underlying
problem or answer.

### 6. Empirical contribution

A controlled comparison of prompting, answer-only SFT, proof-trace SFT,
structural generalisation, external corrected-SpatialEval transfer, and optional
verifiable RL.

The thesis need not succeed equally at all six levels. The minimum defensible
contribution is the semantic contract, validation, audit, and a controlled pilot
of proof-backed supervision. A full proof engine and every training method are
not prerequisites for a meaningful result.

## What would make the project weak

The project becomes much weaker if it reduces to any of the following:

- another synthetic spatial benchmark with a new name;
- a thin Z3 wrapper with no independent explanation or verification artifact;
- proof templates evaluated only on other examples from the same templates;
- post-hoc rationales that are accepted because the final answer is correct;
- an ambiguity correction presented as if it recovers the original authors'
  private intent;
- a large experiment matrix without one decisive causal comparison;
- a “first” claim contradicted by SpaRP, PrOntoQA, Chain-of-Symbol,
  proof-carrying reasoning, or recent Z3-monitored SpatialMap work; or
- a claim of general spatial intelligence from one finite qualitative calculus.

The project should be willing to report that trace supervision does not improve
premise-first generalisation. A controlled negative result would be more useful
than a positive result supported only by matched generator templates.

## What success would look like

At the strongest level:

1. The formal semantics and solver survive independent backend, witness,
   countermodel, parser, and manual checks.
2. The SpatialEval audit changes the interpretation of a meaningful subset of
   released questions and produces a reproducible corrected view.
3. SpatialEntail examples carry replayable certificates whose Natural and
   Symbolic forms agree exactly.
4. Models trained with proof-backed traces outperform answer-only controls on
   premise-first and held-out-composition tests, not merely matched templates.
5. The analysis identifies which representation and state schedule helps, where
   it fails, and whether any gain transfers to corrected SpatialEval.

At the minimum successful level:

1. We expose and formalise the benchmark-validity problem clearly.
2. We publish a sound audit and correction methodology.
3. We build a small but trustworthy proof-carrying SpatialEntail pilot.
4. We run a fair answer-only versus trace-supervision comparison.
5. We report the result without overstating its scope.

## Questions still worth arguing about

- Should coordinate witnesses be a separate trace arm, audit-only evidence, or
  both?
- How much proof search is necessary if training data are proof-first?
- What is the smallest proof-rule vocabulary that still exercises meaningful
  Boolean-spatial composition?
- How should proof minimality be defined when several derivations are valid?
- Should premise-first items without concise proofs remain answer-only evaluation
  cases?
- How independent must the proof checker and SMT oracle be before agreement is
  meaningful evidence?
- Which proof structures should be completely held out from training?
- Is `SINGLE` enough for the core benchmark, or is `ALL_POSSIBLE` necessary to
  test set-valued reasoning directly?
- Is the intended contribution still coherent if Natural and Symbolic traces do
  not outperform answer-only SFT?
- What is the one decisive experiment if compute or time forces the project to
  shrink?

## One-sentence version

> Build spatial-reasoning evaluation and supervision around what the visible
> premises actually entail, attach checkable evidence to every generated answer,
> and test whether learning those proofs transfers beyond the templates that
> created them.
