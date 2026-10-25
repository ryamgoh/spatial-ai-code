# Generation provenance as a SpatialEntail research variable

Status: research framing note, 2026-10-11. The names below are operational
definitions introduced for this study, not a claim that the field already uses
one standard taxonomy.

## Conclusion

Generation provenance is a coherent research variable because different
construction regimes place different objects upstream of the final visible
problem. They create different risks even when every emitted item is later
solved correctly. However, provenance alone is not a sufficient novelty claim,
and an unmatched comparison of generated datasets would confound provenance
with their visible distributions.

The original three-way vocabulary should be refined into four regimes:

```text
latent-world-first:   W -> P,Q -> solver -> A,C
proof-template-first: T -> P,Q -> solver -> A,C
certificate-first:    C -> P,Q,A -> independent validation
visible-problem-first:P,Q -> solver -> A,C -> measured bucket
```

Here `W` is a latent complete world, `T` an intended rule or proof template,
`P` the visible premises, `Q` the visible query, `A` the complete semantic
answer, and `C` a checked certificate.

The present SpatialEntail implementation supports latent-world-first,
proof-template-first, and visible-problem-first construction. It does not yet
implement certificate-first construction. Calling the current controlled
generator simply “proof-first” is understandable shorthand, but
“proof-template-first” is more accurate: it plants a support structure and then
asks the solver and certificate engine what proof actually holds.

## The four regimes

### 1. Latent-world-first

Construct a complete coordinate world, render only some relations as premises,
and create a query from the world.

There are two materially different label routes:

1. **Privileged-world labeling:** read the answer from the complete world.
   This is not valid for an entailment benchmark unless the visible premises
   uniquely entail the same answer. A single world shows possibility, not
   necessity.
2. **Visible-problem re-solving:** discard the world answer and solve the
   emitted premises and query. This produces a semantically valid label, but the
   latent world still affects which problems were proposed and admitted.

World-first is therefore not inherently wrong. The invalid step is treating a
privileged-world relation as though it were entailed by incomplete text.
Re-solving fixes correctness but does not remove world-induced selection bias.

### 2. Proof-template-first

Choose an intended rule family, support graph, path shape, or semantic pattern;
instantiate visible premises and a query; then solve, measure, and certify the
result.

This is useful for producing controlled coverage of rare structures such as
case splits, correlated counts, or exact two-way ambiguity. It does not prove
that the planted template is the shortest, unique, or actual explanation. An
unintended shorter derivation may exist, so the admitted item must be measured
after construction.

The v3 `DIR-ARC-ALT` constructor is an example. It deliberately fixes equality
on one axis while leaving the other axis unordered, then relies on the solver
to confirm that the visible problem has exactly the North/South or East/West
possibility set. It does not derive premises from an answer-bearing map.

### 3. Certificate-first

Construct the final replayable derivation or refutation object first, then
derive the premises, query, and answer from it. An independent semantic checker
must still validate the resulting visible problem.

This is stricter than template-first. A template says what kind of reasoning is
intended; a certificate identifies concrete steps, dependencies, and branch
scope. Certificate-first is attractive for exact construction and validation,
but it can make examples unusually clean and can leak generator-specific proof
shape. It is currently proposed vocabulary in SpatialEntail, not implemented
code or a planned certificate-supervision arm.

### 4. Visible-problem-first

Sample the visible premises and query without choosing a target answer. Solve
every candidate under the declared semantics, construct any available
certificate, measure the resulting structure, and only then place the item in a
semantic or difficulty bucket.

This is the strongest stress test of whether a learned reasoner can handle
problems whose proof structure was not planted for it. The implementation name
`premise-first` is slightly incomplete because the query must also be sampled
before labeling. “Visible-problem-first” or “premise-and-query-first” is clearer
in the report.

Visible-problem-first is not bias-free. Rejection for inconsistency,
uninteresting answer sets, missing certificates, or excessive length changes
the admitted distribution. Those rejection rates and reasons must be reported.

## Why the distinction is meaningful

The regimes differ in causal direction and therefore in characteristic failure
modes:

| Regime | Upstream commitment | Main strength | Main validity threat |
|---|---|---|---|
| Latent-world-first | Complete world | Easy coherent scene construction | Hidden-world labels and world-induced selection bias |
| Proof-template-first | Intended support/rule shape | Controlled coverage of rare reasoning | Template leakage and accidental shorter proofs |
| Certificate-first | Concrete derivation | Exact construction and validation | Over-clean proof distribution and certificate artifacts |
| Visible-problem-first | Visible premises and query | Unplanted semantic stress testing | Post-selection and certificate-availability bias |

The distinction may disappear observationally after aggressive balancing. Two
rows generated by different routes can contain the same visible formula and
query. Provenance is therefore metadata about the data-generating process, not
another semantic task label.

A causal claim about provenance requires either paired construction or matching
on the final visible distribution. Otherwise a performance gap might be caused
by different answer balances, rule families, entity counts, proof depths,
branching, distractors, or language templates.

## Primary-source precedents

The ingredients have substantial precedent. The contribution cannot be “the
first proof-first dataset” or “the first solver-generated reasoning data.”

### RuleTaker: theory first, then closure and query selection

RuleTaker generates a formal theory, exhaustively forward-chains its
implications, selects questions based on the resulting closure, and verbalizes
the theory and questions. Its paper names the stages Theory Generation, Forward
Inference, and Question Generation and English Synthesis. This is best
described as theory-first, solve-and-select generation—not proof-first.

- Clark, Tafjord, and Richardson, “Transformers as Soft Reasoners over
  Language,” IJCAI 2020, [paper sections 3.1–3.4](https://arxiv.org/html/2002.05867v2#S3).
- [Official RuleTaker generator](https://github.com/allenai/ruletaker/blob/abaacec9364992eff5ec4555b837e20fee2f2ff0/theory_generator.py#L205-L398).

### ProofWriter: theory-first data with explicit proofs

ProofWriter retains theory-first dataset construction while adding proof DAGs,
open-world `Unknown`, complete proof generation, iterative one-step reasoning,
and restricted abduction. Its model may generate a complete proof, but that is
different from constructing the underlying problem certificate-first.

- Tafjord, Dalvi, and Clark, “ProofWriter,” Findings of ACL 2021,
  [ACL Anthology](https://aclanthology.org/2021.findings-acl.317/) and
  [paper sections 3–4](https://arxiv.org/html/2012.13048v2#S3).
- [Official ProofWriter dataset](https://allenai.org/data/proofwriter).

### PrOntoQA: ontology first, then proof-conditioned rendering

PrOntoQA explicitly generates an ontology, traverses it to construct a proof of
a requested length, translates the ontology into context, and translates the
proof into query, chain of thought, and label. It is direct prior art against a
generic novelty claim for proof-conditioned natural-language generation.

- Saparov and He, “Language Models Are Greedy Reasoners,” ICLR 2023,
  [paper section 3](https://arxiv.org/html/2210.01240v2#S3) and
  [OpenReview](https://openreview.net/forum?id=qFVVBzXxR2V).
- Original-paper implementation: [ontology construction](https://github.com/asaparov/prontoqa/blob/066f73fc70bfaa92ed2476fd3c41b4951ad6e63a/theory.py#L34-L104) and
  [proof construction](https://github.com/asaparov/prontoqa/blob/066f73fc70bfaa92ed2476fd3c41b4951ad6e63a/proof.py#L29-L187).

### ProverGen: goal-first, prover-constrained construction

ProverGen chooses a goal and truth value, recursively constructs supporting
rules top-down, uses Prover9 during truth assignment, adds distractors,
verbalizes the logic skeleton, and validates the result. This is strong direct
precedent for solver-in-the-loop target-conditioned generation.

- Qi et al., “Large Language Models Meet Symbolic Provers for Logical
  Reasoning Evaluation,” ICLR 2025,
  [paper section 3.2](https://arxiv.org/html/2502.06563#S3.SS2).
- [Official ProverGen repository](https://github.com/opendatalab/ProverGen).

### AlphaGeometry: premises, closure, then proof traceback

AlphaGeometry samples premises, computes a symbolic deduction closure, and
traces reachable conclusions back to proofs. It is a strong precedent for
forward solver-generated theorem/proof data outside natural-language spatial
question answering.

- Trinh et al., “Solving olympiad geometry without human demonstrations,”
  Nature 2024, [open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794143/#Sec2) and
  [DOI](https://doi.org/10.1038/s41586-023-06747-5).

### StepGame and SpaRP: closest spatial precedents

StepGame constructs spatial relation chains, chooses a query over chain
entities, verbalizes and shuffles the edges, and optionally adds controlled
noise. It establishes graph/path-first spatial generation and depth-controlled
evaluation, but not all-model entailment under incomplete premises.

SpaRP constructs and verbalizes symbolic spatial reasoning paths over existing
SpaRTUN and StepGame instances. It is the closest precedent for spatial trace
supervision, but it derives paths post hoc rather than constructing new problems
from certificates.

- Shi et al., “StepGame,” AAAI 2022,
  [official paper record](https://ojs.aaai.org/index.php/AAAI/article/view/21383).
- Rizvi et al., “SpaRC and SpaRP,” ACL 2024,
  [ACL Anthology](https://aclanthology.org/2024.acl-long.261/).

### SpatialEval: likely map-first, but not fully auditable

SpatialEval describes configurable map-like scenes and pairwise textual
relations, which suggests scene/map-first construction. The publication does
not establish whether released labels were read from a privileged complete map
or re-solved from emitted text, and the official repository still does not
publish its announced generation script. It should therefore be described as
likely map-first rather than used as definitive evidence for one generation
algorithm.

- [SpatialEval paper](https://arxiv.org/html/2406.14852).
- [Official repository](https://github.com/jiayuww/SpatialEval).

## Defensible SpatialEntail contribution

The defensible contribution is not any one regime in isolation. It is the
combination of:

1. explicit generation-provenance metadata;
2. open-world all-model semantics for all eight ordinal directions and Boolean
   combinations;
3. re-solving every visible problem instead of trusting a hidden generator
   answer;
4. a checked proof/refutation/witness certificate covering the complete answer
   domain;
5. independent replay and global SMT validation with a round-tripped text
   representation; and
6. evaluation of whether training on controlled, solver-validated problems
   transfers to unplanted visible-problem-first and external instances.

This is an integration and evaluation claim. No located primary source uses
this exact combination, but that bounded literature finding is not proof of
absolute priority.

## Recommended FYP experiment

For a manageable FYP, provenance should be a methodological axis rather than a
four-arm project by itself.

### Core story

1. **Part I — diagnose latent-world labeling.** Show that a complete map can
   support an answer that is not entailed by the emitted text. This motivates
   all-model validation.
2. **Part II — train on proof-template-first problems.** Compare answer-only
   with verified Natural worked solutions while certificates remain offline.
3. **Evaluate on visible-problem-first and external data.** This tests whether
   the model learned the spatial logic rather than the planted generator
   template.

### Minimum convincing comparison

Hold the model and base problems constant, then compare answer-only with verified
Natural SFT on proof-template-first training data. Evaluate both on
visible-problem-first and external data; a provenance-mixture study is optional,
not part of the central SFT contrast.

Test on:

- matched in-distribution examples;
- visible-problem-first structural holdouts;
- held-out rule compositions and longer depths; and
- untouched compatible external data.

Match or stratify the final rows by query family, semantic answer class, entity
and premise count, operator family, measured shortest proof depth, branching,
distractor count, and surface template. Split by canonical structure rather
than row identity.

### Strong research questions

1. Does proof-template-first verified-Natural training transfer to
   visible-problem-first problems whose proof structures were not planted?
2. Does adding visible-problem-first training reduce template dependence?
3. Does verified Natural supervision improve answer accuracy and ambiguity
   recognition beyond answer-only supervision on the same problems?
4. How often does intended template depth differ from measured proof depth or
   admit a shorter alternative proof?
5. How often does a privileged-world answer disagree with the complete
   semantics of the emitted premises?

## Claims to avoid

- Do not call the four regimes established community terminology.
- Do not claim the first proof-first or solver-generated reasoning dataset.
- Do not equate a planted proof template with the actual shortest proof.
- Do not say that re-solving a world-first item removes every generator bias.
- Do not compare unmatched regimes and attribute the entire result to
  provenance.
- Do not treat one satisfying map as evidence of entailment.
- Do not call a prover-accepted formal object proof that its natural-language
  rendering is faithful.

## FYP presentation boundary

The implementation is larger than the mathematical argument. The presentation
should not give every generator control, certificate schema, proof rule, or
review component equal weight. The defensible core requires only introductory
propositional logic, satisfiability, and ordinal comparisons.

### Minimum mathematics to explain

For visible premises `P` and a candidate answer `A`:

```text
possible(A)   iff SAT(P AND A)
impossible(A) iff UNSAT(P AND A)
entailed(A)   iff UNSAT(P AND NOT A)
```

These checks have direct verbal interpretations:

- `SAT(P AND A)`: at least one valid arrangement supports the answer;
- `UNSAT(P AND A)`: no valid arrangement supports the answer; and
- `UNSAT(P AND NOT A)`: no counterexample exists, so every valid arrangement
  supports the answer.

The spatial encoding adds only two ordinal comparisons per exact direction:

```text
East(A, B)      := x_A > x_B AND y_A = y_B
North(A, B)     := x_A = x_B AND y_A > y_B
Northeast(A, B) := x_A > x_B AND y_A > y_B
```

The other five directions use the corresponding combinations of `<`, `=`, and
`>`. Transitivity is ordinary order transitivity, such as `x_A > x_B` and
`x_B > x_C` implying `x_A > x_C`.

No presentation-level derivation of Z3's internal algorithms is required. Z3
is the semantic decision procedure used to answer the checks above; the study's
responsibility is the spatial encoding, declared semantics, candidate scope,
and surrounding validation.

### One distinction that must be mastered

| Status | Presenter-friendly meaning |
|---|---|
| True in one hidden map | One privileged arrangement supports the answer. |
| Possible | At least one arrangement satisfying the visible premises supports it. |
| Entailed | Every arrangement satisfying the visible premises supports it. |
| Impossible | No arrangement satisfying the visible premises supports it. |
| Ambiguous | Different valid arrangements support different candidate answers. |

Ambiguity is not solver indecision or probability. It is a positive result
supported by multiple valid witnesses.

### Simplified three-part thesis story

The main presentation should use three stages rather than presenting every
software component as a separate contribution:

1. **Diagnose:** a relation true in a complete hidden map need not be entailed
   by the partial text shown to the model.
2. **Validate:** translate directions into X/Y constraints and classify every
   candidate as possible, impossible, or entailed.
3. **Teach and test:** generate checked reasoning examples, then test whether
   learning transfers to problems whose proof structures were not planted.

Generation provenance supports stage 3. For the oral presentation, three
questions are enough:

```text
Latent-world-first:    Did we begin from a complete hidden world?
Proof-template-first:  Did we plant a reasoning structure to teach?
Visible-problem-first: Did we propose the visible problem before discovering its answer?
```

Certificate-first should remain a future-work distinction unless it becomes an
implemented experimental arm. It is unnecessary for explaining the completed
core contribution.

### Sixty-second explanation

> Spatial reasoning datasets are often generated from complete maps, while the
> model sees only a partial textual description. A relation can be true in the
> hidden map without being logically determined by the visible text. I encode
> each direction as X and Y ordering constraints and use Z3 to determine whether
> every candidate answer is possible, impossible, or entailed. Entailment is
> checked by asking whether the premises together with the negation of the
> answer are unsatisfiable. I then construct checked proof, refutation, and
> witness certificates to validate the data, render verified Natural worked
> solutions for SFT, and test whether models trained on controlled
> proof-template examples generalize to independently proposed visible problems
> and compatible external datasets.

### Recommended main-slide sequence

1. One hidden-map/partial-text counterexample.
2. The distinction between truth, possibility, and entailment.
3. The three SAT/UNSAT checks.
4. One example of a direction encoded as X/Y comparisons.
5. The solver, certificate engine, and renderer as separate responsibilities.
6. Controlled training versus unplanted evaluation.
7. Results, limitations, and the precise contribution claim.

Detailed certificate schemas, registry cells, Count correlation, HITL routing,
all generation knobs, and solver implementation internals belong in backup
slides or the written report. They should be introduced in the main talk only
when they directly support a reported result.

### Short viva answers

**Why not reconstruct one map?**

Several maps may satisfy the same text. One reconstructed map proves
possibility, not entailment.

**Is Z3 generating the explanation?**

No. Z3 checks the complete semantic answer space. A separate proof engine
constructs replayable evidence, and the renderer expresses that evidence.

**Is proof-template-first generation biased?**

Potentially. It provides controlled coverage, but can create recognizable
templates. This motivates visible-problem-first structural holdouts and
external evaluation.

**What is novel if proof-conditioned generation already exists?**

The claim is not first use of proof-conditioned generation. It is the
integration of open-world spatial semantics, complete candidate checking,
checked certificates, explicit provenance, and evaluation across construction
regimes.

**What does the student need to defend about Z3?**

The formulas submitted to it, the interpretation of SAT and UNSAT, and why the
candidate checks establish the reported semantics—not Z3's internal search
algorithm.

### Presenter readiness checklist

The presenter is ready for the mathematical core when they can explain, without
notes:

- why one satisfying map does not prove entailment;
- why `UNSAT(P AND NOT A)` proves `A` is entailed;
- how one exact direction becomes two axis comparisons;
- why two different witnesses establish genuine ambiguity;
- why the proof engine and Z3 have different responsibilities; and
- why proof-template training must be evaluated on unplanted problems.

This boundary keeps the thesis technically serious without requiring advanced
mathematics. The complexity remains in the implementation and verification,
where it belongs, rather than in the oral explanation.

## Decision

The idea is valid and worth retaining. The clean version is:

> SpatialEntail studies whether solver-validated Natural spatial supervision
> transfers beyond its construction regime to independently proposed visible
> problems under open-world entailment, compared with matched answer-only SFT.

That is narrower, more testable, and better supported than presenting
world-first, proof-first, and premise-first as three universally recognized
dataset categories.
