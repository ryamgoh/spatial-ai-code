# FYP research canvas

Status: working research direction, not a fixed paper outline

Last updated: 10 October 2026

Raw contribution notes:
[`What this project is trying to do`](../notes/raw-contribution-notes.md)

Annotated literature ledger:
[`What each source supports and does not support`](annotated-sources.md)

## Working title

**Formal Semantics for Evaluating and Improving Spatial Reasoning in Large
Language Models**

Generic backup: **Evaluating and Improving Spatial Reasoning in Large Language
Models**

Working benchmark name:

> **SpatialEntail: A Benchmark for Qualitative Spatial Reasoning under Partial
> Information**

`SpatialEntail` is provisional until the benchmark tracks are frozen and a
public-name collision check is completed. Internal schemas and implementation
modules continue to use `spatial-v2` for now.

## Naming convention

Use these terms consistently in experiments and the report:

| Term | Meaning | Examples |
|---|---|---|
| `SpatialEntail` | The benchmark as a whole | “evaluated on SpatialEntail” |
| Pilot | Preliminary data used to calibrate generation, difficulty, and protocols | SpatialEntail Pilot |
| Suite | A fixed evaluation module | Core, Propositional, Depth, Ambiguity |
| Arm | One prompting, training, or RL condition | Answer-Only, Checked Natural, Checked Symbolic, Symbolic Local Proof |
| Version | Public artifact identity, used only when needed for reproducibility | SpatialEntail v1.0 |

Do not rename the benchmark for each ablation. Natural/Symbolic traces, state
schedules, data scales, and SFT/RL methods are arm names over the same benchmark.
Avoid “dev release”; use “pilot” for preliminary experiments and “development
split” only for data used to make research decisions.

## Research focus

This project studies spatial reasoning in LLMs, separately from visual
perception. Its domain is a finite, static, two-dimensional qualitative world:
eight compass directions, no co-located entities, no metric information, and
open-world treatment of unstated relations. Visual, embodied, metric, and 3D
reasoning are outside the experiments.

The motivating problem is that a benchmark may derive its answer from a
complete latent world while showing the LLM only a partial description. An
answer can be true in that world without being entailed by the prompt. This
conflates spatial reasoning with guessing hidden construction state and can
also contaminate generated training labels or rationales.

SpatialEval Spatial-Map TQA is the first case study. Its public release contains
generated data and evaluation code but no public generator. Our solver is an
independent reconstruction of the answer semantics needed to audit the released
text, not a reproduction of the authors' generator.

The central proposition is:

> Explicit formal spatial semantics can improve both the validity of LLM
> spatial-reasoning evaluation and the supervision used to improve that
> reasoning.

## Formal method

The method has no name yet. It combines:

- qualitative spatial atoms over `N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, and
  `NW`;
- X/Y-axis decomposition;
- open-world, model-theoretic answer semantics;
- arbitrary propositional combinations using `NOT`, `AND`, `OR`, `IF`, and
  `IFF`;
- SMT satisfiability checks; and
- constructive witnesses and counterexamples.

For premises `P` and candidate claim `A`:

```text
Possible(P, A)  iff  SAT(P AND A)
Entailed(P, A)  iff  UNSAT(P AND NOT A)
```

The theoretical obligation is a soundness-and-completeness argument for the
encoding relative to the declared ontology. Backend comparison, witness tests,
parser coverage, and manual case checks separately validate the implementation
and its interpretation of dataset language.

The result serves four roles:

```text
an evaluator  -> audit labels and expose ambiguity
a generator   -> accept only problems with the requested semantics
a supervisor  -> render Natural or Symbolic reasoning traces
a verifier    -> score exact outcomes during optional RL
```

Coordinates must not be a privileged hidden source from which both premises and
gold answers are read. Numeric coordinates remain post-solve audit certificates,
not SFT targets. The checked Symbolic arm may expose a qualitative rank-compressed
witness; a 4B-only local-proof arm omits that witness. This comparison is limited
to unique or entailed cells for which both targets are well-defined.

## Generation provenance: world-first, proof-template-first, certificate-first,
and premise-first

Status: provisional design decision for SpatialEntail, retained here for further
iteration before the benchmark suites are frozen.

The order in which a synthetic problem is constructed changes what its answer
and reasoning trace mean. Four generation families must therefore be named and
tracked explicitly. Generated V2 rows record this choice in
`metadata.generation_provenance`.

### World-first generation

```text
sample a complete coordinate world W*
-> derive a visible premise subset P from W*
-> choose a query Q
-> read the answer A from W*
```

This establishes only that the selected-world answer is possible:

```text
W* satisfies P AND A
```

It does not establish that the visible premises entail that answer:

```text
P entails A
```

This is the failure mode exposed by the SpatialEval audit. SpatialEntail must
not use an answer read from a privileged generating map as its gold label.
World-first sampling may at most propose a candidate problem if the original
answer is discarded and the complete answer semantics are recomputed from the
emitted premises. Even then, it is not the preferred provenance because it
biases the problem distribution around one arbitrary complete world.

Coordinates are not prohibited. After a problem exists, a coordinate assignment
returned for `P AND A` is a valid certificate that `A` is possible. Two such
assignments can certify ambiguity. A single assignment cannot establish that an
answer is entailed.

### Proof-template-first generation

```text
choose a proof-rule or path template
-> instantiate premises and a query expected to exercise that template
-> solve the resulting problem
-> construct and replay the actual certificate
```

This is the provenance of the current Boolean curricula and controlled
Direction/Which/Count paths. It controls the intended proof shape before the
problem is solved, but it does not instantiate the final typed certificate
first. The name therefore avoids overstating the implementation.

### Certificate-first generation

```text
sample a typed proof structure
-> instantiate its Boolean and spatial atoms
-> derive premises P, query Q, and a proof certificate
-> replay the certificate with a small proof checker
-> ask Z3 for the complete answer semantics
```

The answer is known by construction because it is derived from the exact
premises shown to the model, not because it was true in a hidden map. For
example, a disjunctive-syllogism template may instantiate

```text
P1: NE(A,B) OR NW(A,B)
P2: NOT NW(A,B)
therefore: NE(A,B)
```

Certificate-first generation would be the strongest source of trace-supervised
training data because it guarantees a structured derivation before the problem
text exists. It remains a distinct target provenance rather than a label for
the current proof-template-first generator. Together, the two controlled
families can directly control:

- Boolean and spatial proof depth;
- rule family and rule composition;
- number of branches;
- independent X/Y support;
- ambiguity and contradiction forms; and
- relevant versus distracting premises.

Its principal risk is construction leakage. A model may learn recurring proof
templates, premise order, or answer-conditioned surface patterns rather than a
general inference procedure. A constructed proof may also cease to describe the
true difficulty if another premise creates a shorter proof. Every candidate must
therefore be re-solved, checked for unintended answer alternatives, and measured
from the reconstructed proof rather than trusted from generation metadata.

### Premise-first generation

```text
sample premises P and query Q without selecting an answer
-> classify every candidate with the formal solver
-> search for proof or countermodel certificates
-> measure the proof that actually exists
-> admit the problem to the resulting semantic and difficulty bucket
```

Premise-first generation allows unplanned interactions between negation,
disjunction, implication, equivalence, and spatial composition. It is therefore
less answer-conditioned and better suited to evaluating generalisation beyond
known proof templates. It can expose unexpected entailments, shorter proofs,
ambiguity, inconsistency, and solver bugs.

Its costs are substantial: many sampled formulas may be trivial, inconsistent,
irrelevant to the query, excessively ambiguous, or difficult to explain
compactly. Semantic classes will be imbalanced, and proof difficulty is known
only after proof search. Premise-first data therefore requires solve-and-bucket
admission rather than balancing by planting a target answer.

### Comparison

| Criterion | World-first | Proof-template-first | Certificate-first | Premise-first |
|---|---|---|---|---|
| Gold-label basis | One sampled world unless recomputed | Solver semantics plus extracted checked certificate | Certificate fixed before problem construction | Semantics discovered from emitted premises |
| Trace availability | Post-hoc and potentially unfaithful | Checked after instantiation | Guaranteed by construction | Requires proof search |
| Difficulty control | Indirect and unreliable | Intended template, then remeasured | Direct, then remeasured | Post-hoc only |
| Rule coverage | Accidental | Explicitly balanceable | Explicitly balanceable | Emergent and potentially sparse |
| Template leakage | Hidden-world and construction bias | Main risk | Main risk | Lower, though sampler artifacts remain |
| Best role | Proposal source only after re-solving | Current controlled supervision | Stricter future supervision | Generalisation and robustness evaluation |

### Provisional SpatialEntail design

The working recommendation is not template-controlled generation everywhere.

```text
Trace-supervised training:
  proof-template-first problems with replayable certificates

Secondary training pool:
  premise-first problems for which concise proofs are successfully extracted

Development and final evaluation:
  primarily premise-first, solve-and-bucket problems

External transfer:
  untouched SpatialMap-TQA-Corr
```

Proof-template-first data supplies controlled coverage and checked Natural or
Symbolic targets. Premise-first evaluation tests whether any gain transfers
beyond the construction templates seen during supervision. A
proof-template-only final test would risk measuring template recognition; a
premise-first-only trace corpus would make checked supervision unnecessarily
expensive and selective.

No fixed mixture is assumed. The balance must be chosen from a pilot measuring
rejection rates, proof-search success, trace length, class balance, duplicate
formula structures, and transfer to held-out rule compositions.

### Shared validation contract

Both generation tracks must pass through the same validation boundary:

```text
SpatialProblem
  |-- Z3 oracle: possible, entailed, impossible, inconsistent
  |-- proof certificate: typed derivation or constructive branch evidence
  |-- proof checker: replay every claimed step
  |-- renderer: Natural and Symbolic views of the same certificate
  `-- text round trip: parse and re-solve the emitted prompt
```

Z3 and the proof engine may share the formula AST and ontology, but they should
not share their core reasoning implementation if agreement is claimed as
independent evidence. The present explainer is only partially independent: Z3
classifies the claim and the explainer reconstructs axis paths for the subset it
supports. Full SpatialEntail traces require a typed Boolean-plus-spatial proof
certificate rather than a solver-status statement alone.

For each candidate claim `A`, validation requires:

```text
possible    -> SAT(P AND A) and a checked open-branch or witness certificate
entailed    -> UNSAT(P AND NOT A) and a checked derivation or closed proof tree
impossible  -> UNSAT(P AND A) and a checked contradiction certificate
contingent  -> SAT(P AND A) and SAT(P AND NOT A), with evidence for both
```

Natural and Symbolic targets originate from the same accepted certificate.
Natural now narrates the checked dependencies and scopes that Symbolic encodes,
with query-specific witnesses and exclusions in both views. Shared provenance
and step coverage do not establish equal information load or token cost. Audit
the mapping across the frozen corpus before claiming a notation-only effect;
only Symbolic model outputs have replay-based process validation. Generated Natural
targets inherit checked source evidence, but arbitrary model-produced Natural
prose has no replay parser. External evidence validity does not establish
faithfulness to a model's internal computation.

### Leakage and split commitments

Random-seed and entity-name separation are insufficient. SpatialEntail should
record and, where appropriate, hold out:

- proof-rule sequences and compositions;
- normalised formula-tree shapes;
- spatial and propositional proof depths;
- branch counts and ambiguity sizes;
- supporting-premise sets;
- distractor connection patterns; and
- surface realisation templates.

Proof-first training and premise-first evaluation must not share exact
normalised problems. The strongest structural test also withholds selected rule
compositions rather than merely renaming entities or shuffling premises.

The workload supports random interpolation splits and structural clustering.
Canonical signatures describe the visible source formulas and query, ignoring
entity names, premise order, and commutative operand order. Canonicalization is
exact within its permutation budget; a conservative invariant fallback can
merge distinct structures. Three-way overlap validation keeps train,
development, and test separate under that identity. This does not automatically
hold out a rule composition, depth, or provenance. Such claims require explicit
predeclared holdout cells and a distribution audit after context admission.

### Pilot questions

- Which primitive Boolean and spatial rules belong in the first proof schema?
- Can the proof checker remain small enough to audit independently?
- How often does premise-first sampling yield concise replayable proofs?
- How often does proof-template-first construction admit an unintended shorter proof?
- What semantic-class imbalance appears before solve-and-bucket admission?
- Does a qualitative rank witness add value beyond query-local Symbolic proof?
- How much performance survives held-out formula trees and proof-rule
  compositions?
- Should problems without a concise extracted proof remain answer-only
  evaluation items rather than being discarded?

## Data boundary

```text
evaluation stream: released SpatialEval questions
                   -> semantic audit
                   -> separately stored SpatialMap-TQA-Corr labels and option E

learning stream: newly generated, solver-validated SpatialEntail questions
                 -> prompting, SFT, ablations, and optional RL
```

The evaluation stream does not generate new spatial questions. Its coordinate
witnesses are audit evidence for alternative models of the existing text.
SpatialMap-TQA-Corr preserves the released 1,500 questions and derives a new
evaluation view from them.

## Main research question

> **Can formal spatial semantics improve both the validity of spatial-reasoning
> benchmarks and the spatial reasoning learned by large language models?**

## Sub-questions and hypotheses

| Question | Hypothesis |
|---|---|
| Are answers in an existing spatial benchmark entailed by the information given to the LLM? | SpatialEval will contain a substantial set of published answers that are possible but not unique under the declared semantics. |
| Does semantic correction change how LLM spatial reasoning is measured? | Original and corrected evaluation will produce different scores and error interpretations; ranking changes are possible but not assumed. |
| Can a generated benchmark measure meaningful spatial difficulty? | Accuracy will decline systematically with proof depth, independent-axis composition, and relevant distractors rather than only with prompt length. |
| Does formal reasoning supervision teach more than answer imitation? | Natural or Symbolic trace SFT will outperform answer-only SFT on at least some structurally held-out conditions. |
| Which representation supports the strongest generalisation? | Natural and Symbolic supervision packages may trade accuracy, validity, and token cost; information-matched notation effects remain a separate experiment. |
| Does solver-verifiable RL add value after SFT? | Unknown; it is viable only if hard prompts produce within-group reward variance and SFT leaves headroom. |

The first five questions form the intended study. RL is optional.

## Decisive studies

Detailed plans:

- [`SpatialEval evaluation stream`](part-1-evaluation.md)
- [`SpatialEntail learning study`](part-2-training.md)

| Study | Design | Question answered |
|---|---|---|
| Formal validation and SpatialEval | Prove and validate the encoding; audit the untouched four-option release; validate counter-witnesses; create a separate five-option correction; compare general, reasoning, and compatible spatially fine-tuned LLMs on original, corrected, entailed-only, and ambiguous-only views | Does formal semantics expose consequential ambiguity and change conclusions about LLM capability? |
| Benchmark calibration and prompting | Generate a development pool over query type, direction, ambiguity, proof depth, independent axes, and distractors; survey untuned LLMs; compare direct, generic-CoT, Natural-axis, and Symbolic-axis prompts; then freeze the benchmark | Is the benchmark valid, solvable, non-saturated, and structurally discriminative? |
| SFT and representation | Compare answer-only, checked Natural, and checked Symbolic on Qwen3.5-2B and Qwen3.5-4B; run corrupted Symbolic and Symbolic local-proof mechanisms on 4B | Do checked traces add value beyond task exposure, does evidence validity matter, and does a global qualitative witness help? |
| Generalisation, scale, and optional RL | Calibrate nested 4K/8K/17K training; test matched cells, depth 1--8, held-out formula compositions, and five external datasets; run GRPO only with reward variance and compare it with compute-matched continued SFT | Do gains extrapolate beyond training depth and construction families, and does RL add anything beyond supervised optimisation? |

Difficulty is not an end in itself. The benchmark should contain solvable,
discriminative, and hard cells whose performance changes for identifiable
structural reasons.

### Learning-study design status

- The [canonical task taxonomy](task-taxonomy.md) defines three query families
  (`DIR`, `SEL`, `CNT`), three capability lenses (`RR`, `LR`, `MTR`), and
  factorial benchmark cells. The former four-tier, 17-bucket allocation is
  withdrawn. The 4K/8K/17K sizes remain candidate maximum budgets; cell quotas
  are frozen only after a bounded capability/yield pilot.
- The central arms are answer-only, checked Natural, and checked Symbolic on
  Qwen3.5-2B and Qwen3.5-4B with seeds 42 and 43. Corrupted Symbolic and
  Symbolic local proof are planned 4B-only mechanism arms: 16 confirmatory runs
  total after the local-proof serializer and acceptance contract are implemented.
- Depth-controlled training stops at depth 4. Direction, Selection, and Count are
  tested separately at depths 1--8 with 100 clean examples per family-depth
  cell and one paired noisy counterpart per item. Depths 5--8 are extrapolation.
  Paired Selection/Count distractors remain an implementation prerequisite.
- External evaluation uses SpatialMap-TQA-Corr, StepGame, Text2Space,
  SpartQA-Human, and ReSQ under their native semantics and metrics.
- Every row receives formal checking. If `C` is the number of frozen reporting
  cells, source-blinded review uses at least the larger of 2% of the selected
  pool or `5C` examples, with at least five per cell. One
  pinned OpenAI judge, one pinned Anthropic judge, and one human see only the
  prompt and checked Natural trace.
- SFT is the main study. GRPO is conditional future work.

## Evaluation commitments

- Use a development set for decisions and an untouched final test for claims.
- Split by canonical formal problem and generation provenance before producing
  prompt, answer, or trace variants; do not rely only on entity renaming, premise
  shuffling, or random seed separation.
- Keep original SpatialEval unchanged and publish corrections separately.
- Use the same LLM checkpoints, task instructions, and decoding settings for
  paired original-versus-corrected comparisons.
- Use strict exact-answer-set accuracy as the primary model metric.
- Macro-average predeclared structural cells so easy mass cannot hide failures.
- Report ambiguity recognition, output validity, token cost, and paired
  fixes/regressions as secondary diagnostics.
- Decompose Symbolic process validity into replayable reasoning, checked menu
  decision, cross-record domain agreement, and fully valid trace rates.
- Report seeds 42 and 43 separately, with their mean and range; use paired
  item-level bootstrap intervals within each seed rather than a strong
  seed-level confidence claim.
- Treat `SINGLE` as the primary contract, `ALL_POSSIBLE` as a strict set-valued
  extension, and `VISIBLE_POSSIBLE` as a menu-relative diagnostic.

## Success and limits

The minimum successful FYP establishes the formal semantics and validated
decision procedure, a defensible SpatialEval audit and correction, a frozen
controlled benchmark, fair prompting and answer-only baselines, and a clear
result on whether reasoning supervision improves accuracy or structural
generalisation. RL is not required. A controlled negative SFT or RL result is
still evidence about the tested method.

Reassess the central claim if manual checks expose a bad formal interpretation,
solver backends or witnesses disagree, correction has negligible evaluation
impact, generated difficulty lacks an interpretable gradient, or training gains
disappear under new seeds and a frozen test.

The project does not claim visual or embodied reasoning, navigation, metric or
3D geometry, unrestricted natural-language semantics, or formal verification
of the complete Python implementation. Relation verification remains a
low-priority KIV extension.

## Open decisions

- Final wording of the title and central claim
- Final collision check and approval of the provisional `SpatialEntail` name
- Pilot balance between proof-template-first and premise-first generation
- Held-out proof compositions and formula structures for the final test
- Manual case-selection and checking protocol
- Frozen inference prompts and decoding conditions
- Version-pinned OpenAI and Anthropic judge identifiers
- Whether `ALL_POSSIBLE` belongs in the core benchmark
- Go/no-go criteria and compute budget for RL
