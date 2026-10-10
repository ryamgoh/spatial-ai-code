#pagebreak(weak: true)
= Results and Discussion

== SpatialEval Semantic Audit

All 1,500 released SpatialMap-TQA questions parse and solve under the declared
`SINGLE` semantics. The published oracle is uniquely entailed in 667 questions
and possible but non-unique in 833.

#figure(
  table(
    columns: (1fr, auto, auto),
    table.header([*Query*], [*Exact oracle*], [*Possible but non-unique*]),
    [Direction], [332], [168],
    [Which], [140], [360],
    [Count], [195], [305],
    [*Total*], [*667*], [*833*],
  ),
  caption: [SpatialMap-TQA audit under visible-premise single-answer semantics.],
) <results-spatialeval-audit>

No released oracle is contradicted under the current formalisation. The result
does not show that the hidden source map is wrong; it shows that one source-map
answer is often not identifiable from the text alone. The affected rate differs
substantially across query families, so aggregate accuracy conceals different
semantic failure modes.

SpatialMap-TQA-Corr preserves the source rows and adds _Cannot be determined_.
It maps the 667 uniquely entailed questions to their published answers and the
833 underdetermined questions to the explicit fifth option. Manual assumption
sensitivity and the final model re-evaluation remain pending. Two independent
Z3 executions produced byte-identical audit and correction artifacts, while
returned witnesses were revalidated against every parsed premise.

== Implemented Artifact Evidence

The completed software evidence concerns preparation and replay, not learning.
The bounded integration pilot admitted 80 base problems and 320 paired rows,
split across training, development, and test. Its largest full training sequence
was 4,032 tokens under a 4,096-token limit. A separate 16-base premise-first
holdout also passed admission.

Across the final generated pilot and holdout artifacts, 384 rows passed the
evaluation loader and their expected control outcomes: checked Symbolic traces
replayed, corrupted Symbolic reasoning failed while retaining its valid
decision, and Natural/answer-only process metrics were correctly inapplicable.
These checks establish that the proposed experimental interfaces execute; they
do not estimate LLM performance.

The final nested 4K/8K/17K corpus, human-facing quality audit, and model training
have not been executed. Their tables below are explicit placeholders rather
than results; cell-level rows are added only after the taxonomy allocation is
frozen.

== Dataset Yield and Quality — Not Run

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    table.header([*Query family*], [*Requested*], [*Accepted*], [*Rejected*], [*Human acceptable*]),
    [`DIR`], [—], [—], [—], [—],
    [`SEL`], [—], [—], [—], [—],
    [`CNT`], [—], [—], [—], [—],
  ),
  caption: [Dataset-yield and source-blinded quality placeholder. No values are available before allocation, generation, and review.],
) <results-dataset-yield>

The final analysis will report the frozen cell registry, capability and semantic
strata, rejection reasons, and length distributions rather than only accepted
query totals. Context admission can selectively remove Count or branching
examples; the accepted distribution must therefore be compared with the
requested factorial allocation.

== Main SFT Comparison — Not Run

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    table.header([*Model*], [*Arm*], [*Seed 42*], [*Seed 43*], [*Mean / range*]),
    [2B], [Answer-only], [—], [—], [—],
    [2B], [Checked Natural], [—], [—], [—],
    [2B], [Checked Symbolic], [—], [—], [—],
    [4B], [Answer-only], [—], [—], [—],
    [4B], [Checked Natural], [—], [—], [—],
    [4B], [Checked Symbolic], [—], [—], [—],
  ),
  caption: [Primary internal macro-accuracy table. Placeholder cells must not be interpreted as zero.],
) <results-main-sft>

The primary claim requires consistent direction across both seeds. Model size is
reported as robustness, not as a scaling law, because equal LoRA rank represents
different relative adapter capacity and only two model sizes are observed.

== Depth Extrapolation and Distractor Robustness — Not Run

The depth result will show separate Direction, Selection, and Count curves over
depths 1–8. Depths 1–4 are matched to training; depths 5–8 are extrapolation.
For each point, the clean result is paired with an otherwise identical version
containing two removable distractors.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    table.header([*Family*], [*d1*], [*d2*], [*d3*], [*d4*], [*d5*], [*d6*], [*d7*], [*d8*]),
    [Direction clean], [—], [—], [—], [—], [—], [—], [—], [—],
    [Direction noisy], [—], [—], [—], [—], [—], [—], [—], [—],
    [Selection clean], [—], [—], [—], [—], [—], [—], [—], [—],
    [Selection noisy], [—], [—], [—], [—], [—], [—], [—], [—],
    [Count clean], [—], [—], [—], [—], [—], [—], [—], [—],
    [Count noisy], [—], [—], [—], [—], [—], [—], [—], [—],
  ),
  caption: [StepGame-style matched and extrapolative depth ladder.],
) <results-depth-ladder>

A decline from depth 5 is evidence of length extrapolation failure only if all
training and auxiliary supervision exclude depths 5–8. A clean/noisy gap is a
separate robustness result and must not be folded into the depth effect.

== Premise-First Composition — Not Run

The premise-first table will separate conditional, negation/disjunction,
branching, and mixed formula holdouts. Every primitive operator occurs in
training; only selected compounds are unseen. This prevents operator novelty
from being misreported as compositional extrapolation.

== External Transfer — Not Run

#figure(
  table(
    columns: (1fr, 1fr, 1fr),
    table.header([*Dataset*], [*Native primary metric*], [*Result*]),
    [SpatialMap-TQA-Corr], [Strict answer accuracy], [—],
    [StepGame], [Accuracy by hop], [—],
    [Text2Space], [Eight-direction query accuracy], [—],
    [SpartQA-Human], [YN/FR/CO/FB native metrics], [—],
    [ReSQ], [Binary accuracy], [—],
  ),
  caption: [External transfer results under native task contracts.],
) <results-external-transfer>

No cross-dataset aggregate is reported. A gain on synthetic StepGame does not
establish realistic-language transfer; a gain on ReSQ does not establish
open-world calibration or Count reasoning.

== Process Validity and Mechanism Ablations — Not Run

Symbolic outputs are divided into malformed envelope, invalid reasoning,
invalid decision, domain disagreement, wrong answer after valid evidence, and
fully valid trace. Answer accuracy and process validity are reported separately.

The corrupted-Symbolic comparison estimates the effect of the predeclared typed
semantic-mutation procedure within the Symbolic package; it is not a generic
control for extra tokens or proof validity. Mutation classes, frequencies, and
length changes must be reported. The local-proof comparison is restricted to
unique/entailed cells and tests whether adding a complete qualitative rank
witness improves performance or merely adds a difficult serialization burden.
Ambiguous cells are excluded because witness removal would remove necessary
evidence rather than isolate representation. This arm remains unimplemented
until its consistency-evidence and process-acceptance contract is specified.

== Discussion and Threats

The semantic audit already establishes that a latent-world answer can remain
possible without being textually entailed. The certificate pipeline establishes
that checked supervision and process scoring are executable. Neither result
establishes that an LLM benefits from those traces.

The planned learning study remains limited by synthetic controlled language,
finite templates, incomplete automatic proof construction, shared data
definitions between solver and checker, and potentially selective context
admission. Natural model output has no process parser. Symbolic process validity
does not reveal the model's hidden computation. Rank witnesses may be useful
world-state scaffolds or arbitrary-completion noise; the ablation is intended to
measure that trade-off rather than assume an outcome.

Two seeds provide replication but weak seed-level uncertainty. Human and LLM
readability judgements are source-blinded rather than classically double-blind.
External datasets differ in ontology, language, and answer contract, so their
results support bounded transfer claims only.
