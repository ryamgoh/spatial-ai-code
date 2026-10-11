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

The current no-quota benchmark registry contains thirteen representative cells,
all marked ready after adding multiple-certain Selection admission,
correlation-sensitive Count admission, and removable Selection distractors. A
seed-1911 pilot generated and certificate-checked 50 examples per cell,
producing 650 persisted rows with a content-hashed manifest. The generator made
865 candidate attempts and rejected 215 proposals, for a 75.14% overall
acceptance rate. A constructive atomic template raised `DIR-ARC-ALT` from 1.05%
acceptance in the preceding pilot to 100% while retaining solver validation.

A deterministic source-blinded sample contained five items from each cell.
Sequential isolated `DeepSeek-V4-Pro` and `GPT-5.5` terminal-agent runs each
returned 65 approvals and no rejections. Both receipts passed exact packet,
item-ID, payload-hash, and rubric-field validation, so all 65 items were
accepted and the human appeal queue was empty. These results concern the
representative feasibility registry and automated Natural-trace screen only.
They do not estimate human acceptability, determine the final allocation, or
measure LLM task performance.

The final nested 4K/8K (16K optional) corpus, human-facing quality audit, and model training
have not been executed. Their tables below are explicit placeholders rather
than results; cell-level rows are added only after the taxonomy allocation is
frozen.

== Dataset Yield and Quality — Not Run

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    table.header([*Query family*], [*Requested*], [*Accepted*], [*Rejected*], [*Final review accept*]),
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
    [4B], [Answer-only], [—], [—], [—],
    [4B], [Checked Natural], [—], [—], [—],
    [4B], [Checked Symbolic], [—], [—], [—],
  ),
  caption: [Primary internal macro-accuracy table. Placeholder cells must not be interpreted as zero.],
) <results-main-sft>

Results are reported individually for each seed and with descriptive mean/range.
Item-level uncertainty conditions on the trained checkpoints. A second model
size is optional, not a prerequisite for the main result.

== Depth Extrapolation — Not Run

The common direct-answer protocol reports Direction, Selection, and Count at
depths 3–8. Depths 3/4 are matched anchors and 5–8 are extrapolation. Full-trace
process scores are not applicable to this module; gold certificates are checked
offline. No paired noisy suite is promised in the central study.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    table.header([*Family*], [*d3*], [*d4*], [*d5*], [*d6*], [*d7*], [*d8*]),
    [Direction], [—], [—], [—], [—], [—], [—],
    [Selection], [—], [—], [—], [—], [—], [—],
    [Count], [—], [—], [—], [—], [—], [—],
  ),
  caption: [Matched anchors and answer-only depth extrapolation.],
) <results-depth-ladder>

Depth results also report input length, premise count, and entity count. A drop
is not interpreted as a length-independent effect of logical depth alone.

== Held-Out Composition — Not Run

MP→TRC→MP, MP→DJE→TRC, and TRC→MP→TRC are the reserved skeletons,
each instantiated across DIR/SEL/CNT. They share the
construction provenance of the seen compositions, so the comparison does not
introduce a simultaneous premise-first shift. Results support transfer to those
three skeleton families only. No combined composition/depth SHIFT score is reported.

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

Two seeds provide replication but weak seed-level uncertainty. The LLM quality
screen is source-blinded, while the human appeal is selected from unanimous
rejections and therefore cannot estimate overall human acceptability or
LLM–human agreement.
External datasets differ in ontology, language, and answer contract, so their
results support bounded transfer claims only.
