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
split across development partitions used to test the artifact pipeline. Its
largest serialized artifact was 4,032 tokens under a 4,096-token engineering
limit. A separate 16-base premise-first holdout also passed admission.

Across the final generated pilot and holdout artifacts, 384 rows passed the
evaluation loader and their expected control outcomes: checked Symbolic traces
replayed, corrupted Symbolic reasoning failed while retaining its valid
decision, and Natural/answer-only process metrics were correctly inapplicable.
These checks establish that the proposed experimental interfaces execute; they
do not make Symbolic certificates training targets and do not estimate LLM
performance.

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

The sealed production release contains 8,000 Train, 775 Dev, and 4,300 Bench
bases, together with a 4,000-base nested training subset. Its final Natural
quality screen contains 485 items, all approved independently by both judges;
this is a model-based readability screen, not a population-level human rating.
The central analysis compares answer-only with verified Natural SFT on CORE,
COMP, and SHIFT. Each internal section reports its cumulative score and
cell-level breakdown; SHIFT slice labels and depth curves remain diagnostic.
External datasets remain separate from the internal benchmark. Gold certificates
remain offline checks throughout. A compact Chain-of-Symbol-like extension is
optional only if its frozen targets are shorter than Natural corpus-wide.

== SpatialEntail-SINGLE: First Frozen-Benchmark Result

The first completed internal benchmark run evaluates the prompt-matched untuned
Qwen3.5-4B baseline, `SpatialEntail-SINGLE-SFT` answer-only SFT, and
`SpatialEntail-SINGLE-SFT` verified Natural SFT on
`SpatialEntail-SINGLE-Bench`. The final score uses two-stage evaluation: the
model first produces its native response, then the same evaluated checkpoint
performs constrained one-letter answer selection. This makes semantic accuracy
separate from answer-format compliance.

#figure(
  table(
    columns: (1.45fr, auto, auto, auto, auto, auto, auto, auto),
    align: (x, y) => if x == 0 { left } else { center },
    table.header(
      table.cell(rowspan: 2)[*Model*],
      table.cell(colspan: 4)[*Benchmark sections*],
      table.cell(colspan: 3)[*Answer-selection diagnostics*],
      [*Overall*],
      [*CORE*],
      [*COMP*],
      [*SHIFT*],
      [*Stage 1 native*],
      [*Stage 1 format*],
      [*Stage 2 constrained*],
    ),
    [Untuned Qwen3.5-4B], [91.42%], [96.46%], [98.42%], [83.11%], [76.47%], [0.00%], [91.42%],
    [Answer-only SFT], [86.02%], [93.92%], [97.58%], [72.61%], [73.44%], [59.79%], [86.02%],
    [Verified Natural SFT], [97.91%], [99.46%], [99.75%], [95.56%], [95.23%], [94.28%], [97.91%],
  ),
  caption: [`SpatialEntail-SINGLE-Bench` accuracy and answer-selection diagnostics. Stage-2 constrained-choice accuracy is the primary metric; stage-1 columns are diagnostics.],
) <results-frozen-single>

The earlier all-zero untuned row was therefore an extraction artifact. The base
model often produced semantically usable answers, but did not use the canonical
`Answer: X` footer expected by the strict stage-1 scorer. Under constrained
answer selection, the untuned baseline is strong on CORE and COMP, reaching
96.46% and 98.42% respectively. These sections remain useful sanity checks and
cell-level diagnostics, but the first run suggests that they are not the main
source of discrimination for this base model.

SHIFT carries the clearest first-pass signal. Natural SFT reaches 95.56% on
SHIFT, compared with 83.11% for the untuned baseline and 72.61% for answer-only
SFT. Answer-only SFT is the ablation arm: it exposes the model to the same
questions and final labels as Natural SFT, but removes the checked worked
solution. This controls for ordinary supervised task exposure. Its
underperformance means that the Natural result cannot be explained by answer
labels alone. In this run, answer exposure by itself does not improve the
prompt-matched baseline and appears to reduce robustness on the structurally
shifted section. Natural SFT is the only tested supervision format that improves
both the overall benchmark and the SHIFT section.

The strongest remaining weakness is depth-labelled Count. Answer-only is near
zero on the count-depth cells, the untuned baseline remains between 24% and 40%,
and Natural ranges from 76% to 100%. Direction depth is close to solved by the
untuned and Natural models, so the useful difficulty is not simply longer
prompts. The first result supports a narrower claim than "teaching spatial
reasoning from scratch": verified Natural supervision improves robustness and
structural generalisation over both answer-only SFT and the prompt-matched
untuned baseline on `SpatialEntail-SINGLE-Bench`. The result remains a first
seed; seed replication and external transfer are still required before making
stronger claims. The natural next internal extension is
`SpatialEntail-ALL-POSSIBLE`, with `SpatialEntail-VISIBLE-POSSIBLE` reserved as
a menu-sensitive diagnostic.

== Discussion and Threats

The semantic audit already establishes that a latent-world answer can remain
possible without being textually entailed. The certificate pipeline establishes
that gold answers and stored audit artifacts can be replayed. It does not
establish that an LLM benefits from a particular supervision format.

The planned learning study remains limited by synthetic controlled language,
finite templates, incomplete automatic proof construction, shared data
definitions between solver and checker, and potentially selective context
admission. Natural model output has no process parser, and its wording does not
reveal the model's hidden computation.

Two seeds provide replication but weak seed-level uncertainty. The LLM quality
screen is source-blinded, while the human appeal is selected from unanimous
rejections and therefore cannot estimate overall human acceptability or
LLM–human agreement.
External datasets differ in ontology, language, and answer contract, so their
results support bounded transfer claims only.
