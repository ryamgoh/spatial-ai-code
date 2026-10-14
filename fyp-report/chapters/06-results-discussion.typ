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

== SpatialEntail-SINGLE: Two-Seed Frozen-Benchmark Result

The completed internal benchmark evaluates the prompt-matched untuned
Qwen3.5-4B baseline and two training seeds each of
`SpatialEntail-SINGLE-SFT` answer-only and verified Natural SFT on the same
4,300-example `SpatialEntail-SINGLE-Bench`. The final score uses two-stage
evaluation: the model first produces its native response, then the same
evaluated checkpoint performs constrained one-letter answer selection. This
separates semantic accuracy from answer-format compliance without changing the
checkpoint between stages.

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
    [Answer-only SFT, seed 42], [86.02%], [93.92%], [97.58%], [72.61%], [73.44%], [59.79%], [86.02%],
    [Answer-only SFT, seed 43], [99.44%], [99.92%], [99.75%], [98.89%], [97.63%], [96.63%], [99.44%],
    [Verified Natural SFT, seed 42], [97.91%], [99.46%], [99.75%], [95.56%], [95.23%], [94.28%], [97.91%],
    [Verified Natural SFT, seed 43], [96.77%], [99.54%], [99.67%], [92.83%], [96.14%], [98.07%], [96.77%],
  ),
  caption: [`SpatialEntail-SINGLE-Bench` accuracy across two training seeds. Stage-2 constrained-choice accuracy is the primary metric; stage-1 columns are diagnostics.],
) <results-frozen-single>

The earlier all-zero untuned row was an extraction artifact. The base model
often produced semantically usable answers, but did not use the canonical
`Answer: X` footer expected by the strict stage-1 format scorer. Under
constrained answer selection, the untuned baseline is already strong on CORE
and COMP, reaching 96.46% and 98.42% respectively. These sections remain useful
sanity checks and cell-level diagnostics, but are close to saturation for this
base model.

#figure(
  table(
    columns: (1.45fr, auto, auto, auto, auto),
    align: (x, y) => if x == 0 { left } else { center },
    table.header(
      [*Supervision arm*],
      [*Mean overall*],
      [*Observed overall range*],
      [*Mean SHIFT*],
      [*Observed SHIFT range*],
    ),
    [Answer-only SFT], [92.73%], [86.02--99.44%], [85.75%], [72.61--98.89%],
    [Verified Natural SFT], [97.34%], [96.77--97.91%], [94.19%], [92.83--95.56%],
  ),
  caption: [Descriptive two-seed means and observed ranges. Two seeds are insufficient for a strong variance estimate.],
) <results-frozen-single-seeds>

The two seeds change the interpretation of the ablation. Verified Natural SFT
is consistently strong: seeds 42 and 43 score 97.91% and 96.77% overall, both
above the 91.42% untuned baseline. Its two-seed mean is 97.34%, with an observed
range of 1.14 percentage points. On SHIFT, both Natural checkpoints improve on
the 83.11% baseline, producing a 94.19% mean and a 2.72-point observed range.
This is evidence of replication across the two tested seeds, although it is not
a population-level estimate of seed variance.

Answer-only SFT behaves very differently. Seed 42 falls below the untuned model
at 86.02%, whereas seed 43 is the strongest individual checkpoint at 99.44%,
making only 24 errors across 4,300 examples. The corresponding 13.42-point
overall range expands to 26.28 points on SHIFT. The seed-43 result is not merely
a constrained-decoding rescue: it already reaches 97.63% native stage-1
accuracy and 96.63% canonical-format validity. Answer-only therefore remains a
valid matched-exposure ablation, but the two runs do not support the earlier
claim that answer labels alone cannot produce the improvement. Instead, they
show that answer-only supervision can reach very high accuracy while being much
less stable across the observed seeds. Natural has the stronger two-seed mean
and substantially narrower observed range, but does not win every individual
run.

SHIFT still carries most of the discriminative signal. Natural seed 43 remains
near-perfect on CORE and COMP but falls with unseen depth: Count declines to
72%, 70%, and 66% at depths 6--8, while Selection reaches 78%, 72%, and 68%.
Natural seed 42 is stronger on these depth cells, yet both Natural checkpoints
remain above the untuned SHIFT aggregate. Answer-only exposes the instability
most clearly: seed 42 is near zero on depth-labelled Count, while seed 43 scores
between 92% and 98% on the same cells. Direction depth is close to solved in
most conditions, so the remaining difficulty is not simply longer prompts but
deep Selection and Count reasoning.

The result supports a narrower claim than "teaching spatial reasoning from
scratch": on `SpatialEntail-SINGLE-Bench`, verified Natural supervision gives
consistently strong performance and structural-shift improvement across the two
tested seeds, whereas answer-only supervision has a higher observed sensitivity
to training seed. The high untuned score and near-saturated tuned results also
limit further discrimination under the single-answer contract. This motivates
`SpatialEntail-ALL-POSSIBLE` as the next harder internal benchmark, with
`SpatialEntail-VISIBLE-POSSIBLE` retained as a menu-sensitive diagnostic;
external transfer remains necessary for broader generalisation claims.

== SpatialEntail-ALL-POSSIBLE: Zero-Shot Contract-Transfer Pilot

The next diagnostic asks whether a checkpoint trained only on
`SpatialEntail-SINGLE-SFT` transfers to a set-valued answer contract. The draft
`SpatialEntail-ALL-POSSIBLE` calibration pilot contains 34 cells with 25 fresh
examples per cell, for 850 examples in total. Every question displays exactly
five options, but the correct output is the complete non-empty subset of
options that remain possible. This is an engineering pilot rather than a frozen
benchmark: its purpose is to test transfer and cell difficulty before fixing a
final allocation.

#figure(
  table(
    columns: (1.45fr, auto, auto, auto, auto, auto, auto),
    align: (x, y) => if x == 0 { left } else { center },
    table.header(
      [*Model*],
      [*Exact set*],
      [*Stage 1 exact*],
      [*Stage 1 format*],
      [*Gold covered*],
      [*Over-inclusive*],
      [*Missing gold*],
    ),
    [Untuned Qwen3.5-4B], [93.65%], [65.65%], [0.00%], [94.94%], [1.29%], [5.06%],
    [Natural SINGLE-SFT, seed 42], [14.59%], [10.71%], [68.00%], [15.88%], [1.29%], [84.12%],
  ),
  caption: [`SpatialEntail-ALL-POSSIBLE` zero-shot transfer pilot. Stage-2 constrained exact-set accuracy is primary. Gold-covered predictions contain every correct option; over-inclusive predictions cover the gold set but add at least one option; missing-gold predictions omit at least one correct option.],
) <results-all-possible-transfer>

The untuned model reaches 93.65% exact-set accuracy, so expanding the answer
space alone did not make the current draft difficult for the base model. Its
stage-1 exact-set score is lower at 65.65%, while canonical stage-1 format
validity is zero; as in SINGLE, the second constrained pass separates semantic
selection from a preferred answer footer. The 94.94% gold-covered rate and
1.29% over-inclusive rate show that its remaining errors are not explained by
simply selecting every option.

The Natural seed-42 adapter behaves in the opposite way. It produces the
canonical stage-1 format on 68.00% of examples, yet reaches only 10.71% native
and 14.59% constrained exact-set accuracy. Its predictions omit at least one
gold answer on 84.12% of examples, while only 1.29% are over-inclusive. The
failure is therefore not primarily formatting or indiscriminate over-selection.
It is consistent with answer-contract specialisation: the adapter was trained
to resolve and emit one answer under SINGLE, then was asked zero-shot to return
a complete set.

#figure(
  table(
    columns: (1.45fr, auto, auto, auto),
    align: (x, y) => if x == 0 { left } else { center },
    table.header([*Slice*], [*$N$*], [*Untuned*], [*Natural seed 42*]),
    [Direction], [275], [89.09%], [13.45%],
    [Selection], [275], [99.27%], [30.18%],
    [Count], [300], [92.67%], [1.33%],
    [Gold set size 2], [325], [99.08%], [28.00%],
    [Gold set size 3], [25], [100.00%], [0.00%],
    [Gold set size 4], [300], [93.67%], [6.33%],
    [Depth 3], [75], [85.33%], [8.00%],
    [Depth 5], [75], [77.33%], [6.67%],
    [Joint Count reasoning], [50], [100.00%], [2.00%],
    [Removable noise], [50], [92.00%], [6.00%],
  ),
  caption: [ALL-POSSIBLE exact-set accuracy by query family and overlapping structural diagnostics.],
) <results-all-possible-breakdown>

Count shows the sharpest transfer failure: the untuned model scores 92.67%,
whereas Natural seed 42 scores 1.33%. Natural also falls from 28.00% with two
gold answers to 6.33% with four, and reaches zero on the small three-answer
Joint Count cell. The same adapter remains strong on the frozen SINGLE
benchmark, so this pilot does not establish a general loss of spatial reasoning
or a defect in Natural worked solutions. It establishes the narrower point that
successful SINGLE SFT does not automatically transfer to complete set-valued
prediction.

No answer-only arm or additional training seed is added to this calibration
pilot. Answer-only is the matched-supervision ablation for the completed SINGLE
study; this pilot instead tests whether the selected Natural SINGLE checkpoint
and the untuned base already satisfy a new answer contract. Repeating the same
diagnostic would not address the more immediate design result: the Natural
checkpoint needs explicit multi-answer exposure, while the untuned model nearly
saturates the current draft. Consequently, these values are retained as pilot
evidence rather than final benchmark scores. The ALL-POSSIBLE cells must be
audited and hardened before freezing, and any subsequent MULTI-SFT study must
evaluate set completeness directly.

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
