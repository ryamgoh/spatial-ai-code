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

=== Hardened untuned calibration

The response audit found that 600 of the first pilot's 850 examples placed the
complete query-level alternative set in one visible Boolean disjunction. Those
cells measured basic contract reading rather than composed possible-world
reasoning. A separate hardening registry therefore preserves the first pilot as
diagnostic evidence and introduces 14 new cells: five COMP cells combine
exclusive cases with relational chains or complete correlated Count worlds,
while nine SHIFT cells extend Direction to depth 9 and Selection/Count to depth
7. The prompt explicitly states the open-world and qualitative-distance
semantics, and every cell still uses a fixed five-option menu.

The hardened engineering pool contains 25 examples per cell, or 350 examples.
It was evaluated only on untuned Qwen3.5-4B because its purpose is benchmark
calibration, not another supervision ablation. All prepared-file hashes and all
14 task outputs validate against the generation receipt.

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
    [Untuned Qwen3.5-4B], [82.29%], [42.29%], [0.57%], [88.57%], [6.29%], [11.43%],
  ),
  caption: [Untuned accuracy on the 350-example hardened ALL-POSSIBLE calibration pool. Stage-2 constrained exact-set accuracy remains primary.],
) <results-all-possible-hardened>

Hardening reduces untuned exact-set accuracy from 93.65% to 82.29%. Of the 350
examples, 148 are exact in both stages, 140 additional decisions are recovered
by constrained stage 2, and 62 remain semantically incorrect. Stage 2 does not
degrade any stage-1 exact decision. Among the remaining errors, 40 omit at least
one gold option and 22 contain the gold set plus one or more extras. Response
inspection shows depth-related semantic failures and unstable repetitive
reasoning, commonly collapsing a multi-answer result to one option or expanding
it to all five; the discrepancy is not a scorer or oracle mismatch.

#figure(
  table(
    columns: (1.45fr, auto, auto),
    align: (x, y) => if x == 0 { left } else { center },
    table.header([*Slice*], [*$N$*], [*Exact-set accuracy*]),
    [COMP], [125], [92.00%],
    [SHIFT], [225], [76.89%],
    [Direction], [75], [53.33%],
    [Selection], [150], [85.33%],
    [Count], [125], [96.00%],
    [Depth 2], [50], [84.00%],
    [Depth 3], [50], [82.00%],
    [Depth 5], [75], [82.67%],
    [Depth 7], [75], [80.00%],
    [Depth 9], [25], [40.00%],
  ),
  caption: [Hardened ALL-POSSIBLE accuracy by overlapping section, query-family, and depth slices.],
) <results-all-possible-hardened-breakdown>

The resulting gradient is useful rather than uniformly difficult. COMP remains
a strong matched/compositional check at 92.00%, whereas SHIFT falls to 76.89%.
Direction carries the clearest headroom: its D5 and D7 cells each score 60%, and
D9 scores 40%. Selection falls from 99.27% in the first pilot to 85.33% after
removing direct enumeration. Count remains close to saturation at 96.00%, so
a separate two-cell Count calibration tests independent correlated factors
without redesigning the successful Direction and Selection cells.

The correlated Count G3 cell reaches 88% exact-set accuracy, while G4 remains
fully solved at 100%; combined accuracy is 94%. The three G3 errors reflect one
incorrect independence assumption, one incorrect merger of the latent groups,
and one repetitive over-inclusive response. This provides modest Count
headroom, but also confirms that Count is intrinsically easier for this base
model under the current ontology. The fixed 30-cell freeze candidate is
therefore retained only as engineering history. Its `D` control plants a longer
relational chain, but simultaneously increases premises, entities, removable
noise, and prompt length; it cannot by itself identify general proof depth as
the cause of the D9 drop.

The replacement v2 design samples ordered linear paths over MP, IFF
elimination, disjunctive elimination, double-negation elimination, and
transitive relational composition, then admits them by their replayed
certificates. Case split remains a measured branching rule rather than a
sampled path token. Semantic length $L$ is the longest semantic dependency path;
proof bookkeeping does not increase $L$. Verified distractor count $X$ is kept separate. Training covers
L1--L6, X0--X6, and training-seen ordered transitions. Four disjoint evaluation
partitions then compare fresh matched structures against one-factor length
(L7--L8), noise (X7--X8), or ordered-transition OOD cases. Complete feasible
rule paths are sampled with dynamic-programming completion counts rather than
selected after observing model failures.

Executable capability probes also expose a necessary non-factorial boundary:
complete G2, G3, and G4 Count certificates currently have minimum semantic
lengths L2, L3, and L4 because excluded counts require assignment refutations.
The allocation preserves each query/gold total by redistributing examples over
its feasible L range instead of fabricating impossible cells. Support size,
proof size, branching, and token length remain continuous diagnostics rather
than new task labels or a composite difficulty score.

=== Rejected first full randomized-v2 calibration

The first full L6/X6 calibration generated all 2,400 requested rows but was
rejected after a row-level audit. All rows matched their requested L and X,
used five-option menus, passed prompt round trips, and contained no sampled CSR
steps. For each query family, 23 of the 25 ordered linear transitions were
eligible and succeeded in 20/20 attempts. The two excluded transitions were
the expected typed limitations `TRC→DJE` and `TRC→DN`.

The run used 128 workers and recorded 155.8 seconds for row generation. It
accepted 2,400 rows from 3,052 candidate trials, an aggregate acceptance rate
of 78.64%. Two thousand rows succeeded on their first attempt; the
95th-percentile rejection count was two and the maximum was eleven. It produced
2,395 unique canonical structures. The five surplus duplicates were confined
to four minimum-length, X0 Count collision groups and would have been replaced
under the stricter headline-candidate policy.

The blocking defect concerned gold-set preservation. Twenty-eight rows (1.17%)
had fewer solved answers than their declared G. Selection contributed 24:
five declared G2 rows realised G1, ten G3 rows realised G1 or G2, and nine G4
rows realised G2 or G3. Count contributed four: two G3 rows realised G2 and
two G4 rows realised G3. A deterministic Selection G2/L2/X0 reproduction
showed that the exclusion-first base correctly allowed both planted answers,
after which a lifted random proof prefix reused a query candidate and made one
answer impossible. Admission checked only whether the result remained broadly
ambiguous; it did not assert that the solved answer-set size still equalled the
requested G.

This run is retained as engineering evidence but is not used to freeze the
benchmark or generate SFT data. The required correction is to isolate
proof-prefix auxiliary objects from query candidates, add an exact post-solver
G check, and lock the recorded seed into a regression test before repeating the
calibration. The
row-level audit also measured Natural-target whitespace counts of median 771,
p95 1,080, and maximum 1,465, and complete-message counts of median 1,004 and
maximum 1,746. These values are lexical diagnostics rather than exact counts
from the pinned Qwen tokenizer.

=== Lessons and next actions

The two pilots show that a larger answer space is not sufficient by itself:
difficulty comes from how alternatives are derived. Directly displaying a gold
disjunction leaves little headroom, while branch composition and qualitative
depth expose substantial weaknesses. They also show that successful SINGLE SFT
does not imply set-complete prediction: the Natural seed-42 transfer failure is
dominated by omitted answers. MULTI-SFT must therefore include explicit
set-valued supervision and be evaluated by exact-set completeness.

The generation run also measures the prospective Natural supervision cost
without culling any candidate. Across 350 full Natural rows, the complete
training sequence has median 2,172 tokens, p95 19,202, p99 19,925, and maximum
20,264. Seventy-five rows exceed 12,288 tokens, 62 exceed 16,384, and only three
exceed 20,000; none exceeds 24,576. A 24,576-token MULTI-SFT context would
therefore admit this pool in full, although shortening the three longest
correlated-Count renderings remains an alternative. Certificates remain offline
audit artifacts; full Natural source rows and answer-only evaluation views are
stored separately.

This remains an engineering pool, not the final benchmark. Three of its 350
rows repeat another canonical structure within one Selection composition cell,
even though their surface names and option mappings differ. Randomized v2 must
first pass its 2,400-example structural calibration and transition capability
gates. It then prepares 900 fresh SINGLE and 2,700 fresh ALL-POSSIBLE headline
examples, audits canonical uniqueness and structural disjointness, and freezes
their hashes before generating SFT data. SINGLE-v2 retains the matched
Natural-versus-answer-only ablation at seeds 42 and 43 because the v1
answer-only result was strongly seed-sensitive. MULTI-v2 uses Natural
supervision only and is compared with the prompt-matched untuned model and
Natural SINGLE-SFT transfer baseline.

VISIBLE-POSSIBLE is implemented as a paired menu view of an ALL-POSSIBLE base
but is deferred from the first v2 SFT study. Special L0, ambiguous, no-match,
and zero-visible cases form a separate diagnostic. Inconsistent strict cycles
are never mislabeled as no-answer spatial questions: a separate paired
consistency diagnostic adds one closing edge to a satisfiable chain and requires
deletion-minimal contradiction evidence.

Randomized v2 also separates semantic depth from proof branching. Case split
is measured in realised certificates but is not sampled as a linear
transformation; treating it as interchangeable would create accidental branch
expansion rather than a controlled L curve. Direction
and Selection construct ALL-POSSIBLE answers through necessary non-gold
exclusions, while Count retains complete correlated-assignment evidence with
five declared candidates. Fresh leaf objects make X distractors constructively
removable, unused construction names are pruned before solving, and the known
base-skeleton offset avoids recertifying the base on every retry. On
representative local L6/X6 cases, these changes reduced generation from 4.62s
to 0.84s for SINGLE Direction, 14.12s to 1.42s for ALL Direction G4, 69.24s to
6.66s for ALL Selection G4, and 5.28s to 3.37s for ALL Count G4.

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
