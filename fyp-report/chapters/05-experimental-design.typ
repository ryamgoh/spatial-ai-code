#pagebreak(weak: true)
= Experimental Design

== Hypotheses and Contrasts

The experiment separates three questions that are easily conflated: whether
task exposure helps, whether valid reasoning evidence helps, and whether a
global rank witness is a useful training target.

#figure(
  table(
    columns: (auto, 1fr),
    inset: (x: 7pt, y: 5pt),
    table.header([*Contrast*], [*Interpretation*]),
    [Checked Natural vs. answer-only], [Effect of adding readable checked evidence to matched task exposure.],
    [Checked Symbolic vs. answer-only], [Effect of adding replayable typed evidence.],
    [Checked vs. corrupted Symbolic], [Whether semantic validity matters beyond structured answer-conditioned tokens.],
    [Proof plus ranks vs. local proof], [Whether constructing a global qualitative witness helps beyond query-local reasoning.],
  ),
  caption: [Predeclared supervision contrasts.],
) <experimental-contrasts>

The first two are central comparisons across model sizes. The latter two are
mechanism studies on the 4B model. Natural and Symbolic are described as
supervision packages because their tokenization, validation, and presentation
properties differ.

== Nested Data-Size Calibration

The maximum candidate budget is one 17K base-problem pool with deterministic
subsets:

$ 4"K" subset 8"K" subset 17"K". $

The final benchmark-cell registry and quotas are frozen only after a bounded
capability-coverage and generation-yield pilot. Once frozen, each subset
preserves the query, reasoning-structure, semantic-status, and evidence-burden
allocation using stratified prefixes and largest-remainder integer allocation.
Calibration trains only checked Symbolic using Qwen3.5-2B and Qwen3.5-4B with
seed 42 and development data.

The selected size is the smallest for which the next size improves both macro
answer accuracy and full Symbolic validity by less than two percentage points
for both models, without a material regression in a predeclared query family or
capability lens. Sparse cells and uncertainty are reported rather than hidden by
the aggregate. If 4K→8K saturates, 17K is not trained. Otherwise, 8K is selected
only if 8K→17K saturates; 17K is the fallback. The selected-size calibration
checkpoints are reused in the confirmatory matrix. Final test data are not
accessed during this decision. Exact cell quotas and the material-regression
threshold are not yet frozen; consequently, generation and calibration remain
`Not Run`.

== Models and Training Arms

The study uses the instruction-tuned `Qwen/Qwen3.5-2B` and
`Qwen/Qwen3.5-4B` checkpoints. Both use the same QLoRA recipe, two fixed epochs,
and training seeds 42 and 43. Model size is a robustness factor rather than a
claim of monotonic parameter scaling; total and trainable parameter counts are
reported.

Three central arms run on both sizes:

1. answer-only;
2. checked Natural; and
3. checked Symbolic with qualitative rank witness.

This gives $3 times 2 times 2 = 12$ confirmatory runs. Two mechanism arms run
only on 4B:

4. corrupted Symbolic; and
5. Symbolic local proof.

This adds $2 times 1 times 2 = 4$ runs, for sixteen confirmatory SFT runs. With
size calibration, the complete programme requires approximately eighteen to
twenty unique runs. Untuned prompt-matched checkpoints are evaluation baselines
and do not count as training runs. The local-proof comparison is evaluated only
on unique or entailed cells, where removing the qualitative rank witness does
not remove evidence required to express ambiguity.

The local-proof arm is a planned mechanism condition, not an exposed current
supervision variant. Before it enters the run matrix, its serializer and checker
must specify which consistency evidence remains required, how eligible base IDs
are paired, and which process-validity metrics remain comparable with the
rank-witness condition.

The same base IDs and epochs are used across arms. The experiment does not
repeat short answer-only targets to match Symbolic tokens. It instead reports
prompt tokens, supervised target tokens, optimizer updates, wall time, and peak
memory. Conclusions concern supervision packages under fixed task exposure,
not equal-compute notation effects.

== Internal Evaluation Regimes

=== Matched interpolation

A balanced interpolation test will sample new base problems across every frozen
reporting cell under the training distributions. Per-cell counts and the total
are not fixed until the post-pilot registry is frozen; they must be identical
across supervision arms and sufficient for separately reported query and
capability results.

=== StepGame-style depth ladder

Depth-controlled training stops at depth 4 for every central arm, mechanism
arm, and auxiliary demonstration. The frozen test covers all three query
families from depth 1 through 8:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    table.header([*Family*], [*Depths*], [*Per depth*], [*Clean total*]),
    [Direction], [1–8], [100], [800],
    [Selection], [1–8], [100], [800],
    [Count], [1–8], [100], [800],
    [*Total*], [], [], [*2,400*],
  ),
  caption: [Depth ladder. Depths 1–4 are matched; depths 5–8 are unseen-depth extrapolation.],
) <depth-ladder-design>

Direction depth is the required X/Y support depth. Selection depth is the entailed
membership-proof depth. Count depth is the positive membership depth underlying
a unique count; correlated-only counts without an entailed member remain in the
uncertainty suite rather than receiving a fabricated depth.

Every clean item has one matched noisy counterpart with identical core premises,
query, and answer plus two solver-verified removable distractors. The noisy set
adds 2,400 examples. Clean depth accuracy is the headline extrapolation curve;
the paired clean/noisy delta measures robustness separately.

The current generator exposes controlled removable distractors only for
Direction. The stated paired Selection/Count conditions are therefore implementation
requirements, not completed artifacts; the depth suite cannot be frozen until
those controls pass semantic and context-admission checks.

Any family/depth cell that cannot produce 100 context-admitted examples fails
before freezing. It is never backfilled with a shallower or different-family
problem.

=== Premise-first composition holdout

A separate 1,000-example test reserves 250 examples each for premise-first
conditional formulas, negation/disjunction, branching, and mixed compositions.
Every primitive operator appears during training; only selected formula trees
and proof-rule compounds are held out. This follows the systematic-composition
logic of CFQ rather than testing wholly unseen operators @keysers2020cfq.

== External Transfer

External datasets are evaluation-only and retain native contracts. Results are
never collapsed into one cross-dataset score.

#figure(
  table(
    columns: (auto, 1.2fr, 1.5fr),
    inset: (x: 6pt, y: 5pt),
    table.header([*Dataset*], [*Role*], [*Protocol*]),
    [SpatialMap-TQA-Corr], [Same-family audited transfer], [Strict answers and underdetermination calibration.],
    [StepGame], [Eight-direction multi-hop transfer], [Corrected 2024 release; hops 2–10 headline, one-hop overlap separate, overlap label separate.],
    [Text2Space], [Independent language/graph shift], [Description-only `full` queries; no ASCII or hidden-layout fields; verify query inferability.],
    [SpartQA-Human], [Human-authored open-world transfer], [Report YN Yes/No/DK, directional FR, and CO/FB separately.],
    [ReSQ], [Realistic human-language stress test], [Native binary scoring; map No to contradiction only with explicit evidence.],
  ),
  caption: [External evaluation suite and non-interchangeable task contracts.],
) <external-evaluation-suite>

No verified external Count analogue is claimed. Count generalisation is measured
internally. SODA/SPOD-Bench remains a related-work and future stress-test
comparator until a semantics-preserving public adapter can be frozen.

== Metrics and Statistical Reporting

Primary outcomes are macro strict accuracy across the frozen benchmark cells,
depth-5–8 extrapolation accuracy, premise-first holdout accuracy, and strict
SpatialMap-TQA-Corr accuracy. Secondary metrics include per-capability/query results,
ambiguity precision/recall, option-position consistency, Symbolic parse and
replay rates, decision/domain consistency, full-trace validity, rank-witness
validity, generated tokens, runtime, and memory.

Both training seeds are reported individually, together with their mean and
range. Two seeds do not support a strong seed-level confidence interval. Paired
item-level bootstrap intervals are reported within each seed, and an effect is
described as replicated only when its direction agrees across both seeds.

== Source-Blinded Quality Review

Automatic semantic, certificate, round-trip, split, menu, and context checks
apply to every row. Let $C$ be the number of frozen reporting cells. Human-facing
quality is reviewed on at least the larger of 2% of the selected training pool
or $5 C$ examples, stratified with at least five examples per cell. The numeric
floor is therefore determined only after the cell registry is frozen.

The same anonymized prompt and checked Natural trace are judged independently
by one version-pinned OpenAI model, one version-pinned Anthropic model, and one
human reviewer. Tier, supervision arm, generation mode, source label, model
identity, expected hypothesis, and certificate metadata are hidden; item order
is randomized.

The structured rubric covers prompt clarity, answer-contract clarity,
readability, self-containment, unnecessary repetition, answer-before-evidence
leakage, and overall acceptability. Formal correctness is never delegated to an
LLM judge. The report gives raw agreement, Fleiss' kappa for acceptability,
pairwise Spearman correlations for readability, and disagreement categories.
If LLM–human kappa is below 0.60, automated judgements remain descriptive.

Release requires 100% automatic validity and at least 95% human acceptability
overall. Any task with two or more unacceptable sampled examples is revised and
re-audited. This protocol is described as _source-blinded independent LLM
judging with human calibration_, not double-blind review.

== Relation to SODA's Data Process

SODA begins with 100 gold seeds for each of 13 benchmark tasks, generates four
grid-control families algorithmically, expands task-specific templates through
scripts and LLM paraphrasing, and processes 1,000-item batches
@bai2026soda. Fifty items per batch are manually inspected; a batch is retained
when at least 98% pass. Its unequal final volumes range from roughly 1.5K to
16K per task and reflect heterogeneous generation pipelines, especially four
16K grid-control sets. The 98% threshold is a quality gate, not a difficulty
weight.

SpatialEntail instead uses a predeclared factorial cell allocation, complete
formal checking, and a smaller source-blinded readability sample. This makes
coverage across queries, reasoning structures, semantic statuses, and evidence
obligations an explicit experimental choice.

== GRPO Gate

GRPO is not part of the main study. It is considered only after SFT if the
selected model retains meaningful error headroom, completion groups exhibit
non-degenerate outcome variance, and parsing is stable. Any later GRPO result
must be compared with compute-matched continued SFT.
