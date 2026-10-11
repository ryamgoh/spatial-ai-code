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

The first two are central comparisons on the primary 4B model. The latter two are optional
mechanism studies on the 4B model. Natural and Symbolic are described as
supervision packages because their tokenization, validation, and presentation
properties differ.

== Corpus and Benchmark Roles

SpatialEntail-Train and SpatialEntail-Bench are distinct candidate artifacts.
The existing 650-row pilot is development evidence only. The reviewed execution
protocol supersedes the earlier seventeen-thousand-example and 5.8K-test plans.
Candidate training budgets are a nested 4K/8K pair; 16K is optional future
expansion. Quotas are frozen only after structural diversity, semantic coverage,
and exact context admission pass. Equal cell counts describe an experimental
sampling distribution, not the population of all spatial reasoning problems.

The production assembler records exact role quotas, paired variants,
contamination inputs, canonical instance signatures, declared skeleton families,
software revision, and content hashes. It rejects insufficient unique diversity
rather than substituting renamed copies. Its output remains a candidate until
quality review and the final seal are complete.

== Models and Training Arms

The primary study uses Qwen3.5-4B with matched base problems and fixed epochs.
Answer-only, checked Natural, and checked Symbolic each run with seeds 42 and
43, giving six central SFT runs. The final fixed-epoch checkpoint is evaluated;
format-dependent token loss does not select different checkpoints. Appropriate
prompt-matched untuned baselines accompany each package. Natural versus
Symbolic compares supervision packages, not notation isolated from information
content, token cost, or replayability.

The 4K to 8K checked-Symbolic comparison is a bounded development-set budget
ablation. It does not prove two-point saturation or an optimal training size.
Unresolved uncertainty is reported. A second model size, local-proof and
corrupted-trace mechanisms, broader answer-contract training, and GRPO are
optional after the central study.

== Matched and Structural Evaluations

Four seen skeleton families (MP→TRC, IFF→TRC, TRC→MP, DJE→TRC) are crossed
with all three query contracts, producing twelve COMP cells. Three reserved
families (MP→TRC→MP, MP→DJE→TRC, TRC→MP→TRC) produce nine holdout cells.
Every component rule is seen; its longer dependency ordering is held out.
All use proof-template-first construction, separating composition from provenance.
Distinct checked steps must follow the declared order, each an ancestor of the next. Removing each
logical support premise must change the consistent query answer domain.
This establishes bounded semantic dependence, not globally shortest proofs.

Candidate internal evaluation has 1,300 CORE bases, 1,200 matched composition
bases, 900 held-out composition bases, and 900 depth bases: 4,300 total.
These remain quota targets pending full admission and diversity audits.
CORE entries include primary structures, semantic strata, and diagnostic noise;
no claim is made that the thirteen strata are thirteen equal primitives.

Depth evaluation uses fifty examples for each query family at depths 3 through
8. Depths 3 and 4 anchor the matched generator; depths 5 through 8 test
extrapolation. Input length, entity count, premise count, and achieved depth are
reported together. Deep Selection and Count traces exceed the existing output
budget, so this module uses a common direct-answer protocol. Gold certificates
are checked offline, and model process validity is not applicable. Full
Natural/Symbolic replay applies only to paired CORE/composition examples that
fit the frozen budget. No truncation or shallower backfill is allowed.

The benchmark is sealed before central model selection. Two grouping keys serve
different roles: a canonical instance signature excludes renamed/reordered
near-duplicates, while a declared skeleton signature reserves composition
families. Canonicalization is direction-sensitive and does not establish logical
equivalence or rotation invariance. Held-out results apply to the three frozen
skeleton families, not arbitrary unseen compositions. Engineering checkpoints and
previously inspected pilots are excluded from final evaluation and initialization.

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

Primary outcomes are separately reported cell-macro accuracy on matched CORE,
matched compositions, held-out compositions, depth curves, and strict
SpatialMap-TQA-Corr accuracy. Secondary metrics include per-capability/query results,
ambiguity precision/recall, option-position consistency, Symbolic parse and
replay rates, decision/domain consistency, full-trace validity, rank-witness
validity, generated tokens, runtime, and memory.

Both training seeds are reported individually, together with their mean and
range. Two seeds do not support a strong seed-level confidence interval. Paired
item-level bootstrap intervals are reported within each seed, and consistent directions are reported as observations from two runs. Item-level
intervals condition on the checkpoints and do not estimate training-run variance.
The two primary held-out comparisons are Natural and Symbolic versus answer-only,
with Holm adjustment within each seed. Cell-stratified paired resampling keeps
base variants together. At one hundred examples, cell profiles are descriptive;
a cell near fifty percent has roughly a ten-point confidence half-width.

== Source-Blinded Quality Review

Automatic semantic, certificate, round-trip, split, menu, and context checks
apply to every row. Let $C$ be the number of frozen reporting cells. Human-facing
quality is screened on at least the larger of 2% of the formally admitted
maximum candidate pool or $5 C$ examples, stratified with at least five examples
per cell. The numeric floor is therefore determined only after the cell registry
is frozen.

The same anonymized prompt and checked Natural trace are screened sequentially
by isolated terminal-agent runs using version-pinned `DeepSeek-V4-Pro` and
`GPT-5.5`. Each judge sees the frozen rubric and current item but not the other
judge's output. Capability, supervision arm, generation mode, source label,
expected hypothesis, and certificate metadata are hidden; item order is
randomized.

The structured rubric covers prompt clarity, answer-contract clarity,
readability, self-containment, unnecessary repetition, answer-before-evidence
leakage, and overall acceptability. Formal correctness is never delegated to an
LLM judge. Two approvals accept the item. A split verdict holds it out for
revision or replacement without involving the human. Two rejections route the
item to a separate payload-blinded human appeal, presented as a one-item
Approve/Reject menu without either model's reasons. Human approval records an
LLM consensus false-positive rejection; human rejection confirms removal or
revision.

Release requires 100% automatic validity and at least 95% final acceptance in
the quality sample after this routing. Two or more final non-acceptances in one
reporting cell trigger revision and a newly versioned review packet. The report
gives double-approval, split, double-rejection, human-override, and final
disposition rates, plus agreement and Cohen's kappa between the two model
judges. Because the human sees only unanimous rejections, the study does not
report overall human acceptability, LLM–human kappa, or a human false-acceptance
rate.

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
