#pagebreak(weak: true)
= Experimental Design

== Hypotheses and Contrasts

The experiment separates whether task exposure helps from whether a verified
worked solution helps. A compact symbolic representation is an optional
token-efficiency extension rather than a central formal-reasoning claim.

#figure(
  table(
    columns: (auto, 1fr),
    inset: (x: 7pt, y: 5pt),
    table.header([*Contrast*], [*Interpretation*]),
    [Verified Natural vs. answer-only], [Effect of adding a readable, query-directed worked solution to matched task exposure.],
    [Compact Symbolic vs. Natural (optional)], [Whether a CoS-like representation reduces target and generation tokens while retaining accuracy.],
  ),
  caption: [Predeclared supervision contrasts.],
) <experimental-contrasts>

Verified Natural versus answer-only is the completed
`SpatialEntail-SINGLE-SFT` comparison on the primary 4B model. Randomized v2
retains this contrast for the SINGLE contract at seeds 42 and 43. MULTI uses
Natural supervision only and is not treated as a second supervision-format
ablation; its comparators are the prompt-matched untuned model and Natural
SINGLE-SFT transfer baseline.
Prompt-matched untuned evaluation provides context rather than another SFT
arm. Compact Symbolic is optional and proceeds only after its
notation and corpus-wide token reduction are demonstrated. Replayable NDJSON
answer certificates are not an SFT arm.

== Corpus and Benchmark Roles

`SpatialEntail-SINGLE-SFT` and `SpatialEntail-SINGLE-Bench` are distinct sealed
artifacts. The production release contains 8,000 training bases, a nested
4,000-base subset, 775 development bases, and 4,300 benchmark bases. The
reviewed execution protocol supersedes the earlier seventeen-thousand-example
and 5.8K-test plans. Equal cell counts describe an experimental sampling
distribution, not the population of all spatial reasoning problems.

The production assembler records exact role quotas, paired variants,
contamination inputs, canonical instance signatures, declared skeleton families,
software revision, and content hashes. It rejects insufficient unique diversity
rather than substituting renamed copies. The production output passed quality
review and sealing before job preparation.

== Models and Training Arms

The primary study uses Qwen3.5-4B with matched base problems and fixed epochs.
Answer-only and verified Natural are the central SFT arms. The final fixed-epoch
checkpoint is evaluated; training loss does not select different checkpoints.
Appropriate prompt-matched untuned baselines accompany each response contract.

Compact Symbolic and GRPO are optional after the central study. Compact Symbolic
is compared with Natural on answer accuracy, target/generated tokens, runtime,
and memory; it is not evaluated as a formal certificate format.

== Matched and Structural Evaluations

Four seen skeleton families (MP→TRC, IFF→TRC, TRC→MP, DJE→TRC) are crossed
with all three query contracts, producing twelve COMP cells. Three reserved
families (MP→TRC→MP, MP→DJE→TRC, TRC→MP→TRC) produce nine holdout cells.
Every component rule is seen; its longer dependency ordering is held out.
All use proof-template-first construction, separating composition from provenance.
Distinct checked steps must follow the declared order, each an ancestor of the next. Removing each
logical support premise must change the consistent query answer domain.
This establishes bounded semantic dependence, not globally shortest proofs.

Internal evaluation has three formal groups: 1,300 CORE bases, 1,200 COMP bases,
and 1,800 SHIFT bases, for 4,300 total. SHIFT combines 900 held-out-composition
and 900 depth-ladder bases; these labels remain cell metadata, not extra
headline groups.
These quotas passed admission and diversity audits before sealing.
CORE entries include primary structures, semantic strata, and diagnostic noise;
no claim is made that the thirteen strata are thirteen equal primitives.

Depth evaluation uses fifty examples for each query family at depths 3 through
8. Depths 3 and 4 anchor the matched generator; depths 5 through 8 test
extrapolation. Input length, entity count, premise count, and achieved depth are
reported together. Deep Selection and Count worked solutions can exceed the
output budget, so this module uses a common direct-answer protocol. Gold
certificates are checked offline, and model process validity is not required.
No truncation or shallower backfill is allowed.

The benchmark is sealed before central model selection. Two grouping keys serve
different roles: a canonical instance signature excludes renamed/reordered
near-duplicates, while a declared skeleton signature reserves composition
families. Canonicalization is direction-sensitive and does not establish logical
equivalence or rotation invariance. Held-out results apply to the three frozen
skeleton families, not arbitrary unseen compositions. Engineering checkpoints and
previously inspected pilots are excluded from final evaluation and initialization.

== Randomized Proof-Graph v2 Extension

The fixed v1 registry remains the evidence source for its reported
CORE/COMP/SHIFT results; v2 does not reuse those labels. It samples ordered
linear paths over MP, IFF elimination, disjunctive elimination,
double-negation elimination, and transitive relational composition. Case split
remains a measured branching rule rather than an interchangeable path token.
The replayed certificate measures semantic dependency length $L$, while $X$ counts
atomic distractor premises whose joint removal leaves the complete answer
unchanged. Support size and proof size remain continuous diagnostics.

Training is restricted to L1--L6, X0--X6, and training-seen ordered
transitions. Each query/gold evaluation group has 120 fresh matched examples
and three disjoint sixty-example OOD partitions. LENGTH-OOD changes only L to
L7--L8; NOISE-OOD changes only X to X7--X8; COMPOSITION-OOD introduces exactly
one held-out ordered transition while retaining training L/X ranges. A
predeclared exponential maximum-entropy tilt favours the upper end without
using model accuracy to select items.

A held-out ordered transition requires two consecutive sampled prefix steps.
Direction, Selection, and SINGLE Count therefore permit COMPOSITION-OOD from
L2. ALL-POSSIBLE Count first spends G certificate steps establishing its
answer-set backbone, so its feasible composition minima are G2/L4, G3/L5, and
G4/L6. The allocation excludes lower cells rather than repeatedly sampling an
impossible transition requirement.

A 100-seed sweep of the minimum-length, X0 ALL-POSSIBLE Count construction
observed four canonical structures under the frozen direction-sensitive
isomorphism key. This is an empirical-operational cap rather than a proved
upper bound. The design conservatively caps each such exact L/X pairing at four
rows while preserving the predeclared L and X marginals. Deterministic pairing
retries only their joint association; it does not drop rows or weaken
final-benchmark uniqueness.

Complete Count certificates impose a feasibility boundary: G2/G3/G4 have
audited minimum lengths L2/L3/L4 under the current replay obligations. Quotas
are redistributed over each query/gold group's feasible L values rather than
creating unrealizable cells. MULTI training normalizes its exponential L tilt
within each gold size's feasible interval; imposing one global L marginal would
force a misleading U-shaped G1 allocation. The headline candidate contains 900 SINGLE and
2,700 ALL-POSSIBLE bases and follows a 2,400-example capability calibration.

VISIBLE-POSSIBLE is excluded from SFT and retained as a third, non-headline
evaluation contract. After the headline candidate is frozen, a separate,
structurally disjoint pool contributes five bases from each
query-by-G-by-partition stratum, giving 180 diagnostic bases. Each G2--G4 base
produces one ALL-POSSIBLE control and every VISIBLE level V0 through VG. This
gives 720 VISIBLE views plus 180 paired ALL controls, or 900 prompts in total.
V0 requires `None of the Options`, intermediate levels test partial visibility,
and VG isolates the change in instruction wording when all possible answers
remain displayed. All views retain the same problem, semantic analysis,
certificate, and five-option A--E menu.

This pairing requires a larger underlying answer universe than the displayed
menu: Selection uses at least G+4 query candidates, and Count uses at least G+3
candidate entities because its answer universe spans zero through the candidate
count. Direction already has eight possible direction values. Evaluation covers
the untuned model, Natural SINGLE-SFT, and Natural MULTI-SFT; existing SINGLE
answer-only checkpoints are secondary transfer references rather than another
training arm. Exact-set accuracy is reported by query, G, visibility V, and
benchmark partition. A paired consistency metric compares the VISIBLE result
with its diagnostic ALL control restricted to the displayed menu, and
resampling keeps all views of one base together.

Direct L0, ambiguous SINGLE, and true Selection no-match cases remain a
separate special-response diagnostic. A binary diagnostic evaluates 280
clean/dirty strict-cycle pairs; inconsistent premises are never mapped to
Cannot be determined or None of the Options.

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

Primary v1 outcomes are separately reported accuracy on CORE, COMP, SHIFT, and
SpatialMap-TQA-Corr. Randomized v2 instead reports MATCHED, LENGTH-OOD,
NOISE-OOD, and COMPOSITION-OOD, together with macro-L and macro-X curves.
Secondary metrics include per-capability/query
results, ambiguity precision/recall, option-position consistency, generated
tokens, runtime, and memory. If compact Symbolic runs, its token reduction and
answer accuracy are reported together.

Both training seeds are reported individually, together with their mean and
range. Two seeds do not support a strong seed-level confidence interval. Paired
item-level bootstrap intervals are reported within each seed, and consistent directions are reported as observations from two runs. Item-level
intervals condition on the checkpoints and do not estimate training-run variance.
For SINGLE, the primary held-out supervision comparison is Natural versus
answer-only. MULTI instead compares Natural MULTI-SFT against the prompt-matched
untuned model and Natural SINGLE-SFT transfer baseline. Cell-stratified paired
resampling keeps base variants together. At one hundred examples, cell profiles
are descriptive; a cell near fifty percent has roughly a ten-point confidence
half-width.

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

SpatialEntail v1 instead uses a predeclared factorial cell allocation, complete
formal checking, and a smaller source-blinded readability sample. Randomized v2
retains predeclared marginal quotas but samples complete feasible proof paths
uniformly within them. Both designs fix selection rules before model evaluation.

== GRPO Gate

GRPO is not part of the main study. It is considered only after SFT if the
selected model retains meaningful error headroom, completion groups exhibit
non-degenerate outcome variance, and parsing is stable. Any later GRPO result
must be compared with compute-matched continued SFT.
