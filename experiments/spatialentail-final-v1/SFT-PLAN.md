# SFT Plan

**Status: generation-first CPU workflow implemented; no training-ready release
or final-v1 SFT workflow yet.** This document owns the scientific requirements.
See [README.md](README.md) for generation commands, artifact layout and tested
scope. Full server generation and remaining release/runtime checks are pending.

## 1. Objective And Boundaries

Train two matched QLoRA adapters to answer atomic spatial entailment questions:
Natural explanations versus Symbolic explanations. Measure answer accuracy,
depth generalization and generated-token cost. The primary contribution remains
the solver-verified dataset; Symbolic is not assumed to be shorter or better.

Preserve the intended composition study. This reset adds an auditable release
boundary and trainer checks, not a new hypothesis, hyperparameter search,
multi-seed campaign, answer-only arm, RL phase or possibility-based task.
The study scope and user-confirmed changes need no separate reapproval.
Required validation still applies; this plan does not certify that its checks
are implemented or have passed. Scope changes or exceptions still need an
explicit user decision.

## 2. Reuse And Provenance

The inherited specification is [Unique-only v1](../spatialentail-composition-v1/UNIQUE-ONLY-v1.md).
This plan's revised distributions take precedence. Preserve historical data,
receipts and results; never resume from the failed adapter.

Initial source inspection used commit
`4f3b1db7403730d8495cf6e4edc6ecd79986c041` on 2026-11-02. Shared generation,
solver, evidence, native-thinking preparation and two-stage evaluation helpers
are reusable, but their experiment paths and old inventory assumptions need
parameterization. Epoch dev integration and actual loader checks remain work.
Export equality receipts are not independent SMT verification.
The candidate-generation workflow now lives in
`spatial/v2/entailment_generation.py`, configured explicitly by final-v1's
`generation_policy.py` through a thin `generate.py` launcher. Shared independent
audit functions live in `spatial/v2/entailment_artifact_audit.py`; historical
import paths delegate there. Source fingerprints change with this extraction;
historical receipts and payloads must remain unchanged.
CPU generation uses PEP 723 dependency metadata in the thin launcher and uv's
separate cached script environment. It retains Z3 4.16.0.0; the root GPU
dependency definitions and lock remain unchanged. Launcher fingerprints bind
the direct pins, and manifests record installed runtime versions.

Historical admission uses a Qwen3.5 tokenizer before active-model preparation.
Retaining that gate requires explicit provenance; removing it requires a
versioned admission change. All final tokens must be checked with the pinned
active tokenizer regardless. Old audits do not certify new artifacts.

## 3. Semantic Contract

Use `spatialentail-entailment-v1`, scope `unique-only-entailment-v1`, prompt
`plain-unique-system-rules-v1`. Premises are flat conjunctions of exact compass
relations between distinct spatial points, with open-world semantics. Which and
Count range over every nonreference object in the scene, not only menu entries.
No hidden construction coordinates may determine gold or enter the prompt.

Check consistency first. For consistent premises compute the complete set T of
individually entailed domain values, independently of the menu:

| Condition | Label | Required evidence |
|---|---|---|
| Inconsistent premises | E | Checked contradiction; E is a convention here. |
| T empty | E | Checked non-entailment across the complete domain. |
| At least two values in T | E | Prove two distinct qualifying objects; possible only for Which. |
| Exactly one value in T, displayed | A-D | Sole entailment and correct option mapping. |
| Exactly one value in T, unlisted | F | Sole entailment and absence from A-D. |

Sole Which entailment permits contingent alternatives. Every other eligible
object must be shown **not entailed**, which is not the same as impossible.
Exact Count is invariant total membership across satisfying worlds, not a count
of individually entailed objects. Count zero is a real sole value. Unknown,
timeout, unsupported proof and audit failure are errors, never E or F.

Keep short questions, ordinary place names and an unnumbered premise paragraph.
System rules explain entailment and E/F. P citations belong in explanations
only. Both styles use the same evidence obligations and end with `Answer: X`.

## 4. Fixed Data Design

### Training

6,000 distinct canonical underlying questions, one selected menu each. Each arm
receives exactly these same 6,000 questions, menus, labels, evidence and ordering.
Only explanation style and the corresponding style instruction differ.

All synthetic splits use the same controlled generation framework, with
disjoint questions, scenes and families. Per training query, retain 1,400 depth-cohort
and 600 semantic-cohort questions. Which's depth cohort contains 900 A-D,
200 F and 300 multiple-entailment E; its semantic cohort contains 300 A-D,
225 no-entailment E and 75 inconsistent E. These evidence cohorts do not
segregate semantic cases into easy versus hard construction levels.

The user now requires matched synthetic answer distributions and semantic
cases spread across all assigned levels. This versioned policy,
`matched-controlled-uniform-levels-v2`, supersedes the old fixed
test inventories, mixed-source targets and weighted per-depth allocation tables.
The historical frozen policy is a reference, not an equality target for the
new joint allocation. Retain Count 0-4 coverage, 50-family floors per applicable
training proof-depth stratum, direction coverage and compatible evidence
minimums. Audit any conflict explicitly; do not weaken these silently.
A-D remains the 60% answer group. Within that group, balance correct A/B/C/D
positions per split/query where feasible by permuting valid menus, never by
changing semantic labels or duplicating questions. The per-letter targets are
300 train, 60 dev, 120 in-range and 60 extrapolation per query. Freeze menu
order before model evaluation and use identical menus in both arms.

Depth is verified shortest atomic-axis support, not prompt or trace length.
Which multiple-entailment E uses its cheapest two-member certificate. Ambiguity
and inconsistency do not receive fabricated depth labels. An unseen primary-menu
explanation cannot supply depth credit to a selected omission view.

Keep the declared deterministic slot schedule and at most three constructions
per slot: at most 18,000 constructions and exactly 6,000 successful admissions.
Fail a release on shortage or rejected selected rows; no silent refill,
relabeling, quota relaxation or moving failed rows into another split.
After a fix, create a new candidate version and rerun the complete audit.

### Evaluation

Dev is held-out validation evaluated during SFT at each epoch, not a final
benchmark or a source of gradient updates. **SFT-Bench** consists only of the
in-range test and extrapolation test; train and dev are excluded. After training,
evaluate the selected model on SFT-Bench and separately on SpatialMap-TQA-Corr.

Use one selected menu per underlying question in every split. To construct F,
generate a valid sole-answer question and omit its correct answer from A-D;
retain only that menu, not both the original and omitted versions. Keep four
valid distinct A-D options and recheck the final prompt and label.
Ambiguity is within E's `no-entailed-answer` subtype, not an extra allocation.
No split adds duplicate omission companions.

Synthetic Count menus use policy `synthetic-count-options-0-4-v1`: choose four
distinct values from 0-4 for every answer case. A-D menus contain the sole
entailed value; F replaces it with the remaining unlisted value and retains only
that omission menu. E menus use the same option pool. The solver still evaluates
the complete Count domain over all nonreference objects, including counts above
4; the menu pool must never restrict entailment or counterexample checks.
With uniformly distributed sole values and balanced correct positions, ordered
menu distributions match across A-D/E/F in the sampling law. This removes the
known numeric-range cue, not a claim that all possible shortcuts are eliminated.
The change is opt-in for final-v1; historical callers and external source menus
are unchanged. Slot identities, scene seeds and answer quotas are unchanged,
but revised menus require fresh artifacts and hashes.

The following totals apply **per query** (Direction, Which and Count):

| Split | Questions per query | A-D (60%) | E (30%) | F (10%) |
|---|---:|---:|---:|---:|
| Train | 2,000 | 1,200 | 600 | 200 |
| Dev | 400 | 240 | 120 | 40 |
| SFT-Bench in-range | 800 | 480 | 240 | 80 |
| SFT-Bench extrapolation | 400 | 240 | 120 | 40 |

Match the E subtypes as well, not just the E total:

| Split | Direction/Count no-entailment / inconsistent | Which multiple / no-entailment / inconsistent |
|---|---|---|
| Train | 450 / 150 | 300 / 225 / 75 |
| Dev | 90 / 30 | 60 / 45 / 15 |
| SFT-Bench in-range | 180 / 60 | 120 / 90 / 30 |
| SFT-Bench extrapolation | 90 / 30 | 60 / 45 / 15 |

The old SFT-Bench inventory cannot be preserved unchanged while adding its
missing E cases and enforcing these proportions. Build and freeze a new
version before model evaluation. Keep historical inventories, labels, receipts
and results untouched. Never relabel a question just to fill a quota.

### Semantic Cases Across Levels

Every applicable E subtype, A-D and F must occur at every assigned construction
level: 1-8 for train/dev/in-range and 9-10 for extrapolation. Sole Which with
contingent alternatives must also span those levels instead of being restricted
to span 1. Multiple-entailment E applies only to Which; do not manufacture
impossible Direction or Count subtypes.

Record two separate fields:

- `construction_level`: the assigned structural stratum, available for all
  cases. Train, dev and in-range use uniform levels 1-8 (12.5% each);
  extrapolation uses equal levels 9 and 10 (50% each).
- `verified_depth`: the measured answer-support depth where the existing
  calculus applies. Inconsistent and no-entailment cases have null here.
  Sole Which with contingent alternatives also must not borrow a full
  membership-depth label from a positive path alone.

The level margins below apply per query and include all answer categories:

| Split | Levels | Questions per level | A-D target | E target | F target |
|---|---|---:|---:|---:|---:|
| Train | 1-8 | 250 | 150 | 75 | 25 |
| Dev | 1-8 | 50 | 30 | 15 | 5 |
| SFT-Bench in-range | 1-8 | 100 | 60 | 30 | 10 |
| SFT-Bench extrapolation | 9-10 | 200 | 120 | 60 | 20 |

These are construction-level margins, not counts of verified depth for E
cases. The 70/30 evidence-cohort totals remain separate constraints.

For positive cases, verify the intended support depth, including the selected
two-object certificate for multiple-entailment Which E. Semantic-cohort sole
Which cases must retain their intended positive support length, but keep full
`verified_depth` null; a positive path does not certify contingent negatives.
For inconsistency, check the contradiction. For no-entailment, verify the
complete-domain non-entailment evidence and counterexample worlds. Both cases
must occur at every assigned construction level and keep `verified_depth` null.
Increasing contradiction or ambiguity difficulty is not a release requirement:
a short valid contradiction or unresolved relationship is not a defect at a
higher construction level. Construction level groups the scene; it does not
claim a proof length or increasing difficulty for these two E subtypes. Retain the checks
for unintended shortcuts, valid evidence and exact level/case allocations.

Match A-D/E/F and E-subtype proportions within levels as closely as integer
counts allow while preserving exact split/query totals and nonzero applicable
case coverage. Use largest-remainder allocation for level margins, with ties
in ascending level order, then a constrained integer allocation for cases.
Do not independently round every cell: for example 30 Which inconsistent
in-range cases cannot divide equally over eight levels.
Small subtype cells provide coverage, not precise per-level performance
estimates: dev has only 15 inconsistent Which questions across eight levels.
Report sample counts and pooled subtype performance; do not draw strong
conclusions from one- or two-question cells.

Freeze the complete joint tables and deterministic tie-breaking before
generation. [generation_policy.py](generation_policy.py) now computes and
validates exact split/level/answer/E-subtype tables and writes them into each
run. E remainders go to the level with greatest remaining capacity, ties by
ascending level; Count 0-4 cycles deterministically within each applicable
case/level. Construction and full-corpus feasibility still require the run.
The earlier 24-train/5-dev span-1-only contingent recipe and independent
per-depth tables are superseded; five dev cases cannot cover eight levels.
Allocate that subcase across all levels within the existing Which A-D totals,
and match its rate across synthetic splits up to integer rounding. The executable
policy fixes sole Which with contingent alternatives at 2% of each query's
questions: 40 train, 8 dev, 16 in-range and 8 extrapolation. These belong to
the semantic-cohort A-D allocation, not an additional category.

Dev retains 280 depth-cohort and 120 semantic-cohort examples per query and
10 families per applicable proof-depth stratum. Scale compatible inherited
coverage minima by `ceil(train_minimum / 5)`; presence requirements remain
positive. Retain Count 0-4 and all-direction coverage. Test-specific joint
coverage minima must be frozen with the new benchmark inventory.
Use equal target weights for Count values 0-4 within applicable strata, with
ascending Count value breaking allocation ties.

Use disjoint canonical questions, scenes and construction families across
train, dev and tests. Both explanation arms share the same split inventory,
menus and labels. Generate and freeze held-out identities before training
admission, then rerun all cross-split exclusions against the new inventories;
the old exclusion file alone is insufficient.

### External Evaluation

SpatialMap-TQA-Corr evaluates 1,500 corrected questions from 500 scenes under the
`unique-entailed` answer contract, once each per model. Do not add omission
companions, alter menus to manufacture F cases, rebalance labels, or require
coverage of all answer types in SFT-Bench. Its answer distribution is determined
by the corrected dataset, not synthetic benchmark quotas.

Pin the corrected source artifact and question/menu/label hashes. Do not assume
that removing companions from an old external artifact recovers this dataset.
The generation entry point verifies the tracked randomized-v2 correction export
against `data/spatialeval_org.jsonl` and its correction receipt, then snapshots
those inputs inside the run. It preserves source A-D menus, independently
recomputes the Unique labels, and records differences from historical correction
labels. It does not reuse the old possibility-based evaluation wrapper.

### Split Comparability

Internal query allocations remain equal. Train, dev, in-range and extrapolation
all use the same controlled generation framework and semantic-case proportions.
The old mixed-source in-range target is retired for final-v1. Uniform in-range
levels and controlled extrapolation make comparisons easier to interpret, but
scene size and case-specific evidence complexity can still vary with level.
Report these properties; an aggregate score gap is not automatically a pure
causal effect of proof depth. No split adds omission companions.
Preserve historical frozen Corr SINGLE artifacts unchanged.

Recheck canonical query, scene and family/anchor disjointness against all
held-out splits, including external scenes. Renaming, reordered sentences and
menu variants are not new underlying questions. Cross-query relatives must
not straddle splits. Structural composition patterns
may recur in-range as designed; do not claim every proof topology is held out.
If the fixed benchmark fails semantic verification, block release and propose a
versioned correction instead of silently changing labels or selection.

## 5. Data Release Gates

### G0: Failure Evidence And Policy Freeze

Identify the failed run's commit, effective configs, dataset hashes, checkpoints,
logs and representative dirty rows when accessible. Classify observed defects:
prompt/label mismatch, unsupported rationale, stale/mixed payload, leakage,
template/mask error, or shortcut distribution. These are investigation
categories, not established causes. Each reproducible defect becomes a fixture.
If artifacts cannot be recovered, explicitly record that limitation; do not
claim the previous failure is resolved.

Freeze the complete allocation, model, tokenizer, prompts, rendering version,
seed schedule and evaluation inventory before full generation.

### G1: Small End-To-End Candidate

Before building all 6,000, exercise every query/answer subtype, Count 0-4, all
eight directions, sole Which plus contingent alternatives, and positive depths
1 and 8. Exercise semantic-case construction at every planned level, including
9-10 test construction, before admitting a new benchmark. Include deep Which
multiple-entailment and shortest/longest target cases. These are
construction/serialization fixtures, not additional training.

Render both styles and their final native chat form. Inspect all pilot examples
side by side. Resolve systematic defects before the full build. A rename alone
is not a data fix.

### G2: Complete Independent Semantic And Explanation Audit

For **every exported question**, parse the model-visible system/user text back
to the typed problem and menu. In a fresh solver context, recompute consistency,
the full entailed set and A-F label. Do not use saved `answer`, intended recipe
case or saved `entailed` values as the oracle. Compare against them afterward.
The auditor may reuse the solver kernel, but must not merely call the generator
again or trust its success flag.

Replay positive paths, contradictions and counterexample arrangements against
all visible premises and distinctness constraints. Check complete Which/Count
object coverage, selected depth and exact Count reasoning. For Direction/Count
ambiguity, verify two full satisfying worlds with different outcomes. A printed
projection must have a checked full-world extension.

Bind every explanation's claims, P citations, names, directions, counts and
conclusion to that verified evidence. Require equivalent obligations across
styles. A correct final letter does not rescue a wrong rationale. Use structured
render plans and semantic claim checks; exact string equality with the same
renderer is not an independent correctness test. Reject unsupported wording
rather than deleting its evidence obligation.

Add mutation tests that corrupt gold, a relation, a citation, an object, a
counterexample, a depth, a menu and a split, and require the relevant check to
fail. Compare small independent hand-worked fixtures with the SMT results to
reduce shared-oracle risk.

### G3: Distribution And Readability Review

Publish query/letter/subtype/depth/Count/direction/family counts, duplicates,
overlap, token-length percentiles and E/non-E proportions by scene size.
Cross-tabulate answer and semantic subtype by construction level and source.
Verify every applicable subtype occurs at every level and that allocated
ratios hold. Compare nuisance-length distributions across subtypes within each
level, so level-spanning counts do not conceal a residual length cue.
Use train-only, family-grouped diagnostic splits to probe shallow predictors
based on lengths, object counts, names and menus. Compare with query-specific
majority baselines; do not use test performance to redesign training.
This diagnostic does not prove the absence of shortcuts. Obvious deterministic
label cues or strong unexplained predictability block release pending investigation.

Create a reproducible review packet with at least 10 paired examples from each
of the 13 query-by-answer-subtype cells (130 questions), plus all pilot/failure
fixtures and the longest cases. Supplement to cover every depth and Count value.
Record actual reviewed IDs and findings, not just a generic "looks good".
The generator's `review-selection-v2` packet uses deterministic hash-ranked
sampling for ten questions per query/case (or all available in smoke). It adds
coverage for every split/query/construction-level group, contingent cases per
split/level, sole Count values 0-4 per split/level, and the longest native
tokenized example per split/style. `audit/review-selection.json` records selected
IDs and reasons with status `selected-not-reviewed`. Pilot/failure fixtures and
actual review findings remain additional review obligations, not implied by
successful selection.
Any discovered correctness defect blocks release and triggers a full rerun
after correction. Stylistic limitations and sampling limits remain explicit.

### G4: Native Tokens And Immutable Release

Use the pinned Qwen thinking template with no `enable_thinking` override.
Mask system/user text and the native assistant thinking prefix. Supervise the
explanation, closing thinking tag, final `Answer: X` and turn terminator.
Check exact prompt/completion boundary and causal labels before collation;
padding must later receive attention 0 and label -100.

Measure **all** train and dev examples in both styles with the active tokenizer.
Reject zero-target rows, reserved-token injection, duplicate answers, truncation,
dropped rows and overflow. Reserve Stage-2 suffix plus output space as well as
Stage-1 generation space. Historical reference lengths do not certify this data.

Seal after G0-G4 checks pass and explanation-review findings are recorded;
no separate manual release approval is required. The manifest binds payloads,
row identities, canonical keys, complete policy, source dependency inventory,
tokenizer/template, environment lock and audit receipts by SHA-256.
Keep evidence separate from model-facing inputs. Both arms consume one sealed
release. The training jobs must never regenerate or repair data.

## 6. Matched Training Recipe

Preserve the actual intended recipe, rather than replacing it with generic SFT
defaults:

| Setting | Both arms |
|---|---|
| Base/tokenizer | `Qwen/Qwen3-4B-Thinking-2507` |
| Revision | `768f209d9ea81521153ed38c47d515654e938aea` |
| Initialization | Fresh adapter on pinned BF16 base; no old adapter/resume |
| Method | QLoRA, 4-bit loading, BF16 compute |
| LoRA | Rank 64, alpha 128; q/k/v/o and gate/down/up projections |
| Epochs / seeds | 2 / training and data seed 42 |
| LR / optimizer | 5e-5 / `paged_adamw_8bit` |
| Schedule / warmup | Cosine / 0.05 |
| Microbatch / accumulation | 1 / 8 on one GPU per arm |
| Gradient clipping | 0.05, retained from intended config |
| Attention / checkpointing | SDPA / non-reentrant gradient checkpointing |
| Sequence length | 8,192; no truncation |
| Packing / length grouping | Disabled |
| Checkpoints | Save and retain epoch 1 and epoch 2 |
| Dev evaluation | During SFT each epoch: teacher-forced loss and generated Stage-1 accuracy |
| Selection | Generated Stage-1 dev view accuracy, never dev loss |

Expected updates: `6000 / (1 * 8 * 1) = 750` per epoch, 1,500 per arm.
Verify the actual loader/sampler accounting rather than assuming this occurred.
Equal examples/epochs is the matching rule, not equal supervised tokens or FLOPs.
Report both. Pin and record all previously implicit effective defaults, especially
LoRA dropout, quantization settings, optimizer betas/epsilon and weight decay;
do not silently choose different values while moving configs.

Use the repository's pinned Axolotl revision and lock as the initial environment
candidate, then validate on the actual server. No package upgrade is part of this
reset. Namespace W&B under `spatialentail-final-v1-sft`; use distinct arm/run IDs,
private credential loading, Slurm logs and validated email settings.

### G5: Actual Trainer Boundary

Before any optimizer step, compare all Axolotl-loaded rows and ordered token
hashes to the sealed release via an ID sidecar. Detect cache reuse,
retokenization, deduplication, altered labels and row filtering. Verify the real
collator, including a mixed-length diagnostic batch for padding even though the
training microbatch is one. Assert fixed-dev identity and no random split.

Verify model revision, trainable LoRA targets, device, quantization and effective
config equality with an explicit style/path/logging allowlist. Add finite-loss
and finite-gradient abort checks; `logging_nan_inf_filter: false` is not one.
Observe the first 20 optimizer steps, gradient norms, LR, supervised tokens,
memory and W&B connectivity inside each actual training job. No separate GPU
smoke job is required. Stop on correctness or numerical failure.

Retain both epoch adapters and verify they reload with the pinned base/tokenizer.
Full optimizer-state resume is not promised by adapter-only checkpoint saving.
Any recovery must preserve release/config identity and disclose changed exposure.

## 7. Evaluation And Interpretation

Have the current-model evaluator prepared and regression-tested before training.
Integrate fixed-dev evaluation into the SFT epoch workflow for each arm:
teacher-forced loss and generated Stage-1 accuracy, with no gradient updates
from dev. Record both epochs' results and select highest Unique Stage-1 accuracy
over the fixed 1,200 dev examples, earlier epoch on ties. Teacher-forced dev loss
is diagnostic only. Do not defer ordinary dev validation to a separate
post-training campaign; reusing saved-checkpoint evaluation helpers still
requires integration into this epoch workflow.

After training and checkpoint selection, evaluate each selected adapter and two
style-matched untuned baselines on SFT-Bench (in-range and extrapolation) and
SpatialMap-TQA-Corr. Same model revision, prompts, limits and decoding policy:
greedy (`temperature=0`), seed 42, provisional 4,096 Stage-1 new tokens and
16 Stage-2 new tokens within 8,192 context. These are not certified budgets.
Validate length/format behavior on synthetic dev before final evaluation and
freeze the result. A budget change applies to both arms, all epochs and baselines;
rerun affected comparisons, never selectively grant one system more tokens.

Stage 2 must retain exact Stage-1 generated token context. Test normal,
unfinished-thinking, malformed and length-exhausted responses; record any
protocol closure inserted by the evaluator. Never insert the gold/reference
reasoning. Keep failures in scoring denominators and report them separately.

Report Stage-1 and Stage-2 accuracy, per-query and E/F/subtype slices, supported
depth slices, format failures, budget exhaustion, and generated-token mean,
median and tails. For SpatialMap-TQA-Corr, report only slices actually present;
absent answer types are not coverage failures and have no accuracy denominator.
Report paired arm differences with base/family-clustered
uncertainty, keeping related scene/family questions together. There is one menu
per question, so per-example and per-question accuracy are identical within
each split. Report verified proof-depth slices separately from semantic-case
construction-level slices; do not put null-depth E cases into L9-L10 proof-depth
averages. A single training seed does not estimate training-seed variance.
Fixed test labels and external data never
control checkpoint selection or data repair.

The comparison estimates the two supervision recipes, including their style
instructions, not the isolated causal effect of symbolic notation. Correct
answers do not certify generated-reasoning faithfulness. Token savings must be
reported alongside accuracy and incomplete/invalid generations. Confidence
intervals over evaluation samples do not estimate training-seed uncertainty.

## 8. Implementation Order And Ownership

1. Record the agreed scope and recover available failure evidence (G0).
2. Create final-v1-owned policy/model/prompt metadata and pinned evaluation
   inventory with explicit provenance. Preserve historical files.
3. Parameterize shared preparation/evaluation helpers to accept experiment and
   release paths; retain old defaults for historical callers. Do not duplicate
   the solver or monkeypatch global experiment constants. Test two experiment
   roots concurrently to detect accidental old-path/cache use.
4. Implement the pilot, independent exported-text audit, mutation tests and
   review packet (G1-G3). Generate the complete candidate only after pilot pass.
5. Prepare native train/dev tokens, immutable manifests and enforced release
   verification (G4). Add negative launch tests for missing/failed receipts,
   tampered payloads, mixed arms, stale caches and wrong model revisions.
6. Implement loader/collator and numerical checks, in-training epoch dev loss
   and generated accuracy, checkpoint reload and final evaluation (G5).
7. After release validation passes, submit both training jobs against the sealed release.
   Evaluate dev at each epoch during SFT. Use the recorded dev results to select
   the checkpoint, then run SFT-Bench and SpatialMap-TQA-Corr after training.

Future layout: `releases/<release-id>/` for immutable data and receipts;
`runs/<run-id>/` for per-arm caches/configs/logs/checkpoints/evaluation.
Large generated data and weights belong on the training server's artifact
storage, with small manifests and review reports retained here.

Generation and training are separate commands/jobs. A future submitter must
require the explicit release path and verify passing audit receipts and hashes;
no "latest candidate" discovery or training from an unvalidated generation job.
No additional manual sign-off is needed for the agreed experiment. Use a fixed
checkout for each run; never pull into it.

### Remaining Specifications

- Full-corpus construction supply, new test identities and coverage feasibility.
  Slot tables and the contingent rate are executable; actual generated data
  still needs the full campaign.
- Full-run SpatialMap-TQA-Corr source-preservation and label audit.
- Reproducible shortcut probes: nuisance features, classifier, family-grouped
  splits, seed, metrics and investigation trigger. Counts alone are insufficient.
- Primary final comparisons and paired uncertainty procedure, including cluster
  identities, confidence level, bootstrap seed and replicate count. Compare
  tuned arms with their style-matched untuned baselines and each other, keeping
  split results separate. These analysis details must be fixed before results.
- Effective trainer defaults, pinned environment, enforced data-release checks,
  in-training dev evaluation, checkpoint reload and Stage-2 runtime validation.

### Generation Status

The CPU entry point builds `candidate-audited` artifacts (or `smoke-passed`
artifacts), not a sealed training release. It checks new labels/evidence,
one-menu allocation, overlap, native tokens and export integrity. Ambiguity and
inconsistency have verified semantics, not a claim of increasing difficulty
across construction levels. No semantic-difficulty scaling gate is required.
Shortcut probes, paired-example review and trainer/evaluation integration
remain pending; `training_ready` remains false.

Generation does not submit training, install the GPU environment, modify
historical payloads, or claim that local tests certify server execution. A later
SFT job must consume the run-local artifacts and enforce release readiness;
successful CPU exit status alone is not sufficient.

## 9. Exit Criteria

Ready means: complete policy preserved, zero unresolved semantic/evidence/
overlap/mask failures, reviewed paired examples, sealed train and fixed-dev
payloads, enforced loader and launch checks, and a tested evaluation/selection
path. Runtime validity is established only by G5 and checkpoint reload evidence.
Until then the verdict remains **not ready to train**, regardless of whether
unit tests pass or generation exits successfully.
