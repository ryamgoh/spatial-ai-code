# Unique-Only Spatial Entailment Study

## Authority

The user approved dropping Listed and Exhaustive from the active study.
This document supersedes conflicting mixed-contract requirements in
`ENTAILMENT-ONLY-v1.md`, the pivot, and historical runbooks.
Study scope: `unique-only-entailment-v1`.
Only `unique-entailed` is active for Direction, Which, and Count, in both
synthetic training/evaluation and the new external entailment adaptation.
Historical Corr SINGLE and all earlier three-contract artifacts remain unchanged.

## Active Model

The user selected **Qwen/Qwen3-4B-Thinking-2507** for all future training, baseline
and evaluation runs, replacing the provisional Qwen3-4B selection and the earlier
Qwen3.5-4B model. The machine-readable selection is
[`active-model.json`](active-model.json), with model/tokenizer revision
`768f209d9ea81521153ed38c47d515654e938aea` resolved from the official repository.
Both Natural and Symbolic arms use the same standard BF16, thinking-only
checkpoint. Its native template opens the thinking block automatically;
no `enable_thinking` override is required. The previous non-thinking setting
is withdrawn. The existing 8,192 context, 4,096 Stage-1 output and 16 Stage-2
output limits are retained provisionally, pending thinking-model validation;
they are not certified generation budgets for this checkpoint.

This changes the model, not the questions, explanations, allocation or splits.
Historical candidate token measurements use Qwen3.5-4B. The new
[thinking preparation](prepared-thinking-v1/README.md) independently measured
all 6,000 examples per arm with the pinned Qwen3 tokenizer and validated the
native explanation/answer placement and explicit assistant loss masks.
Maximum training lengths are 1,766 Natural / 1,930 Symbolic; supervised maxima
are 1,105 / 1,264. Both fit the current training limits without truncation.
The prepared configs retain matched two-epoch QLoRA training, using four-bit
loading from the BF16 checkpoint and BF16 compute. Existing v3 payloads,
questions, explanations and evaluation selections remain unchanged.

The user approved GPU validation during the opening steps of the actual
training job, with no separate smoke-test job. Actual Axolotl loading/CUDA,
current-model fixed-dev evaluation integration and Stage-2 thinking replay
remain unvalidated. Reference fit does not certify untuned generation length.
Do not truncate or regenerate examples silently if later validation fails.

Historical configurations, model constants and audit receipts retain their
original Qwen3.5 identity. New trainer configuration must read the active
selection, not reuse historical `composition_training.MODEL` or old YAML files.
Only pinned tokenizer/configuration assets were downloaded for preparation;
no model weights were loaded and no training was started.

## Current Verified Artifact

The current [all-controlled replacement](unique-training-candidate-v3/README.md)
is a reference-only revision of v2. All 6,000 questions, menus, labels, saved
evidence, canonical identities, split assignments and depth metadata are
unchanged. The [v3 re-render audit](SPATIAL-PROSE-RERENDER-v3.md) passed for every
row and both styles. [Actual revised examples](SPATIAL-PROSE-RERENDER-v3.examples.md)
include all query/answer cases and deep proofs.

Version `spatial-prose-shared-chains-v3` uses spatial Natural wording, shared
Count chains, two distinct saved outcomes for all 900 ambiguous Direction/Count
questions, grouped Which rejections, and shorter conclusions. The evidence-plan
audit verifies retained positive paths, full Count/Which object coverage, and
the selected positive depth. Average Natural/Symbolic target tokens are
333.3 / 386.8; maxima are 1,146 / 1,301. Old exports are preserved.

### Underlying v2 Question Set

The [v2 all-controlled question set](unique-training-candidate-v2/README.md)
contains 6,000 distinct canonical questions, one selected menu each, and 6,000
matched Natural/Symbolic training examples per style. The build selected them
from 6,045 fixed construction attempts; all 6,000 passed independent spatial
evidence replay, selected-depth checks, token admission and the preserved
numerical allocation/coverage requirements. No supply refill was needed.

Each query has exactly 1,200 A-D, 600 E and 200 F, with the E subtypes specified
below. The previous generation-source percentages are retired for this
replacement. Numerical depth, Count-value, family and evidence safeguards remain.
The v2 references used the concise version with full Which uniqueness obligations,
but retained mechanical wording; the v3 reference-only revision above supersedes
them. Historical v2 maximum Natural/Symbolic target tokens are 1,653 / 1,825.

Build receipt: `CONTROLLED-TRAINING-BUILD-v2.json`. The separate
[published-file audit](CONTROLLED-TRAINING-EXPORT-AUDIT-v2.md) passed.
At the previous 13-premise boundary, E shares are 30% on both sides for Direction
and Count, and 26.4% / 30.4% for Which. This is a descriptive size check, not
a claim that every structural shortcut has been eliminated. Existing training-v1 and evaluation
payloads remain unchanged. This is an unsealed chat-data export, not a GPU
smoke test, trainer launch configuration or a completed training run.

### Historical Candidate v1

The following review and supply-bound history applies to v1, not the replacement.
The v1 payload is preserved for reproducibility and should not be used for the
new training run.

The [6,000-question training candidate](unique-training-candidate-v1/README.md)
is selected, exported and audited under the approved allocation below.
Each style has 6,000 examples, one menu per distinct question, with exact
1,200 A-D / 600 E / 200 F per query and all requested E-subtype counts.
The 70/15/10/5 source mix, controlled depth counts, Count-value quotas and
family/negative-evidence safeguards passed joint selection and final audit.
Selected-explanation depth coverage is 1,400 Direction, 1,416 Which and 1,400
Count. Both styles' token and message checks pass.

Evidence: `UNIQUE-TRAINING-SELECTION-REPLAY-v1.json`,
`unique-training-candidate-v1/manifest.json` and
`UNIQUE-TRAINING-EXPORT-AUDIT-v1.json`. All 15,450 eligible training candidates
completed replay before selection. Existing synthetic and external evaluation
selections remain unchanged. This completes allocation and unsealed export,
not prose-review, release, sealing or model training.
The feasibility sections below preserve the chronological decisions and
intermediate shortages; this current artifact supersedes their pending status.

The subsequent [pre-training quality review](UNIQUE-PRETRAINING-REVIEW-v1.md)
recommends holding the main SFT launch. Training-only family/story-held-out
probes predict E versus non-E from scene size alone at 94.2% Direction and
94.7% Count accuracy (70% majority baseline). This is a measured shortcut
opportunity, not a full-answer or SFT evaluation result. Redundant exact-value
rejection sections and internal explanation terminology also remain.
No candidate payloads or quotas were changed. Reference cleanup and the response
to shortcut risk require a separate decision; allocation correctness does not
by itself establish experiment readiness.

The [bounded existing-pool check](UNIQUE-LENGTH-SUPPLY-CHECK-v1.md) now proves
that reselection under unchanged quotas cannot remove the observed fixed
premise-count shortcut: its E/non-E accuracy remains at least 88.55% for
Direction and 89.0% for Count, even under optimistic supply assumptions.
These are bounds, not attained alternative selections. No dataset replacement,
reference edits, additional generation or quota changes were performed.
A generator/scene-size change or another explicit quality decision is required
before claiming the shortcut has been addressed.

## All-Controlled Prototype Authorization

After inspecting the inherited source quotas, the user approved pursuing a
bounded all-controlled, randomized-construction prototype. This is a generation
method change within the same entailment research question, not another task
or answer-contract pivot.

The 70/15/10/5 proportions describe the existing artifact and remain in its
historical executable policy. They are not required of the new prototype.
The intended replacement training target remains 6,000 distinct questions,
1,200 A-D / 600 E / 200 F per query, the specified E subtypes, and genuine
L1-L8 training / L9-L10 extrapolation coverage where depth is defined.
Existing evaluation selections are not changed.

The separate prototype has exactly 260 scheduled questions: ten construction
spans, two scene sizes per span, and thirteen query/case combinations (four each
for Direction and Count, five for Which). It constructs listed sole entailment,
unlisted sole entailment, no entailment, inconsistency, and Which multiple
entailment directly. Within each span/size group, all cases have identical
object and premise counts. Names, inverse phrasing, sentence order and excluded
context edges are randomized. All scene objects remain in Which/Count scope.

Construction span is not an E proof-depth label. Positive selected-certificate
depth is verified for sole answers and multiple-Which certificates; ambiguous
and inconsistent E examples must have no positive-depth label. Count values
cycle through 0-4 in this feasibility run, not the full depth/value quota table.

The runner records its fixed slots and source fingerprints before admission,
uses the pinned Qwen tokenizer, independently replays semantic evidence, and
reports failed slots without retries or refill. It also measures size-only
and structural-count E/non-E probes. Development evidence stays outside the
workspace; the manifest and results are
`ALL-CONTROLLED-PROTOTYPE-v1.manifest.json` and
`ALL-CONTROLLED-PROTOTYPE-v1.json` / `.md`.

This authorization does not cover bulk replacement generation, changing
evaluation artifacts, sealing or training. The existing renderer's prose issues
remain separate. A production policy still needs recipe-diversity and shortcut
review, canonical/family overlap controls, and a decision on how to replace the
inherited source-linked depth allocation without weakening the research target.

### Prototype Result

The [bounded prototype report](ALL-CONTROLLED-PROTOTYPE-v1.md) records 260/260
accepted questions, independent evidence replay, and exactly matched object and
premise counts across all thirteen query/case combinations in each size group.
All positive requested depths L1-L10 passed. Natural/Symbolic maximum target
tokens were 2,780 / 3,000. Fingerprints of the existing training and external
payloads, frozen config, and selection receipts remained unchanged.

The size-only E/non-E probe has 50% balanced accuracy for every query.
Structural-count balanced accuracy remains 70.0% Direction, 64.2% Which and
71.2% Count in this small ten-group development experiment. These results
support all-controlled feasibility, not production readiness. Scene sizes span
15-31 premises; smaller scene coverage, broader ambiguity/conflict mechanisms,
and recipe-related shortcuts remain unresolved. No additional generation or
automatic replacement follows this completed prototype.

### Controlled Coverage Extension

The user subsequently confirmed continuing with the unchanged 1,200/600/200
answer targets. The next bounded development check is separately versioned as
`all-controlled-coverage-v2`, with 968 scheduled questions and no retry/refill.
It preserves the first prototype and all training/evaluation artifacts.

This extension decouples direction, depth and exact Count value: positive
coverage crosses ten depths with eight directions, and Count additionally
crosses values 0-4 (400 cells). It adds 6- and 9-premise compact scenes across
all applicable cases and eight directions, and adds disconnected-component
ambiguity and variable-length strict cycles. E construction span remains
distinct from positive proof depth. This check does not require another
classifier or treat chance-level structural prediction as a release gate.

The executable inventory is in `run_controlled_coverage.py` and
`spatial/v2/entailment_controlled_coverage.py`. A separate manifest records the
fixed slots before execution. The report is `CONTROLLED-COVERAGE-v2.json` / `.md`;
the `.examples.md` companion contains actual verified compact prompts and both
reference styles. No bulk replacement, split change, sealing or training is
authorized by this bounded coverage step.

The completed [coverage report](CONTROLLED-COVERAGE-v2.md) records 968/968
accepted scheduled examples, all 400 Count depth/direction/value cells passed,
and complete matched sizes in every scheduled size group. All examples passed
independent evidence replay and token admission; maximum Natural/Symbolic
reference tokens were 2,527 / 2,796. The observed premise counts are 6, 9, 12,
18-20, 22-24 and 26; this is not a claim of every integer scene length.
The original artifacts and source fingerprints remained unchanged. The old/new
prototype and runner regression suite passed 226 tests.

The [compact examples](CONTROLLED-COVERAGE-v2.examples.md) contain actual
verified questions and references. They retain the old renderer's redundant
exact-value rejection paragraphs and internal-contract phrasing, which remain
to be cleaned up separately. The 968 slots are coverage demonstrations, not
968 certified distinct production questions or the final answer distribution.

The next production-preparation step is a new allocation adapter, not an edit
to the old frozen policy. It must retain the answer/subtype targets below,
translate the inherited source-linked depth rules into explicit proof-depth
requirements, and schedule scene sizes across answer cases. That translation
must be recorded before a bulk run; it cannot silently drop a numerical depth,
Count-value, family or negative-evidence requirement. Canonical questions and
construction families must be checked against each other and the fixed
evaluation inventory before admission to a replacement training selection.
Renaming, sentence shuffling and omission menus do not establish new underlying
questions. Existing development coverage rows are not a training candidate pool.

### Consolidated Construction Interface

The tested construction logic is now local to `spatial/v2/entailment_generator.py`,
documented in [Controlled Generator](CONTROLLED-GENERATOR.md). New construction
callers use `SceneRequest` and `build_scene`, rather than private functions across
the coverage module, first prototype, and old sampler. The interface requests a
semantic case; A-D/F selection remains a subsequent menu decision.

This is a behavior-preserving refactor, not an additional sampled batch or
production-policy migration. All 968 fixed coverage scenes are compared for
exact equality of names, relations, premise order and query scope; only the
new story recipe provenance tag differs. Separate evidence/depth tests exercise
the new interface. The historical construction files and their receipts remain
unchanged for reproducibility, and production commands are not silently rerouted.

The small recipe collection, compact-size presets, Which cardinality rule and
excluded-side context restriction remain documented design limits. Consolidation
does not establish broader diversity, remove all custom construction logic, or
complete the replacement allocation and explanation-prose work.

### Replacement Build

The user authorized proceeding with replacement preparation and generation,
but not model training. The separately versioned all-controlled allocation
retains every existing numerical training stratum, Count-value quota, depth
family floor and active Unique evidence minimum. The retired 70/15/10/5 source
quotas are removed; generation source is controlled for every new question.
The 1,400 depth / 600 semantic strata per query are evidence cohorts, not sources.
For Which, the depth cohort contains 900 A-D, 200 F and 300 multiple-entailment
E; its semantic cohort contains 300 A-D, 225 no-entailed E and 75 inconsistent E.
The semantic cohort includes the required sole-entailment-plus-contingent case.
Numerical depth and Count targets are imported from the preserved frozen policy.

The build uses `entailment_replacement_allocation.py` and
`build_controlled_training.py`. A fixed manifest declares at most 18,000
construction candidates (three per nominal quota slot). Within each quota
group, preflight selects the first distinct, nonoverlapping canonical questions.
Only the resulting 6,000 questions receive solver admission. There are no
solver retries, unbounded sampling or automatic supply refills. Shortages or
any failed admission/audit stop the export.

The prior generator varied mainly presentation and a small context forest.
The new optional structural seed also varies safe excluded-side relations,
without editing positive supports; complete-scene verification remains required.
A shared-anchor context supplies sole Which cases with contingent alternatives.
Generated coordinates used to choose context relations are not inputs,
additional metric assumptions, gold labels, or supervision.

References use `unique-readable-explanations-v2`. Exact Direction/Count
references keep their positive proofs and a mutual-exclusivity conclusion,
without redundant per-value rejections. Which retains full uniqueness
obligations. Internal-contract and minimality boilerplate is removed. The old
reference version and its default rendering remain supported.

New artifacts are written only after semantic, evidence, depth, quota,
canonical overlap, family, token and message-export checks. Existing training
and all evaluation payloads remain untouched. Training, sealing and a claim
of release readiness do not follow automatically from starting this build.

## Research Question

Can models learn atomic spatial entailment across Direction, Which, and Count
and generalize to longer reasoning depths? Compare answer accuracy and generated
token cost under matched Natural and Symbolic supervision. There is no active
contract-comparison or multi-answer-set prediction experiment.

The semantic interface stays `spatialentail-entailment-v1`. This is not a return
to the historical possibility-based `single` contract.
For consistent premises, compute the complete set T of individually entailed
values. If T has exactly one value, choose its A-D letter, or F if unlisted.
If T is empty or has multiple values, choose E. Inconsistent premises also
return E by convention. Unknown/timeout remains an error, not an answer.
Stage 1 ends in `Answer: X`; Stage 2 returns one A-F letter.

Unique Which means one individually entailed object, not exactly one object
that could qualify. A second merely contingent object does not invalidate a
sole entailment. Multiple entailed objects remain valid E examples; an E answer
alone does not establish that a model identified every entailed object.

## Approved Training Allocation Goal

The user approved the following target on 2026-10-31. It supersedes the earlier
1,500/250/250 and 1,200/400/400 proposals and the projected 6,942-example training
allocation. This fixes the intended distribution, not a verified selection,
executable configuration freeze, dataset seal, or training authorization.

| Query | A-D | E | F | Final training examples per style |
|---|---:|---:|---:|---:|
| Direction | 1,200 | 600 | 200 | 2,000 |
| Which | 1,200 | 600 | 200 | 2,000 |
| Count | 1,200 | 600 | 200 | 2,000 |
| Total | 3,600 | 1,800 | 600 | 6,000 |

The target is 6,000 distinct underlying training questions, with exactly one
selected menu per question. Natural and Symbolic receive the same questions,
menus, labels, and evidence, with style-specific explanations. They each receive
6,000 examples, not 6,000 bases plus omission companions. A-D is an aggregate
quota, not an additional equal-letter quota. A-D and F together provide 70%
sole-entailment examples; E provides 30%.

Within E, use these subtype targets:

| Query | Multiple entailed | No entailed value | Inconsistent | Total E |
|---|---:|---:|---:|---:|
| Direction | 0 | 450 | 150 | 600 |
| Which | 300 | 225 | 75 | 600 |
| Count | 0 | 450 | 150 | 600 |
| Total | 300 | 1,125 | 375 | 1,800 |

Multiple entailments are possible only for Which under consistent premises.
F has one semantic case, but must retain varied spatial reasoning, query types,
and supported depths. These targets emphasize entailment reasoning over the
additional menu-membership check, without claiming an empirically optimal mix.

First check supply and constrained feasibility in the existing verified v4
candidate pool, using all eligible training candidates, not only the previously
selected training subset. Choose at most one existing menu per base; do not count
primary and omission menus as independent questions. Preserve source, Count-value,
family, negative-evidence and depth safeguards. If these conflict with the answer
targets, report the concrete incompatibility for approval. Do not silently add
seeds, refill candidates, move splits, or relax numerical requirements.

Leave dev, in-range test, extrapolation and the audited external Unique set
unchanged. Their existing omission companions remain valid correlated examples.
The new answer and E-subtype quotas apply only to training.

### Feasibility Result

The [2026-10-31 full-pool bounds check](UNIQUE-TRAINING-ALLOCATION-FEASIBILITY-v1.md)
proves this target cannot coexist with all retained requirements on the existing
pool. Direction has only 265 of the required 450 no-entailment questions.
Fixed source quotas also limit Count E and Which's required answer mix.
Separately, the inherited Direction/Count minimum of 1,400 primary substantive
depth examples requires A-D answers and contradicts the new 1,200 A-D quota.

These were verified conflicts, not a solver timeout. At that checkpoint no
requirements or supply had changed. The approved repair and its outcome below
supersede that checkpoint's supply status, not the target.

The user subsequently chose **Keep new allocation**, then **Approve both
repairs**: the training-only selected-explanation depth rule and one fixed
supplementary batch capped at 2,048 queries, with no automatic refill. The
[bounded repair proposal](UNIQUE-TRAINING-REPAIR-PROPOSAL-v1.md) preserves the
70/15/10/5 source mix, presents a source/answer count construction with 323
missing accepted questions, and records the approved boundaries. The arithmetic
does not establish joint feasibility or guarantee sufficient admission yield.
Only the three training-wide primary-menu depth predicates change, retaining
the 1,400 minimum; all other numerical safeguards and evaluation rules remain.
Sealing and training remain unauthorized.

The [bounded repair result](UNIQUE-TRAINING-REPAIR-RESULT-v1.md) now records
completion: 815 accepted additions from 2,048 fixed slots, with 1,187
missing-object pre-admission rejections and 46 held-out overlap exclusions.
The selected-explanation depth rule is implemented and tested. The combined pool
still falls short under fixed source quotas: Direction E has an upper bound
of 597 against 600, and Which's sole-plus-multiple group has an upper bound of
1,651 against 1,700. Count passes the necessary supply bounds, not full selection.
The approved batch is exhausted; no refill or further source changes are
authorized. No final selection or training export was produced.

The user then authorized a read-only source-proportion assessment. The
[diagnostic](UNIQUE-SOURCE-PROPORTION-DIAGNOSTIC-v1.md) shows that three Direction
source-slot changes suffice at the aggregate level, but Which remains at least
35 short even with unrestricted noncontrolled source proportions and 1,400
controlled questions. Source proportions alone cannot resolve the full target.
No changes were applied; more supply or a controlled-cohort/answer-target revision
would require a separate decision.

The user chose **Plan more supply**, preserving answer, source and depth quotas.
The [second-batch proposal](UNIQUE-TRAINING-SECOND-BATCH-PROPOSAL-v1.md) uses
observed first-batch yields to propose 1,600 fixed Direction/Which attempts,
with no Count generation, sampler change or automatic refill. This is planning
history: the user subsequently chose **Approve second batch**, authorizing those
1,600 fixed slots and the ensuing supply, evidence-replay and selection checks.
No further batch, source-quota change, seal or training is authorized.

The [second batch result](UNIQUE-TRAINING-SECOND-BATCH-RESULT-v1.md) records
585 accepted additions from 1,600 fixed slots, 975 missing-object rejections
and 40 held-out overlap exclusions. Combined necessary supply bounds now pass
on 15,450 training candidates. Evidence reconstruction and exact one-menu
selection are in progress; neither full feasibility nor the final 6,000-example
allocation is yet established. Both supplementary generation caps are exhausted.

### Depth Allocation

- Controlled training depths L1-L8 retain weights 8/9/10/11/13/14/16/19%.
- Controlled dev and in-range test depths L1-L8 remain uniform.
- Extrapolation remains controlled-only, split equally between L9 and L10.
- Do not impose the answer or E-subtype proportions separately at every depth.
- Depth means verified shortest atomic-axis support under the declared calculus,
  not premise count or explanation length. Multiple-entailment E uses the cheapest
  two-member certificate. Do not assign artificial depths to ambiguity or
  inconsistency.
- Distinguish base support depth from selected-menu decision/explanation depth.
  An omission-only F example cannot receive primary-menu coverage credit from an
  explanation the model never sees. The approved training-wide rule now counts
  evidence in selected A-D/F explanations or a cheapest-pair Which E certificate,
  while preserving 1,400 qualifying examples per query.

### Complete Answer Cases And Evidence

Check consistency first, then the complete domain's entailed set T, then the menu.
Failed uniqueness takes priority over whether entailed values are displayed.

| Condition | Answer | Explanation obligation |
|---|---|---|
| Inconsistent premises | E | Establish a contradiction; this label is a dataset convention. |
| Consistent, T empty | E | Establish non-entailment for every domain value using checked counterexamples. |
| Consistent, at least two values in T | E | Prove at least two distinct objects qualify; enumerating all is unnecessary. |
| Consistent, exactly one value in T, displayed | A-D | Establish sole entailment and identify its option. |
| Consistent, exactly one value in T, unlisted | F | Establish sole entailment and show that value is absent from A-D. |

For Which A-D/F, proving one object qualifies is insufficient: establish that
every other nonreference scene object is not entailed. A checked arrangement
where an alternative fails suffices to refute its entailment; alternatives may
use different arrangements. Non-entailment is not impossibility, and absence
of a found proof is not evidence. One entailed object plus contingent alternatives
still gives A-D/F. Two entailed objects give E even if neither is displayed.

For Direction and exact Count, proving one value holds in every valid arrangement
excludes other exact values by mutual exclusivity. An entailed Count of zero is
a sole entailed value, not an empty T. Count concerns the exact total in every
arrangement, not merely the number of individually entailed Which members.
Unknown results, timeouts, unsupported proofs and failed audits are rejected,
never converted to E/F. Both explanation styles must express the same verified
obligations before mapping the result to the menu.

## Short Questions And System Rules
The user approved SpatialMap-style questions with the entailment definition,
all-scene scope, and complete E/F rules in the shared system message.
New prompt version: `plain-unique-system-rules-v1`.

- Direction: "In which direction is Library relative to Park?"
- Which: "Which object is west of Police Station?"
- Count: "How many objects are southeast of Clinic?"

The question does not repeat "guaranteed", "exactly one", or instructions to
choose E/F. A-D menus and the E/F option labels remain. The system message
requires one individually entailed domain value, not one merely possible
value, and explicitly covers inconsistency, no entailment, multiple
entailments, and a sole unlisted entailment. A contingent second object does
not invalidate the sole entailed Which answer. Exact Count is not computed
by merely counting entailed Which members.

Synthetic and external examples use the same rules. Their premise sentences,
menus, labels, and evidence are unchanged by this presentation revision.
Natural/Symbolic reference text is not rewritten in this revision.
Old `plain-all-scene-entailment-v1` prompts remain readable; mixed prompt
versions are rejected within an evaluation run. Earlier selection metadata
retains its original prompt/token provenance. New exports must render and
remeasure the short protocol rather than reuse old prompt token counts.

## Unchanged Requirements

- All-scene scope, atomic exact directions, distinct points, and open-world
  semantics; all nonreference objects count for Which/Count.
- Ordinary synthetic place names, one paragraph of untouched premises,
  explanation-only P citations, and source-faithful external names/layout.
- Matched Natural/Symbolic evidence and checked counterexamples for negatives.
- 6,000 distinct train questions with one selected menu each, 1,200 dev,
  2,400 in-range test, and 1,200 extrapolation bases.
- Equal query allocation; in-range 70/15/10/5 source mixture and controlled-only
  extrapolation; fixed seeds and family-held-out splits.
- Controlled depths 1-8 in-range and 9-10 extrapolation, approved training
  weights, Count 0-4 allocation, direction presence, and family floors.
- Unique depth minima, with the approved training-wide selected-explanation
  amendment above and unchanged evaluation predicates. Multiple-entailment E decisions need
  a genuinely deep cheapest two-member certificate; shallow E cannot borrow
  depth from unrelated premises.
- Semantic case, contingent/impossible negative, sole-entailed-plus-contingent,
  and Unique A-F presence requirements.
- Existing evaluation omission companions, including valid repeated E decisions.
  Training now selects one menu per question under the approved allocation above.

## Retired Requirements

Listed/Exhaustive supervision, evaluation, literal-gold presence and depth
requirements are retired, as are all contract-pair difference minima.
The negative-evidence minima previously attached to Listed views are checked
on the corresponding Unique views with the same numerical minima.

G2-G4 output balancing and the diagnostic exact 900/3,600 multi-Selection
target are no longer active goals. Existing multi-entailment bases are retained,
not deleted or relabelled. Their 25% share is a characteristic inherited from
the verified v4 selection, not a new Unique-only quota.

## Previous Validated Selection

This records the preserved pre-rebalance projection, not achievement of the new
6,000-final-example training goal. Its artifacts and receipts remain unchanged.

The same v4 selected bases were projected onto Unique rows only. No new scenes,
seeds, solver labels, split assignments, or premise changes were introduced.
Unique references and menus are preserved; token metadata is filtered to the
matching two styles. The remaining coverage and exact quota checks pass.

| Split | Bases | Unique views per style | A-D | E | F |
|---|---:|---:|---:|---:|---:|
| Train | 6,000 | 6,942 | 4,191 | 1,907 | 844 |
| Dev | 1,200 | 1,394 | 849 | 369 | 176 |
| In-range test | 2,400 | 2,790 | 1,670 | 772 | 348 |
| Extrapolation test | 1,200 | 1,452 | 1,047 | 181 | 224 |
| Total | 10,800 | 12,578 | 7,757 | 3,229 | 1,592 |

The former 37,734 views per style become 12,578; 25,156 Listed/Exhaustive
views per style are excluded. Training shrinks from 20,826 to 6,942 views
per style, not from 6,000 to 2,000 base questions.
No epochs or repetitions are increased automatically.

Training E/F exposure is 39.63%. Training Which has 1,080 E answers in
2,285 views (47.26%), plus 187 F. Report a training-chosen majority baseline
and query/escape-reason slices; do not mistake E accuracy for full member
identification. No rebalancing was performed.

The executable projection and policy derivation are in
`spatial/v2/entailment_unique_study.py`. The selection inventory, policy and
distributions are in `unique-only-selection-v1/`. They are explicitly
nonfrozen metadata, not a training package.

The projection passed the actual selected-base quota/coverage validator.
156 targeted tests passed across Unique projection, both-stage scoring,
checkpoint selection, message construction, legacy scoring, all-scene
protocols, and retention.

## Training And Evaluation

Keep two matched SFT arms, one Natural and one Symbolic, and two style-matched
untuned evaluations. Both Stage 1 and Stage 2 remain in scope.
Checkpoint selection uses Unique Stage-1 synthetic dev view accuracy,
with earliest epoch breaking ties. It is not divided by three contracts.
Report per-query accuracy as well as overall accuracy and depth/escape slices.
Omission views remain correlated observations, not independent base questions.

Scoring recognizes Unique-only input only when explicitly tagged with the
study scope. Mixed scopes, retired contracts in this scope, and legacy scoped
prompts are rejected. Historical three-contract scoring remains supported.
Listed option-classification metrics are not emitted for Unique-only runs.

External evaluation uses the Unique projection of the newly verified all-scene
SpatialMap entailment adaptation, not copied historical Corr SINGLE labels.
External data remains excluded from training and checkpoint selection.

The external artifact is now available in
[spatialmap-unique-v1](spatialmap-unique-v1/README.md): 1,500 base questions
from 500 scenes, with 500 questions per query type and 1,690 views per style
including 190 omission companions. It uses the short question protocol and
passed the full independent audit in
`EXTERNAL-UNIQUE-INDEPENDENT-AUDIT-v1.json`. All IDs, menus, and labels match
the preserved Unique subset of the earlier entailment adaptation.
The short-question integration passed 900 regressions, including unchanged
synthetic labels/evidence/references, both-stage scoring, historical prompt
compatibility, and all 1,500 external source-paragraph round trips.

## Remaining Artifact Gates

The training selection, short-protocol reference export and full artifact/message/
token audit are complete in `unique-training-candidate-v1/`. The scope change
and audited external Unique artifact are also complete. No dataset seal or
training run was performed.
The legacy pilot preparation CLI is not a release path for this 10,800-base
all-scene study. Manual prose review and separately approved release/training
preparation remain required. Evaluation reference materialization is a separate
release concern; this export does not silently alter the original evaluation
selections or declare the whole experiment ready to launch.
Do not rerun the historical three-contract feasibility workflow as the active
Unique-only experiment.
