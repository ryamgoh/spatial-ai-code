# Unique-Only Spatial Entailment Study

## Authority

The user approved dropping Listed and Exhaustive from the active study.
This document contains the active requirements; superseded specifications
and intermediate artifacts were removed during cleanup.
Study scope: `unique-only-entailment-v1`.
Only `unique-entailed` is active for Direction, Which, and Count, in both
synthetic training/evaluation and the new external entailment adaptation.
Historical Corr SINGLE outside this experiment is not modified.

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
Historical candidate token measurements use Qwen3.5-4B. Native
thinking preparation independently measured
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
The retained local measurements do not certify server runtime or job status.

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
333.3 / 386.8; maxima are 1,146 / 1,301. Superseded exports were removed;
the final v3 payload and its build/reference receipts remain.

## Controlled Generation

All training questions use controlled randomized construction. The old
70/15/10/5 training source proportions are retired. Retain the exact allocation
and safeguards in `CONTROLLED-TRAINING-BUILD-v2.manifest.json` and the frozen
config; those files remain live inputs despite their historical version names.

Each query retains 1,400 depth-cohort and 600 semantic-cohort questions.
Which's depth cohort is 900 A-D, 200 F and 300 multiple-entailment E;
its semantic cohort is 300 A-D, 225 no-entailment E and 75 inconsistent E.
The latter includes sole-entailment examples with contingent alternatives.
Depth counts L1-L8 are 112, 126, 140, 154, 182, 196, 224 and 266 per query.
Preserve Count 0-4 targets, 50-family floors per depth stratum, and every
active semantic/negative-evidence minimum in the policy.

`generate_and_prepare.py` allows at most three declared constructions per slot
(18,000 total), then exactly 6,000 solver admissions. Shortages or failed
semantic/evidence/depth/overlap/token audits stop the job; no retry/refill.
`generation-exclusions-v1.json` preserves the fixed evaluation fingerprints.
Renaming, sentence order and omission menus do not create new canonical questions.

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

The approved allocation below applies to final training examples per style.
It is verified for the local v3 artifact and rechecked on server generation.

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

### Depth Allocation

Dev, in-range test, extrapolation and external selections remain fixed.
Their existing omission companions are correlated evaluation views; training
uses exactly one selected menu per question.

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

The final training reference artifact is `unique-training-candidate-v3/`.
Server jobs generate fresh run-local data and matched training files.
Epoch dev evaluation, current-model generated-answer checkpoint selection and
Stage-2 thinking replay are not yet integrated. Actual trainer/GPU behavior
requires runtime evidence. See `TRAINING-RUNBOOK.md`; do not use old commands.
