# SpatialEntail

## Current research authority

The current planning target is
[`spatialentail-final-v1`](experiments/spatialentail-final-v1/README.md), a clean
restart of the intended composition SFT study. Read its
[SFT plan](experiments/spatialentail-final-v1/SFT-PLAN.md) before implementation.
The folder now has CPU-only generation, four-run SFT launchers, in-epoch Dev
evaluation and selected/untuned final-evaluation entry points. It does not yet
have a training-ready release or a validated server SFT/evaluation run.
The agreed scope needs no manual reapproval; validation remains required.

The planned training matrix is Natural/Symbolic crossed with seeds 42/43:
four fresh adapters on the same frozen dataset (generation seed 42). Dev is
evaluation-only during each run; select each run's epoch independently, not a
winning seed. Evaluate all four selected adapters on unified SFT-Bench and
1,500-question SpatialMap-TQA-Corr with original A-D plus shared Unique E/F.
`sft.py` prepares hash-bound configurations without copying/regenerating data.
The Axolotl plugin checks loader/collator/model settings before optimizer updates,
evaluates Dev loss and generated Stage-1 accuracy each epoch, and verifies PEFT
checkpoint reload. Separate evaluation jobs reuse the existing two-stage vLLM
runner; lm-eval-harness task registration remains pending. Current candidates
cannot pass the launch gate; release promotion/review and GPU validation remain.

Final-v1 admission policy `pinned-training-tokenizer-v1` uses only the pinned
Qwen3-4B-Thinking-2507 tokenizer for admission and exports. Its native-thinking
template and token budgets are checked at admission and after menu rebinding.
The inherited Qwen3.5 tokenizer and non-thinking admission format are removed
from this workflow; historical generation defaults and artifacts stay unchanged.

The inherited scientific requirements are recorded in
[`spatialentail-composition-v1`](experiments/spatialentail-composition-v1/README.md).
Read its [Unique-only specification](experiments/spatialentail-composition-v1/UNIQUE-ONLY-v1.md)
before planning or changing research code. It is the baseline for inherited
semantics and historical allocation. The final-v1 plan and user-confirmed
overrides below take precedence for the new experiment. The historical
vocabulary does not override them. Preserve old artifacts and receipts unchanged.

For final-v1, the user's external-evaluation clarification takes precedence:
SpatialMap-TQA-Corr is exactly 1,500 corrected SpatialMap-TQA questions under
`unique-entailed`, with no added omission companions and no requirement to cover
all SFT-Bench answer types. Do not inherit the old augmented external inventory.

Dev is held-out validation evaluated during SFT, not a final benchmark.
For final-v1 it is a new all-controlled set of 1,200 questions, one menu each,
with no omission companions and no question, scene or family overlap with
train or test. It matches train's query/answer/subtype proportions: 400 per
query, including 240 A-D, 120 E and 40 F. Retain 280 depth-cohort and 120
semantic-cohort examples per query, with 10 families per applicable proof-depth
stratum. The old independent per-depth tables and span-1-only contingent cases
are superseded by the joint level/semantic-case policy below.

SFT-Bench contains only the in-range test and extrapolation test, excluding
train and dev. SFT-Bench and SpatialMap-TQA-Corr evaluate the selected model
after training. Final-v1 exports one 3,600-question `data/sft-bench/` dataset;
reference rows retain `split: in-range-test` or `split: extrapolation-test`
for breakdowns from one evaluation run. Construction quotas and exclusion
checks remain separate. Train/dev/external exports are unchanged.

The latest final-v1 distribution policy is
`matched-controlled-uniform-levels-v2`: train, dev, in-range test
and extrapolation test all have 60% A-D, 30% E and 10% F per query, with the
same query-specific E-subtype proportions. Each question has one menu;
SFT-Bench is exactly 2,400 + 1,200 examples, not augmented menu views.
New test inventories and exclusion fingerprints are required; never relabel
old questions to fill quotas or use model results to select new test questions.

Construction now uses `explicit-size-checked-augmentation-v1`. At level L,
sample N objects uniformly from 2L+4 through 2L+8, then sample P-N uniformly
from 2 through 6. This reaches the explicit premise bounds 2L+6 through 2L+14
with 25 compatible pairs per level, not independent size ranges. Preserve the
successful trial's size-seed namespace. Core-first augmentation uses checked
counterexamples, contingent and proof-dependency witnesses, shortest-support
checks, and at most 512 solver fallback proposals. Multiple-answer Which has
exactly two entailed members; Count retains a depth-bearing membership obligation
without requiring every positive member to have a long proof. Existing full
semantic/evidence/token audits remain mandatory. Historical defaults are unchanged.
Five attempts are allowed per slot. Only the final attempt may accept a training-only canonical
query repeat if query, scene and family remain disjoint from held-out splits.
Record repeats and canonical frequencies; 6,000 training examples no longer
means 6,000 distinct canonical questions. Dev/test stay canonically unique.
No solver/evidence/token failure can be bypassed by this duplicate policy.

Final-v1 synthetic Count menus now use `synthetic-count-options-0-4-v1`: four
distinct options from 0-4 for every answer case. F omits the sole entailed
answer; the full Count semantics/domain is unchanged. This removes the
demonstrated large-number distractor cue without changing quotas or scene
seeds. Historical defaults and external menus are unchanged. Review sampling
now covers split/query/levels, contingent cases, Count values and longest
examples, with an explicit selected-not-reviewed receipt. Actual review remains
pending. Direction/variant seed coupling is unchanged: it is a coverage
limitation, not a demonstrated correctness defect.

Every applicable semantic case must span every assigned construction level
(1-8 in-range; 9-10 extrapolation), including inconsistent and no-entailment
cases and sole Which with contingent alternatives. Construction level is
not verified proof depth: inconsistent/ambiguous examples have null proof
depth. They require checked semantic evidence and coverage across levels, not
increasing contradiction or ambiguity difficulty. A short valid contradiction
at a higher level is not a defect; no semantic-difficulty scaling release gate
is required. Positive support depth remains checked, including semantic-cohort
sole Which (whose full verified depth stays null). Joint integer tables are
implemented in `generation_policy.py`; full-corpus feasibility, shortcut probes,
paired review and trainer validation remain pending. Generation never silently
promotes a candidate to a training-ready release.
All synthetic splits use the same controlled generation framework. Train,
dev and in-range use uniform construction levels 1-8; extrapolation uses
equal levels 9 and 10. This supersedes weighted train/dev levels and mixed-source
in-range targets. Balance correct A/B/C/D positions per split/query where
feasible. Rare subtype cells provide coverage, not precise per-level estimates.

The research direction is settled: a solver-verified atomic spatial-composition
dataset is the primary contribution, with matched Natural/Symbolic supervision
as a supporting comparison. Do not reopen the direction or expand the study
without explicit user approval. Surface correctness or feasibility blockers
with evidence; do not silently weaken release requirements to meet the schedule.

Reuse shared solver and training infrastructure. Preserve historical results,
identifiers, and frozen Corr SINGLE behavior. New contracts and workflows need
explicit versioning, not retroactive changes to old experiments.

## Historical vocabulary

The following terms describe earlier SpatialEntail problems, evidence, and
experimental dimensions. Old benchmark sizes, capability lenses, and optional
Symbolic supervision are not requirements for the composition study. Use the
Unique-only specification's descriptors and matched explanation requirements for new work.

## Language

**Spatial problem**:
A finite set of named objects, one visible premise formula, and one declared
query interpreted under the SpatialEntail semantic contract.
_Avoid_: Hidden map as the problem

**Spatial world**:
One complete assignment of X/Y relations that satisfies the visible premises
and the non-coincidence constraint.
_Avoid_: Gold answer source

**Query family**:
The requested projection of the shared entailment semantics: Direction
Identification (`DIR`), Entity Selection (`SEL`), or Cardinality Determination
(`CNT`). SpatialMap's source label `Which` adapts to `SEL`.
_Avoid_: Capability, difficulty tier

**Candidate status**:
The all-model classification of one query claim as entailed, contingent, or
impossible under consistent visible premises.
_Avoid_: Answer label, confidence

**Answer contract**:
The declared rule for turning candidate statuses into an output, such as one
invariant answer or all possible values.
_Avoid_: Output formatting alone

**Capability lens**:
A grouping used to present and evaluate related reasoning structures. It is not
a chronological training stage, universal hardness tier, or independent latent
cognitive module.
_Avoid_: Capability tier, difficulty level

**Relational Reasoning (RR)**:
Reasoning through relation access, inversion, decomposition, transitivity, and
axis recomposition.

**Logical Reasoning (LR)**:
Reasoning governed by Boolean operators, licensed rule applications, and branch
scope.

**Model-Theoretic Reasoning (MTR)**:
Reasoning about which query claims hold in every, some, or no satisfying
spatial model.
_Avoid_: Model-Based Reasoning, MR

**Reasoning-structure tag**:
An evidence-supported description of a relation or logical structure used in
an accepted derivation, such as `TRC`, `AXC`, `MP`, or `CSR`.
_Avoid_: Operator presence as verified use

**Semantic-status tag**:
A query-specific resolution condition such as unique/invariant, alternative
answers, multiple certain selections, or no match.
_Avoid_: Proof rule

**Evidence obligation**:
The claim that an explanation must establish, such as necessity,
impossibility, possibility, consistency, or joint Count coverage.
_Avoid_: Generic rationale, solver search log

**Answer certificate**:
A machine-checkable audit artifact establishing that a proposed answer satisfies
the declared semantics. It belongs to generation, verification, and evaluation;
it is not a default model response or SFT target.
_Avoid_: Reasoning trace, chain of thought, Symbolic supervision

**Natural solution trace**:
A readable, query-directed explanation rendered from verified reasoning for use
as an ordinary worked-solution target in SFT.
_Avoid_: Natural-language certificate dump, audit witness

**Compact symbolic trace**:
An optional CoS-like intermediate representation that abbreviates the
query-relevant spatial reasoning to reduce target and generation tokens. It is
not an answer certificate and is not required to be a replayable proof object.
_Avoid_: Certificate, formal proof, checked Symbolic

**Benchmark cell**:
A predeclared combination of query contract, support or rule structure,
semantic status, and evidence burden. Difficulty measurements annotate the
cell rather than define its task identity.
_Avoid_: Flat task bucket, depth tier

**Core benchmark**:
The 1,300-base coverage suite over the standard reasoning cells.

**Composition benchmark**:
The 1,200-base suite of new instances using composition families represented in
training.
_Avoid_: Comp benchmark, matched composition tier

**Shift benchmark**:
The 1,800-base structural-generalisation suite combining 900 held-out
composition instances with 900 depth-ladder instances. Its two components are
reported separately as diagnostics.
_Avoid_: Held-out composition tier, depth tier

**Structural difficulty**:
Measured properties such as spatial support depth, formula nesting, live branch
count, candidate-domain size, independent-axis support, distractors, and target
length.
_Avoid_: Trace length alone, depth as task identity

**External capability alignment**:
The relationship between an external task's native semantics and a declared
capability. Alignment may be direct, restricted, or absent.
_Avoid_: Shared spatial vocabulary as validation

**Native benchmark contract**:
An external dataset's stated input, assumptions, answer meanings, and scoring
rules, distinct from any adapted or newly audited view.
_Avoid_: Universal spatial answer contract
