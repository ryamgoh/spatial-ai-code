# SpatialEntail learning-study plan

Archive status: superseded by the
[composition pivot](../../../experiments/spatialentail-composition-v1/pivot_20261026.md).
The corpus and job statuses below belong to the historical study, not the new
composition dataset or its two matched adapters.

Historical status: production corpus sealed and four central SFT jobs prepared, 2026-10-12.
The repository receipt does not track remote submission or completion. No frozen
benchmark evaluation result is claimed here.

The canonical [task taxonomy](task-taxonomy.md) separates query family,
capability, reasoning structure, semantic status, evidence burden, and measured
difficulty. The former four-tier, 17-bucket allocation is withdrawn. The final
cell quotas, matched-test totals, and review strata are frozen in the sealed
production release.

This document develops the learning study within the unified FYP pipeline. It
is subordinate to the main [`FYP research plan`](research-plan.md) and does not
define dissertation chapter numbering.

## Purpose

The project first defines and audits formal spatial semantics through the
SpatialEval case study. SpatialEntail then asks whether supervision constructed
under the same semantics improves spatial reasoning in LLMs. Released
SpatialEval rows remain external evaluation data; all training problems are new
solver-validated SpatialEntail examples.

The formal system contributes to learning in three ways:

- it generates problems with known answer semantics;
- it validates the answers and Natural worked solutions used for supervision;
  and
- it provides exact answer verification for evaluation and optional RL.

The aim is not merely to train a model that fits generated examples. The model
must improve on held-out spatial structures without relying on answer-format,
template, entity-name, or menu-position shortcuts.

## Research question

> **Which forms of formally validated data and reasoning supervision improve
> spatial reasoning and structural generalisation in large language models?**

Supporting questions:

1. How much can prompting elicit without parameter updates?
2. Does SFT improve beyond prompt-only and answer-only baselines?
3. Does a verified Natural worked solution improve over answer-only SFT?
4. As an optional extension, can a compact CoS-like representation reduce
   token cost without reducing accuracy?
5. Does propositionally richer training add logical-spatial capability without
   damaging ordinary relational reasoning?
6. How do gains change across structural difficulty and transfer conditions?
7. Does solver-verifiable RL add anything beyond strong SFT?

## Required semantic inputs

The sealed release was prepared only after these artifacts were stable:

- the qualitative spatial ontology and open-world semantics;
- the validated solver and text round trip;
- the three answer-mode definitions;
- the corrected SpatialEval case-study set; and
- a generator manifest that records problem structure and split identity.

SpatialEval is an external case-study and transfer evaluation. It is not used
as SFT training data.

## Generation provenance

The main research canvas distinguishes world-first, proof-template-first,
certificate-first, and premise-first generation. The learning study preserves
that distinction:

- world-first answers read from a privileged coordinate map are not valid gold
  labels;
- proof-template-first problems are the current source of controlled,
  solver-validated answer-only and Natural supervision;
- certificate-first generation remains a stricter future provenance in which
  the typed proof exists before the problem text;
- premise-first solve-and-bucket problems are the preferred basis for testing
  generalisation beyond construction templates; and
- premise-first generation is implemented for supported atomic and Boolean operator-family cells, with
  acceptance conditioned on solver classification and replayable evidence; its
  distribution and extraction success still need empirical characterization.

Every track uses the same solver semantics, proof-checking boundary, and text
round trip. Z3 supplies an independent answer oracle; it does not by itself
supply the SFT response. Coordinates and replayable certificates remain
post-solve audit artifacts. Natural worked solutions may be rendered from the
verified derivation, but they should include only query-relevant reasoning.

## Authoritative execution scope

The reviewed [corpus and benchmark protocol](size-freeze-protocol.md) controls
all sizes, roles, composition reservations, and central contrasts. Its sealed
allocations supersede the former 17K and 5.8K-test programme. The primary study
is one 4B model, answer-only and Natural packages, and separately reported
`SpatialEntail-SINGLE-Bench` CORE, COMP, and SHIFT evaluations. Each section
reports its cumulative score and cell-level breakdown; SHIFT slice labels and
depth curves are diagnostics. The release contains
8,000 Train, 775 Dev, and 4,300 Bench bases and passed uniqueness, context,
coverage, semantic, replay, and quality gates.

## Baseline prompting study

Prompting measures what the untuned LLM can already do and provides matched
baselines for SFT. Demonstration count and reasoning instruction are separate
variables and should not be changed in the same comparison.

### Reasoning instruction

Hold the demonstration count at zero:

| Condition | Purpose |
|---|---|
| Zero-shot Direct | Native task performance with minimal instruction |
| Zero-shot Generic CoT | Effect of generic step-by-step prompting |
| Zero-shot Natural Axis | Effect of an explicit natural-language X/Y procedure |
| Zero-shot Compact Axis (optional) | Future token-efficiency probe using a separately defined CoS-like X/Y procedure |

This comparison asks whether the formal procedure can be elicited without
examples or parameter updates.

### Demonstration count

Hold the instruction style fixed and compare:

| Condition | Demonstrations |
|---|---:|
| Zero-shot | 0 |
| One-shot | 1 |
| Few-shot | A fixed `k`, provisionally 6 |

Use the exact value of `k` in all reports rather than the vague label
“N-shot.” One-shot performance must be averaged across several independently
selected demonstrations because a single example can dominate the result.

For the primary `SINGLE` task, a six-shot prompt can cover one unique and one
ambiguous example for each of Direction, Selection, and Count. Demonstration
selection must not depend on the test item's answer or semantic status.

### Demonstration content

At one fixed shot count, compare:

| Demonstration | Contents |
|---|---|
| Answer-only | Problem and final answer |
| Natural worked example | Natural reasoning trace and final answer |

This separates the effect of seeing an answer from seeing a verified worked
solution. A compact-symbol demonstration belongs only in the future
token-efficiency extension and must not use replayable NDJSON certificates.

All demonstrations must be solver-validated, disjoint from evaluation worlds,
balanced in answer letters and semantic status, and drawn from a frozen pool.
Record demonstration IDs, order, prompt tokens, answer mode, reasoning format,
and decoding settings. Report variation across one-shot exemplars and check
whether gains reflect spatial accuracy, answer formatting, or increased use of
`Cannot be determined`.

Develop prompts on the development suite and freeze them before final
evaluation. Run the full prompt comparison on the target model. A wider LLM
survey should use one common direct prompt and at most one selected structured
prompt instead of crossing every model with every prompt.

**Hypotheses:**

- Explicit axis decomposition will help compositional problems relative to
  direct and generic CoT prompting.
- Few-shot examples will improve answer-contract and format compliance more
  consistently than deep structural reasoning.
- Verified Natural demonstrations may outperform answer-only demonstrations on
  at least some compositional conditions.
- One-shot results will vary more across exemplar choices than balanced
  few-shot results.

Freeze the strongest prompting condition as an SFT baseline. Each SFT model
must also be compared with an untuned model receiving the same inference
instruction so that prompt and training effects remain separable.

## SFT study 1: does reasoning supervision help?

Use the same underlying training problems and compare:

| Arm | Target |
|---|---|
| Untuned, prompt-matched | No parameter update |
| Answer-only SFT | Final answer only |
| Verified Natural SFT | Concise, query-directed worked solution rendered from checked reasoning |
| Compact Symbolic SFT (optional) | CoS-like abbreviation of the query-relevant reasoning, evaluated for token reduction |

The central arms are answer-only and verified Natural SFT on Qwen3.5-4B, with a
prompt-matched untuned baseline. Use fixed epochs and the final checkpoint. A
second seed remains desirable for run-to-run stability, but it is not used to
inflate the project into a factorial study. Qwen3.5-2B, compact Symbolic SFT,
and GRPO remain optional after the central study.

**Hypothesis:** a verified Natural worked-solution condition may outperform
answer-only SFT on structurally held-out or external problems, even if matched
accuracy is similar.

## Optional SFT extension: compact symbolic representation

This optional arm follows Chain-of-Symbol's practical question: can the
query-relevant reasoning be represented with fewer tokens than a Natural worked
solution while retaining answer accuracy? It must use a compact, declared
spatial notation. Replayable NDJSON answer certificates, global audit witnesses,
and menu-decision records are not this arm's target.

Keep base problems, model initialization, optimizer, and evaluation decisions
paired. Report corpus-wide target tokens, generated tokens, runtime, and answer
accuracy. The arm is worth running only if the frozen representation is actually
shorter than Natural on the admitted corpus. It is an optional token-efficiency
exploration, not a central claim about formal verification or hidden reasoning.

## Composition, allocation, and depth

See [size-freeze-protocol.md](size-freeze-protocol.md) for the executable twelve
seen composition cells and nine held-out cells, paired semantic interventions,
775-base Dev set, and 4,300-base benchmark. These are sealed artifact counts,
not provisional allocation targets.

The sealed study uses 8,000 training bases. The nested 4K artifact supports
engineering checks but is not a required scaling experiment, and no saturation
or optimal-size claim follows from it. Depth 3/4 anchors and 5–8 extrapolation
are scored separately; deep answers use a common direct-answer instruction.

## Answer-contract study

The modes are different tasks, not interchangeable scoring options:

- `SpatialEntail-SINGLE` applies the query-specific unique-answer rule: one possible Direction
  or Count, or exactly one entailed Selection entity with no other possible
  entity; otherwise it returns `Cannot be determined`;
- `ALL_POSSIBLE` requests every possible value in the query's declared candidate
  domain; for Selection this is the union of individually possible entities,
  not a domain of complete membership sets; and
- `VISIBLE_POSSIBLE` requests the possible values displayed in the menu.

Use explicit wording for each mode. Keep `SpatialEntail-SINGLE` as the primary
task, `SpatialEntail-ALL-POSSIBLE` as the strict set-valued extension, and
`SpatialEntail-VISIBLE-POSSIBLE` as a menu-sensitivity diagnostic.

The set-valued prompts define "possible" against the stated relations rather
than a hidden true map: "Select every answer that could be true given the
stated relations." `VISIBLE_POSSIBLE` adds that possible answers which are not
listed do not affect the selection. `SpatialEntail-SINGLE-Bench` was sealed
before this wording was fixed and uses only the unchanged `SINGLE` instruction.

`SpatialEntail-ALL-POSSIBLE-Bench` uses full menus only: every possible value
is displayed, so the complete set is always expressible. Partial-menu
`ALL_POSSIBLE` items would let `Cannot be determined` absorb uncertainty and
weaken the stricter contract. Partial menus belong to the
`SpatialEntail-VISIBLE-POSSIBLE` probe, built from the same bases as the
`ALL_POSSIBLE` bench so that menu coverage is the only changed variable.

A fixed-budget comparison can contrast `SINGLE`-only training with a
`SINGLE`/`ALL_POSSIBLE` mixture. Add `VISIBLE_POSSIBLE` training only if the
diagnostic is important enough to justify another arm. Evaluate retention on
`SINGLE` whenever broader answer contracts are introduced.

## Source-blinded quality review

Formal semantic, certificate, round-trip, split, menu, and context checks cover
every selected row. Let `C` be the number of frozen reporting cells. Review uses
at least the larger of 2% of the formally admitted maximum candidate pool or
`5C` examples,
stratified with at least five examples from each cell. Version-pinned
`DeepSeek-V4-Pro` and `GPT-5.5` review the same anonymized prompt and verified
Natural trace in sequential, isolated terminal-agent runs. Replayable Symbolic
certificate quality remains an offline parser-and-replay check.

Two approvals accept an item. A split verdict is held out for revision or
replacement without human review. Two rejections trigger a separate
payload-blinded human Approve/Reject appeal; model reasons remain hidden. Human
approval records a consensus false-positive rejection, while human rejection
confirms removal or revision. Release requires 100% automatic validity and at
least 95% final sample acceptance after routing. Two or more final
non-acceptances in one reporting cell trigger revision and re-audit. Report
model–model agreement and Cohen's kappa, disposition counts, and the human
override rate among double rejections. Selective human review cannot estimate
overall human acceptability or LLM–human kappa.

## Optional RL study

RL is attempted only when:

- the selected SFT model leaves meaningful hard-holdout headroom;
- sampled completion groups contain both correct and incorrect answers;
- answer parsing is stable; and
- the RL holdout has not influenced SFT or reward design.

Compare the frozen SFT model, compute-matched continued SFT, and SFT+GRPO. Use
the solver for exact answer-set rewards and report format reward separately.
Track reward variance, strict accuracy, format validity, completion length, and
retention.

## Experimental controls

- Split base problems before expanding paired variants. Train, development, and
  test must have no overlapping base IDs, prompts, or canonical source signatures.
- Structural signatures canonicalize visible formulas and queries, ignoring
  entity names and premise or commutative ordering. Exact canonicalization has a
  permutation budget; its conservative fallback may merge distinct problems.
  Signature separation is not a guarantee of unseen proof compositions. Reserve
  explicit cells before model selection for each structural generalisation claim.
- Shuffle special and ordinary options together and report answer-position counts.
- Hold model, training recipe, and decoding fixed within each comparison; select
  checkpoints on development data only. Report seeds 42 and 43 separately, with
  their mean and range.
- Report strict exact-answer-set accuracy, cell macro-averages, paired changes,
  uncertainty, output validity, and token/resource costs.
- Score central models by generated answers under the benchmark's declared
  contract. Gold certificates verify the dataset and oracle offline; the model
  is not required to reproduce them.
- `answer-only` omits a worked solution. Natural contains a verified
  worked solution followed by the answer. If the optional compact Symbolic arm
  runs, measure its token reduction and answer accuracy; do not substitute the
  replayable NDJSON audit certificate for that representation.
- Use `ContextBudget` with the selected tokenizer and chat template. Admit the
  full training chat, the evaluation prompt plus generation reserve, and any
  configured target limit. Reject the whole paired base group when any required
  variant overflows. Record rejected groups and exact lengths; never truncate.
  Evaluation consumes `metadata.evaluation_prompt` without retemplating or
  adding special tokens. Character and whitespace counts are only diagnostics.
- Report actual updates and runtime alongside token-matching multipliers; those
  multipliers do not establish matched training compute.

Admission also checks that the training chat begins with the exact evaluation
prompt, and measures the remaining rendered completion including termination
tokens against the generation limit. Raw assistant-text token counts alone do
not capture that cost. Lossy or incompatible chat templates are rejected.

## Implemented preparation and pending model runs

The bounded integration pilot predates the current boundary and wires checked
Natural, replayable NDJSON Symbolic, answer-only, and invalid Symbolic verifier
fixtures. It exercises unique depth-2 and two-way ambiguous Direction cells
with three entities and two premises. Its configured limits are 4,096 training
tokens, 8,192 evaluation-context tokens, and a 4,096-token generation reserve.
This is solver and data-pipeline validation, not the compact CoS-like optional
arm or the central SFT design.

That bounded configuration did not exercise the final Selection, Count,
Boolean, transfer, depth, or quality-review protocols. The later sealed
production release does. SFT and model evaluation have not been run. See the
[preparation audit](../audits/sft-job-preparation.md) for final verification
evidence and the [remediation audit](../audits/critique-remediation.md) for the
earlier pilot.

The separate `matrix-premise-holdout.yaml` reserves premise-first IFF development
cells and premise-first implication test cells against atomic training data.
Its 8/4/4 base counts and common three-entity/two-premise budgets are a preparation
check, not an empirical training comparison. Consult the remediation ledger for
the executed tokenizer smoke status.

The final tokenizer/generator smoke admitted all 80 pilot bases (320 rows):
48 training, 16 development, and 16 test bases. Maximum full rendered-row lengths
were 4,032 tokens for valid/invalid Symbolic verifier fixtures, 1,863 for Natural,
and 129 for answer-only; the maximum reserved evaluation length was 4,218 and maximum
rendered generation target was 3,910. The 16-base holdout configuration also
passed, with maximum training length 965 and reserved evaluation length 4,265.
These are preparation diagnostics for the executed configuration, not model
accuracy or learning results. The earlier four-entity/three-premise proposal
failed admission and was narrowed equally across arms.

The current taxonomy feasibility pilot generated 50 checked Natural examples
for each of thirteen ready representative cells (650 total), after implementing
multiple-certain Selection admission, correlation-sensitive Count admission,
and removable Selection distractors. A deterministic five-per-cell packet was
screened sequentially by `DeepSeek-V4-Pro` and `GPT-5.5`; both approved all 65
items, so the human appeal queue was empty. Exact hashes and local artifact paths
are summarized in the tracked [v3 audit](../audits/capability-pilot-v3.md).
This historical pilot validated the representative registry and automated
quality workflow. The later production release froze the final allocation and
completed its review; neither artifact establishes learning results.

## Execution status and next steps

Corpus generation, formal validation, source-blinded review, sealing, and
creation of four answer-only/Natural configs and scripts are complete. The
remaining central work is to freeze inference prompts, run the untuned
baselines, submit the four prepared jobs, and evaluate their generated answers
on matched, depth, composition, and external suites without retuning. A compact
CoS-like arm may be designed later only if it is measurably shorter than Natural;
GRPO remains conditional on its go/no-go checks.

## Open decisions

- Frozen inference prompts and decoding settings
- Optional compact Symbolic notation and its token-reduction threshold
- Whether `ALL_POSSIBLE` is a core training objective
- RL calibration thresholds and compute budget
