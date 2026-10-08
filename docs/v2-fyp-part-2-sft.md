# Part II research plan: improving LLM spatial reasoning

Status: living experimental plan

This document develops the learning half of the project. It is subordinate to
the main [`FYP research canvas`](v2-fyp-research-canvas.md) and does not fix the
dissertation chapter structure or experiment numbering.

## Purpose

Part I defines formal spatial semantics and tests them through the SpatialEval
case study. Part II asks whether those semantics can improve spatial reasoning
in LLMs.

All new synthetic problem generation belongs to Part II. Part I only audits and
relabels the existing SpatialEval questions.

The formal system contributes to learning in three ways:

- it generates problems with known answer semantics;
- it produces proof-backed reasoning supervision, coordinate-free by default;
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
3. Do Natural or Symbolic traces generalise better?
4. How should intermediate spatial state be represented?
5. Does propositionally richer training add logical-spatial capability without
   damaging ordinary relational reasoning?
6. How do gains change with data scale and structural difficulty?
7. Does solver-verifiable RL add anything beyond strong SFT?

## Required inputs from Part I

The main training study starts only after these artifacts are stable:

- the qualitative spatial ontology and open-world semantics;
- the validated solver and text round trip;
- the three answer-mode definitions;
- the corrected SpatialEval case-study set; and
- a generator manifest that records problem structure and split identity.

SpatialEval is an external case-study and transfer evaluation. It is not used
as SFT training data.

## Generation provenance

The main research canvas distinguishes
[world-first, proof-first, and premise-first generation](v2-fyp-research-canvas.md#generation-provenance-world-first-proof-first-and-premise-first).
Part II inherits that distinction:

- world-first answers read from a privileged coordinate map are not valid gold
  labels;
- proof-first problems are the initial source of controlled, replayable trace
  supervision;
- premise-first solve-and-bucket problems are the preferred basis for testing
  generalisation beyond construction templates; and
- a secondary premise-first training pool may be added when concise proofs can
  be extracted and replayed.

Every track uses the same solver semantics, proof-checking boundary, and text
round trip. Z3 supplies an independent answer oracle; it does not by itself
supply the Natural or Symbolic trace. Coordinates may be retained as post-solve
witnesses and considered as a separate trace arm, but they must not be the
hidden source of both premises and labels.

## Evaluation suites

Do not combine every case into one arbitrarily weighted test. Freeze separate
modules and report each module directly.

| Suite | Purpose |
|---|---|
| Core relational | Positive spatial premises with Direction, Which, and Count queries |
| Spatial depth | Multi-hop X/Y proofs at increasing held-out depths |
| Independent axes | X and Y conclusions supported by different premise chains |
| Distractors | None, disconnected branches, and query-connected interference |
| Ambiguity | Unique answers and controlled possibility-set sizes |
| Answer contracts | `SINGLE`, `ALL_POSSIBLE`, and diagnostic `VISIBLE_POSSIBLE` |
| Propositional | Negation, disjunction, implication, equivalence, and mixed formulas |
| Language shift | Held-out templates and entity-name domains |
| External transfer | SpatialMap-TQA-Corr |

Use a development suite for generator and prompt decisions, a validation split
for checkpoint selection, and an untouched final test for reported claims. If
RL is designed after SFT errors are inspected, reserve a separate RL holdout.

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
| Zero-shot Symbolic Axis | Effect of a compact symbolic X/Y procedure |

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
ambiguous example for each of Direction, Which, and Count. Demonstration
selection must not depend on the test item's answer or semantic status.

### Demonstration content

At one fixed shot count, compare:

| Demonstration | Contents |
|---|---|
| Answer-only | Problem and final answer |
| Natural worked example | Natural reasoning trace and final answer |
| Symbolic worked example | Symbolic reasoning trace and final answer |

This separates the effect of seeing examples from the effect of seeing a
particular reasoning representation.

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
- Worked Natural or Symbolic demonstrations will outperform answer-only
  demonstrations on at least some compositional conditions.
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
| Natural Delta SFT | Natural-language derivation and incremental state |
| Symbolic Delta SFT | Symbolic derivation and incremental state |

The answer-only arm controls for learning the task distribution and output
format. Each reasoning-SFT arm must also be compared with the untuned model
using the same inference instruction; otherwise training and prompting effects
are confounded.

**Hypothesis:** at least one reasoning-trace condition will outperform
answer-only SFT on structurally held-out problems, even if matched accuracy is
similar.

## SFT study 2: reasoning representation

The representation comparison has two checked trace forms: Natural and
Symbolic. Both are rendered from the same certificate, so changing the view
does not change proof dependencies or the accepted answer evidence.

Keep the base problems, row order, model, optimizer, effective batch, epochs,
context limit, decoding, and evaluation prompts fixed. Report target-token
counts and training time because representation length is part of the proposed
mechanism.

**Hypotheses:**

- Delta will be more robust than Full at large depth because it avoids repeated
  state serialization.
- Symbolic traces may improve token efficiency and deep composition.
- Natural traces may transfer better to naturally worded external data.

The final two hypotheses are comparative, not assumptions that Symbolic or
Natural must win overall.

## SFT study 3: propositional training

The core solver supports arbitrary finite propositional formulas over its
spatial atoms, but the ordinary generator currently emits positive
conjunctions. Treat richer logic as a separate track.

### Evaluation modules

- explicit negation;
- disjunction and elimination;
- implication, including invalid-converse controls;
- equivalence;
- mixed Boolean formulas;
- unseen formula structures or greater Boolean depth; and
- the unchanged core relational suite for retention.

Track spatial proof depth and propositional depth separately. A problem may be
spatially deep but propositionally shallow, or the reverse.

### Controlled training comparison

| Arm | Composition | Question answered |
|---|---|---|
| Core SFT | Fixed number of positive-conjunction examples | Baseline spatial learning |
| Mixed SFT | Same total number, split between core and propositional examples | Effect of replacing some core data with logical diversity |
| Optional volume control | Core-only data matched to the larger Core+Proposition row or token budget | Whether a gain comes from logic rather than more data |

If an additive Core+Proposition model is trained without the volume control, it
can identify the strongest model but not the causal value of propositional data.

**Hypothesis:** mixed training will improve the propositional suites while
retaining core relational performance. It need not improve the original
SpatialEval case study, whose premises use a narrower language.

Before trace SFT, propositional generation must support deterministic text
round trips and faithful branched explanations. A satisfying coordinate model
is evidence that a formula is possible; it is not by itself a valid CoT proof.

## SFT study 4: data scale

Use nested training sets so every smaller set is contained in the next. The
working sizes are 1.5K and 6K, with a larger run only if the comparison retains
headroom.

**Hypothesis:** additional validated data will first improve matched accuracy,
then plateau unless the added examples cover the structures responsible for
held-out errors.

Scaling should use the selected representation. Do not multiply every data
size by every exploratory trace condition.

## Answer-contract study

The modes are different tasks, not interchangeable scoring options:

- `SINGLE` requests one invariant answer or `Cannot be determined`;
- `ALL_POSSIBLE` requests the complete possibility set; and
- `VISIBLE_POSSIBLE` requests the possible values displayed in the menu.

Use explicit wording for each mode. Keep `SINGLE` as the primary task,
`ALL_POSSIBLE` as the strict set-valued extension, and `VISIBLE_POSSIBLE` as a
menu-sensitivity diagnostic.

A fixed-budget comparison can contrast `SINGLE`-only training with a
`SINGLE`/`ALL_POSSIBLE` mixture. Add `VISIBLE_POSSIBLE` training only if the
diagnostic is important enough to justify another arm. Evaluate retention on
`SINGLE` whenever broader answer contracts are introduced.

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

- Split base problems before expanding answer or trace variants.
- Keep paired variants in the same split.
- Prevent overlap by canonical world as well as prompt text.
- Hold model, training recipe, data order, and decoding fixed within each
  ablation.
- Use one seed for screening and repeated seeds for reported comparisons.
- Use strict exact-answer-set accuracy as the primary metric.
- Macro-average declared structural cells.
- Report paired fixes and regressions, uncertainty intervals, output validity,
  and token cost.
- Never tune on the final test.

## Decision order

1. Validate the generator and create a development pool.
2. Calibrate structural cells with untuned LLMs.
3. Freeze evaluation modules and prompts.
4. Run the prompting baselines.
5. Run answer-only, Natural, and Symbolic SFT.
6. Run the propositional-data comparison.
7. Scale only the selected training design.
8. Test broader answer-contract training if it remains relevant.
9. Attempt RL only after its go/no-go conditions pass.

## Open decisions

- Final core and propositional evaluation cells
- Dataset sizes and repeated-seed budget
- Full factorial or staged representation comparison
- Exact Natural and Symbolic trace grammars for propositional proofs
- Baseline LLMs and frozen prompting conditions
- Whether `ALL_POSSIBLE` is a core training objective
- RL calibration thresholds and compute budget
