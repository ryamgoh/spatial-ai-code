# SpatialEntail generation and experiment knobs

This is the authoritative catalogue of controls exposed by the active V2 data
generator and experiment matrix. The formal meaning of accepted problems is
defined by [Semantics](semantics.md); certificates are defined by
[Certificate system](certificates.md); rendered outputs are defined by
[Trace formats](trace-formats.md).

The controls below do not belong to the solver. They select, filter, render,
split, or admit problems after the solver has established their semantics.

## Knob classes

The configuration surface contains four different kinds of control:

1. **Problem controls** change the query, formula, answer contract, or menu.
2. **Difficulty controls** require measured properties of accepted evidence.
3. **Supervision controls** change what is printed for the same accepted base
   problem.
4. **Experiment controls** govern sampling, splitting, and token admission.

These categories should not be conflated. In particular, a requested template
is a generation condition; measured proof depth and semantic support describe
the accepted problem.

## Complete policy catalogue

### Problem semantics and questions

| Field | Values | Effect | Important constraints |
|---|---|---|---|
| `query_kind` | `direction`, `which`, `count` | Selects the question and certificate family. | Direction compares one target/reference pair. Which and Count use declared candidates and a requested relation set. |
| `semantic_shape` | `any`, `unique`, `ambiguous`, `no-match` | Filters the all-model answer domain. | `no-match` is currently realizable only for Which. Controlled proof-depth cells require `unique`. |
| `ambiguity_size` | integer ≥ 2 or omitted | Requires exactly this many possible semantic answers. | Applies to ambiguous cells; the feasible maximum depends on the query domain. |
| `query_direction` | one of eight exact directions or omitted | Fixes the requested membership direction for Which/Count. | Invalid for Direction queries. Matrix files can enumerate `query_directions`. |
| `target_direction` | one of eight exact directions or omitted | Requires a particular unique Direction answer. | Requires a Direction query and `semantic_shape: unique`. Matrix files can enumerate `target_directions`. |

The exact directions are North, Northeast, East, Southeast, South, Southwest,
West, and Northwest. Direction-sensitive structural signatures do not identify
rotated problems.

### Answer and menu contract

| Field | Values | Effect | Important constraints |
|---|---|---|---|
| `answer_mode` | `single`, `all-possible`, `visible-possible` | Defines whether the answer is one invariant result, the complete semantic possibility set, or only visible possible menu values. | `visible-possible` is diagnostic rather than headline exact semantics. |
| `menu_coverage` | `full`, `partial`, `zero` | Controls whether all, some, or none of the semantic answers occur as ordinary menu options. | `partial` is undefined for `single`. |
| `ordinary_option_target` | 1–25 | Sets the requested number of ordinary menu entries before any special answer. | Exposed by `GenerationPolicy`; the strict matrix currently uses its policy default. |

All ordinary and special options participate in one seeded final shuffle.
`Cannot be determined` and `None of the Options` therefore have no fixed letter.

### Formula and provenance controls

| Field | Values | Effect | Important constraints |
|---|---|---|---|
| `boolean_shape` | `atomic`, `modus-ponens`, `iff`, `double-negation`, `disjunctive-syllogism`, `case-split`, `nested-case-split` | Selects an atomic or Boolean syntax/proof-template family. | Outside premise-first mode, non-atomic shapes require unique/any policy semantics, at least four entities (six for nested cases), and cannot combine with axis-depth, direct-relation omission, independent-axis, or distractor controls. A crossed `WorkloadSpec` is stricter: all non-premise-first non-atomic cells must be `unique`. In premise-first mode the name identifies the sampled syntax family, not a promise that the shortest accepted proof uses the similarly named rule. |
| `generation_mode` | `auto`, `world-first`, `proof-template-first`, `premise-first` | Selects how a candidate problem is proposed. | World-first supports atomic premises only. Unsupported combinations fail instead of silently rerouting. |

The recorded provenance reflects the route actually used:

- **world-first:** propose a coordinate world, discard its answer, and re-solve
  the visible premises;
- **proof-template-first:** instantiate a controlled formula/proof family, then
  solve and certify it;
- **premise-first:** sample visible formulas before choosing the query, then
  solve, classify, and certify without an answer-bearing map;
- **auto:** choose the supported route implied by the policy.

`certificate-first` remains vocabulary for a stricter proposed construction and
is not an implemented `GenerationMode`.

### Size and measured difficulty

| Field | Values | Effect | Important constraints |
|---|---|---|---|
| `num_entities` | 2–30 | Controls the named-object domain. | Thirty is the current name-pool size, not a semantic solver limit. |
| `num_premises` | connected minimum through all unordered pairs | Controls visible formula count. | The generator requires enough premises to connect the selected entities. |
| `omit_direct_query_relation` | Boolean | Forbids a premise that directly states the query relation. | Requires at least three entities and unique controlled semantics. |
| `min_axis_depth`, `max_axis_depth` | positive integers | Bounds measured X/Y proof depth for Direction. | Applies only to unique Direction cells. |
| `min_membership_depth`, `max_membership_depth` | positive integers | Bounds positive membership-proof depth for Which/Count. | Requires at least one entailed member; correlated counts without one do not receive a fabricated depth. |
| `require_independent_axes` | Boolean | Requires X and Y conclusions to depend on disjoint supporting premises. | Requires the relevant minimum depth to be at least two. |
| `distractor_premises` | non-negative integer | Requires this many jointly removable premises after preserving answer and membership semantics. | Currently Direction-only and no greater than `num_premises`. Support is sufficient and order-dependent, not globally minimum. Incomplete semantic checks retain premises and set `support_checks_complete: false`. |

Difficulty is measured on accepted problems. Generator intent alone does not
establish the reported depth, independence, or distractor count.

### Coupled feasibility rules

The constructor rejects impossible combinations before sampling:

- `num_premises` must be at least `num_entities - 1` and no greater than the
  number of unordered entity pairs;
- requested proof depth cannot exceed either `num_entities - 1` or
  `num_premises`;
- direct-relation omission requires at least three entities, and a protected
  pair cannot be omitted when the premise budget already consumes every pair;
- independent axes require depth at least two and reserve twice the requested
  depth, so `2 × depth + distractor_premises` must fit the premise budget;
- without independent axes, `depth + distractor_premises` must fit;
- `ambiguity_size` requires ambiguous semantics and cannot exceed eight for
  Direction, `num_entities - 1` for Which, or `num_entities` for Count; and
- depth, independent-axis, direct-omission, and distractor controls require
  unique semantics. Membership depth is invalid for Direction; axis depth is
  invalid for Which/Count.

These are compatibility rules, not suggestions. Invalid cells fail before an
attempt loop; feasible but selective cells may still exhaust
`max_attempts_per_sample` and report their rejection histogram.

## Rendering and supervision catalogue

| Field | Values | Effect | Important constraints |
|---|---|---|---|
| `trace_format` | `natural`, `symbolic` | Selects controlled prose or versioned replayable NDJSON. | Only arbitrary Symbolic model output has a process replay parser. |
| `supervision_arm` | `checked-trace`, `answer-only`, `corrupted-trace` | Selects valid evidence, no evidence, or deliberately invalid evidence with the same answer. | Corruption is Symbolic-only and must preserve syntax/decision while failing reasoning replay. Answer-only is emitted once per answer/menu variant, not once per trace format. |
| `include_audit` | Boolean | Adds full certificate and coordinate-bearing generation witness metadata. | Ordinary training evidence remains qualitative and coordinate-free. |

Natural and Symbolic rows originate from the same checked certificate, problem,
menu, and gold answer. This makes them paired supervision packages; it does not
by itself make their rendered information or token cost identical.

## Workload and split catalogue

| Field | Values | Effect | Important constraints |
|---|---|---|---|
| `samples_per_cell` / matrix `count` | non-negative / positive integer | Requests accepted base problems per crossed cell. | A direction-enumerated cell applies `count` per listed direction. |
| `seed` | integer | Controls reproducible proposal, menu, group, and split shuffling. | Record it with the generated manifest and software revision. |
| `max_attempts_per_sample` | positive integer | Bounds rejection sampling for each requested base. | Exhaustion reports the rejection histogram. |
| `dev_split`, `test_split` | fractions in `[0,1)` | Requests three-way train/development/test allocation. | Their sum must be below one. Development selects checkpoints; test is final only. |
| `split_strategy` | `random`, `structural`, `holdout` | Selects base-group splitting, canonical-structure clustering, or declared-cell routing. | Paired variants never cross splits. Holdout cell declarations must agree with requested dev/test presence. |
| `holdout_cells` | named dev/test cell lists | Reserves predeclared matrix cells. | Used only with `holdout`; listed cells must exist and be disjoint. |
| `replace` | Boolean | Authorizes replacing this workload's exact output paths. | Failed admission or validation preserves prior successful data and writes diagnostics. |

### Python workload versus matrix names

`GenerationPolicy` describes one cell and therefore uses singular fields. The
`WorkloadSpec` API crosses collections of values and uses the following plural
fields:

| `WorkloadSpec` field | Per-cell or matrix equivalent |
|---|---|
| `query_kinds` | cell `query_kind` |
| `answer_modes` | named `variants.answers[].mode` |
| `semantic_shapes` | cell `semantic_shape` |
| `menu_coverages` | named `variants.answers[].menu_coverage` |
| `trace_formats` | `variants.traces[].format` |
| `supervision_arms` | `variants.supervision_arms` |
| `boolean_shapes` | cell `boolean_shape` |
| `query_directions` | cell `query_directions` |
| `target_directions` | cell `target_directions` |
| `context_budget` | root `context_budget` mapping |

The remaining `WorkloadSpec` fields use the same singular names documented in
the tables above: `generation_mode`, `num_entities`, `num_premises`,
`omit_direct_query_relation`, both depth ranges, `require_independent_axes`,
`ambiguity_size`, `distractor_premises`, `max_attempts_per_sample`, `dev_split`,
`test_split`, `split_strategy`, `seed`, `include_audit`, and `replace`.

At matrix level, `version`, `seed`, `dev_split`, `test_split`,
`split_strategy`, `holdout_cells`, `context_budget`, `include_audit`, `defaults`,
`variants`, and `cells` are the only accepted root keys. Each cell requires a
unique `name`, requests a base count with `count`, and may restrict named
`answer_variants`; its other fields are the per-cell controls above. Unknown
keys fail validation.

Structural grouping ignores entity names, premise/candidate order, and
commutative operand order. It retains exact directions and implication order.
Exact canonicalization is bounded at 40,320 residual permutations; the fallback
is invariant but coarse and may merge distinct structures. A structural split
is not logical equivalence or a proof-composition holdout.

## Context-admission catalogue

| Field | Meaning |
|---|---|
| `tokenizer_name` | Tokenizer whose chat template and encoding define admission. |
| `tokenizer_revision` | Pinned model/tokenizer revision used by generation, training, and evaluation. |
| `train_max_tokens` | Maximum complete rendered training sequence. |
| `eval_max_tokens` | Maximum rendered evaluation prompt plus generation reserve. |
| `max_new_tokens` | Reserved and enforced generation suffix budget. |
| `target_max_tokens` | Optional stricter completion cap. |

Admission renders with the selected chat template, counts the full sequence,
checks the exact generatable suffix including terminators, and binds the stored
evaluation prompt to its source messages with integrity fingerprints. Every
required variant for a base is accepted or rejected as one group. Failure writes
a durable rejection report instead of silently dropping one representation or
semantic stratum.

## Matrix configuration

```yaml
version: 1
seed: 1515
dev_split: 0.2
test_split: 0.2
split_strategy: structural
include_audit: false

context_budget:
  tokenizer_name: Qwen/Qwen3.5-4B
  tokenizer_revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
  train_max_tokens: 4096
  eval_max_tokens: 8192
  max_new_tokens: 4096

defaults:
  num_entities: 3
  num_premises: 2
  max_attempts_per_sample: 2000

variants:
  answers:
    - {name: single, mode: single, menu_coverage: full}
  traces:
    - {format: natural}
    - {format: symbolic}
  supervision_arms: [checked-trace, answer-only, corrupted-trace]

cells:
  - name: direction-depth-2
    count: 5
    query_kind: direction
    semantic_shape: unique
    depth: 2
    omit_direct_query_relation: true
    target_directions: [North, Northeast, East, Southeast, South, Southwest, West, Northwest]

  - name: direction-ambiguity-2
    count: 40
    query_kind: direction
    semantic_shape: ambiguous
    ambiguity_size: 2
```

`depth` is matrix shorthand for axis depth on Direction or membership depth on
Which/Count. Unknown keys and unsupported combinations fail validation.

## Current executed preparation coverage

The bounded pilot exercises:

- Direction queries;
- unique depth-two versus ambiguity-size-two semantics;
- all eight target directions for the unique cell;
- Natural versus Symbolic checked traces;
- answer-only and corrupted-Symbolic controls;
- structural train/development/test splitting;
- identical three-entity/two-premise budgets across semantic strata; and
- pinned Qwen3.5-4B context admission.

Its tokenizer/generator smoke admitted 80 bases and 320 rows, split 48/16/16
bases. Maximum full training length was 4,032 tokens under the 4,096 limit.

The separate `matrix-premise-holdout.yaml` exercises atomic training,
premise-first IFF development, and premise-first implication testing over 16
bases. These are preparation checks, not trained-model results.

## Commands and outputs

```bash
uv run --python 3.12 --no-project --with typer --with pyyaml \
  --with z3-solver --with transformers --with jinja2 \
  python -m spatial.v2.generate_matrix \
  experiments/15-v2-ablation/matrix.yaml \
  --out data/spatial_v2_pilot.jsonl
```

Generation writes master train/development/test JSONL, a manifest, rejection
diagnostics when applicable, and materialized `by_variant`/`by_cell` views. The
manifest—not the requested YAML—is the authority for actual counts, paths,
distributions, fingerprints, and admitted token volumes.

The runnable training/evaluation orchestration is documented beside its code in
[`experiments/15-v2-ablation`](../../experiments/15-v2-ablation/README.md).
Training benefits, readable explanation quality, and internal reasoning
faithfulness remain empirical questions.
