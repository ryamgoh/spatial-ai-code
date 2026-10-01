# Spatial V2 ablation matrices

An experiment matrix declares exact base-problem cells and the answer/trace
variants rendered from each base problem. It belongs above the data-agnostic
solver and generator.

Run the checked-in example with:

```bash
uv run --python 3.12 --no-project --with typer --with pyyaml --with z3-solver \
  python spatial/generate_matrix_v2.py \
  experiments/spatial-v2-ablation.example.yaml \
  --out data/spatial_v2_ablation.jsonl
```

This writes `_train.jsonl`, `_test.jsonl`, and `_manifest.json` artifacts.

## Contract

```yaml
version: 1
seed: 42
test_split: 0.2
include_audit: false

defaults:
  max_attempts_per_sample: 2000
  omit_direct_query_relation: true

variants:
  answers:
    - {name: single, mode: single, menu_coverage: full}
    - {name: complete, mode: all-possible, menu_coverage: full}
    - {name: visible-partial, mode: visible-possible, menu_coverage: partial}
  traces:
    - {format: natural, state_mode: delta}
    - {format: symbolic, state_mode: delta}

cells:
  - name: direction-depth-2
    count: 100
    query_kind: direction
    semantic_shape: unique
    depth: 2
    num_entities: 6
    num_premises: 7
    target_directions: [North, Northeast, East, Southeast, South, Southwest, West, Northwest]
    answer_variants: [single, complete]
```

`count` applies to every direction value listed by the cell. The example above
therefore requests `100 × 8 = 800` base problems. Each base is then crossed
with the selected answer and trace variants. With two answer variants and two
trace variants, it produces `800 × 2 × 2 = 3,200` rows.

`depth` is exact: Direction cells set both axis-depth bounds to that value;
Which and Count cells set both positive-membership-depth bounds. Separate cells
for depths 2 through 10 therefore produce a balanced depth ablation. A range is
not sampled implicitly.

Answer variants may be selected per cell. Partial menu coverage is valid only
for ambiguous cells. This permits one matrix to compare:

- `SINGLE`, which requires one invariant answer;
- `ALL_POSSIBLE`, which requires the complete possibility set; and
- `VISIBLE_POSSIBLE`, which selects only displayed possible values.

Every answer variant is crossed with every selected trace variant on the same
`base_id`. The complete group remains in one train/test split. The manifest
checks the requested Cartesian product and reports exact generated counts per
cell.

Unknown fields, duplicate names, incompatible policies, incomplete variant
groups, duplicate rows, and cross-split leakage are errors. They are never
silently ignored.
