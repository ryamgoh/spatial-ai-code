# Spatial Reasoning

The active study follows the
[composition pivot](../experiments/spatialentail-composition-v1/pivot_20261026.md).
New dataset premises are atomic pairwise exact-direction relations.

## Implementation

| Path | Responsibility |
| --- | --- |
| `v2/solver.py` | Shared spatial semantics, Z3/reference backends, query analysis, witnesses |
| `v2/proofs.py` and certificate modules | Shared evidence construction and verification |
| `v2/generation.py` | Atomic-only spatial generation |
| `v2/generate_all.py` | Atomic-only workload CLI using the existing V2 answer interface |
| `v2/text.py`, `v2/grading.py`, renderers | Existing parsing, answer policy, and evidence rendering |
| `v2/tests/test_pivot_evidence.py` | Composition specification fixtures |
| `v2/tests/test_atomic_generation.py` | Atomic generation boundary regression tests |
| `v2/legacy/` | Retired Boolean curricula, randomized proof paths, and benchmark runners |
| `v2/study/` | Historical study workflows and reusable study utilities |
| `v1/` | Retired pre-V2 implementations |

Boolean curriculum controls have been removed from the active generator.
Historical callers use the [legacy namespace](v2/legacy/README.md). The shared
solver still needs conjunction, negation, and disjunction for consistency,
entailment, candidate exclusions, distinctness, and correlated Count checks.
Removing visible Boolean premises does not justify removing those operations.

The atomic-only generator retains the old V2 menus and explanation renderers.
It is not yet the new six-option, matched-explanation release pipeline.
The pivot's complete synthetic release gates still need to be implemented.
Do not use the legacy study builders as the new study's training recipe.

## Local Checks

```sh
uv run --python 3.12 --no-project --with pytest --with typer --with pyyaml --with z3-solver \
  python -m pytest spatial -q
python -m spatial.v2.generate_all --help
```

No historical datasets, results, or Corr SINGLE labels are rewritten by this
cleanup. New interface behavior must be versioned separately.
