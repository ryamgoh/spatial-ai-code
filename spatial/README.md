# Spatial reasoning

The solver and synthetic data generators live together here because they share
one domain model and one gold-label contract.

| File | Responsibility | Typical caller |
|---|---|---|
| `spatial_solver.py` | Frozen v6 answer oracle | Legacy v6 generation/evaluation |
| `spatial_solver_v2.py` | Data-agnostic formulas, Direction/Which/Count queries, Z3/reference engines, witnesses | Any structured spatial workload |
| `spatial_explanations_v2.py` | Structured claim evidence, axis proofs, qualitative domains, and ambiguity witnesses | Generators and audit reports |
| `spatial_explanation_renderers_v2.py` | Coordinate-free natural and symbolic traces with final, delta, or full state | SFT trace ablations |
| `spatial_audit_rendering_v2.py` | Coordinate-bearing witness reports | Benchmark audits only |
| `spatial_text_v2.py` | Current natural-language prompt to `SpatialProblem` adapter | Synthetic text round trips |
| `spatial_grading_v2.py` | Semantic resolution, menu encoding, and exact-set response scoring | Dataset-specific evaluation |
| `spatialeval_adapter_v2.py` | Original SpatialEval rows to structured cases and oracle audit status | SpatialEval audit pipeline |
| `test_spatial_solver_v2.py` | Core, adapter, policy, witness, and differential contracts | Local/CI verification |
| `test_spatial_grading_v2.py` | Answer semantics, menu cardinality, and exact-set scoring contracts | Local/CI verification |
| `test_spatial_explanations_v2.py` | Claim assessment and Direction/Which/Count explanation contracts | Local/CI verification |

Legacy generators remain separate:

- `generate_all.py`: frozen legacy SFT generator
- `generate_all_v6.py`: solver-validated SFT generator
- `generate_grpo.py`: prompt-only GRPO data generator
- `test_spatial_laws.py`: solver/generator contract tests

V2 deliberately remains separate from the V13 data contract. Its semantics,
including exact cardinals, coarse `*ward` relations, and negation, are defined
in `docs/spatial-solver-v2-contract.md`. The structured core supports
Direction, Which, and Count queries over arbitrary finite propositional spatial
formulas; dataset adapters choose the formula/query subset they expose.

The training launchers still work from `finetune/` and call these scripts
through `../spatial/`. Axolotl commands and config path semantics are
unchanged.

Run the domain tests from the repository root:

    uv run --python 3.12 --no-project --with pytest --with typer --with z3-solver \
      pytest spatial -q

The V2 core intentionally has no JSONL, prompt, option-letter, oracle, or
dataset-policy knowledge. New generators should construct `SpatialProblem`
directly, render text above that seam, parse the rendered text back through an
adapter, and verify the reparsed problem before emitting a row. They can pass
the same problem to `SpatialExplainerV2`, serialize its typed evidence, and use
`render_training_trace` with `TraceFormat.NATURAL` or `TraceFormat.SYMBOLIC`
and an independent `StateMode`. Coordinate witnesses belong only in structured
audit metadata or `render_audit_explanation`, never in an SFT reasoning target.
