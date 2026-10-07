# Spatial reasoning

The solver and synthetic data generators live together here because they share
one domain model and one gold-label contract.

| File | Responsibility | Typical caller |
|---|---|---|
| `spatial_solver.py` | Frozen v6 answer oracle | Legacy v6 generation/evaluation |
| `spatial_solver_v2.py` | Data-agnostic formulas, Direction/Which/Count queries, Z3/reference engines, witnesses | Any structured spatial workload |
| `spatial_explanations_v2.py` | Structured claim evidence, axis proofs, qualitative domains, and ambiguity witnesses | Generators and audit reports |
| `spatial_explanation_renderers_v2.py` | Coordinate-free natural and symbolic traces with final, delta, or full state | SFT trace ablations |
| `spatial_proofs_v2.py` | Typed Direction proof certificates, deterministic construction, replay checking, and serialization | Proof-first SpatialEntail generation |
| `spatial_proof_renderers_v2.py` | Natural and Symbolic views of one checked proof certificate | Proof-first SFT traces |
| `spatial_audit_rendering_v2.py` | Coordinate-bearing witness reports | Benchmark audits only |
| `spatial_text_v2.py` | Current natural-language prompt to `SpatialProblem` adapter | Synthetic text round trips |
| `spatial_grading_v2.py` | Answer-mode resolution, menu encoding, and exact letter-set scoring | Dataset-specific evaluation |
| `spatial_generation_v2.py` | Policy-driven problem, menu, trace, and audit construction with mandatory round-trip validation | V2 synthetic workloads |
| `generate_all_v2.py` | Balanced JSONL workload CLI over V2 policy cells | SFT train/test generation |
| `spatial_workload_manifest_v2.py` | Distribution summaries, paired-variant checks, and split-leakage validation | V2 workload generation and audit |
| `spatial_matrix_v2.py` | Strict YAML matrix parsing and paired ablation expansion | Reproducible multi-cell experiments |
| `generate_matrix_v2.py` | Thin CLI for a V2 experiment matrix | Pilot and full dataset generation |
| `spatialeval_adapter_v2.py` | Original SpatialEval rows to structured cases and oracle audit status | SpatialEval audit pipeline |
| `audit_spatialeval_v2.py` | Deterministic full audit, witness validation, and SpatialMap-TQA-Corr derivation | Part I benchmark audit |
| `test_spatial_solver_v2.py` | Core, adapter, answer semantics, witness, and differential contracts | Local/CI verification |
| `test_audit_spatialeval_v2.py` | Source preservation, correction, determinism, and overwrite contracts | Part I audit verification |
| `test_spatial_grading_v2.py` | Single/complete/visible modes and exact letter-set scoring | Local/CI verification |
| `test_spatial_explanations_v2.py` | Claim assessment and Direction/Which/Count explanation contracts | Local/CI verification |
| `test_spatial_proofs_v2.py` | Proof construction, compact certificates, mutation rejection, and paired rendering | Local/CI verification |
| `test_spatial_generation_v2.py` | Generator, difficulty, menu, pairing, and CLI contracts | Local/CI verification |
| `test_spatial_workload_manifest_v2.py` | Manifest and leakage validation contracts | Local/CI verification |
| `test_spatial_matrix_v2.py` | YAML matrix parsing and exact expansion contracts | Local/CI verification |

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

    uv run --python 3.12 --no-project --with pytest --with typer --with pyyaml --with z3-solver \
      pytest spatial -q

The V2 core intentionally has no JSONL, prompt, option-letter, oracle, or
dataset-policy knowledge. New generators should construct `SpatialProblem`
directly, render text above that seam, parse the rendered text back through an
adapter, and verify the reparsed problem before emitting a row. They can pass
the same problem to `SpatialExplainerV2`, serialize its typed evidence, and use
`render_training_trace` with `TraceFormat.NATURAL` or `TraceFormat.SYMBOLIC`
and an independent `StateMode`. Coordinate witnesses belong only in structured
audit metadata or `render_audit_explanation`, never in an SFT reasoning target.

The proof-first seam is the replacement path for training explanations.
`build_direction_proof` currently accepts only exact positive-conjunction
Direction problems, constructs a typed certificate without consulting Z3, and
replays every premise, decomposition, inversion, transitivity, and recomposition
step through `check_direction_proof`. `render_direction_proof` produces Natural
or Symbolic text from that same certificate. It is not wired behind a fallback
to the post-hoc training renderer: Boolean branching, ambiguity certificates,
Which, and Count must be implemented before proof-first workload generation is
enabled for those shapes.

Audit the untouched SpatialMap-TQA release and derive SpatialMap-TQA-Corr with:

    uv --system-certs run --python 3.12 --no-project --with typer --with z3-solver \
      python spatial/audit_spatialeval_v2.py \
      --input data/spatialeval_org.jsonl \
      --output-dir results/part1-spatialeval-audit

The command writes the row-level audit, corrected 1,500-row dataset, and a
summary containing input/output hashes. Existing outputs are preserved unless
`--replace` is explicit. The corrected dataset adds `E. Cannot be determined`
but excludes coordinate witnesses; witnesses remain in the audit artifact.

Generate a balanced V2 workload with:

    uv run --python 3.12 --no-project --with typer --with z3-solver \
      python spatial/generate_all_v2.py \
      --out data/spatial_v2.jsonl \
      --samples-per-cell 100 \
      --query-kinds direction,which,count \
      --answer-modes single \
      --semantic-shapes unique,ambiguous \
      --query-directions north,northeast,east,southeast,south,southwest,west,northwest \
      --target-directions north,northeast,east,southeast,south,southwest,west,northwest \
      --trace-formats natural,symbolic \
      --state-modes delta

This writes `data/spatial_v2_train.jsonl`, `data/spatial_v2_test.jsonl`, and
`data/spatial_v2_manifest.json`. Natural/Symbolic variants of one base problem
stay in the same split. Coordinate-bearing audit metadata is excluded by
default. `--include-audit` is for diagnostic artifacts, not training data.
Programmatic callers pass one `WorkloadSpec` to `generate_workload` rather than
duplicating the CLI's individual settings.
The manifest includes generation acceptance rate and rejection reasons so
expensive policy cells are visible before scaling the workload.
The workload CLI also fails on existing outputs unless `--replace` is passed.

For a heterogeneous ablation, use the checked-in matrix example:

    uv run --python 3.12 --no-project --with typer --with pyyaml --with z3-solver \
      python spatial/generate_matrix_v2.py \
      experiments/spatial-v2-ablation.example.yaml \
      --out data/spatial_v2_ablation.jsonl

See `docs/spatial-v2-matrix.md` for the matrix contract and exact row-count
semantics. Matrix runs also create `*_views/by_variant` and `*_views/by_cell`
train/test pairs that can be passed directly to trainers and evaluators.
Existing matrix outputs are preserved unless `--replace` is explicitly passed.

For a transitive Direction workload, use a single compatible policy cell:

    uv run --python 3.12 --no-project --with typer --with z3-solver \
      python spatial/generate_all_v2.py \
      --out data/spatial_v2_depth.jsonl \
      --query-kinds direction \
      --semantic-shapes unique \
      --omit-direct-query-relation \
      --min-axis-depth 2 \
      --max-axis-depth 4 \
      --distractor-premises 5

Which and Count use `--min-membership-depth` and
`--max-membership-depth`. Exact ambiguity buckets use `--ambiguity-size` with
`--semantic-shapes ambiguous`. Incompatible cross-products fail instead of
silently weakening a requested constraint.

A practical depth-2 Which/Count pilot uses seven entities and ten premises;
that layout is constructed directly for all eight query directions rather than
found through expensive rejection sampling.
