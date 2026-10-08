# Spatial reasoning

Active SpatialEntail code lives in `v2/`. Retired pre-V2 implementations are
kept under `v1/` only for historical reproducibility.

| File | Responsibility | Typical caller |
|---|---|---|
| `v2/solver.py` | Data-agnostic formulas, Direction/Which/Count queries, Z3/reference engines, witnesses | Any structured spatial workload |
| `v2/proofs.py` | Typed proof and refutation certificates with replay checking | Proof-first SpatialEntail generation |
| `v2/model_certificates.py` | Constructive model, countermodel, and contingency validation | Possibility and non-entailment evidence |
| `v2/answer_certificates.py` | Complete Direction candidate coverage | Direction answer sets |
| `v2/which_certificates.py` | Entailed, contingent, or impossible membership evidence | Which answer sets |
| `v2/count_certificates.py` | Correlation-preserving evidence over complete membership assignments | Count answer sets |
| `v2/certificate_generation.py` | Convert solver witnesses into independently replayed answer certificates | Proof-first generator |
| `v2/*_renderers.py` | Natural and Symbolic views of checked evidence | Proof-first traces |
| `v2/text.py` | Controlled prompt grammar to `SpatialProblem` | Round-trip validation |
| `v2/grading.py` | Answer-mode resolution, menu encoding, and scoring | Dataset adapters |
| `v2/generation.py` | Policy-driven, certificate-backed workload construction | V2 synthetic workloads |
| `v2/generate_all.py` | Balanced workload CLI | SFT train/test generation |
| `v2/matrix.py` and `v2/generate_matrix.py` | Strict ablation matrices and runner | Reproducible experiments |
| `v2/spatialeval_adapter.py` and `v2/audit_spatialeval.py` | SpatialEval audit and correction | Part I benchmark audit |
| `v2/tests/` | V2 semantic, certificate, generation, and audit contracts | Local/CI verification |

Retired implementations live under `v1/`:

- `v1/generate_all.py`: frozen original SFT generator
- `v1/generate_all_v6.py`: solver-validated V6 generator
- `v1/generation_v13.py`: V13 typed generator
- `v1/generate_grpo.py`: historical prompt-only GRPO generator
- `v1/tests/`: retained regression tests

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
dataset-policy knowledge. New generators construct `SpatialProblem` directly,
render text above that seam, parse the rendered text back through an adapter,
and verify the reparsed problem before emitting a row. Accepted rows carry a
checked answer certificate and render it as either `TraceFormat.NATURAL` or
`TraceFormat.SYMBOLIC`. Coordinate witnesses belong only in structured audit
metadata or `render_audit_explanation`, never in an SFT reasoning target.

The proof-first seam is the training-explanation path.
`build_direction_proof` currently accepts only exact positive-conjunction
Direction problems, constructs a typed certificate without consulting Z3, and
replays every premise, decomposition, inversion, transitivity, and recomposition
step through `check_direction_proof`. `render_direction_proof` produces Natural
or Symbolic text from that same certificate. The certificate checker also
supports conjunction introduction/elimination, modus ponens, disjunctive
syllogism, biconditional elimination, double negation, and explicit
contradiction. Scoped assumptions, contradiction closure, explosion, and
complete case splits support branched proofs while rejecting cross-branch
dependencies. Boolean-derived atoms feed the same axis rules as direct spatial
premises. Unsupported proof construction rejects the sample; there is no
post-hoc training-renderer fallback. `SpatialModelCertificate` independently checks constructive witnesses
and countermodels against the complete premise formula; a
`ContingencyCertificate` requires both sides for the same claim. These
certificates establish possibility and non-entailment, not completeness of an
answer set. `DirectionAnswerSetCertificate` adds completeness by requiring every
declared candidate, in canonical order, to carry exactly one checked model or
checked refutation. `WhichAnswerSetCertificate` classifies every declared entity
as entailed, contingent, or impossible. Entailment excludes every non-matching
direction when no shorter exact-direction proof is available; impossibility
excludes every matching direction; and contingency requires both a supporting
model and a countermodel. `CountAnswerSetCertificate` preserves correlation by
covering every value from zero through the candidate count. Possible values use
checked models; impossible values require checked refutations for every complete
membership assignment with that count. The generic Count checker is complete
relative to supplied evidence, while automatic refutation construction remains
limited by the proof builder's supported fragment.

`v2/generation.py` rejects any candidate for which
`build_answer_certificate` cannot produce complete checked evidence. Accepted
training traces are rendered only from that certificate; there is no post-hoc
explanation fallback or independent state-snapshot mode.

Audit the untouched SpatialMap-TQA release and derive SpatialMap-TQA-Corr with:

    uv --system-certs run --python 3.12 --no-project --with typer --with z3-solver \
      python -m spatial.v2.audit_spatialeval \
      --input data/spatialeval_org.jsonl \
      --output-dir results/part1-spatialeval-audit

The command writes the row-level audit, corrected 1,500-row dataset, and a
summary containing input/output hashes. Existing outputs are preserved unless
`--replace` is explicit. The corrected dataset adds `E. Cannot be determined`
but excludes coordinate witnesses; witnesses remain in the audit artifact.

Generate a balanced V2 workload with:

    uv run --python 3.12 --no-project --with typer --with z3-solver \
      python -m spatial.v2.generate_all \
      --out data/spatial_v2.jsonl \
      --samples-per-cell 100 \
      --query-kinds direction,which,count \
      --answer-modes single \
      --semantic-shapes unique,ambiguous \
      --query-directions north,northeast,east,southeast,south,southwest,west,northwest \
      --target-directions north,northeast,east,southeast,south,southwest,west,northwest \
      --trace-formats natural,symbolic

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
      python -m spatial.v2.generate_matrix \
      experiments/spatial-v2-ablation.example.yaml \
      --out data/spatial_v2_ablation.jsonl

See `docs/spatial-v2-matrix.md` for the matrix contract and exact row-count
semantics. Matrix runs also create `*_views/by_variant` and `*_views/by_cell`
train/test pairs that can be passed directly to trainers and evaluators.
Existing matrix outputs are preserved unless `--replace` is explicitly passed.

For a transitive Direction workload, use a single compatible policy cell:

    uv run --python 3.12 --no-project --with typer --with z3-solver \
      python -m spatial.v2.generate_all \
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
