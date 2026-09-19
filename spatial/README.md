# Spatial reasoning

The solver and synthetic data generators live together here because they share
one domain model and one gold-label contract.

- `spatial_solver.py`: source of truth for v6 answers
- `generate_all.py`: frozen legacy SFT generator
- `generate_all_v6.py`: solver-validated SFT generator
- `generate_grpo.py`: prompt-only GRPO data generator
- `test_spatial_laws.py`: solver/generator contract tests

The training launchers still work from `finetune/` and call these scripts
through `../spatial/`. Axolotl commands and config path semantics are
unchanged.

Run the domain tests from the repository root:

    uv run --python 3.12 --no-project --with pytest --with typer \
      pytest spatial/test_spatial_laws.py -q
