# Historical Result Analysis

Dataset construction, release sealing, HITL review, and training/evaluation
preparation have been retired from this package. Their source is available in
Git at `a85c6c7`; reproduce historical workflows in a separate checkout of the
original run's revision. Do not rewrite saved receipts to match current code.

Retained utilities:

| Module | Purpose |
|---|---|
| `summarize_randomized.py` | Summarize saved randomized-v2 results |
| `analyze_randomized.py` | Audit and analyze saved responses |
| `summarize_corr.py` | Validate and summarize external Corr results |
| `summarize_transfer_diagnostic.py` | Validate paired transfer-diagnostic results |
| `metrics.py` | Paired comparisons and statistical corrections |
| `gen_fig_randomized.py` | Plot previously produced analysis tables |
| `training_token_audit.py` | Audit historical training-consumption records |

Historical datasets, results, review receipts, and prepared packages are
preserved. These utilities are not current composition experiment launchers.
Some require the exact frozen source/configuration assets of the original run.

Current workflow:
[Composition v1 Slurm instructions](../../../experiments/spatialentail-composition-v1/slurm/README.md).
Current authority:
[Unique-only specification](../../../experiments/spatialentail-composition-v1/UNIQUE-ONLY-v1.md).
