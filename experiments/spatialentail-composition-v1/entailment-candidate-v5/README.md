# Synthetic Evaluation Source

Retained nontraining source only: 1,200 dev, 2,400 in-range test and 1,200
extrapolation bases. Only Unique rows and their token records remain.
Original problems, evidence, Unique labels/menus/references and retained model
input lines are unchanged. Old training data and Listed/Exhaustive views were
removed.

This is still the older scoped source, not a ready all-scene evaluation release.
Its IDs do not completely match `../unique-only-selection-v1/selection.json`.
Keep that fixed inventory; reconciliation and current-model evaluation
materialization remain unfinished. Do not use these source prompts directly
as the active study's evaluation inputs.

`manifest.json` records retained payload hashes and the original artifact
fingerprints. `attempts.jsonl` retains the evidence needed for materialization;
`references.jsonl` and split/style inputs contain 5,767 Unique views per style.
