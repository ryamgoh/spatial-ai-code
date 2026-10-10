# Experiment 15: Spatial V2 proof-trace pilot (Part II)

This is a preparation and integration pilot for proof-trace SFT. It contains no
model results. The checked-in run trains four supervision arms on paired base
problems: checked Natural, checked Symbolic, answer-only, and corrupted Symbolic.
Three untuned baselines evaluate the same Natural, Symbolic, and answer-only
prompts without adapters. Corrupted Symbolic preserves parseable syntax and the
final gold answer while changing checked reasoning; there is no Natural
corruption arm.

The complete generator and matrix field catalogue is
[`docs/spatialentail/generation-and-knobs.md`](../../docs/spatialentail/generation-and-knobs.md).

`matrix.yaml` requests 40 unique Direction problems at depth 2 and 40 ambiguous
Direction problems with two possibilities. Both use three entities and two
premises. The exact requested strata must all generate and pass admission;
generation fails instead of dropping difficult cells and writes a `_rejected.json`
diagnostic with the failing rows, token counts, and reasons. Which and Count remain
supported by the generator and the same admission contract, but are outside
this bounded pilot.

`split_strategy: structural` keeps canonical visible-formula/query signature
clusters together across train, development, and final test. Requested dev/test
fractions are 0.2 each; cluster sizes can change realized fractions. Development
loss selects the checkpoint. Final test is used only by the evaluation jobs.
This split is a structural-overlap defense, not evidence of broad compositional
generalization.

`matrix-premise-holdout.yaml` separately declares atomic training, independently
sampled premise-first IFF development, and premise-first implication test cells.
These cells are selected before generation and use strict cell holdouts plus
structural-overlap rejection. Its Boolean shape names specify sampling syntax;
they do not promise that every accepted example needs the named inference rule.
It is an additional source/operator diagnostic, not a reported benchmark result.

All variants of each base problem pass admission together using the actual
`Qwen/Qwen3.5-4B` tokenizer at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and its default chat template. Admission counts the
full training conversation against 4096 tokens and the exact evaluation prompt
plus a 4096-token generation allowance against 8192 tokens. Targets must also
fit the generation allowance. No truncation or format-specific filtering is
used. The manifest records actual model-token volumes by variant and split;
fixed examples/epochs are not equal-token or equal-compute training.

The evaluation task consumes the admitted, fully templated prompt verbatim,
retains the raw completion, and scores answer accuracy plus Symbolic reasoning
replay, decision validity, domain consistency, and full validity. Natural and
answer-only process validity are unavailable; conditional process aggregates
exclude them. Deterministic decoding uses temperature 0 and repetition penalty
1. Retention evaluation is deferred because its separately authored task needs
its own prompt/template contract.

## Local checks

From the repository root, preview the graph without creating run artifacts or
calling Slurm:

```bash
uv run --no-project --with typer --with pyyaml --with z3-solver \
  python experiments/15-v2-ablation/submit.py --dry-run
```

Generate and admit the pilot on CPU (downloads tokenizer files only):

```bash
uv run --no-project --with typer --with pyyaml --with z3-solver \
  --with transformers --with jinja2 \
  python -m spatial.v2.generate_matrix experiments/15-v2-ablation/matrix.yaml \
  --out /tmp/spatial-v2-pilot/pilot.jsonl
```

Use `matrix-premise-holdout.yaml` instead to prepare the independent-source
holdout diagnostic. The generator writes `_train.jsonl`, `_dev.jsonl`,
`_test.jsonl`, `_manifest.json`, and paired view directories. View names are
`<answer>__<trace>__<supervision-arm>_<split>.jsonl`, for example
`single__symbolic__checked-trace_test.jsonl`. Answer-only is emitted once using
the first configured trace format as its view namespace.

```bash
uv run --no-project --with pytest --with typer --with pyyaml --with z3-solver \
  python -m pytest spatial/v2/tests/test_matrix.py \
  experiments/15-v2-ablation/test_submit.py experiments/tasks/test_utils_v2.py -q
```

## Slurm execution

On the Slurm login server, the same submit command without `--dry-run` renders
immutable run configs and submits generation, one train/eval chain for each SFT
arm, three generation-dependent untuned evaluation jobs, and a summary job.
The summary waits `afterany`; dependent training/evaluation jobs require
`afterok`. No GPU training or evaluation is required for the local checks.

`--only single-natural` selects one arm. `--skip-generation` requires all
selected train/dev/test views and a validated manifest. `--replace-data`
explicitly replaces generated output. `--resume --run-id <existing-id>` reuses
saved configs and preserves submission history. A scheduler submission failure
preserves captured job IDs and prints the cancellation command.

`run.yaml` controls model, arms, output paths, overrides, and Slurm resources.
An untuned arm has `train: false`. Dataset paths, adapter paths, output paths,
and evaluation tasks are computed by the submitter; model/context/template
settings must agree with matrix admission. The generated data and evaluation
prompts must be regenerated after changing the model or tokenizer template.
