# Matched Transfer Diagnostic

## Status

Prepared locally, solver-checked, and not submitted. There are no model results
for this diagnostic yet. This is a post-hoc investigation motivated by Corr,
not a preregistered benchmark or a change to the original SFT experiment.

The runnable package is `../prepared-transfer-diagnostic/`. Commit and push
its source code, data, tasks, configs, receipt, and documentation before using
Git to deploy it. No archive handoff is needed for these compact inputs.
Do not commit model weights, runtime logs, or raw generated responses.

## Questions

1. On identical ambiguous problems, does an adapter succeed when asked to
   enumerate possibilities but fail when asked to select "Cannot be determined"?
2. Does changing only premise layout alter accuracy on those same problems?
3. Do errors persist across both answer policies and layouts, which would
   motivate a deeper reasoning or grounding audit?

## Controlled Design

There are 60 fresh bases: 10 each in Direction, Selection, and Count crossed
with unique and ambiguous answers. Ambiguous bases have exactly two possible
substantive values. All bases use eight entities and nine atomic premises.
The generator uses fresh seeds, world-first construction, and the Z3 solver.

Each base has four views:

| View | Answer policy | Premise layout |
|---|---|---|
| single / paragraph | One letter; E for ambiguity | Paragraph |
| single / lines | One letter; E for ambiguity | One premise per line |
| all-possible / paragraph | Every possible substantive answer | Paragraph |
| all-possible / lines | Every possible substantive answer | One premise per line |

Within a base, the entity names, ordered premises, query and candidate scope,
ordinary option values and positions are identical. A-D contain the complete
possible answer set plus impossible distractors; E is always "Cannot be
determined." Unique cases therefore have the same gold letter across policies.
Ambiguous cases require E under SINGLE and two ordinary letters under
ALL-POSSIBLE. Only instructions change between policies; only whitespace
changes between layouts.

All four text views were parsed back into the identical structured problem
and menu, and their labels were replayed with the solver. Offline audit fields
and gold labels are not included in the model-visible prompt.

### Fixed Decoder

All views use the same set-capable Stage-2 decoder over A-E, including SINGLE
views. A multi-letter answer to a SINGLE question is scored wrong rather than
silently truncated. This holds the decoder constant while varying instructions.

This differs intentionally from the frozen Corr SINGLE-only decoder.
Diagnostic scores must not replace or be directly merged with Corr scores.
Stage 1 is retained as a diagnostic; Stage 2 is primary within this new suite.
All models retain the pinned Qwen3.5-4B revision, 4,096-token thinking budget,
8,192-token context, temperature zero, and original adapter paths.

### Exclusion and Scope

Canonical structure signatures were checked against the audited training and
development files, the 3,600 internal benchmark bases, and all 1,500 corrected
Corr questions: 20,500 distinct excluded signatures. Accepted overlap is zero.
Candidate archive hashes match the prior provenance audit; this does not prove
actual cluster training consumption or absence from model pretraining.

The longest prompt is 269 tokens with the pinned tokenizer, comfortably inside
the context reserve. There are 240 views per checkpoint and seven checkpoint
jobs: 1,680 total requested responses, representing only 60 distinct base
problems, not 1,680 independent research observations.

This initial format probe tests paragraph versus line layout only. It does not
test shop-name vocabulary, paraphrasing, Boolean versus atomic reasoning,
question-type rewrites, or varied candidate scopes. The fresh problems are
simpler than many Corr examples; success here would not resolve Corr failures.

## Reading Results

Report both seeds separately, all query/status slices, and paired wins/losses.
Avoid inferring training-seed uncertainty from only two seeds.

| Pattern on matched examples | Supported interpretation |
|---|---|
| ALL-POSSIBLE correct, SINGLE wrong on ambiguous rows | Policy-sensitive failure consistent with weak abstention transfer; not proof of intact reasoning in every run |
| Lines outperform paragraph under a fixed policy | Layout-sensitive behavior on this probe |
| Both policies fail under both layouts | Policy wording/layout alone did not resolve errors; audit reasoning and menu grounding |
| Unique controls also change with policy wording | Broader instruction sensitivity, not only ambiguity recognition |
| All four views succeed but Corr remains weak | This simple probe does not reproduce the external difficulty; broaden the next diagnostic deliberately |

The summarizer reports policy differences as SINGLE minus ALL-POSSIBLE and
layout differences as lines minus paragraph. It validates receipts, source
documents, targets, prompts, runtime model arguments, paired ID coverage,
original scorer replay, and aggregate scores. E rates and multi-letter rates
help identify output-policy violations. Small per-cell counts support
descriptive diagnosis, not a new leaderboard or definitive causal claim.

Do not tune prompts or retrain on these same results and then call them a
held-out evaluation. New adaptations need a separate validation/test split.

## Cluster Handoff

After the new files have been committed, pushed, and pulled on the cluster,
run from the cluster repository root. First submit only the untuned smoke
of the complete 240-view diagnostic:

```bash
mkdir -p experiments/spatialentail-randomized-v2/prepared-transfer-diagnostic/logs
sbatch experiments/spatialentail-randomized-v2/prepared-transfer-diagnostic/slurm/untuned.sh
```

Check its logs and `results/untuned/COMPLETED`. A completion marker alone does
not establish full sample coverage; the importer checks that later. Once
baseline execution is successful, submit the six adapters:

```bash
for arm in single-natural single-answer-only multi-natural; do for seed in 42 43; do sbatch "experiments/spatialentail-randomized-v2/prepared-transfer-diagnostic/slurm/${arm}-seed-${seed}.sh"; done; done
```

These jobs evaluate existing weights. They do not retrain or modify adapters.
Do not bypass the input/environment hash checks or resubmit running jobs.

After all seven jobs finish:

```bash
tar -czf ~/transfer-diagnostic-results.tar.gz -C experiments/spatialentail-randomized-v2/prepared-transfer-diagnostic jobs.json results/ logs/
```

Download from the local terminal:

```bash
scp xlogin:~/transfer-diagnostic-results.tar.gz ~/Downloads/
```

Summarize locally from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --offline --python 3.12 --no-project \
  --with pyyaml python -m spatial.v2.study.summarize_transfer_diagnostic \
  --archive "$HOME/Downloads/transfer-diagnostic-results.tar.gz" \
  --prepared experiments/spatialentail-randomized-v2/prepared-transfer-diagnostic \
  --output experiments/spatialentail-randomized-v2/transfer-diagnostic-summary
```

## Reproduction

The preparation program refuses an existing output directory. Preserve prior
packages and generate into a fresh repository subdirectory for replay.
Config/script paths and receipt hashes will differ with the output path;
the data and offline audit should match exactly.

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --offline --python 3.12 --no-project \
  --with z3-solver==5.3.2.0 --with tokenizers==0.23.3 --with pyyaml \
  python -m spatial.v2.study.prepare_transfer_diagnostic \
  --experiment experiments/spatialentail-randomized-v2 \
  --output experiments/spatialentail-randomized-v2/prepared-transfer-diagnostic \
  --training-archive "$HOME/Downloads/prepared-training-candidate.tar.gz" \
  --benchmark-archive "$HOME/Downloads/prepared-benchmark.tar.gz" \
  --tokenizer "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer.json"
```

The receipt includes source hashes, data/config hashes, source-archive member
hashes, the tokenizer hash, Z3 version, generation seed, and exclusion counts.
Existing Corr data, raw scores, and training files remain unchanged.
