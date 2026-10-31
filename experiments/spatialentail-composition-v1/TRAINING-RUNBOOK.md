# SpatialEntail Training Preparation

## Server Generation and Training

From the server repository root, after pulling the current launcher code:

```bash
bash experiments/spatialentail-composition-v1/slurm/submit.sh
```

This submits CPU generation/auditing/preparation on `normal`, then independent
Natural and Symbolic jobs on `gpu-long` with one `h100-47` GPU each. Both use
`afterok` on the CPU job. No dataset transfer is needed. See
[the three-job workflow](slurm/README.md) for outputs and failure handling.
This server path creates new run-local candidates and prepared packages.
The historical local `prepared-thinking-v1` retains its original source hashes;
it is not the package used by this launcher after the preparation code extension.

## Status and Scope

**Active model: Qwen/Qwen3-4B-Thinking-2507 (BF16, thinking-only)**, selected in
[`active-model.json`](active-model.json), not Qwen3.5-4B. Future matched training,
baseline and evaluation runs must use its pinned model/tokenizer revision.
No `enable_thinking` override is needed; the native template opens the thinking
block. The earlier non-thinking setting is withdrawn.
The [current prepared package](prepared-thinking-v1/README.md) now contains
matched configs and 6,000 pre-tokenized examples per arm. All 12,000 examples
passed native thinking-format, answer, token-boundary and assistant-loss checks.
Maximum full training sequences are 1,766 Natural / 1,930 Symbolic tokens;
maximum supervised targets are 1,105 / 1,264. No source data was regenerated.
The existing QLoRA recipe uses four-bit loading from the BF16 checkpoint and
BF16 compute; both arms retain the same two-epoch schedule and seeded order.

The user approved checking GPU operation in the opening steps of the actual
training job, not a separate smoke-test job. Actual Axolotl loading and CUDA
execution remain untested locally. The fixed development evaluation package
and Stage-2 thinking replay still need integration; no evaluation selection
was changed. Current reference lengths fit, but untuned generation length is
not certified by those measurements.

Token counts in the legacy sections below use Qwen3.5-4B and must not be
relabelled as Qwen3-4B-Thinking-2507 results.
The historical model settings and commands later in this runbook are not the
new study's trainer configuration.

The active study is now [Unique-only](UNIQUE-ONLY-v1.md) for Direction,
Which, and Count, with matched Natural/Symbolic arms. Its approved training
target is 6,000 distinct questions and 6,000 examples per style, with one menu
per question and 1,200 A-D / 600 E / 200 F per query.
The all-controlled replacement is tracked by `CONTROLLED-TRAINING-BUILD-v2.json`
and `CONTROLLED-TRAINING-EXPORT-AUDIT-v2.json`. Its current reference revision is
`unique-training-candidate-v3/`, tracked by `SPATIAL-PROSE-RERENDER-v3.json`.
Use it only when the build/export audits and reference-only audit report success.
V3 preserves the v2 questions and evidence; only explanation text/version and
token measurements change.
Its chat datasets are `natural-training.jsonl` and `symbolic-training.jsonl`.
None of the legacy commands below launches this new study.

**Legacy commands below.** This runbook describes the prior sealed 200-story
package, not the all-controlled replacement. Do not run its training commands
for the amended experiment. Follow [the active specification](UNIQUE-ONLY-v1.md).
An audited chat-data export is not a GPU smoke test, trainer configuration,
sealed release or authorization to submit a training job.

The prior package's `prepared-training-v2/seal.json` binds its original
preparation. It does not establish sufficient depth or answer coverage for
training at 1-8 and extrapolation at 9-10. **No training, GPU inference,
scheduler submission, commit, or push was performed.**

Preparation v2 is a scoring-code cleanup of v1, not a new dataset or experiment.
Each answer is parsed once and transition flags are shared across summaries.
Data, token measurements and audit selection are byte-identical; config changes
only relocate paths to v2. The original 88-pair audit is carried forward after
hash verification, not presented as a new independent review.

The original `prepared-training-v1/`, archive and receipt are unchanged.
Its source hashes intentionally do not match the cleaned working tree: reproduce
v1 using its archived source in a separate directory. Use v2 with current source;
never bypass the source-hash check. The v2 archive receipt records the comparison.

This seals synthetic data and local preparation, not the empirical study.
GPU compatibility and actual Axolotl preprocessing remain launch gates.
Separately versioned external SpatialMap contract views still require their
own preparation before final evaluation. Do not pass the frozen legacy
five-option Corr SINGLE data to this six-option evaluator, and never use
SpatialMap for training or checkpoint selection.

## Dataset

A premise set means one map description. Its three questions and all
contract/menu/style versions stay in one split.

| Split | Premise Sets | Families | Questions | Examples Per Style |
|---|---:|---:|---:|---:|
| Train | 120 | 114 | 360 | 1,596 |
| Development | 39 | 37 | 117 | 522 |
| Test | 41 | 37 | 123 | 534 |
| Total | 200 | 188 | 600 | 2,652 |

The fixed budget contains 100 randomly sampled maps, 50 independently sampled
atomic-relation sets, and 50 targeted examples. All 600 attempted questions
were admitted; no retries or post-result expansion. There are 192 canonical
story groups, conservatively grouped into 188 families. All 33
query/contract/semantic-case cells occur in every split.

There are 1,284 full-coverage and 1,368 partial-coverage views. Gold E occurs
758 times and F 780 times; this is not a balanced-label benchmark.
Supporting x/y paths, where applicable, have 1-4 links. Of 600 questions,
510 have zero designated verified distractors, 60 have one, and 30 have two.
Zero designated distractors does not mean every premise is necessary.

The unstratified `study-candidate-v1` is a retained, failed-coverage candidate,
not the release. `study-candidate-v2` is the source for the prepared package.
The initial `generator-pilot-v1` remains historical feasibility evidence.
Do not regenerate or overwrite any of them to hide provenance differences.

## Artifacts

- `prepared-training-v2/manifest.json`: model, seed, limits, source/environment,
  tokenizer and data hashes, fixed policy, file inventory.
- `prepared-training-v2/seal.json`: binds that manifest and completed audit.
- `prepared-training-v2/audit-review.json`: 88 agent-reviewed pairs, per-item
  notes and hashes, all 785 distinct paired lines and full sequence mappings.
- `prepared-training-v2/data/*-train-tokenized.jsonl`: exact assistant-only
  masks; these are the actual Axolotl datasets.
- `prepared-training-v2/data/*-train-messages.jsonl`: readable training views.
- `prepared-training-v2/data/*-{dev,test}-inputs.jsonl`: no labels/references.
- `prepared-training-v2/references.jsonl`: protected scoring/reference bank.
- `prepared-training-v2/configs/{natural,symbolic}.yaml`: matched training jobs.
- `prepared-training-v2/token-summary.json`: pinned-tokenizer distributions.

The audit is by the implementing agent, not an independent human reviewer.
Every released row passed programmatic evidence checks and paired alignment.
The independent reference search covered 102 small Direction questions only;
it shares domain/sign definitions. No claim is made about arbitrary
model-generated explanation validity.

## Frozen Settings

Qwen/Qwen3.5-4B, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, seed 42. Two epochs per arm,
one GPU, QLoRA rank 64 / alpha 128, microbatch one, accumulation eight,
learning rate 5e-5, paged AdamW 8-bit, cosine schedule, 5% warmup, bf16.
Keep the generated configs intact; they include the existing Qwen projection
targets and clipping setting. No packing or length-based grouping.

Both epoch checkpoints are saved. Choose each arm using **only Stage-1 dev
accuracy**, equally macro-averaged over the three contracts; earliest epoch
wins ties. Test results, Stage 2 and training loss cannot select a checkpoint.
Use that same selected adapter for both test stages.

Stage 1 is greedy with a 4,096-new-token cap. Stage 2 runs on every example,
replays every Stage-1 token, then requests a constrained answer with a
16-new-token cap. It handles an already-generated turn terminator without
duplicating it. Overflow raises an error; context is never truncated.
`enable_thinking=False` matches the pinned template's empty thinking prefix;
reasoning is generated in the supervised assistant content.

Context/training capacity is 8,192 tokens. Maximum full reference sequence:
4,184 Natural / 3,737 Symbolic. Maximum reference target: 3,618 / 3,172.
Maximum reserved Stage-2 context: 4,892 / 4,891.

Training target-text tokens per epoch: 898,788 Natural / 777,108 Symbolic.
Including the supervised terminator/newline: 901,980 / 780,300.
These are reference lengths, not generated-token, runtime or accuracy results.
The 2-percentage-point descriptive accuracy-loss tolerance applies separately
to both stages; one seed and this sample size cannot establish equivalence.

## GPU Preflight

Run commands from the repository root on the chosen GPU host. Transfer the
repository source, lockfile, candidate v2 and prepared package unchanged.
No machine, partition, account or memory allocation has been assumed.

1. Install the existing locked Linux/CUDA environment with `uv sync --frozen`.
   Do not silently upgrade Axolotl, vLLM, Transformers, or the tokenizer.
   Record GPU model, driver, CUDA, package versions and available memory.
2. Verify the seal and all frozen source/environment/data hashes:

```bash
P=experiments/spatialentail-composition-v1/prepared-training-v2
uv run --frozen python -c \
  'from spatial.v2.composition_evaluation import verify_seal; import sys; print(verify_seal(sys.argv[1]))' "$P"
```

3. Run `uv run --frozen axolotl preprocess "$P/configs/natural.yaml" --debug`
   and the corresponding Symbolic command. Inspect the actual loaded rows:
   1,596 each, identical example order, prompt labels all `-100`, and every
   assistant target token including its terminator supervised. Reject any
   truncation, retemplating, row dropping or retokenization.
4. Confirm the pinned Qwen checkpoint loads, bf16/4-bit support is available,
   all configured LoRA targets exist, and Qwen linear-attention LoRA is supported
   by the installed training and evaluation runtimes. Fail closed on unsupported
   modules; do not remove target modules from one arm.
5. Run a three-example dev-only inference smoke check for each style:

```bash
mkdir -p "$P/results"
uv run --frozen python -m spatial.v2.composition_evaluation run \
  --prepared "$P" --style natural --split dev --smoke \
  --output "$P/results/natural-base-smoke.json"
uv run --frozen python -m spatial.v2.composition_evaluation run \
  --prepared "$P" --style symbolic --split dev --smoke \
  --output "$P/results/symbolic-base-smoke.json"
```

The runner verifies tokenizer file hashes and records generated IDs,
finish reasons and input lengths. Inspect both stages and structured decoding.
Smoke outputs are marked `split=smoke` and cannot select checkpoints.
Record successful preflight evidence before authorizing training.

## Training Commands

**Commands below start training. They have not been executed.**

```bash
uv run --frozen python finetune/finetune.py "$P/configs/natural.yaml"
uv run --frozen python finetune/finetune.py "$P/configs/symbolic.yaml"
```

Run with one visible GPU per process. Preserve logs and both epoch checkpoints.
Record actual elapsed time, GPU memory and processed tokens for each arm.
Do not train both arms concurrently on one GPU. A scheduler wrapper should be
added only after the target cluster's resource/account settings are known.

## Checkpoint Selection and Test

Find each epoch's actual `checkpoint-*` directory from `trainer_state.json`;
do not guess step numbers. Evaluate both epochs on the full dev set:

```bash
uv run --frozen python -m spatial.v2.composition_evaluation run \
  --prepared "$P" --style natural --split dev --epoch 1 \
  --adapter PATH_TO_NATURAL_EPOCH_1 \
  --output "$P/results/natural-dev-epoch-1.json"
```

Repeat for epoch 2 and both Symbolic epochs, using the matching style.
Then select separately for each style:

```bash
uv run --frozen python -m spatial.v2.composition_evaluation select \
  --prepared "$P" --style natural \
  "$P/results/natural-dev-epoch-1.json" \
  "$P/results/natural-dev-epoch-2.json"
```

Selection records dev result hashes, model/manifest provenance, adapter content
hash and epoch. It refuses overwrite. The test runner refuses an adapter that
does not match that selection.

Use `run --split test` without `--smoke` for each selected adapter and for
the untuned base under both styles (omit `--adapter` for untuned). Four
checkpoint/instruction combinations, not four training arms. Output paths
must be new; partial runs are retained and cannot be silently overwritten.

Report both stages, format compliance, gold E/F versus substantive performance,
semantic/query/coverage slices, truncation, answer changes, rescues/regressions,
generated tokens and replayed input tokens. Paired comparisons must retain base
and family grouping; views are not independent observations. Reference-token
savings must not be reported as model-generated savings.
