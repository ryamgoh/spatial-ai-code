# SpatialEntail Training Readiness Report

Date: 2026-11-01.
Reviewed commit: `bd25fb823b2aa98e9fbcde1199a50f34064abca8`.
Method: local source/specification inspection, targeted regression execution,
and primary-source research. No Slurm access, GPU execution, W&B upload, email
delivery or server checkpoint inspection was performed.
Cleanup note: superseded local prepared packages were subsequently deleted.
Their measurements below are historical observations, not instructions to
reuse a currently available prepared directory.

Historical status: findings and source line references below describe the reviewed
commit, not current readiness. The thinking-model evaluator and post-training dev
selection now live in `spatial/v2/entailment_evaluation.py`. The obsolete
`composition_evaluation.run_gpu` and its selection/CLI entry points have been
removed; shared scoring remains. This update does not supply GPU verification
for the original report's other gaps.

## Verdict

**The data and local supervision preparation have substantial verification.
The complete training-and-evaluation experiment is not yet integrated.**

The current scripts can attempt matched two-epoch QLoRA training, conditional
on the real loader and GPU stack behaving correctly. They do not implement
epoch dev evaluation, current-model generated-answer checkpoint selection,
or the new thinking-model Stage-2 evaluator. Passing local tests is not evidence
that those missing pieces exist.

There is no evidence here that the training data is ruined or that a server
run has produced invalid weights. Missing dev loss does not invalidate
gradient updates. Saved epoch checkpoints can be evaluated afterward.
However, the agreed study conclusions cannot be produced until the correct
dev/baseline/test evaluation and selection path is completed. [L1-L4, S3]

I previously described partial preparation as readiness and did not carry
forward expected monitoring and development-evaluation behavior. Those were
workflow and communication failures. W&B/email configuration is now restored;
dev integration is still not implemented. Future launch reviews must inspect
the effective configs and evaluator together, not infer readiness from another
experiment's conventions.

## Findings, In Priority Order

### 1. Development evaluation and checkpoint selection are missing

**Confirmed, high priority for completing the study.**
`training_config` sets `eval_strategy: no`, provides no `test_datasets`,
and sets `load_best_model_at_end: false`. Epoch checkpoints are configured to
be retained, but the three-job workflow does not evaluate or select them.
The study explicitly requires selection by Unique Stage-1 synthetic dev
view accuracy, with earliest epoch breaking ties. [L1, L2, L3]

Randomized-v2 attaches development data and evaluates each epoch; composition
did not carry that reporting behavior forward. Adding that configuration
would supply held-out teacher-forced loss, not automatically generate task
answers or implement our selection rule. [L5, S3]

Required correction: materialize the existing fixed dev IDs under the approved
all-scene/short-question protocol, measure them with the pinned thinking
template, attach them for epoch dev-loss diagnostics, and implement prompt-only
generated Stage-1 dev scoring for both saved epochs. Preserve the fixed
selection; do not substitute a random split or select on test/Stage-2/training loss.
Existing trained checkpoints can use post-training generated dev evaluation.

### 2. Historical GPU evaluation is not the active-model evaluator

**Confirmed, high priority before reporting results.**
`composition_evaluation.run_gpu` requires the legacy seal/package schema and
uses model constants imported from the historical composition trainer.
Those constants remain Qwen3.5-4B. Its prompt and Stage-2 suffix are historical
non-thinking behavior. It is not a drop-in evaluator for the current run-local
Qwen3-4B-Thinking-2507 packages. [L4]

Required correction: connect evaluation to the actual prepared model revision,
tokenizer, run manifest and selected adapter. Preserve Stage-1 thinking tokens
according to the intended Stage-2 protocol, handle malformed/unfinished thinking
blocks explicitly, and verify constrained A-F output. Run matched untuned
baselines with the same declared budget. Do not silently reuse legacy commands.

The model card recommends longer thinking budgets and non-greedy sampling.
These are not mandatory training settings. Our 4,096-token/greedy evaluation
choice must be declared and checked for exhaustion; a supervised target fitting
that budget does not show that the untuned model will finish within it. [S1]

### 3. Real trainer ingestion and checkpoint reload are unverified

**Verification gap, not a demonstrated runtime failure.**
Local preparation validates native serialization and explicit labels, but
the trainer calls `load_datasets` and `train` without a composition-specific
post-loader assertion of paired membership or actual batch masks. The launch
scripts verify file hashes, not the materialized trainer dataset or padded
batch. They also do not validate epoch adapter completeness/reloadability
after training. [L1, L6, L7]

The pinned Axolotl wrapper recognizes a Dataset with the three tokenized
columns and bypasses prompt reconstruction. Other loader steps can still
load cached data, handle long rows or deduplicate. That is why actual loaded
IDs/counts/masks must be observed, not inferred from `train_on_inputs: false`.
There is no evidence from this report that these transformations dropped rows
in our runs. [S2]

Required correction: inspect actual loaded counts, paired retained IDs/sidecar,
token labels and padding before optimizer steps. Check every configured LoRA
target and correct base revision, then load the saved base/adapter/tokenizer
combination before evaluation. Full resume testing is required only if full
resume is promised; checkpoint existence alone is insufficient. [S4]

### 4. Numerical failure handling is observational, not automatic

**Confirmed operational limitation.**
`logging_nan_inf_filter: false` makes bad losses visible. It does not implement
an automatic finite-loss/gradient abort. The launcher reports GPU allocation
and starts training; the documentation asks an operator to watch opening steps.
We must not describe this as a proven fail-fast numerical safety mechanism.
[L1, L7, S3]

Required correction: define and test finite-loss/gradient checks and their exit
policy, and record early optimizer progress. This can happen inside the
training job. No separate smoke-test job is necessary.

### 5. Checkout changes invalidate active generation/preparation provenance

**Confirmed compatibility limitation; usually an operational failure, not silent
data corruption.**
Generation fingerprints all Python modules in `spatial/v2` and all experiment
Python scripts. Training preparation binds those fingerprints and its own
source/environment hashes. Verification rejects changed inputs.
Consequently, pulling a monitoring/preparation code update into an active
checkout can fail generation or later verification, even without changed
spatial questions. No monitoring-only migration currently exists. [L1, L8]

Required procedure: use a fixed checkout/environment for each running cohort.
Never recommend pulling into a running checkout. Preserve old manifests and
payloads. A migration must be versioned, verify its historical inputs, explicitly
allow only intended changes, and retain the old receipt. Do not disable checks
or overwrite historical manifests to make a run pass.

### 6. Monitoring is configured, but delivery is not certified

**Configuration restored; runtime evidence pending.**
New server-generated configs enable W&B online in
`spatialentail-composition-v1-sft`, with separate arm/submission IDs.
The shared randomized-v2 credential loader requires an environment key or
private credential file. Slurm email directives are present for the CPU and
both GPU jobs. Historical generated packages still have their old settings.
[L1, L3, L7]

The credential loader proves local credential availability, not valid remote
authentication. `afterok` proves successful predecessor exit status, not valid
metrics or checkpoints. Observe the run URL, actual logs and artifacts.
Email depends on the cluster mail service; no delivery was tested here. [S5]

### 7. Matching controls are reasonable, but evidence is limited to one seed

**Interpretation limitation, not a reason to change the approved experiment.**
Both arms share question order, model revision, two epochs, batch/accumulation,
optimizer, precision, clipping and scheduler. They deliberately differ in
language/style and supervised token counts. Equal examples/epochs is our
matching rule; it is not equal tokens or equal compute. [L1, L9]

Report actual examples, supervised tokens, updates and elapsed time.
Do not change quotas or add seeds without approval. One shared seed does not
support a strong estimate of seed variance or bitwise reproducibility across
machines/software. PyTorch explicitly warns about environment-bounded
reproducibility. [S6]

## What Is Verified

| Component | Evidence | Limit |
|---|---|---|
| Local v3 data | Published 6,000-question build/reference audit receipts | Does not audit an unseen server run |
| Native thinking labels | Prompt prefix masked; reasoning, closing tag, answer and EOS supervised | Actual Axolotl/padding batch still needs verification |
| Reference lengths | Historical Qwen3 local preparation: full maxima 1,766 Natural / 1,930 Symbolic; supervised maxima 1,105 / 1,264 | Not untuned generation lengths; old package source hashes are historical |
| Matched trainer recipe | Pinned model/revision; two-epoch QLoRA; common seed/order and hyperparameters | Not executed optimizer trajectories |
| Fresh Slurm output and dependencies | Mocked submission tests; separate GPU arms wait on CPU `afterok` | Not scheduler or GPU execution |
| Monitoring | Config/directive and mocked credential tests | Not actual W&B/email delivery |

I reran the current targeted suite: **38 passed in 16.09 seconds**:

```bash
uv --system-certs run --python 3.12 --no-project \
  --with transformers==5.16.1 --with tokenizers==0.23.1 \
  --with jinja2==3.1.6 --with pyyaml==6.0.3 \
  --with z3-solver==4.16.0.0 --with scipy==1.18.1 --with pytest \
  python -m pytest spatial/v2/tests/test_thinking_slurm.py \
  spatial/v2/tests/test_entailment_thinking_training.py \
  spatial/v2/tests/test_server_generation.py -q
```

The suite covers answer/format rejection, native template/mask mutations,
overflow, config matching, monitoring, mocked dependency failures and a selection
of fresh generated cases including depths 1 and 8. It is not a full 6,000-record
new server admission, a GPU run, a checkpoint reload, or an evaluation run.

## Future Launch Standard

Use [TRAINING-LAUNCH-CHECKLIST.md](TRAINING-LAUNCH-CHECKLIST.md).
It is now linked from the training runbook. It is a process gate, not an
implemented submission blocker.

Before saying "ready," issue one explicit verdict covering data, effective
trainer config, development data/evaluator, checkpoint policy, monitoring,
runtime checks and provenance. List known exceptions before submission.
The current three-job scripts should be described as training-only, not as a
complete experiment. A user-approved training-only run remains possible; its
unfinished evaluator must not be hidden or discovered as a surprise afterward.

For an already running job, preserve its checkout and epoch artifacts. This
report alone is not a reason to cancel training or repeat generation. Reconstruct
the actual commit/config/job state before recommending any change.

## Sources

Local references refer to the reviewed working tree at the commit above:

- **L1:** [Native thinking preparation](../../spatial/v2/entailment_thinking_training.py), especially lines 100-161, 174-255, 310-363 and 442-512.
- **L2:** [Active study specification](UNIQUE-ONLY-v1.md), Training And Evaluation section.
- **L3:** [Slurm workflow and monitoring](slurm/README.md), [submission](slurm/submit.sh), and both training wrappers.
- **L4:** [Historical GPU evaluator](../../spatial/v2/composition_evaluation.py), lines 303-418; [historical model/template constants](../../spatial/v2/composition_training.py).
- **L5:** [Randomized-v2 job generator](../../spatial/v2/study/randomized_jobs.py), `_config` and `prepare_jobs`.
- **L6:** [Trainer entry point](../../finetune/finetune.py), lines 168-220.
- **L7:** [Training common launcher](slurm/train-common.bash) and [credential loader](../15-v2-ablation/jobs/common.bash).
- **L8:** [Portable generation](generate_and_prepare.py), lines 61-152.
- **L9:** Historical `prepared-thinking-v1` token summary and preparation manifest
  inspected for this report; the superseded package was removed during cleanup.

Primary external sources:

- **S1:** [Pinned Qwen thinking model card](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507/blob/768f209d9ea81521153ed38c47d515654e938aea/README.md) and [pinned native template](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507/blob/768f209d9ea81521153ed38c47d515654e938aea/tokenizer_config.json).
- **S2:** [Axolotl pretokenized docs](https://docs.axolotl.ai/docs/dataset-formats/tokenized.html); pinned [wrapper](https://github.com/axolotl-ai-cloud/axolotl/blob/7d775808d0ff71f21f553bd7d761631d7fb089e6/src/axolotl/utils/data/wrappers.py) and [SFT loader](https://github.com/axolotl-ai-cloud/axolotl/blob/7d775808d0ff71f21f553bd7d761631d7fb089e6/src/axolotl/utils/data/sft.py).
- **S3:** [Transformers Trainer API](https://huggingface.co/docs/transformers/main/en/main_classes/trainer), evaluation and `logging_nan_inf_filter` contracts.
- **S4:** [Transformers checkpoint arguments and resume API](https://huggingface.co/docs/transformers/main/en/main_classes/trainer).
- **S5:** [Slurm official dependency/email documentation](https://slurm.schedmd.com/sbatch.html).
- **S6:** [PyTorch official reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

See [TRAINING-PRIMARY-SOURCES.md](TRAINING-PRIMARY-SOURCES.md) for detailed
source-version distinctions. Current online docs are not proof of behavior in
the server's installed versions; the Axolotl source cited here is pinned.
