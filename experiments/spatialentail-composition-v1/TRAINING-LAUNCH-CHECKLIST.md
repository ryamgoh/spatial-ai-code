# Training Launch Checklist

Use this checklist before describing a new or amended training run as ready.
This is a process requirement, not an implemented automatic submission gate.
Do not copy readiness from a historical package, another experiment, or passing
unit tests. Record the actual prepared run path and evidence for each item.

## Run Record

- Commit and checkout:
- Prepared package and manifest SHA-256:
- Model and tokenizer revisions:
- Natural/Symbolic configuration hashes:
- Fixed development inventory and prompt version:
- Generation and training job IDs:
- W&B run IDs and URLs:
- Exceptions explicitly accepted by the user:

## Before Submission

- [ ] Read the active study specification and compare with the actual configs,
      not just README claims.
- [ ] Pin the model, tokenizer, template and dependency versions.
- [ ] Check distinct question counts, exact answer allocation, evidence,
      proof-depth requirements and fixed scene/query/family exclusions.
- [ ] Verify both arms share question IDs, menus, labels, evidence and order;
      compare training settings allowing only approved style/logging differences.
- [ ] Inspect actual serialized examples with the current native template.
      Verify explanation/answer placement, prompt masks, supervised terminators,
      padding masks and causal-label handling.
- [ ] Measure the current training and dev data; reject silent truncation,
      dropped rows, changed splits or automatic quota refills.
- [ ] Attach the fixed dev set for epoch dev-loss reporting when required.
      Explicitly state whether dev evaluation is enabled in the actual config.
- [ ] Have the correct-model generated-answer dev evaluator and checkpoint
      selection procedure ready, or disclose their absence before launch.
      Dev loss must not replace the agreed Stage-1 accuracy selection rule.
- [ ] Confirm both epoch checkpoints are retained and all evaluation stages
      use the same selected adapter.
- [ ] Keep external/test labels out of training and checkpoint selection.
- [ ] Confirm W&B online settings, separate arm IDs, credential loading and
      email directives against randomized-v2. Never put credentials in configs.
- [ ] Verify actual Slurm partitions/resources, fresh output paths and both
      `afterok` dependencies. Missing data or CPU failure must prevent training.
- [ ] Freeze the checkout/environment while jobs run. Do not pull into it.
      A config-only update requires a versioned compatible preparation, not
      disabling hash verification.
- [ ] Run relevant regression tests and record their scope. Distinguish local
      token tests, full data audits, mocked scheduling and actual GPU execution.
- [ ] State one readiness verdict listing unresolved work and accepted
      exceptions; do not introduce known omissions after submission.

## Opening Training Steps

These checks can occur inside the actual job; no separate smoke-test job is
required by this checklist.

- [ ] Inspect the actual Axolotl-loaded row count and batch labels against the
      prepared data, including padding; reject retokenization/retemplating.
- [ ] Verify correct model/revision, adapter targets, quantization, device count
      and memory usage.
- [ ] Observe finite loss and gradient norms through optimizer updates.
      Do not mistake `logging_nan_inf_filter: false` for an automatic abort.
- [ ] Record W&B connectivity/run URL, Slurm logs, actual software/CUDA/GPU
      versions, processed examples/steps and memory.

## Before Results

- [ ] Verify both epoch checkpoint files and load the saved adapters.
- [ ] Evaluate both epochs on fixed synthetic dev under the current template.
      Select by Unique Stage-1 dev view accuracy; earlier epoch wins ties.
- [ ] Evaluate style-matched untuned baselines and selected adapters with the
      same declared decoding budget; record length exhaustion and format errors.
- [ ] Verify Stage-2 replay retains the intended Stage-1 token context and
      uses the selected adapter; report both stages without using Stage 2 to select.
- [ ] Report per-query, depth and E/F slices; keep omission companions grouped
      as correlated views and disclose the one-seed limitation.
- [ ] Preserve run manifests, logs, checkpoint hashes and evaluation outputs.
      Distinguish configuration intent from observed server execution.
