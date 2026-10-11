# SpatialEntail study release and SFT job preparation

Run these commands from the repository root. This workflow does not submit
Slurm jobs or train a model locally.

1. Build a new candidate release:

   ```sh
   python -m spatial.v2.study.build spatial/v2/study/release-candidate.yaml --out results/spatialentail-study/release-candidate-v2
   ```

2. Refresh shared Natural evidence into a new candidate, then audit all role
   quotas, hashes, paired targets, and overlap:

   ```sh
   python -m spatial.v2.study.refresh results/spatialentail-study/release-candidate-v2 --out results/spatialentail-study/release-final
   python -m spatial.v2.study.seal results/spatialentail-study/release-final
   ```

3. Prepare immutable role/cell-stratified review packets:

   ```sh
   python -m spatial.v2.study.review_packets results/spatialentail-study/release-final --out results/spatialentail-study/review-final
   ```

   Run DeepSeek-V4-Pro then GPT-5.5 sequentially in isolated contexts over each
   frozen packet, using the rubric in `../hitl/quality-judge-v1.md`. Preserve
   payload hashes, model identifiers, sequence positions, and receipt files
   listed in the review index. Human appeals remain restricted to double
   rejection; split decisions are held out for replacement. Unresolved appeals
   or held-out source rows prevent the release seal. This is two independent
   blinded LLM review passes, not a human double-blind study.

   If a later release changes only unreviewed strata, `study.reuse_reviews`
   may rebind prior decisions only after proving every reviewed item is
   identical. It bundles the original packet and receipts as provenance.

4. Seal only after the actual quality receipts pass:

   ```sh
   python -m spatial.v2.study.seal results/spatialentail-study/release-final --review-index results/spatialentail-study/review-final/index.json
   ```

5. Create the six paired SFT configurations and Slurm scripts:

   ```sh
   python -m spatial.v2.study.jobs results/spatialentail-study/release-final --out experiments/spatialentail-sft/prepared-final
   ```

   The output contains three supervision packages times seeds 42/43, paired
   Train/Dev views, config/script hashes, and server instructions. Generation
   does not call `sbatch`. Transfer the repository and ignored result/job
   artifacts together. Scripts assume the Slurm submission directory is the
   repository root and reuse the existing `finetune` environment. Their default
   request is one H200 on `gpu` for three hours; verify that against the server.
   The scripts follow the Experiment 14 convention and email BEGIN, END, FAIL,
   TIME_LIMIT, and TIME_LIMIT_90 events to `e0958116@u.nus.edu`; stdout and
   stderr are written under `prepared-final/logs/`.
   Put `export WANDB_API_KEY=...` in the server-local, mode-600 file
   `~/.config/spatialentail/wandb.env`. The shared launcher loads it for every
   job, while still accepting an already-exported key or a custom path through
   `WANDB_ENV_FILE`. The generated configurations use project
   `spatialentail-sft`, deterministic per-arm/seed run IDs, and resumable online
   logging. They do not upload model checkpoints to W&B.

The candidate build may fail on diversity or token admission. It never replaces
existing output, silently relaxes uniqueness, backfills a different depth, or
truncates a paired trace. Resolve the failed stratum and version the configuration
before retrying. Candidate artifacts are not training-ready merely because they
exist. Final job preparation requires a sealed release and its audit receipt.

The executed release used 485 reviewed Natural examples. Both models approved
all 485, so the predefined human-appeal path was not triggered. Do not describe
this as a completed human review or as evidence of population-level human
acceptability.

For an eligible double-rejection item, the active user-facing review records
Approve/Reject using the existing append-only `HumanAppeal` schema. Add the
batch's optional `appeals` JSONL filename to the review index. Sealing validates
eligibility and payload binding, preserves that record inside `release/quality`,
and allows approved false-positive rejections to proceed. Rejected appeals or
split-verdict source rows must be replaced and screened in a new version.

Completed strata are stored in `<output>-work` with config, registry,
contamination, and source hashes. They are not a completed release. A retry
with identical inputs resumes them after admission/overlap checks; changed
inputs refuse the old checkpoints and require a new output version.

The seal audit also re-solves persisted prompts and replays persisted Symbolic
targets. Generation-time validity flags and updated checksums cannot substitute
for that audit. Human-language screening remains a separate sampled gate.

If the language-quality screen identifies repeated Count witness prose,
`study.refresh` writes a new unsealed candidate with shared witness references,
re-admits changed paired groups, and rebuilds nested views without altering gold
answers or Symbolic evidence. The old candidate and judge packets remain intact;
changed payloads require fresh sequential judgments.
