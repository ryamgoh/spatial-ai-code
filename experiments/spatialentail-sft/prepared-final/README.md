# Prepared SFT jobs

Six jobs compare answer-only, Natural, and Symbolic on the same base problems,
with seeds 42 and 43. These files have not been submitted. Transfer the whole
repository plus the release/job artifacts to the Slurm server. From its repo
root, submit the six scripts individually using `sbatch <script-path>`.
The default resource request is one H200 on gpu for three hours. Confirm
that request against the server's limits before submission. Authenticate once
with `wandb login` or export `WANDB_API_KEY` before submission. Training and
system metrics are logged to the `spatialentail-sft` project; stable run IDs and
`WANDB_RESUME=allow` keep resumed allocations in the same run. Model checkpoints
are not uploaded to W&B. Training uses the existing finetune environment and
fixed final-epoch checkpoints. Bench data is absent from the Axolotl
configurations. The data/config hashes are in jobs.json.
