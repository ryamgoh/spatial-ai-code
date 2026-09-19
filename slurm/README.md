# Slurm support

The local development machine does not have Slurm. These scripts are validated
locally and submitted only after the repository is available on the Slurm
server. A local `sbatch: command not found` error is therefore expected.

Submit experiment launchers directly from the repository root with:

    sbatch experiments/<experiment>/slurm/<launcher>.sh

Each experiment tracks its own `logs/.gitkeep`, and its launchers write to
`experiments/<experiment>/logs/%x-%j.{out,err}`. Historical
multi-experiment launchers under `slurm/archive/` write to
`slurm/archive/logs/`. All jobs assume the repository root is
`SLURM_SUBMIT_DIR`; this is also required because config paths are resolved
from `eval/` or `finetune/`.

- lib/pin-srun-cpus.sh keeps srun CPU settings consistent with the job
  allocation. Every launcher that invokes srun must reach this helper.
- archive/ contains historical launchers that mixed unrelated experiments.
  They are retained for provenance, not as recommended entrypoints.

The cluster's gpu partition is limited to three hours. Longer jobs belong on
gpu-long.
