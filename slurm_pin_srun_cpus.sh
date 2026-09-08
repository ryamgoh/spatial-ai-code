# Source from every SLURM launcher BEFORE any srun.
# Canonical rule: experiments/README.md → "srun cpus-per-task".
#
# These two must NEVER differ (Slurm 23+ fatals if they do):
#   #SBATCH --cpus-per-task=N     →  this job's TRES  cpu=N
#   SLURM_CPUS_PER_TASK           →  must be that same N
#
# Do not hardcode a second N. sbatch --export=ALL copies leftover
# SLURM_CPUS_PER_TASK from the login shell (train jobs use 16; eval
# often uses 8). Copy cpu= from this job's TRES so they cannot drift.
#
#   srun: fatal: cpus-per-task set by two different environment variables
#   SLURM_CPUS_PER_TASK=16 != SLURM_TRES_PER_TASK=cpu=8

if [[ "${SLURM_SRUN_CPUS_PINNED:-}" == "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_ID:-}" ]]; then
  return 0
fi

_cpt=""
if [[ "${SLURM_TRES_PER_TASK:-}" =~ cpu=([0-9]+) ]]; then
  _cpt="${BASH_REMATCH[1]}"
elif [[ -n "${SLURM_JOB_CPUS_PER_NODE:-}" ]]; then
  _cpt="${SLURM_JOB_CPUS_PER_NODE%%(*}"
  _cpt="${_cpt%%,*}"
fi

if [[ -z "$_cpt" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "slurm_pin_srun_cpus.sh: WARN no cpu= in SLURM_TRES_PER_TASK=${SLURM_TRES_PER_TASK-unset}"
  fi
  unset SLURM_TRES_PER_TASK || true
  unset _cpt
  return 0
fi

export SLURM_CPUS_PER_TASK="$_cpt"
export SRUN_CPUS_PER_TASK="$_cpt"
unset SLURM_TRES_PER_TASK || true
export SLURM_SRUN_CPUS_PINNED="${SLURM_JOB_ID:-1}"
echo "srun cpus-per-task pinned to ${SLURM_CPUS_PER_TASK} (job TRES; login leftover ignored)"
unset _cpt
