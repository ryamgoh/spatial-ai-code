#!/usr/bin/env bash
#SBATCH --job-name=composition-cpu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=composition-cpu-%j.log

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${TOKENIZER:?Set TOKENIZER to the pinned tokenizer snapshot on the server}"
: "${CONFIG:?Set CONFIG to the frozen execution configuration}"
: "${OUTPUT:?Set OUTPUT to a fresh candidate directory or benchmark JSON file}"
cd "$SLURM_SUBMIT_DIR"
source slurm/lib/pin-srun-cpus.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

workers=${WORKERS:-${SLURM_CPUS_PER_TASK}}
if (( workers < 1 || workers > SLURM_CPUS_PER_TASK || workers > 128 )); then
    echo "WORKERS must be between 1 and min(128, allocated CPUs)" >&2
    exit 2
fi

module=spatial.v2.composition_execution
args=(--config "$CONFIG" --tokenizer "$TOKENIZER" --output "$OUTPUT")
case "${MODE:-generate}" in
    generate)
        args+=(--workers "$workers")
        if [[ "${RESUME:-0}" == 1 ]]; then
            args+=(--resume)
        fi
        ;;
    benchmark)
        module=spatial.v2.composition_execution_benchmark
        args+=(--workers "$workers" --repeats "${REPEATS:-4}")
        ;;
    *)
        echo "MODE must be generate or benchmark" >&2
        exit 2
        ;;
esac

echo "CPU workers=$workers allocated=$SLURM_CPUS_PER_TASK mode=${MODE:-generate}"
lscpu || true
exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    uv --system-certs run --python 3.12 --no-project \
    --with 'z3-solver==4.16.0.0' --with 'transformers==5.16.1' \
    --with 'tokenizers==0.23.3' --with jinja2 --with pyyaml \
    python -u -m "$module" "${args[@]}"
