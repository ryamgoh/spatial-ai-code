# CPU Execution and Solver Reuse

The original serial candidate is stopped. Its 554 completed records and original
receipts remain untouched. This implementation prepares a new execution version;
it does not submit a Slurm job, restart generation, seal data or authorize training.

## Entailment-Only Routing

The executor now selects its production implementation from the configuration
schema. `entailment-production-v1` with interface `spatialentail-entailment-v1`
uses entailment-only admission, coverage gates and exports; the historical
`composition-production-v1` configuration retains possibility-based semantics.
The launcher still invokes the same `composition_execution` module.

The [new configuration draft](entailment-config-draft-v1.json) is not executable:
it requires a frozen status and approved revised coverage before generation.
Do not change those markers merely to bypass review. Source/evidence supply
verification and external-adapter work remain outstanding.

The new path shares ordered workers, quota accounting, hash-pinned checkpoints,
exclusive output locking, progress, resume and early stopping. Checkpoint identity
includes production schema and semantic interface. Old checkpoints cannot be
resumed as entailment-only output.

Atomic counterexamples use deterministic axis-graph rank assignments, fixing
only required query relations with incremental SAT checks when necessary.
The initial full-coordinate optimization alternative was rejected after it
slowed the development workload. The graph path matches serial and one-/two-worker
records on tested fixtures without restoring all-pairs canonicalization.
See [the current migration checkpoint](ENTAILMENT-MIGRATION-STATUS-v2.md).

The proof/evidence details and timing receipts below describe the historical
possibility-based path unless explicitly labelled otherwise.

## What Changed

`composition_solver_reuse.QuestionSolver` caches successful analyses and claim
assessments within one question, with at most 256 entries. Mutable results are
copied. It never caches unknown/timeout errors and is cleared between admissions.
When a Selection analysis already supplies the canonical positive membership
map, certificate construction reuses that exact map. Missing counterexamples
still go through the original solver and canonical completion algorithm.

The production admission path checks rendering against its already constructed
outline instead of rebuilding the outline. Full evidence validation, independent
printed-text/scenario audits, coordinate verification, depth checks, semantic
transformations and actual-tokenizer admission remain enabled. The standalone
`check_pair` API still rebuilds an outline when checking externally supplied text.

`composition_execution` runs independent admissions in spawned CPU processes,
each with its own Z3 solver and tokenizer. It accepts 1-128 workers and defaults to
`SLURM_CPUS_PER_TASK` when set. It refuses more workers than the Slurm allocation.
Read-ahead is bounded to twice the worker count relative to the next ordered
admission, not just by the number of currently running tasks. No GPU or model
weights are needed.

Global scene/family preflight runs once. Canonical query deduplication precedes
dispatch. The existing generator applies quotas in its original fixed order,
not worker completion order. Source, depth and Count quotas are unchanged.
Completion timing must not decide which bases are retained. Per-check solver
timeouts still depend on hardware/load; version pinning does not guarantee that
timeout outcomes are identical across different machines.

## Early Stopping

The user approved treating the 31,251 scheduled attempts as a maximum reserve,
not mandatory work. `composition-execution-v2` stops at the first ordered prefix
whose retained split sizes exactly match the targets and whose full existing
summary has no shortfalls. This includes source, depth, Count-value and family
requirements; semantic, gold, contract-difference and direction coverage; and
no unresolved verification failures in the consumed prefix. Reaching 10,800
alone is insufficient. The full audit runs only when exact split sizes make
success possible, avoiding an expensive full report after every question.

The final summary is checked again before export. Initial reconstruction and
structural preflight still cover the entire fixed reserve. The stopping rule
saves solver admission work; it does not skip family-isolation preflight.

When success is reached, queued/running reserve work is cancelled. Already
delivered extra results are preserved as raw checkpoints, not included in the
retained dataset or its distribution report. Bounded parallel read-ahead means
a small amount of unnecessary work can occur. Manifest and summary distinguish
ordered attempts, unprocessed reserve slots, completed checkpoints, consumed
admissions and speculative completed admissions. Completion need not show
31,251/31,251; `stop_reason=requirements-satisfied` records success.

If the reserve is exhausted first, `stop_reason=attempt-budget-exhausted` is
recorded and any shortfalls still produce exit code 1. This change does not
replace first-accepted quota selection: if size quotas fill while coverage is
missing, later candidates cannot automatically replace retained questions.
The run must report that failure, not relax the coverage rules.

Resume reconstructs the same ordered prefix and stops at the same point,
without consuming speculative checkpoints beyond it. The execution schema
and source identity changed, so v1 checkpoints cannot be resumed under v2.
Existing outputs and the frozen reserve configuration remain untouched.

## Checkpoints and Progress

- Each finished raw admission is atomically saved under `checkpoints/`, with
  an integrity hash, exact task identity, elapsed time and worker peak RSS.
- Completed tasks can be reused only with matching source, configuration,
  exclusion, tokenizer, Python and recorded dependency identities.
- Worker count may change on resume. The fixed question plan may not.
- The ordered journal and quota counters are reconstructed from raw checkpoints;
  they are not copied from partially updated counters.
- A partial `.tmp` checkpoint is ignored; a corrupted completed checkpoint fails
  closed. A failed task is not replaced by a new scene.
- Concurrent coordinators targeting one output are blocked with an exclusive
  file lock. Use storage that supports advisory locking and atomic rename.
- Worker exceptions and detected worker death abort with checkpoints preserved.
  SIGTERM/SIGINT are handled; SIGKILL can leave the manifest stale, but completed
  checkpoints remain independently verifiable.
- `progress.json` and stderr report execution stage, completed admissions,
  reused checkpoints, throughput, elapsed time, worker count and ordered statuses.
  During a long initial structural stage, the last reported stage remains visible.
- Exports are staged and rebuilt after an interrupted export. Completion is
  distinct from passing coverage: shortfalls produce exit code 1 and an explicit
  `acceptance=shortfalls` manifest entry.

The old `composition_production` CLI remains available for compatibility and
still consumes its full fixed budget. Use the new executor/launcher for
early stopping and parallel, resumable runs. Old serial output is
not a compatible resume checkpoint and must not be mixed into the new run.

## Local Development Check

The benchmark compares the legacy serial path (uncached solver plus the original
outline-rebuilding check), optimized serial admission, and worker processes.
It uses only existing scaffold seeds 3 and 7, depths 2 and 8, all three query
types and paired menus. It requires exact full-record equality, including golds,
evidence, prose, depth metadata and token measurements.

```sh
python -m spatial.v2.composition_execution_benchmark \
  --config experiments/spatialentail-composition-v1/production-allocation-v1/execution-config-v1.json \
  --tokenizer /absolute/path/to/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --output /tmp/composition-cpu-check.json \
  --workers 1 2
```

The benchmark always times six unique baseline jobs. `--repeats` repeats the
same six jobs for parallel throughput measurements, without adding families
or changing menu seeds. Account for the differing job counts when comparing
totals. Every repeated admission clears its cache. These measurements are
engineering checks, not a new dataset pilot or a production distribution audit.

## Slurm Normal Partition

Submit from the repository root. The launcher defaults to 16 CPUs, one node,
one task, 2 GB per allocated CPU and a 24-hour limit. Adjust CPU, memory and
wall-time requests to the actual partition policy and measurements.

The launcher uses `uv --native-tls`, supported by older cluster installations
as well as the current local version (where it is a deprecated alias).
Job 1041432 failed before Python started because its installed `uv` rejected
the newer `--system-certs` spelling. The replacement was checked locally with
uv 0.8.22 and 0.12.17; this is not a completed server generation test.

If downloading the tokenizer, use the Python API to avoid differences in how
Hugging Face CLI versions parse multiple `--include` patterns:

```sh
TOKENIZER="$(uv --native-tls run --no-project --with huggingface-hub python -c \
  'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", allow_patterns=["*.json", "*.jinja", "*.txt"]))')" &&
export TOKENIZER
```

On login nodes with thread-creation limits, first export
`UV_CONCURRENT_INSTALLS=1`, `UV_CONCURRENT_BUILDS=1`,
`UV_CONCURRENT_DOWNLOADS=1`, `TOKIO_WORKER_THREADS=1`, and
`RAYON_NUM_THREADS=1`. These setup limits do not change the admission worker count.

Set these environment variables to server paths:

```sh
export CONFIG="$PWD/experiments/spatialentail-composition-v1/production-allocation-v1/execution-config-v1.json"
export TOKENIZER="/absolute/server/path/to/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
export OUTPUT="/absolute/server/path/to/cpu-check-16.json"
export MODE=benchmark
export REPEATS=64
sbatch --cpus-per-task=16 experiments/spatialentail-composition-v1/run-cpu.slurm.sh
```

Repeat the same development workload with 32, 64 and 128 allocated CPUs, giving
each report a fresh output path. `REPEATS=64` supplies 384 jobs so a 128-worker
measurement is not limited to six tasks. Read `lscpu` in the log to distinguish
physical cores from hardware threads. Compare throughput, peak RSS, queue wait
and timeout outcomes; do not assume a 128-fold speedup.

The launcher caps tokenizer and BLAS/OpenMP threading to one per worker and
uses the repository's Slurm CPU-pinning helper. It pins Z3, Transformers and
Tokenizers; the execution identity also records Python, Jinja2, PyYAML and NumPy.
For reliable resume, retain the same software environment.

After reviewing performance, a separately authorized new candidate can run:

```sh
export MODE=generate
export OUTPUT="/absolute/server/path/to/production-candidate-v2"
sbatch --cpus-per-task=64 experiments/spatialentail-composition-v1/run-cpu.slurm.sh
```

The 64-worker example is not a measured recommendation. Choose the worker count
from the server check. For an interrupted run under unchanged inputs, use the
same output and `export RESUME=1`. Missing jobs are retried with the same fixed
inputs, not replaced by new samples. Completed unsupported/timeout outcomes
remain recorded and are not automatically retried to improve acceptance.

No Slurm job has been submitted as part of this implementation.

## Verification

- Launcher compatibility: both generation and benchmark tests reproduce the
  unsupported-flag error before the fix and pass afterward (2 tests). Shell
  syntax validation also passes. No Slurm resubmission was performed locally.
- Early-stopping v2 execution and production regression selection: 120 passed
  in 108.55 seconds. Includes each individual audit gate blocking success,
  exact split-size checks, bounded out-of-order work, successful early stop with
  one/two workers and resume after an export interruption at the same prefix.
- The following results are from the preceding solver-reuse/execution version:
- Execution, reuse, existing production and readable-audit regression selection:
  278 passed in 82.62 seconds.
- Proof diversity, evidence and depth selection: 311 passed in 208.19 seconds.
- Final focused execution/reuse rerun, after adding ambiguous scenes to the
  worker fixtures: 18 passed in 67.61 seconds. These 18 overlap the first group.
- Checks include identical serial/parallel records, quota-order preservation,
  canonical deduplication, corrupted checkpoints, mismatched resume inputs,
  partial resume, interrupted exports, concurrent coordinators, worker exceptions,
  killed workers, mutable-cache isolation and non-caching of unknown results.
- Ruff and `bash -n` passed. No server-side Slurm test has been performed.

These are targeted regression selections, not a claim that the entire historical
repository test suite was rerun.

## Local Timing Results

The pre-early-stopping [measurement receipt](CPU-PERFORMANCE-v1.json) records six existing
development queries, exact full-record agreement and pinned input identities.
No other test jobs were running during this comparison.

| Execution | Wall Time for Six Questions |
|---|---:|
| Legacy serial, uncached solver and outline rebuild | 29.66 s |
| Reuse, serial in-process | 20.66 s |
| Reuse, one worker process with checkpoints | 23.02 s |
| Reuse, two worker processes with checkpoints | 14.88 s |

Reuse reduced serial time by about 30%. Two workers delivered about 2.0x
overall speedup relative to the legacy path. Process timings include worker
startup and checkpoint writes; the in-process measurements do not. Maximum
observed worker peak RSS was 306,790,400 bytes (about 293 MiB).

This is a small local workload, not a production runtime forecast. It does not
measure Slurm queue delay, network filesystem overhead, the full scene
distribution, or scaling to 16-128 CPUs. The canonical witness algorithm still
does substantive work for distinct obligations; this change removes redundant
work, not all solver cost. The next check is the same fixed repeated development
workload on the normal partition, before launching a new candidate.
