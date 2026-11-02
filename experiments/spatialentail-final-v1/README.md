# SpatialEntail Final v1

**Status: CPU generation workflow implemented. Full server generation and
training-release validation are still pending. No SFT jobs are submitted here.**

A clean restart of the composition study: matched Natural and Symbolic SFT
on Qwen3-4B-Thinking-2507, using the Unique-entailed answer contract.

The [SFT plan](SFT-PLAN.md) is the single source of truth for data allocations,
training, evaluation, validation requirements and remaining work.
[generation_policy.py](generation_policy.py) is the executable allocation used
by the thin [generate.py](generate.py) launcher, not a second manually maintained
status document. Shared generation, export and verification live in
[`spatial/v2/entailment_generation.py`](../../spatial/v2/entailment_generation.py);
independent evidence audits live in
[`entailment_artifact_audit.py`](../../spatial/v2/entailment_artifact_audit.py).
The policy supplies model, slots and source paths explicitly. Shared code does
not import experiment scripts or modify Python's import path to find old audits.

| Dataset | Questions | Purpose |
|---|---:|---|
| Train | 6,000 | SFT weight updates |
| Dev | 1,200 | Validation during SFT |
| SFT-Bench in-range | 2,400 | Internal test, levels 1-8 |
| SFT-Bench extrapolation | 1,200 | Internal test, levels 9-10 |
| SpatialMap-TQA-Corr | 1,500 | External evaluation |

Synthetic splits share 60% A-D / 30% E / 10% F per query and matched E
subtypes, with one menu per question. All use the same controlled generation
framework with disjoint questions, scenes and families. Train/dev/in-range
use uniform levels 1-8; extrapolation uses equal levels 9 and 10. Balance
correct A/B/C/D positions where feasible. External data is not rebalanced.

Synthetic Count options now use `synthetic-count-options-0-4-v1`: four distinct
values from 0-4 for A-D, E and F. This removes the known cue from distractors
above the true-answer range. F still omits the sole entailed answer. The solver
continues to check the complete Count domain over every nonreference object.
Historical generation defaults and external menus are unchanged. The allocation
and scene seeds are unchanged, but use fresh run paths for these revised menus.

Reuse shared infrastructure, not old adapters, caches or readiness claims.
Historical artifacts stay untouched. The agreed scope needs no repeated manual
approval, but new data and the actual training pipeline must pass validation.
Do not use the old composition submit script for this experiment.

## Run On Slurm

Use a fixed checkout of the whole repository on your server, with `uv` and
`sbatch` available. Shared solver code and tracked correction sources remain
repository dependencies; copying only this folder is insufficient.

From the repository root, submit generation for all five datasets:

```bash
bash experiments/spatialentail-final-v1/slurm/submit-generation.sh full
```

This submits **one CPU job** on `normal`: 64 CPUs, 128 GB RAM, 24 hours, with
64 generation workers by default (following `SLURM_CPUS_PER_TASK`).
It requests no GPU, installs only the small CPU
generation environment, and never calls the old training submitter.
CPU dependencies are declared once in `generate.py`'s PEP 723 script metadata.
`uv run --no-project --script` uses a separate cached script environment,
preserving Z3 4.16.0.0 without resolving against the GPU training stack.
There is no separate requirements file and no root dependency change.
The existing cluster email address is retained in `slurm/generate.sh`.
Adjust its resource directives for your cluster if needed.

To test the workflow on a smaller set first:

```bash
bash experiments/spatialentail-final-v1/slurm/submit-generation.sh smoke
```

Smoke covers all four synthetic splits at their level endpoints, every answer
case, Count values 0-4, contingent Which cases, and three external questions.
It still excludes all 1,500 external questions during synthetic construction.
Smoke output is explicitly marked and is never a full dataset.

Each submission prints its job ID and artifact path. Monitor it with
`squeue -j JOB_ID` and the `generation-JOB_ID.out` / `.err` files in the printed
run directory. Slurm email delivery itself has not been verified.
`FINAL_GENERATION_WORKERS` can override the worker count within the CPU allocation.
Synthetic admission uses these workers; preflight, external preparation and
export readback remain serial. Peak memory and throughput at 64 workers have
not yet been measured on the server.

The job needs network access for Python dependencies and two pinned tokenizer
snapshots; it does not download model weights. Package/Hugging Face caches stay
inside the submission directory. The historical Qwen3.5 tokenizer is used only
for the inherited admission check; final train/dev tokens use the pinned Qwen3
thinking tokenizer.

## Run Contents

Everything generated for a submission is kept under this experiment:

```text
runs/generation-full-<id>/
  submission.txt
  generation-<job>.out
  generation-<job>.err
  cache/
  artifacts/
    generation.json
    manifest.json
    inputs/                 # policy, slots, raw source and correction receipt
    tokenizers/             # actual pinned active/admission tokenizer files
    audit/                  # attempts, quotas, export checks, review samples
    data/
      train/
      dev/
      in-range-test/
      extrapolation-test/
      external-test/
```

Every split has `references.jsonl`, `evidence.jsonl`, `token-measurements.jsonl`
and gold-free `natural-inputs.jsonl` / `symbolic-inputs.jsonl`.
Train/dev additionally have native thinking `*-chat.jsonl`, pretokenized
`*-tokenized.jsonl` (`input_ids`, `attention_mask`, `labels`), and `index.jsonl`
binding the shared row order to each style's token hash.
Later SFT/evaluation jobs should consume this explicit artifact path and record
their outputs under the same submission, without regenerating the data.
Large run outputs are git-ignored; preserve or back them up on the server.

`audit/review-samples.jsonl` includes ten paired examples per query/case when
available, plus split/query/level coverage, contingent cases and Count values
per split/level, and longest native examples per split/style.
`audit/review-selection.json` records the IDs and selection reasons. It is marked
`selected-not-reviewed`; it does not certify human inspection or replace the
additional pilot/failure-fixture review.

SpatialMap-TQA-Corr is read from the tracked correction source and verified
against its original source/receipt. Source names, premises and A-D menu order
are preserved, with the shared Unique question/system rules applied. Unique
labels are recomputed; differences from historical source labels are recorded
in `audit/external.json` and reference metadata, never used to rebalance data.
The source export's old `evaluation_prompt` wrapper is not used.

## Verification And Failure Handling

Use the printed artifact path:

```bash
uv --system-certs run --python 3.12 --no-project \
  --script experiments/spatialentail-final-v1/generate.py verify \
  --run-dir /absolute/path/to/artifacts
```

The build performs canonical query/scene/family exclusions, evidence and
independent SMT replay, exact slot/menu/answer checks, active-model token
measurements and exported-file readback. It hashes payloads, sources and
tokenizers and writes the success manifest only after those checks pass.
Runtime package versions are recorded in the manifest.

New manifests record `source_inventory_version: script-metadata-generation-v3`.
The launcher fingerprint includes the inline dependency pins; installed versions
are recorded in the environment receipt. Transitive dependencies are not fully
locked by this script metadata, just as with the previous requirements file.
Verify older packages under their original checkout; never rewrite old manifests
or disable fingerprint checks to make them pass. The historical coverage-policy
JSON remains an explicit input, not an import of the old experiment's workflow.
Generation retains Z3 4.16.0.0; the root training lock remains unchanged.

A shortage after three declared constructions, failed admission, invalid
external source or failed audit exits nonzero. `generation.json` and partial
audit/data files remain for diagnosis; there is no successful manifest.
Do not overwrite failed runs, auto-refill slots, or pull code into a running
checkout. Submit a fresh run after a diagnosed fix. Hard kills can leave status
`running`, but without a valid completion manifest they cannot pass verification.

**A successful full build is an audited candidate, not a training release.**
Ambiguity and inconsistency require checked evidence and coverage across levels,
not increasing difficulty; their `verified_depth` stays null. Semantic-cohort
records use `semantic_complexity_status: verified-semantics-no-depth-claim`.
Positive support length is still checked, including semantic-cohort sole Which.
Shortcut probes, paired-example review and actual trainer/GPU checks remain explicit
`pending_release_checks`; `training_ready` remains false.
Adding `--require-training-ready` to verification rejects these candidates.
`afterok` on the CPU job alone is not a sufficient training gate.
Final-v1 SFT configs, launchers and in-training dev integration are the next
step, not implemented by this generation-only submission.

## Local Tests

In a Python 3.12 environment containing the dependency pins from `generate.py`
and pytest:

```bash
python -m pytest experiments/spatialentail-final-v1/tests -q
```

Tokenizer-dependent tests require both pinned snapshots in the Hugging Face
cache. Unit tests and local smoke runs do not certify Slurm execution, full
dataset supply or GPU behavior.

The final-v1 tests exercise 601 representative scheduled questions through real
construction, solver admission, independent evidence replay, both explanation
styles and the pinned tokenizers. They cover every split/construction-level/
answer-case cell, Count 0-4 for A-D/F, and Which's depth, semantic and contingent
positive cases, plus all eight directions in each split/query group.
Assertions check all-object scope, menu-to-answer mapping,
omission-only F, E subtypes, selected support length and token budgets.
Semantic-cohort positives must retain their intended positive support length
even though their full `verified_depth` remains null. Negative tests substitute
a level-1 construction into a level-8 semantic Which slot and require rejection,
both with and without contingent alternatives.

This tests that the requested kinds of questions can be generated correctly;
it is not a full 10,800-question supply test. Ambiguous and inconsistent cases
are tested for their semantics at every level, not claimed to have progressively
harder ambiguity or contradiction evidence.

Local verification on 2026-11-02:

- Revised Count menus and review selection: 775 tests passed in the final
  combined regression run; a fresh 176+3 smoke build passed. All 80 non-Count
  synthetic references and all five external data files matched the previous
  smoke artifacts; all 96 Count rows retained their labels with revised menus.
- Menu-only replay over all 2,000 planned train Count menu seeds, reusing checked
  semantics rather than admitting a full corpus: the previously observed
  first-small-option rule now scores 15% (always A); the earlier unique-small
  rule scores 30% (always E), equal to the majority baseline. All five possible
  unordered menus occur in every Count answer case. Other shortcut probes and
  full-corpus checks remain required.
- 26 final-v1 tests passed, plus 35 shared native-thinking/evaluation tests.
- End-to-end smoke: 176 synthetic questions (44 per split) and 3 external
  questions passed, including multiprocessing, export readback and manifest
  verification. Temporary smoke artifacts were kept outside the repository.
- All 1,500 external questions separately passed source-menu preservation,
  independent Unique label verification and current-tokenizer prompt budgets.
  There are 212 label differences from the historical correction export;
  this is a contract projection, not a claim that the old source was already
  labeled under the current Unique rule.
- No Slurm or GPU job was run locally. Full synthetic solver admission remains
  the first server campaign; these tests do not certify a training-ready release.
- After extracting shared modules, the 176+3 smoke build passed again. All
  35 data files were byte-identical to the earlier smoke output, and the
  emitted policy was unchanged. Independent audit function bodies were also
  checked for AST equivalence against their pre-refactor implementations.
