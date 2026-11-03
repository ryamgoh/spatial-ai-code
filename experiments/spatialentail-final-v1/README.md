# SpatialEntail Final v1

**Status: full server candidate verified; receipt-backed data release and four-run
SFT handoff implemented. Actual GPU training/evaluation remain unvalidated.
No training jobs have been submitted by this implementation.**

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
| SFT-Bench | 3,600 | Internal test: 2,400 in-range (L1-8), 1,200 extrapolation (L9-10) |
| SpatialMap-TQA-Corr | 1,500 | External evaluation |

Planned SFT runs: Natural/42, Natural/43, Symbolic/42 and Symbolic/43.
All four share the same frozen dataset. Train supplies gradient updates; Dev
is evaluated each epoch to select a checkpoint independently for each run.
Each selected adapter is evaluated on SFT-Bench and SpatialMap-TQA-Corr.
External questions retain original A-D choices and display the shared E/F
options under the Unique contract; this remains 1,500 questions per adapter.
The [sft.py](sft.py) entry point owns preparation, launch verification, training
and selected/untuned evaluation, using shared implementation in
[`entailment_sft.py`](../../spatial/v2/entailment_sft.py).

Synthetic splits share 60% A-D / 30% E / 10% F per query and matched E
subtypes, with one menu per slot. All use the same controlled generation
framework with questions, scenes and families disjoint across splits. Train/dev/in-range
use uniform levels 1-8; extrapolation uses equal levels 9 and 10. Balance
correct A/B/C/D positions where feasible. External data is not rebalanced.

Synthetic Count options now use `synthetic-count-options-0-4-v1`: four distinct
values from 0-4 for A-D, E and F. This removes the known cue from distractors
above the true-answer range. F still omits the sole entailed answer. The solver
continues to check the complete Count domain over every nonreference object.
Historical generation defaults and external menus are unchanged. The allocation
and scene seeds are unchanged, but use fresh run paths for these revised menus.

`explicit-size-checked-augmentation-v1` uses the explicit table in
`generation_policy.py`: L1 has 6-10 objects / 8-16 premises; both ranges rise
by two per level, reaching 24-28 objects / 26-34 premises at L10.
Choose objects N uniformly, then choose 2-6 additional premises beyond N.
This gives 25 compatible pairs per level, with uniform object and triangular
premise marginals, using the same sampling law for every answer case.
The successful trial's size-seed namespace is retained to preserve its draws.
Semantic cores are augmented using checked witnesses and shortest-support
checks, with a bounded solver fallback. Multiple-answer Which has exactly two
entailed members. Answer-support depth still comes from verified evidence,
not total scene size. Full details and the table are in the SFT plan.
Canonical repeats may be accepted on the fifth attempt within training only,
never across splits. Train contains 6,000 examples, not a guarantee of 6,000
unique canonical questions; audits record repeat counts and query frequencies.

Reuse shared infrastructure, not old adapters, caches or readiness claims.
Historical artifacts stay untouched. The agreed scope needs no repeated manual
approval, but new data and the actual training pipeline must pass validation.
Do not use the old composition submit script for this experiment.

## Run On Slurm

Use a fixed checkout of the whole repository on your server, with `uv` and
`sbatch` available. Shared solver code and tracked correction sources remain
repository dependencies; copying only this folder is insufficient.

From the repository root, submit generation for all four exported datasets:

```bash
bash experiments/spatialentail-final-v1/slurm/submit-generation.sh full
```

This submits **one CPU job** on `normal`: 128 CPUs, 128 GB RAM, 24 hours, with
128 workers for preflight construction and then synthetic admission by default
(following `SLURM_CPUS_PER_TASK`).
It requests no GPU, installs only the small CPU
generation environment, and never calls the old training submitter.
CPU dependencies are declared once in `generate.py`'s PEP 723 script metadata.
`uv run --no-project --script` uses a separate cached script environment,
preserving Z3 4.16.0.0 without resolving against the GPU training stack.
The launcher uses `--native-tls` for the system certificate store, compatible
with server uv 0.9.30. That version does not recognize `--system-certs`.
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
run directory. Generation, training, and evaluation workers request email to
`e0958116@u.nus.edu` for `BEGIN,END,FAIL,TIME_LIMIT`: job start, termination,
failure, and reaching the time limit. Slurm calls the start event `BEGIN`,
not `START`. These directives apply to new submissions; queued jobs retain
their submitted script copies. Slurm email delivery itself has not been verified.
`FINAL_GENERATION_WORKERS` can override the worker count within the CPU allocation.
Preflight construction and synthetic admission use separate, sequential worker
pools, not 256 concurrent workers. Preflight keeps at most one pending candidate
per allocated worker and returns only canonical keys and metadata to the
coordinator. The coordinator processes slots in the original order, makes all
exclusion/ownership decisions, and requests retries only on conflicts. The five
attempts and final-attempt training-only duplicate rule are unchanged.
External preparation, journal/export writes and final export readback remain
serial. Peak memory and throughput at 128 workers have not yet been measured
on the server.

The job needs network access for Python dependencies and one pinned tokenizer
snapshot; it does not download model weights. Package/Hugging Face caches stay
inside the submission directory. Admission policy `pinned-training-tokenizer-v1`
uses the same pinned Qwen3-4B-Thinking-2507 tokenizer for admission, context
checks and train/dev exports. Admission uses native-thinking template and
loss-mask validation, not the inherited non-thinking format. There is no
separate historical tokenizer.

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
    tokenizers/active/      # one pinned tokenizer for admission and training
    audit/                  # attempts, quotas, export checks, review samples
    data/
      train/
      dev/
      sft-bench/            # one benchmark, with in-range/extrapolation metadata
      external-test/
```

Every split has `references.jsonl`, `evidence.jsonl`, `token-measurements.jsonl`
and gold-free `natural-inputs.jsonl` / `symbolic-inputs.jsonl`.
SFT-Bench is one 3,600-question export. Its reference rows retain
`split: in-range-test` (2,400) or `split: extrapolation-test` (1,200) for scoring
breakdowns, plus the existing level/query/case metadata. These fields are not
added to model inputs. Generation still enforces the original subgroup quotas
and held-out exclusions. The manifest's `split_counts` describes exported
datasets; `generation_split_counts` retains the construction subgroups.
The evaluation entry point scores the combined benchmark once and aggregates
overall and subgroup results. It currently reuses the shared two-stage vLLM
runner, not a registered lm-eval-harness task.
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
uv --native-tls run --python 3.12 --no-project \
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

A shortage after five declared constructions, failed admission, invalid
external source or failed audit exits nonzero. `generation.json` and partial
audit/data files remain for diagnosis; there is no successful manifest.
The fifth attempt may accept a query duplicate confined to training only when
its scene and family are also free of held-out overlap. All cross-split conflicts
still fail, including conflicts hidden behind an existing training query key.
Dev/test duplicates still fail. Accepted repeats are marked
`training_duplicate` in evidence and `selected-training-duplicate` in the
construction journal. `audit/summary.json` reports frequencies, total repeats,
and scene-size counts by split/query/level/case. Requested sizes are bound in
evidence records and checked against policy during inventory auditing.
Do not overwrite failed runs, auto-refill slots, or pull code into a running
checkout. Submit a fresh run after a diagnosed fix. Hard kills can leave status
`running`, but without a valid completion manifest they cannot pass verification.

**A successful full build is an audited candidate, not a training release.**
Ambiguity and inconsistency require checked evidence and coverage across levels,
not increasing difficulty; their `verified_depth` stays null. Semantic-cohort
records use `semantic_complexity_status: verified-semantics-no-depth-claim`.
Positive support length is still checked, including semantic-cohort sole Which.
The original candidate retains its historical `pending_release_checks` and
`training_ready: false`. Completed data checks are recorded separately below;
actual trainer/GPU checks run inside SFT.
Adding `--require-training-ready` to verification rejects these candidates.
`afterok` on the CPU job alone is not a sufficient training gate.
Generation-only submission still never starts SFT. The SFT entry points below
do not promote candidates or bypass readiness checks.

## Data Release

The server candidate from `generation-full-OO5CGrnn` has manifest SHA-256
`30d1eab73e3a59d55678640ac92b296947c9c3163e1b58e28215c5d67a331598`.
Its 48 payload files and 84 generation-source files match the original commit
`21227fe07888f249c126b7c995a54ce09963007d`. The uploaded archive was unpacked
and checked locally without changing its manifest, payloads or readiness flags.

Candidate-specific evidence is in
[`release-evidence/30d1eab73e3a/`](release-evidence/30d1eab73e3a/):
full 12,300-record independent replay, fixed train-only nuisance diagnostics,
token/distribution profiles, and an attributed AI-assisted review of 149 paired
examples (130 case samples, ten contingent-Which levels, seven longest pairs,
the repaired Count slot and an L10 multiple-Which case). This is not human sign-off
or a claim that all 303 selected samples were manually read.

The diagnostic policy uses five family-grouped folds, seed 42, depth-3 decision
trees and minimum leaf size 30. Lengths, names/mention counts, menus, and their
combination are tested separately; no reference explanations or hidden labels
enter the features. A 10-percentage-point lift over the query-specific majority
baseline triggers investigation. None triggered: six-way accuracy was
26.75-29.7% versus 30%; A-D/E/F class accuracy was 58.2-60.15% versus 60%.
These probes do not establish the absence of every possible shortcut.

The actual receipt and all four local SFT configuration previews passed the
launch verifier. Targeted release/SFT/workflow/audit tests: 136 passed, one
optional Axolotl-import test skipped. No GPU training was run.
Train sequence maxima are 1,478 Natural / 1,637 Symbolic tokens, below 8,192.
Mean supervised target lengths are 257.112 / 296.985 tokens respectively;
Symbolic is longer here, so do not claim reference-token savings.

Verify the portable receipt against the unchanged server artifacts:

```bash
uv --native-tls run --python 3.12 --no-project \
  --script experiments/spatialentail-final-v1/generate.py verify-release \
  --run-dir /absolute/artifacts \
  --release experiments/spatialentail-final-v1/release-evidence/30d1eab73e3a/release.json
```

`release.json` binds the original manifest/commit and the evidence hashes.
Generation sources are checked against that Git commit; current training code is
separately frozen in each campaign. Updating the SFT handoff does not require
regenerating good data or rewriting an old manifest. Keep the original commit
available in Git. Candidate-only `verify` still expects its original checkout.

For a future candidate, `generate.py audit-release --run-dir ... --evidence ...
--generation-revision FULL_COMMIT --workers N` produces automated evidence in a
fresh directory outside the candidate. Run it through the same uv script command
with `--with scikit-learn==1.7.2` before `--script`. An attributed `review.json`
is still required; automated checks never manufacture a manual review.
Then `generate.py release --run-dir ... --evidence ...
--generation-revision FULL_COMMIT` checks and seals those receipts.

## SFT And Evaluation

On the Linux GPU server, install the existing root lock with `uv sync --frozen`.
No GPU dependency pins have been changed. All commands require explicit paths;
no latest-run discovery, implicit resume or data regeneration is used.

The training submitter prepares and checks all four configurations before its
first `sbatch` call:

```bash
bash experiments/spatialentail-final-v1/slurm/submit-training.sh \
  /absolute/artifacts \
  experiments/spatialentail-final-v1/release-evidence/30d1eab73e3a/release.json
```

It submits four independent one-GPU jobs, with separate configs, caches, W&B
IDs and output directories, submitted in seed-paired order: `natural-seed-42`,
`symbolic-seed-42`, `natural-seed-43`, `symbolic-seed-43`. Slurm determines actual
start order; no dependencies are added between jobs. Defaults are the existing cluster's
`gpu-long` partition and one `h100-47` GPU, eight CPUs, 64 GB RAM, 24 hours.
These resource/time defaults have not been measured with epoch Dev generation.

The runtime audit checks the pinned Transformers v5 effective settings:
`get_warmup_steps(1500) == 75` and `train_sampling_strategy == "random"`.
Axolotl converts the YAML `warmup_ratio` into `warmup_steps`; the trainer no
longer exposes `warmup_ratio` or `group_by_length`. Regression tests use real
Transformers arguments with mocked GPU/model details, not a full CUDA run.

Training and evaluation workers use the repository working directory set by
the submitters' `sbatch --chdir`, not their own script path: Slurm copies worker
scripts into its spool directory. Direct `sbatch` calls must also set
`--chdir=/absolute/repository`. After an early worker-path failure, retain the
data and release receipt and resubmit once the old jobs have failed or been
cancelled. Updating a checkout does not update already-submitted script copies.

W&B uses `sft-<campaign-path-hash>-<style>-seed-<seed>` for run names/IDs,
with the shared `sft-<campaign-path-hash>` group set by the Python launcher.
The hash uses the resolved campaign path, so separate submissions do not reuse
the generic `campaign` ID. Existing prepared configurations are not rewritten;
prepare a fresh campaign with its verified data-release receipt.

**Candidates without a valid release receipt are rejected before submission.**
Launch requires complete counts, matching original source/payload hashes,
completed review/replay/probe evidence and a hash-bound release receipt. The
candidate's own `training_ready` field stays false. GPU/loader/collator and live
epoch checks are mandatory runtime checks, not circular pre-submission gates.
Do not manually flip either candidate or release flags.

To inspect four configurations without training, including for smoke data:

```bash
uv run --frozen --no-sync python experiments/spatialentail-final-v1/sft.py prepare \
  --artifacts /absolute/artifacts --output /absolute/new-campaign
```

Before the first optimizer update, the Axolotl plugin checks all loaded token
rows/order/masks, mixed-length collator padding, one-GPU configuration, model
revision, quantization and LoRA targets. It aborts on nonfinite losses/gradients,
logs the opening steps and verifies 6,000 examples / 750 updates per epoch.
Dev loss uses the fixed Dev token file. After each epoch save, generated Dev
Stage-1 accuracy is measured from gold-free prompts in batches of four, with
the model's training state and random state restored afterward.

Each run saves `dev-epoch-{1,2}.jsonl` predictions and `.json` receipts, retains
both epoch adapters, exercises PEFT reload, and writes `selection.json`.
Selection is maximum Stage-1 Dev accuracy, earliest epoch on ties; seeds are
never selected against one another. No test questions enter training or selection.

After all four training jobs finish:

```bash
bash experiments/spatialentail-final-v1/slurm/submit-evaluation.sh \
  /absolute/campaign FULL_TRAINING_COMMIT
```

Use the full Git commit used to prepare the completed training campaign.
It must reproduce every recorded training-source hash; evaluation code changes
do not require modifying old campaign receipts or retraining. The revision can
be omitted only when the current checkout still matches those training hashes.
For a campaign prepared at `db3a966`, obtain the full value with
`git rev-parse db3a966`; do not assume that revision for another campaign.

The submitter verifies all four selections and every prompt budget before its
first GPU `sbatch`. This tokenizer-only preflight runs through `srun` on
`normal` with two CPUs, 8 GB RAM and 30 minutes. The submitter waits for that
allocation and successful checks before submitting any evaluations; the login
node only coordinates Slurm submission.
Preflight uses `generate.py verify-evaluation` in the existing pinned CPU
script environment, not the root GPU environment. It generates no data.
GPU workers still use the unchanged root lock and runtime checks.
It submits seven jobs, all on `gpu` with eight CPUs, 64 GB RAM,
and a three-hour limit:

| Jobs | GPU Per Job | Evaluations Per Job |
|---|---|---|
| Four selected adapters | `h100-47:1` | SFT-Bench and external |
| Untuned zero-shot | `h200-141:1` | Both styles on both benchmarks |
| Untuned one-shot | `h200-141:1` | Both styles on both benchmarks |
| Untuned three-shot | `h200-141:1` | Both styles on both benchmarks |

Each job loads one BF16 engine and reuses it sequentially, producing 20 result
files in total. Untuned baselines are not repeated by training seed.
Final policy `final-v1-evaluation-8192-fewshot-v1` gives every model 8,192
Stage-1 tokens, 16 Stage-2 tokens, and a 16,384-token context allowance. Stage 1
includes both thinking and the final answer, not 8,192 thinking tokens plus an
extra answer allowance. Training and historical in-epoch Dev budgets stay
unchanged at 8,192 context / 4,096 Stage-1 tokens. This final budget change does
not reselect checkpoints. The BF16/longer-budget transition remains unvalidated
on Dev; full evaluation time within three hours remains unmeasured.

Demonstrations are fixed before inference: hash-rank A-D Train examples, choose
one Direction, one Which, and one Count, in that order. One-shot uses the first
Direction example; three-shot uses all three. Both styles share question IDs,
menus and labels but use their matching reference explanations. This single
fixed demonstration set is not an estimate over demonstration choices. Native
thinking in demonstrations is preserved, and no held-out reference enters a
prompt. All prompts reserve the full Stage-1 cap, Stage-2 suffix and Stage-2 cap;
overflow fails before model loading instead of truncating.

Results live under `campaign/evaluation/final-v1-evaluation-8192-fewshot-v1/`,
separated by trained model or untuned shot setting, then style and benchmark.
Resume journals bind policy, demonstration IDs, Train reference hash, training
revision and evaluation-source hashes. Re-running the same command resumes
completed batches without mixing older budgets or prompts. Already-submitted
jobs retain their old scripts; use the updated launcher for new submissions.
The existing raw-token Stage-1/Stage-2 replay runner is reused. Each result has
a `.summary.json` with release identity, training seed, subgroup and separate
construction-level/verified-depth slices, generated-token mean/median/p95/max,
length-finish counts and thinking-token lengths before `</think>`.
Token counts are comparable across the hardware split; wall-clock latency is
not a matched H100-versus-H200 comparison.
Cross-seed statistical reporting and
lm-eval-harness task registration remain separate work.

### Collect Results While Baselines Run

From the repository root on the Slurm server:

```bash
sbatch experiments/spatialentail-final-v1/slurm/collect-results.sh
```

The default input is
`experiments/spatialentail-final-v1/runs/sft-y63EhpI5/campaign/evaluation/final-v1-evaluation-8192-fewshot-v1/`.
Pass another evaluation directory as the first argument when needed.
This CPU-only job uses `normal`, one CPU, 2 GB RAM and no GPU. It needs only
`python3` and does not modify or wait for running evaluations.

The job prints a 20-result status/accuracy table to
`spatialentail-results-JOBID.out`. It writes a timestamped directory and
`.tar.gz` archive under `campaign/result-reports/`. Send back that archive for
analysis, or paste its `report.txt`. The archive includes full summary metrics,
subgroup/depth slices, token profiles, provenance, selection/Dev receipts,
bounded evaluation log tails and available Slurm status. Raw model responses
and adapters are excluded. Completed result hashes are checked against their
summaries; partial journal counts are live progress only, never final scores.
Re-submit later to collect a fresh snapshot after the baselines finish.

## Local Tests

In a Python 3.12 environment containing the dependency pins from `generate.py`
and pytest:

```bash
python -m pytest experiments/spatialentail-final-v1/tests/test_generation.py -q
```

SFT tests additionally need the pinned Torch/Transformers/PEFT/Accelerate stack:
`python -m pytest experiments/spatialentail-final-v1/tests/test_sft.py -q`.
The optional plugin test requires Axolotl; the CPU tests do not certify its
complete GPU model/data-loader path.

Tokenizer-dependent tests require the pinned training snapshot in the Hugging Face
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

## Validation Summary

Local evidence recorded on 2026-11-02 and 2026-11-03. These overlapping runs
are not an additive unique-test total.

| Check | Result | Scope |
|---|---|---|
| Flexible-constructor trial | 10,800 selected; zero accepted duplicates | Construction/exclusions, not full final admission |
| Trial representative admission | 928 passed | Solver, explanations, independent replay and native tokens |
| Integrated constructor | 7,437 tests passed, including 4,800 endpoint cases | All 928 selected trial scenes/family keys matched exactly |
| Unified benchmark export | 726 tests passed | Merged files exactly concatenate the two former subgroup exports |
| SFT/evaluation integration | 785 tests passed; optional plugin test passed separately | CPU PEFT reload, masks, selection, launch guards; not full GPU training |
| W&B naming fix | 22 tests passed; one optional Axolotl test skipped | Eight distinct IDs across two campaigns; shared group within each campaign |
| SFT integration smoke | 176 synthetic + 3 external passed | Build, exports and separate manifest verification |
| Dependency-preservation smoke | 176 synthetic + 3 external passed | Fresh build and separate verification after the depth fix |
| Dependency-preservation replay | 10,800 selected seeds passed | Reconstruction, exact sizes and applicable depth checks; not full admission/export |
| Dependency-fix regressions | 7,543 generator/admission + 43 independent audit/token tests passed | Includes the failing scheduled seed, tied-proof rejection and 4,800 size endpoints |
| Parallel preflight | 101 workflow/audit tests passed | Real 2/4-process equivalence with retries; reversed completion order, queue bounds and failures |
| Parallel preflight smoke | 176 synthetic + 3 external passed with four workers | Data, construction/admission journals and preflight receipt byte-identical to serial preflight; separate verification passed |

The successful size trial observed all 250 declared level/size pairs, including
43 scenes with six objects and nine premises. It selected 10,767 first candidates,
32 second candidates and one third candidate, excluding all 1,500 external
questions. Temporary receipts are in `/tmp/flexible-core-campaign-v3/` and
`/tmp/flexible-core-endpoints-v3/`. The integrated source was then checked against
those trial scenes; a fresh full integrated-source admission campaign is still
pending. Earlier template endpoint limits were implementation restrictions,
not mathematical impossibility. Sorted dependency-witness selection fixed a
trial cross-process reproducibility failure.

The server admission failure `depth mismatch: 2 vs None` was reproduced locally
on the selected in-range L2 Count-zero slot 35. An added relation preserved the
answer and shortest length but introduced a tied proof bypassing the protected
dependency edge. Augmentation now preserves that edge in every shortest proof
of at least one deepest obligation, including solver-fallback proposals.
The admission requirement is unchanged. This fix can change constructed scenes:
retain the failed run for diagnosis and submit a fresh generation run after
updating the shared source; do not resume or reuse partial exports.
The fresh smoke at `/tmp/spatialentail-final-depth-dependency-smoke/` passed
full admission, native exports and separate manifest verification with pinned
Z3 4.16.0.0.

The parallel-preflight smoke is
`/tmp/spatialentail-final-parallel-preflight-smoke/`. Its comparison against the
preceding dependency-fix smoke checks output equivalence, not a 128-worker
server speedup or a new full-corpus admission run.

The SFT integration smoke was `/tmp/spatialentail-final-sft-integration-smoke/`: 44 train,
44 dev, 88 SFT-Bench (44 per subgroup), and three external questions. Its 30 data
files matched the preceding unified-benchmark smoke. Four SFT configuration
previews were prepared without training, and the launch verifier rejected their
smoke release. Subsequent source changes require fresh artifacts; these temporary
paths are evidence records, not durable releases or current launch inputs.

All 1,500 external questions separately passed source-menu preservation,
independent Unique label verification and tokenizer prompt budgets. There were
212 label differences from the historical correction export, reflecting the
Unique contract projection rather than synthetic rebalancing.

The old Count-menu rule scored 47% on a disjoint 400-question train-only probe,
against a 30% majority baseline. After the 0-4 menu fix, replay over 2,000 planned
train Count menu seeds reduced the former rules to always-A (15%) and always-E
(30%). This removes the demonstrated range cue, not all possible shortcuts.
Direction/recipe-variant seed coupling remains a shared coverage limitation,
not a demonstrated wrong-label or harmful-shortcut bug.

No Slurm, full CUDA training, online W&B initialization or final model evaluation
was run locally. The pinned Axolotl collator source passed a mixed-length CPU
probe, but the complete loader/model path still needs server validation.
The data-release evidence above completes the candidate-specific review and
shortcut checks. Remaining GPU/runtime and uncertainty work is listed in the
[SFT plan](SFT-PLAN.md); data readiness is not a claim of successful training.
