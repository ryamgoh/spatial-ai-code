# experiments/

Every experiment lives in its own directory: its train/eval config yamls, its
launcher notes, and a README recording the research question, setup, and result.

Numbering follows the dissertation (H036660, Ch. 4): Experiments **0–3**.
Experiments **4a/4b** (Qwen3.5-4B pipeline smoke), **5** (GRPO), **6**
(baseline task feasibility), **7** (SFT starting state), and **8**
(dual scaling: data × params) are post-thesis follow-ups and are not
part of the report.

`eval/` and `finetune/` contain only code and their uv environments — no
configs. New experiments get a new numbered directory here.

## Layout

```
experiments/
├── 00-baseline-model/            Exp 0: un-tuned model comparison (73.7% DeepSeek)
├── 01-few-shot-prompting/        Exp 1: prompt-strategy sweep (77.6% Non-shot-2)
├── 02-spatial-baseline-models/   Exp 2: specialised spatial models on TQA-CORR
├── 03-sft-vs-baseline/           Exp 3: baseline vs SFT (73.7% → 87.9%)
├── 04a-smoke/                    Exp 4a: Qwen3.5-4B pipeline smoke (finetune.py)
├── 04b-cli/                      Exp 4b: same smoke via axolotl CLI (no finetune.py)
├── 05-grpo/                      post-thesis GRPO (inconclusive — not in report)
├── 06-baseline-task-feasibility/ Exp 6: zero-shot Single vs Multi TQA-Corr
├── 07-sft-starting-state/        Exp 7: SFT Base vs Instruct on TQA-Corr-Single
├── 08-dual-scaling/              Exp 8: SFT data × param scaling on TQA-Corr-Single
├── archive/                      unrunnable / abandoned configs
├── tasks/                        lm-eval task definitions (shared; via include_path)
└── prompts/                      standalone copies of Exp 1 system prompts
                                  (inlined into the yamls; not read at runtime)
```

## How configs resolve paths (do not "fix" these)

All relative paths inside configs are resolved from the **launcher's working
directory**, not from the config file:

- Eval launchers `cd eval` → `include_path: ../experiments/tasks`, task
  `data_files: "../data/spatialeval_cleaned.jsonl"` → repo `data/` (eval CWD is `eval/`).
- Finetune launchers `cd finetune` → `base_model: ../experiments/<exp>/models/...`, dataset
  `path: ../data/...jsonl`, `reward_funcs: rewards.*` (imported from CWD).

So a config run from the right CWD works regardless of where in `experiments/`
it sits. Moving a config between experiment dirs is safe; moving a dataset or
checkpoint is not — update the config in the same commit.

## Running

Eval (from repo root, or submit to SLURM):

```bash
cd eval
uv run python eval_new.py --config ../experiments/<exp>/<config>.yaml
```

Finetune (name or path; bare names are searched recursively under experiments/):

```bash
cd finetune
uv run python finetune.py ../experiments/03-sft-vs-baseline/train-sft-8b.yaml
```

Outputs: eval results go to `<experiment-dir>/results/<timestamp>/` (next to
the config that produced them); training checkpoints and merged models go to
`<experiment-dir>/models/<name>` as set by each train yaml's `output_dir`.
Record the resulting `results/` timestamp in the experiment README so runs are
traceable.
