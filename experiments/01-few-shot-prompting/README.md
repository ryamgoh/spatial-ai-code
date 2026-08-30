# Experiment 1 — Few-Shot Prompting Strategy Optimisation

**Source.** Dissertation H036660, §4.3 / Appendix Fig. 5 + §6.2.2.

**Question.** With DeepSeek-R1-0528-Qwen3-8B fixed from Exp 0, does the
prompting strategy change accuracy on SpatialMap-TQA-CORR?

**Setup.** Same base model, two-pass vLLM. The "shots" are inlined into the
system prompt (`num_fewshot: 0` in all configs). Seven strategies + baseline
control; full prompt text also kept as standalone copies in `../prompts/`.

| config | prompt (from thesis appendix) | reported acc |
|---|---|---|
| `eval-baseline.yaml` (Exp 0) | Default instruction following | 73.7% |
| `nonshot_1.yaml` | Full structured pipeline with explicit decomposition | 74.3% |
| `nonshot_1_mul.yaml` | Variant of Non-shot-1 supporting multiple correct answers | 75.1% |
| **`nonshot_2.yaml`** | **Simplified pipeline, merged instruction steps** | **77.6%** |
| `nonshot_3.yaml` | ASCII-formatted instructions with explicit round logging | 73.7% |
| `nonshot_4.yaml` | Final zero-shot formulation without verification step | 75.3% |
| `oneshot_3.yaml` | Non-shot-3 + one worked example | 74.4% |
| `threeshot_3.yaml` | Non-shot-3 + three worked examples | 68.3% |
| `fewshot.yaml` / `nonshot.yaml` | Earlier full-set (2000-row) variants | — |

**Result.** **Non-shot-2: 77.6%** (+3.9 pp over base). No few-shot strategy
beat zero-shot; 3-shot degraded to 68.3% with context explosion.

Non-shot-2 is the prompt locked in for Exp 2's "+ Non-Shot-2" arm and for
the structured-prompting narrative in the report.

## Run log

| date | results dir | notes |
|---|---|---|
| | | |
