# Dynamic Spatial Reasoning — Research Report & Close-out Plan

*For the progress meeting with Prof. Lee Wee Sun. Based on the state of the
codebase (`work-on-dynamic` branch, GRPO run in progress) and a survey of the
2025–2026 literature.*

---

## 1. What "Dynamic" Actually Means

The literature uses "dynamic spatial reasoning" for **two distinct things**.
Establishing this distinction with your professor first determines what is in
and out of scope.

| | **A. Temporal state tracking** | **B. Intrinsic mental transformation** |
|---|---|---|
| What changes | Relations **between** objects: objects move / appear / disappear; observer moves or rotates | The **internal structure** of one object: rotation, folding, assembly |
| Spatial-DISE quadrant | **E-D** (extrinsic–dynamic) | **I-D** (intrinsic–dynamic) |
| Spatial-DISE tasks | 3D Combination (mostly video-based) | 3D/2D Rotation, 3D Folding, Fold-and-Punch |
| Text-only viable? | **Yes** — SnorkelSpatial, STARK, SpaMEM, MentalMap L4 are text | Barely — essentially needs images / 3D renders |
| Fit with axis decomposition | **Direct** — axis graphs over time | **None** — needs 3D geometry, not ordering |

**Spatial-DISE** (Huang et al., 2026, arXiv:2510.13394, ICLR 2026 poster)
defines the DISE 2x2 taxonomy your dissertation already cites:
- **Intrinsic** = within-object (its parts); **Extrinsic** = between objects or
  to the environment.
- **Static** = fixed configuration, no transformation; **Dynamic** = requires
  mental simulation of a change (rotation, folding, reassembly, movement).
- Its 10 benchmark tasks map onto the quadrants; the "dynamic" ones
  (rotation, folding, projection, combination) are I-D/E-D *geometry* tasks
  solved by simulating how a *shape* transforms. Key finding: 28–32 VLMs
  average ~28.4% accuracy vs 76.8% human; 72.5% of errors are reasoning
  deficits (rule application, mental simulation, spatial working memory), not
  perception. Fine-tuning on their 12K synthetic set gives large gains
  (Qwen2.5-VL-7B: 26.1% -> 47.0%) with transfer from 3D E-D to 2D.

**Consequence for you.** Your work — and axis decomposition in particular —
lives in **E-S** (extrinsic-static): relations among multiple objects given a
fixed configuration. Spatial-DISE's "dynamic" is mostly I-D shape
transformation, which is a different (and visual) problem. The text-only,
LLM-native meaning of dynamic that is *reachable from your current framework*
is **A: temporal state tracking** — "the scene evolves by explicit actions;
reason about the new configuration." That is the defensible, coherent next
chapter for this dissertation.

### 1.1 The closest prior work (what you would be talking to)

| Work | What it is | Relation to us |
|---|---|---|
| **SnorkelSpatial** (Snorkel AI, Oct 2025) | Text-only, programmatically verified. 20x20 board, 3 particles, 10–200 actions (move F/B/L/R, rotate 0–270), board and particles move independently with wrap-around. Queries: absolute/relative position, tile, orientation. Leaderboard: GPT-5.4 ~99%, o3 76.7%, gpt-oss-120b 52.7%, most models <50%, accuracy collapses as action count grows | **The closest text benchmark to a dynamic version of our task.** It uses coordinates, not qualitative direction facts. Nobody has done it with *qualitative* relation sentences + axis decomposition |
| **STARK** (2025, arXiv:2505.11618) | LLM benchmark, 26 tasks, 14k+ challenges: state estimation/tracking along trajectories with partial observability, Allen's interval algebra + geometric predicates | Validates that *temporal* reasoning is a recognized LLM evaluation axis; heavier on formal temporal logic |
| **MentalMap** (2026, arXiv:2605.28277) | Pure-text world-model diagnostic, L0–L5 staircase from atomic facts to generative world-graph output. Universal "L3 cliff" at viewpoint/frame reasoning; L4 = dynamic state updates; even humans plateau ~41% on L4 in text | **Strong external validation of your whole approach**: world-graph (your axis graph) + text is the right substrate; L4 dynamic state updates is the frontier and is hard even for humans |
| **SpaMEM** (2026, arXiv:2604.22409) | Action-conditioned state changes (spawn/place/remove) over long sequences; L2 = temporal reasoning with *textual state histories*, L3 = visual belief maintenance | Same "state-update-over-time" task family; confirms the split between giving the model state vs. making it maintain it |
| **Building-Blocks-to-Planning** (arXiv:2512.24532) | SFT on synthetic atomic spatial transformations, then **GRPO RL** for multi-step planning in *dynamic* environments with explicit state updates | Closest methodological match to your SFT->GRPO pipeline; proves the SFT->RL-on-dynamic-env recipe works |
| **DSR-Bench / "Learning to Reason in 4D"** (arXiv:2512.20557) | Dynamic *spatial* reasoning: evolving object geometry + 3D relations, with training data | Confirms "train on dynamic data -> improve on dynamic tasks" |
| **Video-based** (DSI-Bench, Dyn-Bench, SpatialScore, Spatial4D-Bench, VLM4D) | 4D video benchmarks for VLMs/MLLMs (camera + object motion) | Out of scope for a text-only dissertation, but useful for the related-work narrative on why text state tracking is a *minimal* version of 4D reasoning |
| **QSTR theory** (Röhrig 1994; Ligozat; Dylla et al. 2017) | Qualitative calculi: your two axis graphs *are* the point/linear-order calculus (PA); cardinal directions = product of two linear orders; dynamic extension = spatio-temporal constraint updating (STCC, motion classes, conceptual neighbourhoods) | **Theoretical grounding**: axis decomposition is not ad hoc, it is the linear-order calculus of qualitative spatial reasoning, and the "dynamic" extension has a known name in that literature |

---

## 2. What Changes in Axis Decomposition — and What Doesn't

Your current pipeline (per `finetune/generate_all.py` and `eval/clean_v3.py`):

1. Assign each entity a **fixed** (x, y) in [0, 100].
2. Emit relational sentences ("The A is to the Northeast of the B").
3. Parse each sentence into two atomic facts: X-axis edge (a < b) and
   Y-axis edge.
4. Maintain two **time-free** partial orders (AxisGraph = DAG per axis).
5. Take transitive closure once, over the final static world.
6. Answer direction / which-entity / count queries against the closure.

The core insight that survives a move to dynamic settings is: **the world is
just (axis order, time)**. A dynamic world is the same two linear-order
calculi, indexed by time, with actions that edit them.

### 2.1 The minimal extension (what "dynamic" concretely is)

Add a time axis. The model keeps the same per-axis representation, but as a
**sequence of states** (t_0, t_1, ... t_n):

```
t0:  Police < Library < Hospital   (X),  Park < Library   (Y)   ...
action 1: "The Library moves East"        -> edit X-order
t1:  Police < Library < Hospital ... (Library's x updated)
action 2: "The Park moves North of the Police Station"
t2:  ...
Question (at t_n): "In which direction is the Library relative to the Park?"
```

Four design decisions to make explicit (each is a potential experiment):

| Decision | Static version | Dynamic options |
|---|---|---|
| **Action language** | — | (a) *qualitative*: "A moves East of B" / "A is now North of B" — pure relation edits, consistent with your fact format; (b) *metric*: "A moves 3 tiles East on the grid" — SnorkelSpatial-style, needs coordinate state; (c) *egocentric*: "A moves forward 2" while A faces West — requires tracking orientation (harder, matches SnorkelSpatial findings that relative queries are the failure mode) |
| **State the model maintains** | single closure | (a) full re-derivation per step (your current trace style, extended with a state line per timestep); (b) *incremental* updates — only recompute affected edges (the "bounded output" property of M1 becomes the selling point); (c) compact temporal log (SpaMEM-style "oracle textual history") |
| **What is queryable** | final configuration | (a) only final state (cheapest); (b) final state + *temporal* queries ("When was the Library first East of the Park?", "How many times did the Hospital change its relative position to the Zoo?") |
| **Information structure** | all sentences before question | (a) interleaved: sentence/action/question in time order (natural); (b) partial observability: actions given as *relative* statements without full state, forcing genuine belief revision |

**Key observation from your own results that motivates this:** your M1 model
already emits an *explicit per-step state line* ("X-State / Y-State" after each
sentence) and this is exactly what made its traces bounded (708 tokens vs M0's
5,816) and 5.33x faster. In the static setting, the "steps" are sentences. In
the dynamic setting, the "steps" are **actions**. The M1 procedure generalises
verbatim: initialize world -> for each event, extract axis facts, update the
two orders, write the state line -> final deduction. This is the single
strongest argument that the dynamic extension is *native* to your framework,
not a new project.

### 2.2 Concrete generator design (builds on `generate_all.py`)

`generate_sample_dynamic(...)`:
1. Sample k entities, initial coordinates (as now).
2. Generate an action sequence of length m in {2..10}:
   - move one entity (qualitative: pick a new anchor + direction, or metric:
     delta on the integer grid);
   - optionally spawn/remove an entity (entity-set change — SpaMEM's hardest
     case, and the case where a naive model breaks);
   - optionally a no-op / redundant action (tests whether the model tracks
     equivalences, cf. SnorkelSpatial's 0-degree rotations).
3. Re-derive coordinates; for each timestep, re-extract the two axis orders
   and the reduced state line (your `format_state` already produces the
   compact grouped-chain form).
4. Question types (extend your existing three):
   - **D0** direction at final state (same as Type 0, but after the sequence);
   - **D1** which-entity / count at final state (Type 1/2);
   - **D2** temporal: "At how many points was A East of B?" / "After which
     action did A first become North of B?";
   - **D3** counterfactual (stretch): "If A had not moved, where would it be
     relative to B?" — pure closure manipulation, no coordinate arithmetic.
5. Gold: recompute everything deterministically from coordinates (you already
   trust the oracle — the oracle is now just *run per timestep*, or maintain
   the two orders incrementally; both are cheap).
6. Emit SFT traces in your existing format (Initialization / Step i with
   action text + X-Extraction/Y-Extraction + state lines / Final Deduction).
7. For GRPO: prompt-only rows + `oracle_option` exactly as
   `generate_grpo.py` does today — **your reward functions
   (`outcome_reward`, `format_reward`, `length_penalty`) need zero changes.**
   The structured, bounded trace is what keeps GRPO reward signal dense, and
   dynamic traces inherit that.

Difficulty knob = number of actions (SnorkelSpatial's finding: accuracy
degrades monotonically with action count — gives you a free
length-controlled difficulty curve and a clean "scaling" figure).

### 2.3 Where the axis method has limits in the dynamic world (say this up front)

- **Ordering is not enough for metric motion.** If actions are "move 3 tiles
  east", the linear-order representation loses the *distance*; you need
  coordinates or betweenness for "is A still closer to B than C". The
  qualitative-only action language sidesteps this, but then you cannot ask
  distance questions.
- **Rotations / egocentric actions** require an orientation state per entity
  (a cyclic-order calculus — Röhrig's CYCORD — i.e., a third graph type).
  This is a genuine new component, not just "reuse the same code".
- **Spawn/remove** change the entity set; your "track only active entities"
  strategy extends naturally (activation now also covers new entities).
- **Non-transitive relations** ("immediately north of") become much more
  common once motion is involved ("A passes B") — your dissertation already
  lists this as a limitation; the dynamic setting makes it unavoidable.

---

## 3. Candidate Research Questions (pick 1–2 to commit)

Each is framed so it can *close out* the dissertation: it extends the
existing story (validate static benchmark -> structured reasoning -> SFT ->
RL) into the dynamic quadrant and answers a specific question.

### RQ1 — "Does explicit axis-state tracking transfer from static to
### dynamic spatial reasoning?"  (RECOMMENDED)
- **Claim to test:** a model fine-tuned on *static* axis-decomposition
  traces (your M1) + fine-tuned on *dynamic* traces (new generator) outperforms
  (a) the base model, (b) M1 alone on dynamic tests, (c) a model SFT'd on
  dynamic data *without* the structured state-line format.
- **Why it closes the dissertation:** directly answers the EWoK motivation
  ("do LLMs maintain world models?") with a *controllable* test: dynamic
  queries require the model to actually maintain and update the world model,
  not just parse static relations. Your 87.9% static result becomes the
  "can it build the model?" half; RQ1 is the "can it *update* the model?" half.
- **Effort:** low-to-mid. Generator ~1 week (the hard machinery — parsing,
  state formatting, rewards, eval harness — all exists). One new SFT dataset
  (1–2k samples) + one GRPO run. One new benchmark set (Dynamic-SpatialMap-TQA,
  say 500–1,000 held-out samples, plus optional zero-shot eval on
  SnorkelSpatial-style problems for external validation).
- **Risk:** if structured static SFT does *not* transfer, that is still a
  publishable, citable negative result (cf. Spatial-DISE's "prompting is
  model-specific" finding, your own Exp. 2 result).

### RQ2 — "How does dynamic spatial accuracy scale with action-sequence
### length, and does structured reasoning change the degradation curve?"
- **Claim:** unstructured models degrade steeply and early (M0-style
  unbounded reasoning); the M1-style procedure degrades later / more slowly
  because state is written down rather than re-derived each step.
- **Deliverable:** a SnorkelSpatial-style difficulty curve (accuracy vs.
  action count 2/4/6/8/10/15) for M0, M1, and M1+dynamic-SFT, plus your own
  dynamic benchmark. Cheap to produce *given* RQ1's data; nearly free as a
  byproduct analysis.
- **Fits your profile:** you already report token/latency scaling; this
  extends that analysis to sequence length and produces the headline figure
  for a "why structured reasoning helps" argument.

### RQ3 — "Temporal (when / how-many-times) queries: can closure
### manipulation be taught in text?"
- **Claim:** D2/D3-type queries (when did the relation first hold, how many
  times did it flip, counterfactual "if A had not moved") are answerable by a
  fine-tuned model using only the maintained state lines, and they fail on
  base models *even when the final-state query is easy*.
- **Why interesting:** this is a genuinely new task type in the text-only
  literature (SnorkelSpatial is final-state only; STARK is closer but uses
  formal temporal logic, not natural-language relations). Strongest novelty
  claim of the three, but also the least validated — highest risk.
- **Effort:** mid. The generator needs per-timestep query sampling and
  temporal answer checking; the oracle work is small but new.

### RQ4 — (stretch, likely too big for this semester) **Egocentric +
### rotation.** Add orientation state and egocentric actions ("move forward",
  "turn left"), i.e., the third (cyclic) axis. This is the full
  SnorkelSpatial task set in qualitative form. Do not commit to this for the
  close-out; mention it as future work.

**Recommendation for the prof meeting:** commit to **RQ1**, treat **RQ2** as
its analysis layer (same data, different figure), and list **RQ3** as a
stretch goal you'll attempt if RQ1's SFT lands well. This is one coherent
chapter: "Dynamic SpatialMap-TQA: extending axis decomposition to
state-tracking, with SFT+GRPO and a difficulty-length scaling analysis."

---

## 4. Timeline (working backwards from a ~3-week close-out)

Assumptions: GRPO run finishes ~this week; one H100 available for SFT/GRPO;
you can spend ~2–3 focused days/week on code.

### Week 1 — freeze the static story + scaffold the dynamic generator
- [ ] **D1:** Finish/collect GRPO results. Table: M0, M1 (SFT), M2 (SFT+GRPO)
      on SpatialMap-TQA-Corr (1,329). This is your "static results" section —
      whatever the GRPO run gives, it's a real result to report tomorrow.
- [ ] **D1–D2:** Write `generate_dynamic.py`: action model (start with
      qualitative actions + spawn/remove), per-timestep state lines reusing
      `AxisGraph.format_state`, question types D0/D1, 2,000 samples + 500
      held-out test. (D2/D3 optional.)
- [ ] **D2:** Sanity-check the oracle: generate 20 dynamic problems, solve
      them by hand / with an independent coordinate simulation; verify
      distractors and multi-answer cases.
- **Milestone (end of week 1):** dynamic dataset v1 + a one-page
      "dynamic story" draft for the dissertation skeleton. *This alone is
      enough material for the progress meeting.*

### Week 2 — baseline the dynamic task (no new training)
- [ ] **D1:** Evaluate M0, M1 (existing checkpoints) zero-shot on the dynamic
      test set (same two-pass harness, `nonshot_2` prompt vs. baseline).
      This answers "how bad are current models at the dynamic version of
      *our own* benchmark?" — an immediate result.
- [ ] **D2–D3:** SFT on dynamic data (M3 = base + dynamic-SFT; M4 = M1 +
      dynamic-SFT as a continuation). Same Axolotl config as M1, 1k–2k
      samples.
- [ ] **D3:** Also run M0/M1 on SnorkelSpatial-format problems (convert a
      small subset; optional, only if time allows) for external validation.
- **Milestone (end of week 2):** the RQ1 experiment table is *almost*
      complete (only GRPO on dynamic data outstanding).

### Week 3 — RL + write-up
- [ ] **D1–D2:** GRPO on the dynamic generator's prompt-only rows (reuses
      `generate_grpo.py` + `rewards.py` unchanged). M2-dynamic.
- [ ] **D2:** RQ2 analysis: accuracy vs. action-count curves; token/latency
      per action step; error taxonomy of M4 regressions (you already have
      this machinery from §4.5.1 of the dissertation).
- [ ] **D3–D5:** Write the new chapter: problem definition, generator,
      results (RQ1 table + RQ2 curves), limitations (ordering vs. metric,
      rotations, non-transitive "passing" relations — you already listed
      these in §5.2), future work (egocentric/rotation = RQ4; multimodal =
      the video-benchmark line).
- [ ] Buffer days: re-runs, figure polish, advisor feedback loop.

### What to say about the *current* GRPO run tomorrow
Frame it exactly as the plan implies: "I am confirming the SFT gain is
robust under RL — if GRPO holds or improves the +14.2pp, the static story is
complete and I am moving to the dynamic extension, which is the
extrinsic-dynamic quadrant of the DISE taxonomy I cited in the intro. I have
already designed the extension: it reuses the axis graph as a
time-indexed state, the same trace format, and the same reward functions.
Here is the one-page design." (This report's §2–§4 is that one page.)

---

## 5. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Dynamic SFT doesn't beat static-SFT transfer (RQ1 negative) | Medium | Still a result: "explicit state-line format transfers to dynamic only when retrained on dynamic data" — and RQ2's degradation curves are unaffected |
| 8B model struggles with long action sequences (context > 1k tokens) | Medium | Cap sequence at 8 actions; the difficulty curve *is* the finding; note the token budget in the config (`sequence_len: 1024` may need bumping for dynamic prompts) |
| GRPO reward signal sparse on dynamic (multi-correct answers) | Low | You already handle multi-answer gold ("A,D"); outcome_reward is set-exact |
| Time overruns | Medium | D2/D3 temporal queries and SnorkelSpatial external eval are the first things to cut; RQ1+RQ2 on the in-house benchmark is the floor |

## 6. One-paragraph pitch (for the meeting)

> The dissertation so far validates a benchmark, builds a deterministic
> oracle (axis decomposition), and shows that *teaching* that procedure to
> an 8B model via structured SFT raises accuracy 73.7% -> 87.9% at 1/3 the
> tokens. GRPO is currently confirming the RL side. The natural next step —
> and the part the DISE taxonomy's extrinsic-dynamic quadrant asks for — is
> to make the world *change*: objects move, appear, and disappear over a
> sequence of actions, and the model must maintain and update the world
> model. The same axis graph, indexed by time, is the oracle; the same
> state-line trace format, the same data pipeline, and the same rewards
> carry over unchanged. I propose a new Dynamic-SpatialMap-TQA benchmark
> plus one SFT+RL experiment (does the structured procedure transfer to
> state tracking?) and a difficulty-length scaling analysis, which closes
> the "do LLMs maintain world models?" question from the introduction.

