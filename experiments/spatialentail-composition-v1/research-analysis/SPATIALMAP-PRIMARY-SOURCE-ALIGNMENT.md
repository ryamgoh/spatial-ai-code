# Spatial-Map TQA: Primary-Source Alignment

Research date: 2026-10-29. Scope: original author documentation, released Hugging Face rows, and official remote repository. No local adapter, local dataset, generation, training, or model evaluation was inspected/run. Parent report owns local adapter/solver findings.

## Sources and Provenance

- **[P1] Paper, dataset setup:** https://arxiv.org/html/2406.14852v2#S2.SS1 (the requested unversioned URL currently serves v2, dated 2024-11-04).
- **[P2] Paper, evaluation:** https://arxiv.org/html/2406.14852v2#S2.SS2 ; results aggregation: https://arxiv.org/html/2406.14852v2#S3 .
- **[P3] Paper, complete examples:** https://arxiv.org/html/2406.14852v2#A3.SS4 and Figure 14, https://arxiv.org/html/2406.14852v2#A3.F14 .
- **[H] Author-released dataset:** https://huggingface.co/datasets/MilaWang/SpatialEval ; revision observed through the Hub API: `59ce0450bf65ae8dfa3590062442cf642474d7fe`. [Pinned dataset card](https://huggingface.co/datasets/MilaWang/SpatialEval/blob/59ce0450bf65ae8dfa3590062442cf642474d7fe/README.md).
- **[R] Official repository**, linked by paper, dataset card, and [project site](https://spatialeval.github.io/): https://github.com/jiayuww/SpatialEval . Inspected using `gh`; `main` was `d82ba382805265f205693cb83a30c4866ea6c35b`, the only listed branch.
- **[D] Dataset-server observations:** `https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=OFFSET&length=100`. Read offsets 0, 100, ..., 1500; analyzed the 1,500 Spatial-Map rows at indices 0-1499. Server requests were to the current release, not revision-pinned requests. Exact example URLs appear below.

## Native Contract: Author Documentation

1. **Input:** TQA is text-only input plus a question. Authors say it contains all information necessary for a person to answer. VQA and VTQA are alternative modalities for the same questions. Spatial-Map textual scenes contain pairwise relations between uniquely named locations. The paper describes configurable `K` objects and illustrates `K=6`. It explicitly contrasts Spatial-Grid's rigid grid with Spatial-Map, “where objects are positioned arbitrarily.” [P1]
2. **Questions/output:** Each synthetic sample has three questions, Q1-Q3. [P3] The paper describes spatial relationships and counts; the released rows establish the exact direction/which/count templates below. Each question has four options and one reference answer, not a set of possible answers. [P2, H, D]
3. **Prompt:** The original evaluation appends: “First, provide a concise answer in one sentence. Then, elaborate on the reasoning behind your answer in a detailed, step-by-step explanation.” This is a prose-response evaluation, not a documented letter-only protocol. [P2; [README lines 57-60](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/README.md#L57-L60); [inference lines 47-52](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/inference_lm.py#L47-L52)]
4. **Metric:** Accuracy is the main metric; task results average across questions. For nondeterministic decoding, the paper reports three independent runs for open-source models and one for proprietary models. [P2]
5. **No native Unique/Listed/Exhaustive:** These operators/contracts are not defined in the inspected paper, dataset card, README, or Spatial-Map scoring path. None of the 1,500 inspected Spatial-Map prompts contains those words, case-insensitively. The paper's “unique location name” concerns entity naming, not uniqueness of a qualifying object or logical answer. There are no E/F ambiguity or overflow options in these rows. [P1, D; scoring references below]

## Native Contract: Released Data Observations

These are measurements of released rows, not claims about an unpublished generator.

| Aspect | Observed release |
|---|---|
| TQA schema | Five string fields: `id`, `text`, `oracle_answer`, `oracle_option`, `oracle_full_answer`; no coordinates, relation graph, or image field. |
| Split versus task | Server/card report 4,635 TQA test rows across tasks. Indices 0-1499 are Spatial-Map; index 1500 is `mazenav.tqa.0.0`. |
| Spatial-Map block | 500 scene IDs, 2000-2499, each with suffix `.0`, `.1`, `.2`; 500 rows per question family. |
| Scenes | All 500 `.0` scene texts contain six distinct names and nine directional relation sentences. Premises use the four diagonal words only. |
| `.0`: direction | `In which direction is X relative to Y?` Four options, exactly Northeast/Northwest/Southeast/Southwest, in varying order. |
| `.1`: which | `Which object is in the D of X?` Query direction is diagonal. Four distinct scene-object names. This menu is not all six scene objects. |
| `.2`: count | `How many objects are in the D of X?` Query direction uses **all eight** compass words. Four numeric choices per row, drawn from 0-5; one numeric reference stored as a string. |
| Cardinal counts | North: 73 rows; South: 56; East: 54; West: 79. **All 262 have reference 0.** Diagonal count questions: 238. Across all 500 counts, 351 references are 0. |
| Direct nonuniqueness | At least 45/500 which rows contain two or more **explicit same-direction premises relative to the query anchor**. This is a direct-text lower bound, not a solver-derived ambiguity census. |

Measurement method: parse server JSON, group by ID suffix, extract fixed question/option templates and pairwise scene sentences. For the 45-row count, only identical anchor/direction matches in explicit premises were counted; no inverse/transitive inference was used. Source pages are reproducible with [D]; e.g. [first which page](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=500&length=100), [first count page](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=1000&length=100), [last count page](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=1400&length=100). This remote inspection did not scan indices 1600-4634 for additional task occurrences.

## Scoring: Documentation Versus Executable Release

**Documentation:** README calls the evaluation “exact match.” An [owner comment](https://github.com/jiayuww/SpatialEval/issues/2#issuecomment-2933500896) also describes the three synthetic tasks as exact-match parsing, contrasting them with GPT-4o judging for Spatial-Real. Do not transfer Spatial-Real's judge procedure to Spatial-Map.

**Executable release:** [`evals/evaluation.py`, lines 26-136](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/evals/evaluation.py#L26-L136) selects a parser using the last ID component:

- `.0`: extract a diagonal direction, preferring `A-D. direction`, otherwise the first matching direction word.
- `.1`: heuristic object-name extraction using answer-letter and prose patterns, with some model-specific fallbacks.
- `.2`: prefer `A-D. integer`, otherwise extract the earliest matching digit or number word; number words include “no” as zero.

Then [lines 266-298](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/evals/evaluation.py#L266-L298) compare **lowercased reference-answer substring membership in the extracted answer**:

```python
ref_ans = str(data['oracle_answer']).lower()
model_answer = str(model_answer).lower()
eval_result = int(ref_ans.lower() in model_answer.lower())
```

Accuracy is the number counted correct divided by model-output line count. `oracle_option` and `oracle_full_answer` are not the compared fields here. Consequently, reproducing strict option-letter accuracy or exact string equality is not identical to reproducing this implementation. It does not validate logical entailment, alternate valid objects, uniqueness, or exhaustive answer sets.

An external contributor's [PR #3](https://github.com/jiayuww/SpatialEval/pull/3) proposes a revised prompt and parser and explicitly identifies the substring problem. It was **open and unmerged** when checked. Its proposed letter-response protocol is not the original released protocol, and the contributor's observations are not author documentation.

## Raw Story Origins and Examples

The following JSON objects preserve the five original field values, including scene wording, spacing, options, and supplied label. They are dataset-server rows, not reconstructed scenes or adapter outputs.

### A. Scene 2000, Direction

Source: [row 0](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=0&length=1).

```json
{"id":"spatialmap.tqa.2000.0","text":"Consider a map with multiple objects: \nPolice Supply Store is in the map.  Narwhal's Novelties is to the Northwest of Police Supply Store.  Coral Crafts is to the Northwest of Narwhal's Novelties. Coral Crafts is to the Northwest of Police Supply Store.  Planetarium Prints is to the Southeast of Coral Crafts. Planetarium Prints is to the Northeast of Police Supply Store.  Oz Oddities is to the Southwest of Planetarium Prints. Oz Oddities is to the Southwest of Police Supply Store.  Ice Queen Ice Cream is to the Northwest of Planetarium Prints. Ice Queen Ice Cream is to the Southeast of Coral Crafts. \n\n Please answer the following multiple-choice question based on the provided information. In which direction is Planetarium Prints relative to Police Supply Store? Available options:\nA. Northeast\nB. Northwest\nC. Southwest\nD. Southeast.","oracle_answer":"Northeast","oracle_option":"A","oracle_full_answer":"A. Northeast"}
```

The same original scene also supplies these exact question/option suffixes:

```text
[row 500; spatialmap.tqa.2000.1]
Which object is in the Southwest of Ice Queen Ice Cream? Available options:
A. Coral Crafts
B. Narwhal's Novelties
C. Planetarium Prints
D. Police Supply Store.
oracle_answer = "Narwhal's Novelties"; oracle_option = "B"; oracle_full_answer = "B. Narwhal's Novelties"

[row 1000; spatialmap.tqa.2000.2]
How many objects are in the Southeast of Oz Oddities? Available options:
A. 1
B. 5
C. 3
D. 0.
oracle_answer = "0"; oracle_option = "D"; oracle_full_answer = "D. 0"
```

Sources: [row 500](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=500&length=1), [row 1000](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=1000&length=1).

### B. Scene 2002, Which With Multiple Explicit Qualifying Objects

Source: [row 502](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=502&length=1).

```json
{"id":"spatialmap.tqa.2002.1","text":"Consider a map with multiple objects: \nIris's Ice Skates is in the map.  Dolphin's Dresses is to the Southwest of Iris's Ice Skates.  Mordor Supplies is to the Northwest of Iris's Ice Skates. Mordor Supplies is to the Northeast of Dolphin's Dresses.  Art Supplies is to the Southwest of Iris's Ice Skates. Art Supplies is to the Southwest of Mordor Supplies.  Andy's Autos is to the Southeast of Mordor Supplies. Andy's Autos is to the Southeast of Iris's Ice Skates.  Narwhal's Novels is to the Northeast of Art Supplies. Narwhal's Novels is to the Northeast of Iris's Ice Skates. \n\n Please answer the following multiple-choice question based on the provided information. Which object is in the Southwest of Iris's Ice Skates? Available options:\nA. Art Supplies\nB. Mordor Supplies\nC. Andy's Autos\nD. Narwhal's Novels.","oracle_answer":"Art Supplies","oracle_option":"A","oracle_full_answer":"A. Art Supplies"}
```

**Direct observation:** Both Dolphin's Dresses and Art Supplies satisfy the wording by explicit premises. Only Art Supplies is in the menu. Unique names do not imply a scene-wide unique answer.

Additional original questions on this scene:

```text
[row 2; spatialmap.tqa.2002.0]
In which direction is Narwhal's Novels relative to Iris's Ice Skates? Available options:
A. Northwest
B. Northeast
C. Southeast
D. Southwest.
oracle_answer = "Northeast"; oracle_option = "B"; oracle_full_answer = "B. Northeast"

[row 1002; spatialmap.tqa.2002.2]
How many objects are in the Southeast of Mordor Supplies? Available options:
A. 0
B. 4
C. 2
D. 5.
oracle_answer = "2"; oracle_option = "C"; oracle_full_answer = "C. 2"
```

Sources: [row 2](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=2&length=1), [row 1002](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=1002&length=1).

### C. Scene 2003, Native Direction Label and Another Nonunique Which

Source: [row 3](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=3&length=1).

```json
{"id":"spatialmap.tqa.2003.0","text":"Consider a map with multiple objects: \nRecycle Center is in the map.  Sally's Salon is to the Southeast of Recycle Center.  Andy's Autos is to the Northwest of Sally's Salon. Andy's Autos is to the Southeast of Recycle Center.  Unicorn Umbrellas is to the Northwest of Sally's Salon. Unicorn Umbrellas is to the Southwest of Andy's Autos.  Nightingale Novelties is to the Southwest of Andy's Autos. Nightingale Novelties is to the Southwest of Sally's Salon.  Trail Hiking Gear is to the Southeast of Recycle Center. Trail Hiking Gear is to the Southwest of Sally's Salon. \n\n Please answer the following multiple-choice question based on the provided information. In which direction is Recycle Center relative to Nightingale Novelties? Available options:\nA. Southeast\nB. Southwest\nC. Northeast\nD. Northwest.","oracle_answer":"Northeast","oracle_option":"C","oracle_full_answer":"C. Northeast"}
```

```text
[row 503; spatialmap.tqa.2003.1]
Which object is in the Southwest of Andy's Autos? Available options:
A. Recycle Center
B. Trail Hiking Gear
C. Sally's Salon
D. Unicorn Umbrellas.
oracle_answer = "Unicorn Umbrellas"; oracle_option = "D"; oracle_full_answer = "D. Unicorn Umbrellas"

[row 1003; spatialmap.tqa.2003.2]
How many objects are in the Southeast of Nightingale Novelties? Available options:
A. 0
B. 1
C. 2
D. 5.
oracle_answer = "1"; oracle_option = "B"; oracle_full_answer = "B. 1"
```

Sources: [row 503](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=503&length=1), [row 1003](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=1003&length=1). **Direct observation:** Unicorn Umbrellas and Nightingale Novelties are both explicitly Southwest of Andy's Autos; the latter is absent from the which menu. The native direction row stores C, without any uncertainty metadata.

### D. Scene 2012, Cardinal Count

Source: [row 1012](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=1012&length=1).

```json
{"id":"spatialmap.tqa.2012.2","text":"Consider a map with multiple objects: \nHippopotamus's Hats is in the map.  Yacht Club is to the Southeast of Hippopotamus's Hats.  Warrior's Memorial is to the Southeast of Hippopotamus's Hats. Warrior's Memorial is to the Northeast of Yacht Club.  Koala's Kitchenware is to the Southeast of Hippopotamus's Hats. Koala's Kitchenware is to the Southwest of Yacht Club.  Stargazer Observatory is to the Northwest of Koala's Kitchenware. Stargazer Observatory is to the Southeast of Hippopotamus's Hats.  Fresh Foods is to the Southeast of Stargazer Observatory. Fresh Foods is to the Southeast of Hippopotamus's Hats. \n\n Please answer the following multiple-choice question based on the provided information. How many objects are in the North of Koala's Kitchenware? Available options:\nA. 3\nB. 1\nC. 0\nD. 2.","oracle_answer":"0","oracle_option":"C","oracle_full_answer":"C. 0"}
```

**Our interpretation, not an author definition:** This zero label is incompatible with interpreting “North” as the inclusive northern half-plane under ordinary diagonal inequalities, because Stargazer Observatory is explicitly Northwest of the anchor. Exclusive direction categories or axis-alignment rules could explain it, but neither is established by this row alone.

## Geometry, Distractors, and Scene Structure: Evidence Limits

**Explicitly documented:** Synthetic configurable maps; uniquely named locations; pairwise spatial prose; arbitrary positioning rather than Spatial-Grid's rigid cells; text intended to suffice for human answers; three questions per scene; four-choice accuracy. [P1-P3]

**Visible in the rows:** Scenes contain indirect paths and additional cross-relations, not just one ordered chain. Some queried relations are explicitly stated (e.g. 2000.0); others are not (e.g. 2003.0). Which menus select four scene objects and can omit other explicitly qualifying objects. Count menus select four numbers, not an exhaustive enumeration of all counts 0-5. These observations do not establish a distractor-generation algorithm or a guarantee that every non-reference option is logically impossible.

**Not established by inspected sources:** Exact coordinate distribution, integer versus continuous coordinates, step lengths, angle-sector versus coordinate-sign classification, equality/boundary/tie handling, co-location policy, formal closure rules, proof that each TQA reference is uniquely entailed, or whether labels were computed from coordinates versus another representation. The authors' sufficiency claim is not a formal semantics specification.

**Generator unavailable in inspected official release:** [README lines 138-140](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/README.md#L138-L140) says the dataset generation script “will be released in Feburary”; the inspected [recursive tree](https://api.github.com/repos/jiayuww/SpatialEval/git/trees/d82ba382805265f205693cb83a30c4866ea6c35b?recursive=1) contains inference/evaluation code and assets, but no scene generator. Do not infer a generation recipe from the wording.

No documented hard-negative strategy, disconnected distractor components, controlled contradiction injection, or required query hop count was found. The paper's random/noise-image experiments are **modality ablations**, not evidence of such textual-scene constructions: https://arxiv.org/html/2406.14852v2#S4.SS1 .

## Compatibility Conclusions (Our Assessment)

- Original scoring targets one supplied answer to the original four-option prompt. Unique/Listed/Exhaustive, set-valued labels, and E/F outputs are new evaluation contracts, not recovered original terminology.
- Preserve menu and answer scope when claiming native compatibility. Scene-wide uniqueness is demonstrably stronger than original which questions, even without a solver (examples B/C).
- Do not conflate direction-question menus (four diagonals), count query vocabulary (eight directions), and an inferred solver's relation universe. They are different facts; the sources specify no formal universe of possible completions.
- Keep original labels distinct from independently proved entailments. Source rows establish what was labeled, not that the chosen answer is forced in every geometric completion.
- Do not call the benchmark **closed-world** or **unit-step** on this evidence. A finite depicted map, incomplete pairwise prose, and a counting task do not by themselves establish either formal assumption.
- Preserve raw name strings and the original natural-language interface. No ASCII-only identifier contract is documented; punctuation and spaces are ordinary source data.
- Report scoring provenance explicitly: paper accuracy / README “exact match” versus the released heuristic-extraction-plus-substring implementation. A cleaner scorer changes the operational protocol even if it better matches the stated intent.
