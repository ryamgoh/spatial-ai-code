# What each question type looks like

All items are **5-way MCQ**. A–D are content (directions, objects, or counts). The fifth slot is a special: **Cannot be determined** or **None of the Options** (wording may be paraphrased; `--shuffle-special` can put it on any letter).

Gold is whatever `SpatialSolver` proves. Examples below are from `data/spatial_sft_data_v6_6k_train.jsonl`.

---

## Shared fifth-slot rule

If the theorem is **already on A–D**, the fifth option is only a **distractor** (either special phrase, both Out).

The fifth option is **gold** only when:

| Situation | Fifth option |
|---|---|
| No compound is a theorem (missing edges, cycle, SE listed without SW) | **Cannot be determined** |
| A theorem exists but is not listed | **None of the Options** |

Type 1 / Type 2 never use Cannot be determined as gold. A count is always an integer (including 0). “Nobody listed is in that direction” is None of the Options, not undetermined.

---

## Type 0 — `In which direction is X relative to Y?`

A compound (NE/NW/SE/SW) is a theorem only if **both** axes are proven. `<` on X is west→east; on Y is south→north.

### 1-ans — both axes proven

Bakery is west and north of Gas Station → **Northwest**.

```
Question: In which direction is the Bakery relative to the Gas Station?
Options: A. North, B. Southeast, C. Cannot be determined, D. Northeast, E. Northwest

X-axis: Bakery < Gas Station → West.
Y-axis: Gas Station < Bakery → North.
Derived: Northwest (both axes proven).
E. Northwest — In.
C. Cannot be determined — Out (an axis was derived).
Answer: E
```

### 2-ans — one axis proven

East is proven; North/South is not. Live pair: **Northeast or Southeast**. Both are listed.

```
Question: In which direction is the Zoo relative to the Bakery?
Options: A. Northeast, B. South, C. None of these options, D. Southeast, E. Southwest

X-axis: Bakery < Zoo → East.
Y-axis: neither … → North/South unknown.
Derived: East proven → must be Northeast or Southeast.
A. Northeast — In.
D. Southeast — In.
C. None of these options — Out (the pair is listed).
Answer: A, D
```

### Undetermined — neither axis proven

High School vs Church: no `<` either way.

```
Question: In which direction is the High School relative to the Church?
Options: A. Cannot be determined, B. Northeast, C. Southeast, D. Northwest, E. Southwest

X-axis: neither Church < High School nor High School < Church.
Y-axis: neither …
Derived: none (both axes unknown).
A. Cannot be determined — In.
Answer: A
```

Missing edges. Not a fight, just no path.

### Cycle — both directions exist (contradiction)

Post Office SE of Hospital **and** Hospital SE of Post Office.

```
Question: In which direction is the Hospital relative to the Post Office?
Options: A. Northwest, B. Cannot be determined, C. Southwest, D. Northeast, E. Southeast

X-State: {Hospital, Post Office}
X-axis: both Post Office < Hospital and Hospital < Post Office (contradiction).
B. Cannot be determined — In.
Answer: B
```

Different from undetermined: here **both** `<` edges exist.

### Incomplete pair — one axis proven, only one of the two live compounds listed

South is proven → must be SE **or** SW. Only Southeast is on the menu.

```
Question: In which direction is the Bakery relative to the Library?
Options: A. East, B. Northwest, C. Cannot be determined, D. Southeast, E. North

Y-axis: Bakery < Library → South.
Derived: South proven, East/West unknown → Southeast or Southwest.
C. Cannot be determined — only one of the two remaining compounds is listed. In.
D. Southeast — remains possible, but Out (cannot pick SE alone).
Answer: C
```

Picking D would invent East.

### Live dir omitted — a theorem exists, not on A–D

Church **is** Southeast of Cinema. Southeast is not listed.

```
Question: In which direction is the Church relative to the Cinema?
Options: A. Northwest, B. East, C. No listed option is correct, D. North, E. Northeast

Derived: Southeast (both axes proven).
C. No listed option is correct — the proven direction is not listed. In.
Answer: C
```

This is **None of the Options** (paraphrase). Not “cannot determine.”

---

## Type 1 — `Which object is in the [Dir] of X?`

A–D are **object names**. Gold is every listed name that is **proven** on every required axis (0–4). Fifth option is None of the Options (or a distractor phrase if N≥1).

### 1 proven listed

```
Question: Which object is in the East of the Shopping Mall?
Options: A. Hospital, B. Not among the options, C. Pharmacy, D. University, E. Supermarket

East of Shopping Mall: Hospital.
A. Hospital — In.
B. Not among the options — Out.
Answer: A
```

### 2 proven listed

```
Question: Which object is in the South of the Park?
Options: A. Cinema, B. Coffee Shop, C. Bank, D. Post Office, E. Cannot be determined

South of Park: Bank, Coffee Shop.
B. Coffee Shop — In.
C. Bank — In.
E. Cannot be determined — Out (entities are proven).
Answer: B, C
```

(E here is a **distractor**; gold is already B,C.)

### 3 proven listed

```
Question: Which object is in the North of the Shopping Mall?
Options: A. Library, B. Supermarket, C. Post Office, D. No listed option is correct, E. High School

North: Library, Post Office, Supermarket.
A, B, C — In.
D. No listed option is correct — Out.
Answer: A, B, C
```

### 4 proven listed

All four names on A–D are theorems. None of the Options is Out.

```
Question: Which object is in the North of the Fire Department?
Options: A. Pharmacy, B. Museum, C. Post Office, D. Not among the options, E. Cinema

A, B, C, E — In (Cinema is letter E; the special is D).
Answer: A, B, C, E
```

This is **not** the old fake “mark everyone because nothing was proven.”

### 0 proven listed → None of the Options

Gas Station is actually Northwest of Bakery, but it is not on A–D.

```
Question: Which object is in the Northwest of the Bakery?
Options: A. Pharmacy, B. Fire Department, C. Zoo, D. None of the given options, E. Police Station

Intersection: Gas Station (not listed).
D. None of the given options — no listed entity is proven. In.
Answer: D
```

---

## Type 2 — `How many objects are in the [Dir] of X?`

One theorem: `|definite set|`, always an integer, including **0**. Never 1–N letters. Never Cannot be determined as gold.

### Count listed

Count is 0, and 0 is option B.

```
Question: How many objects are in the Northwest of the Bakery?
Options: A. 4, B. 0, C. 3, D. 2, E. Cannot be determined

Count: |intersection| = 0.
B. 0 — In.
E. Cannot be determined — Out (distractor).
Answer: B
```

0 on the paper is a real answer, not E.

### Count omitted → None of the Options

True count is 0; A–D are 6, 4, 1, 7.

```
Question: How many objects are in the Southwest of the Shopping Mall?
Options: A. 6, B. 4, C. 1, D. 7, E. Not among the options

Count: 0, not among A–D.
E. Not among the options — In.
Answer: E
```

---

## 6k mix (`spatial_sft_data_v6_6k_*.jsonl`)

2,000 per type, 4,800 train / 1,200 test, specials shuffled.

| Type 0 | n | Type 1 | n | Type 2 | n |
|---|---|---|---|---|---|
| 1-ans | 600 | 1 listed | 600 | count listed | 1,600 |
| 2-ans | 500 | 2 listed | 500 | count omitted | 400 |
| undetermined | 300 | 3 listed | 300 | | |
| cycle | 200 | 4 listed | 200 | | |
| incomplete pair | 200 | none listed | 400 | | |
| live dir omitted | 200 | | | | |

Solver: `spatial/spatial_solver.py`. Generator: `spatial/generate_all_v6.py`. CoT format: `docs/symbolic-cot.md`.
