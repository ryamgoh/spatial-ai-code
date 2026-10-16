# SpatialEntail-ALL-POSSIBLE-Bench: draft cell registry and feasibility pilot

Status: hardening, updated 2026-10-15. Not frozen or sealed. The first
calibration registry is `spatial/v2/all-possible-cells.yaml`; the separate
hardening registry is `spatial/v2/all-possible-hardening-cells.yaml`. The
generator is `spatial/v2/possibility.py`.

## Purpose

`SpatialEntail-SINGLE-Bench` was largely solved by the untuned base model on
CORE and COMP once answer selection was fair. Under `SINGLE`, every
underdetermined question collapses to one escape option, `Cannot be
determined`. `ALL_POSSIBLE` removes that escape: the model must return the
complete set of answers that could be true given the stated relations, and
leave out every impossible one. The answer space grows from one letter to any
non-empty subset of the menu.

This bench reuses the structures that made SINGLE's SHIFT section
discriminative (depth, Boolean rules, correlated Count, distractors), but every
cell plants two or more possible answers.

## Contract

- Answer mode `ALL_POSSIBLE`. Every menu shows exactly five options, matching
  SINGLE and the external five-option benchmarks, so option count never reveals
  how many answers are correct and transfer is not confounded by menu shape.
  For `ALL_POSSIBLE` the complete possible set is always among the five shown
  (so the full set is expressible); the gold set size is therefore capped at
  four, leaving at least one distractor. Partial menus, where some possible
  answers are omitted, are reserved for the `SpatialEntail-VISIBLE-POSSIBLE`
  probe.
- Prompt instruction: "Select every answer that could be true given the stated
  relations." The hardening prompt also makes the semantic contract explicit:
  unstated relations remain unknown rather than false, possibility means truth
  in at least one arrangement satisfying every statement, and direction words
  constrain qualitative coordinate order rather than equal unit distances.
- Primary metric: exact-set accuracy. Diagnostics: per-option precision and
  recall, over- and under-inclusion, and accuracy by gold set size.
- Stage-2 constrained decoding already supports 1–N answer subsets over A–E.

## Cell design

Cell IDs follow the existing grammar, with `ALT` as the semantic status.

| Suffix | Meaning |
|---|---|
| `G2`, `G4` | One Boolean rule plants a gold set of exactly 2 or 4 possible answers |
| `D3`, `D5` | One axis is left unordered along a 3- or 5-link chain |
| `NOISE` | A depth-3 chain plus support-attached removable distractors |
| `JCR` | Planted memberships move together, so intermediate counts are impossible |

Skeletons, each for Direction, Selection, and Count unless noted:

- `MP`: a true trigger implies a disjunction of planted answers.
- `IFF`: the same, through an equivalence.
- `DJE`: a disjunction of planted answers plus one alternative that a premise
  rules out.
- `CSR`: one object pair takes exactly one of k exact directions, and each case
  implies a different planted answer.
- `TRC` depth: a chain fixes one axis strictly and leaves the other open.
  Selection and Count use one open chain and one fixed chain.
- `NOISE` (Direction, Selection): the depth skeleton plus premises that hang
  spare objects off the support. They look relevant but are removable.
- `CNT-CSR-JCR` (Count): the planted group is all in or all out, so the
  possible counts are {0, k} rather than {0, ..., k}.

The draft has 34 cells: 24 Boolean (4 skeletons × 2 gold sizes × 3 query
kinds), 6 depth, 2 noise, and 2 joint-count.

Every menu holds exactly five options, so the gold set size cannot be read off
the menu. Direction shows the possible directions plus other directions as
distractors; Selection and Count pad with distractor candidates.

## Feasibility pilot

50 samples per cell, seed `31000 + cell index`, generated after the inert
tautology fix described in
[`certificate-search-bottleneck.md`](certificate-search-bottleneck.md). Every
sample was solved, matched its declared shape, built a replayable answer
certificate, and passed the prompt round trip.

- Accepted: 1,700 of 1,700 requested; every cell was accepted within its
  attempt budget.
- Every menu held exactly five options, and every gold set was a subset of the
  shown options.
- Unique structure signatures: 50 of 50 in every cell.
- Wall time: 1,443 s on one laptop CPU, about 0.85 s per row.
- Longest Natural trace: 15,810 characters, roughly 4K tokens, well inside the
  12,288-token training budget.

| Cell | Accepted | Gold set size | Menu | Median / max trace chars | Seconds |
|---|---:|---:|---:|---:|---:|
| `DIR-MP-ALT-G2` | 50/50 | 2 | 5 | 10,011 / 12,350 | 14.4 |
| `DIR-MP-ALT-G4` | 50/50 | 4 | 5 | 11,945 / 12,945 | 20.3 |
| `DIR-IFF-ALT-G2` | 50/50 | 2 | 5 | 10,034 / 11,169 | 14.3 |
| `DIR-IFF-ALT-G4` | 50/50 | 4 | 5 | 12,014 / 12,954 | 19.4 |
| `DIR-DJE-ALT-G2` | 50/50 | 2 | 5 | 10,134 / 11,076 | 13.0 |
| `DIR-DJE-ALT-G4` | 50/50 | 4 | 5 | 10,345 / 11,137 | 17.7 |
| `DIR-CSR-ALT-G2` | 50/50 | 2 | 5 | 11,409 / 12,404 | 13.3 |
| `DIR-CSR-ALT-G4` | 50/50 | 4 | 5 | 14,323 / 15,229 | 19.3 |
| `DIR-TRC-ALT-D3` | 50/50 | 3 | 5 | 6,798 / 7,681 | 18.0 |
| `DIR-TRC-ALT-D5` | 50/50 | 3 | 5 | 8,176 / 10,318 | 70.5 |
| `DIR-TRC-ALT-NOISE` | 50/50 | 3 | 5 | 6,657 / 7,599 | 90.6 |
| `SEL-MP-ALT-G2` | 50/50 | 2 | 5 | 4,117 / 4,492 | 35.8 |
| `SEL-MP-ALT-G4` | 50/50 | 4 | 5 | 4,506 / 4,977 | 42.2 |
| `SEL-IFF-ALT-G2` | 50/50 | 2 | 5 | 4,068 / 4,471 | 36.0 |
| `SEL-IFF-ALT-G4` | 50/50 | 4 | 5 | 4,456 / 5,055 | 43.7 |
| `SEL-DJE-ALT-G2` | 50/50 | 2 | 5 | 4,123 / 4,453 | 30.2 |
| `SEL-DJE-ALT-G4` | 50/50 | 4 | 5 | 4,498 / 5,095 | 37.4 |
| `SEL-CSR-ALT-G2` | 50/50 | 2 | 5 | 4,201 / 4,441 | 36.5 |
| `SEL-CSR-ALT-G4` | 50/50 | 4 | 5 | 4,666 / 5,090 | 43.6 |
| `SEL-TRC-ALT-D3` | 50/50 | 2 | 5 | 4,093 / 4,563 | 36.5 |
| `SEL-TRC-ALT-D5` | 50/50 | 2 | 5 | 4,963 / 5,696 | 107.7 |
| `SEL-TRC-ALT-NOISE` | 50/50 | 2 | 5 | 4,073 / 4,446 | 80.0 |
| `CNT-MP-ALT-G2` | 50/50 | 2 | 5 | 5,906 / 6,253 | 45.8 |
| `CNT-MP-ALT-G4` | 50/50 | 4 | 5 | 6,595 / 7,068 | 56.4 |
| `CNT-IFF-ALT-G2` | 50/50 | 2 | 5 | 5,908 / 6,250 | 43.8 |
| `CNT-IFF-ALT-G4` | 50/50 | 4 | 5 | 6,554 / 7,132 | 53.5 |
| `CNT-DJE-ALT-G2` | 50/50 | 2 | 5 | 5,915 / 6,495 | 37.0 |
| `CNT-DJE-ALT-G4` | 50/50 | 4 | 5 | 6,532 / 7,074 | 45.8 |
| `CNT-CSR-ALT-G2` | 50/50 | 2 | 5 | 6,212 / 6,538 | 43.3 |
| `CNT-CSR-ALT-G4` | 50/50 | 4 | 5 | 7,023 / 7,478 | 51.8 |
| `CNT-TRC-ALT-D3` | 50/50 | 2 | 5 | 4,711 / 5,122 | 45.8 |
| `CNT-TRC-ALT-D5` | 50/50 | 2 | 5 | 5,673 / 6,116 | 125.5 |
| `CNT-CSR-JCR-ALT-G2` | 50/50 | 2 | 5 | 7,216 / 7,908 | 25.7 |
| `CNT-CSR-JCR-ALT-G3` | 50/50 | 2 | 5 | 14,104 / 15,810 | 28.5 |

## Known limitations before freezing

1. **Gold set size is fixed per cell.** Each cell plants exactly one set size,
   for example `G4` is always size 4. That keeps cells interpretable, but a
   model could learn the size from structure. The bench should mix gold sizes
   across cells, and diagnostics must report accuracy by set size.
2. **Depth and noise cells plant small sets.** Direction depth cells always
   give 3 possible directions; Selection and Count depth cells always give 2.
   Larger open sets would need more open chains.
3. **The original Boolean cells directly expose alternatives.** A true trigger
   often implies one disjunction containing the complete query-level gold set.
   This is useful as a matched contract check, but it is not sufficient as a
   compositional benchmark cell.
4. **No disjointness audit yet.** Freezing requires canonical-structure
   disjointness from every `SpatialEntail-SINGLE` split, as in the SINGLE
   release.
5. **Boolean Direction traces are long.** They reach about 10–15K characters
   because every one of the eight directions needs a witness or a refutation,
   even though only five are shown in the menu.

## First model calibration and response audit

The 34-cell pool was prepared with 25 examples per cell and evaluated using the
same two-stage protocol as SINGLE. This is retained as engineering evidence,
not reported as a frozen benchmark score.

| Model | Exact set | Stage 1 exact | Stage 1 format | Gold covered | Missing gold |
|---|---:|---:|---:|---:|---:|
| Untuned Qwen3.5-4B | 93.65% | 65.65% | 0.00% | 94.94% | 5.06% |
| Natural SINGLE-SFT seed 42 | 14.59% | 10.71% | 68.00% | 15.88% | 84.12% |

The Natural adapter result is a contract-transfer failure dominated by omitted
gold answers, not formatting or indiscriminate over-selection. It does not show
that Natural supervision generally damages spatial reasoning: the adapter was
trained to emit one SINGLE answer and was evaluated zero-shot on complete-set
prediction.

The untuned response archive rules out a scorer or metadata leak: all 850
predictions rescore exactly, all base IDs and structure signatures are unique,
the inference argument contains the rendered prompt rather than oracle metadata,
and option letters are balanced. Stage 2 preserves all 558 native exact answers
and recovers 238 additional responses whose prose or compact footer contains the
right decision; it leaves 54 semantic errors (43 missing-gold and 11
over-inclusive).

The near-saturation is structural. Direct G2/G4 Boolean cells make up 600 of
850 examples and score 96.33% untuned. They commonly place the complete set of
possible query answers in one visible disjunction. Depth is more useful: D3
scores 85.33%, D5 scores 77.33%, and `DIR-TRC-ALT-D5` scores 52%. Count CSR
errors also expose a meaningful open-world distinction, while Direction depth
errors show that the qualitative, non-unit-distance contract must be explicit.

## Separate hardening registry

The original registry remains unchanged so the calibration is reproducible.
The 14-cell hardening registry has two sections:

- **COMP:** `case-chain` Selection cells put each possible candidate behind an
  exclusive case and a two- or three-link relational chain. No query-answer atom
  appears as an operand of a disjunction. `case-count` cells make each exclusive
  branch specify a complete five-candidate world with a different count, so the
  set of possible counts must be computed rather than read from one formula.
- **SHIFT:** certified Direction, Selection, and Count depth cells extend to D7
  and Direction to D9. The existing certificate builder supports these cells;
  branch-composed Direction was deliberately not admitted because its
  cross-case transitive refutations are outside the current proof-construction
  boundary.

Every one of the 14 cells generated a checked example with a replayable answer
certificate. The complete one-example-per-cell prepared-bundle integration test
passes in 55 seconds with 14 worker processes, a deterministic tokenizer stub,
and the hardening pilot's 20,000-token target / 32,768-token evaluation budget.
Real Qwen tokenizer admission remains a cluster-side check because the local
macOS environment does not install the Linux/CUDA evaluation dependency set.

The first 25-per-cell cluster preparation generated all 350 certified samples
but rejected a correlated Count row at admission because the writer treated its
checked Natural trace as though a provisional 20,000-token training limit were
already final. Benchmark evaluation does not consume that target: the loader
uses the prompt and solver-authored oracle metadata. The corrected writer keeps
both views. It preserves every full Natural row and certificate under
`source-natural/`, records overall and per-cell min/p50/p90/p95/p99/max token
statistics plus counts over 12K/16K/20K/24K/32K, and derives answer-only rows
under `data/` for evaluation admission. Natural rows are not silently culled;
their measured distribution informs the later MULTI-SFT context decision.

Generation is also cached before tokenizer measurement. The cache contract pins
the registry, seed, requested rows per cell, generator/solver/proof sources, and
every cached file hash. A rerun with the same contract reuses the certified rows;
a changed contract fails closed. Thus a tokenizer or packaging failure after a
long generation pass no longer discards the generated source pool.

The first cluster run also exposed a parallelism tail: one process handled all
25 samples for a cell, so the job used at most 14 CPUs and then waited for the
slowest composed cell for 2,175 seconds. Generation now treats each
`(cell, sample)` pair as an independently and deterministically seeded work
item. A 14-cell × 25-sample pool therefore exposes 350 process-pool tasks and
can use more workers than cells while restoring rows in stable cell/sample
order. The integration regression runs 28 samples with 28 workers and completes
in essentially the same wall time as the former 14-sample/14-worker path.

## Next steps

1. Prepare 25 examples per hardening cell and evaluate only the untuned model
   with stage-2 constrained decoding.
2. Audit the hardening responses and retain structures with clear headroom;
   scores are a saturation diagnostic, not a rule for selecting individual
   model failures.
3. Fix per-cell allocations, run the disjointness audit, and seal
   `SpatialEntail-ALL-POSSIBLE-Bench`.
4. Train and evaluate MULTI-SFT only after the benchmark contract and allocation
   are frozen. The completed SINGLE answer-only ablation is not repeated for this
   calibration.
5. Build the paired `SpatialEntail-VISIBLE-POSSIBLE` probe from the same bases
   with partial menus.

The engineering pool can be prepared as a detached one-hour Slurm job. The
script requests CPU and memory only, uses the cluster's default partition, and
sends END/FAIL/TIME_LIMIT mail to the configured NUS address:

```sh
sbatch experiments/spatialentail-allpossible-pilot/slurm/prepare.sh
```

Inspect the largest CPU count and memory available on one normal node. The
Python process pool is single-node; a 256-CPU user quota does not imply that one
node has 256 CPUs.

```sh
sinfo -N -h -p normal -o "%N %c %m %t" | sort -k2,2nr | head
```

Prepare the separate hardening registry with sample-level parallelism. On a
node offering at least 128 CPUs and 128 GiB, use:

```sh
sbatch --partition=normal --cpus-per-task=128 --mem=128G \
  --export=ALL,PILOT_REGISTRY=spatial/v2/all-possible-hardening-cells.yaml,PILOT_OUT=experiments/spatialentail-allpossible-pilot/prepared-hardening,SAMPLES_PER_CELL=25,MAX_NEW_TOKENS=20000,EVAL_MAX_TOKENS=32768 \
  experiments/spatialentail-allpossible-pilot/slurm/prepare.sh
```

If the largest eligible node has 64 CPUs, request 64 CPUs and about 64 GiB
instead. Requesting 256 CPUs from a one-node process-pool job will remain queued
or be rejected unless a single eligible node actually has that capacity.

## Pre-freeze uniqueness and targeted Count extension

The hardened untuned calibration scores 82.29% overall: COMP 92.00%, SHIFT
76.89%, Direction 53.33%, Selection 85.33%, and Count 96.00%. Direction D5/D7
at 60% and D9 at 40% provide sufficient headroom to freeze after the normal
disjointness checks. Selection G4 and D7 are also discriminative; its easier G2
and D5 cells remain useful anchors but should receive lower final allocation.
The existing Count cells are retained as sanity checks rather than the primary
Count discriminator.

Generation now enforces canonical-structure uniqueness globally. Rows are
scanned in stable cell/sample order; any repeated signature is deterministically
regenerated with a retry-specific seed. Cache loading rejects missing or repeated
signatures, and the receipt records how many slots required replacement. This
removes the three duplicates observed in the 350-example engineering pool before
any final benchmark is sealed.

A separate two-cell Count hardening registry lives at
`spatial/v2/all-possible-count-hardening-cells.yaml`. Its `factor-count`
skeleton uses two independent latent spatial facts, each equivalent to a
correlated candidate group. The possible counts are `{0, 2, 4}` for G3 and
`{0, 2, 3, 5}` for G4. No visible disjunction enumerates those counts, and the
groups must be combined to obtain the complete set. An attempted multi-hop
version was rejected because certificate construction was not operationally
feasible; the IFF-based version stays within the replay-tested proof boundary
and probes both cells in about two seconds locally.

Prepare only this 50-example Count calibration pool:

```sh
sbatch --partition=normal --cpus-per-task=50 --mem=50G \
  --export=ALL,PILOT_REGISTRY=spatial/v2/all-possible-count-hardening-cells.yaml,PILOT_OUT=experiments/spatialentail-allpossible-pilot/prepared-count-hardening,SAMPLES_PER_CELL=25,MAX_NEW_TOKENS=20000,EVAL_MAX_TOKENS=32768 \
  experiments/spatialentail-allpossible-pilot/slurm/prepare.sh
```

After preparation, submit only its untuned job. These two cells are merged into
the final allocation only if they create material Count headroom.

## Freeze-candidate matrix

The Count calibration scores 88% on correlated G3 and 100% on G4. G3 is kept as
the Count SHIFT discriminator; G4 is dropped rather than iterating toward a
Qwen-specific failure. The freeze-candidate registry is
`spatial/v2/all-possible-final-cells.yaml` with a uniform 100 fresh examples per
cell (2,200 total). The matrix is structural rather than selected only from
Qwen failures:

- COMP anchors: Selection G2/G3/G4 D2 and correlated Count G2/G3/G4;
- SHIFT Direction: every depth D5 through D9;
- SHIFT Selection: G2/G3/G4 D3 and every relational depth D5 through D7;
- SHIFT Count: every relational depth D5 through D7 and independent-factor
  G3/G4.

This gives five Direction, nine Selection, and eight Count cells. Easier and
saturated cells are retained as transparent anchors, while macro query-family
and slice results prevent the larger Selection/Count grids from hiding the
Direction result. The calibration examples themselves are not frozen.

Final generation enforces two boundaries in the generator: global canonical
uniqueness within the new pool and structural exclusion against every JSONL row
in the sealed SINGLE release. Exclusion paths are included in the generation
cache contract by signature count and hash, so changing the reference corpus
invalidates cache reuse.

Prepare the fresh freeze candidate on a 128-CPU node:

```sh
sbatch --partition=normal --cpus-per-task=128 --mem=128G \
  --export=ALL,PILOT_REGISTRY=spatial/v2/all-possible-final-cells.yaml,PILOT_OUT=experiments/spatialentail-allpossible-pilot/prepared-final-candidate,SAMPLES_PER_CELL=100,MAX_NEW_TOKENS=20000,EVAL_MAX_TOKENS=32768,PILOT_DISJOINT_FROM=results/spatialentail-study/release-final \
  experiments/spatialentail-allpossible-pilot/slurm/prepare.sh
```

The output is still called a freeze *candidate* until its 2,200 rows, cell
allocation, source/evaluation pairing, certificate presence, hashes, uniqueness,
and disjointness receipt are checked. Model evaluation begins only after that
validation is recorded and the candidate is sealed.

After preparation, submit only untuned:

```sh
sbatch experiments/spatialentail-allpossible-pilot/prepared-hardening/slurm/untuned.sh
```
