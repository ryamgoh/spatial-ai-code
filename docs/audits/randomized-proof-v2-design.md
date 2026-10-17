# SpatialEntail randomized proof-graph v2

## Status

This is the active design. It preserves the completed SINGLE v1 results and
supersedes the fixed 30-cell ALL-POSSIBLE freeze candidate. No v2 benchmark is
called frozen until the structural calibration, certificate replay, quotas,
uniqueness, disjointness, and hashes pass.

## Controlled variables

- `G`: G1 for SINGLE; G2--G4 for ALL-POSSIBLE.
- `L`: maximum semantic-rule dependency length measured from replayed proofs.
- `X`: exact solver-verified removable atomic distractor count.
- ordered rule transitions: training-seen or composition-held-out.

Support size, proof size, branches, relational length, and token length are
recorded continuously. They are not cells or invented difficulty scores.

The semantic grammar is MP, IFF elimination, disjunctive elimination, case
split, double-negation elimination, and transitive relational composition.
Complete feasible rule paths are sampled uniformly using dynamic-programming
completion counts. Prompt premise order is shuffled independently from proof
order. The certificate, not generator intent, determines realised L.

Generation separates sequential depth from branching: a randomized prefix may
contain at most one case-split step, while branch count and proof size remain
reported diagnostics. Without this bound, repeated case splits create an
accidental exponential branch tree rather than a controlled L curve. Direction
and Selection use exclusion-first ALL-POSSIBLE skeletons: gold values remain
open-world possible and the proof derives necessary non-gold exclusions. Count
retains correlated assignment evidence with a five-candidate cap.

The construction pool reserves fresh leaf objects for X distractors and a small
proof-atom core, then removes unused names before solving. The known
possibility-skeleton offset avoids rebuilding a base certificate on every
rejected lift; the final complete certificate still decides admission.
Representative local L6/X6 timings fell from 4.62s to 0.84s for SINGLE
Direction, 14.12s to 1.42s for ALL Direction G4, 69.24s to 6.66s for ALL
Selection G4, and 5.28s to 3.37s for ALL Count G4.

A separate L8/X8 regression covers SINGLE Direction G1 plus ALL Direction,
Selection, and Count G4. Representative local times were 2.72s, 7.49s, 9.84s,
and 9.29s respectively; every case replayed at exact L8/X8.

## Calibration and feasibility correction

The structural pool targets 2,400 examples over three query families, four gold
sizes, and L1--L8. Infeasible low-L Count cells are not fabricated: complete
G2/G3/G4 Count certificates currently have audited minima L2/L3/L4 because
excluded counts require assignment refutations. Each query/gold group retains
200 calibration examples by reallocating uniformly over its feasible L range.

All 36 ordered rule pairs receive 20 seeded capability attempts. A transition
is eligible after 16 successful replays. Seed 42 partitions eligible edges 75%
seen and 25% held out while retaining an L8-capable seen graph.

## Frozen-candidate allocation

The candidate contains 3,600 fresh bases: 900 SINGLE and 2,700 ALL-POSSIBLE.
Every query/gold group contains:

| Partition | N | L | X | Transitions |
|---|---:|---|---|---|
| MATCHED | 120 | feasible L1--L5 | X0--X5 | seen |
| LENGTH-OOD | 60 | L6--L8 | X0--X5 | seen |
| NOISE-OOD | 60 | feasible L1--L5 | X6--X8 | seen |
| COMPOSITION-OOD | 60 | feasible L2--L5 | X0--X5 | exactly one held-out |

No headline item combines OOD factors. Within each allowed interval, quotas use
the predeclared maximum-entropy tilt `P(k) proportional to r^k`: benchmark
`r_L=1.15`, benchmark `r_X=1.10`, training `r_L=1.40`, and training `r_X=1.25`.
Largest-remainder rounding makes integer marginals deterministic.

## Contracts and diagnostics

Every spatial query has exactly five A--E options. ALL-POSSIBLE requires the
complete semantic set. VISIBLE-POSSIBLE uses the same problem, analysis, and
certificate but selects only visible possible values. `visible_gold_size`
controls V0 through VG; V0 reserves one of the five slots for None of the
Options.

VISIBLE is generator-capable but excluded from the first SFT study. A separate
300-example diagnostic covers direct L0, ambiguous SINGLE, true Selection
no-match, and zero-visible menus.

Inconsistent strict cycles are not mislabeled as no-answer spatial questions.
A separate binary diagnostic constructs 280 clean/dirty pairs over cycle
lengths C2--C8 and both axes. Adding one closing edge creates the dirty variant;
deleting any planted cycle edge must restore satisfiability. This produces 560
prompts and remains outside the headline score and SFT data.

## SFT and ablation

SINGLE-v2 and MULTI-v2 each use 8,000 train and 800 development bases. Training
contains only L1--L5, X0--X5, and seen transitions. MULTI is balanced at 2,000
examples for each G1--G4. Its exponential L tilt is normalized separately over
each gold size's feasible interval (G1: L1--L5, G2: L2--L5, G3: L3--L5, G4:
L4--L5). X0--X5 is likewise normalized within each gold size, where all X
values are feasible, giving identical conditional noise distributions. This
avoids a misleading U-shaped conditional L distribution that would
be forced by combining equal gold-size rows, structural zeros, and one global L
marginal. The same base rows yield Natural and answer-only
targets, trained at seeds 42 and 43, giving eight adapters in total.

Certificates remain offline audit artifacts and are never SFT targets. Natural
rows are never silently culled: use 24,576 context tokens when all fit, otherwise
32,768, and fail preparation above that bound.

## Commands

From the repository root on Slurm:

```sh
sbatch --partition=normal --cpus-per-task=128 --mem=128G \
  experiments/spatialentail-randomized-v2/slurm/prepare-calibration.sh
```

After the complete calibration receipt exists:

```sh
sbatch --partition=normal --cpus-per-task=128 --mem=128G \
  experiments/spatialentail-randomized-v2/slurm/prepare-benchmark.sh
```

Prepare non-headline diagnostics separately:

```sh
sbatch --partition=normal --cpus-per-task=16 --mem=32G \
  experiments/spatialentail-randomized-v2/slurm/prepare-diagnostics.sh
```
