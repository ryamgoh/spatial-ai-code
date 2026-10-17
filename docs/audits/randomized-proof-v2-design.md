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

The sampled linear grammar is MP, IFF elimination, disjunctive elimination,
double-negation elimination, and transitive relational composition. Complete
feasible paths over these five transformations are sampled uniformly using
dynamic-programming completion counts. Prompt premise order is shuffled
independently from proof order. The certificate, not generator intent,
determines realised L.

Generation separates sequential transformations from branching. Case split
remains a semantic certificate rule and contributes to realised L, proof size,
and branch diagnostics, but it is not sampled as an interchangeable linear
token: its branches already contain implication steps. Direction and Selection
use exclusion-first ALL-POSSIBLE skeletons, where gold values remain open-world
possible and the proof derives necessary non-gold exclusions. Count retains
correlated assignment evidence with a five-candidate cap.

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
Calibration is disposable engineering evidence, so repeated canonical
structures are counted in its receipt rather than causing the completed pool to
be discarded. The fresh headline benchmark remains strictly unique: duplicate
slots are deterministically regenerated until unique or the bounded replacement
budget fails.

All 25 ordered linear-rule pairs receive 20 seeded capability attempts. A transition
is eligible after 16 successful exact-path replays; failing pairs are filtered
rather than treated as a failed calibration. Calibration fails only when a
query family retains fewer than six reliable edges. Seed 42 subsequently
partitions eligible edges 75% seen and 25% held out while retaining an
L8-capable seen graph.

### Rejected first full calibration

The first 2,400-row L6/X6 calibration completed on 17 October 2026 but is
retained only as engineering evidence. All rows matched their requested L and
X, used five-option menus, passed prompt round trips, and contained no sampled
CSR steps. The transition probe found the same 23 eligible edges for all three
query families; every eligible edge succeeded in 20/20 attempts, while the two
typed exclusions `TRC->DJE` and `TRC->DN` failed in 0/20 attempts as expected.
At 128 workers, the recorded row-generation phase took 155.8 seconds. The pool
accepted 2,400 rows from 3,052 candidate trials (78.64% aggregate acceptance),
with 2,000 rows accepted on their first attempt, a 95th-percentile rejection
count of two, and a maximum of eleven.

The audit nevertheless found a release-blocking gold-set preservation defect:
28 rows (1.17%) had fewer solved answers than their declared G. Selection
accounted for 24 rows: five declared G2 rows realised G1, ten declared G3 rows
realised G1 or G2, and nine declared G4 rows realised G2 or G3. Count accounted
for four rows: two G3 rows realised G2 and two G4 rows realised G3. A pinned
Selection G2/L2/X0 seed reproduced the fault deterministically: the base
exclusion skeleton allowed both High School and University, but the lifted
random proof prefix introduced an additional relation involving High School
and left only University possible. Admission checked only the broad ambiguous
shape, not equality between the solved answer-set size and requested G.

The same audit found 2,395 unique canonical structures. The five surplus
duplicates occurred across four small Count collision groups at their minimum
feasible L and X0; duplicates are permitted in disposable calibration but not
in the headline candidate. Natural-target whitespace counts had median 771,
p95 1,080, and maximum 1,465; complete-message whitespace counts had median
1,004 and maximum 1,746. These are lexical proxies, not pinned-tokenizer
measurements.

This calibration must not seed the headline benchmark or SFT release. The
replacement must isolate proof-prefix auxiliaries from query candidates and
enforce `realised gold-set size == requested G` after solving. A regression test
must retain the failing seed, and a multi-seed property check must cover
Selection and Count before the 2,400 rows are regenerated.

SpatialEntail uses a bounded propositional fragment whose basic propositions
are concrete pairwise spatial relations. It supplements the Boolean rules with
checked operations for decomposing, inverting, composing, and recombining
spatial directions. This is not a claim of complete coverage of classical
propositional logic. The formula language supports NOT, AND, OR, implication,
and equivalence; the checker supports double-negation elimination, disjunctive
syllogism, contradiction, and scoped refutation. It does not provide every
associated inference or equivalence, such as reusable negation introduction,
double-negation introduction, general Boolean normalization, or
complement-aware disjunct elimination.

This boundary creates a specific typing limitation. TRC derives an exact
positive spatial atom, whereas the current DJE primitive consumes an explicit
disjunction and negated alternative, and DN consumes an explicitly
double-negated formula. Consequently, `TRC->DJE` and `TRC->DN` are not direct
edges in this constructor. This does not prevent Boolean and spatial reasoning
from co-occurring: `DJE->TRC`, `DN->TRC`, and longer paths such as
`TRC->MP->DJE` remain available. Extending the calculus is valid future work,
but complete classical propositional coverage is outside this FYP's scope.

## Frozen-candidate allocation

The candidate contains 3,600 fresh bases: 900 SINGLE and 2,700 ALL-POSSIBLE.
Every query/gold group contains:

| Partition | N | L | X | Transitions |
|---|---:|---|---|---|
| MATCHED | 120 | feasible L1--L6 | X0--X6 | seen |
| LENGTH-OOD | 60 | L7--L8 | X0--X6 | seen |
| NOISE-OOD | 60 | feasible L1--L6 | X7--X8 | seen |
| COMPOSITION-OOD | 60 | feasible L2--L6 | X0--X6 | exactly one held-out |

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
contains only L1--L6, X0--X6, and seen transitions. MULTI is balanced at 2,000
examples for each G1--G4. Its exponential L tilt is normalized separately over
each gold size's feasible interval (G1: L1--L6, G2: L2--L6, G3: L3--L6, G4:
L4--L6). X0--X6 is likewise normalized within each gold size, where all X
values are feasible, giving identical conditional noise distributions. This
avoids a misleading U-shaped conditional L distribution that would
be forced by combining equal gold-size rows, structural zeros, and one global L
marginal. SINGLE-v2 retains matched Natural and answer-only targets as the
supervision ablation. MULTI-v2 uses Natural targets only and is not treated as
a second supervision-format ablation; its comparators are the prompt-matched
untuned model and Natural SINGLE-SFT transfer baseline. Both contracts are
trained at seeds 42 and 43, giving six adapters in total.

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
