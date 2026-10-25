# Randomized benchmark generation: allocation and pipeline review

Status: deferred future-work engineering review, 2026-10-18. This note reviews
a possible post-FYP hardening path for SpatialEntail randomized-v2. The current
FYP deliberately retains the smaller deterministic cap/retry workflow so that
experiments can proceed. This note does not freeze a benchmark, change an
allocation, or establish a population sampling claim.

## Verdict

Use a systematic preflight and staged generation pipeline, but right-size it
for an FYP.

The research workflow is sound in outline: calibrate the constructive grammar,
freeze the capability and holdout contract, allocate a fresh headline suite,
generate deterministically, replay certificates, enforce uniqueness and
disjointness, and publish only after a whole-artifact audit. The active design
document already states most of these gates. The missing part is operational:
the current randomized-v2 writer keeps a complete run in memory, writes directly
to the final output directory, and records receipts that do not hash their own
dataset file. A cancelled or partially written run is therefore neither safely
resumable nor safely publishable.

This is a sound synthesis of benchmark practice, not a named standard copied
from one framework. HELM selects scenarios using coverage and feasibility rather
than attempting every design-space cross-product
([paper](https://doi.org/10.48550/arXiv.2211.09110)) and supports dry runs that
construct requests without model calls
([official documentation](https://github.com/stanford-crfm/helm/blob/63754d05db6f874e41a395880fb573890a13e791/docs/benchmark.md#L9-L27)).
BIG-bench combines executable correctness checks, inspection, and review before
task inclusion
([official criteria](https://github.com/google/BIG-bench/blob/092b196c1f8f14a54bbc62f24759d43bde46dd3b/docs/doc.md#L53-L71)).
The lm-evaluation-harness task guide treats the task definition plus code commit
as the reproducibility boundary
([official guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/docs/task_guide.md#L1-L7)).
SpatialEntail's proposed gates are consistent with these practices, but its
particular sequence and constraints remain project-specific.

The appropriate fix is small:

1. freeze canonical identity and exclusions, then establish usable capacity;
2. solve and freeze the complete integer allocation and ordered work plan;
3. generate deterministic, checksum-bound chunks in a work directory;
4. assemble into a sibling staging directory;
5. reread and validate the persisted candidate; and
6. publish it only after the manifest, hashes, and audit agree.

Use the existing Z3 dependency, or an equivalently small deterministic
transportation solver, for allocation. Do not introduce Beam, Snakemake,
Nextflow, DVC, a database, or a general workflow platform. Those systems provide
useful design precedents, but their runtimes solve a larger problem than this
single-node, deterministic batch generator has.

The observed four-structure result for minimum-length, zero-distractor
ALL-POSSIBLE Count is an **empirical-operational cap**, not a proved exact upper
bound. One hundred additional seeds returning four canonical structures does
not exhaust the generator's state space. A conservative cap of four is valid for
making the present run feasible, but the documentation and tests must not call
the capacity exactly four unless exhaustive enumeration or a mathematical proof
establishes that result.

## What benchmark practice supports

Synthetic reasoning benchmarks commonly use constructive generation,
oversampling, rejection, exact balancing, and structural holdouts. They do not
require every requested cell to be produced by unbounded random retry.

| Precedent | Primary-source practice | Lesson for SpatialEntail |
|---|---|---|
| RuleTaker | It generates a formal theory, computes its complete forward consequences and proof depths, then selects questions. Depth uses generate-and-test; questions are balanced true/false, and its 70/10/20 split has no theory overlap. [Paper, §§3.3–3.4 and Appendix A.5](https://arxiv.org/pdf/2002.05867); [official reimplementation](https://github.com/allenai/ruletaker/blob/abaacec9364992eff5ec4555b837e20fee2f2ff0/theory_generator.py#L261-L398) | It is legitimate to reject infeasible candidates and balance after semantic analysis. Identity and split separation belong at the theory/structure level, not only the rendered-row level. |
| ProofWriter | It extends theory-first generation with proof DAGs and open-world `Unknown`. For abduction it exhaustively enumerates the finite formal query/missing-fact pairs and checks each with a theorem prover. [ACL paper, §§3.8 and 4](https://aclanthology.org/2021.findings-acl.317.pdf); [official dataset](https://allenai.org/data/proofwriter) | When a finite support space is tractable, enumerate and check it. Repeated random search is not intrinsically more principled. |
| PrOntoQA | It builds a linear ontology, constructs a unique proof of a requested hop length, and only then renders context, query, chain of thought, and label. The implementation rejects wrong-length constructions and retries. [Paper, §3](https://arxiv.org/pdf/2210.01240); [exact-length check](https://github.com/asaparov/prontoqa/blob/066f73fc70bfaa92ed2476fd3c41b4951ad6e63a/run_experiment.py#L315-L435); [retry loop](https://github.com/asaparov/prontoqa/blob/066f73fc70bfaa92ed2476fd3c41b4951ad6e63a/run_experiment.py#L1003-L1028) | Construct toward the requested structural contract, then verify the realized proof. A failed retry budget should report infeasibility or insufficient yield rather than silently relax the cell. |
| CLUTRR | It samples and rejects until it has enough stories, filters for exact relation-chain length, and can downsample every relation-pattern type to the minimum pattern count. It also holds out paraphrases and logical clauses. It does not claim canonical story-level deduplication. [ACL paper](https://aclanthology.org/D19-1458.pdf); [official generator](https://github.com/facebookresearch/clutrr/blob/d045fae289d3746503677ceed7631c999202501e/clutrr/generator.py#L65-L172); [pattern balancing](https://github.com/facebookresearch/clutrr/blob/d045fae289d3746503677ceed7631c999202501e/clutrr/relations/builder.py#L262-L276) | Oversample/reject/downsample is normal benchmark engineering. Hold out structural templates explicitly and keep the final allocation separate from the order in which the generator happened to succeed. |
| StepGame | It derives an analytic path-instance capacity and controls nominal hop bins. Its separately generated splits have no dataset-level seen-set/hash check; the paper reports 1.09% train–test overlap, all from `k=1`, and 0% after removing `k=1`. [AAAI paper](https://cdn.aaai.org/ojs/21383/21383-13-25396-1-2-20220628.pdf); [official split loops](https://github.com/ZhengxiangShi/StepGame/blob/5e6aff1563a6d7f46ee1f1aeff98b94e68c29005/Code/parameterized_step_game_8relation.py#L213-L322) | Analytic raw capacity does not imply canonical unique capacity after rendering and isomorphism. SpatialEntail's stricter global uniqueness gate is useful, but its capacity claims need evidence at the same canonicalization boundary. |
| SODA | SPOD-143k starts from task seeds, abstracts templates, expands them with scripts, and uses LLM rewriting. It generates batches of 1,000, inspects a 5% sample, and retains a batch at at least 98% compliance. The paper does not report an equivalent canonical deduplication or train/benchmark non-overlap audit. [ACL paper, §§3.2.2–3.2.3](https://aclanthology.org/2026.acl-long.1382.pdf) | Sampled language QC is useful but cannot replace exact semantic replay, identity accounting, or deduplication for a headline benchmark. |

These precedents support the proposal's synthesis: constructive generation plus
post-construction semantic validation, exact quota accounting, and structural
holdouts. They do not support claiming that fixed marginals make the resulting
benchmark representative of a natural population. The L, X, query-family,
gold-size, and partition marginals are an explicit experimental design. They
support controlled comparisons under that design, not ecological frequency or
population prevalence claims.

## Capacity and allocation

### Capacity must precede allocation

The rejected headline attempts expose two distinct feasibility constraints:

- semantic feasibility: a requested proof length must leave enough steps for
  the required held-out transition; and
- identity feasibility: a joint cell must contain at least as many distinct
  canonical structures as its allocated row count.

Both constraints should be computed or conservatively estimated before any
3,600-row generation job starts. The canonical identity function and the
cross-artifact exclusion set must be frozen first: raw generator yield can
overstate capacity if candidates later collapse to one canonical signature or
overlap calibration, training, development, SINGLE, or inspected engineering
pools. A work plan is valid only if every requested joint cell respects known
semantic minima and usable post-canonicalization capacity.

Capacity metadata should distinguish its evidence:

```json
{
  "cell": "count/g4/matched/l4/x0",
  "cap": 4,
  "basis": "empirical-operational",
  "evidence": {
    "search_seeds": 100,
    "canonicalizer": "visible-problem-structure-v1",
    "generator_revision": "<source fingerprint>"
  }
}
```

Use `basis: "proved"` only for an analytic argument or exhaustive enumeration
over a completely specified finite construction space. Until then, change the
active design's “exactly four canonical structures” wording to “an operational
cap of four, supported by the bounded sweep.” An operational cap may be lower
than the true capacity; that conservatism is acceptable when it preserves the
study marginals and prevents another doomed run.

### Use a capacitated transportation model

Pairing exact L and X marginals is an integer transportation problem. Let
`n[l,x]` be the number of rows assigned to joint cell `(l,x)`. Solve:

```text
n[l,x] >= 0 and integer
sum_x n[l,x] = required_L[l]
sum_l n[l,x] = required_X[x]
n[l,x] <= capacity[l,x] where a cap is known
```

This is a small, principled use of Z3 integer constraints. The official Z3
guide supports lexicographic and Pareto optimization over arithmetic objectives
([arithmetic optimization](https://microsoft.github.io/z3guide/docs/optimization/arithmeticaloptimization/),
[combining objectives](https://microsoft.github.io/z3guide/docs/optimization/combiningobjectives/)).
Feasibility alone is insufficient because many joint tables may satisfy the same
marginals. Add an explicit deterministic objective, for example:

1. minimize total scaled absolute deviation from the uncapped independent table
   `required_L[l] * required_X[x] / N`;
2. minimize the maximum cell deviation; and
3. break all remaining ties lexicographically over a documented `(l,x)` order.

Then persist the **entire selected joint table**. Do not recompute an arbitrary
satisfying Z3 model during resume. Hash the table into the run contract and
derive the ordered row work items from it. This makes the allocation auditable,
stable across restarts, and independent of solver model enumeration order.

The current retry-and-shuffle pairing in `pair_marginal_quotas()` is seeded and
does preserve the two marginals when it succeeds. It is therefore a reasonable
prototype, but it does not explain which feasible joint table it prefers, cannot
prove infeasibility, and records only marginal rules rather than the final joint
allocation. Replace it for the frozen run; do not build a general allocation
framework.

## Pipeline reliability principles

The relevant workflow systems converge on a few invariants:

- **Resume from identity plus verified output, not existence.** Nextflow hashes
  task identity, inputs, script, referenced values, and execution environment,
  and reuses a task only when its cache entry and outputs remain valid
  ([caching and resuming](https://nextflow.io/docs/latest/cache-and-resume.html)).
- **Make finalization atomic or idempotent.** Beam recommends atomic sink
  finalization, or idempotence when atomicity is unavailable, because finalize
  may be retried. Its file sinks write temporary bundle files before moving them
  to final names ([I/O base](https://beam.apache.org/releases/pydoc/current/apache_beam.io.iobase.html),
  [FileBasedSink](https://beam.apache.org/releases/javadoc/current/org/apache/beam/sdk/io/FileBasedSink.html)).
- **Bind transformations as well as inputs.** Hugging Face Datasets derives a
  transformed dataset fingerprint from the previous fingerprint plus the
  transform and its arguments ([cache and fingerprints](https://huggingface.co/docs/datasets/about_cache)).
- **Use content hashes to identify immutable artifacts.** DVC's cache is
  content-addressed, and its metadata records artifact hashes and paths
  ([internal files](https://dvc.org/doc/user-guide/project-structure/internal-files),
  [DVC files](https://dvc.org/doc/user-guide/project-structure/dvc-files)).
  SpatialEntail needs the manifest-and-SHA-256 principle, not DVC itself.
- **Validate before declaring success.** Snakemake's `ensure()` checks output
  properties such as non-emptiness and checksum compliance before a job is
  successful ([output properties](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#ensuring-output-file-properties-like-non-emptyness-or-checksum-compliance)).
- **Fail early.** Snakemake supports JSON-Schema configuration validation and
  dry runs; Dataflow validates access, permissions, staging, and sources/sinks
  before launching a job
  ([Snakemake validation](https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html#validation),
  [Dataflow preflight](https://cloud.google.com/dataflow/docs/guides/deploying-a-pipeline)).
- **Publish dataset checksums.** MLCommons Croissant recommends SHA-256 for
  versioned dataset file objects
  ([Croissant specification](https://docs.mlcommons.org/croissant/docs/croissant-spec.html#checksums)).

Dataflow's exactly-once documentation also makes an important boundary clear:
records may be retried while committed pipeline results appear once, but
arbitrary external side effects do not automatically receive exactly-once
semantics ([Dataflow exactly once](https://cloud.google.com/dataflow/docs/concepts/exactly-once)).
For this batch generator, deterministic recomputation plus delayed publication
is simpler than emulating a distributed exactly-once protocol.

## Recommended SpatialEntail pipeline

```text
validate inputs/environment; freeze identity and exclusions
        |
        v
estimate or prove usable post-canonicalization capacity
        |
        v
solve and freeze joint allocation + ordered work plan
        |
        v
generate deterministic chunks into <output>.work/
        |
        v
validate each chunk and record its SHA-256
        |
        v
assemble <output>.staging-<run-key>/
        |
        v
reread, replay, audit quotas/identity/disjointness/hashes
        |
        v
rename staging directory to final candidate and mark complete
```

### 1. Preflight

Before starting the transition probe or row workers:

- validate the study schema and all numeric ranges;
- freeze the canonical identity version and every exclusion input before
  capacity estimation;
- validate that calibration is complete and that every transition probe entry
  meets its declared threshold, rather than checking only that the probe is a
  dictionary;
- calculate semantic length minima and load capacity records with provenance;
- solve the full joint allocation and fail if it is infeasible;
- materialize the ordered work plan, including base seed and deterministic
  replacement-seed rule;
- verify input artifacts, disjointness sources, source files, dependency lock,
  and writable work/final locations;
- refuse a conflicting final output, but recognize a matching work directory;
  and
- compute a `run_key` over canonical JSON containing schema version, study
  design, full joint allocation, ordered work plan, seeds, input hashes, and
  generator/solver/canonicalizer source fingerprints.

Keep wall-clock time, hostnames, and worker counts in a run receipt, not in the
artifact identity. They describe execution, not dataset content.

### 2. Chunked generation and resume

Partition the frozen work plan into deterministic chunks of roughly 25--100
rows. Each chunk should contain its run key, index range, rows, row count, and
SHA-256. Generate to a temporary sibling file, reread and validate it, then
rename it to the stable chunk path. On resume:

- reuse only chunks whose run key, range, count, and hash agree;
- reject work from a changed design, allocation, input, or implementation;
- regenerate missing or corrupt chunks; and
- restore rows by work-plan index, never worker completion order.

Checkpointing every row would create unnecessary filesystem and bookkeeping
cost. Checkpointing only after all 2,400 or 3,600 rows repeats too much work.
Small deterministic chunks are the appropriate middle ground.

Global canonical uniqueness cannot be finalized independently inside each
chunk. After assembly, scan rows in frozen work-plan order. Regenerate duplicate
slots with the documented replacement-seed sequence, writing replacement chunks
or an auditable replacement overlay. Re-run the global scan until unique or the
bounded budget fails. Never relax the canonicalizer or silently reduce a quota.

### 3. Staged validation

Generation-time assertions are necessary but not sufficient. Reread the bytes
that will actually be published and independently check:

- JSON/schema validity and expected file/row counts;
- every requested and realized query, G, L, X, and partition field;
- prompt round trip, solver answer, certificate replay, and process-validity
  contract;
- exact joint allocation and declared marginals;
- base IDs, prompt hashes, canonical structure uniqueness, and disjointness from
  calibration, prior releases, training, development, and other benchmark
  roles;
- seen versus held-out transition rules;
- answer/menu and query-family distributions; and
- byte size and SHA-256 of every published file.

This should reuse the fail-closed style of `spatial.v2.study.seal`, which rereads
the persisted SINGLE release and verifies more than a checksum. Updating a
checksum after corrupting or semantically changing a row must not make it
sealable.

### 4. Manifest and publication

The artifact manifest should include at least:

- schema and artifact status (`candidate`, not `frozen`);
- run key and exact work-plan/allocation hashes;
- seeds and deterministic replacement policy;
- all capacity records and their `proved` or `empirical-operational` basis;
- calibration, disjointness-input, code, dependency-lock, solver, and
  canonicalizer fingerprints;
- per-file path, byte size, row count, and SHA-256;
- quota and realized-distribution summaries;
- validation/audit receipt hash; and
- software revision plus a dirty-diff hash when applicable.

Build this under a sibling staging directory. After the whole-artifact audit
passes, write the manifest there and rename the directory to the final candidate
path on the same filesystem. The manifest status or a final `COMPLETED` marker
is the publication boundary. Directory rename is a practical atomic publication
mechanism on the intended local filesystem; do not claim it is atomic on every
object store or network backend.

## Current documentation and code corrections

The following corrections are needed before the active proposal and
implementation agree:

1. **Downgrade the capacity claim.**
   `docs/archive/audits/randomized-proof-v2-design.md` currently says the 100-seed sweep
   found a cell “with capacity four” and later says it has “exactly four
   canonical structures.” Replace both with an empirical-operational cap unless
   exhaustive enumeration proves the upper bound.
2. **Rename the capacity test.**
   `test_minimum_count_cell_declares_canonical_capacity` tests only that the
   configured cap equals four. It does not prove canonical capacity. Name and
   metadata should say `operational_cap` and record its evidence basis.
3. **Freeze the joint allocation.**
   `pair_marginal_quotas()` currently retries seeded shuffles until caps hold.
   Replace the frozen-run path with a deterministic capacitated allocation and
   persist the full joint table and ordered work plan.
4. **Make active randomized-v2 resumable.**
   `prepare_calibration()` and `prepare_headline_benchmark()` hold all rows in
   memory and reject any existing output directory. Add contract-bound chunk
   checkpoints and resume tests. The cache described in the older
   `all-possible-bench-draft.md` belongs to a superseded pilot and must not be
   described as active randomized-v2 behavior.
5. **Separate work, staging, and final output.**
   Both current functions create the final directory and write the JSONL before
   the manifest. A crash leaves an incomplete directory that blocks retry. Use
   a stable work directory and a sibling staging directory, then publish.
6. **Strengthen receipts.**
   The current calibration and benchmark manifests do not record the SHA-256 or
   byte size of their own `source-natural.jsonl`, and do not bind the generator
   source/dependency environment. The benchmark records the calibration-manifest
   hash, which should be retained as one link in the full provenance chain.
7. **Validate calibration contents, not shape alone.**
   `prepare_headline_benchmark()` currently accepts any dictionary as a completed
   transition probe. Validate query coverage, attempts, thresholds, supported
   transitions, and the frozen seen/held-out partition.
8. **Add a post-write semantic audit.**
   Reopen and replay the staged corpus before candidate publication. Do not rely
   only on generation-time checks or a mutable checksum.
9. **Make disjointness an input, not only a final observation.**
   `_replace_headline_duplicates()` checks only the newly generated row list.
   Bind calibration, training, development, SINGLE, and relevant engineering
   pools into preflight capacity and final disjointness checks.
10. **Strengthen Slurm completion checks.**
   `prepare-calibration.sh` checks only that `manifest.json` exists, while the
   benchmark wrapper does not validate a completed artifact receipt. Require the
   final status and audit/hash checks.

## Current FYP decision

The full allocator/checkpoint/publication design below is methodologically
sound but disproportionate to the remaining FYP schedule. The active study
will retain deterministic marginal pairing, explicit empirical-operational
caps, bounded duplicate replacement, and strict final uniqueness. Further
narrow failures may receive targeted, documented corrections. This accepts
some engineering retry risk in exchange for moving promptly to model
experiments; no quota, answer contract, or canonical check may be silently
relaxed.

The larger design is retained here for future work. It becomes worthwhile if
the generator is reused for a public benchmark release, substantially larger
corpora, or repeated benchmark revisions.

## Deferred future-work implementation plan

### First hardening increment

- Add capacity provenance and correct the four-structure wording/tests.
- Freeze the canonical identity version and bind all cross-artifact exclusions
  before estimating usable capacity or solving the allocation.
- Implement the deterministic capacitated allocation, tie-breaking objective,
  persisted joint table, and ordered work plan.
- Compute a run key from design, plan, seeds, inputs, and implementation.
- Add deterministic chunk checkpoints with hash-verified resume.
- Add sibling staging, persisted-data replay, whole-artifact audit, file hashes,
  and final publication marker.
- Validate the calibration transition probe semantically.
- Add tests for interruption, stale contracts, corrupt/truncated chunks,
  worker-count invariance, duplicate replacement, and interrupted finalization.

These changes would protect larger or repeated runs and strengthen a public
release, but they are not prerequisites for completing the current FYP.

### Later release hardening

- Add a `--preflight`/`--check` command that emits the frozen allocation and a
  machine-readable feasibility report without generating rows.
- Add a concise audit/report command that verifies an existing candidate and
  renders its allocations, distributions, hashes, and capacity provenance.
- Publish a compact dataset card covering selection, filtering, splits,
  identity/disjointness, intended uses, and limitations, and freeze the
  evaluation task/config, decoding settings, metric version, and model revision
  with the benchmark.
- Exhaustively enumerate the bounded minimum Count construction space if the
  thesis needs to upgrade the four-structure cap from empirical to proved.
- Optionally key reusable chunk directories by run key. This is a narrow
  content-addressed cache, not a general artifact store.
- Reuse the same small pipeline helper for later training and diagnostic pools
  only after the benchmark path is stable.

### Not now

- Beam, Dataflow, Snakemake, Nextflow, DVC, or another workflow-engine migration;
- a database-backed job coordinator or distributed exactly-once protocol;
- one-file-per-row checkpointing;
- a general content-addressed blob store or Merkle-DAG artifact system;
- container orchestration solely for this generator; or
- a generic pipeline framework spanning unrelated experiments.

Those additions would increase implementation and operational surface without
improving the central FYP claim. A small deterministic allocator, chunked work
directory, independent audit, and atomic publication boundary provide the
needed reliability and reproducibility.

## Final methodological boundary

The pipeline can establish that the published candidate exactly implements its
declared experimental design: fixed quotas, solver-validated answers,
replayable certificates, structural separation, deterministic provenance, and
byte-identical artifacts. It cannot establish that the chosen marginals are
representative of naturally occurring spatial reasoning, that an empirical cap
is a mathematical capacity, or that held-out transition families exhaust
compositional generalization. Those remain explicit limitations, not pipeline
failures.
