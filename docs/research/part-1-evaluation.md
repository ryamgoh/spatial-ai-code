# Part I research plan: evaluating LLM spatial reasoning

Status: living experimental plan

This document develops the evaluation half of the project. It is subordinate to
the main [`FYP research plan`](research-plan.md) and does not fix the
dissertation chapter structure or experiment numbering.

## Purpose

Part I asks what should count as a correct answer when an LLM receives a partial
description of a spatial world.

A benchmark may derive its oracle from a complete latent map while exposing
only some relations in text. The oracle can be true in that map without being
entailed by the text. Evaluating it as the only correct answer then measures
agreement with hidden construction state as well as spatial reasoning.

SpatialEval Spatial-Map TQA is the first case study. Its original four-option
data remains unchanged. The formal solver provides an independent semantic
audit because the official release contains generated data and evaluation code
but no public Spatial-Map generator.

Part I does **not** generate new spatial questions. It has one fixed input: the
released 1,500-item SpatialMap-TQA set. SpatialMap-TQA-Corr is a derived, separately stored
view of those same questions with solver-audited labels and an additional
`Cannot be determined` option. Coordinate witnesses are produced only to audit
possible worlds; they are not new benchmark items.

## Research question

> **What changes when LLM spatial reasoning is evaluated using answers entailed
> by the presented information rather than answers selected from a hidden
> generating world?**

Supporting questions:

1. Is the formal spatial-to-SMT encoding correct within its declared ontology?
2. Does the adapter preserve every relevant statement and query?
3. How often is the published SpatialMap-TQA oracle uniquely entailed,
   possible but non-unique, impossible, or based on inconsistent premises?
4. Do constructive counter-witnesses and manual case checks support the audit?
5. How sensitive are the results to plausible semantic choices?
6. Does the corrected benchmark change LLM scores, error patterns, or rankings?

## Evidence boundary

The paper describes how SpatialEval constructs configurable synthetic tasks,
but its current public repository does not contain the generator and still
describes its release as forthcoming. The linked Hugging Face repository
contains generated TQA, VQA, and VTQA test data.

Therefore, this study can establish what follows from the released TQA text
under a stated ontology. It cannot recover the original coordinates, reproduce
undocumented generator choices, or prove that the private implementation has a
specific defect.

The precise claim for an ambiguous item is:

> The published oracle is compatible with at least one model of the text, but
> another model satisfying the same text gives a different answer. The oracle
> is therefore not uniquely entailed under the declared semantics.

## Study 1: formal correctness

### Theoretical obligation

For premises `P` and candidate claim `A`:

```text
Possible(P, A)  iff  SAT(P AND A)
Entailed(P, A)  iff  UNSAT(P AND NOT A)
```

Give a soundness-and-completeness argument for the translation from the
supported spatial language to Boolean integer constraints:

- every satisfying encoded model denotes a spatial configuration satisfying
  the source formula; and
- every spatial configuration satisfying the source formula is admitted by
  the encoding.

The argument must cover:

- the eight exact directions and their inverses;
- X/Y decomposition and axis equality;
- no co-location;
- transitive order constraints;
- coarse direction sets and negation;
- `AND`, `OR`, `IF`, and `IFF`;
- Direction, Which, and correlated Count queries; and
- inconsistent premise sets.

This proves the encoding relative to the ontology, not the correctness of every
line of Python or every interpretation of English.

### Implementation checks

- Compare the exhaustive reference and Z3 backends on bounded generated worlds.
- Exhaustively enumerate small worlds where feasible.
- Test inverses, rotations, reflections, premise permutation, redundant
  premises, and monotonic strengthening.
- Revalidate every returned witness against the source formula.
- Confirm that timeouts and unsupported inputs fail closed.

**Pass condition:** no unexplained backend disagreement or invalid witness.

## Study 2: adapter fidelity

For every released SpatialMap-TQA row, verify:

- complete premise consumption;
- correct entity identity and reference direction;
- correct parsing of relation phrases;
- correct Direction, Which, or Count query construction;
- correct candidate scope;
- preservation of option text and labels; and
- stable semantics after render-to-parse-to-solve round trips.

Unparsed text is an error, not an ignorable warning. This study separates solver
correctness from the harder question of whether the formal input matches the
dataset language.

## Study 3: complete SpatialMap-TQA audit

Preserve the original 1,500-row, four-option dataset. For each row, record:

- query family;
- published oracle;
- possible and entailed answer sets;
- consistency and solver status;
- audit classification;
- one validated witness per possible answer; and
- parser, solver, and ontology versions.

Use mutually exclusive classifications:

- uniquely entailed oracle;
- oracle possible but not unique;
- oracle underinclusive or overinclusive;
- oracle contradicted;
- inconsistent premises; and
- parsing or solver error.

Because the 1,500 rows are the complete released local slice rather than a
sample, report counts and proportions by Direction, Which, and Count. The
current implementation provides the reproducible command below; the
interpretation remains conditional on the stated ontology and case checks.

**Hypothesis:** a substantial portion of the published oracles are possible but
not uniquely entailed by the released text.

### Reproducible audit command

```bash
uv --system-certs run --python 3.12 --no-project --with typer --with z3-solver \
  python -m spatial.v2.audit_spatialeval \
  --input data/spatialeval_org.jsonl \
  --output-dir results/part1-spatialeval-audit
```

The command requires 1,500 input rows by default, leaves the source unchanged,
reparses every corrected prompt, validates every witness, and refuses to
overwrite outputs unless `--replace` is explicit. It writes:

- `spatialeval_audit.jsonl`;
- `spatialmap_tqa_corr.jsonl`; and
- `spatialeval_audit_summary.json` with input/output SHA-256 hashes.

The current Z3 run reports:

| Query | Exact oracle | Oracle possible but underdetermined |
|---|---:|---:|
| Direction | 332 | 168 |
| Which | 140 | 360 |
| Count | 195 | 305 |
| **Total** | **667** | **833** |

Two independent runs produced byte-identical audit and corrected-data files.

## Study 4: counter-witness and case validation

For every ambiguous audit result, validate at least two witnesses that satisfy
all premises and produce different answers. Representative cases should be
reduced to the smallest useful set of premises for presentation, while the full
audit artifact retains the original problem.

Manually inspect a small, stratified set of unique and ambiguous cases from each
query family. Check the parsed premises, possible answers, and witness maps.
Record any disagreement as a logic, parsing, wording, candidate-scope, or source
annotation issue. This is qualitative error checking by the researcher, not a
human-subject annotation study.

## Study 5: assumption sensitivity

The audit is conditional on its ontology. Test whether its main conclusion is
stable under plausible alternative interpretations:

| Choice | Main interpretation | Sensitivity question |
|---|---|---|
| Missing relations | Open world | Which conclusions would require treating absence as negation? |
| Candidate domain | All declared candidates | Does option-scoped Which interpretation change the audit? |
| Axis equality | Permitted | Does an ordinal-only restriction change candidate answers? |
| Co-location | Forbidden | Would a same-location state change any classification? |
| Direction language | Exact compass relation | Does any wording support a coarse interpretation? |
| Answer contract | `SINGLE` | How would complete-set answering change the expected response? |

Alternatives need not become official semantics. The purpose is to show which
findings are robust and which depend on a contestable modelling choice. A naive
closed-world interpretation is not adopted merely for comparison because every
pair must occupy some relation in a complete spatial model.

## Study 6: corrected benchmark integrity

Create SpatialMap-TQA-Corr as a separate artifact. Do not alter the original.

Under `SINGLE`:

- preserve an ordinary answer only when it is uniquely entailed;
- select E, `Cannot be determined`, when multiple answers remain possible;
- keep contradictions and parser failures separate from ambiguity; and
- retain complete possible-answer sets and coordinates only in audit metadata.

Verify that every corrected label can be regenerated from the frozen audit and
that the only model-facing change is the declared answer contract and option E.
If a uniquely entailed solver answer disagrees with the publication oracle,
report it separately rather than folding it into ambiguity.

## Study 7: effect on LLM evaluation

This is a required Part I experiment, not a broad leaderboard. It asks whether
the semantic correction changes conclusions about LLM spatial reasoning.

### Model groups

Use a representative LLM-only panel:

| Group | Purpose |
|---|---|
| General instruction-tuned LLM | Ordinary instruction-following baseline |
| Size-matched reasoning LLM where available | Test whether reasoning training improves entailment and uncertainty handling |
| Larger model from the target family | Separate some scale effects from method effects |
| Public text-spatial fine-tuned model, if compatible | Test whether existing spatial training transfers to corrected semantics |
| Legacy in-house spatial SFT | Measure transfer from the earlier forced-choice task contract |
| Optional frontier API LLM | External reference point rather than the main reproducible result |

Do not include a VLM or MLLM merely because it is spatially specialised. All
learned systems in this comparison receive text only. Record model version,
parameter count, training disclosures, context length, and any known exposure
to SpatialEval. Match model size or backbone when making claims about reasoning
training rather than overall model strength.

### Evaluation views

Evaluate the same checkpoints and decoding settings on:

1. original four-option SpatialMap-TQA;
2. five-option SpatialMap-TQA-Corr;
3. the unchanged uniquely entailed subset; and
4. the ambiguous subset.

The entailed subset controls for the additional menu option: its semantic
answer is unchanged between Original and SpatialMap-TQA-Corr. The ambiguous subset isolates
the ability to distinguish possibility from entailment.

For each ambiguous response, classify it as the published oracle, another
solver-possible answer, an impossible answer, `Cannot be determined`, or an
invalid output. Report original and corrected strict accuracy, entailed-subset
accuracy, abstention precision and recall, paired fixes and regressions, and
model-ranking changes.

### Prompt protocol

The primary comparison uses one shared task instruction and each model's
official chat template. A secondary condition may use the model's documented
reasoning mode, but its results remain separate. Do not tune a different task
prompt for every model.

The full Direct/Generic-CoT/Natural-axis/Symbolic-axis comparison belongs in
Part II. Part I uses the common direct prompt and, at most, one frozen structured
prompt across the panel.

Counterbalance the position of `Cannot be determined` in a robustness variant.
If performance changes when the option moves, the model may be exploiting menu
position rather than recognising ambiguity.

### Hypotheses

- Reasoning-oriented LLMs will identify underdetermination more reliably than
  comparable general instruction-tuned LLMs.
- Spatial models trained under forced-choice labels may improve ordinary
  entailed questions without improving ambiguity recognition.
- Corrected semantics will materially change measured performance or error
  interpretation. The direction of aggregate scores and ranking changes is not
  assumed.

Do not use individual SpatialMap-TQA-Corr errors to tune the later synthetic curriculum.
Curriculum and prompt decisions use the generated development suite so that
SpatialMap-TQA-Corr remains an external case-study and transfer evaluation.

## Success criteria

Part I succeeds if:

1. the encoding has a defensible correctness argument;
2. implementation checks reveal no unresolved semantic discrepancy;
3. every audited row is either fully parsed or explicitly rejected;
4. ambiguous classifications have valid counter-witnesses;
5. manual case checks reveal no unresolved parsing or semantic error; and
6. original-versus-corrected evaluation quantifies the practical consequence.

A small ambiguity count or negligible model-score change would weaken the
motivation but would still be a valid result. Backend disagreement, invalid
witnesses, or unresolved errors found during manual checks would block the
benchmark claim.

## Outputs

- Formal ontology and encoding argument
- Solver and adapter validation report
- Immutable copy/reference of original SpatialMap-TQA
- Machine-readable audit with witnesses
- Selected case checks and qualitative error notes
- SpatialMap-TQA-Corr
- Paired original-versus-corrected LLM results
- Representative unique and ambiguous case studies

## Open decisions

- Exact theorem statements and proof depth expected for the dissertation
- Number and selection criteria for manual case checks
- Main and alternative ontology settings for sensitivity analysis
- LLM checkpoints in the compact survey
- Direct prompt and decoding protocol
- Whether option-position counterbalancing is a main result or appendix check
