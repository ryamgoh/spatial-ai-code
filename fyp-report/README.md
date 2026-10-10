# FYP report

This directory contains the buildable Typst dissertation draft. The report uses
one seven-chapter narrative from benchmark diagnosis through SpatialEntail and
its predeclared learning study. SFT, transfer, and quality-review results are
explicitly marked as not run.

## Build

From the repository root:

```sh
typst compile fyp-report/report.typ
```

The generated `fyp-report/report.pdf` is ignored by Git.

## Structure

| Path | Purpose |
| --- | --- |
| `report.typ` | Project metadata, global formatting, and document assembly |
| `frontmatter/` | Cover, title page, abstract, acknowledgements, and contents |
| `chapters/01-introduction.typ` | Motivation, scope, research questions, and contributions |
| `chapters/02-related-work.typ` | Text spatial benchmarks, structured traces, maps, and verified reasoning |
| `chapters/03-problem-formulation.typ` | All-model semantics and the SpatialEval diagnosis |
| `chapters/04-spatialentail-method.typ` | Solver, certificates, provenance, curriculum, and admission |
| `chapters/05-experimental-design.typ` | Models, supervision arms, depth ladder, transfer, and quality protocol |
| `chapters/06-results-discussion.typ` | Completed audit evidence and frozen placeholders for unrun experiments |
| `chapters/07-conclusion.typ` | Claims, limits, and remaining empirical work |
| `figures/` | Reusable Typst diagram definitions and shared diagram styling |
| `appendices/` | Formal correctness, dataset records, frozen configuration, and additional results |
| `docs/` | Report-authoring notes such as the Typst diagram-package comparison |
| `references.bib` | Bibliography used by the report |
| `reference/` | Official formatting guidance and other non-source references |

Working research notes remain in the repository-level `docs/` directory so
they can be shared with the solver, data-generation, and experiment work.
