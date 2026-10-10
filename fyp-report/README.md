# FYP report

This directory contains the buildable Typst dissertation draft. Part II methods
are drafted from the current implementation; V2 SFT and model-evaluation results
are explicitly marked as not run.

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
| `chapters/` | Sequentially numbered main-matter chapters |
| `figures/` | Reusable Typst diagram definitions and shared diagram styling |
| `appendices/` | Appendices named by their actual subject |
| `docs/` | Report-authoring notes such as the Typst diagram-package comparison |
| `references.bib` | Bibliography used by the report |
| `reference/` | Official formatting guidance and other non-source references |

Working research notes remain in the repository-level `docs/` directory so
they can be shared with the solver, data-generation, and experiment work.
