# FYP Report

The active Typst draft follows the
[composition pivot](../experiments/spatialentail-composition-v1/pivot_20261026.md):
a verified atomic spatial-composition dataset and a supporting matched
Natural/Symbolic training comparison.

The Boolean-rule curriculum, old training-arm comparisons, randomized-rule
calibration tables, and broad generalization campaigns have been removed from
the report narrative. Solver operations needed to explain spatial entailment,
distinctness, and joint Count remain.

This is a shortened methodology draft, not a completed empirical report.
The new release and training results are explicitly pending. The historical
SpatialMap audit is labelled historical rather than presented as a result of
the new six-option contract views.

## Build

```sh
typst compile fyp-report/report.typ
```

`report.pdf` is ignored by Git. The seven chapters cover motivation, related
work, semantics, evidence and release gates, experimental design, current
evidence and limitations, and conclusions. Appendices specify spatial
correctness, dataset records, settings to freeze, and required reporting.

## Historical Sources

The complete pre-cleanup tracked report tree is preserved in
[`precomposition-report-2372c13.tar.gz`](../experiments/archive/precomposition-report-2372c13.tar.gz),
from revision `2372c13439bb94eb0369c7acb88fa696ff7c3b93`.
Archive SHA-256: `7c543fe4cdfb2e43772f48a4a9eef6aaa870663a617482c49c56d6a8bd117872`.
Extract it into a separate directory to inspect the original report sources;
some auxiliary presentation assets may depend on ignored generated files.
Historical results and receipts remain in their original experiment folders.

The old `professor-brief.typ`, `figures/`, and brief-export assets have been
removed from this directory. Their latest pre-pruning copies are preserved in
the [pruned-materials snapshot](../docs/archive/pruned-materials-20261028.tar.gz),
including the exporter's legacy-import update from the code cleanup.
See the [archive index](../docs/archive/README.md#superseded-plans-and-presentations)
for the checksum, contents, and archived research plans and diagrams.
