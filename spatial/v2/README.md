# Spatial V2

The active research direction is
[Composition v1](../../experiments/spatialentail-composition-v1/README.md),
governed by its [pivot](../../experiments/spatialentail-composition-v1/pivot_20261026.md).

## Active Generation

`generation.py` now constructs only atomic pairwise premises. Its policy has
no Boolean-shape, mixed-rule composition, or legacy possibility-skeleton
controls, and the generator has no randomized-proof entry point.
`generate_all.py` exposes the corresponding atomic-only workload CLI.

This is a cleanup of the existing V2 generator, not completion of the pivot.
Six-option menus, matched predicate-and-axis supervision, story-family release
gates, and the new evaluation protocol still require implementation.

## Shared Infrastructure

Keep `solver.py`, `proofs.py`, the certificate modules, `text.py`, `grading.py`,
and existing renderers available to shared and historical callers.
Internal logical operations support semantic checks and evidence even when
visible premises are atomic. Existing Symbolic NDJSON is an audit grammar,
not automatically the new study's Symbolic supervision format.

Software contracts in [docs/spatialentail](../../docs/spatialentail/) describe
existing behavior. Version the new interface without changing frozen historical
contracts or Corr SINGLE.

## Historical Workflows

The [legacy namespace](legacy/README.md) contains the retired Boolean
curriculum, mixed-rule compositions, randomized proofs, possibility skeletons,
and benchmark/matrix runners. Historical `study/` workflows and tests import
that namespace. The old benchmark YAML files retain their paths.

```sh
python -m spatial.v2.generate_all --help
python -m spatial.v2.legacy.check_registry spatial/v2/benchmark-cells.yaml --probe
python -m pytest spatial/v2 -q
```

Do not import `spatial.v1` or `spatial.v2.legacy` into the active generation path.
Historical reproduction uses those packages explicitly.
