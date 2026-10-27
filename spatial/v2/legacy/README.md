# Retired SpatialEntail Curricula

These modules preserve the pre-composition Boolean and randomized-proof
curricula for historical experiment reproduction. They are not dependencies
of the active atomic generator or requirements for the current study.

- `generation.py` and `generate_all.py`: historical policies and workload CLI.
- `random_proof.py` and `random_proof_generation.py`: sampled logical-rule paths.
- `composition.py` and `possibility.py`: historical mixed-rule and possibility
  skeletons, including compound premises.
- `benchmark_registry.py`, `check_registry.py`, `matrix.py`, and
  `generate_matrix.py`: historical registry and ablation runners.

Historical study callers and their regression tests now import this namespace.
For example:

```sh
python -m spatial.v2.legacy.generate_all --help
python -m spatial.v2.legacy.check_registry --probe
python -m spatial.v2.legacy.generate_matrix --help
```

Registry YAML files stay at their existing `spatial/v2/` paths. Shared solver,
certificate, grading, rendering, and training infrastructure is not duplicated.
The historical generator snapshot is deliberately separate from the active
generator so removing active curriculum controls cannot silently alter old
generation policies or seeded outputs.

For byte-for-byte source provenance, the pre-cleanup revision is
`2372c13439bb94eb0369c7acb88fa696ff7c3b93`. Existing receipts remain unchanged;
the namespace migration changes source hashes and is not a reason to rewrite
old receipts or resume an incompatible generation cache.

The exact original source and environment inputs are also preserved in
[`precomposition-source-2372c13.tar.gz`](../../../experiments/archive/precomposition-source-2372c13.tar.gz).
Its SHA-256 is
`752cafdd8fb4a8f0896d934ceaeb87eef471ec6055d7d0689fd34817afb53d3d`.
Frozen preparation-receipt tests check that archive; runtime/scorer receipt
checks still validate the files actually used for evaluation. The frozen Corr
preparer retains its unchanged prompt-helper import, covered by a compatibility
test.

The active study is
[Composition v1](../../../experiments/spatialentail-composition-v1/README.md).
