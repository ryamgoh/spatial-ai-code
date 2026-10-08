# Retired spatial implementations

This directory contains the original, V6, and V13 solvers and generators. They
are retained so historical experiments can still be inspected or reproduced;
new solver, proof, dataset, and evaluation work belongs in `spatial/v2/`.

The V6 and V13 suffixes are intentionally preserved inside this directory
because they distinguish historical contracts. V2 code must not import these
modules.

Run the retained regression suite from the repository root:

```bash
pytest spatial/v1 -q
```
