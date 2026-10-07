# Typst diagram package recommendation

Checked 2026-10-05 against Typst Universe package sources. The repository host
currently has Typst **0.15.1** (`typst --version`), although the target baseline
was stated as 0.15.0.

## Recommendation

Use **Fletcher 0.5.8** for academic solver/correctness figures:

```typ
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
```

It directly supports the three useful diagram families here: explicit-node
flowcharts, math-native commutative diagrams, and directed graphs/trees/state
machines. Its Universe manifest declares Typst 0.13.0, so Typst 0.15.0 is above
the stated minimum. The examples below also compile together with the installed
Typst 0.15.1.

Add **CeTZ 0.5.2** only when a figure needs custom geometry or low-level drawing
that Fletcher cannot express cleanly:

```typ
#import "@preview/cetz:0.5.2"
```

CeTZ is the broader TikZ/Processing-like canvas and declares Typst 0.14.0, so it
also covers the 0.15.0 baseline. Fletcher already depends on CeTZ 0.3.4; directly
importing CeTZ 0.5.2 alongside it loads two CeTZ versions and exposes two APIs.
Keep most figures Fletcher-only and isolate any direct CeTZ drawing.

## Minimal Fletcher patterns

Flowchart:

```typ
#diagram(
  node-stroke: .7pt,
  node((0, 0), [Input]),
  edge("-|>"),
  node((1, 0), [Solve]),
  edge("-|>"),
  node((2, 0), [Verified]),
)
```

Commutative square:

```typ
#diagram($
  A edge(f, ->) edge("d", g, ->) & B edge("d", h, ->) \
  C edge(k, ->) & D
$)
```

Partial-order / proof-dependency graph:

```typ
#diagram(
  node-stroke: .7pt,
  node((0, 0), [$bot$]),
  node((-1, 1), [$a$]),
  node((1, 1), [$b$]),
  node((0, 2), [$top$]),
  edge((0, 0), (-1, 1), "->"),
  edge((0, 0), (1, 1), "->"),
  edge((-1, 1), (0, 2), "->"),
  edge((1, 1), (0, 2), "->"),
)
```

## Risks and offline builds

- Fletcher is the smaller, domain-specific API; CeTZ is more flexible but
  requires manual coordinates and more drawing code.
- Pin the exact versions above. Fletcher's own dependency is pinned to CeTZ
  0.3.4, not the latest CeTZ 0.5.2.
- `@preview` packages download on first compile and are then cached. A cold,
  offline CI build will fail: pre-warm and persist the package cache, set
  `--package-cache-path`, or vendor local packages and use `--package-path`.
  The local probe required Fletcher 0.5.8, CeTZ 0.3.4, and Oxifmt 0.2.1.
- No additional proof-tree or automata package is needed for the current scope:
  Fletcher already covers trees and finite-state graphs, avoiding another API
  and another cached dependency set.

## Primary sources

- [Fletcher 0.5.8 on Typst Universe](https://typst.app/universe/package/fletcher/)
  and its [package manifest](https://github.com/typst/packages/blob/main/packages/preview/fletcher/0.5.8/typst.toml)
- [Fletcher examples and changelog](https://github.com/typst/packages/blob/main/packages/preview/fletcher/0.5.8/README.md)
  and its [CeTZ 0.3.4 dependency](https://github.com/typst/packages/blob/main/packages/preview/fletcher/0.5.8/src/deps.typ)
- [CeTZ 0.5.2 on Typst Universe](https://typst.app/universe/package/cetz/), its
  [package manifest](https://github.com/typst/packages/blob/main/packages/preview/cetz/0.5.2/typst.toml),
  and [official usage documentation](https://github.com/typst/packages/blob/main/packages/preview/cetz/0.5.2/README.md)
- [Typst package download/cache guide](https://typst.app/docs/guides/for-latex-users/#packages)
