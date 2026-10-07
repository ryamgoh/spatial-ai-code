# AIMA citation note: propositional semantics and inference

## Recommended bibliographic record

Use the **fourth US edition** and the publication year attached to its print
ISBN, not a later Pearson eText/subscription date:

```bibtex
@book{russellNorvig2020aima,
  author    = {Russell, Stuart and Norvig, Peter},
  title     = {Artificial Intelligence: A Modern Approach},
  edition   = {4},
  publisher = {Pearson},
  year      = {2020},
  isbn      = {978-0-13-461099-3},
}
```

Pearson's own InformIT record confirms the title, authors, fourth edition,
publisher, publication date (28 April 2020), and both ISBN-10 and ISBN-13. The
authors' official companion site independently labels the book the “4th US
ed.” and names Stuart Russell and Peter Norvig. Pearson's current subject page
contains several later digital product variants, so its generic page-level
copyright/date must not replace the print record's 2020 year.

## Relevant coverage in the fourth US edition

The official detailed table of contents places the applicable material in
Chapter 7, **“Logical Agents”** (starting p. 208):

| Topic used here | AIMA location |
|---|---|
| Models, truth in a model, and logical entailment | §7.3, “Logic” (p. 214) |
| Propositional truth conditions / model semantics | §7.4 and especially §7.4.2, “Semantics” (pp. 217–220) |
| Validity, satisfiability, unsatisfiability, and the reduction of entailment to unsatisfiability | opening of §7.5, “Propositional Theorem Proving” (p. 222) |
| Soundness and completeness as properties of inference procedures | §7.5.1, “Inference and proofs” (p. 223) |
| A concrete completeness result for propositional resolution | §7.5.2, “Proof by resolution,” especially “Completeness of resolution” (p. 228) |
| SAT/model-checking algorithms | §7.6, “Effective Propositional Model Checking” (p. 232), including §7.6.1, “A complete backtracking algorithm” (p. 233) |

This citation therefore supports the standard model-theoretic vocabulary and
relationships used by the report: a model satisfies a sentence; entailment
means truth in every model of the premises; satisfiability means existence of
a model; and testing whether `P ∧ ¬A` is unsatisfiable is the standard route to
establishing `P ⊨ A`. It also supports the standard meanings of sound and
complete inference.

It does **not** establish this project's spatial results. The eight-direction
ontology, no-co-location assumption, sign-pair encoding, direction partition
and inversion, finite rank-compression lemma, atomic/formula adequacy,
path-consistency propagation argument, exact direction/which/count query
reductions, declared-candidate completeness boundary, Z3 trust boundary, and
implementation-conformance evidence remain original claims that require the
project's own definitions, proofs, and tests. Cite AIMA for the logical
framework, not as evidence for those solver-specific theorems.

## Primary sources

- Russell and Norvig, official AIMA site: [fourth US edition landing
  page](https://aima.cs.berkeley.edu/) and [detailed table of
  contents](https://aima.cs.berkeley.edu/contents.html).
- Pearson InformIT: [*Artificial Intelligence: A Modern Approach*, 4th
  Edition, ISBN 978-0-13-461099-3](https://www.informit.com/store/artificial-intelligence-a-modern-approach-9780134610993).
- Pearson: [current subject/product
  record](https://www.pearson.com/en-us/subject-catalog/p/artificial-intelligence-a-modern-approach/P200000003500/9780134610993)
  (useful for checking the active product variants; the print format is the
  2020 ISBN above).

Sources checked 2026-10-05.
