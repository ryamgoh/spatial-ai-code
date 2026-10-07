# AIMA citation for propositional semantics

Checked 2026-10-05 against the official AIMA and Pearson pages. Cite the
fourth **US** edition as follows:

```bibtex
@book{russell2020artificial,
  author    = {Stuart Russell and Peter Norvig},
  title     = {Artificial Intelligence: A Modern Approach},
  edition   = {4},
  publisher = {Pearson},
  year      = {2020},
  isbn      = {978-0-13-461099-3},
}
```

Pearson identifies the hardcover as ISBN-13 `9780134610993`, published
28 April 2020, with copyright year 2021. `year = {2020}` is therefore the
publication year for this ISBN; 2021 should not replace it merely because it
is the copyright year. Pearson is the publisher/imprint name in the catalog
metadata. The Pearson+ subscription (`9780137505135`, published 2021) is a
different product and should not be mixed into the print-book citation.

## Exact relevant coverage

All locations below are in Chapter 7, **Logical Agents** (starts p. 208), of
the fourth US edition. Section boundaries and page starts are confirmed by the
[official detailed contents](https://aima.cs.berkeley.edu/contents.html).

| Topic | Exact AIMA location | What it supports |
|---|---|---|
| Models, satisfaction, entailment, model checking, and the general meanings of soundness and completeness | §7.3, **Logic**, pp. 214–216 | A model gives truth values; entailment means truth in every model satisfying the premises; sound inference derives only entailments; complete inference can derive every entailment. |
| Propositional truth conditions | §7.4, **Propositional Logic: A Very Simple Logic**, p. 217; especially §7.4.2, **Semantics**, pp. 218–219 | A propositional model assigns true/false to proposition symbols, and connective truth is defined compositionally. |
| Entailment by exhaustive models | §7.4.4, **A simple inference procedure**, pp. 220–221 | Truth-table model enumeration directly decides propositional entailment and is sound and complete for the finite propositional vocabulary. |
| Validity, satisfiability, and refutation | Opening of §7.5, **Propositional Theorem Proving**, p. 222 | A sentence is satisfiable iff some model satisfies it; entailment can be reduced to unsatisfiability of the premises conjoined with the negated conclusion. |
| Proof rules and their soundness | §7.5.1, **Inference and proofs**, pp. 223–224 | Syntactic inference rules generate proofs; rules such as Modus Ponens are justified as sound. |
| Resolution completeness | §7.5.2, **Proof by resolution**, pp. 225–228; **Completeness of resolution**, p. 228 | Resolution provides the chapter's principal completeness result for propositional theorem proving. |
| SAT/model-checking algorithms | §7.6, **Effective Propositional Model Checking**, p. 232; §7.6.1, **A complete backtracking algorithm**, p. 233 | Algorithmic SAT/model checking, including a complete backtracking procedure. |

For a compact report citation covering the project's use of models,
entailment, satisfiability, and sound/complete inference, cite **Russell and
Norvig (2020, Chapter 7, especially §§7.3–7.6)**. Use the narrower section and
page above when attaching a citation to a specific definition.

## Citation boundary for this project

The AIMA citation supports the standard logical framework: truth in a model,
entailment over all models, satisfiability as existence of a model, the
equivalence between entailment and unsatisfiability after negating the
conclusion, and the meanings of sound and complete inference.

It does **not** establish this project's spatial ontology or solver theorems.
The eight compass relations, no-co-location assumption, X/Y integer encoding,
rank-compression argument, propagation rules, query/answer contracts,
SpatialEval corrections, and claims that this implementation is sound or
complete relative to that encoding are project-original and require the
project's own definitions, proofs, and empirical validation. AIMA may motivate
the logical vocabulary for those claims, but it is not evidence that the
spatial translation or Python/Z3 implementation is correct.

## Primary sources

- [Official AIMA fourth US edition site](https://aima.cs.berkeley.edu/): title,
  edition, authors, and Chapter 7 placement.
- [Official AIMA detailed contents](https://aima.cs.berkeley.edu/contents.html):
  chapter, section, subsection, and page starts.
- [Pearson US hardcover product page](https://www.pearson.com/en-us/subject-catalog/p/artificial-intelligence-a-modern-approach/P200000003500/9780134610993):
  fourth edition, authors, publisher, 28 April 2020 publication date, 2021
  copyright, hardcover format, and ISBN-13 `9780134610993`.
