#heading(numbering: none, outlined: false)[Abstract]

This project studies how formal semantics can improve the evaluation and
supervision of text-based spatial reasoning in large language models. It
interprets qualitative two-dimensional relations over finite named objects and
defines answers by the spatial models satisfying the visible premises. A
satisfying example establishes possibility; an entailed answer must hold across
all such models, with inconsistent premises handled separately.

Part I develops this semantics and applies it to the SpatialMap-TQA benchmark.
The repository audit classifies 667 of 1,500 released answers as uniquely
entailed and 833 as possible but non-unique under the declared interpretation.
The correction preserves the original questions and records underdetermination
explicitly. Manual assumption checks and the comparative LLM evaluation remain
pending, so these counts describe a conditional semantic audit.

Part II develops SpatialEntail, a generator and certificate-checking pipeline
for Direction, Which, and correlated Count queries. Natural and Symbolic targets
originate from accepted certificates, while Symbolic model outputs can be
parsed and replayed. Natural now narrates checked dependencies, while only
Symbolic output has a replay parser. A corpus-wide evidence and cost comparison
is still needed before isolating notation from the supervision package. A bounded
pilot specifies answer-only, checked Natural, checked Symbolic, and corrupted
Symbolic supervision, with separate development and test sets and exact
context admission. Training and model-evaluation results have not been run for
this draft. The implementation supports testing these hypotheses; certificate
validity does not establish fidelity to a model's internal reasoning.

#v(1fr)
*Subject Descriptors:* Artificial intelligence; knowledge representation and reasoning.

*Keywords:* spatial reasoning, formal semantics, benchmark auditing, proof certificates.

#pagebreak()
