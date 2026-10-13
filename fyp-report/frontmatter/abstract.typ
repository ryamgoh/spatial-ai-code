#heading(numbering: none, outlined: false)[Abstract]

This project studies whether formal semantics can improve the evaluation and
supervision of text-based spatial reasoning in large language models. It defines
qualitative two-dimensional relations over finite named objects and evaluates
answers across all spatial models satisfying the visible premises. A witness
establishes possibility; entailment requires agreement across every such model.

Applied to SpatialMap-TQA, the audit classifies 667 of 1,500 released answers as
uniquely entailed and 833 as possible but non-unique. SpatialMap-TQA-Corr
preserves the questions and adds explicit underdetermination. Manual assumption
checks and comparative model evaluation remain pending, so this is a
conditional semantic audit.

The same contract supports SpatialEntail, a generator and certificate-checking
pipeline for Direction, Entity Selection, and correlated Count queries over eight
directions and finite Boolean formulas. Replayable certificates and constructive
witnesses verify gold answers and remain offline audit artifacts. The
predeclared central study compares answer-only with verified Natural targets on
Qwen3.5-4B using two training seeds. A compact Chain-of-Symbol-like target is an
optional token-efficiency extension, not a certificate format. The sealed
release contains 8,000 training, 775 development, and 4,300 benchmark bases;
the benchmark reports CORE (1,300), COMP (1,200), and SHIFT (1,800), with SHIFT
slice labels retained as cell metadata. Its 485-item Natural quality sample was approved
by both independent model judges. Model benchmark and transfer results remain
pending.
Certificate validity supports auditability, not a model response contract or
claims about internal reasoning.

#v(1fr)
*Subject Descriptors:* Artificial intelligence; knowledge representation and reasoning.

*Keywords:* spatial reasoning, formal semantics, benchmark auditing, proof certificates.

#pagebreak()
