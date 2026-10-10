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
directions and finite Boolean formulas. Accepted certificates yield Natural and
replayable Symbolic supervision. The predeclared study compares answer-only,
checked Natural, and checked Symbolic targets on Qwen3.5-2B and Qwen3.5-4B, with
4B controls for corrupted evidence and local proofs. Candidate nested
4K/8K/17K budgets and
depth-1--8 tests separate matched reasoning from depth-5--8 extrapolation.
Training, transfer, and quality-review results are not yet available.
Certificate validity supports auditability, not claims about a model's internal
reasoning.

#v(1fr)
*Subject Descriptors:* Artificial intelligence; knowledge representation and reasoning.

*Keywords:* spatial reasoning, formal semantics, benchmark auditing, proof certificates.

#pagebreak()
