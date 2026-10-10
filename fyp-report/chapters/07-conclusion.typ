#pagebreak(weak: true)
= Conclusion

This project begins from a simple evaluation principle: a text-only model should
be scored against what follows from the text it receives. Under the declared
qualitative semantics, a relation is possible when some satisfying spatial
world supports it and entailed only when every satisfying world supports it.
Applying that distinction to SpatialMap-TQA shows that one latent-map answer can
remain possible without being uniquely recoverable from the released text.

SpatialEntail uses the same semantics constructively. It separates dataset
adapters, all-model solving, answer policy, local proof replay, constructive
model checking, rendering, and audit storage. Generated problems carry explicit
provenance and are accepted only after semantic analysis, certificate replay,
text round trip, structural split validation, and exact context admission.
Natural and Symbolic traces are evidence projections rather than claims about a
model's hidden thought process.

The implemented generator and bounded pilot establish preparation feasibility,
not a learning benefit. The final study therefore predeclares answer-only,
checked Natural, checked Symbolic, corrupted Symbolic, and local-proof
conditions over 2B and 4B models. It separates matched interpolation,
depth-5–8 extrapolation, premise-first formula composition, and external
transfer. A source-blinded LLM/human review measures readability separately
from formal validity.

The unresolved empirical question is whether checked evidence teaches a more
general spatial procedure than task exposure alone. The rank-witness comparison
asks a narrower question: whether explicit qualitative world construction helps
or whether a query-local proof is the better supervision target. Until those
experiments run, the report makes no claim of SFT improvement, human-preferred
explanation, or causal faithfulness to internal model reasoning.

Future work may extend the controlled language, add visual grounding, evaluate
larger world models, and consider GRPO after supervised results demonstrate
both remaining headroom and non-degenerate reward variance. Those extensions
should retain the central contract: visible information defines the task, and
every scored explanation should have an auditable relation to that task.
