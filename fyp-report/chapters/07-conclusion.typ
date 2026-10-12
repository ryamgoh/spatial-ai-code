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
Answer certificates verify and audit solutions; they are not presented as the
model's hidden thought process or its default response format.

The implemented generator and bounded pilot establish preparation feasibility,
not a learning benefit. The central study compares answer-only and verified
Natural SFT on Qwen3.5-4B, separating matched interpolation, depth-5–8
extrapolation, premise-first formula composition, and external transfer. A
source-blinded two-model quality screen with selective human appeal measures
readability separately from formal validity. A compact Chain-of-Symbol-like arm
is optional and asks only whether a shorter representation can retain accuracy.

The unresolved empirical question is whether a verified Natural worked solution
teaches a more general spatial procedure than task exposure alone. Until the
evaluations run, the report makes no claim of SFT improvement, human-preferred
explanation, or causal faithfulness to internal model reasoning.

Future work may extend the controlled language, add visual grounding, evaluate
larger world models, and consider GRPO after supervised results demonstrate
both remaining headroom and non-degenerate reward variance. Those extensions
should retain the central contract: visible information defines the task, and
every scored explanation should have an auditable relation to that task.
