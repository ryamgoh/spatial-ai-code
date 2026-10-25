# Human-in-the-loop quality review

This package owns the review seam after formal validation. It creates blinded
Natural-trace packets, validates independent model-judge receipts, routes
double approvals and split decisions, and exposes only unanimous model
rejections for human appeal.

The package does not decide formal correctness, generate benchmark examples,
or infer a training-set size. Those remain the responsibilities of the
certificate stack, benchmark registry, and size-calibration protocol.

The public interface is exported from `spatial.v2.hitl`. The execution and
reporting contract is documented in
[`docs/archive/research/human-review-workflow.md`](../../../docs/archive/research/human-review-workflow.md).
The frozen model-judge instructions are
[`quality-judge-v1.md`](quality-judge-v1.md).
