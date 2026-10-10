# Explicit world-state supervision in text-only spatial reasoning

_Primary-source review, 10 October 2026._

## Question and representation taxonomy

This review asks a narrow question: do text-only spatial-reasoning systems
train a language model to **construct a complete explicit world state**, or do
they keep that state in a data generator, symbolic solver, verifier, or latent
neural memory while supervising only answers or local reasoning?

The following distinctions matter:

1. **Numeric coordinates:** an explicit assignment such as
   `A=(0,2), B=(1,0)`.
2. **Qualitative per-axis order/rank state:** a global, coordinate-free state
   such as `X: A < B < C; Y: B < A < C`, ideally with a declared convention
   for ties, disconnected components, and underdetermined pairs.
3. **Relation graph or path:** local or global triples/edges such as
   `(A, left, B)`, or a query-directed path through those edges. This need not
   select one globally consistent arrangement.
4. **Solver-generated rationale:** a natural-language or symbolic derivation
   produced from a graph/solver and used as a prompt or training target.
5. **Latent/internal map:** a hidden neural state or a behavioural hypothesis
   about what a model represents. It is not an externally checkable map unless
   the model is required to emit it.

An important semantic caveat follows. Coordinate compression removes arbitrary
translation and distance, but it does **not** necessarily make a world unique.
If the text leaves two objects incomparable, several rank maps can satisfy the
same premises. “Exact rank-map generation” is therefore well-defined only when
the task guarantees a unique order, defines a canonical completion, asks for
the entailed partial-order graph, or accepts any solver-validated satisfying
map. Exact string match to one arbitrary witness would conflate reasoning error
with harmless model choice.

## Bottom line

The inspected literature contains one direct explicit-layout training
precedent, but no located paper trains a text-only LLM to emit the specific
object requested here: a **complete, canonical, two-axis rank-compressed
spatial map and then answer from it**.

- Text2Space directly fine-tunes an LLM to construct complete ASCII grid
  layouts. It is strong evidence that explicit map construction is teachable,
  but also that construction is initially the bottleneck and erroneous sketches
  can harm answering.
- Full numeric coordinates remain common in **data generation and solvers**,
  not as the supervised output of the inspected spatial LLMs.
- The nearest qualitative-order precedent is Chain-of-Symbol's one-dimensional
  Brick World chain. It is few-shot prompting, not fine-tuning, and it does not
  construct a general two-axis rank state.
- Explicit relation graphs are genuine model outputs in graph-based StepGame
  SFT and in MentalMap's world-graph evaluation. These are the strongest direct
  precedents, but neither representation is a canonical per-axis rank map.
- SpaRP trains on query-directed, solver-generated verbalized paths. PistaQ
  trains relation extractors whose triples feed Prolog. Q-Chain uses formal
  chains to constrain the loss but does not require them at inference.
- SODA trains an explicit Observe–Orient–Decide–Act procedure. Some tasks track
  coordinate deltas or the agent's current grid location, but the target is a
  task-local cognitive/action trace rather than a complete canonical map.
- `interwhen` keeps Z3 coordinates in a test-time verifier and elicits local
  directional claims to audit; it neither trains nor asks the base model for a
  full map.
- “Mental map” claims in SpatialText are behavioural diagnostics of latent
  representation. MentalMap goes further by asking for an explicit JSON world
  graph, but it is an evaluation benchmark rather than task-specific map
  training.

Thus a rank-map target is not merely another spelling of SpaRP or CoS. It is
closest to **global structured-state generation**. Text2Space is the nearest
explicit-map SFT precedent; graph-triple SFT is the nearest relational-state
precedent; MentalMap L5 is the nearest JSON world-state evaluation precedent.

## Evidence by work

### Text2Space: Learning to Draw ASCII (Huang et al., 2026)

[Paper](https://arxiv.org/abs/2604.14641)

Text2Space pairs natural-language descriptions, spatial questions, and
ground-truth ASCII grid layouts derived from one spatial graph. It explicitly
tests text-to-ASCII construction, ASCII-to-text comprehension, and two joint
orders: answer before ASCII and ASCII before answer. The paper calls ASCII a
verifiable structured intermediate or “cognitive map.”

This is genuine layout-construction SFT: the authors fine-tune
Qwen3-30B-A3B on 4,000 training and 500 validation examples using LoRA. Their
base-model results reveal a read–write asymmetry: models read supplied ASCII
more reliably than they construct it, and asking a base model to draw ASCII
before answering reduces answer accuracy. After text-to-ASCII training, the
paper reports improved text-only spatial reasoning despite producing no ASCII
at inference, plus transfer gains on StepGame, bAbI task 19, and SpartQA.
Ground-truth ASCII still outperforms model-generated sketches.

Classification:

- **explicit global map output:** yes, as an ASCII grid;
- **task-specific map SFT:** yes;
- **intermediate-map generation:** evaluated directly before and after training;
- **not shown:** canonical X/Y rank compression, open-world multiple-witness
  acceptance, proof/refutation dependencies, or premise-relative entailment.

Text2Space is therefore strong evidence that construction can be learned and
can act as an auxiliary training signal. It is also direct evidence for the
user's concern: using an imperfect generated map as an inference-time
scratchpad can hurt. Its ground-truth grid is identified by the dataset's source
spatial graph. SpatialEntail's incomplete visible premises can admit several
equally valid rank completions, creating an additional target-identifiability
problem that Text2Space does not resolve.

### SODA / SPOD-143k (Bai et al., ACL 2026)

[Paper](https://aclanthology.org/2026.acl-long.1382/) ·
[PDF](https://aclanthology.org/2026.acl-long.1382.pdf)

SODA trains Qwen3-4B and Qwen3-14B on gold
Observe–Orient–Decide–Act (OODA) chains in the 143,000-example SPOD-143k
dataset, first with SFT and then with GRPO. The four stages have procedural
roles: Observe parses the current environment, Orient localizes or transforms
it, Decide selects a plan, and Act emits or executes the result.

Several SODA examples contain explicit spatial state. A relative-direction
example computes coordinate differences and bearing in the Orient stage. Its
grid-world maze traces repeatedly print the current coordinate, goal distance,
open directions, chosen movement, and updated coordinate. This is real
supervision of local state tracking and structured spatial computation.

It is not, however, supervision for a complete canonical map reconstructed from
partial qualitative premises:

- most OODA traces are query- or action-directed rather than exhaustive world
  states;
- coordinates are frequently given in the question or generated by the task
  program, rather than inferred as one of several valid open-world witnesses;
- the published data pipeline creates gold OODA chains by giving powerful LLMs
  the question and real answer, then performs sampled double-blind quality
  control;
- its GRPO reward checks conclusion correctness and the presence/order of the
  four OODA stages, not formal replay of each spatial inference.

Classification:

- **structured spatial CoT SFT:** yes;
- **explicit coordinate/state updates:** yes, for relevant task families;
- **complete global map output:** generally no;
- **formal proof or all-model witness checking:** no;
- **answer-conditioned rationale generation:** yes in the reported data
  expansion process.

SODA is therefore an important precedent for teaching a model a reusable
spatial procedure and for allocating substantial output to observation and
orientation. It does not answer whether a model can reliably construct our
two-axis rank witness, nor whether such a witness is preferable to a shorter
query-directed proof.

### Chain-of-Symbol (Hu et al., COLM 2024)

[Paper](https://openreview.net/forum?id=Hvq9RtSoHG) ·
[official repository](https://github.com/hanxuhu/chain-of-symbol-planning)

Chain-of-Symbol (CoS) is explicitly a prompting method that “does not need
additional training.” Its demonstrations are made by generating and manually
correcting natural-language CoT, replacing spatial expressions with compact
symbols, and using five demonstrations at inference. The paper's Brick World
examples concatenate relations into chains such as `A//C//B`; its 2-D variant
uses separate symbols for “on top of” and “in front of.” Navigation examples
emit candidate graph paths with summed distances, and SpartQA examples emit
symbolic object-relation statements.

Classification:

- **(b) qualitative order:** yes, narrowly, for 1-D stacks and simple 2-D
  task-specific chains;
- **(c) graph/path:** yes, especially for navigation and relational QA;
- **training:** no; few-shot in-context generation only;
- **not shown:** a normalized X/Y rank state, exhaustive global closure,
  solver validation, or a matched SFT comparison.

CoS is therefore evidence that models can be prompted to externalize compact
qualitative structure and that representation form can matter. It is not
evidence that a model can learn exact canonical rank maps from supervised data.

### SpaRC/SpaRP (Rizvi et al., ACL 2024)

[Paper](https://aclanthology.org/2024.acl-long.261/) ·
[official repository](https://github.com/UKPLab/acl2024-sparc-and-sparp) ·
[official dataset](https://huggingface.co/datasets/UKPLab/sparp)

SpaRP starts from context-question-answer examples. It extracts a symbolic
context of relation triples, constructs a network graph, finds a traversal from
the queried head to tail, composes relations with dataset-specific spatial
reasoners, and verbalizes the resulting reasoning path link by link. For its
StepGame-derived conditions the reasoner internally represents relative
positions as signed X/Y integers; those numbers are computation inside the
path generator, not the LLM target. The released schema's `reasoning` field is
a “verbalized reasoning path as deductively-verified CoT,” and Llama-2 models
are QLoRA-fine-tuned on those verbalized paths.

Classification:

- **(a) numeric coordinates/deltas:** solver-internal only;
- **(c) relation graph/path:** the symbolic source and query path;
- **(d) solver-generated rationale:** yes, and genuinely used for SFT;
- **not shown:** model emission of the source graph, a full world state, or an
  X/Y rank map.

SpaRP is the closest published precedent for solver-derived spatial
**proof-path supervision**, but the target remains query-local. A full rank map
would ask the model to preserve and serialize more state than SpaRP's selected
path. The paper reports that path fine-tuning improves F1 by 21--32 absolute
points for Llama-2-13B and 7--13 points for Llama-2-70B across its four
datasets. That establishes learnability of verbalized local paths, not full-map
serialization.

### StepGame and graph/symbolic training built on it

#### Original StepGame (Shi et al., AAAI 2022)

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/21383) ·
[official repository](https://github.com/ZhengxiangShi/StepGame)

StepGame samples a chain of directional relations, verbalizes it as a story,
and trains models to classify one of nine relative-position labels. Its
repository records coordinates in the data generator, while the proposed
TP-MANN stores learned entity-relation tensors in a recurrent memory. Neither
coordinates nor a readable world state are supervised outputs.

Classification: numeric state in the generator, latent relational memory in
the model, answer-label supervision only.

#### Graph-based synthetic StepGame SFT (Zhou et al., 2024)

[Paper](https://arxiv.org/abs/2409.12437) ·
[official repository](https://github.com/riddickzhou/LLM-Graph-Synthetic-Reasoning)

This work constructs a relational graph, samples non-repeating random-walk
chains, removes one edge as the prediction target, and verbalizes the remaining
chain. Its Extract-Then-Answer prompt requires the model to output the
**ordered structured triples** before the final answer. The paper fine-tunes
Mistral-7B on original stories plus synthetic chains, so this is real
supervision of an explicit relation-graph view rather than merely a solver
artifact.

Classification:

- **(a) numeric coordinates:** used to deduce StepGame relations in data
  construction, not emitted;
- **(c) relation graph:** yes, emitted as ordered triples under SFT;
- **not shown:** transitive closure, two canonical axis orders, or exact global
  reconstruction of every entity.

This is the strongest located training precedent for “parse text into explicit
structured spatial state, then answer.” It should be treated as the nearest
baseline for rank-map supervision, while noting that copying local story
triples is weaker than constructing a closed and canonical world map.

#### Q-Chain neuro-symbolic training (Premsri & Kordjamshidi, Findings of NAACL 2025)

[Paper](https://aclanthology.org/2025.findings-naacl.128/) ·
[official repository](https://github.com/HLR/SpaRTUNQChain)

Q-Chain forward-chains from SpaRTUN's annotated logical facts using 79 spatial
rules, derives a resolution tree and per-step consistency constraints, and adds
soft constraint-violation terms to the ordinary task loss. The paper is
explicit that logical representations are used **during training only** and
are unnecessary at inference. The model still outputs the QA answer, not the
Q-Chain or a map. StepGame is an evaluation set, not the source of its formal
training annotations.

Classification: symbolic world knowledge and proof chains in the training
objective, but no explicit state-generation target.

### PistaQ (Mirzaee & Kordjamshidi, Findings of EMNLP 2023)

[Paper](https://aclanthology.org/2023.findings-emnlp.221/) ·
[official repository](https://github.com/RshNk73/PistaQ-SREQA)

PistaQ separates language extraction from reasoning. BERT-based modules are
trained to identify spatial roles, classify relation triplets, and resolve
coreference; the resulting facts and query are converted to Prolog and passed
to a symbolic spatial reasoner. A related end-to-end variant receives
solver-expanded supervision for direct and indirect pairwise relations.

Classification:

- **(c) relation graph/facts:** yes, as supervised extraction/classification;
- **solver state:** Prolog facts and backtracking rules;
- **not shown:** autoregressive world-map generation, coordinates, axis ranks,
  or proof-trace SFT.

PistaQ demonstrates that explicit formal state can be a learned interface
between text and solver. It does not show that a generative LLM can serialize a
complete canonical state, and the authors emphasize that extraction errors
bound solver correctness.

### SpatialText and MentalMap

#### SpatialText (Jiang et al., 2026)

[Paper](https://arxiv.org/abs/2603.03002)

SpatialText is a diagnostic benchmark, not a spatial-state training method. Its
synthetic scenes first assign objects to 2-D or 3-D coordinates and then derive
pairwise relations deterministically. Evaluated models receive text/QA prompts;
non-CoT models are instructed to “think step by step to construct a mental
map,” but they are not required to emit coordinates or a standardized map.
The paper infers limitations in latent/internal spatial representations from
behaviour across reference-frame, uncertainty, and counterfactual tasks.

Classification: **(a) coordinates in dataset construction** and **(e) latent
map as the diagnostic hypothesis**, not explicit map supervision.

#### MentalMap (Pan et al., 2026)

[Paper](https://arxiv.org/abs/2605.28277)

MentalMap is much closer to explicit world-state construction. Its L5 tasks ask
models to generate static, dynamic, counterfactual, or local-slice world graphs
as JSON with `nodes` and `edges`, scored by validity plus node and edge F1. The
graphs emphasize object/receptacle containment and state transitions over
ProcTHOR/AI2-THOR scenes; they are not compass-axis rank maps. The study
evaluates existing models with zero-shot, few-shot, CoT, and structured-guidance
prompts rather than fine-tuning them on L5 maps.

Classification:

- **(c) explicit global relation graph:** yes, as an evaluation output;
- **training:** no task-specific map SFT in the paper;
- **not shown:** qualitative X/Y orders or numeric-coordinate generation.

MentalMap reports a useful warning for any exact-state target: node and edge
accuracy dissociate, and reasoning preambles can damage strict JSON validity.
Across its evaluated model groups, the reported mean node-minus-edge F1 gaps
range from 15.8 to 27.5 points.
That is evidence of structured-output fragility, not evidence about whether
rank-map SFT will or will not succeed.

#### Adjacent explicit-map pipeline: Language to Map (Deguchi et al., ICRA 2024)

[Paper](https://arxiv.org/abs/2403.10008) ·
[DOI](https://doi.org/10.1109/ICRA57147.2024.10611377)

Language to Map is a robotics/navigation paper rather than a qualitative
spatial-QA benchmark, but it is an important explicit-state precedent. Prompted
LLMs extract waypoint sequences and turn points into a canonical intermediate
representation; an external algorithm then constructs a topological graph of
nodes, edges, and actions and searches that graph for new paths. It compares
this pipeline with asking the LLM to retain an implicit map. The explicit-map
pipeline exceeds 90% success in the reported setting. By comparison, the paper
summarizes implicit-map reverse-path generation at approximately 65% success
and reports about 40% reachable-path success for GPT-4 on its harder combined-
path task.

Classification: explicit graph construction is beneficial, but the map is
assembled and searched algorithmically from prompted extractions. There is no
map SFT, numeric/rank-map target, or evidence that the LLM itself learns the
complete graph. The paper's own design splits node and turn extraction into
separate prompts to reduce task difficulty, which is directionally consistent
with decomposing a global-state target.

### `interwhen` (Bhat et al., 2026)

[Paper](https://arxiv.org/abs/2602.11202) ·
[official repository](https://github.com/microsoft/interwhen)

On the text-only SpatialMap split, `interwhen` periodically forks the model's
reasoning stream and asks a side stream to list derived claims in the form
“A is to the northeast of B.” Parsed problem relations are represented as Z3
real-valued X/Y variables; each extracted claim is checked against the
constraints and contradiction feedback is injected into the continuing trace.
The paper explicitly presents this as test-time steering “without any
finetuning.” Its structured prompt asks for given and derived relationships,
not one full coordinate or rank map.

Classification:

- **(a) numeric coordinates:** verifier-internal symbolic variables only;
- **(c/d) local relation claims/rationale:** elicited at test time and audited;
- **not training:** no spatial map supervision;
- **not a full state:** monitoring checks individual claims rather than
  requiring exhaustive global reconstruction.

This work is direct prior art for Z3-backed auditing on the same task family,
but not for teaching map construction.

## Analogous proof supervision outside spatial map construction

These works establish useful mechanisms but should not be presented as direct
rank-map precedents.

- **ProofWriter** ([paper](https://aclanthology.org/2021.findings-acl.317/),
  [official data](https://allenai.org/data/proofwriter)) fine-tunes T5 either to
  emit an entire answer-plus-proof DAG at once or to emit one depth-1
  implication at a time and assemble the fragments iteratively. Its explicit
  comparison is relevant to target granularity: when trained only through
  depth 3 and tested through depth 5, the iterative local-step model
  generalizes substantially better than the all-at-once proof generator at the
  unseen depths. This is evidence for decomposing structured reasoning, but it
  is not a spatial or map-generation result.
- **PrOntoQA** ([paper](https://openreview.net/forum?id=qFVVBzXxR2V),
  [official repository](https://github.com/asaparov/prontoqa)) generates an
  ontology, then a proof, then translates both into context, query, CoT, and
  label. It evaluates few-shot LLMs rather than training them. The authors find
  that models often make valid individual deductions but struggle with proof
  planning when several valid next steps are available. This supports a
  local-step-versus-global-planning distinction, not a quantitative claim
  about rank maps.
- **Faithful CoT** ([paper](https://aclanthology.org/2023.ijcnlp-main.20/),
  [official repository](https://github.com/veronica320/Faithful-COT)) prompts
  an LM to translate text into interleaved natural-language decomposition and
  executable programs, then lets a deterministic solver produce the answer.
  It is strong precedent for an explicit executable interface, but the main
  study is few-shot prompting and its relation programs are not spatial world
  maps.

## Is exact rank-compressed-map generation harder to teach than local proof traces?

### Defensible answer

**Probably, as a first supervised target; not yet demonstrated directly.** No
located primary paper runs the needed controlled comparison between
answer-only, query-local proof/path targets, relation triples, and a canonical
two-axis rank map on matched spatial problems.

There are four evidence-backed considerations:

1. **Explicit map construction is learnable, but initially fragile.**
   Text2Space finds that base models' generated ASCII can reduce downstream
   answer accuracy, while construction SFT improves both sketching and later
   text-only reasoning. Supplied ground-truth layouts remain better than
   generated ones. This is the closest direct result, although ASCII grids with
   source layouts are not open-world rank certificates.
2. **Local generation generalizes better in the closest proof analogue.**
   ProofWriter's explicit all-at-once versus one-step comparison favours
   iterative local implications at proof depths not seen in training. When
   trained through depth 3 and evaluated at depth 5, exact proof accuracy at
   depth 5 is 27.4% versus 87.8% under closed-world evaluation and 35.1% versus
   86.4% under open-world evaluation (all-at-once versus iterative). PrOntoQA
   likewise separates strong individual deductions from weaker global proof
   planning. These are logic-proof results, not spatial-map measurements.
3. **Global graph emission introduces independent failure modes.** MentalMap's
   L5 evaluation separates node recovery, edge recovery, schema validity, and
   JSON validity, and finds that these do not move together. A correct local
   spatial deduction therefore does not imply a correct serialized world
   state.
4. **Most other spatial SFT evidence uses smaller targets.** SpaRP supervises
   query-directed verbal paths; graph-based StepGame SFT supervises ordered
   local triples; Q-Chain supervises consistency in the loss. SODA demonstrates
   learnable multi-stage procedural and local-state traces, but not global
   canonicalization and complete closure. None establishes that models reliably
   learn our exact rank representation.
   The graph-based paper also reports that Extract-Then-Answer hurts few-shot
   use when models fail to extract graph relations, while SFT usually helps.
   Its StepGame ablation is mixed rather than universal: Extract-Then-Answer
   changes SFT accuracy from 35 to 48 at two hops and 26 to 32 at ten hops, but
   from 35 to 34 at seven hops.

The structural burden of a rank map is also visibly larger. It requires the
model to extract every relevant relation, split diagonal relations across two
axes, merge transitive information, retain disconnected or irrelevant
components, choose a canonical representation, and serialize all entities
exactly. A local proof path can ignore facts that do not support the query. This
is a task-complexity argument, not an empirical effect size.

### Why “harder” is not the whole story

A global state can still be a useful **scaffold**. Once correct, it amortizes
reasoning across multiple queries, exposes contradictions, and may make Which,
Count, ambiguity, and countermodel questions easier than repeatedly constructing
local paths. CoS's compact chains and graph-based StepGame's Extract-Then-Answer
results make this plausible. The literature does not determine whether that
benefit outweighs the harder state-construction target.

The most conservative research claim is therefore:

> Existing work supports training or prompting local spatial relations,
> paths, triples, and executable programs, and separately supports evaluating
> global world-graph emission. It does not yet establish the teachability or
> advantage of exact canonical two-axis rank-map supervision. Local-step proof
> results predict that such a target may need decomposition or auxiliary losses,
> but this remains a hypothesis to test.

## What a decisive experiment would need to separate

This is a literature-derived experimental distinction, not an implementation
proposal:

1. Use identical base problems and compare answer-only, SpaRP-like local path,
   graph-triple, per-axis delta, and full rank-map targets.
2. Report answer accuracy separately from map parse validity, entity coverage,
   pairwise X/Y order accuracy, global consistency, and solver acceptance.
3. Include both exact-match and semantic validation; exact match is appropriate
   only for a genuinely canonical and identifiable state.
4. Match supervised tokens or compute, since a full map is longer than a local
   path.
5. Test structural and depth extrapolation. ProofWriter suggests this is where
   local-step and all-at-once supervision are most likely to diverge.
6. Evaluate whether a correct emitted map causally supports the answer, rather
   than serving as a post-hoc decorative trace.

## Scope of the absence finding

The absence statement is bounded to the primary papers and official
repositories above, including the April 2026 Text2Space preprint, plus the reference trail in
`docs/research/annotated-sources.md`. It is not a proof that no unpublished,
concurrent, or differently named work uses rank maps. The strongest safe claim
is: **no exact two-axis rank-compressed-map training precedent was found in the
inspected text-only spatial-reasoning literature.**
