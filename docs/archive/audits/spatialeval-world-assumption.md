# SpatialEval Spatial-Map: modality matching and world assumptions

*Primary-source research note, checked 5 October 2026. This note separates what
the SpatialEval authors and release directly establish from a semantic
interpretation of the released TQA text.*

## Finding

SpatialEval's TQA, VQA, and VTQA formats are intended to ask the **same
underlying questions** with different input modalities. The paper says this
twice in Section 3: Figure 1's caption says the models are evaluated "on the
same set of questions," and the dataset-setup paragraph repeats that wording.
The official release shows the same alignment at row level.

This does **not** establish that SpatialEval explicitly adopts a closed-world
assumption. The paper uses neither *closed-world* nor *open-world* terminology
and does not specify that an unstated relation is false. It instead makes a
different claim: TQA contains "all necessary information" for a person to
answer, while the image and text representations of a problem are each
sufficient. That completeness claim can be tested against the released text;
it should not be silently renamed a closed-world semantics.

## Direct evidence: the modalities are matched

The paper defines TQA as text plus question, VQA as image plus question, and
VTQA as image plus the textual representation plus question. It states that
VTQA's modalities are redundant and that evaluation uses the same question
set across formats [1, Section 3].

The official Hugging Face release has three configurations, each with 4,635
test rows [2]:

| Config | Released fields |
|---|---|
| `tqa` | `id`, `text`, `oracle_answer`, `oracle_option`, `oracle_full_answer` |
| `vqa` | the same fields plus `image` |
| `vtqa` | the same fields plus `image` |

Representative Spatial-Map row 0 is an exact cross-modal match [3]:

| Config | ID | Question | Oracle |
|---|---|---|---|
| TQA | `spatialmap.tqa.2000.0` | direction of Planetarium Prints relative to Police Supply Store | `A`, Northeast |
| VQA | `spatialmap.vqa.2000.0` | same question and options | `A`, Northeast |
| VTQA | `spatialmap.vtqa.2000.0` | same question, options, and TQA map description, plus the image | `A`, Northeast |

The modality token is the only difference in those IDs. The same pattern is
visible for question-family row 500: `spatialmap.{tqa,vqa,vtqa}.2000.1` asks
which object is southwest of Ice Queen Ice Cream and carries oracle `B`,
Narwhal's Novelties [4]. The VQA and VTQA images for row 0 are also byte-for-byte
identical (SHA-256
`acaef414809aabe0f1d6d96dfa5d92eac4c08de18c5e0c9a595dbfbcdd6de0f8`).
Thus the paper directly asserts matching globally, and the released rows
demonstrate how the matching is encoded. This is stronger evidence than merely
observing equal row counts.

## Direct evidence: construction versus model-facing information

The paper says that each problem has an image and a text representation, that
Spatial-Map contains a configurable number of objects, and that its textual
representation consists of pairwise relations [1, Sections 1 and 3]. The
release contains the rendered image in VQA/VTQA and a common oracle across the
three modalities. This directly establishes a concrete benchmark instance
shared across modalities.

It is reasonable to infer that dataset construction operated on a particular
map arrangement from which the image and oracle were produced. That is an
**inference about construction**, not a published logical semantics. The
paper's exact description is that Spatial-Map objects are "positioned
arbitrarily"; it does not say that coordinates are randomly sampled. The
official repository still says that the generation script will be released,
but it is not present in the checked release [5]. Consequently, the public
sources do not expose a hidden coordinate state, label-generation rule, or a
sampling distribution, nor a statement that missing TQA relations should be
treated as false.

## Row evidence: the released TQA text can underdetermine its oracle

Official row `spatialmap.tqa.2000.1` gives pairwise compass relations and asks:
"Which object is in the Southwest of Ice Queen Ice Cream?" Its released oracle
is `B`, Narwhal's Novelties [4]. Under the ordinary qualitative reading

- northeast: both x and y are greater;
- northwest: x is smaller and y is greater;
- southeast: x is greater and y is smaller; and
- southwest: both x and y are smaller,

the text permits at least these two coordinate completions. Every listed
coordinate satisfies every pairwise sentence in the released row; all x and y
coordinates are distinct.

| Object | Completion 1 | Completion 2 |
|---|---:|---:|
| Police Supply Store | `(0, 0)` | `(0, 0)` |
| Narwhal's Novelties | `(-1, 1)` | `(-1, 4)` |
| Coral Crafts | `(-3, 4)` | `(-3, 5)` |
| Planetarium Prints | `(3, 2)` | `(3, 1)` |
| Oz Oddities | `(-2, -2)` | `(-2, -2)` |
| Ice Queen Ice Cream | `(-0.5, 3)` | `(1, 2)` |

In Completion 1, Narwhal's Novelties (`B`) is the unique offered object
southwest of Ice Queen Ice Cream. In Completion 2, Police Supply Store (`D`)
is the unique offered object southwest of it. Therefore the released text does
not uniquely entail the released oracle under this open-world,
constraint-based interpretation.

That counterexample is an analysis of the official row, not evidence that the
authors intended open-world semantics. Conversely, applying a conventional
closed-world rule ("unstated/unentailed means false") would not recover oracle
`B`; the row does not state or entail that Narwhal's Novelties is southwest of
Ice Queen Ice Cream. The more accurate interpretation is that `B` appears to
come from one intended/generated map, while the text leaves multiple maps
possible.

## Defensible wording

- **Directly sourced:** SpatialEval constructs matched image/text versions and
  evaluates the same questions in TQA, VQA, and VTQA; the authors claim the TQA
  text contains all necessary information.
- **Directly observed in the release:** modality-prefixed rows share the same
  question, options, and oracle; VTQA combines the TQA description with the
  corresponding image.
- **Interpretation supported by a counterexample:** at least one released TQA
  prompt is incomplete for uniquely deriving its oracle under qualitative
  constraint semantics, even though a concrete generated map can select that
  oracle.
- **Not sourced and should not be claimed:** "SpatialEval explicitly adopts a
  closed-world assumption." No such statement or absence-as-negation rule was
  found in the paper, dataset card, or official repository.

## Primary sources

1. Wang et al., *Is A Picture Worth A Thousand Words? Delving Into Spatial
   Reasoning for Vision Language Models*, NeurIPS 2024, especially Sections 1
   and 3 and Figure 1: <https://proceedings.neurips.cc/paper_files/paper/2024/file/89cc5e613d34f90de90c21e996e60b30-Paper-Conference.pdf>
2. Official SpatialEval dataset card and pinned release tree (revision
   `59ce0450bf65ae8dfa3590062442cf642474d7fe`):
   <https://huggingface.co/datasets/MilaWang/SpatialEval/blob/59ce0450bf65ae8dfa3590062442cf642474d7fe/README.md> and
   <https://huggingface.co/datasets/MilaWang/SpatialEval/tree/59ce0450bf65ae8dfa3590062442cf642474d7fe>
3. Official dataset viewer, row 0:
   [TQA](https://huggingface.co/datasets/MilaWang/SpatialEval/viewer/tqa/test?row=0),
   [VQA](https://huggingface.co/datasets/MilaWang/SpatialEval/viewer/vqa/test?row=0), and
   [VTQA](https://huggingface.co/datasets/MilaWang/SpatialEval/viewer/vtqa/test?row=0).
4. Official dataset viewer, row 500:
   [TQA](https://huggingface.co/datasets/MilaWang/SpatialEval/viewer/tqa/test?row=500),
   [VQA](https://huggingface.co/datasets/MilaWang/SpatialEval/viewer/vqa/test?row=500), and
   [VTQA](https://huggingface.co/datasets/MilaWang/SpatialEval/viewer/vtqa/test?row=500).
5. Official SpatialEval repository README at commit
   `d82ba382805265f205693cb83a30c4866ea6c35b`, including the unreleased-generator
   notice: <https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/README.md#L138-L140>
