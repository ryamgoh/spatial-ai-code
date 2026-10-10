# SODA task taxonomy: primary-paper verification

**Source:** Shunwen Bai et al., *One Cognitive Loop Is Enough: SODA unlocks
Pure-Text Spatial Reasoning in Large Language Models* (ACL 2026).
[Primary record](https://aclanthology.org/2026.acl-long.1382/);
[official PDF](https://aclanthology.org/2026.acl-long.1382.pdf);
DOI: **10.18653/v1/2026.acl-long.1382**. Verified directly from the official PDF
on 2026-10-10; no secondary-source names used. Page references below give the
printed proceedings page, followed by the PDF page in parentheses.

## Exact SPOD-Bench tiers and task names

The authoritative definitions are **§3.2.1, “Dataset Overview,” p. 29977
(PDF p. 4)**. SPOD-Bench has **13 task types in three tiers**. Names and
acronyms below follow that section, with line-wrap hyphenation removed.

| Exact tier heading | Exact task names and acronyms | Authors' grouping rationale (§3.2.1) |
|---|---|---|
| **Tier 1 (Basic Single-Turn Tasks)** | Memory Path (**MP**); Object Location Distance (**OLD**); Relative Direction (**RD**); Shortest Path (**SP**) | “geometric foundational defects”: misjudging basic concepts such as Euclidean distances and left–right relationships; rebuilding foundational knowledge. |
| **Tier 2 (Complex Single-Turn Tasks)** | Packing Shapes (**PS**); Mental Rotation (**MR**); Double-Point Relation (**DPR**); Multi-Point Relation (**MPR**); Door Rotation (**DR**) | “unstable spatial transformation defects”: rotation, reflection and coordinate translation disrupting the internal reference framework; explicit mental transformations. |
| **Tier 3 (Multi-Turn Dialogue Tasks)** | Door Rotation Multi-turn (**DRM**); Spatial Relation Multi-turn (**SRM**); Coordinates Movement Multi-turn (**CMM**); Mental Rotation Multi-turn (**MRM**) | “long-term planning defects”: premature interruption of multi-step decision chains; continuous state tracking and planning depth. |

**Table 2, p. 29979 (PDF p. 6)** uses shortened group labels: **Basic
Single-Turn**, **Complex Single-Turn**, and **Multi-Turn**. Its column order is
OLD/SP/RD/MP; PS/MR/MPR/DPR/DR; DRM/SRM/MRM/CMM. These are the same groups,
not alternative tier definitions.

### Naming and scope caveats

- Appendix **B.0.1, pp. 29986–29987 (PDF pp. 13–14)** calls DPR and MPR
  **“Doublepoint-relation Test”** and **“Multipoint-relation Test”**;
  example labels are `doublepoint_relation` and `multipoint_relation`.
  Retain the §3.2.1 names above when quoting the main taxonomy; the paper itself
  has spelling variants.
- **§3.2.1, p. 29977 (PDF p. 4)** separately lists additional **SPOD-143k**
  grid-world control setups: **OP, FOP, MCF, MCO**. **Figure 2, p. 29978
  (PDF p. 5)** labels their group **“Grid-World Maze”**, separate from Tier 1–3.
- Appendix **B.0.3, p. 29991 (PDF p. 18)** nevertheless has the exact heading
  **“Tier 4(Manipulation Tasks)”**. It is an appendix grouping of additional
  control tasks, **not a fourth tier of the 13-task SPOD-Bench in Table 2**.
  The appendix explicitly pairs **FOP** with
  `single_object_free_obstacle_path_planning`; it also describes obstacle
  path planning and multi-object control with/without obstacles
  (**pp. 29991–29993, PDF pp. 18–20**). **Unverified:** explicit acronym-to-full-name
  pairings for OP, MCF and MCO were not found in these excerpts; no expansions
  are asserted here.

## Capability organisation versus empirical difficulty

**Authors' conceptual claim:** these are not merely neutral capability buckets.
The abstract (**p. 29974, PDF p. 1**) says “three levels of difficulty,” and
§3.2.1 says the layers are organised “according to the difficulty of the
 defects.” The stated basis is a capability/failure-mode progression:
foundational geometry → spatial transformations → sustained multi-turn
tracking/planning. This is the authors' design rationale, not by itself an
empirically calibrated difficulty scale.

**Empirical evidence:** **§4.2, “Task-Level Performance Analysis,” and Table 2,
p. 29979 (PDF p. 6)** report high Tier 1 accuracy (mostly above 90%) and lower
performance for non-reasoning models on the more complex tiers. For example,
GPT-5 nano scores **92.8–97.6%** across Tier 1, **46.5–68.1%** across Tier 2,
and **25.0–47.0%** across Tier 3. This supports the broad progression for that
model, but not a universal task ordering: **o4-mini scores 26.9% on Tier 2 DR
versus 96.9% on Tier 3 DRM**, and reaches **100.0% on Tier 2 PS**.

**Safe interpretation:** SPOD-Bench is a capability/failure-mode taxonomy with
an authors-asserted difficulty progression and model-dependent empirical
support. **Unverified/not established by the cited evidence:** a calibrated,
model-independent difficulty scale, or a guarantee that every higher-tier task
is harder than every lower-tier task. “Single-turn” describes interaction
structure, not necessarily one computational step: Appendix B's MP uses
5–15 moves, while SP requires graph shortest-path computation
(**pp. 29985–29986, PDF pp. 12–13**).

**Access:** the official ACL PDF was successfully downloaded and inspected;
no primary-source blockage or fallback was needed.
