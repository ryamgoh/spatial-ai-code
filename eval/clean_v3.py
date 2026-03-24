"""
Spatial Reasoning Question Solver
==================================
Supports three question types:
  Type 0: "In which direction is X relative to Y?"
  Type 1: "Which object is in the [Direction] of X?"
  Type 2: "How many objects are in the [broad direction] of X?"

Returns comma-separated valid answer option letters, e.g. "A", "A,C", "A,B,C,D".

Algorithm overview
------------------
1. Model each object as a point (x, y) in 2-D space:
       NE(A, B) → x_A > x_B  AND  y_A > y_B
       SE(A, B) → x_A > x_B  AND  y_A < y_B
       SW(A, B) → x_A < x_B  AND  y_A < y_B
       NW(A, B) → x_A < x_B  AND  y_A > y_B

2. Build two directed "greater-than" graphs:
       x-graph: edge A→B  means  x_A > x_B  (A is east  of B)
       y-graph: edge A→B  means  y_A > y_B  (A is north of B)

3. Compute the transitive closure of each graph.
   After closure, A→B reachable  ↔  the ordering is provably true.

4. For each object pair (A, B), derive one of three states per axis:
       'gt'  — A > B  is provably true
       'lt'  — A < B  is provably true  (i.e., B > A is provably true)
       'unknown' — neither can be derived

5. Answer the question:
   • Type 0 / Type 1 — an option is VALID if the direction is *possible*
     (not contradicted by any known constraint).
     Since x and y are independent, both axes are checked independently.
   • Type 2 — count objects that are *definitely* in the stated broad
     direction (must hold in every consistent assignment).
"""

import re


def _normalize(text: str) -> str:
    """Normalize curly/smart quotes to ASCII apostrophe so object names
    parsed from the map description always match those parsed from options."""
    return (
        text
        .replace("\u2018", "'").replace("\u2019", "'")   # '' → '
        .replace("\u201c", '"').replace("\u201d", '"')   # "" → "
    )


# ──────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────

def parse_problem(text: str):
    """
    Returns (objects: list[str], relations: list[(a, dir, b)], question_part: str).
    'direction' values are lowercase: 'northeast', 'southeast', 'southwest', 'northwest'.
    """
    text = _normalize(text)
    split_marker = "Please answer"
    if split_marker in text:
        idx = text.index(split_marker)
        map_part = text[:idx]
        question_part = text[idx:]
    else:
        map_part = text
        question_part = ""

    objects: set[str] = set()

    # First object declared with "X is in the map."
    # Use [^\n.] so we never match across newlines (avoids capturing the
    # leading "Consider a map with multiple objects:\n" prefix).
    m = re.search(r"([^\n.]+?) is in the map", map_part)
    if m:
        objects.add(m.group(1).strip())

    # All directional relations: "X is to the [Dir] of Y."
    rel_re = re.compile(
        r"([^.]+?) is to the (Northeast|Northwest|Southeast|Southwest) of ([^.]+?)\.",
        re.IGNORECASE,
    )
    relations = []
    for m in rel_re.finditer(map_part):
        a = m.group(1).strip()
        d = m.group(2).strip().lower()
        b = m.group(3).strip()
        relations.append((a, d, b))
        objects.add(a)
        objects.add(b)

    return list(objects), relations, question_part


def detect_type(question_part: str) -> int:
    if "In which direction is" in question_part:
        return 0
    if "Which object is in the" in question_part:
        return 1
    if "How many objects are in the" in question_part:
        return 2
    return -1


def parse_options(question_part: str) -> dict[str, str]:
    """Parse 'A. Value' lines → {'A': 'Value', ...}."""
    opts = {}
    for m in re.finditer(r"\b([A-D])\.\s*([^\n]+)", question_part):
        key = m.group(1)
        val = m.group(2).strip().rstrip(".")
        opts[key] = val
    return opts


# ──────────────────────────────────────────────────────────
# Constraint graphs & transitive closure
# ──────────────────────────────────────────────────────────

def transitive_closure(objects: list[str], edges: list[tuple]) -> dict[str, set]:
    """
    reach[a] = set of b such that a > b is provably derivable.
    Uses iterative fixed-point expansion.
    """
    reach: dict[str, set] = {obj: set() for obj in objects}
    for (a, b) in edges:
        if a in reach:
            reach[a].add(b)

    changed = True
    while changed:
        changed = False
        for a in objects:
            before = len(reach[a])
            extras: set = set()
            for b in reach[a]:
                extras |= reach.get(b, set())
            reach[a] |= extras
            if len(reach[a]) > before:
                changed = True
    return reach


def build_order_graphs(objects: list[str], relations: list[tuple]):
    """
    Returns (x_reach, y_reach).
      x_reach[a] contains b  →  x_a > x_b  (a is east  of b, provably)
      y_reach[a] contains b  →  y_a > y_b  (a is north of b, provably)
    """
    x_edges, y_edges = [], []

    for (a, d, b) in relations:
        if d == "northeast":      # x_a > x_b,  y_a > y_b
            x_edges.append((a, b))
            y_edges.append((a, b))
        elif d == "southeast":    # x_a > x_b,  y_a < y_b  →  y_b > y_a
            x_edges.append((a, b))
            y_edges.append((b, a))
        elif d == "southwest":    # x_a < x_b  →  x_b > x_a,  y_a < y_b  →  y_b > y_a
            x_edges.append((b, a))
            y_edges.append((b, a))
        elif d == "northwest":    # x_a < x_b  →  x_b > x_a,  y_a > y_b
            x_edges.append((b, a))
            y_edges.append((a, b))

    return (
        transitive_closure(objects, x_edges),
        transitive_closure(objects, y_edges),
    )


def get_rel(reach: dict, a: str, b: str) -> str:
    """
    Relation of a vs b on one axis.
      'gt'      → a > b  (proved)
      'lt'      → a < b  (proved, because b > a)
      'unknown' → cannot determine
    """
    if b in reach.get(a, set()):
        return "gt"
    if a in reach.get(b, set()):
        return "lt"
    return "unknown"


# ──────────────────────────────────────────────────────────
# Direction checks
# ──────────────────────────────────────────────────────────

def direction_possible(x_rel: str, y_rel: str, direction: str) -> bool:
    """
    Is it POSSIBLE for A to be in `direction` of B,
    given x_rel / y_rel are A's relation to B?
    'gt' on x means A is east of B, etc.
    """
    d = direction.strip().lower()
    # Broad single-axis directions
    if d == "north":
        return y_rel != "lt"
    if d == "south":
        return y_rel != "gt"
    if d == "east":
        return x_rel != "lt"
    if d == "west":
        return x_rel != "gt"
    # Diagonal directions
    needs_east  = d in ("northeast", "southeast")
    needs_north = d in ("northeast", "northwest")
    x_ok = (needs_east  and x_rel != "lt") or (not needs_east  and x_rel != "gt")
    y_ok = (needs_north and y_rel != "lt") or (not needs_north and y_rel != "gt")
    return x_ok and y_ok


def direction_definite(x_rel: str, y_rel: str, direction: str) -> bool:
    """
    Is it DEFINITELY TRUE that A is in `direction` of B?
    Used for Type-2 counting.
    """
    d = direction.strip().lower()
    if d == "north":
        return y_rel == "gt"
    if d == "south":
        return y_rel == "lt"
    if d == "east":
        return x_rel == "gt"
    if d == "west":
        return x_rel == "lt"
    if d == "northeast":
        return x_rel == "gt" and y_rel == "gt"
    if d == "southeast":
        return x_rel == "gt" and y_rel == "lt"
    if d == "southwest":
        return x_rel == "lt" and y_rel == "lt"
    if d == "northwest":
        return x_rel == "lt" and y_rel == "gt"
    return False


# ──────────────────────────────────────────────────────────
# Main solver
# ──────────────────────────────────────────────────────────

def solve(text: str) -> str:
    """
    Parse and solve one spatial question.
    Returns comma-separated valid option letters, e.g. "A", "A,C", "A,B,C,D".
    """
    objects, relations, question_part = parse_problem(text)
    if not question_part:
        return "Error: no question found"

    q_type = detect_type(question_part)
    options = parse_options(question_part)
    if not options:
        return "Error: no options found"

    x_reach, y_reach = build_order_graphs(objects, relations)
    valid: list[str] = []

    # ── Type 0: "In which direction is X relative to Y?" ──────────────────
    if q_type == 0:
        m = re.search(
            r"In which direction is ([^?]+?) relative to ([^?]+?)\?", question_part
        )
        if not m:
            return "Error: cannot parse type-0 question"
        obj_x = m.group(1).strip()
        obj_y = m.group(2).strip()
        x_rel = get_rel(x_reach, obj_x, obj_y)
        y_rel = get_rel(y_reach, obj_x, obj_y)
        for key in sorted(options):
            if direction_possible(x_rel, y_rel, options[key]):
                valid.append(key)

    # ── Type 1: "Which object is in the [Dir] of X?" ──────────────────────
    elif q_type == 1:
        m = re.search(
            r"Which object is in the (\w+) of ([^?]+?)\?", question_part
        )
        if not m:
            return "Error: cannot parse type-1 question"
        direction = m.group(1).strip()
        obj_ref = m.group(2).strip()
        for key in sorted(options):
            candidate = options[key]
            x_rel = get_rel(x_reach, candidate, obj_ref)
            y_rel = get_rel(y_reach, candidate, obj_ref)
            if direction_possible(x_rel, y_rel, direction):
                valid.append(key)

    # ── Type 2: "How many objects are in the [Dir] of X?" ─────────────────
    elif q_type == 2:
        m = re.search(
            r"How many objects are in the (\w+) of ([^?]+?)\?", question_part
        )
        if not m:
            return "Error: cannot parse type-2 question"
        direction = m.group(1).strip()
        obj_ref = m.group(2).strip()

        count = 0
        for obj in objects:
            if obj == obj_ref:
                continue
            x_rel = get_rel(x_reach, obj, obj_ref)
            y_rel = get_rel(y_reach, obj, obj_ref)
            if direction_definite(x_rel, y_rel, direction):
                count += 1

        for key in sorted(options):
            try:
                if int(options[key]) == count:
                    valid.append(key)
            except ValueError:
                pass

    else:
        return "Error: unknown question type"

    return ",".join(valid) if valid else "No valid options found"


# ──────────────────────────────────────────────────────────
# Tests with the three provided examples
# ──────────────────────────────────────────────────────────

TYPE0 = """Consider a map with multiple objects:
Photogenic Studio is in the map. Fishing Frenzy is to the Southeast of Photogenic Studio. Gale Gifts is to the Northeast of Photogenic Studio. Gale Gifts is to the Northeast of Fishing Frenzy. Unicorn's Utensils is to the Northeast of Photogenic Studio. Unicorn's Utensils is to the Northeast of Gale Gifts. K University is to the Southwest of Gale Gifts. K University is to the Southwest of Unicorn's Utensils. Peet's Coffee is to the Northwest of Gale Gifts. Peet's Coffee is to the Southwest of Unicorn's Utensils.

Please answer the following multiple-choice question based on the provided information. In which direction is Fishing Frenzy relative to K University? Available options:
A. Northeast
B. Southeast
C. Northwest
D. Southwest."""

TYPE1 = """Consider a map with multiple objects:
Unicorn's Umbrellas is in the map. Eccentric Electronics is to the Northwest of Unicorn's Umbrellas. Mantis's Maps is to the Southeast of Eccentric Electronics. Mantis's Maps is to the Southeast of Unicorn's Umbrellas. Tremor Toys is to the Northwest of Mantis's Maps. Tremor Toys is to the Southwest of Unicorn's Umbrellas. K University is to the Northeast of Eccentric Electronics. K University is to the Northeast of Mantis's Maps. Wild Water Park is to the Southeast of K University. Wild Water Park is to the Northeast of Tremor Toys.

Please answer the following multiple-choice question based on the provided information. Which object is in the Northeast of Unicorn's Umbrellas? Available options:
A. K University
B. Tremor Toys
C. Mantis's Maps
D. Eccentric Electronics."""

TYPE2 = """Consider a map with multiple objects:
Rose Garden Florist is in the map. Eagle's Electronics is to the Southeast of Rose Garden Florist. Albatross's Astronomy Accessories is to the Northeast of Rose Garden Florist. Albatross's Astronomy Accessories is to the Northwest of Eagle's Electronics. K University is to the Southeast of Albatross's Astronomy Accessories. K University is to the Northwest of Eagle's Electronics. Jasmine's Jewellery is to the Southeast of Albatross's Astronomy Accessories. Jasmine's Jewellery is to the Southwest of K University. Fred's Fishing Supplies is to the Northwest of Eagle's Electronics. Fred's Fishing Supplies is to the Northwest of Albatross's Astronomy Accessories.

Please answer the following multiple-choice question based on the provided information. How many objects are in the North of Fred's Fishing Supplies? Available options:
A. 3
B. 2
C. 0
D. 1."""


def clean_jsonl(input_path: str, output_path: str) -> None:
    """
    Read every entry from input_path (JSONL), run the solver to determine the
    correct answer option(s), and write cleaned oracle fields to output_path.

    Cleaned fields:
      oracle_option       — comma-separated valid option letter(s) from solver
      oracle_answer       — answer text for the (first) valid option
      oracle_full_answer  — "<option>. <answer>" for the (first) valid option;
                            if multiple valid options exist, all are joined with " | "
    """
    import json

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            text = entry.get("text", "")

            result = solve(text)

            if result.startswith("Error: unknown question type"):
                # Non-spatial entry — pass through unchanged, no annotation
                pass
            elif result.startswith("Error") or result == "No valid options found":
                # Solver couldn't resolve — keep original oracle fields, flag entry
                entry["clean_note"] = result
            else:
                valid_keys = result.split(",")
                # Re-parse options to get answer text
                _, _, question_part = parse_problem(text)
                options = parse_options(question_part)

                first_key = valid_keys[0]
                first_ans = options.get(first_key, "")

                entry["oracle_option"] = result  # e.g. "A" or "A,C"
                entry["oracle_answer"] = first_ans
                entry["oracle_full_answer"] = " | ".join(
                    f"{k}. {options.get(k, '')}" for k in valid_keys
                )

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if line_no % 500 == 0:
                print(f"  processed {line_no} entries …")

    print(f"Done. Cleaned file written to: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "batch":
        input_file  = sys.argv[2] if len(sys.argv) > 2 else "spatialeval_org.jsonl"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "spatialeval_cleaned.jsonl"
        print(f"Cleaning {input_file} → {output_file}")
        clean_jsonl(input_file, output_file)
    else:
        print(f"Type 0 answer: {solve(TYPE0)}")
        print(f"Type 1 answer: {solve(TYPE1)}")
        print(f"Type 2 answer: {solve(TYPE2)}")
        print()
        print("Tip: run with argument 'batch' to clean the JSONL file.")
        print("  python clean_v3.py batch [input.jsonl] [output.jsonl]")
 

