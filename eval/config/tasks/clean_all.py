import re


def solve(text):
    # 1. Extract relationship sentences (use [^.]+? to prevent matching across sentence boundaries)
    rel_pattern = r"([^.]+?) is to the (Northeast|Northwest|Southeast|Southwest) of ([^.]+?)\."
    relations = re.findall(rel_pattern, text)

    # 2. Collect all objects
    all_objects = set()
    for subj, direction, obj in relations:
        all_objects.add(subj.strip())
        all_objects.add(obj.strip())
    map_match = re.search(r"([^.]+?) is in the map\.", text)
    if map_match:
        all_objects.add(map_match.group(1).strip())

    # 3. Build directed edges for X-axis (West < East) and Y-axis (South < North)
    #    (a, b) in x_edges means b is east of a
    #    (a, b) in y_edges means b is north of a
    x_edges = set()
    y_edges = set()

    for subj, direction, obj in relations:
        subj, obj = subj.strip(), obj.strip()
        if direction == "Northeast":
            x_edges.add((obj, subj))  # subj is east of obj
            y_edges.add((obj, subj))  # subj is north of obj
        elif direction == "Northwest":
            x_edges.add((subj, obj))  # subj is west of obj
            y_edges.add((obj, subj))  # subj is north of obj
        elif direction == "Southeast":
            x_edges.add((obj, subj))  # subj is east of obj
            y_edges.add((subj, obj))  # subj is south of obj
        elif direction == "Southwest":
            x_edges.add((subj, obj))  # subj is west of obj
            y_edges.add((subj, obj))  # subj is south of obj

    # 4. Compute transitive closure
    def transitive_closure(edges):
        closure = set(edges)
        changed = True
        while changed:
            changed = False
            new = set()
            for a, b in list(closure):
                for c, d in list(closure):
                    if b == c and (a, d) not in closure:
                        new.add((a, d))
            if new:
                closure.update(new)
                changed = True
        return closure

    x_closure = transitive_closure(x_edges)
    y_closure = transitive_closure(y_edges)

    # 5. Detect question type
    q_type0 = re.search(r"In which direction is (.+?) relative to (.+?)\?", text)
    q_type1 = re.search(r"Which object is in the (Northeast|Northwest|Southeast|Southwest) of (.+?)\?", text)
    q_type2 = re.search(r"How many objects are in the (Northeast|Northwest|Southeast|Southwest) of (.+?)\?", text)

    # 6. Parse options generically
    options_text = text[text.index("Available options:"):]
    options_dict = {}
    for line in options_text.strip().split('\n'):
        line = line.strip()
        m = re.match(r'([A-D])\.\s*(.+?)\.?\s*$', line)
        if m:
            options_dict[m.group(1)] = m.group(2).strip()

    # Helper: check if obj is definitively in a given direction relative to ref
    def is_in_direction(obj, ref, direction):
        if direction == "Northeast":
            return (ref, obj) in x_closure and (ref, obj) in y_closure
        elif direction == "Northwest":
            return (obj, ref) in x_closure and (ref, obj) in y_closure
        elif direction == "Southeast":
            return (ref, obj) in x_closure and (obj, ref) in y_closure
        elif direction == "Southwest":
            return (obj, ref) in x_closure and (obj, ref) in y_closure
        return False

    if q_type0:
        # Type 0: In which direction is target relative to reference?
        target = q_type0.group(1).strip()
        reference = q_type0.group(2).strip()

        if (reference, target) in x_closure:
            x_dirs = ["East"]
        elif (target, reference) in x_closure:
            x_dirs = ["West"]
        else:
            x_dirs = ["East", "West"]

        if (reference, target) in y_closure:
            y_dirs = ["North"]
        elif (target, reference) in y_closure:
            y_dirs = ["South"]
        else:
            y_dirs = ["North", "South"]

        possible = set()
        for y in y_dirs:
            for x in x_dirs:
                possible.add(y + x.lower())
        possible = {d[0].upper() + d[1:] for d in possible}

        correct = sorted([letter for letter, direction in options_dict.items() if direction in possible])
        return ",".join(correct)

    elif q_type1:
        # Type 1: Which object is in the [Direction] of [Reference]?
        asked_direction = q_type1.group(1).strip()
        reference = q_type1.group(2).strip()

        correct = []
        for letter, obj_name in options_dict.items():
            if is_in_direction(obj_name, reference, asked_direction):
                correct.append(letter)
        correct.sort()
        return ",".join(correct)

    elif q_type2:
        # Type 2: How many objects are in the [Direction] of [Reference]?
        asked_direction = q_type2.group(1).strip()
        reference = q_type2.group(2).strip()

        count = 0
        for obj in all_objects:
            if obj != reference and is_in_direction(obj, reference, asked_direction):
                count += 1

        correct = sorted([letter for letter, num in options_dict.items() if num == str(count)])
        return ",".join(correct)

    return ""


# ========================= Tests =========================

# Type 0 test
text0 = r"""Consider a map with multiple objects:
Camelot Antiques is in the map. Andy's Autos is to the Northeast of Camelot Antiques. Oscar's Office Supplies is to the Southeast of Camelot Antiques. Oscar's Office Supplies is to the Southwest of Andy's Autos. Quokka's Quilts is to the Southwest of Andy's Autos. Quokka's Quilts is to the Northwest of Oscar's Office Supplies. Marshland Mart is to the Northeast of Quokka's Quilts. Marshland Mart is to the Northeast of Camelot Antiques. Parrot's Pottery is to the Northwest of Marshland Mart. Parrot's Pottery is to the Northeast of Quokka's Quilts.

 Please answer the following multiple-choice question based on the provided information. In which direction is Marshland Mart relative to Parrot's Pottery? Available options:
A. Northwest
B. Northeast
C. Southwest
D. Southeast."""
print("Type 0:", solve(text0))

# Type 1 test
text1 = r"""Consider a map with multiple objects:
Unicorn's Umbrellas is in the map. Eccentric Electronics is to the Northwest of Unicorn's Umbrellas. Mantis's Maps is to the Southeast of Eccentric Electronics. Mantis's Maps is to the Southeast of Unicorn's Umbrellas. Tremor Toys is to the Northwest of Mantis's Maps. Tremor Toys is to the Southwest of Unicorn's Umbrellas. K University is to the Northeast of Eccentric Electronics. K University is to the Northeast of Mantis's Maps. Wild Water Park is to the Southeast of K University. Wild Water Park is to the Northeast of Tremor Toys.

Please answer the following multiple-choice question based on the provided information. Which object is in the Northeast of Unicorn's Umbrellas? Available options:
A. K University
B. Tremor Toys
C. Mantis's Maps
D. Eccentric Electronics."""
print("Type 1:", solve(text1))

# Type 2 test
text2 = r"""Consider a map with multiple objects:
Zebra's Zen Zone is in the map. Eagle's Electronics is to the Southeast of Zebra's Zen Zone. Baker Street Bookstore is to the Northwest of Eagle's Electronics. Baker Street Bookstore is to the Northeast of Zebra's Zen Zone. Oz Oddities is to the Southeast of Zebra's Zen Zone. Oz Oddities is to the Southwest of Eagle's Electronics. Waterfall Wonders is to the Northeast of Oz Oddities. Waterfall Wonders is to the Southeast of Zebra's Zen Zone. Gale Gifts is to the Northwest of Oz Oddities. Gale Gifts is to the Southwest of Eagle's Electronics.

Please answer the following multiple-choice question based on the provided information. How many objects are in the Northwest of Zebra's Zen Zone? Available options:
A. 3
B. 0
C. 1
D. 4."""
print("Type 2:", solve(text2))