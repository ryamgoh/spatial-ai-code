import re


def solve(text):
    text = text.replace('\\n', '\n')

    # ===== 1. Extract directional relations =====
    rel_pattern = r"([^.\n]+?) is to the (Northeast|Northwest|Southeast|Southwest) of ([^.\n]+?)\."
    relations = re.findall(rel_pattern, text)

    # ===== 2. Collect all place names =====
    all_objects = set()
    for subj, d, obj in relations:
        all_objects.add(subj.strip())
        all_objects.add(obj.strip())
    map_m = re.search(r"([^.\n]+?) is in the map", text)
    if map_m:
        all_objects.add(map_m.group(1).strip())

    # ===== 3. Build directed edges =====
    # x_edges: (a, b) means b is to the east of a
    # y_edges: (a, b) means b is to the north of a
    x_edges, y_edges = set(), set()
    for subj, d, obj in relations:
        s, o = subj.strip(), obj.strip()
        if d == "Northeast":     # s is NE of o -> s is north of o AND s is east of o
            x_edges.add((o, s));  y_edges.add((o, s))
        elif d == "Northwest":   # s is NW of o -> s is north of o AND s is west of o
            x_edges.add((s, o));  y_edges.add((o, s))
        elif d == "Southeast":   # s is SE of o -> s is south of o AND s is east of o
            x_edges.add((o, s));  y_edges.add((s, o))
        elif d == "Southwest":   # s is SW of o -> s is south of o AND s is west of o
            x_edges.add((s, o));  y_edges.add((s, o))

    # ===== 4. Transitive closure =====
    def transitive_closure(edges):
        cl = set(edges)
        changed = True
        while changed:
            changed = False
            new = set()
            for a, b in list(cl):
                for c, dd in list(cl):
                    if b == c and (a, dd) not in cl:
                        new.add((a, dd))
            if new:
                cl |= new
                changed = True
        return cl

    xc = transitive_closure(x_edges)
    yc = transitive_closure(y_edges)

    # ===== Direction query helpers =====
    def is_east(o, r):  return (r, o) in xc
    def is_west(o, r):  return (o, r) in xc
    def is_north(o, r): return (r, o) in yc
    def is_south(o, r): return (o, r) in yc

    def definitely_in(o, r, d):
        """
        Determine whether o is definitely in direction d of r.

        Cardinal directions (North/South/East/West): only require the
        corresponding axis to be provable; the other axis is ignored.
          e.g. "North of X" = provably north of X (NE, NW, or due north all count)
        Compound directions (NE/NW/SE/SW): require both axes to be provable.
        """
        if d == "Northeast": return is_north(o, r) and is_east(o, r)
        if d == "Northwest": return is_north(o, r) and is_west(o, r)
        if d == "Southeast": return is_south(o, r) and is_east(o, r)
        if d == "Southwest": return is_south(o, r) and is_west(o, r)
        if d == "North":     return is_north(o, r)
        if d == "South":     return is_south(o, r)
        if d == "East":      return is_east(o, r)
        if d == "West":      return is_west(o, r)
        return False

    def contradicted_for(o, r, d):
        """Check whether o is provably NOT in direction d of r (contradicted)."""
        if d == "Northeast": return is_south(o, r) or is_west(o, r)
        if d == "Northwest": return is_south(o, r) or is_east(o, r)
        if d == "Southeast": return is_north(o, r) or is_west(o, r)
        if d == "Southwest": return is_north(o, r) or is_east(o, r)
        if d == "North":     return is_south(o, r)
        if d == "South":     return is_north(o, r)
        if d == "East":      return is_west(o, r)
        if d == "West":      return is_east(o, r)
        return False

    def count_between(cand, ref, direction):
        """Count the number of objects between cand and ref along each axis required by direction."""
        total = 0
        need_north = direction in ("Northeast", "Northwest", "North")
        need_south = direction in ("Southeast", "Southwest", "South")
        need_east  = direction in ("Northeast", "Southeast", "East")
        need_west  = direction in ("Northwest", "Southwest", "West")
        for o in all_objects:
            if o == cand or o == ref:
                continue
            if need_north and is_north(o, ref) and is_south(o, cand):
                total += 1
            if need_south and is_south(o, ref) and is_north(o, cand):
                total += 1
            if need_east and is_east(o, ref) and is_west(o, cand):
                total += 1
            if need_west and is_west(o, ref) and is_east(o, cand):
                total += 1
        return total

    # ===== 5. Match question type =====
    ALL_DIR = "Northeast|Northwest|Southeast|Southwest|North|South|East|West"
    q_dir   = re.search(r"In which direction is (.+?) relative to (.+?)\?", text)
    q_which = re.search(r"Which object is in the (" + ALL_DIR + r") of (.+?)\?", text)
    q_count = re.search(r"How many objects are in the (" + ALL_DIR + r") of (.+?)\?", text)

    # ===== 6. Parse answer options =====
    oi_m = re.search(r"Available options:", text)
    if not oi_m:
        return ""
    od = {}
    for ln in text[oi_m.start():].strip().split('\n'):
        m2 = re.match(r'([A-D])\.\s*(.+?)\.?\s*$', ln.strip())
        if m2:
            od[m2.group(1)] = m2.group(2).strip()
    if not od:
        return ""

    # ============ Type 0: In which direction ============
    if q_dir:
        tgt = q_dir.group(1).strip()
        ref = q_dir.group(2).strip()
        x_known = "east" if is_east(tgt, ref) else ("west" if is_west(tgt, ref) else None)
        y_known = "north" if is_north(tgt, ref) else ("south" if is_south(tgt, ref) else None)
        possible = set()
        if y_known and x_known:
            possible.add(y_known.capitalize() + x_known)
        if y_known and not x_known:
            possible.add(y_known.capitalize())
        if x_known and not y_known:
            possible.add(x_known.capitalize())
        res = sorted(l for l, d in od.items() if d in possible)
        return ",".join(res) if res else ""

    # ============ Type 1: Which object ============
    elif q_which:
        asked = q_which.group(1).strip()
        ref   = q_which.group(2).strip()

        # Step 1: strict proof — find all options provably in the asked direction
        res = sorted(l for l, o in od.items() if definitely_in(o, ref, asked))
        if len(res) == 1:
            return res[0]
        if len(res) > 1:
            # Multiple matches — break tie by fewest intervening objects (closest)
            scored = [(count_between(od[l], ref, asked), l) for l in res]
            scored.sort()
            return scored[0][1]

        # Step 2: elimination — keep options not contradicted
        res = sorted(l for l, o in od.items() if not contradicted_for(o, ref, asked))
        if len(res) == 1:
            return res[0]
        if len(res) > 1:
            scored = [(count_between(od[l], ref, asked), l) for l in res]
            scored.sort()
            return scored[0][1]
        return ""

    # ============ Type 2: How many objects ============
    elif q_count:
        asked = q_count.group(1).strip()
        ref   = q_count.group(2).strip()
        cnt = sum(1 for o in all_objects
                  if o != ref and definitely_in(o, ref, asked))
        res = sorted(l for l, n in od.items() if n == str(cnt))
        return ",".join(res) if res else ""

    return ""


# ========================= Tests =========================

# text0 = r"""Consider a map with multiple objects:
# Camelot Antiques is in the map. Andy's Autos is to the Northeast of Camelot Antiques. Oscar's Office Supplies is to the Southeast of Camelot Antiques. Oscar's Office Supplies is to the Southwest of Andy's Autos. Quokka's Quilts is to the Southwest of Andy's Autos. Quokka's Quilts is to the Northwest of Oscar's Office Supplies. Marshland Mart is to the Northeast of Quokka's Quilts. Marshland Mart is to the Northeast of Camelot Antiques. Parrot's Pottery is to the Northwest of Marshland Mart. Parrot's Pottery is to the Northeast of Quokka's Quilts.
#
#  Please answer the following multiple-choice question based on the provided information. In which direction is Marshland Mart relative to Parrot's Pottery? Available options:
# A. Northwest
# B. Northeast
# C. Southwest
# D. Southeast."""
# print("Test 0 (direction):", solve(text0))  # Expected: D (Southeast)
#
# text1 = r"""Consider a map with multiple objects:
# Unicorn's Umbrellas is in the map. Eccentric Electronics is to the Northwest of Unicorn's Umbrellas. Mantis's Maps is to the Southeast of Eccentric Electronics. Mantis's Maps is to the Southeast of Unicorn's Umbrellas. Tremor Toys is to the Northwest of Mantis's Maps. Tremor Toys is to the Southwest of Unicorn's Umbrellas. K University is to the Northeast of Eccentric Electronics. K University is to the Northeast of Mantis's Maps. Wild Water Park is to the Southeast of K University. Wild Water Park is to the Northeast of Tremor Toys.
#
# Please answer the following multiple-choice question based on the provided information. Which object is in the Northeast of Unicorn's Umbrellas? Available options:
# A. K University
# B. Tremor Toys
# C. Mantis's Maps
# D. Eccentric Electronics."""
# print("Test 1 (which NE):", solve(text1))  # Expected: A
#
# text2 = r"""Consider a map with multiple objects:
# Zebra's Zen Zone is in the map. Eagle's Electronics is to the Southeast of Zebra's Zen Zone. Baker Street Bookstore is to the Northwest of Eagle's Electronics. Baker Street Bookstore is to the Northeast of Zebra's Zen Zone. Oz Oddities is to the Southeast of Zebra's Zen Zone. Oz Oddities is to the Southwest of Eagle's Electronics. Waterfall Wonders is to the Northeast of Oz Oddities. Waterfall Wonders is to the Southeast of Zebra's Zen Zone. Gale Gifts is to the Northwest of Oz Oddities. Gale Gifts is to the Southwest of Eagle's Electronics.
#
# Please answer the following multiple-choice question based on the provided information. How many objects are in the Northwest of Zebra's Zen Zone? Available options:
# A. 3
# B. 0
# C. 1
# D. 4."""
# print("Test 2 (count NW):", solve(text2))  # Expected: B (0), NW is a compound direction
#
# text_new1 = "Consider a map with multiple objects: \\nYeti Yogurt is in the map.  Walrus Watches is to the Southeast of Yeti Yogurt.  Fresh Foods is to the Southeast of Yeti Yogurt. Fresh Foods is to the Southeast of Walrus Watches.  Ursa Uniforms is to the Northwest of Yeti Yogurt. Ursa Uniforms is to the Northwest of Fresh Foods.  Hummingbird Hats is to the Southwest of Ursa Uniforms. Hummingbird Hats is to the Northwest of Walrus Watches.  Umbrella Universe is to the Southeast of Ursa Uniforms. Umbrella Universe is to the Southeast of Yeti Yogurt. \\n\\n Please answer the following multiple-choice question based on the provided information. Which object is in the Southwest of Yeti Yogurt? Available options:\\nA. Hummingbird Hats\\nB. Fresh Foods\\nC. Umbrella Universe\\nD. Walrus Watches."
# print("New1 (SW of YY):", solve(text_new1))  # Expected: A
#
# text_new2 = "Consider a map with multiple objects: \\nPool Hall Provisions is in the map.  Miner's Market is to the Southwest of Pool Hall Provisions.  Safari Supplies is to the Southwest of Pool Hall Provisions. Safari Supplies is to the Northwest of Miner's Market.  Cobra's Cameras is to the Northeast of Safari Supplies. Cobra's Cameras is to the Southwest of Pool Hall Provisions.  Mordor Supplies is to the Southeast of Cobra's Cameras. Mordor Supplies is to the Northeast of Miner's Market.  Brews Brothers Pub is to the Southeast of Miner's Market. Brews Brothers Pub is to the Southeast of Safari Supplies. \\n\\n Please answer the following multiple-choice question based on the provided information. Which object is in the Southwest of Cobra's Cameras? Available options:\\nA. Brews Brothers Pub\\nB. Pool Hall Provisions\\nC. Mordor Supplies\\nD. Miner's Market."
# print("New2 (SW of CC):", solve(text_new2))  # Expected: D
#
# text_new3 = "Consider a map with multiple objects: \\nFactory Finds is in the map.  Fresh Foods is to the Northeast of Factory Finds.  Recycle Center is to the Northwest of Fresh Foods. Recycle Center is to the Northwest of Factory Finds.  Fox's Florist is to the Southeast of Factory Finds. Fox's Florist is to the Southeast of Recycle Center.  Jaguar Juice Bar is to the Southwest of Factory Finds. Jaguar Juice Bar is to the Southeast of Recycle Center.  Lucy's Lingerie is to the Southeast of Fresh Foods. Lucy's Lingerie is to the Southeast of Factory Finds. \\n\\n Please answer the following multiple-choice question based on the provided information. How many objects are in the South of Recycle Center? Available options:\\nA. 5\\nB. 2\\nC. 4\\nD. 0."
# print("New3 (South of RC):", solve(text_new3))
# # New interpretation: "South of RC" = provably south of RC (SE, SW, or due south all count)
# # RC is the northernmost; all other 5 objects are provably south of it -> answer A (5)
#
# text_new4 = "Consider a map with multiple objects: \\nAndy's Autos is in the map.  Ursa Uniforms is to the Northeast of Andy's Autos.  Ibex Instruments is to the Northwest of Andy's Autos. Ibex Instruments is to the Northwest of Ursa Uniforms.  Delilah's Deli is to the Northwest of Ursa Uniforms. Delilah's Deli is to the Northeast of Andy's Autos.  Soothing Springs Spa is to the Northeast of Ibex Instruments. Soothing Springs Spa is to the Northwest of Ursa Uniforms.  Buffalo's Books is to the Northeast of Delilah's Deli. Buffalo's Books is to the Northwest of Ursa Uniforms. \\n\\n Please answer the following multiple-choice question based on the provided information. How many objects are in the North of Andy's Autos? Available options:\\nA. 0\\nB. 4\\nC. 1\\nD. 3."
# print("New4 (North of AA):", solve(text_new4))
# # New interpretation: "North of AA" = provably north of AA (NE, NW, or due north all count)
# # AA is the southernmost; all other 5 objects are provably north of it -> should be 5,
# # but 5 is not among the options, so this test sample's options may be incorrect,
# # or the source dataset uses a different rule
#
