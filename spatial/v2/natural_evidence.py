"""Shared qualitative witnesses and readable scopes in Natural evidence."""

import re


def share_natural_evidence(text: str) -> str:
    """Name each qualitative X/Y witness once, preserving all claim checks."""
    lines = text.splitlines()
    models = {}
    result = []
    index = 0
    while index < len(lines):
        if (
            lines[index].startswith("X west to east:")
            and index + 1 < len(lines)
            and lines[index + 1].startswith("Y south to north:")
        ):
            axes = (lines[index], lines[index + 1])
            name = models.get(axes)
            if name is None:
                name = f"M{len(models) + 1}"
                models[axes] = name
                if result and result[-1] == f"Qualitative witness {name}:":
                    result.extend(axes)
                else:
                    result.extend((f"Qualitative witness {name}:", *axes))
            else:
                result.append(f"Reuse qualitative witness {name} defined above.")
            index += 2
        else:
            result.append(lines[index])
            index += 1
    # Long encoded branch paths are implementation IDs, not explanation prose.
    text = "\n".join(result)
    branches = dict.fromkeys(re.findall(r"\[(refutation(?:/case-[^\]]+)+)\]", text))
    names = {branch: f"Case{index}" for index, branch in enumerate(branches, 1)}
    # Replace nested scopes before parent prefixes; step suffixes remain steps.
    for branch in sorted(names, key=len, reverse=True):
        text = text.replace(branch, names[branch])
    text = re.sub(
        r"^membership assignment \{([^}]*)\}$",
        lambda match: (
            match.group(0)
            + ": suppose exactly "
            + (
                match.group(1)
                + " satisfy the requested relation and all other candidates do not."
                if match.group(1)
                else "none of the candidates satisfies the requested relation."
            )
            + " We will refute this assumption."
        ),
        text,
        flags=re.MULTILINE,
    )
    repeated_premises = set()
    readable = []
    for line in text.splitlines():
        if re.match(r"P[0-9]+:", line):
            if line in repeated_premises:
                readable.append(
                    line.split(":", 1)[0] + ": Use the same premise stated above."
                )
                continue
            repeated_premises.add(line)
        readable.append(line)
    legend = (
        "Direction contract: North means equal X and greater Y; South means equal X and smaller Y; "
        "East means greater X and equal Y; West means smaller X and equal Y. "
        "If both axes differ, the relation is diagonal. Being higher on Y alone is not North, "
        "and being greater on X alone is not East."
    )
    if readable and readable[0] != legend:
        readable.insert(0, legend)
    return "\n".join(readable)
