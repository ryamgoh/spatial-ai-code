"""Bucket tags for v6 SFT jsonl (dir / which / count subtypes)."""
from __future__ import annotations

import re

ANSWER_RE = re.compile(r"Answer:\s*([A-E](?:\s*,\s*[A-E])*)", re.I)
INN_RE = re.compile(r"- ([A-E])\.\s*(.+?) — .+\. (In|Out)\.")

HARD = ("dir-2", "dir-incomplete", "dir-cycle")


def user_text(row: dict) -> str:
    for msg in row.get("messages") or []:
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def assistant_text(row: dict) -> str:
    for msg in row.get("messages") or []:
        if msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def qtype(text: str) -> str:
    if "In which direction is" in text:
        return "dir"
    if "Which object is in the" in text:
        return "which"
    if "How many objects are in the" in text:
        return "count"
    return "?"


def gold_letters(asst: str) -> list[str]:
    found = list(ANSWER_RE.finditer(asst))
    if not found:
        return []
    return [p.strip() for p in found[-1].group(1).split(",") if p.strip()]


def kind(row: dict) -> str:
    user = user_text(row)
    asst = assistant_text(row)
    t = qtype(user)
    n = len(gold_letters(asst))
    asst_tail = (
        asst[asst.find("### Final Deduction") :]
        if "### Final Deduction" in asst
        else asst
    )
    inns = INN_RE.findall(asst_tail)
    in_vals = [v.strip() for _let, v, mark in inns if mark == "In"]
    if t == "dir":
        if any("Cannot be determined" in v for v in in_vals):
            if "contradiction" in asst_tail:
                return "dir-cycle"
            if "only one of the two remaining" in asst_tail:
                return "dir-incomplete"
            return "dir-undetermined"
        if any(
            "none of" in v.lower()
            or "not among" in v.lower()
            or "no listed option" in v.lower()
            for v in in_vals
        ):
            return "dir-omit"
        if n == 1:
            return "dir-1"
        if n == 2:
            return "dir-2"
        return "dir-?"
    if t == "which":
        if any(
            "none of" in v.lower()
            or "not among" in v.lower()
            or "no listed" in v.lower()
            for v in in_vals
        ):
            return "which-0"
        k = sum(
            1
            for v in in_vals
            if "none of" not in v.lower()
            and "not among" not in v.lower()
            and "cannot be determined" not in v.lower()
            and "no listed" not in v.lower()
        )
        if k in (1, 2, 3, 4):
            return f"which-{k}"
        return "which-?"
    if t == "count":
        if any(
            "none of" in v.lower()
            or "not among" in v.lower()
            or "no listed" in v.lower()
            for v in in_vals
        ):
            return "count-omit"
        return "count-1"
    return "?"
