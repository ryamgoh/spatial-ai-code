"""Trace representation choices shared by checked certificate renderers."""

from enum import Enum


class TraceFormat(str, Enum):
    NATURAL = "natural"
    SYMBOLIC = "symbolic"
