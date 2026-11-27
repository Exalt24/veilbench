"""Shared types: the entity labels, a span, and one labelled record."""
from __future__ import annotations

from dataclasses import dataclass, field

ENTITY_TYPES = [
    "PERSON", "EMAIL", "PHONE", "ADDRESS", "NATIONAL_ID", "CREDIT_CARD",
    "IBAN", "API_KEY", "JWT", "PASSWORD", "DATE_OF_BIRTH", "IP_ADDRESS",
]

# Types that are secrets rather than personal data: they get the strictest handling.
SECRET_TYPES = {"API_KEY", "JWT", "PASSWORD", "CREDIT_CARD"}


@dataclass(frozen=True, order=True)
class Span:
    """A half open character range [start, end) with a label.

    `canon` is only filled on ground truth: the clean form of an obfuscated value.
    """
    start: int
    end: int
    label: str
    source: str = field(default="", compare=False)
    canon: str = field(default="", compare=False)

    def text(self, doc: str) -> str:
        return doc[self.start:self.end]

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end

    def iou(self, other: "Span") -> float:
        inter = min(self.end, other.end) - max(self.start, other.start)
        if inter <= 0:
            return 0.0
        union = max(self.end, other.end) - min(self.start, other.start)
        return inter / union


@dataclass
class Record:
    """One generated document plus its ground truth."""
    id: str
    kind: str                 # ticket, transcript, json, csv, log, filename
    text: str
    spans: list[Span]         # every identifier that must not survive anonymization
    negatives: list[Span]     # look-alikes that are not identifiers (hard negatives)
    slices: list[str]
    meta: dict = field(default_factory=dict)
