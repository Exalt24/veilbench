"""The detector interface and the one place spans from several detectors are merged."""
from __future__ import annotations

from typing import Protocol

from veilbench.schema import Span

# Earlier names win a disagreement: a rule with a validator outranks a model, which outranks a heuristic.
PRIORITY = ["rules", "gliner", "stat", "spacy", "llm"]


class Detector(Protocol):
    name: str

    def detect(self, text: str) -> list[Span]: ...


def _rank(source: str) -> int:
    return PRIORITY.index(source) if source in PRIORITY else len(PRIORITY)


def _subtract(a: int, b: int, taken: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pieces = [(a, b)]
    for ta, tb in taken:
        nxt = []
        for pa, pb in pieces:
            if tb <= pa or ta >= pb:
                nxt.append((pa, pb))
                continue
            if pa < ta:
                nxt.append((pa, ta))
            if tb < pb:
                nxt.append((tb, pb))
        pieces = nxt
    return pieces


def merge_spans(spans: list[Span], keep_fraction: float = 0.5) -> list[Span]:
    """Combine detectors by priority, keeping every character any detector flagged that a better detector did not.

    A span from a lower priority detector is trimmed to the part not already covered. If less than
    `keep_fraction` of it is new, it is dropped: it was the same finding with sloppier edges, and keeping
    the sliver would only add a false positive. Recall comes from detectors finding different things, not from
    stretching every boundary to the union.
    """
    accepted: list[Span] = []
    taken: list[tuple[int, int]] = []
    for s in sorted(spans, key=lambda x: (_rank(x.source), x.start, -(x.end - x.start))):
        pieces = _subtract(s.start, s.end, taken)
        new_len = sum(b - a for a, b in pieces)
        if new_len == 0:
            continue
        if new_len == s.end - s.start:
            accepted.append(s)
            taken.append((s.start, s.end))
        elif new_len >= keep_fraction * (s.end - s.start):
            for a, b in pieces:
                accepted.append(Span(a, b, s.label, s.source))
                taken.append((a, b))
    return sorted(accepted)
