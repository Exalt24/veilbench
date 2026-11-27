"""Span metrics, recall weighted scores and two leakage measures.

Matching is typed: a predicted span counts as a hit for a ground truth span of the same label when their
character overlap divided by their union is at least 0.5. F2 weights recall four times as much as precision,
because for a redaction pipeline a missed identifier costs more than a needless mask.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from veilbench import ids
from veilbench.schema import ENTITY_TYPES, Record, Span

PARTICLES = {"de", "del", "dela", "la", "los", "san", "van", "von", "ma"}


def prf(tp: int, fp: int, fn: int, beta: float = 1.0) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if p + r else 0.0
    return p, r, f


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def summary(self) -> dict:
        p, r, f1 = prf(self.tp, self.fp, self.fn)
        _, _, f2 = prf(self.tp, self.fp, self.fn, beta=2.0)
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "f2": round(f2, 4)}


def match_record(gt: list[Span], pred: list[Span], iou: float = 0.5) -> dict[str, Counts]:
    """Greedy one to one typed matching. Returns counts per label."""
    out: dict[str, Counts] = defaultdict(Counts)
    used: set[int] = set()
    for g in gt:
        best, best_iou = -1, 0.0
        for i, p in enumerate(pred):
            if i in used or p.label != g.label:
                continue
            v = g.iou(p)
            if v >= iou and v > best_iou:
                best, best_iou = i, v
        if best >= 0:
            used.add(best)
            out[g.label].tp += 1
        else:
            out[g.label].fn += 1
    for i, p in enumerate(pred):
        if i not in used:
            out[p.label].fp += 1
    return out


def covered_fraction(g: Span, pred: list[Span], doc: str) -> float:
    """Share of the alphanumeric characters of `g` that lie inside any predicted span, whatever its label."""
    chars = [i for i in range(g.start, g.end) if doc[i].isalnum()]
    if not chars:
        return 1.0
    hit = sum(1 for i in chars if any(p.start <= i < p.end for p in pred))
    return hit / len(chars)


def _uncovered_runs(g: Span, pred: list[Span], doc: str) -> list[str]:
    """Maximal runs of alphanumeric characters of `g` not inside any predicted span."""
    runs, cur = [], []
    for i in range(g.start, g.end):
        covered = any(p.start <= i < p.end for p in pred)
        if doc[i].isalnum() and not covered:
            cur.append(doc[i])
        else:
            if cur:
                runs.append("".join(cur))
            cur = []
    if cur:
        runs.append("".join(cur))
    return runs


def leaks(g: Span, pred: list[Span], doc: str) -> bool:
    """Does the ground truth span `g` still show through, given the ranges that were replaced?

    PERSON: any name token of two or more letters survives (a particle such as "dela" does not count).
    ADDRESS: more than half of its letters and digits survive.
    Digit identifiers (phone, card, national id, IBAN): four or more of their digits survive.
    Everything else (email, key, token, password, IP, date): any run of four or more characters survives.
    """
    runs = _uncovered_runs(g, pred, doc)
    if g.label == "PERSON":
        return any(len(r) >= 2 and r.lower() not in PARTICLES for r in runs)
    if g.label == "ADDRESS":
        total = sum(1 for i in range(g.start, g.end) if doc[i].isalnum())
        return total > 0 and sum(len(r) for r in runs) * 2 > total
    if g.label in ("PHONE", "CREDIT_CARD", "NATIONAL_ID", "IBAN"):
        return sum(sum(c.isdigit() for c in r) for r in runs) >= 4
    return any(len(r) >= 4 for r in runs)


def verbatim_leak(rec: Record, output: str) -> list[Span]:
    """Ground truth identifiers whose text, clean form or digits can still be read in `output`."""
    hits = []
    low = output.lower()
    out_digits = ids.digits_only(output)

    def present(needle: str) -> bool:
        # whole word match, so the name "Li" is not found inside "Lisa"
        return bool(needle) and re.search(r"(?<![a-z0-9])" + re.escape(needle.lower()) + r"(?![a-z0-9])", low) is not None

    for s in rec.spans:
        surface = rec.text[s.start:s.end]
        found = present(surface)
        if not found and s.canon and present(s.canon):
            found = True
        if not found and s.label in ("PHONE", "CREDIT_CARD", "NATIONAL_ID", "IBAN"):
            d = ids.digits_only(s.canon or surface)
            found = len(d) >= 7 and d in out_digits
        if found:
            hits.append(s)
    return hits


@dataclass
class Evaluation:
    per_type: dict[str, Counts] = field(default_factory=lambda: defaultdict(Counts))
    per_slice: dict[str, Counts] = field(default_factory=lambda: defaultdict(Counts))
    records: int = 0
    gt_spans: int = 0
    leaking_records: int = 0
    leaking_by_slice: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    caught_spans: int = 0
    negative_total: int = 0
    negative_hit: int = 0
    leak_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_record(self, rec: Record, pred: list[Span]) -> None:
        self.records += 1
        self.gt_spans += len(rec.spans)
        counts = match_record(rec.spans, pred)
        for label, c in counts.items():
            self.per_type[label].add(c)
            for sl in rec.slices:
                self.per_slice[sl].add(c)
        leaking = False
        for g in rec.spans:
            if covered_fraction(g, pred, rec.text) >= 0.8:
                self.caught_spans += 1
            if leaks(g, pred, rec.text):
                leaking = True
                self.leak_by_type[g.label] += 1
        self.leaking_records += int(leaking)
        for sl in rec.slices:
            self.leaking_by_slice[sl][0] += int(leaking)
            self.leaking_by_slice[sl][1] += 1
        for n in rec.negatives:
            self.negative_total += 1
            if any(p.overlaps(n) and not any(p.overlaps(g) for g in rec.spans) for p in pred):
                self.negative_hit += 1

    def totals(self) -> Counts:
        t = Counts()
        for c in self.per_type.values():
            t.add(c)
        return t

    def report(self) -> dict:
        types = {k: self.per_type[k].summary() for k in ENTITY_TYPES if k in self.per_type}
        slices = {k: v.summary() for k, v in sorted(self.per_slice.items())}
        return {
            "records": self.records,
            "gt_spans": self.gt_spans,
            "overall": self.totals().summary(),
            "per_type": types,
            "per_slice": slices,
            "caught_rate": round(self.caught_spans / self.gt_spans, 4) if self.gt_spans else 0.0,
            "leakage_rate": round(self.leaking_records / self.records, 4) if self.records else 0.0,
            "leaking_records": self.leaking_records,
            "leakage_by_slice": {k: {"leaking": v[0], "records": v[1], "rate": round(v[0] / v[1], 4)} for k, v in sorted(self.leaking_by_slice.items())},
            "leaking_spans_by_type": dict(sorted(self.leak_by_type.items())),
            "hard_negatives": {"total": self.negative_total, "flagged": self.negative_hit,
                               "rate": round(self.negative_hit / self.negative_total, 4) if self.negative_total else 0.0},
        }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


