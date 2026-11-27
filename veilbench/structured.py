"""A schema agnostic anonymizer: it never assumes which field holds what, and it fails closed to a plain text pass.

For raw input it tries, in order: JSON (walks every value and every key, and opens a string that is itself JSON, such as a
payload column), then delimited text (comma, semicolon, tab or pipe; header or none; a column that is mostly one entity type
is treated as that type for its quiet cells too), then plain text. A field name is a hint, never a requirement: renamed keys
("f2", "Nme") still get cleaned because every leaf is also run through the detectors.
"""
from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from veilbench.detect.base import merge_spans
from veilbench.schema import Span
from veilbench.transform.ops import Transformer

Detect = Callable[[str], list[Span]]

KEY_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)pass(word|wd)?|pwd|secret|contrase"), "PASSWORD"),
    (re.compile(r"(?i)api[_-]?key|token|authorization|bearer|credential|private[_-]?key|access[_-]?key"), "API_KEY"),
    (re.compile(r"(?i)e[-_]?mail|\bmail\b|correo"), "EMAIL"),
    (re.compile(r"(?i)phone|mobile|tel|msisdn|cell|fax"), "PHONE"),
    (re.compile(r"(?i)\bssn\b|national|tax[_-]?id|\bnric\b|\bdni\b|\btin\b|passport"), "NATIONAL_ID"),
    (re.compile(r"(?i)iban|account[_-]?number"), "IBAN"),
    (re.compile(r"(?i)card|\bpan\b|cc[_-]?num"), "CREDIT_CARD"),
    (re.compile(r"(?i)birth|\bdob\b"), "DATE_OF_BIRTH"),
    (re.compile(r"(?i)address|\baddr\b|street|ship[_-]?to|deliver"), "ADDRESS"),
    (re.compile(r"(?i)full[_-]?name|first[_-]?name|last[_-]?name|surname|customer[_-]?name|^name$|\bnm\b"), "PERSON"),
    (re.compile(r"(?i)\bip\b|ip[_-]?addr"), "IP_ADDRESS"),
]
HINT_MAX_CHARS = 80
OPEN_VALUES = {"", "***", "anon", "null", "none", "n/a", "redacted"}


def hint_for(key: str) -> str | None:
    for rx, label in KEY_HINTS:
        if rx.search(key or ""):
            return label
    return None


@dataclass
class Result:
    text: str
    mode: str                      # json, csv, text, text-fallback
    changed_leaves: int = 0
    by_hint: int = 0               # leaves cleaned only because of a field name or a column vote
    notes: list[str] = field(default_factory=list)


class StructuredAnonymizer:
    def __init__(self, detect: Detect, transformer: Transformer, use_hints: bool = True, column_vote: bool = True) -> None:
        self.detect, self.tr = detect, transformer
        self.use_hints, self.column_vote = use_hints, column_vote
        self._res = Result("", "text")

    # ---- leaves
    def _clean(self, s: str, hint: str | None = None, voted: bool = False) -> str:
        spans = merge_spans(self.detect(s))
        out = self.tr.apply(s, spans) if spans else s
        stripped = s.strip()
        if hint and (self.use_hints or voted) and stripped.lower() not in OPEN_VALUES and len(stripped) <= HINT_MAX_CHARS:
            # the field name or the column says what this short value is: unless a detector already covered all of it, replace all of it
            covered = sum(sp.end - sp.start for sp in spans)
            if covered < 0.8 * len(stripped):
                self._res.by_hint += 1
                self._res.changed_leaves += 1
                return self.tr.replacement(s, hint)
        if out != s:
            self._res.changed_leaves += 1
        return out

    def _value(self, v, hint: str | None):
        if isinstance(v, str):
            t = v.strip()
            if t[:1] in "{[" and t[-1:] in "}]":
                try:
                    return json.dumps(self._walk(json.loads(t)), ensure_ascii=False)
                except (json.JSONDecodeError, RecursionError):
                    pass
            return self._clean(v, hint)
        return self._walk(v, hint)

    def _walk(self, obj, hint: str | None = None):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                new_key = self._clean(k) if isinstance(k, str) else k
                # a list of {"k": "mail", "v": ...} pairs: the sibling "k" names what "v" holds
                sub_hint = hint_for(k) if isinstance(k, str) else None
                if k in ("v", "value") and isinstance(obj.get("k", obj.get("key")), str):
                    sub_hint = hint_for(obj.get("k", obj.get("key"))) or sub_hint
                out[new_key] = self._value(v, sub_hint)
            return out
        if isinstance(obj, list):
            return [self._value(v, hint) if isinstance(v, str) else self._walk(v, hint) for v in obj]
        return obj

    # ---- formats
    def _json(self, text: str) -> Result | None:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, RecursionError):
            return None
        if not isinstance(data, (dict, list)):
            return None
        self._res = Result("", "json")
        self._res.text = json.dumps(self._walk(data), ensure_ascii=False, indent=2 if "\n" in text.strip() else None)
        return self._res

    @staticmethod
    def _delimiter(text: str) -> str | None:
        lines = [ln for ln in text.splitlines() if ln.strip()][:6]
        if len(lines) < 2:
            return None
        best, best_n = None, 0
        for d in ",;\t|":
            counts = [ln.count(d) for ln in lines]
            n = Counter(counts).most_common(1)[0]
            if n[0] > 0 and n[1] >= 0.8 * len(lines) and n[0] > best_n:
                best, best_n = d, n[0]
        return best

    def _csv(self, text: str) -> Result | None:
        d = self._delimiter(text)
        if not d:
            return None
        rows = list(csv.reader(io.StringIO(text), delimiter=d))
        if len(rows) < 2:
            return None
        header = None
        if not any(re.search(r"\d|@", c) for c in rows[0]) and all(len(c) < 40 for c in rows[0]):
            header, rows = rows[0], rows[1:]
        self._res = Result("", "csv")
        ncols = max(len(r) for r in rows)
        # a column that detection mostly agrees on is that type for every row, including the rows detection missed
        col_label: dict[int, str] = {}
        if self.column_vote:
            for c in range(ncols):
                votes = Counter()
                cells = [r[c] for r in rows if c < len(r) and r[c].strip()]
                for cell in cells:
                    for s in merge_spans(self.detect(cell)):
                        if s.end - s.start >= 0.8 * len(cell.strip()):
                            votes[s.label] += 1
                if cells and votes:
                    label, n = votes.most_common(1)[0]
                    if n >= 2 and n >= 0.4 * len(cells):
                        col_label[c] = label
        out_rows = []
        for r in rows:
            new = []
            for c, cell in enumerate(r):
                named = hint_for(header[c]) if header and c < len(header) and self.use_hints else None
                new.append(self._clean(cell, named or col_label.get(c), voted=named is None and c in col_label) if cell else cell)
            out_rows.append(new)
        buf = io.StringIO()
        w = csv.writer(buf, delimiter=d, lineterminator="\n")
        if header:
            w.writerow([self._clean(h) for h in header])
        w.writerows(out_rows)
        self._res.text = buf.getvalue()
        return self._res

    def run(self, text: str) -> Result:
        """Anonymize `text`. Whatever goes wrong inside a structured path, the plain text pass still runs: it fails closed."""
        for attempt in (self._json, self._csv):
            try:
                res = attempt(text)
            except Exception as e:  # noqa: BLE001 - fail closed: any failure falls through to the text pass
                self._res = Result("", "text-fallback", notes=[f"{attempt.__name__} failed: {type(e).__name__}"])
                continue
            if res is not None:
                return res
        self._res = Result("", "text")
        self._res.text = self._clean(text)
        return self._res
