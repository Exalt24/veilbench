"""Detector configurations: which components run, how their spans merge, and a resumable component store.

Each component (rules, stat, spacy, gliner) runs once per record and is timed alone; a configuration is a merge of some
components' spans and its latency is the sum of theirs. That is exact because the components do not depend on each other.
The LLM pass is the exception: it reads the text with the configuration's other findings hidden, so it is run on its own.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from veilbench.corpus.generate import generate_corpus
from veilbench.detect.base import merge_spans
from veilbench.detect.llm import LLMReviewer
from veilbench.detect.rules import RuleDetector
from veilbench.detect.stat import StatDetector
from veilbench.schema import Record, Span

CONFIGS: dict[str, list[str]] = {
    "rules": ["rules"],
    "rules+stat": ["rules", "stat"],
    "rules+stat+spacy": ["rules", "stat", "spacy"],
    "rules+stat+gliner": ["rules", "stat", "gliner"],
    "rules+stat+gliner+llm": ["rules", "stat", "gliner", "llm"],
    "spacy-only": ["spacy"],
    "gliner-only": ["gliner"],
}


def build_component(name: str, train_records: int = 1500, train_seed: int = 1):
    if name == "rules":
        return RuleDetector()
    if name == "stat":
        return StatDetector().fit(generate_corpus(train_records, train_seed, "train"))
    if name == "spacy":
        from veilbench.detect.ml import SpacyDetector
        return SpacyDetector()
    if name == "gliner":
        from veilbench.detect.ml import GlinerDetector
        return GlinerDetector()
    raise KeyError(name)


class ComponentStore:
    """Per record, per component: the spans and the milliseconds it took. Appended to a jsonl file so a long run can resume."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.rows: dict[tuple[str, str], dict] = {}
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[(r["rec"], r["comp"])] = r

    def get(self, rec: str, comp: str) -> dict | None:
        return self.rows.get((rec, comp))

    def put(self, rec: str, comp: str, spans: list[Span], ms: float) -> None:
        row = {"rec": rec, "comp": comp, "ms": round(ms, 3), "spans": [[s.start, s.end, s.label] for s in spans]}
        self.rows[(rec, comp)] = row
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")


def run_component(name: str, detector, records: list[Record], store: ComponentStore, max_seconds: float | None = None) -> int:
    """Run `detector` over the records the store does not have yet. Returns how many were done; stops early after `max_seconds`."""
    done, t_start = 0, time.perf_counter()
    for rec in records:
        if store.get(rec.id, name):
            continue
        t0 = time.perf_counter()
        spans = detector.detect(rec.text)
        store.put(rec.id, name, spans, (time.perf_counter() - t0) * 1000)
        done += 1
        if max_seconds is not None and time.perf_counter() - t_start > max_seconds:
            break
    return done


def stored_spans(store: ComponentStore, rec_id: str, comp: str) -> tuple[list[Span], float]:
    row = store.get(rec_id, comp)
    if row is None:
        raise KeyError(f"component {comp} has not been run on {rec_id}")
    return [Span(a, b, lab, comp) for a, b, lab in row["spans"]], row["ms"]


def config_prediction(store: ComponentStore, rec: Record, comps: list[str], llm: LLMReviewer | None = None) -> tuple[list[Span], float, dict]:
    """Merged spans, milliseconds, and the LLM usage (zeros when the configuration has no LLM pass)."""
    spans: list[Span] = []
    ms = 0.0
    for c in comps:
        if c == "llm":
            continue
        s, t = stored_spans(store, rec.id, c)
        spans += s
        ms += t
    t0 = time.perf_counter()
    merged = merge_spans(spans)
    ms += (time.perf_counter() - t0) * 1000
    usage = {"in": 0, "out": 0}
    if "llm" in comps:
        assert llm is not None
        extra, u = llm.review(rec.text, merged)
        merged = merge_spans(merged + extra)
        ms += u["ms"]
        usage = {"in": u["in"], "out": u["out"]}
    return merged, ms, usage
