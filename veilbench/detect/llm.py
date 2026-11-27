"""An LLM second pass: shown the text with everything already found hidden, it names what is still there.

It sees `[X]` where the other detectors already found something, so it only has to hunt for leftovers and its
tokens go on the part that is hard. Its answers are located in the original text by exact substring and kept as
character offsets; the cache stores offsets and token counts keyed by a hash of the prompt, never the text.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from veilbench.schema import ENTITY_TYPES, Span

URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
PROMPT_VERSION = "v1"
# USD per million tokens, read from console.groq.com/docs/model/openai/gpt-oss-120b on 2026-11-28.
PRICE_IN, PRICE_OUT = 0.15, 0.60

SYSTEM = (
    "You review text for leftover personal data before it is used to train AI models. Values that were already found "
    "are replaced with [X]. Find any REMAINING personal data in the text: names of people, email addresses, phone numbers, "
    "street addresses, national ID or tax numbers, bank account or card numbers, dates of birth, IP addresses, passwords, "
    "API keys and tokens. Reply with JSON only, in this shape: "
    '{"items":[{"text":"<the exact characters as written in the text>","type":"<one of ' + "|".join(ENTITY_TYPES) + '>"}]}. '
    "Do not list order numbers, ticket numbers, version strings, commit hashes, UUIDs, timestamps, company names or [X]. "
    'If nothing remains reply {"items":[]}.'
)


def mask(text: str, known: list[Span]) -> str:
    out, pos = [], 0
    for s in sorted(known):
        if s.start >= pos:
            out.append(text[pos:s.start])
            out.append("[X]")
            pos = s.end
    out.append(text[pos:])
    return "".join(out)


def prompt_key(masked: str) -> str:
    return hashlib.sha256(f"{MODEL}|{PROMPT_VERSION}|{masked}".encode("utf-8")).hexdigest()[:24]


class LLMReviewer:
    name = "llm"

    def __init__(self, cache_path: Path | None = None, api_key: str | None = None, offline: bool = False) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.offline = offline
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path and cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row
        self.calls = 0

    @staticmethod
    def _pace(headers) -> None:
        """Stay inside the token per minute limit: when little is left, wait for the window to reset."""
        try:
            remaining = int(headers.get("x-ratelimit-remaining-tokens", "99999"))
        except ValueError:
            return
        if remaining < 1800:
            reset = headers.get("x-ratelimit-reset-tokens", "")
            m = re.fullmatch(r"(?:(\d+)m)?([\d.]+)s", reset)
            wait = (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else 5.0
            time.sleep(min(wait + 0.3, 30))

    def _call(self, masked: str) -> tuple[list[dict], dict]:
        body = json.dumps({
            "model": MODEL, "temperature": 0, "max_completion_tokens": 300, "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": masked}],
        }).encode("utf-8")
        for attempt in range(8):
            req = urllib.request.Request(URL, data=body, headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json", "User-Agent": "veilbench/0.1"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    headers = resp.headers
                ms = (time.perf_counter() - t0) * 1000
                self._pace(headers)  # the wait is ours, not the model's, so it is not part of the measured latency
                content = data["choices"][0]["message"].get("content") or "{}"
                usage = data.get("usage", {})
                try:
                    items = json.loads(content).get("items", [])
                except json.JSONDecodeError:
                    items = []
                return items, {"ms": round(ms, 1), "in": usage.get("prompt_tokens", 0), "out": usage.get("completion_tokens", 0), "attempts": attempt + 1}
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503):
                    wait = float(e.headers.get("retry-after", "0") or 0) or min(2 ** attempt, 30)
                    time.sleep(min(wait + 0.5, 120))
                    continue
                raise
        raise RuntimeError("the LLM call kept failing after 8 attempts")

    def review(self, text: str, known: list[Span]) -> tuple[list[Span], dict]:
        masked = mask(text, known)
        key = prompt_key(masked)
        row = self.cache.get(key)
        if row is None:
            if self.offline or not self.api_key:
                return [], {"ms": 0.0, "in": 0, "out": 0, "cached": False, "missing": True}
            items, usage = self._call(masked)
            self.calls += 1
            spans = self._locate(text, known, items)
            row = {"key": key, "spans": [[s.start, s.end, s.label] for s in spans], **usage}
            self.cache[key] = row
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        return [Span(a, b, lab, "llm") for a, b, lab in row["spans"]], {k: row.get(k, 0) for k in ("ms", "in", "out")}

    @staticmethod
    def _locate(text: str, known: list[Span], items: list[dict]) -> list[Span]:
        found: list[Span] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            needle, label = str(it.get("text", "")), str(it.get("type", "")).upper()
            if len(needle) < 2 or label not in ENTITY_TYPES or needle == "[X]":
                continue
            for m in re.finditer(re.escape(needle), text):
                a, b = m.start(), m.end()
                if not any(a < k.end and k.start < b for k in known):
                    found.append(Span(a, b, label, "llm"))
        return found


def cost_usd(tokens_in: int, tokens_out: int) -> float:
    return tokens_in * PRICE_IN / 1e6 + tokens_out * PRICE_OUT / 1e6
