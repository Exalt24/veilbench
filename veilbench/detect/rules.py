"""Rule based detection: patterns plus validators (Luhn, IBAN mod 97, DNI and NRIC check letters).

A pattern alone is a candidate; a validator turns a candidate into a finding. That is what keeps order numbers
and version strings out of the results. There is no name gazetteer: a name is found only from a cue (a title,
"my name is", a signature line) or because its parts appear in an e-mail address elsewhere in the same text.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from veilbench import ids
from veilbench.schema import Span

UP = "A-ZÀ-ÖØ-Þ"
NAME_TOKEN = rf"(?:(?:Mr|Ms|Mrs|Dr|Sr|Sra|Dra|Ma|G|Bb|[{UP}])\.|[{UP}][\w'’-]+)"
PARTICLE = r"(?:de|del|dela|de la|de los|la|los|San|van|von)"
NAME = rf"{NAME_TOKEN}(?:\s+(?:{PARTICLE}\s+)?{NAME_TOKEN}){{0,3}}"

JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{10,}")
VENDOR_KEYS = [
    r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}", r"\bAKIA[0-9A-Z]{16}\b", r"\bgh[pousr]_[A-Za-z0-9]{36,}",
    r"\bxox[abpr]-[A-Za-z0-9-]{20,}", r"\bAIza[0-9A-Za-z_-]{35}", r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}",
    r"\bglpat-[A-Za-z0-9_-]{20}", r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}", r"\bhf_[A-Za-z]{30,}",
]
VENDOR_RE = re.compile("|".join(VENDOR_KEYS))
CONTEXT_KEY_RE = re.compile(r"(?i)\b(?:api[_-]?key|secret|token|access[_-]?key|auth(?:orization)?)\b[\"']?\s*[:=]\s*[\"']?(?:Bearer\s+)?([A-Za-z0-9_\-]{20,})")
GENERIC_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9]{32,64}(?![A-Za-z0-9_-])")
PASSWORD_RE = re.compile(r"(?i)\b(?:password|passwd|pwd|contrase[nñ]a)(?:\s+ko)?\s*(?:is|es|ay|:|=)\s*[\"']?([^\s\"'&,;]{4,64})")
EMAIL_RE = re.compile(r"[A-Za-z0-9._+-]+(?:@|%40)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
EMAIL_BRACKET_RE = re.compile(r"(?i)[A-Za-z0-9._+-]+\s*(?:\[at\]|\(at\))\s*[A-Za-z0-9-]+(?:\s*(?:\[dot\]|\.)\s*[A-Za-z0-9-]+)+")
TLDS = "com|org|net|test|example|invalid|local|io|ph|es|de|co|uk|sg|dev|app|info"
EMAIL_WORDS_RE = re.compile(rf"(?i)\b[a-z0-9]+(?: (?:dot|underscore) [a-z0-9]+)* at [a-z0-9-]+(?: dot [a-z0-9-]+)*? dot (?:{TLDS})(?: dot (?:{TLDS}))*\b")
IBAN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?(?![A-Za-z0-9])")
DIGIT_RUN_RE = re.compile(r"(?<![\w.])\d(?:[ -]?\d){11,18}(?![\w])")
SSN_RE = re.compile(r"(?<![\d-])(\d{3})-(\d{2})-(\d{4})(?![\d-])")
SSS_RE = re.compile(r"(?<![\d-])\d{2}-\d{7}-\d(?![\d-])")
PHILSYS_RE = re.compile(r"(?<![\d-])\d{4}-\d{4}-\d{4}-\d{4}(?![\d-])")
TIN_RE = re.compile(r"(?<![\d-])\d{3}-\d{3}-\d{3}-\d{3}(?![\d-])")
DNI_RE = re.compile(r"(?<![A-Za-z0-9-])\d{8}-?[A-Za-z](?![A-Za-z0-9])")
NRIC_RE = re.compile(r"(?<![A-Za-z0-9-])[STst]\d{7}[A-Za-z](?![A-Za-z0-9])")
PHONE_RE = re.compile(r"(?<![\w+.-])\+?\(?\d[\d ().-]{6,17}\d(?![\w-])")
ID_CONTEXT = re.compile(r"(?i)\b(?:id|dni|ssn|nric|tin|sss|philsys|passport)\b\W{0,3}(?:number|no\.?|#)?\s*(?:is|es|ay|:|=)?\s*$")
SPACED_ID_RE = re.compile(r"(?<![\w.])(?:[STst] )?\d(?: \d){6,8}(?: [A-Za-z])?(?![\w])")
WORD_DIGITS = "zero|oh|one|two|three|four|five|six|seven|eight|nine"
PHONE_WORDS_RE = re.compile(rf"(?i)\b(?:(?:{WORD_DIGITS})[ ,-]+){{7,15}}(?:{WORD_DIGITS})\b")
SPACED_DIGITS_RE = re.compile(r"(?<![\w.])\d(?: \d){7,18}(?![\w])")
IP_RE = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\d)(?!\.\d)")
VERSION_CONTEXT = re.compile(r"(?i)(?:version|firmware|build|ver|release|v)\s*(?:is|=|:)?\s*$")
DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}|\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4})\b")
DOB_CONTEXT = re.compile(r"(?i)(?:date of birth|dob|born|birth ?date|birthday|fecha de nacimiento|nac[ií]|nacido|ipinanganak|birth date)")

ADDRESS_RES = [
    re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road|Lane|Ln|Dr|Drive)\b\.?(?:,\s*(?:Apt|Unit|Suite)\s*\d+)?(?:,\s*[A-Z][A-Za-z .-]+,\s*[A-Z]{2}\s+\d{5})?"),
    re.compile(r"\b(?:Blk\s*\d+\s+Lot\s*\d+\s+)?\d{0,4}\s*[A-Z][A-Za-z-]+\s+(?:St\.|Street)(?:,\s*(?:Brgy\.|Barangay)\s+[A-Z][A-Za-z-]+)?(?:,\s*[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+)*)?(?:,\s*Philippines)?(?:\s+\d{4})?"),
    re.compile(r"\b(?:Calle|C/)\s+(?:de\s+)?[A-Z][a-z]+(?:\s+\d{1,3}|,\s*\d{1,3}(?:,\s*\d+º\s*[A-D])?)(?:,\s*\d{5}\s+[A-Z][a-z]+)?"),
]

CUE_RES = [
    re.compile(rf"(?i:\b(?:my name is|me llamo|soy|ako si|ako po si|this is|it['’]s|name:|dear|hi|hello|hola|kumusta|account holder|signed by|greetings from|good day, i am)\s+)({NAME})"),
    re.compile(rf"(?i:\b(?:regards|thanks|thank you|best|gracias|salamat po|un saludo|maraming salamat),?\s*)\n?({NAME})(?=\s*(?:$|\n|<))"),
    re.compile(rf"\b((?:Mr|Ms|Mrs|Dr|Sr|Sra|Dra|G|Bb)\.\s*{NAME_TOKEN}(?:\s+{NAME_TOKEN})?|(?:Ginoong|Binibining)\s+{NAME_TOKEN})"),
    re.compile(rf"(?i:\b(?:name|from|user|customer|cc|to)\s*[:=]\s*\"?)({NAME}(?:,\s*{NAME_TOKEN}(?:\s+{NAME_TOKEN})?)?)"),
]
STOP_NAMES = {"team", "support", "there", "all", "again", "customer", "sir", "madam", "everyone", "help", "your", "the", "a", "an", "po", "ako", "na", "sa", "ang"}


def _entropy(s: str) -> float:
    c = Counter(s)
    return -sum(v / len(s) * math.log2(v / len(s)) for v in c.values())


def _digit_words_to_digits(s: str) -> str:
    table = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine".split())}
    table["oh"] = "0"
    return "".join(table.get(w.lower(), "") for w in re.findall(r"[A-Za-z]+", s))


def _overlaps_any(a: int, b: int, spans: list[Span]) -> bool:
    return any(a < s.end and s.start < b for s in spans)


class RuleDetector:
    name = "rules"

    def detect(self, text: str) -> list[Span]:
        out: list[Span] = []

        def add(a: int, b: int, label: str) -> None:
            if b > a:
                out.append(Span(a, b, label, "rules"))

        for m in JWT_RE.finditer(text):
            add(m.start(), m.end(), "JWT")
        for m in VENDOR_RE.finditer(text):
            if not _overlaps_any(m.start(), m.end(), out):
                add(m.start(), m.end(), "API_KEY")
        for m in CONTEXT_KEY_RE.finditer(text):
            if not _overlaps_any(m.start(1), m.end(1), out):
                add(m.start(1), m.end(1), "API_KEY")
        for m in GENERIC_TOKEN_RE.finditer(text):
            v = m.group(0)
            if _overlaps_any(m.start(), m.end(), out):
                continue
            if not (any(ch.isdigit() for ch in v) and any(ch.isupper() for ch in v)):
                continue  # all lowercase hex (a commit hash) and plain numbers are not secrets
            if _entropy(v) >= 4.0:
                add(m.start(), m.end(), "API_KEY")
        for m in PASSWORD_RE.finditer(text):
            add(m.start(1), m.end(1), "PASSWORD")

        for rx in (EMAIL_RE, EMAIL_BRACKET_RE, EMAIL_WORDS_RE):
            for m in rx.finditer(text):
                if not _overlaps_any(m.start(), m.end(), out):
                    add(m.start(), m.end(), "EMAIL")

        for m in IBAN_RE.finditer(text):
            v = m.group(0)
            while v and not ids.iban_valid(v):
                v = v[:-1].rstrip()
                if len(v) < 15:
                    v = ""
            if v and not _overlaps_any(m.start(), m.start() + len(v), out):
                add(m.start(), m.start() + len(v), "IBAN")

        # digit runs: a Luhn valid run of 13 to 19 digits is a card number
        for m in DIGIT_RUN_RE.finditer(text):
            run = m.group(0)
            digits = ids.digits_only(run)
            if 13 <= len(digits) <= 19 and ids.luhn_valid(digits) and not _overlaps_any(m.start(), m.end(), out):
                add(m.start(), m.end(), "CREDIT_CARD")
        for m in SPACED_ID_RE.finditer(text):
            # an identifier written one character at a time after "ID" or "DNI": a spaced letter at either end marks a check letter
            compact = m.group(0).replace(" ", "")
            if _overlaps_any(m.start(), m.end(), out):
                continue
            if (compact[0].isalpha() and ids.nric_valid(compact)) or (compact[-1].isalpha() and ids.dni_valid(compact)):
                add(m.start(), m.end(), "NATIONAL_ID")
        for m in SPACED_DIGITS_RE.finditer(text):
            digits = ids.digits_only(m.group(0))
            if _overlaps_any(m.start(), m.end(), out):
                continue
            if 13 <= len(digits) <= 19 and ids.luhn_valid(digits):
                add(m.start(), m.end(), "CREDIT_CARD")
            elif ID_CONTEXT.search(text[max(0, m.start() - 20):m.start()]):
                add(m.start(), m.end(), "NATIONAL_ID")
            else:
                add(m.start(), m.end(), "PHONE")
        for m in PHONE_WORDS_RE.finditer(text):
            if _overlaps_any(m.start(), m.end(), out):
                continue
            digits = _digit_words_to_digits(m.group(0))
            if len(digits) >= 13 and ids.luhn_valid(digits):
                add(m.start(), m.end(), "CREDIT_CARD")
            else:
                add(m.start(), m.end(), "NATIONAL_ID" if ID_CONTEXT.search(text[max(0, m.start() - 20):m.start()]) else "PHONE")

        for rx, label in ((PHILSYS_RE, "NATIONAL_ID"), (TIN_RE, "NATIONAL_ID"), (SSS_RE, "NATIONAL_ID")):
            for m in rx.finditer(text):
                if not _overlaps_any(m.start(), m.end(), out):
                    add(m.start(), m.end(), label)
        for m in SSN_RE.finditer(text):
            if ids.ssn_valid(*m.groups()) and not _overlaps_any(m.start(), m.end(), out):
                add(m.start(), m.end(), "NATIONAL_ID")
        for m in DNI_RE.finditer(text):
            if ids.dni_valid(m.group(0)) and not _overlaps_any(m.start(), m.end(), out):
                add(m.start(), m.end(), "NATIONAL_ID")
        for m in NRIC_RE.finditer(text):
            if ids.nric_valid(m.group(0)) and not _overlaps_any(m.start(), m.end(), out):
                add(m.start(), m.end(), "NATIONAL_ID")

        for m in IP_RE.finditer(text):
            if VERSION_CONTEXT.search(text[max(0, m.start() - 20):m.start()]):
                continue
            if not _overlaps_any(m.start(), m.end(), out):
                add(m.start(), m.end(), "IP_ADDRESS")

        for m in PHONE_RE.finditer(text):
            v = m.group(0)
            digits = ids.digits_only(v)
            if not (8 <= len(digits) <= 15) or _overlaps_any(m.start(), m.end(), out):
                continue
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}", v.strip()) or IP_RE.fullmatch(v.strip()):
                continue
            if re.fullmatch(r"\d+", v) and not v.startswith(("+", "0")):
                continue  # a bare digit run such as an epoch timestamp
            if re.fullmatch(r"\d+\.\d+", v):
                continue
            add(m.start(), m.end(), "PHONE")

        for m in DATE_RE.finditer(text):
            before = text[max(0, m.start() - 45):m.start()]
            if DOB_CONTEXT.search(before) and not _overlaps_any(m.start(), m.end(), out):
                add(m.start(), m.end(), "DATE_OF_BIRTH")

        cands = []
        for rx in ADDRESS_RES:
            for m in rx.finditer(text):
                v = m.group(0).rstrip(" ,")
                if len(v) >= 8:
                    cands.append((m.start(), m.start() + len(v)))
        for a, b in sorted(cands, key=lambda c: c[0] - c[1]):  # longest first
            if not _overlaps_any(a, b, out):
                add(a, b, "ADDRESS")

        out.extend(self._persons(text, out))
        return sorted(out)

    def _persons(self, text: str, existing: list[Span]) -> list[Span]:
        found: list[Span] = []
        for rx in CUE_RES:
            for m in rx.finditer(text):
                a, b = m.start(1), m.end(1)
                value = text[a:b]
                first = value.split()[0].lower().strip(".")
                if first in STOP_NAMES:
                    continue
                value = value.rstrip(". ,")
                b = a + len(value)
                if len(value) >= 2 and not _overlaps_any(a, b, existing + found):
                    found.append(Span(a, b, "PERSON", "rules"))
        # a name written into an e-mail address is also a name when it appears elsewhere in the text
        names: set[str] = set()
        for s in existing:
            if s.label == "EMAIL":
                local = re.split(r"@|%40|\[at\]|\(at\)", text[s.start:s.end], flags=re.I)[0]
                parts = [p for p in re.split(r"[._\-]|\s+dot\s+|\s+underscore\s+", local) if p.isalpha() and len(p) >= 3]
                if len(parts) >= 2:
                    names.add(" ".join(parts[:2]))
        for n in names:
            a, b = n.split(" ")[:2]
            rx = re.compile(rf"(?i)\b{re.escape(a)}[ _.-]{re.escape(b)}\b|\b{re.escape(b)},\s*{re.escape(a)}\b")
            for m in rx.finditer(text):
                if not _overlaps_any(m.start(), m.end(), existing + found):
                    found.append(Span(m.start(), m.end(), "PERSON", "rules"))
        return found
