"""The five transformations a detected span can get, each a different answer to "what should the downstream system see?".

redact        remove: a typed placeholder. Nothing of the value survives, and neither does any link between two mentions.
mask          obscure: same length and punctuation, characters replaced, the last four digits of a number kept. Format survives.
pseudonymize  replace with a keyed stand-in that is the same every time for the same value: joins survive, reversal needs the key.
anonymize     generalize: keep only a coarser fact (the year, the city, the domain, the country code). Irreversible, not linkable.
synthesize    replace with a freshly generated value of the same type that has no relation to the original.

Secrets (API keys, tokens, passwords) are never pseudonymized, masked partly or kept: every technique removes them.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import random
import re

from veilbench.corpus import names as N
from veilbench.corpus import people as P
from veilbench.schema import SECRET_TYPES, Span
from veilbench.transform import fpe
from veilbench.transform.xdoc import NameResolver, fold, parse_mention

MONTHS_LOWER = [m.lower() for m in P.MONTHS]
SYLLABLES = ["ka", "vu", "ri", "len", "mo", "sa", "tel", "dor", "ni", "pha", "zu", "bel", "ran", "ix", "or", "ve", "mar", "quin", "fo", "lis", "tha", "ger", "yo", "dan", "wel", "ur", "pe", "ko", "sil", "ba"]
SUFFIXES = ["son", "ford", "wick", "ley", "ton", "mere", "den", "ham"]
REAL_TOKENS = {fold(t) for pool in N.POOLS.values() for key in ("first", "last") for t in pool[key]}
TECHNIQUES = ["redact", "mask", "pseudonymize", "anonymize", "synthesize"]
DIGIT_TYPES = {"PHONE", "CREDIT_CARD", "NATIONAL_ID", "IBAN"}


def canon_email(s: str) -> str:
    t = s.lower().replace("%40", "@")
    t = re.sub(r"\s*\[at\]\s*|\s*\(at\)\s*", "@", t)
    t = re.sub(r"\s*\[dot\]\s*", ".", t)
    t = re.sub(r"\s+underscore\s+", "_", t)
    t = re.sub(r"\s+dot\s+", ".", t)
    t = re.sub(r"\s+at\s+", "@", t)
    return t.replace(" ", "")


def _digit_words(s: str) -> str | None:
    table = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine".split())}
    table["oh"] = "0"
    words = re.findall(r"[A-Za-z]+", s)
    if words and all(w.lower() in table for w in words):
        return "".join(table[w.lower()] for w in words)
    return None


def mask_surface(surface: str, label: str) -> str:
    """Length and punctuation preserving mask."""
    if label in SECRET_TYPES - {"CREDIT_CARD"}:
        return "".join(ch if ch.isspace() else "*" for ch in surface)
    if label in DIGIT_TYPES:
        idx = [i for i, ch in enumerate(surface) if ch.isalnum()]
        keep = set(idx[-4:]) if label != "NATIONAL_ID" else set()
        return "".join(ch if (not ch.isalnum() or i in keep) else "*" for i, ch in enumerate(surface))
    if label == "PERSON":
        out, new_word = [], True
        for ch in surface:
            if ch.isalpha():
                out.append(ch if new_word else "*")
                new_word = False
            else:
                out.append(ch)
                new_word = not ch.isalnum() and ch not in "'’"
        return "".join(out)
    if label == "EMAIL" and "@" in surface:
        local, dom = surface.split("@", 1)
        return "".join(ch if (not ch.isalnum() or i == 0) else "*" for i, ch in enumerate(local)) + "@" + dom
    return "".join("*" if ch.isalnum() else ch for ch in surface)


def _parse_date(s: str) -> tuple[dt.date, str] | None:
    s = s.strip()
    if m := re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s):
        return dt.date(int(m[1]), int(m[2]), int(m[3])), "iso"
    if m := re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        if a > 12:
            return dt.date(y, b, a), "dmy_slash"
        if b > 12:
            return dt.date(y, a, b), "mdy_slash"
        return dt.date(y, b, a), "dmy_slash" if len(m[1]) == 2 else "mdy_slash_swapped"
    if m := re.fullmatch(r"([A-Za-z]+) (\d{1,2}), (\d{4})", s):
        if m[1].lower() in MONTHS_LOWER:
            return dt.date(int(m[3]), MONTHS_LOWER.index(m[1].lower()) + 1, int(m[2])), "mdy_text"
    if m := re.fullmatch(r"(\d{1,2}) ([A-Za-z]+) (\d{4})", s):
        if m[2].lower() in MONTHS_LOWER:
            return dt.date(int(m[3]), MONTHS_LOWER.index(m[2].lower()) + 1, int(m[1])), "dmy_text"
    return None


def _format_date(d: dt.date, style: str) -> str:
    month = P.MONTHS[d.month - 1]
    return {
        "iso": f"{d.year:04d}-{d.month:02d}-{d.day:02d}", "dmy_slash": f"{d.day:02d}/{d.month:02d}/{d.year}",
        "mdy_slash": f"{d.month}/{d.day}/{d.year}", "mdy_slash_swapped": f"{d.month}/{d.day}/{d.year}",
        "mdy_text": f"{month} {d.day}, {d.year}", "dmy_text": f"{d.day} {month} {d.year}",
    }[style]


def _city_of(address: str) -> str | None:
    for city, *_ in N.CITIES_PH + N.CITIES_US + N.CITIES_ES:
        if city.lower().replace(" ", "") in re.sub(r"[^a-z]", "", address.lower()):
            return city
    return None


class Transformer:
    """Applies one technique to detected spans. Pseudonymization state (the key and the name clusters) lives here."""

    def __init__(self, technique: str, key: bytes = b"veilbench-demo-key-0123456789abcd", seed: int = 0, name_threshold: float = 0.96) -> None:
        if technique not in TECHNIQUES:
            raise ValueError(f"technique must be one of {TECHNIQUES}")
        self.technique = technique
        self.key = (key * 2)[:32] if len(key) < 16 else key[:32]
        self.resolver = NameResolver(name_threshold)
        self.rng = random.Random(f"veilbench-synthetic-{seed}")  # a string seed: never the same stream as a corpus generated with the same integer seed
        self._synth_memo: dict[tuple[str, str], str] = {}
        self._avoid: set[str] = set()

    # ------------------------------------------------------------ helpers
    def _h(self, *parts: str) -> bytes:
        return hmac.new(self.key, "\x1f".join(parts).encode("utf-8"), hashlib.sha256).digest()

    def _fpe_key(self) -> bytes:
        return self._h("fpe-key")[:16]

    def _pseudo_token(self, digest: bytes, kind: str) -> str:
        """A made-up name built from syllables, re-drawn until it is not a real name from any pool, so a pseudonym never collides with a person."""
        salt = 0
        while True:
            d = digest if salt == 0 else hashlib.sha256(digest + bytes([salt])).digest()
            n_syl = 2 + d[0] % 2
            word = "".join(SYLLABLES[d[1 + i] % len(SYLLABLES)] for i in range(n_syl))
            if kind == "last":
                word += SUFFIXES[d[5] % len(SUFFIXES)]
            word = word.capitalize()
            if fold(word) not in REAL_TOKENS:
                return word
            salt += 1

    def _pseudo_person(self, surface: str) -> str:
        cluster = self.resolver.resolve(surface)
        # the two parts are keyed separately, so a lone "Maria" and a later "Maria Santos" agree on the first name
        first = self._pseudo_token(self._h("pgiven", " ".join(cluster.given))[:6], "first")
        last = self._pseudo_token(self._h("pfamily", " ".join(cluster.family))[:6], "last")
        m = parse_mention(surface)
        raw = surface.strip()
        if m.form == "first" or (m.only and m.form == "upper"):
            out = first
        elif m.form == "last_first":
            out = f"{last}, {first}"
        elif m.form == "title_last":
            title = re.match(r"([A-Za-z]+\.?)\s", raw)
            out = f"{title.group(1)} {last}" if title else last
        elif m.form == "initial_last":
            out = f"{first[0]}. {last}"
        elif m.form == "first_initial":
            out = f"{first} {last[0]}."
        else:
            out = f"{first} {last}"
        return out.upper() if raw.isupper() and len(raw) > 3 else out

    def _pseudo_date(self, surface: str) -> str:
        parsed = _parse_date(surface)
        if not parsed:
            return "[DATE_OF_BIRTH]"
        d, style = parsed
        shift = int.from_bytes(self._h("dob", d.isoformat())[:2], "big") % 700 - 350 or 1
        return _format_date(d + dt.timedelta(days=shift), style)

    def _pseudo(self, surface: str, label: str) -> str:
        if label == "PERSON":
            return self._pseudo_person(surface)
        if label == "EMAIL":
            d = self._h("email", canon_email(surface)).hex()
            return f"u{d[:10]}@h{d[10:16]}.example"
        if label == "PHONE":
            words = _digit_words(surface)
            text = surface if words is None else " ".join(words)
            positions = [i for i, ch in enumerate(text) if ch.isdigit()]
            if len(positions) < 8:
                return f"[{label}]"
            tail = positions[-8:]
            enc = fpe.ff1_encrypt(self._fpe_key(), b"phone", [int(text[i]) for i in tail])
            out = list(text)
            for i, dgt in zip(tail, enc):
                out[i] = str(dgt)
            return "".join(out)
        if label == "CREDIT_CARD":
            digits = fpe.ids.digits_only(surface)
            if len(digits) < 13:
                return "[CREDIT_CARD]"
            return fpe.mask_card(self._fpe_key(), surface)
        if label == "IBAN":
            return fpe.mask_iban(self._fpe_key(), surface)
        if label == "NATIONAL_ID":
            dni = fpe.mask_spanish_dni(self._fpe_key(), surface.replace("-", "")) if re.fullmatch(r"\d{8}-?[A-Za-z]", surface) else None
            if dni:
                return dni
            return fpe.mask_digits(self._fpe_key(), b"nid", surface)
        if label == "DATE_OF_BIRTH":
            return self._pseudo_date(surface)
        if label == "IP_ADDRESS":
            d = self._h("ip", surface)
            return f"10.{d[0]}.{d[1]}.{d[2]}"
        if label == "ADDRESS":
            return "ADDR-" + self._h("addr", fold(re.sub(r"[^A-Za-z0-9]+", " ", surface)).strip()).hex()[:10]
        return f"[{label}]"  # secrets (keys, tokens, passwords) have no stand-in: they are removed, never pseudonymized

    def _anon(self, surface: str, label: str) -> str:
        if label == "DATE_OF_BIRTH":
            m = re.search(r"(19|20)(\d)\d", surface)
            return f"{m.group(1)}{m.group(2)}0s" if m else "[DATE_OF_BIRTH]"
        if label == "EMAIL":
            dom = canon_email(surface).split("@")[-1] if "@" in canon_email(surface) else ""
            return f"[EMAIL]@{dom}" if dom else "[EMAIL]"
        if label == "PHONE":
            m = re.match(r"\+(\d{1,3})", surface.strip())
            return f"+{m.group(1)} [PHONE]" if m else "[PHONE]"
        if label == "ADDRESS":
            city = _city_of(surface)
            return f"[ADDRESS in {city}]" if city else "[ADDRESS]"
        if label == "IP_ADDRESS":
            m = re.match(r"(\d+)\.(\d+)\.", surface)
            return f"{m.group(1)}.{m.group(2)}.0.0/16" if m else "[IP_ADDRESS]"
        return f"[{label}]"

    def _synth(self, surface: str, label: str) -> str:
        memo_key = (label, fold(surface))
        if memo_key in self._synth_memo:
            return self._synth_memo[memo_key]
        out = ""
        for _ in range(60):
            out = self._draw(surface, label)
            # a stand-in that happens to contain a real name from this document would leak it: draw again
            if not ({fold(t) for t in re.findall(r"\w+", out)} & self._avoid):
                break
        self._synth_memo[memo_key] = out
        return out

    def _draw(self, surface: str, label: str) -> str:
        rng = self.rng
        p = P.make_person(rng, ["en", "es", "ph", "de", "in", "zh"])
        if label == "PERSON":
            out = P.person_surface(rng, p, parse_mention(surface).form if parse_mention(surface).form in ("full", "first", "last_first") else "full")
        elif label == "EMAIL":
            out = p.email
        elif label == "PHONE":
            out = P.phone_surface(rng, p)[0]
        elif label == "ADDRESS":
            out = p.address
        elif label == "CREDIT_CARD":
            out = p.card
        elif label == "IBAN":
            out = p.iban
        elif label == "NATIONAL_ID":
            out = p.national_id
        elif label == "DATE_OF_BIRTH":
            out = P.dob_surface(rng, p)[0]
        elif label == "IP_ADDRESS":
            out = p.ip
        elif label == "PASSWORD":
            out = P.make_password(rng)
        elif label == "JWT":
            out = P.make_secret(rng, "jwt")[1]
        elif label == "API_KEY":
            out = P.make_secret(rng, "generic")[1]
        else:
            out = f"[{label}]"
        return out

    # ------------------------------------------------------------ api
    def replacement(self, surface: str, label: str) -> str:
        t = self.technique
        if t == "redact":
            return f"[{label}]"
        if t == "mask":
            return mask_surface(surface, label)
        if t == "pseudonymize":
            return self._pseudo(surface, label)
        if t == "anonymize":
            return self._anon(surface, label)
        if label in SECRET_TYPES - {"CREDIT_CARD"}:
            return f"[{label}]"
        return self._synth(surface, label)

    def apply(self, text: str, spans: list[Span]) -> str:
        if self.technique == "synthesize":
            self._synth_memo = {}  # a fresh value per document: nothing links two documents
        self._avoid = {fold(t) for s in spans if s.label == "PERSON" for t in re.findall(r"\w+", text[s.start:s.end])}
        out, pos = [], 0
        for s in sorted(spans):
            if s.start < pos:
                continue
            out.append(text[pos:s.start])
            out.append(self.replacement(text[s.start:s.end], s.label))
            pos = s.end
        out.append(text[pos:])
        return "".join(out)
