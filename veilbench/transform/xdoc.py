"""Cross document name resolution: variants of one name map to one pseudonym, close but different names do not.

A mention is parsed into given and family parts (titles dropped, "Last, First" put back in order, initials kept),
then compared with the clusters seen so far by Jaro-Winkler similarity on each part. A mention joins a cluster only if the
family parts and the given parts both clear the threshold, an initial agrees with the first letter of the given name, and
exactly one cluster qualifies. A lone first or last name joins only when exactly one cluster could own it.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

TITLES = {"mr", "ms", "mrs", "dr", "sr", "sra", "dra", "ginoong", "binibining", "g", "bb", "miss", "mx", "prof"}
PARTICLES = {"de", "del", "dela", "la", "los", "san", "van", "von", "da", "di"}


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


def jaro_winkler(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    window = max(max(la, lb) // 2 - 1, 0)
    ma, mb = [False] * la, [False] * lb
    matches = 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - window), min(lb, i + window + 1)):
            if not mb[j] and b[j] == ca:
                ma[i] = mb[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    t, k = 0, 0
    for i in range(la):
        if ma[i]:
            while not mb[k]:
                k += 1
            if a[i] != b[k]:
                t += 1
            k += 1
    jaro = (matches / la + matches / lb + (matches - t / 2) / matches) / 3
    prefix = 0
    for ca, cb in zip(a[:4], b[:4]):
        if ca != cb:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


@dataclass
class Mention:
    given: list[str]
    family: list[str]
    only: str = ""        # set when the mention is a single token, which could be either part
    form: str = "full"    # full, first, last_first, title_last, upper, initial_last, first_initial


def parse_mention(text: str) -> Mention:
    raw = text.strip()
    upper = raw.isupper() and len(raw) > 3
    comma = "," in raw
    toks = [t for t in re.split(r"[\s_.-]+|,", fold(raw)) if t]
    had_title = bool(toks) and toks[0] in TITLES
    toks = [t for t in toks if t not in TITLES] if had_title else toks
    if comma and len(toks) >= 2:
        # "Last, First" or "dela Cruz, Juan": the family part is whatever came before the comma
        before = [t for t in re.split(r"[\s_.-]+", fold(raw.split(",")[0])) if t and t not in TITLES]
        after = [t for t in re.split(r"[\s_.-]+", fold(raw.split(",", 1)[1])) if t]
        return Mention(given=after, family=before, form="last_first")
    if len(toks) == 1:
        return Mention([], [], only=toks[0], form="title_last" if had_title else ("upper" if upper else "first"))
    split = len(toks) - 1
    while split > 1 and toks[split - 1] in PARTICLES:
        split -= 1
    given, family = toks[:split], toks[split:]
    form = "upper" if upper else "full"
    if had_title:
        form = "title_last"
    elif len(given) == 1 and len(given[0]) == 1:
        form = "initial_last"
    elif len(family) == 1 and len(family[0]) == 1:
        form = "first_initial"
    return Mention(given, family, form=form)


def _part_sim(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    ja, jb = " ".join(a), " ".join(b)
    if len(a) == 1 and len(a[0]) == 1:
        return 1.0 if jb.startswith(a[0]) else 0.0
    if len(b) == 1 and len(b[0]) == 1:
        return 1.0 if ja.startswith(b[0]) else 0.0
    best = jaro_winkler(ja, jb)
    if len(a) != len(b):
        # "Silvia" against "Silvia Pablo", or "Navarro" against "Navarro Cabrera": every word of the shorter side must match a word of the longer
        short, long_ = (a, b) if len(a) < len(b) else (b, a)
        if all(len(t) > 1 for t in short):
            best = max(best, min(max(jaro_winkler(t, u) for u in long_) for t in short) * 0.99)
    return best


@dataclass
class Cluster:
    cid: int
    given: list[str]
    family: list[str]
    matchable: bool = True   # a stand-alone cluster made for an ambiguous mention is never a target for later mentions


class NameResolver:
    """Stateful: the first full mention of a person founds a cluster, later mentions join it."""

    def __init__(self, threshold: float = 0.96) -> None:
        self.threshold = threshold
        self.clusters: list[Cluster] = []
        self._memo: dict[str, Cluster] = {}

    def _match(self, m: Mention) -> list[Cluster]:
        out = []
        for c in self.clusters:
            if not c.matchable:
                continue
            if m.only:
                tok = [m.only]
                if max(_part_sim(tok, c.family), _part_sim(tok, c.given)) >= self.threshold:
                    out.append(c)
            elif m.given and m.family:
                fam_ok = _part_sim(m.family, c.family) >= self.threshold if c.family else None
                giv_ok = _part_sim(m.given, c.given) >= self.threshold if c.given else None
                # a cluster founded by a lone first or last name is completed by the first full mention that agrees on the part it has
                if (fam_ok is None or fam_ok) and (giv_ok is None or giv_ok) and (fam_ok or giv_ok):
                    out.append(c)
        return out

    def resolve(self, text: str) -> Cluster:
        """The cluster for a mention. When more than one cluster could own it, it is NOT linked to any of them: a wrong link
        joins two people's records, which is worse than a missed link. The same ambiguous text always gets the same stand-alone cluster."""
        key = fold(text)
        if key in self._memo:
            return self._memo[key]
        m = parse_mention(text)
        hits = self._match(m)
        if len(hits) == 1:
            c = hits[0]
            if m.given and m.family and not (c.given and c.family):
                c.given, c.family = c.given or m.given, c.family or m.family
        elif len(hits) > 1:
            c = self._found(m)
            c.matchable = False
        else:
            c = self._found(m)
        self._memo[key] = c
        return c

    def fit(self, texts: list[str]) -> "NameResolver":
        """Build the clusters from every mention first, complete names before initials and bare names.

        A complete name is a far better anchor than "M. Santos", and streaming order should not decide who joins whom.
        After `fit`, `resolve` on any fitted text returns its cluster, and a new text still joins by the same rules.
        """
        def strong(t: str) -> bool:
            m = parse_mention(t)
            return bool(m.given and m.family and len(m.given[0]) > 1 and len(m.family[0]) > 1)

        unique = list(dict.fromkeys(texts))
        for t in [t for t in unique if strong(t)] + [t for t in unique if not strong(t)]:
            self.resolve(t)
        return self

    def _found(self, m: Mention) -> Cluster:
        lone_family = bool(m.only) and m.form == "title_last"
        c = Cluster(len(self.clusters), m.given or ([m.only] if m.only and not lone_family else []), m.family or ([m.only] if lone_family else []))
        self.clusters.append(c)
        return c
