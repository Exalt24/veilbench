"""A small statistical detector: a token classifier trained on generated records (CPU, seconds).

Features are the shape of a token and the words around it, never the token's own spelling beyond a short prefix
and suffix, so it can learn "a capitalised word after 'Thanks,' is a name" without memorising a name list.
It is trained on the `train` split (four cultures, some templates, no obfuscation) and measured on `eval`.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier

from veilbench.schema import Record, Span

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def shape(tok: str) -> str:
    s = re.sub(r"[A-ZÀ-ÖØ-Þ]", "X", tok)
    s = re.sub(r"[a-zß-öø-ÿ]", "x", s)
    s = re.sub(r"\d", "d", s)
    return re.sub(r"(.)\1{2,}", r"\1\1", s)  # collapse long runs: Xxxxxxx -> Xxx


def token_features(tokens: list[str], i: int) -> list[str]:
    t = tokens[i]
    low = t.lower()
    f = [
        "bias", f"shape={shape(t)}", f"len={min(len(t), 12)}", f"pre3={low[:3]}", f"suf3={low[-3:]}", f"suf2={low[-2:]}",
        f"title={t[:1].isupper() and t[1:].islower()}", f"upper={t.isupper() and len(t) > 1}", f"digit={t.isdigit()}", f"hasdigit={any(c.isdigit() for c in t)}",
        f"alpha={t.isalpha()}", f"nlower={sum(c.islower() for c in t) >= 6}",
    ]
    for off in (-3, -2, -1, 1, 2, 3):
        j = i + off
        if 0 <= j < len(tokens):
            n = tokens[j]
            f.append(f"w{off}={n.lower()[:12]}")
            f.append(f"s{off}={shape(n)}")
        else:
            f.append(f"w{off}=<edge>")
    if i > 0:
        f.append(f"bi-1={tokens[i - 1].lower()[:10]}|{low[:0]}shape={shape(t)}")
    return f


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def _labels(rec: Record, toks: list[tuple[str, int, int]]) -> list[str]:
    out = []
    for _, a, b in toks:
        lab = "O"
        for s in rec.spans:
            if a < s.end and s.start < b:
                lab = s.label
                break
        out.append(lab)
    return out


class StatDetector:
    name = "stat"

    def __init__(self) -> None:
        self.hasher = FeatureHasher(n_features=2**19, input_type="string", alternate_sign=False)
        self.clf: SGDClassifier | None = None

    def _matrix(self, tokens: list[str]):
        return self.hasher.transform([token_features(tokens, i) for i in range(len(tokens))])

    def fit(self, records: list[Record], epochs: int = 12, seed: int = 0) -> "StatDetector":
        import scipy.sparse as sp
        mats, ys = [], []
        for rec in records:
            toks = tokenize(rec.text)
            if not toks:
                continue
            mats.append(self._matrix([t for t, _, _ in toks]))
            ys.extend(_labels(rec, toks))
        X = sp.vstack(mats).tocsr()
        y = np.array(ys)
        self.clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=epochs, tol=None, random_state=seed, class_weight={c: (1.0 if c == "O" else 4.0) for c in set(ys)})
        self.clf.fit(X, y)
        return self

    def detect(self, text: str) -> list[Span]:
        if self.clf is None:
            raise RuntimeError("fit() or load() first")
        toks = tokenize(text)
        if not toks:
            return []
        labels = self.clf.predict(self._matrix([t for t, _, _ in toks]))
        spans: list[Span] = []
        i = 0
        while i < len(toks):
            lab = labels[i]
            if lab == "O":
                i += 1
                continue
            j = i
            while j + 1 < len(toks) and (labels[j + 1] == lab or (labels[j + 1] == "O" and toks[j + 1][0] in ",.-/'@:" and j + 2 < len(toks) and labels[j + 2] == lab)):
                j += 1
            start, end = toks[i][1], toks[j][2]
            while text[start:end].endswith((",", ".", ":", "-")) and end > start + 1:
                end -= 1
            if any(ch.isalnum() for ch in text[start:end]):
                spans.append(Span(start, end, lab, "stat"))
            i = j + 1
        return spans
