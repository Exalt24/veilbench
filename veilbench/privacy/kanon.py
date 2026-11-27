"""k-anonymity by Mondrian multidimensional partitioning (LeFevre et al., 2006), written from the paper's description.

The quasi-identifiers are recoded into ranges so every released row shares its range tuple with at least k-1 others.
The measured k is the size of the smallest group of identical quasi-identifier tuples.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from veilbench.privacy.population import Person


@dataclass
class Generalized:
    pid: int
    zip: tuple[int, int]            # inclusive range
    birth_year: tuple[int, int]
    gender: frozenset[str]
    condition: str

    def key(self) -> tuple:
        return (self.zip, self.birth_year, tuple(sorted(self.gender)))


def measured_k(keys: list[tuple]) -> int:
    return min(Counter(keys).values()) if keys else 0


def raw_keys(people: list[Person]) -> list[tuple]:
    return [(p.zip, p.birth_year, p.gender) for p in people]


def unique_share(keys: list[tuple]) -> float:
    c = Counter(keys)
    return sum(1 for k in keys if c[k] == 1) / len(keys) if keys else 0.0


def mondrian(people: list[Person], k: int) -> list[Generalized]:
    """Recursive median split on the widest dimension, only where both halves keep at least k rows."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if len(people) < k:
        raise ValueError("fewer rows than k: no k-anonymous release exists")
    zips = sorted({p.zip for p in people})
    span = {"zip": max(len(zips) - 1, 1), "birth_year": max(max(p.birth_year for p in people) - min(p.birth_year for p in people), 1)}
    zip_rank = {z: i for i, z in enumerate(zips)}
    genders = sorted({p.gender for p in people})
    out: list[Generalized] = []

    def finish(part: list[Person]) -> None:
        zs = [p.zip for p in part]
        ys = [p.birth_year for p in part]
        gs = frozenset(p.gender for p in part)
        for p in part:
            out.append(Generalized(p.pid, (min(zs), max(zs)), (min(ys), max(ys)), gs, p.condition))

    def width(part: list[Person]) -> dict[str, float]:
        zr = [zip_rank[p.zip] for p in part]
        yr = [p.birth_year for p in part]
        return {"zip": (max(zr) - min(zr)) / span["zip"], "birth_year": (max(yr) - min(yr)) / span["birth_year"],
                "gender": (len({p.gender for p in part}) - 1) / max(len(genders) - 1, 1)}

    def split(part: list[Person]) -> None:
        for dim, _ in sorted(width(part).items(), key=lambda kv: -kv[1]):
            if _ == 0:
                break
            if dim == "gender":
                # split off the least common gender value if both sides keep k rows
                counts = Counter(p.gender for p in part)
                for g, _n in sorted(counts.items(), key=lambda kv: kv[1]):
                    a = [p for p in part if p.gender == g]
                    b = [p for p in part if p.gender != g]
                    if len(a) >= k and len(b) >= k:
                        split(a)
                        split(b)
                        return
                continue
            key = (lambda p: zip_rank[p.zip]) if dim == "zip" else (lambda p: p.birth_year)
            ordered = sorted(part, key=key)
            mid_val = key(ordered[len(ordered) // 2])
            left = [p for p in ordered if key(p) < mid_val]
            right = [p for p in ordered if key(p) >= mid_val]
            if len(left) < k or len(right) < k:
                # all equal around the median: try splitting just above the median value instead
                left = [p for p in ordered if key(p) <= mid_val]
                right = [p for p in ordered if key(p) > mid_val]
            if len(left) >= k and len(right) >= k:
                split(left)
                split(right)
                return
        finish(part)

    split(list(people))
    return out


def info_loss(rows: list[Generalized], people: list[Person]) -> float:
    """Mean normalised certainty penalty over the three quasi-identifiers: 0 is exact, 1 is everything suppressed."""
    zs = sorted({p.zip for p in people})
    zr = {z: i for i, z in enumerate(zs)}
    ys = [p.birth_year for p in people]
    total = 0.0
    for r in rows:
        zloss = (zr[r.zip[1]] - zr[r.zip[0]]) / max(len(zs) - 1, 1)
        yloss = (r.birth_year[1] - r.birth_year[0]) / max(max(ys) - min(ys), 1)
        gloss = (len(r.gender) - 1) / 2
        total += (zloss + yloss + gloss) / 3
    return total / len(rows)
