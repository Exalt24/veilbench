"""A generated population with quasi-identifiers (zip, birth year, gender) and one sensitive attribute, plus the named list an attacker would hold."""
from __future__ import annotations

import random
from dataclasses import dataclass

from veilbench.corpus import people as P

GENDERS = ["F", "M", "X"]
CONDITIONS = ["asthma", "diabetes", "hypertension", "migraine", "anxiety", "none", "none", "none"]
QIS = ["zip", "birth_year", "gender"]


@dataclass(frozen=True)
class Person:
    pid: int
    name: str
    zip: int
    birth_year: int
    gender: str
    condition: str


def make_population(n: int = 3000, seed: int = 11, zips: int = 60) -> list[Person]:
    """Zip codes are drawn with a skewed weight so a few are crowded and many are small, which is what makes some people unique."""
    rng = random.Random(seed)
    zip_pool = sorted(rng.sample(range(10000, 99999), zips))
    weights = [1.0 / (i + 1) ** 0.8 for i in range(zips)]
    out = []
    for pid in range(n):
        p = P.make_person(rng, ["en", "es", "ph", "de", "in", "zh"])
        out.append(Person(
            pid=pid, name=p.full, zip=rng.choices(zip_pool, weights)[0],
            birth_year=int(min(2005, max(1950, rng.gauss(1983, 14)))), gender=rng.choices(GENDERS, [0.47, 0.47, 0.06])[0],
            condition=rng.choice(CONDITIONS)))
    return out


def qi(p: Person) -> tuple[int, int, str]:
    return (p.zip, p.birth_year, p.gender)
