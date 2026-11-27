"""Synthetic quasi-identifier rows sampled from a differentially private histogram.

The histogram is over (zip, five year birth band, gender). It gets Laplace noise, negative cells are set to zero, and rows are
sampled in proportion to what is left, with the birth year drawn uniformly inside its band. No row belongs to a real person.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from veilbench.privacy.dp import dp_histogram
from veilbench.privacy.population import Person

BAND = 5


def band(year: int) -> int:
    return (year // BAND) * BAND


def synthesize(people: list[Person], epsilon: float, n: int | None = None, seed: int = 9) -> list[tuple[int, int, str]]:
    rng = np.random.default_rng(seed)
    cells = sorted({(p.zip, band(p.birth_year), p.gender) for p in people})
    counts = Counter((p.zip, band(p.birth_year), p.gender) for p in people)
    true = np.array([counts[c] for c in cells], dtype=float)
    noisy = dp_histogram(true, epsilon, rng, clamp=True)
    if noisy.sum() == 0:
        noisy = np.ones_like(noisy)
    probs = noisy / noisy.sum()
    picks = rng.choice(len(cells), size=n or len(people), p=probs)
    return [(cells[i][0], cells[i][1] + int(rng.integers(0, BAND)), cells[i][2]) for i in picks]


def marginal_error(people: list[Person], synthetic: list[tuple[int, int, str]]) -> dict:
    """Relative error of two aggregate questions answered from the synthetic rows instead of the real ones."""
    n_real, n_syn = len(people), len(synthetic)
    zr, zs = Counter(p.zip for p in people), Counter(r[0] for r in synthetic)
    big = [z for z, c in zr.items() if c >= 20]
    zip_err = float(np.mean([abs(zs[z] / n_syn - zr[z] / n_real) / (zr[z] / n_real) for z in big])) if big else 0.0
    dr, ds = Counter(band(p.birth_year) for p in people), Counter(band(r[1]) for r in synthetic)
    dec_err = float(np.mean([abs(ds[b] / n_syn - c / n_real) / (c / n_real) for b, c in dr.items() if c >= 20]))
    return {"zip_share_relative_error": round(zip_err, 4), "birth_band_share_relative_error": round(dec_err, 4)}
