"""Adversarial re-identification: what an attacker holding a named list (the auxiliary table) can do with each kind of release.

Three attacks, all against the same generated population:

linkage       the release carries exact quasi-identifiers. The attacker looks each row up in the named list; a row with exactly
              one matching name is re-identified. After k-anonymity the attacker keeps every name whose exact values fall inside
              the row's ranges, so a row is only re-identified if one name fits.
membership    the release is a histogram of counts per quasi-identifier cell, built from a random half of the population. The
              attacker asks, for each person who is unique in the named list, "was this person in the data?", and answers
              "yes" if the cell's count (exact or noisy) is at least one half.
synthetic     rows sampled from a noisy histogram belong to nobody. The check is how often a synthetic row happens to match
              exactly one real person, which is a false claim the attacker could not tell from a true one.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from veilbench.privacy.dp import dp_histogram
from veilbench.privacy.kanon import Generalized
from veilbench.privacy.population import Person, qi


def _index(aux: list[Person]) -> dict[tuple, list[int]]:
    idx: dict[tuple, list[int]] = defaultdict(list)
    for p in aux:
        idx[qi(p)].append(p.pid)
    return idx


def linkage_exact(released: list[Person], aux: list[Person]) -> dict:
    idx = _index(aux)
    unique_hits = 0
    guess_prob = 0.0
    for p in released:
        cands = idx.get(qi(p), [])
        if len(cands) == 1 and cands[0] == p.pid:
            unique_hits += 1
        if p.pid in cands:
            guess_prob += 1.0 / len(cands)
    n = len(released)
    return {"rows": n, "reidentified": unique_hits, "reid_rate": round(unique_hits / n, 4), "mean_guess_probability": round(guess_prob / n, 4)}


def linkage_generalized(released: list[Generalized], aux: list[Person]) -> dict:
    by_pid = {p.pid: p for p in aux}
    hits, guess_prob = 0, 0.0
    for r in released:
        cands = [p.pid for p in aux if r.zip[0] <= p.zip <= r.zip[1] and r.birth_year[0] <= p.birth_year <= r.birth_year[1] and p.gender in r.gender]
        if len(cands) == 1 and cands[0] == r.pid:
            hits += 1
        if r.pid in cands:
            guess_prob += 1.0 / len(cands)
    n = len(released)
    assert all(r.pid in by_pid for r in released)
    return {"rows": n, "reidentified": hits, "reid_rate": round(hits / n, 4), "mean_guess_probability": round(guess_prob / n, 4)}


def cell_counts(people: list[Person], cells: list[tuple]) -> np.ndarray:
    index = {c: i for i, c in enumerate(cells)}
    out = np.zeros(len(cells))
    for p in people:
        out[index[qi(p)]] += 1
    return out


def membership_attack(aux: list[Person], epsilons: list[float | None], seed: int = 3, trials: int = 20) -> list[dict]:
    """None means the exact table. Accuracy is over the people who are unique in the named list; always guessing "yes" scores the base rate."""
    cells = sorted({qi(p) for p in aux})
    cell_pos = {c: i for i, c in enumerate(cells)}
    aux_counts = Counter(qi(p) for p in aux)
    targets = [p for p in aux if aux_counts[qi(p)] == 1]
    rows = []
    rng = np.random.default_rng(seed)
    for eps in epsilons:
        accs = []
        for t in range(trials):
            sample_rng = random.Random(seed * 1000 + t)
            members = {p.pid for p in aux if sample_rng.random() < 0.5}
            counts = cell_counts([p for p in aux if p.pid in members], cells)
            released = counts if eps is None else dp_histogram(counts, eps, rng)
            correct = sum((released[cell_pos[qi(p)]] >= 0.5) == (p.pid in members) for p in targets)
            accs.append(correct / len(targets))
        rows.append({"epsilon": eps, "targets": len(targets), "trials": trials, "accuracy": round(float(np.mean(accs)), 4), "base_rate": 0.5})
    return rows


def synthetic_false_claims(synthetic: list[tuple], aux: list[Person]) -> dict:
    idx = _index(aux)
    matched = sum(1 for row in synthetic if len(idx.get(row, [])) == 1)
    return {"synthetic_rows": len(synthetic), "match_exactly_one_real_person": matched, "rate": round(matched / len(synthetic), 4)}
