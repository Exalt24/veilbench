"""Differential privacy with the Laplace mechanism: a noisy count, a noisy histogram, and the measured error versus epsilon.

A count has sensitivity 1 (one person changes it by at most 1), so Laplace noise of scale 1/epsilon gives epsilon-DP.
A histogram where each person falls in exactly one bin has sensitivity 1 too (parallel composition), so every bin gets the same scale.
"""
from __future__ import annotations

import math

import numpy as np


def laplace(rng: np.random.Generator, scale: float, size=None):
    if scale <= 0:
        raise ValueError("the scale must be positive")
    return rng.laplace(0.0, scale, size)


def dp_count(true_count: float, epsilon: float, rng: np.random.Generator, sensitivity: float = 1.0) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return float(true_count + laplace(rng, sensitivity / epsilon))


def dp_histogram(counts: np.ndarray, epsilon: float, rng: np.random.Generator, clamp: bool = False) -> np.ndarray:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    noisy = counts.astype(float) + laplace(rng, 1.0 / epsilon, counts.shape)
    return np.maximum(noisy, 0.0) if clamp else noisy


def epsilon_error_curve(epsilons: list[float], trials: int = 20000, true_count: float = 100.0, seed: int = 5) -> list[dict]:
    """Measured error of a noisy count at each epsilon, next to what theory says: MAE = 1/eps, RMSE = sqrt(2)/eps, bias 0."""
    rng = np.random.default_rng(seed)
    rows = []
    for eps in epsilons:
        draws = np.array([dp_count(true_count, eps, rng) for _ in range(trials)])
        err = draws - true_count
        rows.append({
            "epsilon": eps, "trials": trials,
            "mae_measured": round(float(np.mean(np.abs(err))), 3), "mae_theory": round(1.0 / eps, 3),
            "rmse_measured": round(float(np.sqrt(np.mean(err ** 2))), 3), "rmse_theory": round(math.sqrt(2) / eps, 3),
            "bias_measured": round(float(np.mean(err)), 3),
        })
    return rows
