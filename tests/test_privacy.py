import math
import random

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from veilbench.privacy import dp, kanon, linkage, synth
from veilbench.privacy.population import make_population, qi


@pytest.fixture(scope="module")
def pop():
    return make_population(1200, 11)


@settings(max_examples=25, deadline=None)
@given(st.integers(2, 40), st.integers(0, 500))
def test_k_anonymity_every_released_group_has_at_least_k_rows(k, seed):
    people = make_population(400, seed, zips=20)
    gen = kanon.mondrian(people, k)
    assert len(gen) == len(people)
    assert kanon.measured_k([g.key() for g in gen]) >= k


def test_k_anonymity_ranges_contain_the_true_values_and_nothing_is_dropped(pop):
    by_pid = {p.pid: p for p in pop}
    gen = kanon.mondrian(pop, 5)
    assert {g.pid for g in gen} == set(by_pid)
    for g in gen:
        p = by_pid[g.pid]
        assert g.zip[0] <= p.zip <= g.zip[1] and g.birth_year[0] <= p.birth_year <= g.birth_year[1] and p.gender in g.gender
        assert g.condition == p.condition


def test_measured_k_is_the_smallest_group_before_and_after(pop):
    raw = kanon.raw_keys(pop)
    assert kanon.measured_k(raw) == 1  # some people are unique on zip, birth year and gender
    assert kanon.measured_k([g.key() for g in kanon.mondrian(pop, 10)]) >= 10
    assert kanon.measured_k([]) == 0


def test_a_larger_k_loses_more_information(pop):
    losses = [kanon.info_loss(kanon.mondrian(pop, k), pop) for k in (2, 10, 40)]
    assert losses == sorted(losses) and losses[0] < losses[-1]


def test_k_anonymity_refuses_what_cannot_be_done(pop):
    with pytest.raises(ValueError):
        kanon.mondrian(pop[:3], 5)
    with pytest.raises(ValueError):
        kanon.mondrian(pop, 0)


def test_linkage_attack_succeeds_on_raw_rows_and_fails_after_generalization(pop):
    released = [p for p in pop if p.pid % 2 == 0]
    before = linkage.linkage_exact(released, pop)
    assert before["reid_rate"] > 0.3
    for k in (5, 10):
        after = linkage.linkage_generalized(kanon.mondrian(released, k), pop)
        assert after["reid_rate"] == 0.0
        assert after["mean_guess_probability"] <= 1 / k + 1e-9  # no row can be guessed better than 1 in k
        assert after["mean_guess_probability"] < before["mean_guess_probability"]


@settings(max_examples=30, deadline=None)
@given(st.floats(0.05, 20.0), st.integers(0, 10_000))
def test_laplace_noise_is_unbiased_with_the_theoretical_spread(eps, seed):
    rng = np.random.default_rng(seed)
    draws = np.array([dp.dp_count(0.0, eps, rng) for _ in range(6000)])
    sd = math.sqrt(2) / eps
    assert abs(draws.mean()) < 5 * sd / math.sqrt(len(draws))
    assert abs(np.mean(np.abs(draws)) * eps - 1.0) < 0.1  # mean absolute error is 1 / epsilon


def test_error_curve_matches_theory_and_shrinks_as_epsilon_grows():
    rows = dp.epsilon_error_curve([0.1, 1.0, 10.0], trials=8000)
    for r in rows:
        assert abs(r["mae_measured"] / r["mae_theory"] - 1) < 0.06
        assert abs(r["rmse_measured"] / r["rmse_theory"] - 1) < 0.06
        assert abs(r["bias_measured"]) < 6 * r["rmse_theory"] / math.sqrt(r["trials"])
    assert rows[0]["mae_measured"] > rows[1]["mae_measured"] > rows[2]["mae_measured"]


def test_histogram_noise_is_independent_per_bin_and_clamping_only_when_asked():
    rng = np.random.default_rng(1)
    counts = np.zeros(2000)
    noisy = dp.dp_histogram(counts, 1.0, rng)
    assert (noisy < 0).mean() > 0.4 and abs(noisy.mean()) < 0.15
    assert (dp.dp_histogram(counts, 1.0, rng, clamp=True) >= 0).all()
    with pytest.raises(ValueError):
        dp.dp_histogram(counts, 0.0, rng)
    with pytest.raises(ValueError):
        dp.dp_count(5, -1.0, rng)


def test_membership_attack_is_certain_on_exact_counts_and_near_chance_at_tiny_epsilon(pop):
    rows = {r["epsilon"]: r for r in linkage.membership_attack(pop, [None, 0.05, 5.0], trials=6)}
    assert rows[None]["accuracy"] == 1.0
    assert rows[0.05]["accuracy"] < 0.58
    assert rows[5.0]["accuracy"] > rows[0.05]["accuracy"] + 0.2


def test_synthetic_rows_sample_the_histogram_and_belong_to_nobody(pop):
    rows = synth.synthesize(pop, 5.0, seed=2)
    assert len(rows) == len(pop)
    assert {r[2] for r in rows} <= {"F", "M", "X"}
    err = synth.marginal_error(pop, rows)
    assert err["birth_band_share_relative_error"] < 0.25
    real = {qi(p) for p in pop}
    assert len(set(rows) - real) > 0  # it invents combinations a real person does not have
    assert linkage.synthetic_false_claims(rows, pop)["rate"] < 1.0


def test_population_is_reproducible():
    assert [qi(p) for p in make_population(50, 4)] == [qi(p) for p in make_population(50, 4)]
    assert random.Random(1).random() < 1
