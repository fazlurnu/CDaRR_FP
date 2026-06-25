'''Tests for the CNS reception model (sim_models.cns.reception_model).'''
import numpy as np
import pytest

from conftest import make_cns_states
from sim_models.cns.reception_model import (
    ReceptionModel,
    ensure_size,
    make_reception,
    p_from_range,
    sample_mask,
    set_pair,
)


# ---------------------------------------------------------------------------
# make_reception
# ---------------------------------------------------------------------------

def test_make_reception_returns_frozen():
    rm = make_reception(0.9)
    assert isinstance(rm, ReceptionModel)
    with pytest.raises(Exception):
        rm.default_prob = 0.5


def test_make_reception_starts_empty():
    rm = make_reception(0.8)
    assert rm.P.shape == (0, 0)


def test_make_reception_rejects_out_of_range():
    with pytest.raises(ValueError):
        make_reception(-0.1)
    with pytest.raises(ValueError):
        make_reception(1.1)


def test_make_reception_accepts_boundary_values():
    make_reception(0.0)
    make_reception(1.0)


# ---------------------------------------------------------------------------
# ensure_size
# ---------------------------------------------------------------------------

def test_ensure_size_builds_correct_matrix():
    rm = ensure_size(make_reception(0.7), 3)
    assert rm.P.shape == (3, 3)
    np.testing.assert_array_equal(np.diag(rm.P), [1.0, 1.0, 1.0])
    # Off-diagonal all equal default_prob.
    off = rm.P[~np.eye(3, dtype=bool)]
    assert np.all(off == 0.7)


def test_ensure_size_same_size_returns_same_instance():
    rm = ensure_size(make_reception(0.5), 4)
    rm2 = ensure_size(rm, 4)
    assert rm2 is rm


def test_ensure_size_grow_preserves_subblock():
    rm = ensure_size(make_reception(0.5), 2)
    rm = set_pair(rm, 0, 1, 0.25)  # custom entry in top-left block
    rm_big = ensure_size(rm, 4)
    assert rm_big.P.shape == (4, 4)
    assert rm_big.P[0, 1] == pytest.approx(0.25)   # preserved
    assert rm_big.P[2, 3] == pytest.approx(0.5)    # new off-diag = default
    np.testing.assert_array_equal(np.diag(rm_big.P), [1.0, 1.0, 1.0, 1.0])


def test_ensure_size_does_not_mutate_original():
    rm = ensure_size(make_reception(0.5), 2)
    original_P = rm.P.copy()
    ensure_size(rm, 4)
    np.testing.assert_array_equal(rm.P, original_P)


# ---------------------------------------------------------------------------
# set_pair
# ---------------------------------------------------------------------------

def test_set_pair_returns_new_instance():
    rm = ensure_size(make_reception(0.9), 3)
    rm2 = set_pair(rm, 0, 1, 0.0)
    assert rm2 is not rm
    assert rm2.P[0, 1] == pytest.approx(0.0)


def test_set_pair_does_not_mutate_original():
    rm = ensure_size(make_reception(0.9), 3)
    original_val = rm.P[0, 1]
    set_pair(rm, 0, 1, 0.0)
    assert rm.P[0, 1] == pytest.approx(original_val)


# ---------------------------------------------------------------------------
# sample_mask
# ---------------------------------------------------------------------------

def test_sample_mask_force_full_all_true():
    rm = make_reception(0.0)  # zero prob, but force_full overrides
    rng = np.random.default_rng(0)
    mask, _ = sample_mask(rm, 4, rng, force_full=True)
    assert mask.shape == (4, 4)
    assert mask.all()


def test_sample_mask_zero_prob_only_diagonal():
    rm = make_reception(0.0)
    rng = np.random.default_rng(0)
    # Run many times: off-diagonal should never be True.
    for _ in range(50):
        mask, rm = sample_mask(rm, 5, rng)
        off_diag = mask[~np.eye(5, dtype=bool)]
        assert not off_diag.any()


def test_sample_mask_diagonal_always_true():
    rm = make_reception(0.0)
    rng = np.random.default_rng(1)
    for _ in range(20):
        mask, rm = sample_mask(rm, 4, rng)
        assert np.all(np.diag(mask))


def test_sample_mask_threads_resized_rm_back():
    # First call grows P from (0,0) -> (3,3); the returned rm reflects that.
    rm = make_reception(0.8)
    rng = np.random.default_rng(0)
    _, rm_out = sample_mask(rm, 3, rng)
    assert rm_out.P.shape == (3, 3)
    assert rm.P.shape == (0, 0)  # original unchanged


def test_sample_mask_empirical_reception_rate():
    # Core semantic claim: the fraction of off-diagonal cells set to True
    # over many ticks should match default_prob.
    prob = 0.7
    n = 10
    rm = make_reception(prob)
    rng = np.random.default_rng(42)
    off_diag_mask = ~np.eye(n, dtype=bool)
    total, received = 0, 0
    for _ in range(2000):
        mask, rm = sample_mask(rm, n, rng)
        total += off_diag_mask.sum()
        received += mask[off_diag_mask].sum()
    empirical = received / total
    assert empirical == pytest.approx(prob, abs=0.01)


def test_sample_mask_asymmetric_pair_rates():
    # set_pair(i,j) and set_pair(j,i) with different probs -> each direction
    # converges to its own configured rate.
    n = 2
    rm = ensure_size(make_reception(0.5), n)
    rm = set_pair(rm, 0, 1, 0.2)   # AC0 receiving AC1: low
    rm = set_pair(rm, 1, 0, 0.9)   # AC1 receiving AC0: high
    rng = np.random.default_rng(7)
    hits_01, hits_10, trials = 0, 0, 5000
    for _ in range(trials):
        mask, rm = sample_mask(rm, n, rng)
        hits_01 += int(mask[0, 1])
        hits_10 += int(mask[1, 0])
    assert hits_01 / trials == pytest.approx(0.2, abs=0.02)
    assert hits_10 / trials == pytest.approx(0.9, abs=0.02)


# ---------------------------------------------------------------------------
# p_from_range
# ---------------------------------------------------------------------------

# Lat offset of 0.005 deg  ≈  557 m  (< 1000 m threshold)
# Lat offset of 0.015 deg  ≈ 1670 m  (> 1000 m threshold)
_WITHIN  = make_cns_states([0.0, 0.005], [0.0, 0.0], [0.0, 0.0], [100.0, 100.0])
_BEYOND  = make_cns_states([0.0, 0.015], [0.0, 0.0], [0.0, 0.0], [100.0, 100.0])
_MAX_RANGE = 1000.0   # metres


def test_p_from_range_within_sets_default_prob():
    rm = p_from_range(_WITHIN, _MAX_RANGE, default_prob=0.9)
    assert rm.P[0, 1] == pytest.approx(0.9)
    assert rm.P[1, 0] == pytest.approx(0.9)


def test_p_from_range_beyond_sets_zero():
    rm = p_from_range(_BEYOND, _MAX_RANGE, default_prob=0.9)
    assert rm.P[0, 1] == pytest.approx(0.0)
    assert rm.P[1, 0] == pytest.approx(0.0)


def test_p_from_range_diagonal_always_one():
    for states in (_WITHIN, _BEYOND):
        rm = p_from_range(states, _MAX_RANGE)
        np.testing.assert_array_equal(np.diag(rm.P), [1.0, 1.0])


def test_p_from_range_boundary_included():
    # A pair whose distance is exactly max_range should be in range (<=).
    # 1000 m north = 1000 / 111320 ≈ 0.008983 deg lat.
    lat_delta = 1000.0 / 111_320.0
    states = make_cns_states([0.0, lat_delta], [0.0, 0.0], [0.0, 0.0],
                             [100.0, 100.0])
    rm = p_from_range(states, 1000.0, default_prob=1.0)
    assert rm.P[0, 1] == pytest.approx(1.0)


def test_p_from_range_max_range_change():
    # Shrinking max_range moves the pair from in-range to out-of-range.
    rm_wide   = p_from_range(_WITHIN, 1000.0)
    rm_narrow = p_from_range(_WITHIN, 400.0)
    assert rm_wide.P[0, 1] == pytest.approx(1.0)
    assert rm_narrow.P[0, 1] == pytest.approx(0.0)


def test_p_from_range_three_aircraft_mixed():
    # Three aircraft: AC0-AC1 within range, AC0-AC2 and AC1-AC2 beyond.
    states = make_cns_states(
        [0.0, 0.003, 0.020],   # lat offsets: ~334 m, ~2226 m from AC0
        [0.0, 0.0,   0.0],
        [0.0, 0.0,   0.0],
        [100.0, 100.0, 100.0],
    )
    rm = p_from_range(states, 1000.0)
    assert rm.P.shape == (3, 3)
    assert rm.P[0, 1] == pytest.approx(1.0)   # within
    assert rm.P[1, 0] == pytest.approx(1.0)   # within
    assert rm.P[0, 2] == pytest.approx(0.0)   # beyond
    assert rm.P[2, 0] == pytest.approx(0.0)   # beyond
    np.testing.assert_array_equal(np.diag(rm.P), [1.0, 1.0, 1.0])


def test_p_from_range_empirical_rate_matches_prob():
    # With default_prob=0.8 for in-range pairs, the empirical reception rate
    # from sample_mask should converge to 0.8.
    rng = np.random.default_rng(99)
    rm = p_from_range(_WITHIN, _MAX_RANGE, default_prob=0.8)
    hits, trials = 0, 5000
    for _ in range(trials):
        mask, rm = sample_mask(rm, 2, rng)
        hits += int(mask[0, 1])
    assert hits / trials == pytest.approx(0.8, abs=0.02)
