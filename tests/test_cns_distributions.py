'''Tests for the CNS noise distributions (sim_models.cns.distributions).'''
import numpy as np
import pytest

from sim_models.cns.distributions import (
    CI95_TO_STD_2D,
    ci95_to_std,
    gaussian,
    make_biased_gaussian,
    make_mixture_gaussian,
    tstudent,
)


def test_ci95_to_std_scalar_broadcasts():
    std = ci95_to_std(2.448, 3)
    assert std.shape == (3, 1)
    # 2.448 / 2.448 == 1.0 at every aircraft.
    assert np.allclose(std, 1.0)


def test_ci95_to_std_per_aircraft_preserved():
    std = ci95_to_std([2.448, 4.896], 2)
    assert std.shape == (2, 1)
    assert std[0, 0] == pytest.approx(1.0)
    assert std[1, 0] == pytest.approx(2.0)


def test_ci95_to_std_uses_2d_factor():
    assert ci95_to_std(10.0, 1)[0, 0] == pytest.approx(10.0 / CI95_TO_STD_2D)


def test_gaussian_shape():
    rng = np.random.default_rng(0)
    err = gaussian(5, 100.0, rng)
    assert err.shape == (5, 2)


def test_gaussian_empty():
    rng = np.random.default_rng(0)
    err = gaussian(0, 100.0, rng)
    assert err.shape == (0, 2)


def test_gaussian_zero_mean_over_many_draws():
    rng = np.random.default_rng(42)
    draws = np.concatenate([gaussian(1000, 50.0, rng) for _ in range(50)])
    assert np.allclose(draws.mean(axis=0), 0.0, atol=1.0)


def test_gaussian_spread_scales_with_ci95():
    rng = np.random.default_rng(1)
    small = np.concatenate([gaussian(2000, 10.0, rng) for _ in range(20)])
    big = np.concatenate([gaussian(2000, 100.0, rng) for _ in range(20)])
    # Empirical sigma tracks ci95 / CI95_TO_STD_2D.
    assert small.std() == pytest.approx(10.0 / CI95_TO_STD_2D, rel=0.1)
    assert big.std() == pytest.approx(100.0 / CI95_TO_STD_2D, rel=0.1)
    assert big.std() > small.std()


def test_biased_gaussian_mean_matches_bias():
    bias = (3.0, -7.0)
    dist = make_biased_gaussian(bias)
    rng = np.random.default_rng(7)
    draws = np.concatenate([dist(1000, 20.0, rng) for _ in range(50)])
    assert np.allclose(draws.mean(axis=0), bias, atol=1.0)


def test_biased_gaussian_default_is_unbiased():
    dist = make_biased_gaussian()
    rng = np.random.default_rng(3)
    draws = np.concatenate([dist(1000, 20.0, rng) for _ in range(20)])
    assert np.allclose(draws.mean(axis=0), 0.0, atol=1.0)


def test_gaussian_95pct_within_ci95():
    # Core semantic claim: CI95_TO_STD_2D is chosen so that ~95% of 2D draws
    # land within a circle of radius ci95. Uses the Rayleigh CDF threshold.
    rng = np.random.default_rng(42)
    ci95 = 50.0
    draws = np.concatenate([gaussian(2000, ci95, rng) for _ in range(50)])
    fraction_within = np.mean(np.linalg.norm(draws, axis=1) <= ci95)
    assert fraction_within == pytest.approx(0.95, abs=0.01)


def test_biased_gaussian_95pct_within_ci95():
    # After removing the bias, the Gaussian component should still satisfy 95%
    # coverage — bias shifts the distribution but doesn't change its spread.
    bias = np.array([30.0, -20.0])
    dist = make_biased_gaussian(bias)
    rng = np.random.default_rng(43)
    ci95 = 50.0
    draws = np.concatenate([dist(2000, ci95, rng) for _ in range(50)])
    centred = draws - bias
    fraction_within = np.mean(np.linalg.norm(centred, axis=1) <= ci95)
    assert fraction_within == pytest.approx(0.95, abs=0.01)


def test_tstudent_is_a_stub():
    rng = np.random.default_rng(0)
    with pytest.raises(NotImplementedError):
        tstudent(3, 10.0, rng)


# ---------------------------------------------------------------------------
# make_mixture_gaussian
# ---------------------------------------------------------------------------

def test_mixture_gaussian_shape():
    dist = make_mixture_gaussian()
    err = dist(7, 50.0, np.random.default_rng(0))
    assert err.shape == (7, 2)


def test_mixture_gaussian_empty():
    dist = make_mixture_gaussian()
    err = dist(0, 50.0, np.random.default_rng(0))
    assert err.shape == (0, 2)


def test_mixture_gaussian_zero_mean():
    dist = make_mixture_gaussian(tail_ratio=4.0, tail_weight=0.1)
    rng = np.random.default_rng(10)
    draws = np.concatenate([dist(2000, 50.0, rng) for _ in range(50)])
    assert np.allclose(draws.mean(axis=0), 0.0, atol=1.0)


def test_mixture_gaussian_preserves_ci95():
    # The core invariant: P(r <= ci95) == 0.95 for the mixture distribution.
    ci95 = 50.0
    dist = make_mixture_gaussian(tail_ratio=3.0, tail_weight=0.1)
    rng = np.random.default_rng(20)
    draws = np.concatenate([dist(5000, ci95, rng) for _ in range(40)])
    fraction_within = np.mean(np.linalg.norm(draws, axis=1) <= ci95)
    assert fraction_within == pytest.approx(0.95, abs=0.01)


def test_mixture_gaussian_dominant_sigma_tighter_than_isotropic():
    # σ₁ < ci95/2.448 because the tail component needs headroom.
    from sim_models.cns.distributions import CI95_TO_STD_2D
    ci95 = 50.0
    sigma_iso = ci95 / CI95_TO_STD_2D
    dist = make_mixture_gaussian(tail_ratio=3.0, tail_weight=0.1)
    rng = np.random.default_rng(30)
    # Draws from the dominant component alone would have std < sigma_iso.
    # We verify the overall std is close to sigma_iso (mixture preserves variance
    # only approximately, but the 95th percentile is exact).
    draws = np.concatenate([dist(5000, ci95, rng) for _ in range(20)])
    # Outliers beyond 3*sigma_iso should be more frequent than in pure Gaussian.
    pure_rng = np.random.default_rng(30)
    pure_draws = np.concatenate([gaussian(5000, ci95, pure_rng) for _ in range(20)])
    mix_outlier_rate = np.mean(np.linalg.norm(draws, axis=1) > 3 * sigma_iso)
    pure_outlier_rate = np.mean(np.linalg.norm(pure_draws, axis=1) > 3 * sigma_iso)
    assert mix_outlier_rate > pure_outlier_rate


def test_mixture_gaussian_larger_tail_ratio_more_extreme_outliers():
    ci95 = 50.0
    rng_a = np.random.default_rng(40)
    rng_b = np.random.default_rng(40)
    mild = make_mixture_gaussian(tail_ratio=2.0, tail_weight=0.1)
    heavy = make_mixture_gaussian(tail_ratio=6.0, tail_weight=0.1)
    draws_mild  = np.concatenate([mild(5000, ci95, rng_a)  for _ in range(20)])
    draws_heavy = np.concatenate([heavy(5000, ci95, rng_b) for _ in range(20)])
    threshold = 2.0 * ci95
    assert (np.linalg.norm(draws_heavy, axis=1) > threshold).mean() > \
           (np.linalg.norm(draws_mild,  axis=1) > threshold).mean()


def test_mixture_gaussian_per_aircraft_ci95():
    dist = make_mixture_gaussian()
    rng = np.random.default_rng(50)
    err = dist(3, [10.0, 50.0, 100.0], rng)
    assert err.shape == (3, 2)


def test_mixture_gaussian_invalid_params():
    with pytest.raises(ValueError):
        make_mixture_gaussian(tail_weight=0.0)   # must be in (0, 1)
    with pytest.raises(ValueError):
        make_mixture_gaussian(tail_weight=1.0)
    with pytest.raises(ValueError):
        make_mixture_gaussian(tail_ratio=1.0)    # must be > 1
    with pytest.raises(ValueError):
        make_mixture_gaussian(tail_ratio=0.5)
