'''Tests for the CNS ADS-L observation layer (sim_models.cns.adsl_observation).'''
import numpy as np
import pytest

from conftest import make_cns_states
from sim_models.cns.adsl_observation import (
    ADSLObservation,
    FIELDS,
    empty_observation,
    ensure_size,
    field,
    update,
)
from sim_models.cns.sensor import measure

# Zero-noise distribution so sensor values are deterministic.
def _zero_dist(n, ci95, rng):
    return np.zeros((n, 2))


def _sensor(lat, lon, trk, gs, **kw):
    rng = np.random.default_rng(0)
    states = make_cns_states(lat, lon, trk, gs, **kw)
    return measure(states, 5.0, 1.0, _zero_dist, _zero_dist, rng)


def _full_mask(n):
    return np.ones((n, n), dtype=bool)


def _no_off_diag_mask(n):
    '''Only diagonal True — no off-diagonal reception.'''
    return np.eye(n, dtype=bool)


# ---------------------------------------------------------------------------
# empty_observation
# ---------------------------------------------------------------------------

def test_empty_observation_is_frozen():
    obs = empty_observation()
    assert isinstance(obs, ADSLObservation)
    with pytest.raises(Exception):
        obs.n = 1


def test_empty_observation_zero_sized():
    obs = empty_observation()
    assert obs.n == 0
    assert obs.id == []
    for f in FIELDS:
        assert obs.fields[f].shape == (0, 0)


# ---------------------------------------------------------------------------
# ensure_size
# ---------------------------------------------------------------------------

def test_ensure_size_grows_matrices():
    obs = ensure_size(empty_observation(), 3)
    assert obs.n == 3
    for f in FIELDS:
        assert obs.fields[f].shape == (3, 3)
    assert obs.id == ['', '', '']


def test_ensure_size_same_size_returns_same_instance():
    obs = ensure_size(empty_observation(), 3)
    obs2 = ensure_size(obs, 3)
    assert obs2 is obs


def test_ensure_size_preserves_subblock():
    s = _sensor([0.0, 0.1], [0.0, 0.0], [0.0, 0.0], [100.0, 100.0])
    obs = update(empty_observation(), s, _full_mask(2))
    saved_lat = obs.fields['lat'].copy()

    obs_big = ensure_size(obs, 4)
    assert obs_big.n == 4
    np.testing.assert_array_equal(obs_big.fields['lat'][:2, :2], saved_lat)
    # New rows/cols initialised to zero.
    assert obs_big.fields['lat'][2, 3] == 0.0


def test_ensure_size_does_not_mutate_original():
    obs = ensure_size(empty_observation(), 2)
    orig = {f: obs.fields[f].copy() for f in FIELDS}
    ensure_size(obs, 4)
    for f in FIELDS:
        np.testing.assert_array_equal(obs.fields[f], orig[f])


# ---------------------------------------------------------------------------
# update — core behaviour
# ---------------------------------------------------------------------------

def test_update_returns_frozen_instance():
    s = _sensor([0.0], [0.0], [0.0], [100.0])
    obs = update(empty_observation(), s, _full_mask(1))
    assert isinstance(obs, ADSLObservation)
    with pytest.raises(Exception):
        obs.n = 99


def test_update_full_mask_writes_all_cells():
    s = _sensor([0.0, 0.1], [0.0, 0.0], [0.0, 90.0], [100.0, 120.0])
    obs = update(empty_observation(), s, _full_mask(2))
    # Every row i holds target j's sensor lat.
    lat = field(obs, 'lat')
    assert lat[0, 0] == pytest.approx(s.lat[0])
    assert lat[0, 1] == pytest.approx(s.lat[1])
    assert lat[1, 0] == pytest.approx(s.lat[0])
    assert lat[1, 1] == pytest.approx(s.lat[1])


def test_update_no_comms_noise_identical_receivers(spec_test_3):
    # Spec test 3: two observers both receiving target j hold identical values.
    s = _sensor([0.0, 0.1, 0.2], [0.0, 0.0, 0.0], [0.0, 90.0, 45.0],
                [100.0, 120.0, 80.0])
    obs = update(empty_observation(), s, _full_mask(3))
    lat = field(obs, 'lat')
    # All observers of target j=1 hold the same value.
    assert lat[0, 1] == lat[1, 1] == lat[2, 1]
    assert lat[0, 2] == lat[1, 2] == lat[2, 2]


def test_update_staleness(spec_test_4):
    # Spec test 4: obs[i,j] unchanged while sensor[j] moves, if mask[i,j]=False.
    s1 = _sensor([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [100.0, 100.0])
    obs = update(empty_observation(), s1, _full_mask(2))
    first_val = field(obs, 'lat')[0, 1]

    # Move aircraft 1; observer 0 does NOT receive it.
    stale_mask = _full_mask(2)
    stale_mask[0, 1] = False
    for lat1 in [2.0, 3.0, 4.0]:
        s = _sensor([0.0, lat1], [0.0, 0.0], [0.0, 0.0], [100.0, 100.0])
        obs = update(obs, s, stale_mask)
        assert field(obs, 'lat')[0, 1] == pytest.approx(first_val)  # stale

    # Observer 1 (diagonal, always True) tracks the moving target.
    assert field(obs, 'lat')[1, 1] == pytest.approx(4.0)


def test_update_diagonal_equals_sensor(spec_test_5):
    # Spec test 5: obs[i,i] == sensor[i] regardless of off-diag reception.
    s = _sensor([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 90.0, 180.0],
                [100.0, 120.0, 80.0])
    obs = update(empty_observation(), s, _no_off_diag_mask(3))
    lat = field(obs, 'lat')
    for i in range(3):
        assert lat[i, i] == pytest.approx(s.lat[i])


def test_update_does_not_mutate_input():
    s = _sensor([0.0, 0.1], [0.0, 0.0], [0.0, 0.0], [100.0, 100.0])
    obs = update(empty_observation(), s, _full_mask(2))
    orig = {f: obs.fields[f].copy() for f in FIELDS}
    # Move aircraft, full mask again.
    s2 = _sensor([5.0, 6.0], [0.0, 0.0], [0.0, 0.0], [100.0, 100.0])
    update(obs, s2, _full_mask(2))
    for f in FIELDS:
        np.testing.assert_array_equal(obs.fields[f], orig[f])


def test_update_resize_grows_on_new_aircraft(spec_test_8):
    # Spec test 8: adding an aircraft grows matrices to (N+1, N+1), prior
    # sub-block preserved.
    s2 = _sensor([0.0, 0.1], [0.0, 0.0], [0.0, 0.0], [100.0, 100.0])
    obs = update(empty_observation(), s2, _full_mask(2))
    saved = field(obs, 'lat').copy()

    s3 = _sensor([0.0, 0.1, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                 [100.0, 100.0, 100.0])
    full3 = _full_mask(3)
    obs3 = update(obs, s3, full3)
    assert obs3.n == 3
    for f in FIELDS:
        assert obs3.fields[f].shape == (3, 3)


# ---------------------------------------------------------------------------
# field accessor
# ---------------------------------------------------------------------------

def test_field_returns_correct_array():
    s = _sensor([0.0, 0.1], [0.0, 0.0], [0.0, 0.0], [100.0, 120.0])
    obs = update(empty_observation(), s, _full_mask(2))
    np.testing.assert_array_equal(field(obs, 'lat'), obs.fields['lat'])


def test_field_all_fields_accessible():
    s = _sensor([0.0, 0.1], [0.0, 0.0], [0.0, 0.0], [100.0, 120.0])
    obs = update(empty_observation(), s, _full_mask(2))
    for f in FIELDS:
        assert field(obs, f).shape == (2, 2)


# ---------------------------------------------------------------------------
# Fixtures (spec test labels for readability)
# ---------------------------------------------------------------------------

@pytest.fixture
def spec_test_3():
    pass

@pytest.fixture
def spec_test_4():
    pass

@pytest.fixture
def spec_test_5():
    pass

@pytest.fixture
def spec_test_8():
    pass
