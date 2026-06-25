'''Tests for the CNS sensor layer (sim_models.cns.sensor).'''
import numpy as np
import pytest

from conftest import make_cns_states
from sim_models.cns.sensor import SensorState, measure


def zero_dist(n, ci95, rng):
    '''A noise-free distribution for the deterministic pass-through tests.'''
    return np.zeros((n, 2))


def _states():
    return make_cns_states(
        lat=[0.0, 0.05], lon=[0.0, 0.10], trk=[0.0, 90.0], gs=[100.0, 120.0],
        alt=[300.0, 600.0], vs=[1.0, -1.0],
    )


def test_measure_returns_immutable_sensor_state():
    s = measure(_states(), 5.0, 1.0, zero_dist, zero_dist, np.random.default_rng(0))
    assert isinstance(s, SensorState)
    with pytest.raises(Exception):
        s.lat = np.array([1.0])


def test_jitter_redrawn_each_tick():
    # Two measurements of identical truth differ (fresh draw every call).
    st = _states()
    rng = np.random.default_rng(0)
    a = measure(st, 50.0, 5.0, rng=rng)
    b = measure(st, 50.0, 5.0, rng=rng)
    assert not np.allclose(a.lat, b.lat)
    assert not np.allclose(a.lon, b.lon)


def test_mean_measurement_is_unbiased():
    st = _states()
    rng = np.random.default_rng(1)
    lats = np.array([measure(st, 50.0, 5.0, rng=rng).lat for _ in range(4000)])
    # Averaged over many draws the position error vanishes -> truth lat.
    assert np.allclose(lats.mean(axis=0), st.lat, atol=1e-4)


def test_larger_pos_ci95_widens_spread():
    st = _states()
    rng = np.random.default_rng(2)
    tight = np.array([measure(st, 10.0, 5.0, rng=rng).lat for _ in range(2000)])
    loose = np.array([measure(st, 200.0, 5.0, rng=rng).lat for _ in range(2000)])
    assert loose[:, 0].std() > tight[:, 0].std()


def test_passthrough_fields_equal_truth():
    st = _states()
    s = measure(st, 5.0, 1.0, zero_dist, zero_dist, np.random.default_rng(0))
    assert np.array_equal(s.alt, st.alt)
    assert np.array_equal(s.hdg, st.hdg)
    assert np.array_equal(s.trk, st.trk)
    assert np.array_equal(s.gs, st.gs)
    assert np.array_equal(s.tas, st.tas)
    assert np.array_equal(s.vs, st.vs)
    assert s.id == st.id


def test_velocity_components_reconstructed_without_noise():
    st = _states()
    s = measure(st, 5.0, 1.0, zero_dist, zero_dist, np.random.default_rng(0))
    trk_rad = np.deg2rad(st.trk)
    assert np.allclose(s.gsnorth, st.gs * np.cos(trk_rad))
    assert np.allclose(s.gseast, st.gs * np.sin(trk_rad))


def test_pos_acc_records_broadcast_ci95():
    st = _states()
    s = measure(st, 42.0, 7.0, zero_dist, zero_dist, np.random.default_rng(0))
    assert np.array_equal(s.pos_acc, [42.0, 42.0])
    assert np.array_equal(s.vel_acc, [7.0, 7.0])


def test_per_aircraft_accuracy_preserved():
    st = _states()
    s = measure(st, [10.0, 20.0], [1.0, 2.0], zero_dist, zero_dist,
                np.random.default_rng(0))
    assert np.array_equal(s.pos_acc, [10.0, 20.0])
    assert np.array_equal(s.vel_acc, [1.0, 2.0])


def test_does_not_alias_truth_arrays():
    st = _states()
    s = measure(st, 5.0, 1.0, zero_dist, zero_dist, np.random.default_rng(0))
    s.alt[0] = 9999.0
    assert st.alt[0] == 300.0  # truth untouched


def test_resize_tracks_ntraf():
    rng = np.random.default_rng(0)
    one = measure(make_cns_states([0.0], [0.0], [0.0], [100.0]), 5.0, 1.0, rng=rng)
    assert one.n == 1 and one.lat.shape == (1,)
    three = measure(
        make_cns_states([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                        [100.0, 100.0, 100.0]),
        5.0, 1.0, rng=rng)
    assert three.n == 3 and three.gsnorth.shape == (3,)


def test_empty_traffic():
    s = measure(make_cns_states([], [], [], []), 5.0, 1.0, rng=np.random.default_rng(0))
    assert s.n == 0
    assert s.lat.shape == (0,)
    assert s.id == []
