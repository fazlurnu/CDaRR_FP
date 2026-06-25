'''End-to-end tests for the CNS coordinator (sim_models.cns.cns).'''
from dataclasses import replace

import numpy as np
import pytest

from conftest import make_cns_states
from sim_models.cns.cns import CNSState, adsl_field, make_cns, ownship_field, step
from sim_models.cns.reception_model import set_pair


def _zero_dist(n, ci95, rng):
    return np.zeros((n, 2))


def _st2(lat0=0.0, lat1=10.0):
    return make_cns_states([lat0, lat1], [0.0, 0.0], [0.0, 90.0],
                           [100.0, 120.0])


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_make_cns_returns_frozen():
    cns = make_cns(5.0, 1.0)
    assert isinstance(cns, CNSState)
    with pytest.raises(Exception):
        cns.pos_ci95 = 10.0


def test_step_returns_new_instance():
    cns = make_cns(5.0, 1.0, seed=0)
    cns2 = step(cns, _st2())
    assert cns2 is not cns
    # Original is unmodified.
    assert cns.first_update_done is False
    assert cns2.first_update_done is True


# ---------------------------------------------------------------------------
# Spec test 5 — diagonal always equals ownship sensor
# ---------------------------------------------------------------------------

def test_diagonal_equals_ownship_field_every_tick():
    cns = make_cns(5.0, 1.0, seed=0)
    for _ in range(5):
        cns = step(cns, _st2())
        lat_adsl = adsl_field(cns, 'lat')
        lat_own = ownship_field(cns, 'lat')
        np.testing.assert_array_equal(np.diag(lat_adsl), lat_own)


# ---------------------------------------------------------------------------
# Spec test 6 — first update seeds all cells; second leaves off-diag stale
# ---------------------------------------------------------------------------

def test_first_update_seeds_all_cells():
    cns = make_cns(0.0, 0.0, reception_prob=0.0, seed=0,
                   pos_dist=_zero_dist, vel_dist=_zero_dist)
    st1 = make_cns_states([5.0, 10.0], [0.0, 0.0], [0.0, 0.0], [100.0, 100.0])
    cns = step(cns, st1)
    lat = adsl_field(cns, 'lat')

    # No NaN; off-diagonal cells carry the seeded sensor values.
    assert not np.any(np.isnan(lat))
    assert lat[0, 1] == pytest.approx(10.0)
    assert lat[1, 0] == pytest.approx(5.0)


def test_second_update_leaves_off_diagonal_stale():
    cns = make_cns(0.0, 0.0, reception_prob=0.0, seed=0,
                   pos_dist=_zero_dist, vel_dist=_zero_dist)
    st1 = make_cns_states([5.0, 10.0], [0.0, 0.0], [0.0, 0.0], [100.0, 100.0])
    cns = step(cns, st1)

    # Move both aircraft; reception_prob=0 so off-diagonal stays stale.
    st2 = make_cns_states([6.0, 11.0], [0.0, 0.0], [0.0, 0.0], [100.0, 100.0])
    cns2 = step(cns, st2)
    lat2 = adsl_field(cns2, 'lat')

    assert lat2[0, 1] == pytest.approx(10.0)   # stale: AC0 never received AC1
    assert lat2[1, 0] == pytest.approx(5.0)    # stale: AC1 never received AC0
    assert lat2[0, 0] == pytest.approx(6.0)    # fresh diagonal: AC0's own new lat
    assert lat2[1, 1] == pytest.approx(11.0)   # fresh diagonal: AC1's own new lat


# ---------------------------------------------------------------------------
# Spec test 2 — time-varying accuracy widens spread
# ---------------------------------------------------------------------------

def test_time_varying_accuracy_widens_spread():
    st = make_cns_states([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [100.0, 100.0])

    cns_tight = make_cns(10.0, 1.0, seed=1)
    cns_loose = make_cns(200.0, 1.0, seed=1)

    tight_lats, loose_lats = [], []
    for _ in range(1000):
        cns_tight = step(cns_tight, st)
        cns_loose = step(cns_loose, st)
        tight_lats.append(ownship_field(cns_tight, 'lat')[0])
        loose_lats.append(ownship_field(cns_loose, 'lat')[0])

    assert np.std(loose_lats) > np.std(tight_lats)


def test_replace_pos_ci95_changes_spread():
    # Changing pos_ci95 via dataclasses.replace mid-run takes effect next tick.
    st = make_cns_states([0.0], [0.0], [0.0], [100.0])
    cns = make_cns(10.0, 1.0, seed=2)

    tight_lats = [ownship_field(step(cns, st), 'lat')[0] for _ in range(500)]
    cns_loose = replace(cns, pos_ci95=300.0)
    loose_lats = [ownship_field(step(cns_loose, st), 'lat')[0] for _ in range(500)]

    assert np.std(loose_lats) > np.std(tight_lats)


# ---------------------------------------------------------------------------
# Spec test 3 — no comms noise: identical receivers hold identical values
# ---------------------------------------------------------------------------

def test_identical_receivers_hold_same_value():
    # With full reception, all observers of the same target j hold the exact
    # same value — no per-observer noise is added.
    cns = make_cns(5.0, 1.0, reception_prob=1.0, seed=0)
    for _ in range(3):
        cns = step(cns, _st2())
        lat = adsl_field(cns, 'lat')
        # Both observers of target j=1 must hold identical values.
        assert lat[0, 1] == lat[1, 1]
        assert lat[0, 0] == lat[1, 0]


# ---------------------------------------------------------------------------
# Spec test 7 — asymmetric reception end-to-end
# ---------------------------------------------------------------------------

def test_asymmetric_reception_end_to_end():
    cns = make_cns(0.0, 0.0, reception_prob=1.0, seed=0,
                   pos_dist=_zero_dist, vel_dist=_zero_dist)
    st0 = make_cns_states([0.0, 10.0], [0.0, 0.0], [0.0, 0.0], [100.0, 100.0])
    cns = step(cns, st0)   # seed: everyone knows everyone

    # AC0 never receives AC1; AC1 always receives AC0.
    rm = set_pair(cns.reception, 0, 1, 0.0)
    rm = set_pair(rm, 1, 0, 1.0)
    cns = replace(cns, reception=rm)
    stale_val = adsl_field(cns, 'lat')[0, 1]   # AC0's last known lat of AC1

    # Move AC1 ten times; AC0 lat stays fixed.
    for lat1 in range(11, 21):
        st = make_cns_states([0.0, float(lat1)], [0.0, 0.0], [0.0, 0.0],
                             [100.0, 100.0])
        cns = step(cns, st)

    # obs[0,1]: AC0 never received AC1 → stale.
    assert adsl_field(cns, 'lat')[0, 1] == pytest.approx(stale_val)
    # obs[1,0]: AC1 always received AC0 → fresh (AC0 lat=0.0).
    assert adsl_field(cns, 'lat')[1, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Spec test 8 — resize: adding an aircraft grows all structures
# ---------------------------------------------------------------------------

def test_resize_add_aircraft():
    cns = make_cns(0.0, 0.0, seed=0,
                   pos_dist=_zero_dist, vel_dist=_zero_dist)
    st2 = make_cns_states([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [100.0, 100.0])
    cns = step(cns, st2)

    # Add a third aircraft.
    st3 = make_cns_states([0.0, 1.0, 2.0], [0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0], [100.0, 100.0, 100.0])
    cns3 = step(cns, st3)

    assert cns3.sensor.n == 3
    assert cns3.obs.n == 3
    assert cns3.reception.P.shape == (3, 3)
    assert adsl_field(cns3, 'lat').shape == (3, 3)
    assert ownship_field(cns3, 'lat').shape == (3,)

    # The original 2-aircraft CNSState is unmodified.
    assert cns.sensor.n == 2
    assert adsl_field(cns, 'lat').shape == (2, 2)
