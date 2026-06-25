'''Tests for the functional conflict-recovery models (crr package).'''
import numpy as np
import pytest

from cd import detect
from crr import (
    RecoveryState,
    empty_recovery_state,
    resumenav_cpa,
    resumenav_double_criteria,
    resumenav_probabilistic_ftr,
)
from crr.common import (
    calculate_dcpa,
    compute_pair_positions,
    get_desired_ownship_velocity,
    get_relative_position,
    record_initial_intruder_velocity,
)

from conftest import make_id2idx, make_recorder, make_traffic

RPZ, HPZ, DTLOOK = 200.0, 50.0, 300.0
RESOFACH = 1.05


# -- common pure helpers -----------------------------------------------------

def test_calculate_dcpa_head_on_is_zero():
    dcpa, tcpa = calculate_dcpa(dx=1000.0, dy=0.0, du=-100.0, dv=0.0)
    assert dcpa == pytest.approx(0.0, abs=1e-9)
    assert tcpa == pytest.approx(10.0)


def test_get_relative_position_north_offset(head_on):
    dx, dy = get_relative_position(head_on, head_on, 0, 1)
    assert dx == pytest.approx(0.0, abs=1.0)
    assert dy == pytest.approx(5559.7, abs=2.0)


def test_compute_pair_positions_matches_relative(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    pair_dxdy = compute_pair_positions(cs)
    assert ('AC1', 'AC2') in pair_dxdy
    dx, dy = pair_dxdy[('AC1', 'AC2')]
    assert np.hypot(dx, dy) == pytest.approx(5559.7, abs=2.0)


def test_get_desired_velocity_falls_back_to_track_gs(head_on):
    # The fake has no seltrk/selspd/ap.trk -> falls back to trk + gs.
    u, v = get_desired_ownship_velocity(head_on, 0, {})
    assert u == pytest.approx(0.0, abs=1e-9)   # trk 0 -> due north
    assert v == pytest.approx(100.0)


def test_record_initial_intruder_velocity_logs_new_pairs(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    id2idx = make_id2idx(head_on, head_on)
    state, newpairs = record_initial_intruder_velocity(
        empty_recovery_state(), cs, head_on, id2idx)
    assert ('AC1', 'AC2') in state.resopairs
    assert ('AC1', 'AC2') in newpairs
    # AC2 flies due south at 100 m/s -> (east, north) = (0, -100).
    eu, ev = state.init_vel[('AC1', 'AC2')]
    assert eu == pytest.approx(0.0, abs=1e-9)
    assert ev == pytest.approx(-100.0)


def test_record_returns_fresh_state_input_untouched(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    id2idx = make_id2idx(head_on, head_on)
    original = empty_recovery_state()
    state, _ = record_initial_intruder_velocity(original, cs, head_on, id2idx)
    assert original.resopairs == frozenset()   # input unchanged
    assert state is not original


# -- CPA recovery ------------------------------------------------------------

def test_cpa_keeps_active_before_cpa(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    active = np.zeros(head_on.ntraf, dtype=bool)
    _, recover = make_recorder()
    id2idx = make_id2idx(head_on, head_on)

    state, delpairs = resumenav_cpa(
        empty_recovery_state(), cs, head_on, head_on, active,
        resofach=RESOFACH, id2idx=id2idx, recover=recover)
    # Still converging -> nobody released, stays active.
    assert delpairs == set()
    assert active[0] and active[1]
    assert ('AC1', 'AC2') in state.resopairs


def test_cpa_releases_past_cpa(diverging):
    cs = detect(diverging, diverging, RPZ, HPZ, DTLOOK)  # confpairs == []
    active = np.array([True, True])
    calls, recover = make_recorder()
    id2idx = make_id2idx(diverging, diverging)

    seeded = RecoveryState(frozenset({('AC1', 'AC2'), ('AC2', 'AC1')}), {})
    state, delpairs = resumenav_cpa(
        seeded, cs, diverging, diverging, active,
        resofach=RESOFACH, id2idx=id2idx, recover=recover)
    # Past CPA, no LOS, not bouncing -> released and recovered.
    assert state.resopairs == frozenset()
    assert not active[0] and not active[1]
    assert 0 in calls and 1 in calls


# -- FTR double-criteria recovery --------------------------------------------

def test_ftr_keeps_active_when_converging(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    active = np.zeros(head_on.ntraf, dtype=bool)
    _, recover = make_recorder()
    id2idx = make_id2idx(head_on, head_on)

    state, delpairs = resumenav_double_criteria(
        empty_recovery_state(), cs, head_on, head_on, active,
        id2idx=id2idx, recover=recover)
    assert ('AC1', 'AC2') not in delpairs
    assert active[0]


def test_ftr_releases_when_clear():
    # Parallel northbound tracks ~11 km apart -> both CPA criteria clear.
    clear = make_traffic([0.0, 0.0], [0.0, 0.1], [0.0, 0.0], [100.0, 100.0])
    cs = detect(clear, clear, RPZ, HPZ, DTLOOK)
    active = np.array([True, True])
    _, recover = make_recorder()
    id2idx = make_id2idx(clear, clear)

    seeded = RecoveryState(frozenset({('AC1', 'AC2')}), {})
    state, delpairs = resumenav_double_criteria(
        seeded, cs, clear, clear, active, id2idx=id2idx, recover=recover)
    assert ('AC1', 'AC2') in delpairs
    assert ('AC1', 'AC2') not in state.resopairs


# -- probabilistic FTR recovery ----------------------------------------------

def test_probabilistic_keeps_active_when_converging(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    active = np.zeros(head_on.ntraf, dtype=bool)
    _, recover = make_recorder()
    id2idx = make_id2idx(head_on, head_on)

    state, delpairs = resumenav_probabilistic_ftr(
        empty_recovery_state(), cs, head_on, head_on, active,
        id2idx=id2idx, recover=recover)
    # P(DCPA > rpz) ~ 0 for a head-on -> not released.
    assert ('AC1', 'AC2') not in delpairs
    assert active[0]


def test_probabilistic_releases_when_clear():
    clear = make_traffic([0.0, 0.0], [0.0, 0.1], [0.0, 0.0], [100.0, 100.0])
    cs = detect(clear, clear, RPZ, HPZ, DTLOOK)
    active = np.array([True, True])
    _, recover = make_recorder()
    id2idx = make_id2idx(clear, clear)

    seeded = RecoveryState(frozenset({('AC1', 'AC2')}), {})
    state, delpairs = resumenav_probabilistic_ftr(
        seeded, cs, clear, clear, active, id2idx=id2idx, recover=recover)
    # Well separated -> P(DCPA > rpz) ~ 1 > threshold -> released.
    assert ('AC1', 'AC2') in delpairs
