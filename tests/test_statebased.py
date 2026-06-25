'''Tests for the functional state-based conflict detector (cd.statebased).'''
import numpy as np
import pytest

from cd import ConflictState, detect
from cd.statebased import (
    combine_conflicts,
    horizontal_conflict,
    relative_bearing_distance,
    vertical_conflict,
)

RPZ = 200.0       # horizontal protected-zone radius [m]
HPZ = 50.0        # vertical protected-zone half-height [m]
DTLOOK = 300.0    # look-ahead time [s]


def test_detect_returns_immutable_conflict_state(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    assert isinstance(cs, ConflictState)
    # frozen dataclass -> attributes cannot be rebound.
    with pytest.raises(Exception):
        cs.rpz = np.array([1.0])


def test_head_on_is_a_conflict(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    assert cs.inconf.tolist() == [True, True]
    # Both ordered directions are reported.
    assert ('AC1', 'AC2') in cs.confpairs
    assert ('AC2', 'AC1') in cs.confpairs
    # The unordered set collapses them to a single pair.
    assert cs.confpairs_unique == frozenset({frozenset({'AC1', 'AC2'})})


def test_head_on_cpa_geometry(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    # Closing at 200 m/s over ~5.56 km -> tcpa ~ 27.8 s, dcpa ~ 0.
    assert cs.dist[0] == pytest.approx(5559.7, abs=1.0)
    assert cs.tcpa[0] == pytest.approx(27.8, abs=0.5)
    assert cs.dcpa[0] == pytest.approx(0.0, abs=1.0)
    # Loss of separation precedes CPA.
    assert np.all(cs.tLOS < cs.tcpa)


def test_diverging_is_no_conflict(diverging):
    cs = detect(diverging, diverging, RPZ, HPZ, DTLOOK)
    assert cs.confpairs == []
    assert cs.inconf.tolist() == [False, False]
    assert cs.confpairs_unique == frozenset()


def test_rpz_hpz_dtlookahead_broadcast(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    assert cs.rpz.tolist() == [RPZ, RPZ]
    assert cs.hpz.tolist() == [HPZ, HPZ]
    assert cs.dtlookahead == [DTLOOK, DTLOOK]


def test_relative_bearing_distance_masks_self_pairs(head_on):
    eye = np.eye(head_on.ntraf)
    qdr, dist = relative_bearing_distance(head_on, head_on, eye)
    # The diagonal (self-distance) is pushed to a huge value, never near zero.
    assert dist[0, 0] > 1e8
    assert dist[1, 1] > 1e8
    # Off-diagonal is the true ~5.56 km separation.
    assert dist[0, 1] == pytest.approx(5559.7, abs=1.0)


def test_vertical_separation_with_altitude_offset():
    # Same horizontal track but stacked 1000 m apart vertically -> no vertical
    # conflict because the altitude gap exceeds HPZ and nobody is climbing.
    from conftest import make_traffic
    own = make_traffic([0.0, 0.05], [0, 0], [0, 180], [100, 100],
                       alt=[0.0, 1000.0])
    cs = detect(own, own, RPZ, HPZ, DTLOOK)
    assert cs.inconf.tolist() == [False, False]


def test_horizontal_conflict_flags_closing_pair(head_on):
    eye = np.eye(head_on.ntraf)
    qdr, dist = relative_bearing_distance(head_on, head_on, eye)
    swhorconf, tcpa, dcpa2, tinhor, touthor, vrel, rpz_mat = horizontal_conflict(
        head_on, head_on, qdr, dist, RPZ, eye)
    # The off-diagonal closing pair is flagged horizontally.
    assert bool(swhorconf[0, 1])
    assert bool(swhorconf[1, 0])
    # tLOS (tinhor) is before CPA for a real crossing.
    assert tinhor[0, 1] < tcpa[0, 1]


def test_combine_requires_horizontal_and_vertical_overlap(head_on):
    eye = np.eye(head_on.ntraf)
    qdr, dist = relative_bearing_distance(head_on, head_on, eye)
    swhorconf, tcpa, dcpa2, tinhor, touthor, vrel, _ = horizontal_conflict(
        head_on, head_on, qdr, dist, RPZ, eye)
    dalt, tinver, toutver, _ = vertical_conflict(head_on, head_on, HPZ, eye)
    swconfl, tinconf = combine_conflicts(
        swhorconf, tinhor, touthor, tinver, toutver, DTLOOK, eye)
    # Diagonal is masked; off-diagonal pair survives the intersection.
    assert not swconfl[0, 0]
    assert swconfl[0, 1]
