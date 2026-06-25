'''Tests for the functional VO conflict resolver (cr.vo).'''
import numpy as np
import pytest
from shapely.geometry import Point

from cd import detect
from cr import ResolutionConfig, vo

RPZ, HPZ, DTLOOK = 200.0, 50.0, 300.0


# -- collision-cone geometry -------------------------------------------------

def test_tangent_points_symmetric_outside_zone():
    own = Point(0.0, 0.0)
    intr = Point(1000.0, 0.0)
    tp1, tp2 = vo.tangent_points(own, intr, rpz=200.0)
    assert tp1 is not None and tp2 is not None
    # The two tangent points are mirror images across the line of sight (x-axis).
    assert tp1.x == pytest.approx(tp2.x)
    assert tp1.y == pytest.approx(-tp2.y)
    # Tangent line is perpendicular to the radius: |tp| == sqrt(d^2 - rpz^2).
    assert np.hypot(tp1.x, tp1.y) == pytest.approx(np.sqrt(1000.0 ** 2 - 200.0 ** 2))


def test_tangent_points_none_inside_zone():
    own = Point(0.0, 0.0)
    intr = Point(100.0, 0.0)   # closer than rpz
    assert vo.tangent_points(own, intr, rpz=200.0) == (None, None)


# -- per-pair resolution -----------------------------------------------------

def test_vo_pair_shape_and_action(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    dv = vo.vo_pair(head_on, head_on, cs, cs.qdr[0], cs.dist[0], 0, 1, 1.05)
    # Horizontal-only resolution: a 2-D velocity change.
    assert dv.shape == (2,)
    assert np.hypot(dv[0], dv[1]) > 0.0


# -- whole-fleet resolution --------------------------------------------------

def test_resolve_changes_heading_to_avoid(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    cfg = ResolutionConfig(resofach=1.05)
    newtrack, newgs, newvs, alt = vo.resolve(cs, head_on, head_on, cfg)
    assert newtrack[0] != pytest.approx(0.0)
    assert newtrack[1] != pytest.approx(180.0)
    assert np.all(newgs <= head_on.perf.vmax)


def test_resolve_diverging_keeps_track(diverging):
    cs = detect(diverging, diverging, RPZ, HPZ, DTLOOK)
    newtrack, *_ = vo.resolve(cs, diverging, diverging, ResolutionConfig())
    np.testing.assert_allclose(newtrack, diverging.trk)
