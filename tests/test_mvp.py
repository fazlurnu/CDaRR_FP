'''Tests for the functional MVP conflict resolver (cr.mvp).'''
import numpy as np
import pytest

from cd import detect
from cr import ResolutionConfig, mvp

RPZ, HPZ, DTLOOK = 200.0, 50.0, 300.0


# -- per-pair resolution -----------------------------------------------------

def test_mvp_pair_shape_and_horizontal_action(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    dv = mvp.mvp_pair(head_on, head_on, cs, cs.qdr[0], cs.dist[0],
                      cs.tcpa[0], 0, 1, 1.0)
    # Horizontal-only resolution: a 2-D velocity change.
    assert dv.shape == (2,)
    # A real horizontal conflict produces a non-trivial maneuver.
    assert np.hypot(dv[0], dv[1]) > 0.0


# -- whole-fleet resolution --------------------------------------------------

def test_resolve_changes_heading_to_avoid(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    cfg = ResolutionConfig(resofach=1.05)
    newtrack, newgs, newvs, alt = mvp.resolve(cs, head_on, head_on, cfg)
    # Headings are nudged away from the original 0 / 180 deg.
    assert newtrack[0] != pytest.approx(0.0)
    assert newtrack[1] != pytest.approx(180.0)
    # Ground speed stays within the performance envelope.
    assert np.all(newgs >= head_on.perf.vmin)
    assert np.all(newgs <= head_on.perf.vmax)


def test_resolve_diverging_keeps_track(diverging):
    cs = detect(diverging, diverging, RPZ, HPZ, DTLOOK)
    cfg = ResolutionConfig()
    newtrack, newgs, newvs, alt = mvp.resolve(cs, diverging, diverging, cfg)
    # No conflicts -> headings unchanged.
    np.testing.assert_allclose(newtrack, diverging.trk)


def test_resolve_speed_capping_respects_vmax(head_on):
    cs = detect(head_on, head_on, RPZ, HPZ, DTLOOK)
    head_on.perf.vmax = np.array([100.0, 100.0])  # tight cap
    cfg = ResolutionConfig()
    _, newgs, _, _ = mvp.resolve(cs, head_on, head_on, cfg)
    assert np.all(newgs <= 100.0 + 1e-9)
