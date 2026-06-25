'''Shared, pure building blocks for conflict-recovery (resume-navigation).

The recovery models decide which resolved conflicts may be released and which
aircraft should resume their route. This module holds the pieces they share:

* :class:`RecoveryState` — the immutable, explicitly-threaded book-keeping
  (``resopairs`` still before CPA, and each intruder's velocity at conflict
  initiation). Replaces the mutable ``reso.resopairs`` / ``reso._intr_init_vel``.
* small pure maths/geometry helpers.

The impure parts — looking up aircraft indices and commanding waypoint
recovery — are injected as callables (:func:`default_id2idx`,
:func:`default_recover`) so the decision logic stays testable in isolation.
'''
from dataclasses import dataclass, field
from typing import Callable, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class RecoveryState:
    '''Immutable state threaded through the recovery models.

    ``resopairs`` are conflicts that have been resolved but whose CPA is still
    ahead; ``init_vel`` records each intruder's velocity at conflict initiation
    (used by the "intruder reverts" criterion).
    '''
    resopairs: frozenset = frozenset()
    init_vel: Mapping = field(default_factory=dict)


def empty_recovery_state() -> RecoveryState:
    '''A fresh, empty :class:`RecoveryState`.'''
    return RecoveryState(frozenset(), {})


def _val(a, idx):
    '''Best-effort float read of ``a[idx]``; ``None`` if unavailable.'''
    try:
        return float(a[idx])
    except Exception:
        return None


def get_desired_ownship_velocity(ownship, idx, cache) -> Tuple[float, float]:
    '''Desired (pre-resolution) ownship velocity as ``(east, north)`` m/s.

    Prefers ``seltrk``/``selspd``, then ``ap.trk``/``ap.tas``, then the current
    track/ground speed. ``cache`` memoises results within one timestep.
    '''
    if idx in cache:
        return cache[idx]

    trk = None
    if hasattr(ownship, 'seltrk'):
        trk = _val(ownship.seltrk, idx)
    if trk is None and hasattr(ownship, 'ap') and hasattr(ownship.ap, 'trk'):
        trk = _val(ownship.ap.trk, idx)
    if trk is None:
        trk = _val(ownship.trk, idx)

    spd = None
    if hasattr(ownship, 'selspd'):
        spd = _val(ownship.selspd, idx)
    if spd is None and hasattr(ownship, 'ap') and hasattr(ownship.ap, 'tas'):
        spd = _val(ownship.ap.tas, idx)
    if spd is None:
        spd = _val(getattr(ownship, 'gs', None), idx)
    if spd is None:
        spd = float(np.hypot(ownship.gseast[idx], ownship.gsnorth[idx]))

    r = np.radians(trk)
    u = spd * np.sin(r)
    v = spd * np.cos(r)
    cache[idx] = (u, v)
    return u, v


def compute_pair_positions(conf) -> dict:
    '''Map each conflict pair to its ``(dx, dy)`` relative position from qdr/dist.'''
    if len(conf.confpairs) == 0:
        return {}
    q = np.radians(conf.qdr)
    dxs = conf.dist * np.sin(q)
    dys = conf.dist * np.cos(q)
    return dict(zip(conf.confpairs, zip(dxs.tolist(), dys.tolist())))


def get_relative_position(ownship, intruder, idx1, idx2) -> Tuple[float, float]:
    '''Flat-earth east/north displacement (m) from ownship to intruder.'''
    re = 6371000.0
    dlon = float(intruder.lon[idx2] - ownship.lon[idx1])
    dlat = float(intruder.lat[idx2] - ownship.lat[idx1])
    latm = 0.5 * np.radians(float(intruder.lat[idx2] + ownship.lat[idx1]))
    dx = re * np.radians(dlon) * np.cos(latm)
    dy = re * np.radians(dlat)
    return dx, dy


def get_pair_dxdy(conflict, pair_dxdy, ownship, intruder, idx1, idx2) -> Tuple[float, float]:
    '''``(dx, dy)`` for a pair: precomputed value if present, else flat-earth.'''
    if conflict in pair_dxdy:
        dx, dy = pair_dxdy[conflict]
        return float(dx), float(dy)
    return get_relative_position(ownship, intruder, idx1, idx2)


def calculate_dcpa(dx, dy, du, dv) -> Tuple[float, float]:
    '''Closest-point-of-approach distance and time for a single pair.

    ``(dx, dy)`` is the relative position and ``(du, dv)`` the relative
    velocity. Returns ``(dcpa, tcpa)``.
    '''
    dv2 = du * du + dv * dv
    if abs(dv2) < 1e-6:
        dv2 = 1e-6
    tcpa = -(du * dx + dv * dy) / dv2
    dist2 = dx * dx + dy * dy
    dcpa2 = abs(dist2 - tcpa * tcpa * dv2)
    return float(np.sqrt(dcpa2)), float(tcpa)


def anglediff(a, b) -> float:
    '''Smallest signed difference (deg) between two headings.'''
    d = a - b
    if d > 180:
        return anglediff(a, b + 360)
    if d < -180:
        return anglediff(a + 360, b)
    return d


def record_initial_intruder_velocity(state: RecoveryState, conf, intruder,
                                      id2idx) -> Tuple[RecoveryState, set]:
    '''Fold newly-seen conflict pairs into the state, logging intruder velocity.

    Returns ``(new_state, newpairs)``. Pure: a fresh :class:`RecoveryState` is
    returned, the input is untouched.
    '''
    curpairs = set(conf.confpairs)
    newpairs = curpairs - state.resopairs
    resopairs = frozenset(set(state.resopairs) | curpairs)

    init_vel = dict(state.init_vel)
    for pair in newpairs:
        idx1, idx2 = id2idx(pair)
        if idx1 >= 0 and idx2 >= 0:
            init_vel[pair] = (float(intruder.gseast[idx2]),
                              float(intruder.gsnorth[idx2]))

    return RecoveryState(resopairs, init_vel), newpairs


def default_id2idx(pair):
    '''Default index resolver: look the pair up in the live BlueSky traffic.'''
    import bluesky as bs
    return bs.traf.id2idx(pair)


def default_recover(idx):
    '''Default waypoint-recovery side effect against live BlueSky traffic.'''
    import bluesky as bs
    iwpid = bs.traf.ap.route[idx].findact(idx)
    if iwpid != -1:
        bs.traf.ap.route[idx].direct(idx, bs.traf.ap.route[idx].wpname[iwpid])


def apply_active_changes(changeactive: Mapping, active: np.ndarray,
                         recover: Callable = default_recover) -> None:
    '''Write per-aircraft ASAS-active flags and trigger waypoint recovery.

    The single impure step shared by the recovery models: the decision of
    *what* to change is computed purely upstream; here we apply it.
    '''
    for idx, is_active in changeactive.items():
        active[idx] = is_active
        if not is_active:
            recover(idx)
