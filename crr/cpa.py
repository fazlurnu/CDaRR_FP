'''CPA-based conflict recovery (resume navigation).

Releases a resolved conflict once it is past the closest point of approach,
provided the aircraft are no longer in horizontal loss of separation and not in
a "bouncing" conflict. Functional rewrite of the former ``reso``-mutating
version.
'''
import numpy as np

from .common import (
    RecoveryState,
    anglediff,
    apply_active_changes,
    default_id2idx,
    default_recover,
    get_relative_position,
)


def _past_cpa(dist, vrel):
    '''True once the pair has passed their closest point of approach.'''
    return bool(np.dot(dist, vrel) < 0.0)


def _hor_los(dist, rpz):
    '''True while horizontal loss of separation persists.'''
    return bool(np.linalg.norm(dist) < rpz)


def _is_bouncing(dist, trk_own, trk_int, rpz, resofach):
    '''True for a bouncing conflict: nearly parallel tracks inside the zone.'''
    return bool(
        abs(anglediff(trk_own, trk_int)) < 30.0
        and np.linalg.norm(dist) < rpz * resofach)


def resumenav_cpa(state: RecoveryState, conf, ownship, intruder, active, **params):
    '''Decide which resolved conflicts to release on a past-CPA criterion.

    Uniform recovery interface: ``(state, conf, ownship, intruder, active,
    **params) -> (new_state, delpairs)``. Recognised ``params``:
      ``resofach``  bounce-check resolution factor (default 1.05)
      ``id2idx``    conflict-pair -> indices resolver (default ``default_id2idx``)
      ``recover``   waypoint-recovery side effect (default ``default_recover``)

    Side effects (writing ``active`` and waypoint recovery) go through the
    injected ``recover`` callable.
    '''
    resofach = params.get("resofach", 1.05)
    id2idx   = params.get("id2idx", default_id2idx)
    recover  = params.get("recover", default_recover)
    resopairs = set(state.resopairs) | set(conf.confpairs)

    delpairs = set()
    changeactive = {}

    for conflict in resopairs:
        idx1, idx2 = id2idx(conflict)
        # Ownship deleted: drop the conflict.
        if idx1 < 0:
            delpairs.add(conflict)
            continue

        if idx2 >= 0:
            # Flat-earth relative position vector (ownship -> intruder).
            dx, dy = get_relative_position(ownship, intruder, idx1, idx2)
            dist = np.array([dx, dy])
            # Relative velocity vector (ownship - intruder).
            vrel = np.array([ownship.gseast[idx1] - intruder.gseast[idx2],
                             ownship.gsnorth[idx1] - intruder.gsnorth[idx2]])
            rpz = float(np.max(conf.rpz[[idx1, idx2]]))

        if idx2 >= 0 and (not _past_cpa(dist, vrel)
                          or _hor_los(dist, rpz)
                          or _is_bouncing(dist, ownship.trk[idx1], intruder.trk[idx2],
                                          rpz, resofach)):
            changeactive[idx1] = True
        else:
            changeactive[idx1] = changeactive.get(idx1, False)
            delpairs.add(conflict)

    apply_active_changes(changeactive, active, recover)

    return RecoveryState(frozenset(resopairs - delpairs), dict(state.init_vel)), delpairs
