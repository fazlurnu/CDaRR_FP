'''Free-to-revert (FTR) conflict recovery — deterministic double criteria.

Releases a resolved conflict only when, for the ownship's *desired* velocity,
the predicted CPA distance exceeds the protected-zone radius under both
assumptions: the intruder keeps its current velocity (criterion 1) and the
intruder reverts to its velocity at conflict initiation (criterion 2).

**Desired-velocity approximations**

The ownship's desired velocity is the pre-resolution route velocity, read from
the autopilot target (``seltrk``/``selspd`` or ``ap.trk``/``ap.tas``); it is
available locally because the ownship runs on the same system.

The intruder's desired velocity — needed for criterion 2 — is *not*
communicated via surveillance.  ADS-L (and ADS-B) broadcasts only the
intruder's instantaneous observed velocity, not its flight-plan intent.  The
FTR formulation therefore approximates the intruder's desired velocity by its
observed velocity at the moment conflict was first detected.  This is the
weakest assumption in the two-criteria rule: if the intruder deviates from
that initial velocity for reasons unrelated to the conflict, criterion 2 may
be over- or under-conservative.

Functional rewrite of the former ``reso``-mutating version.
'''
import numpy as np

from .common import (
    RecoveryState,
    apply_active_changes,
    calculate_dcpa,
    compute_pair_positions,
    default_id2idx,
    default_recover,
    get_desired_ownship_velocity,
    get_pair_dxdy,
    record_initial_intruder_velocity,
)


def resumenav_double_criteria(state: RecoveryState, conf, ownship, intruder,
                              active, **params):
    '''Decide which resolved conflicts to release on the two CPA criteria.

    Uniform recovery interface: ``(state, conf, ownship, intruder, active,
    **params) -> (new_state, delpairs)``. Recognised ``params``:
      ``id2idx``    conflict-pair -> indices resolver (default ``default_id2idx``)
      ``recover``   waypoint-recovery side effect (default ``default_recover``)

    Side effects (writing ``active`` and waypoint recovery) go through the
    injected ``recover`` callable.
    '''
    id2idx  = params.get("id2idx", default_id2idx)
    recover = params.get("recover", default_recover)
    state, _ = record_initial_intruder_velocity(state, conf, intruder, id2idx)

    pair_dxdy = compute_pair_positions(conf)
    vod_cache = {}
    init_vel = dict(state.init_vel)

    delpairs = set()
    changeactive = {}

    for conflict in state.resopairs:
        idx1, idx2 = id2idx(conflict)

        if idx1 < 0:
            delpairs.add(conflict)
            init_vel.pop(conflict, None)
            continue

        if idx2 < 0:
            delpairs.add(conflict)
            init_vel.pop(conflict, None)
            changeactive[idx1] = changeactive.get(idx1, False)
            continue

        dx, dy = get_pair_dxdy(conflict, pair_dxdy, ownship, intruder, idx1, idx2)
        rpz = float(np.max(conf.rpz[[idx1, idx2]]))
        Vo_u, Vo_v = get_desired_ownship_velocity(ownship, idx1, vod_cache)

        Vi_c_u = float(intruder.gseast[idx2])
        Vi_c_v = float(intruder.gsnorth[idx2])

        # Criterion 1: intruder maintains current velocity (Vi,c).
        Dcpa1, _ = calculate_dcpa(dx, dy, Vo_u - Vi_c_u, Vo_v - Vi_c_v)
        crit1 = Dcpa1 > rpz

        # Criterion 2: intruder reverts to its initial velocity (Vi,i).
        Vi_i_u, Vi_i_v = init_vel.get(conflict, (Vi_c_u, Vi_c_v))
        Dcpa2, _ = calculate_dcpa(dx, dy, Vo_u - Vi_i_u, Vo_v - Vi_i_v)
        crit2 = Dcpa2 > rpz

        if crit1 and crit2:
            delpairs.add(conflict)
            init_vel.pop(conflict, None)
            changeactive[idx1] = changeactive.get(idx1, False)
        else:
            changeactive[idx1] = True

    apply_active_changes(changeactive, active, recover)

    new_state = RecoveryState(frozenset(set(state.resopairs) - delpairs), init_vel)
    return new_state, delpairs
