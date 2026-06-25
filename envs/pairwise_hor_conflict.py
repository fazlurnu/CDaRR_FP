'''Pairwise horizontal conflict environment — BlueSky traffic setup and step.

Creates N = pair_width × pair_height ownship/intruder pairs on a lat/lon grid.
Ownships head north (hdg = 0); intruders use a fixed or random relative heading.

Usage::

    env = make_pairwise_hor_conflict(
        pair_width=3, pair_height=3,
        asas_pzr_m=50.0, dtlookahead=60.0,
        init_speed_ownship=15.0, init_speed_intruder=15.0,
        aircraft_type_ownship="M600",
        start_lat=52.0, start_lon=4.0, delta_lat_lon=0.01,
    )

    # each tick:
    distances = step(env, action)   # np.ndarray, shape (nb_pair,), metres

    # between episodes:
    reset()
'''
from dataclasses import dataclass

import numpy as np
import bluesky as bs
from bluesky.tools import geo
from bluesky.tools.aero import kts

M2NM = 1 / 1852
NM2M = 1852

_DCPA_NM = 0.0
_ALT = 100


@dataclass(frozen=True)
class PairwiseHorConflictEnv:
    '''Immutable config and lookup tables for a pairwise horizontal conflict scenario.

    Created by :func:`make_pairwise_hor_conflict`; passed into :func:`step`
    each tick. BlueSky traffic state is held externally in ``bs.traf``.
    '''
    nb_pair: int
    init_speed_ownship: float
    init_speed_intruder: float
    ownship_ids: tuple
    intruder_ids: tuple
    ownship_idx: tuple
    intruder_idx: tuple
    init_heading: object  # np.ndarray, shape (2 * nb_pair,)


def make_pairwise_hor_conflict(
    pair_width: int,
    pair_height: int,
    asas_pzr_m: float,
    dtlookahead: float,
    init_speed_ownship: float,
    init_speed_intruder: float,
    aircraft_type_ownship: str,
    start_lat: float,
    start_lon: float,
    delta_lat_lon: float,
    aircraft_type_intruder: str = None,
    init_dpsi: float = None,
    simdt_factor: int = 1,
) -> PairwiseHorConflictEnv:
    '''Spawn aircraft in BlueSky and return an immutable env descriptor.

    If *init_dpsi* is given every intruder uses that relative heading; otherwise
    each intruder gets a random heading in [0, 360). If *aircraft_type_intruder*
    is None, intruders share the ownship type.
    '''
    ac_type_int = aircraft_type_ownship if aircraft_type_intruder is None else aircraft_type_intruder
    n = pair_width * pair_height

    if init_dpsi is not None:
        init_heading = np.array([0 if k % 2 == 0 else init_dpsi for k in range(2 * n)])
    else:
        init_heading = np.array([0 if k % 2 == 0 else np.random.randint(0, 360) for k in range(2 * n)])

    bs.settings.asas_pzr = asas_pzr_m * M2NM
    bs.settings.asas_dtlookahead = dtlookahead
    bs.stack.stack(f"DT {bs.settings.simdt * simdt_factor}")

    ownship_ids, intruder_ids = [], []
    ownship_idx, intruder_idx = [], []
    counter = 0
    idx = 0

    for i in range(pair_width):
        for j in range(pair_height):
            ownship_id = f"DRO{counter:03}"
            intruder_id = f"DRI{counter:03}"

            aclat = start_lat + i * delta_lat_lon
            aclon = start_lon + j * delta_lat_lon

            bs.traf.cre(
                acid=ownship_id, actype=aircraft_type_ownship,
                aclat=aclat, aclon=aclon,
                achdg=init_heading[idx], acalt=_ALT, acspd=init_speed_ownship,
            )
            ownship_ids.append(ownship_id)
            ownship_idx.append(idx)
            idx += 1

            bs.traf.creconfs(
                acid=intruder_id, actype=ac_type_int,
                targetidx=bs.traf.id2idx(ownship_id),
                dpsi=init_heading[idx], dcpa=_DCPA_NM,
                tlosh=dtlookahead, spd=init_speed_intruder,
            )
            intruder_ids.append(intruder_id)
            intruder_idx.append(idx)
            idx += 1

            counter += 1

    return PairwiseHorConflictEnv(
        nb_pair=n,
        init_speed_ownship=init_speed_ownship,
        init_speed_intruder=init_speed_intruder,
        ownship_ids=tuple(ownship_ids),
        intruder_ids=tuple(intruder_ids),
        ownship_idx=tuple(ownship_idx),
        intruder_idx=tuple(intruder_idx),
        init_heading=init_heading,
    )


def step(env: PairwiseHorConflictEnv, action) -> np.ndarray:
    '''Apply *action*, advance BlueSky one tick, return ownship–intruder distances (m).

    Returns a ``(nb_pair,)`` array.
    '''
    _apply_action(env, action)
    bs.sim.step()
    return _compute_distances(env)


def reset() -> None:
    '''Clear all BlueSky traffic. Call between episodes.'''
    bs.traf.reset()


def avoidance_mask(action) -> np.ndarray:
    '''Per-aircraft avoidance flags for *action* — 1.0 if the aircraft is in an
    active resolution pair (currently manoeuvring to avoid), else 0.0.

    Order matches ``bs.traf.id`` (and the lat/lon/gs/hdg state arrays).
    '''
    resopairs = action[4] if action is not None else None
    mask = np.zeros(bs.traf.ntraf, dtype=float)
    if resopairs:
        for i in range(bs.traf.ntraf):
            if any(bs.traf.id[i] in pair for pair in resopairs):
                mask[i] = 1.0
    return mask


def _apply_action(env: PairwiseHorConflictEnv, action) -> None:
    reso_hdg = reso_spd = None
    if action is not None:
        reso_hdg, reso_spd, _, _, _ = action

    avoiding = avoidance_mask(action)
    for i in range(bs.traf.ntraf):
        target_id = bs.traf.id[i]
        if avoiding[i]:
            bs.stack.stack(f"HDG {target_id}, {reso_hdg[i]}")
            bs.stack.stack(f"SPD {target_id}, {reso_spd[i] / kts}")
        else:
            bs.stack.stack(f"HDG {target_id}, {env.init_heading[i]}")
            nom_spd = env.init_speed_ownship if target_id.startswith("DRO") else env.init_speed_intruder
            bs.stack.stack(f"SPD {target_id}, {nom_spd / kts}")


def _compute_distances(env: PairwiseHorConflictEnv) -> np.ndarray:
    lat_own = np.array([bs.traf.lat[i] for i in env.ownship_idx])
    lon_own = np.array([bs.traf.lon[i] for i in env.ownship_idx])
    lat_int = np.array([bs.traf.lat[i] for i in env.intruder_idx])
    lon_int = np.array([bs.traf.lon[i] for i in env.intruder_idx])

    dist = geo.latlondist_matrix(
        np.asmatrix(lat_own), np.asmatrix(lon_own),
        np.asmatrix(lat_int), np.asmatrix(lon_int),
    )
    return np.diag(dist) * NM2M
