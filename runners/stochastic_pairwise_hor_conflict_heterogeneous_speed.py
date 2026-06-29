'''Stochastic pairwise horizontal conflict runner — heterogeneous per-pair speeds.

Identical to :mod:`runners.stochastic_pairwise_hor_conflict` except that each
pair independently draws its ownship and intruder speed from
Uniform(speed_min, speed_max) at the start of every run.  The ``speed_seed``
is derived automatically from the main ``seed`` so that speed draws and CNS
noise draws are independent but both reproducible.

Usage::

    res = run_single(
        pair_width=100, pair_height=100,
        rpz=50.0, hpz=50.0, dtlookahead=121.0,
        speed_min=10.0, speed_max=30.0,
        aircraft_type="M600", dpsi=90.0,
        pos_ci95=10.0, vel_ci95=1.0, reception_prob=1.0,
        record_history=True,
    )

    results = run_parallel(
        n_runs=5, n_jobs=8,
        pair_width=100, pair_height=100,
        rpz=50.0, hpz=50.0, dtlookahead=121.0,
        speed_min=10.0, speed_max=30.0,
        aircraft_type="M600", dpsi=90.0,
        pos_ci95=10.0, vel_ci95=1.0, reception_prob=1.0,
    )
'''
import os
import sys
import contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import numpy as np
from joblib import Parallel, delayed
import bluesky as bs
from bluesky.tools import geo
from bluesky.tools.aero import nm

from cd import detect
from cr import ResolutionConfig, mvp
from crr import empty_recovery_state, make_recovery
from envs.pairwise_hor_conflict_heterogeneous_speed import (
    make_pairwise_hor_conflict_heterogeneous_speed,
    step, reset, avoidance_mask,
)
from sim_models.cns.cns import make_cns
from sim_models.cns.cns import step as cns_step

# Offset applied to seed to get the speed-draw seed; keeps speed draws and
# CNS noise draws from sharing state while remaining fully reproducible.
_SPEED_SEED_OFFSET = 100_000


# ── Private helpers ───────────────────────────────────────────────────────────

def _as_obs(sensor) -> SimpleNamespace:
    return SimpleNamespace(
        ntraf=sensor.n,
        id=sensor.id,
        lat=sensor.lat, lon=sensor.lon, alt=sensor.alt,
        trk=sensor.trk, gs=sensor.gs, vs=sensor.vs,
        gseast=sensor.gseast, gsnorth=sensor.gsnorth,
        perf=bs.traf.perf,
        selalt=bs.traf.selalt,
        adsl=SimpleNamespace(pos_acc=sensor.pos_acc, vel_acc=sensor.vel_acc),
    )


def _noop_recover(_idx):
    pass


def _geom_dcpa(view, env) -> np.ndarray:
    oi = np.asarray(env.ownship_idx)
    ii = np.asarray(env.intruder_idx)
    lat = np.asarray(view.lat); lon = np.asarray(view.lon)
    trk = np.radians(np.asarray(view.trk)); gs = np.asarray(view.gs)

    qdr, dist_nm = geo.kwikqdrdist_matrix(
        np.asmatrix(lat[oi]), np.asmatrix(lon[oi]),
        np.asmatrix(lat[ii]), np.asmatrix(lon[ii]))
    qdr  = np.diag(np.asarray(qdr))
    dist = np.diag(np.asarray(dist_nm)) * nm

    qdrrad = np.radians(qdr)
    dx = dist * np.sin(qdrrad)
    dy = dist * np.cos(qdrrad)

    du = gs[oi] * np.sin(trk[oi]) - gs[ii] * np.sin(trk[ii])
    dv = gs[oi] * np.cos(trk[oi]) - gs[ii] * np.cos(trk[ii])
    dv2 = du * du + dv * dv
    dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)

    tcpa  = -(du * dx + dv * dy) / dv2
    dcpa2 = np.abs(dist * dist - tcpa * tcpa * dv2)
    return np.sqrt(dcpa2)


def _done_with_timeout(done_now, done_start_t, t, timeout):
    if done_now:
        if done_start_t is None:
            done_start_t = t
    else:
        done_start_t = None
    should_stop = done_start_t is not None and (t - done_start_t) >= timeout
    return done_start_t, should_stop


@contextlib.contextmanager
def _silence():
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


# ── Public API ────────────────────────────────────────────────────────────────

def run_single(
    pair_width:      int,
    pair_height:     int,
    rpz:             float,
    hpz:             float,
    dtlookahead:     float,
    speed_min:       float,
    speed_max:       float,
    aircraft_type:   str,
    dpsi:            float,
    pos_ci95:        float,
    vel_ci95:        float,
    reception_prob:  float,
    *,
    pos_dist=None,
    latency_s:           float = 0.0,
    start_lat:           float = 52.0,
    start_lon:           float = 4.0,
    delta_lat_lon:       float = 0.01,
    tmax:                float = 600.0,
    done_timeout:        float = 30.0,
    resofach:            float = 1.05,
    recovery_resofach:   float = 1.05,
    prob_threshold:      float = 0.9,
    Ktheta:              int   = 256,
    cd=detect,
    cr=mvp.resolve,
    crr="double_criteria",
    simdt_factor:        int   = 1,
    seed:                int   = 44,
    record_history:      bool  = False,
) -> SimpleNamespace:
    '''Run one stochastic CD/CR/CRR simulation with heterogeneous per-pair speeds.

    Each pair independently draws ownship and intruder speeds from
    Uniform(speed_min, speed_max).  The speed draws use ``seed +
    _SPEED_SEED_OFFSET`` so they are reproducible but independent of the CNS
    noise RNG (which uses ``seed`` directly).

    All other parameters and return fields are identical to
    :func:`runners.stochastic_pairwise_hor_conflict.run_single`.

    Returns
    -------
    SimpleNamespace with fields:
      ipr, t_end, dist_arr, min_dist, n_los, env,
      rpz, hpz, dtlookahead, dpsi, pos_ci95, vel_ci95, reception_prob,
      speed_min, speed_max  — echoed inputs.
    When ``record_history=True``: t_arr, lat_arr, lon_arr, gs_arr, hdg_arr,
      avoid_arr, dcpa_obs_arr, dcpa_gt_arr.
    '''
    if not getattr(bs, "_sim_inited", False):
        with _silence():
            bs.init(mode="sim", detached=True)
        bs._sim_inited = True

    cfg   = ResolutionConfig(resofach=resofach)
    simdt = bs.settings.simdt * simdt_factor
    if isinstance(crr, str):
        crr = make_recovery(crr, recover=_noop_recover,
                            resofach=recovery_resofach,
                            prob_threshold=prob_threshold, Ktheta=Ktheta)

    env = make_pairwise_hor_conflict_heterogeneous_speed(
        pair_width=pair_width, pair_height=pair_height,
        asas_pzr_m=rpz, dtlookahead=dtlookahead,
        speed_min=speed_min, speed_max=speed_max,
        aircraft_type_ownship=aircraft_type,
        start_lat=start_lat, start_lon=start_lon, delta_lat_lon=delta_lat_lon,
        init_dpsi=dpsi, simdt_factor=simdt_factor,
        speed_seed=seed + _SPEED_SEED_OFFSET,
    )

    cns            = make_cns(pos_ci95=pos_ci95, vel_ci95=vel_ci95,
                               reception_prob=reception_prob, seed=seed,
                               pos_dist=pos_dist, latency_s=latency_s)
    recovery_state = empty_recovery_state()
    active         = np.zeros(bs.traf.ntraf, dtype=bool)

    distance_list = []
    time_list, lat_list, lon_list, gs_list, hdg_list = [], [], [], [], []
    avoid_list = []
    dcpa_obs_list, dcpa_gt_list = [], []
    t            = 0.0
    eps          = np.finfo(float).eps * 100
    next_event_t = 0.0
    asas_dt      = float(bs.settings.asas_dt)
    action       = None
    done_start_t = None

    while t < tmax:
        if t + eps >= next_event_t:
            cns     = cns_step(cns, bs.traf)
            obs     = _as_obs(cns.sensor)
            conf    = cd(obs, obs, rpz, hpz, dtlookahead)
            conf_gt = cd(bs.traf, bs.traf, rpz, hpz, dtlookahead)

            newtrack, newgs, newvs, alt = cr(conf, obs, obs, cfg)
            recovery_state, _ = crr(recovery_state, conf, obs, obs, active)
            action = (newtrack, newgs, newvs, alt, list(recovery_state.resopairs))

            done_now = (len(conf_gt.confpairs) == 0
                        and len(recovery_state.resopairs) == 0)
            done_start_t, should_stop = _done_with_timeout(
                done_now, done_start_t, t, done_timeout)
            if should_stop:
                break

            missed        = int(np.floor((t - next_event_t) / asas_dt)) + 1 if t > next_event_t else 1
            next_event_t += missed * asas_dt

        distances = step(env, action)
        distance_list.append(distances.copy())
        if record_history:
            time_list.append(t)
            lat_list.append(bs.traf.lat.copy())
            lon_list.append(bs.traf.lon.copy())
            gs_list.append(bs.traf.gs.copy())
            hdg_list.append(bs.traf.hdg.copy())
            avoid_list.append(avoidance_mask(action))
            dcpa_gt_list.append(_geom_dcpa(bs.traf, env))
            dcpa_obs_list.append(_geom_dcpa(cns.sensor, env))
        t += simdt

    t_end = t
    reset()

    dist_arr = np.array(distance_list)
    min_dist = np.min(dist_arr, axis=0)
    n_los    = int(np.sum(min_dist < rpz))
    ipr      = 1.0 - n_los / float(env.nb_pair)

    return SimpleNamespace(
        ipr=ipr, t_end=t_end,
        dist_arr=dist_arr, min_dist=min_dist, n_los=n_los, env=env,
        rpz=rpz, hpz=hpz, dtlookahead=dtlookahead, dpsi=dpsi,
        pos_ci95=pos_ci95, vel_ci95=vel_ci95, reception_prob=reception_prob,
        speed_min=speed_min, speed_max=speed_max, latency_s=latency_s,
        t_arr=np.array(time_list) if record_history else None,
        lat_arr=np.array(lat_list) if record_history else None,
        lon_arr=np.array(lon_list) if record_history else None,
        gs_arr=np.array(gs_list)   if record_history else None,
        hdg_arr=np.array(hdg_list) if record_history else None,
        avoid_arr=np.array(avoid_list) if record_history else None,
        dcpa_obs_arr=np.array(dcpa_obs_list) if record_history else None,
        dcpa_gt_arr=np.array(dcpa_gt_list)   if record_history else None,
    )


def get_ipr(**kwargs):
    '''Tuple wrapper for the Monte Carlo driver. Returns (dist_arr, ipr, t_end).'''
    kwargs.pop("record_history", None)
    res = run_single(record_history=False, **kwargs)
    return res.dist_arr, res.ipr, res.t_end


def run_parallel(
    *,
    n_runs:    int,
    n_jobs:    int,
    base_seed: int = 42,
    **kwargs,
) -> dict:
    '''Run :func:`get_ipr` ``n_runs`` times in parallel with independent seeds.

    Returns a dict with aggregated statistics::

        {
            "overall_ipr": float,
            "ipr":         np.ndarray,  shape (n_runs,)
            "worst_cpa":   np.ndarray,  minimum CPA per run (m)
            "t_end":       np.ndarray,  termination time per run (s)
        }
    '''
    def _one(rep):
        dist_arr, ipr, t_end = get_ipr(seed=base_seed + rep, **kwargs)
        worst_cpa = float(np.min(dist_arr))
        return ipr, worst_cpa, t_end

    results = Parallel(n_jobs=n_jobs)(delayed(_one)(r) for r in range(n_runs))

    ipr_arr, worst_cpa_arr, t_end_arr = map(np.array, zip(*results))

    nb_pair     = kwargs.get("pair_width", 1) * kwargs.get("pair_height", 1)
    n_los       = np.sum((1.0 - ipr_arr) * nb_pair)
    overall_ipr = 1.0 - n_los / float(n_runs * nb_pair)

    return {
        "overall_ipr": float(overall_ipr),
        "ipr":         ipr_arr,
        "worst_cpa":   worst_cpa_arr,
        "t_end":       t_end_arr,
    }
