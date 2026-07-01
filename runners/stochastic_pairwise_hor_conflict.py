'''Stochastic pairwise horizontal conflict runner — single run and Monte Carlo.

One call to :func:`run_single` runs a full CD/CR/CRR simulation and returns a
result object with the IPR and (optionally) the full per-tick trajectory
history needed for plotting.  :func:`get_ipr` is a thin tuple-returning wrapper
used by :func:`run_parallel`, which wraps it with joblib to produce Monte Carlo
statistics across many seeds.

Observation model
-----------------
:func:`get_ipr` uses the FP CNS layer for noisy observations.  On each ASAS
tick the CNS sensor measures every aircraft with Gaussian position/velocity
noise (``pos_ci95`` / ``vel_ci95`` in metres / m/s at the 95th percentile).
With ``reception_prob=1.0`` all aircraft receive each other's current noisy
reading; the ``SensorState`` is used directly as both ownship and intruder
views for detection.  For ``reception_prob < 1.0`` some cells become stale
(the CNS ADS-L layer handles this), but extracting per-observer 1-D views from
the N×N observation matrix is left to callers — see the note in :func:`get_ipr`.

Usage::

    # Single run with full history for plotting:
    res = run_single(
        pair_width=3, pair_height=3,
        rpz=50.0, hpz=50.0, dtlookahead=121.0,
        init_speed_ownship=15.0, init_speed_intruder=15.0,
        aircraft_type="M600", dpsi=90.0,
        pos_ci95=10.0, vel_ci95=1.0, reception_prob=1.0,
        record_history=True,
    )
    res.ipr, res.t_arr, res.dist_arr, res.lat_arr, res.env  # ...

    # Tuple wrapper used by the Monte Carlo driver (no history):
    dist_arr, ipr, t_end = get_ipr(
        pair_width=3, pair_height=3,
        rpz=50.0, hpz=50.0, dtlookahead=121.0,
        init_speed_ownship=15.0, init_speed_intruder=15.0,
        aircraft_type="M600", dpsi=90.0,
        pos_ci95=10.0, vel_ci95=1.0, reception_prob=1.0,
    )

    results = run_parallel(
        n_runs=50, n_jobs=10,
        pair_width=3, pair_height=3,
        rpz=50.0, hpz=50.0, dtlookahead=121.0,
        init_speed_ownship=15.0, init_speed_intruder=15.0,
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
from crr import (
    empty_recovery_state,
    make_recovery,
)
from envs.pairwise_hor_conflict import make_pairwise_hor_conflict, step, reset, avoidance_mask
from sim_models.cns.cns import make_cns
from sim_models.cns.cns import step as cns_step


# ── Private helpers ───────────────────────────────────────────────────────────

def _as_obs(sensor) -> SimpleNamespace:
    '''Wrap a SensorState as a traffic-like object for detect / resolve.

    Position and velocity come from the noisy sensor reading. Performance limits
    and autopilot targets are taken from bs.traf (onboard parameters — available
    locally and not transmitted over ADS-L).

    ``adsl`` exposes the per-aircraft 95% accuracy radii (``pos_acc`` / ``vel_acc``)
    broadcast over ADS-L, consumed by the probabilistic recovery rule.
    '''
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
    # pairwise aircraft have no filed route; waypoint recovery is a no-op
    pass


def _geom_dcpa(view, env) -> np.ndarray:
    '''Geometric projected distance at CPA for every pair, computed directly.

    Uses the pair's own relative position and velocity (no conflict-detection
    flagging), so a value is defined at *every* tick — including while the pair
    is not a detected conflict. Same CPA geometry as ``cd.statebased`` but
    evaluated per pair. ``view`` is any traffic-like object indexable by the env
    aircraft indices (``lat``/``lon``/``trk``/``gs``): pass ``bs.traf`` for
    ground truth or the CNS ``sensor`` for the noisy observation. Returns shape
    ``(nb_pair,)`` in metres.
    '''
    oi = np.asarray(env.ownship_idx)
    ii = np.asarray(env.intruder_idx)
    lat = np.asarray(view.lat); lon = np.asarray(view.lon)
    trk = np.radians(np.asarray(view.trk)); gs = np.asarray(view.gs)

    qdr, dist_nm = geo.kwikqdrdist_matrix(
        np.asmatrix(lat[oi]), np.asmatrix(lon[oi]),
        np.asmatrix(lat[ii]), np.asmatrix(lon[ii]))
    qdr  = np.diag(np.asarray(qdr))          # own↔int bearing, per pair (deg)
    dist = np.diag(np.asarray(dist_nm)) * nm # own↔int distance, per pair (m)

    qdrrad = np.radians(qdr)
    dx = dist * np.sin(qdrrad)               # relative position, east/north
    dy = dist * np.cos(qdrrad)

    du = gs[oi] * np.sin(trk[oi]) - gs[ii] * np.sin(trk[ii])  # relative velocity
    dv = gs[oi] * np.cos(trk[oi]) - gs[ii] * np.cos(trk[ii])
    dv2 = du * du + dv * dv
    dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)

    tcpa  = -(du * dx + dv * dy) / dv2
    dcpa2 = np.abs(dist * dist - tcpa * tcpa * dv2)
    return np.sqrt(dcpa2)


def _done_with_timeout(done_now, done_start_t, t, timeout):
    '''Latch-and-timeout: return (done_start_t, should_stop).'''
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
    pair_width: int,
    pair_height: int,
    rpz: float,
    hpz: float,
    dtlookahead: float,
    init_speed_ownship: float,
    init_speed_intruder: float,
    aircraft_type: str,
    dpsi: float,
    pos_ci95: float,
    vel_ci95: float,
    reception_prob: float,
    *,
    start_lat: float = 52.0,
    start_lon: float = 4.0,
    delta_lat_lon: float = 0.01,
    tmax: float = 600.0,
    done_timeout: float = 30.0,
    resofach: float = 1.05,
    recovery_resofach: float = 1.05,
    prob_threshold: float = 0.9,
    Ktheta: int = 256,
    cd=detect,
    cr=mvp.resolve,
    crr="double_criteria",
    simdt_factor: int = 1,
    seed: int = 44,
    pos_dist=None,
    vel_dist=None,
    latency_s: float = 0.0,
    record_history: bool = False,
) -> SimpleNamespace:
    '''Run one stochastic CD/CR/CRR simulation and return a result namespace.

    ``pos_ci95`` / ``vel_ci95`` are the 95th-percentile Gaussian noise bounds
    applied to each aircraft's self-measurement (metres / m/s).

    The CD / CR / CRR stages are injectable so alternative algorithms (e.g. a
    learned policy) can be dropped in without touching the loop:
      cd(ownship, intruder, rpz, hpz, dtlookahead) -> conf
      cr(conf, ownship, intruder, cfg) -> (newtrack, newgs, newvs, alt)
      crr(state, conf, ownship, intruder, active) -> (new_state, delpairs)
    Defaults are the project's ``detect`` / ``mvp.resolve``.

    ``crr`` accepts either a strategy name or a ready-made callable:
      * a name (str) — ``"double_criteria"`` (FTR two-criteria rule, the
        default), ``"cpa"`` (past-CPA rule, parameterised by
        ``recovery_resofach``), or ``"probabilistic"`` (probabilistic FTR rule,
        parameterised by ``prob_threshold`` / ``Ktheta``). The name is resolved
        via ``crr.make_recovery`` with this env's no-op waypoint recovery bound.
      * a callable with the ``crr(...)`` signature above — used as-is, so the
        ``recovery_resofach`` / ``prob_threshold`` / ``Ktheta`` knobs are
        ignored (bind them yourself, e.g. ``make_recovery("cpa", resofach=...)``).
    The probabilistic rule reads each aircraft's ADS-L accuracy (``pos_acc`` /
    ``vel_acc``, populated from ``pos_ci95`` / ``vel_ci95``) via the ``adsl``
    field of the observation.

    With ``record_history=True`` the full per-tick trajectory is captured for
    plotting; leave it ``False`` (the default) for Monte Carlo runs where only
    the aggregate IPR matters, to avoid storing large arrays per seed.

    Note: with ``reception_prob < 1.0`` some aircraft receive stale intruder
    observations. The runner currently passes the fresh sensor snapshot as the
    intruder view for all aircraft, which is exact only at prob=1.0. Support for
    stale observation extraction from the N×N CNS obs matrix can be added by
    constructing per-observer 1-D views from ``adsl_field(cns, <field>)``.

    Returns
    -------
    SimpleNamespace with fields:
      ipr       : float
      t_end     : float                 — simulation time at termination
      dist_arr  : np.ndarray (T, nb_pair)
      min_dist  : np.ndarray (nb_pair,) — minimum CPA per pair (m)
      n_los     : int
      env       : PairwiseHorConflictEnv — index/id lookup tables (for plotting)
      rpz, hpz, dtlookahead, dpsi, pos_ci95, vel_ci95, reception_prob — echoed
        inputs, convenient for figure titles.
    When ``record_history=True`` the following per-tick arrays are also set
    (else they are ``None``):
      t_arr   : np.ndarray (T,)
      lat_arr, lon_arr, gs_arr, hdg_arr : np.ndarray (T, ntraf)
      avoid_arr : np.ndarray (T, ntraf) — 1.0 where the aircraft is actively
        avoiding (in a resolution pair) at that tick, else 0.0.
      dcpa_obs_arr, dcpa_gt_arr : np.ndarray (T, nb_pair) — geometric projected
        distance at CPA per pair, computed directly from each pair's relative
        position/velocity every tick (no conflict-detection flagging, so defined
        at all times). ``dcpa_gt_arr`` uses ground truth (bs.traf); ``dcpa_obs_arr``
        uses the held noisy CNS sensor reading (stepwise — refreshes on ASAS ticks).
    '''
    if not getattr(bs, "_sim_inited", False):
        with _silence():
            bs.init(mode="sim", detached=True)
        bs._sim_inited = True

    cfg   = ResolutionConfig(resofach=resofach)
    simdt = bs.settings.simdt * simdt_factor
    if isinstance(crr, str):
        # crr is a strategy name: build it for this env. Pairwise aircraft have
        # no filed route, so override the waypoint-recovery side effect with a
        # no-op. resofach/prob_threshold/Ktheta are forwarded; each strategy
        # ignores the ones it doesn't use. (A callable crr is used as-is.)
        crr = make_recovery(crr, recover=_noop_recover,
                            resofach=recovery_resofach,
                            prob_threshold=prob_threshold, Ktheta=Ktheta)

    env = make_pairwise_hor_conflict(
        pair_width=pair_width, pair_height=pair_height,
        asas_pzr_m=rpz, dtlookahead=dtlookahead,
        init_speed_ownship=init_speed_ownship,
        init_speed_intruder=init_speed_intruder,
        aircraft_type_ownship=aircraft_type,
        start_lat=start_lat, start_lon=start_lon, delta_lat_lon=delta_lat_lon,
        init_dpsi=dpsi, simdt_factor=simdt_factor,
    )

    cns            = make_cns(pos_ci95=pos_ci95, vel_ci95=vel_ci95,
                               reception_prob=reception_prob, seed=seed,
                               pos_dist=pos_dist, vel_dist=vel_dist,
                               latency_s=latency_s)
    recovery_state = empty_recovery_state()
    active         = np.zeros(bs.traf.ntraf, dtype=bool)

    distance_list = []
    time_list, lat_list, lon_list, gs_list, hdg_list = [], [], [], [], []
    sensor_lat_list, sensor_lon_list = [], []
    avoid_list = []
    dcpa_obs_list, dcpa_gt_list = [], []
    t             = 0.0
    eps           = np.finfo(float).eps * 100
    next_event_t  = 0.0
    asas_dt       = float(bs.settings.asas_dt)
    action        = None
    done_start_t  = None
    conf_gt       = None

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
            # Geometric CPA projection per pair, defined every tick: ground truth
            # from bs.traf (fresh), observed from the held CNS sensor reading
            # (which only refreshes on ASAS ticks, hence its stepwise shape).
            dcpa_gt_list.append(_geom_dcpa(bs.traf, env))
            dcpa_obs_list.append(_geom_dcpa(cns.sensor, env))
            sensor_lat_list.append(cns.sensor.lat.copy())
            sensor_lon_list.append(cns.sensor.lon.copy())
        t += simdt

    t_end = t
    reset()

    dist_arr = np.array(distance_list)              # (T, nb_pair)
    min_dist = np.min(dist_arr, axis=0)
    n_los    = int(np.sum(min_dist < rpz))
    ipr      = 1.0 - n_los / float(env.nb_pair)

    return SimpleNamespace(
        ipr=ipr, t_end=t_end,
        dist_arr=dist_arr, min_dist=min_dist, n_los=n_los, env=env,
        rpz=rpz, hpz=hpz, dtlookahead=dtlookahead, dpsi=dpsi,
        pos_ci95=pos_ci95, vel_ci95=vel_ci95, reception_prob=reception_prob,
        latency_s=latency_s,
        t_arr=np.array(time_list) if record_history else None,
        lat_arr=np.array(lat_list) if record_history else None,
        lon_arr=np.array(lon_list) if record_history else None,
        gs_arr=np.array(gs_list)  if record_history else None,
        hdg_arr=np.array(hdg_list) if record_history else None,
        avoid_arr=np.array(avoid_list) if record_history else None,
        dcpa_obs_arr=np.array(dcpa_obs_list) if record_history else None,
        dcpa_gt_arr=np.array(dcpa_gt_list)   if record_history else None,
        sensor_lat_arr=np.array(sensor_lat_list) if record_history else None,
        sensor_lon_arr=np.array(sensor_lon_list) if record_history else None,
    )


def get_ipr(**kwargs):
    '''Tuple wrapper around :func:`run_single` for the Monte Carlo driver.

    Returns ``(distance_array, ipr, t_end)``. History recording is forced off.
    '''
    kwargs.pop("record_history", None)
    res = run_single(record_history=False, **kwargs)
    return res.dist_arr, res.ipr, res.t_end


def run_parallel(
    *,
    n_runs: int,
    n_jobs: int,
    base_seed: int = 42,
    **kwargs,
) -> dict:
    '''Run :func:`get_ipr` ``n_runs`` times in parallel with independent seeds.

    Returns a dict with aggregated statistics::

        {
            "overall_ipr": float,         — aggregated across all runs
            "ipr":         np.ndarray,    — per-run IPR, shape (n_runs,)
            "worst_cpa":   np.ndarray,    — minimum CPA per pair per run (m)
            "t_end":       np.ndarray,    — termination time per run (s)
        }
    '''
    def _one(rep):
        dist_arr, ipr, t_end = get_ipr(seed=base_seed + rep, **kwargs)
        worst_cpa = float(np.min(dist_arr))
        return ipr, worst_cpa, t_end

    results = Parallel(n_jobs=n_jobs)(delayed(_one)(r) for r in range(n_runs))

    ipr_arr, worst_cpa_arr, t_end_arr = map(np.array, zip(*results))

    nb_pair    = kwargs.get("pair_width", 1) * kwargs.get("pair_height", 1)
    n_los      = np.sum((1.0 - ipr_arr) * nb_pair)
    overall_ipr = 1.0 - n_los / float(n_runs * nb_pair)

    return {
        "overall_ipr": float(overall_ipr),
        "ipr":         ipr_arr,
        "worst_cpa":   worst_cpa_arr,
        "t_end":       t_end_arr,
    }
