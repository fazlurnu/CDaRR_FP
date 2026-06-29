'''CNS coordinator — single per-tick entry point for the full CNS pipeline.

One call to :func:`step` per simulation tick: it measures truth, samples the
reception mask, and refreshes the observation picture, returning a new
:class:`CNSState`. Nothing is mutated in place; each tick produces a fresh,
immutable value.

Usage in CD/CR/CRR::

    cns = make_cns(pos_ci95=5.0, vel_ci95=1.0, reception_prob=0.95, seed=42)

    # each tick:
    cns = step(cns, traffic)               # returns a new CNSState
    lat_seen = adsl_field(cns, "lat")      # (N, N): lat_seen[i, j] = what i sees of j
    own_lat  = ownship_field(cns, "lat")   # (N,):   own_lat[i]      = lat_seen[i, i]

To change accuracy between ticks::

    from dataclasses import replace
    cns = replace(cns, pos_ci95=new_ci95)
    cns = step(cns, traffic)

To configure an asymmetric reception pair::

    from dataclasses import replace
    from sim_models.cns.reception_model import set_pair
    cns = replace(cns, reception=set_pair(cns.reception, i, j, prob))
'''
from dataclasses import dataclass, replace

import numpy as np

from .adsl_observation import ADSLObservation, empty_observation
from .adsl_observation import field as obs_field
from .adsl_observation import update
from .distributions import gaussian
from .reception_model import ReceptionModel, make_reception, sample_mask
from .sensor import SensorState, measure


@dataclass(frozen=True)
class CNSState:
    '''Immutable snapshot of the full CNS pipeline.

    Thread this forward: ``cns = step(cns, states)`` each tick. Change
    parameters between ticks with ``dataclasses.replace``.
    '''
    sensor: SensorState
    reception: ReceptionModel
    obs: ADSLObservation
    pos_ci95: object           # scalar or (N,) — accuracy for position noise draw
    vel_ci95: object           # scalar or (N,) — accuracy for velocity noise draw
    pos_dist: object           # callable (n, ci95, rng) -> (n, 2)
    vel_dist: object           # callable (n, ci95, rng) -> (n, 2)
    rng: object                # np.random.Generator — shared; advances each tick
    first_update_done: bool
    latency_s: float           # ADS-B reporting latency; bias = −latency × gs per aircraft
    cross_track_bias_m: float  # systematic lateral offset (near zero per literature)


def make_cns(pos_ci95, vel_ci95, reception_prob=1.0,
             pos_dist=None, vel_dist=None, seed=None,
             latency_s=0.0, cross_track_bias_m=0.0) -> CNSState:
    '''Create an initial (pre-tick) CNSState.

    All matrices start empty; the first :func:`step` call seeds every cell
    unconditionally (``force_full=True``) so no NaN cells arise.

    ``latency_s``: ADS-B position reporting latency in seconds. The per-aircraft
    along-track bias is ``−latency_s × gs`` each tick (see :func:`sensor.measure`).
    ADS-B v2 mean latency ≈ 0.0661 s; at 20 kts (10.3 m/s) this gives ~0.68 m lag.
    '''
    rng = np.random.default_rng(seed)
    return CNSState(
        sensor=_empty_sensor(),
        reception=make_reception(reception_prob),
        obs=empty_observation(),
        pos_ci95=pos_ci95,
        vel_ci95=vel_ci95,
        pos_dist=pos_dist if pos_dist is not None else gaussian,
        vel_dist=vel_dist if vel_dist is not None else gaussian,
        rng=rng,
        first_update_done=False,
        latency_s=latency_s,
        cross_track_bias_m=cross_track_bias_m,
    )


def step(cns: CNSState, states) -> CNSState:
    '''Advance the CNS by one tick, returning a new CNSState.

    Sequence:
    1. ``measure`` — fresh noisy snapshot of truth.
    2. ``sample_mask`` — which (i, j) pairs receive a packet this tick;
       ``force_full=True`` on the first call to seed all cells.
    3. ``update`` — write received sensor values into the observation picture.
    '''
    n = int(states.ntraf)
    sensor = measure(states, cns.pos_ci95, cns.vel_ci95,
                     cns.pos_dist, cns.vel_dist, cns.rng,
                     cns.latency_s, cns.cross_track_bias_m)
    mask, rm = sample_mask(cns.reception, n, cns.rng,
                           force_full=not cns.first_update_done)
    obs = update(cns.obs, sensor, mask)
    return replace(cns, sensor=sensor, reception=rm, obs=obs,
                   first_update_done=True)


def ownship_field(cns: CNSState, name: str) -> np.ndarray:
    '''1D length-N array: ownship i's own sensor reading of ``name``.

    Equivalent to the diagonal of ``adsl_field(cns, name)``.
    '''
    return getattr(cns.sensor, name)


def adsl_field(cns: CNSState, name: str) -> np.ndarray:
    '''(N, N) array: ``result[i, j]`` is what observer i last received of j's ``name``.'''
    return obs_field(cns.obs, name)


def _empty_sensor() -> SensorState:
    z = np.empty(0, dtype=float)
    return SensorState(
        n=0, id=[],
        lat=z, lon=z, alt=z, hdg=z, trk=z,
        gs=z, tas=z, vs=z, gseast=z, gsnorth=z,
        pos_acc=z, vel_acc=z,
    )
