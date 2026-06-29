'''CNS sensor layer — each aircraft's noisy measurement of its own state.

``sensor = truth + noise``, re-drawn **every tick**. The public surface is a pure
function :func:`measure` that maps a BlueSky-truth snapshot to a fresh, immutable
:class:`SensorState`. There is no mutable ``Sensor`` object and no separate
``_init_arrays`` step: every call rebuilds the 1D arrays from ``states.ntraf``, so
resizing is implicit and side-effect free.

Noise is applied **once, here, at the sensor**. The ADS-L layer adds only
reception (stale-vs-fresh) and never re-noises. Position error is drawn in metres
(east, north) and converted to lat/lon degrees using the legacy conversion
(including the ``cos(lat)`` pole guard); velocity error is added to the
north/east ground-speed components derived from ``gs`` and ``trk``.
'''
from dataclasses import dataclass

import numpy as np

from .distributions import gaussian

# Measured fields carried through sensor -> obs. The first ten are state; the
# last two record the 95% CI actually used for the draw, exposed downstream.
STATE_FIELDS = ['lat', 'lon', 'alt', 'hdg', 'trk', 'gs', 'tas', 'vs',
                'gseast', 'gsnorth']
ACC_FIELDS = ['pos_acc', 'vel_acc']
FIELDS = STATE_FIELDS + ACC_FIELDS

# Metres per degree of latitude (legacy constant; longitude scales by cos(lat)).
_M_PER_DEG = 111_320.0
_COSLAT_FLOOR = 1e-6  # pole guard so the lon conversion never blows up


@dataclass(frozen=True)
class SensorState:
    '''Immutable 1D snapshot of every aircraft's self-measurement (length N).

    Arrays are aligned by aircraft index. ``id`` is the per-aircraft id list.
    Being frozen keeps :func:`measure` a pure function of its inputs.
    '''
    n: int
    id: list
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    hdg: np.ndarray
    trk: np.ndarray
    gs: np.ndarray
    tas: np.ndarray
    vs: np.ndarray
    gseast: np.ndarray
    gsnorth: np.ndarray
    pos_acc: np.ndarray
    vel_acc: np.ndarray


def _apply_position_noise(lat, lon, exy):
    '''Add an (east, north) metre error to lat/lon, returning degrees.

    Preserves the legacy conversion: north metres scale by ``1/111320``; east
    metres additionally divide by ``cos(lat)`` (floored to avoid a pole blow-up).
    '''
    east_m, north_m = exy[:, 0], exy[:, 1]
    coslat = np.maximum(np.cos(np.deg2rad(lat)), _COSLAT_FLOOR)
    lat_out = lat + north_m / _M_PER_DEG
    lon_out = lon + east_m / (_M_PER_DEG * coslat)
    return lat_out, lon_out


def _track_relative_bias(trk, bias_at_m, bias_ct_m):
    '''Rotate a (along-track, cross-track) bias into (east, north) metres.

    Along-track is the direction of travel; a negative value means the reported
    position lags behind the true position (ADS-B latency signature). Cross-track
    is 90° left of the direction of travel; the paper finds this is near zero.

    Returns shape (n, 2) ready to add directly to an exy draw.
    '''
    trk_rad = np.deg2rad(trk)
    east  = bias_at_m * np.sin(trk_rad) - bias_ct_m * np.cos(trk_rad)
    north = bias_at_m * np.cos(trk_rad) + bias_ct_m * np.sin(trk_rad)
    return np.stack([east, north], axis=1)


def _velocity_components(gs, trk, vxy):
    '''Decompose ground speed/track into (north, east) components plus noise.

    ``vxy[:, 0]`` is the north error, ``vxy[:, 1]`` the east error (m/s).
    '''
    north_n, east_n = vxy[:, 0], vxy[:, 1]
    trk_rad = np.deg2rad(trk)
    gsnorth = gs * np.cos(trk_rad) + north_n
    gseast = gs * np.sin(trk_rad) + east_n
    return gsnorth, gseast


def measure(states, pos_ci95, vel_ci95,
            pos_dist=gaussian, vel_dist=gaussian, rng=None,
            latency_s=0.0, cross_track_bias_m=0.0) -> SensorState:
    '''Measure every aircraft from truth ``states``, returning a fresh SensorState.

    ``pos_ci95`` / ``vel_ci95`` are scalar or shape ``(n,)`` and are read here, so
    the noise level can change per tick and per aircraft. ``pos_dist`` / ``vel_dist``
    are distribution callables ``(n, ci95, rng) -> (n, 2)`` (default Gaussian).

    ``latency_s`` is the ADS-B position reporting latency in seconds (a system
    property). The per-aircraft along-track bias is computed as ``−latency_s × gs``
    each tick, so it automatically scales with each aircraft's ground speed.
    ``cross_track_bias_m`` is a fixed lateral offset (near zero per the literature).
    '''
    rng = rng or np.random.default_rng()
    n = int(states.ntraf)

    # Pass-through (non-noised) fields, copied so the result never aliases truth.
    alt = np.array(states.alt, dtype=float)
    hdg = np.array(states.hdg, dtype=float)
    trk = np.array(states.trk, dtype=float)
    gs = np.array(states.gs, dtype=float)
    tas = np.array(states.tas, dtype=float)
    vs = np.array(states.vs, dtype=float)

    # Record the accuracy actually used (broadcast scalars to per-aircraft).
    pos_acc = np.broadcast_to(pos_ci95, (n,)).astype(float).copy()
    vel_acc = np.broadcast_to(vel_ci95, (n,)).astype(float).copy()

    # Position noise: metres (east, north) -> lat/lon degrees.
    exy = pos_dist(n, pos_ci95, rng)
    if latency_s != 0.0 or cross_track_bias_m != 0.0:
        # Along-track bias = −latency × gs (per-aircraft, m/s → m).
        bias_at = -latency_s * gs
        exy = exy + _track_relative_bias(trk, bias_at, cross_track_bias_m)
    lat, lon = _apply_position_noise(
        np.asarray(states.lat, dtype=float),
        np.asarray(states.lon, dtype=float),
        exy,
    )

    # Velocity noise: m/s added to north/east ground-speed components.
    vxy = vel_dist(n, vel_ci95, rng)
    gsnorth, gseast = _velocity_components(gs, trk, vxy)

    return SensorState(
        n=n,
        id=list(states.id),
        lat=lat, lon=lon, alt=alt, hdg=hdg, trk=trk,
        gs=gs, tas=tas, vs=vs, gseast=gseast, gsnorth=gsnorth,
        pos_acc=pos_acc, vel_acc=vel_acc,
    )
