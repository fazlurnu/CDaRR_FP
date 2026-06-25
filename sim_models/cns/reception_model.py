'''CNS reception layer — N×N packet reception matrix and mask sampler.

The public surface is a frozen :class:`ReceptionModel` (the N×N matrix ``P``) plus
pure functions that return new instances rather than mutating in place.
``sample_mask`` returns ``(mask, updated_rm)`` so a resized ``P`` is threaded
back explicitly — no hidden state change.

``P[i, j]`` is the probability that observer ``i`` receives target ``j``'s
message this tick. Asymmetry is allowed; the diagonal is always 1.0 (ownship
always receives itself).
'''
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReceptionModel:
    '''Immutable N×N reception probability matrix.

    ``default_prob`` is the off-diagonal value used when (re)building ``P``.
    ``P`` itself is shape ``(n, n)``; an empty ``(0, 0)`` array is the
    canonical "not yet sized" state returned by :func:`make_reception`.
    '''
    default_prob: float
    P: np.ndarray


def make_reception(default_prob: float) -> ReceptionModel:
    '''Create a reception model with an empty (unsized) P matrix.'''
    if not (0.0 <= default_prob <= 1.0):
        raise ValueError(f'default_prob must be in [0, 1], got {default_prob}')
    return ReceptionModel(default_prob=float(default_prob), P=np.empty((0, 0)))


def ensure_size(rm: ReceptionModel, n: int) -> ReceptionModel:
    '''Return a ReceptionModel whose P is exactly (n, n).

    If P is already the right size, the same instance is returned unchanged.
    Otherwise a new matrix is built: off-diagonal = default_prob, diagonal = 1.0,
    and any existing entries in the top-left sub-block are preserved.
    '''
    if rm.P.shape == (n, n):
        return rm
    new_P = np.full((n, n), rm.default_prob, dtype=float)
    np.fill_diagonal(new_P, 1.0)
    m = min(rm.P.shape[0], n)
    if m:
        new_P[:m, :m] = rm.P[:m, :m]
    return ReceptionModel(default_prob=rm.default_prob, P=new_P)


def set_pair(rm: ReceptionModel, i: int, j: int, prob: float) -> ReceptionModel:
    '''Return a new ReceptionModel with P[i, j] set to prob.

    P is copied so the original ReceptionModel is not affected.
    '''
    new_P = rm.P.copy()
    new_P[i, j] = prob
    return ReceptionModel(default_prob=rm.default_prob, P=new_P)


def sample_mask(rm: ReceptionModel, n: int, rng: np.random.Generator,
                force_full: bool = False):
    '''Sample a boolean (n, n) refresh mask and return ``(mask, updated_rm)``.

    ``True`` at ``[i, j]`` means observer ``i`` receives target ``j`` this tick.
    The diagonal is always ``True``. ``force_full=True`` sets all cells to
    ``True`` (used on the first update to seed every cell without packet loss).

    Returns the updated ``ReceptionModel`` because ``ensure_size`` may have
    grown ``P``; callers must thread the returned value forward.
    '''
    rm = ensure_size(rm, n)
    if force_full:
        return np.ones((n, n), dtype=bool), rm
    mask = rng.random((n, n)) <= rm.P
    np.fill_diagonal(mask, True)
    return mask, rm


# Flat-earth metre-per-degree constants (same as sensor.py).
_M_PER_DEG = 111_320.0
_COSLAT_FLOOR = 1e-6


def p_from_range(states, max_range: float,
                 default_prob: float = 1.0) -> ReceptionModel:
    '''Build a ReceptionModel from pairwise geometry — v1 step function.

    ``P[i, j] = default_prob`` when the flat-earth distance between aircraft
    ``i`` and ``j`` is **≤ max_range** (metres), ``0.0`` otherwise.
    Diagonal is always ``1.0`` (ownship always receives itself).

    Call this every tick (or whenever geometry or ``max_range`` changes) and
    thread the result into CNSState via ``dataclasses.replace``::

        rm = p_from_range(states, max_range)
        cns = replace(cns, reception=rm)
        cns = step(cns, states)

    # TODO(geometry): replace the step function with a continuous model
    #   (logistic decay, Friis power falloff, LoS obstruction, …) by swapping
    #   the ``np.where`` below for the desired mapping dist -> probability.
    '''
    lat = np.asarray(states.lat, dtype=float)
    lon = np.asarray(states.lon, dtype=float)

    # Pairwise flat-earth distances in metres.
    dlat_m = (lat[:, None] - lat[None, :]) * _M_PER_DEG
    mean_lat = (lat[:, None] + lat[None, :]) / 2.0
    coslat = np.maximum(np.cos(np.deg2rad(mean_lat)), _COSLAT_FLOOR)
    dlon_m = (lon[:, None] - lon[None, :]) * _M_PER_DEG * coslat
    dist = np.sqrt(dlat_m ** 2 + dlon_m ** 2)

    P = np.where(dist <= max_range, float(default_prob), 0.0)
    np.fill_diagonal(P, 1.0)
    return ReceptionModel(default_prob=default_prob, P=P)
