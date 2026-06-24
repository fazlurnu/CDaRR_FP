''' State-based conflict detection — functional style.

The public entry point is :func:`detect`, a pure function that maps an
``(ownship, intruder, rpz, hpz, dtlookahead)`` snapshot to an immutable
:class:`~cd.common.ConflictState`. It is composed of small, side-effect-free
helpers (one per physical concern) so each piece can be reasoned about and
tested in isolation.
'''
import numpy as np
from bluesky.tools import geo
from bluesky.tools.aero import nm

from .common import ConflictState

# A large finite value used to mask the diagonal (ownship vs. itself) so that an
# aircraft never detects a conflict with its own track.
_BIG = 1e9


def relative_bearing_distance(ownship, intruder, eye):
    '''Bearing (deg) and distance (m) from every ownship to every intruder.

    The diagonal is pushed to a huge distance via ``eye`` so self-pairs are
    never flagged as conflicts.
    '''
    qdr, dist = geo.kwikqdrdist_matrix(
        np.asmatrix(ownship.lat), np.asmatrix(ownship.lon),
        np.asmatrix(intruder.lat), np.asmatrix(intruder.lon),
    )
    qdr = np.asarray(qdr)
    dist = np.asarray(dist) * nm + _BIG * eye
    return qdr, dist


def _velocity_components(trk, gs, n):
    '''Decompose track/ground-speed into (east, north) components.'''
    trkrad = np.radians(trk)
    u = gs * np.sin(trkrad).reshape((1, n))
    v = gs * np.cos(trkrad).reshape((1, n))
    return u, v


def horizontal_conflict(ownship, intruder, qdr, dist, rpz, eye):
    '''Horizontal CPA geometry and entry/exit times of the protected zone.

    Returns ``(swhorconf, tcpa, dcpa2, tinhor, touthor, vrel, rpz_mat)``.
    '''
    # Relative position at the CPA, projected onto east/north axes.
    qdrrad = np.radians(qdr)
    dx = dist * np.sin(qdrrad)
    dy = dist * np.cos(qdrrad)

    ownu, ownv = _velocity_components(ownship.trk, ownship.gs, ownship.ntraf)
    intu, intv = _velocity_components(intruder.trk, intruder.gs, intruder.ntraf)

    du = ownu - intu.T
    dv = ownv - intv.T

    dv2 = du * du + dv * dv
    dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)
    vrel = np.sqrt(dv2)

    tcpa = -(du * dx + dv * dy) / dv2 + _BIG * eye

    # Squared distance at CPA.
    dcpa2 = np.abs(dist * dist - tcpa * tcpa * dv2)

    # Pairwise protected-zone radius (symmetric: the larger of the two).
    rpz_mat = np.asarray(np.maximum(np.asmatrix(rpz), np.asmatrix(rpz).transpose()))
    R2 = rpz_mat * rpz_mat
    swhorconf = dcpa2 < R2

    # Times of entering / leaving the horizontal conflict zone.
    dxinhor = np.sqrt(np.maximum(0., R2 - dcpa2))
    dtinhor = dxinhor / vrel
    tinhor = np.where(swhorconf, tcpa - dtinhor, 1e8)
    touthor = np.where(swhorconf, tcpa + dtinhor, -1e8)

    return swhorconf, tcpa, dcpa2, tinhor, touthor, vrel, rpz_mat


def vertical_conflict(ownship, intruder, hpz, eye):
    '''Vertical separation and entry/exit times of the protected zone.

    Returns ``(dalt, tinver, toutver, hpz_mat)``.
    '''
    dalt = (
        ownship.alt.reshape((1, ownship.ntraf))
        - intruder.alt.reshape((1, intruder.ntraf)).T
        + _BIG * eye
    )

    dvs = (
        ownship.vs.reshape(1, ownship.ntraf)
        - intruder.vs.reshape(1, intruder.ntraf).T
    )
    dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)

    hpz_mat = np.asarray(np.maximum(np.asmatrix(hpz), np.asmatrix(hpz).transpose()))
    tcrosshi = (dalt + hpz_mat) / -dvs
    tcrosslo = (dalt - hpz_mat) / -dvs
    tinver = np.minimum(tcrosshi, tcrosslo)
    toutver = np.maximum(tcrosshi, tcrosslo)

    return dalt, tinver, toutver, hpz_mat


def combine_conflicts(swhorconf, tinhor, touthor, tinver, toutver, dtlookahead, eye):
    '''Intersect the horizontal and vertical windows into a conflict mask.

    Returns ``(swconfl, tinconf)``.
    '''
    tinconf = np.maximum(tinver, tinhor)
    toutconf = np.minimum(toutver, touthor)

    swconfl = np.array(
        swhorconf
        * (tinconf <= toutconf)
        * (toutconf > 0.0)
        * np.asarray(tinconf < np.asmatrix(dtlookahead).T)
        * (1.0 - eye),
        dtype=bool,
    )
    return swconfl, tinconf


def _conflict_pairs(ids, swconfl):
    '''Ordered (ownship, intruder) id tuples for every flagged cell.'''
    return [(ids[i], ids[j]) for i, j in zip(*np.where(swconfl))]


def detect(ownship, intruder, rpz, hpz, dtlookahead) -> ConflictState:
    '''Detect conflicts between ownship (traf) and intruder (traf/adsb).

    Pure function: given the traffic snapshot and protected-zone parameters it
    returns a fresh, immutable :class:`ConflictState`.
    '''
    ntraf = ownship.ntraf

    rpz_arr = np.array([rpz] * ntraf)
    hpz_arr = np.array([hpz] * ntraf)
    dtlook_arr = [dtlookahead] * ntraf

    # Identity matrix: masks ownship-vs-ownship self pairs.
    eye = np.eye(ntraf)

    qdr, dist = relative_bearing_distance(ownship, intruder, eye)

    swhorconf, tcpa, dcpa2, tinhor, touthor, vrel, rpz_mat = horizontal_conflict(
        ownship, intruder, qdr, dist, rpz, eye)

    dalt, tinver, toutver, hpz_mat = vertical_conflict(ownship, intruder, hpz, eye)

    swconfl, tinconf = combine_conflicts(
        swhorconf, tinhor, touthor, tinver, toutver, dtlookahead, eye)

    # Build result --------------------------------------------------------
    inconf = np.any(swconfl, 1)
    tcpamax = np.max(tcpa * swconfl, 1)

    confpairs = _conflict_pairs(ownship.id, swconfl)
    confpairs_unique = frozenset(frozenset(pair) for pair in confpairs)

    swlos = (dist < rpz_mat) * (np.abs(dalt) < hpz_mat)
    lospairs = _conflict_pairs(ownship.id, swlos)

    return ConflictState(
        rpz=rpz_arr,
        hpz=hpz_arr,
        dtlookahead=dtlook_arr,
        confpairs=confpairs,
        confpairs_unique=confpairs_unique,
        lospairs=lospairs,
        qdr=qdr[swconfl],
        dist=dist[swconfl],
        dcpa=np.sqrt(dcpa2[swconfl]),
        tcpa=tcpa[swconfl],
        tLOS=tinconf[swconfl],
        inconf=inconf,
        tcpamax=tcpamax,
    )
