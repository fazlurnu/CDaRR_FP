''' Conflict resolution based on the Modified Voltage Potential (MVP) algorithm.

Functional rewrite of the former ``MVP(Entity)`` class. The algorithm is now a
set of pure functions:

* :func:`mvp_pair`     — resolution velocity for a single conflict pair.
* :func:`resolve`      — combine per-pair resolutions into per-aircraft commands.

Mutable instance switches became a :class:`~cr.common.ResolutionConfig`.
'''
import numpy as np

from .common import (
    ResolutionConfig,
    cap_velocities,
    horizontal_command,
)


def mvp_pair(ownship, intruder, conf, qdr, dist, tcpa, idx1, idx2, resofach):
    '''Modified Voltage Potential resolution for one conflict pair.

    Returns the 2-D horizontal resolution velocity ``[dv_east, dv_north]`` for
    the ownship. Pure.
    '''
    rpz_m = np.max(conf.rpz[[idx1, idx2]] * resofach)
    qdr = np.radians(qdr)

    # Relative position vector between idx1 and idx2.
    drel = np.array([np.sin(qdr) * dist,
                     np.cos(qdr) * dist])

    # Relative velocity vector (intruder minus ownship).
    v1 = np.array([ownship.gseast[idx1], ownship.gsnorth[idx1]])
    v2 = np.array([intruder.gseast[idx2], intruder.gsnorth[idx2]])
    vrel = v2 - v1

    # Horizontal resolution ----------------------------------------------
    dcpa = drel + vrel * tcpa
    dabsH = np.sqrt(dcpa[0] * dcpa[0] + dcpa[1] * dcpa[1])

    iH = rpz_m - dabsH

    threshold = 0.001
    if dabsH <= threshold:
        dabsH = threshold
        dcpa[0] = drel[1] / dist * dabsH
        dcpa[1] = -drel[0] / dist * dabsH

    if rpz_m < dist and dabsH < dist:
        erratum = np.cos(np.arcsin(rpz_m / dist) - np.arcsin(dabsH / dist))
        dv1 = ((rpz_m / erratum - dabsH) * dcpa[0]) / (abs(tcpa) * dabsH)
        dv2 = ((rpz_m / erratum - dabsH) * dcpa[1]) / (abs(tcpa) * dabsH)
    else:
        dv1 = (iH * dcpa[0]) / (abs(tcpa) * dabsH)
        dv2 = (iH * dcpa[1]) / (abs(tcpa) * dabsH)

    return np.array([dv1, dv2])


def resolve(conf, ownship, intruder, cfg: ResolutionConfig, resofach=None):
    '''Resolve all current conflicts with MVP.

    Returns ``(newtrack, newgs, newvs, alt)`` — the per-aircraft ASAS commands.
    '''
    if resofach is not None:
        cfg = cfg.with_resofach(resofach)

    ntraf = ownship.ntraf

    # Per-aircraft running sum of horizontal resolution velocities.
    dv = np.zeros((ntraf, 2))

    for ((ac1, ac2), qdr, dist, tcpa) in zip(
            conf.confpairs, conf.qdr, conf.dist, conf.tcpa):
        idx1 = ownship.id.index(ac1)
        idx2 = intruder.id.index(ac2)

        if idx1 > -1 and idx2 > -1:
            dv_mvp = mvp_pair(ownship, intruder, conf, qdr, dist, tcpa,
                              idx1, idx2, cfg.resofach)
            # Simultaneous conflicts: an aircraft appears in several pairs, so
            # each pairwise change accumulates into the same slot (vector sum).
            dv[idx1] = dv[idx1] - dv_mvp

    # Combine into per-aircraft commands ---------------------------------
    dv = np.transpose(dv)
    v = np.array([ownship.gseast, ownship.gsnorth])
    newv = v + dv

    newtrack, newgs, newvs = horizontal_command(newv, ownship.vs)

    newgscapped, vscapped = cap_velocities(
        newgs, newvs,
        ownship.perf.vmin, ownship.perf.vmax,
        ownship.perf.vsmin, ownship.perf.vsmax)

    alt = ownship.selalt

    return newtrack, newgscapped, vscapped, alt
