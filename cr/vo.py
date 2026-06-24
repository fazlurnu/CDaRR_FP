''' Conflict resolution based on the Velocity Obstacle (VO) algorithm.

Functional rewrite of the former ``VO(Entity)`` class. The geometry is now a set
of plain functions:

* :func:`tangent_points`  — collision-cone tangent points for a pair.
* :func:`vo_pair`         — resolution velocity for a single conflict pair.
* :func:`resolve`         — combine per-pair resolutions into commands.

Mutable switches became a :class:`~cr.common.ResolutionConfig`.
'''
from math import asin, atan2, cos, sin, sqrt

import numpy as np
from shapely.geometry import LineString, Point
from shapely.affinity import translate
from shapely.ops import nearest_points

from .common import (
    ResolutionConfig,
    cap_velocities,
    horizontal_command,
)


def tangent_points(ownship_position, intruder_position, rpz):
    '''Tangent points of the collision cone from ownship around the intruder.

    Returns ``(Point, Point)`` or ``(None, None)`` when the aircraft are already
    closer than ``rpz`` (cone undefined). Pure.
    '''
    dx = intruder_position.x - ownship_position.x
    dy = intruder_position.y - ownship_position.y

    d = sqrt(dx ** 2 + dy ** 2)

    if d > rpz:
        theta = atan2(dy, dx)
        beta = asin(rpz / d)
        side = sqrt(d ** 2 - rpz ** 2)

        tp_1_x = ownship_position.x + side * cos(theta - beta)
        tp_1_y = ownship_position.y + side * sin(theta - beta)
        tp_2_x = ownship_position.x + side * cos(theta + beta)
        tp_2_y = ownship_position.y + side * sin(theta + beta)

        return Point(tp_1_x, tp_1_y), Point(tp_2_x, tp_2_y)

    return None, None


def vo_pair(ownship, intruder, conf, qdr, dist, idx1, idx2, resofach, method=0):
    '''Velocity Obstacle resolution for one conflict pair.

    Returns the 2-D horizontal resolution velocity ``[dv_east, dv_north]`` for
    the ownship. Pure.
    '''
    rpz = np.max(conf.rpz[[idx1, idx2]] * resofach)

    qdr = np.radians(qdr)

    # Relative position vector between idx1 and idx2.
    drel = np.array([np.sin(qdr) * dist,
                     np.cos(qdr) * dist])

    ownship_position = Point(0, 0)
    intruder_position = Point(drel[1], drel[0])

    tp_1, tp_2 = tangent_points(ownship_position, intruder_position, rpz)

    ownship_velocity = Point(ownship.gsnorth[idx1], ownship.gseast[idx1])
    intruder_velocity = Point(intruder.gsnorth[idx2], intruder.gseast[idx2])

    if (tp_1 is not None) and (tp_2 is not None):
        vo_0 = translate(ownship_position, xoff=intruder_velocity.x, yoff=intruder_velocity.y)
        vo_1 = translate(tp_1, xoff=intruder_velocity.x, yoff=intruder_velocity.y)
        vo_2 = translate(tp_2, xoff=intruder_velocity.x, yoff=intruder_velocity.y)

        vo_line_1 = LineString([vo_0, vo_1])
        vo_line_2 = LineString([vo_0, vo_2])

        # method 0: optimal (closest cone edge), 1: spd change, 2: hdg change.
        if method == 0:
            cp_1 = nearest_points(vo_line_1, ownship_velocity)[0]
            cp_2 = nearest_points(vo_line_2, ownship_velocity)[0]

            cp = cp_1 if cp_1.distance(ownship_velocity) <= cp_2.distance(ownship_velocity) else cp_2

        dv1 = ownship_velocity.y - cp.y
        dv2 = ownship_velocity.x - cp.x
    else:
        dv1 = 0
        dv2 = 0

    return np.array([dv1, dv2])


def resolve(conf, ownship, intruder, cfg: ResolutionConfig, resofach=None):
    '''Resolve all current conflicts with VO.

    Returns ``(newtrack, newgs, newvs, alt)`` — the per-aircraft ASAS commands.
    '''
    if resofach is not None:
        cfg = cfg.with_resofach(resofach)

    ntraf = ownship.ntraf

    # Per-aircraft running sum of horizontal resolution velocities.
    dv = np.zeros((ntraf, 2))

    for ((ac1, ac2), qdr, dist) in zip(
            conf.confpairs, conf.qdr, conf.dist):
        idx1 = ownship.id.index(ac1)
        idx2 = intruder.id.index(ac2)

        if idx1 > -1 and idx2 > -1:
            dv_vo = vo_pair(ownship, intruder, conf, qdr, dist,
                            idx1, idx2, cfg.resofach)
            # Simultaneous conflicts: an aircraft appears in several pairs, so
            # each pairwise change accumulates into the same slot (vector sum).
            #
            # DISCLAIMER: this is only validated on isolated two-drone
            # conflicts, where summing is fine because there is a single pair.
            # For genuine multi/simultaneous conflicts this is NOT correct VO —
            # it should instead resolve against the union of the velocity
            # obstacles (pick a velocity outside all VOs), not add per-pair
            # changes. Revisit before relying on multi-conflict scenarios.
            dv[idx1] = dv[idx1] - dv_vo

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

