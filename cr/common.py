''' Shared, pure building blocks for conflict resolution.

Both the MVP and VO resolvers are expressed as plain functions operating on
explicit inputs. The pieces they have in common live here:

* :class:`ResolutionConfig` — an immutable bag of resolution factors.
* small pure maths helpers used by both resolvers.

Nothing here touches global BlueSky state.
'''
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ResolutionConfig:
    '''Immutable resolution settings (formerly mutable instance switches).'''
    resofach: float = 1.0      # horizontal margin / maneuver fraction

    def with_resofach(self, resofach: float) -> 'ResolutionConfig':
        '''Return a copy with an updated horizontal resolution factor.'''
        return replace(self, resofach=resofach)


def horizontal_command(newv, own_vs):
    '''Convert the resolved cartesian velocity into autopilot commands.

    ``newv`` is the ``(3, ntraf)`` cartesian velocity. Resolutions are
    horizontal-only: track and ground speed come from the resolved velocity,
    while the vertical speed is left at the aircraft's current value. Returns
    ``(newtrack, newgs, newvs)`` as arrays.
    '''
    newtrack = (np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi) % 360
    newgs = np.sqrt(newv[0, :] ** 2 + newv[1, :] ** 2)
    newvs = own_vs
    return newtrack, newgs, newvs


def cap_velocities(newgs, newvs, vmin, vmax, vsmin, vsmax):
    '''Clamp ground speed and vertical speed to the performance envelope.'''
    newgscapped = np.maximum(vmin, np.minimum(vmax, newgs))
    vscapped = np.maximum(vsmin, np.minimum(vsmax, newvs))
    return newgscapped, vscapped
