'''Conflict-recovery (resume-navigation) package — functional models.'''
import functools

from .common import RecoveryState, empty_recovery_state
from .cpa import resumenav_cpa
from .ftr import resumenav_double_criteria
from .probabilistic_ftr import resumenav_probabilistic_ftr

# Strategy registry. Every value is a recovery callable with the uniform
# signature ``(state, conf, ownship, intruder, active, **params) ->
# (new_state, delpairs)``.
RECOVERY_STRATEGIES = {
    'cpa': resumenav_cpa,
    'double_criteria': resumenav_double_criteria,
    'probabilistic': resumenav_probabilistic_ftr,
    'probabilistic_ftr': resumenav_probabilistic_ftr,
}


def make_recovery(name, **params):
    '''Build a uniform ``crr(state, conf, ownship, intruder, active)`` callable.

    ``name`` selects a strategy from ``RECOVERY_STRATEGIES``; ``params`` are the
    strategy's extra arguments (e.g. ``resofach``, ``recover``,
    ``prob_threshold``, ``Ktheta``) bound up front. Each strategy ignores any
    ``params`` key it doesn't use, so the caller may pass a single superset.
    '''
    try:
        fn = RECOVERY_STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown recovery strategy: {name!r}")
    return functools.partial(fn, **params)


__all__ = [
    'RecoveryState',
    'empty_recovery_state',
    'resumenav_cpa',
    'resumenav_double_criteria',
    'resumenav_probabilistic_ftr',
    'RECOVERY_STRATEGIES',
    'make_recovery',
]
