''' Shared, immutable data structures for conflict detection. '''
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConflictState:
    '''Immutable result of a conflict-detection pass.

    Being ``frozen`` keeps the detection step side-effect free: callers receive
    a value they cannot accidentally mutate, which is what lets ``detect`` be
    treated as a pure function of its inputs.
    '''
    rpz: np.ndarray
    hpz: np.ndarray
    dtlookahead: list
    confpairs: list
    confpairs_unique: frozenset
    lospairs: list
    qdr: np.ndarray
    dist: np.ndarray
    dcpa: np.ndarray
    tcpa: np.ndarray
    tLOS: np.ndarray
    inconf: np.ndarray
    tcpamax: np.ndarray
