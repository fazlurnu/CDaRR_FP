'''CNS ADS-L observation layer — N×N last-known surveillance picture.

``obs[i, j]`` holds the last-received sensor value of target ``j`` as seen by
observer ``i``. Each tick :func:`update` writes ``sensor[j]`` into every row
``i`` where ``mask[i, j]`` is True; cells where it is False keep their previous
value verbatim (stale). No extra noise is added here — the value written is
exactly what the sensor produced.

Fields are stored as a ``dict[str, ndarray]`` inside the frozen dataclass so
:func:`update` can loop over :data:`FIELDS` rather than repeating the same
``np.where`` twelve times.
'''
from dataclasses import dataclass
from typing import Dict

import numpy as np

FIELDS = [
    'lat', 'lon', 'alt', 'hdg', 'trk', 'gs', 'tas', 'vs',
    'gseast', 'gsnorth', 'pos_acc', 'vel_acc',
]


@dataclass(frozen=True)
class ADSLObservation:
    '''Immutable N×N last-known store, one matrix per field.

    ``fields`` maps each field name to an ``(n, n)`` float array where
    ``fields[f][i, j]`` is what observer ``i`` last received about target
    ``j``'s field ``f``. ``id`` is the length-N list of target identifiers.
    '''
    n: int
    id: list
    fields: Dict[str, np.ndarray]


def empty_observation() -> ADSLObservation:
    '''Return the canonical zero-sized (unsized) observation state.'''
    return ADSLObservation(
        n=0,
        id=[],
        fields={f: np.zeros((0, 0), dtype=float) for f in FIELDS},
    )


def ensure_size(obs: ADSLObservation, n: int) -> ADSLObservation:
    '''Return an ADSLObservation whose matrices are exactly (n, n).

    If already the right size the same instance is returned unchanged.
    Otherwise a new instance is built: matrices are zero-initialised and the
    existing top-left sub-block is copied in to preserve prior observations.
    '''
    if obs.n == n:
        return obs
    m = min(obs.n, n)
    new_fields = {}
    for f in FIELDS:
        new_mat = np.zeros((n, n), dtype=float)
        if m:
            new_mat[:m, :m] = obs.fields[f][:m, :m]
        new_fields[f] = new_mat
    new_id = (obs.id + [''] * n)[:n]
    return ADSLObservation(n=n, id=new_id, fields=new_fields)


def update(obs: ADSLObservation, sensor, mask: np.ndarray) -> ADSLObservation:
    '''Return a new ADSLObservation with cells refreshed where mask is True.

    For each field ``f`` and each pair ``(i, j)``:
    - ``mask[i, j] == True``  → ``obs[i, j]`` becomes ``sensor.f[j]``
    - ``mask[i, j] == False`` → ``obs[i, j]`` keeps its previous value

    ``sensor.f`` is 1D length N; reshaping to ``(1, N)`` broadcasts target
    ``j``'s value across all observer rows ``i`` in one ``np.where`` call.
    '''
    obs = ensure_size(obs, sensor.n)
    new_fields = {}
    for f in FIELDS:
        sensor_row = np.asarray(getattr(sensor, f), dtype=float).reshape(1, -1)
        new_fields[f] = np.where(mask, sensor_row, obs.fields[f])
    return ADSLObservation(n=sensor.n, id=list(sensor.id), fields=new_fields)


def field(obs: ADSLObservation, name: str) -> np.ndarray:
    '''Return the (n, n) array for a named field.'''
    return obs.fields[name]
