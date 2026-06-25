'''Shared test fixtures and lightweight fakes.

The refactored detect/resolve functions are duck-typed over BlueSky's
``Traffic`` object: they only touch a handful of array attributes. Rather than
spin up a full simulation we build minimal stand-ins with numpy arrays, which is
exactly what makes the functional refactor testable.
'''
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

# Make the top-level ``cd`` / ``cr`` / ``crr`` packages importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_id2idx(ownship, intruder):
    '''Build a pure index resolver mimicking ``bs.traf.id2idx`` for fakes.'''
    def id2idx(pair):
        a, b = pair
        i1 = ownship.id.index(a) if a in ownship.id else -1
        i2 = intruder.id.index(b) if b in intruder.id else -1
        return i1, i2
    return id2idx


def make_recorder():
    '''A recover callback that records which aircraft indices it was called on.'''
    calls = []
    return calls, calls.append


def make_traffic(lat, lon, trk, gs, alt=None, vs=None, ids=None):
    '''Build a fake Traffic-like object from per-aircraft state.

    ``trk`` is in degrees, ``gs`` in m/s. East/north ground-speed components are
    derived from track and ground speed, matching BlueSky's convention
    (east = gs*sin(trk), north = gs*cos(trk)).
    '''
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    trk = np.asarray(trk, dtype=float)
    gs = np.asarray(gs, dtype=float)
    n = len(lat)

    alt = np.zeros(n) if alt is None else np.asarray(alt, dtype=float)
    vs = np.zeros(n) if vs is None else np.asarray(vs, dtype=float)
    ids = [f'AC{i + 1}' for i in range(n)] if ids is None else list(ids)

    trkrad = np.radians(trk)
    gseast = gs * np.sin(trkrad)
    gsnorth = gs * np.cos(trkrad)

    perf = SimpleNamespace(
        vmin=np.zeros(n),
        vmax=np.ones(n) * 1e9,
        vsmin=-np.ones(n) * 1e9,
        vsmax=np.ones(n) * 1e9,
    )
    ap = SimpleNamespace(vs=np.zeros(n))

    return SimpleNamespace(
        ntraf=n,
        id=ids,
        lat=lat, lon=lon,
        trk=trk, gs=gs,
        alt=alt, vs=vs,
        gseast=gseast, gsnorth=gsnorth,
        selalt=alt.copy(),
        perf=perf,
        ap=ap,
    )


@pytest.fixture
def head_on():
    '''Two aircraft on a head-on collision course along the same meridian.

    AC1 flies north (trk=0) from the south, AC2 flies south (trk=180) from the
    north; both at the same longitude and altitude, closing on each other.
    '''
    lat = [0.00, 0.05]      # AC2 is ~5.5 km north of AC1
    lon = [0.00, 0.00]
    trk = [0.0, 180.0]      # toward each other
    gs = [100.0, 100.0]     # m/s
    return make_traffic(lat, lon, trk, gs)


@pytest.fixture
def diverging():
    '''Two aircraft flying directly away from each other — no conflict.'''
    lat = [0.00, 0.05]
    lon = [0.00, 0.00]
    trk = [180.0, 0.0]      # away from each other
    gs = [100.0, 100.0]
    return make_traffic(lat, lon, trk, gs)
