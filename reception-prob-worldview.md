# Option B — Observer-Aware Detection via `WorldView`

**Goal.** Make CD / CR / CRR correct for **general N-aircraft** under
**`reception_prob < 1`**, by replacing the implicit "everyone sees the same
intruder" assumption with an explicit observer axis. The detection/resolution/
recovery kernels become *natively vectorized over `(observer, target)` pairs* —
no Python observer loop, single code path for `prob == 1` (ground truth /
perfect comms) and `prob < 1` (stale, observer-specific comms).

This is the **maximal-abstraction** option: we *do* modify the existing kernels'
signatures and internals, and we end up with one clean contract. Regression
safety at `prob == 1` is a hard requirement (see §2.3 and §9.2).

Audience: an implementer (Sonnet) coding this later. Every change below names the
exact file, function, current behaviour, and the new behaviour with code.

---

## 1. Background — why today's pipeline is `prob == 1` only

### 1.1 The current geometry convention

`cd/statebased.py` builds N×N matrices indexed `[i, j]` where **row `i` =
ownship, col `j` = intruder**:

- **Position** ([`relative_bearing_distance`](cd/statebased.py:20)) via
  `geo.kwikqdrdist_matrix(own, intr)` → `qdr[i,j] / dist[i,j]` is the bearing/
  distance from **ownship `i`** to **intruder `j`**. So
  `dx[i,j], dy[i,j] = pos[j] − pos[i]` (displacement i→j).
- **Velocity** ([`horizontal_conflict`](cd/statebased.py:43)):
  `du = ownu − intu.T` → `du[i,j] = own_east[j] − intr_east[i]`.
- **Vertical** ([`vertical_conflict`](cd/statebased.py:82)):
  `dalt[i,j] = own.alt[j] − intr.alt[i]`.

Note velocity/alt look "transposed" relative to position (own indexed by the
column, intruder by the row). **This is only self-consistent because every call
site passes the *same object* as both `ownship` and `intruder`** — i.e.
`detect(obs, obs, …)` and `detect(bs.traf, bs.traf, …)`. When `own ≡ intr`,
`own_X[j] = intr_X[j] = X[j]`, so every difference collapses to:

```
quantity[i, j]  =  X[col j]  −  X[row i]  =  X[j] − X[i]
```

i.e. "**the value of target `j` minus the value of observer `i`**", which is the
correct relative quantity for the pair (i, j). `tcpa`, `dcpa2`, etc. all come out
right.

### 1.2 What breaks at `prob < 1`

With imperfect/stale comms, **observer `i`'s view of `j` is no longer `X[j]`** —
it is `adsl[i, j]` (what `i` last received about `j`), which differs per
observer and is stale by a different amount for each `i`. There is **no single
1-D intruder vector** `X[j]`; the intruder picture is genuinely the N×N matrix
`adsl_field(cns, "X")` ([cns.py:117](sim_models/cns/cns.py:117)). The same-object
symmetry that made the transpose work is gone.

### 1.3 The invariant Option B enforces

For **every** pairwise quantity, cell `[i, j]` must be computed as:

```
quantity[i, j]  =  seen_X[i, j]   −   self_X[i]
                   └ i's ADS-L view └ i's own sensor
                     of target j       reading of itself
```

- `self_X[i]` = observer `i`'s **own** sensor reading of itself = diagonal
  `adsl[i,i]` = `cns.sensor.X[i]`.
- `seen_X[i, j]` = observer `i`'s last-received value of target `j` =
  `adsl_field(cns, "X")[i, j]`.

At `prob == 1`, `seen_X[i, j] = X[j]` (broadcast) and `self_X[i] = X[i]`, so
`quantity[i,j] = X[j] − X[i]` — **identical to today**. That collapse is what
guarantees regression safety.

The uniform refactor rule is therefore:

> **Everywhere the code computes `X[col] − X[row]` (i.e. an intruder value minus
> an ownship value), replace it with `seen_X[i, j] − self_X[i]`.**
> Everywhere it reads a *scalar per-pair* intruder value `intruder.X[idx2]`,
> replace it with `seen.X[idx1, idx2]`; keep ownship reads `ownship.X[idx1]` as
> `self.X[idx1]`.

---

## 2. Design overview

### 2.1 One new data structure: `WorldView`

A frozen dataclass that carries both halves of the invariant:

- `self_state` — a 1-D traffic-like namespace (length N): the observer's own
  reading of itself plus onboard-only fields (`perf`, `selalt`, autopilot
  targets). This is exactly today's `_as_obs(cns.sensor)`
  ([runner:77](runners/stochastic_pairwise_hor_conflict.py:77)).
- `seen` — `dict[str, ndarray(N, N)]`: `seen[f][i, j]` = observer `i`'s last
  received value of `j`'s field `f`. This is the CNS ADS-L matrices.

`self_state` and `seen` are aligned by aircraft index (both indexed like
`bs.traf`). Diagonal consistency holds: `seen[f][i, i] == self_state.f[i]`.

### 2.2 Kernel signature change

The duplicated `(ownship, intruder)` arguments — always the same object today —
collapse into a single `wv`:

| Kernel | Today | Option B |
|---|---|---|
| `cd.detect` | `detect(ownship, intruder, rpz, hpz, dtlk)` | `detect(wv, rpz, hpz, dtlk)` |
| `cr.mvp.resolve` | `resolve(conf, ownship, intruder, cfg)` | `resolve(conf, wv, cfg)` |
| `cr.vo.resolve` | `resolve(conf, ownship, intruder, cfg)` | `resolve(conf, wv, cfg)` |
| `crr.*` | `crr(state, conf, ownship, intruder, active, **p)` | `crr(state, conf, wv, active, **p)` |

The kernels stay **duck-typed** — they read `wv.self_state.X` and
`wv.seen["X"]` directly and never import the `WorldView` class (matching the
existing style where traffic objects are duck-typed `SimpleNamespace`s). Only
the **builders** and the **runner** construct a real `WorldView`.

### 2.3 Regression-safety contract

A broadcast `WorldView` built from a single traffic object (`prob == 1`, or
ground truth) **must reproduce today's numbers bit-for-bit** (tolerance ≤ 1e-9).
This is enforced by a dedicated equivalence test (§9.2). Implement the geometry
so the broadcast case is arithmetically identical to the current matrix path.

---

## 3. New package: `worldview/`

Create a new top-level package so cd/cr/crr can import the *type* (for typing/
docs) without pulling in BlueSky or CNS. The builders that depend on CNS live in
a separate submodule.

```
worldview/
  __init__.py        # re-exports WorldView + builders
  common.py          # WorldView dataclass + neutral helpers (no bluesky/cns import)
  build.py           # builders from CNS / truth (imports cns, optionally bluesky)
```

### 3.1 `worldview/common.py`

```python
'''WorldView — the observation picture handed to CD/CR/CRR.

Carries both halves of the observer-aware invariant:
  self_state.X[i]   = observer i's own sensor reading of itself  (1-D, length N)
  seen[f][i, j]     = observer i's last ADS-L view of target j    (N×N)

At reception_prob == 1 every observer sees truth, so seen[f] is the column
broadcast of self_state.f and the picture collapses to the legacy 1-D case.
'''
from dataclasses import dataclass
from typing import Dict

import numpy as np

# Geometry/velocity fields the kernels read out of `seen`. Must be a subset of
# the CNS observation FIELDS (sim_models/cns/adsl_observation.py:FIELDS).
SEEN_FIELDS = ['lat', 'lon', 'alt', 'trk', 'gs', 'vs',
               'gseast', 'gsnorth', 'pos_acc', 'vel_acc']


@dataclass(frozen=True)
class WorldView:
    '''Immutable observer-aware traffic picture.

    `self_state` is any 1-D traffic-like object (duck-typed) exposing at least
    ntraf, id, lat, lon, alt, trk, gs, vs, gseast, gsnorth, perf, selalt and the
    autopilot target fields used by recovery (seltrk/selspd or ap.trk/ap.tas).
    `seen` maps each SEEN_FIELDS name to an (N, N) array.
    '''
    n: int
    id: list
    self_state: object
    seen: Dict[str, np.ndarray]

    def self_arr(self, name: str) -> np.ndarray:
        '''1-D length-N array: observer's own reading of `name`.'''
        return np.asarray(getattr(self.self_state, name), dtype=float)

    def seen_mat(self, name: str) -> np.ndarray:
        '''(N, N) array: seen[name][i, j] = i's view of j.'''
        return self.seen[name]
```

### 3.2 `worldview/build.py`

Two builders. Both must produce a `WorldView` whose indices align with
`bs.traf` (so `id2idx` from `bs.traf.id2idx` stays valid in recovery).

```python
'''Builders that turn a CNS state (or ground truth) into a WorldView.'''
from types import SimpleNamespace

import numpy as np

from sim_models.cns.cns import adsl_field
from .common import WorldView, SEEN_FIELDS


def _self_state_from_sensor(sensor, *, perf, selalt, ap=None,
                            seltrk=None, selspd=None) -> SimpleNamespace:
    '''Observer's own state: position/velocity from the noisy sensor, onboard
    parameters (perf, selalt, autopilot targets) passed through from bs.traf.

    This is exactly today's `_as_obs(cns.sensor)`, minus the `adsl` shim (the
    accuracy now lives in WorldView.seen['pos_acc'/'vel_acc'] and on self).
    '''
    ns = SimpleNamespace(
        ntraf=sensor.n,
        id=list(sensor.id),
        lat=sensor.lat, lon=sensor.lon, alt=sensor.alt,
        trk=sensor.trk, gs=sensor.gs, vs=sensor.vs,
        gseast=sensor.gseast, gsnorth=sensor.gsnorth,
        pos_acc=sensor.pos_acc, vel_acc=sensor.vel_acc,   # 1-D self accuracy
        perf=perf, selalt=selalt,
    )
    # Optional autopilot targets used by get_desired_ownship_velocity.
    if ap is not None:
        ns.ap = ap
    if seltrk is not None:
        ns.seltrk = seltrk
    if selspd is not None:
        ns.selspd = selspd
    return ns


def worldview_from_cns(cns, *, perf, selalt, ap=None,
                       seltrk=None, selspd=None) -> WorldView:
    '''Observer-aware view from a stepped CNSState.

    self_state = noisy self-reading (diagonal of the obs); seen = the N×N ADS-L
    matrices. Stale cells are handled by the CNS obs layer, so seen already
    carries the correct per-observer staleness.
    '''
    sensor = cns.sensor
    self_state = _self_state_from_sensor(
        sensor, perf=perf, selalt=selalt, ap=ap, seltrk=seltrk, selspd=selspd)
    seen = {f: adsl_field(cns, f) for f in SEEN_FIELDS}
    return WorldView(n=sensor.n, id=list(sensor.id),
                     self_state=self_state, seen=seen)


def worldview_broadcast(traf) -> WorldView:
    '''Ground-truth / perfect-comms view: every observer sees truth.

    seen[f][i, j] = traf.f[j] for all i (column broadcast). Used for conf_gt and
    for any prob==1 path. Reproduces the legacy single-object detect exactly.
    '''
    n = int(traf.ntraf)
    seen = {}
    for f in SEEN_FIELDS:
        if hasattr(traf, f):
            row = np.asarray(getattr(traf, f), dtype=float).reshape(1, -1)  # (1,N)
            seen[f] = np.broadcast_to(row, (n, n)).copy()                   # (N,N)
        else:
            # pos_acc/vel_acc absent on bs.traf truth → zero accuracy.
            seen[f] = np.zeros((n, n), dtype=float)
    return WorldView(n=n, id=list(traf.id), self_state=traf, seen=seen)
```

> **Index alignment (critical).** `worldview_from_cns` and `worldview_broadcast`
> both index aircraft in `bs.traf` order, because `cns.sensor` is produced by
> `measure(bs.traf, …)` in that order. Recovery's `default_id2idx`
> ([crr/common.py:155](crr/common.py:155)) returns indices into `bs.traf`, so
> `seen[idx1, idx2]` is well-defined. Do **not** reorder ids in the builder.

### 3.3 `worldview/__init__.py`

```python
from .common import WorldView, SEEN_FIELDS
from .build import worldview_from_cns, worldview_broadcast

__all__ = ['WorldView', 'SEEN_FIELDS',
           'worldview_from_cns', 'worldview_broadcast']
```

---

## 4. `cd/statebased.py` — geometry rewrite

`ConflictState` ([cd/common.py](cd/common.py)) is **unchanged** (it already
stores flat per-conflict arrays). Only the geometry that feeds it changes.

### 4.1 New element-wise position helper (replaces `relative_bearing_distance`)

The current helper uses `geo.kwikqdrdist_matrix(own, intr)` which forms the
outer product of two 1-D arrays. We need per-observer targets, so compute the
flat-earth bearing/distance **element-wise** between `self_X[i]` (broadcast down
each row) and the `seen` matrix.

**Preferred implementation** — reuse BlueSky's own scalar function with
broadcasting so the math is guaranteed identical to the matrix version:

```python
def relative_bearing_distance(wv, eye):
    '''Bearing (deg) and distance (m) for every (observer i, target j) pair.

    qdr[i,j] / dist[i,j] is from observer i's own position to its view of j.
    The diagonal is pushed to a huge distance via `eye`.
    '''
    self_lat = wv.self_arr('lat')[:, None]   # (N,1) observer's own lat/lon
    self_lon = wv.self_arr('lon')[:, None]
    seen_lat = wv.seen_mat('lat')            # (N,N)
    seen_lon = wv.seen_mat('lon')

    # geo.kwikqdrdist is element-wise & broadcast-safe: (N,1) vs (N,N) -> (N,N).
    qdr, dist_nm = geo.kwikqdrdist(self_lat, self_lon, seen_lat, seen_lon)
    qdr = np.asarray(qdr)
    dist = np.asarray(dist_nm) * nm + _BIG * eye
    return qdr, dist
```

**Fallback** — if `geo.kwikqdrdist` is not broadcast-safe in this BlueSky build,
inline the flat-earth formula (this *is* what `kwikqdrdist_matrix` computes; same
constants as [crr/common.py:90](crr/common.py:90) and the runner's `_geom_dcpa`):

```python
def _pairwise_qdr_dist(self_lat, self_lon, seen_lat, seen_lon):
    re = 6371000.0
    la1 = self_lat[:, None]      # (N,1)
    lo1 = self_lon[:, None]
    dlat = np.radians(seen_lat - la1)
    dlon = np.radians(((seen_lon - lo1) + 180.0) % 360.0 - 180.0)
    cavelat = np.cos(np.radians(la1 + seen_lat) * 0.5)
    dangle = np.sqrt(dlat * dlat + dlon * dlon * cavelat * cavelat)
    dist_nm = re * dangle / nm
    qdr = np.degrees(np.arctan2(dlon * cavelat, dlat)) % 360.0
    return qdr, dist_nm
```

> Verify against `geo.kwikqdrdist_matrix` on the broadcast case in §9.2; if the
> fallback diverges, prefer whichever reproduces the matrix output.

### 4.2 `horizontal_conflict` — self/seen relative velocity

Current ([cd/statebased.py:43](cd/statebased.py:43)) uses
`_velocity_components(own…)` and `_velocity_components(int…)` then
`du = ownu − intu.T`. Replace the velocity block; keep the CPA math below it
identical.

```python
def horizontal_conflict(wv, qdr, dist, eye):
    '''Horizontal CPA geometry and entry/exit times of the protected zone.'''
    qdrrad = np.radians(qdr)
    dx = dist * np.sin(qdrrad)          # (N,N) relative position, i -> j
    dy = dist * np.cos(qdrrad)

    # Relative velocity for pair (i, j) AS OBSERVER i PERCEIVES IT:
    #   du[i,j] = seen_vel_east[i,j] - self_vel_east[i]
    # At prob==1 seen_*[i,j]=v[j], self_*[i]=v[i]  ->  v[j]-v[i] (legacy value).
    self_e = wv.self_arr('gseast')[:, None]   # (N,1)
    self_n = wv.self_arr('gsnorth')[:, None]
    seen_e = wv.seen_mat('gseast')            # (N,N)
    seen_n = wv.seen_mat('gsnorth')
    du = seen_e - self_e
    dv = seen_n - self_n

    dv2 = du * du + dv * dv
    dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)
    vrel = np.sqrt(dv2)

    tcpa = -(du * dx + dv * dy) / dv2 + _BIG * eye
    dcpa2 = np.abs(dist * dist - tcpa * tcpa * dv2)

    rpz_mat = _rpz_matrix(wv.n)          # see note below
    R2 = rpz_mat * rpz_mat
    swhorconf = dcpa2 < R2

    dxinhor = np.sqrt(np.maximum(0., R2 - dcpa2))
    dtinhor = dxinhor / vrel
    tinhor = np.where(swhorconf, tcpa - dtinhor, 1e8)
    touthor = np.where(swhorconf, tcpa + dtinhor, -1e8)

    return swhorconf, tcpa, dcpa2, tinhor, touthor, vrel, rpz_mat
```

> **`gseast`/`gsnorth` must be present in `seen`.** They are CNS `FIELDS`
> ([adsl_observation.py:18](sim_models/cns/adsl_observation.py:18)) and in
> `SEEN_FIELDS`, so `seen['gseast'][i,j]` is the velocity-east of `j` as received
> by `i`. Do **not** recompute them from `seen['gs']`/`seen['trk']` — the sensor
> already added velocity noise into the components
> ([sensor.py:84](sim_models/cns/sensor.py:84)); recomputing would double-count.

> **`rpz`/`hpz` handling.** Today `rpz` is a scalar passed into `detect`,
> expanded to `rpz_arr = [rpz]*ntraf` and the symmetric `rpz_mat = max(rpz,
> rpz.T)`. Keep that: `detect` still takes scalar `rpz, hpz`. Factor the
> `rpz_mat`/`hpz_mat` construction into small helpers `_rpz_matrix(n)` /
> `_hpz_matrix(n)` that close over the scalar, or pass them in. The protected
> zone is **not** observed/noised, so it stays truth-based.

### 4.3 `vertical_conflict` — self/seen relative altitude & vs

Current ([cd/statebased.py:82](cd/statebased.py:82)):
`dalt = own.alt[(1,N)] − intr.alt[(1,N)].T`. Replace with the invariant:

```python
def vertical_conflict(wv, hpz, eye):
    self_alt = wv.self_arr('alt')[:, None]   # (N,1)
    self_vs  = wv.self_arr('vs')[:, None]
    seen_alt = wv.seen_mat('alt')            # (N,N)
    seen_vs  = wv.seen_mat('vs')

    dalt = (seen_alt - self_alt) + _BIG * eye          # [i,j] = alt[j]-alt[i] @prob1
    dvs = seen_vs - self_vs
    dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)

    hpz_mat = _hpz_matrix(wv.n)
    tcrosshi = (dalt + hpz_mat) / -dvs
    tcrosslo = (dalt - hpz_mat) / -dvs
    tinver = np.minimum(tcrosshi, tcrosslo)
    toutver = np.maximum(tcrosshi, tcrosslo)
    return dalt, tinver, toutver, hpz_mat
```

### 4.4 `detect` — new signature, same assembly

```python
def detect(wv, rpz, hpz, dtlookahead) -> ConflictState:
    ntraf = wv.n
    rpz_arr = np.array([rpz] * ntraf)
    hpz_arr = np.array([hpz] * ntraf)
    dtlook_arr = [dtlookahead] * ntraf
    eye = np.eye(ntraf)

    qdr, dist = relative_bearing_distance(wv, eye)
    swhorconf, tcpa, dcpa2, tinhor, touthor, vrel, rpz_mat = horizontal_conflict(wv, qdr, dist, eye)
    dalt, tinver, toutver, hpz_mat = vertical_conflict(wv, hpz, eye)
    swconfl, tinconf = combine_conflicts(swhorconf, tinhor, touthor, tinver, toutver, dtlookahead, eye)

    inconf = np.any(swconfl, 1)
    tcpamax = np.max(tcpa * swconfl, 1)
    confpairs = _conflict_pairs(wv.id, swconfl)
    confpairs_unique = frozenset(frozenset(pair) for pair in confpairs)
    swlos = (dist < rpz_mat) * (np.abs(dalt) < hpz_mat)
    lospairs = _conflict_pairs(wv.id, swlos)

    return ConflictState(
        rpz=rpz_arr, hpz=hpz_arr, dtlookahead=dtlook_arr,
        confpairs=confpairs, confpairs_unique=confpairs_unique, lospairs=lospairs,
        qdr=qdr[swconfl], dist=dist[swconfl], dcpa=np.sqrt(dcpa2[swconfl]),
        tcpa=tcpa[swconfl], tLOS=tinconf[swconfl], inconf=inconf, tcpamax=tcpamax,
    )
```

`combine_conflicts`, `_conflict_pairs`, `_BIG` are unchanged.
**Important:** `confpairs` are emitted as `(wv.id[i], wv.id[j])` = `(observer,
target)`. The pair order encodes who-observes-whom, and CR/CRR rely on it
(idx1 = observer, idx2 = target). Preserve `_conflict_pairs` exactly.

---

## 5. `cr/` — resolution

### 5.1 `cr/mvp.py`

`mvp_pair` ([cr/mvp.py:20](cr/mvp.py:20)) reads ownship velocity at `idx1` and
intruder velocity at `idx2`. Change the intruder read to `seen[idx1, idx2]`;
ownship stays `self[idx1]`. The signature takes `wv` instead of
`(ownship, intruder)`.

```python
def mvp_pair(wv, conf, qdr, dist, tcpa, idx1, idx2, resofach):
    rpz_m = np.max(conf.rpz[[idx1, idx2]] * resofach)
    qdr = np.radians(qdr)
    drel = np.array([np.sin(qdr) * dist, np.cos(qdr) * dist])

    self_e = wv.self_arr('gseast'); self_n = wv.self_arr('gsnorth')
    seen_e = wv.seen_mat('gseast'); seen_n = wv.seen_mat('gsnorth')

    v1 = np.array([self_e[idx1],          self_n[idx1]])           # observer's own
    v2 = np.array([seen_e[idx1, idx2],    seen_n[idx1, idx2]])     # i's view of j
    vrel = v2 - v1
    # ... rest of the MVP math is UNCHANGED ...
```

`resolve` ([cr/mvp.py:61](cr/mvp.py:61)):

```python
def resolve(conf, wv, cfg: ResolutionConfig, resofach=None):
    if resofach is not None:
        cfg = cfg.with_resofach(resofach)
    ntraf = wv.n
    dv = np.zeros((ntraf, 2))
    for ((ac1, ac2), qdr, dist, tcpa) in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa):
        idx1 = wv.id.index(ac1)        # observer
        idx2 = wv.id.index(ac2)        # target
        if idx1 > -1 and idx2 > -1:
            dv_mvp = mvp_pair(wv, conf, qdr, dist, tcpa, idx1, idx2, cfg.resofach)
            dv[idx1] = dv[idx1] - dv_mvp
    dv = np.transpose(dv)
    v = np.array([wv.self_arr('gseast'), wv.self_arr('gsnorth')])   # own current vel
    newv = v + dv
    newtrack, newgs, newvs = horizontal_command(newv, wv.self_arr('vs'))
    perf = wv.self_state.perf
    newgscapped, vscapped = cap_velocities(newgs, newvs, perf.vmin, perf.vmax, perf.vsmin, perf.vsmax)
    alt = wv.self_state.selalt
    return newtrack, newgscapped, vscapped, alt
```

### 5.2 `cr/vo.py`

Same transformation. `vo_pair` ([cr/vo.py:52](cr/vo.py:52)) reads
`ownship.gsnorth[idx1]` / `intruder.gsnorth[idx2]` → `self[idx1]` /
`seen[idx1, idx2]`. `resolve` mirrors §5.1. Keep the existing multi-conflict
disclaimer comment ([cr/vo.py:119](cr/vo.py:119)).

---

## 6. `crr/` — recovery

The uniform rule again: **ownship/self at `idx1`, intruder via
`seen[idx1, idx2]`.** Recovery threads `RecoveryState`; that is unchanged.

### 6.1 `crr/common.py`

Functions that read traffic objects and need updating:

1. **`get_desired_ownship_velocity(ownship, idx, cache)`**
   ([crr/common.py:46](crr/common.py:46)) — reads the **ownship's own** autopilot
   target. Change the parameter from `ownship` to `wv.self_state` (a 1-D
   namespace), or pass `wv` and read `wv.self_state`. No `seen` needed (it's a
   self quantity). Logic otherwise unchanged.

2. **`compute_pair_positions(conf)`** ([crr/common.py:80](crr/common.py:80)) —
   derives `(dx, dy)` from `conf.qdr/conf.dist`, which are already
   observer-correct once §4 lands. **No change.**

3. **`get_relative_position(ownship, intruder, idx1, idx2)`**
   ([crr/common.py:90](crr/common.py:90)) — flat-earth fallback when a pair is
   not in `conf`. Change to use `wv`: observer position `self[idx1]`, target
   position `seen[idx1, idx2]`:
   ```python
   def get_relative_position(wv, idx1, idx2):
       re = 6371000.0
       o_lat = wv.self_arr('lat')[idx1]; o_lon = wv.self_arr('lon')[idx1]
       t_lat = wv.seen_mat('lat')[idx1, idx2]; t_lon = wv.seen_mat('lon')[idx1, idx2]
       dlon = float(t_lon - o_lon); dlat = float(t_lat - o_lat)
       latm = 0.5 * np.radians(float(t_lat + o_lat))
       dx = re * np.radians(dlon) * np.cos(latm)
       dy = re * np.radians(dlat)
       return dx, dy
   ```

4. **`get_pair_dxdy(conflict, pair_dxdy, wv, idx1, idx2)`**
   ([crr/common.py:101](crr/common.py:101)) — drop the `ownship, intruder`
   params, pass `wv`; the fallback calls the new `get_relative_position(wv, …)`.

5. **`record_initial_intruder_velocity(state, conf, wv, id2idx)`**
   ([crr/common.py:134](crr/common.py:134)) — records the intruder's velocity at
   conflict onset **as the observer saw it**:
   ```python
   seen_e = wv.seen_mat('gseast'); seen_n = wv.seen_mat('gsnorth')
   for pair in newpairs:
       idx1, idx2 = id2idx(pair)
       if idx1 >= 0 and idx2 >= 0:
           init_vel[pair] = (float(seen_e[idx1, idx2]), float(seen_n[idx1, idx2]))
   ```

6. `default_id2idx`, `default_recover`, `apply_active_changes`,
   `anglediff`, `calculate_dcpa`, `_val` — **unchanged**.

### 6.2 `crr/ftr.py` — `resumenav_double_criteria`

([crr/ftr.py:40](crr/ftr.py:40)) Signature `(state, conf, wv, active, **params)`.
Inside the loop:
```python
seen_e = wv.seen_mat('gseast'); seen_n = wv.seen_mat('gsnorth')
...
Vo_u, Vo_v = get_desired_ownship_velocity(wv.self_state, idx1, vod_cache)
Vi_c_u = float(seen_e[idx1, idx2])      # was intruder.gseast[idx2]
Vi_c_v = float(seen_n[idx1, idx2])
dx, dy = get_pair_dxdy(conflict, pair_dxdy, wv, idx1, idx2)
rpz = float(np.max(conf.rpz[[idx1, idx2]]))
```
Criterion-1/2 math otherwise identical.

### 6.3 `crr/cpa.py` — `resumenav_cpa`

([crr/cpa.py:37](crr/cpa.py:37)) Signature `(state, conf, wv, active, **params)`.
```python
seen_e = wv.seen_mat('gseast'); seen_n = wv.seen_mat('gsnorth')
self_e = wv.self_arr('gseast');  self_n = wv.self_arr('gsnorth')
seen_trk = wv.seen_mat('trk');   self_trk = wv.self_arr('trk')
...
dx, dy = get_relative_position(wv, idx1, idx2)
dist = np.array([dx, dy])
vrel = np.array([self_e[idx1] - seen_e[idx1, idx2],
                 self_n[idx1] - seen_n[idx1, idx2]])
rpz = float(np.max(conf.rpz[[idx1, idx2]]))
# bounce check uses observer's own track vs its view of the intruder's track:
_is_bouncing(dist, self_trk[idx1], seen_trk[idx1, idx2], rpz, resofach)
```
`_past_cpa`, `_hor_los`, `_is_bouncing` are unchanged (they take vectors/scalars).

### 6.4 `crr/probabilistic_ftr.py`

([crr/probabilistic_ftr.py:259](crr/probabilistic_ftr.py:259)) Signature
`(state, conf, wv, active, **params)`. Two changes:

1. **Velocities** as in §6.2 (`seen[idx1, idx2]` for the intruder current/initial
   velocity, `self_state` autopilot target for the ownship desired velocity).

2. **Covariances.** `_aircraft_covariance(traffic, idx, attr)`
   ([crr/probabilistic_ftr.py:236](crr/probabilistic_ftr.py:236)) currently reads
   `traffic.adsl.<attr>[idx]` (1-D). Split into self vs seen:
   - ownship (self) accuracy: `wv.self_arr('pos_acc')[idx1]` (1-D).
   - intruder (seen) accuracy: `wv.seen_mat('pos_acc')[idx1, idx2]` — i.e. the
     accuracy `j` advertised, as received by `i`.

   Refactor `_aircraft_covariance` to take a **scalar accuracy value** and return
   `σ²I`:
   ```python
   def _cov_from_acc(acc_95, eps=1e-6):
       '''2×2 isotropic covariance from a 95% accuracy radius (m). acc≈0 -> ~0.'''
       try:
           sigma = float(acc_95) / _SCALE_95
       except (TypeError, ValueError):
           sigma = 0.0
       return _regularize_spd(sigma ** 2 * np.eye(2), eps=eps)
   ```
   Then per pair:
   ```python
   pos_self = wv.self_arr('pos_acc'); pos_seen = wv.seen_mat('pos_acc')
   vel_self = wv.self_arr('vel_acc'); vel_seen = wv.seen_mat('vel_acc')
   Sigma_r = _cov_from_acc(pos_self[idx1]) + _cov_from_acc(pos_seen[idx1, idx2])
   Sigma_v = _cov_from_acc(vel_self[idx1]) + _cov_from_acc(vel_seen[idx1, idx2])
   ```
   The projected-normal math (`analytical_dcpa_prob_gt`, `Phi`, etc.) is
   **unchanged**.

### 6.5 `crr/__init__.py`

`make_recovery` and `RECOVERY_STRATEGIES` are unchanged — the strategy callables
keep the *uniform* signature, just with `wv` replacing `(ownship, intruder)`.
Update the docstring signature line to
`(state, conf, wv, active, **params) -> (new_state, delpairs)`.

---

## 7. Runner changes (both runners)

Apply identically to:
- `runners/stochastic_pairwise_hor_conflict.py` (loop body
  [:297-304](runners/stochastic_pairwise_hor_conflict.py:297))
- `runners/stochastic_pairwise_hor_conflict_heterogeneous_speed.py` (loop body
  [:219-223](runners/stochastic_pairwise_hor_conflict_heterogeneous_speed.py:219))

### 7.1 Replace `_as_obs` with WorldView builders

Delete `_as_obs` ([:77](runners/stochastic_pairwise_hor_conflict.py:77)). In the
ASAS tick:

```python
from worldview import worldview_from_cns, worldview_broadcast

# inside the tick (t + eps >= next_event_t):
cns = cns_step(cns, bs.traf)
wv = worldview_from_cns(
    cns,
    perf=bs.traf.perf, selalt=bs.traf.selalt,
    ap=getattr(bs.traf, 'ap', None),
    seltrk=getattr(bs.traf, 'seltrk', None),
    selspd=getattr(bs.traf, 'selspd', None),
)
wv_gt = worldview_broadcast(bs.traf)            # ground-truth view for scoring

conf    = cd(wv, rpz, hpz, dtlookahead)         # observer-aware detection
conf_gt = cd(wv_gt, rpz, hpz, dtlookahead)      # truth-based, for done/IPR only

newtrack, newgs, newvs, alt = cr(conf, wv, cfg)
recovery_state, _ = crr(recovery_state, conf, wv, active)
action = (newtrack, newgs, newvs, alt, list(recovery_state.resopairs))
```

`conf_gt` stays a ground-truth quantity (used only by `done_now`
[:306](runners/stochastic_pairwise_hor_conflict.py:306) and IPR). Do **not** feed
`wv` into scoring — that would let the controller's noisy view leak into the
truth metric.

### 7.2 Injected-stage defaults

`run_single`'s `cd=detect, cr=mvp.resolve` defaults
([:187-188](runners/stochastic_pairwise_hor_conflict.py:187)) keep working with
the new arity. Update the docstring's stage-signature block
([:200-203](runners/stochastic_pairwise_hor_conflict.py:200)) to:
```
cd(wv, rpz, hpz, dtlookahead) -> conf
cr(conf, wv, cfg) -> (newtrack, newgs, newvs, alt)
crr(state, conf, wv, active) -> (new_state, delpairs)
```

### 7.3 `_geom_dcpa` (history recording)

`_geom_dcpa(view, env)`
([:104](runners/stochastic_pairwise_hor_conflict.py:104)) takes a 1-D
traffic-like object (`bs.traf` or `cns.sensor`) and is **per-pair**, not
observer-matrix. It is independent of detect's internals, so leave it as-is —
keep passing `bs.traf` (truth) and `cns.sensor` (the observer's own reading).
This is a diagnostic, not part of the control loop.

---

## 8. Backward compatibility & migration strategy

The signature change is breaking. Two ways to land it; pick one up front:

- **(A) Hard cutover (recommended for a research repo).** Change all kernels,
  call sites, and tests in one branch. Smaller net surface, no shim to delete
  later. The equivalence test (§9.2) guards behaviour.

- **(B) Dual-arity shim (if you must keep old call sites alive).** Add a thin
  wrapper that accepts either `(wv)` or `(ownship, intruder)` and, in the latter
  case, builds `worldview_broadcast`-style on the fly. Marked deprecated. More
  code, easy to misuse; only worth it if external scripts call `detect` directly.

Recommended order regardless: land `worldview/` first (additive, no behaviour
change), then convert kernels bottom-up (cd → cr → crr), then the runners, then
delete `_as_obs`.

---

## 9. Tests

### 9.1 New `tests/test_worldview.py`

- `worldview_broadcast(traf)`: `seen[f][i, j] == traf.f[j]` for all i, j;
  `self_state is traf`; `seen[f][i, i] == self_arr(f)[i]`.
- `worldview_from_cns(cns, …)`: after a `cns_step`, `seen['lat']` equals
  `adsl_field(cns, 'lat')`; diagonal equals `cns.sensor.lat`
  (`seen[f][i,i] == self_arr(f)[i]`); `pos_acc`/`vel_acc` present in `seen`.
- Frozen: rebinding a field on `WorldView` raises.

### 9.2 Regression equivalence (the safety net) — `tests/test_worldview_equiv.py`

Build a small fixed scenario (reuse `tests/conftest.py` fakes / `make_cns_states`
[refactor plan §0]). Assert **bit-for-bit** equality (`atol=1e-9`) between:

- `detect(worldview_broadcast(traf), rpz, hpz, dtlk)` and the **pre-refactor**
  `detect(traf, traf, rpz, hpz, dtlk)`. Compare `confpairs`, `lospairs`, and each
  numeric `ConflictState` array (`qdr, dist, dcpa, tcpa, tLOS, inconf, tcpamax`).
- Same for `mvp.resolve` / `vo.resolve` outputs `(newtrack, newgs, newvs, alt)`.
- Same for each recovery strategy's `(new_state.resopairs, delpairs)` and the
  `active` writes, with `worldview_broadcast` vs a same-object traffic.

Practical way to get the "pre-refactor" reference: capture golden outputs from
`git stash`/the current `main` for the fixed scenario and store them as the
expected values, OR keep a copy of the old `detect` under
`tests/_legacy_detect.py` for the duration of the migration. The point is to
prove the broadcast path reproduces today's numbers.

### 9.3 Observer-awareness (`prob < 1`) — `tests/test_worldview_observer.py`

Construct a `CNSState` where observer `i` holds a **stale** view of `j` (set
`reception_prob < 1` and step until a cell goes stale, or hand-craft an
`ADSLObservation` with a known stale cell via `set_pair`/manual `update`). Then:

- `detect(wv, …)` uses `seen[i, j]` for pair (i, j): move `j` in truth while its
  cell is stale; assert the detected `dist`/`tcpa` for (i, j) reflect the
  **stale** position, not truth.
- Two observers `i`, `k` with different staleness of the same `j` produce
  **different** per-pair geometry — impossible to express in the old 1-D
  contract. This is the core new capability; assert it explicitly.
- Recovery: `record_initial_intruder_velocity` stores `seen[idx1, idx2]`, so two
  observers of the same intruder can record different `init_vel`.

### 9.4 Update existing kernel tests

`tests/test_statebased.py`, `tests/test_mvp.py`, `tests/test_vo.py`,
`tests/test_recovery.py` call the kernels with `(ownship, intruder)`. Wrap their
inputs: `wv = worldview_broadcast(fake_traffic)` and call the new signatures.
Where a test deliberately passed *different* ownship/intruder objects (if any),
build a `WorldView` whose `self_state` is the ownship and whose `seen` is the
column broadcast of the intruder — semantically identical to the old call.

---

## 10. Commit order (one logical change per commit)

1. `worldview/` package (`common.py`, `build.py`, `__init__.py`) + `test_worldview.py`. Additive; nothing else imports it yet.
2. `cd/statebased.py` → `detect(wv, …)` + geometry rewrite; update `tests/test_statebased.py`; add `tests/test_worldview_equiv.py` (detect half).
3. `cr/mvp.py` + `cr/vo.py` → `resolve(conf, wv, cfg)`; update `tests/test_mvp.py`, `tests/test_vo.py`; extend equivalence test.
4. `crr/common.py` + the three strategies → `(…, wv, …)`; update `tests/test_recovery.py`; extend equivalence test.
5. Runners → WorldView builders; delete `_as_obs`; docstrings.
6. `tests/test_worldview_observer.py` (the `prob < 1` capability tests).
7. Sweep: run the full suite + the experiment scripts (`experiments/exp1/2/3`) headless to confirm IPR numbers are unchanged at `prob == 1`.

After each of 2–5, the equivalence test (§9.2) must stay green.

---

## 11. Acceptance criteria

- All existing tests pass after their signature updates.
- §9.2 equivalence test passes at `atol ≤ 1e-9` for detect, both resolvers, and
  all three recovery strategies.
- §9.3 observer tests pass (stale, per-observer geometry is honoured).
- `experiments/exp1-crossing-angle.py`, `exp2-gamma.py`,
  `exp3-noise-model-random-angle.py` produce **identical** IPR/figures when run
  at the settings they use today (those run at `reception_prob = 1.0`, so the
  broadcast path must reproduce current results).
- No kernel imports `bluesky` or `sim_models.cns` (only `worldview/build.py` and
  the runners do).

---

## 12. Risks & gotchas

1. **Velocity double-counting.** Use `seen['gseast'/'gsnorth']` directly; do not
   reconstruct from `seen['gs']`/`seen['trk']`. The sensor already folded
   velocity noise into the components ([sensor.py:84](sim_models/cns/sensor.py:84)).
2. **Index alignment.** `wv` indices must match `bs.traf` (recovery's `id2idx`
   returns `bs.traf` indices). The builders preserve order; never sort ids.
3. **`kwikqdrdist` broadcasting.** Confirm `geo.kwikqdrdist` accepts `(N,1)` vs
   `(N,N)` inputs in this BlueSky build (§4.1). If not, use the inlined
   `_pairwise_qdr_dist` and verify it matches `kwikqdrdist_matrix` on the
   broadcast case.
4. **Diagonal masking.** Keep `_BIG * eye`. With `seen[i,i] == self[i]`, the
   diagonal distance is ~0 and would self-flag without the mask.
5. **`pos_acc`/`vel_acc` absent on truth.** `worldview_broadcast(bs.traf)` zeroes
   them (bs.traf has no accuracy fields). Probabilistic recovery on `conf_gt` is
   never used, so this is fine; but if someone runs the probabilistic rule on a
   broadcast/truth view, the covariance collapses to ~0 (deterministic FTR
   limit) — document, don't fix.
6. **`rpz`/`hpz` stay scalar truth inputs** to `detect`; they are not observed.
   Don't route them through `seen`.
7. **Per-pair `conf.rpz[[idx1, idx2]]`** indexing in CR/CRR relies on the
   per-aircraft `rpz_arr`; keep `detect` emitting `rpz_arr = [rpz]*ntraf`.
</content>
</invoke>
