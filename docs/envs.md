# Envs — Scenario Construction & BlueSky Stepping

**Package:** `envs/`
**Module:** `envs/pairwise_hor_conflict.py`

An *env* owns the **scenario**: it spawns aircraft in BlueSky, advances the
simulation one tick at a time, and reports back the quantities a runner needs
(ownship–intruder distances, avoidance flags). It deliberately knows nothing
about CD/CR/CRR or the CNS — it only sets up traffic and steps it. The
algorithmic pipeline lives in the [runner](runners.md).

---

## Pairwise horizontal conflict

The one env today builds **N = `pair_width` × `pair_height`** ownship/intruder
pairs on a lat/lon grid. Each ownship heads north (`hdg = 0`); each intruder is
spawned on a collision course with its ownship using BlueSky's `creconfs`
(create-in-conflict).

```
DRO000 ── north ──►        DRO001 ── north ──►
   ▲                          ▲
   │ dcpa=0, tlosh=dtlook     │
DRI000 (dpsi rel. hdg)     DRI001
```

- **Ownships** are `DRO000, DRO001, …`; **intruders** are `DRI000, DRI001, …`.
- `creconfs(targetidx, dpsi, dcpa, tlosh, spd)` places each intruder so that, on
  current velocities, it reaches the ownship's position in `tlosh` seconds with a
  closest-approach distance of `dcpa` (here `_DCPA_NM = 0.0`, a dead-centre hit).
- `init_dpsi` fixes the relative heading for every intruder; if omitted, each
  intruder gets a random heading in `[0, 360)`.

---

## `PairwiseHorConflictEnv`

An immutable descriptor returned by the factory and threaded into `step`. The
mutable traffic state itself lives outside, in `bs.traf`.

```python
@dataclass(frozen=True)
class PairwiseHorConflictEnv:
    nb_pair: int                  # number of ownship/intruder pairs
    init_speed_ownship: float
    init_speed_intruder: float
    ownship_ids: tuple            # ("DRO000", …)
    intruder_ids: tuple           # ("DRI000", …)
    ownship_idx: tuple            # bs.traf indices of the ownships
    intruder_idx: tuple           # bs.traf indices of the intruders
    init_heading: object          # np.ndarray (2·nb_pair,) — nominal heading per ac
```

The `*_idx` / `*_ids` tables are the bridge between the per-pair logical view and
BlueSky's flat aircraft arrays; runners use them to slice trajectories and label
plots.

---

## API

### `make_pairwise_hor_conflict(...)` → `PairwiseHorConflictEnv`

Spawns the aircraft and returns the descriptor.

| Parameter | Meaning |
|---|---|
| `pair_width`, `pair_height` | grid dimensions; `nb_pair = width × height` |
| `asas_pzr_m` | protected-zone radius (m); sets `bs.settings.asas_pzr` |
| `dtlookahead` | look-ahead time (s); also the intruders' `tlosh` |
| `init_speed_ownship`, `init_speed_intruder` | spawn speeds (m/s) |
| `aircraft_type_ownship` | BlueSky performance model, e.g. `"M600"` |
| `start_lat`, `start_lon`, `delta_lat_lon` | grid origin and pair spacing (deg) |
| `aircraft_type_intruder` | defaults to the ownship type if `None` |
| `init_dpsi` | fixed intruder relative heading; random per intruder if `None` |
| `simdt_factor` | integer multiplier on `bs.settings.simdt` (coarser steps) |

Side effects on creation: sets `bs.settings.asas_pzr`, `bs.settings.asas_dtlookahead`,
and stacks a `DT` command so the sim timestep is `simdt × simdt_factor`.

### `step(env, action)` → `np.ndarray`

Applies `action`, advances BlueSky one tick, and returns the per-pair
ownship–intruder distance in **metres**, shape `(nb_pair,)`.

`action` is the 5-tuple the runner assembles from CR + CRR:

```python
action = (reso_hdg, reso_spd, reso_vs, reso_alt, resopairs)
#          ▲ from cr (per-ac)          ▲ list of pairs still resolving (from crr)
```

`None` is accepted (no resolution yet) and treated as "everyone flies nominal."

### `reset()` → `None`

Clears all BlueSky traffic. Call between episodes.

### `avoidance_mask(action)` → `np.ndarray`

Per-aircraft flags (length `bs.traf.ntraf`, order matches `bs.traf.id`): `1.0`
if the aircraft is in an active resolution pair (currently manoeuvring), else
`0.0`. Used by runners for the avoidance/trajectory figures.

---

## How an action is applied (`_apply_action`)

Each tick, every aircraft is commanded either its **resolution** velocity or its
**nominal** velocity, decided by the avoidance mask:

```python
avoiding = avoidance_mask(action)
for i in range(bs.traf.ntraf):
    if avoiding[i]:
        bs.stack.stack(f"HDG {id}, {reso_hdg[i]}")      # follow CR resolution
        bs.stack.stack(f"SPD {id}, {reso_spd[i]/kts}")
    else:
        bs.stack.stack(f"HDG {id}, {env.init_heading[i]}")   # resume nominal
        bs.stack.stack(f"SPD {id}, {nom_spd/kts}")
```

This is why the recovery layer's `recover` callback is a **no-op** in the
pairwise runner (see [crr docs](crr.md)): the env already re-commands the nominal
heading/speed for any aircraft that is no longer in a resolution pair, so route
resumption needs no waypoint `direct()`.

---

## Distances (`_compute_distances`)

Per-pair great-circle distance between each ownship and its intruder, via
`geo.latlondist_matrix`, returned in metres (the diagonal of the cross-matrix):

```python
dist = geo.latlondist_matrix(lat_own, lon_own, lat_int, lon_int)
return np.diag(dist) * NM2M
```

The runner accumulates these per tick; the minimum over time per pair is the
realised CPA, and a CPA below `rpz` counts as a loss of separation (LoS).

---

## Why the env is split from the runner

| Concern | Owner |
|---|---|
| Spawn geometry, BlueSky settings, stepping | **env** |
| Noise / surveillance (CNS) | runner |
| CD / CR / CRR algorithm choice | runner |
| Metrics aggregation, Monte Carlo, plotting | runner |

Keeping the env free of the algorithmic pipeline means the same scenario can be
driven by any detection/resolution/recovery combination — or a learned policy —
without touching scenario setup.
