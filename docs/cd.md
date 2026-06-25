# CD — Conflict Detection

**Package:** `cd/`  
**Entry point:** `cd.detect(ownship, intruder, rpz, hpz, dtlookahead) -> ConflictState`

A conflict exists when two aircraft will simultaneously violate both the **horizontal** protected zone (radius `rpz`) and the **vertical** protected zone (half-height `hpz`) within the look-ahead window `dtlookahead`.

The detection is a state-based linear prediction: both aircraft are assumed to fly at their current velocity until the look-ahead horizon.

---

## 1. Geometry overview

For a pair `(i, j)`:

1. Compute the current relative position (bearing `qdr`, distance `dist`).
2. Compute the **time to closest point of approach** (TCPA) and the **miss distance at CPA** (DCPA).
3. Find the **time window** `[t_in, t_out]` during which the pair is inside the horizontal zone.
4. Do the same vertically.
5. Intersect the two windows and check that the overlap falls within `[0, dtlookahead]`.

This is done for **all N² pairs simultaneously** using numpy matrix operations.

---

## 2. Relative position (`relative_bearing_distance`)

Bearing `qdr` and distance `dist` (in metres) are computed for every `(i, j)` pair using BlueSky's `geo.kwikqdrdist_matrix`. The diagonal (ownship vs. itself) is pushed to a large value `_BIG = 1e9` so self-pairs can never be flagged:

```python
dist = np.asarray(dist) * nm + _BIG * eye   # eye = np.eye(ntraf)
```

Source: [`cd/statebased.py:21`](../cd/statebased.py).

---

## 3. Horizontal conflict (`horizontal_conflict`)

### Relative velocity

```python
du = ownu - intu.T     # east component difference,  shape (N, N)
dv = ownv - intv.T     # north component difference, shape (N, N)
dv2 = du² + dv²        # |Δv|²
```

where `ownu = gs_i · sin(trk_i)` and `ownv = gs_i · cos(trk_i)`.

### Time to CPA

The CPA occurs when d/dt |r(t)|² = 0, where r(t) = r₀ + Δv · t. Solving gives:

$$t_\text{CPA} = -\frac{\Delta\mathbf{r} \cdot \Delta\mathbf{v}}{|\Delta\mathbf{v}|^2}$$

```python
tcpa = -(du * dx + dv * dy) / dv2 + _BIG * eye
```

### Miss distance at CPA

$$d_\text{CPA}^2 = |\Delta\mathbf{r}|^2 - t_\text{CPA}^2 |\Delta\mathbf{v}|^2$$

```python
dcpa2 = np.abs(dist**2 - tcpa**2 * dv2)
```

### Horizontal conflict flag and entry/exit times

A pair is horizontally conflicting when `d_CPA < R_PZ`. The half-chord length inside the zone is `d_x = √max(0, R_PZ² - d_CPA²)`, giving entry and exit times:

$$t_\text{in} = t_\text{CPA} - \frac{d_x}{|\Delta\mathbf{v}|}, \qquad t_\text{out} = t_\text{CPA} + \frac{d_x}{|\Delta\mathbf{v}|}$$

Source: [`cd/statebased.py:44`](../cd/statebased.py).

---

## 4. Vertical conflict (`vertical_conflict`)

Vertical separation at the current instant: Δh = h_i − h_j (with `_BIG` added on the diagonal).

Relative vertical speed Δḣ = ḣ_i − ḣ_j (with a small floor to avoid division by zero). Time to cross the upper/lower boundary of the vertical protected zone:

$$t_{\text{cross},\text{hi}} = \frac{\Delta h + H_\text{PZ}}{-\Delta\dot{h}}, \qquad t_{\text{cross},\text{lo}} = \frac{\Delta h - H_\text{PZ}}{-\Delta\dot{h}}$$

$$t_\text{in,v} = \min(t_{\text{cross},\text{hi}},\, t_{\text{cross},\text{lo}}), \qquad t_\text{out,v} = \max(\ldots)$$

Source: [`cd/statebased.py:82`](../cd/statebased.py).

---

## 5. Combining into a conflict (`combine_conflicts`)

A conflict exists for pair `(i, j)` when the horizontal and vertical windows overlap and that overlap occurs before the look-ahead horizon:

$$t_\text{in,conf} = \max(t_\text{in,h},\, t_\text{in,v}), \qquad t_\text{out,conf} = \min(t_\text{out,h},\, t_\text{out,v})$$

All four conditions must hold simultaneously:

| Condition | Expression |
|---|---|
| Horizontal miss inside zone | `d_CPA < R_PZ` |
| Windows overlap | `t_in,conf ≤ t_out,conf` |
| Conflict not purely in past | `t_out,conf > 0` |
| Conflict within look-ahead | `t_in,conf < dtlookahead` |

```python
swconfl = (swhorconf
           * (tinconf <= toutconf)
           * (toutconf > 0.0)
           * (tinconf < dtlookahead)
           * (1.0 - eye))
```

Source: [`cd/statebased.py:109`](../cd/statebased.py).

---

## 6. Result: `ConflictState`

`detect` packs all outputs into a frozen dataclass ([`cd/common.py`](../cd/common.py)):

| Field | Meaning |
|---|---|
| `confpairs` | Ordered `(ownship_id, intruder_id)` tuples for every conflicting pair |
| `confpairs_unique` | `frozenset` of unordered pairs (collapses `(A,B)` and `(B,A)`) |
| `lospairs` | Pairs currently in loss of separation (inside the zone right now) |
| `inconf` | Length-N boolean: is aircraft `i` in any conflict? |
| `tcpamax` | Per-aircraft maximum TCPA across all its conflicts |
| `qdr`, `dist` | Bearing and distance at detection time for each conflicting pair |
| `dcpa` | Miss distance at CPA for each conflicting pair |
| `tcpa` | Time to CPA for each conflicting pair |
| `tLOS` | Time to loss of separation (`t_in,conf`) for each pair |

Being `frozen=True`, `ConflictState` is a value — callers cannot accidentally mutate detection results.

---

## 7. Usage

```python
from cd import detect

conf = detect(ownship, intruder, rpz=200.0, hpz=50.0, dtlookahead=300.0)
# conf.inconf[i] is True if aircraft i is in conflict
# conf.confpairs lists every (ownship, intruder) ordered pair
```

With the CNS model, pass the degraded view:

```python
conf = detect(traffic, traffic, rpz, hpz, dtlookahead)
# ownship uses traffic.sensor (1D); intruder uses traffic.adsl row i
```
