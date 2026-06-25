# CR — Conflict Resolution

**Package:** `cr/`  
**Entry points:**  
- `cr.mvp.resolve(conf, ownship, intruder, cfg) -> (newtrack, newgs, newvs, alt)`  
- `cr.vo.resolve(conf, ownship, intruder, cfg) -> (newtrack, newgs, newvs, alt)`

Both resolvers find a new velocity for the ownship that will — if maintained — avoid the conflict. They operate on the per-pair geometry produced by `cd.detect`.

Resolution is **horizontal-only**: only track and ground speed change; vertical speed is untouched.

---

## Shared building blocks (`cr/common.py`)

### `ResolutionConfig`

An immutable bag of resolution tuning parameters:

```python
@dataclass(frozen=True)
class ResolutionConfig:
    resofach: float = 1.0   # horizontal margin factor
```

`resofach > 1` inflates the effective protected-zone radius used in resolution, adding safety margin beyond the detection radius.

### `horizontal_command`

Converts the resolved cartesian velocity `newv = (v_E, v_N)` to autopilot commands. Track is the angle of the velocity vector; ground speed is its magnitude:

$$\text{track} = \text{atan2}(v_E, v_N) \pmod{360}, \qquad gs = \sqrt{v_E^2 + v_N^2}$$

```python
newtrack = (np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi) % 360
newgs    = np.sqrt(newv[0, :]**2 + newv[1, :]**2)
```

### `cap_velocities`

Clamps ground speed and vertical speed to the aircraft performance envelope `[vmin, vmax]` and `[vsmin, vsmax]`.

---

## 1. Modified Voltage Potential (MVP) — `cr/mvp.py`

### Intuition

MVP treats the conflict as a repulsive force field: the ownship is pushed away from the intruder's predicted position at CPA by a velocity increment proportional to how much deeper inside the protected zone the CPA would be.

### Per-pair resolution (`mvp_pair`)

Given conflict pair `(i, j)` with `t_CPA`, current relative position Δr, and relative velocity Δv = v_j − v_i:

**Predicted CPA vector (from ownship to intruder at CPA):**

$$\mathbf{d}_\text{CPA} = \Delta\mathbf{r} + \Delta\mathbf{v}\, t_\text{CPA}, \qquad d_\text{CPA} = |\mathbf{d}_\text{CPA}|$$

**Penetration depth into the protected zone:**

$$\iota_H = R_\text{PZ} - d_\text{CPA}$$

**Resolution velocity (normal case — ownship outside the zone, R_PZ < dist and d_CPA < dist):**

An erratum correction prevents over-correction when the CPA is offset from the zone centre:

$$\epsilon = \cos\!\left(\arcsin\!\frac{R_\text{PZ}}{\text{dist}} - \arcsin\!\frac{d_\text{CPA}}{\text{dist}}\right)$$

$$\Delta v_E = \frac{(R_\text{PZ}/\epsilon - d_\text{CPA})\, d_\text{CPA,E}}{|t_\text{CPA}|\, d_\text{CPA}}, \qquad \Delta v_N = \frac{(R_\text{PZ}/\epsilon - d_\text{CPA})\, d_\text{CPA,N}}{|t_\text{CPA}|\, d_\text{CPA}}$$

**Simplified case (ownship already inside the zone):**

$$\Delta v_E = \frac{\iota_H\, d_\text{CPA,E}}{|t_\text{CPA}|\, d_\text{CPA}}, \qquad \Delta v_N = \frac{\iota_H\, d_\text{CPA,N}}{|t_\text{CPA}|\, d_\text{CPA}}$$

Source: [`cr/mvp.py:20`](../cr/mvp.py).

### Multiple conflicts (`resolve`)

When ownship `i` is in conflict with several intruders simultaneously, the per-pair resolution velocities are **vector-summed** into a single Δv_i, then added to the current velocity:

$$\Delta\mathbf{v}_i = \sum_{j \in \text{conflicts}(i)} \Delta\mathbf{v}_{ij}$$

```python
dv[idx1] = dv[idx1] - dv_mvp   # sign: dv_mvp pushes ownship away
...
newv = v + np.transpose(dv)
```

Source: [`cr/mvp.py:61`](../cr/mvp.py).

---

## 2. Velocity Obstacle (VO) — `cr/vo.py`

### Intuition

The **velocity obstacle** (VO) of pair `(i, j)` is the set of all absolute velocities v_i that, if maintained, will lead to a collision (i.e. bring the pair closer than `rpz` at some future time). The VO is a cone in velocity space whose apex is at the intruder's velocity and whose half-angle depends on the protected-zone radius and current separation.

Resolution means finding the closest velocity to the current v_i that lies **outside** the VO cone.

### Collision cone geometry (`tangent_points`)

In the ownship's reference frame, the intruder is at relative position Δr, and the collision zone is a circle of radius `rpz` around it. The tangent points from the origin to that circle define the cone edges:

- d = |Δr|, θ = atan2(Δr_y, Δr_x)
- β = arcsin(R_PZ / d),  side = √(d² − R_PZ²)
- T₁ = side · (cos(θ − β), sin(θ − β))
- T₂ = side · (cos(θ + β), sin(θ + β))

Returns `(None, None)` when `d ≤ R_PZ` (already inside the zone — cone is undefined).

Source: [`cr/vo.py:26`](../cr/vo.py).

### VO in velocity space (`vo_pair`)

The VO cone is translated by the intruder's velocity v_j, shifting the apex and both edges into absolute velocity space:

```
VO apex  = v_j  (in velocity space)
VO edge 1 = ray from v_j through T1
VO edge 2 = ray from v_j through T2
```

Using `shapely`:

```python
vo_0 = translate(origin, xoff=intruder_velocity.x, yoff=intruder_velocity.y)
vo_1 = translate(tp_1,   xoff=intruder_velocity.x, yoff=intruder_velocity.y)
vo_2 = translate(tp_2,   xoff=intruder_velocity.x, yoff=intruder_velocity.y)
```

**Resolution (method 0 — optimal):** project the ownship's current velocity onto both cone edges; take the closer projection:

```python
cp_1 = nearest_points(vo_line_1, ownship_velocity)[0]
cp_2 = nearest_points(vo_line_2, ownship_velocity)[0]
cp   = cp_1 if cp_1.distance(ownship_velocity) <= cp_2.distance(ownship_velocity) else cp_2
```

The resolution delta is the vector from `cp` back to the current velocity (Δv_E = v_E − cp_E, Δv_N = v_N − cp_N).

Source: [`cr/vo.py:52`](../cr/vo.py).

### Multi-conflict note

Like MVP, VO sums per-pair deltas when there are simultaneous conflicts. This is **only validated for isolated two-aircraft conflicts**. For genuine multi-conflict scenarios the correct approach is to resolve against the union of all VOs (pick a velocity outside all cones simultaneously), not to sum per-pair changes — see the disclaimer comment in [`cr/vo.py:126`](../cr/vo.py).

---

## 3. Choosing between MVP and VO

| | MVP | VO |
|---|---|---|
| **Geometry** | Force-field analogy; adjusts velocity proportional to penetration depth | Cone-based; projects velocity to the nearest safe edge |
| **Output** | Velocity increment sized to just exit the zone at CPA | Minimum velocity change to escape the obstacle cone |
| **Multi-conflict** | Vector sum of per-pair forces (reasonable) | Vector sum of per-pair escapes (approximate) |
| **Dependency** | Pure numpy | Requires `shapely` |
| **Known limitation** | Erratum correction may be approximate near the zone boundary | Multi-VO superposition not implemented |

---

## 4. Usage

```python
from cr import mvp, vo, ResolutionConfig

cfg = ResolutionConfig(resofach=1.1)   # 10% safety margin

# MVP
newtrk, newgs, newvs, alt = mvp.resolve(conf, ownship, intruder, cfg)

# VO
newtrk, newgs, newvs, alt = vo.resolve(conf, ownship, intruder, cfg)
```

The returned values are per-aircraft arrays of length N; pass them to the autopilot command interface.
