# CRR — Conflict Recovery (Resume Navigation)

**Package:** `crr/`  
**Entry points:**  
- `crr.resumenav_cpa` — release once past the closest point of approach  
- `crr.resumenav_double_criteria` — deterministic free-to-revert (FTR)  
- `crr.resumenav_probabilistic_ftr` — probabilistic FTR with sensor-noise awareness

All three share the same interface and threading model. They decide **when a resolved conflict may be released** — i.e. when the ownship can switch its autopilot back from the resolution manoeuvre to its original route.

---

## Why recovery is non-trivial

After CR issues a resolution velocity, the conflict is "resolved" in the sense that the pair will no longer collide — but only if both aircraft maintain their new velocities. Releasing too early (before enough separation has built up) risks the pair re-entering conflict immediately. Holding the resolution too long causes unnecessary route deviation.

The recovery models track which pairs have been resolved (`resopairs`) and test a clear-of-conflict condition each tick. When the condition is satisfied, the autopilot is commanded back to the original route.

---

## Shared building blocks (`crr/common.py`)

### `RecoveryState`

The immutable, explicitly-threaded book-keeping:

```python
@dataclass(frozen=True)
class RecoveryState:
    resopairs: frozenset       # pairs still under resolution
    init_vel: Mapping          # intruder velocity at conflict initiation (for FTR)
```

Updating state is done with `dataclasses.replace` or by building a new `RecoveryState`; the input is never mutated.

### Geometry helpers

**`calculate_dcpa(dx, dy, du, dv)`** — DCPA and TCPA for a single pair, using the standard CPA formula:

$$t_\text{CPA} = -\frac{\Delta\mathbf{r} \cdot \Delta\mathbf{v}}{|\Delta\mathbf{v}|^2}, \qquad d_\text{CPA} = \sqrt{|\Delta\mathbf{r}|^2 - t_\text{CPA}^2 |\Delta\mathbf{v}|^2}$$

**`get_desired_ownship_velocity(ownship, idx, cache)`** — reads the pre-resolution route velocity from the autopilot target (`seltrk`/`selspd` → `ap.trk`/`ap.tas` → fallback to current velocity). Cached per-tick to avoid redundant reads.

**`anglediff(a, b)`** — smallest signed difference between two headings, handling wrap-around.

### Injected side effects

The only impure step — writing the ASAS-active flag and commanding waypoint recovery — is isolated in `apply_active_changes` and reaches BlueSky only through injected callables:

```python
def resumenav_cpa(state, conf, ownship, intruder, active,
                  resofach, id2idx=default_id2idx, recover=default_recover):
```

Tests substitute lightweight fakes for `id2idx` and `recover` (see `tests/conftest.py`).

---

## 1. CPA criterion (`crr/cpa.py`)

### When to release

A resolved pair `(i, j)` is released when **all three conditions hold**:

| Condition | Meaning | Test |
|---|---|---|
| Past CPA | The pair is now separating | `dr · dv < 0` |
| No horizontal LoS | Currently outside the protected zone | `|dr| ≥ R_PZ` |
| Not bouncing | Tracks not nearly parallel near the zone | `|Δhdg| ≥ 30°` or `|dr| ≥ R_PZ · resofach` |

**Past CPA:** `dr · dv < 0` means the relative position vector and relative velocity vector point in opposite directions — the aircraft are now moving apart.

```python
def _past_cpa(dist, vrel):
    return bool(np.dot(dist, vrel) < 0.0)
```

**Bouncing** describes a failure mode where two aircraft fly nearly parallel tracks close together: they resolve, separate slightly, re-enter conflict, and cycle repeatedly. The check rejects release when track difference is small and they are still close.

Source: [`crr/cpa.py`](../crr/cpa.py).

---

## 2. Deterministic Free-To-Revert (FTR) — `crr/ftr.py`

### Motivation

The CPA criterion releases when the pair *has* separated. FTR is more conservative: it releases only when the pair *would be safe* if ownship were to immediately revert to its original route velocity.

### Two criteria

Let V_o be the ownship's **desired velocity** (pre-resolution), V_i,c the intruder's **current velocity**, and V_i,i the intruder's **velocity at conflict initiation**.

**Criterion 1** — safe assuming intruder maintains current velocity:

$$d_\text{CPA}(V_o,\, V_{i,c}) > R_\text{PZ}$$

**Criterion 2** — safe assuming intruder reverts to its initial velocity:

$$d_\text{CPA}(V_o,\, V_{i,i}) > R_\text{PZ}$$

Release when **both** hold. Holding both guards against two failure modes:
- Criterion 1 alone: if the intruder is currently deviating (resolving its own conflict), reverting could put you back in conflict.
- Criterion 2 alone: if the intruder is still deviating, the prediction using initial velocity would be too pessimistic.

### Intruder desired velocity approximation

The intruder's desired velocity V_i,i is **not communicated via ADS-L**. ADS-L (and ADS-B) broadcast only the instantaneous observed velocity, not flight-plan intent. FTR approximates V_i,i as the intruder's observed velocity at the moment conflict was first detected, recorded in `RecoveryState.init_vel`:

```python
state, _ = record_initial_intruder_velocity(state, conf, intruder, id2idx)
...
Vi_i_u, Vi_i_v = init_vel.get(conflict, (Vi_c_u, Vi_c_v))
```

This is the **weakest assumption** in the model: if the intruder deviates from that initial velocity for reasons unrelated to the conflict, criterion 2 may be over- or under-conservative. Source: [`crr/ftr.py:1–24`](../crr/ftr.py).

---

## 3. Probabilistic FTR — `crr/probabilistic_ftr.py`

### Motivation

The deterministic FTR uses point estimates for position and velocity. When the CNS model is active, position and velocity have known uncertainty (from `pos_acc` and `vel_acc` in the ADS-L observation). The probabilistic FTR replaces the hard threshold `d_CPA > R_PZ` with a probability exceeding a threshold τ (default 0.9):

$$\text{Criterion 1:}\quad P\!\left(d_\text{CPA} > R_\text{PZ} \mid V_o,\, V_{i,c},\, \Sigma_r,\, \Sigma_v\right) > \tau$$

$$\text{Criterion 2:}\quad P\!\left(d_\text{CPA} > R_\text{PZ} \mid V_o,\, V_{i,i},\, \Sigma_r,\, \Sigma_v\right) > \tau$$

### Relative covariance

Position and velocity uncertainty are read from `ownship.adsl.pos_acc` and `intruder.adsl.pos_acc` (the CI95 radius in metres). Converting to σ via the 2D isotropic relation σ = R₉₅ / 2.448, and noting that ownship and intruder errors are independent, the relative covariances add:

$$\Sigma_r = \sigma_{r,i}^2 I + \sigma_{r,j}^2 I, \qquad \Sigma_v = \sigma_{v,i}^2 I + \sigma_{v,j}^2 I$$

Source: [`crr/probabilistic_ftr.py:307`](../crr/probabilistic_ftr.py).

### Computing P(d_CPA > x) (`analytical_dcpa_prob_gt`)

The relative position r ~ N(μ_r, Σ_r) and relative velocity v ~ N(μ_v, Σ_v) are independent 2D Gaussians. The DCPA equals the component of r perpendicular to v:

$$d_\text{CPA}(\mathbf{r}, \mathbf{v}) = \left\|\mathbf{r} - \mathbf{v}\,\frac{\mathbf{r}\cdot\mathbf{v}}{|\mathbf{v}|^2}\right\| = |\mathbf{r}^\perp|$$

The approach is to **marginalise over the direction θ of v** — for each angle θ, the perpendicular component of r is a 1D projection with a normal distribution. Conditioned on θ, the DCPA follows a folded-normal:

$$P(|r^\perp(\theta)| > x) = 1 - \left[\Phi\!\left(\frac{x - m(\theta)}{s(\theta)}\right) - \Phi\!\left(\frac{-x - m(\theta)}{s(\theta)}\right)\right]$$

where m(θ) and s(θ) are the mean and standard deviation of the projection at angle θ. The total probability integrates over θ, weighted by the **projected-normal density** w(θ) of the direction of v:

$$P(d_\text{CPA} > x) \approx \sum_{k=1}^{K} w(\theta_k)\, P(|r^\perp(\theta_k)| > x)$$

### Projected-normal density (`log_p_theta_projected_normal`)

For V ~ N(μ, Σ) in R², the density of its direction Θ = atan2(V_y, V_x) has a closed form involving the standard normal CDF Φ. It is computed in **log-space** to avoid overflow/underflow when the velocity SNR |μ|/σ is large (peaked distribution around a single direction):

```python
def log_p_theta_projected_normal(theta, mu, Sigma): ...
```

Source: [`crr/probabilistic_ftr.py:91`](../crr/probabilistic_ftr.py).

### Parameters

| Parameter | Default | Effect |
|---|---|---|
| `prob_threshold` | 0.9 | Higher = more conservative release |
| `Ktheta` | 256 | More angle samples = more accurate integration; negligible cost |

---

## 4. Choosing a recovery model

| Model | When to use |
|---|---|
| `resumenav_cpa` | Simplest; no route intent needed; sufficient for basic scenarios |
| `resumenav_double_criteria` | Prevents premature release when intruder may revert; no uncertainty model needed |
| `resumenav_probabilistic_ftr` | Use when CNS model is active and sensor noise is non-negligible; accounts for position/velocity uncertainty from ADS-L |

---

## 5. Usage

```python
from crr import empty_recovery_state, resumenav_cpa, resumenav_double_criteria, resumenav_probabilistic_ftr

state = empty_recovery_state()

# --- each tick ---

# Option A: CPA
state, released = resumenav_cpa(
    state, conf, ownship, intruder, active, resofach=1.0)

# Option B: Deterministic FTR
state, released = resumenav_double_criteria(
    state, conf, ownship, intruder, active)

# Option C: Probabilistic FTR (requires traffic.adsl with pos_acc / vel_acc)
state, released = resumenav_probabilistic_ftr(
    state, conf, ownship, intruder, active,
    prob_threshold=0.9, Ktheta=256)
```

`released` is the set of pairs that were cleared this tick. `state` must be threaded forward — pass the returned value into the next tick's call.
