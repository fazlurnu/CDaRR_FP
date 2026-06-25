# CNS — Communication, Navigation & Surveillance Model

**Package:** `sim_models/cns/`

BlueSky provides ground truth. Real aircraft never see ground truth — they see a noisy, delayed, sometimes-missing version of it. The CNS model sits between truth and the CD/CR/CRR algorithms and produces that degraded view.

## Two-layer model

```
sensor[j]   = truth[j] + ε_j          # 1D, length N. ε re-drawn every tick.
obs[i][j]   = last-received sensor[j]  # N×N. Gated by reception P[i][j]. No extra noise.
```

- `j` is the **target** aircraft (whose state is being measured or received).
- `i` is the **observer** aircraft (whose picture of the world we are describing).
- `obs[i][j]` is "what aircraft `i` currently knows about aircraft `j`."
- The **diagonal** `obs[i][i] == sensor[i]` always: ownship always knows its own state.

Noise is applied **once**, at the sensor. The ADS-L layer adds only reception (stale vs. fresh); it never re-noises.

---

## 1. Noise distributions (`distributions.py`)

A *distribution* is any callable with signature:

```
(n: int, ci95, rng: np.random.Generator) -> np.ndarray of shape (n, 2)
```

It returns `n` two-dimensional errors in native field units (metres for position, m/s for velocity). The two components are the east and north errors.

### 95% CI to sigma conversion

Accuracy is specified as a **95% confidence interval** (CI95). For a 2D isotropic Gaussian the 95% probability circle has radius R₉₅ = σ · √(−2 ln 0.05) ≈ 2.448 σ, so:

$$\sigma = \frac{\text{CI95}}{2.448}$$

Implemented in [`ci95_to_std`](../sim_models/cns/distributions.py):

```python
CI95_TO_STD_2D = 2.448

def ci95_to_std(ci95, n) -> np.ndarray:   # returns shape (n, 1)
    std = np.asarray(ci95, dtype=float) / CI95_TO_STD_2D
    ...
    return std.reshape(n, 1)  # broadcasts over (n, 2) draw
```

### Gaussian

Zero-mean isotropic draw — noise ε ~ N(0, σ²I), with σ = CI95 / 2.448:

$$\varepsilon \sim \mathcal{N}\!\left(\mathbf{0},\, \sigma^2 I_2\right)$$

```python
def gaussian(n, ci95, rng) -> np.ndarray:
    z = rng.standard_normal((n, 2))
    return z * ci95_to_std(ci95, n)
```

### Biased Gaussian

A systematic measurement bias `(b_east, b_north)` in native units — a property of the distribution, not time-varying. Noise ε ~ N(b, σ²I):

$$\varepsilon \sim \mathcal{N}\!\left(\mathbf{b},\, \sigma^2 I_2\right)$$

Implemented as a **higher-order factory** (rather than a subclass) so any callable is a valid distribution:

```python
def make_biased_gaussian(bias=(0.0, 0.0)):
    bias = np.asarray(bias).reshape(1, 2)
    def biased_gaussian(n, ci95, rng):
        return gaussian(n, ci95, rng) + bias
    return biased_gaussian
```

### Accuracy is per-tick and per-aircraft

`ci95` may be a scalar (same noise for all aircraft) or a length-N array (different noise per aircraft). It is read at draw time, so GPS degradation or improved accuracy can be simulated simply by changing `cns.pos_ci95` between ticks via `dataclasses.replace`.

---

## 2. Sensor layer (`sensor.py`)

Each tick, `measure` produces a fresh noisy snapshot of every aircraft's state from BlueSky truth. It is a **pure function**: it takes a truth snapshot and returns a new [`SensorState`](../sim_models/cns/sensor.py); it never mutates an existing state.

### Position noise

Error is drawn in metres (east δ_E, north δ_N) then converted to lat/lon degrees using a flat-earth approximation. The `cos(lat)` term accounts for longitude degree widths shrinking toward the poles:

$$\text{lat}' = \text{lat} + \frac{\delta_N}{111320}, \qquad \text{lon}' = \text{lon} + \frac{\delta_E}{111320 \cdot \cos(\text{lat})}$$

Implemented in `_apply_position_noise`:

```python
coslat = np.maximum(np.cos(np.deg2rad(lat)), 1e-6)   # pole guard
lat_out = lat + north_m / 111_320.0
lon_out = lon + east_m / (111_320.0 * coslat)
```

### Velocity noise

Error (δ_vN, δ_vE) in m/s is added to the north/east ground-speed components derived from `gs` (ground speed) and `trk` (track angle):

$$v_N = gs \cdot \cos(\text{trk}) + \delta_{v_N}, \qquad v_E = gs \cdot \sin(\text{trk}) + \delta_{v_E}$$

Implemented in `_velocity_components`.

### Pass-through fields

`alt`, `hdg`, `trk`, `gs`, `tas`, `vs` are copied verbatim (no noise). `pos_acc` and `vel_acc` record the CI95 actually used (scalar broadcasts to per-aircraft array) so downstream algorithms can read the advertised accuracy.

### Resize is implicit

Because `measure` always rebuilds the `SensorState` from `states.ntraf`, there is no explicit resize step — adding or removing aircraft is handled automatically on the next call.

---

## 3. Reception model (`reception_model.py`)

`P[i, j]` is the probability that observer `i` receives target `j`'s ADS-L message this tick. Asymmetry is allowed: `P[i,j] ≠ P[j,i]`.

### Building P

[`make_reception(default_prob)`](../sim_models/cns/reception_model.py) creates a `ReceptionModel` with an empty `(0,0)` P matrix. [`ensure_size(rm, n)`](../sim_models/cns/reception_model.py) (re)builds it to `(n, n)`:

$$P_{ij} = \begin{cases} 1.0 & i = j \\ \text{default\_prob} & i \neq j \end{cases}$$

Individual pairs are overridden with `set_pair(rm, i, j, prob)`, which copies P before writing (so the original `ReceptionModel` is unmodified).

### Sampling the refresh mask

Each tick, `sample_mask(rm, n, rng)` draws a boolean `(n, n)` mask: cell `[i,j]` is True when a uniform draw U ≤ P[i,j]:

$$M_{ij} = \mathbf{1}\!\left[U_{ij} \leq P_{ij}\right], \quad U_{ij} \sim \text{Uniform}[0, 1]$$

The diagonal is forced True. `True` at `[i,j]` means observer `i` receives target `j` this tick.

`sample_mask` returns `(mask, updated_rm)` — the updated `ReceptionModel` is threaded back explicitly because `ensure_size` may have grown `P`.

**First tick:** `force_full=True` (called by `step` when `first_update_done=False`) sets the entire mask to True so all cells are seeded without packet loss. No NaN cells can arise.

### Geometry-based P (`p_from_range`)

[`p_from_range(states, max_range, default_prob=1.0)`](../sim_models/cns/reception_model.py) builds P from pairwise flat-earth distances. For a pair (i, j):

$$P_{ij} = \begin{cases} 1.0 & i = j \\ \text{default\_prob} & d(i,j) \leq \text{max\_range} \\ 0 & d(i,j) > \text{max\_range} \end{cases}$$

where the flat-earth distance d(i, j) in metres is:

$$d(i,j) = \sqrt{(\Delta\text{lat} \cdot 111320)^2 + (\Delta\text{lon} \cdot 111320 \cdot \cos\bar{\phi})^2}$$

This is a v1 step function. The TODO comment on the `np.where` line marks where a continuous model (logistic decay, Friis falloff) would be substituted. Usage — call every tick before `step`:

```python
rm  = p_from_range(states, max_range=1000.0)
cns = replace(cns, reception=rm)
cns = step(cns, states)
```

---

## 4. ADS-L observation layer (`adsl_observation.py`)

`ADSLObservation` holds one `(N, N)` matrix per field. `obs.fields["lat"][i, j]` is the last value of target `j`'s latitude that observer `i` received.

### Update rule

Each tick, [`update(obs, sensor, mask)`](../sim_models/cns/adsl_observation.py) writes sensor[j] into row i where `mask[i,j]` is True, otherwise keeps the previous (stale) value:

$$\text{obs}[i,j]^{(t)} = \begin{cases} \text{sensor}[j]^{(t)} & M_{ij} = \text{True} \\ \text{obs}[i,j]^{(t-1)} & M_{ij} = \text{False} \end{cases}$$

`sensor[j]` is a 1D length-N array; reshaping to `(1, N)` broadcasts target `j`'s value across all observer rows `i` in a single `np.where` call:

```python
sensor_row = np.asarray(getattr(sensor, f)).reshape(1, -1)  # (1, N)
new_fields[f] = np.where(mask, sensor_row, obs.fields[f])
```

Two observers that both receive target `j` hold **exactly the same value** — no per-observer noise is added in the observation layer.

A stale `obs[i,j]` is therefore "old jitter on old truth": wrong both because the target moved and because the old draw was random.

---

## 5. CNS coordinator (`cns.py`)

[`CNSState`](../sim_models/cns/cns.py) binds the sensor, reception model, and observation into one immutable object. [`step(cns, states)`](../sim_models/cns/cns.py) is the single per-tick function:

```python
def step(cns, states) -> CNSState:
    sensor = measure(states, cns.pos_ci95, cns.vel_ci95,
                     cns.pos_dist, cns.vel_dist, cns.rng)
    mask, rm = sample_mask(cns.reception, n, cns.rng,
                           force_full=not cns.first_update_done)
    obs = update(cns.obs, sensor, mask)
    return replace(cns, sensor=sensor, reception=rm, obs=obs,
                   first_update_done=True)
```

### Accessors for CD/CR/CRR

```python
ownship_field(cns, "lat")   # (N,)  — sensor[i]: ownship i's own lat
adsl_field(cns, "lat")      # (N,N) — obs[i,j]: what i last received of j's lat
```

Ownship `i` uses its **own row** of the N×N picture:
- `ownship_field(cns, "lat")[i]` → ownship i's own position (sensor).
- `adsl_field(cns, "lat")[i, j]` → what ownship i last received about intruder j.

### Changing parameters between ticks

```python
from dataclasses import replace
cns = replace(cns, pos_ci95=new_accuracy)    # change noise level
cns = replace(cns, reception=new_rm)         # change P matrix (e.g. after p_from_range)
```

---

## Summary of invariants

| # | Invariant |
|---|---|
| 1 | First `step` seeds all cells (no NaN). |
| 2 | ε is a fresh draw each tick; CI95 is read at draw time. |
| 3 | `P[i,j] ≠ P[j,i]` permitted; diagonal = 1.0. |
| 4 | Structures resize on aircraft create/delete, preserving existing data. |
| 5 | `obs[i,i] == sensor[i]` every tick. |
| 6 | No noise is added at the observation layer — received value is exactly `sensor[j]`. |
