# ADS-B Latency and the Along-Track Position Bias

## Background

ADS-B position reports carry a systematic error that is absent from the isotropic
Gaussian noise model used elsewhere in the framework. The error has a clear physical
cause: the aircraft's onboard GNSS sensor determines its position at one instant, but
that position is not broadcast until some time later. By the time the message is
received, the aircraft has moved on. The reported position therefore lags behind the
true position — always in the direction the aircraft was travelling, never sideways.

This lag is called **ADS-B latency** and it is the primary source of systematic
along-track error in ADS-B surveillance data.

### Evidence from the literature

Schäfer & Jonáš (2025) quantify this effect using a high-resolution MLAT system
(>170 ground receivers) as an independent position reference. Their key findings for
ADS-B version 2 transponders:

- **Cross-track deviations** follow a zero-mean Gaussian with σ ≈ 3.24 m (95th
  percentile ≈ 6.17 m). This is attributable to residual MLAT noise and is symmetric.
- **Along-track deviations** have a non-zero mean of approximately **−14.93 m**
  (negative = reported position is behind the true position). The distribution is
  non-Gaussian, reflecting a mixture of latency values across aircraft.
- The **mean ADS-B v2 latency** across all aircraft in the dataset is **66.1 ms**
  (median 73.7 ms, standard deviation 76.9 ms).
- At a typical cruising speed of 840 km/h (233 m/s):
  `0.0661 s × 233 m/s ≈ 15.4 m` — consistent with the observed −14.93 m.

> **Reference:** M. Schäfer and P. Jonáš, "ADS-B Positional Accuracy and Anomalies:
> A Comprehensive Analysis Using High-Resolution MLAT Data," *2025 Integrated
> Communications, Navigation and Surveillance Conference (ICNS)*, 2025.
> DOI: 10.1109/ICNS65417.2025.10976935

---

## Why Along-Track Has More Variance Than Cross-Track

### The directional nature of latency error

Latency is a **time** error. The aircraft moves forward in the direction it is
travelling during the latency window — it does not drift sideways. So a delay of
66.1 ms at 840 km/h produces a ~15 m lag in the direction of travel and **zero**
displacement perpendicular to it. The along-track axis accumulates all of the
latency-driven error; the cross-track axis sees none of it.

Cross-track error comes from a separate, independent source: **GNSS positioning
noise** (multipath, atmospheric refraction, satellite geometry). This is roughly
isotropic around the true position and has no connection to latency. The paper
confirms this — the cross-track distribution is zero-mean and well-modelled by a
Gaussian with σ ≈ 3.24 m, attributable almost entirely to residual MLAT measurement
noise.

### Sources of along-track variance

The along-track distribution has larger variance than cross-track for three reasons,
all rooted in the latency being inconsistent across broadcasts and aircraft:

**1. Unsynchronised GNSS and ADS-B subsystems.** The GNSS sensor updates at fixed
intervals, while the ADS-B transmitter broadcasts at random intervals (between 0.4 s
and 0.6 s). Because they operate independently, the gap between position determination
and transmission varies randomly every cycle. The mean gap is 66.1 ms but it is
different for every broadcast, producing a spread of along-track errors around that
mean.

**2. Mixture of aircraft and transponder types.** The paper identifies two latency
clusters across the fleet — a low-variance cluster and a high-variance cluster —
associated with different avionics implementations across manufacturers and models.
Pooling all aircraft produces a mixture distribution in along-track that is broader
than any individual cluster, and is why the paper describes the along-track distribution
as non-Gaussian.

**3. Redundant transponders.** Some aircraft alternate between two onboard ADS-B
subsystems with different latency characteristics. Each subsystem produces its own
tight distribution, but flight-to-flight alternation creates a bimodal spread for those
aircraft (Fig. 4 in the paper — the A319 example).

### Asymmetry in error sources

| Source | Cross-track | Along-track |
|---|---|---|
| GNSS positioning noise | Yes (σ ≈ 3.24 m) | Yes (same) |
| Mean latency × speed | No | Yes — the systematic bias |
| Latency inconsistency across broadcasts | No | Yes — drives the spread |
| Mixture of aircraft/transponder types | No | Yes — inflates variance further |

Cross-track variance is small because only GNSS noise contributes. Along-track
variance is larger because it accumulates GNSS noise on top of all latency-driven
spread, and the latency effect dominates at cruising speed.

---

## Containment Radius Violations and the Role of Latency

### What NIC guarantees — and what it does not

ADS-B position reports include a **Navigation Integrity Category (NIC)** value, which
defines a **horizontal containment radius (HCR)**: the radius within which the true
aircraft position is expected to lie with high probability, as determined by the
onboard GNSS sensor. NIC=10 corresponds to an HCR of 25 metres.

A critical limitation, stated explicitly in the paper:

> *"ADS-B latency is not accounted for in the horizontal containment radius, as the
> NIC value is determined by the GNSS sensor, whereas mitigating ADS-B latency is the
> responsibility of the ADS-B transmitting subsystem."*

The GNSS sensor may report high integrity (high NIC) because it has located itself
accurately. But the reported position is transmitted later — and the displacement
accumulated during that latency window is **not included in the NIC bound**. The
containment radius only covers GNSS positioning error; it is blind to latency error.

### Observed violation: NIC=10 case study

The paper presents a business jet that reported NIC=10 for approximately 5–6 minutes.
During this period:

- **Cross-track deviation**: remained below 10 m — well within the 25 m HCR.
- **Along-track deviation**: fluctuated over a **70 m range** — far exceeding the 25 m HCR.

The cross-track axis is within bounds because GNSS positioning error is small and
latency has no cross-track component. The along-track axis violates the containment
radius because latency accumulates directly in the direction of travel, and that
error is invisible to the NIC/HCR system.

This is not an anomaly — it is the expected consequence of the structural gap between
what NIC measures (GNSS accuracy) and what NIC does not measure (transmission latency).
Any aircraft flying at speed with nonzero latency can in principle violate its stated
along-track containment radius, regardless of how high its reported NIC value is.

### A separate mechanism: dual transponders

A second case in the paper (a large military aircraft observed on two flights three
hours apart) shows a different pattern: the standard deviation of both along- and
cross-track deviations was within the containment radius, but a **nonzero constant
offset** appeared on one flight that was absent on the other. The paper attributes
this to the aircraft likely alternating between two onboard ADS-B subsystems with
different antenna or calibration offsets.

This is distinct from latency — it is a fixed positional offset rather than a
speed-dependent lag, and it affects both axes. It illustrates that along-track
violations can have more than one root cause; latency is the dominant and systematic
one, but hardware-level offsets are an additional source.

---

## The Error Model

### Coordinate frames

The standard position noise in this framework is drawn in the **fixed world frame**:
`exy = (east_m, north_m)`. Applying a fixed bias in that frame (as
`make_biased_gaussian` does) would point in the same compass direction regardless
of where the aircraft is flying, which is physically wrong for a latency effect.

The along-track bias lives in the **track-relative frame**:

```
along-track direction:  (east=sin(trk), north=cos(trk))   — direction of travel
cross-track direction:  (east=-cos(trk), north=sin(trk))  — 90° left of travel
```

where `trk` is the track angle in degrees, measured clockwise from geographic North.

### Magnitude

The along-track bias magnitude for a single aircraft is:

```
b_at = −latency_s × gs        [metres]
```

- `latency_s` — ADS-B position reporting latency in seconds (a property of the
  transponder/avionics system, not of the aircraft geometry).
- `gs` — ground speed in m/s at the moment of transmission.
- The negative sign reflects that the reported position *lags behind* the true
  position.

### Rotation into the world frame

Given `b_at` (along-track) and `b_ct` (cross-track, ≈ 0), the bias in (east, north)
metres is:

```
east_bias  = b_at · sin(trk)  −  b_ct · cos(trk)
north_bias = b_at · cos(trk)  +  b_ct · sin(trk)
```

This is a standard 2D rotation of the vector `(b_at, b_ct)` by angle `trk`. The
result is added on top of the zero-mean noise draw before converting to lat/lon.

---

## Speed scaling

Because the bias is `−latency × gs`, it is not a fixed offset — it scales with
ground speed. The table below shows what the same ADS-B v2 latency (66.1 ms) produces
at different speeds:

| Speed           | gs (m/s) | Along-track bias |
|-----------------|----------|-----------------|
| 840 km/h        | 233      | −15.4 m         |
| 100 kts         | 51.4     | −3.4 m          |
| 20 kts (sim)    | 10.3     | **−0.68 m**     |

At the simulation speed of ~20 kts, the bias is sub-metre and small relative to
the cross-track noise (σ ≈ 3.24 m), but is included for physical correctness.

---

## Implementation

### `latency_s` vs. a fixed `along_track_bias_m`

An earlier design stored a pre-computed `along_track_bias_m` constant. This was
replaced by `latency_s` because:

1. Latency is the actual system invariant; the bias in metres is derived from it.
2. A constant bias would be wrong for any aircraft not flying at the speed it was
   calibrated for.
3. Storing `latency_s` lets the framework compute the correct per-aircraft bias at
   every tick, automatically handling mixed-speed scenarios.

### `_track_relative_bias` — rotation helper

**File:** [`sim_models/cns/sensor.py:69`](sim_models/cns/sensor.py)

```python
def _track_relative_bias(trk, bias_at_m, bias_ct_m):
    trk_rad = np.deg2rad(trk)
    east  = bias_at_m * np.sin(trk_rad) - bias_ct_m * np.cos(trk_rad)
    north = bias_at_m * np.cos(trk_rad) + bias_ct_m * np.sin(trk_rad)
    return np.stack([east, north], axis=1)
```

`bias_at_m` may be a scalar or a shape `(n,)` per-aircraft array (numpy broadcasting
handles both). The return shape is `(n, 2)` — directly addable to the noise draw `exy`.

### `measure` — where the bias is applied

**File:** [`sim_models/cns/sensor.py:96`](sim_models/cns/sensor.py)

```python
exy = pos_dist(n, pos_ci95, rng)          # zero-mean isotropic noise, shape (n, 2)
if latency_s != 0.0 or cross_track_bias_m != 0.0:
    bias_at = -latency_s * gs             # per-aircraft, shape (n,)
    exy = exy + _track_relative_bias(trk, bias_at, cross_track_bias_m)
```

`gs` (m/s) and `trk` (degrees) are already read from `states` earlier in `measure`,
so no extra data extraction is needed. The bias is applied before `_apply_position_noise`
converts the (east, north) metre error to lat/lon degrees.

### `CNSState` and `make_cns` — configuration

**File:** [`sim_models/cns/cns.py:57`](sim_models/cns/cns.py)

`CNSState` carries `latency_s` and `cross_track_bias_m` as frozen fields. They are
passed through on every `step` call:

```python
# sim_models/cns/cns.py:99
sensor = measure(states, cns.pos_ci95, cns.vel_ci95,
                 cns.pos_dist, cns.vel_dist, cns.rng,
                 cns.latency_s, cns.cross_track_bias_m)
```

To construct a CNS with the ADS-B v2 noise model:

```python
from sim_models.cns.cns import make_cns

cns = make_cns(
    pos_ci95=10.0,           # 10 m ci95 for the isotropic cross-track draw
    vel_ci95=1.0,
    latency_s=0.0661,        # ADS-B v2 mean; bias auto-scales with each aircraft's gs
)
```

To change the latency between ticks (e.g., simulating different transponder versions):

```python
from dataclasses import replace
cns = replace(cns, latency_s=0.512)  # ADS-B v1 mean latency
cns = step(cns, traffic)
```

---

## Comparison with the isotropic Gaussian

| Property              | `make_biased_gaussian(bias=(ΔE, ΔN))` | `latency_s` on `make_cns`         |
|-----------------------|---------------------------------------|------------------------------------|
| Bias frame            | Fixed world (East, North)             | Track-relative (along, cross)      |
| Rotates with heading  | No                                    | Yes — recomputed every tick        |
| Scales with speed     | No — fixed metres                     | Yes — `−latency × gs` per aircraft |
| Physical meaning      | Generic constant offset               | ADS-B transmission lag             |
| Where applied         | Inside `pos_dist` callable            | In `measure()` after the draw      |

`make_biased_gaussian` remains available for scenarios where a fixed world-frame
offset is intentional (e.g., a known sensor installation offset). The two can be
combined: use `pos_dist=make_biased_gaussian(bias=...)` for any fixed-frame component
and `latency_s=...` for the latency-driven along-track component simultaneously.
