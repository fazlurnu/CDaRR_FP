# Running Experiments with Different Noise Distributions

This document describes how to run the stochastic pairwise horizontal conflict
simulation under three different position noise models and compare their effect
on the Intrusion Prevention Rate (IPR).

---

## The Three Models

| Model | What it captures | Key parameter |
|---|---|---|
| Normal Gaussian | Isotropic zero-mean noise | `pos_ci95` only |
| Latency bias | Gaussian noise + along-track shift proportional to speed | `latency_s` |
| Mixture Gaussian | Isotropic, zero-mean, heavy-tailed | `pos_dist` |

All three share the same `pos_ci95` so the nominal 95% containment radius is
held constant across experiments. See [`latency-alongtrack-bias.md`](latency-alongtrack-bias.md)
for the physics behind the latency model.

---

## Step 1 — Modify `run_single`

In [`runners/stochastic_pairwise_hor_conflict.py`](runners/stochastic_pairwise_hor_conflict.py),
add two keyword arguments to the `run_single` signature alongside the existing
`pos_ci95` / `vel_ci95`:

```python
def run_single(
    ...
    pos_ci95: float,
    vel_ci95: float,
    reception_prob: float,
    pos_dist=None,            # pass a distribution callable, or None → gaussian
    latency_s: float = 0.0,   # ADS-B latency in seconds; 0.0 = no bias
    ...
) -> SimpleNamespace:
```

Then pass them through to `make_cns` (line 278–279):

```python
cns = make_cns(
    pos_ci95=pos_ci95,
    vel_ci95=vel_ci95,
    reception_prob=reception_prob,
    seed=seed,
    pos_dist=pos_dist,      # ← new
    latency_s=latency_s,    # ← new
)
```

`get_ipr` and `run_parallel` both use `**kwargs` and require no changes — new
arguments flow through automatically.

---

## Step 2 — Call the Runner

```python
from runners.stochastic_pairwise_hor_conflict import run_single, run_parallel
from sim_models.cns.distributions import make_mixture_gaussian

COMMON = dict(
    pair_width=3, pair_height=3,
    rpz=50.0, hpz=50.0, dtlookahead=121.0,
    init_speed_ownship=15.0, init_speed_intruder=15.0,
    aircraft_type="M600", dpsi=90.0,
    pos_ci95=10.0, vel_ci95=1.0, reception_prob=1.0,
)

# 1. Normal Gaussian — baseline, no extra arguments needed
res_normal = run_single(**COMMON)

# 2. Latency bias — ADS-B v2 mean latency (0.0661 s)
#    bias = −0.0661 × gs per aircraft per tick, rotated into (east, north)
res_latency = run_single(**COMMON, latency_s=0.0661)

# 3. Mixture Gaussian — same ci95, but 10% chance of a 3× wider draw
res_mixture = run_single(
    **COMMON,
    pos_dist=make_mixture_gaussian(tail_ratio=3.0, tail_weight=0.1),
)
```

---

## Step 3 — Monte Carlo Comparison

Use `run_parallel` to get statistically stable IPR estimates across seeds.
The `**kwargs` pass-through means the same interface works:

```python
MC = dict(n_runs=200, n_jobs=8, **COMMON)

stats_normal  = run_parallel(**MC)
stats_latency = run_parallel(**MC, latency_s=0.0661)
stats_mixture = run_parallel(
    **MC,
    pos_dist=make_mixture_gaussian(tail_ratio=3.0, tail_weight=0.1),
)

for label, stats in [
    ("Normal",  stats_normal),
    ("Latency", stats_latency),
    ("Mixture", stats_mixture),
]:
    print(f"{label:10s}  IPR={stats['overall_ipr']:.3f}  "
          f"worst_CPA={stats['worst_cpa'].min():.1f} m")
```

---

## Combining Latency and Mixture

The two parameters are independent and can be activated together:

```python
# Heavy-tailed noise AND along-track latency bias simultaneously
res_full = run_single(
    **COMMON,
    pos_dist=make_mixture_gaussian(tail_ratio=3.0, tail_weight=0.1),
    latency_s=0.0661,
)
```

This is the most realistic model: the dominant noise component is isotropic
with occasional outliers (mixture), and the reported position also lags behind
the aircraft in the direction of travel (latency).

---

## Parameter Reference

### `latency_s`

Mean ADS-B position reporting latency in seconds. The per-aircraft along-track
bias is recomputed as `−latency_s × gs` every tick, so it scales automatically
with each aircraft's current ground speed.

| ADS-B version | Typical `latency_s` |
|---|---|
| v2 (current standard) | `0.0661` |
| v1 | `0.512` |
| v0 (legacy) | `0.655` |

### `make_mixture_gaussian(tail_ratio, tail_weight)`

| Parameter | Effect | Typical value |
|---|---|---|
| `tail_ratio` | How much wider the tail component is (σ₂ = ratio × σ₁) | `3.0` |
| `tail_weight` | Probability of drawing from the tail component | `0.1` |

The dominant component sigma σ₁ is solved numerically so that the 95th
percentile of the 2D radial distance always equals `pos_ci95`, regardless
of `tail_ratio` and `tail_weight`. See [`distributions.py`](sim_models/cns/distributions.py)
for the bisection derivation.
