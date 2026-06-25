# CDaRR — Documentation Index

CDaRR is a functional implementation of the **Conflict Detection, Resolution, and Recovery** pipeline for UAM/drone traffic management, built to sit on top of BlueSky.

## What the pipeline does

```
  envs                    ← spawn the scenario, step BlueSky
        │
        ▼
BlueSky truth (lat, lon, gs, trk, …)
        │
        ▼
  sim_models/cns          ← noisy, degraded view of the world
  (sensor + ADS-L)
        │
        ▼
  cd  (conflict detection) ← are we on a collision course?
        │
        ▼
  cr  (conflict resolution) ← what velocity change resolves it?
        │
        ▼
  crr (conflict recovery)   ← when is it safe to resume the route?
        │
        ▼
  runners                 ← compose the stages, loop, aggregate metrics
```

The **envs** and **runners** packages are the harness around this pipeline: an
env owns scenario setup and stepping, and a runner composes the CNS + cd/cr/crr
stages into the per-tick loop and collects metrics.

Each stage is a **pure function** of its inputs and returns an immutable result. Nothing mutates global state. Side effects (commanding the autopilot, writing ASAS-active flags) are injected as callables so the decision logic can be tested in isolation.

## Documents

| File | What it covers |
|---|---|
| [cns.md](cns.md) | Communication, Navigation & Surveillance model — sensor noise, ADS-L reception, `p_from_range` |
| [cd.md](cd.md) | State-based conflict detection — CPA geometry, horizontal/vertical windows |
| [cr.md](cr.md) | Conflict resolution — Modified Voltage Potential (MVP) and Velocity Obstacle (VO) |
| [crr.md](crr.md) | Conflict recovery — CPA criterion, deterministic FTR, probabilistic FTR |
| [envs.md](envs.md) | Scenario construction & BlueSky stepping — pairwise horizontal conflict |
| [runners.md](runners.md) | Driving the full pipeline — simulation loop, IPR metrics, Monte Carlo |

## Functional programming conventions

All modules share the same idiom, drawn from `cd/statebased.py` and `cd/common.py`.

**Immutable state containers**

```python
@dataclass(frozen=True)
class ConflictState: ...   # cd
class ResolutionConfig: ... # cr
class RecoveryState: ...    # crr
class CNSState: ...         # sim_models/cns
```

`frozen=True` prevents attribute rebinding after construction, making each instance a value. Updating means creating a new instance with `dataclasses.replace(old, field=new_value)`.

**Pure functions over duck-typed traffic**

The algorithms never import BlueSky directly. They operate on any object that exposes the right numpy arrays (`lat`, `lon`, `gs`, `trk`, `gseast`, `gsnorth`, …). This is what makes them unit-testable with lightweight fakes (see `tests/conftest.py`).

**Injected side effects**

The one unavoidable impure action in `crr` (writing the ASAS-active flag and commanding waypoint recovery) is passed in as a callable, defaulting to the live BlueSky call:

```python
def resumenav_cpa(state, conf, ownship, intruder, active,
                  resofach, id2idx=default_id2idx, recover=default_recover):
```

Tests pass their own `recover` callback; production passes the BlueSky one.

## Quick-start usage

```python
from sim_models.cns import make_cns, step, ownship_field, adsl_field
from cd import detect
from cr import mvp, ResolutionConfig
from crr import empty_recovery_state, resumenav_double_criteria

# --- initialise ---
cns   = make_cns(pos_ci95=5.0, vel_ci95=1.0, reception_prob=0.95, seed=42)
state = empty_recovery_state()
cfg   = ResolutionConfig()

# --- each tick ---
cns   = step(cns, traffic)                         # update sensor picture
traffic.sensor = cns.sensor                        # attach 1D ownship view
traffic.adsl   = cns.obs                           # attach N×N surveillance view

conf = detect(traffic, traffic, rpz=200, hpz=50, dtlookahead=300)

newtrk, newgs, newvs, alt = mvp.resolve(conf, traffic, traffic, cfg)

state, released = resumenav_double_criteria(
    state, conf, traffic, traffic, active)
```
