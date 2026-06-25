# Runners — Driving the Full CD/CR/CRR Pipeline

**Package:** `runners/`
**Module:** `runners/stochastic_pairwise_hor_conflict.py`

A *runner* is the orchestrator. It takes a scenario from an [env](envs.md), wraps
truth in the [CNS](cns.md) degraded view, threads the result through detection
→ resolution → recovery each tick, accumulates metrics, and (optionally) records
the trajectory history for plotting. Everything below the runner is a pure
function; the runner is where they are composed and the loop is run.

```
env (scenario) ──► runner ──► CNS ──► cd ──► cr ──► crr ──► env.step ──┐
        ▲                                                              │
        └──────────────────────  next tick  ───────────────────────────┘
```

---

## Public API

| Function | Purpose |
|---|---|
| `run_single(...)` | one full simulation; returns a result namespace (IPR + optional history) |
| `get_ipr(...)` | thin tuple wrapper around `run_single` (history forced off) for Monte Carlo |
| `run_parallel(...)` | runs `get_ipr` over many seeds with joblib; returns aggregate stats |

### `run_single(...)` → `SimpleNamespace`

The core entry point. Key parameters:

```python
run_single(
    pair_width, pair_height,
    rpz, hpz, dtlookahead,
    init_speed_ownship, init_speed_intruder,
    aircraft_type, dpsi,
    pos_ci95, vel_ci95, reception_prob,   # ← CNS noise / reception
    *,
    resofach=1.05,
    recovery_resofach=1.05,               # cpa strategy only
    prob_threshold=0.9, Ktheta=256,       # probabilistic strategy only
    cd=detect, cr=mvp.resolve,            # injectable pipeline stages
    crr="double_criteria",                # strategy name ("cpa"|"double_criteria"|
                                          #   "probabilistic") OR a crr(...) callable
    seed=44, record_history=False,
    ...
)
```

**Returned fields** (always): `ipr`, `t_end`, `dist_arr` `(T, nb_pair)`,
`min_dist` `(nb_pair,)`, `n_los`, `env`, and the echoed scenario inputs
(`rpz`, `hpz`, `dtlookahead`, `dpsi`, `pos_ci95`, `vel_ci95`, `reception_prob`).
With `record_history=True`, the per-tick arrays `t_arr`, `lat_arr`, `lon_arr`,
`gs_arr`, `hdg_arr`, `avoid_arr` are also populated (else `None`).

### `get_ipr(**kwargs)` → `(dist_arr, ipr, t_end)`

Same arguments as `run_single`, history forced off. Exists so the Monte Carlo
driver has a small, picklable return value.

### `run_parallel(n_runs, n_jobs, base_seed=42, **kwargs)` → `dict`

Runs `get_ipr` `n_runs` times with independent seeds (`base_seed + rep`) via
`joblib.Parallel`, and aggregates:

```python
{
    "overall_ipr": float,        # aggregated across all runs and pairs
    "ipr":        np.ndarray,    # per-run IPR, (n_runs,)
    "worst_cpa":  np.ndarray,    # min CPA across pairs per run (m)
    "t_end":      np.ndarray,    # termination time per run (s)
}
```

---

## The simulation loop

Time advances in `simdt` steps, but the ASAS pipeline (CD/CR/CRR) only fires on
an `asas_dt` cadence. Between ASAS events the last `action` is re-applied so the
aircraft keep flying their commanded velocities.

```python
while t < tmax:
    if t + eps >= next_event_t:                 # ── ASAS tick ──
        cns     = cns_step(cns, bs.traf)        # 1. degrade truth
        obs     = _as_obs(cns.sensor)           # 2. traffic-like noisy view
        conf    = cd(obs, obs, rpz, hpz, dtlookahead)         # 3. detect
        conf_gt = cd(bs.traf, bs.traf, rpz, hpz, dtlookahead) # ground-truth (termination only)

        newtrack, newgs, newvs, alt = cr(conf, obs, obs, cfg)         # 4. resolve
        recovery_state, _ = crr(recovery_state, conf, obs, obs, active) # 5. recover
        action = (newtrack, newgs, newvs, alt, list(recovery_state.resopairs))

        # stop once truth is conflict-free AND nothing is still resolving, held for done_timeout
        ...
    distances = step(env, action)               # advance BlueSky one simdt
    t += simdt
```

Two detection calls are made deliberately:

- `cd(obs, obs, …)` — on the **noisy** view; this drives CR/CRR (what the aircraft
  actually perceive).
- `cd(bs.traf, bs.traf, …)` — on **ground truth**; used only for the termination
  test, so the run ends on actual safety, not on a noisy artefact.

### Termination

A latch-with-timeout (`_done_with_timeout`): once truth has no conflict pairs and
no pairs remain under resolution, a timer starts; the loop stops when that
condition has held continuously for `done_timeout` seconds (or at `tmax`).

---

## Observation model (`_as_obs`)

The runner is where the CNS meets the algorithms. `_as_obs` wraps the per-tick
sensor snapshot as a duck-typed traffic object: **measured** state from the CNS,
**onboard** parameters (`perf`, `selalt`) from `bs.traf`, and an `adsl`
sub-namespace carrying the advertised accuracy (`pos_acc` / `vel_acc`) that the
probabilistic recovery rule consumes. See [cns.md §6](cns.md) for the full data
flow and the `reception_prob < 1.0` caveat.

---

## Injectable pipeline stages

The three algorithmic stages are parameters, so alternative algorithms (including
a learned policy) drop in without touching the loop:

```python
cd(ownship, intruder, rpz, hpz, dtlookahead) -> conf
cr(conf, ownship, intruder, cfg)             -> (newtrack, newgs, newvs, alt)
crr(state, conf, ownship, intruder, active)  -> (new_state, delpairs)
```

All three stages match their call signatures directly, so the module-level
defaults `detect` / `mvp.resolve` work as-is. `crr` is dual-typed: pass a
**callable** with the signature above to use it as-is, or pass a **strategy
name** (the default, `"double_criteria"`) and the runner resolves it via
`crr.make_recovery`, binding this env's no-op `recover` (the env handles route
resumption; see [envs.md](envs.md)) and forwarding the strategy knobs below:

| `crr` name | Underlying rule | Strategy knobs |
|---|---|---|
| `"cpa"` | `resumenav_cpa` — release past CPA | `recovery_resofach` |
| `"double_criteria"` | `resumenav_double_criteria` — deterministic FTR | — |
| `"probabilistic"` | `resumenav_probabilistic_ftr` — sensor-noise-aware FTR | `prob_threshold`, `Ktheta` |

For full control, build a callable yourself and pass it as `crr`, e.g.
`crr=make_recovery("cpa", resofach=1.1, recover=my_recover)`. The
`recovery_resofach` / `prob_threshold` / `Ktheta` knobs apply only to the
name form.

---

## Metrics

- **Realised CPA per pair:** `min_dist = dist_arr.min(axis=0)`.
- **Loss of separation (LoS):** a pair whose realised CPA fell below `rpz`.
- **IPR (Intrusion Prevention Rate):**

$$\text{IPR} = 1 - \frac{n_\text{LoS}}{n_\text{pair}}$$

`run_parallel` aggregates LoS across all runs and pairs, so `overall_ipr` is the
pooled rate rather than a mean of per-run IPRs.

---

## Plotting

`run_single(record_history=True)` results feed the helpers in
[`plot_utils.py`](../plot_utils.py): `plot_distances`, `plot_gs_hdg`,
`plot_avoidance`, `plot_trajectories` (or `plot_run` for all four), and
`plot_avoidance_compare` to overlay strategies. The figure-generating example is
[`tests/test_stochastic_pairwise_hor_conflict_sim.py`](../tests/test_stochastic_pairwise_hor_conflict_sim.py),
which runs all three recovery strategies over a shared scenario and writes the
per-strategy and comparison PNGs to `figures/tests/`.

---

## Usage

```python
from runners.stochastic_pairwise_hor_conflict import run_single, run_parallel

# Single run with history for plotting:
res = run_single(
    pair_width=3, pair_height=3,
    rpz=50.0, hpz=50.0, dtlookahead=121.0,
    init_speed_ownship=15.0, init_speed_intruder=15.0,
    aircraft_type="M600", dpsi=90.0,
    pos_ci95=10.0, vel_ci95=1.0, reception_prob=1.0,
    crr="probabilistic", prob_threshold=0.9,
    record_history=True,
)
print(res.ipr, res.n_los)

# Monte Carlo across 50 seeds:
stats = run_parallel(
    n_runs=50, n_jobs=10,
    pair_width=3, pair_height=3,
    rpz=50.0, hpz=50.0, dtlookahead=121.0,
    init_speed_ownship=15.0, init_speed_intruder=15.0,
    aircraft_type="M600", dpsi=90.0,
    pos_ci95=10.0, vel_ci95=1.0, reception_prob=1.0,
)
print(stats["overall_ipr"])
```
