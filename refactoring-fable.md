# Refactoring plan: pluggable CD/CR/CRR stages behind a single policy function

**Audience:** an implementation agent (Sonnet) executing with minimal supervision.
**Repo:** `/Users/mfrahman/Projects/CDaRR_FP` (CDaRR — Conflict Detection, Resolution and Recovery on BlueSky).

## 1. Goals

1. **Pluggability.** A new user must be able to swap in their own conflict-detection (CD),
   conflict-resolution (CR), or recovery (CRR) logic without touching the simulation loop.
   Today CRR has a registry (`crr.RECOVERY_STRATEGIES` + `crr.make_recovery`) but CD and CR
   do not, the stage signatures are inconsistent, and the required "traffic view" duck-type
   is documented nowhere except the `_as_obs` helper buried in the runners.
2. **A single mapping function.** Wrap CD + CR + CRR into one function that maps the
   *observed states* of ownship and intruder to (a) the commanded action of each aircraft
   and (b) the avoidance/recovery status of each aircraft:

   ```
   policy(memory, ownship_obs, intruder_obs) -> (new_memory, Decision)
   ```

   This becomes the only seam between the runner loop and the CDR logic.
   (*memory*, not *state*: the observations already carry the aircraft
   kinematic state; the first argument is the pipeline's cross-tick memory —
   unresolved pairs, initial intruder velocities, ASAS-active flags.)

### Non-goals / constraints

- **Behavior-preserving.** Given the same seed and parameters, `run_single` must produce
  bit-identical `dist_arr`, `ipr`, `t_end`, and history arrays before and after
  (verified in Phase 0 / Phase 9). Do not reorder any RNG draws or BlueSky calls.
- **No public-API breaks.** These call sites must keep working unchanged:
  - `runners.stochastic_pairwise_hor_conflict.{run_single, get_ipr, run_parallel}` —
    used by `analysis/pairwise_hor_conflict_analysis.py`, `analysis/latency_analysis.py`,
    `plot_utils.py`, `experiments/exp1-crossing-angle.py`, `experiments/exp2-gamma.py`,
    `tests/test_stochastic_pairwise_hor_conflict_sim.py`,
    `tests/test_parallel_stochastic_pairwise_hor_conflict_sim.py`,
    `tests/test_detailed_pair_plot_sim.py`.
  - `runners.stochastic_pairwise_hor_conflict_heterogeneous_speed.{run_single, get_ipr, run_parallel}` —
    used by `analysis/latency_analysis_heterogeneous.py`, `experiments/exp3-noise-model-random-angle.py`.
  - The keyword names `cd=`, `cr=`, `crr=`, `crr="double_criteria"|"cpa"|"probabilistic"`,
    `recovery_resofach=`, `prob_threshold=`, `Ktheta=` on both `run_single`s.
  - Legacy injected callables: `cd(ownship, intruder, rpz, hpz, dtlookahead)`,
    `cr(conf, ownship, intruder, cfg)`,
    `crr(state, conf, ownship, intruder, active)` must still be accepted by `run_single`.
  - The env legacy action 5-tuple `(newtrack, newgs, newvs, alt, resopairs)` must still be
    accepted by `envs.*.step` and `envs.*.avoidance_mask` (used by
    `tests/test_pairwise_hor_conflict_sim.py` and notebooks in `brouillon/`).
- Do **not** rename the `crr` keyword argument of `run_single` even though it shadows the
  `crr` package inside those modules (external callers pass `crr='probabilistic'` etc.).
- Do not touch `analysis/`, `experiments/`, `plot_utils.py`, `paper/`, `brouillon/` except
  where explicitly stated (they should not need changes).

### Environment

- Python: `/Users/mfrahman/anaconda3/envs/cdarr/bin/python` (conda env `cdarr`).
- Run tests with: `/Users/mfrahman/anaconda3/envs/cdarr/bin/python -m pytest tests/ -x -q`.
  Some sim tests take a minute or two; that is normal. Run the full suite at least once
  before starting and once at the end.
- Temporary files (baselines, scratch scripts) go in a git-ignored scratch location, not in
  the repo tree.

---

## 2. Current architecture (for orientation)

```
envs/          scenario setup + BlueSky stepping (2 near-identical modules)
sim_models/cns noisy observation pipeline (CNSState, immutable, threaded)
cd/            detect(ownship, intruder, rpz, hpz, dtlookahead) -> ConflictState (frozen)
cr/            mvp.resolve / vo.resolve (conf, ownship, intruder, cfg) -> 4-tuple of commands
crr/           RECOVERY_STRATEGIES registry; uniform signature
               (state, conf, ownship, intruder, active, **params) -> (new_state, delpairs)
runners/       2 near-identical ~400-line modules; the CD→CR→CRR wiring lives inline in
               the while-loop of each run_single; action is an opaque positional 5-tuple
```

Pain points this plan fixes:
- The per-tick wiring `cns_step → _as_obs → cd → cr → crr → action-tuple` is copy-pasted in
  two runners; helpers `_as_obs`, `_noop_recover`, `_geom_dcpa`, `_done_with_timeout`,
  `_silence`, `get_ipr`, `run_parallel` are duplicated nearly verbatim.
- The action is an opaque tuple; `envs` unpack it by position (`action[4]`), and the
  per-aircraft avoidance status is *re-derived* in the env instead of being part of the
  decision.
- No formal contract for the "traffic view" object or the three stage signatures.
- CD and CR have no strategy registry / factory, unlike CRR.

---

## 3. Target architecture

```
                       ┌─────────────────────────────────────────────────┐
 observed states       │ pipeline.make_policy(cd=…, cr=…, crr=…,         │   commands +
 (ownship_obs,   ────▶ │                      rpz, hpz, dtlookahead)     │──▶ per-aircraft
  intruder_obs)        │ policy(memory, own, intr) -> (memory, Decision) │   avoidance/recovery
                       └─────────────────────────────────────────────────┘   status
```

New package `pipeline/`:

- `pipeline/types.py` — `TrafficView` Protocol, `Decision` frozen dataclass,
  `CDRMemory` frozen dataclass, status constants.
- `pipeline/policy.py` — `make_policy(...)` and `initial_memory(ntraf)`.
- `pipeline/__init__.py` — re-exports.

Registries: `cd.CD_STRATEGIES`/`cd.make_cd`, `cr.CR_STRATEGIES`/`cr.make_cr`
(mirroring the existing `crr.RECOVERY_STRATEGIES`/`crr.make_recovery`).

Runners: shared `runners/common.py` holding all duplicated helpers plus the core
`simulate()` loop; the two runner modules shrink to env construction + parameter echo.

Envs: `step()` / `avoidance_mask()` accept a `Decision` (and, for backward compatibility,
the legacy 5-tuple or `None`).

---

## 4. Phase 0 — capture a behavioral baseline

Before changing anything, create a scratch script (do **not** commit it) that exercises both
runners and all three recovery strategies, and saves the results:

```python
# baseline_capture.py  (run from repo root with the cdarr python)
import numpy as np
from runners.stochastic_pairwise_hor_conflict import run_single as run_hom
from runners.stochastic_pairwise_hor_conflict_heterogeneous_speed import run_single as run_het

COMMON = dict(pair_width=2, pair_height=2, rpz=50.0, hpz=50.0, dtlookahead=121.0,
              aircraft_type="M600", dpsi=90.0, pos_ci95=10.0, vel_ci95=1.0,
              reception_prob=1.0, tmax=200.0, seed=44, record_history=True)

cases = {}
for name, crr_kw in [("ftr", {}), ("cpa", dict(crr="cpa")),
                     ("prob", dict(crr="probabilistic", prob_threshold=0.9))]:
    r = run_hom(init_speed_ownship=15.0, init_speed_intruder=15.0, **COMMON, **crr_kw)
    cases[f"hom_{name}"] = r

r = run_het(speed_min=10.0, speed_max=30.0, **COMMON)
cases["het_ftr"] = r

np.savez("/path/to/scratch/baseline.npz", **{
    f"{k}_{f}": getattr(v, f)
    for k, v in cases.items()
    for f in ("dist_arr", "min_dist", "t_end", "lat_arr", "lon_arr",
              "gs_arr", "hdg_arr", "avoid_arr")
    if getattr(v, f) is not None
} | {f"{k}_ipr": np.float64(v.ipr) for k, v in cases.items()})
print("baseline saved")
```

Also run the full test suite once and record the pass/fail summary. After the refactor
(Phase 9) rerun the same script into `after.npz` and assert every array is **exactly equal**
(`np.array_equal`). If any array differs, the refactor changed behavior — stop and fix.

---

## 5. Phase 1 — `pipeline/types.py`: contracts and data types

Create `pipeline/__init__.py` and `pipeline/types.py`.

`pipeline/types.py` (complete implementation, adjust docstrings to taste but keep the
repo's docstring style — module docstring explaining the "why", `'''` quotes):

```python
'''Formal contracts and data types for the CD/CR/CRR pipeline.

* :class:`TrafficView`  — the duck-typed traffic object every stage consumes.
* :class:`Decision`     — the per-tick output of the composed pipeline: commands
  plus each aircraft's avoidance/recovery status.
* :class:`CDRMemory`    — the pipeline's cross-tick memory (not to be confused
  with the aircraft kinematic state, which lives in the observations).
* Stage protocols (:class:`ConflictDetector`, :class:`ConflictResolver`,
  :class:`RecoveryModel`) documenting exactly what a user-supplied stage must
  look like.
'''
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from cd.common import ConflictState
from crr.common import RecoveryState

# Per-aircraft status codes reported in Decision.status.
STATUS_NOMINAL    = 0   # flying its nominal route
STATUS_AVOIDING   = 1   # in a currently-detected conflict, manoeuvring
STATUS_RECOVERING = 2   # conflict no longer detected, but not yet released
                        # by the recovery criterion (still holding the manoeuvre)

STATUS_NAMES = {STATUS_NOMINAL: 'nominal',
                STATUS_AVOIDING: 'avoiding',
                STATUS_RECOVERING: 'recovering'}


@runtime_checkable
class TrafficView(Protocol):
    '''What every pipeline stage expects of a traffic snapshot.

    Both BlueSky's live ``bs.traf`` and the CNS sensor wrapper satisfy this.
    All arrays are length ``ntraf`` and share one index order.
    '''
    ntraf: int
    id: list           # aircraft id strings
    lat: np.ndarray    # deg
    lon: np.ndarray    # deg
    alt: np.ndarray    # m
    trk: np.ndarray    # deg
    gs: np.ndarray     # m/s
    vs: np.ndarray     # m/s
    gseast: np.ndarray
    gsnorth: np.ndarray
    # Required by the CR stage and Decision assembly:
    perf: object       # .vmin/.vmax/.vsmin/.vsmax arrays
    selalt: np.ndarray
    # Required by the probabilistic recovery rule (may be absent otherwise):
    # adsl: object with .pos_acc / .vel_acc arrays (95% accuracy radii)


class ConflictDetector(Protocol):
    def __call__(self, ownship, intruder, rpz, hpz, dtlookahead) -> ConflictState: ...


class ConflictResolver(Protocol):
    '''Bound resolver: configuration is already baked in (see cr.make_cr).'''
    def __call__(self, conf, ownship, intruder) -> tuple: ...
    # returns (newtrack, newgs, newvs, alt), each an (ntraf,) array


class RecoveryModel(Protocol):
    def __call__(self, state, conf, ownship, intruder, active) -> tuple: ...
    # returns (new_recovery_state, released_pairs)


@dataclass(frozen=True)
class CDRMemory:
    '''The pipeline's cross-tick memory (immutable, threaded explicitly).

    Deliberately named *memory*, not *state*: the aircraft kinematic state is
    what the observations carry. This object holds only what the pipeline must
    remember between ticks and cannot recover from the current observation —
    unresolved conflict pairs, intruder velocities at conflict initiation, and
    the crr-managed ASAS-active flags.
    '''
    recovery: RecoveryState
    asas_active: np.ndarray   # (ntraf,) bool — the crr-managed ASAS-active flags


@dataclass(frozen=True)
class Decision:
    '''One tick's output: what each aircraft should do, and why.

    All arrays are per-aircraft, in the same index order as the observation
    passed to the policy (which matches ``bs.traf`` order in these envs).
    '''
    newtrack: np.ndarray   # deg
    newgs: np.ndarray      # m/s
    newvs: np.ndarray      # m/s
    alt: np.ndarray        # m
    avoiding: np.ndarray   # bool — True while the aircraft is in a resolution pair
                           # (i.e. must fly the resolution command, not its route)
    status: np.ndarray     # int — STATUS_NOMINAL / STATUS_AVOIDING / STATUS_RECOVERING
    resopairs: frozenset   # unresolved (ownship_id, intruder_id) pairs after this tick
    conf: ConflictState    # the detection result this decision was based on
```

`pipeline/__init__.py`:

```python
'''Composable CD/CR/CRR pipeline — one function from observations to actions.'''
from .types import (CDRMemory, Decision, STATUS_AVOIDING, STATUS_NAMES,
                    STATUS_NOMINAL, STATUS_RECOVERING)
from .policy import initial_memory, make_policy

__all__ = ['CDRMemory', 'Decision', 'initial_memory', 'make_policy',
           'STATUS_NOMINAL', 'STATUS_AVOIDING', 'STATUS_RECOVERING', 'STATUS_NAMES']
```

---

## 6. Phase 2 — strategy registries for CD and CR

### 6.1 `cd/__init__.py`

Add (keeping existing exports):

```python
import functools

CD_STRATEGIES = {
    'statebased': detect,
}

def make_cd(name, **params):
    '''Build a detector callable ``cd(ownship, intruder, rpz, hpz, dtlookahead)``.

    ``name`` selects a strategy from ``CD_STRATEGIES``; ``params`` (if any) are
    bound up front with functools.partial.
    '''
    try:
        fn = CD_STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown CD strategy: {name!r}; "
                         f"known: {sorted(CD_STRATEGIES)}")
    return functools.partial(fn, **params) if params else fn
```

### 6.2 `cr/__init__.py`

Registry values keep the existing *unbound* signature
`(conf, ownship, intruder, cfg, resofach=None)`; `make_cr` binds the config so the
pipeline sees the uniform bound signature `(conf, ownship, intruder)`:

```python
CR_STRATEGIES = {
    'mvp': mvp.resolve,
    'vo':  vo.resolve,
}

def make_cr(name, resofach=1.0):
    '''Build a bound resolver ``cr(conf, ownship, intruder)``.

    ``name`` selects a strategy from ``CR_STRATEGIES``; ``resofach`` is baked
    into a :class:`ResolutionConfig`. User strategies registered in
    ``CR_STRATEGIES`` must accept ``(conf, ownship, intruder, cfg)`` and may
    ignore ``cfg``.
    '''
    try:
        fn = CR_STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown CR strategy: {name!r}; "
                         f"known: {sorted(CR_STRATEGIES)}")
    cfg = ResolutionConfig(resofach=resofach)

    def resolver(conf, ownship, intruder):
        return fn(conf, ownship, intruder, cfg)

    return resolver
```

Update `cr/__init__.py.__all__` and module docstring accordingly. `crr` already has its
registry — leave it as is.

---

## 7. Phase 3 — `pipeline/policy.py`: the single mapping function

Complete implementation:

```python
'''Compose CD, CR and CRR into a single observations→decision policy.

:func:`make_policy` binds the three stages plus the protected-zone parameters
and returns a function with the uniform signature::

    policy(memory: CDRMemory, ownship_obs, intruder_obs) -> (CDRMemory, Decision)

The returned policy is the only thing a simulation loop needs to call per ASAS
tick; everything the environment needs (per-aircraft commands and
avoidance/recovery status) is in the :class:`~pipeline.types.Decision`.
'''
import numpy as np

from crr.common import empty_recovery_state

from .types import (CDRMemory, Decision, STATUS_AVOIDING, STATUS_NOMINAL,
                    STATUS_RECOVERING)


def initial_memory(ntraf: int) -> CDRMemory:
    '''Fresh, empty pipeline memory for a scenario with ``ntraf`` aircraft.'''
    return CDRMemory(recovery=empty_recovery_state(),
                    asas_active=np.zeros(ntraf, dtype=bool))


def _membership_mask(ids, pairs) -> np.ndarray:
    '''(ntraf,) bool — True where the aircraft id appears in any pair.'''
    mask = np.zeros(len(ids), dtype=bool)
    if pairs:
        for i, acid in enumerate(ids):
            mask[i] = any(acid in pair for pair in pairs)
    return mask


def make_policy(*, cd, cr, crr, rpz, hpz, dtlookahead):
    '''Bind the three stages into one observations→decision function.

    Parameters
    ----------
    cd  : callable ``(ownship, intruder, rpz, hpz, dtlookahead) -> ConflictState``
    cr  : bound resolver ``(conf, ownship, intruder) -> (newtrack, newgs, newvs, alt)``
    crr : recovery callable ``(state, conf, ownship, intruder, active) ->
          (new_state, released)`` (build with :func:`crr.make_recovery`)
    rpz, hpz, dtlookahead : protected-zone geometry, bound once per scenario.
    '''
    def policy(memory: CDRMemory, ownship, intruder):
        conf = cd(ownship, intruder, rpz, hpz, dtlookahead)
        newtrack, newgs, newvs, alt = cr(conf, ownship, intruder)

        # crr mutates the active array in place (its documented contract);
        # copy so the incoming CDRMemory stays immutable.
        asas_active = memory.asas_active.copy()
        recovery, _released = crr(memory.recovery, conf, ownship, intruder,
                                  asas_active)

        ids = list(ownship.id)
        avoiding = _membership_mask(ids, recovery.resopairs)
        in_conf  = _membership_mask(ids, conf.confpairs)
        status = np.where(in_conf, STATUS_AVOIDING,
                 np.where(avoiding, STATUS_RECOVERING, STATUS_NOMINAL))

        new_memory = CDRMemory(recovery=recovery, asas_active=asas_active)
        decision = Decision(
            newtrack=newtrack, newgs=newgs, newvs=newvs, alt=alt,
            avoiding=avoiding, status=status,
            resopairs=recovery.resopairs, conf=conf,
        )
        return new_memory, decision

    return policy
```

**Invariant to preserve:** `avoiding` must be computed from the *post-crr* `resopairs`
(exactly what the legacy `envs.avoidance_mask(action)` computed from `action[4] =
list(recovery_state.resopairs)`). Do not compute it from `conf.confpairs`.

---

## 8. Phase 4 — envs consume `Decision` (legacy tuple still accepted)

Apply the same edit to **both** `envs/pairwise_hor_conflict.py` and
`envs/pairwise_hor_conflict_heterogeneous_speed.py`.

Add at the top:

```python
from pipeline.types import Decision
```

Rewrite `avoidance_mask`:

```python
def avoidance_mask(action) -> np.ndarray:
    '''Per-aircraft avoidance flags — 1.0 while in an active resolution pair.

    Accepts a :class:`pipeline.types.Decision`, the legacy 5-tuple
    ``(newtrack, newgs, newvs, alt, resopairs)``, or ``None``. Order matches
    ``bs.traf.id``.
    '''
    if action is None:
        return np.zeros(bs.traf.ntraf, dtype=float)
    if isinstance(action, Decision):
        return action.avoiding.astype(float)
    resopairs = action[4]
    mask = np.zeros(bs.traf.ntraf, dtype=float)
    if resopairs:
        for i in range(bs.traf.ntraf):
            if any(bs.traf.id[i] in pair for pair in resopairs):
                mask[i] = 1.0
    return mask
```

In `_apply_action`, replace the tuple unpack with:

```python
    reso_hdg = reso_spd = None
    if isinstance(action, Decision):
        reso_hdg, reso_spd = action.newtrack, action.newgs
    elif action is not None:
        reso_hdg, reso_spd, _, _, _ = action
```

Everything else in `_apply_action` (the per-aircraft HDG/SPD stacking, nominal-speed
restore) stays byte-for-byte the same. Update the module docstrings' usage examples to show
`step(env, decision)`.

**Ordering invariant:** `Decision` arrays are in the observation's index order; the CNS
sensor preserves `bs.traf` creation order, so `decision.newtrack[i]` corresponds to
`bs.traf.id[i]`. State this in the docstring.

---

## 9. Phase 5 — deduplicate the runners into `runners/common.py`

Create `runners/common.py` containing everything currently duplicated between the two
runner modules, moved verbatim unless noted:

1. `silence()` — the stdout/stderr suppressor (rename from `_silence`, it's shared now).
2. `ensure_bluesky()` — the `bs._sim_inited` guard + `bs.init(mode="sim", detached=True)`.
3. `as_traffic_view(sensor, traf)` — the former `_as_obs`, with `bs.traf` made an explicit
   parameter (`perf=traf.perf`, `selalt=traf.selalt`). Callers pass `bs.traf`.
4. `noop_recover(_idx)` — the former `_noop_recover`.
5. `geom_dcpa(view, env)` — the former `_geom_dcpa` (identical in both runners).
6. `done_with_timeout(...)` — the former `_done_with_timeout`.
7. `normalize_stages(...)` — new; turns the user-facing `cd`/`cr`/`crr` arguments into
   bound callables:

```python
def normalize_stages(cd, cr, crr, *, resofach, cfg,
                     recovery_resofach, prob_threshold, Ktheta, recover):
    '''Resolve str-or-callable stage specs into bound stage callables.

    Legacy callables keep their documented contracts:
      cd(ownship, intruder, rpz, hpz, dtlookahead)
      cr(conf, ownship, intruder, cfg)          — cfg is bound here
      crr(state, conf, ownship, intruder, active)
    '''
    from cd import make_cd
    from cr import make_cr
    from crr import make_recovery

    cd_fn = make_cd(cd) if isinstance(cd, str) else cd

    if isinstance(cr, str):
        cr_fn = make_cr(cr, resofach=resofach)
    else:
        def cr_fn(conf, ownship, intruder, _fn=cr, _cfg=cfg):
            return _fn(conf, ownship, intruder, _cfg)

    if isinstance(crr, str):
        crr_fn = make_recovery(crr, recover=recover,
                               resofach=recovery_resofach,
                               prob_threshold=prob_threshold, Ktheta=Ktheta)
    else:
        crr_fn = crr
    return cd_fn, cr_fn, crr_fn
```

8. `simulate(...)` — the shared core loop, extracted from `run_single`. Signature:

```python
def simulate(*, env, env_step, env_avoidance_mask, env_reset,
             policy, cd_gt, cns, rpz, hpz, dtlookahead,
             tmax, done_timeout, simdt, record_history) -> SimpleNamespace
```

Body = the existing `while t < tmax:` loop of
`runners/stochastic_pairwise_hor_conflict.py::run_single` (lines ~285–346), with these
mechanical substitutions and nothing else:

- `recovery_state` / `active` replaced by `memory = initial_memory(bs.traf.ntraf)`.
- The five wiring lines replaced by:

  ```python
  cns  = cns_step(cns, bs.traf)
  obs  = as_traffic_view(cns.sensor, bs.traf)
  memory, decision = policy(memory, obs, obs)
  conf_gt = cd_gt(bs.traf, bs.traf, rpz, hpz, dtlookahead)
  action  = decision
  done_now = (len(conf_gt.confpairs) == 0
              and len(memory.recovery.resopairs) == 0)
  ```

  ⚠️ Keep the call order **cns_step → policy → cd_gt** equivalent to today's
  **cns_step → cd(obs) → cd_gt → cr → crr**. `cd_gt` runs on ground truth and draws no
  random numbers, so moving it after `cr`/`crr` does not affect RNG state — but if in doubt
  keep it in the original position by having `simulate` call the pieces in the original
  order (policy already encapsulates cd/cr/crr in the original relative order; cd_gt may
  run before or after the policy with identical results; choose after, as above).
- History recording identical, with `avoid_list.append(env_avoidance_mask(action))`.
- Always record `sensor_lat_list`/`sensor_lon_list` when `record_history` (the
  heterogeneous runner gains these two result fields — an additive, allowed change).
- Return a `SimpleNamespace` with the loop-level outputs (`ipr`, `t_end`, `dist_arr`,
  `min_dist`, `n_los`, plus the history arrays or `None`s). The wrappers add echoed
  input fields.

9. `run_parallel_impl(get_ipr_fn, *, n_runs, n_jobs, base_seed=42, **kwargs)` — the shared
   body of `run_parallel` (identical today in both modules), parameterised by the
   module's `get_ipr`.

### 9.1 Rewrite `runners/stochastic_pairwise_hor_conflict.py`

Keep the module docstring (update the injectable-stage paragraph to mention that `cd`/`cr`
also accept registry names, and point to `docs/extending.md`). Keep `run_single`'s exact
signature and defaults (`cd=detect`, `cr=mvp.resolve`, `crr="double_criteria"`, …).

Body becomes:

```python
    ensure_bluesky()
    cfg = ResolutionConfig(resofach=resofach)
    simdt = bs.settings.simdt * simdt_factor

    cd_fn, cr_fn, crr_fn = normalize_stages(
        cd, cr, crr, resofach=resofach, cfg=cfg,
        recovery_resofach=recovery_resofach, prob_threshold=prob_threshold,
        Ktheta=Ktheta, recover=noop_recover)

    env = make_pairwise_hor_conflict(...)          # unchanged

    policy = make_policy(cd=cd_fn, cr=cr_fn, crr=crr_fn,
                         rpz=rpz, hpz=hpz, dtlookahead=dtlookahead)

    cns = make_cns(...)                            # unchanged

    core = simulate(env=env, env_step=step, env_avoidance_mask=avoidance_mask,
                    env_reset=reset, policy=policy, cd_gt=cd_fn if isinstance(cd, str) or cd is detect else cd_fn,
                    cns=cns, rpz=rpz, hpz=hpz, dtlookahead=dtlookahead,
                    tmax=tmax, done_timeout=done_timeout, simdt=simdt,
                    record_history=record_history)

    return SimpleNamespace(**vars(core),
                           rpz=rpz, hpz=hpz, dtlookahead=dtlookahead, dpsi=dpsi,
                           pos_ci95=pos_ci95, vel_ci95=vel_ci95,
                           reception_prob=reception_prob, latency_s=latency_s)
```

Note on `cd_gt`: today the ground-truth check uses the same `cd` callable as the observed
check. Preserve that: pass the normalized `cd_fn` as `cd_gt` (the conditional in the sketch
above is redundant — just pass `cd_gt=cd_fn`).

`get_ipr` and `run_parallel` keep their signatures; `run_parallel` delegates to
`run_parallel_impl(get_ipr, ...)`.

### 9.2 Rewrite `runners/stochastic_pairwise_hor_conflict_heterogeneous_speed.py`

Same treatment: delete the duplicated helpers, import from `runners.common`, build the
heterogeneous env, call `simulate`, echo `speed_min`/`speed_max` in the result. Keep
`_SPEED_SEED_OFFSET` and the env construction exactly as they are.

Both modules keep the `sys.path.insert(0, ...)` header (scripts are run from anywhere).

---

## 10. Phase 6 — tests

Run the suite after each phase; everything existing must stay green **unmodified** —
`tests/test_pairwise_hor_conflict_sim.py` still builds a legacy action tuple and must pass
via the env compatibility path.

Add `tests/test_pipeline.py` covering, with the lightweight fakes from
`tests/conftest.py` (`make_traffic`, `make_id2idx`, `make_recorder`) and **no BlueSky**:

1. **Composition:** build a policy from `cd.detect`, `cr.make_cr('mvp', resofach=1.05)`,
   and `crr.make_recovery('double_criteria', id2idx=..., recover=...)` on a two-aircraft
   head-on fake; assert the returned `Decision` arrays have shape `(ntraf,)`, that both
   aircraft are `STATUS_AVOIDING` while the conflict is detected, and that
   `decision.avoiding` matches membership of `decision.resopairs`.
2. **Purity:** calling the policy twice with the same `CDRMemory` and inputs gives
   equal `Decision`s and does not mutate the input memory (check `memory.asas_active`
   unchanged, `memory.recovery` is the same object).
3. **Custom CD injection:** a stub detector that returns an empty `ConflictState` (no
   pairs) → all-`STATUS_NOMINAL` decision, commands equal the CR's no-conflict output.
4. **Custom CR injection:** register a toy resolver in `CR_STRATEGIES` (e.g. constant
   +30° track offset), `make_cr` it, assert the decision carries its output; unregister in
   a `finally`.
5. **Registries:** `make_cd('nope')` / `make_cr('nope')` raise `ValueError` listing known
   names (mirror the existing `make_recovery` test if one exists; if not, add one).
6. **Status transition:** drive the head-on fake a few "ticks" by manually moving the fake
   aircraft past CPA (see `tests/test_recovery.py` for the pattern) and assert an aircraft
   goes `AVOIDING → RECOVERING → NOMINAL` (`avoiding` flag drops when the recovery rule
   releases the pair).
7. **Legacy tuple compat (env-level, needs BlueSky):** optional — already covered by the
   untouched `test_pairwise_hor_conflict_sim.py`.

---

## 11. Phase 7 — documentation

1. New `docs/extending.md` — "Bring your own CD / CR / recovery". Contents:
   - The three stage contracts (copy the Protocol signatures from `pipeline/types.py`),
     the `TrafficView` field table, and the `Decision` field table.
   - A worked example: write a toy CR (`always_right_30deg`), add it to
     `cr.CR_STRATEGIES`, build a policy with `make_policy`, and run it through
     `run_single(cr='always_right_30deg')`... note that `run_single` accepts registry
     names for `cd`/`cr`/`crr` and bare callables with the legacy signatures.
   - A worked example of a custom recovery rule registered in
     `crr.RECOVERY_STRATEGIES`.
   - A short "testing your stage without BlueSky" section pointing at
     `tests/conftest.py::make_traffic`.
2. Update `docs/index.md`: add `pipeline` to the pipeline diagram (the box that wraps
   cd/cr/crr), add `extending.md` and a `pipeline.md` row to the table, and update the
   quick-start to the policy API:

   ```python
   from pipeline import make_policy, initial_memory
   policy = make_policy(cd=make_cd('statebased'), cr=make_cr('mvp', resofach=1.05),
                        crr=make_recovery('double_criteria'), rpz=50, hpz=50,
                        dtlookahead=121)
   memory = initial_memory(traffic.ntraf)
   memory, decision = policy(memory, obs, obs)   # each ASAS tick
   ```
3. New `docs/pipeline.md` documenting `Decision`, `CDRMemory`, the status semantics
   (nominal/avoiding/recovering) and the ordering invariant.
4. Update `docs/runners.md` for the `runners/common.py` split (one paragraph + the new
   module map).

---

## 12. Phase 8 (optional — only if everything above is green and time permits)

Merge the two env modules: `PairwiseHorConflictEnv` gains a per-aircraft
`nominal_speed` array (length `2·nb_pair`, in `bs.traf` order) that
`make_pairwise_hor_conflict` fills with the two constant speeds and
`make_pairwise_hor_conflict_heterogeneous_speed` fills with the uniform draws; then
`_apply_action` restores `env.nominal_speed[i]` and both `step`s and both runners' env
duplication collapse. Keep both public factory names. **Skip this phase if any baseline
comparison in Phase 9 is not exact** — it touches the command-restoration path.

---

## 13. Phase 9 — final verification checklist

1. `/Users/mfrahman/anaconda3/envs/cdarr/bin/python -m pytest tests/ -q` — all green,
   including the new `tests/test_pipeline.py`.
2. Rerun the Phase 0 script into `after.npz`; compare with `np.array_equal` per key against
   `baseline.npz` — every key exact.
3. Grep checks:
   - `grep -rn "_as_obs\|_noop_recover\|_geom_dcpa\|_done_with_timeout\|_silence" runners/`
     → only definitions in `runners/common.py` (or references to the renamed versions).
   - `grep -rn "action\[4\]" envs/` → only inside the documented legacy-tuple branch.
4. Smoke-run one analysis entry point that imports the runners (import only, no full run):
   `/Users/mfrahman/anaconda3/envs/cdarr/bin/python -c "import analysis.pairwise_hor_conflict_analysis"`
   (and the same for `experiments.config`; the exp scripts execute sweeps at import, so do
   **not** import them — instead check them with
   `python -m py_compile experiments/exp1-crossing-angle.py experiments/exp2-gamma.py experiments/exp3-noise-model-random-angle.py`).
5. Delete the scratch baseline script/files; do not commit them or `refactoring-fable.md`
   changes beyond what the user asked.

---

## Appendix — behavioral invariants and gotchas

- **RNG order.** All stochasticity flows through `cns.rng` (one `np.random.default_rng`
  advanced by `cns_step`) and the env's speed/heading draws. The refactor must not add,
  remove, or reorder any call that consumes these generators.
- **`active` array semantics.** `crr` strategies *mutate* the passed `active` array via
  `apply_active_changes` and only touch indices they decide about; untouched indices keep
  their previous value. The policy therefore copies `memory.asas_active` each tick and
  stores the mutated copy — same values as today's single persistent array.
- **`avoiding` ≠ `asas_active`.** The env command-switch uses resopair *membership*
  (legacy `avoidance_mask`), not the crr `active` flags. Keep both; don't "simplify" one
  into the other.
- **Pair symmetry.** `detect` runs all-vs-all, so both `(A,B)` and `(B,A)` appear in
  `confpairs`/`resopairs`; membership checks rely on this — each aircraft is `idx1` of its
  own pair.
- **Held decisions between ASAS ticks.** The decision/action is computed on ASAS ticks and
  re-applied every sim tick until the next ASAS tick. `simulate` must keep this
  latch-and-reapply structure (including the `missed`/`next_event_t` catch-up arithmetic —
  copy it verbatim).
- **`frozenset` iteration order** of `resopairs` is only used for membership tests, never
  for indexing, so the switch from `list(resopairs)` (legacy tuple slot 4) to the frozenset
  on `Decision` is behavior-neutral.
- **`id2idx` returning −1** means the aircraft was deleted; recovery strategies handle
  `idx < 0`. Don't filter pairs before calling `crr`.
- **Name shadowing.** Inside runner modules the parameters `cd`, `cr`, `crr` shadow the
  packages of the same names — that's why `normalize_stages` does its imports locally.
  Keep it that way; do not rename the public keyword arguments.
- **BlueSky init.** `bs.init` may only run once per process (`bs._sim_inited` guard).
  `ensure_bluesky()` must be called before `make_pairwise_hor_conflict*`.
