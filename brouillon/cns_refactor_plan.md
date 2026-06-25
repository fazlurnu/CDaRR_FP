# CNS Refactor Plan — Functional Programming Style

Implements `cns_spec_req_agent.md`, but **translated from the spec's OOP draft into
the repo's established FP idiom** (see `cd/statebased.py`, `cd/common.py`,
`tests/conftest.py`):

- **No mutable objects.** Every module exposes an immutable `@dataclass(frozen=True)`
  state container plus **pure functions** `(state, inputs) -> new_state`. No method
  mutates `self`; "resize" / "update" return a fresh value.
- **Distributions are functions, not an ABC.** A distribution is any callable
  `(n, ci95, rng) -> ndarray (n, 2)`. "Biased gaussian" is a higher-order factory
  returning a closure. This replaces the `NoiseDistribution(ABC)` hierarchy.
- **Small one-concern helpers**, each independently testable (matches `cd.statebased`).
- **RNG**: `np.random.Generator` is threaded as an explicit argument and stored on the
  CNS state, exactly as the legacy `noise_model`/`reception_model` and `conftest` already
  do. Determinism comes from the seed. (A pragmatic concession — same one the repo
  already makes — rather than JAX-style key splitting.)

Target layout (new, legacy `cns_old/` untouched):

```
sim_models/cns/
  __init__.py
  distributions.py
  sensor.py
  reception_model.py
  adsl_observation.py
  cns.py
tests/
  conftest.py            # extend: add make_cns_states fake
  test_cns_distributions.py
  test_cns_sensor.py
  test_cns_reception.py
  test_cns_observation.py
  test_cns.py
```

One commit per file (impl + its test), in dependency order:
`distributions → sensor → reception_model → adsl_observation → cns`.

---

## 0. `tests/conftest.py` — shared fake (prereq, do first)

`Sensor.measure` reads `states.{ntraf, lat, lon, alt, hdg, trk, gs, tas, vs, id}`. The
existing `make_traffic` lacks `hdg`/`tas`, so add a minimal states fake.

**Add:**
```python
def make_cns_states(lat, lon, trk, gs, *, hdg=None, tas=None, alt=None, vs=None, ids=None):
    """BlueSky-truth fake for the CNS layer: 1D arrays + ntraf + id."""
```
Returns a `SimpleNamespace(ntraf, id, lat, lon, alt, hdg, trk, gs, tas, vs)`.
Reuse the existing array-coercion pattern; default `hdg=trk`, `tas=gs`.

---

## 1. `distributions.py`

**Purpose:** pure 2D-error samplers in native units (m / m·s⁻¹).

**FP shape (replaces the ABC + 3 classes):**
```python
CI95_TO_STD_2D = 2.448

def ci95_to_std(ci95, n) -> np.ndarray          # scalar|(n,) -> (n,1), pure
def gaussian(n, ci95, rng) -> np.ndarray         # (n,2) zero-mean isotropic
def make_biased_gaussian(bias=(0., 0.)):         # -> closure (n, ci95, rng) -> (n,2)
def tstudent(n, ci95, rng):                      # raise NotImplementedError (stub)
# TODO(correlated): factory for joint 4D pos-vel sampling — left as hook
```
A "distribution" is just any `(n, ci95, rng) -> (n,2)` callable; `gaussian` is the
default. `make_biased_gaussian((bx,by))` returns a closure that adds the constant offset.
Handle `n == 0 -> np.empty((0,2))`.

**`tests/test_cns_distributions.py`:**
- `ci95_to_std`: scalar broadcasts to `(n,1)`; per-aircraft array preserved; factor 2.448.
- `gaussian`: shape `(n,2)`; with fixed seed mean error ≈ 0 over many draws; spread scales
  with `ci95` (spec test 1's draw-level claim + test 2 mechanism).
- `gaussian` `n=0` → empty `(0,2)`.
- **(spec test 9)** `make_biased_gaussian(bias)`: mean error over many draws ≈ `bias`.
- `tstudent` raises `NotImplementedError`.

---

## 2. `sensor.py`

**Purpose:** fresh noisy 1D snapshot of all aircraft from truth, each tick.

**FP shape (replaces mutable `Sensor` + `_init_arrays`):**
```python
FIELDS = ["lat","lon","alt","hdg","trk","gs","tas","vs","gseast","gsnorth","pos_acc","vel_acc"]

@dataclass(frozen=True)
class SensorState:
    n: int
    id: list
    lat: np.ndarray; lon: np.ndarray; ...; pos_acc: np.ndarray; vel_acc: np.ndarray

def measure(states, pos_ci95, vel_ci95, pos_dist=gaussian, vel_dist=gaussian, rng=...) -> SensorState
```
Pure: rebuilds a fresh `SensorState` from `states.ntraf` each call — **resize is implicit**,
no `_init_arrays` step. Internal pure helpers, mirroring legacy `noise_model`:
- `_position_noise(lat, lon, exy) -> (lat', lon')` with the `cos(lat)` pole guard and
  the `111_320.0` conversion (preserve legacy math exactly).
- `_velocity_components(gs, trk, vxy) -> (gsnorth, gseast)`.

Pass-through fields (`alt,hdg,trk,gs,tas,vs,id`) copied; `pos_acc/vel_acc` record the CI95
actually used (`np.broadcast_to(ci95, (n,))`).

**`tests/test_cns_sensor.py`:**
- **(spec test 1)** two `measure()` on identical truth → different `lat/lon` (fresh draw);
  mean over many draws ≈ truth.
- **(spec test 2)** larger `pos_ci95` → larger observed spread.
- Pass-through fields equal truth exactly; `gsnorth/gseast` reconstruct from `gs,trk` when
  vel noise is zero (`make_biased_gaussian((0,0))` w/ tiny ci, or monkeypatched zero dist).
- `pos_acc/vel_acc` equal the broadcast CI95.
- **(spec test 8, sensor part)** `ntraf` change → returned `SensorState.n` and array lengths track.
- Returned `SensorState` is frozen (rebind raises).

---

## 3. `reception_model.py`

**Purpose:** N×N reception matrix `P` + boolean refresh-mask sampling.

**FP shape (replaces mutable `ReceptionModel`):**
```python
@dataclass(frozen=True)
class ReceptionModel:
    default_prob: float
    P: np.ndarray            # (n,n), off-diag=default_prob, diag=1.0

def make_reception(default_prob) -> ReceptionModel        # validates [0,1], P empty
def ensure_size(rm, n) -> ReceptionModel                   # new rm, preserves [:m,:m]
def set_pair(rm, i, j, prob) -> ReceptionModel             # new rm w/ copied P
def sample_mask(rm, n, rng, force_full=False) -> (np.ndarray, ReceptionModel)
```
`sample_mask` returns `(mask, resized_rm)` so the grown `P` is threaded back (no hidden
mutation). `force_full` → all-True; else `rng.random((n,n)) <= P`, then `fill_diagonal(True)`.
`set_pair` copies `P` before writing (immutability).

**`tests/test_cns_reception.py`:**
- `make_reception` rejects prob outside `[0,1]`.
- `ensure_size`: `(0,0)→(n,n)`, off-diag `=default_prob`, diag `=1.0`; **(spec test 8)** grow
  preserves the existing sub-block, originals unchanged (purity).
- `sample_mask(force_full=True)` → all True incl. diagonal.
- `sample_mask` with `default_prob=0.0` → only diagonal True.
- **(spec test 7, P part)** `set_pair(i,j,0.0)` returns new rm; original `P` untouched.

---

## 4. `adsl_observation.py`

**Purpose:** N×N last-known store, one matrix per field; stale-keeps-previous.

**FP shape (replaces mutable `ADSLObservation`):**
```python
FIELDS = [...same 12...]

@dataclass(frozen=True)
class ADSLObservation:
    n: int
    id: list
    fields: dict[str, np.ndarray]   # name -> (n,n); dict avoids 12 declared fields

def empty_observation() -> ADSLObservation
def ensure_size(obs, n) -> ADSLObservation                 # new, preserves [:m,:m] per field
def update(obs, sensor, mask) -> ADSLObservation           # np.where(mask, row_j, cur)
def field(obs, name) -> np.ndarray                          # accessor
```
`update` is pure: for each field, `np.where(mask, sensor.<f>.reshape(1,-1), cur)` →
fresh dict, fresh `ADSLObservation`. Broadcasts target `j`'s value across observer rows `i`;
keeps previous where mask False (staleness). `id = list(sensor.id)`.

**`tests/test_cns_observation.py`:**
- **(spec test 3)** mask True for two observers of same `j` → `obs[i,j] == obs[k,j]` exactly
  (no comms noise).
- **(spec test 4)** mask `[i,j]=False` over several `update`s while sensor `j` moves →
  `obs[i,j]` unchanged (stale), other cells track.
- **(spec test 5)** diagonal `obs[i,i] == sensor[i]` when diagonal mask True.
- **(spec test 8)** `ensure_size` grow preserves sub-block; input `obs` unmutated.
- `update` returns a frozen value distinct from input.

---

## 5. `cns.py` (coordinator)

**Purpose:** single per-tick entry; binds sensor + reception + obs; exposes accessors.

**FP shape (replaces mutable `CNS` + `first_update_done` flag):**
```python
@dataclass(frozen=True)
class CNSState:
    sensor: SensorState
    reception: ReceptionModel
    obs: ADSLObservation
    pos_ci95; vel_ci95
    pos_dist; vel_dist; rng
    first_update_done: bool

def make_cns(pos_ci95, vel_ci95, reception_prob=1.0,
             pos_dist=gaussian, vel_dist=gaussian, seed=None) -> CNSState
def step(cns, states) -> CNSState        # the per-tick fn; returns NEW CNSState
def ownship_field(cns, field) -> np.ndarray   # 1D, == diagonal == sensor[i]
def adsl_field(cns, field) -> np.ndarray      # (n,n) row i = world seen by i
```
`step` composes the pure pieces:
```
sensor'   = measure(states, cns.pos_ci95, cns.vel_ci95, cns.pos_dist, cns.vel_dist, cns.rng)
mask, rm' = sample_mask(cns.reception, n, cns.rng, force_full=not cns.first_update_done)
obs'      = update(cns.obs, sensor', mask)
return replace(cns, sensor=sensor', reception=rm', obs=obs', first_update_done=True)
```
Changing accuracy between ticks = caller rebuilds via `replace(cns, pos_ci95=...)` (immutable).
Integration note for CD/CR/CRR: after `cns = step(cns, traf)`, set
`traf.sensor = cns.sensor` (1D) and `traf.adsl = cns.obs` (N×N); ownship `i` uses row `i`.

**`tests/test_cns.py` (end-to-end, remaining spec tests):**
- **(spec test 6)** `reception_prob=0.0`: first `step` fills all cells (no 0/NaN off-diag);
  second `step` leaves off-diagonal stale, diagonal fresh.
- **(spec test 5)** every tick `adsl_field(... )[i,i] == ownship_field(...)[i]`, any prob.
- **(spec test 2)** raising `pos_ci95` (via `replace`) between steps widens observed spread.
- **(spec test 7)** `set_pair(i,j,0)` + `set_pair(j,i,1)` then many steps → `obs[i,j]` stale,
  `obs[j,i]` tracks (asymmetry end-to-end).
- **(spec test 3)** two observers both receiving `j` hold identical `obs[*,j]`.
- **(spec test 8)** add an aircraft (bigger `states`) → all structures grow to `(N+1,N+1)`,
  prior sub-block preserved.
- `CNSState` frozen; `step` returns a new instance (input unchanged).

---

## 6. `__init__.py`

Export the FP surface (no classes-as-constructors):
```python
from .cns import make_cns, step, ownship_field, adsl_field, CNSState
from .distributions import gaussian, make_biased_gaussian
from .sensor import measure, SensorState
from .reception_model import make_reception, sample_mask, set_pair, ReceptionModel
from .adsl_observation import update, empty_observation, ADSLObservation
```

---

## Coverage of spec §7 tests

| Spec test | Where |
|---|---|
| 1 sensor jitter | test_cns_sensor (+ distributions) |
| 2 time-varying accuracy | test_cns_sensor + test_cns |
| 3 no comms noise / identical receivers | test_cns_observation + test_cns |
| 4 staleness | test_cns_observation |
| 5 diagonal | test_cns_observation + test_cns |
| 6 first update full | test_cns |
| 7 asymmetry | test_cns_reception + test_cns |
| 8 resize | each module's test |
| 9 biased gaussian | test_cns_distributions |

## Migration notes (unchanged from spec §8)
- Receive-side noise removed — noise is sensor-only.
- Legacy scalar `reception_prob` → `ReceptionModel.default_prob`.
- 1D legacy message arrays → diagonal/rows of N×N obs; call sites index `[i,j]`.
- No geometry-`P`, t-student, or correlated sampling this pass — keep TODO hooks.
- `cns_old/` retained until call sites migrate.
```
