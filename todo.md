# todo

## question on abstraction of cdr

What if someone wants to do the cd, cr, and reso in one algorithm, for instnace using reinforcement learning model. Can the existing structure accomodate it?

conf    = detect(obs, obs, rpz, hpz, dtlookahead)
conf_gt = detect(bs.traf, bs.traf, rpz, hpz, dtlookahead)

newtrack, newgs, newvs, alt = mvp.resolve(conf, obs, obs, cfg)
recovery_state, _ = resumenav_double_criteria(
    recovery_state, conf, obs, obs, active,
    id2idx=_id2idx, recover=_noop_recover,
)
action = (newtrack, newgs, newvs, alt, list(recovery_state.resopairs))

maybe those blocks above should be made into one function

cdarr(obs, cns, rpz, hpz, dtlookahead, detect, resolve, recovery) etc etc

then the cdarr can be replaced with something like:

rl(obs, cns, rpz, hpz, dtlookahead, detect, resolve, recovery)

what about if someone wants to "not separate the avoidance and recovery"

to what level this abstraction can be done at the moment?

### answer

**Short version: yes, and you're closer than the snippet suggests — because the
simulation only ever sees one thing, the `action` tuple.** The env contract is
`step(env, action)` with `action = (newtrack, newgs, newvs, alt, resopairs)`.
`_apply_action` (envs/pairwise_hor_conflict.py:162) never sees `conf`,
`ConflictState`, or `recovery_state` — it only reads that 5-tuple. So whatever
*produces* the tuple is already a black box to BlueSky. cd/cr/crr is just the
current *implementation* of "obs → action", not a structural requirement.

**1. What couples the three stages today.** In the loop
(runners/stochastic_pairwise_hor_conflict.py:297-304) the glue is two things:
- the `ConflictState` value — `cd` emits it, both `cr` and `crr` consume it;
- the threaded `recovery_state` — `crr`'s own book-keeping, passed in/out.
Everything else flows from the single shared `obs` (`_as_obs`). cd/cr/crr are
*already* injectable params of `run_single` (`cd=`, `cr=`, `crr=`), so the
project already treats the stages as swappable.

**2. The `cdarr(...)` wrapper is a pure refactor you can do right now.** Collapse
lines 297-304 into one policy with the interface `policy(obs, state, cfg) ->
(action, new_state)`:

```python
def cdarr(obs, state, cfg, active, *, cd, cr, crr):
    conf = cd(obs, obs, rpz, hpz, dtlookahead)
    newtrack, newgs, newvs, alt = cr(conf, obs, obs, cfg)
    state, _ = crr(state, conf, obs, obs, active)
    action = (newtrack, newgs, newvs, alt, list(state.resopairs))
    return action, state
```

The loop becomes `action, recovery_state = policy(obs, recovery_state, ...)`.
No env change, no behaviour change.

**3. The RL drop-in is then a different implementation of the same interface:**

```python
def rl_policy(obs, state, cfg, active):
    action = model(featurize(obs))   # emits (trk, gs, vs, alt, resopairs) directly
    return action, state
```

It is *not* obligated to build a `ConflictState`, call `cd`, or run MVP. It only
has to emit the action tuple. So fusing CD+CR+reso into one learned model is
fully accommodated by the current env boundary.

**4. "Not separating avoidance and recovery."** At the env boundary they are
*already* merged — both arrive inside the one action tuple; the separation only
exists *inside* the default policy (two calls). A fused model just emits the
merged decision directly. The one real wrinkle: the env encodes "avoiding vs
reverted-to-nav" via `resopairs` (a set of id-pairs) → `avoidance_mask`
(envs/pairwise_hor_conflict.py:147). A fused per-aircraft policy more naturally
wants a *per-aircraft* "commanded velocity + active flag" instead of a pair-set.
So full fusion wants the action contract to evolve from
`(trk, gs, vs, alt, resopairs)` to a per-aircraft velocity/active representation
— a small `_apply_action`/`avoidance_mask` change, not a structural one.

**5. To what level it can be done at the moment.**
- *Today, zero env changes:* swap the `cd`/`cr`/`crr` params for a single
  injected `policy(obs, state) -> (action, state)` that internally does anything
  (rules or RL), as long as it returns the existing action tuple. ~80% there.
- *Friction / not yet abstracted:*
  - `obs` is a bluesky-traffic-shaped `SimpleNamespace` (`_as_obs`). RL wants a
    flat feature vector → you need a featurizer (obs already carries
    lat/lon/trk/gs/perf and `adsl.pos_acc/vel_acc`, so the data is there).
  - The threaded state is `RecoveryState`, recovery-specific. A generic policy
    needs its own opaque `state` — generalize the threaded object from
    "recovery state" to "policy state".
  - `ResolutionConfig` / `recovery_resofach` / `prob_threshold` / `Ktheta` are
    MVP/FTR-specific knobs; a learned policy ignores them. Fine, but it shows the
    cfg surface is per-strategy, not generic — a unified policy would take its
    own config bag.
  - The `resopairs` encoding (see point 4) is the only thing blocking a *fully*
    fused avoidance+recovery action.

**Recommendation:** introduce a `Policy` protocol `policy(obs, state) ->
(action, state)`; make the default an explicit composition of cd→cr→crr (the
`cdarr` wrapper); for full fusion, generalize (a) the threaded state to an opaque
policy state and (b) the action's `resopairs` field to a per-aircraft
active/velocity representation. Both are local changes — the action-as-sole-env-
contract design already does the heavy lifting.