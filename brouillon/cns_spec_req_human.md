# CNS Model — Specification (Human)

*Communication, Navigation & Surveillance model for CDaRR. Translates `cns_architecture_idea.md` plus the design decisions agreed in discussion.*

## 1. Purpose

BlueSky gives us **ground-truth** traffic state (`lat`, `lon`, `gs`, `trk`, …). Real aircraft never see ground truth — they see a *noisy, delayed, sometimes-missing* version of it. The CNS model sits between ground truth and the conflict-detection/resolution/recovery (CD/CR/CRR) algorithms and produces that realistic, degraded view.

It has **two layers**:

| Layer | What it is | Shape |
|-------|-----------|-------|
| **Sensor** | Each aircraft's measurement of **its own** state. `sensor = ground_truth + noise`. | 1D, length N |
| **ADS-L** | The surveillance picture: what each aircraft **receives about every other** aircraft. `adsl = sensor + reception probability` (no extra noise). | **N×N** |

## 2. Sensor layer

- For every aircraft `j`: `sensor[j] = truth[j] + ε`.
- The error `ε` is **re-drawn every tick** (fresh random draw each time step — "jitter," not a fixed offset).
- Because it's re-drawn each tick, the **noise level can change at any time step** (e.g. simulate GPS degradation): the accuracy is an input to each draw, not fixed at startup.
- Accuracy is given as a **95% confidence interval**: `pos_acc` (position) and `vel_acc` (velocity). These are **per-aircraft and per-tick** settable.
- The error distribution is **pluggable**:
  - `gaussian` — zero-mean normal.
  - `biased-gaussian` — normal with a non-zero mean (a built-in offset of the *distribution*; nothing to do with time).
  - `t-student`, correlated position–velocity sampling — future, leave hooks.
  - For now implement **gaussian** and **biased-gaussian**.

## 3. ADS-L layer (N×N)

- `obs[i][j]` = what observer **i** currently knows about target **j** = the **last-received** value of `sensor[j]`.
- **No communication noise is added.** The received value is *exactly* `j`'s sensor value, so two different observers who both receive `j` hold the identical number.
- The only per-pair effect is **reception**: each tick, with probability `P[i][j]`, `obs[i][j]` refreshes to the current `sensor[j]`; otherwise it keeps its **last-known (stale)** value.
- A stale `obs[i][j]` is therefore "old jitter on old truth" — wrong both because the target moved and because the old draw was random.
- **Diagonal:** `obs[i][i] = sensor[i]`, always fresh (`P = 1`). This *is* the ownship's own view of itself.
- **First update is full** — all pairs are seeded once with no packet loss, so there are no empty/NaN cells.

## 4. Reception matrix P (N×N)

- `P[i][j]` = probability that observer `i` receives target `j`'s message, **from the ownship's perspective**.
- **Asymmetric is allowed:** `P[i][j] ≠ P[j][i]` (A→B = 0.95 while C→B = 0.92).
- **For now:** a single value applied to all off-diagonal pairs (same everywhere); diagonal = 1.0.
- **TODO (later):** compute `P` from **geometry** (range-dependent reception). This is a code TODO, not for this pass.

## 5. Integration with traffic & algorithms

- Attach to the traffic object as `traffic.sensor` (1D) and `traffic.adsl` (N×N).
- `pos_acc` / `vel_acc` ride along on the observation so downstream code can read the advertised accuracy.
- CD/CR/CRR consume the model as: ownship `i` uses **`ownship.sensor`** for itself (`obs[i][i]`) and **`intruder.adsl`** for each intruder `j` (`obs[i][j]`). Concretely, ownship `i` operates on **row `i`** of the N×N picture.

## 6. Lifecycle

- When aircraft are created/deleted, all structures resize: `sensor` (1D) and `obs`, `P` (N×N), preserving existing values.
- Note: N×N storage grows quadratically with aircraft count — fine for moderate N.

## 7. Out of scope (this pass)

- t-student distribution, correlated pos–velocity sampling (leave interface hooks).
- Geometry-derived `P` (TODO in code).
- Anything in the legacy code that adds noise on the *receive* side (removed — noise lives only at the sensor).
