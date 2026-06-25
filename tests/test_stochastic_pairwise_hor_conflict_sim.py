'''Stochastic pairwise horizontal conflict — simulation test with figures.

Runs the same scenario under three recovery strategies (``cpa``,
``double_criteria``, ``probabilistic``) via
:func:`runners.stochastic_pairwise_hor_conflict.run_single`, then renders
figures through :mod:`plot_utils`.

For each strategy ``<label>`` four per-run figures are saved to figures/tests/:
  stochastic_pairwise_hor_conflict_distances_<label>.png
  stochastic_pairwise_hor_conflict_gs_hdg_<label>.png
  stochastic_pairwise_hor_conflict_avoidance_<label>.png
  stochastic_pairwise_hor_conflict_trajectories_<label>.png
plus a single cross-strategy comparison figure:
  stochastic_pairwise_hor_conflict_avoidance_compare.png

Run directly::

    python tests/test_stochastic_pairwise_hor_conflict_sim.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runners.stochastic_pairwise_hor_conflict import run_single
from plot_utils import plot_run, plot_avoidance_compare

# ── Parameters ────────────────────────────────────────────────────────────────
PAIR_WIDTH    = 5
PAIR_HEIGHT   = 5
RPZ           = 50.0
HPZ           = 50.0
DTLOOKAHEAD   = 121.0
INIT_SPD_OWN  = 15.0
INIT_SPD_INT  = 15.0
DPSI          = 90
AIRCRAFT_TYPE = "M600"
SIMDT_FACTOR  = 1
START_LAT     = 52.0
START_LON     = 4.0

# At 52°N, 0.1° ≈ 11 km (lat) / 6.9 km (lon) — 4× safety margin in both axes.
DELTA_LAT_LON = 0.1

POS_CI95      = 10.0   # m
VEL_CI95      = 1.0    # m/s
RECEPTION_PROB = 1.0
SEED          = 44

TMAX         = DTLOOKAHEAD * 4
DONE_TIMEOUT = 30.0

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(_ROOT, "figures", "tests")
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Run ───────────────────────────────────────────────────────────────────────
# Shared scenario; only the recovery strategy differs between runs.
RUN_KWARGS = dict(
    pair_width=PAIR_WIDTH, pair_height=PAIR_HEIGHT,
    rpz=RPZ, hpz=HPZ, dtlookahead=DTLOOKAHEAD,
    init_speed_ownship=INIT_SPD_OWN, init_speed_intruder=INIT_SPD_INT,
    aircraft_type=AIRCRAFT_TYPE, dpsi=DPSI,
    pos_ci95=POS_CI95, vel_ci95=VEL_CI95, reception_prob=RECEPTION_PROB,
    start_lat=START_LAT, start_lon=START_LON, delta_lat_lon=DELTA_LAT_LON,
    tmax=TMAX, done_timeout=DONE_TIMEOUT,
    resofach=1.0, recovery_resofach=1.05,
    simdt_factor=SIMDT_FACTOR, seed=SEED,
    record_history=True,
)

results = {
    "cpa":             run_single(crr="cpa", **RUN_KWARGS),
    "double_criteria": run_single(crr="double_criteria", **RUN_KWARGS),
    "probabilistic":   run_single(crr="probabilistic", **RUN_KWARGS),
}

# ── Figures ───────────────────────────────────────────────────────────────────
# Per-run figures (distances, gs/hdg, avoidance, trajectories) for every strategy.
for label, res in results.items():
    plot_run(res, FIGURE_DIR, label)

# Cross-strategy avoidance comparison.
plot_avoidance_compare(results, FIGURE_DIR)

# ── Summary ───────────────────────────────────────────────────────────────────
nb_pair = next(iter(results.values())).env.nb_pair
for label, res in results.items():
    print(f"IPR  {label:<16} : {res.ipr:.4f}  (LOS={res.n_los}/{nb_pair})")
print(f"Figures  → {FIGURE_DIR}")
