'''Parallel stochastic pairwise horizontal conflict — aggregated avoidance.

Like :mod:`tests.test_stochastic_pairwise_hor_conflict_sim`, but runs the
scenario ``N_RUNS`` times in parallel (``N_JOBS`` workers, independent seeds)
for the two FTR recovery strategies only:

  * ``double_criteria`` — FTR two-criteria rule.
  * ``probabilistic``   — probabilistic FTR rule (Prob FTR).

Each run carries ``PAIR_WIDTH × PAIR_HEIGHT`` pairs, so ``N_RUNS`` runs pool to
``N_RUNS × PAIR_WIDTH × PAIR_HEIGHT`` pairs. With the defaults below that is
25 runs × (10 × 10) = 2500 pairs per strategy.

The only figure produced is the aggregated average avoidance flag across all
pooled pairs, one line per strategy:
  stochastic_pairwise_hor_conflict_avoidance_aggregate.png

Run directly::

    python tests/test_parallel_stochastic_pairwise_hor_conflict_sim.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed

from runners.stochastic_pairwise_hor_conflict import run_single
from plot_utils import plot_avoidance_aggregate

# ── Parameters ────────────────────────────────────────────────────────────────
N_RUNS        = 25     # parallel runs per strategy (independent seeds)
# Use 4 workers normally, but 100 on a big (>100-core) box.
N_JOBS        = 100 if (os.cpu_count() or 1) > 100 else 4
BASE_SEED     = 42

PAIR_WIDTH    = 10     # 10 × 10 = 100 pairs per run → 1000 pairs over 10 runs
PAIR_HEIGHT   = 10
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
DELTA_LAT_LON = 0.1

POS_CI95      = 10.0   # m
VEL_CI95      = 1.0    # m/s
RECEPTION_PROB = 1.0

TMAX         = DTLOOKAHEAD * 4
DONE_TIMEOUT = 30.0

# FTR strategies only.
STRATEGIES = ("double_criteria", "probabilistic")

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(_ROOT, "figures", "tests")
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Run ───────────────────────────────────────────────────────────────────────
# Shared scenario; only the recovery strategy and seed differ between runs.
RUN_KWARGS = dict(
    pair_width=PAIR_WIDTH, pair_height=PAIR_HEIGHT,
    rpz=RPZ, hpz=HPZ, dtlookahead=DTLOOKAHEAD,
    init_speed_ownship=INIT_SPD_OWN, init_speed_intruder=INIT_SPD_INT,
    aircraft_type=AIRCRAFT_TYPE, dpsi=DPSI,
    pos_ci95=POS_CI95, vel_ci95=VEL_CI95, reception_prob=RECEPTION_PROB,
    start_lat=START_LAT, start_lon=START_LON, delta_lat_lon=DELTA_LAT_LON,
    tmax=TMAX, done_timeout=DONE_TIMEOUT,
    resofach=1.0, recovery_resofach=1.05,
    simdt_factor=SIMDT_FACTOR,
    record_history=True,
)

results_by_label = {}
for label in STRATEGIES:
    runs = Parallel(n_jobs=N_JOBS)(
        delayed(run_single)(crr=label, seed=BASE_SEED + rep, **RUN_KWARGS)
        for rep in range(N_RUNS)
    )
    results_by_label[label] = runs

# ── Figure ────────────────────────────────────────────────────────────────────
# Single aggregated avoidance-flag plot, pooling every pair across all runs.
path = plot_avoidance_aggregate(results_by_label, FIGURE_DIR, select="all")

# ── Summary ───────────────────────────────────────────────────────────────────
for label, runs in results_by_label.items():
    n_pairs = sum(r.env.nb_pair for r in runs)
    n_los   = sum(r.n_los for r in runs)
    ipr     = 1.0 - n_los / float(n_pairs)
    print(f"IPR  {label:<16} : {ipr:.4f}  (LOS={n_los}/{n_pairs}, runs={len(runs)})")
print(f"Figure  → {path}")
