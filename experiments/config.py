'''Shared simulation parameters — matches the journal paper setup.

All experiment scripts import from here so parameter changes propagate everywhere.
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Simulation environment ────────────────────────────────────────────────────
PAIR_WIDTH    = 100      # 100 × 100 = 10 000 independent pairs per run
PAIR_HEIGHT   = 100
RPZ           = 50.0     # m  protected-zone radius
HPZ           = 50.0     # m
DTLOOKAHEAD   = 121.0    # s  (≈ 120 s tactical look-ahead horizon)
SPEED         = 20.0     # kts — ownship and intruder (identical)
AIRCRAFT_TYPE = 'M600'
TMAX          = 600.0    # s  hard run-time limit per simulation
DONE_TIMEOUT  = 10.0     # s  post-clearance settle window before early stop

# ── Monte Carlo ────────────────────────────────────────────────────────────────
# 10 000 pairs per run is already statistically stable; n_runs=1 matches the paper.
# Increase for explicit seed-to-seed variance estimates.
N_RUNS        = 1
import multiprocessing as _mp
_ncpu = _mp.cpu_count()
N_JOBS        = 100 if _ncpu > 100 else (4 if _ncpu > 4 else 1)
BASE_SEED     = 42

# ── Probabilistic recovery ─────────────────────────────────────────────────────
KTHETA        = 256      # Monte Carlo samples for exceedance probability
DEFAULT_GAMMA = 0.999    # default confidence threshold

# ── Independent variables ─────────────────────────────────────────────────────
CROSSING_ANGLES = list(range(2, 181, 2))   # 2, 4, …, 180 degrees (90 values)

UNCERTAINTY_LEVELS = [
    dict(pos_ci95=3,  vel_ci95=1, label='pos3_vel1',  title='pos=3 m, vel=1 m/s'),
    dict(pos_ci95=3,  vel_ci95=3, label='pos3_vel3',  title='pos=3 m, vel=3 m/s'),
    dict(pos_ci95=10, vel_ci95=1, label='pos10_vel1', title='pos=10 m, vel=1 m/s'),
    dict(pos_ci95=10, vel_ci95=3, label='pos10_vel3', title='pos=10 m, vel=3 m/s'),
]

RECOVERY_METHODS = ['cpa', 'double_criteria', 'probabilistic']
METHOD_LABELS    = {'cpa': 'Past-CPA', 'double_criteria': 'FTR', 'probabilistic': 'Probabilistic'}

GAMMA_VALUES = [0.999, 0.99, 0.9, 0.75, 0.5]

# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
