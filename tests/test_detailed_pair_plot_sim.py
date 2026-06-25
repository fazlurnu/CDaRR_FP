'''Detailed single-run, single-pair plots (todo.md "## detailed plot").

Runs ONE stochastic pairwise-horizontal-conflict simulation per FTR strategy
(``double_criteria`` and ``probabilistic``) and, for each selected pair, renders
one stacked detail figure with three vertically-stacked panels sharing a time
axis (top → bottom):

  1. actual ownship–intruder distance
  2. projected distance at CPA — ground truth vs observed
  3. avoidance status

A vertical dashed line marks the actual closest point of approach on every panel.

One file per (strategy ``<label>``, pair ``<pp>``) in figures/tests/:
  stochastic_pairwise_hor_conflict_pair<pp>_detail_<label>.png

Run directly::

    python tests/test_detailed_pair_plot_sim.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runners.stochastic_pairwise_hor_conflict import run_single
from plot_utils import plot_pair_detail

# ── Parameters ────────────────────────────────────────────────────────────────
PAIR_WIDTH    = 10     # 10 × 10 = 100 pairs in the single run
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
SEED          = 44

TMAX         = DTLOOKAHEAD * 4
DONE_TIMEOUT = 30.0

# Pairs to render a detail figure for.
PAIRS = (12, 24)

# FTR strategies only.
STRATEGIES = ("double_criteria", "probabilistic")

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(_ROOT, "figures", "tests")
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Run ───────────────────────────────────────────────────────────────────────
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

for label in STRATEGIES:
    res = run_single(crr=label, **RUN_KWARGS)
    print(f"{label:<16} : IPR = {res.ipr:.4f}")
    for pair in PAIRS:
        path = plot_pair_detail(res, FIGURE_DIR, pair, label)
        print(f"    pair {pair:03d} "
              f"({res.env.ownship_ids[pair]} ↔ {res.env.intruder_ids[pair]}), "
              f"actual CPA = {res.min_dist[pair]:.1f} m  → {path}")
