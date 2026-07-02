'''Pairwise horizontal conflict — analysis figures.

Two analyses, both for the FTR recovery strategies ``double_criteria`` (FTR) and
``probabilistic`` (Prob FTR), sharing one scenario:

1. Aggregate avoidance comparison — ``N_RUNS`` parallel runs per strategy
   (independent seeds), pooled to ``N_RUNS × PAIR_WIDTH × PAIR_HEIGHT`` pairs;
   plots the average avoidance flag over time, one line per strategy, so FTR and
   Prob FTR can be compared directly (plot_avoidance_aggregate).

2. Detailed per-pair figures — ONE single run per strategy; for each selected
   pair a stacked figure of actual distance, projected CPA distance (truth vs
   observed) and avoidance status, with the CPA marked (plot_pair_detail). The
   same run also yields an ownship-centric trajectory plot (plot_trajectories).

All figures are written to figures/analysis/.

Run directly::

    python analysis/pairwise_hor_conflict_analysis.py            # default style
    python analysis/pairwise_hor_conflict_analysis.py --latex    # LaTeX-friendly
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed

from runners.stochastic_pairwise_hor_conflict import run_single
from plot_utils import (
    plot_avoidance_aggregate, plot_pair_detail, plot_pair_trajectory,
    plot_trajectories, set_latex_style,
)

# Publication / LaTeX-friendly typography (serif CM fonts, inward ticks, 300 dpi).
# Pass --latex on the command line to enable; off by default.
LATEX_STYLE = "--latex" in sys.argv
set_latex_style(LATEX_STYLE)

# ── Shared scenario ─────────────────────────────────────────────────────────────
PAIR_WIDTH    = 10     # 10 × 10 = 100 pairs per run
PAIR_HEIGHT   = 10
RPZ           = 50.0
HPZ           = 50.0
DTLOOKAHEAD   = 120.0
INIT_SPD_OWN  = 10.2889 # m/s
INIT_SPD_INT  = 10.2889 # m/s
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

GAMMA        = 0.999   # confidence threshold for the probabilistic (Prob FTR) strategy

# FTR strategies only.
STRATEGIES = ("double_criteria", "probabilistic")

# Aggregate analysis: 1 runs/strategy → 100 pooled pairs.
N_RUNS    = 1
# Use 4 workers normally, but 100 on a big (>100-core) box.
N_JOBS    = 100 if (os.cpu_count() or 1) > 100 else 4
BASE_SEED = 42

# Detailed analysis: single run/strategy at this seed; these pairs get a figure.
SEED  = 44
PAIRS = (12, 94)

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(_ROOT, "figures", "analysis")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Scenario kwargs shared by both analyses (seed supplied per call).
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
    prob_threshold=GAMMA,
)

# ── 1. Aggregate avoidance comparison (FTR vs Prob FTR) ──────────────────────────
print("== Aggregate avoidance comparison ==")
results_by_label = {}
for label in STRATEGIES:
    runs = Parallel(n_jobs=N_JOBS)(
        delayed(run_single)(crr=label, seed=BASE_SEED + rep, **RUN_KWARGS)
        for rep in range(N_RUNS)
    )
    results_by_label[label] = runs

agg_path = plot_avoidance_aggregate(results_by_label, FIGURE_DIR, select="all")

for label, runs in results_by_label.items():
    n_pairs = sum(r.env.nb_pair for r in runs)
    n_los   = sum(r.n_los for r in runs)
    ipr     = 1.0 - n_los / float(n_pairs)
    print(f"  IPR  {label:<16} : {ipr:.4f}  (LOS={n_los}/{n_pairs}, runs={len(runs)})")
print(f"  → {agg_path}")

# ── 2. Detailed per-pair figures ─────────────────────────────────────────────────
print("== Detailed per-pair figures ==")
for label in STRATEGIES:
    res = run_single(crr=label, seed=SEED, **RUN_KWARGS)
    print(f"  {label:<16} : IPR = {res.ipr:.4f}")
    for pair in PAIRS:
        path = plot_pair_detail(res, FIGURE_DIR, pair, label)
        print(f"    pair {pair:03d} "
              f"({res.env.ownship_ids[pair]} ↔ {res.env.intruder_ids[pair]}), "
              f"actual CPA = {res.min_dist[pair]:.1f} m  → {path}")
        traj_pair_path = plot_pair_trajectory(res, FIGURE_DIR, pair, label)
        print(f"      trajectory  → {traj_pair_path}")
    # Ownship-centric trajectories for the whole run (avoiding segments in dark).
    traj_path = plot_trajectories(res, FIGURE_DIR, label)
    print(f"    trajectories  → {traj_path}")
