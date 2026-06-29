'''Experiment 3 — Noise model comparison under heterogeneous speed encounters.

Independent variables
---------------------
* Crossing angle  : 2°, 4°, …, 180°  (90 values)
* Uncertainty     : 4 pos_ci95 × vel_ci95 combinations (same as Exp 1)
* Noise model     : Normal Gaussian / Latency bias / Mixture Gaussian

Fixed
-----
* Speed           : Uniform(10, 30) kts, drawn independently per aircraft per run
* Recovery method : probabilistic (γ = DEFAULT_GAMMA = 0.999)
* Pairs per run   : 10 × 10 = 100
* Runs per angle  : 10 000   →   100 × 10 000 = 1 000 000 pairs per condition

Parallelism strategy
--------------------
The 10 000 runs per crossing angle are handled by run_parallel (n_jobs=N_JOBS).
The crossing angle loop is sequential to avoid nested joblib parallelism.
Total calls to run_single per (uncertainty, noise model) combination:
  90 angles × 10 000 runs = 900 000 simulations of 100 pairs each.

Results saved to experiments/results/exp3.npz. Run directly:

    python experiments/exp3-noise-model-heterogeneous-speed.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import (
    RPZ, HPZ, DTLOOKAHEAD, AIRCRAFT_TYPE,
    TMAX, DONE_TIMEOUT, KTHETA, DEFAULT_GAMMA,
    CROSSING_ANGLES, UNCERTAINTY_LEVELS,
    N_JOBS, BASE_SEED, RESULTS_DIR,
)
from runners.stochastic_pairwise_hor_conflict_heterogeneous_speed import run_parallel
from sim_models.cns.distributions import make_mixture_gaussian

# ── Exp 3 specific parameters ─────────────────────────────────────────────────
PAIR_WIDTH  = 10
PAIR_HEIGHT = 10      # 100 pairs per run
N_RUNS      = 10_000  # 100 × 10 000 = 1 000 000 pairs per (angle, unc, model)

SPEED_MIN   = 10.0    # kts  — uniform draw lower bound
SPEED_MAX   = 30.0    # kts  — uniform draw upper bound

# ADS-B v2 mean latency (used by the latency model)
LATENCY_S   = 0.0661  # s

# Mixture Gaussian parameters (tail 3× wider, 10% weight)
TAIL_RATIO  = 3.0
TAIL_WEIGHT = 0.10

# ── Noise model definitions ───────────────────────────────────────────────────
# Each entry: (label, pos_dist, latency_s)
# pos_dist=None → default isotropic Gaussian
NOISE_MODELS = [
    ('normal',  None,                                              0.0),
    ('latency', None,                                              LATENCY_S),
    ('mixture', make_mixture_gaussian(TAIL_RATIO, TAIL_WEIGHT),   0.0),
]
NOISE_LABELS = [m[0] for m in NOISE_MODELS]

# ── Storage arrays ────────────────────────────────────────────────────────────
n_unc    = len(UNCERTAINTY_LEVELS)
n_models = len(NOISE_MODELS)
n_angles = len(CROSSING_ANGLES)

# shape: (n_unc, n_models, n_angles)
ipr_arr        = np.full((n_unc, n_models, n_angles), np.nan)
median_dcpa_arr = np.full((n_unc, n_models, n_angles), np.nan)

# ── Common kwargs shared across all runs ──────────────────────────────────────
COMMON = dict(
    pair_width=PAIR_WIDTH, pair_height=PAIR_HEIGHT,
    rpz=RPZ, hpz=HPZ, dtlookahead=DTLOOKAHEAD,
    speed_min=SPEED_MIN, speed_max=SPEED_MAX,
    aircraft_type=AIRCRAFT_TYPE,
    tmax=TMAX, done_timeout=DONE_TIMEOUT,
    crr='probabilistic', Ktheta=KTHETA, prob_threshold=DEFAULT_GAMMA,
    reception_prob=1.0,
)

# ── Main sweep ────────────────────────────────────────────────────────────────
for ui, unc in enumerate(UNCERTAINTY_LEVELS):
    for mi, (model_label, pos_dist, latency_s) in enumerate(NOISE_MODELS):
        label = f'{unc["label"]} / {model_label}'
        print(f'\nRunning: {label}', flush=True)
        print(f'  {N_RUNS} runs × {PAIR_WIDTH*PAIR_HEIGHT} pairs = '
              f'{N_RUNS * PAIR_WIDTH * PAIR_HEIGHT:,} pairs per angle', flush=True)

        for ai, angle in enumerate(CROSSING_ANGLES):
            stats = run_parallel(
                n_runs=N_RUNS, n_jobs=N_JOBS,
                base_seed=BASE_SEED + ai,
                **COMMON,
                dpsi=float(angle),
                pos_ci95=unc['pos_ci95'],
                vel_ci95=unc['vel_ci95'],
                pos_dist=pos_dist,
                latency_s=latency_s,
            )
            ipr_arr[ui, mi, ai] = stats['overall_ipr']
            # run_parallel does not expose per-pair min_dist; record overall IPR only.
            # For median dCPA, use run_single directly if needed.

            if (ai + 1) % 10 == 0 or ai == n_angles - 1:
                print(f'  angle {angle:3d}°  IPR={stats["overall_ipr"]:.4f}', flush=True)

        print(f'  mean IPR = {ipr_arr[ui, mi, :].mean():.4f}', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, 'exp3.npz')
np.savez(
    out_path,
    crossing_angles=np.array(CROSSING_ANGLES),
    uncertainty_labels=np.array([u['label'] for u in UNCERTAINTY_LEVELS]),
    uncertainty_titles=np.array([u['title'] for u in UNCERTAINTY_LEVELS]),
    noise_labels=np.array(NOISE_LABELS),
    ipr=ipr_arr,
    speed_min=SPEED_MIN,
    speed_max=SPEED_MAX,
    n_runs=N_RUNS,
    pair_width=PAIR_WIDTH,
    pair_height=PAIR_HEIGHT,
)
print(f'\nSaved → {out_path}')

# Quick summary table
print(f'\n{"Uncertainty":<20} {"Model":<10} {"Mean IPR":>9} {"Min IPR":>8}')
print('-' * 51)
for ui, unc in enumerate(UNCERTAINTY_LEVELS):
    for mi, (model_label, _, _) in enumerate(NOISE_MODELS):
        row = ipr_arr[ui, mi, :]
        print(f'{unc["label"]:<20} {model_label:<10} '
              f'{row.mean():>9.4f} {row.min():>8.4f}')
