'''Experiment 1 — IPR and median dCPA vs crossing angle.

Reproduces the main result from the journal paper: sweeps crossing angle from 2°
to 180° in 2° steps for all four uncertainty levels and three recovery methods
(Past-CPA, FTR, Probabilistic FTR with γ=0.999). Each configuration runs
10 000 independent pairwise encounters (pair_width=100, pair_height=100).

Dependent variables collected per configuration:
  - IPR   : intrusion prevention rate
  - median dCPA : median closest-point-of-approach distance across pairs [m]

Results are saved to experiments/results/exp1.npz. Run directly:

    python experiments/exp1-crossing-angle.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from joblib import Parallel, delayed

from config import (
    PAIR_WIDTH, PAIR_HEIGHT, RPZ, HPZ, DTLOOKAHEAD, SPEED, AIRCRAFT_TYPE,
    TMAX, DONE_TIMEOUT, KTHETA, DEFAULT_GAMMA,
    CROSSING_ANGLES, UNCERTAINTY_LEVELS, RECOVERY_METHODS, METHOD_LABELS,
    N_RUNS, N_JOBS, BASE_SEED, RESULTS_DIR,
)
from runners.stochastic_pairwise_hor_conflict import run_parallel

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

n_unc     = len(UNCERTAINTY_LEVELS)
n_methods = len(RECOVERY_METHODS)
n_angles  = len(CROSSING_ANGLES)

ipr_arr        = np.full((n_unc, n_methods, n_angles), np.nan)
median_dcpa_arr = np.full((n_unc, n_methods, n_angles), np.nan)

for ui, unc in enumerate(UNCERTAINTY_LEVELS):
    for mi, method in enumerate(RECOVERY_METHODS):
        label = f'{unc["label"]} / {METHOD_LABELS[method]}'
        print(f'Running: {label} ...', flush=True)

        for ai, angle in enumerate(CROSSING_ANGLES):
            res = run_parallel(
                n_runs=N_RUNS, n_jobs=N_JOBS,
                base_seed=BASE_SEED,
                pair_width=PAIR_WIDTH, pair_height=PAIR_HEIGHT,
                rpz=RPZ, hpz=HPZ, dtlookahead=DTLOOKAHEAD,
                init_speed_ownship=SPEED, init_speed_intruder=SPEED,
                aircraft_type=AIRCRAFT_TYPE, dpsi=float(angle),
                pos_ci95=unc['pos_ci95'], vel_ci95=unc['vel_ci95'],
                reception_prob=1.0,
                tmax=TMAX, done_timeout=DONE_TIMEOUT,
                crr=method, Ktheta=KTHETA, prob_threshold=DEFAULT_GAMMA,
            )
            ipr_arr[ui, mi, ai]        = res['overall_ipr']
            median_dcpa_arr[ui, mi, ai] = float(np.median(res['worst_cpa']))

        mean_ipr = ipr_arr[ui, mi, :].mean()
        print(f'  done — mean IPR = {mean_ipr:.4f}', flush=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, 'exp1.npz')
np.savez(
    out_path,
    crossing_angles=np.array(CROSSING_ANGLES),
    uncertainty_labels=np.array([u['label'] for u in UNCERTAINTY_LEVELS]),
    uncertainty_titles=np.array([u['title'] for u in UNCERTAINTY_LEVELS]),
    methods=np.array(RECOVERY_METHODS),
    method_labels=np.array([METHOD_LABELS[m] for m in RECOVERY_METHODS]),
    ipr=ipr_arr,
    median_dcpa=median_dcpa_arr,
)
print(f'\nSaved → {out_path}')

# Quick summary table
print(f'\n{"Uncertainty":<20} {"Method":<16} {"Mean IPR":>9} {"Min IPR":>8}')
print('-' * 57)
for ui, unc in enumerate(UNCERTAINTY_LEVELS):
    for mi, method in enumerate(RECOVERY_METHODS):
        row = ipr_arr[ui, mi, :]
        print(f'{unc["label"]:<20} {METHOD_LABELS[method]:<16} '
              f'{row.mean():>9.4f} {row.min():>8.4f}')
