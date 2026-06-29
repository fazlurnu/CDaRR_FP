'''Experiment 2 — Effect of confidence threshold γ on IPR and median dCPA.

Reproduces the gamma-sensitivity result from the journal paper: sweeps crossing
angle from 2° to 180° for all four uncertainty levels and five γ values
{0.999, 0.99, 0.9, 0.75, 0.5}, using the probabilistic recovery method only.
The deterministic FTR (double_criteria) is included as a benchmark.

Results are saved to experiments/results/exp2.npz. Run directly:

    python experiments/exp2-gamma.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from joblib import Parallel, delayed

from config import (
    PAIR_WIDTH, PAIR_HEIGHT, RPZ, HPZ, DTLOOKAHEAD, SPEED, AIRCRAFT_TYPE,
    TMAX, DONE_TIMEOUT, KTHETA,
    CROSSING_ANGLES, UNCERTAINTY_LEVELS, GAMMA_VALUES,
    N_JOBS, BASE_SEED, RESULTS_DIR,
)
from runners.stochastic_pairwise_hor_conflict import run_single

# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _run_probabilistic(angle, unc, gamma, seed):
    res = run_single(
        pair_width=PAIR_WIDTH, pair_height=PAIR_HEIGHT,
        rpz=RPZ, hpz=HPZ, dtlookahead=DTLOOKAHEAD,
        init_speed_ownship=SPEED, init_speed_intruder=SPEED,
        aircraft_type=AIRCRAFT_TYPE, dpsi=float(angle),
        pos_ci95=unc['pos_ci95'], vel_ci95=unc['vel_ci95'],
        reception_prob=1.0,
        tmax=TMAX, done_timeout=DONE_TIMEOUT,
        crr='probabilistic', Ktheta=KTHETA, prob_threshold=gamma,
        seed=seed, record_history=False,
    )
    return res.ipr, float(np.median(res.min_dist))


def _run_ftr(angle, unc, seed):
    res = run_single(
        pair_width=PAIR_WIDTH, pair_height=PAIR_HEIGHT,
        rpz=RPZ, hpz=HPZ, dtlookahead=DTLOOKAHEAD,
        init_speed_ownship=SPEED, init_speed_intruder=SPEED,
        aircraft_type=AIRCRAFT_TYPE, dpsi=float(angle),
        pos_ci95=unc['pos_ci95'], vel_ci95=unc['vel_ci95'],
        reception_prob=1.0,
        tmax=TMAX, done_timeout=DONE_TIMEOUT,
        crr='double_criteria',
        seed=seed, record_history=False,
    )
    return res.ipr, float(np.median(res.min_dist))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

n_unc    = len(UNCERTAINTY_LEVELS)
n_gamma  = len(GAMMA_VALUES)
n_angles = len(CROSSING_ANGLES)

# Probabilistic: shape (n_unc, n_gamma, n_angles)
ipr_prob        = np.full((n_unc, n_gamma, n_angles), np.nan)
median_dcpa_prob = np.full((n_unc, n_gamma, n_angles), np.nan)

# FTR benchmark: shape (n_unc, n_angles)
ipr_ftr        = np.full((n_unc, n_angles), np.nan)
median_dcpa_ftr = np.full((n_unc, n_angles), np.nan)

for ui, unc in enumerate(UNCERTAINTY_LEVELS):

    # FTR benchmark for this uncertainty level
    print(f'Running FTR benchmark: {unc["label"]} ...', flush=True)
    ftr_results = Parallel(n_jobs=N_JOBS)(
        delayed(_run_ftr)(angle, unc, BASE_SEED + ai)
        for ai, angle in enumerate(CROSSING_ANGLES)
    )
    for ai, (ipr, med) in enumerate(ftr_results):
        ipr_ftr[ui, ai]        = ipr
        median_dcpa_ftr[ui, ai] = med

    # Probabilistic sweep over gamma values
    for gi, gamma in enumerate(GAMMA_VALUES):
        label = f'{unc["label"]} / γ={gamma}'
        print(f'Running: {label} ...', flush=True)

        results = Parallel(n_jobs=N_JOBS)(
            delayed(_run_probabilistic)(angle, unc, gamma, BASE_SEED + ai)
            for ai, angle in enumerate(CROSSING_ANGLES)
        )

        for ai, (ipr, med) in enumerate(results):
            ipr_prob[ui, gi, ai]        = ipr
            median_dcpa_prob[ui, gi, ai] = med

        mean_ipr = ipr_prob[ui, gi, :].mean()
        print(f'  done — mean IPR = {mean_ipr:.4f}', flush=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, 'exp2.npz')
np.savez(
    out_path,
    crossing_angles=np.array(CROSSING_ANGLES),
    uncertainty_labels=np.array([u['label'] for u in UNCERTAINTY_LEVELS]),
    uncertainty_titles=np.array([u['title'] for u in UNCERTAINTY_LEVELS]),
    gamma_values=np.array(GAMMA_VALUES),
    ipr_prob=ipr_prob,
    median_dcpa_prob=median_dcpa_prob,
    ipr_ftr=ipr_ftr,
    median_dcpa_ftr=median_dcpa_ftr,
)
print(f'\nSaved → {out_path}')

# Quick summary table
print(f'\n{"Uncertainty":<20} {"γ":>6} {"Mean IPR":>9} {"Min IPR":>8}')
print('-' * 47)
for ui, unc in enumerate(UNCERTAINTY_LEVELS):
    for gi, gamma in enumerate(GAMMA_VALUES):
        row = ipr_prob[ui, gi, :]
        print(f'{unc["label"]:<20} {gamma:>6.3f} {row.mean():>9.4f} {row.min():>8.4f}')
