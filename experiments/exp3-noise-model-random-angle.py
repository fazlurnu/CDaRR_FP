'''Experiment 3 — Noise model comparison, random crossing angle, heterogeneous speed.

Design
------
* Uncertainty    : pos_ci95=10 m, vel_ci95=1 m/s  (single level)
* Noise model    : Normal Gaussian / Latency bias / Mixture Gaussian  (3 conditions)
* Crossing angle : drawn i.i.d. from Uniform(0, 360°) per run
* Speed          : drawn i.i.d. from Uniform(10, 30) kts per aircraft per run
* Pairs per run  : 10 × 10 = 100
* Runs per model : 10 000   →   100 × 10 000 = 1 000 000 pairs per condition

Results saved to experiments/results/exp3.npz.  Run directly::

    python experiments/exp3-noise-model-random-angle.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from joblib import Parallel, delayed

from experiments.config import (
    RPZ, HPZ, DTLOOKAHEAD, AIRCRAFT_TYPE,
    TMAX, DONE_TIMEOUT, KTHETA, DEFAULT_GAMMA,
    N_JOBS, BASE_SEED, RESULTS_DIR,
)
from runners.stochastic_pairwise_hor_conflict_heterogeneous_speed import get_ipr
from sim_models.cns.distributions import make_mixture_gaussian

# ── Exp 4 specific parameters ─────────────────────────────────────────────────
PAIR_WIDTH  = 10
PAIR_HEIGHT = 10        # 100 pairs per run
N_RUNS      = 10_000   # 10 000 × 100 = 1 000 000 pairs per noise model

SPEED_MIN   = 10.0     # kts
SPEED_MAX   = 30.0     # kts

LATENCY_S   = 0.0661   # s  — ADS-B v2 mean latency
TAIL_RATIO  = 3.0
TAIL_WEIGHT = 0.10

POS_CI95 = 10.0
VEL_CI95 =  1.0

# ── Noise model definitions ───────────────────────────────────────────────────
NOISE_MODELS = [
    ('normal',  None,                                            0.0),
    ('latency', None,                                            LATENCY_S),
    ('mixture', make_mixture_gaussian(TAIL_RATIO, TAIL_WEIGHT), 0.0),
]
NOISE_LABELS = [m[0] for m in NOISE_MODELS]

# ── Pre-generate random crossing angles (shared across models for comparability) ─
rng_angle = np.random.default_rng(BASE_SEED)
dpsi_values = rng_angle.uniform(0.0, 360.0, size=N_RUNS)  # shape (N_RUNS,)

# ── Common kwargs ─────────────────────────────────────────────────────────────
COMMON = dict(
    pair_width=PAIR_WIDTH, pair_height=PAIR_HEIGHT,
    rpz=RPZ, hpz=HPZ, dtlookahead=DTLOOKAHEAD,
    speed_min=SPEED_MIN, speed_max=SPEED_MAX,
    aircraft_type=AIRCRAFT_TYPE,
    tmax=TMAX, done_timeout=DONE_TIMEOUT,
    crr='probabilistic', Ktheta=KTHETA, prob_threshold=DEFAULT_GAMMA,
    reception_prob=1.0,
    pos_ci95=POS_CI95,
    vel_ci95=VEL_CI95,
)

# ── Storage ───────────────────────────────────────────────────────────────────
# ipr_arr: shape (n_models, N_RUNS) — per-run IPR for each noise model
n_models = len(NOISE_MODELS)
ipr_arr = np.full((n_models, N_RUNS), np.nan)

# ── Main sweep ────────────────────────────────────────────────────────────────
def _one(rep, pos_dist, latency_s):
    dist_arr, ipr, t_end = get_ipr(
        seed=BASE_SEED + rep,
        dpsi=float(dpsi_values[rep]),
        pos_dist=pos_dist,
        latency_s=latency_s,
        **COMMON,
    )
    return ipr


for mi, (model_label, pos_dist, latency_s) in enumerate(NOISE_MODELS):
    print(f'\nRunning noise model: {model_label}', flush=True)
    print(f'  {N_RUNS} runs × {PAIR_WIDTH * PAIR_HEIGHT} pairs = '
          f'{N_RUNS * PAIR_WIDTH * PAIR_HEIGHT:,} pairs', flush=True)

    results = Parallel(n_jobs=N_JOBS)(
        delayed(_one)(r, pos_dist, latency_s) for r in range(N_RUNS)
    )
    ipr_arr[mi] = np.array(results)

    nb_pair   = PAIR_WIDTH * PAIR_HEIGHT
    n_los     = np.sum((1.0 - ipr_arr[mi]) * nb_pair)
    total_ipr = 1.0 - n_los / float(N_RUNS * nb_pair)
    print(f'  overall IPR = {total_ipr:.4f}', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, 'exp3.npz')
np.savez(
    out_path,
    noise_labels=np.array(NOISE_LABELS),
    ipr=ipr_arr,
    dpsi_values=dpsi_values,
    pos_ci95=POS_CI95,
    vel_ci95=VEL_CI95,
    speed_min=SPEED_MIN,
    speed_max=SPEED_MAX,
    n_runs=N_RUNS,
    pair_width=PAIR_WIDTH,
    pair_height=PAIR_HEIGHT,
)
print(f'\nSaved → {out_path}')

# ── Summary ───────────────────────────────────────────────────────────────────
nb_pair = PAIR_WIDTH * PAIR_HEIGHT
print(f'\n{"Model":<10} {"Overall IPR":>12} {"Min run IPR":>12}')
print('-' * 38)
for mi, model_label in enumerate(NOISE_LABELS):
    n_los     = np.sum((1.0 - ipr_arr[mi]) * nb_pair)
    total_ipr = 1.0 - n_los / float(N_RUNS * nb_pair)
    print(f'{model_label:<10} {total_ipr:>12.4f} {ipr_arr[mi].min():>12.4f}')
