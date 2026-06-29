'''Noise model comparison — Monte Carlo 95% CI verification.

Compares three position noise models side by side:
  1. Normal Gaussian — isotropic, zero-mean
  2. Latency bias    — Gaussian noise + along-track shift (bias = −latency × gs)
  3. Mixture Gaussian — zero-mean, heavy-tailed (dominant + tail component)

Layout: 3 columns × 2 rows
  Row 1: 2D scatter of (east, north) positional error
  Row 2: empirical CDF of radial distance r = √(east² + north²)

The empirical 95th percentile is computed via Monte Carlo and shown on every
subplot. For the latency model the total positional error (noise + bias) is
plotted — the shift causes p95 to exceed the nominal ci95, which is the point.

Run directly:
    python tests/test_noise_model_comparison.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sim_models.cns.distributions import gaussian, make_mixture_gaussian

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
CI95        = 50.0     # nominal 95% CI radius [m] — same for all three models
N_SCATTER   = 3_000    # points drawn in scatter plots
N_MC        = 100_000  # total Monte Carlo draws for empirical CI

# Latency model parameters (cruising speed so the bias is clearly visible)
GS_MS       = 233.0    # ground speed [m/s] ≈ 840 km/h
LATENCY_S   = 0.0661   # ADS-B v2 mean latency [s]
TRK_DEG     = 0.0      # track angle [deg], 0 = flying north

# Mixture Gaussian parameters
TAIL_RATIO  = 3.0      # σ₂ / σ₁
TAIL_WEIGHT = 0.10     # probability of drawing from the tail component

SEED        = 42
OUT_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'figures', 'tests',
                           'noise_model_comparison.png')

# ---------------------------------------------------------------------------
# Draw samples
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

# 1. Normal Gaussian
draws_normal = gaussian(N_MC, CI95, rng)

# 2. Latency: Gaussian noise + along-track bias rotated to (east, north)
#    bias_at = −latency × gs  (negative → position lags behind the aircraft)
bias_at  = -LATENCY_S * GS_MS
trk_rad  = np.deg2rad(TRK_DEG)
east_b   = bias_at * np.sin(trk_rad)   #  0 m  (flying north, no east component)
north_b  = bias_at * np.cos(trk_rad)   # −15.4 m south of truth
draws_latency = gaussian(N_MC, CI95, rng) + np.array([east_b, north_b])

# 3. Mixture Gaussian
mix_dist = make_mixture_gaussian(tail_ratio=TAIL_RATIO, tail_weight=TAIL_WEIGHT)
draws_mix = mix_dist(N_MC, CI95, rng)

# ---------------------------------------------------------------------------
# Empirical 95th percentile via Monte Carlo
# ---------------------------------------------------------------------------
def empirical_p95(draws):
    return np.percentile(np.linalg.norm(draws, axis=1), 95)

models = [
    ('Normal Gaussian\n(isotropic, zero-mean)',
     draws_normal,
     '#2563eb'),
    (f'Latency Bias\n(gs={GS_MS:.0f} m/s, τ={LATENCY_S*1e3:.1f} ms, trk={TRK_DEG:.0f}°)',
     draws_latency,
     '#dc2626'),
    (f'Mixture Gaussian\n(ratio={TAIL_RATIO}, tail weight={TAIL_WEIGHT:.0%})',
     draws_mix,
     '#16a34a'),
]

p95_values = [empirical_p95(d) for _, d, _ in models]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    f'Noise Model Comparison — Monte Carlo 95% CI  '
    f'(N={N_MC:,}, nominal ci95={CI95:.0f} m)',
    fontsize=13, fontweight='bold', y=1.01,
)

theta = np.linspace(0, 2 * np.pi, 400)

SCATTER_LIM = CI95 * 3.0   # axis half-width for scatter plots
CDF_XLIM    = CI95 * 4.0   # x-axis limit for CDF plots

for col, ((title, draws, color), p95) in enumerate(zip(models, p95_values)):

    r_all = np.linalg.norm(draws, axis=1)

    # ── Row 0: 2D scatter ──────────────────────────────────────────────────
    ax = axes[0, col]

    # Scatter (subsample for legibility)
    sc_idx = np.random.default_rng(SEED + col).choice(N_MC, N_SCATTER, replace=False)
    ax.scatter(draws[sc_idx, 0], draws[sc_idx, 1],
               s=3, alpha=0.25, color=color, linewidths=0)

    # Nominal ci95 circle
    ax.plot(CI95 * np.cos(theta), CI95 * np.sin(theta),
            'k--', lw=1.5, label=f'nominal ci95={CI95:.0f} m')

    # Empirical p95 circle
    ax.plot(p95 * np.cos(theta), p95 * np.sin(theta),
            color='tomato', lw=1.5, ls='-',
            label=f'empirical p95={p95:.1f} m')

    # Origin cross
    ax.axhline(0, color='k', lw=0.4, alpha=0.4)
    ax.axvline(0, color='k', lw=0.4, alpha=0.4)

    # Bias arrow for latency model
    if col == 1:
        ax.annotate('', xy=(east_b, north_b), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='maroon', lw=2.0))
        ax.text(east_b + 2, north_b - 4,
                f'bias\n({east_b:.1f}, {north_b:.1f}) m',
                fontsize=8, color='maroon')

    ax.set_xlim(-SCATTER_LIM, SCATTER_LIM)
    ax.set_ylim(-SCATTER_LIM, SCATTER_LIM)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('East error (m)', fontsize=9)
    if col == 0:
        ax.set_ylabel('North error (m)', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, lw=0.3, alpha=0.5)

    # ── Row 1: Empirical CDF of radial distance ────────────────────────────
    ax = axes[1, col]

    r_sorted = np.sort(r_all)
    cdf      = np.arange(1, N_MC + 1) / N_MC

    ax.plot(r_sorted, cdf, color=color, lw=1.8, label='empirical CDF')

    # Reference lines
    ax.axvline(CI95, color='k', lw=1.5, ls='--',
               label=f'nominal ci95={CI95:.0f} m')
    ax.axvline(p95, color='tomato', lw=1.5, ls='-',
               label=f'empirical p95={p95:.1f} m')
    ax.axhline(0.95, color='gray', lw=0.8, ls=':', alpha=0.8)

    # Annotation of the gap between nominal and empirical
    gap = p95 - CI95
    sign = '+' if gap >= 0 else ''
    ax.text(max(CI95, p95) + 1, 0.60,
            f'Δ = {sign}{gap:.1f} m', fontsize=9,
            color='tomato' if abs(gap) > 1 else 'green')

    ax.set_xlim(0, CDF_XLIM)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Radial distance r (m)', fontsize=9)
    if col == 0:
        ax.set_ylabel('CDF', fontsize=9)
    ax.set_title(f'Radial CDF  (p95 = {p95:.1f} m)', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, lw=0.3, alpha=0.5)

# Row labels on the left margin
for row, label in enumerate(['2D Positional Error', 'Radial Distance CDF']):
    axes[row, 0].set_ylabel(
        f'{label}\n{axes[row, 0].get_ylabel()}', fontsize=9
    )

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved → {OUT_PATH}')

# Summary table
print(f'\n{"Model":<35} {"Nominal ci95":>13} {"Empirical p95":>14} {"Δ":>8}')
print('-' * 73)
for (title, _, _), p95 in zip(models, p95_values):
    name = title.split('\n')[0]
    print(f'{name:<35} {CI95:>13.1f} m {p95:>13.1f} m {p95-CI95:>+7.1f} m')
