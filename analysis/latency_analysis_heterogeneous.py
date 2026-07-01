'''Latency analysis — heterogeneous speeds (ownship 10 kts, intruder 30 kts).

Same investigation as latency_analysis.py but with asymmetric aircraft speeds:
ownship at 10 kts (5.1 m/s) and intruder at 30 kts (15.4 m/s).  This creates
an unequal along-track latency bias:

    ownship lag  ≈ 10 s × 5.1 m/s  =  51 m
    intruder lag ≈ 10 s × 15.4 m/s = 154 m

The net relative position error at a 90° crossing is therefore larger and
more asymmetric than in the equal-speed case, making it a stronger stress
test for the latency model.

All figures are written to figures/latency_analysis_heterogeneous/.

Run directly::

    python analysis/latency_analysis_heterogeneous.py
    python analysis/latency_analysis_heterogeneous.py --latex
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runners.stochastic_pairwise_hor_conflict import run_single
from plot_utils import (
    plot_pos_error,
    plot_latency_comparison,
    plot_avoidance_aggregate,
    set_latex_style,
)

LATEX_STYLE = "--latex" in sys.argv
set_latex_style(LATEX_STYLE)

# ── Scenario ──────────────────────────────────────────────────────────────────
PAIR_WIDTH    = 10
PAIR_HEIGHT   = 10      # 100 pairs per run
RPZ           = 50.0
HPZ           = 50.0
DTLOOKAHEAD   = 121.0
INIT_SPD_OWN  = 10.0   # kts — slow ownship
INIT_SPD_INT  = 30.0   # kts — fast intruder
DPSI          = 90      # perpendicular crossing — worst case for latency error
AIRCRAFT_TYPE = "M600"

POS_CI95      = 10.0   # m
VEL_CI95      = 1.0    # m/s
RECEPTION_PROB = 1.0

TMAX          = DTLOOKAHEAD * 4
DONE_TIMEOUT  = 30.0

LATENCY_HIGH  = 10.0   # s — exaggerated for visibility (real ADS-B ≈ 0.066 s)

STRATEGIES    = ("probabilistic", "double_criteria")
STRATEGY_LABELS = {"probabilistic": "probabilistic", "double_criteria": "ftr"}

SEED   = 44
PAIRS  = (12, 24)

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(_ROOT, "figures", "latency_analysis_heterogeneous")
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Common kwargs ─────────────────────────────────────────────────────────────
BASE_KWARGS = dict(
    pair_width=PAIR_WIDTH, pair_height=PAIR_HEIGHT,
    rpz=RPZ, hpz=HPZ, dtlookahead=DTLOOKAHEAD,
    init_speed_ownship=INIT_SPD_OWN, init_speed_intruder=INIT_SPD_INT,
    aircraft_type=AIRCRAFT_TYPE, dpsi=DPSI,
    pos_ci95=POS_CI95, vel_ci95=VEL_CI95, reception_prob=RECEPTION_PROB,
    tmax=TMAX, done_timeout=DONE_TIMEOUT,
    resofach=1.0, recovery_resofach=1.05,
    seed=SEED,
    record_history=True,
)

# ── Run all four conditions ───────────────────────────────────────────────────
print(f"Speeds: ownship={INIT_SPD_OWN} kts, intruder={INIT_SPD_INT} kts")
print("Running 4 conditions (2 noise × 2 recovery) …")
results = {}

for crr in STRATEGIES:
    lbl = STRATEGY_LABELS[crr]
    for lat_s, noise_label in [(0.0, "normal"), (LATENCY_HIGH, "latency")]:
        print(f"  {lbl:<16} | noise={noise_label} (latency_s={lat_s})")
        res = run_single(crr=crr, latency_s=lat_s, **BASE_KWARGS)
        results[(noise_label, lbl)] = res
        print(f"    IPR = {res.ipr:.4f}  (LoS={res.n_los}/{res.env.nb_pair})")

# ── Figure 1: positional error for each latency condition ─────────────────────
print("\nGenerating positional-error figures …")
for crr_lbl in STRATEGY_LABELS.values():
    res_lat = results[("latency", crr_lbl)]
    path = plot_pos_error(res_lat, FIGURE_DIR, f"latency_{crr_lbl}")
    print(f"  → {path}")

# ── Figure 2: pair-detail comparison (baseline vs high latency) ───────────────
print("\nGenerating pair-detail comparison figures …")
for crr_lbl in STRATEGY_LABELS.values():
    res_base = results[("normal",  crr_lbl)]
    res_lat  = results[("latency", crr_lbl)]
    for pair in PAIRS:
        path = plot_latency_comparison(res_base, res_lat, FIGURE_DIR, pair, crr_lbl)
        min_base = res_base.min_dist[pair]
        min_lat  = res_lat.min_dist[pair]
        print(f"  {crr_lbl} pair {pair:03d}: "
              f"CPA baseline={min_base:.1f} m  latency={min_lat:.1f} m  → {path}")

# ── Figure 3: aggregate avoidance — all four conditions ───────────────────────
print("\nGenerating aggregate avoidance comparison …")
results_by_label = {}
for (noise_label, crr_lbl), res in results.items():
    key = f"{crr_lbl} ({noise_label})"
    results_by_label[key] = [res]

path = plot_avoidance_aggregate(
    results_by_label, FIGURE_DIR,
    name="latency_analysis_het_avoidance_comparison.png",
    select="all",
)
print(f"  → {path}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\nOwnship {INIT_SPD_OWN} kts  |  Intruder {INIT_SPD_INT} kts")
print(f"\n{'Recovery':<16} {'Noise':<10} {'IPR':>8}  {'LoS':>6}")
print("-" * 44)
for crr_lbl in STRATEGY_LABELS.values():
    for noise_label in ("normal", "latency"):
        res = results[(noise_label, crr_lbl)]
        print(f"{crr_lbl:<16} {noise_label:<10} {res.ipr:>8.4f}  {res.n_los:>6}")
