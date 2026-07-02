'''Plotting helpers for stochastic pairwise horizontal conflict runs.

Each ``plot_*`` function takes a result namespace produced by
:func:`runners.stochastic_pairwise_hor_conflict.run_single` (with
``record_history=True``) and writes one PNG to ``figure_dir``, returning the
saved path. A ``label`` (e.g. the recovery strategy name) is appended to every
per-run filename and title so results from different runs don't collide.

Titles are built from fields the result already carries (``dpsi``, ``rpz``,
``pos_ci95``, ``vel_ci95``, ``reception_prob``), so the callers don't need to
re-pass scenario parameters.

Usage::

    from plot_utils import plot_run, plot_avoidance_compare

    plot_run(res_cpa, figure_dir, "cpa")
    plot_run(res_dc,  figure_dir, "double_criteria")
    plot_avoidance_compare(
        {"cpa": res_cpa, "double_criteria": res_dc}, figure_dir)
'''
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Earth radius (m) for the flat-earth trajectory projection.
_R_EARTH = 6371000.0
# Metres per degree latitude (flat-earth approximation; longitude scales by cos(lat)).
_M_PER_DEG = 111_320.0

# Output resolution and file formats for saved figures; both overridden by
# set_latex_style(). Every plot_* function writes one file per format.
_SAVE_DPI = 150
_SAVE_FORMATS = ("png",)


def set_latex_style(enable: bool = True, usetex: bool = None,
                    formats=None) -> None:
    '''Switch figures to a publication / LaTeX-friendly style.

    Applies serif (Computer Modern) fonts, inward major+minor ticks, a
    colourblind-safe colour cycle, tight 300-dpi output and other paper-ready
    rcParams --following https://basemrajjoub.com/programming/2026/03/17/matplotlib-latex-plots.

    Call once *before* creating any figures.

    ``usetex`` controls real-LaTeX text rendering: ``None`` (default) auto-uses
    LaTeX only if a ``latex`` binary is on PATH, else falls back to matplotlib's
    mathtext with the CM font set (no TeX install needed, and safe with the
    underscores / unicode in these labels). Pass ``True``/``False`` to force it.

    ``formats`` is the set of file types each figure is written as. ``None``
    (default) selects ``("pgf", "png")`` when a LaTeX install is present
    (``.pgf`` is meant to be ``\\input`` directly into a document so its text
    is typeset by LaTeX), else ``("png",)``. Pass e.g. ``("pgf",)`` to force it.

    Pass ``enable=False`` to restore matplotlib defaults and PNG-only output.
    '''
    global _SAVE_DPI, _SAVE_FORMATS
    if not enable:
        matplotlib.rcdefaults()
        matplotlib.use("Agg")
        _SAVE_DPI = 150
        _SAVE_FORMATS = ("png",)
        return

    has_latex = shutil.which("latex") is not None

    rc = {
        "font.family":         "serif",
        "font.size":           10,
        "axes.labelsize":      10,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "legend.fontsize":     9,
        "axes.prop_cycle":     matplotlib.cycler("color", [
            "#0072B2", "#D55E00", "#009E73",
            "#E69F00", "#CC79A7", "#56B4E9",
        ]),
        "lines.linewidth":     1.5,
        "axes.linewidth":      0.8,
        "xtick.direction":     "in",
        "ytick.direction":     "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size":    4,   "ytick.major.size":  4,
        "xtick.minor.size":    2,   "ytick.minor.size":  2,
        "xtick.major.width":   0.8, "ytick.major.width": 0.8,
        "xtick.minor.width":   0.6, "ytick.minor.width": 0.6,
        "lines.markersize":    4,
        "errorbar.capsize":    3,
        "axes.xmargin":        0.02,
        "axes.ymargin":        0.02,
        "legend.frameon":      False,
        "savefig.bbox":        "tight",
        "savefig.dpi":         300,
    }
    if usetex is None:
        usetex = has_latex
    if usetex:
        rc.update({
            "text.usetex":         True,
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
            "pgf.texsystem":       "pdflatex",
            "pgf.rcfonts":         False,
        })
    else:
        rc.update({"text.usetex": False, "mathtext.fontset": "cm"})

    matplotlib.rcParams.update(rc)
    _SAVE_DPI = 300
    if formats is None:
        formats = ("pgf", "png") if has_latex else ("png",)
    _SAVE_FORMATS = tuple(formats)

# Fixed colours for the strategy comparison plot (others fall back to the cycle).
_COMPARE_COLORS = {
    "cpa": "tab:green",
    "double_criteria": "tab:purple",
    "probabilistic": "tab:orange",
}

_FILE_PREFIX = "stochastic_pairwise_hor_conflict"


def _title_suffix(res) -> str:
    lat_s = getattr(res, "latency_s", 0.0)
    lat_str = f", latency={lat_s} s" if lat_s else ""
    return (f"pos_ci95={res.pos_ci95} m, vel_ci95={res.vel_ci95} m/s, "
            f"p_rx={res.reception_prob}{lat_str}")


def _ac_colors(env, ntraf) -> list:
    '''Per-aircraft colours: ownships blue, intruders red, others grey.'''
    colors = ["tab:gray"] * ntraf
    for i in env.ownship_idx:
        colors[i] = "tab:blue"
    for i in env.intruder_idx:
        colors[i] = "tab:red"
    return colors


def _write(fig, figure_dir, stem) -> str:
    '''Save *fig* as ``stem.<fmt>`` for every configured format; close it and
    return the first path. ``.pgf`` is saved with ``bbox_inches=None`` (per the
    LaTeX-plots recipe); other formats honour the current save DPI.'''
    paths = []
    for fmt in _SAVE_FORMATS:
        path = os.path.join(figure_dir, f"{stem}.{fmt}")
        if fmt == "pgf":
            fig.savefig(path, bbox_inches=None)
        else:
            fig.savefig(path, dpi=_SAVE_DPI)
        paths.append(path)
    plt.close(fig)
    return paths[0]


def _save(fig, figure_dir, name) -> str:
    fig.tight_layout()
    return _write(fig, figure_dir, os.path.splitext(name)[0])


def plot_distances(res, figure_dir, label) -> str:
    '''Ownship–intruder distance vs time for every pair, with the RPZ line.'''
    env = res.env
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(env.nb_pair):
        ax.plot(res.t_arr, res.dist_arr[:, i], color="tab:blue", alpha=0.4)
    ax.axhline(res.rpz, color="r", linestyle="--", label=f"RPZ = {res.rpz} m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title(
        f"Ownship-intruder distance --{label}  -- dpsi={res.dpsi} deg, RPZ={res.rpz} m\n"
        f"{_title_suffix(res)}"
    )
    ax.legend()
    return _save(fig, figure_dir, f"{_FILE_PREFIX}_distances_{label}.png")


def plot_gs_hdg(res, figure_dir, label) -> str:
    '''Ground speed (top) and wrapped heading (bottom) for every aircraft.'''
    env = res.env
    T, ntraf = res.lat_arr.shape
    colors = _ac_colors(env, ntraf)
    hdg_wrapped = ((res.hdg_arr + 180.0) % 360.0) - 180.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for k in range(ntraf):
        ax1.plot(res.t_arr, res.gs_arr[:T, k], color=colors[k], alpha=0.3)
        ax2.plot(res.t_arr, hdg_wrapped[:T, k], color=colors[k], alpha=0.3)
    ax1.set_ylabel("Ground speed (m/s)")
    ax1.set_title(
        f"Ground speed --{label}  -- dpsi={res.dpsi} deg  (DRO blue, DRI red)\n"
        f"{_title_suffix(res)}"
    )
    ax2.set_ylabel("Heading (deg, wrapped $\\pm$180 deg)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Heading")
    return _save(fig, figure_dir, f"{_FILE_PREFIX}_gs_hdg_{label}.png")


def plot_avoidance(res, figure_dir, label) -> str:
    '''Fraction of ownships and intruders actively avoiding over time.'''
    env = res.env
    own_idx = list(env.ownship_idx)
    int_idx = list(env.intruder_idx)
    own_avoid_avg = res.avoid_arr[:, own_idx].mean(axis=1)
    int_avoid_avg = res.avoid_arr[:, int_idx].mean(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(res.t_arr, own_avoid_avg, color="tab:blue", label="Ownship (DRO)")
    ax.plot(res.t_arr, int_avoid_avg, color="tab:red",  label="Intruder (DRI)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fraction avoiding")
    ax.set_title(
        f"Average avoidance --{label}  -- dpsi={res.dpsi} deg  (1 = all aircraft avoiding)\n"
        f"{_title_suffix(res)}"
    )
    ax.legend()
    return _save(fig, figure_dir, f"{_FILE_PREFIX}_avoidance_{label}.png")


def plot_trajectories(res, figure_dir, label) -> str:
    '''Ownship-centric trajectories; avoiding segments overplotted in dark.'''
    env = res.env
    lat_arr, lon_arr, avoid_arr = res.lat_arr, res.lon_arr, res.avoid_arr

    fig, ax = plt.subplots(figsize=(7, 7))
    for p in range(env.nb_pair):
        i_own = env.ownship_idx[p]
        i_int = env.intruder_idx[p]

        lat0  = float(lat_arr[0, i_own])
        lon0  = float(lon_arr[0, i_own])
        lat0r = np.deg2rad(lat0)

        x_own = np.deg2rad(lon_arr[:, i_own] - lon0) * _R_EARTH * np.cos(lat0r)
        y_own = np.deg2rad(lat_arr[:, i_own] - lat0) * _R_EARTH
        x_int = np.deg2rad(lon_arr[:, i_int] - lon0) * _R_EARTH * np.cos(lat0r)
        y_int = np.deg2rad(lat_arr[:, i_int] - lat0) * _R_EARTH

        own_av = avoid_arr[:, i_own] == 1.0
        int_av = avoid_arr[:, i_int] == 1.0

        # Base (nominal) trajectory in a very light tab colour.
        ax.plot(x_own, y_own, color="lightskyblue", alpha=0.3)
        ax.plot(x_int, y_int, color="lightsalmon",  alpha=0.3)
        # Overplot the avoiding portions in the full tab colour; NaN breaks the
        # line elsewhere.
        ax.plot(np.where(own_av, x_own, np.nan), np.where(own_av, y_own, np.nan),
                color="tab:blue", alpha=0.3)
        ax.plot(np.where(int_av, x_int, np.nan), np.where(int_av, y_int, np.nan),
                color="tab:red", alpha=0.3)

    ax.set_xlim(-3000, 1000)
    ax.set_ylim(-100, 4100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m) relative to ownship start")
    ax.set_ylabel("North (m) relative to ownship start")
    ax.set_title(
        f"Ownship-centric trajectories --{label}  -- dpsi={res.dpsi} deg  (DRO blue, DRI red)\n"
        f"{_title_suffix(res)}"
    )
    return _save(fig, figure_dir, f"{_FILE_PREFIX}_trajectories_{label}.png")


def plot_run(res, figure_dir, label) -> list:
    '''Render all four per-run figures for a single result; return saved paths.'''
    return [
        plot_distances(res, figure_dir, label),
        plot_gs_hdg(res, figure_dir, label),
        plot_avoidance(res, figure_dir, label),
        plot_trajectories(res, figure_dir, label),
    ]


# ── Detailed single-run, single-pair plots ────────────────────────────────────

def _pair_label(res, pair) -> str:
    '''Human-readable id of one pair, e.g. "pair 007 (DRO007 $\leftrightarrow$ DRI007)".'''
    return (f"pair {pair:03d} "
            f"({res.env.ownship_ids[pair]} $\leftrightarrow$ {res.env.intruder_ids[pair]})")


def _cpa_time(res, pair) -> float:
    '''Sim time (s) of the actual closest point of approach for the pair.'''
    return float(res.t_arr[int(np.argmin(res.dist_arr[:, pair]))])


def _draw_distance(ax, res, pair) -> None:
    '''Top panel --actual ownship–intruder separation flown each tick.'''
    ax.plot(res.t_arr, res.dist_arr[:, pair], color="tab:blue",
            label="actual distance")
    ax.axhline(res.rpz, color="tab:red", linestyle="--", label=f"RPZ = {res.rpz} m")
    ax.set_ylabel("Actual distance (m)")
    ax.set_title(f"Actual distance  (CPA = {res.min_dist[pair]:.1f} m)")


def _draw_dcpa_compare(ax, res, pair) -> None:
    '''Middle panel --projected CPA distance, ground truth vs noisy observation.

    Both lines are the geometric CPA projection computed directly from the pair
    (defined at every tick): truth from ``bs.traf`` (fresh) vs the held CNS
    observation (stepwise), exposing the effect of sensor noise / update latency.
    '''
    ax.plot(res.t_arr, res.dcpa_gt_arr[:, pair], color="tab:green",
            label="projected CPA (truth)")
    ax.plot(res.t_arr, res.dcpa_obs_arr[:, pair], color="tab:orange",
            alpha=0.85, label="projected CPA (observed)")
    ax.axhline(res.rpz, color="tab:red", linestyle="--", label=f"RPZ = {res.rpz} m")
    ax.set_ylabel("Projected dist. at CPA (m)")
    ax.set_title("Projected CPA distance --truth vs observed")


def _draw_avoidance(ax, res, pair) -> None:
    '''Bottom panel --avoidance status (symmetric, so one line is enough).'''
    i_own = res.env.ownship_idx[pair]
    ax.step(res.t_arr, res.avoid_arr[:, i_own], where="post", color="tab:blue")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Not active", "Active"], rotation=90, va="center")
    ax.set_title("Avoidance status")


def plot_pair_detail(res, figure_dir, pair, label) -> str:
    '''Stacked single-pair detail figure: actual distance (top), projected CPA
    distance truth-vs-observed (middle), and avoidance status (bottom), sharing
    a time axis. A vertical dashed line marks the actual closest point of
    approach on every panel.

    ``pair`` is the pair index (0 .. nb_pair-1). Requires a result produced with
    ``record_history=True``. Returns the saved path.
    '''
    t_cpa = _cpa_time(res, pair)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    _draw_distance(ax1, res, pair)
    _draw_dcpa_compare(ax2, res, pair)
    _draw_avoidance(ax3, res, pair)

    ax2.set_ylim(-2, 252)

    for k, ax in enumerate((ax1, ax2, ax3)):
        # CPA marker on every panel; label it only once (top) to avoid clutter.
        ax.axvline(t_cpa, color="dimgray", linestyle=":", linewidth=1.5,
                   label=(f"CPA @ {t_cpa:.0f} s" if k == 0 else None))
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    ax3.set_xlim(res.t_arr[0], 150)   # shared axis: clips every panel at 150 s
    ax3.set_xlabel("Time (s)")
    fig.tight_layout()
    return _write(fig, figure_dir, f"{_FILE_PREFIX}_pair{pair:03d}_detail_{label}")


def plot_pair_trajectory(res, figure_dir, pair, label) -> str:
    '''Ownship-centric trajectory for a single pair (avoiding segments in dark,
    actual CPA marked). Companion to :func:`plot_pair_detail`, zoomed to one
    pair instead of plotting every pair in the run.
    '''
    env = res.env
    lat_arr, lon_arr, avoid_arr = res.lat_arr, res.lon_arr, res.avoid_arr

    i_own = env.ownship_idx[pair]
    i_int = env.intruder_idx[pair]

    lat0  = float(lat_arr[0, i_own])
    lon0  = float(lon_arr[0, i_own])
    lat0r = np.deg2rad(lat0)

    x_own = np.deg2rad(lon_arr[:, i_own] - lon0) * _R_EARTH * np.cos(lat0r)
    y_own = np.deg2rad(lat_arr[:, i_own] - lat0) * _R_EARTH
    x_int = np.deg2rad(lon_arr[:, i_int] - lon0) * _R_EARTH * np.cos(lat0r)
    y_int = np.deg2rad(lat_arr[:, i_int] - lat0) * _R_EARTH

    own_av = avoid_arr[:, i_own] == 1.0
    int_av = avoid_arr[:, i_int] == 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_own, y_own, color="lightskyblue", alpha=0.4, label="Ownship (nominal)")
    ax.plot(x_int, y_int, color="lightsalmon",  alpha=0.4, label="Intruder (nominal)")
    ax.plot(np.where(own_av, x_own, np.nan), np.where(own_av, y_own, np.nan),
            color="tab:blue", label="Ownship (avoiding)")
    ax.plot(np.where(int_av, x_int, np.nan), np.where(int_av, y_int, np.nan),
            color="tab:red", label="Intruder (avoiding)")

    i_cpa = int(np.argmin(res.dist_arr[:, pair]))
    ax.scatter([x_own[i_cpa], x_int[i_cpa]], [y_own[i_cpa], y_int[i_cpa]],
               color="dimgray", marker="x", s=80, zorder=5,
               label=f"CPA @ {res.t_arr[i_cpa]:.0f} s")

    ax.set_xlim(-3000, 1000)
    ax.set_ylim(-100, 4100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m) relative to ownship start")
    ax.set_ylabel("North (m) relative to ownship start")
    ax.set_title(
        f"Ownship-centric trajectory --{_pair_label(res, pair)}  --{label}\n"
        f"CPA = {res.min_dist[pair]:.1f} m  -- {_title_suffix(res)}"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _write(fig, figure_dir, f"{_FILE_PREFIX}_pair{pair:03d}_trajectory_{label}")


def plot_avoidance_aggregate(results_by_label, figure_dir,
                             name=f"{_FILE_PREFIX}_avoidance_aggregate.png",
                             select="all") -> str:
    '''Aggregate the avoidance flag across many runs per strategy and overlay.

    ``results_by_label`` maps a label (recovery strategy) to a list of run
    namespaces produced with ``record_history=True``. For each strategy all the
    selected aircraft of all its runs are pooled and the mean avoidance fraction
    over time is plotted as a single line --e.g. 10 runs × 100 pairs gives the
    average avoidance flag over 1000 pairs.

    Runs of unequal length are zero-padded to the longest one: once a run has
    terminated its aircraft are no longer avoiding, so trailing zeros are exact.

    ``select`` chooses which aircraft are pooled: ``"all"`` (every aircraft),
    ``"ownship"`` (DROs only), or ``"intruder"`` (DRIs only).
    '''
    all_runs = [r for runs in results_by_label.values() for r in runs]
    t_max    = max(r.avoid_arr.shape[0] for r in all_runs)
    t_axis   = max((r.t_arr for r in all_runs), key=len)

    def _sel(env):
        if select == "ownship":
            return list(env.ownship_idx)
        if select == "intruder":
            return list(env.intruder_idx)
        return list(env.ownship_idx) + list(env.intruder_idx)

    def _pooled_mean(runs):
        cols = []
        for r in runs:
            a = r.avoid_arr[:, _sel(r.env)]            # (T_run, n_sel)
            if a.shape[0] < t_max:                     # zero-pad past termination
                a = np.vstack([a, np.zeros((t_max - a.shape[0], a.shape[1]))])
            cols.append(a)
        return np.hstack(cols).mean(axis=1)            # mean over all pooled aircraft

    any_res = all_runs[0]
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, runs in results_by_label.items():
        n_pairs = sum(r.env.nb_pair for r in runs)
        ax.plot(t_axis, _pooled_mean(runs),
                color=_COMPARE_COLORS.get(label),
                label=f"{label}  (n={n_pairs} pairs, {len(runs)} runs)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Average avoidance flag ({select} aircraft)")
    ax.set_title(
        f"Aggregated average avoidance --recovery strategy comparison  -- "
        f"dpsi={any_res.dpsi} deg\n{_title_suffix(any_res)}"
    )
    ax.legend()
    return _save(fig, figure_dir, name)


def plot_pos_error(res, figure_dir, label) -> str:
    '''Per-aircraft positional error (sensor vs truth) over time.

    The error magnitude at each tick is sqrt(Δeast² + Δnorth²) in metres, where
    Δeast and Δnorth are derived from the difference between sensor lat/lon and
    truth lat/lon. For a high-latency run this reveals the systematic along-track
    bias; for normal noise it shows the random scatter around zero.

    Requires ``record_history=True`` and that the runner recorded
    ``sensor_lat_arr`` / ``sensor_lon_arr`` (available from
    :mod:`runners.stochastic_pairwise_hor_conflict` ≥ latency support).
    '''
    truth_lat = res.lat_arr          # (T, ntraf)
    truth_lon = res.lon_arr
    sens_lat  = res.sensor_lat_arr
    sens_lon  = res.sensor_lon_arr

    lat0r = np.deg2rad(truth_lat)
    d_north = (sens_lat - truth_lat) * _M_PER_DEG
    d_east  = (sens_lon - truth_lon) * _M_PER_DEG * np.cos(lat0r)
    err_m   = np.sqrt(d_east**2 + d_north**2)    # (T, ntraf)

    fig, ax = plt.subplots(figsize=(12, 5))
    for k in range(err_m.shape[1]):
        ax.plot(res.t_arr, err_m[:, k], color="tab:blue", alpha=0.15, linewidth=0.8)
    ax.plot(res.t_arr, err_m.mean(axis=1), color="tab:blue", linewidth=2,
            label="mean positional error")
    ax.axhline(res.rpz, color="tab:red", linestyle="--",
               label=f"RPZ = {res.rpz} m")

    lat_s = getattr(res, "latency_s", 0.0)
    expected_bias = lat_s * np.mean(res.gs_arr) if lat_s else None
    if expected_bias is not None:
        ax.axhline(expected_bias, color="tab:orange", linestyle=":",
                   label=f"expected bias $\\approx$ {expected_bias:.0f} m  (latency $\\times$ mean gs)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Positional error, sensor vs truth (m)")
    ax.set_title(
        f"Sensor positional error --{label}  -- dpsi={res.dpsi} deg\n"
        f"{_title_suffix(res)}"
    )
    ax.legend()
    return _save(fig, figure_dir, f"{_FILE_PREFIX}_pos_error_{label}.png")


def plot_latency_comparison(res_base, res_lat, figure_dir, pair,
                            crr_label) -> str:
    '''Two-column pair-detail comparison: baseline noise (left) vs high latency (right).

    Each column has three vertically stacked panels (shared time axis):
      • actual ownship–intruder distance with RPZ line
      • projected DCPA --ground truth (green) vs CNS observation (orange)
      • avoidance status (step plot)

    A vertical dashed line marks the actual CPA time in each result. The left
    column is the baseline (``res_base``); the right is the latency run
    (``res_lat``).  Both must have been run with ``record_history=True``.
    '''
    def _cpa_t(res, p):
        return float(res.t_arr[int(np.argmin(res.dist_arr[:, p]))])

    def _fill_col(axes, res, col_title):
        ax_dist, ax_dcpa, ax_av = axes
        t = res.t_arr
        t_cpa = _cpa_t(res, pair)

        # Row 1 – actual distance
        ax_dist.plot(t, res.dist_arr[:, pair], color="tab:blue")
        ax_dist.axhline(res.rpz, color="tab:red", linestyle="--",
                        label=f"RPZ = {res.rpz} m")
        ax_dist.axvline(t_cpa, color="dimgray", linestyle=":", linewidth=1.5,
                        label=f"CPA @ {t_cpa:.0f} s")
        ax_dist.set_ylabel("Actual dist. (m)")
        ax_dist.set_title(f"{col_title}\nActual distance  (CPA = {res.min_dist[pair]:.1f} m)")
        ax_dist.legend(loc="upper right", fontsize=8)

        # Row 2 – projected DCPA
        ax_dcpa.plot(t, res.dcpa_gt_arr[:, pair], color="tab:green",
                     label="DCPA truth")
        ax_dcpa.plot(t, res.dcpa_obs_arr[:, pair], color="tab:orange",
                     alpha=0.85, label="DCPA observed")
        ax_dcpa.axhline(res.rpz, color="tab:red", linestyle="--")
        ax_dcpa.axvline(t_cpa, color="dimgray", linestyle=":", linewidth=1.5)
        ax_dcpa.set_ylim(-2, 252)
        ax_dcpa.set_ylabel("Projected DCPA (m)")
        ax_dcpa.set_title("Projected DCPA --truth vs observed")
        ax_dcpa.legend(loc="upper right", fontsize=8)

        # Row 3 – avoidance
        i_own = res.env.ownship_idx[pair]
        ax_av.step(t, res.avoid_arr[:, i_own], where="post", color="tab:blue")
        ax_av.axvline(t_cpa, color="dimgray", linestyle=":", linewidth=1.5)
        ax_av.set_ylim(-0.05, 1.05)
        ax_av.set_yticks([0, 1])
        ax_av.set_yticklabels(["Off", "On"], rotation=90, va="center")
        ax_av.set_ylabel("Avoidance")
        ax_av.set_title("Avoidance status")
        ax_av.set_xlabel("Time (s)")
        ax_av.set_xlim(t[0], 150)

    lat_s = getattr(res_lat, "latency_s", "?")
    fig, axes = plt.subplots(3, 2, figsize=(14, 11),
                             sharex="col", sharey="row")

    _fill_col(axes[:, 0], res_base, f"Baseline (latency = 0 s)")
    _fill_col(axes[:, 1], res_lat,  f"High latency ({lat_s} s)")

    pair_label = (f"pair {pair:03d} "
                  f"({res_base.env.ownship_ids[pair]} $\leftrightarrow$ {res_base.env.intruder_ids[pair]})")
    fig.suptitle(
        f"Latency comparison --{crr_label}  -- {pair_label}  -- dpsi={res_base.dpsi} deg\n"
        f"{_title_suffix(res_base).split(',')[0]}  |  latency = 0 s  vs  {lat_s} s",
        fontsize=10,
    )
    fig.tight_layout()
    return _write(fig, figure_dir,
                  f"{_FILE_PREFIX}_latency_comparison_pair{pair:03d}_{crr_label}")


def plot_avoidance_compare(results, figure_dir,
                           name=f"{_FILE_PREFIX}_avoidance_compare.png") -> str:
    '''Overlay the mean avoidance fraction of several runs.

    ``results`` maps a label to a result namespace; all runs are assumed to
    share a scenario (the title is taken from the first one).
    '''
    any_res = next(iter(results.values()))
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, res in results.items():
        ax.plot(res.t_arr, res.avoid_arr.mean(axis=1),
                color=_COMPARE_COLORS.get(label), label=label)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fraction avoiding (all aircraft)")
    ax.set_title(
        f"Average avoidance --recovery strategy comparison  -- dpsi={any_res.dpsi} deg\n"
        f"{_title_suffix(any_res)}"
    )
    ax.legend()
    return _save(fig, figure_dir, name)
