'''Pairwise horizontal conflict — full-loop simulation (MVP + FTR recovery).

Deterministic scenario: 3×3 grid of ownship/intruder pairs at dpsi=90°.
Uses the functional pipeline: detect → mvp.resolve → resumenav_double_criteria.
BlueSky truth (bs.traf) is passed directly as both ownship and intruder views
(no CNS noise); swap in adsl_field outputs to add a communication layer.

Saves three figures to figures/tests/:
  pairwise_hor_conflict_distances.png    — ownship–intruder distance over time
  pairwise_hor_conflict_gs_hdg.png       — ground speed and heading over time
  pairwise_hor_conflict_trajectories.png — ownship-centric spatial trajectories

Run directly::

    python tests/test_pairwise_hor_conflict_sim.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bluesky as bs

from cd import detect
from cr import ResolutionConfig, mvp
from crr import empty_recovery_state, resumenav_double_criteria
from envs.pairwise_hor_conflict import make_pairwise_hor_conflict, step, reset

# ── BlueSky init ─────────────────────────────────────────────────────────────
if not getattr(bs, "_sim_inited", False):
    bs.init(mode="sim", detached=True)
    bs._sim_inited = True

# ── Parameters ────────────────────────────────────────────────────────────────
PAIR_WIDTH    = 3
PAIR_HEIGHT   = 3
RPZ           = 50.0    # m
HPZ           = 50.0    # m
DTLOOKAHEAD   = 121.0   # s
INIT_SPD_OWN  = 15.0    # m/s
INIT_SPD_INT  = 15.0    # m/s
DPSI          = 90      # deg
AIRCRAFT_TYPE = "M600"
SIMDT_FACTOR  = 1
START_LAT     = 52.0
START_LON     = 4.0
DELTA_LAT_LON = 0.01

TMAX    = DTLOOKAHEAD * 4
SIMDT   = bs.settings.simdt * SIMDT_FACTOR
ASAS_DT = float(bs.settings.asas_dt)
CFG     = ResolutionConfig(resofach=1.05)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(_ROOT, "figures", "tests")
os.makedirs(FIGURE_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _id2idx(pair):
    a, b = pair
    return bs.traf.id2idx(a), bs.traf.id2idx(b)


def _noop_recover(_idx):
    # pairwise aircraft have no filed route; waypoint recovery is a no-op
    pass


# ── Spawn ─────────────────────────────────────────────────────────────────────
env = make_pairwise_hor_conflict(
    pair_width=PAIR_WIDTH, pair_height=PAIR_HEIGHT,
    asas_pzr_m=RPZ, dtlookahead=DTLOOKAHEAD,
    init_speed_ownship=INIT_SPD_OWN, init_speed_intruder=INIT_SPD_INT,
    aircraft_type_ownship=AIRCRAFT_TYPE,
    start_lat=START_LAT, start_lon=START_LON, delta_lat_lon=DELTA_LAT_LON,
    init_dpsi=DPSI, simdt_factor=SIMDT_FACTOR,
)

# ── Sim loop ──────────────────────────────────────────────────────────────────
recovery_state = empty_recovery_state()
active = np.zeros(bs.traf.ntraf, dtype=bool)

time_list, distance_list = [], []
lat_list, lon_list       = [], []
gs_list, hdg_list        = [], []

t            = 0.0
eps          = np.finfo(float).eps * 100
next_event_t = 0.0
action       = None

while t < TMAX:
    if t + eps >= next_event_t:
        conf = detect(bs.traf, bs.traf, RPZ, HPZ, DTLOOKAHEAD)
        newtrack, newgs, newvs, alt = mvp.resolve(conf, bs.traf, bs.traf, CFG)
        recovery_state, _ = resumenav_double_criteria(
            recovery_state, conf, bs.traf, bs.traf, active,
            id2idx=_id2idx, recover=_noop_recover,
        )
        action = (newtrack, newgs, newvs, alt, list(recovery_state.resopairs))

        missed = int(np.floor((t - next_event_t) / ASAS_DT)) + 1 if t > next_event_t else 1
        next_event_t += missed * ASAS_DT

    distances = step(env, action)

    time_list.append(t)
    distance_list.append(distances.copy())
    lat_list.append(bs.traf.lat.copy())
    lon_list.append(bs.traf.lon.copy())
    gs_list.append(bs.traf.gs.copy())
    hdg_list.append(bs.traf.hdg.copy())

    t += SIMDT

reset()

# ── Arrays ────────────────────────────────────────────────────────────────────
t_arr    = np.array(time_list)
dist_arr = np.array(distance_list)   # (T, nb_pair)
lat_arr  = np.array(lat_list)        # (T, ntraf)
lon_arr  = np.array(lon_list)
gs_arr   = np.array(gs_list)
hdg_arr  = np.array(hdg_list)

T, ntraf    = lat_arr.shape
hdg_wrapped = ((hdg_arr + 180.0) % 360.0) - 180.0

colors = ["tab:gray"] * ntraf
for i in env.ownship_idx:
    colors[i] = "tab:blue"
for i in env.intruder_idx:
    colors[i] = "tab:red"

# ── Figure 1: distances ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for i in range(env.nb_pair):
    ax.plot(t_arr, dist_arr[:, i], color="tab:blue", alpha=0.4)
ax.axhline(RPZ, color="r", linestyle="--", label=f"RPZ = {RPZ} m")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Distance (m)")
ax.set_title(f"Ownship–intruder distance  —  dpsi={DPSI}°, RPZ={RPZ} m")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "pairwise_hor_conflict_distances.png"), dpi=150)
plt.close(fig)

# ── Figure 2: ground speed and heading ───────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for k in range(ntraf):
    ax1.plot(t_arr, gs_arr[:T, k], color=colors[k], alpha=0.3)
    ax2.plot(t_arr, hdg_wrapped[:T, k], color=colors[k], alpha=0.3)
ax1.set_ylabel("Ground speed (m/s)")
ax1.set_title(f"Ground speed  —  dpsi={DPSI}°  (DRO blue, DRI red)")
ax2.set_ylabel("Heading (deg, wrapped ±180°)")
ax2.set_xlabel("Time (s)")
ax2.set_title("Heading")
fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "pairwise_hor_conflict_gs_hdg.png"), dpi=150)
plt.close(fig)

# ── Figure 3: ownship-centric trajectories ────────────────────────────────────
R = 6371000.0
fig, ax = plt.subplots(figsize=(7, 7))
for p in range(env.nb_pair):
    i_own = env.ownship_idx[p]
    i_int = env.intruder_idx[p]

    lat0  = float(lat_arr[0, i_own])
    lon0  = float(lon_arr[0, i_own])
    lat0r = np.deg2rad(lat0)

    x_own = np.deg2rad(lon_arr[:, i_own] - lon0) * R * np.cos(lat0r)
    y_own = np.deg2rad(lat_arr[:, i_own] - lat0) * R
    x_int = np.deg2rad(lon_arr[:, i_int] - lon0) * R * np.cos(lat0r)
    y_int = np.deg2rad(lat_arr[:, i_int] - lat0) * R

    ax.plot(x_own, y_own, color="tab:blue", alpha=0.4)
    ax.plot(x_int, y_int, color="tab:red",  alpha=0.4)

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("East (m) relative to ownship start")
ax.set_ylabel("North (m) relative to ownship start")
ax.set_title(f"Ownship-centric trajectories  —  dpsi={DPSI}°  (DRO blue, DRI red)")
fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "pairwise_hor_conflict_trajectories.png"), dpi=150)
plt.close(fig)

# ── Summary ───────────────────────────────────────────────────────────────────
min_dist = np.min(dist_arr, axis=0)
n_los    = int(np.sum(min_dist < RPZ))
ipr      = 1.0 - n_los / float(env.nb_pair)

print(f"IPR      : {ipr:.4f}  (LOS={n_los}/{env.nb_pair})")
print(f"Min dist : {np.round(min_dist, 1).tolist()} m")
print(f"Figures  → {FIGURE_DIR}")
