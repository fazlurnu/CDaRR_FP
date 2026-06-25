'''Probabilistic FTR conflict recovery.

Like :mod:`crr.ftr`, but each clear-of-conflict criterion is a
probability rather than a hard threshold:

    crit1 := P(DCPA > rpz | intruder keeps current velocity)  > threshold
    crit2 := P(DCPA > rpz | intruder reverts to initial velocity) > threshold

The probability model (projected-normal angle density + folded-normal tail) is a
set of pure functions; the recovery decision threads :class:`RecoveryState` and
injects its BlueSky side effects, matching the rest of the package.

**Desired-velocity approximations**

The mean relative velocity passed to each criterion uses the *ownship's*
desired velocity (read from the autopilot target — available locally) minus
the *intruder's* approximated desired velocity.  As with the deterministic
FTR rule (:mod:`crr.ftr`), the intruder's desired velocity is not communicated
via surveillance: ADS-L broadcasts only the instantaneous observed velocity.
Criterion 1 therefore uses the intruder's current observed velocity, and
criterion 2 uses the intruder's observed velocity at conflict initiation as a
proxy for its intended velocity.  The probabilistic criterion accounts for
sensor noise on top of this structural approximation, but it cannot compensate
for intruder intent that genuinely differs from the recorded initial velocity.
'''
import math

import numpy as np

from .common import (
    RecoveryState,
    apply_active_changes,
    compute_pair_positions,
    default_id2idx,
    default_recover,
    get_desired_ownship_velocity,
    get_pair_dxdy,
    record_initial_intruder_velocity,
)

# -------------------------
# Core math helpers
# -------------------------

try:
    from scipy.special import erf as sp_erf
    _erf = sp_erf
except Exception:
    _erf = np.vectorize(math.erf)

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
# 2D isotropic Gaussian: R_95 = σ · √(−2 ln 0.05) ≈ 2.448 σ
_SCALE_95 = math.sqrt(-2.0 * math.log(0.05))


def Phi(x):
    """Standard normal CDF, vectorized."""
    return 0.5 * (1.0 + _erf(np.asarray(x) / SQRT2))


def _to_cov(s, dim=2):
    """
    Convert scalar/std-vector/cov-matrix into a (dim,dim) covariance matrix.
    Accepts:
      - scalar std: s -> (s^2) I
      - (dim,) stds: -> diag(stds^2)
      - (dim,dim) cov: -> itself
    """
    if s is None:
        return np.zeros((dim, dim), float)

    if np.isscalar(s):
        return (float(s) ** 2) * np.eye(dim)

    s = np.asarray(s, float)
    if s.shape == (dim,):
        return np.diag(s ** 2)
    if s.shape == (dim, dim):
        return s

    raise ValueError(f"Invalid covariance/std shape: {s.shape}")


def _regularize_spd(S, eps=1e-9):
    """Make covariance numerically SPD-ish (adds eps*I)."""
    S = np.asarray(S, float).reshape(2, 2)
    return 0.5 * (S + S.T) + eps * np.eye(2)


def log_p_theta_projected_normal(theta, mu, Sigma):
    """
    Log of the projected-normal angular density p_Theta(theta).

    Numerically stable version that works in log-space to avoid the
    exp(-large) * exp(+large) overflow/underflow that occurs when
    the velocity SNR (|mu|/sigma) is high.

    theta : array in [0, 2pi)
    mu    : (2,) mean velocity
    Sigma : (2,2) velocity covariance (SPD)

    Returns log p(theta), shape (K,).
    """
    mu = np.asarray(mu, float).reshape(2)
    Sigma = _regularize_spd(Sigma, eps=1e-10)

    detS = float(np.linalg.det(Sigma))
    if detS <= 0:
        Sigma = _regularize_spd(Sigma, eps=1e-6)
        detS = float(np.linalg.det(Sigma))
        if detS <= 0:
            raise ValueError("Sigma_v must be positive definite.")

    Q = np.linalg.inv(Sigma)
    c = float(mu @ Q @ mu)

    u = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2,K)
    Qu = Q @ u                                            # (2,K)
    a = np.sum(u * Qu, axis=0)                             # (K,)
    b = (u.T @ (Q @ mu))                                   # (K,)

    a = np.maximum(a, 1e-15)
    z = b / np.sqrt(a)

    # --- log-space computation of term = 1/a + (b*sqrt(2pi)/a^1.5)*exp(0.5*z^2)*Phi(z) ---
    # term1 = 1/a
    log_term1 = -np.log(a)

    # term2 = |b| * sqrt(2pi) / a^1.5 * exp(0.5*z^2) * Phi(z)
    log_phi_z = np.log(np.maximum(Phi(z), 1e-300))
    log_term2_abs = (np.log(np.maximum(np.abs(b), 1e-300))
                     + math.log(SQRT2PI)
                     - 1.5 * np.log(a)
                     + 0.5 * z * z
                     + log_phi_z)

    sign_b = np.sign(b)

    # log(term) via log-sum-exp when b >= 0, log-sub-exp when b < 0
    log_term = np.where(
        sign_b >= 0,
        np.logaddexp(log_term1, log_term2_abs),
        # b < 0: term = 1/a - |term2|; guaranteed positive by theory
        log_term1 + np.log(np.maximum(
            1.0 - np.exp(np.minimum(log_term2_abs - log_term1, 500)),
            1e-300
        ))
    )

    log_const = -math.log(2.0 * math.pi) - 0.5 * math.log(detS)
    log_p = log_const - 0.5 * c + log_term

    return log_p


def p_theta_projected_normal(theta, mu, Sigma):
    """
    Angle density p_Theta(theta) for V ~ N(mu, Sigma) in R^2 (projected normal).
    theta: array in [0,2pi). Sigma must be SPD-ish.
    Returns p(theta) (not necessarily normalized unless you normalize in caller).

    This is a convenience wrapper around log_p_theta_projected_normal.
    """
    return np.exp(log_p_theta_projected_normal(theta, mu, Sigma))


def analytical_dcpa_prob_gt(x, mu_r, Sigma_r, mu_v, Sigma_v, Ktheta=256):
    """
    Compute P(DCPA > x) for unconstrained CPA model:
      d_CPA = r - v*(r.v)/(v.v), D = ||d_CPA||,
    with r ~ N(mu_r, Sigma_r), v ~ N(mu_v, Sigma_v), independent.
    Uses 1D integration over theta (direction of v) and a folded-normal tail.

    Parameters
    ----------
    x : float (threshold, x >= 0)
    mu_r : (2,) mean relative position [dx, dy]
    Sigma_r : (2,2) covariance of relative position
    mu_v : (2,) mean relative velocity [du, dv]
    Sigma_v : (2,2) covariance of relative velocity
    Ktheta : number of angle samples

    Returns
    -------
    float : P(D > x)
    """
    x = float(x)
    if x < 0:
        return 1.0

    mu_r = np.asarray(mu_r, float).reshape(2)
    Sigma_r = _regularize_spd(Sigma_r, eps=1e-9)

    mu_v = np.asarray(mu_v, float).reshape(2)
    Sigma_v = _regularize_spd(Sigma_v, eps=1e-9)

    theta = np.linspace(0.0, 2.0 * math.pi, int(Ktheta), endpoint=False)
    dtheta = 2.0 * math.pi / float(Ktheta)

    # Log-space density and normalization to avoid underflow/overflow
    # when velocity SNR is high (peaked p_Theta)
    log_pth = log_p_theta_projected_normal(theta, mu_v, Sigma_v)
    log_pth_max = np.max(log_pth)

    if not np.isfinite(log_pth_max):
        # Fallback: uniform weights that sum to 1
        weights = np.full_like(theta, 1.0 / float(Ktheta))
    else:
        # Normalize in log-space: w_k = exp(log_pth_k) / sum_j(exp(log_pth_j))
        # so that sum(w_k) = 1  (a discrete probability distribution over theta)
        log_pth_shifted = log_pth - log_pth_max
        pth_shifted = np.exp(log_pth_shifted)
        pth_sum_shifted = float(np.sum(pth_shifted))
        if pth_sum_shifted <= 0:
            weights = np.full_like(theta, 1.0 / float(Ktheta))
        else:
            weights = pth_shifted / pth_sum_shifted

    u_perp = np.stack([-np.sin(theta), np.cos(theta)], axis=0)  # (2,K)

    m = (u_perp.T @ mu_r)                                       # (K,)
    s2 = np.sum(u_perp * (Sigma_r @ u_perp), axis=0)            # (K,)
    s = np.sqrt(np.maximum(s2, 1e-15))                          # (K,)

    z1 = (x - m) / s
    z0 = (-x - m) / s
    cdf = Phi(z1) - Phi(z0)
    cdf = np.clip(cdf, 0.0, 1.0)

    tail = 1.0 - cdf
    p = float(np.sum(tail * weights))
    return float(np.clip(p, 0.0, 1.0))


def _aircraft_covariance(traffic, idx, attr, eps=1e-6):
    '''Extract a 2×2 isotropic covariance from an ADS-L accuracy field.

    ``attr`` is an attribute on ``traffic.adsl`` (e.g. ``'pos_acc'``,
    ``'vel_acc'``) holding a 95%-confidence accuracy radius (metres) per
    aircraft, indexed by aircraft index.  Converts to σ via the 2D isotropic
    relation R_95 = σ · _SCALE_95 ≈ 2.448 σ, then returns σ² I.
    Returns a near-zero SPD matrix when the attribute is absent.
    '''
    adsl = getattr(traffic, 'adsl', None)
    if adsl is not None and hasattr(adsl, attr):
        try:
            acc = float(getattr(adsl, attr)[idx])
            sigma = acc / _SCALE_95
            return _regularize_spd(sigma ** 2 * np.eye(2), eps=eps)
        except (IndexError, TypeError):
            pass
    return _regularize_spd(np.zeros((2, 2)), eps=eps)


# -------------------------
# Recovery method
# -------------------------
def resumenav_probabilistic_ftr(state: RecoveryState, conf, ownship, intruder,
                                active, **params):
    """
    Probabilistic version of the FTR double-criteria recovery.

    Per-aircraft uncertainty is read from ``ownship.adsl.pos_acc`` and
    ``ownship.adsl.vel_acc`` (and the same on ``intruder``).  Each is a
    95%-confidence accuracy radius (metres, isotropic) indexed by aircraft.
    Converted to σ via R_95 = σ · √(−2 ln 0.05) ≈ 2.448 σ, then to σ² I.
    The relative covariance for each pair is the sum of the two individual
    covariances.  Missing attributes fall back to a near-zero matrix.

    Uniform recovery interface: ``(state, conf, ownship, intruder, active,
    **params) -> (new_state, delpairs)``. Recognised ``params``:
      ``id2idx``         conflict-pair -> indices resolver (default ``default_id2idx``)
      ``recover``        waypoint-recovery side effect (default ``default_recover``)
      ``prob_threshold`` minimum P(DCPA > rpz) required to release (default 0.9)
      ``Ktheta``         angle samples for projected-normal integration (default 256)

    Side effects (writing ``active`` and waypoint recovery) go through the
    injected ``recover`` callable.
    """
    id2idx         = params.get("id2idx", default_id2idx)
    recover        = params.get("recover", default_recover)
    prob_threshold = params.get("prob_threshold", 0.9)
    Ktheta         = params.get("Ktheta", 256)
    state, _ = record_initial_intruder_velocity(state, conf, intruder, id2idx)

    pair_dxdy = compute_pair_positions(conf)
    vod_cache = {}
    init_vel = dict(state.init_vel)

    delpairs = set()
    changeactive = {}

    for conflict in state.resopairs:
        idx1, idx2 = id2idx(conflict)

        if idx1 < 0:
            delpairs.add(conflict)
            init_vel.pop(conflict, None)
            continue

        if idx2 < 0:
            delpairs.add(conflict)
            init_vel.pop(conflict, None)
            changeactive[idx1] = changeactive.get(idx1, False)
            continue

        # Per-pair covariances: relative = ownship + intruder (independent sources).
        # This assumption is valid since the model assume the noise has a Gaussian distribution
        Sigma_r = (_aircraft_covariance(ownship,  idx1, 'pos_acc') +
                   _aircraft_covariance(intruder, idx2, 'pos_acc'))
        Sigma_v = (_aircraft_covariance(ownship,  idx1, 'vel_acc') +
                   _aircraft_covariance(intruder, idx2, 'vel_acc'))

        dx, dy = get_pair_dxdy(conflict, pair_dxdy, ownship, intruder, idx1, idx2)
        rpz = float(np.max(conf.rpz[[idx1, idx2]]))
        Vo_u, Vo_v = get_desired_ownship_velocity(ownship, idx1, vod_cache)

        Vi_c_u = float(intruder.gseast[idx2])
        Vi_c_v = float(intruder.gsnorth[idx2])

        mu_r = np.array([dx, dy], dtype=float)

        # Criterion 1: intruder maintains current velocity (Vi,c).
        mu_v1 = np.array([Vo_u - Vi_c_u, Vo_v - Vi_c_v], dtype=float)
        p1 = analytical_dcpa_prob_gt(rpz, mu_r, Sigma_r, mu_v1, Sigma_v, Ktheta=Ktheta)
        crit1 = (p1 > prob_threshold)

        # Criterion 2: intruder reverts to initial velocity (Vi,i).
        Vi_i_u, Vi_i_v = init_vel.get(conflict, (Vi_c_u, Vi_c_v))
        mu_v2 = np.array([Vo_u - float(Vi_i_u), Vo_v - float(Vi_i_v)], dtype=float)
        p2 = analytical_dcpa_prob_gt(rpz, mu_r, Sigma_r, mu_v2, Sigma_v, Ktheta=Ktheta)
        crit2 = (p2 > prob_threshold)

        if crit1 and crit2:
            delpairs.add(conflict)
            init_vel.pop(conflict, None)
            changeactive[idx1] = changeactive.get(idx1, False)
        else:
            changeactive[idx1] = True

    apply_active_changes(changeactive, active, recover)

    new_state = RecoveryState(frozenset(set(state.resopairs) - delpairs), init_vel)
    return new_state, delpairs
