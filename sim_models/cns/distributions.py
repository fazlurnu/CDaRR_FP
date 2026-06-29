'''Noise distributions for the CNS sensor layer — functional style.

A *distribution* here is simply any callable ``(n, ci95, rng) -> ndarray`` that
returns ``n`` two-dimensional measurement errors in native field units (metres
for position, m/s for velocity). There is deliberately **no class hierarchy**:
``gaussian`` is the default distribution, and :func:`make_biased_gaussian` is a
higher-order factory that closes over a constant offset. This mirrors the
side-effect-free, small-function approach used throughout the package.

Accuracy is supplied as a 95% confidence interval (``ci95``) and may be a scalar
*or* a per-aircraft array — it is read at draw time so the noise level can change
every tick. Because per-sample covariances are not supported by
``multivariate_normal``, we draw standard-normal samples and scale them.
'''
import numpy as np

# 95% confidence interval -> 1 sigma, for a 2D isotropic distribution. Matches
# the legacy noise model so behaviour is preserved across the refactor.
CI95_TO_STD_2D = 2.448


def ci95_to_std(ci95, n) -> np.ndarray:
    '''Convert a 95% CI (scalar or shape ``(n,)``) to a per-aircraft 1-sigma.

    Returns shape ``(n, 1)`` so it broadcasts cleanly over the two error
    components of an ``(n, 2)`` draw.
    '''
    std = np.asarray(ci95, dtype=float) / CI95_TO_STD_2D
    if std.ndim == 0:
        std = np.full(n, float(std))
    return std.reshape(n, 1)


def gaussian(n, ci95, rng) -> np.ndarray:
    '''Zero-mean isotropic 2D normal error, shape ``(n, 2)``.'''
    if n == 0:
        return np.empty((0, 2))
    z = rng.standard_normal((n, 2))
    return z * ci95_to_std(ci95, n)


def make_biased_gaussian(bias=(0.0, 0.0)):
    '''Return a distribution that adds a constant offset to a Gaussian draw.

    ``bias`` is in native field units, shape ``(2,)``. The offset is a property
    of the *distribution* (a systematic measurement bias), unrelated to time.
    '''
    bias = np.asarray(bias, dtype=float).reshape(1, 2)

    def biased_gaussian(n, ci95, rng) -> np.ndarray:
        return gaussian(n, ci95, rng) + bias

    return biased_gaussian


def make_mixture_gaussian(tail_ratio=3.0, tail_weight=0.1):
    '''Two-component zero-mean isotropic Gaussian mixture with a preserved 2D radial ci95.

    With probability ``(1 - tail_weight)``: draw from N(0, σ₁²I) — dominant component.
    With probability ``tail_weight``:       draw from N(0, σ₂²I) — tail component.
    σ₂ = tail_ratio · σ₁, so the tail is wider by that factor.

    σ₁ is solved numerically so that the 95th percentile of the 2D radial distance
    r = √(e_east² + e_north²) equals ci95 exactly, preserving the same containment
    guarantee as the plain ``gaussian``. The dominant component is therefore tighter
    than σ_iso = ci95/2.448 — it gives headroom for the heavy tail.

    The constraint solved by bisection:

        p · exp(−u) + (1−p) · exp(−u/k²) = 0.05
        where  u = ci95² / (2σ₁²),  k = tail_ratio,  p = 1 − tail_weight

    The result is cached by ci95 value so the bisection only runs once per
    unique accuracy level.

    Can be used as both ``pos_dist`` and ``vel_dist`` in ``make_cns`` — the
    signature is identical to ``gaussian``.
    '''
    if not 0.0 < tail_weight < 1.0:
        raise ValueError(f'tail_weight must be in (0, 1), got {tail_weight}')
    if tail_ratio <= 1.0:
        raise ValueError(f'tail_ratio must be > 1, got {tail_ratio}')

    p = 1.0 - tail_weight
    k = float(tail_ratio)
    _cache = {}

    def _sigma1(ci95_val):
        key = round(ci95_val, 8)
        if key in _cache:
            return _cache[key]
        # Bisect: val(σ₁) is monotonically increasing from 0 to 1 as σ₁ grows.
        # We want val = 0.05 (the 5% tail probability).
        lo, hi = ci95_val * 1e-5, ci95_val
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            u = ci95_val ** 2 / (2.0 * mid ** 2)
            val = p * np.exp(-u) + (1.0 - p) * np.exp(-u / k ** 2)
            if val < 0.05:
                lo = mid
            else:
                hi = mid
        _cache[key] = 0.5 * (lo + hi)
        return _cache[key]

    def mixture_gaussian(n, ci95, rng) -> np.ndarray:
        if n == 0:
            return np.empty((0, 2))
        ci95_arr = np.broadcast_to(np.asarray(ci95, dtype=float), (n,))
        s1 = np.array([_sigma1(float(v)) for v in ci95_arr])
        s2 = k * s1
        use_tail = rng.random(n) < tail_weight
        sigmas = np.where(use_tail, s2, s1).reshape(n, 1)
        return rng.standard_normal((n, 2)) * sigmas

    return mixture_gaussian


def tstudent(n, ci95, rng) -> np.ndarray:  # pragma: no cover - stub
    '''Heavy-tailed draw — not implemented this pass (left as a hook).'''
    raise NotImplementedError('tstudent distribution not yet implemented')


# TODO(correlated): a factory that draws the 4D (pos_x, pos_y, vel_x, vel_y)
#   vector jointly with a cross-covariance block, for correlated position-velocity
#   sampling. Left as a hook for a later pass.
