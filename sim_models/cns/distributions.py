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


def tstudent(n, ci95, rng) -> np.ndarray:  # pragma: no cover - stub
    '''Heavy-tailed draw — not implemented this pass (left as a hook).'''
    raise NotImplementedError('tstudent distribution not yet implemented')


# TODO(correlated): a factory that draws the 4D (pos_x, pos_y, vel_x, vel_y)
#   vector jointly with a cross-covariance block, for correlated position-velocity
#   sampling. Left as a hook for a later pass.
