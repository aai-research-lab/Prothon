"""Distributions from samples, on a line and on a circle.

A comparison needs densities, and a density needs a kernel, a bandwidth and a
grid. All three are different for a circular order parameter, and getting any of
them wrong is quiet rather than loud: a torsion treated as linear produces a
plausible number that is wrong by up to a factor of eighty-five
(``docs/circular.md``).

Kept apart from the comparison itself because it answers a different question.
This module turns a column of numbers into a density; :mod:`prothon.compare.dissimilarity`
decides what the distance between two of them means.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ive
from scipy.stats import gaussian_kde

from ..sampling.statistics import effective_sample_size, validate_weights
from ..utils import get_logger

__all__ = ["estimate_pdf"]

logger = get_logger("density")

#: A column whose range is below this is constant to within floating point.
#: Gaussian KDE on it raises a singular-covariance error, so it is handled.
_CONSTANT_TOLERANCE = 1e-12

#: Bounds on the fitted von Mises concentration. Below the first, the kernel is
#: so broad every distribution looks uniform; above the second, so narrow that
#: the estimate is a comb of spikes at the sample points.
_KAPPA_MIN, _KAPPA_MAX = 0.5, 5.0e3


def _kappa_from_resultant(mean_resultant: float) -> float:
    """Concentration of the von Mises fitted to data with this resultant length.

    Fisher's piecewise approximation to the maximum-likelihood estimate, which
    is accurate enough for a bandwidth rule and avoids iterating on a ratio of
    Bessel functions.
    """
    r = float(np.clip(mean_resultant, 1e-8, 1 - 1e-8))
    if r < 0.53:
        return 2 * r + r**3 + 5 * r**5 / 6
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1 - r)
    return 1.0 / (r**3 - 4 * r**2 + 3 * r)


def _vonmises_bandwidth(angles: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Taylor's plug-in concentration for a von Mises kernel.

    The exponentially scaled Bessel functions ``ive`` are used so the ratio can
    be formed without either factor overflowing: the ``exp`` terms in
    ``I_2(2k) / I_1(k)^2`` cancel exactly.
    """
    # Both the concentration and the sample size are weighted quantities. Using
    # the raw frame count for n would widen the kernel as if every conformation
    # counted, which is the same over-confidence in reverse.
    if weights is None:
        n = float(angles.size)
        resultant = float(np.abs(np.mean(np.exp(1j * angles))))
    else:
        n = effective_sample_size(weights)
        resultant = float(np.abs(np.sum(weights * np.exp(1j * angles))))
    kappa_hat = _kappa_from_resultant(resultant)

    numerator = 3.0 * n * kappa_hat**2 * ive(2, 2 * kappa_hat)
    denominator = 4.0 * np.sqrt(np.pi) * ive(1, kappa_hat) ** 2
    if denominator <= 0 or not np.isfinite(numerator / denominator):
        return float(np.clip(kappa_hat, _KAPPA_MIN, _KAPPA_MAX))

    kappa = (numerator / denominator) ** 0.4
    if not np.isfinite(kappa):
        kappa = kappa_hat
    return float(np.clip(kappa, _KAPPA_MIN, _KAPPA_MAX))


#: Smallest density reported anywhere on a grid. Present only to keep an
#: underflowed kernel from producing an infinite Kullback-Leibler term; it is
#: some three hundred orders of magnitude below any density that matters.
_DENSITY_FLOOR = 1e-300

def _circular_pdf(
    values: np.ndarray, grid: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Von Mises kernel density on the circle.

    ``exp(k*cos(d)) / I_0(k)`` is written as ``exp(k*(cos(d)-1)) / ive(0, k)``,
    which is the same quantity with the overflow removed: at the
    concentrations a tight torsion distribution produces, ``exp(k)`` alone
    exceeds a float64.
    """
    kappa = _vonmises_bandwidth(values, weights)
    deviation = grid[:, None] - values[None, :]
    kernel = np.exp(kappa * (np.cos(deviation) - 1.0))
    if weights is None:
        total = kernel.sum(axis=1) / values.size
    else:
        total = kernel @ weights
    density = total / (2.0 * np.pi * ive(0, kappa))

    # A tight torsion gives a concentration of order a thousand, at which the
    # kernel underflows to exact zero a few grid points from the peak. An
    # exact zero opposite a positive value makes a Kullback-Leibler term
    # infinite, and the Jensen-Shannon distance with it. The floor is far
    # below any density that carries weight and removes the discontinuity.
    return np.maximum(density, _DENSITY_FLOOR)


def _constant_pdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Density for a column that never varies: all mass in the nearest bin.

    A buried residue with zero solvent exposure in every frame is a real and
    common case. Version 2.0 passed it to ``gaussian_kde``, which raised a
    singular-covariance error and took the whole run down.
    """
    density = np.zeros_like(grid)
    density[int(np.argmin(np.abs(grid - values[0])))] = 1.0
    return density


def estimate_pdf(
    arr: np.ndarray,
    x_min: float,
    x_max: float,
    x_num: int,
    circular: bool = False,
    weights=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a one-dimensional density on a fixed grid.

    Parameters
    ----------
    arr
        Sample values for one feature.
    x_min, x_max
        Grid bounds. Ignored when ``circular``, which always spans a full turn:
        a grid that stopped at the observed extremes would cut the circle at an
        arbitrary point and reintroduce the wraparound artefact.
    x_num
        Number of grid points.
    circular
        Whether the values live on a circle.
    weights
        Probability per sample, or ``None`` for uniform. A deposited ensemble
        stores these; treating its conformers as equally likely discards the
        part of the answer that was hardest to obtain.

    Returns
    -------
    grid, density
        Evaluation points and the estimated density at each.
    """
    values = np.asarray(arr, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("Cannot estimate a density from an empty sample.")
    w = validate_weights(weights, values.size)

    if circular:
        grid = np.linspace(-np.pi, np.pi, x_num)
        values = np.mod(values + np.pi, 2 * np.pi) - np.pi
        if values.size < 2:
            return grid, _constant_pdf(values, grid)
        return grid, _circular_pdf(values, grid, w)

    grid = np.linspace(x_min, x_max, x_num)
    if values.size < 2 or float(np.ptp(values)) < _CONSTANT_TOLERANCE:
        return grid, _constant_pdf(values, grid)

    try:
        # SciPy sizes the Silverman factor by the effective sample size when
        # weights are given, which is the behaviour wanted here.
        kde = gaussian_kde(values, bw_method="silverman", weights=w)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
        return grid, _constant_pdf(values, grid)

    # The same floor the circular estimator carries, and for the same reason:
    # a Gaussian kernel underflows to exact zero far enough into the tail, and
    # an exact zero opposite a positive value makes the Jensen-Shannon distance
    # infinite. With the floor in place the distance is always defined, and two
    # genuinely disjoint distributions approach 1 because that is what the
    # arithmetic gives rather than because a fallback said so.
    return grid, np.maximum(kde(grid), _DENSITY_FLOOR)
