"""How much of a trajectory is actually independent, and what follows from it.

The permutation null asks whether the labels carry information: pool the frames
of both ensembles, relabel them at random, and see how far apart the relabelled
groups fall. That argument needs the frames to be exchangeable, and frames from
a molecular dynamics trajectory are not. Consecutive conformations are nearly
the same conformation.

The consequence is measured rather than supposed. With frames from an
Ornstein-Uhlenbeck process at correlation time tau, both ensembles drawn from
the *same* distribution, against a nominal 5%:

    tau = 1 frame     5.5% of features called different
    tau = 5 frames   72.1%
    tau = 20 frames  99.0%
    tau = 50 frames  99.9%

The null is too narrow, and everything clears it.

The fix is to permute what is exchangeable. Whole blocks of consecutive frames,
each long enough to have forgotten its start, can be relabelled without
destroying the correlation inside them -- so the null is built from ensembles
that look like trajectories rather than like independent draws.

Two things follow that are worth stating plainly, because they are costs.

**A trajectory has fewer independent observations than it has frames**, by a
factor of the correlation time. Ten thousand frames at tau = 20 carry about two
hundred and fifty. That number, not the frame count, is what a p-value can be
built on.

**Below a handful of blocks there is no test to run.** A permutation p-value
over six blocks cannot fall below about 1/720 before the correction, and after
correcting across a few hundred residues nothing survives. Prothon reports the
measured floor and declines the p-value rather than printing one that cannot be
supported.

    Sokal, A. Monte Carlo methods in statistical mechanics: foundations and
    new algorithms. In Functional Integration; Springer: Boston, 1997.

    Chodera, J. D. A simple method for automated equilibration detection in
    molecular simulations. J. Chem. Theory Comput. 2016, 12, 1799-1805.

    Politis, D. N.; Romano, J. P. The stationary bootstrap. J. Am. Stat.
    Assoc. 1994, 89, 1303-1313.
"""

from __future__ import annotations

import numpy as np

from ..utils import get_logger

logger = get_logger("correlation")

__all__ = [
    "MINIMUM_BLOCKS",
    "block_labels",
    "correlation_time",
    "effective_frames",
    "plan_blocks",
]

#: Sokal's window criterion. The autocorrelation sum is truncated at the first
#: window w with ``w >= c * tau(w)``, which balances the bias of stopping early
#: against the variance of summing noise at long lag. 5 is the usual choice.
SOKAL_WINDOW = 5.0

#: Blocks are this many correlation times long. Two is enough for a block to
#: have largely forgotten how it started, and short blocks are worth more than
#: long ones because the number of them is what the null is built from.
BLOCK_MULTIPLIER = 2.0

#: Below this many blocks the permutation null has too few distinct
#: arrangements to give a usable p-value, and Prothon declines to report one.
#: With 8 blocks there are 35 balanced splits; the resolution before any
#: multiplicity correction is already coarser than a 5% threshold.
MINIMUM_BLOCKS = 8

#: Features whose values never change carry no autocorrelation and would
#: otherwise contribute a meaningless estimate to the summary.
_CONSTANT_TOLERANCE = 1e-12


def _autocorrelation_time(series: np.ndarray, c: float = SOKAL_WINDOW) -> float:
    """Integrated autocorrelation time of one series, in frames.

    ``tau_int = 1 + 2 * sum_k rho(k)``, truncated by Sokal's window. The
    autocorrelation is computed through an FFT, which is O(n log n) rather than
    the O(n^2) of the direct sum and matters at the trajectory lengths this is
    meant for.

    Returns 1.0 -- meaning uncorrelated -- for a series too short or too flat
    to say anything about.
    """
    x = np.asarray(series, dtype=np.float64)
    n = x.size
    if n < 16:
        return 1.0
    x = x - x.mean()
    variance = float(np.dot(x, x))
    if variance < _CONSTANT_TOLERANCE:
        return 1.0

    padded = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, padded)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), padded)[:n].real
    acf /= acf[0]

    taus = 2.0 * np.cumsum(acf) - 1.0
    window = np.nonzero(np.arange(n) >= c * taus)[0]
    if window.size:
        tau = taus[window[0]]
    else:
        # The window never closed: the correlation time is at least comparable
        # to the series, and the sum has run out of data before running out of
        # correlation. Anything reported here is a lower bound, and saying so
        # is what lets the caller notice rather than trust a saturated number.
        tau = max(taus[-1], n / SOKAL_WINDOW)
    # A negative estimate means the series is anticorrelated or the sum ran
    # into noise; either way there is nothing to correct for.
    return float(max(1.0, tau))


def correlation_time(
    matrix: np.ndarray, quantile: float = 0.75, max_features: int = 200
) -> float:
    """Correlation time of a representation, in frames.

    Estimated per feature and summarised across features. The choice of
    summary matters more than it looks, because the per-feature estimates are
    noisy: on 2000 frames of a system whose true integrated correlation time is
    40, the median across features lands near 33 while the 90th percentile
    lands between 53 and 101 and *grows with the number of features*. A high
    quantile over a few hundred noisy estimates is chasing the worst estimate
    rather than the slowest residue, and the difference is not visible in the
    number that comes out.

    The default is the upper quartile: above the median, so a protein's many
    rigid residues cannot hide its slower ones, and far enough from the tail
    that it estimates the protein rather than the estimator.

    Parameters
    ----------
    matrix
        ``(n_frames, n_features)`` representation, in frame order. **The row
        order must be the order the frames were generated in** -- a shuffled
        or concatenated matrix has no correlation time to estimate, and this
        will report 1.0 and quietly disable the correction.
    quantile
        Which quantile of the per-feature times to take.
    max_features
        Estimate on at most this many features, evenly spaced. The estimate is
        a summary and does not need every residue.
    """
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    n_frames, n_features = matrix.shape
    if n_frames < 16:
        return 1.0

    columns = (
        np.arange(n_features)
        if n_features <= max_features
        else np.linspace(0, n_features - 1, max_features).astype(int)
    )
    times = np.array([_autocorrelation_time(matrix[:, i]) for i in columns])
    tau = float(np.quantile(times, quantile))
    logger.debug(
        "correlation time: median %.1f, q%.0f %.1f, max %.1f frames",
        np.median(times), 100 * quantile, tau, times.max(),
    )
    return tau


def effective_frames(n_frames: int, tau: float) -> float:
    """Independent observations in ``n_frames`` frames at correlation time tau.

    The standard result for a correlated series: ``n / tau_int``. At tau = 20,
    ten thousand frames are worth five hundred.
    """
    return float(n_frames / max(1.0, tau))


def plan_blocks(n_frames: int, tau: float, multiplier: float = BLOCK_MULTIPLIER):
    """Block length and count for a trajectory of this length and correlation.

    Returns ``(block_length, n_blocks)``. The block is ``multiplier * tau``
    frames and the count is whatever that gives.

    **The block is never shortened to manufacture more of them.** An earlier
    version capped the length so that :data:`MINIMUM_BLOCKS` blocks always
    existed, which is the one thing that must not happen: a block shorter than
    the correlation time does not contain the correlation, so the null it
    builds is the frame-permutation null wearing a block-shaped label, and the
    count that was supposed to reveal the problem is exactly the count that was
    forced to look healthy. A short trajectory of a slow system returns few
    blocks, and the caller refuses.
    """
    if tau <= 1.0:
        return 1, n_frames
    length = max(1, int(np.ceil(multiplier * tau)))
    return length, max(0, n_frames // length)


def block_labels(n_frames: int, block_length: int) -> np.ndarray:
    """Which block each frame belongs to.

    A trailing partial block is merged into the one before it rather than kept,
    so every block is at least ``block_length`` long and none is a stub whose
    correlation structure is unrepresentative.
    """
    if block_length <= 1:
        return np.arange(n_frames)
    labels = np.arange(n_frames) // block_length
    last = labels[-1]
    if np.count_nonzero(labels == last) < block_length and last > 0:
        labels[labels == last] = last - 1
    return labels
