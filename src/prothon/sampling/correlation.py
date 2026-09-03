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

from dataclasses import dataclass, field

import numpy as np

from ..utils import get_logger

logger = get_logger("correlation")

__all__ = [
    "MINIMUM_BLOCKS",
    "PLATEAU_PREFIXES",
    "PLATEAU_SLOPE",
    "CorrelationEstimate",
    "CorrelationProfile",
    "block_labels",
    "correlation_profile",
    "correlation_time",
    "correlation_time_estimate",
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

#: Slope of log(tau_hat) against log(n) below which the estimate counts as
#: settled. A converged estimator has slope zero: the answer does not depend on
#: how much data you gave it. A saturating one climbs, and in the limit where
#: the autocorrelation sum never closes its window the estimate is proportional
#: to n, giving slope one. 0.15 allows a shallow climb from noise.
#:
#: **A ratio between two prefixes is not enough**, which was learned the
#: expensive way. The per-feature estimates are noisy and their quantile dips;
#: on prefixes of a real trajectory the sequence ran 5, 17, 21, 19, 30, 45, and
#: the single dip from 21 to 19 made a two-point ratio report a plateau in the
#: middle of a clear climb. A slope over several prefixes cannot be fooled by
#: one point.
PLATEAU_SLOPE = 0.15

#: Prefixes to estimate on, as fractions of the trajectory. Four points spanning
#: a factor of eight is enough for a slope and cheap: the three short ones cost
#: less together than the full-length estimate.
PLATEAU_PREFIXES = (0.125, 0.25, 0.5, 1.0)

#: Below this many frames an estimate is too poor to contribute to the slope.
_PLATEAU_MINIMUM = 64

#: Features whose values never change carry no autocorrelation and would
#: otherwise contribute a meaningless estimate to the summary.
_CONSTANT_TOLERANCE = 1e-12

#: A second population must be this much slower than both the median feature
#: and the ordinary upper-quartile summary before it changes the global block
#: plan. Smaller separations are already represented reasonably by q75; this
#: guard is for a genuinely distinct slow region, not the noisy upper tail of
#: one homogeneous population.
SLOW_TAIL_RATIO = 3.0

#: Robust outlier boundary on log correlation times. Four scaled MADs keeps a
#: homogeneous population on its stable q75 summary even when one FFT estimate
#: wanders upward.
SLOW_TAIL_MAD_MULTIPLIER = 4.0

#: One extreme column can be estimator noise. Two independently estimated
#: columns are the smallest coherent slow region that may alter every block.
MINIMUM_SLOW_FEATURES = 2


@dataclass(frozen=True)
class CorrelationProfile:
    """Per-feature evidence behind one correlation-time summary.

    Feature indices are zero-based matrix-column indices. Constant and
    non-finite columns are excluded before feature subsampling: they carry no
    autocorrelation estimate and therefore must not vote the summary down.
    ``slow_features`` is non-empty only when a coherent, separated slow group
    changed the plan; isolated high estimates remain visible in
    ``feature_times`` but do not dictate every block.
    """

    tau: float
    quantile_tau: float
    median_tau: float
    summary: str
    assessable_features: tuple[int, ...] = ()
    sampled_features: tuple[int, ...] = ()
    slow_features: tuple[int, ...] = ()
    feature_times: tuple[float, ...] = ()

    @property
    def n_assessable_features(self) -> int:
        return len(self.assessable_features)

    @property
    def n_sampled_features(self) -> int:
        return len(self.sampled_features)


@dataclass
class CorrelationEstimate:
    """A correlation time, and whether it can be believed.

    ``tau`` alone cannot be checked. A saturated estimate is a plausible
    number, in the right units, smaller than the truth, and there is nothing
    about it that looks wrong. The only way to find out is to estimate it again
    on less data and see whether it moved.

    Attributes
    ----------
    tau
        The estimate on the whole matrix, in frames.
    converged
        Whether ``tau`` stopped growing between half the frames and all of
        them. False means it is a **lower bound**, and everything computed from
        it -- the effective sample size, the block count -- is correspondingly
        an *upper* bound, which is the optimistic direction.
    prefix_taus
        The estimate at each prefix, keyed by frame count, so a caller can see
        the trend rather than take the verdict on trust.
    slope
        Slope of ``log tau`` against ``log n`` across the prefixes. Zero means
        the answer does not depend on how much data it was given, which is what
        a converged estimate looks like. One means the estimate is reporting
        the trajectory length rather than the correlation.
    growth
        ``tau(n) / tau(n/2)``, kept for reporting. **The verdict does not rest
        on it**: two points cannot tell a dip from a plateau.
    summary, n_assessable_features, n_sampled_features, slow_features
        Audit trail for the cross-feature summary. ``slow_features`` contains
        zero-based matrix columns only when a coherent slow group, rather than
        the ordinary upper quartile, selected ``tau``.
    """

    tau: float
    converged: bool = True
    prefix_taus: dict[int, float] = field(default_factory=dict)
    growth: float = 1.0
    slope: float = 0.0
    summary: str = "upper quartile"
    assessable_features: tuple[int, ...] = ()
    sampled_features: tuple[int, ...] = ()
    slow_features: tuple[int, ...] = ()

    @property
    def n_assessable_features(self) -> int:
        return len(self.assessable_features)

    @property
    def n_sampled_features(self) -> int:
        return len(self.sampled_features)

    def __float__(self) -> float:
        return float(self.tau)


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


def _assessable_columns(matrix: np.ndarray, circular: bool = False) -> np.ndarray:
    """Columns with finite values and enough variation to estimate a time."""
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.array([], dtype=int)
    finite = np.all(np.isfinite(matrix), axis=0)
    # This is the same energy test used by _autocorrelation_time, evaluated in
    # one pass so constants can be removed *before* the feature limit is
    # applied. Otherwise 190 constants can crowd slow columns out of a
    # 200-column sample and also vote as tau=1 in its quantile.
    finite_columns = np.flatnonzero(finite)
    if finite_columns.size == 0:
        return finite_columns
    values = matrix[:, finite_columns]
    if circular:
        # -pi and +pi are the same value. Raw linear variance would call a
        # trajectory alternating across that boundary highly variable even
        # when it occupies one narrow angular state.
        energy = np.maximum(
            np.var(np.sin(values), axis=0),
            np.var(np.cos(values), axis=0),
        ) * matrix.shape[0]
    else:
        energy = np.var(values, axis=0) * matrix.shape[0]
    return finite_columns[energy >= _CONSTANT_TOLERANCE]


def _feature_correlation_time(series: np.ndarray, circular: bool) -> float:
    if not circular:
        return _autocorrelation_time(series)
    # The sine/cosine embedding respects angular equivalence at the branch
    # cut. Taking the slower component protects both angular degrees of
    # freedom without unwrapping, which would invent a history of full turns.
    return max(
        _autocorrelation_time(np.sin(series)),
        _autocorrelation_time(np.cos(series)),
    )


def correlation_profile(
    matrix: np.ndarray,
    quantile: float = 0.75,
    max_features: int = 200,
    circular: bool = False,
) -> CorrelationProfile:
    """Correlation-time summary and the feature evidence behind it.

    Estimated per feature and summarised across features. The choice of
    summary matters more than it looks, because the per-feature estimates are
    noisy: on 2000 frames of a system whose true integrated correlation time is
    40, the median across features lands near 33 while the 90th percentile
    lands between 53 and 101 and *grows with the number of features*. A high
    quantile over a few hundred noisy estimates is chasing the worst estimate
    rather than the slowest residue, and the difference is not visible in the
    number that comes out.

    The default starts at the upper quartile: above the median and far enough
    from the tail that it estimates the protein rather than the estimator.
    An upper quartile alone has another failure, however: any coherent slow
    region smaller than one quarter of the representation is mathematically
    invisible to it. Prothon therefore also looks for a distinct upper group
    on log correlation times. At least two features must lie beyond both a
    threefold separation and four scaled median absolute deviations, and their
    median must be at least three times q75. When those conditions hold, the
    slow-group median sets the block plan. This protects a small slow loop
    without making one noisy column -- or the noisiest column in a larger
    protein -- dictate every block.

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
        Estimate on at most this many *assessable* features, evenly spaced.
        Constant and non-finite features are removed first.
    circular
        Treat feature values as angles in radians. Correlation is estimated on
        their sine/cosine embedding so crossing -pi/+pi is not mistaken for a
        decorrelating jump.
    """
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    if matrix.ndim != 2:
        raise ValueError("A correlation profile requires a 2-D matrix.")
    n_frames = matrix.shape[0]
    if not isinstance(max_features, (int, np.integer)) or max_features < 1:
        raise ValueError("max_features must be a positive integer.")
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie between zero and one.")

    assessable = _assessable_columns(matrix, circular)
    if n_frames < 16 or assessable.size == 0:
        return CorrelationProfile(
            tau=1.0,
            quantile_tau=1.0,
            median_tau=1.0,
            summary="no assessable features",
            assessable_features=tuple(int(i) for i in assessable),
        )

    columns = (
        assessable
        if assessable.size <= max_features
        else assessable[
            np.linspace(0, assessable.size - 1, max_features).astype(int)
        ]
    )
    times = np.array([
        _feature_correlation_time(matrix[:, i], circular) for i in columns
    ])
    quantile_tau = float(np.quantile(times, quantile))
    median_tau = float(np.median(times))

    log_times = np.log(np.maximum(times, 1.0))
    log_median = float(np.median(log_times))
    log_mad = float(np.median(np.abs(log_times - log_median)))
    slow_boundary = np.exp(
        log_median
        + max(
            np.log(SLOW_TAIL_RATIO),
            SLOW_TAIL_MAD_MULTIPLIER * 1.4826 * log_mad,
        )
    )
    slow_mask = times >= slow_boundary
    slow_times = times[slow_mask]
    slow_tau = (
        float(np.median(slow_times))
        if slow_times.size >= MINIMUM_SLOW_FEATURES
        else quantile_tau
    )
    protects_slow_group = bool(
        slow_times.size >= MINIMUM_SLOW_FEATURES
        and slow_tau >= SLOW_TAIL_RATIO * quantile_tau
    )
    tau = slow_tau if protects_slow_group else quantile_tau
    slow_features = columns[slow_mask] if protects_slow_group else np.array([], dtype=int)
    ordinary_summary = (
        "upper quartile"
        if quantile == 0.75
        else f"q{100.0 * quantile:g}"
    )
    summary = "coherent slow-feature median" if protects_slow_group else ordinary_summary
    logger.debug(
        "correlation time: median %.1f, q%.0f %.1f, max %.1f; %s %.1f frames",
        median_tau, 100 * quantile, quantile_tau, times.max(), summary, tau,
    )
    return CorrelationProfile(
        tau=float(tau),
        quantile_tau=quantile_tau,
        median_tau=median_tau,
        summary=summary,
        assessable_features=tuple(int(i) for i in assessable),
        sampled_features=tuple(int(i) for i in columns),
        slow_features=tuple(int(i) for i in slow_features),
        feature_times=tuple(float(value) for value in times),
    )


def correlation_time(
    matrix: np.ndarray,
    quantile: float = 0.75,
    max_features: int = 200,
    circular: bool = False,
) -> float:
    """Correlation time of a representation, in frames.

    This compatibility wrapper returns the scalar selected by
    :func:`correlation_profile`. Use that function when the assessable and slow
    feature sets are needed for an audit trail.
    """
    return correlation_profile(matrix, quantile, max_features, circular).tau


def correlation_time_estimate(
    matrix: np.ndarray,
    quantile: float = 0.75,
    max_features: int = 200,
    tolerance: float = PLATEAU_SLOPE,
    circular: bool = False,
) -> CorrelationEstimate:
    """Correlation time, plus whether the trajectory was long enough to find it.

    The estimate is a sum of the autocorrelation function, and on a short
    series the sum runs out of data before it runs out of correlation. What
    comes back is then a plausible number in the right units that is smaller
    than the truth, with nothing about it that looks wrong.

    **A ratio test on the estimate itself does not catch this.** Comparing
    ``n / tau_hat`` against a threshold puts the saturated value in the
    denominator, so the worse the saturation the healthier the ratio looks. On
    prefixes of a real trajectory whose settled value is 45 frames, a
    250-frame prefix returns 5 and the ratio reads 50, comfortably above any
    threshold, while the true ratio is 5.6. The test fires when saturation is
    mild and passes when it is severe, which is the wrong way round.

    So this estimates twice, on half the frames and on all of them, and asks
    whether the answer moved. That is the criterion Flyvbjerg and Petersen give
    for block averaging -- the curve must reach a plateau before the number is
    accepted -- applied to the quantity rather than to a proxy for it.

    Parameters
    ----------
    matrix
        ``(n_frames, n_features)`` representation, **in frame order**.
    quantile, max_features, circular
        As :func:`correlation_time`.
    tolerance
        Growth between the half and the whole that still counts as settled.

    Returns
    -------
    CorrelationEstimate
        ``converged=False`` means ``tau`` is a lower bound. Everything derived
        from it is then an upper bound: a block count that clears
        :data:`MINIMUM_BLOCKS` on a lower-bound tau has not necessarily cleared
        it on the true one.
    """
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    n_frames = matrix.shape[0]
    profile = correlation_profile(matrix, quantile, max_features, circular)
    tau = profile.tau

    lengths = sorted(
        {
            int(n_frames * fraction)
            for fraction in PLATEAU_PREFIXES
            if int(n_frames * fraction) >= _PLATEAU_MINIMUM
        }
    )
    if len(lengths) < 3:
        # Too short to establish a trend. Nothing can be said about
        # convergence, and claiming it would be the one answer certainly
        # unearned.
        return CorrelationEstimate(
            tau=tau, converged=False,
            prefix_taus={n_frames: tau}, growth=float("inf"), slope=float("inf"),
            summary=profile.summary,
            assessable_features=profile.assessable_features,
            sampled_features=profile.sampled_features,
            slow_features=profile.slow_features,
        )

    taus = {
        length: (
            tau if length == n_frames
            else correlation_time(
                matrix[:length], quantile, max_features, circular
            )
        )
        for length in lengths
    }

    # Slope of log tau against log n. Zero is settled; one is an estimate that
    # is simply reporting how much data it was given.
    x = np.log(np.array(lengths, dtype=np.float64))
    y = np.log(np.maximum([taus[length] for length in lengths], 1e-12))
    slope = float(np.polyfit(x, y, 1)[0])
    converged = bool(slope <= tolerance)

    largest_half = max(length for length in lengths if length < n_frames)
    growth = tau / max(taus[largest_half], 1e-12)

    logger.debug(
        "correlation time: %s -> slope %.2f (%s)",
        {k: round(v, 1) for k, v in taus.items()}, slope,
        "settled" if converged else "still rising",
    )
    return CorrelationEstimate(
        tau=tau, converged=converged, prefix_taus=taus,
        growth=float(growth), slope=slope,
        summary=profile.summary,
        assessable_features=profile.assessable_features,
        sampled_features=profile.sampled_features,
        slow_features=profile.slow_features,
    )


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
