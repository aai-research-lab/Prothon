"""How far apart two ensembles are, and how much of that is real.

Each column of an ensemble matrix is a sample from the distribution of one
local order parameter. Turning that sample into a density and comparing the
densities of two ensembles with the Jensen-Shannon distance gives a bounded
number per residue -- 0 for identical distributions, 1 for disjoint ones --
whose mean over residues is the global dissimilarity.

The number on its own is not a result. Two independent halves of a *single*
ensemble also have a non-zero Jensen-Shannon distance, because a finite sample
of a continuous distribution never reproduces it exactly. That self-distance is
the resolution limit of the comparison: a difference between two ensembles
smaller than it is not a difference, it is sampling noise. Prothon measures it
by resampling within each ensemble and reports it alongside every result, and
declares a residue different only where the between-ensemble distances exceed
the within-ensemble ones by more than chance.

Four things changed here in 2.1, and all four can alter published numbers:

**The null distribution is now a permutation null.** Version 2.0 built its
null by drawing two bootstrap resamples from the *same* ensemble and measuring
the distance between them. Two resamples of n frames drawn with replacement
from the same n frames share about 63% of their points, so they resemble each
other far more than two independent samples of the same size do -- measured on
a 400-frame Gaussian ensemble, the bootstrap null averages 0.046 where two
genuinely independent samples of the same distribution average 0.097. The null
is too tight by roughly a factor of two, and any honest between-ensemble
distance clears it. Two independent draws from an identical distribution are
reported as differing significantly at every residue.

The replacement is the standard test for this question: pool the frames of both
ensembles, relabel them at random into two groups of the original sizes, and
measure the distance. That is the exact distribution of the statistic when the
ensembles are the same, and it needs no assumption about the shape of anything.


**Torsions are estimated on a circle.** A Gaussian kernel on a linear grid
treats -179 and +179 degrees as far apart, splitting a single population across
the ends of the grid and putting a false trough at the wraparound. Circular
measures now use a von Mises kernel with Taylor's plug-in bandwidth.

**Significance is decided per residue.** Version 2.0 ran one Mann-Whitney test
on the pooled distances and then wrote ``local[p >= 0.05] = 0`` with a scalar
``p``, which NumPy interprets as a mask over the whole array: one test decided
the fate of every residue at once. Each residue is now tested against its own
noise, and the resulting p-values are corrected for multiplicity, because a
300-residue protein tested at 0.05 yields fifteen false positives by
construction.

**The test is one-sided.** The hypothesis is that between-ensemble distances
*exceed* within-ensemble ones. A two-sided test spends half its power on the
impossible alternative that two ensembles resemble each other more than an
ensemble resembles itself.

Passing ``legacy=True`` reproduces the 2.0 behaviour exactly, for anyone
regenerating a published figure.

    Taylor, C. C. Automatic bandwidth selection for circular density
    estimation. Comput. Stat. Data Anal. 2008, 52, 3493-3500.

    Benjamini, Y.; Hochberg, Y. Controlling the false discovery rate.
    J. R. Stat. Soc. B 1995, 57, 289-300.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import ive
from scipy.stats import gaussian_kde, mannwhitneyu

from ..utils import get_logger

logger = get_logger("dissimilarity")

__all__ = [
    "ComparisonResult",
    "benjamini_hochberg",
    "effective_sample_size",
    "dissimilarity",
    "estimate_pdf",
    "jsd_local",
    "random_sample",
]

#: Default number of frames drawn from each ensemble for the test. Larger
#: ensembles are subsampled to this, without replacement, because the null
#: needs many repeats and the density estimate is the expensive part.
DEFAULT_SAMPLE_SIZE = 1000

#: Default relabellings used to build the null. The per-feature p-values are
#: pooled across features after studentising (see :func:`_studentised_p_values`),
#: so the resolution is roughly ``1 / (n_permutations * n_features)`` rather
#: than ``1 / n_permutations`` -- which is what makes 100 enough to survive a
#: false-discovery-rate correction over a few hundred residues.
DEFAULT_PERMUTATIONS = 100

#: Below this many *effective* samples, a resampled density is mostly repeats
#: of the same few conformations and the noise floor is optimistic. Warned
#: about, not refused: a 20-model NMR ensemble is a legitimate thing to
#: compare, as long as nobody mistakes its error bars for those of a long
#: simulation.
MIN_EFFECTIVE_SAMPLES = 50.0

#: Below this many effective samples the comparison is refused. A weighted
#: ensemble in which one conformer carries most of the probability has the
#: information content of a handful of structures however many frames it
#: holds, and a per-residue profile computed from it would be a description of
#: those few structures wearing the name of an ensemble.
MINIMUM_EFFECTIVE_SAMPLES = 10.0

#: A column whose range is below this is constant to within floating point.
#: Gaussian KDE on it raises a singular-covariance error, so it is handled.
_CONSTANT_TOLERANCE = 1e-12

#: Bounds on the fitted von Mises concentration. Below the first, the kernel is
#: so broad every distribution looks uniform; above the second, so narrow that
#: the estimate is a comb of spikes at the sample points.
_KAPPA_MIN, _KAPPA_MAX = 0.5, 5.0e3


@dataclass
class ComparisonResult:
    """One ensemble measured against the reference.

    Behaves like the dictionary version 2.0 returned -- ``result["p_value"]``
    and ``result["global_dissimilarity"]`` still work -- so existing scripts
    keep running, while new code can use attributes and the fields that did
    not exist before.

    Attributes
    ----------
    global_dissimilarity
        Mean local dissimilarity over features, after non-significant features
        have been set to zero.
    local_dissimilarity
        Per-feature dissimilarity, masked.
    raw_local_dissimilarity
        Per-feature dissimilarity before masking. Worth plotting alongside the
        masked version: a residue just under the threshold looks identical to
        one that genuinely does not move, and only the raw values distinguish
        them.
    p_values
        Per-feature p-values after false-discovery-rate correction. In legacy
        mode this is the single pooled p-value broadcast across features.
    feature_index
        Where each surviving feature sits on the *reference* ensemble,
        one-based. After reconciliation the columns are a subset of the
        reference's, and plotting them at 1..n would silently renumber the
        protein.
    noise_floor
        Mean within-ensemble Jensen-Shannon distance: the smallest difference
        this much sampling could resolve. A global dissimilarity below it is
        not evidence of anything.
    resolved
        Whether the global dissimilarity exceeds the noise floor at all.
    """

    ensemble_index: int
    reference_index: int
    global_dissimilarity: float
    local_dissimilarity: np.ndarray
    raw_local_dissimilarity: np.ndarray
    p_values: np.ndarray
    significant: np.ndarray
    noise_floor: float
    n_frames: tuple[int, int]
    #: Kish effective sample size of each ensemble. Equal to the frame count
    #: when unweighted; much smaller when a few conformers carry the mass.
    effective_samples: tuple[float, float] = (0.0, 0.0)
    measure: str = ""
    #: Position of each feature on the reference ensemble, one-based. Set when
    #: the ensembles were reconciled and the columns are a subset; ``None``
    #: means every column is present and the index is simply 1..n.
    feature_index: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def p_value(self) -> float:
        """Smallest per-feature p-value, for the 2.0 dictionary key."""
        return float(np.min(self.p_values)) if self.p_values.size else 1.0

    @property
    def resolved(self) -> bool:
        return bool(self.global_dissimilarity > self.noise_floor)

    @property
    def n_significant(self) -> int:
        return int(np.count_nonzero(self.significant))

    def __getitem__(self, key: str) -> Any:
        """Dictionary access, so 2.0 code indexing the result still works."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the JSON manifest, arrays flattened to lists."""
        return {
            "ensemble_index": self.ensemble_index,
            "reference_index": self.reference_index,
            "measure": self.measure,
            "global_dissimilarity": float(self.global_dissimilarity),
            "local_dissimilarity": self.local_dissimilarity.tolist(),
            "raw_local_dissimilarity": self.raw_local_dissimilarity.tolist(),
            "p_values": self.p_values.tolist(),
            "significant": self.significant.tolist(),
            "n_significant": self.n_significant,
            "noise_floor": float(self.noise_floor),
            "resolved": self.resolved,
            "n_frames": list(self.n_frames),
            "effective_samples": list(self.effective_samples),
            "feature_index": (
                None if self.feature_index is None else self.feature_index.tolist()
            ),
            **self.metadata,
        }


def effective_sample_size(weights=None, n: int | None = None) -> float:
    """How many independent conformations a weighted ensemble is worth.

    Kish's formula, ``(sum w)^2 / sum(w^2)``. Equal weights give back the frame
    count; concentrated weights give much less. A thousand frames in which one
    conformer carries half the probability is worth about four independent
    samples, and sizing a noise floor by the frame count instead would produce
    error bars for an ensemble nobody sampled.

    This is the same failure as the bootstrap null the 2.1 release replaced --
    a quantity that looks like a sample size, is smaller than it appears, and
    makes everything downstream look more certain than it is.

        Kish, L. Survey Sampling; Wiley: New York, 1965.
    """
    if weights is None:
        if n is None:
            raise ValueError("Give either weights or a frame count.")
        return float(n)
    w = np.asarray(weights, dtype=np.float64).ravel()
    total = w.sum()
    if total <= 0:
        return 0.0
    return float(total**2 / np.sum(w**2))


def _normalise(weights, n: int) -> np.ndarray | None:
    """Weights summing to one, or ``None`` for uniform."""
    if weights is None:
        return None
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != n:
        raise ValueError(f"{w.size} weights for {n} frames.")
    total = w.sum()
    if total <= 0:
        raise ValueError("Weights sum to zero.")
    return w / total


def random_sample(
    arr: np.ndarray,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw frames with replacement.

    A generator can be supplied so a run is reproducible; version 2.0 drew from
    the global NumPy state, which meant two runs of the same study produced
    different p-values and nothing recorded why.
    """
    rng = np.random.default_rng() if rng is None else rng
    indices = rng.integers(0, arr.shape[0], sample_size)
    return arr[indices, :]


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
    return total / (2.0 * np.pi * ive(0, kappa))


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
    w = _normalise(weights, values.size)

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
    return grid, kde(grid)


def jsd_local(
    ensemble1: np.ndarray,
    ensemble2: np.ndarray,
    x_min: float,
    x_max: float,
    x_num: int,
    circular: bool = False,
    weights1=None,
    weights2=None,
) -> np.ndarray:
    """Jensen-Shannon distance per feature between two representation matrices.

    Returns one value per column, in [0, 1]. Base-2 logarithms, so the bound is
    1 for distributions with disjoint support.
    """
    if ensemble1.shape[1] != ensemble2.shape[1]:
        raise ValueError(
            f"Cannot compare representations with different feature counts: "
            f"{ensemble1.shape[1]} and {ensemble2.shape[1]}."
        )

    n_features = ensemble1.shape[1]
    distances = np.zeros(n_features, dtype=np.float64)
    for i in range(n_features):
        _, pdf1 = estimate_pdf(ensemble1[:, i], x_min, x_max, x_num, circular, weights1)
        _, pdf2 = estimate_pdf(ensemble2[:, i], x_min, x_max, x_num, circular, weights2)
        value = jensenshannon(pdf1, pdf2, base=2)
        distances[i] = 0.0 if not np.isfinite(value) else value
    return distances


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, controlling the false discovery rate.

    Written out rather than taken from SciPy so that Prothon keeps working on
    the SciPy versions that predate ``false_discovery_control``.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    scaled = p[order] * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downwards, so an adjusted value
    # can never fall below one ranked above it.
    stepped = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty_like(stepped)
    adjusted[order] = np.clip(stepped, 0.0, 1.0)
    return adjusted


def _subsample(
    arr: np.ndarray, size: int, rng: np.random.Generator, weights=None
):
    """Take at most ``size`` frames, without replacement.

    Without replacement matters: the whole point of the permutation null is
    that the two groups are disjoint, and drawing with replacement would put
    the same frame on both sides.
    """
    if arr.shape[0] <= size:
        return arr, weights
    keep = rng.choice(arr.shape[0], size, replace=False)
    return arr[keep], (None if weights is None else weights[keep] / weights[keep].sum())


def _permutation_null(
    reference: np.ndarray,
    other: np.ndarray,
    x_min: float,
    x_max: float,
    x_num: int,
    circular: bool,
    n_permutations: int,
    rng: np.random.Generator,
    weights_a=None,
    weights_b=None,
) -> np.ndarray:
    """Distances between two groups formed by relabelling the pooled frames.

    Under the hypothesis that both ensembles sample the same distribution, the
    labels carry no information, so any relabelling is as likely as the one
    observed. The distances obtained from many relabellings are the null
    distribution of the statistic, and no assumption about its shape is needed.

    Returns ``(n_permutations, n_features)``.
    """
    n_reference = reference.shape[0]
    pooled = np.vstack([reference, other])
    total = pooled.shape[0]

    # A frame and its weight are one observation. Under the null they are
    # exchangeable together; permuting frames while leaving weights in place
    # would attach each conformation's probability to a different structure.
    weighted = weights_a is not None or weights_b is not None
    pooled_w = (
        np.concatenate([
            np.full(n_reference, 1.0 / n_reference) if weights_a is None else weights_a,
            np.full(total - n_reference, 1.0 / (total - n_reference))
            if weights_b is None else weights_b,
        ])
        if weighted else None
    )

    null = np.empty((n_permutations, reference.shape[1]), dtype=np.float64)
    for k in range(n_permutations):
        order = rng.permutation(total)
        left, right = order[:n_reference], order[n_reference:]
        wl = wr = None
        if weighted:
            wl, wr = pooled_w[left], pooled_w[right]
            wl, wr = wl / wl.sum(), wr / wr.sum()
        null[k] = jsd_local(
            pooled[left], pooled[right], x_min, x_max, x_num, circular, wl, wr
        )
    return null


def _studentised_p_values(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Pooled permutation p-values, FDR-corrected.

    A per-feature p-value from ``n`` relabellings cannot fall below
    ``1/(n+1)``, and after correcting across a few hundred residues nothing at
    that resolution survives -- so a naive permutation test would need tens of
    thousands of relabellings to detect anything.

    Standardising each feature by its own null mean and spread puts every
    feature on one scale, at which point the null values from all features can
    be pooled into a single reference distribution. The resolution becomes
    ``1/(n_permutations * n_features + 1)``, which is ample.

    The assumption this rests on is that the standardised null is comparable
    across features. For Jensen-Shannon distances computed from equal sample
    sizes on a shared grid that holds well; it would not hold if features had
    wildly different sample sizes.
    """
    mean = null.mean(axis=0)
    spread = null.std(axis=0, ddof=1)
    spread = np.where(spread < 1e-12, 1e-12, spread)

    z_observed = (observed - mean) / spread
    z_null = np.sort(((null - mean) / spread).ravel())

    total = z_null.size
    at_least_as_extreme = total - np.searchsorted(z_null, z_observed, side="left")
    raw = (1.0 + at_least_as_extreme) / (total + 1.0)
    return benjamini_hochberg(np.nan_to_num(raw, nan=1.0))


def _split_half_floor(
    ensembles: tuple[np.ndarray, ...],
    x_min: float,
    x_max: float,
    x_num: int,
    circular: bool,
    repeats: int,
    rng: np.random.Generator,
    weights: tuple = (None, None),
) -> np.ndarray:
    """Distance between two disjoint halves of a single ensemble.

    This is the resolution limit: two independent samples of the same
    distribution, at half the sampling the study actually has. A difference
    between two ensembles smaller than this is not a difference.

    Disjoint halves, not bootstrap resamples -- that distinction is exactly
    what version 2.0 got wrong, and it is worth roughly a factor of two.
    """
    values = []
    for ensemble, w in zip(ensembles, weights):
        half = ensemble.shape[0] // 2
        if half < 2:
            continue
        for _ in range(repeats):
            order = rng.permutation(ensemble.shape[0])
            left, right = order[:half], order[half : 2 * half]
            wl = wr = None
            if w is not None:
                wl, wr = w[left], w[right]
                wl, wr = wl / wl.sum(), wr / wr.sum()
            values.append(
                jsd_local(
                    ensemble[left], ensemble[right],
                    x_min, x_max, x_num, circular, wl, wr,
                )
            )
    if not values:
        return np.zeros((1, ensembles[0].shape[1]))
    return np.stack(values, axis=0)


def _legacy_bootstrap(
    reference: np.ndarray,
    other: np.ndarray,
    x_min: float,
    x_max: float,
    x_num: int,
    s_num: int,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Version 2.0's bootstrap, kept so published figures can be regenerated.

    Retained deliberately unchanged, including the too-tight within-ensemble
    null. Reachable only through ``legacy=True``.
    """
    between = [
        jsd_local(
            random_sample(reference, sample_size, rng),
            random_sample(other, sample_size, rng),
            x_min, x_max, x_num, False,
        )
        for _ in range(s_num)
        for _ in range(s_num)
    ]
    within = []
    for ensemble in (reference, other):
        for i in range(s_num):
            for _ in range(i + 1, s_num):
                within.append(
                    jsd_local(
                        random_sample(ensemble, sample_size, rng),
                        random_sample(ensemble, sample_size, rng),
                        x_min, x_max, x_num, False,
                    )
                )
    return np.stack(between, axis=0), np.stack(within, axis=0)


def dissimilarity(
    ref_rep: np.ndarray,
    rep: np.ndarray,
    x_min: float,
    x_max: float,
    x_num: int = 100,
    s_num: int = 5,
    circular: bool = False,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    weights_ref=None,
    weights=None,
    alpha: float = 0.05,
    random_state: int | np.random.Generator | None = None,
    legacy: bool = False,
    ensemble_index: int = 0,
    reference_index: int = 0,
    measure: str = "",
) -> ComparisonResult:
    """Compare two ensemble representations.

    Parameters
    ----------
    ref_rep, rep
        Representation matrices, ``(n_frames, n_features)``, with matching
        feature counts.
    x_min, x_max
        Grid bounds for density estimation, normally the range over every
        ensemble in the study so that all comparisons share one grid. Ignored
        for circular measures.
    x_num
        Grid points per density.
    s_num
        Repeats of the split-half noise floor per ensemble (and, in legacy
        mode, resamples per ensemble).
    n_permutations
        Relabellings used to build the null. More gives finer p-values at
        linear cost; 100 is enough for a few hundred residues once the null is
        pooled across features.
    circular
        Whether the feature values live on a circle. Read from the measure's
        entry in :data:`~prothon.core.representation.MEASURES` when called
        through :class:`~prothon.Prothon`.
    sample_size
        Ensembles larger than this are subsampled, without replacement, before
        the test. The reported dissimilarity is computed on the subsample too,
        so observation and null are measured on the same data.
    weights_ref, weights
        Probability per frame, or ``None`` for uniform. A deposited ensemble
        stores these and a reweighted simulation produces them; ignoring them
        answers a question about a distribution nobody sampled.
    alpha
        False-discovery rate for the per-feature test.
    random_state
        Seed or generator. Supply one for a reproducible result.
    weights_ref, weights
        Probability per frame for each ensemble, or ``None`` for uniform. The
        densities, the permutation null and the noise floor all carry them.
    legacy
        Reproduce version 2.0: one pooled two-sided test, all features masked
        together, no FDR correction, linear grid even for torsions.

    Returns
    -------
    ComparisonResult

    Notes
    -----
    The permutation null assumes frames are exchangeable. Molecular dynamics
    frames are correlated in time, so an ensemble holds fewer independent
    conformations than it has frames, and these p-values remain somewhat
    optimistic for a single continuous trajectory. Correcting that properly
    needs a block permutation over the correlation time, which is planned for
    3.0. Meanwhile the split-half noise floor is measured rather than assumed,
    and is the more trustworthy of the two guides.
    """
    ref_rep = np.asarray(ref_rep, dtype=np.float64)
    rep = np.asarray(rep, dtype=np.float64)

    if ref_rep.ndim != 2 or rep.ndim != 2:
        raise ValueError("Representations must be 2-D (frames, features) matrices.")
    if ref_rep.shape[1] != rep.shape[1]:
        raise ValueError(
            f"Feature counts differ ({ref_rep.shape[1]} and {rep.shape[1]}); these "
            f"representations do not describe the same residues."
        )

    weights_ref = _normalise(weights_ref, ref_rep.shape[0])
    weights = _normalise(weights, rep.shape[0])

    if (weights_ref is None) != (weights is None):
        # One deposited ensemble against one trajectory is a real comparison
        # and a real asymmetry. Saying nothing produces a defensible-looking
        # number that nobody questions.
        warnings.warn(
            "One ensemble carries per-frame weights and the other does not. The "
            "unweighted one is treated as uniform, which is right for unbiased "
            "sampling and wrong for anything that stored probabilities and lost "
            "them on the way in.",
            UserWarning,
            stacklevel=2,
        )

    n_eff = (
        effective_sample_size(weights_ref, ref_rep.shape[0]),
        effective_sample_size(weights, rep.shape[0]),
    )
    for (name, matrix, w), neff in zip(
        (("reference", ref_rep, weights_ref), ("comparison", rep, weights)), n_eff
    ):
        if neff < MINIMUM_EFFECTIVE_SAMPLES:
            raise ValueError(
                f"The {name} ensemble is worth {neff:.1f} independent conformations "
                f"({matrix.shape[0]} frames"
                + (", weights concentrated on a few" if w is not None else "")
                + f"). Below {MINIMUM_EFFECTIVE_SAMPLES:.0f} there is nothing to "
                f"estimate a distribution from, and a per-residue profile computed "
                f"here would describe those few conformations wearing the name of "
                f"an ensemble."
            )
        if neff < MIN_EFFECTIVE_SAMPLES:
            detail = (
                f"{matrix.shape[0]} frames"
                if w is None
                else f"{matrix.shape[0]} frames, worth {neff:.0f} after weighting"
            )
            warnings.warn(
                f"The {name} ensemble is worth {neff:.0f} independent conformations "
                f"({detail}). Resampled densities will largely repeat the same "
                f"structures, so the noise floor understates the true uncertainty "
                f"and the p-values are optimistic. Treat this comparison as "
                f"qualitative.",
                UserWarning,
                stacklevel=2,
            )

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    use_circular = circular and not legacy
    if use_circular:
        x_min, x_max = -np.pi, np.pi

    if legacy:
        raw_local = jsd_local(
            ref_rep, rep, x_min, x_max, x_num, False, weights_ref, weights
        )  # noqa: E501
        between, within = _legacy_bootstrap(
            ref_rep, rep, x_min, x_max, x_num, s_num, sample_size, rng
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pooled = mannwhitneyu(between.flatten(), within.flatten()).pvalue
        p_values = np.full(raw_local.shape, float(pooled))
        noise_floor = float(np.mean(within))
    else:
        # Subsample once, then use the same matrices for the observed statistic
        # and for every relabelling: the null must be built from exactly the
        # data the observation was made on, or it calibrates the wrong thing.
        reference_sample, w_ref = _subsample(ref_rep, sample_size, rng, weights_ref)
        other_sample, w_other = _subsample(rep, sample_size, rng, weights)

        raw_local = jsd_local(
            reference_sample, other_sample, x_min, x_max, x_num, use_circular,
            w_ref, w_other,
        )
        null = _permutation_null(
            reference_sample, other_sample, x_min, x_max, x_num,
            use_circular, n_permutations, rng, w_ref, w_other,
        )
        p_values = _studentised_p_values(raw_local, null)
        noise_floor = float(
            np.mean(
                _split_half_floor(
                    (reference_sample, other_sample),
                    x_min, x_max, x_num, use_circular, max(1, s_num // 2), rng,
                    weights=(w_ref, w_other),
                )
            )
        )

    significant = p_values < alpha
    local = np.where(significant, raw_local, 0.0)

    logger.debug(
        "%s: global=%.4f floor=%.4f significant=%d/%d",
        measure or "comparison",
        float(np.mean(local)),
        noise_floor,
        int(np.count_nonzero(significant)),
        raw_local.size,
    )

    return ComparisonResult(
        ensemble_index=ensemble_index,
        reference_index=reference_index,
        global_dissimilarity=float(np.mean(local)),
        local_dissimilarity=local,
        raw_local_dissimilarity=raw_local,
        p_values=p_values,
        significant=significant,
        noise_floor=noise_floor,
        n_frames=(int(ref_rep.shape[0]), int(rep.shape[0])),
        effective_samples=n_eff,
        measure=measure,
        metadata={
            "alpha": alpha,
            "s_num": s_num,
            "n_permutations": 0 if legacy else n_permutations,
            "sample_size": sample_size,
            "null": "bootstrap (2.0)" if legacy else "permutation",
            "weighted": (weights_ref is not None) or (weights is not None),
            "effective_samples": [
                effective_sample_size(weights_ref, ref_rep.shape[0]),
                effective_sample_size(weights, rep.shape[0]),
            ],
            "legacy": legacy,
        },
    )
