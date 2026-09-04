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
from scipy.stats import mannwhitneyu

from ..sampling.correlation import MINIMUM_BLOCKS, block_labels
from ..sampling.floor import (
    FLOOR_QUANTILE,
    MINIMUM_FLOOR_REPEATS,
    MINIMUM_FLOOR_UNITS,
    FloorPlan,
    plan_floor,
    split_half_floor,
)
from ..sampling.null import permutation_null, studentised_p_values
from ..sampling.statistics import (
    DEFAULT_SAMPLE_SIZE,
    benjamini_hochberg,
    effective_sample_size,
    random_sample,
    validate_weights,
)
from ..utils import get_logger
from .density import estimate_pdf
from .distance import feature_distance, resolve_metric

logger = get_logger("dissimilarity")

#: Correlation time below which an unsettled estimate is not worth warning
#: about: the block correction is immaterial there and the warning would be
#: noise. Above it, an unsettled estimate makes the effective sample size and
#: the block count upper bounds, which is worth saying.
_WARN_ABOVE_TAU = 2.0

__all__ = [
    "ComparisonResult",
    "benjamini_hochberg",
    "effective_sample_size",
    "dissimilarity",
    "estimate_pdf",
    "jsd_local",
    "random_sample",
]


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
        Mean local dissimilarity over *every* feature, before any significance
        filter. This is the magnitude of the difference between the two
        ensembles, and it is the quantity the noise floor is comparable to:
        the floor is itself an unmasked mean over every feature, so comparing
        a filtered value against it compares two different quantities.

        It was a mean over the *masked* values through 2.1, which made a large
        difference read as zero whenever nothing survived the filter -- and
        whenever the sampling was too poor to run the filter at all, which is
        exactly when a magnitude beside a floor is the only thing left to say.
    masked_global_dissimilarity
        Mean over the features called significant, and zero elsewhere. What
        2.1 called ``global_dissimilarity``. Reported alongside rather than
        instead, because it answers a different question: not how far apart
        the ensembles are, but how much of that distance the sampling supports.
        Undefined in any useful sense when nothing survives.
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
        one-based. It is explicit even when the topologies are identical,
        because a representation such as CBCN can omit a residue.
    feature_labels
        Human-readable labels for those same features. Multichain labels are
        chain-qualified so repeated chain-local residue numbers are not
        ambiguous.
    noise_floor
        Mean within-ensemble Jensen-Shannon distance: the smallest difference
        this much sampling could resolve. Retained as a descriptive summary;
        ``resolved`` uses the upper-tail threshold rather than this mean.
    resolved
        Whether the global dissimilarity exceeds the 95th percentile of the
        noise-floor distribution. ``None`` when too few independent units were
        available to support that verdict.
    """

    ensemble_index: int
    reference_index: int
    global_dissimilarity: float
    masked_global_dissimilarity: float
    local_dissimilarity: np.ndarray
    raw_local_dissimilarity: np.ndarray
    p_values: np.ndarray
    significant: np.ndarray
    noise_floor: float
    n_frames: tuple[int, int]
    #: Upper quantile of the split-half distribution used for the global
    #: resolved/unresolved verdict. ``noise_floor`` remains its mean for
    #: backward compatibility and descriptive reporting.
    noise_floor_threshold: float | None = None
    #: False when there were too few independent units to support a floor
    #: verdict. The measured values remain available but are descriptive only.
    noise_floor_assessable: bool = True
    noise_floor_distribution: np.ndarray | None = None
    #: Kish effective count of each ensemble's native frames, temporal blocks,
    #: or replicas. Equal to the frame count only for unweighted IID input.
    effective_samples: tuple[float, float] = (0.0, 0.0)
    #: Estimated correlation time in frames, and the number of independent
    #: blocks the null was built from. A correlation time of 1.0 means the
    #: frames were treated as independent.
    correlation_time: float = 1.0
    #: Whether that correlation time stopped growing between half the frames
    #: and all of them. False makes it a *lower bound*, and makes ``n_blocks``
    #: and ``effective_samples`` upper bounds -- the optimistic direction. A
    #: block count clearing MINIMUM_BLOCKS on a lower-bound tau has not
    #: necessarily cleared it on the true one.
    correlation_time_converged: bool = True
    n_blocks: int = 0
    #: Set when the sampling could not support a p-value at all, in which case
    #: every entry of ``p_values`` is 1. The floor distribution remains a
    #: descriptive diagnostic, but it does not issue a verdict from the same
    #: insufficient blocks.
    p_values_withheld: bool = False
    order_parameter: str = ""
    #: Position of each feature on the reference ensemble, one-based.
    feature_index: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Display label for each feature, chain-qualified where needed.
    feature_labels: np.ndarray | None = None

    @property
    def p_value(self) -> float:
        """Smallest per-feature p-value, for the 2.0 dictionary key."""
        return float(np.min(self.p_values)) if self.p_values.size else 1.0

    @property
    def resolved(self) -> bool | None:
        """Whether the difference exceeds what this much sampling could produce.

        Compares the unmasked mean against the floor, because the floor is an
        unmasked mean. It is deliberately independent of the significance
        filter. A comparison whose p-values and floor verdict were withheld
        for want of independent blocks still retains its descriptive
        magnitude, while this property returns ``None``.
        """
        if not self.noise_floor_assessable:
            return None
        threshold = (
            self.noise_floor
            if self.noise_floor_threshold is None
            else self.noise_floor_threshold
        )
        return bool(self.global_dissimilarity > threshold)

    @property
    def p_values_reported(self) -> bool:
        """Whether the sampling supported a p-value at all.

        False when too few independent blocks were available, in which case
        the floor is the only guide and every p-value is 1.
        """
        return not self.p_values_withheld

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
            "order_parameter": self.order_parameter,
            "global_dissimilarity": float(self.global_dissimilarity),
            "masked_global_dissimilarity": float(self.masked_global_dissimilarity),
            "local_dissimilarity": self.local_dissimilarity.tolist(),
            "raw_local_dissimilarity": self.raw_local_dissimilarity.tolist(),
            "p_values": self.p_values.tolist(),
            "significant": self.significant.tolist(),
            "n_significant": self.n_significant,
            "noise_floor": float(self.noise_floor),
            "noise_floor_threshold": (
                float(self.noise_floor)
                if self.noise_floor_threshold is None
                else float(self.noise_floor_threshold)
            ),
            "noise_floor_assessable": bool(self.noise_floor_assessable),
            "noise_floor_distribution": (
                None
                if self.noise_floor_distribution is None
                else np.asarray(self.noise_floor_distribution).tolist()
            ),
            "resolved": self.resolved,
            "n_frames": list(self.n_frames),
            "effective_samples": list(self.effective_samples),
            "correlation_time": float(self.correlation_time),
            "correlation_time_converged": bool(self.correlation_time_converged),
            "n_blocks": int(self.n_blocks),
            "p_values_withheld": bool(self.p_values_withheld),
            "feature_index": (
                None if self.feature_index is None else self.feature_index.tolist()
            ),
            "feature_labels": (
                None if self.feature_labels is None else self.feature_labels.tolist()
            ),
            **self.metadata,
        }


def jsd_local(
    ensemble1: np.ndarray,
    ensemble2: np.ndarray,
    x_min: float,
    x_max: float,
    x_num: int,
    circular: bool = False,
    weights1=None,
    weights2=None,
    metric: str = "jsd",
) -> np.ndarray:
    """Per-feature distance between two representation matrices.

    Jensen-Shannon by default, in [0, 1] with base-2 logarithms so the bound is
    1 for distributions with disjoint support. ``metric`` selects another from
    :data:`~prothon.compare.distance.METRICS`; the name of this function is kept
    for the code written against 2.1.
    """
    if ensemble1.shape[1] != ensemble2.shape[1]:
        raise ValueError(
            f"Cannot compare representations with different feature counts: "
            f"{ensemble1.shape[1]} and {ensemble2.shape[1]}."
        )

    n_features = ensemble1.shape[1]
    distances = np.zeros(n_features, dtype=np.float64)
    for i in range(n_features):
        distances[i] = feature_distance(
            ensemble1[:, i], ensemble2[:, i], metric,
            x_min, x_max, x_num, circular, weights1, weights2,
        )
    return distances


@dataclass(frozen=True)
class _ComparisonSample:
    matrix: np.ndarray
    weights: np.ndarray | None
    replica_labels: np.ndarray | None
    frame_indices: np.ndarray
    strategy: str


def _structured_subsample(
    matrix: np.ndarray,
    weights: np.ndarray | None,
    sample_size: int,
    rng: np.random.Generator,
    sampling_kind: str,
    replica_labels,
) -> _ComparisonSample:
    """Subsample one side without damaging its declared sampling units."""
    labels = None if replica_labels is None else np.asarray(replica_labels)
    if labels is not None and (labels.ndim != 1 or labels.size != matrix.shape[0]):
        raise ValueError(
            "Replica labels must be one-dimensional with one label per frame."
        )
    n_frames = matrix.shape[0]
    if n_frames <= sample_size:
        keep = np.arange(n_frames)
        strategy = "all frames"
    elif labels is not None:
        _, inverse = np.unique(labels, return_inverse=True)
        replicas = [
            np.flatnonzero(inverse == label)
            for label in range(int(inverse.max()) + 1)
        ]
        selected = []
        selected_frames = 0
        for position in rng.permutation(len(replicas)):
            replica = replicas[int(position)]
            if selected_frames + replica.size <= sample_size:
                selected.append(replica)
                selected_frames += replica.size
        if not selected:
            selected = [min(replicas, key=len)]
        keep = np.sort(np.concatenate(selected))
        strategy = "complete replicas"
    elif sampling_kind == "trajectory":
        start = int(rng.integers(0, n_frames - sample_size + 1))
        keep = np.arange(start, start + sample_size)
        strategy = "contiguous window"
    else:
        keep = np.sort(rng.choice(n_frames, sample_size, replace=False))
        strategy = "uniform without replacement"

    sampled_weights = None if weights is None else weights[keep]
    if sampled_weights is not None:
        total = float(sampled_weights.sum())
        if total <= 0.0:
            raise ValueError("Selected sampling units carry zero probability mass.")
        sampled_weights = sampled_weights / total
    return _ComparisonSample(
        matrix=matrix[keep],
        weights=sampled_weights,
        replica_labels=None if labels is None else labels[keep],
        frame_indices=keep,
        strategy=strategy,
    )


def _native_unit_length(sample: _ComparisonSample, plan: FloorPlan) -> int:
    if sample.replica_labels is not None:
        _, counts = np.unique(sample.replica_labels, return_counts=True)
        return max(1, int(np.median(counts)))
    return plan.block_length if plan.sampling_kind == "trajectory" else 1


def _comparison_units(
    sample: _ComparisonSample,
    common_length: int,
) -> list[np.ndarray]:
    if sample.replica_labels is not None:
        _, inverse = np.unique(sample.replica_labels, return_inverse=True)
        return [
            np.flatnonzero(inverse == label)
            for label in range(int(inverse.max()) + 1)
        ]
    labels = block_labels(sample.matrix.shape[0], common_length)
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]


def _effective_units(
    units: list[np.ndarray],
    weights: np.ndarray | None,
    n_frames: int,
) -> float:
    mass = np.full(n_frames, 1.0 / n_frames) if weights is None else weights
    unit_mass = np.array([mass[unit].sum() for unit in units])
    return effective_sample_size(unit_mass)


def _sample_ranges(indices: np.ndarray) -> list[list[int]]:
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    return [
        [int(run[0]), int(run[-1]) + 1]
        for run in np.split(indices, breaks)
        if run.size
    ]


def _floor_plan_metadata(plan: FloorPlan) -> dict[str, Any]:
    return {
        "sampling_kind": plan.sampling_kind,
        "strategy": plan.strategy,
        "correlation_time": plan.correlation_time,
        "correlation_time_converged": plan.correlation_time_converged,
        "correlation_summary": plan.correlation_summary,
        "native_block_length": plan.block_length,
        "native_units": plan.n_units,
        "assessable_feature_columns": list(plan.assessable_features),
        "sampled_feature_columns": list(plan.sampled_features),
        "slow_feature_columns": list(plan.slow_features),
    }


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
    n_jobs: int = 1,
    weights_ref=None,
    weights=None,
    metric: str = "jsd",
    block_permutation: bool | None = None,
    correlation_time_frames: float | None = None,
    sampling_kind_ref: str | None = None,
    sampling_kind: str | None = None,
    correlation_time_frames_ref: float | None = None,
    replica_labels_ref=None,
    replica_labels=None,
    time_stride_ref: int = 1,
    time_stride: int = 1,
    alpha: float = 0.05,
    random_state: int | np.random.Generator | None = None,
    legacy: bool = False,
    ensemble_index: int = 0,
    reference_index: int = 0,
    order_parameter: str = "",
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
        Requested repeats of the split-half noise floor per ensemble (and, in
        legacy mode, resamples per ensemble). Modern mode uses at least ten so
        an upper-tail decision threshold can be estimated.
    n_permutations
        Relabellings used to build the null. More gives finer p-values at
        linear cost; 100 is enough for a few hundred residues once the null is
        pooled across features.
    circular
        Whether the feature values live on a circle. Read from the measure's
        entry in :data:`~prothon.represent.order_parameters.MEASURES` when called
        through :class:`~prothon.Prothon`.
    sample_size
        Ensembles larger than this are subsampled, without replacement, before
        the test. The reported dissimilarity is computed on the subsample too,
        so observation and null are measured on the same data. When block
        permutation is active, the subsample is a contiguous window: temporal
        order and the time step are part of a trajectory's sampling structure.
        Frames may be selected independently only when block permutation is
        explicitly disabled or the data are estimated to be uncorrelated.
    weights_ref, weights
        Probability per frame, or ``None`` for uniform. A deposited ensemble
        stores these and a reweighted simulation produces them; ignoring them
        answers a question about a distribution nobody sampled.
    metric
        Which per-feature distance to use: ``jsd`` (default, bounded),
        ``wasserstein`` (unbounded, in the feature's own units) or ``ks``.
    block_permutation
        Relabel contiguous blocks rather than individual frames, so the null
        is built from data that looks like a trajectory. ``None`` (the
        default) enables it whenever a correlation time longer than one frame
        is detected. Set ``False`` for an ensemble whose frames genuinely are
        independent -- a set of generated structures, or an already-subsampled
        trajectory -- where blocking costs resolution for nothing.

        **Rows must be in the order the frames were generated.** A shuffled or
        concatenated matrix has no correlation time, and this will silently
        find none.
    correlation_time_frames
        Correlation time for the comparison ensemble. For source compatibility,
        when no per-ensemble sampling arguments are supplied it applies to both
        sides as it did before 2.4.
    sampling_kind_ref, sampling_kind
        ``"trajectory"`` or ``"iid"`` for each ensemble. The conservative
        default is trajectory. The older ``block_permutation=False`` maps both
        sides to IID; new mixed designs should state each side explicitly.
    correlation_time_frames_ref
        Optional supplied correlation time for the reference. Otherwise it is
        estimated from the sampled reference trajectory.
    replica_labels_ref, replica_labels
        One label per frame. Complete replicas replace temporal blocks as
        indivisible permutation and floor units.
    time_stride_ref, time_stride
        Stored-frame strides, recorded as provenance. Correlation times remain
        expressed in stored frames.
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
    Individual molecular-dynamics frames are not exchangeable. When temporal
    correlation is detected, the null relabels contiguous blocks and any
    computational subsample remains contiguous. The block count and refusal
    decision are computed from that sampled window rather than from frames the
    test did not use.
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

    weights_ref = validate_weights(
        weights_ref, ref_rep.shape[0], "Reference weights"
    )
    weights = validate_weights(weights, rep.shape[0], "Comparison weights")

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

    resolve_metric(metric)  # fail here rather than inside the permutation loop

    explicit_per_ensemble_sampling = any(
        value is not None
        for value in (
            sampling_kind_ref,
            sampling_kind,
            correlation_time_frames_ref,
            replica_labels_ref,
            replica_labels,
        )
    )
    if block_permutation is False and explicit_per_ensemble_sampling:
        raise ValueError(
            "block_permutation=False cannot be combined with per-ensemble "
            "sampling provenance; declare both sampling kinds as IID instead."
        )
    default_kind = "iid" if block_permutation is False else "trajectory"
    kind_ref = str(
        default_kind if sampling_kind_ref is None else sampling_kind_ref
    ).strip().lower()
    kind_other = str(
        default_kind if sampling_kind is None else sampling_kind
    ).strip().lower()
    if kind_ref not in {"trajectory", "iid"} or kind_other not in {
        "trajectory",
        "iid",
    }:
        raise ValueError("sampling_kind_ref and sampling_kind must be trajectory or iid.")
    # The old single correlation-time argument described the common block plan.
    # Preserve that contract unless the caller has opted into the new per-side
    # provenance model, where it naturally belongs to the comparison side.
    if block_permutation is False and not explicit_per_ensemble_sampling:
        correlation_time_frames = None
        correlation_time_frames_ref = None
    elif not explicit_per_ensemble_sampling and correlation_time_frames is not None:
        correlation_time_frames_ref = correlation_time_frames

    for name, value in (
        ("time_stride_ref", time_stride_ref),
        ("time_stride", time_stride),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
    time_stride_ref, time_stride = int(time_stride_ref), int(time_stride)

    tau = 1.0
    tau_converged = True
    correlation_profiles: dict[str, dict[str, Any]] = {}
    block_length, n_blocks = 1, min(ref_rep.shape[0], rep.shape[0])
    use_blocks = False
    if legacy:
        raw_local = jsd_local(
            ref_rep, rep, x_min, x_max, x_num, False, weights_ref, weights, metric
        )  # noqa: E501
        between, within = _legacy_bootstrap(
            ref_rep, rep, x_min, x_max, x_num, s_num, sample_size, rng
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pooled = mannwhitneyu(between.flatten(), within.flatten()).pvalue
        p_values = np.full(raw_local.shape, float(pooled))
        noise_floor = float(np.mean(within))
        noise_floor_threshold = noise_floor
        noise_floor_distribution = np.asarray(within, dtype=np.float64)
        noise_floor_assessable = True
        withheld = False
    else:
        sample_ref = _structured_subsample(
            ref_rep, weights_ref, sample_size, rng, kind_ref, replica_labels_ref
        )
        sample_other = _structured_subsample(
            rep, weights, sample_size, rng, kind_other, replica_labels
        )
        reference_sample, w_ref = sample_ref.matrix, sample_ref.weights
        other_sample, w_other = sample_other.matrix, sample_other.weights

        plans = (
            plan_floor(
                reference_sample,
                sampling_kind=kind_ref,
                correlation_time_frames=correlation_time_frames_ref,
                replica_labels=sample_ref.replica_labels,
                circular=use_circular,
            ),
            plan_floor(
                other_sample,
                sampling_kind=kind_other,
                correlation_time_frames=correlation_time_frames,
                replica_labels=sample_other.replica_labels,
                circular=use_circular,
            ),
        )
        tau = max(plan.correlation_time for plan in plans)
        tau_converged = all(plan.correlation_time_converged for plan in plans)
        correlation_profiles = {
            name: {
                "summary": plan.correlation_summary,
                "assessable_features": plan.n_assessable_features,
                "sampled_features": plan.n_sampled_features,
                "assessable_feature_columns": list(plan.assessable_features),
                "sampled_feature_columns": list(plan.sampled_features),
                "slow_feature_columns": list(plan.slow_features),
            }
            for name, plan in zip(("reference", "comparison"), plans)
        }
        native_unit_lengths = (
            _native_unit_length(sample_ref, plans[0]),
            _native_unit_length(sample_other, plans[1]),
        )
        native_units = (
            _comparison_units(sample_ref, native_unit_lengths[0]),
            _comparison_units(sample_other, native_unit_lengths[1]),
        )
        effective_native_units = (
            _effective_units(
                native_units[0], w_ref, reference_sample.shape[0]
            ),
            _effective_units(
                native_units[1], w_other, other_sample.shape[0]
            ),
        )
        common_unit_length = max(native_unit_lengths)
        units_ref = _comparison_units(sample_ref, common_unit_length)
        units_other = _comparison_units(sample_other, common_unit_length)
        use_blocks = bool(
            common_unit_length > 1
            or sample_ref.replica_labels is not None
            or sample_other.replica_labels is not None
        )
        block_length = common_unit_length
        n_blocks = min(len(units_ref), len(units_other))
        effective_permutation_units = (
            _effective_units(units_ref, w_ref, reference_sample.shape[0]),
            _effective_units(units_other, w_other, other_sample.shape[0]),
        )
        sampled_smaller = min(reference_sample.shape[0], other_sample.shape[0])

        raw_local = jsd_local(
            reference_sample, other_sample, x_min, x_max, x_num, use_circular,
            w_ref, w_other, metric,
        )
        # The null and the floor need to measure, not to know what the
        # measurement means. Both take the statistic rather than importing it,
        # which is what keeps `sampling` free of any dependency on `compare`.
        def statistic(left, right, weights_left=None, weights_right=None):
            return jsd_local(
                left, right, x_min, x_max, x_num, use_circular,
                weights_left, weights_right, metric,
            )

        null = permutation_null(
            n_jobs, statistic,
            reference_sample, other_sample,
            n_permutations, rng, w_ref, w_other,
            units_a=units_ref if use_blocks else None,
            units_b=units_other if use_blocks else None,
        )
        p_values = studentised_p_values(raw_local, null)

        # One check, because the block length is no longer shortened to
        # manufacture blocks: a trajectory too short for its own correlation
        # time now shows up as too few blocks, which is what it is.
        # Only worth saying when the correlation correction is doing
        # something. Below two frames the blocks are three frames long, the
        # effective sample size is within a factor of two of the frame count,
        # and an unsettled estimate changes nothing a reader would act on. The
        # log-slope of a noisy estimate hovering near 1 trips easily, so
        # without this the warning fires on every small dataset -- which is how
        # a warning stops being read.
        for name, plan in zip(("reference", "comparison"), plans):
            if (
                not plan.correlation_time_converged
                and plan.correlation_time >= _WARN_ABOVE_TAU
            ):
                warnings.warn(
                    f"The {name} correlation time is still rising with "
                    f"trajectory length, so its estimate of about "
                    f"{plan.correlation_time:.0f} frames is a lower bound. "
                    f"Everything derived from it is correspondingly "
                    f"optimistic: the independent-unit count and the "
                    f"effective sample size. Sample this system for longer "
                    f"before treating the per-residue calls as settled.",
                    UserWarning,
                    stacklevel=3,
                )

        withheld = use_blocks and (
            n_blocks < MINIMUM_BLOCKS
            or min(effective_permutation_units) < MINIMUM_BLOCKS
        )
        if withheld:
            # Fewer independent units than a p-value can be built from. The
            # floor is still measured and still means something; the p-value
            # would not, so it is withheld rather than printed.
            p_values = np.ones_like(p_values)
            effective_detail = (
                ""
                if (
                    min(effective_permutation_units) >= MINIMUM_BLOCKS
                    or (w_ref is None and w_other is None)
                )
                else f", only {min(effective_permutation_units):.1f} effective "
                f"after weighting"
            )
            warnings.warn(
                f"A correlation time of about {tau:.0f} frames leaves only "
                f"{n_blocks} independent blocks{effective_detail} in "
                f"{sampled_smaller} sampled frames, fewer than the "
                f"{MINIMUM_BLOCKS} a permutation p-value can be built from. No "
                f"p-value is reported, and the split-half floor has too few "
                f"independent units for a resolved/unresolved verdict. Its "
                f"measured values are retained as descriptive diagnostics. "
                f"Sample this system for longer, or compare independent replicas.",
                UserWarning,
                stacklevel=3,
            )
        floor_distribution = split_half_floor(
            n_jobs,
            statistic,
            (reference_sample, other_sample),
            max(MINIMUM_FLOOR_REPEATS, s_num),
            rng,
            weights=(w_ref, w_other),
            block_lengths=tuple(plan.block_length for plan in plans),
            replica_labels=(
                sample_ref.replica_labels,
                sample_other.replica_labels,
            ),
        )
        noise_floor_distribution = np.asarray(floor_distribution, dtype=np.float64)
        global_floor_distribution = noise_floor_distribution.mean(axis=1)
        noise_floor = float(global_floor_distribution.mean())
        noise_floor_threshold = float(
            np.quantile(global_floor_distribution, FLOOR_QUANTILE)
        )
        noise_floor_assessable = bool(
            all(plan.assessable for plan in plans)
            and min(effective_native_units) >= MINIMUM_FLOOR_UNITS
        )

    significant = p_values < alpha
    local = np.where(significant, raw_local, 0.0)
    if legacy:
        withheld = False

    logger.debug(
        "%s: global=%.4f floor=%.4f significant=%d/%d",
        order_parameter or "comparison",
        float(np.mean(raw_local)),
        noise_floor,
        int(np.count_nonzero(significant)),
        raw_local.size,
    )

    if legacy:
        sampled_frames = [int(ref_rep.shape[0]), int(rep.shape[0])]
        sampling_strategy = "bootstrap (2.0)"
        sampling_strategies = [sampling_strategy, sampling_strategy]
        sample_selection = []
        sampling_plans = []
        sampling_units = sampled_frames
        native_sampling_units = sampled_frames
        sampling_unit_sizes = []
        effective_permutation_result = n_eff
        effective_result = n_eff
    else:
        sampled_frames = [
            int(reference_sample.shape[0]), int(other_sample.shape[0])
        ]
        sampling_strategies = [sample_ref.strategy, sample_other.strategy]
        sampling_strategy = (
            sampling_strategies[0]
            if sampling_strategies[0] == sampling_strategies[1]
            else "per-ensemble"
        )
        sample_selection = [
            {
                "original_frames": int(original.shape[0]),
                "sampled_frames": int(sample.matrix.shape[0]),
                "sampling_strategy": sample.strategy,
                "frame_ranges": _sample_ranges(sample.frame_indices),
            }
            for original, sample in (
                (ref_rep, sample_ref),
                (rep, sample_other),
            )
        ]
        sampling_plans = [_floor_plan_metadata(plan) for plan in plans]
        sampling_units = [len(units_ref), len(units_other)]
        native_sampling_units = [len(units) for units in native_units]
        sampling_unit_sizes = [
            [int(unit.size) for unit in units_ref],
            [int(unit.size) for unit in units_other],
        ]
        effective_permutation_result = effective_permutation_units
        effective_result = effective_native_units

    return ComparisonResult(
        ensemble_index=ensemble_index,
        reference_index=reference_index,
        global_dissimilarity=float(np.mean(raw_local)),
        masked_global_dissimilarity=float(np.mean(local)),
        local_dissimilarity=local,
        raw_local_dissimilarity=raw_local,
        p_values=p_values,
        significant=significant,
        noise_floor=noise_floor,
        n_frames=(int(ref_rep.shape[0]), int(rep.shape[0])),
        noise_floor_threshold=noise_floor_threshold,
        noise_floor_assessable=noise_floor_assessable,
        noise_floor_distribution=noise_floor_distribution,
        effective_samples=effective_result,
        correlation_time=float(tau),
        correlation_time_converged=bool(tau_converged),
        n_blocks=int(n_blocks),
        p_values_withheld=bool(withheld),
        order_parameter=order_parameter,
        metadata={
            "alpha": alpha,
            "s_num": s_num,
            "n_permutations": 0 if legacy else n_permutations,
            "sample_size": sample_size,
            "sampled_frames": sampled_frames,
            "sampling_strategy": sampling_strategy,
            "sampling_strategies": sampling_strategies,
            "sample_selection": sample_selection,
            "sampling_plans": sampling_plans,
            "input_time_stride": [time_stride_ref, time_stride],
            "block_length": int(block_length),
            "permutation_units": sampling_units,
            "effective_permutation_units": list(effective_permutation_result),
            "permutation_unit_sizes": sampling_unit_sizes,
            "native_sampling_units": native_sampling_units,
            "correlation_profiles": correlation_profiles,
            "noise_floor_quantile": FLOOR_QUANTILE,
            "null": "bootstrap (2.0)" if legacy else "permutation",
            "metric": "jsd" if legacy else metric,
            "weighted": (weights_ref is not None) or (weights is not None),
            "effective_samples": list(effective_result),
            "frame_weight_effective_samples": list(n_eff),
            "weights_attached_to_observations": True,
            "legacy": legacy,
        },
    )
