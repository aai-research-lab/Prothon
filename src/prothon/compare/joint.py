"""Is this ensemble distinguishable from that one, and by what?

The per-residue metrics answer a question about each feature separately. They
cannot see a difference that lives in the relationship *between* features -- a
mutant whose two loops each visit the same positions as the wild type, but no
longer at the same time, has an identical profile at every residue and is a
different ensemble. Both methods here look at the joint distribution.

**Maximum mean discrepancy** embeds each conformation in a reproducing kernel
Hilbert space and measures the distance between the mean embeddings. With a
characteristic kernel it is zero only when the distributions are equal, so it
sees any difference given enough samples. Its squared MMD statistic remains
available for every comparison. A p-value is reported only when a
sampling-unit permutation has enough independent blocks, replicas or IID rows
to resolve the requested threshold; MMD gives no indication of where the
difference is.

**The classifier two-sample test** trains a classifier to tell the two
ensembles apart. If it cannot do better than chance, they are indistinguishable
at this sampling; if it can, out-of-fold area under the ROC curve (AUC) is a
bounded effect size, and the classifier can be asked which features it used.
That last part is what the per-residue metrics give for free and MMD cannot give
at all.

Three details that decide whether the numbers mean anything:

**Features are standardised**, because MMD's kernel measures Euclidean
distance and a contact number ranging over 0-12 would otherwise drown a torsion
ranging over one radian.

**Circular features are encoded as (cos, sin)** rather than passed as raw
angles. A torsion at +179 degrees and one at -179 are two degrees apart, and to
a kernel reading raw numbers they are 358.

Two things about that are worth being accurate on, because the earlier circular
problems in this package were more severe and it would be easy to imply this
one is the same. The encoding matters much less to the classifier than to MMD:
a decision tree splits at thresholds and can carve the wraparound out with an
extra split, so a forest is largely immune where a Euclidean kernel is not.
And for MMD the *statistic* moves reproducibly -- populations near the
wraparound come out about 15% closer once encoded -- while the *verdict* moves
hardly at all, because the permutation null is built from the same encoding and
a systematic distortion partly cancels between the observation and its null.

That is a point in favour of permutation nulls rather than a reason to skip the
encoding. The encoding is still right: the statistic is the thing being
reported, it should mean what it says, and a null cannot be relied on to
absorb every mistake in the statistic it calibrates.

**The classifier is scored out of fold.** Complete blocks or replicas remain
in exactly one fold. A separate label-blind unit split trains the evidence
classifier without any test-unit label, so its fixed predictions can be tested
by permuting whole held-out unit labels without repeatedly fitting the forest.

    Gretton, A.; Borgwardt, K. M.; Rasch, M. J.; Scholkopf, B.; Smola, A.
    A kernel two-sample test. J. Mach. Learn. Res. 2012, 13, 723-773.

    Lopez-Paz, D.; Oquab, M. Revisiting classifier two-sample tests.
    International Conference on Learning Representations, 2017.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..sampling.correlation import block_labels
from ..sampling.floor import FloorPlan, plan_floor
from ..sampling.statistics import validate_weights
from ..utils import get_logger
from .dissimilarity import (
    MINIMUM_EFFECTIVE_SAMPLES,
    effective_sample_size,
)

logger = get_logger("ensemble_metrics")

__all__ = [
    "MINIMUM_MMD_UNITS",
    "EnsembleComparison",
    "classifier_two_sample",
    "distinguishability",
    "maximum_mean_discrepancy",
]

#: Conformations drawn from each ensemble. Both methods are quadratic in this,
#: so it buys accuracy at a cost that grows quickly.
DEFAULT_SAMPLE_SIZE = 1000

#: Relabellings used for the MMD null.
DEFAULT_PERMUTATIONS = 200

#: Folds for the classifier. Every conformation is scored by a classifier that
#: did not see it.
DEFAULT_FOLDS = 5

#: Retained for source compatibility with 2.3. C2ST no longer uses the
#: asymptotic normal tail that made this reporting floor necessary.
P_VALUE_FLOOR = 1e-6

#: The inferential threshold used by :attr:`EnsembleComparison.distinguishable`.
MMD_ALPHA = 0.05

#: Four units on each side give 35 distinct balanced partitions after
#: complementary labels are identified. Three give only ten, so no MMD
#: permutation p-value from that design can resolve a 5% threshold.
MINIMUM_MMD_UNITS = 4


@dataclass
class EnsembleComparison:
    """One whole-ensemble comparison.

    Attributes
    ----------
    statistic
        MMD squared, or classifier out-of-fold balanced accuracy. For the
        second, 0.5 is the chance reference; ``effect`` carries the primary
        classifier effect size.
    p_value
        From a sampling-unit permutation null. ``None`` means the effect was
        measured but the independent units or requested permutations could not
        resolve the requested threshold.
    effect
        A bounded, readable summary. Area under the ROC curve for the
        classifier; ``None`` for MMD, which has no natural scale.
    feature_importance
        How much each feature contributed, where the method can say. ``None``
        for MMD, which cannot.
    feature_index, feature_labels
        Stable one-based reference residue positions and readable labels for
        classifier importances. Labels carry chain identity where needed.
    """

    method: str
    statistic: float
    p_value: float | None
    effect: float | None = None
    null_mean: float = 0.0
    null_std: float = 0.0
    n_samples: tuple[int, int] = (0, 0)
    effective_samples: tuple[float, float] = (0.0, 0.0)
    feature_importance: np.ndarray | None = None
    feature_index: np.ndarray | None = None
    order_parameter: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    feature_labels: np.ndarray | None = None

    @property
    def distinguishable(self) -> bool | None:
        if self.p_value is None:
            return None
        return bool(self.p_value < float(self.metadata.get("alpha", MMD_ALPHA)))

    @property
    def p_value_withheld(self) -> bool:
        return self.p_value is None

    def leading_features(self, n: int = 5) -> list[tuple[int, float]]:
        """The features the method leaned on most, largest first."""
        if self.feature_importance is None:
            return []
        order = np.argsort(self.feature_importance)[::-1][:n]
        labels = (
            np.arange(1, self.feature_importance.size + 1)
            if self.feature_index is None
            else self.feature_index
        )
        return [(int(labels[i]), float(self.feature_importance[i])) for i in order]

    def leading_feature_labels(self, n: int = 5) -> list[tuple[str, float]]:
        """Readable labels for the strongest features, largest first."""
        if self.feature_importance is None:
            return []
        order = np.argsort(self.feature_importance)[::-1][:n]
        labels = (
            np.asarray(
                np.arange(1, self.feature_importance.size + 1)
                if self.feature_index is None
                else self.feature_index
            ).astype(str)
            if self.feature_labels is None
            else np.asarray(self.feature_labels, dtype=str)
        )
        return [(str(labels[i]), float(self.feature_importance[i])) for i in order]

    def summary(self) -> str:
        if self.p_value is None:
            reason = self.metadata.get(
                "p_value_withheld_reason",
                "sampling design cannot resolve alpha",
            )
            effect = (
                f"MMD² = {self.statistic:.4g}"
                if self.method.lower() == "mmd"
                else f"AUC = {self.effect:.3f}"
            )
            line = f"{self.method.upper()}: {effect}, p-value withheld ({reason})"
            leading = self.leading_feature_labels(5)
            if leading:
                named = ", ".join(f"{i}" for i, _ in leading)
                line += f"\n  driven mostly by residues {named}"
            return line
        verdict = (
            "distinguishable" if self.distinguishable else "not distinguishable"
        )
        shown = f"p = {self.p_value:.3g}"
        line = f"{self.method.upper()}: {verdict} ({shown})"
        if self.effect is not None:
            line += f", AUC = {self.effect:.3f}"
        leading = self.leading_feature_labels(5)
        if leading:
            named = ", ".join(f"{i}" for i, _ in leading)
            line += f"\n  driven mostly by residues {named}"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "order_parameter": self.order_parameter,
            "statistic": float(self.statistic),
            "p_value": None if self.p_value is None else float(self.p_value),
            "p_value_withheld": self.p_value_withheld,
            "effect": None if self.effect is None else float(self.effect),
            "distinguishable": self.distinguishable,
            "null_mean": float(self.null_mean),
            "null_std": float(self.null_std),
            "n_samples": list(self.n_samples),
            "effective_samples": list(self.effective_samples),
            "feature_importance": (
                None
                if self.feature_importance is None
                else self.feature_importance.tolist()
            ),
            "feature_index": (
                None if self.feature_index is None else self.feature_index.tolist()
            ),
            "feature_labels": (
                None if self.feature_labels is None else self.feature_labels.tolist()
            ),
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Preparing the joint representation
# ---------------------------------------------------------------------------
def _encode(matrix: np.ndarray, circular: bool) -> np.ndarray:
    """Put a representation into a space where Euclidean distance is sensible.

    Circular columns become a cosine and a sine, so that two angles either side
    of the wraparound are close where they should be close. Everything else is
    passed through and standardised by the caller.
    """
    if not circular:
        return matrix
    return np.hstack([np.cos(matrix), np.sin(matrix)])


def _prepare(a, b, circular):
    """Encode, pool and standardise. Standardisation uses the pooled sample,
    so neither ensemble sets the scale for the other."""
    x, y = _encode(np.asarray(a, float), circular), _encode(np.asarray(b, float), circular)
    pooled = np.vstack([x, y])
    centre = pooled.mean(axis=0)
    scale = pooled.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (x - centre) / scale, (y - centre) / scale


def _check_sampling(a, b, weights_a, weights_b, labels=("first", "second")):
    n_eff = (
        effective_sample_size(weights_a, a.shape[0]),
        effective_sample_size(weights_b, b.shape[0]),
    )
    for label, eff in zip(labels, n_eff):
        if eff < MINIMUM_EFFECTIVE_SAMPLES:
            raise ValueError(
                f"The {label} ensemble is worth {eff:.1f} independent conformations. "
                f"Below {MINIMUM_EFFECTIVE_SAMPLES:.0f} there is not enough "
                f"independent information to tell two joint distributions apart."
            )
    return n_eff


@dataclass(frozen=True)
class _JointSample:
    """One side of a joint test, with its sampling structure intact."""

    matrix: np.ndarray
    weights: np.ndarray | None
    replica_labels: np.ndarray | None
    plan: FloorPlan
    original_frames: int
    frame_indices: np.ndarray
    sampling_strategy: str


def _probability_weights(weights, n_frames: int, label: str) -> np.ndarray | None:
    """Validated probability mass that remains attached to its observation."""
    return validate_weights(weights, n_frames, f"The {label} weights")


def _replica_subsample_indices(
    replica_labels: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select complete replicas up to a target frame count.

    ``sample_size`` is a cap unless no complete replica fits, in which case the
    smallest replica is retained whole. Splitting it would manufacture
    independent units that the data do not contain.
    """
    _, inverse = np.unique(replica_labels, return_inverse=True)
    replicas = [
        np.flatnonzero(inverse == label)
        for label in range(int(inverse.max()) + 1)
    ]
    order = rng.permutation(len(replicas))
    selected: list[np.ndarray] = []
    total = 0
    for position in order:
        replica = replicas[int(position)]
        if total + replica.size <= sample_size:
            selected.append(replica)
            total += replica.size
    if not selected:
        selected = [min(replicas, key=len)]
    return np.sort(np.concatenate(selected))


def _sample_for_joint_test(
    matrix: np.ndarray,
    weights: np.ndarray | None,
    sample_size: int,
    rng: np.random.Generator,
    sampling_kind: str,
    correlation_time_frames: float | None,
    replica_labels,
    circular: bool,
) -> _JointSample:
    """Subsample without destroying the units used by the joint null."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(
            "A joint comparison requires a non-empty 2-D (frames, features) matrix."
        )
    kind = str(sampling_kind).strip().lower()
    if kind not in {"trajectory", "iid"}:
        raise ValueError("sampling_kind must be 'trajectory' or 'iid'.")
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
        keep = _replica_subsample_indices(labels, sample_size, rng)
        strategy = "complete replicas"
    elif kind == "trajectory":
        start = int(rng.integers(0, n_frames - sample_size + 1))
        keep = np.arange(start, start + sample_size)
        strategy = "contiguous window"
    else:
        keep = np.sort(rng.choice(n_frames, sample_size, replace=False))
        strategy = "uniform without replacement"

    sampled_weights = None if weights is None else weights[keep]
    if sampled_weights is not None:
        sampled_total = float(sampled_weights.sum())
        if sampled_total <= 0.0:
            raise ValueError(
                "The selected complete sampling units carry zero probability mass."
            )
        sampled_weights = sampled_weights / sampled_total
    sampled_labels = None if labels is None else labels[keep]
    sampled_matrix = matrix[keep]
    # The plan belongs to the data actually used. Estimating on the original
    # trajectory and carrying that block count into a shorter computational
    # sample would claim independent units that never reached the test.
    plan = plan_floor(
        sampled_matrix,
        sampling_kind=kind,
        correlation_time_frames=correlation_time_frames,
        replica_labels=sampled_labels,
        circular=circular,
    )
    return _JointSample(
        matrix=sampled_matrix,
        weights=sampled_weights,
        replica_labels=sampled_labels,
        plan=plan,
        original_frames=n_frames,
        frame_indices=keep,
        sampling_strategy=strategy,
    )


def _native_unit_length(sample: _JointSample) -> int:
    if sample.replica_labels is not None:
        _, counts = np.unique(sample.replica_labels, return_counts=True)
        return max(1, int(np.median(counts)))
    if sample.plan.sampling_kind == "trajectory":
        return sample.plan.block_length
    return 1


def _sample_units(sample: _JointSample, common_length: int) -> list[np.ndarray]:
    """Local frame indices for complete replicas or common-sized blocks."""
    if sample.replica_labels is not None:
        _, inverse = np.unique(sample.replica_labels, return_inverse=True)
        return [
            np.flatnonzero(inverse == label)
            for label in range(int(inverse.max()) + 1)
        ]
    labels = block_labels(sample.matrix.shape[0], common_length)
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]


def _mmd_signed_weights(
    pooled_weights: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Signed, group-normalised mass for one label assignment.

    The probability stays on its observation. Only the group label moves, and
    each relabelled empirical distribution is then normalised to mass one.
    """
    left_mass = float(pooled_weights[left].sum())
    right_mass = float(pooled_weights[right].sum())
    if left_mass <= 0.0 or right_mass <= 0.0:
        raise ValueError("Each relabelled MMD group must carry positive weight.")
    signed = np.zeros(pooled_weights.size, dtype=np.float64)
    signed[left] = pooled_weights[left] / left_mass
    signed[right] = -pooled_weights[right] / right_mass
    return signed


def _mmd_unit_assignment(
    units: list[np.ndarray],
    n_left_units: int,
    rng: np.random.Generator,
    pooled_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel whole units, never their constituent frames."""
    for _ in range(1000):
        order = rng.permutation(len(units))
        left = np.concatenate([units[i] for i in order[:n_left_units]])
        right = np.concatenate([units[i] for i in order[n_left_units:]])
        if pooled_weights is None or (
            pooled_weights[left].sum() > 0.0
            and pooled_weights[right].sum() > 0.0
        ):
            return left, right
    raise ValueError(
        "Could not form two relabelled MMD groups with positive probability mass."
    )


def _distinct_mmd_assignments(n_left_units: int, n_right_units: int) -> int:
    arrangements = math.comb(n_left_units + n_right_units, n_left_units)
    # MMD(u) == MMD(-u), so complementary balanced labels are the same
    # statistic and provide one, not two, points of p-value resolution.
    return arrangements // 2 if n_left_units == n_right_units else arrangements


def _effective_mmd_units(
    units: list[np.ndarray],
    observation_mass: np.ndarray,
) -> float:
    """Kish effective count after probability mass is collected by unit."""
    unit_mass = np.array(
        [observation_mass[unit].sum() for unit in units],
        dtype=np.float64,
    )
    return effective_sample_size(unit_mass)


def _plan_metadata(plan: FloorPlan) -> dict[str, Any]:
    return {
        "sampling_kind": plan.sampling_kind,
        "strategy": plan.strategy,
        "correlation_time": plan.correlation_time,
        "correlation_time_converged": plan.correlation_time_converged,
        "correlation_summary": plan.correlation_summary,
        "assessable_feature_columns": list(plan.assessable_features),
        "sampled_feature_columns": list(plan.sampled_features),
        "slow_feature_columns": list(plan.slow_features),
        "native_block_length": plan.block_length,
    }


def _selection_metadata(sample: _JointSample) -> dict[str, Any]:
    indices = sample.frame_indices
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    runs = np.split(indices, breaks)
    ranges = [[int(run[0]), int(run[-1]) + 1] for run in runs if run.size]
    selected = (
        {"frame_range": ranges[0]}
        if len(ranges) == 1
        else {"frame_ranges": ranges}
    )
    return {
        "original_frames": sample.original_frames,
        "sampled_frames": int(indices.size),
        "sampling_strategy": sample.sampling_strategy,
        **selected,
    }


# ---------------------------------------------------------------------------
# Maximum mean discrepancy
# ---------------------------------------------------------------------------
def _squared_distances(points: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, in ``O(n^2)`` memory.

    The obvious expression, ``((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)``,
    materialises an ``(n, n, d)`` array before reducing it. At the default
    thousand conformations a side that is 2000 x 2000 x d, which for a
    76-residue protein is 2.4 GB and for a 300-residue one is nine times that:
    the process is killed rather than slowed, and only on real proteins, since
    a test fixture with a dozen residues never approaches it.

    Expanding the square instead -- ``|x-y|^2 = |x|^2 + |y|^2 - 2 x.y`` -- turns
    it into one matrix product and never allocates the third dimension.
    Rounding can make a diagonal entry very slightly negative, so the result is
    clipped at zero.
    """
    square_norms = np.einsum("ij,ij->i", points, points)
    squared = square_norms[:, None] + square_norms[None, :] - 2.0 * (points @ points.T)
    np.maximum(squared, 0.0, out=squared)
    return squared


def _median_bandwidth(pooled: np.ndarray, rng, cap: int = 2000) -> float:
    """The median heuristic: set the kernel width to the median distance
    between points, so the kernel is neither flat nor a delta over this data."""
    sample = pooled if pooled.shape[0] <= cap else pooled[
        rng.choice(pooled.shape[0], cap, replace=False)
    ]
    squared = _squared_distances(sample)
    upper = squared[np.triu_indices(sample.shape[0], 1)]
    median = float(np.median(upper))
    return median if median > 0 else 1.0


def maximum_mean_discrepancy(
    a: np.ndarray,
    b: np.ndarray,
    weights_a=None,
    weights_b=None,
    circular: bool = False,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_state=None,
    order_parameter: str = "",
    bandwidth: float | None = None,
    standardise: bool = True,
    sampling_kind_a: str = "trajectory",
    sampling_kind_b: str = "trajectory",
    correlation_time_frames_a: float | None = None,
    correlation_time_frames_b: float | None = None,
    replica_labels_a=None,
    replica_labels_b=None,
    time_stride_a: int = 1,
    time_stride_b: int = 1,
    alpha: float = MMD_ALPHA,
) -> EnsembleComparison:
    """Kernel two-sample test between two ensembles.

    Parameters
    ----------
    bandwidth
        Gaussian kernel width. ``None`` uses the median heuristic: the kernel
        is set to the median distance between points, so it is neither flat nor
        a delta over this data. Fixing it is useful for checking the statistic
        against a case with a known value.
    standardise
        Put the features on a common scale using the pooled sample. On by
        default because the kernel measures Euclidean distance and a contact
        number ranging over 0-12 would otherwise drown a torsion ranging over
        one radian. Turn it off only when the scales are already comparable and
        the absolute value of the statistic matters.
    sampling_kind_a, sampling_kind_b
        ``"trajectory"`` (the conservative default) or ``"iid"``. Temporal
        trajectories are relabelled in complete blocks; IID frames are
        relabelled individually unless paired with a blocked input, in which
        case they are grouped to the same unit length.
    correlation_time_frames_a, correlation_time_frames_b
        Optional supplied correlation times. Otherwise each trajectory is
        estimated from its ordered representation.
    replica_labels_a, replica_labels_b
        One label per frame. Complete replicas replace temporal blocks as the
        indivisible permutation units.
    time_stride_a, time_stride_b
        Separation between stored frames in each source trajectory, recorded
        as provenance. Correlation times remain expressed in stored frames.

    Notes
    -----
    The squared MMD is a quadratic form ``u' K u`` in a signed weight vector
    ``u`` -- positive on one ensemble, negative on the other. The null moves
    group labels between complete sampling units, then rebuilds ``u`` from the
    probability mass that remained attached to each observation. The kernel
    itself is still built only once.
    """
    integer_controls = {
        "n_permutations": n_permutations,
        "sample_size": sample_size,
        "time_stride_a": time_stride_a,
        "time_stride_b": time_stride_b,
    }
    for name, value in integer_controls.items():
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
    n_permutations = int(n_permutations)
    sample_size = int(sample_size)
    time_stride_a = int(time_stride_a)
    time_stride_b = int(time_stride_b)
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must lie strictly between zero and one.")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha must lie strictly between zero and one.") from error
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    rng = np.random.default_rng(random_state)
    raw_a = np.asarray(a, dtype=np.float64)
    raw_b = np.asarray(b, dtype=np.float64)
    if raw_a.ndim != 2 or raw_b.ndim != 2:
        raise ValueError("MMD inputs must be 2-D (frames, features) matrices.")
    if raw_a.shape[1] != raw_b.shape[1]:
        raise ValueError(
            f"MMD feature counts differ ({raw_a.shape[1]} and {raw_b.shape[1]})."
        )
    if not np.all(np.isfinite(raw_a)) or not np.all(np.isfinite(raw_b)):
        raise ValueError("MMD inputs must contain only finite values.")
    wa = _probability_weights(weights_a, raw_a.shape[0], "first")
    wb = _probability_weights(weights_b, raw_b.shape[0], "second")
    sample_a = _sample_for_joint_test(
        raw_a, wa, sample_size, rng,
        sampling_kind_a, correlation_time_frames_a, replica_labels_a, circular,
    )
    sample_b = _sample_for_joint_test(
        raw_b, wb, sample_size, rng,
        sampling_kind_b, correlation_time_frames_b, replica_labels_b, circular,
    )
    wa, wb = sample_a.weights, sample_b.weights
    frame_weight_effective = _check_sampling(
        sample_a.matrix, sample_b.matrix, wa, wb
    )

    for label, plan in (("first", sample_a.plan), ("second", sample_b.plan)):
        if not plan.correlation_time_converged and plan.correlation_time >= 2.0:
            warnings.warn(
                f"The {label} MMD correlation time is still rising with "
                f"trajectory length. Its {plan.correlation_time:.0f}-frame "
                f"estimate is a lower bound, so the block null remains "
                f"optimistic. Sample longer before treating its p-value as "
                f"settled.",
                UserWarning,
                stacklevel=2,
            )

    if standardise:
        x, y = _prepare(sample_a.matrix, sample_b.matrix, circular)
    else:
        x = _encode(sample_a.matrix, circular)
        y = _encode(sample_b.matrix, circular)
    m, n = x.shape[0], y.shape[0]

    pooled = np.vstack([x, y])
    sigma_squared = (
        float(bandwidth) ** 2 if bandwidth is not None
        else _median_bandwidth(pooled, rng)
    )
    if not np.isfinite(sigma_squared) or sigma_squared <= 0.0:
        raise ValueError("bandwidth must be finite and greater than zero.")
    kernel = np.exp(-_squared_distances(pooled) / (2.0 * sigma_squared))

    wa = np.full(m, 1.0 / m) if wa is None else wa / wa.sum()
    wb = np.full(n, 1.0 / n) if wb is None else wb / wb.sum()
    # Put both inputs on the same mean-one mass scale before labels move.
    # Their supplied scales are arbitrary (each vector is normalised within
    # its original ensemble); pooling 1/m beside 1/n would leave the original
    # label encoded in every weight whenever lengths differ. Mean-one scaling
    # preserves relative probability within each input, makes unweighted
    # observations exactly equal, and recovers wa/wb after group normalisation.
    pooled_weights = np.concatenate([wa * m, wb * n])
    original_left = np.arange(m)
    original_right = np.arange(m, m + n)
    signed = _mmd_signed_weights(
        pooled_weights, original_left, original_right
    )

    observed = float(signed @ (kernel @ signed))
    common_unit_length = max(
        _native_unit_length(sample_a),
        _native_unit_length(sample_b),
    )
    units_a = _sample_units(sample_a, common_unit_length)
    units_b = [m + unit for unit in _sample_units(sample_b, common_unit_length)]
    units = units_a + units_b
    n_units_a, n_units_b = len(units_a), len(units_b)
    effective_units = (
        _effective_mmd_units(units_a, pooled_weights),
        _effective_mmd_units(units_b, pooled_weights),
    )
    null = np.empty(n_permutations)
    for k in range(n_permutations):
        left, right = _mmd_unit_assignment(
            units, n_units_a, rng, pooled_weights
        )
        shuffled = _mmd_signed_weights(pooled_weights, left, right)
        null[k] = float(shuffled @ (kernel @ shuffled))

    distinct_assignments = _distinct_mmd_assignments(n_units_a, n_units_b)
    assignment_resolution = math.exp(-math.log(distinct_assignments))
    p_value_resolution = max(
        1.0 / (n_permutations + 1),
        assignment_resolution,
    )
    # The +1 keeps a Monte Carlo p-value from being zero. The exact assignment
    # count is a second, independent limit: repeatedly drawing the same few
    # labelings cannot create finer evidence than the design contains.
    measured_p = max(
        float((1 + np.count_nonzero(null >= observed)) / (n_permutations + 1)),
        p_value_resolution,
    )
    p_value_supported = bool(
        min(n_units_a, n_units_b) >= MINIMUM_MMD_UNITS
        and min(effective_units) >= MINIMUM_MMD_UNITS
        and p_value_resolution < alpha
    )
    p_value = measured_p if p_value_supported else None
    withheld_reasons = []
    if min(n_units_a, n_units_b) < MINIMUM_MMD_UNITS:
        withheld_reasons.append("fewer than four sampling units")
    if min(effective_units) < MINIMUM_MMD_UNITS:
        withheld_reasons.append("fewer than four weight-effective units")
    if p_value_resolution >= alpha:
        withheld_reasons.append("permutation resolution is not finer than alpha")
    p_value_withheld_reason = "; ".join(withheld_reasons) or None
    if not p_value_supported:
        warnings.warn(
            f"MMD has only {n_units_a} and {n_units_b} sampling units "
            f"({effective_units[0]:.1f} and {effective_units[1]:.1f} effective; "
            f"{distinct_assignments} distinct labelings; attainable p-value "
            f"resolution {p_value_resolution:.3g}), too little to resolve "
            f"alpha={alpha:g}. MMD squared is "
            f"retained as an effect measure, but the p-value and "
            f"distinguishable verdict are withheld.",
            UserWarning,
            stacklevel=2,
        )

    return EnsembleComparison(
        method="mmd",
        statistic=observed,
        p_value=p_value,
        effect=None,
        null_mean=float(null.mean()),
        # A single relabelling has no spread. That is a legitimate way to call
        # this -- asking for the statistic alone, with no test -- so it returns
        # zero rather than a NaN from a division by zero degrees of freedom.
        null_std=float(null.std(ddof=1)) if null.size > 1 else 0.0,
        n_samples=(m, n),
        effective_samples=effective_units,
        order_parameter=order_parameter,
        metadata={
            "kernel": "gaussian",
            "bandwidth_squared": sigma_squared,
            "bandwidth_rule": "fixed" if bandwidth is not None else "median heuristic",
            "standardised": standardise,
            "n_permutations": n_permutations,
            "circular_encoding": circular,
            "alpha": alpha,
            "sampling_plans": [
                _plan_metadata(sample_a.plan),
                _plan_metadata(sample_b.plan),
            ],
            "sample_selection": [
                _selection_metadata(sample_a),
                _selection_metadata(sample_b),
            ],
            "input_time_stride": [time_stride_a, time_stride_b],
            "permutation_unit_length": common_unit_length,
            "permutation_units": [n_units_a, n_units_b],
            "effective_permutation_units": list(effective_units),
            "frame_weight_effective_samples": list(frame_weight_effective),
            "permutation_unit_sizes": [
                [int(unit.size) for unit in units_a],
                [int(unit.size) for unit in units_b],
            ],
            "distinct_labelings": distinct_assignments,
            "p_value_resolution": p_value_resolution,
            "p_value_withheld_reason": p_value_withheld_reason,
            "weights_attached_to_observations": True,
        },
    )


# ---------------------------------------------------------------------------
# Classifier two-sample test
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _GroupedFolds:
    """A fold for every frame and, separately, every complete unit."""

    frame_folds: np.ndarray
    unit_folds_a: tuple[int, ...]
    unit_folds_b: tuple[int, ...]
    n_folds: int


def _grouped_fold_plan(
    units_a: list[np.ndarray],
    units_b: list[np.ndarray],
    n_frames: int,
    requested_folds: int,
    rng: np.random.Generator,
    observation_mass: np.ndarray | None = None,
) -> _GroupedFolds:
    """Assign whole sampling units to class-stratified folds.

    Positive-mass units are distributed first. This makes every weighted test
    fold informative and leaves positive mass from both classes in every
    training split. Zero-mass units still receive a fold and are never split.
    """
    mass = (
        np.ones(n_frames, dtype=np.float64)
        if observation_mass is None
        else np.asarray(observation_mass, dtype=np.float64)
    )
    if mass.shape != (n_frames,):
        raise ValueError("observation_mass must contain one value per frame.")

    def positive_count(units):
        return sum(float(mass[unit].sum()) > 0.0 for unit in units)

    usable_folds = min(
        int(requested_folds),
        positive_count(units_a),
        positive_count(units_b),
    )
    if usable_folds < 2:
        raise ValueError(
            "Cannot cross-validate C2ST with fewer than two positive-mass "
            "sampling units from each ensemble."
        )

    def assign(units):
        positive = np.array(
            [i for i, unit in enumerate(units) if mass[unit].sum() > 0.0],
            dtype=int,
        )
        zero = np.array(
            [i for i, unit in enumerate(units) if mass[unit].sum() <= 0.0],
            dtype=int,
        )
        assignments = np.empty(len(units), dtype=int)
        shuffled = rng.permutation(positive)
        assignments[shuffled] = np.arange(shuffled.size) % usable_folds
        if zero.size:
            shuffled_zero = rng.permutation(zero)
            assignments[shuffled_zero] = (
                np.arange(shuffled_zero.size) % usable_folds
            )
        return assignments

    assigned_a = assign(units_a)
    assigned_b = assign(units_b)
    frame_folds = np.full(n_frames, -1, dtype=int)
    for units, assignments in (
        (units_a, assigned_a),
        (units_b, assigned_b),
    ):
        for unit, fold in zip(units, assignments):
            if np.any(frame_folds[unit] != -1):
                raise ValueError("C2ST sampling units overlap.")
            frame_folds[unit] = fold
    if np.any(frame_folds < 0):
        raise ValueError("C2ST sampling units do not cover every sampled frame.")
    return _GroupedFolds(
        frame_folds=frame_folds,
        unit_folds_a=tuple(int(fold) for fold in assigned_a),
        unit_folds_b=tuple(int(fold) for fold in assigned_b),
        n_folds=usable_folds,
    )


def _balanced_class_weights(
    observation_mass: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Equal class mass while retaining relative mass within each class."""
    balanced = np.zeros(labels.size, dtype=np.float64)
    for label in (0, 1):
        selected = labels == label
        total = float(observation_mass[selected].sum())
        if total <= 0.0:
            raise ValueError(
                "Each C2ST class must carry positive probability mass in this split."
            )
        balanced[selected] = 0.5 * observation_mass[selected] / total
    return balanced


def _weighted_balanced_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    observation_mass: np.ndarray,
) -> float:
    weights = _balanced_class_weights(observation_mass, labels)
    return float(np.sum(weights * (predictions == labels)))


def _unit_balance(
    units: list[np.ndarray],
    labels: np.ndarray,
    observation_mass: np.ndarray,
) -> dict[str, list[float] | list[int]]:
    unit_labels = np.array([labels[unit[0]] for unit in units], dtype=int)
    return {
        "class_units": [
            int(np.count_nonzero(unit_labels == label)) for label in (0, 1)
        ],
        "class_weight_mass": [
            float(
                sum(
                    observation_mass[unit].sum()
                    for unit, unit_label in zip(units, unit_labels)
                    if unit_label == label
                )
            )
            for label in (0, 1)
        ],
    }


def classifier_two_sample(
    a: np.ndarray,
    b: np.ndarray,
    weights_a=None,
    weights_b=None,
    circular: bool = False,
    folds: int = DEFAULT_FOLDS,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_state=None,
    order_parameter: str = "",
    feature_index=None,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    sampling_kind_a: str = "trajectory",
    sampling_kind_b: str = "trajectory",
    correlation_time_frames_a: float | None = None,
    correlation_time_frames_b: float | None = None,
    replica_labels_a=None,
    replica_labels_b=None,
    time_stride_a: int = 1,
    time_stride_b: int = 1,
    alpha: float = MMD_ALPHA,
    feature_labels=None,
) -> EnsembleComparison:
    """Train a classifier to tell the two ensembles apart, and score it fairly.

    A random forest rather than a linear model, because two ensembles that
    differ in *spread* rather than in mean -- a loop that is rigid in one and
    mobile in the other -- are not linearly separable and are a difference
    anybody would want found.

    Complete temporal blocks or replicas, rather than rows, define the
    cross-validation folds. The out-of-fold AUC is the primary effect size.

    Evidence is deliberately separate from effect estimation. A second forest
    is trained on a label-blind half of the pooled units. Its fixed predictions
    on the untouched units are compared with whole-unit label permutations.
    This avoids both adjacent-frame leakage and the invalid independent-row
    normal approximation. If the held-out unit assignment cannot resolve
    ``alpha``, the AUC is retained and the p-value is withheld.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    if feature_index is not None and len(feature_index) != a.shape[1]:
        raise ValueError("feature_index must contain one value per feature.")
    if feature_labels is not None and len(feature_labels) != a.shape[1]:
        raise ValueError("feature_labels must contain one value per feature.")

    integer_controls = {
        "folds": folds,
        "n_permutations": n_permutations,
        "sample_size": sample_size,
        "time_stride_a": time_stride_a,
        "time_stride_b": time_stride_b,
    }
    for name, value in integer_controls.items():
        minimum = 2 if name == "folds" else 1
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            qualifier = (
                "an integer of at least two"
                if minimum == 2
                else "a positive integer"
            )
            raise ValueError(f"{name} must be {qualifier}.")
    folds = int(folds)
    n_permutations = int(n_permutations)
    sample_size = int(sample_size)
    time_stride_a = int(time_stride_a)
    time_stride_b = int(time_stride_b)
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must lie strictly between zero and one.")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha must lie strictly between zero and one.") from error
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")

    raw_a = np.asarray(a, dtype=np.float64)
    raw_b = np.asarray(b, dtype=np.float64)
    if raw_a.ndim != 2 or raw_b.ndim != 2:
        raise ValueError("C2ST inputs must be 2-D (frames, features) matrices.")
    if raw_a.shape[1] != raw_b.shape[1]:
        raise ValueError(
            f"C2ST feature counts differ ({raw_a.shape[1]} and {raw_b.shape[1]})."
        )
    if not np.all(np.isfinite(raw_a)) or not np.all(np.isfinite(raw_b)):
        raise ValueError("C2ST inputs must contain only finite values.")

    rng = np.random.default_rng(random_state)
    wa = _probability_weights(weights_a, raw_a.shape[0], "first")
    wb = _probability_weights(weights_b, raw_b.shape[0], "second")
    sample_a = _sample_for_joint_test(
        raw_a, wa, sample_size, rng,
        sampling_kind_a, correlation_time_frames_a, replica_labels_a, circular,
    )
    sample_b = _sample_for_joint_test(
        raw_b, wb, sample_size, rng,
        sampling_kind_b, correlation_time_frames_b, replica_labels_b, circular,
    )
    wa, wb = sample_a.weights, sample_b.weights
    frame_weight_effective = _check_sampling(
        sample_a.matrix, sample_b.matrix, wa, wb
    )
    for label, plan in (("first", sample_a.plan), ("second", sample_b.plan)):
        if not plan.correlation_time_converged and plan.correlation_time >= 2.0:
            warnings.warn(
                f"The {label} C2ST correlation time is still rising with "
                f"trajectory length. Its {plan.correlation_time:.0f}-frame "
                f"estimate is a lower bound, so grouped folds and the unit "
                f"null remain optimistic. Sample longer before treating its "
                f"p-value as settled.",
                UserWarning,
                stacklevel=2,
            )

    x, y = _prepare(sample_a.matrix, sample_b.matrix, circular)
    m, n = x.shape[0], y.shape[0]
    features = np.vstack([x, y])
    labels = np.concatenate([np.zeros(m, int), np.ones(n, int)])
    wa = np.full(m, 1.0 / m) if wa is None else wa / wa.sum()
    wb = np.full(n, 1.0 / n) if wb is None else wb / wb.sum()
    observation_mass = np.concatenate([wa * m, wb * n])

    common_unit_length = max(
        _native_unit_length(sample_a),
        _native_unit_length(sample_b),
    )
    units_a = _sample_units(sample_a, common_unit_length)
    units_b = [m + unit for unit in _sample_units(sample_b, common_unit_length)]
    units = units_a + units_b
    effective_units = (
        _effective_mmd_units(units_a, observation_mass),
        _effective_mmd_units(units_b, observation_mass),
    )

    cv_assignment_seed = int(rng.integers(0, 2**31 - 1))
    cv_plan = _grouped_fold_plan(
        units_a,
        units_b,
        m + n,
        folds,
        np.random.default_rng(cv_assignment_seed),
        observation_mass,
    )
    predictions = np.zeros(m + n)
    importance = np.zeros(features.shape[1])
    cv_forest_seeds = []
    cv_fold_balance = []
    for fold in range(cv_plan.n_folds):
        test = np.flatnonzero(cv_plan.frame_folds == fold)
        train = np.flatnonzero(cv_plan.frame_folds != fold)
        train_weight = _balanced_class_weights(
            observation_mass[train], labels[train]
        )
        forest_seed = int(rng.integers(0, 2**31 - 1))
        cv_forest_seeds.append(forest_seed)
        forest = RandomForestClassifier(
            n_estimators=200,
            random_state=forest_seed,
            n_jobs=1,
            min_samples_leaf=2,
        )
        forest.fit(features[train], labels[train], sample_weight=train_weight)
        predictions[test] = forest.predict_proba(features[test])[:, 1]
        importance += forest.feature_importances_ / cv_plan.n_folds
        fold_units = [
            unit
            for unit, unit_fold in zip(
                units,
                (*cv_plan.unit_folds_a, *cv_plan.unit_folds_b),
            )
            if unit_fold == fold
        ]
        cv_fold_balance.append(
            {
                "fold": fold,
                **_unit_balance(fold_units, labels, observation_mass),
            }
        )

    cv_labels = (predictions >= 0.5).astype(int)
    accuracy = _weighted_balanced_accuracy(
        labels, cv_labels, observation_mass
    )
    auc = float(
        roc_auc_score(labels, predictions, sample_weight=observation_mass)
    )

    # The evidence split is deliberately label-blind: select pooled units
    # before looking at which ensemble supplied them. Only positive-mass units
    # matter to a weighted empirical distribution.
    evidence_units = [
        unit for unit in units if observation_mass[unit].sum() > 0.0
    ]
    unit_names = [
        ["a", local] for local in range(len(units_a))
    ] + [
        ["b", local] for local in range(len(units_b))
    ]
    evidence_names = [
        name
        for name, unit in zip(unit_names, units)
        if observation_mass[unit].sum() > 0.0
    ]
    evidence_split_seed = int(rng.integers(0, 2**31 - 1))
    split_rng = np.random.default_rng(evidence_split_seed)
    split_order = split_rng.permutation(len(evidence_units))
    n_test_units = len(evidence_units) // 2
    test_positions = split_order[:n_test_units]
    train_positions = split_order[n_test_units:]
    train_units = [evidence_units[int(i)] for i in train_positions]
    test_units = [evidence_units[int(i)] for i in test_positions]
    train = (
        np.concatenate(train_units)
        if train_units
        else np.empty(0, dtype=int)
    )
    test = (
        np.concatenate(test_units)
        if test_units
        else np.empty(0, dtype=int)
    )
    evidence_forest_seed = int(rng.integers(0, 2**31 - 1))
    permutation_seed = int(rng.integers(0, 2**31 - 1))
    unit_test_labels = np.array(
        [labels[unit[0]] for unit in test_units], dtype=int
    )
    train_has_both = train.size > 0 and np.unique(labels[train]).size == 2
    test_has_both = test.size > 0 and np.unique(labels[test]).size == 2
    distinct_labelings = (
        math.comb(len(test_units), int(unit_test_labels.sum()))
        if test_has_both
        else 1
    )
    assignment_resolution = math.exp(-math.log(distinct_labelings))
    p_value_resolution = max(
        1.0 / (n_permutations + 1), assignment_resolution
    )
    p_value_supported = bool(
        train_has_both and test_has_both and p_value_resolution < alpha
    )
    null = np.empty(0, dtype=np.float64)
    evidence_accuracy = None
    if train_has_both and test_has_both:
        evidence_forest = RandomForestClassifier(
            n_estimators=200,
            random_state=evidence_forest_seed,
            n_jobs=1,
            min_samples_leaf=2,
        )
        evidence_forest.fit(
            features[train],
            labels[train],
            sample_weight=_balanced_class_weights(
                observation_mass[train], labels[train]
            ),
        )
        test_predictions = (
            evidence_forest.predict_proba(features[test])[:, 1] >= 0.5
        ).astype(int)
        evidence_accuracy = _weighted_balanced_accuracy(
            labels[test], test_predictions, observation_mass[test]
        )
        permutation_rng = np.random.default_rng(permutation_seed)
        null = np.empty(n_permutations, dtype=np.float64)
        for iteration in range(n_permutations):
            shuffled_units = permutation_rng.permutation(unit_test_labels)
            shuffled_labels = np.empty(test.size, dtype=int)
            offset = 0
            for unit, unit_label in zip(test_units, shuffled_units):
                shuffled_labels[offset:offset + unit.size] = unit_label
                offset += unit.size
            null[iteration] = _weighted_balanced_accuracy(
                shuffled_labels, test_predictions, observation_mass[test]
            )
        measured_p = max(
            float(
                (1 + np.count_nonzero(null >= evidence_accuracy))
                / (n_permutations + 1)
            ),
            p_value_resolution,
        )
        p_value = measured_p if p_value_supported else None
    else:
        p_value = None

    withheld_reasons = []
    if not train_has_both:
        withheld_reasons.append("label-blind training units contain only one class")
    if not test_has_both:
        withheld_reasons.append("label-blind test units contain only one class")
    if p_value_resolution >= alpha:
        withheld_reasons.append("permutation resolution is not finer than alpha")
    p_value_withheld_reason = "; ".join(withheld_reasons) or None
    if not p_value_supported:
        warnings.warn(
            f"C2ST held out {len(test_units)} sampling units with "
            f"{distinct_labelings} distinct labelings and attainable p-value "
            f"resolution {p_value_resolution:.3g}, which cannot resolve "
            f"alpha={alpha:g}. The grouped-CV AUC is retained as an effect "
            f"measure, but the p-value and distinguishable verdict are withheld "
            f"({p_value_withheld_reason}).",
            UserWarning,
            stacklevel=2,
        )

    if circular:
        # Undo the (cos, sin) doubling so importances line up with features.
        half = importance.size // 2
        importance = importance[:half] + importance[half:]

    return EnsembleComparison(
        method="c2st",
        statistic=accuracy,
        p_value=p_value,
        effect=auc,
        null_mean=float(null.mean()) if null.size else 0.5,
        null_std=float(null.std(ddof=1)) if null.size > 1 else 0.0,
        n_samples=(m, n),
        effective_samples=effective_units,
        feature_importance=importance,
        feature_index=None if feature_index is None else np.asarray(feature_index),
        feature_labels=(
            None if feature_labels is None else np.asarray(feature_labels, dtype=str)
        ),
        order_parameter=order_parameter,
        metadata={
            "classifier": "random forest (200 trees)",
            "folds": cv_plan.n_folds,
            "circular_encoding": circular,
            "alpha": alpha,
            "n_permutations": n_permutations,
            "sampling_plans": [
                _plan_metadata(sample_a.plan),
                _plan_metadata(sample_b.plan),
            ],
            "sample_selection": [
                _selection_metadata(sample_a),
                _selection_metadata(sample_b),
            ],
            "input_time_stride": [time_stride_a, time_stride_b],
            "sampling_unit_length": common_unit_length,
            "sampling_units": [len(units_a), len(units_b)],
            "effective_sampling_units": list(effective_units),
            "frame_weight_effective_samples": list(frame_weight_effective),
            "sampling_unit_sizes": [
                [int(unit.size) for unit in units_a],
                [int(unit.size) for unit in units_b],
            ],
            "cv_assignment_seed": cv_assignment_seed,
            "cv_unit_folds": [
                list(cv_plan.unit_folds_a),
                list(cv_plan.unit_folds_b),
            ],
            "cv_forest_seeds": cv_forest_seeds,
            "cv_fold_balance": cv_fold_balance,
            "inference_split_seed": evidence_split_seed,
            "inference_forest_seed": evidence_forest_seed,
            "permutation_seed": permutation_seed,
            "inference_train_units": [
                evidence_names[int(i)] for i in train_positions
            ],
            "inference_test_units": [
                evidence_names[int(i)] for i in test_positions
            ],
            "inference_train_balance": (
                _unit_balance(train_units, labels, observation_mass)
                if train_units
                else {"class_units": [0, 0], "class_weight_mass": [0.0, 0.0]}
            ),
            "inference_test_balance": (
                _unit_balance(test_units, labels, observation_mass)
                if test_units
                else {"class_units": [0, 0], "class_weight_mass": [0.0, 0.0]}
            ),
            "evidence_balanced_accuracy": evidence_accuracy,
            "distinct_labelings": distinct_labelings,
            "p_value_resolution": p_value_resolution,
            "p_value_withheld_reason": p_value_withheld_reason,
            "weights_attached_to_observations": True,
        },
    )


_METHODS = {"mmd": maximum_mean_discrepancy, "c2st": classifier_two_sample}


def distinguishability(
    a: np.ndarray,
    b: np.ndarray,
    method: str = "c2st",
    **kwargs,
) -> EnsembleComparison:
    """Whole-ensemble comparison by the named method: ``mmd`` or ``c2st``."""
    key = method.strip().lower()
    if key not in _METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Available: {', '.join(sorted(_METHODS))}."
        )
    if key == "mmd" and (
        "feature_index" in kwargs or "feature_labels" in kwargs
    ):
        # MMD has no per-feature view; accepting the argument and discarding it
        # would suggest otherwise.
        kwargs.pop("feature_index", None)
        kwargs.pop("feature_labels", None)
        warnings.warn(
            "MMD reports no per-feature contribution, so feature_index has no "
            "effect. Use c2st to find out which residues carry the difference.",
            UserWarning,
            stacklevel=2,
        )
    return _METHODS[key](a, b, **kwargs)
