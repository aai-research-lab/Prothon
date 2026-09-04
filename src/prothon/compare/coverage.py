"""What a model misses, and what it invents.

A single dissimilarity says two ensembles differ. It does not say *how*, and
the two ways of differing call for opposite fixes. A generative model that
never opens a cryptic pocket and one that opens pockets no physics produces are
both wrong, both score badly on any symmetric distance, and need entirely
different work.

The distinction is standard in machine learning and absent from protein
ensemble comparison. Precision asks how much of what a model emits lands where
the reference has support; recall asks how much of the reference's support the
model reaches. Low recall is a missed state -- mode collapse, a loop that never
unfolds. Low precision is a hallucinated one -- a conformation no simulation
visited.

**Per residue, which is the part that is Prothon's to add.** The machine
learning versions of this operate on a whole sample at once and return two
numbers. Local order parameters give the same decomposition at every residue,
so the answer is not "recall 0.62" but "the model covers the fold and misses
the 40-55 loop". That is a sentence a model developer can act on.

**The support is a highest-density region**, the smallest region holding a
stated fraction of the mass. That makes the null value exact rather than
approximate: if two ensembles are drawn from the same distribution, both
precision and recall are the coverage level itself, by construction. A number
below it means something, and how far below is measurable.

**And it carries a floor.** Two independently assignable halves of one
ensemble do not reach each other's support perfectly either, because a finite
sample never covers a continuous distribution. For trajectories those halves
contain complete temporal blocks, not interleaved frames. Prothon measures
that self-precision and self-recall and reports both beside the result, for the
same reason it reports a noise floor beside a dissimilarity: a model cannot be
asked to do better than the reference can do against itself.

    Sajjadi, M. S. M.; Bachem, O.; Lucic, M.; Bousquet, O.; Gelly, S.
    Assessing generative models via precision and recall. Advances in Neural
    Information Processing Systems, 2018.

    Kynkaanniemi, T.; Karras, T.; Laine, S.; Lehtinen, J.; Aila, T. Improved
    precision and recall metric for assessing generative models. Advances in
    Neural Information Processing Systems, 2019.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..sampling.floor import (
    FLOOR_QUANTILE,
    MINIMUM_FLOOR_REPEATS,
    MINIMUM_FLOOR_UNITS,
    effective_floor_units,
    plan_floor,
    split_half_floor,
)
from ..sampling.statistics import validate_weights
from ..utils import get_logger
from .dissimilarity import MINIMUM_EFFECTIVE_SAMPLES, effective_sample_size, estimate_pdf

logger = get_logger("precision_recall")

__all__ = ["PrecisionRecall", "precision_recall"]

#: Fraction of an ensemble's probability defining its support. Also the exact
#: value both precision and recall take when the two ensembles are drawn from
#: the same distribution, which is what makes a departure readable.
DEFAULT_COVERAGE = 0.95

#: Grid points per density.
DEFAULT_GRID = 200

#: Repeats of the split-half self-comparison that establishes the floor. Ten
#: give the lower tail enough support not to be merely the observed minimum.
DEFAULT_FLOOR_REPEATS = 10

#: Compatibility fallback for results constructed without the stored lower-
#: tail threshold introduced in 3.0.
FLOOR_MARGIN_SD = 2.0

#: And at least this far, so that a residue whose floor happens to be very
#: stable across repeats does not get called on a difference of 0.001.
FLOOR_MARGIN_MIN = 0.02


@dataclass
class PrecisionRecall:
    """Coverage and fidelity, per feature and overall.

    Attributes
    ----------
    precision
        Fraction of the model's mass inside the reference's support, per
        feature. Below the floor means conformations the reference never
        visited.
    recall
        Fraction of the reference's mass inside the model's support, per
        feature. Below the floor means conformations the model never reaches.
    floor_precision, floor_recall
        The same quantities measured between two halves of each ensemble and
        pooled, per feature. The best a perfect model could score at this
        sampling. Per feature rather than averaged: a rigid residue is easy to
        cover and a mobile one is not.
    floor_precision_threshold, floor_recall_threshold
        Lower-tail split-half quantiles, with a small numerical margin. These,
        rather than the mean floors, decide missed and invented calls.
    floor_assessable
        False when fewer than eight independent frames, temporal blocks, or
        replicas were available. Floor values remain descriptive, while
        missed and invented calls are withheld.
    effective_samples
        Kish effective count after weights have been aggregated within each
        ensemble's native frames, temporal blocks, or replicas.
    coverage
        The level defining the support, and the value both quantities take
        under identical distributions.
    feature_index, feature_labels
        Stable one-based reference residue positions and readable labels for
        the same features. Labels carry chain identity in multichain systems.
    """

    precision: np.ndarray
    recall: np.ndarray
    floor_precision: np.ndarray
    floor_recall: np.ndarray
    floor_precision_sd: np.ndarray | None = None
    floor_recall_sd: np.ndarray | None = None
    floor_precision_threshold: np.ndarray | None = None
    floor_recall_threshold: np.ndarray | None = None
    floor_precision_distribution: np.ndarray | None = None
    floor_recall_distribution: np.ndarray | None = None
    floor_assessable: bool = True
    coverage: float = DEFAULT_COVERAGE
    feature_index: np.ndarray | None = None
    effective_samples: tuple[float, float] = (0.0, 0.0)
    order_parameter: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    feature_labels: np.ndarray | None = None

    @property
    def mean_precision(self) -> float:
        return float(np.mean(self.precision))

    @property
    def mean_recall(self) -> float:
        return float(np.mean(self.recall))

    @property
    def mean_floor_precision(self) -> float:
        return float(np.mean(self.floor_precision))

    @property
    def mean_floor_recall(self) -> float:
        return float(np.mean(self.floor_recall))

    def _margin(self, spread) -> np.ndarray:
        if spread is None:
            return np.full(self.precision.shape, FLOOR_MARGIN_MIN)
        return np.maximum(FLOOR_MARGIN_SD * np.asarray(spread), FLOOR_MARGIN_MIN)

    def _labels(self) -> np.ndarray:
        return (
            np.arange(1, self.precision.size + 1)
            if self.feature_index is None
            else np.asarray(self.feature_index)
        )

    def _display_labels(self) -> np.ndarray:
        return (
            self._labels().astype(str)
            if self.feature_labels is None
            else np.asarray(self.feature_labels, dtype=str)
        )

    def missed(self) -> np.ndarray:
        """Features where the model fails to reach the reference's support."""
        if not self.floor_assessable:
            return self._labels()[:0]
        threshold = (
            self.floor_recall - self._margin(self.floor_recall_sd)
            if self.floor_recall_threshold is None
            else self.floor_recall_threshold
        )
        below = self.recall < threshold
        return self._labels()[below]

    def invented(self) -> np.ndarray:
        """Features where the model puts mass the reference does not have."""
        if not self.floor_assessable:
            return self._labels()[:0]
        threshold = (
            self.floor_precision - self._margin(self.floor_precision_sd)
            if self.floor_precision_threshold is None
            else self.floor_precision_threshold
        )
        below = self.precision < threshold
        return self._labels()[below]

    def missed_labels(self) -> np.ndarray:
        """Readable labels for missed features, preserving chain identity."""
        if not self.floor_assessable:
            return self._display_labels()[:0]
        threshold = (
            self.floor_recall - self._margin(self.floor_recall_sd)
            if self.floor_recall_threshold is None
            else self.floor_recall_threshold
        )
        return self._display_labels()[self.recall < threshold]

    def invented_labels(self) -> np.ndarray:
        """Readable labels for invented features, preserving chain identity."""
        if not self.floor_assessable:
            return self._display_labels()[:0]
        threshold = (
            self.floor_precision - self._margin(self.floor_precision_sd)
            if self.floor_precision_threshold is None
            else self.floor_precision_threshold
        )
        return self._display_labels()[self.precision < threshold]

    def summary(self) -> str:
        lines = [
            f"precision {self.mean_precision:.3f} (floor {self.mean_floor_precision:.3f}), "
            f"recall {self.mean_recall:.3f} (floor {self.mean_floor_recall:.3f})"
        ]
        if not self.floor_assessable:
            lines.append(
                "  floor verdict withheld: too few independent sampling units"
            )
            return "\n".join(lines)
        missed, invented = self.missed_labels(), self.invented_labels()
        if missed.size:
            lines.append(
                f"  misses conformations at {missed.size} residue(s): "
                + ", ".join(missed[:10])
                + ("..." if missed.size > 10 else "")
            )
        if invented.size:
            lines.append(
                f"  invents conformations at {invented.size} residue(s): "
                + ", ".join(invented[:10])
                + ("..." if invented.size > 10 else "")
            )
        if not missed.size and not invented.size:
            lines.append("  nothing outside the floor: not resolvable at this sampling")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_parameter": self.order_parameter,
            "coverage": self.coverage,
            "precision": self.precision.tolist(),
            "recall": self.recall.tolist(),
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "floor_precision": self.floor_precision.tolist(),
            "floor_recall": self.floor_recall.tolist(),
            "floor_precision_sd": (
                None
                if self.floor_precision_sd is None
                else np.asarray(self.floor_precision_sd).tolist()
            ),
            "floor_recall_sd": (
                None
                if self.floor_recall_sd is None
                else np.asarray(self.floor_recall_sd).tolist()
            ),
            "mean_floor_precision": self.mean_floor_precision,
            "mean_floor_recall": self.mean_floor_recall,
            "floor_precision_threshold": (
                None
                if self.floor_precision_threshold is None
                else np.asarray(self.floor_precision_threshold).tolist()
            ),
            "floor_recall_threshold": (
                None
                if self.floor_recall_threshold is None
                else np.asarray(self.floor_recall_threshold).tolist()
            ),
            "floor_precision_distribution": (
                None
                if self.floor_precision_distribution is None
                else np.asarray(self.floor_precision_distribution).tolist()
            ),
            "floor_recall_distribution": (
                None
                if self.floor_recall_distribution is None
                else np.asarray(self.floor_recall_distribution).tolist()
            ),
            "floor_assessable": bool(self.floor_assessable),
            "missed": self.missed().astype(int).tolist(),
            "invented": self.invented().astype(int).tolist(),
            "missed_labels": self.missed_labels().tolist(),
            "invented_labels": self.invented_labels().tolist(),
            "feature_index": (
                None if self.feature_index is None else np.asarray(self.feature_index).tolist()
            ),
            "feature_labels": (
                None
                if self.feature_labels is None
                else np.asarray(self.feature_labels).tolist()
            ),
            "effective_samples": list(self.effective_samples),
            **self.metadata,
        }


def _support_threshold(density: np.ndarray, spacing: float, coverage: float) -> float:
    """Density level bounding the smallest region holding ``coverage`` mass.

    Sorting the grid by density and walking down it accumulates mass fastest,
    which is exactly the definition of a highest-density region.
    """
    order = np.argsort(density)[::-1]
    cumulative = np.cumsum(density[order]) * spacing
    total = cumulative[-1]
    if total <= 0:
        return 0.0
    reached = np.searchsorted(cumulative, coverage * total)
    return float(density[order[min(reached, order.size - 1)]])


def _mass_inside(
    values, weights, grid, density, threshold, circular: bool
) -> float:
    """Weighted fraction of a sample lying where another density exceeds a level."""
    if circular:
        interpolated = np.interp(values, grid, density, period=2 * np.pi)
    else:
        interpolated = np.interp(values, grid, density)
    inside = interpolated >= threshold
    if weights is None:
        return float(np.mean(inside))
    return float(np.sum(weights[inside]) / np.sum(weights))


def _one_feature(x, y, wx, wy, x_min, x_max, x_num, circular, coverage):
    """Precision and recall for a single feature."""
    grid_x, density_x = estimate_pdf(x, x_min, x_max, x_num, circular, wx)
    grid_y, density_y = estimate_pdf(y, x_min, x_max, x_num, circular, wy)
    spacing = float(grid_x[1] - grid_x[0])

    threshold_x = _support_threshold(density_x, spacing, coverage)
    threshold_y = _support_threshold(density_y, spacing, coverage)

    # Precision: how much of y sits where x has support.
    precision = _mass_inside(y, wy, grid_x, density_x, threshold_x, circular)
    # Recall: how much of x sits where y has support.
    recall = _mass_inside(x, wx, grid_y, density_y, threshold_y, circular)
    return precision, recall


def precision_recall(
    reference: np.ndarray,
    other: np.ndarray,
    weights_ref=None,
    weights=None,
    circular: bool = False,
    coverage: float = DEFAULT_COVERAGE,
    x_min: float | None = None,
    x_max: float | None = None,
    x_num: int = DEFAULT_GRID,
    floor_repeats: int = DEFAULT_FLOOR_REPEATS,
    random_state=None,
    feature_index=None,
    order_parameter: str = "",
    sampling_kind_ref: str = "trajectory",
    sampling_kind: str = "trajectory",
    correlation_time_frames_ref: float | None = None,
    correlation_time_frames: float | None = None,
    replica_labels_ref=None,
    replica_labels=None,
    n_jobs: int = 1,
    feature_labels=None,
) -> PrecisionRecall:
    """Split a difference into what is missed and what is invented.

    Parameters
    ----------
    reference, other
        Representation matrices. ``reference`` is the ensemble being matched --
        molecular dynamics, or an experimentally derived ensemble -- and
        ``other`` is the one being assessed against it. The two are not
        interchangeable: precision and recall swap when they are.
    coverage
        Fraction of an ensemble's mass defining its support, and the value both
        quantities take when the ensembles are drawn from the same
        distribution.
    floor_repeats
        Requested split-half repeats used to measure what a perfect model could
        score at this sampling. At least ten are used for the lower tail.
    feature_index, feature_labels
        Stable one-based residue indices and their display labels. The former
        remain numeric for downstream analysis; the latter may carry chain
        identity for summaries and serialisation.
    sampling_kind_ref, sampling_kind
        Sampling provenance of the reference and assessed ensemble.
        ``trajectory`` (default) estimates temporal correlation and splits
        complete blocks. Use ``iid`` only for independently generated
        structures. Supplying a nontrivial correlation time with ``iid`` is
        refused.
    correlation_time_frames_ref, correlation_time_frames
        Known correlation time for each trajectory, or ``None`` to estimate
        it.
    replica_labels_ref, replica_labels
        Optional label per frame on each side. Complete independent replicas
        then replace temporal blocks as the split units.

    Returns
    -------
    PrecisionRecall
    """
    reference = np.asarray(reference, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    if reference.shape[1] != other.shape[1]:
        raise ValueError(
            f"Feature counts differ ({reference.shape[1]} and {other.shape[1]}); "
            f"these representations do not describe the same residues."
        )
    if feature_index is not None and len(feature_index) != reference.shape[1]:
        raise ValueError("feature_index must contain one value per feature.")
    if feature_labels is not None and len(feature_labels) != reference.shape[1]:
        raise ValueError("feature_labels must contain one value per feature.")
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must lie strictly between 0 and 1; got {coverage}.")

    rng = np.random.default_rng(random_state)
    weights_ref = validate_weights(
        weights_ref, reference.shape[0], "Reference weights"
    )
    weights = validate_weights(weights, other.shape[0], "Compared weights")

    n_eff = (
        effective_sample_size(weights_ref, reference.shape[0]),
        effective_sample_size(weights, other.shape[0]),
    )
    for label, eff in zip(("reference", "compared"), n_eff):
        if eff < MINIMUM_EFFECTIVE_SAMPLES:
            raise ValueError(
                f"The {label} ensemble is worth {eff:.1f} independent conformations. "
                f"A support estimated from that describes those conformations, not a "
                f"distribution."
            )

    if circular:
        x_min, x_max = -np.pi, np.pi
    else:
        if x_min is None:
            x_min = float(min(reference.min(), other.min()))
        if x_max is None:
            x_max = float(max(reference.max(), other.max()))

    n_features = reference.shape[1]
    precision = np.zeros(n_features)
    recall = np.zeros(n_features)
    for i in range(n_features):
        precision[i], recall[i] = _one_feature(
            reference[:, i], other[:, i], weights_ref, weights,
            x_min, x_max, x_num, circular, coverage,
        )

    # The floor: independently assignable halves of both ensembles. Pooling
    # them makes the worse-sampled side part of the resolution limit rather
    # than pretending that only the reference contributes uncertainty.
    plans = (
        plan_floor(
            reference,
            sampling_kind=sampling_kind_ref,
            correlation_time_frames=correlation_time_frames_ref,
            replica_labels=replica_labels_ref,
            circular=circular,
        ),
        plan_floor(
            other,
            sampling_kind=sampling_kind,
            correlation_time_frames=correlation_time_frames,
            replica_labels=replica_labels,
            circular=circular,
        ),
    )
    effective_units = tuple(
        effective_floor_units(
            matrix.shape[0],
            ensemble_weights,
            plan.block_length,
            labels,
        )
        for matrix, ensemble_weights, plan, labels in zip(
            (reference, other),
            (weights_ref, weights),
            plans,
            (replica_labels_ref, replica_labels),
        )
    )
    for name, plan in zip(("reference", "comparison"), plans):
        if not plan.correlation_time_converged and plan.correlation_time >= 2.0:
            warnings.warn(
                f"The {name} correlation time is still rising with trajectory "
                f"length. Its {plan.correlation_time:.0f}-frame estimate is a "
                f"lower bound, so the precision/recall floor remains optimistic. "
                f"Sample longer before treating coverage verdicts as settled.",
                UserWarning,
                stacklevel=2,
            )
    repeats = max(MINIMUM_FLOOR_REPEATS, floor_repeats)

    def floor_statistic(left, right, wl, wr):
        values = np.zeros(2 * n_features)
        for i in range(n_features):
            values[i], values[n_features + i] = _one_feature(
                left[:, i], right[:, i], wl, wr,
                x_min, x_max, x_num, circular, coverage,
            )
        return values

    floor_distribution = split_half_floor(
        n_jobs,
        floor_statistic,
        (reference, other),
        repeats,
        rng,
        weights=(weights_ref, weights),
        block_lengths=tuple(plan.block_length for plan in plans),
        replica_labels=(replica_labels_ref, replica_labels),
        output_size=2 * n_features,
    )
    floor_p = floor_distribution[:, :n_features]
    floor_r = floor_distribution[:, n_features:]
    lower_tail = 1.0 - FLOOR_QUANTILE
    floor_p_threshold = np.clip(
        np.quantile(floor_p, lower_tail, axis=0) - FLOOR_MARGIN_MIN, 0.0, 1.0
    )
    floor_r_threshold = np.clip(
        np.quantile(floor_r, lower_tail, axis=0) - FLOOR_MARGIN_MIN, 0.0, 1.0
    )

    floor_assessable = bool(
        all(plan.assessable for plan in plans)
        and min(effective_units) >= MINIMUM_FLOOR_UNITS
    )
    if not floor_assessable:
        detail = ", ".join(
            f"{name}: {plan.n_units} {plan.strategy}, {effective:.1f} "
            f"weight-effective"
            for name, plan, effective in zip(
                ("reference", "comparison"), plans, effective_units
            )
        )
        warnings.warn(
            f"Too few independent units are available for the "
            f"precision/recall floor ({detail}; at least 8 per side are "
            f"required). Floor values are "
            f"reported descriptively, but missed/invented verdicts are withheld.",
            UserWarning,
            stacklevel=2,
        )

    result = PrecisionRecall(
        precision=precision,
        recall=recall,
        floor_precision=floor_p.mean(axis=0),
        floor_recall=floor_r.mean(axis=0),
        floor_precision_sd=floor_p.std(axis=0, ddof=1),
        floor_recall_sd=floor_r.std(axis=0, ddof=1),
        floor_precision_threshold=floor_p_threshold,
        floor_recall_threshold=floor_r_threshold,
        floor_precision_distribution=floor_p,
        floor_recall_distribution=floor_r,
        floor_assessable=floor_assessable,
        coverage=coverage,
        feature_index=None if feature_index is None else np.asarray(feature_index),
        feature_labels=(
            None if feature_labels is None else np.asarray(feature_labels, dtype=str)
        ),
        effective_samples=effective_units,
        order_parameter=order_parameter,
        metadata={
            "grid_points": x_num,
            "floor_repeats": repeats,
            "floor_quantile": lower_tail,
            "floor_sampling_kind": [plan.sampling_kind for plan in plans],
            "floor_strategy": [plan.strategy for plan in plans],
            "floor_correlation_time": [plan.correlation_time for plan in plans],
            "floor_correlation_time_converged": [
                plan.correlation_time_converged for plan in plans
            ],
            "floor_correlation_summary": [
                plan.correlation_summary for plan in plans
            ],
            "floor_assessable_features": [
                plan.n_assessable_features for plan in plans
            ],
            "floor_sampled_features": [
                plan.n_sampled_features for plan in plans
            ],
            "floor_assessable_feature_columns": [
                list(plan.assessable_features) for plan in plans
            ],
            "floor_sampled_feature_columns": [
                list(plan.sampled_features) for plan in plans
            ],
            "floor_slow_feature_columns": [
                list(plan.slow_features) for plan in plans
            ],
            "floor_block_length": [plan.block_length for plan in plans],
            "floor_units": [plan.n_units for plan in plans],
            "floor_effective_units": list(effective_units),
            "frame_weight_effective_samples": list(n_eff),
        },
    )
    logger.info("%s", result.summary().replace("\n", "; "))
    return result
