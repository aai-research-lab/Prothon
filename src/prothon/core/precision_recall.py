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

**And it carries a floor.** Two halves of one ensemble do not reach each
other's support perfectly either, because a finite sample never covers a
continuous distribution. Prothon measures that self-precision and
self-recall and reports both beside the result, for the same reason it reports
a noise floor beside a dissimilarity: a model cannot be asked to do better than
the reference can do against itself.

    Sajjadi, M. S. M.; Bachem, O.; Lucic, M.; Bousquet, O.; Gelly, S.
    Assessing generative models via precision and recall. Advances in Neural
    Information Processing Systems, 2018.

    Kynkaanniemi, T.; Karras, T.; Laine, S.; Lehtinen, J.; Aila, T. Improved
    precision and recall metric for assessing generative models. Advances in
    Neural Information Processing Systems, 2019.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

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

#: Repeats of the split-half self-comparison that establishes the floor. Enough
#: to get a spread as well as a level, since the spread is what separates a
#: residue that genuinely falls short from one that is merely sampled.
DEFAULT_FLOOR_REPEATS = 5

#: A residue must fall this many standard deviations below its own floor to be
#: called. The floor varies from residue to residue -- a rigid one is easy to
#: cover and a mobile one is not -- so a single averaged threshold flags about
#: half of the unchanged residues in a protein by construction.
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
        The same quantities measured between two halves of the reference, per
        feature. The best a perfect model could score at this sampling. Per
        feature rather than averaged: a rigid residue is easy to cover and a
        mobile one is not, and one threshold for both calls about half the
        unchanged residues in a protein.
    coverage
        The level defining the support, and the value both quantities take
        under identical distributions.
    """

    precision: np.ndarray
    recall: np.ndarray
    floor_precision: np.ndarray
    floor_recall: np.ndarray
    floor_precision_sd: np.ndarray | None = None
    floor_recall_sd: np.ndarray | None = None
    coverage: float = DEFAULT_COVERAGE
    feature_index: np.ndarray | None = None
    effective_samples: tuple[float, float] = (0.0, 0.0)
    measure: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

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

    def missed(self) -> np.ndarray:
        """Features where the model fails to reach the reference's support."""
        below = self.recall < self.floor_recall - self._margin(self.floor_recall_sd)
        return self._labels()[below]

    def invented(self) -> np.ndarray:
        """Features where the model puts mass the reference does not have."""
        below = (
            self.precision < self.floor_precision - self._margin(self.floor_precision_sd)
        )
        return self._labels()[below]

    def summary(self) -> str:
        lines = [
            f"precision {self.mean_precision:.3f} (floor {self.mean_floor_precision:.3f}), "
            f"recall {self.mean_recall:.3f} (floor {self.mean_floor_recall:.3f})"
        ]
        missed, invented = self.missed(), self.invented()
        if missed.size:
            lines.append(
                f"  misses conformations at {missed.size} residue(s): "
                + ", ".join(str(int(i)) for i in missed[:10])
                + ("..." if missed.size > 10 else "")
            )
        if invented.size:
            lines.append(
                f"  invents conformations at {invented.size} residue(s): "
                + ", ".join(str(int(i)) for i in invented[:10])
                + ("..." if invented.size > 10 else "")
            )
        if not missed.size and not invented.size:
            lines.append("  nothing outside the floor: not resolvable at this sampling")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "measure": self.measure,
            "coverage": self.coverage,
            "precision": self.precision.tolist(),
            "recall": self.recall.tolist(),
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "floor_precision": self.floor_precision.tolist(),
            "floor_recall": self.floor_recall.tolist(),
            "mean_floor_precision": self.mean_floor_precision,
            "mean_floor_recall": self.mean_floor_recall,
            "missed": self.missed().astype(int).tolist(),
            "invented": self.invented().astype(int).tolist(),
            "feature_index": (
                None if self.feature_index is None else np.asarray(self.feature_index).tolist()
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
    measure: str = "",
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
        Split-half repeats used to measure what a perfect model could score at
        this sampling.

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
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must lie strictly between 0 and 1; got {coverage}.")

    rng = np.random.default_rng(random_state)
    weights_ref = None if weights_ref is None else np.asarray(weights_ref, float)
    weights = None if weights is None else np.asarray(weights, float)

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

    # The floor: two halves of the reference, which is the best a model could
    # score against this much sampling of it.
    repeats = max(2, floor_repeats)
    floor_p = np.zeros((repeats, n_features))
    floor_r = np.zeros((repeats, n_features))
    half = reference.shape[0] // 2
    for k in range(repeats):
        order = rng.permutation(reference.shape[0])
        left, right = order[:half], order[half : 2 * half]
        wl = None if weights_ref is None else weights_ref[left]
        wr = None if weights_ref is None else weights_ref[right]
        for i in range(n_features):
            floor_p[k, i], floor_r[k, i] = _one_feature(
                reference[left, i], reference[right, i], wl, wr,
                x_min, x_max, x_num, circular, coverage,
            )

    result = PrecisionRecall(
        precision=precision,
        recall=recall,
        floor_precision=floor_p.mean(axis=0),
        floor_recall=floor_r.mean(axis=0),
        floor_precision_sd=floor_p.std(axis=0, ddof=1),
        floor_recall_sd=floor_r.std(axis=0, ddof=1),
        coverage=coverage,
        feature_index=None if feature_index is None else np.asarray(feature_index),
        effective_samples=n_eff,
        measure=measure,
        metadata={"grid_points": x_num, "floor_repeats": floor_repeats},
    )
    logger.info("%s", result.summary().replace("\n", "; "))
    return result
