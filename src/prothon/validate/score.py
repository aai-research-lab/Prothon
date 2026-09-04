"""How well an ensemble agrees with experiment, and how well it could.

A reduced chi-squared of 1 is the usual target: predictions differ from
measurements by about the experimental uncertainty, no more and no less. For an
ensemble that target is wrong in both directions, and the reason is the same one
that runs through the rest of this package -- the prediction is itself an
estimate, made from a finite sample.

Measured on a synthetic ensemble whose true average *is* the experimental
value, so the only error is sampling::

    conformations    chi2_red of a perfect ensemble
               20                              0.77
               50                              0.33
              200                              0.08
             1000                              0.02
             5000                              0.00

**A perfect ensemble of twenty conformations scores 0.77, and a perfect
ensemble of five thousand scores 0.00.** Fitting either to 1.0 is fitting to
noise: in the first case the ensemble is being asked to reproduce experimental
scatter it cannot know about, and in the second it is being asked to reproduce
scatter it has already averaged away.

So a chi-squared is reported here beside a **floor**, obtained by assigning
independent frames, complete temporal blocks, or complete replicas to two
halves and scoring one half's prediction against the other's. The mean
describes what sampling contributes; the 95th percentile decides whether the
experimental disagreement clears it.

    Bottaro, S.; Lindorff-Larsen, K. Biophysical experiments and biomolecular
    simulations: a perfect match? Science 2018, 361, 355-360.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..compare.dissimilarity import MINIMUM_EFFECTIVE_SAMPLES, effective_sample_size
from ..sampling.floor import (
    FLOOR_QUANTILE,
    MINIMUM_FLOOR_REPEATS,
    plan_floor,
    split_half_floor,
)
from ..sampling.statistics import validate_weights
from ..utils import get_logger

logger = get_logger("validate.score")

__all__ = ["AgreementResult", "score_observable"]

#: Split-half repeats behind the floor.
DEFAULT_FLOOR_REPEATS = 20


@dataclass
class AgreementResult:
    """How an ensemble's predictions compare with measurements.

    Attributes
    ----------
    chi2_reduced
        Mean squared deviation in units of the experimental uncertainty.
    floor
        The same quantity between two halves of this ensemble: what the
        sampling contributes on its own, reported as a descriptive mean.
    floor_threshold
        The 95th percentile of the floor distribution used for the verdict.
    within_floor
        Whether the agreement is already inside the sampling limit. ``None``
        when fewer than eight independent units make a verdict unsupported.
    labels, feature_index
        Readable measurement labels and stable one-based numeric indices.
    """

    observable: str
    chi2_reduced: float
    floor: float
    floor_sd: float
    n_points: int
    n_frames: int
    effective_samples: float
    predicted: np.ndarray
    experimental: np.ndarray
    uncertainty: np.ndarray
    residuals: np.ndarray
    labels: np.ndarray | None = None
    floor_threshold: float | None = None
    floor_distribution: np.ndarray | None = None
    floor_assessable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    feature_index: np.ndarray | None = None

    @property
    def within_floor(self) -> bool | None:
        if not self.floor_assessable:
            return None
        threshold = self.floor if self.floor_threshold is None else self.floor_threshold
        return bool(self.chi2_reduced <= threshold)

    @property
    def worst(self) -> list[tuple[Any, float]]:
        """The points contributing most, largest first, up to five."""
        order = np.argsort(np.abs(self.residuals))[::-1][:5]
        labels = (
            (
                np.arange(1, self.residuals.size + 1)
                if self.feature_index is None
                else self.feature_index
            )
            if self.labels is None
            else self.labels
        )
        return [(labels[i], float(self.residuals[i])) for i in order]

    def summary(self) -> str:
        if self.within_floor is None:
            verdict = "floor verdict withheld: too few independent sampling units"
        elif self.within_floor:
            verdict = "agrees to within its own sampling"
        else:
            verdict = "disagrees beyond what the sampling explains"
        threshold = self.floor if self.floor_threshold is None else self.floor_threshold
        lines = [
            f"{self.observable}: chi2_red = {self.chi2_reduced:.2f} "
            f"(floor mean {self.floor:.2f}, q95 {threshold:.2f}) — {verdict}"
        ]
        if self.within_floor is False:
            worst = ", ".join(f"{label} ({r:+.1f}σ)" for label, r in self.worst[:3])
            lines.append(f"  largest deviations: {worst}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observable": self.observable,
            "chi2_reduced": float(self.chi2_reduced),
            "floor": float(self.floor),
            "floor_sd": float(self.floor_sd),
            "floor_threshold": (
                float(self.floor)
                if self.floor_threshold is None
                else float(self.floor_threshold)
            ),
            "floor_distribution": (
                None
                if self.floor_distribution is None
                else np.asarray(self.floor_distribution).tolist()
            ),
            "floor_assessable": bool(self.floor_assessable),
            "within_floor": self.within_floor,
            "n_points": self.n_points,
            "n_frames": self.n_frames,
            "effective_samples": float(self.effective_samples),
            "predicted": self.predicted.tolist(),
            "experimental": self.experimental.tolist(),
            "uncertainty": self.uncertainty.tolist(),
            "residuals": self.residuals.tolist(),
            "labels": None if self.labels is None else np.asarray(self.labels).tolist(),
            "feature_index": (
                None
                if self.feature_index is None
                else np.asarray(self.feature_index).tolist()
            ),
            **self.metadata,
        }


def _chi2_reduced(predicted, experimental, uncertainty) -> float:
    return float(np.mean(((predicted - experimental) / uncertainty) ** 2))


def score_observable(
    per_frame: np.ndarray,
    experimental,
    uncertainty,
    observable: str = "observable",
    weights=None,
    averaging: str = "linear",
    labels=None,
    floor_repeats: int = DEFAULT_FLOOR_REPEATS,
    random_state=None,
    sampling_kind: str = "trajectory",
    correlation_time_frames: float | None = None,
    replica_labels=None,
    n_jobs: int = 1,
    feature_index=None,
) -> AgreementResult:
    """Score an ensemble's predictions against measurements.

    Parameters
    ----------
    per_frame
        ``(n_frames, n_points)`` predicted values, one column per measurement.
        Predictions from anywhere are accepted, so chemical shifts from
        SPARTA+ or a SAXS profile from CRYSOL can be scored here even though
        Prothon does not compute them.
    experimental
        ``(n_points,)`` measured values.
    uncertainty
        ``(n_points,)`` experimental uncertainties. Not optional: a
        chi-squared without them is a sum of squares in arbitrary units, and
        the floor it is compared against would be meaningless.
    averaging
        ``linear``, or ``r6`` for a distance reported through an inverse
        sixth-power interaction.
    labels, feature_index
        Optional display labels and stable one-based numeric indices for the
        measured points. Computed residue observables use both so multichain
        labels remain readable without sacrificing a machine key.
    floor_repeats
        Split-half repeats behind the floor.
    sampling_kind
        ``trajectory`` (default) estimates temporal correlation and splits
        complete blocks. Use ``iid`` only for independently generated
        structures.
    correlation_time_frames
        Known trajectory correlation time, or ``None`` to estimate it.
    replica_labels
        Optional label per frame. Complete independent replicas then replace
        temporal blocks as the split units.

    Returns
    -------
    AgreementResult
    """
    from .observables import average_observable

    per_frame = np.atleast_2d(np.asarray(per_frame, dtype=np.float64))
    if per_frame.shape[0] == 1 and per_frame.shape[1] > 1:
        # A single row is ambiguous; assume it is one frame of many points
        # only if the caller gave that shape deliberately.
        pass
    experimental = np.asarray(experimental, dtype=np.float64).ravel()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).ravel()

    if per_frame.shape[1] != experimental.size:
        raise ValueError(
            f"{per_frame.shape[1]} predicted values per frame against "
            f"{experimental.size} measurements. These do not describe the same "
            f"observables."
        )
    if uncertainty.size != experimental.size:
        raise ValueError(
            f"{uncertainty.size} uncertainties for {experimental.size} "
            f"measurements."
        )
    if labels is not None and len(labels) != experimental.size:
        raise ValueError("labels must contain one value per measurement.")
    if feature_index is not None and len(feature_index) != experimental.size:
        raise ValueError("feature_index must contain one value per measurement.")
    if np.any(uncertainty <= 0):
        raise ValueError(
            "Experimental uncertainties must be positive. A chi-squared "
            "without them is a sum of squares in arbitrary units, and the "
            "floor it would be compared against means nothing."
        )

    n_frames = per_frame.shape[0]
    weights = validate_weights(weights, n_frames)
    n_eff = effective_sample_size(weights, n_frames)
    if n_eff < MINIMUM_EFFECTIVE_SAMPLES:
        raise ValueError(
            f"The ensemble is worth {n_eff:.1f} independent conformations. An "
            f"ensemble average from that describes those conformations rather "
            f"than a distribution, and the agreement would not mean what it "
            f"appears to."
        )

    predicted = average_observable(per_frame, weights, averaging)
    chi2 = _chi2_reduced(predicted, experimental, uncertainty)

    # The floor: one independently assignable half's prediction scored against
    # the other's. Not against the experiment -- the question is what the
    # sampling alone contributes, and the experiment is common to both halves.
    rng = np.random.default_rng(random_state)
    plan = plan_floor(
        per_frame,
        sampling_kind=sampling_kind,
        correlation_time_frames=correlation_time_frames,
        replica_labels=replica_labels,
    )
    if not plan.correlation_time_converged and plan.correlation_time >= 2.0:
        warnings.warn(
            f"The correlation time is still rising with trajectory length. Its "
            f"{plan.correlation_time:.0f}-frame estimate is a lower bound, so "
            f"the experimental-agreement floor remains optimistic. Sample "
            f"longer before treating the verdict as settled.",
            UserWarning,
            stacklevel=2,
        )

    def floor_statistic(left, right, wl, wr):
        a = average_observable(left, wl, averaging)
        b = average_observable(right, wr, averaging)
        return np.array([_chi2_reduced(a, b, uncertainty)])

    repeats = max(MINIMUM_FLOOR_REPEATS, floor_repeats)
    floors = split_half_floor(
        n_jobs,
        floor_statistic,
        (per_frame,),
        repeats,
        rng,
        weights=(None if weights is None else np.asarray(weights),),
        block_lengths=plan.block_length,
        replica_labels=(replica_labels,),
        output_size=1,
    ).ravel()
    floor = float(np.mean(floors))
    floor_sd = float(np.std(floors, ddof=1)) if floors.size > 1 else 0.0
    floor_threshold = float(np.quantile(floors, FLOOR_QUANTILE))

    if not plan.assessable:
        warnings.warn(
            f"Only {plan.n_units} independent {plan.strategy} are available "
            f"for the experimental-agreement floor, fewer than 8. Floor values "
            f"are reported descriptively, but within-floor verdicts are withheld.",
            UserWarning,
            stacklevel=2,
        )

    result = AgreementResult(
        observable=observable,
        chi2_reduced=chi2,
        floor=floor,
        floor_sd=floor_sd,
        n_points=int(experimental.size),
        n_frames=int(n_frames),
        effective_samples=float(n_eff),
        predicted=np.asarray(predicted, dtype=np.float64),
        experimental=experimental,
        uncertainty=uncertainty,
        residuals=(np.asarray(predicted) - experimental) / uncertainty,
        labels=None if labels is None else np.asarray(labels),
        feature_index=(
            None if feature_index is None else np.asarray(feature_index, dtype=int)
        ),
        floor_threshold=floor_threshold,
        floor_distribution=floors,
        floor_assessable=plan.assessable,
        metadata={
            "averaging": averaging,
            "floor_repeats": repeats,
            "floor_quantile": FLOOR_QUANTILE,
            "floor_sampling_kind": plan.sampling_kind,
            "floor_strategy": plan.strategy,
            "floor_correlation_time": plan.correlation_time,
            "floor_correlation_time_converged": plan.correlation_time_converged,
            "floor_correlation_summary": plan.correlation_summary,
            "floor_assessable_features": plan.n_assessable_features,
            "floor_sampled_features": plan.n_sampled_features,
            "floor_assessable_feature_columns": list(plan.assessable_features),
            "floor_sampled_feature_columns": list(plan.sampled_features),
            "floor_slow_feature_columns": list(plan.slow_features),
            "floor_block_length": plan.block_length,
            "floor_units": plan.n_units,
        },
    )
    logger.info("%s", result.summary().replace("\n", "; "))
    return result
