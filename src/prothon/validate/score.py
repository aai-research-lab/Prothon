"""How well an ensemble agrees with experiment, and how well it could.

A reduced chi-squared of 1 is the usual target: predictions differ from
measurements by about the experimental uncertainty, no more and no less. For an
ensemble that target is wrong in both directions, and the reason is the same one
that runs through the rest of this package -- the prediction is itself an
estimate, made from a finite sample.

Measured on a synthetic ensemble whose true average *is* the experimental
value, so the only error is sampling:

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

So a chi-squared is reported here beside a **floor**, obtained by splitting the
ensemble in half and scoring one half's prediction against the other's. That is
what the sampling alone contributes, and an ensemble already inside it cannot
be improved by more conformations of the same kind.

    Bottaro, S.; Lindorff-Larsen, K. Biophysical experiments and biomolecular
    simulations: a perfect match? Science 2018, 361, 355-360.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.dissimilarity import MINIMUM_EFFECTIVE_SAMPLES, effective_sample_size
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
        sampling contributes on its own. A value at or below the floor is as
        good as this much sampling permits.
    within_floor
        Whether the agreement is already inside the sampling limit. When true,
        the ensemble cannot be improved by drawing more conformations of the
        same kind, and a smaller chi-squared would mean overfitting.
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
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def within_floor(self) -> bool:
        return bool(self.chi2_reduced <= self.floor)

    @property
    def worst(self) -> list[tuple[Any, float]]:
        """The points contributing most, largest first, up to five."""
        order = np.argsort(np.abs(self.residuals))[::-1][:5]
        labels = (
            np.arange(1, self.residuals.size + 1)
            if self.labels is None
            else self.labels
        )
        return [(labels[i], float(self.residuals[i])) for i in order]

    def summary(self) -> str:
        verdict = (
            "agrees to within its own sampling"
            if self.within_floor
            else "disagrees beyond what the sampling explains"
        )
        lines = [
            f"{self.observable}: chi2_red = {self.chi2_reduced:.2f} "
            f"(floor {self.floor:.2f} +/- {self.floor_sd:.2f}) — {verdict}"
        ]
        if not self.within_floor:
            worst = ", ".join(f"{label} ({r:+.1f}σ)" for label, r in self.worst[:3])
            lines.append(f"  largest deviations: {worst}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observable": self.observable,
            "chi2_reduced": float(self.chi2_reduced),
            "floor": float(self.floor),
            "floor_sd": float(self.floor_sd),
            "within_floor": self.within_floor,
            "n_points": self.n_points,
            "n_frames": self.n_frames,
            "effective_samples": float(self.effective_samples),
            "predicted": self.predicted.tolist(),
            "experimental": self.experimental.tolist(),
            "uncertainty": self.uncertainty.tolist(),
            "residuals": self.residuals.tolist(),
            "labels": None if self.labels is None else np.asarray(self.labels).tolist(),
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
    floor_repeats
        Split-half repeats behind the floor.

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
    if np.any(uncertainty <= 0):
        raise ValueError(
            "Experimental uncertainties must be positive. A chi-squared "
            "without them is a sum of squares in arbitrary units, and the "
            "floor it would be compared against means nothing."
        )

    n_frames = per_frame.shape[0]
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

    # The floor: one half's prediction scored against the other's. Not against
    # the experiment -- the question is what the sampling alone contributes,
    # and the experiment is common to both halves.
    rng = np.random.default_rng(random_state)
    half = n_frames // 2
    floors = []
    if half >= 2:
        for _ in range(max(2, floor_repeats)):
            order = rng.permutation(n_frames)
            left, right = order[:half], order[half : 2 * half]
            wl = None if weights is None else np.asarray(weights)[left]
            wr = None if weights is None else np.asarray(weights)[right]
            a = average_observable(per_frame[left], wl, averaging)
            b = average_observable(per_frame[right], wr, averaging)
            floors.append(_chi2_reduced(a, b, uncertainty))
    floor = float(np.mean(floors)) if floors else 0.0
    floor_sd = float(np.std(floors, ddof=1)) if len(floors) > 1 else 0.0

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
        metadata={"averaging": averaging, "floor_repeats": floor_repeats},
    )
    logger.info("%s", result.summary().replace("\n", "; "))
    return result
