"""The smallest difference the sampling can resolve.

A finite sample never reproduces a continuous distribution exactly, so two
independent halves of a *single* ensemble are not at distance zero from each
other. That self-distance is the resolution limit of a comparison, and it is
measured rather than assumed: each ensemble is split at random into disjoint
halves, the statistic is evaluated between them, and the procedure repeated.

Disjoint halves are used because they are the only pair of samples guaranteed
to come from one distribution without assumption. A bootstrap treats the sample
as the population and gives a floor about half what it should be; a parametric
reference imposes a shape.

For a trajectory, however, random rows are not independent halves. They mix
every slow excursion into both sides and make the ensemble look more precisely
sampled than it is. The exchangeable unit is therefore a complete temporal
block, or a complete independent replica when replica labels are available.

As with :mod:`prothon.sampling.null`, the statistic arrives as a callable. What
a floor is does not depend on which distance is being floored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import get_logger
from .correlation import block_labels, correlation_time_estimate, plan_blocks
from .statistics import effective_sample_size

__all__ = [
    "FLOOR_QUANTILE",
    "MINIMUM_FLOOR_REPEATS",
    "MINIMUM_FLOOR_UNITS",
    "FloorPlan",
    "effective_floor_units",
    "floor_unit_count",
    "plan_floor",
    "split_half_floor",
]

logger = get_logger("floor")

#: Upper tail used when a larger statistic means worse agreement. The mean is
#: descriptive; a 95th-percentile threshold controls the chance that ordinary
#: split-half sampling variation alone is called resolved.
FLOOR_QUANTILE = 0.95

#: Per ensemble. Two ensembles therefore contribute at least twenty values to
#: a comparison floor, enough for the upper tail not to be merely its maximum.
MINIMUM_FLOOR_REPEATS = 10

#: Eight balanced units have 35 unique two-half partitions after mirror images
#: are identified. Below that, even an exact split distribution cannot resolve
#: a 5% tail reliably enough to issue a verdict.
MINIMUM_FLOOR_UNITS = 8


@dataclass(frozen=True)
class FloorPlan:
    """How an ensemble may honestly be divided for a sampling floor."""

    sampling_kind: str
    strategy: str
    correlation_time: float
    correlation_time_converged: bool
    block_length: int
    n_units: int
    assessable: bool
    correlation_summary: str = "supplied"
    assessable_features: tuple[int, ...] = ()
    sampled_features: tuple[int, ...] = ()
    slow_features: tuple[int, ...] = ()

    @property
    def n_assessable_features(self) -> int:
        return len(self.assessable_features)

    @property
    def n_sampled_features(self) -> int:
        return len(self.sampled_features)


def _exchangeable_units(
    n_frames: int,
    block_length: int = 1,
    replica_labels: np.ndarray | None = None,
) -> list[np.ndarray] | None:
    if replica_labels is not None:
        replicas = np.asarray(replica_labels)
        if replicas.ndim != 1 or replicas.size != n_frames:
            raise ValueError(
                "Replica labels must be one-dimensional with one label per frame."
            )
        _, inverse = np.unique(replicas, return_inverse=True)
        return [
            np.flatnonzero(inverse == label)
            for label in range(int(inverse.max()) + 1)
        ]
    if block_length > 1:
        labels = block_labels(n_frames, block_length)
        return [np.flatnonzero(labels == label) for label in np.unique(labels)]
    return None


def floor_unit_count(
    n_frames: int,
    block_length: int = 1,
    replica_labels: np.ndarray | None = None,
) -> int:
    """Number of independently assignable units available to a floor."""
    units = _exchangeable_units(n_frames, block_length, replica_labels)
    return n_frames if units is None else len(units)


def effective_floor_units(
    n_frames: int,
    weights=None,
    block_length: int = 1,
    replica_labels: np.ndarray | None = None,
) -> float:
    """Kish-effective independently assignable units in an ensemble."""
    units = _exchangeable_units(n_frames, block_length, replica_labels)
    if units is None:
        return effective_sample_size(weights, n_frames)
    mass = (
        np.full(n_frames, 1.0 / n_frames)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    mass = mass / mass.sum()
    unit_mass = np.array([mass[unit].sum() for unit in units])
    return effective_sample_size(unit_mass)


def plan_floor(
    matrix: np.ndarray,
    sampling_kind: str = "trajectory",
    correlation_time_frames: float | None = None,
    replica_labels: np.ndarray | None = None,
    circular: bool = False,
) -> FloorPlan:
    """Choose independent frames, temporal blocks, or complete replicas.

    ``trajectory`` is the conservative default because an array does not carry
    provenance. Callers with genuinely independent generated structures must
    say ``iid`` explicitly; combining that claim with a nontrivial correlation
    time is refused as internally inconsistent metadata.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("A floor plan requires a 2-D (frames, features) matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("A floor plan requires at least one frame and one feature.")
    kind = str(sampling_kind).strip().lower()
    if kind not in {"trajectory", "iid"}:
        raise ValueError("sampling_kind must be 'trajectory' or 'iid'.")

    if correlation_time_frames is not None:
        tau = float(correlation_time_frames)
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("correlation_time_frames must be finite and positive.")
        if kind == "iid" and tau > 1.0:
            raise ValueError(
                "IID sampling cannot have a correlation time greater than one frame."
            )
    else:
        tau = 1.0

    if replica_labels is not None:
        n_units = floor_unit_count(matrix.shape[0], 1, replica_labels)
        return FloorPlan(
            sampling_kind=kind,
            strategy="independent replicas",
            correlation_time=tau,
            correlation_time_converged=True,
            block_length=1,
            n_units=n_units,
            assessable=n_units >= MINIMUM_FLOOR_UNITS,
            correlation_summary="not required (replica labels)",
        )

    converged = True
    correlation_summary = (
        "supplied" if correlation_time_frames is not None else "not estimated"
    )
    assessable_features: tuple[int, ...] = ()
    sampled_features: tuple[int, ...] = ()
    slow_features: tuple[int, ...] = ()
    if kind == "trajectory":
        if correlation_time_frames is None:
            estimate = correlation_time_estimate(matrix, circular=circular)
            tau = float(estimate.tau)
            converged = bool(estimate.converged)
            correlation_summary = estimate.summary
            assessable_features = estimate.assessable_features
            sampled_features = estimate.sampled_features
            slow_features = estimate.slow_features
        block_length, _ = plan_blocks(matrix.shape[0], tau)
        strategy = "temporal blocks" if block_length > 1 else "uncorrelated frames"
    else:
        tau = 1.0
        block_length = 1
        strategy = "IID frames"

    n_units = floor_unit_count(matrix.shape[0], block_length)
    return FloorPlan(
        sampling_kind=kind,
        strategy=strategy,
        correlation_time=tau,
        correlation_time_converged=converged,
        block_length=block_length,
        n_units=n_units,
        assessable=n_units >= MINIMUM_FLOOR_UNITS,
        correlation_summary=correlation_summary,
        assessable_features=assessable_features,
        sampled_features=sampled_features,
        slow_features=slow_features,
    )


def split_half_floor(
    n_jobs: int,
    statistic,
    ensembles: tuple[np.ndarray, ...],
    repeats: int,
    rng: np.random.Generator,
    weights: tuple | None = None,
    block_lengths: int | tuple[int, ...] = 1,
    replica_labels: tuple[np.ndarray | None, ...] | np.ndarray | None = None,
    output_size: int | None = None,
) -> np.ndarray:
    """Distance between two disjoint halves of each ensemble.

    The resolution limit: what two samples of the same distribution look like
    at this much sampling. A difference between two ensembles smaller than
    this is not a difference.

    Halves of one ensemble are the only pair of samples guaranteed to come
    from the same distribution without assuming what that distribution is. A
    bootstrap assumes the sample is the population; a parametric reference
    assumes a shape. Two halves differ only by sampling.

    **The result is conservative by about a quarter.** Halves have half the
    frames, so this measures the limit at n/2 while the study has n: a
    1000-frame ensemble reports about 0.063 where the limit at 1000 frames is
    0.050. The error is in the safe direction -- a difference called resolvable
    is resolvable -- and correcting it would mean assuming the distance scales
    as n^(-1/2) for every metric, which has been measured only for the
    Jensen-Shannon distance on Gaussians.

    Both ensembles are split, not only the reference, and the results pooled:
    the resolution limit of a comparison is set by whichever side is sampled
    worse.

    ``block_lengths`` gives the indivisible temporal-block length for each
    ensemble. A scalar applies to all ensembles. When ``replica_labels`` are
    supplied, complete replicas are the units instead and blocks never cross a
    replica boundary. The original random disjoint split is retained only when
    the block length is one and no replica labels are supplied.

    ``output_size`` is needed only when the statistic does not return one value
    per input feature. It lets an unsplittable floor return a correctly shaped
    row of missing values rather than inventing zeros.
    """
    if not ensembles:
        raise ValueError("At least one ensemble is required to measure a floor.")
    if repeats < 1:
        raise ValueError("Floor repeats must be a positive integer.")
    ensembles = tuple(np.asarray(ensemble) for ensemble in ensembles)
    if any(ensemble.ndim != 2 for ensemble in ensembles):
        raise ValueError("Floor ensembles must be 2-D (frames, features) matrices.")
    if any(ensemble.shape[0] == 0 or ensemble.shape[1] == 0 for ensemble in ensembles):
        raise ValueError("Floor ensembles must contain frames and features.")
    n_features = ensembles[0].shape[1]
    output_size = n_features if output_size is None else int(output_size)
    if output_size < 1:
        raise ValueError("output_size must be a positive integer.")
    if any(ensemble.shape[1] != n_features for ensemble in ensembles):
        raise ValueError("Every floor ensemble must have the same feature count.")

    n_ensembles = len(ensembles)
    if weights is None:
        weights = (None,) * n_ensembles
    elif len(weights) != n_ensembles:
        raise ValueError(
            f"Expected weights for {n_ensembles} ensembles; got {len(weights)}."
        )
    weights = tuple(None if w is None else np.asarray(w, dtype=float) for w in weights)
    for ensemble, w in zip(ensembles, weights):
        if w is not None and (w.ndim != 1 or w.size != ensemble.shape[0]):
            raise ValueError("Weights must contain one value per ensemble frame.")

    if np.isscalar(block_lengths):
        block_lengths = (int(block_lengths),) * n_ensembles
    else:
        block_lengths = tuple(int(length) for length in block_lengths)
    if len(block_lengths) != n_ensembles:
        raise ValueError(
            f"Expected block lengths for {n_ensembles} ensembles; "
            f"got {len(block_lengths)}."
        )
    if any(length < 1 for length in block_lengths):
        raise ValueError("Block lengths must be positive integers.")

    if replica_labels is None:
        replica_labels = (None,) * n_ensembles
    elif n_ensembles == 1 and isinstance(replica_labels, np.ndarray):
        replica_labels = (replica_labels,)
    else:
        replica_labels = tuple(replica_labels)
    if len(replica_labels) != n_ensembles:
        raise ValueError(
            f"Expected replica labels for {n_ensembles} ensembles; "
            f"got {len(replica_labels)}."
        )

    def one_split(ensemble, w, seed, units):
        local_rng = np.random.default_rng(seed)
        if units is None:
            half = ensemble.shape[0] // 2
            order = local_rng.permutation(ensemble.shape[0])
            left, right = order[:half], order[half : 2 * half]
        else:
            half = len(units) // 2
            if half == 0:
                return None
            order = local_rng.permutation(len(units))
            left = np.concatenate([units[i] for i in order[:half]])
            right = np.concatenate([units[i] for i in order[half : 2 * half]])
        wl = wr = None
        if w is not None:
            wl, wr = w[left], w[right]
            if wl.sum() <= 0 or wr.sum() <= 0:
                return None
            wl, wr = wl / wl.sum(), wr / wr.sum()
        return statistic(ensemble[left], ensemble[right], wl, wr)

    jobs = []
    for ensemble, w, block_length, replicas in zip(
        ensembles, weights, block_lengths, replica_labels
    ):
        if ensemble.shape[0] // 2 < 2:
            continue
        units = _exchangeable_units(ensemble.shape[0], block_length, replicas)
        if units is not None and len(units) < 2:
            continue
        for seed in rng.integers(0, 2**63 - 1, size=repeats):
            jobs.append((ensemble, w, seed, units))

    if n_jobs != 1 and len(jobs) > 1:
        import os

        from joblib import Parallel, delayed

        workers = os.cpu_count() or 1 if n_jobs < 0 else n_jobs
        groups = np.array_split(np.arange(len(jobs)), min(workers, len(jobs)))
        batched = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(lambda g: [one_split(*jobs[i]) for i in g])(group)
            for group in groups
            if group.size
        )
        values = [row for group in batched for row in group if row is not None]
        if not values:
            return np.full((1, output_size), np.nan)
        return np.stack(values, axis=0)

    values = [value for job in jobs if (value := one_split(*job)) is not None]
    if not values:
        return np.full((1, output_size), np.nan)
    return np.stack(values, axis=0)
