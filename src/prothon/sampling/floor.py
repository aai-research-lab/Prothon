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

As with :mod:`prothon.sampling.null`, the statistic arrives as a callable. What
a floor is does not depend on which distance is being floored.
"""

from __future__ import annotations

import numpy as np

from ..utils import get_logger

__all__ = ["split_half_floor"]

logger = get_logger("floor")


def split_half_floor(
    n_jobs: int,
    statistic,
    ensembles: tuple[np.ndarray, ...],
    repeats: int,
    rng: np.random.Generator,
    weights: tuple = (None, None),
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
    """
    def one_split(ensemble, w, half, seed):
        rng = np.random.default_rng(seed)
        order = rng.permutation(ensemble.shape[0])
        left, right = order[:half], order[half : 2 * half]
        wl = wr = None
        if w is not None:
            wl, wr = w[left], w[right]
            wl, wr = wl / wl.sum(), wr / wr.sum()
        return statistic(ensemble[left], ensemble[right], wl, wr)

    jobs = []
    for ensemble, w in zip(ensembles, weights):
        half = ensemble.shape[0] // 2
        if half < 2:
            continue
        for seed in rng.integers(0, 2**63 - 1, size=repeats):
            jobs.append((ensemble, w, half, seed))

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
        values = [row for group in batched for row in group]
        if not values:
            return np.zeros((1, ensembles[0].shape[1]))
        return np.stack(values, axis=0)

    values = [one_split(*job) for job in jobs]
    if not values:
        return np.zeros((1, ensembles[0].shape[1]))
    return np.stack(values, axis=0)
