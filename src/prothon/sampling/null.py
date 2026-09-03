"""The null distribution, built by relabelling rather than assumed.

Under the hypothesis that two ensembles sample the same distribution, the
labels carry no information, so any relabelling is as likely as the one
observed. Distances from many relabellings are the null distribution of the
statistic, and no assumption about its shape is needed.

**Blocks, not frames.** Consecutive conformations of a trajectory are nearly
the same conformation and are not exchangeable. Relabelling individual frames
gives a null far too narrow: at a correlation time of twenty frames it calls
99% of residues different when nothing differs. Relabelling contiguous blocks
holds between 1.7% and 2.3% across correlation times spanning a factor of fifty
(``docs/calibration.md``).

Nothing here knows what a distance is. The statistic arrives as a callable
taking two samples and their weights and returning one value per feature, which
is why this module does not import from :mod:`prothon.compare` and the two
packages do not depend on each other in a circle. It is also the honest shape:
block permutation is a resampling procedure, and the choice of statistic is the
caller's business.
"""

from __future__ import annotations

import numpy as np

from ..utils import get_logger
from .correlation import block_labels
from .statistics import benjamini_hochberg

__all__ = ["permutation_null", "studentised_p_values"]

logger = get_logger("null")


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


def _one_permutation(
    k,
    seeds,
    blocks,
    total,
    n_reference,
    n_reference_blocks,
    pooled,
    pooled_w,
    weighted,
    statistic,
):
    """One relabelling, computed from its own seed.

    Split out so it can run in a worker process. It takes a seed rather than a
    generator because a generator does not survive being sent to one, and
    because seeding per permutation is what makes a parallel run and a serial
    run agree.
    """
    rng = np.random.default_rng(seeds[k])
    if blocks is None:
        order = rng.permutation(total)
        left, right = order[:n_reference], order[n_reference:]
    else:
        # Blocks are the observations in this design. Preserve the number of
        # blocks assigned to each ensemble, but allow their frame counts to
        # vary when a trailing stub made some blocks longer than others.
        # Cutting the concatenated blocks at ``n_reference`` would keep the
        # frame count exact only by splitting whichever block crosses it.
        shuffled = rng.permutation(len(blocks))
        left = np.concatenate([blocks[i] for i in shuffled[:n_reference_blocks]])
        right = np.concatenate([blocks[i] for i in shuffled[n_reference_blocks:]])
    wl = wr = None
    if weighted:
        wl, wr = pooled_w[left], pooled_w[right]
        wl, wr = wl / wl.sum(), wr / wr.sum()
    return statistic(pooled[left], pooled[right], wl, wr)


def _permutation_chunk(
    indices,
    seeds,
    blocks,
    total,
    n_reference,
    n_reference_blocks,
    pooled,
    pooled_w,
    weighted,
    statistic,
):
    """A run of permutations, computed in one worker.

    The unit of work is a chunk rather than a single permutation because the
    pooled representation has to reach the worker, and sending it once per
    permutation costs more than the permutation does.
    """
    return np.stack(
        [
            _one_permutation(
                k,
                seeds,
                blocks,
                total,
                n_reference,
                n_reference_blocks,
                pooled,
                pooled_w,
                weighted,
                statistic,
            )
            for k in indices
        ],
        axis=0,
    )


def permutation_null(
    n_jobs: int,
    statistic,
    reference: np.ndarray,
    other: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    weights_a=None,
    weights_b=None,
    block_length: int = 1,
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

    # Blocks of consecutive frames are the exchangeable unit, not frames. Each
    # ensemble is blocked separately, because they are separate trajectories
    # and a block must not straddle the join between them.
    if block_length > 1:
        labels_a = block_labels(n_reference, block_length)
        labels_b = block_labels(total - n_reference, block_length) + labels_a[-1] + 1
        labels = np.concatenate([labels_a, labels_b])
        blocks = [np.nonzero(labels == b)[0] for b in np.unique(labels)]
        n_reference_blocks = np.unique(labels_a).size
    else:
        blocks = None
        n_reference_blocks = None

    # Each permutation is independent, so the work divides cleanly. Seeds are
    # drawn from the caller's generator up front rather than letting workers
    # draw from a shared one: a parallel run and a serial run then produce the
    # same null, and a result stays reproducible from its seed however many
    # cores it was computed on.
    seeds = rng.integers(0, 2**63 - 1, size=n_permutations)

    def one(k: int) -> np.ndarray:
        return _one_permutation(
            k,
            seeds,
            blocks,
            total,
            n_reference,
            n_reference_blocks,
            pooled,
            pooled_w,
            weighted,
            statistic,
        )

    if n_jobs != 1 and n_permutations > 1:
        import os

        from joblib import Parallel, delayed

        # One task per permutation sends the pooled array to a worker a hundred
        # times and costs more than it saves. Chunking sends it once per
        # worker, which is where the gain is.
        workers = os.cpu_count() or 1 if n_jobs < 0 else n_jobs
        chunks = np.array_split(np.arange(n_permutations), min(workers, n_permutations))
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_permutation_chunk)(
                chunk,
                seeds,
                blocks,
                total,
                n_reference,
                n_reference_blocks,
                pooled,
                pooled_w,
                weighted,
                statistic,
            )
            for chunk in chunks
            if chunk.size
        )
        return np.concatenate(rows, axis=0)

    null = np.empty((n_permutations, reference.shape[1]), dtype=np.float64)
    for k in range(n_permutations):
        null[k] = one(k)
    return null


def studentised_p_values(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
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
