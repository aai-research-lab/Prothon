"""What a sample is worth, before any distance is computed.

Three quantities that decide how much of a comparison the data supports:

* **Effective sample size.** A thousand conformations in which one carries half
  the probability are worth four. Kish (1965).
* **Weighted resampling.** Drawing from an ensemble in proportion to its
  weights, which is what the noise floor and the null both need.
* **Multiplicity.** A protein furnishes as many simultaneous tests as it has
  residues, and at a five per cent threshold a three-hundred-residue protein
  yields fifteen false positives by construction. Benjamini and Hochberg (1995).

None of these knows what a distance is, which is why they are here and not in
:mod:`prothon.compare`.
"""

from __future__ import annotations

import numpy as np

from ..utils import get_logger

__all__ = [
    "DEFAULT_SAMPLE_SIZE",
    "benjamini_hochberg",
    "effective_sample_size",
    "random_sample",
]

logger = get_logger("sampling")

#: Frames drawn from each ensemble for the test by default. Larger ensembles
#: are subsampled to this, without replacement, because the null needs many
#: repeats and the density estimate is the expensive part of each one.
DEFAULT_SAMPLE_SIZE = 1000


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
