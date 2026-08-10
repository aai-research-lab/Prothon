"""Is this ensemble distinguishable from that one, and by what?

The per-residue metrics answer a question about each feature separately. They
cannot see a difference that lives in the relationship *between* features -- a
mutant whose two loops each visit the same positions as the wild type, but no
longer at the same time, has an identical profile at every residue and is a
different ensemble. Both methods here look at the joint distribution.

**Maximum mean discrepancy** embeds each conformation in a reproducing kernel
Hilbert space and measures the distance between the mean embeddings. With a
characteristic kernel it is zero only when the distributions are equal, so it
sees any difference given enough samples. It gives a calibrated p-value from a
permutation test and nothing else -- no indication of where the difference is.

**The classifier two-sample test** trains a classifier to tell the two
ensembles apart. If it cannot do better than chance, they are indistinguishable
at this sampling; if it can, the accuracy is a bounded and immediately readable
effect size -- "these ensembles are 97% separable" needs no scale to interpret
-- and the classifier can be asked which features it used. That last part is
what the per-residue metrics give for free and MMD cannot give at all.

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

**The classifier is scored out of fold.** A classifier scored on the data it
was fitted to separates any two ensembles perfectly, including two halves of
one ensemble.

    Gretton, A.; Borgwardt, K. M.; Rasch, M. J.; Scholkopf, B.; Smola, A.
    A kernel two-sample test. J. Mach. Learn. Res. 2012, 13, 723-773.

    Lopez-Paz, D.; Oquab, M. Revisiting classifier two-sample tests.
    International Conference on Learning Representations, 2017.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm

from ..utils import get_logger
from .dissimilarity import (
    MINIMUM_EFFECTIVE_SAMPLES,
    effective_sample_size,
)

logger = get_logger("ensemble_metrics")

__all__ = [
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

#: Below this, a p-value from the classifier's asymptotic null is reported as
#: a bound rather than a number. The null is a normal approximation whose far
#: tail is not to be taken literally -- and the folds share training data, so
#: the out-of-fold predictions are not quite independent either. A raw value of
#: 1e-222 is arithmetic, not evidence. The area under the curve is the number
#: to quote.
P_VALUE_FLOOR = 1e-6


@dataclass
class EnsembleComparison:
    """One whole-ensemble comparison.

    Attributes
    ----------
    statistic
        MMD squared, or classifier accuracy. Read ``interpretation`` rather
        than the raw number for the second: accuracy has a floor of 0.5.
    p_value
        From a permutation null for MMD; from the asymptotic null of Lopez-Paz
        and Oquab for the classifier. Read the classifier's as a bound below
        :data:`P_VALUE_FLOOR`: the normal approximation's far tail is not
        literal, and cross-validation folds share training data, so the
        out-of-fold predictions are not quite independent. Quote the effect.
    effect
        A bounded, readable summary. Area under the ROC curve for the
        classifier; ``None`` for MMD, which has no natural scale.
    feature_importance
        How much each feature contributed, where the method can say. ``None``
        for MMD, which cannot.
    """

    method: str
    statistic: float
    p_value: float
    effect: float | None = None
    null_mean: float = 0.0
    null_std: float = 0.0
    n_samples: tuple[int, int] = (0, 0)
    effective_samples: tuple[float, float] = (0.0, 0.0)
    feature_importance: np.ndarray | None = None
    feature_index: np.ndarray | None = None
    measure: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def distinguishable(self) -> bool:
        return bool(self.p_value < 0.05)

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

    def summary(self) -> str:
        verdict = (
            "distinguishable" if self.distinguishable else "not distinguishable"
        )
        shown = (
            f"p < {P_VALUE_FLOOR:g}"
            if self.p_value < P_VALUE_FLOOR
            else f"p = {self.p_value:.3g}"
        )
        line = f"{self.method.upper()}: {verdict} ({shown})"
        if self.effect is not None:
            line += f", AUC = {self.effect:.3f}"
        leading = self.leading_features(5)
        if leading:
            named = ", ".join(f"{i}" for i, _ in leading)
            line += f"\n  driven mostly by residues {named}"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "measure": self.measure,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
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


def _subsample(matrix, weights, size, rng):
    if matrix.shape[0] <= size:
        return matrix, weights
    keep = rng.choice(matrix.shape[0], size, replace=False)
    return matrix[keep], (None if weights is None else weights[keep] / weights[keep].sum())


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
    measure: str = "",
    bandwidth: float | None = None,
    standardise: bool = True,
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

    Notes
    -----
    The squared MMD is a quadratic form ``u' K u`` in a signed weight vector
    ``u`` -- positive on one ensemble, negative on the other. A relabelling is
    therefore a permutation of ``u`` rather than a rebuild of the kernel, and
    the whole null costs one kernel matrix plus one matrix-vector product per
    permutation. Two hundred permutations over a thousand conformations a side
    take well under a second.
    """
    rng = np.random.default_rng(random_state)
    if standardise:
        x, y = _prepare(a, b, circular)
    else:
        x = _encode(np.asarray(a, float), circular)
        y = _encode(np.asarray(b, float), circular)
    wa = None if weights_a is None else np.asarray(weights_a, float)
    wb = None if weights_b is None else np.asarray(weights_b, float)
    n_eff = _check_sampling(x, y, wa, wb)

    x, wa = _subsample(x, wa, sample_size, rng)
    y, wb = _subsample(y, wb, sample_size, rng)
    m, n = x.shape[0], y.shape[0]

    pooled = np.vstack([x, y])
    sigma_squared = (
        float(bandwidth) ** 2 if bandwidth is not None
        else _median_bandwidth(pooled, rng)
    )
    kernel = np.exp(-_squared_distances(pooled) / (2.0 * sigma_squared))

    wa = np.full(m, 1.0 / m) if wa is None else wa / wa.sum()
    wb = np.full(n, 1.0 / n) if wb is None else wb / wb.sum()
    signed = np.concatenate([wa, -wb])

    observed = float(signed @ (kernel @ signed))
    null = np.empty(n_permutations)
    for k in range(n_permutations):
        shuffled = signed[rng.permutation(m + n)]
        null[k] = float(shuffled @ (kernel @ shuffled))

    # The +1 keeps the p-value from ever being zero, which no finite number of
    # permutations can justify.
    p_value = float((1 + np.count_nonzero(null >= observed)) / (n_permutations + 1))

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
        effective_samples=n_eff,
        measure=measure,
        metadata={
            "kernel": "gaussian",
            "bandwidth_squared": sigma_squared,
            "bandwidth_rule": "fixed" if bandwidth is not None else "median heuristic",
            "standardised": standardise,
            "n_permutations": n_permutations,
            "circular_encoding": circular,
        },
    )


# ---------------------------------------------------------------------------
# Classifier two-sample test
# ---------------------------------------------------------------------------
def classifier_two_sample(
    a: np.ndarray,
    b: np.ndarray,
    weights_a=None,
    weights_b=None,
    circular: bool = False,
    folds: int = DEFAULT_FOLDS,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_state=None,
    measure: str = "",
    feature_index=None,
) -> EnsembleComparison:
    """Train a classifier to tell the two ensembles apart, and score it fairly.

    A random forest rather than a linear model, because two ensembles that
    differ in *spread* rather than in mean -- a loop that is rigid in one and
    mobile in the other -- are not linearly separable and are a difference
    anybody would want found.

    The p-value uses the asymptotic null of Lopez-Paz and Oquab: under the
    hypothesis that the ensembles are the same, out-of-fold accuracy is
    normally distributed about one half with variance ``1 / (4 n)``. That
    avoids refitting the classifier for every permutation, which for a forest
    would dominate the runtime of a whole study.

    It is an approximation, and its far tail is where approximations are worst.
    A clearly separable pair returns something like 1e-200, which is
    arithmetic rather than evidence -- and the folds share training data, so
    the predictions are not quite the independent draws the null assumes.
    :meth:`EnsembleComparison.summary` reports anything below
    :data:`P_VALUE_FLOOR` as a bound. The area under the curve is bounded,
    needs no scale to read, and is the number to quote.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(random_state)
    seed = int(rng.integers(0, 2**31 - 1))
    x, y = _prepare(a, b, circular)
    wa = None if weights_a is None else np.asarray(weights_a, float)
    wb = None if weights_b is None else np.asarray(weights_b, float)
    n_eff = _check_sampling(x, y, wa, wb)

    x, wa = _subsample(x, wa, sample_size, rng)
    y, wb = _subsample(y, wb, sample_size, rng)
    m, n = x.shape[0], y.shape[0]

    features = np.vstack([x, y])
    labels = np.concatenate([np.zeros(m, int), np.ones(n, int)])
    sample_weight = np.concatenate([
        np.full(m, 1.0 / m) if wa is None else wa / wa.sum(),
        np.full(n, 1.0 / n) if wb is None else wb / wb.sum(),
    ])

    usable_folds = int(min(folds, m, n))
    if usable_folds < 2:
        raise ValueError(
            f"Cannot cross-validate with {m} and {n} conformations; at least two "
            f"of each are needed to score a classifier out of fold."
        )

    predictions = np.zeros(m + n)
    importance = np.zeros(features.shape[1])
    splitter = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(features, labels):
        forest = RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=1, min_samples_leaf=2
        )
        forest.fit(features[train], labels[train], sample_weight=sample_weight[train])
        predictions[test] = forest.predict_proba(features[test])[:, 1]
        importance += forest.feature_importances_ / usable_folds

    accuracy = float(
        np.sum(sample_weight * ((predictions > 0.5).astype(int) == labels))
        / np.sum(sample_weight)
    )
    auc = float(roc_auc_score(labels, predictions, sample_weight=sample_weight))

    # Sized by the effective count, not the frame count: a weighted ensemble
    # supports a smaller claim than its number of rows suggests.
    total_effective = float(sum(n_eff))
    z = 2.0 * np.sqrt(total_effective) * (accuracy - 0.5)
    p_value = float(norm.sf(z))

    if circular:
        # Undo the (cos, sin) doubling so importances line up with features.
        half = importance.size // 2
        importance = importance[:half] + importance[half:]

    return EnsembleComparison(
        method="c2st",
        statistic=accuracy,
        p_value=p_value,
        effect=auc,
        null_mean=0.5,
        null_std=float(1.0 / (2.0 * np.sqrt(total_effective))),
        n_samples=(m, n),
        effective_samples=n_eff,
        feature_importance=importance,
        feature_index=None if feature_index is None else np.asarray(feature_index),
        measure=measure,
        metadata={
            "classifier": "random forest (200 trees)",
            "folds": usable_folds,
            "circular_encoding": circular,
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
    if key == "mmd" and "feature_index" in kwargs:
        # MMD has no per-feature view; accepting the argument and discarding it
        # would suggest otherwise.
        kwargs.pop("feature_index")
        warnings.warn(
            "MMD reports no per-feature contribution, so feature_index has no "
            "effect. Use c2st to find out which residues carry the difference.",
            UserWarning,
            stacklevel=2,
        )
    return _METHODS[key](a, b, **kwargs)
