"""How far apart two samples of one feature are, and in what units.

Version 2.1 had one answer: the Jensen-Shannon distance between kernel density
estimates. It is a good default -- bounded in [0, 1], so a residue's value
means the same thing on a contact number as on a torsion, and comparable across
proteins. It is not the only sensible answer, and it is not always the best
one.

**Jensen-Shannon needs a density**, so it inherits a grid and a bandwidth. Both
are choices, both bias the estimate, and neither is visible in the number that
comes out. **Wasserstein-1 needs neither.** It is the average distance the
probability mass has to move, computed from the samples directly, and it
reports in the feature's own units: a contact number, radians of torsion,
square nanometres of exposed surface. "This residue gains 1.4 contacts" is a
sentence about the protein. "This residue has a Jensen-Shannon distance of
0.31" is a sentence about the comparison. The cost is that it is unbounded and
not comparable across measures.

**Kolmogorov-Smirnov** is here because PENSA reports it, and a claim that one
method finds something another misses should be checkable on the same
statistic rather than argued from first principles.

Every metric takes per-frame weights and knows whether its feature lives on a
circle -- because getting that wrong is not a small error. Two tight
populations either side of the wraparound are 0.28 radians apart on the circle;
a linear Wasserstein distance reports 4.43, twenty-one times too large, and
reports it without complaint.

    Delon, J.; Salomon, J.; Sobolevski, A. Fast transport optimization for
    Monge costs on the circle. SIAM J. Appl. Math. 2010, 70, 2239-2258.

    Kuiper, N. H. Tests concerning random points on a circle. Proc. K. Ned.
    Akad. Wet. A 1960, 63, 38-47.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

__all__ = [
    "METRICS",
    "Metric",
    "describe_metric",
    "feature_distance",
    "resolve_metric",
]


@dataclass(frozen=True)
class Metric:
    """One way of measuring the distance between two samples of a feature.

    Attributes
    ----------
    bounded
        Whether the value lies in [0, 1]. A bounded metric is comparable
        across measures and proteins; an unbounded one is interpretable in the
        feature's own units and comparable to nothing else.
    units
        What the number is measured in, for axis labels.
    needs_density
        Whether the metric works from an estimated density, and so inherits a
        grid and a bandwidth, rather than from the samples directly.
    """

    name: str
    description: str
    bounded: bool
    units: str
    needs_density: bool
    function: Callable


def _weighted_ecdf(values: np.ndarray, weights: np.ndarray | None, grid: np.ndarray):
    """Right-continuous empirical CDF, evaluated on a grid."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    if weights is None:
        cumulative = np.arange(1, values.size + 1, dtype=np.float64) / values.size
    else:
        cumulative = np.cumsum(weights[order])
        cumulative = cumulative / cumulative[-1]
    position = np.searchsorted(sorted_values, grid, side="right")
    return np.where(position > 0, cumulative[np.clip(position - 1, 0, None)], 0.0)


def _wrap(values: np.ndarray) -> np.ndarray:
    return np.mod(values + np.pi, 2 * np.pi) - np.pi


def _circular_wasserstein(x, y, wx, wy) -> float:
    """Wasserstein-1 on the circle.

    On the line, W1 is the integral of the absolute difference between the two
    cumulative distributions. On a circle there is no natural place to start
    integrating, and the answer depends on where you cut -- which is exactly
    the failure the linear version produces at the wraparound. Delon's result
    is that the circular distance is the same integral minimised over a
    constant offset, and the minimising offset is the median of the CDF
    difference weighted by the interval lengths.
    """
    x, y = _wrap(np.asarray(x, float)), _wrap(np.asarray(y, float))
    knots = np.unique(np.concatenate([x, y, [-np.pi, np.pi]]))
    difference = _weighted_ecdf(x, wx, knots[:-1]) - _weighted_ecdf(y, wy, knots[:-1])
    widths = np.diff(knots)

    # Weighted median of the CDF difference: the offset that minimises the
    # integrated absolute deviation.
    order = np.argsort(difference)
    ordered_difference, ordered_widths = difference[order], widths[order]
    cumulative = np.cumsum(ordered_widths)
    half = cumulative[-1] / 2.0
    offset = ordered_difference[int(np.searchsorted(cumulative, half))]

    return float(np.sum(widths * np.abs(difference - offset)))


def _jsd(x, y, wx, wy, circular, grid, x_num):
    from .dissimilarity import estimate_pdf

    x_min, x_max = grid
    _, p = estimate_pdf(x, x_min, x_max, x_num, circular, wx)
    _, q = estimate_pdf(y, x_min, x_max, x_num, circular, wy)
    value = jensenshannon(p, q, base=2)
    return 0.0 if not np.isfinite(value) else float(value)


def _wasserstein(x, y, wx, wy, circular, grid, x_num):
    if circular:
        return _circular_wasserstein(x, y, wx, wy)
    return float(wasserstein_distance(x, y, wx, wy))


def _supremum(x, y, wx, wy, circular, grid, x_num):
    """Kolmogorov-Smirnov, or Kuiper's statistic on a circle.

    The KS statistic is the largest gap between two cumulative distributions,
    which on a circle depends on where the circle was cut and so is not a
    property of the data. Kuiper's statistic -- the largest gap upward plus the
    largest gap downward -- is invariant to rotation, and reduces to something
    directly comparable to KS on the line.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if circular:
        x, y = _wrap(x), _wrap(y)
    knots = np.unique(np.concatenate([x, y]))
    difference = _weighted_ecdf(x, wx, knots) - _weighted_ecdf(y, wy, knots)
    if circular:
        return float(min(1.0, np.max(difference) - np.min(difference)))
    return float(np.max(np.abs(difference)))


#: Every metric Prothon knows. The single source of truth for the CLI's
#: choices, the config validator and the report.
METRICS: dict[str, Metric] = {
    "jsd": Metric(
        "jsd",
        "Jensen-Shannon distance between estimated densities",
        bounded=True,
        units="",
        needs_density=True,
        function=_jsd,
    ),
    "wasserstein": Metric(
        "wasserstein",
        "Wasserstein-1: the average distance the probability mass moves",
        bounded=False,
        units="feature units",
        needs_density=False,
        function=_wasserstein,
    ),
    "ks": Metric(
        "ks",
        "Kolmogorov-Smirnov statistic (Kuiper's, for circular features)",
        bounded=True,
        units="",
        needs_density=False,
        function=_supremum,
    ),
}


def resolve_metric(metric: str) -> Metric:
    """Look up a metric, suggesting alternatives when it is unknown."""
    key = metric.strip().lower()
    if key in METRICS:
        return METRICS[key]
    import difflib

    close = difflib.get_close_matches(key, METRICS, n=2, cutoff=0.5)
    hint = f" Did you mean {' or '.join(close)}?" if close else ""
    raise ValueError(
        f"Unknown metric {metric!r}. Available: {', '.join(sorted(METRICS))}.{hint}"
    )


def describe_metric(metric: str) -> str:
    spec = resolve_metric(metric)
    scale = "bounded [0, 1]" if spec.bounded else f"unbounded, in {spec.units}"
    return f"{spec.name}: {spec.description} ({scale})"


def feature_distance(
    x,
    y,
    metric: str = "jsd",
    x_min: float = 0.0,
    x_max: float = 1.0,
    x_num: int = 100,
    circular: bool = False,
    weights_x=None,
    weights_y=None,
) -> float:
    """Distance between two samples of one feature, under the chosen metric."""
    spec = resolve_metric(metric)
    return spec.function(
        np.asarray(x, dtype=np.float64).ravel(),
        np.asarray(y, dtype=np.float64).ravel(),
        weights_x,
        weights_y,
        circular,
        (x_min, x_max),
        x_num,
    )
