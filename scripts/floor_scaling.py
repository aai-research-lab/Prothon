#!/usr/bin/env python3
"""How the noise floor depends on sample size, and why it is not corrected.

The split-half floor measures the resolution limit at half the sampling a study
actually has, so it overestimates. Dividing by the square root of two would fix
that if the distance between two samples of one distribution went as
``n**-0.5``.

This measures whether it does. It does not, for the metric that matters most.

Usage::

    python scripts/floor_scaling.py
    python scripts/floor_scaling.py --repeats 60 --out docs/floor_scaling.md
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from prothon.compare.distance import METRICS, feature_distance

SIZES = (250, 500, 1000, 2000, 4000)


def draw(kind: str, n: int, rng):
    if kind == "gaussian":
        return rng.normal(size=n)
    if kind == "bimodal":
        return np.where(
            rng.random(n) < 0.4, rng.normal(-2, 0.5, n), rng.normal(2, 0.5, n)
        )
    if kind == "skewed":
        return rng.exponential(1.0, n)
    if kind == "uniform":
        return rng.uniform(-2, 2, n)
    raise ValueError(kind)


def exponent(metric: str, kind: str, repeats: int, seed: int = 1) -> float:
    """Slope of log distance against log sample size."""
    rng = np.random.default_rng(seed)
    means = []
    for n in SIZES:
        values = [
            feature_distance(draw(kind, n, rng), draw(kind, n, rng), metric, -6, 6, 80)
            for _ in range(repeats)
        ]
        means.append(np.mean(values))
    return float(np.polyfit(np.log(SIZES), np.log(means), 1)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--out")
    args = parser.parse_args()

    kinds = ("gaussian", "bimodal", "skewed", "uniform")
    lines = [
        "# How the noise floor scales",
        "",
        "Exponent of the distance between two independent samples of one",
        "distribution against sample size. A value of −0.5 would make the",
        "split-half floor exactly √2 too high, and a √2 correction safe.",
        "",
        "| metric | " + " | ".join(kinds) + " | implied correction |",
        "|---" * (len(kinds) + 2) + "|",
    ]
    for metric in sorted(METRICS):
        slopes = [exponent(metric, kind, args.repeats) for kind in kinds]
        print(f"  {metric}: {[f'{s:.3f}' for s in slopes]}", file=sys.stderr)
        factors = [2.0 ** (-s) for s in slopes]
        lines.append(
            f"| `{metric}` | "
            + " | ".join(f"{s:.3f}" for s in slopes)
            + f" | {min(factors):.2f}–{max(factors):.2f} |"
        )

    lines += [
        "",
        "Wasserstein and Kolmogorov–Smirnov sit near −0.5. Jensen–Shannon, the",
        "default, does not: it is estimated from a kernel density whose",
        "bandwidth also depends on the sample size, so its floor carries a",
        "smoothing bias that fades more slowly than the sampling error.",
        "",
        "A single √2 correction would therefore be about right for two metrics",
        "and too large for the third, pushing its floor below the true limit —",
        "which is the failure the floor exists to prevent. The floor is left",
        "conservative and documented as such.",
        "",
    ]
    document = "\n".join(lines)
    print(document)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
