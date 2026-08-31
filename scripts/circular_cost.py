#!/usr/bin/env python3
"""What treating a circular feature as linear costs, measured.

A torsion lives on a circle, and -179 degrees is two degrees from +179 rather
than 358. Every estimator here takes a ``circular`` flag; the question this
answers is what happens when it is not set, which is what the published 2023
analysis did for the virtual torsion angle.

Three failures, and they are different failures:

**Wasserstein** transports probability mass the long way round. Two ensembles
either side of the wrap are close on the circle and maximally far on the line.

**Jensen-Shannon** estimates the density on a linear grid, so a population
straddling the wrap arrives as two separated modes. Two ensembles on opposite
sides then appear to share no support at all and the distance saturates at its
maximum -- 1.00 for a pair of ensembles that differ by a few degrees.

**Kolmogorov-Smirnov** takes a supremum over a cumulative distribution, and on
a circle there is no canonical place for the cumulation to begin. The statistic
therefore depends on where the origin happens to sit, which is a property of
the coordinate convention rather than of the data. Kuiper's statistic is
invariant to it, which is why it replaces KS for circular features.

Nothing here needs a trajectory: the ground truth is the angular separation,
fixed by construction.

Usage::

    python scripts/circular_cost.py
    python scripts/circular_cost.py --repeats 200 --out docs/circular.md
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from prothon.compare.distance import feature_distance

#: Separations in radians. The point of the small ones is that a genuinely
#: small difference is where a linear treatment does the most damage.
SEPARATIONS = (0.05, 0.10, 0.20, 0.40)

#: Concentrations of the von Mises the samples are drawn from. Large is tight.
#: The sweep is the measurement, not a parameter of it: how badly a linear
#: treatment fails depends on whether a population is narrow enough to sit on
#: one side of the wrap, and a single concentration hides that entirely.
KAPPAS = (10.0, 30.0, 100.0, 400.0, 1600.0)

GRID = 200


def _pair(separation: float, offset: float, n: int, rng, kappa: float):
    """Two tight populations a fixed angular distance apart.

    ``offset`` places the midpoint, so the same pair can be put safely inside
    the interval or straddling the wrap at +/- pi without changing anything
    else about it.
    """
    a = rng.vonmises(offset - separation / 2, kappa, n)
    b = rng.vonmises(offset + separation / 2, kappa, n)
    return a, b


def study(metric: str, repeats: int, n: int, separation: float, seed: int = 0):
    """The same pair, away from the wrap and across it, at each concentration."""
    rng = np.random.default_rng(seed)
    rows = []
    for kappa in KAPPAS:
        away_c, away_l, across_c, across_l = [], [], [], []
        for _ in range(repeats):
            a, b = _pair(separation, 0.0, n, rng, kappa)
            away_c.append(
                feature_distance(a, b, metric, -np.pi, np.pi, GRID, circular=True)
            )
            away_l.append(
                feature_distance(a, b, metric, -np.pi, np.pi, GRID, circular=False)
            )
            a, b = _pair(separation, np.pi, n, rng, kappa)
            a = np.arctan2(np.sin(a), np.cos(a))
            b = np.arctan2(np.sin(b), np.cos(b))
            across_c.append(
                feature_distance(a, b, metric, -np.pi, np.pi, GRID, circular=True)
            )
            across_l.append(
                feature_distance(a, b, metric, -np.pi, np.pi, GRID, circular=False)
            )
        rows.append({
            "kappa": kappa,
            "spread": float(1.0 / np.sqrt(kappa)),
            "away_circular": float(np.mean(away_c)),
            "away_linear": float(np.mean(away_l)),
            "across_circular": float(np.mean(across_c)),
            "across_linear": float(np.mean(across_l)),
        })
    return rows


def origin_dependence(repeats: int, n: int, seed: int = 0):
    """How much the KS statistic moves when only the origin moves.

    The data is not touched. The interval is rotated, which is a change of
    convention, and a statistic that moves under it is measuring the
    convention. Kuiper's is the circular branch of the same metric here.
    """
    rng = np.random.default_rng(seed)
    linear, circular = [], []
    for _ in range(repeats):
        a, b = _pair(0.20, 0.0, n, rng, 100.0)
        values_l, values_c = [], []
        for rotation in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            ar = np.arctan2(np.sin(a + rotation), np.cos(a + rotation))
            br = np.arctan2(np.sin(b + rotation), np.cos(b + rotation))
            values_l.append(
                feature_distance(ar, br, "ks", -np.pi, np.pi, GRID, circular=False)
            )
            values_c.append(
                feature_distance(ar, br, "ks", -np.pi, np.pi, GRID, circular=True)
            )
        linear.append(max(values_l) - min(values_l))
        circular.append(max(values_c) - min(values_c))
    return float(np.mean(linear)), float(np.mean(circular))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--out")
    args = parser.parse_args()

    lines = [
        "# What a linear treatment of a circular feature costs",
        "",
        "Two tight von Mises populations a fixed angular distance apart, "
        f"{args.n} draws each, {args.repeats} replicates. The same pair is "
        "placed once away from the wrap at ±π and once straddling it. Only "
        "the position changes; the separation does not.",
        "",
    ]

    for metric, title in (
        ("wasserstein", "Wasserstein-1"),
        ("jsd", "Jensen–Shannon"),
    ):
        print(f"{metric}:", file=sys.stderr)
        lines += [
            f"## {title}",
            "",
            "| separation | κ | spread (rad) | away | across, circular | "
            "across, linear | ratio |",
            "|---|---|---|---|---|---|---|",
        ]
        worst = 0.0
        worst_at = None
        for separation in SEPARATIONS:
            for row in study(metric, args.repeats, args.n, separation):
                ratio = row["across_linear"] / max(row["across_circular"], 1e-12)
                if ratio > worst:
                    worst, worst_at = ratio, (separation, row["kappa"])
                lines.append(
                    f"| {separation:.2f} | {row['kappa']:.0f} | "
                    f"{row['spread']:.3f} | {row['away_circular']:.4f} | "
                    f"{row['across_circular']:.4f} | "
                    f"{row['across_linear']:.4f} | {ratio:.1f}× |"
                )
            print(
                f"  separation {separation:.2f}: worst so far {worst:.1f}",
                file=sys.stderr,
            )
        lines += [
            "",
            f"Largest overestimate across the wrap: **{worst:.1f}×**, at a "
            f"separation of {worst_at[0]:.2f} rad and κ = {worst_at[1]:.0f}. "
            "The `away` column is the circular distance away from the wrap, "
            "and the linear treatment reproduces it to four decimals there — "
            "which is why the failure survives inspection. It depends on "
            "where the population sits, and on whether the population is "
            "narrow enough to fall on one side of the wrap rather than "
            "straddling it.",
            "",
        ]

    print("ks origin dependence:", file=sys.stderr)
    linear_range, circular_range = origin_dependence(args.repeats, args.n)
    lines += [
        "## Kolmogorov–Smirnov, and where the origin sits",
        "",
        "The same two populations, rotated together through twelve positions. "
        "Rotating both is a change of coordinate convention and nothing else, "
        "so a statistic that moves under it is reporting the convention.",
        "",
        f"- Linear KS: the statistic moves by **{linear_range:.4f}** between "
        "the best and worst origin.",
        f"- Kuiper's, the circular branch: **{circular_range:.4f}**.",
        "",
        "Kuiper's statistic is invariant to the choice of origin by "
        "construction, which is why it replaces KS for circular features "
        "rather than the circular distance being bolted onto KS.",
        "",
    ]

    document = "\n".join(lines)
    print()
    print(document)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
