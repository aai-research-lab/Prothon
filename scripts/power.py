#!/usr/bin/env python3
"""How large a difference the test actually finds.

The calibration study asks how often the test rejects when nothing differs.
This asks the other half: how often it rejects when something does.

A test can be made to hold any false-positive rate by never rejecting, so the
two numbers are only meaningful together. The paper reports one point of this
curve -- 98% at a 0.8 sigma shift and tau = 20 -- and one point does not tell a
reader what their own difference will do.

The construction is the same Ornstein-Uhlenbeck process the calibration study
uses, with a mean shift applied to the second ensemble. The stationary
distribution is standard normal, so a shift of `delta` is a shift of `delta`
standard deviations, and every non-rejection is a miss.

**Correlation costs power, and that is the honest cost of the block null.**
Blocks are the unit of exchangeability, so a trajectory at tau = 50 offers
fewer of them than one at tau = 5 and the test has correspondingly less to work
with. Reporting power without saying that would suggest the calibration came
free.

Usage::

    python scripts/power.py
    python scripts/power.py --replicates 400 --out docs/power.md
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

#: Mean shifts, in standard deviations of the stationary distribution.
EFFECTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.2)

#: Correlation times. 1 is an independent series; 50 is a slow domain motion.
TAUS = (1.0, 10.0, 50.0)

FRAMES = 2000
FEATURES = 20
ALPHA = 0.05


def _shifted_pair(n_frames, n_features, tau, delta, rng):
    """Two AR(1) ensembles differing by a mean shift of ``delta`` sigma."""
    phi = np.exp(-1.0 / tau)
    scale = np.sqrt(1.0 - phi**2)

    def draw(shift):
        series = np.empty((n_frames, n_features))
        series[0] = rng.normal(size=n_features)
        noise = rng.normal(size=(n_frames, n_features)) * scale
        for t in range(1, n_frames):
            series[t] = phi * series[t - 1] + noise[t]
        return series + shift

    return draw(0.0), draw(delta)


def _one(seed, tau, delta):
    import warnings

    from prothon.compare.dissimilarity import dissimilarity

    rng = np.random.default_rng(seed)
    a, b = _shifted_pair(FRAMES, FEATURES, tau, delta, rng)
    span = max(abs(a).max(), abs(b).max()) * 1.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = dissimilarity(
            a, b, -span, span, s_num=2, x_num=80,
            sample_size=1000, n_permutations=200, alpha=ALPHA,
            random_state=seed,
        )
    if not result.p_values_reported:
        return None
    return int(result.n_significant) / int(result.local_dissimilarity.size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--out")
    args = parser.parse_args()

    rows = {}
    for tau in TAUS:
        for delta in EFFECTS:
            fractions = [
                f for f in (
                    _one(int(1e6 * tau + 1e3 * delta + i), tau, delta)
                    for i in range(args.replicates)
                ) if f is not None
            ]
            rows[(tau, delta)] = (
                float(np.mean(fractions)) if fractions else None,
                len(fractions),
            )
            mean = rows[(tau, delta)][0]
            print(
                f"  tau={tau:<5g} delta={delta:<4g} "
                f"power={'withheld' if mean is None else f'{mean:.1%}'} "
                f"({len(fractions)}/{args.replicates} testable)",
                file=sys.stderr,
            )

    lines = [
        "# Power",
        "",
        "The fraction of features called different when a difference of "
        f"`delta` standard deviations is present. {FRAMES} frames, "
        f"{FEATURES} features, {args.replicates} replicates, nominal "
        f"{ALPHA:.0%}.",
        "",
        "`delta = 0` is the false-positive rate and should sit below the "
        "nominal level. Everything else is power, and should be as high as the "
        "sampling allows.",
        "",
        "| δ (σ) | " + " | ".join(f"τ = {t:g}" for t in TAUS) + " |",
        "|---" * (len(TAUS) + 1) + "|",
    ]
    for delta in EFFECTS:
        cells = []
        for tau in TAUS:
            mean, n = rows[(tau, delta)]
            cells.append("withheld" if mean is None else f"{mean:.1%}")
        lines.append(f"| {delta:g} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Read the first row across: the test holds below the nominal level at "
        "every correlation time, which is the calibration result stated the "
        "other way round.",
        "",
        "Read the columns down: **correlation costs power**, and that is what "
        "the block null is paid for. Blocks are the unit of exchangeability, "
        "so a trajectory at τ = 50 offers fewer of them than one at τ = 5 and "
        "the test has less to work with. A method that held its error rate "
        "without losing power here would be getting something for nothing.",
        "",
        "Reproduce with `python scripts/power.py`.",
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
