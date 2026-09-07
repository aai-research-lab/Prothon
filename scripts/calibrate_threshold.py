#!/usr/bin/env python3
"""What threshold delivers the error rate the caller asked for.

A test that promises five per cent and delivers eight and a half is not
delivering. Six mechanisms have been examined and swept and none accounts for
the excess: block length plateaus, permutation count plateaus at two hundred,
and the rest were refuted outright. See section 0b of the roadmap.

This does the other thing an instrument maker does. Rather than deriving the
error from first principles, measure it against a standard and correct for it.

**The standard is exact.** Two ensembles drawn from one distribution differ in
no way, so every call is wrong by construction and the correct rate is whatever
the caller asked for. Nothing here is estimated from data whose answer is
unknown.

**What is measured.** For each configuration, run many nulls and record the
smallest corrected p-value in each. The distribution of that minimum *is* the
family-wise null distribution. Its `alpha` quantile is the threshold that
delivers `alpha`: reject below it and, by construction, a fraction `alpha` of
null studies produce a call.

**Why a table and not a constant.** The excess depends on the correlation time,
so a slow trajectory needs a different correction from a fast one. Instruments
ship with a curve.

**Why this is not fudging the numbers.** A fudge would move the threshold until
the answer looked good and stop. This measures on one set of nulls and verifies
on a second, independent set, and reports the verification. A correction that
does not hold on fresh data has not worked, and the script says so.

Usage::

    python scripts/calibrate_threshold.py --replicates 500
    python scripts/calibrate_threshold.py --replicates 2000 --out docs/thresholds.md
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np

#: Correlation times to calibrate at. The correction varies with this, which is
#: why it is a table.
TAUS = (1.0, 5.0, 10.0, 25.0, 50.0)

#: The rate the caller asks for and must receive.
ALPHA = 0.05

FRAMES = 4000
SAMPLE_SIZE = 2000
FEATURES = 8
PERMUTATIONS = 200


def _null_pair(n_frames, n_features, tau, rng):
    """Two ensembles from one distribution. Every call on these is wrong."""
    phi = np.exp(-1.0 / tau)
    scale = np.sqrt(1.0 - phi**2)

    def draw():
        series = np.empty((n_frames, n_features))
        series[0] = rng.normal(size=n_features)
        noise = rng.normal(size=(n_frames, n_features)) * scale
        for t in range(1, n_frames):
            series[t] = phi * series[t - 1] + noise[t]
        return series

    return draw(), draw()


def _smallest_p(seed, tau):
    """The smallest corrected p-value this null study produced, or None."""
    from prothon.compare.dissimilarity import dissimilarity

    rng = np.random.default_rng(seed)
    a, b = _null_pair(FRAMES, FEATURES, tau, rng)
    span = max(abs(a).max(), abs(b).max()) * 1.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = dissimilarity(
            a, b, -span, span, s_num=2, x_num=60,
            sample_size=SAMPLE_SIZE, n_permutations=PERMUTATIONS,
            alpha=ALPHA, random_state=seed,
        )
    if not result.p_values_reported:
        return None
    return float(np.min(result.p_values))


def _minima(tau, seeds, workers=None):
    """Smallest p-value per null study, in parallel.

    Serial, a single correlation time at five hundred replicates outran a
    twenty-minute budget. The work is embarrassingly parallel: each study is
    independent and seeded, so the result does not depend on the worker count.
    """
    import multiprocessing as mp

    seeds = list(seeds)
    workers = workers or max(1, mp.cpu_count() - 1)
    if workers == 1:
        values = [_smallest_p(seed, tau) for seed in seeds]
    else:
        with mp.Pool(workers) as pool:
            values = pool.starmap(
                _smallest_p, [(seed, tau) for seed in seeds], chunksize=4
            )
    return np.array([v for v in values if v is not None]), len(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--out")
    parser.add_argument("--json")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="processes to use. Results are seeded, so this changes only speed.",
    )
    args = parser.parse_args()

    rows = []
    for tau in TAUS:
        # Two disjoint seed ranges: one to measure the threshold, one to check
        # it. Calibrating and verifying on the same numbers proves nothing.
        fit_seeds = range(0, args.replicates)
        check_seeds = range(1_000_000, 1_000_000 + args.replicates)

        fit, fit_total = _minima(tau, fit_seeds, args.workers)
        if fit.size < 50:
            print(f"  tau={tau:<5g} too few testable studies", file=sys.stderr)
            rows.append({"tau": tau, "testable": fit.size / fit_total})
            continue

        # The alpha quantile of the smallest p-value under the null is, by
        # construction, the threshold at which alpha of null studies call.
        threshold = float(np.quantile(fit, args.alpha))
        uncorrected = float(np.mean(fit < args.alpha))

        check, check_total = _minima(tau, check_seeds, args.workers)
        achieved = float(np.mean(check < threshold)) if check.size else None

        rows.append({
            "tau": tau,
            "threshold": threshold,
            "uncorrected_rate": uncorrected,
            "achieved_on_fresh_nulls": achieved,
            "testable": fit.size / fit_total,
            "n_fit": int(fit.size),
            "n_check": int(check.size),
        })
        print(
            f"  tau={tau:<5g} threshold={threshold:.4f}  "
            f"uncorrected={uncorrected:.1%}  fresh={achieved:.1%}  "
            f"testable={fit.size / fit_total:.0%}",
            file=sys.stderr,
        )

    lines = [
        "# Calibrated thresholds",
        "",
        f"Measured on null pairs at {args.replicates} replicates per "
        f"correlation time. Both ensembles come from one distribution, so every "
        f"call is wrong by construction and the correct rate is exactly "
        f"{args.alpha:.0%}.",
        "",
        "| τ | threshold for "
        f"{args.alpha:.0%}"
        " | rate at the nominal threshold | rate on fresh nulls | testable |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        if "threshold" not in row:
            lines.append(
                f"| {row['tau']:.0f} | — | — | — | {row['testable']:.0%} |"
            )
            continue
        lines.append(
            f"| {row['tau']:.0f} | {row['threshold']:.4f} | "
            f"{row['uncorrected_rate']:.1%} | "
            f"{row['achieved_on_fresh_nulls']:.1%} | {row['testable']:.0%} |"
        )
    lines += [
        "",
        "**The third column is the promise and the fourth is whether it is "
        "kept.** The threshold is measured on one set of nulls and applied to a "
        f"second, disjoint set. If the fourth column reads near {args.alpha:.0%} "
        "the correction holds on data it has never seen, which is the only "
        "evidence worth having. If it reads near the second column instead, the "
        "correction has not worked and nothing here should be shipped.",
        "",
        "Reproduce with `python scripts/calibrate_threshold.py`.",
        "",
    ]
    document = "\n".join(lines)
    print()
    print(document)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(document)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"alpha": args.alpha, "rows": rows}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
