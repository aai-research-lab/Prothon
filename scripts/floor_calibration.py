#!/usr/bin/env python3
"""Whether the floor verdicts mean what they say.

`scripts/calibration.py` measures the per-residue significance test. This
measures the verdicts built on the noise floor, which are separate machinery
and can fail while the p-values are fine:

* **`resolved`** — the global dissimilarity exceeds the floor. On two ensembles
  drawn from the same distribution it should almost never be true, and on a
  real difference it should almost always be.
* **coverage floors** — `missed()` names the features whose recall falls below
  its floor and `invented()` those whose precision does. On two ensembles from
  the same distribution both should name almost nothing; on a real difference
  the calls should appear.

Both directions are needed. A floor set high enough never to fire on a null
also never fires on a difference, so a null-only measurement cannot tell a
calibrated verdict from a mute one. Every table here reports the null and the
alternative side by side for that reason.

The generator is the Ornstein-Uhlenbeck process used by the calibration study,
so the two are comparable, and the sample size is varied below the frame count
so the default path is exercised.

Usage::

    python scripts/floor_calibration.py
    python scripts/floor_calibration.py --replicates 200 --out docs/floors.md
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np

#: Correlation times, in frames. 1 is effectively independent.
TAUS = (1.0, 10.0, 50.0)

#: Mean shift of the alternative, in standard deviations. Small enough that a
#: mute verdict cannot pass by being obvious.
EFFECT = 0.5

FRAMES = 2000
SAMPLE_SIZE = 1000
FEATURES = 20

#: A floor verdict is a whole-comparison call, so its null rate is a study
#: rate. Ten per cent is generous; a calibrated verdict should sit well below.
MAXIMUM_NULL_RESOLVED = 0.10

#: And it must still fire on a real difference, or the band above is met by a
#: verdict that never says yes. Only where the sampling can support it: an
#: AR(1) series at tau = 50 sampled to 1000 frames carries about ten
#: independent conformations, and no method resolves half a standard deviation
#: from ten. Refusing there is the correct answer, not a failure, so power is
#: required only above this effective sample size and the table reports it.
MINIMUM_POWER = 0.80
POWER_NEEDS_EFFECTIVE_SAMPLES = 50.0


def _effective(sample_size: int, tau: float) -> float:
    """Independent conformations in an AR(1) sample of this size."""
    phi = np.exp(-1.0 / tau)
    return sample_size * (1.0 - phi) / (1.0 + phi)


def _ou(n_frames, n_features, tau, shift, rng):
    phi = np.exp(-1.0 / tau)
    scale = np.sqrt(1.0 - phi**2)
    series = np.empty((n_frames, n_features))
    series[0] = rng.normal(size=n_features)
    noise = rng.normal(size=(n_frames, n_features)) * scale
    for t in range(1, n_frames):
        series[t] = phi * series[t - 1] + noise[t]
    return series + shift


def _compare(seed, tau, shift):
    from prothon.compare.dissimilarity import dissimilarity

    rng = np.random.default_rng(seed)
    a = _ou(FRAMES, FEATURES, tau, 0.0, rng)
    b = _ou(FRAMES, FEATURES, tau, shift, rng)
    span = max(abs(a).max(), abs(b).max()) * 1.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dissimilarity(
            a, b, -span, span, s_num=4, x_num=80,
            sample_size=SAMPLE_SIZE, n_permutations=100,
            random_state=seed,
        )


def _coverage(seed, tau, shift):
    from prothon.compare.coverage import precision_recall

    rng = np.random.default_rng(seed + 7919)
    reference = _ou(FRAMES, FEATURES, tau, 0.0, rng)
    other = _ou(FRAMES, FEATURES, tau, shift, rng)
    span = max(abs(reference).max(), abs(other).max()) * 1.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return precision_recall(
            reference, other, x_min=-span, x_max=span, random_state=seed,
        )


def _rate(values):
    values = [v for v in values if v is not None]
    return (float(np.mean(values)), len(values)) if values else (None, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=60)
    parser.add_argument("--out")
    parser.add_argument("--json")
    args = parser.parse_args()

    rows = []
    for tau in TAUS:
        for label, shift in (("null", 0.0), ("alternative", EFFECT)):
            resolved, missed = [], []
            for replicate in range(args.replicates):
                seed = int(1e5 * tau + 1e3 * shift * 10 + replicate)
                resolved.append(bool(_compare(seed, tau, shift).resolved))
                coverage = _coverage(seed, tau, shift)
                if coverage.floor_assessable:
                    missed.append(
                        len(coverage.missed()) / int(coverage.recall.size)
                    )
            resolved_rate, _ = _rate(resolved)
            missed_rate, assessable = _rate(missed)
            rows.append({
                "tau": tau,
                "case": label,
                "shift": shift,
                "resolved_rate": resolved_rate,
                "missed_rate": missed_rate,
                "assessable": assessable,
                "effective_samples": _effective(SAMPLE_SIZE, tau),
            })
            print(
                f"  tau={tau:<5g} {label:<12} resolved={resolved_rate:.1%} "
                f"missed={'n/a' if missed_rate is None else f'{missed_rate:.1%}'}",
                file=sys.stderr,
            )

    failures = []
    for row in rows:
        if row["case"] == "null" and row["resolved_rate"] > MAXIMUM_NULL_RESOLVED:
            failures.append(
                f"tau={row['tau']:g}: resolved on {row['resolved_rate']:.1%} of "
                f"nulls, above {MAXIMUM_NULL_RESOLVED:.0%}"
            )
        if (
            row["case"] == "alternative"
            and row["effective_samples"] >= POWER_NEEDS_EFFECTIVE_SAMPLES
            and row["resolved_rate"] < MINIMUM_POWER
        ):
            failures.append(
                f"tau={row['tau']:g}: resolved on only "
                f"{row['resolved_rate']:.1%} of real differences, below "
                f"{MINIMUM_POWER:.0%}, with "
                f"{row['effective_samples']:.0f} effective samples"
            )

    lines = [
        "# Floor calibration",
        "",
        f"{FRAMES} frames sampled to {SAMPLE_SIZE}, {FEATURES} features, "
        f"{args.replicates} replicates. The alternative is a mean shift of "
        f"{EFFECT}σ. Both columns are reported for the null and the "
        f"alternative, because a floor high enough never to fire on a null "
        f"also never fires on a difference.",
        "",
        "| τ | effective samples | case | `resolved` | features called missed |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        missed = row["missed_rate"]
        lines.append(
            f"| {row['tau']:.0f} | {row['effective_samples']:.0f} | "
            f"{row['case']} | {row['resolved_rate']:.1%} | "
            f"{'withheld' if missed is None else f'{missed:.1%}'} |"
        )
    lines += [
        "",
        f"**Predeclared:** `resolved` on at most "
        f"{MAXIMUM_NULL_RESOLVED:.0%} of nulls at every correlation time, and "
        f"on at least {MINIMUM_POWER:.0%} of real differences wherever the "
        f"sample carries {POWER_NEEDS_EFFECTIVE_SAMPLES:.0f} or more "
        f"independent conformations. Below that the difference is not "
        f"resolvable and declining to resolve it is the correct answer.",
        "",
        ("**Every gate met.**" if not failures else
         "**Failed:**\n\n" + "\n".join(f"- {f}" for f in failures)),
        "",
        "Reproduce with `python scripts/floor_calibration.py`.",
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
            json.dump({"rows": rows, "failures": failures}, handle, indent=2)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
