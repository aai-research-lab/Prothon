#!/usr/bin/env python3
"""Measure the false-positive rate, rather than assuming it.

A significance test promises that when two ensembles are drawn from the same
distribution, it will call them different only about as often as the threshold
allows. That is a promise about a rate, and a rate is measured by repetition.

The test suite runs a handful of null replicates -- enough to catch a
catastrophe, not enough to tell 1% from 8%. This runs thousands, across the
parameters a user actually varies, and reports the rate with a confidence
interval so the claim in the documentation can be a measurement.

The third study is the one that matters most. Everything in Prothon's null
assumes frames are exchangeable. Frames from a molecular dynamics trajectory
are correlated in time, so they are not, and the documentation says the
p-values are optimistic without saying by how much. This measures it, using an
Ornstein-Uhlenbeck process where the correlation time is known and the
effective sample size has a closed form.

Usage::

    python scripts/calibration.py --quick              # minutes, for a check
    python scripts/calibration.py --replicates 1000    # the published numbers
    python scripts/calibration.py --study correlation  # just the OU study
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Generating null data
# ---------------------------------------------------------------------------
def independent_null(n_frames, n_features, rng):
    """Two ensembles from the same distribution, frames independent."""
    return rng.normal(size=(n_frames, n_features)), rng.normal(
        size=(n_frames, n_features)
    )


def correlated_features_null(n_frames, n_features, rng, strength=0.7):
    """Two ensembles from the same distribution, features correlated.

    Residues in a real protein do not move independently: a contact number at
    one position is informative about its neighbours. The per-feature tests are
    corrected for multiplicity as though they were separate questions, and
    Benjamini-Hochberg is valid under positive dependence -- but that is a
    theorem about the procedure, and this is a measurement of the whole
    pipeline.
    """
    # An exponentially decaying correlation along the chain, which is roughly
    # what neighbouring residues show.
    lag = np.abs(np.subtract.outer(np.arange(n_features), np.arange(n_features)))
    covariance = strength**lag
    factor = np.linalg.cholesky(covariance + 1e-8 * np.eye(n_features))
    return (
        rng.normal(size=(n_frames, n_features)) @ factor.T,
        rng.normal(size=(n_frames, n_features)) @ factor.T,
    )


def time_correlated_null(n_frames, n_features, rng, tau=10.0):
    """Two ensembles from the same distribution, frames correlated in time.

    An Ornstein-Uhlenbeck process discretised to ``x[t+1] = phi x[t] + noise``
    with ``phi = exp(-1/tau)``, which is the standard model of a coordinate
    relaxing in a harmonic well and the simplest thing that behaves like a
    trajectory. The stationary distribution is standard normal whatever tau is,
    so the two ensembles are identical in distribution and every rejection is a
    false positive -- while the number of independent conformations is

        n_eff / n = (1 - phi) / (1 + phi),

    which falls fast. At tau = 20 frames, ten thousand frames carry about two
    hundred and fifty independent conformations.
    """
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


def theoretical_neff(n_frames, tau):
    """Independent conformations in an AR(1) series of this length."""
    phi = np.exp(-1.0 / tau)
    return n_frames * (1.0 - phi) / (1.0 + phi)


GENERATORS = {
    "independent": independent_null,
    "correlated_features": correlated_features_null,
    "time_correlated": time_correlated_null,
}


# ---------------------------------------------------------------------------
# One replicate
# ---------------------------------------------------------------------------
def one_replicate(job):
    """Run one null comparison and report how much of it was rejected."""
    from prothon.compare.dissimilarity import dissimilarity

    seed, settings = job
    rng = np.random.default_rng(seed)
    generator = GENERATORS[settings["generator"]]

    extra = {}
    if settings["generator"] == "time_correlated":
        extra["tau"] = settings["tau"]
    a, b = generator(settings["frames"], settings["features"], rng, **extra)

    span = max(abs(a).max(), abs(b).max()) * 1.1
    result = dissimilarity(
        a, b, -span, span,
        block_permutation=settings.get("block_permutation"),
        x_num=settings["x_num"],
        s_num=2,
        metric=settings["metric"],
        sample_size=settings["sample_size"],
        n_permutations=settings["permutations"],
        alpha=settings["alpha"],
        random_state=seed,
    )
    return {
        "rejected": int(result.n_significant),
        "features": int(settings["features"]),
        "any": int(result.n_significant > 0),
        "resolved": int(result.resolved),
        "floor": float(result.noise_floor),
    }


def wilson(successes, trials, z=1.96):
    """Wilson interval: behaves at rates near zero, where the normal
    approximation returns negative lower bounds."""
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    spread = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def run(settings, replicates, workers, offset=0):
    jobs = [(offset + i, settings) for i in range(replicates)]
    started = time.perf_counter()
    if workers == 1:
        results = [one_replicate(j) for j in jobs]
    else:
        with mp.Pool(workers) as pool:
            results = pool.map(one_replicate, jobs, chunksize=4)
    rejected = sum(r["rejected"] for r in results)
    total = sum(r["features"] for r in results)
    any_rejected = sum(r["any"] for r in results)
    return {
        **settings,
        "replicates": replicates,
        "feature_rate": rejected / total,
        "feature_ci": wilson(rejected, total),
        "study_rate": any_rejected / replicates,
        "study_ci": wilson(any_rejected, replicates),
        "mean_floor": float(np.mean([r["floor"] for r in results])),
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------
BASE = {
    "generator": "independent",
    "frames": 400,
    "features": 8,
    "metric": "jsd",
    "x_num": 60,
    "sample_size": 400,
    "permutations": 100,
    "alpha": 0.05,
    "tau": 1.0,
    "block_permutation": None,
}


def study_parameters(replicates, workers, quick):
    """Does the rate hold across the settings a user varies?"""
    metrics = ["jsd"] if quick else ["jsd", "wasserstein", "ks"]
    alphas = [0.05] if quick else [0.01, 0.05, 0.10]
    permutations = [100] if quick else [100, 200]
    rows = []
    for metric, alpha, n_perm in itertools.product(metrics, alphas, permutations):
        settings = {**BASE, "metric": metric, "alpha": alpha, "permutations": n_perm}
        print(f"  {metric:<12} alpha={alpha:<5} perms={n_perm}", file=sys.stderr)
        rows.append(run(settings, replicates, workers))
    return rows


def study_correlated_features(replicates, workers, quick):
    """Does it hold when residues move together, as they do in a protein?"""
    strengths = [0.7] if quick else [0.0, 0.5, 0.9]
    rows = []
    for strength in strengths:
        generator = "independent" if strength == 0.0 else "correlated_features"
        settings = {**BASE, "generator": generator, "features": 20}
        print(f"  feature correlation {strength}", file=sys.stderr)
        row = run(settings, replicates, workers)
        row["strength"] = strength
        rows.append(row)
    return rows


def study_time_correlation(replicates, workers, quick):
    """How optimistic are the p-values on a trajectory?

    This is the assumption the documentation declares and does not quantify.
    """
    taus = [1.0, 20.0] if quick else [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    rows = []
    for tau in taus:
        # Both nulls on the same data, so the comparison is like for like.
        for blocked in (False, None):
            settings = {
                **BASE, "generator": "time_correlated", "tau": tau,
                "frames": 2000, "sample_size": 2000,
                "block_permutation": blocked,
            }
            label = "block" if blocked is None else "frame"
            print(f"  correlation time {tau} frames, {label} permutation",
                  file=sys.stderr)
            row = run(settings, replicates, workers)
            row["tau"] = tau
            row["blocked"] = blocked is None
            row["theoretical_neff"] = theoretical_neff(2000, tau)
            rows.append(row)
    return rows


STUDIES = {
    "parameters": study_parameters,
    "features": study_correlated_features,
    "correlation": study_time_correlation,
}


def render(name, rows) -> str:
    lines = []
    if name == "parameters":
        lines.append("| metric | alpha | permutations | features called | 95% CI | studies with ≥1 |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            lo, hi = r["feature_ci"]
            lines.append(
                f"| `{r['metric']}` | {r['alpha']} | {r['permutations']} | "
                f"{r['feature_rate']:.3%} | {lo:.2%}–{hi:.2%} | {r['study_rate']:.1%} |"
            )
    elif name == "features":
        lines.append("| correlation between features | features called | 95% CI |")
        lines.append("|---|---|---|")
        for r in rows:
            lo, hi = r["feature_ci"]
            lines.append(
                f"| {r['strength']} | {r['feature_rate']:.3%} | {lo:.2%}–{hi:.2%} |"
            )
    else:
        lines.append(
            "| correlation time (frames) | independent conformations in 2000 "
            "| null | features called | 95% CI |"
        )
        lines.append("|---|---|---|---|---|")
        for r in rows:
            lo, hi = r["feature_ci"]
            null = "block" if r.get("blocked") else "frame"
            lines.append(
                f"| {r['tau']:.0f} | {r['theoretical_neff']:.0f} | {null} | "
                f"{r['feature_rate']:.2%} | {lo:.2%}–{hi:.2%} |"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--study", choices=[*STUDIES, "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="a reduced grid")
    parser.add_argument("--out", help="write markdown here")
    parser.add_argument("--json", help="write raw results here")
    args = parser.parse_args()

    chosen = list(STUDIES) if args.study == "all" else [args.study]
    print(
        f"{args.replicates} replicates per setting on {args.workers} worker(s)",
        file=sys.stderr,
    )

    sections, raw = [], {}
    for name in chosen:
        print(f"\n{name}:", file=sys.stderr)
        rows = STUDIES[name](args.replicates, args.workers, args.quick)
        raw[name] = rows
        sections.append(f"## {name.replace('_', ' ').title()}\n\n{render(name, rows)}")

    document = (
        "# Calibration\n\n"
        f"Measured over {args.replicates} null replicates per setting. In every "
        "case the two ensembles are drawn from the same distribution, so every "
        "rejection is a false positive.\n\n" + "\n\n".join(sections) + "\n"
    )
    print(document)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(document)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(raw, handle, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
