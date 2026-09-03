#!/usr/bin/env python3
"""Reproduce the audited MMD correlation failure and its corrected null.

The fixture sizes and acceptance bands are fixed in advance. Output is JSON so
every seed and p-value can travel with a release record instead of leaving only
a selected summary in prose.

Usage::

    python scripts/joint_calibration.py
    python scripts/joint_calibration.py --out mmd-calibration.json
"""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from pathlib import Path

import mdtraj
import numpy as np
import scipy

from prothon import __version__ as prothon_version
from prothon.compare.joint import maximum_mean_discrepancy


def ou(n_frames, n_features, relaxation, rng, mean=0.0):
    """A stationary AR(1) trajectory with a known relaxation time."""
    phi = np.exp(-1.0 / relaxation)
    values = np.empty((n_frames, n_features))
    values[0] = rng.normal(size=n_features)
    noise = rng.normal(size=(n_frames, n_features)) * np.sqrt(1.0 - phi**2)
    for frame in range(1, n_frames):
        values[frame] = phi * values[frame - 1] + noise[frame]
    return values + mean


def wilson(successes, trials, z=1.96):
    """A 95% Wilson binomial interval, including at zero and one."""
    rate = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = (rate + z**2 / (2.0 * trials)) / denominator
    spread = (
        z
        * np.sqrt(rate * (1.0 - rate) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return [max(0.0, centre - spread), min(1.0, centre + spread)]


def record(permutation_seed, result, data_seeds):
    """The complete inferential record without the large frame-range audit."""
    return {
        "permutation_seed": permutation_seed,
        "data_seeds": data_seeds,
        "mmd_squared": result.statistic,
        "p_value": result.p_value,
        "null_mean": result.null_mean,
        "null_std": result.null_std,
        "sampling_units": result.metadata["permutation_units"],
        "effective_sampling_units": result.metadata["effective_permutation_units"],
        "distinguishable": result.distinguishable,
    }


def run_calibration():
    """Run every predeclared MMD gate and return its raw record."""
    blocked_settings = {
        "n_permutations": 99,
        "sample_size": 500,
        "sampling_kind_a": "trajectory",
        "sampling_kind_b": "trajectory",
        "correlation_time_frames_a": 20.0,
        "correlation_time_frames_b": 20.0,
    }
    blocked_null = []
    row_null = []
    for seed in range(20):
        a = ou(500, 4, 10.0, np.random.default_rng(100 + seed))
        b = ou(500, 4, 10.0, np.random.default_rng(200 + seed))
        result = maximum_mean_discrepancy(
            a, b, random_state=seed, **blocked_settings
        )
        data_seeds = {"a": 100 + seed, "b": 200 + seed}
        blocked_null.append(record(seed, result, data_seeds))
        if seed < 8:
            # This deliberately false IID declaration preserves the audited
            # row-permutation failure beside the corrected calculation.
            result = maximum_mean_discrepancy(
                a,
                b,
                n_permutations=99,
                sample_size=500,
                sampling_kind_a="iid",
                sampling_kind_b="iid",
                random_state=seed,
            )
            row_null.append(record(seed, result, data_seeds))

    shifted_power = []
    for seed in range(10):
        a = ou(500, 4, 10.0, np.random.default_rng(300 + seed))
        b = ou(500, 4, 10.0, np.random.default_rng(400 + seed), mean=0.5)
        result = maximum_mean_discrepancy(
            a, b, random_state=seed, **blocked_settings
        )
        shifted_power.append(
            record(seed, result, {"a": 300 + seed, "b": 400 + seed})
        )

    fixed_designs = []

    a = ou(500, 4, 10.0, np.random.default_rng(1))
    b = ou(700, 4, 10.0, np.random.default_rng(2))
    result = maximum_mean_discrepancy(
        a,
        b,
        n_permutations=99,
        sample_size=700,
        correlation_time_frames_a=20.0,
        correlation_time_frames_b=20.0,
        random_state=0,
    )
    fixed_designs.append(
        {"design": "unequal lengths", **record(0, result, {"a": 1, "b": 2})}
    )

    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(320, 4)), rng.normal(size=(470, 4))
    result = maximum_mean_discrepancy(
        a,
        b,
        weights_a=np.linspace(0.2, 2.0, 320),
        weights_b=np.linspace(2.0, 0.2, 470) ** 2,
        n_permutations=99,
        sample_size=500,
        sampling_kind_a="iid",
        sampling_kind_b="iid",
        random_state=0,
    )
    fixed_designs.append(
        {"design": "unequal weights", **record(0, result, {"pooled": 3})}
    )

    rng = np.random.default_rng(4)
    a, b = ou(500, 4, 10.0, rng), rng.normal(size=(650, 4))
    result = maximum_mean_discrepancy(
        a,
        b,
        n_permutations=99,
        sample_size=700,
        sampling_kind_a="trajectory",
        sampling_kind_b="iid",
        correlation_time_frames_a=20.0,
        random_state=0,
    )
    fixed_designs.append(
        {"design": "trajectory versus IID", **record(0, result, {"pooled": 4})}
    )

    labels = np.repeat(np.arange(8), 50)

    def replicas(seed):
        rng = np.random.default_rng(seed)
        return np.vstack([ou(50, 4, 10.0, rng) for _ in range(8)])

    result = maximum_mean_discrepancy(
        replicas(33),
        replicas(34),
        replica_labels_a=labels,
        replica_labels_b=labels,
        n_permutations=99,
        sample_size=500,
        random_state=0,
    )
    fixed_designs.append(
        {
            "design": "independent replicas",
            **record(0, result, {"a": 33, "b": 34}),
        }
    )

    rng = np.random.default_rng(7)
    a = np.pi + 0.5 * ou(500, 4, 10.0, rng)
    b = np.pi + 0.5 * ou(500, 4, 10.0, rng)
    a = (a + np.pi) % (2.0 * np.pi) - np.pi
    b = (b + np.pi) % (2.0 * np.pi) - np.pi
    result = maximum_mean_discrepancy(
        a,
        b,
        circular=True,
        n_permutations=99,
        sample_size=500,
        correlation_time_frames_a=20.0,
        correlation_time_frames_b=20.0,
        random_state=0,
    )
    fixed_designs.append(
        {"design": "circular trajectory", **record(0, result, {"pooled": 7})}
    )

    short_a = ou(120, 3, 10.0, np.random.default_rng(31))
    short_b = ou(120, 3, 10.0, np.random.default_rng(32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        short = maximum_mean_discrepancy(
            short_a,
            short_b,
            correlation_time_frames_a=20.0,
            correlation_time_frames_b=20.0,
            n_permutations=20,
            sample_size=120,
            random_state=0,
        )

    blocked_calls = sum(bool(row["distinguishable"]) for row in blocked_null)
    row_calls = sum(bool(row["distinguishable"]) for row in row_null)
    power_calls = sum(bool(row["distinguishable"]) for row in shifted_power)
    fixed_pass = all(
        row["p_value"] is not None and row["p_value"] >= 0.05
        for row in fixed_designs
    )
    gates = {
        "blocked_null": {
            "observed": blocked_calls,
            "trials": 20,
            "acceptable_calls": [0, 3],
            "wilson_95": wilson(blocked_calls, 20),
            "passed": blocked_calls <= 3,
        },
        "row_null_failure_reproduced": {
            "observed": row_calls,
            "trials": 8,
            "wilson_95": wilson(row_calls, 8),
            "passed": row_calls == 8,
        },
        "shifted_power": {
            "observed": power_calls,
            "trials": 10,
            "minimum_calls": 8,
            "wilson_95": wilson(power_calls, 10),
            "passed": power_calls >= 8,
        },
        "fixed_null_designs": {
            "required": "all p-values supported and >= 0.05",
            "passed": fixed_pass,
        },
        "short_trajectory": {
            "required": "MMD squared retained; p-value withheld",
            "mmd_squared": short.statistic,
            "p_value": short.p_value,
            "sampling_units": short.metadata["permutation_units"],
            "permutation_seed": 0,
            "data_seeds": {"a": 31, "b": 32},
            "passed": short.statistic > 0.0 and short.p_value is None,
        },
    }
    return {
        "schema_version": 1,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "prothon": prothon_version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mdtraj": getattr(mdtraj, "__version__", "unknown"),
        },
        "predeclared": {
            "alpha": 0.05,
            "blocked_null_calls": "0-3 of 20",
            "shifted_power_calls": "at least 8 of 10",
        },
        "settings": blocked_settings,
        "fixtures": {
            "ou": {
                "process": "AR(1)",
                "frames": 500,
                "features": 4,
                "relaxation_frames": 10.0,
                "supplied_integrated_time_frames": 20.0,
            },
            "blocked_null_data_seeds": "a=100+seed, b=200+seed; seed=0..19",
            "misdeclared_iid_subset": "blocked-null seeds 0..7",
            "shifted_power": {
                "mean_shift": 0.5,
                "data_seeds": "a=300+seed, b=400+seed; seed=0..9",
            },
            "fixed_designs": {
                "unequal_lengths": "500 versus 700 frames; data seeds 1, 2",
                "unequal_weights": "320 versus 470 IID rows; pooled data seed 3",
                "trajectory_versus_iid": "500 versus 650 rows; pooled data seed 4",
                "replicas": "8 replicas x 50 frames; data seeds 33, 34",
                "circular": "500 frames around pi branch cut; pooled data seed 7",
                "underpowered": "120 frames, 3 blocks per side; data seeds 31, 32",
            },
        },
        "gates": gates,
        "runs": {
            "blocked_null": blocked_null,
            "misdeclared_iid_row_null": row_null,
            "shifted_power": shifted_power,
            "fixed_null_designs": fixed_designs,
        },
        "passed": all(gate["passed"] for gate in gates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="also write the JSON record here")
    args = parser.parse_args()
    payload = run_calibration()
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    print(rendered)
    if args.out is not None:
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
