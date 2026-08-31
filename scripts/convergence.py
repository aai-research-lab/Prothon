#!/usr/bin/env python3
"""How long a trajectory must run before a difference of a given size shows.

The rest of the software compares two systems. This turns the same machinery on
one trajectory and asks whether it is finished, which is a different question
and the one a user has before they have two ensembles to compare.

Three measurements, from prefixes of a single trajectory.

**The floor against length.** The split-half floor is the smallest
dissimilarity a given amount of sampling can resolve. Measured on prefixes of
250 to 5000 conformations it becomes a curve, and any real dissimilarity drawn
across that curve meets it at the length where that comparison becomes
possible. That is the practical output of this script.

**The correlation time against length.** The estimator saturates on a short
series, and this reports whether the guard notices. It is the measurement that
retired the guard that came before: a ratio test on `n / tau_hat` puts the
saturated value in its own denominator, so severe saturation produces a
healthy-looking ratio. On this trajectory a 250-frame prefix returned 5 where
the whole returns 45, and the ratio read 50. The plateau test asks instead
whether the estimate moved between half the frames and all of them.

**Precision and recall, where the answer is known in advance.** A prefix has
visited a subset of what the whole trajectory visited. It has therefore missed
states and invented none, so recall should fall with length while precision
should stay near its floor. Any other result is a finding about the
implementation rather than about the trajectory, which is what makes this a
test rather than an illustration.

Usage::

    python scripts/convergence.py --data ~/ubiquitin
    python scripts/convergence.py --data ~/ubiquitin --reference Q99 \\
        --compare Q95 --out docs/convergence.md --json docs/convergence.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

#: Prefix lengths, in conformations. Each is a contiguous prefix rather than a
#: random subsample: a user with a short trajectory has the first N frames, not
#: N frames scattered through a long one, and the correlation structure of the
#: two is not the same.
LENGTHS = (250, 500, 1000, 2000, 5000)


def load(directory: str, names: list[str]):
    from prothon.ingest import Ensemble

    topology = os.path.join(directory, "topology.pdb")
    if not os.path.exists(topology):
        raise SystemExit(f"No topology.pdb in {directory}")

    loaded = []
    for name in names:
        path = os.path.join(directory, f"{name}.dcd")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}")
        ensemble = Ensemble.from_trajectory(path, topology, label=name)
        loaded.append(ensemble)
        print(f"  {name}: {ensemble.n_frames} frames", file=sys.stderr)
    return loaded


def represent(ensemble, order_parameter: str):
    from prothon.represent.order_parameters import compute_representation

    return compute_representation(ensemble.trajectory, order_parameter)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.expanduser("~/ubiquitin"))
    parser.add_argument("--reference", default="Q99")
    parser.add_argument(
        "--compare", default="Q95",
        help="Drawn across the floor curve as the difference to be resolved. "
             "Q95 is the hardest pair in the series and therefore the one "
             "whose crossing point is worth knowing.",
    )
    parser.add_argument("--order-parameter", default="cbcn")
    parser.add_argument("--s-num", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--json")
    args = parser.parse_args()

    from prothon.compare.coverage import precision_recall
    from prothon.compare.dissimilarity import dissimilarity
    from prothon.sampling.correlation import (
        correlation_time_estimate,
        effective_frames,
    )

    print("Loading:", file=sys.stderr)
    reference, comparison = load(args.data, [args.reference, args.compare])

    from prothon.represent.order_parameters import resolve_order_parameter

    spec = resolve_order_parameter(args.order_parameter)
    full_ref = represent(reference, args.order_parameter)
    full_cmp = represent(comparison, args.order_parameter)
    n_features = int(full_ref.shape[1])

    # One grid for every comparison below, spanning everything that will be
    # compared on it. Prefixes are subsets of the reference, so the full
    # reference already bounds them; taking the bounds per prefix would put
    # each length on a different grid and make the floor curve meaningless.
    grid_min = float(min(full_ref.min(), full_cmp.min()))
    grid_max = float(max(full_ref.max(), full_cmp.max()))
    print(
        f"  {n_features} features under {args.order_parameter}, "
        f"grid [{grid_min:.3f}, {grid_max:.3f}]",
        file=sys.stderr,
    )

    rows = []
    for length in LENGTHS:
        if length > full_ref.shape[0]:
            print(f"  skipping {length}: trajectory is shorter", file=sys.stderr)
            continue
        print(f"\n{length} conformations:", file=sys.stderr)
        prefix = full_ref[:length]

        # The floor: the reference against itself at this length.
        result = dissimilarity(
            prefix, prefix, grid_min, grid_max,
            s_num=args.s_num, n_permutations=0, circular=spec.circular,
            sample_size=length, random_state=args.seed, n_jobs=args.n_jobs,
            order_parameter=args.order_parameter,
        )
        floor = float(result.noise_floor)

        # The correlation time at this length, and whether it has settled.
        # Not a ratio test: see the module docstring for why that one was
        # wrong. A flagged estimate is a lower bound, so `independent` below
        # is an upper bound.
        estimate = correlation_time_estimate(prefix)
        tau = float(estimate.tau)
        saturated = not estimate.converged
        independent = float(effective_frames(length, tau))

        # Precision and recall of the prefix against the whole trajectory.
        # A prefix has visited a subset, so recall should fall and precision
        # should not.
        coverage = precision_recall(
            full_ref, prefix, circular=spec.circular,
            x_min=grid_min, x_max=grid_max,
            random_state=args.seed, order_parameter=args.order_parameter,
        )

        rows.append({
            "length": length,
            "floor": floor,
            "tau": tau,
            "tau_is_lower_bound": bool(saturated),
            "tau_growth": float(estimate.growth),
            "tau_slope": float(estimate.slope),
            "tau_prefixes": {int(k): float(v) for k, v in estimate.prefix_taus.items()},
            "independent": independent,
            "precision": float(coverage.mean_precision),
            "recall": float(coverage.mean_recall),
            "floor_precision": float(coverage.mean_floor_precision),
            "floor_recall": float(coverage.mean_floor_recall),
        })
        print(
            f"  floor {floor:.4f}  tau {tau:.0f}"
            f"{' (lower bound)' if saturated else ''}"
            f"  precision {coverage.mean_precision:.3f}"
            f"  recall {coverage.mean_recall:.3f}",
            file=sys.stderr,
        )

    # The dissimilarity to be resolved, measured once on everything available.
    target = dissimilarity(
        full_ref, full_cmp, grid_min, grid_max,
        s_num=args.s_num, n_permutations=0, circular=spec.circular,
        sample_size=int(max(full_ref.shape[0], full_cmp.shape[0])),
        random_state=args.seed, n_jobs=args.n_jobs,
        order_parameter=args.order_parameter,
    )
    target_d = float(target.global_dissimilarity)

    resolvable = [r["length"] for r in rows if r["floor"] < target_d]
    crossing = min(resolvable) if resolvable else None

    lines = [
        f"# How long is long enough? {args.reference}, "
        f"`{args.order_parameter}`",
        "",
        f"Contiguous prefixes of the {args.reference} trajectory, "
        f"{n_features} features. The floor is the smallest dissimilarity that "
        f"much sampling can resolve; precision and recall are the prefix "
        f"measured against the whole trajectory.",
        "",
        "| conformations | floor | τ | slope | independent | precision | recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['length']} | {row['floor']:.4f} | "
            f"{row['tau']:.0f}{'+' if row['tau_is_lower_bound'] else ''} | "
            + (
                "— | "
                if not np.isfinite(row["tau_slope"])
                else f"{row['tau_slope']:.2f} | "
            )
            + f"{row['independent']:.0f} | {row['precision']:.3f} | "
            + f"{row['recall']:.3f} |"
        )

    flagged = sum(1 for r in rows if r["tau_is_lower_bound"])
    lines += [
        "",
        "`slope` is the slope of log τ against log n across four nested "
        "prefixes. Zero means the answer does not depend on how much data it "
        "was given, which is what a settled estimate looks like; one means the "
        "estimate is reporting the trajectory length rather than the "
        "correlation. `—` means the prefix was too short to fit three "
        "sub-prefixes, so no trend could be fitted and nothing is claimed. A "
        "`τ` marked `+` is a **lower bound**, and the `independent` column "
        "beside it is correspondingly an upper bound. "
        + (
            f"{flagged} of {len(rows)} lengths are flagged."
            if flagged
            else "No length is flagged."
        ),
        "",
        "## Where the crossing is",
        "",
        f"The {args.reference} against {args.compare} dissimilarity is "
        f"**{target_d:.4f}**.",
        "",
    ]
    if crossing is None:
        lines.append(
            f"No prefix resolves it. Every floor measured here exceeds "
            f"{target_d:.4f}, so this comparison is beyond the reach of "
            f"{max(r['length'] for r in rows)} conformations of this "
            f"trajectory."
        )
    elif crossing == min(r["length"] for r in rows):
        lines.append(
            f"Every prefix tested resolves it, including the shortest at "
            f"{crossing} conformations, where the floor is "
            f"{next(r['floor'] for r in rows if r['length'] == crossing):.4f}. "
            f"The crossing is therefore at or below the shortest length "
            f"measured, and this series does not locate it."
        )
    else:
        lines.append(
            f"It first exceeds the floor at **{crossing} conformations**, "
            f"where the floor is {next(r['floor'] for r in rows if r['length'] == crossing):.4f}. "
            f"At the length below that the floor is larger than the "
            f"dissimilarity, so the two ensembles are not distinguishable at "
            f"all and the value returned would be sampling rather than "
            f"structure."
        )

    # Precision and recall have a known right answer here, so say whether it
    # came out that way rather than only reporting the numbers. With one
    # length there is no trend to check, and comparing a row against itself
    # would report a failure that is an artefact of the trajectory being
    # shorter than the shortest prefix after it.
    if len(rows) < 2:
        lines += [
            "",
            "## Missed states, not invented ones",
            "",
            f"Not assessed. Only one prefix length fits in this trajectory, "
            f"and the prediction is about how recall moves *with* length. Run "
            f"against a trajectory of at least {LENGTHS[1]} conformations.",
            "",
        ]
        _finish(lines, args, rows, n_features, target_d, crossing)
        return 0

    first, last = rows[0], rows[-1]
    recall_rises = last["recall"] > first["recall"]
    precision_flat = abs(last["precision"] - first["precision"]) < abs(
        last["recall"] - first["recall"]
    )
    lines += [
        "",
        "## Missed states, not invented ones",
        "",
        "A prefix has visited a subset of what the whole trajectory visited, "
        "so it should be missing states and inventing none. Recall should "
        "therefore rise with length while precision stays near its floor. "
        "That is a prediction with a correct answer, which makes this a test "
        "of the implementation rather than an illustration of it.",
        "",
        f"- Recall {'rises' if recall_rises else '**does not rise**'}: "
        f"{first['recall']:.3f} at {first['length']} conformations to "
        f"{last['recall']:.3f} at {last['length']}.",
        f"- Precision moves {'less' if precision_flat else '**more**'} than "
        f"recall: {first['precision']:.3f} to {last['precision']:.3f}.",
        "",
    ]
    if not (recall_rises and precision_flat):
        lines.append(
            "**One or both predictions failed.** That is a finding about the "
            "precision-recall implementation, not about this trajectory, and "
            "it should be resolved before the result is quoted.\n"
        )

    _finish(lines, args, rows, n_features, target_d, crossing)
    return 0


def _finish(lines, args, rows, n_features, target_d, crossing) -> None:
    document = "\n".join(lines)
    print()
    print(document)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(document)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "order_parameter": args.order_parameter,
                    "reference": args.reference,
                    "compare": args.compare,
                    "n_features": n_features,
                    "target_dissimilarity": target_d,
                    "crossing_length": crossing,
                    "rows": rows,
                },
                handle, indent=2, default=float,
            )


if __name__ == "__main__":
    raise SystemExit(main())
