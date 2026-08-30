#!/usr/bin/env python3
"""Re-run the 2023 ubiquitin comparisons and report how far the calls move.

The dataset is the one published with the paper (Zenodo 10.5281/zenodo.7792288):
six molecular dynamics ensembles of ubiquitin at decreasing degrees of folding,
Q99 through Q75, sharing one topology.

That grading is what makes it a test rather than a demonstration. Q95 is nearly
Q99 and Q75 is not, so the number of residues called different should rise
monotonically along the series. A null that is too narrow flattens it: if
everything is called different, the ordering carries no information.

**Two things are expected to move and one is not.** The Jensen-Shannon
distances themselves were always computed correctly and should be unchanged to
several decimal places -- confirming that is what lets the published figures
stand. What changes is which residues survive the significance filter, for two
independent reasons that both make the published null too permissive:

    the null was a bootstrap of each ensemble against itself
    the frames come from continuous trajectories and are correlated in time

Usage::

    python scripts/ubiquitin_rerun.py --data ~/ubiquitin
    python scripts/ubiquitin_rerun.py --data ~/ubiquitin --out ubiquitin.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

#: Ensembles in the order the paper presents them, most folded first.
SERIES = ("Q99", "Q95", "Q90", "Q85", "Q80", "Q75")


def load(directory: str, stride: int):
    from prothon.ingest import Ensemble

    topology = os.path.join(directory, "topology.pdb")
    if not os.path.exists(topology):
        raise SystemExit(f"No topology.pdb in {directory}")

    ensembles = []
    for name in SERIES:
        path = os.path.join(directory, f"{name}.dcd")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}")
        ensembles.append(
            Ensemble.from_trajectory(path, topology, label=name, stride=stride)
        )
        print(
            f"  {name}: {ensembles[-1].n_frames} frames, "
            f"{ensembles[-1].trajectory.topology.n_residues} residues",
            file=sys.stderr,
        )
    return ensembles


def compare(ensembles, order_parameter, legacy, seed, s_num, permutations,
            sample_size, n_jobs):
    """Every ensemble against Q99, under one statistical treatment.

    ``sample_size`` must exceed the frame count for the two treatments to be
    comparable. The published treatment computes the distance on every frame;
    the current one subsamples to ``sample_size`` first, so leaving the default
    of 1000 against 5000-frame ensembles compares 5000 frames with 1000 and
    reports the difference as though the calculation had changed.
    """
    from prothon import Prothon

    study = Prothon(ensembles=ensembles, random_state=seed)
    return study.compare_ensembles(
        order_parameters=order_parameter,
        ref=0,
        s_num=s_num,
        n_permutations=permutations,
        sample_size=sample_size,
        n_jobs=n_jobs,
        legacy=legacy,
    )[order_parameter]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.expanduser("~/ubiquitin"))
    parser.add_argument("--order-parameters", default="cbcn,cacn,caba,cata,sasa")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--s-num", type=int, default=10)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument(
        "--sample-size", type=int, default=0,
        help="Conformations used per comparison. 0 uses every frame, which is "
             "what makes the two treatments comparable.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Worker processes for the permutation null and the noise floor. "
             "-1 uses every core. The result does not depend on this.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--json")
    args = parser.parse_args()

    print("Loading:", file=sys.stderr)
    ensembles = load(args.data, args.stride)

    sample_size = args.sample_size or max(e.n_frames for e in ensembles)
    print(f"  using {sample_size} conformations per comparison", file=sys.stderr)

    names = [p.strip() for p in args.order_parameters.split(",") if p.strip()]
    lines = [
        "# Ubiquitin, re-run",
        "",
        "Every ensemble against Q99. `published` is the statistical treatment "
        "of the 2023 paper; `current` is the default now.",
        "",
    ]
    collected: dict = {}
    summary: dict = {}

    for name in names:
        print(f"\n{name}:", file=sys.stderr)
        print("  published treatment...", file=sys.stderr)
        old = compare(ensembles, name, True, args.seed, args.s_num,
                      args.n_permutations, sample_size, args.n_jobs)
        print("  current treatment...", file=sys.stderr)
        new = compare(ensembles, name, False, args.seed, args.s_num,
                      args.n_permutations, sample_size, args.n_jobs)

        n_features = int(old[0].local_dissimilarity.size)
        lines += [
            f"## {name.upper()}",
            "",
            "| against Q99 | raw d | floor | resolved | published | current "
            "| τ | blocks | independent |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        rows = []
        for a, b in zip(old, new):
            label = ensembles[b.ensemble_index].label
            # The *unmasked* per-residue values are the quantity that should
            # not have moved. The masked mean necessarily changes when the
            # significance filter does, so comparing that would be comparing
            # the filter to itself.
            delta = float(
                np.abs(
                    a.raw_local_dissimilarity - b.raw_local_dissimilarity
                ).max()
            )
            calls_old = int(a.n_significant)
            calls_new = int(b.n_significant)
            withheld = not b.p_values_reported
            raw_mean = float(b.global_dissimilarity)
            lines.append(
                f"| {label} | {raw_mean:.4f} | {b.noise_floor:.4f} | "
                f"{'yes' if b.resolved else 'no'} | "
                f"{calls_old}/{n_features} | "
                + ("withheld" if withheld else f"{calls_new}/{n_features}")
                + f" | {b.correlation_time:.0f} | {b.n_blocks} | "
                + f"{sample_size / max(1.0, b.correlation_time):.0f} |"
            )
            rows.append({
                "ensemble": label,
                "raw_mean_distance": raw_mean,
                "raw_max_difference": delta,
                "n_blocks": int(b.n_blocks),
                "dissimilarity_published": float(a.global_dissimilarity),
                "dissimilarity_current": float(b.global_dissimilarity),
                "masked_published": float(a.masked_global_dissimilarity),
                "masked_current": float(b.masked_global_dissimilarity),
                "noise_floor": float(b.noise_floor),
                "resolved": bool(b.resolved),
                "significant_published": calls_old,
                "significant_current": calls_new,
                "p_values_withheld": withheld,
                "correlation_time": float(b.correlation_time),
                "n_features": n_features,
            })
        collected[name] = rows

        magnitudes = max(
            float(np.abs(a.raw_local_dissimilarity - b.raw_local_dissimilarity).max())
            for a, b in zip(old, new)
        )
        flagged_old = sum(int(a.n_significant) for a in old)
        # Only the comparisons a test could actually be run on. Dividing by
        # every comparison would count a withheld p-value as a residue that
        # was tested and found not to differ, which is the one distinction
        # this whole re-run exists to draw.
        tested = [b for b in new if b.p_values_reported]
        flagged_new = sum(int(b.n_significant) for b in tested)
        total = n_features * len(old)
        tested_total = n_features * len(tested)
        unresolved = sum(1 for b in new if not b.resolved)
        withheld_count = len(new) - len(tested)
        from prothon.core.representation import resolve_order_parameter

        spec = resolve_order_parameter(name)
        lines += [
            "",
            f"Largest change in any per-residue distance: **{magnitudes:.2e}**"
            + (
                " — unchanged, as it should be: the Jensen–Shannon calculation "
                "was always correct and only the significance filter moved."
                if magnitudes < 1e-9
                else (
                    " — expected here, and a finding in its own right. This is "
                    "a circular feature, and the published treatment estimated "
                    "its density on a linear grid: a torsion whose values "
                    "straddle the wrap at ±π appears as two separated modes, "
                    "and two ensembles sitting on opposite sides of the wrap "
                    "appear to share no support at all. The magnitudes change "
                    "for those residues, not only the significance calls."
                    if spec.circular
                    else " — this should be zero. Investigate before quoting "
                    "anything else here."
                )
            ),
            "",
            f"Residues called different: **{flagged_old}/{total}** "
            f"({flagged_old / total:.0%}) under the published treatment, "
            f"across all {len(old)} comparisons and {n_features} features.",
            "",
            (
                (
                    f"Under the current treatment, **no comparison could be "
                    f"tested at all**: the correlation time leaves too few "
                    f"independent blocks in every one of the {len(old)}. There "
                    f"is no proportion to report, which is the point -- a "
                    f"percentage here would be a statement the data cannot "
                    f"support."
                )
                if not tested_total
                else (
                    f"Under the current treatment, "
                    f"**{flagged_new}/{tested_total}** "
                    f"({flagged_new / tested_total:.0%}) of the features that "
                    f"could be tested at all"
                    + (
                        f", the remaining {withheld_count} comparison"
                        f"{'s' if withheld_count != 1 else ''} having been "
                        f"withheld."
                        if withheld_count
                        else "."
                    )
                )
            ),
            "",
        ]
        summary[name] = {
            "n_features": n_features,
            "comparisons": len(old),
            "tested": len(tested),
            "withheld": withheld_count,
            "flagged_published": flagged_old,
            "flagged_current": flagged_new,
            "fraction_published": flagged_old / total,
            "fraction_current": flagged_new / tested_total if tested_total else None,
            "tau_min": min(float(b.correlation_time) for b in new),
            "tau_max": max(float(b.correlation_time) for b in new),
            "max_magnitude_change": magnitudes,
        }
        if unresolved:
            lines.append(
                f"{unresolved} of {len(new)} comparisons fall below their own "
                f"noise floor and are not resolvable at this sampling.\n"
            )
        if withheld_count:
            lines.append(
                f"{withheld_count} of {len(new)} report no p-value at all: the "
                f"correlation time leaves too few independent blocks to build "
                f"a permutation null from. That is a statement about the "
                f"sampling, not about the ensembles.\n"
            )

    lines += [
        "## Across order parameters",
        "",
        "`published` divides by every comparison, as that treatment reported a "
        "p-value for all of them. `current` divides by the comparisons a test "
        "could be run on, which is not the same denominator -- a withheld "
        "p-value is not a residue that was tested and found not to differ.",
        "",
        "| order parameter | features | τ range | published | current | withheld |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in summary.items():
        current = (
            f"{row['fraction_current']:.0%}"
            if row["fraction_current"] is not None
            else "—"
        )
        lines.append(
            f"| `{name}` | {row['n_features']} | "
            f"{row['tau_min']:.0f}–{row['tau_max']:.0f} | "
            f"{row['fraction_published']:.0%} | {current} | "
            f"{row['withheld']}/{row['comparisons']} |"
        )
    total_tests = sum(
        row["n_features"] * row["comparisons"] for row in summary.values()
    )
    lines += [
        "",
        f"**{total_tests} per-residue tests** in total across "
        f"{len(summary)} order parameters.",
        "",
    ]
    collected["_summary"] = summary

    document = "\n".join(lines)
    print()
    print(document)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(document)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(collected, handle, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
