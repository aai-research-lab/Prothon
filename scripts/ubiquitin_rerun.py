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


def compare(ensembles, order_parameter, legacy, seed, s_num, permutations):
    """Every ensemble against Q99, under one statistical treatment."""
    from prothon import Prothon

    study = Prothon(ensembles=ensembles, random_state=seed)
    return study.compare_ensembles(
        order_parameters=order_parameter,
        ref=0,
        s_num=s_num,
        n_permutations=permutations,
        legacy=legacy,
    )[order_parameter]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.expanduser("~/ubiquitin"))
    parser.add_argument("--order-parameters", default="cbcn,cacn,caba,cata,sasa")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--s-num", type=int, default=10)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--json")
    args = parser.parse_args()

    print("Loading:", file=sys.stderr)
    ensembles = load(args.data, args.stride)

    names = [p.strip() for p in args.order_parameters.split(",") if p.strip()]
    lines = [
        "# Ubiquitin, re-run",
        "",
        "Every ensemble against Q99. `published` is the statistical treatment "
        "of the 2023 paper; `current` is the default now.",
        "",
    ]
    raw: dict = {}

    for name in names:
        print(f"\n{name}:", file=sys.stderr)
        print("  published treatment...", file=sys.stderr)
        old = compare(ensembles, name, True, args.seed, args.s_num,
                      args.n_permutations)
        print("  current treatment...", file=sys.stderr)
        new = compare(ensembles, name, False, args.seed, args.s_num,
                      args.n_permutations)

        n_features = int(old[0].local_dissimilarity.size)
        lines += [
            f"## {name.upper()}",
            "",
            "| against Q99 | d | Δd | floor | published | current | τ |",
            "|---|---|---|---|---|---|---|",
        ]
        rows = []
        for a, b in zip(old, new):
            label = ensembles[b.ensemble_index].label
            # The *unmasked* per-residue values are the quantity that should
            # not have moved. `global_dissimilarity` is a mean over the masked
            # ones, so it necessarily changes when the significance filter
            # does -- comparing that would be comparing the filter to itself.
            delta = float(
                np.abs(
                    a.raw_local_dissimilarity - b.raw_local_dissimilarity
                ).max()
            )
            calls_old = int(a.n_significant)
            calls_new = int(b.n_significant)
            withheld = not b.p_values_reported
            # The *raw* mean, not `global_dissimilarity`: that is a mean over
            # masked values and reads as zero whenever nothing survives the
            # filter, which is the opposite of what it looks like.
            raw = float(np.mean(b.raw_local_dissimilarity))
            lines.append(
                f"| {label} | {raw:.4f} | {b.noise_floor:.4f} | "
                f"{'yes' if raw > b.noise_floor else 'no'} | "
                f"{calls_old}/{n_features} | "
                + ("withheld" if withheld else f"{calls_new}/{n_features}")
                + f" | {b.correlation_time:.0f} | {b.n_blocks} | "
                + f"{min(b.effective_samples):.0f} |"
            )
            rows.append({
                "ensemble": label,
                "raw_mean_distance": raw,
                "raw_max_difference": delta,
                "n_blocks": int(b.n_blocks),
                "dissimilarity_published": float(a.global_dissimilarity),
                "dissimilarity_current": float(b.global_dissimilarity),
                "noise_floor": float(b.noise_floor),
                "resolved": bool(b.resolved),
                "significant_published": calls_old,
                "significant_current": calls_new,
                "p_values_withheld": withheld,
                "correlation_time": float(b.correlation_time),
                "n_features": n_features,
            })
        raw[name] = rows

        magnitudes = max(
            float(np.abs(a.raw_local_dissimilarity - b.raw_local_dissimilarity).max())
            for a, b in zip(old, new)
        )
        flagged_old = sum(int(a.n_significant) for a in old)
        flagged_new = sum(int(b.n_significant) for b in new)
        total = n_features * len(old)
        unresolved = sum(
            1 for b in new
            if float(np.mean(b.raw_local_dissimilarity)) <= b.noise_floor
        )
        withheld_count = sum(1 for b in new if not b.p_values_reported)
        lines += [
            "",
            f"Largest change in any per-residue distance: **{magnitudes:.2e}**"
            + (
                " — unchanged, as it should be: the Jensen–Shannon calculation "
                "was always correct and only the significance filter moved."
                if magnitudes < 1e-9
                else " — this should be zero. Investigate before quoting "
                "anything else here."
            ),
            "",
            f"Residues called different: **{flagged_old}/{total}** "
            f"({flagged_old / total:.0%}) under the published treatment, "
            f"**{flagged_new}/{total}** ({flagged_new / total:.0%}) now.",
            "",
        ]
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

    document = "\n".join(lines)
    print()
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
