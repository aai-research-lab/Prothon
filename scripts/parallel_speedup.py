#!/usr/bin/env python3
"""How much the permutation null and the noise floor gain from more cores.

Both are loops over independent draws and together they are over ninety per
cent of the cost of a comparison, so the ceiling is high. What the ceiling is
worth in practice depends on the machine: each worker process imports NumPy and
SciPy before it does anything, and on a small job that costs more than the work
saved.

This measures the crossover rather than assuming it. Run it on the machine that
will do the work.

Usage::

    python scripts/parallel_speedup.py
    python scripts/parallel_speedup.py --residues 76 --frames 5000 --jobs 1 2 4 8 16
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from prothon.compare.dissimilarity import dissimilarity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residues", type=int, default=76)
    parser.add_argument("--frames", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--s-num", type=int, default=5)
    parser.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--out")
    args = parser.parse_args()

    cores = os.cpu_count() or 1
    print(f"  {cores} cores visible", file=sys.stderr)

    rng = np.random.default_rng(0)
    a = rng.normal(size=(args.frames, args.residues))
    b = rng.normal(0.3, 1.0, (args.frames, args.residues))
    common = dict(
        x_min=-5.0, x_max=5.0, x_num=100, s_num=args.s_num,
        n_permutations=args.n_permutations, sample_size=args.frames,
        random_state=0, block_permutation=False,
    )

    lines = [
        "# Parallel speedup",
        "",
        f"{args.residues} residues, {args.frames} conformations, "
        f"{args.n_permutations} permutations, {args.s_num} split-half repeats, "
        f"on a machine with {cores} cores.",
        "",
        "| workers | seconds | speedup | identical to serial |",
        "|---|---|---|---|",
    ]

    reference = None
    baseline = None
    for jobs in args.jobs:
        if jobs > cores:
            continue
        start = time.time()
        result = dissimilarity(a, b, n_jobs=jobs, **common)
        elapsed = time.time() - start
        if reference is None:
            reference, baseline = result, elapsed
        same = np.allclose(
            reference.raw_local_dissimilarity, result.raw_local_dissimilarity
        ) and abs(reference.noise_floor - result.noise_floor) < 1e-12
        lines.append(
            f"| {jobs} | {elapsed:.1f} | {baseline / elapsed:.2f}× | "
            f"{'yes' if same else 'NO'} |"
        )
        print(f"  {jobs:>3} workers: {elapsed:6.1f}s", file=sys.stderr)

    lines += [
        "",
        "The result is identical whatever the worker count, because the seeds "
        "are drawn from the caller's generator before the work is divided.",
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
