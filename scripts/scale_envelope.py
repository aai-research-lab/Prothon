#!/usr/bin/env python3
"""Measure what a Prothon run costs, in seconds and in memory.

The documentation says the method is linear in the number of conformations.
That is a claim about the algorithm; this measures the implementation, which is
what a user waits for.

Every measurement runs in its own subprocess and reports its own peak resident
memory, because the expensive allocations happen inside MDTraj's compiled
extensions where ``tracemalloc`` cannot see them, and because a high-water mark
in one long-lived process would attribute the largest allocation to every
measurement after it.

Usage::

    python scripts/scale_envelope.py                 # a few minutes
    python scripts/scale_envelope.py --full          # the published grid
    python scripts/scale_envelope.py --out docs/performance.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import time

# Peak resident memory is reported in bytes on macOS and kilobytes on Linux.
_RSS_SCALE = 1.0 if sys.platform == "darwin" else 1024.0

_CHILD = textwrap.dedent(
    '''
    import json, resource, sys, time
    import numpy as np
    import mdtraj as md

    spec = json.loads(sys.argv[1])
    n_res, n_frames, task = spec["residues"], spec["frames"], spec["task"]

    rng = np.random.default_rng(0)
    top = md.Topology()
    chain = top.add_chain()
    for _ in range(n_res):
        residue = top.add_residue("ALA", chain)
        for name in ("N", "CA", "C", "O", "CB"):
            top.add_atom(name, md.element.carbon, residue)

    # A loose globule rather than an extended chain, so that a realistic
    # fraction of pairs falls inside the contact cutoff and the sigmoid is
    # actually evaluated on both sides of it.
    xyz = np.zeros((n_frames, top.n_atoms, 3), dtype=np.float32)
    radius = 0.30 * n_res ** (1 / 3)
    for atom in top.atoms:
        i = atom.residue.index
        angle = 2.4 * i
        xyz[:, atom.index, :] = [
            radius * np.cos(angle) * (i / n_res) ** 0.5,
            radius * np.sin(angle) * (i / n_res) ** 0.5,
            0.38 * i / n_res**0.5,
        ]
    xyz += rng.normal(0, 0.25, xyz.shape).astype(np.float32)
    traj = md.Trajectory(xyz, top)

    from prothon.core.dissimilarity import dissimilarity
    from prothon.core.representation import compute_representation

    start = time.perf_counter()
    if task in ("cbcn", "cacn", "caba", "cata", "sasa"):
        compute_representation(traj, task)
    elif task == "compare":
        left = compute_representation(traj, "cbcn")
        right = left + rng.normal(0, 0.4, left.shape)
        dissimilarity(
            left, right, float(min(left.min(), right.min())),
            float(max(left.max(), right.max())),
            x_num=100, s_num=3, n_permutations=100, random_state=0,
        )
    else:
        raise SystemExit(f"unknown task {task}")
    elapsed = time.perf_counter() - start

    print(json.dumps({
        "seconds": elapsed,
        "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }))
    '''
)


def measure(residues: int, frames: int, task: str, timeout: float) -> dict | None:
    spec = json.dumps({"residues": residues, "frames": frames, "task": task})
    started = time.perf_counter()
    try:
        done = subprocess.run(
            [sys.executable, "-c", _CHILD, spec],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"seconds": float("inf"), "gb": float("nan"), "timed_out": True}
    if done.returncode != 0:
        print(f"    failed: {done.stderr.strip().splitlines()[-1:]}", file=sys.stderr)
        return None
    payload = json.loads(done.stdout.strip().splitlines()[-1])
    return {
        "seconds": payload["seconds"],
        "gb": payload["rss"] * _RSS_SCALE / 1e9,
        "wall": time.perf_counter() - started,
        "timed_out": False,
    }


def fit_exponent(sizes, seconds) -> float:
    """Slope of log time against log size: 1 is linear, 2 is quadratic."""
    import numpy as np

    usable = [(s, t) for s, t in zip(sizes, seconds) if t and t > 1e-4]
    if len(usable) < 2:
        return float("nan")
    x = np.log([s for s, _ in usable])
    y = np.log([t for _, t in usable])
    return float(np.polyfit(x, y, 1)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="the published grid")
    parser.add_argument("--out", help="write a markdown table here")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    if args.full:
        residue_grid = [25, 50, 100, 200, 400]
        frame_grid = [500, 2000, 10000, 50000]
        tasks = ["cbcn", "cata", "sasa", "compare"]
    else:
        residue_grid = [25, 50, 100, 200]
        frame_grid = [500, 2000, 8000]
        tasks = ["cbcn", "cata", "compare"]

    lines: list[str] = []
    emit = lines.append

    emit("# Performance")
    emit("")
    emit("What a run costs, measured rather than asserted. Every point runs in")
    emit("its own process and reports its own peak resident memory.")
    emit("")
    emit(f"Measured on `{sys.platform}`, Python "
         f"{sys.version_info.major}.{sys.version_info.minor}.")
    emit("")

    # --- residues, at fixed frames ---
    held = 2000
    emit(f"## Against chain length ({held} conformations)")
    emit("")
    emit("| residues | " + " | ".join(f"{t} (s)" for t in tasks) + " | peak (GB) |")
    emit("|---" * (len(tasks) + 2) + "|")
    by_task: dict[str, list] = {t: [] for t in tasks}
    for n_res in residue_grid:
        row, peak = [], 0.0
        for task in tasks:
            print(f"  {task} {n_res} residues x {held} frames", file=sys.stderr)
            result = measure(n_res, held, task, args.timeout)
            if result is None or result["timed_out"]:
                row.append("—")
                by_task[task].append(None)
            else:
                row.append(f"{result['seconds']:.2f}")
                by_task[task].append(result["seconds"])
                peak = max(peak, result["gb"])
        emit(f"| {n_res} | " + " | ".join(row) + f" | {peak:.2f} |")
    emit("")
    emit("Slope of log time against log chain length "
         "(1 is linear, 2 is quadratic):")
    emit("")
    for task in tasks:
        exponent = fit_exponent(residue_grid, by_task[task])
        emit(f"- `{task}`: {exponent:.2f}")
    emit("")

    # --- frames, at fixed residues ---
    held = 100
    emit(f"## Against ensemble size ({held} residues)")
    emit("")
    emit("| conformations | " + " | ".join(f"{t} (s)" for t in tasks) + " | peak (GB) |")
    emit("|---" * (len(tasks) + 2) + "|")
    by_task = {t: [] for t in tasks}
    for n_frames in frame_grid:
        row, peak = [], 0.0
        for task in tasks:
            print(f"  {task} {held} residues x {n_frames} frames", file=sys.stderr)
            result = measure(held, n_frames, task, args.timeout)
            if result is None or result["timed_out"]:
                row.append("—")
                by_task[task].append(None)
            else:
                row.append(f"{result['seconds']:.2f}")
                by_task[task].append(result["seconds"])
                peak = max(peak, result["gb"])
        emit(f"| {n_frames} | " + " | ".join(row) + f" | {peak:.2f} |")
    emit("")
    emit("Slope of log time against log ensemble size:")
    emit("")
    for task in tasks:
        exponent = fit_exponent(frame_grid, by_task[task])
        emit(f"- `{task}`: {exponent:.2f}")
    emit("")

    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(table + "\n")
        print(f"\nwritten to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
