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
    python scripts/scale_envelope.py --json performance-evidence.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import platform
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

    spec = json.loads(sys.argv[1])
    n_res, n_frames, task = spec["residues"], spec["frames"], spec["task"]

    rng = np.random.default_rng(0)
    if task in ("mmd", "c2st"):
        # The joint methods consume representations, not coordinates. Use an
        # ordered AR(1) matrix and a supplied correlation time so the measured
        # path performs real block-aware MMD or grouped C2ST without timing a
        # representation method a second time.
        phi = np.exp(-1.0 / 5.0)
        scale = np.sqrt(1.0 - phi**2)

        def series():
            values = np.empty((n_frames, n_res), dtype=np.float32)
            values[0] = rng.normal(size=n_res)
            noise = rng.normal(size=values.shape).astype(np.float32) * scale
            for frame in range(1, n_frames):
                values[frame] = phi * values[frame - 1] + noise[frame]
            return values

        left, right = series(), series()
        from prothon.compare.joint import (
            classifier_two_sample,
            maximum_mean_discrepancy,
        )

        start = time.perf_counter()
        options = {
            "correlation_time_frames_a": 5.0,
            "correlation_time_frames_b": 5.0,
            "random_state": 0,
        }
        if task == "mmd":
            maximum_mean_discrepancy(left, right, **options)
        else:
            classifier_two_sample(left, right, **options)
    else:
        import mdtraj as md

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

        from prothon.compare.dissimilarity import dissimilarity
        from prothon.represent.order_parameters import compute_representation

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


def measure(residues: int, frames: int, task: str, timeout: float) -> dict:
    spec = json.dumps({"residues": residues, "frames": frames, "task": task})
    started = time.perf_counter()
    try:
        done = subprocess.run(
            [sys.executable, "-c", _CHILD, spec],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timed_out",
            "seconds": None,
            "gb": None,
            "wall_seconds": time.perf_counter() - started,
            "error": f"exceeded {timeout:g} seconds",
        }
    if done.returncode != 0:
        error = "\n".join(done.stderr.strip().splitlines()[-3:])
        print(f"    failed: {error}", file=sys.stderr)
        return {
            "status": "failed",
            "seconds": None,
            "gb": None,
            "wall_seconds": time.perf_counter() - started,
            "error": error or f"child exited {done.returncode}",
        }
    payload = json.loads(done.stdout.strip().splitlines()[-1])
    return {
        "status": "ok",
        "seconds": payload["seconds"],
        "gb": payload["rss"] * _RSS_SCALE / 1e9,
        "wall_seconds": time.perf_counter() - started,
        "error": None,
    }


#: Timings below this are dominated by process start-up and imports rather
#: than by the work, and a slope fitted through them describes the harness. On
#: two machines the same measurement gave 1.80 and 2.41 for a quantity whose
#: true exponent is 2, entirely because the fastest points were 0.02 s.
_TIMING_FLOOR = 0.1


def fit_exponent(sizes, seconds) -> tuple[float, int]:
    """Slope of log time against log size: 1 is linear, 2 is quadratic.

    Returns the slope and the number of points it was fitted through, because
    a slope from two points is worth reporting differently from one through
    five.
    """
    import numpy as np

    usable = [(s, t) for s, t in zip(sizes, seconds) if t and t >= _TIMING_FLOOR]
    if len(usable) < 2:
        return float("nan"), len(usable)
    x = np.log([s for s, _ in usable])
    y = np.log([t for _, t in usable])
    return float(np.polyfit(x, y, 1)[0]), len(usable)


# Seconds vary with hardware and load, so they are evidence rather than a
# portable pass/fail threshold. Scaling exponents and the exact regression
# memory point transfer much better between machines. These upper bounds leave
# room for timing noise without accepting the old 1.38 ensemble-size exponent.
_SCALING_BUDGETS = {
    ("frames", "cbcn"): 1.30,
    ("frames", "compare"): 1.30,
    ("frames", "mmd"): 1.30,
    ("frames", "c2st"): 1.30,
    ("frames", "sasa"): 1.30,
    ("residues", "cbcn"): 2.50,
    ("residues", "compare"): 1.60,
    ("residues", "mmd"): 1.60,
    ("residues", "c2st"): 1.60,
}
_MEMORY_FRAMES = 8000
_MAX_MEMORY_GB = 1.0


def performance_gates(measurements: list[dict], scaling: dict) -> dict:
    """Evaluate predeclared, machine-portable performance budgets."""
    checks = []
    failed = [row for row in measurements if row["status"] != "ok"]
    checks.append(
        {
            "name": "every measurement completed",
            "actual": len(failed),
            "maximum": 0,
            "passed": not failed,
        }
    )

    measured_tasks = {row["task"] for row in measurements}
    for (axis, task), maximum in _SCALING_BUDGETS.items():
        if task not in measured_tasks:
            continue
        fit = scaling.get(axis, {}).get(task, {})
        exponent = fit.get("exponent")
        points = fit.get("points", 0)
        passed = (
            points >= 2
            and exponent is not None
            and math.isfinite(exponent)
            and exponent <= maximum
        )
        checks.append(
            {
                "name": f"{task} scaling against {axis}",
                "actual": exponent,
                "maximum": maximum,
                "points": points,
                "passed": passed,
            }
        )

    memory_rows = [
        row
        for row in measurements
        if row["axis"] == "frames"
        and row["frames"] == _MEMORY_FRAMES
        and row["status"] == "ok"
    ]
    peak_gb = max((row["gb"] for row in memory_rows), default=None)
    checks.append(
        {
            "name": f"peak RSS at {_MEMORY_FRAMES} frames",
            "actual": peak_gb,
            "maximum": _MAX_MEMORY_GB,
            "unit": "GB",
            "passed": peak_gb is not None and peak_gb <= _MAX_MEMORY_GB,
        }
    )
    return {"checks": checks, "passed": all(check["passed"] for check in checks)}


def _software_versions() -> dict:
    versions = {"python": platform.python_version()}
    for distribution in ("prothon-ensembles", "numpy", "scipy", "mdtraj"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _display_exponent(exponent: float, points: int) -> str:
    if points < 2:
        return "too fast to fit at these sizes"
    return f"{exponent:.2f} (from {points} points)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="the published grid")
    parser.add_argument("--out", help="write a markdown table here")
    parser.add_argument("--json", help="write measurements and gate results as JSON")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    if args.full:
        residue_grid = [25, 50, 100, 200, 400]
        # Keep the original published points and the exact 8,000-frame memory
        # regression point. A nearby value would not test the historical bug.
        frame_grid = [500, 2000, 8000, 10000, 50000]
        tasks = ["cbcn", "cata", "sasa", "compare", "mmd", "c2st"]
    else:
        residue_grid = [25, 50, 100, 200]
        frame_grid = [500, 2000, 8000]
        tasks = ["cbcn", "cata", "compare", "mmd", "c2st"]

    lines: list[str] = []
    measurements: list[dict] = []
    scaling: dict[str, dict[str, dict]] = {"residues": {}, "frames": {}}
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
            measurements.append(
                {"axis": "residues", "residues": n_res, "frames": held, "task": task, **result}
            )
            if result["status"] != "ok":
                row.append("—")
                by_task[task].append(None)
            else:
                row.append(f"{result['seconds']:.2f}")
                by_task[task].append(result["seconds"])
                peak = max(peak, result["gb"])
        emit(f"| {n_res} | " + " | ".join(row) + f" | {peak:.2f} |")
    emit("")
    emit("Slope of log time against log chain length "
         "(1 is linear, 2 is quadratic), fitted only through points slower "
         f"than {_TIMING_FLOOR} s:")
    emit("")
    for task in tasks:
        exponent, points = fit_exponent(residue_grid, by_task[task])
        scaling["residues"][task] = {
            "exponent": exponent if math.isfinite(exponent) else None,
            "points": points,
        }
        emit(f"- `{task}`: {_display_exponent(exponent, points)}")
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
            measurements.append(
                {"axis": "frames", "residues": held, "frames": n_frames, "task": task, **result}
            )
            if result["status"] != "ok":
                row.append("—")
                by_task[task].append(None)
            else:
                row.append(f"{result['seconds']:.2f}")
                by_task[task].append(result["seconds"])
                peak = max(peak, result["gb"])
        emit(f"| {n_frames} | " + " | ".join(row) + f" | {peak:.2f} |")
    emit("")
    emit("Slope of log time against log ensemble size, fitted only through "
         f"points slower than {_TIMING_FLOOR} s:")
    emit("")
    for task in tasks:
        exponent, points = fit_exponent(frame_grid, by_task[task])
        scaling["frames"][task] = {
            "exponent": exponent if math.isfinite(exponent) else None,
            "points": points,
        }
        emit(f"- `{task}`: {_display_exponent(exponent, points)}")
    emit("")

    gates = performance_gates(measurements, scaling)
    emit("## Regression gates")
    emit("")
    emit("Raw seconds are retained but not gated because they are machine-dependent.")
    emit("")
    for check in gates["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        actual = check["actual"]
        shown = "not measured" if actual is None else f"{actual:.3g}"
        unit = f" {check['unit']}" if check.get("unit") else ""
        emit(f"- **{mark}** — {check['name']}: {shown}{unit} (maximum {check['maximum']:g}{unit})")
    emit("")

    evidence = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "full" if args.full else "quick",
        "platform": platform.platform(),
        "commit": _git_commit(),
        "software": _software_versions(),
        "grids": {"residues": residue_grid, "frames": frame_grid, "tasks": tasks},
        "measurements": measurements,
        "scaling": scaling,
        "gates": gates,
        "passed": gates["passed"],
    }

    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(table + "\n")
        print(f"\nwritten to {args.out}", file=sys.stderr)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(evidence, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"written to {args.json}", file=sys.stderr)
    return 0 if gates["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
