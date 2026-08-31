#!/usr/bin/env python3
"""The manuscript figures, drawn from the measurements rather than beside them.

Two of the three panels are measured here, in this process, by the same code
the library uses -- the floor scaling calls ``feature_distance`` directly, so a
figure cannot drift from the implementation it describes without the drift
showing up in the figure.

The calibration panel cannot be: a thousand replicates at six correlation times
is an hour of compute, and putting that in a figure script means the figure
stops being regenerated. Those numbers are therefore constants below, tagged
with the command that produced them, and ``tests/test_figures.py`` asserts they
still match the table in ``docs/calibration.md``. A stale figure is then a test
failure rather than something a reader finds.

The manuscript itself lives in Overleaf rather than in this repository, so the
output of this script is uploaded there by hand. That is the one manual step
in the loop, and it is why the consistency test exists.

Usage::

    python scripts/figures.py                    # all figures, into paper/figures
    python scripts/figures.py --out-dir figs
    python scripts/figures.py --only calibration
    python scripts/figures.py --repeats 25       # floor scaling, as published
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --------------------------------------------------------------------------
# Measured elsewhere. Provenance is the point of these comments.
# --------------------------------------------------------------------------

#: False-positive rate against correlation time, both nulls on the same data.
#: Produced by `python scripts/calibration.py --study correlation
#: --replicates 1000`, 2000 frames per ensemble, nominal 5%. Tabulated in
#: docs/calibration.md and as Table 1 of the manuscript.
CALIBRATION = {
    "tau": (1, 2, 5, 10, 20, 50),
    "independent": (924, 490, 199, 100, 50, 20),
    "frame_permutation": (5.45, 23.67, 72.11, 93.73, 99.01, 99.92),
    "block_permutation": (1.66, 2.04, 2.30, 2.17, 2.24, 2.31),
    "nominal": 5.0,
}

#: Power of the block-permutation null against a genuine 0.8-sigma difference
#: at tau = 20, from the same run.
POWER_AT_TAU_20 = 98.0

# ACS: 3.25 in single column, 7.0 in double. A figure that has to be shrunk to
# fit is a figure with unreadable axis labels.
SINGLE = 3.25
DOUBLE = 7.0

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

BLOCK = "#1f5c8b"
FRAME = "#b3452c"
GREY = "#666666"


def _save(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    # EPS as well: ACS accepts PDF, but some production workflows still ask.
    fig.savefig(os.path.splitext(path)[0] + ".eps")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 1 -- the calibration panel, and the table of contents graphic
# --------------------------------------------------------------------------


def _calibration_axes(ax, *, annotate: bool) -> None:
    tau = np.asarray(CALIBRATION["tau"], dtype=float)

    ax.axhline(
        CALIBRATION["nominal"], color=GREY, lw=0.8, ls=(0, (4, 3)), zorder=1,
    )
    ax.plot(
        tau, CALIBRATION["frame_permutation"],
        "o-", color=FRAME, lw=1.4, ms=4, zorder=3,
        label="permuting conformations",
    )
    ax.plot(
        tau, CALIBRATION["block_permutation"],
        "s-", color=BLOCK, lw=1.4, ms=4, zorder=3,
        label="permuting blocks",
    )

    ax.set_xscale("log")
    ax.set_xticks(tau)
    ax.set_xticklabels([f"{int(t)}" for t in tau])
    ax.set_xlim(0.85, 60)
    ax.set_ylim(-4, 108)
    ax.set_yticks([0, 5, 25, 50, 75, 100])
    ax.set_xlabel(r"autocorrelation time $\tau$ (conformations)")
    ax.set_ylabel("false positives (%)")

    if annotate:
        ax.annotate(
            "nominal 5%", xy=(34, 8.5), color=GREY, fontsize=7,
        )
        ax.annotate(
            f"{CALIBRATION['frame_permutation'][-1]:.1f}%",
            xy=(22, 88), color=FRAME, fontsize=7,
        )
        ax.annotate(
            "1.7–2.3% across the range",
            xy=(1.55, 13.5), color=BLOCK, fontsize=7,
        )


def figure_calibration(out_dir: str) -> str:
    """The single most informative panel: what correlation costs a null."""
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    _calibration_axes(ax, annotate=True)
    ax.legend(loc="center left", frameon=False, bbox_to_anchor=(0.02, 0.62))
    return _save(fig, out_dir, "figure1_calibration.pdf")


def figure_toc(out_dir: str) -> str:
    """ACS table of contents graphic. At most 3.25 x 1.75 inches."""
    fig, ax = plt.subplots(figsize=(3.25, 1.75))
    _calibration_axes(ax, annotate=False)
    ax.set_ylabel("false positives (%)", fontsize=7)
    ax.set_xlabel(r"autocorrelation time $\tau$", fontsize=7)
    ax.legend(loc="center left", frameon=False, bbox_to_anchor=(0.03, 0.60))
    return _save(fig, out_dir, "toc_graphic.pdf")


# --------------------------------------------------------------------------
# Figure 2 -- the floor's scaling, and the correction that was refused
# --------------------------------------------------------------------------


def figure_floor_scaling(out_dir: str, repeats: int) -> str:
    """Measured here, by the same call the library makes.

    The claim is that the Jensen-Shannon distance does not decay as n^(-1/2),
    so plotting it against a drawn n^(-1/2) reference is the whole argument:
    the reader sees the two lines diverge rather than being asked to compare
    an exponent against a number in prose.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from floor_scaling import SIZES, draw

    from prothon.compare.distance import feature_distance

    sizes = np.asarray(SIZES, dtype=float)
    metrics = ("jsd", "wasserstein", "ks")
    labels = {
        "jsd": "Jensen–Shannon",
        "wasserstein": "Wasserstein-1",
        "ks": "Kolmogorov–Smirnov",
    }
    colours = {"jsd": BLOCK, "wasserstein": FRAME, "ks": "#4a7a4a"}
    markers = {"jsd": "s", "wasserstein": "o", "ks": "^"}

    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))

    for metric in metrics:
        rng = np.random.default_rng(1)  # the seed floor_scaling.py publishes
        means = []
        for n in SIZES:
            values = [
                feature_distance(
                    draw("gaussian", n, rng), draw("gaussian", n, rng),
                    metric, -6, 6, 80,
                )
                for _ in range(repeats)
            ]
            means.append(float(np.mean(values)))
        means = np.asarray(means)
        slope = float(np.polyfit(np.log(sizes), np.log(means), 1)[0])
        # Normalised at the smallest size, so three metrics on different
        # scales can share one axis and only the *shape* is compared.
        ax.plot(
            sizes, means / means[0],
            marker=markers[metric], color=colours[metric], lw=1.4, ms=4,
            label=f"{labels[metric]}  ({slope:.2f})",
        )
        print(f"  {metric}: exponent {slope:.3f}", file=sys.stderr)

    reference = (sizes / sizes[0]) ** -0.5
    ax.plot(
        sizes, reference, color=GREY, lw=1.0, ls=(0, (4, 3)),
        label=r"$n^{-1/2}$  (−0.50)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("conformations in each sample")
    ax.set_ylabel("distance, relative to $n=250$")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{int(n)}" for n in sizes])
    ax.set_yticks([1.0, 0.8, 0.6, 0.5, 0.4, 0.3])
    ax.set_yticklabels(["1.0", "0.8", "0.6", "0.5", "0.4", "0.3"])
    ax.minorticks_off()
    ax.legend(loc="upper right", frameon=False)
    return _save(fig, out_dir, "figure2_floor_scaling.pdf")


# --------------------------------------------------------------------------


FIGURES = {
    "calibration": lambda d, r: figure_calibration(d),
    "toc": lambda d, r: figure_toc(d),
    "floor": lambda d, r: figure_floor_scaling(d, r),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="paper/figures")
    parser.add_argument(
        "--repeats", type=int, default=25,
        help="replicates per sample size for the floor panel; 25 is published",
    )
    parser.add_argument("--only", choices=sorted(FIGURES), action="append")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wanted = args.only or sorted(FIGURES)
    for name in wanted:
        print(f"{name}:", file=sys.stderr)
        path = FIGURES[name](args.out_dir, args.repeats)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
