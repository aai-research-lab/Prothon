"""Figures, and the files that go with them.

Two things separate the functions here from those in version 2.0.

**Replotting no longer writes to disk.** ``replot_global_dissimilarity`` called
the same function that produced the original figure, so asking for a
differently styled copy silently overwrote the saved one -- and the styling
arguments the README documented (``xlabel``, ``ylabel``, ``title``) were
accepted and discarded. The replot functions now build a figure, honour every
styling argument, and return it without touching the output directory.

**Every dissimilarity figure carries its noise floor.** A bar chart of global
dissimilarities invites the eye to compare heights, including heights that are
indistinguishable from the sampling error. The floor is drawn as a dashed line,
so a bar below it reads as what it is.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import matplotlib

if not os.environ.get("DISPLAY") and os.name != "nt":  # pragma: no cover
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from ..utils import get_logger

logger = get_logger("plotting")

__all__ = [
    "dimensionality_reduction_plot",
    "get_ensemble_colors",
    "get_method_output_dir",
    "plot_combined_local_dissimilarity",
    "plot_global_dissimilarity_bar",
    "plot_global_dissimilarity_line",
    "plot_local_dissimilarity",
    "replot_global_dissimilarity",
    "replot_local_dissimilarity",
    "save_matrix_data_and_plot",
]

#: The first twelve ensemble colours, fixed so that a bar chart, a combined
#: line plot and a dimensionality-reduction scatter all give the same ensemble
#: the same colour.
PALETTE: tuple[str, ...] = (
    "red", "gold", "darkgreen", "blue", "darkorchid", "lightcoral",
    "orange", "lime", "deepskyblue", "magenta", "navy", "cyan",
)

#: Above this many frames, MDS is refused rather than attempted. It builds a
#: dense pairwise distance matrix, so 50,000 frames would ask for 20 GB and a
#: run that appears to hang. PCA and t-SNE have no such limit.
MDS_FRAME_LIMIT = 5000


def get_method_output_dir(output_dir: str | None, measure: str) -> str:
    """Create and return ``<output_dir>/<measure>_output``, or ``<measure>_output``."""
    path = (
        os.path.join(output_dir, f"{measure}_output")
        if output_dir
        else f"{measure}_output"
    )
    os.makedirs(path, exist_ok=True)
    return path


def get_ensemble_colors(n: int, seed: int = 0) -> list[str]:
    """Colours for ``n`` ensembles, the first twelve fixed.

    Beyond twelve, extra colours are drawn from a seeded generator so that two
    runs of the same study produce the same figure. Version 2.0 used the global
    ``random`` state, so a thirteenth ensemble changed colour between runs.
    """
    if n <= len(PALETTE):
        return list(PALETTE[:n])
    rng = np.random.default_rng(seed)
    extra = [f"#{int(rng.integers(0, 0xFFFFFF)):06x}" for _ in range(n - len(PALETTE))]
    return list(PALETTE) + extra


def _tick_step(maximum: float) -> int:
    """Tick spacing that keeps labels legible across the usual axis ranges."""
    if maximum <= 20:
        return 2
    if maximum <= 50:
        return 5
    return 10


def _new_axes(figsize: tuple[float, float] = (8, 6)) -> tuple[Figure, Any]:
    return plt.subplots(figsize=figsize)


def _finish(fig: Figure, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


def _draw_noise_floor(ax: Any, noise_floor: float | None) -> None:
    """Mark the resolution limit, where one is known."""
    if noise_floor is None or not np.isfinite(noise_floor) or noise_floor <= 0:
        return
    ax.axhline(
        noise_floor,
        linestyle="--",
        linewidth=1.0,
        color="0.45",
        label=f"noise floor ({noise_floor:.3f})",
    )


def save_matrix_data_and_plot(
    rep: np.ndarray,
    measure: str,
    ensemble_index: int,
    output_dir: str | None,
    verbose: bool = False,
) -> tuple[str, str]:
    """Write one ensemble's representation matrix as CSV, and a heatmap beside it."""
    out_dir = get_method_output_dir(output_dir, measure)
    csv_path = os.path.join(out_dir, f"ensemble_{ensemble_index}_matrix.csv")
    np.savetxt(csv_path, rep, delimiter=",")
    logger.info("Wrote %s", csv_path)

    fig, ax = _new_axes()
    image = ax.imshow(rep, aspect="auto", cmap="viridis")
    ax.set_xlabel("Residue / feature index")
    ax.set_ylabel("Frame index")
    ax.set_title(f"{measure.upper()} representation — ensemble {ensemble_index}")

    bar = fig.colorbar(image, ax=ax)
    low, high = float(np.floor(np.min(rep))), float(np.ceil(np.max(rep)))
    if high > low:
        step = _tick_step(high - low)
        bar.set_ticks(np.arange(low, high + step, step))
        bar.ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

    png_path = os.path.join(out_dir, f"ensemble_{ensemble_index}_matrix.png")
    return csv_path, _finish(fig, png_path)


def _global_values(comparisons: Sequence[Any]) -> tuple[list[int], list[float], float | None]:
    indices = [int(c["ensemble_index"]) for c in comparisons]
    values = [float(c["global_dissimilarity"]) for c in comparisons]
    floors = [
        float(c["noise_floor"])
        for c in comparisons
        if c.get("noise_floor") is not None
    ]
    return indices, values, (float(np.mean(floors)) if floors else None)


def plot_global_dissimilarity_bar(
    measure: str,
    comparisons: Sequence[Any],
    output_dir: str | None,
    verbose: bool = False,
) -> str:
    """Bar chart of global dissimilarity, coloured to match the other figures."""
    out_dir = get_method_output_dir(output_dir, measure)
    indices, values, floor = _global_values(comparisons)

    fig, ax = _new_axes()
    ax.bar(indices, values, color=get_ensemble_colors(len(indices)))
    _draw_noise_floor(ax, floor)
    ax.set_xlabel("Ensemble index")
    ax.set_ylabel("Global dissimilarity")
    ax.set_title(f"{measure.upper()} global dissimilarity vs reference")
    ax.set_xticks(indices)
    if floor:
        ax.legend(frameon=False, fontsize="small")
    return _finish(fig, os.path.join(out_dir, f"{measure}_global_dissimilarity_bar.png"))


def plot_global_dissimilarity_line(
    measure: str,
    comparisons: Sequence[Any],
    output_dir: str | None,
    verbose: bool = False,
    color: str = "k",
) -> str:
    """Line plot of global dissimilarity, for ensembles along an ordered axis."""
    out_dir = get_method_output_dir(output_dir, measure)
    indices, values, floor = _global_values(comparisons)

    fig, ax = _new_axes()
    ax.plot(indices, values, marker="o", linestyle="-", color=color)
    _draw_noise_floor(ax, floor)
    ax.set_xlabel("Ensemble index")
    ax.set_ylabel("Global dissimilarity")
    ax.set_title(f"{measure.upper()} global dissimilarity vs reference")
    ax.set_xticks(indices)
    if floor:
        ax.legend(frameon=False, fontsize="small")
    return _finish(fig, os.path.join(out_dir, f"{measure}_global_dissimilarity_line.png"))


def plot_local_dissimilarity(
    measure: str,
    ensemble_index: int,
    local_diss: np.ndarray,
    output_dir: str | None,
    verbose: bool = False,
    color: str = "k",
    raw_local_diss: np.ndarray | None = None,
    feature_index: np.ndarray | None = None,
) -> str:
    """Per-residue dissimilarity for one ensemble.

    Where the unmasked values are supplied they are drawn faintly underneath,
    so a residue that fell just short of significance is visibly different from
    one that did not move at all. The masked curve alone cannot distinguish
    them, and reading a flat zero as "no change" is the easy mistake.
    """
    out_dir = get_method_output_dir(output_dir, measure)
    # After reconciliation the columns are a subset of the reference's, so
    # plotting them at 1..n would renumber the protein and put the label of
    # one residue under the value of another.
    x = (
        np.arange(1, len(local_diss) + 1)
        if feature_index is None
        else np.asarray(feature_index)
    )

    fig, ax = _new_axes()
    if raw_local_diss is not None:
        ax.plot(
            x, raw_local_diss, linewidth=0.9, color="0.7",
            label="all features", zorder=1,
        )
    ax.plot(
        x, local_diss, marker="o", markersize=3.5, linestyle="-", color=color,
        label="significant only", zorder=2,
    )
    ax.set_xlabel("Residue / feature index")
    ax.set_ylabel("Local dissimilarity")
    ax.set_title(f"{measure.upper()} local dissimilarity — ensemble {ensemble_index}")
    ax.xaxis.set_major_locator(MultipleLocator(_tick_step(int(x[-1]))))
    if raw_local_diss is not None:
        ax.legend(frameon=False, fontsize="small")
    return _finish(
        fig,
        os.path.join(out_dir, f"{measure}_ensemble_{ensemble_index}_local_dissimilarity.png"),
    )


def plot_combined_local_dissimilarity(
    measure: str,
    comparisons: Sequence[Any],
    output_dir: str | None,
    verbose: bool = False,
) -> str:
    """Every ensemble's per-residue dissimilarity on one axes."""
    out_dir = get_method_output_dir(output_dir, measure)
    colors = get_ensemble_colors(len(comparisons))

    fig, ax = _new_axes()
    longest = 1
    for position, comparison in enumerate(comparisons):
        values = np.asarray(comparison["local_dissimilarity"])
        index = comparison.get("feature_index")
        x = np.arange(1, len(values) + 1) if index is None else np.asarray(index)
        longest = max(longest, int(x[-1]) if len(x) else 1)
        ax.plot(
            x, values, marker="o", markersize=3, linestyle="-",
            color=colors[position], label=f"ensemble {comparison['ensemble_index']}",
        )

    ax.set_xlabel("Residue / feature index (reference numbering)")
    ax.set_ylabel("Local dissimilarity")
    ax.set_title(f"{measure.upper()} local dissimilarity vs reference")
    ax.xaxis.set_major_locator(MultipleLocator(_tick_step(longest)))
    ax.legend(frameon=False, fontsize="small")
    return _finish(
        fig, os.path.join(out_dir, f"{measure}_combined_local_dissimilarity.png")
    )


def dimensionality_reduction_plot(
    reps: Sequence[np.ndarray],
    technique: str,
    output_dir: str,
    verbose: bool = False,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, Figure]:
    """Project every frame of every ensemble into two dimensions and plot them.

    Raises
    ------
    ValueError
        For an unknown technique, or for an MDS request over more frames than
        a dense distance matrix can hold. Refusing is better than starting a
        computation that will exhaust memory an hour later.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS, TSNE

    technique = technique.strip().lower()
    data = np.vstack(list(reps))
    labels = np.concatenate(
        [np.full(rep.shape[0], i) for i, rep in enumerate(reps)]
    )

    if technique == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    elif technique == "mds":
        if data.shape[0] > MDS_FRAME_LIMIT:
            raise ValueError(
                f"MDS over {data.shape[0]} frames needs a dense "
                f"{data.shape[0]}x{data.shape[0]} distance matrix "
                f"(~{data.shape[0] ** 2 * 8 / 1e9:.1f} GB). Subsample the "
                f"ensembles, or use pca or tsne, which have no such limit."
            )
        reducer = MDS(n_components=2, random_state=random_state, normalized_stress="auto")
    elif technique == "tsne":
        perplexity = min(30.0, max(5.0, (data.shape[0] - 1) / 3.0))
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    else:
        raise ValueError(
            f"Unknown dimensionality reduction technique {technique!r}. "
            f"Available: pca, mds, tsne."
        )

    logger.info("Running %s on %d frames x %d features", technique, *data.shape)
    reduced = reducer.fit_transform(data)

    colors = get_ensemble_colors(len(reps))
    fig, ax = _new_axes()
    for i in range(len(reps)):
        points = reduced[labels == i]
        ax.scatter(
            points[:, 0], points[:, 1], s=8, color=colors[i],
            label=f"ensemble {i}", alpha=0.6, edgecolors="none",
        )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"{technique.upper()} of ensemble representations")
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()

    fig.savefig(
        os.path.join(output_dir, f"dim_reduction_{technique}.png"),
        bbox_inches="tight", dpi=150,
    )
    np.savetxt(
        os.path.join(output_dir, f"dim_reduction_{technique}_data.csv"),
        reduced, delimiter=",",
    )
    np.savetxt(
        os.path.join(output_dir, f"dim_reduction_{technique}_labels.csv"),
        labels, delimiter=",", fmt="%d",
    )
    plt.close(fig)
    return reduced, labels, fig


def replot_global_dissimilarity(
    measure: str,
    results: Sequence[Any],
    plot_type: str = "line",
    **kwargs: Any,
) -> Figure:
    """Rebuild the global dissimilarity figure with custom styling.

    Returns the figure without writing it, so the caller decides whether and
    where it is saved. Every documented keyword takes effect.
    """
    indices, values, floor = _global_values(results)
    show_floor = kwargs.get("show_noise_floor", True)

    fig, ax = _new_axes(kwargs.get("figsize", (8, 6)))
    if plot_type == "bar":
        color = kwargs.get("color") or get_ensemble_colors(len(indices))
        ax.bar(indices, values, color=color)
    else:
        ax.plot(
            indices, values, marker=kwargs.get("marker", "o"),
            linestyle=kwargs.get("linestyle", "-"),
            color=kwargs.get("color") or "k",
        )

    if show_floor:
        _draw_noise_floor(ax, floor)
        if floor:
            ax.legend(frameon=False, fontsize="small")

    ax.set_xlabel(kwargs.get("xlabel", "Ensemble index"))
    ax.set_ylabel(kwargs.get("ylabel", "Global dissimilarity"))
    ax.set_title(
        kwargs.get("title", f"{measure.upper()} global dissimilarity vs reference")
    )
    ax.set_xticks(indices)
    fig.tight_layout()
    return fig


def replot_local_dissimilarity(
    measure: str,
    local_diss: np.ndarray,
    ensemble_index: int,
    **kwargs: Any,
) -> Figure:
    """Rebuild a per-residue dissimilarity figure with custom styling."""
    values = np.asarray(local_diss)
    x = np.arange(1, len(values) + 1)

    fig, ax = _new_axes(kwargs.get("figsize", (8, 6)))
    ax.plot(
        x, values, marker=kwargs.get("marker", "o"),
        linestyle=kwargs.get("linestyle", "-"),
        color=kwargs.get("color", "k"),
    )
    ax.set_xlabel(kwargs.get("xlabel", "Residue / feature index"))
    ax.set_ylabel(kwargs.get("ylabel", "Local dissimilarity"))
    ax.set_title(
        kwargs.get(
            "title",
            f"{measure.upper()} local dissimilarity — ensemble {ensemble_index}",
        )
    )
    ax.xaxis.set_major_locator(MultipleLocator(kwargs.get("tick_step", _tick_step(x[-1]))))
    fig.tight_layout()
    return fig
