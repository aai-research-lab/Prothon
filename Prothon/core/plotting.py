"""
Plotting module for the Prothon package.

Provides functions to:
  • Create or retrieve output directories for a given measure.
  • Save ensemble representation matrix data as CSV and generate heatmap plots—with the color bar ticks spaced
    at regular intervals (e.g. 2,4,6; 5,10,15; or 10,20,30).
  • Generate global dissimilarity plots in both bar and line formats.
      - The bar plot is color-coded using the fixed palette.
      - The line plot uses black by default.
  • Generate individual local dissimilarity plots with x-axis ticks at regular intervals.
  • Generate a combined local dissimilarity plot (all ensembles in one plot, each line in its assigned color).
  • Perform dimensionality reduction (PCA, MDS, t-SNE) on the ensemble representations and generate a 2D scatter plot.
    The scatter plot is saved in the same measure output directory.
  • Replot global and local plots with custom styling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MultipleLocator
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import random

def get_method_output_dir(output_dir: str, measure: str):
    """
    Get or create the output directory for a given measure.

    Parameters
    ----------
    output_dir : str or None
        Root output directory. If None, default is "<measure>_output".
    measure : str
        Measure name (e.g., 'cbcn').

    Returns
    -------
    out_dir : str
        Full path of the output directory.
    """
    if output_dir:
        out_dir = os.path.join(output_dir, f"{measure}_output")
    else:
        out_dir = f"{measure}_output"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def get_ensemble_colors(n: int):
    """
    Generate n distinct colors using a predefined order.
    
    Predefined order (first 12):
      red, gold, darkgreen, blue, darkorchid, lightcoral, orange, lime,
      deepskyblue, magenta, navy, cyan.
    For more than 12 ensembles, extra colors are generated randomly.
    
    Parameters
    ----------
    n : int

    Returns
    -------
    colors : list
        List of colors.
    """
    predefined = [
        "red", "gold", "darkgreen", "blue", "darkorchid", 
        "lightcoral", "orange", "lime", "deepskyblue", "magenta", "navy", "cyan"
    ]
    if n <= len(predefined):
        return predefined[:n]
    else:
        colors = predefined.copy()
        for _ in range(n - len(predefined)):
            colors.append("#%06x" % random.randint(0, 0xFFFFFF))
        return colors

def save_matrix_data_and_plot(rep: np.ndarray, measure: str, ensemble_index: int, output_dir: str, verbose: bool=False):
    """
    Save the ensemble representation matrix to a CSV and generate a heatmap.
    The color bar ticks are set at regular intervals based on the matrix range.

    Parameters
    ----------
    rep : np.ndarray
        The representation matrix.
    measure : str
        Measure name.
    ensemble_index : int
        Ensemble index.
    output_dir : str
        Root output directory.
    verbose : bool, optional
    """
    out_dir = get_method_output_dir(output_dir, measure)
    csv_file = os.path.join(out_dir, f"ensemble_{ensemble_index}_matrix.csv")
    np.savetxt(csv_file, rep, delimiter=",")
    if verbose:
        print(f"[Output] Saved matrix CSV to {csv_file}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(rep, aspect='auto', cmap='viridis')
    ax.set_xlabel("Residue/Feature Index")
    ax.set_ylabel("Frame Index")
    ax.set_title(f"{measure.upper()} Matrix Representation - Ensemble {ensemble_index}")
    cbar = fig.colorbar(im, ax=ax)
    vmin, vmax = np.floor(np.min(rep)), np.ceil(np.max(rep))
    r = vmax - vmin
    if r <= 10:
        step = 1
    elif r <= 50:
        step = 5
    else:
        step = 10
    ticks = np.arange(vmin, vmax + step, step)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
    plot_file = os.path.join(out_dir, f"ensemble_{ensemble_index}_matrix.png")
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"[Output] Saved matrix heatmap to {plot_file}")

def plot_global_dissimilarity_bar(measure: str, comparisons: list, output_dir: str, verbose: bool=False):
    """
    Generate a bar plot for global dissimilarity, using fixed colors that match the dimensionality reduction palette.

    Parameters
    ----------
    measure : str
    comparisons : list of dict
    output_dir : str
    verbose : bool, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    out_dir = get_method_output_dir(output_dir, measure)
    indices = [comp['ensemble_index'] for comp in comparisons]
    global_vals = [comp['global_dissimilarity'] for comp in comparisons]
    colors = get_ensemble_colors(len(indices))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(indices, global_vals, color=colors)
    ax.set_xlabel("Ensemble (Integer Index)")
    ax.set_ylabel("Global Dissimilarity")
    ax.set_title(f"{measure.upper()} Global Dissimilarity vs Reference")
    ax.set_xticks(indices)
    plt.tight_layout()
    plot_file = os.path.join(out_dir, f"{measure}_global_dissimilarity_bar.png")
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"[Output] Saved global dissimilarity bar plot to {plot_file}")
    return fig

def plot_global_dissimilarity_line(measure: str, comparisons: list, output_dir: str, verbose: bool=False, color: str = 'k'):
    """
    Generate a line plot for global dissimilarity (line plot is in black by default).

    Parameters
    ----------
    measure : str
    comparisons : list of dict
    output_dir : str
    verbose : bool, optional
    color : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    out_dir = get_method_output_dir(output_dir, measure)
    indices = [comp['ensemble_index'] for comp in comparisons]
    global_vals = [comp['global_dissimilarity'] for comp in comparisons]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(indices, global_vals, marker='o', linestyle='-', color=color)
    ax.set_xlabel("Ensemble (Integer Index)")
    ax.set_ylabel("Global Dissimilarity")
    ax.set_title(f"{measure.upper()} Global Dissimilarity vs Reference")
    ax.set_xticks(indices)
    plt.tight_layout()
    plot_file = os.path.join(out_dir, f"{measure}_global_dissimilarity_line.png")
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"[Output] Saved global dissimilarity line plot to {plot_file}")
    return fig

def plot_local_dissimilarity(measure: str, ensemble_index: int, local_diss: np.ndarray, output_dir: str, verbose: bool=False, color: str='k'):
    """
    Generate a local (per-feature) dissimilarity plot.
    The x-axis (residue index) starts at 1 and is marked at regular intervals (e.g. 2,4,6; 5,10,15; or 10,20,30).

    Parameters
    ----------
    measure : str
    ensemble_index : int
    local_diss : np.ndarray
    output_dir : str
    verbose : bool, optional
    color : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    out_dir = get_method_output_dir(output_dir, measure)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(1, len(local_diss)+1)
    ax.plot(x, local_diss, marker='o', linestyle='-', color=color)
    ax.set_xlabel("Residue / Feature Index")
    ax.set_ylabel("Local Dissimilarity")
    ax.set_title(f"{measure.upper()} Local Dissimilarity - Ensemble {ensemble_index}")
    # Set x-axis ticks based on maximum x value
    if x[-1] <= 20:
        step = 2
    elif x[-1] <= 50:
        step = 5
    else:
        step = 10
    ax.xaxis.set_major_locator(plt.MultipleLocator(step))
    plt.tight_layout()
    plot_file = os.path.join(out_dir, f"{measure}_ensemble_{ensemble_index}_local_dissimilarity.png")
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"[Output] Saved local dissimilarity plot for ensemble {ensemble_index} to {plot_file}")
    return fig

def plot_combined_local_dissimilarity(measure: str, comparisons: list, output_dir: str, verbose: bool=False):
    """
    Generate a combined local dissimilarity plot with one line per ensemble using distinct colors.

    Parameters
    ----------
    measure : str
    comparisons : list of dict
        Each dictionary must include 'ensemble_index' and 'local_dissimilarity'.
    output_dir : str
    verbose : bool, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    out_dir = get_method_output_dir(output_dir, measure)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = get_ensemble_colors(len(comparisons))
    for idx, comp in enumerate(comparisons):
        x = np.arange(1, len(comp['local_dissimilarity'])+1)
        ax.plot(x, comp['local_dissimilarity'], marker='o', linestyle='-', color=colors[idx], label=f"Ensemble {comp['ensemble_index']}")
    ax.set_xlabel("Residue / Feature Index")
    ax.set_ylabel("Local Dissimilarity")
    ax.set_title(f"{measure.upper()} Combined Local Dissimilarity vs Reference")
    # Set x-axis ticks based on maximum value
    if x[-1] <= 20:
        step = 2
    elif x[-1] <= 50:
        step = 5
    else:
        step = 10
    ax.xaxis.set_major_locator(plt.MultipleLocator(step))
    ax.legend()
    plt.tight_layout()
    plot_file = os.path.join(out_dir, f"{measure}_combined_local_dissimilarity.png")
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"[Output] Saved combined local dissimilarity plot to {plot_file}")
    return fig

def dimensionality_reduction_plot(reps: list, technique: str, output_dir: str, verbose: bool=False):
    """
    Perform dimensionality reduction (PCA, MDS, or t-SNE) on the concatenated ensemble representation matrices and
    generate a 2D scatter plot. The plot (and associated data) is saved in the same method output directory.

    Parameters
    ----------
    reps : list of np.ndarray
        List of ensemble representation matrices.
    technique : str, one of {'pca', 'mds', 'tsne'}
    output_dir : str
        The method-specific output directory (e.g., as returned by get_method_output_dir).
    verbose : bool, optional

    Returns
    -------
    reduced_data : np.ndarray, shape=(total_frames, 2)
    labels : np.ndarray, shape=(total_frames,)
    fig : matplotlib.figure.Figure
    """
    technique = technique.lower()
    data_list, labels_list = [], []
    for i, rep in enumerate(reps):
        data_list.append(rep)
        labels_list.append(np.full(rep.shape[0], i))
    data = np.vstack(data_list)
    labels = np.concatenate(labels_list)
    if technique == 'pca':
        reducer = PCA(n_components=2)
    elif technique == 'mds':
        reducer = MDS(n_components=2, random_state=42)
    elif technique == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unsupported technique: {technique}")
    reduced_data = reducer.fit_transform(data)
    colors = get_ensemble_colors(len(reps))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(reps)):
        pts = reduced_data[labels == i, :]
        ax.scatter(pts[:, 0], pts[:, 1], color=colors[i], label=f"Ensemble {i}", alpha=0.6)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"Dimensionality Reduction ({technique.upper()}) of Ensemble Representations")
    ax.legend()
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"dim_reduction_{technique}.png")
    fig.savefig(plot_file, bbox_inches='tight')
    if verbose:
        print(f"[Output] Saved dimensionality reduction plot ({technique.upper()}) to {plot_file}")
    plt.close(fig)
    np.savetxt(os.path.join(output_dir, f"dim_reduction_{technique}_data.csv"), reduced_data, delimiter=",")
    np.savetxt(os.path.join(output_dir, f"dim_reduction_{technique}_labels.csv"), labels, delimiter=",")
    if verbose:
        print(f"[Output] Saved dimensionality reduction data and labels to {output_dir}")
    return reduced_data, labels, fig

def replot_global_dissimilarity(measure: str, results: list, plot_type: str = 'line', **kwargs):
    """
    Replot the global dissimilarity figure with custom styling.

    Parameters
    ----------
    measure : str
    results : list of dict
    plot_type : str, one of {'line', 'bar'}
    Additional keyword arguments are passed to the replot functions.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if plot_type == 'bar':
        return plot_global_dissimilarity_bar(measure, results, kwargs.get('output_dir', None), verbose=kwargs.get('verbose', False))
    else:
        return plot_global_dissimilarity_line(measure, results, kwargs.get('output_dir', None), verbose=kwargs.get('verbose', False), color=kwargs.get('color', 'k'))

def replot_local_dissimilarity(measure: str, local_diss: np.ndarray, ensemble_index: int, **kwargs):
    """
    Replot the local dissimilarity figure with custom styling.

    Parameters
    ----------
    measure : str
    local_diss : np.ndarray
    ensemble_index : int
    Additional keyword arguments are passed to the function.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(1, len(local_diss)+1)
    ax.plot(x, local_diss, marker='o', linestyle='-', color=kwargs.get('color', 'k'))
    ax.set_xlabel(kwargs.get('xlabel', "Residue / Feature Index"))
    ax.set_ylabel(kwargs.get('ylabel', "Local Dissimilarity"))
    title = kwargs.get('title', f"{measure.upper()} Local Dissimilarity - Ensemble {ensemble_index}")
    ax.set_title(title)
    # Set x-axis ticks at regular multiples
    if x[-1] <= 20:
        step = 2
    elif x[-1] <= 50:
        step = 5
    else:
        step = 10
    ax.xaxis.set_major_locator(MultipleLocator(step))
    plt.tight_layout()
    return fig

