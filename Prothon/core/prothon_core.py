"""
Main Prothon class for the Prothon package.

This class provides an API for:
  • Computing ensemble representations using one or more local measures.
  • Saving ensemble matrix representation data and generating heatmap plots.
  • Computing global and local dissimilarity vs a reference ensemble.
  • Generating both bar and line plots for global dissimilarity; the bar plot is color-coded with a fixed palette,
    and the line plot is in black.
  • Generating individual and combined local dissimilarity plots (with the x-axis beginning at 1 and marked at regular intervals).
  • Performing dimensionality reduction (PCA, MDS, t-SNE) on ensemble data and saving 2D scatter plots in the same measure output directory.
  • Retrieving stored representation data, dissimilarity results, and dimensionality reduction results.
  • Replotting generated plots with custom styling.
"""

import os
import numpy as np
from .representation import compute_ensemble_representation
from .dissimilarity import dissimilarity
from .plotting import (save_matrix_data_and_plot, plot_global_dissimilarity_bar, plot_global_dissimilarity_line,
                       plot_local_dissimilarity, plot_combined_local_dissimilarity, dimensionality_reduction_plot,
                       replot_global_dissimilarity, replot_local_dissimilarity, get_method_output_dir)


class Prothon:
    """
    Main class for representing and comparing protein ensembles.

    Parameters
    ----------
    traj_files : list or str
        List or comma-separated string of trajectory filenames.
    topology : str
        Path to the topology (PDB) file.
    output_dir : str, optional
        Root directory for output files. If not specified, each measure will create its own directory (e.g., cbcn_output).
    verbose : bool, optional
        If True, prints detailed progress information.
    """
    def __init__(self, traj_files, topology, output_dir=None, verbose=False):
        if isinstance(traj_files, str):
            self.traj_files = [traj.strip() for traj in traj_files.split(",")]
        else:
            self.traj_files = traj_files
        self.topology = topology
        self.output_dir = output_dir
        self.verbose = verbose
        self.ensembles_data = {}      # key: measure, value: list of matrices
        self.comparison_results = {}  # key: measure, value: list of comparison results
        self.dimred_results = {}       # key: measure, value: dict of {technique: (reduced_data, labels, fig)}

    def compute_ensemble_representation(self, measure: str):
        """
        Compute and store ensemble representations using the specified measure.

        Parameters
        ----------
        measure : str
            One of 'cbcn', 'cacn', 'caba', 'cata', or 'sasa'.

        Returns
        -------
        reps : list of np.ndarray
            Computed representation matrices.
        """
        measure = measure.lower()
        if self.verbose:
            print(f"[Prothon] Computing ensemble representation using {measure.upper()}.")
        reps = compute_ensemble_representation(self.traj_files, self.topology, measure, self.verbose)
        self.ensembles_data[measure] = reps
        return reps

    def compare_ensembles(self, methods='cbcn', ref: int = 0, x_num: int = 100, s_num: int = 5, dimred=None):
        """
        For one or more measures, compute representations, save matrix data and plots,
        compute dissimilarity (global and local) vs the reference ensemble,
        generate individual and combined local dissimilarity plots, and perform dimensionality reduction.

        Parameters
        ----------
        methods : str or list of str
            Comma-separated string or list of measures (e.g., "cbcn,sasa").
        ref : int, optional
            Reference ensemble index (default: 0).
        x_num : int, optional
            Number of discretization points for KDE.
        s_num : int, optional
            Number of random samples for statistical testing.
        dimred : str or list of str or None, optional
            Dimensionality reduction techniques to use. Acceptable values are "pca", "mds", "tsne".
            Can be provided as a comma-separated string or list. Default is all three.
            If set to None, dimensionality reduction is skipped.

        Returns
        -------
        overall_results : dict
            Dictionary with keys as measure names and values as lists of comparison results.
        """
        if isinstance(methods, str):
            methods = [m.strip() for m in methods.split(",")]
        overall_results = {}
        for measure in methods:
            if self.verbose:
                print(f"\n[Prothon] Processing measure: {measure.upper()}")
            reps = self.compute_ensemble_representation(measure)
            # Save each ensemble’s matrix representation and heatmap.
            for i, rep in enumerate(reps):
                save_matrix_data_and_plot(rep, measure, i, self.output_dir, self.verbose)
            overall_min = min(np.min(r) for r in reps)
            overall_max = max(np.max(r) for r in reps)
            if self.verbose:
                print(f"[Prothon] KDE range for {measure.upper()}: min={overall_min}, max={overall_max}")
            comparisons = []
            ref_rep = reps[ref]
            for i, rep in enumerate(reps):
                if i == ref:
                    continue
                global_d, local_d, p_val = dissimilarity(ref_rep, rep, overall_min, overall_max, x_num, s_num)
                comparisons.append({
                    'ensemble_index': i,
                    'global_dissimilarity': global_d,
                    'local_dissimilarity': local_d,
                    'p_value': p_val
                })
                # Generate individual local dissimilarity plot (line plot in black).
                plot_local_dissimilarity(measure, i, local_d, self.output_dir, self.verbose, color='k')
            # Generate combined local dissimilarity plot.
            plot_combined_local_dissimilarity(measure, comparisons, self.output_dir, self.verbose)
            # Generate global dissimilarity plots (both bar and line)
            plot_global_dissimilarity_bar(measure, comparisons, self.output_dir, self.verbose)
            plot_global_dissimilarity_line(measure, comparisons, self.output_dir, self.verbose, color='k')
            overall_results[measure] = comparisons
            self.comparison_results[measure] = comparisons

            # Dimensionality reduction:
            if dimred is not None:
                if isinstance(dimred, str):
                    techniques = [tech.strip().lower() for tech in dimred.split(",")]
                else:
                    techniques = [tech.lower() for tech in dimred]
                # Use the measure-specific output directory for saving these plots.
                method_out_dir = get_method_output_dir(self.output_dir, measure)
                dimred_dict = {}
                for tech in techniques:
                    rd_data, labels, fig = dimensionality_reduction_plot(reps, tech, method_out_dir, self.verbose)
                    dimred_dict[tech] = {"reduced_data": rd_data, "labels": labels, "figure": fig}
                self.dimred_results[measure] = dimred_dict

        return overall_results

    def get_representation_data(self, measure: str):
        """
        Retrieve stored ensemble representation matrices for a given measure.

        Parameters
        ----------
        measure : str

        Returns
        -------
        list of np.ndarray or None
        """
        return self.ensembles_data.get(measure.lower(), None)

    def get_comparison_results(self, measure: str):
        """
        Retrieve dissimilarity comparison results for a given measure.

        Parameters
        ----------
        measure : str

        Returns
        -------
        Comparison results (list of dictionaries) if available.
        """
        return self.comparison_results.get(measure.lower(), None)

    def get_dimred_results(self, measure: str):
        """
        Retrieve the dimensionality reduction results for a given measure.

        Parameters
        ----------
        measure : str

        Returns
        -------
        dict or None
            Dictionary keyed by technique (e.g., "pca", "mds", "tsne") with reduced data, labels, and figure.
        """
        return self.dimred_results.get(measure.lower(), None)

    def replot_global_dissimilarity(self, measure: str, plot_type: str = 'line', **kwargs):
        """
        Replot global dissimilarity with custom styling.

        Parameters
        ----------
        measure : str
        plot_type : str, one of {'line', 'bar'}
        Additional keyword arguments are passed to the replot function.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from .plotting import replot_global_dissimilarity
        results = self.get_comparison_results(measure)
        if results is None:
            raise ValueError(f"No comparison results for measure {measure}.")
        return replot_global_dissimilarity(measure, results, plot_type=plot_type, **kwargs)

    def replot_local_dissimilarity(self, measure: str, ensemble_index: int, **kwargs):
        """
        Replot local dissimilarity for a specified ensemble with custom styling.

        Parameters
        ----------
        measure : str
        ensemble_index : int
        Additional keyword arguments are passed to the replot function.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from .plotting import replot_local_dissimilarity
        results = self.get_comparison_results(measure)
        if results is None:
            raise ValueError(f"No comparison results for measure {measure}.")
        comp = next((item for item in results if item['ensemble_index'] == ensemble_index), None)
        if comp is None:
            raise ValueError(f"No dissimilarity data for ensemble {ensemble_index} using measure {measure}.")
        local_diss = comp['local_dissimilarity']
        return replot_local_dissimilarity(measure, local_diss, ensemble_index, **kwargs)

