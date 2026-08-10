"""The study object: representations, comparisons, figures and a manifest.

:class:`Prothon` holds the inputs to one comparison study and runs it. The
public method names and their arguments are those of version 2.0, so existing
scripts keep working; what is new is that every run also writes a
``manifest.json`` recording what was compared, with which settings, under which
version of the code. Version 2.0 wrote figures and CSVs with no record of the
parameters that produced them, which is enough to reproduce a picture and not
enough to reproduce a result.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..utils import configure_logging, get_logger, split_list_arg
from .dissimilarity import ComparisonResult, dissimilarity
from .plotting import (
    dimensionality_reduction_plot,
    get_method_output_dir,
    plot_combined_local_dissimilarity,
    plot_global_dissimilarity_bar,
    plot_global_dissimilarity_line,
    plot_local_dissimilarity,
    replot_global_dissimilarity,
    replot_local_dissimilarity,
    save_matrix_data_and_plot,
)
from .representation import (
    MEASURES,
    compute_ensemble_representation,
    compute_representation,
    resolve_measure,
)

logger = get_logger("core")

__all__ = ["Prothon"]

#: Techniques offered by ``dimred``. Declared once so the CLI, the API and the
#: validator agree on the list.
DIMRED_TECHNIQUES = ("pca", "mds", "tsne")


class Prothon:
    """A comparison study over two or more conformational ensembles.

    Parameters
    ----------
    traj_files
        Trajectory filenames, one per ensemble, as a list or a comma-separated
        string. Each file is one ensemble: they are never concatenated.
    topology
        Topology file (PDB) shared by every trajectory.
    output_dir
        Root for the output tree. Each measure gets ``<output_dir>/<measure>_output``.
        When omitted, those directories are created in the working directory.
    verbose
        Raise the logging level to DEBUG.
    random_state
        Seed for the resampling that produces the noise floor and the p-values.
        Set it, and a rerun of the same study gives the same numbers.

    Examples
    --------
    >>> study = Prothon(["a.dcd", "b.dcd"], "top.pdb", random_state=0)
    >>> results = study.compare_ensembles(methods="cbcn")      # doctest: +SKIP
    >>> results["cbcn"][0].resolved                            # doctest: +SKIP
    True
    """

    @classmethod
    def from_ensembles(
        cls,
        ensembles: Sequence[Any],
        output_dir: str | None = None,
        verbose: bool = False,
        random_state: int | None = None,
    ) -> Prothon:
        """A study over :class:`~prothon.ingest.Ensemble` objects.

        The route that allows ensembles which are *not the same molecule*. A
        study built from filenames shares one topology and so compares one
        protein against itself under different conditions; ensembles carry
        their own topologies, so a mutant, a truncated construct or a
        coarse-grained model can be compared against a reference, and the
        residue correspondence is worked out rather than assumed.

        Examples
        --------
        >>> from prothon.ingest import Ensemble
        >>> wt = Ensemble.from_trajectory("wt.xtc", "wt.pdb", label="wild type")
        >>> mut = Ensemble.from_trajectory("mut.xtc", "mut.pdb", label="R42A")
        >>> study = Prothon.from_ensembles([wt, mut], random_state=0)   # doctest: +SKIP
        """
        from ..ingest import Ensemble

        items = list(ensembles)
        if len(items) < 2:
            raise ValueError(
                f"A comparison needs at least two ensembles; {len(items)} given."
            )
        wrong = [e for e in items if not isinstance(e, Ensemble)]
        if wrong:
            raise TypeError(
                "from_ensembles takes Ensemble objects. Use Prothon(paths, topology) "
                "for filenames, or Ensemble.from_trajectory to build them."
            )

        study = cls.__new__(cls)
        study.ensembles = items
        study.traj_files = [e.label for e in items]
        study.topology = None
        study.output_dir = output_dir
        study.verbose = verbose
        study.random_state = random_state
        study.ensembles_data = {}
        study.comparison_results = {}
        study.dimred_results = {}
        study.correspondences = {}
        configure_logging(verbose)

        return study

    def __init__(
        self,
        traj_files: str | Sequence[str],
        topology: str,
        output_dir: str | None = None,
        verbose: bool = False,
        random_state: int | None = None,
    ) -> None:
        files = split_list_arg(traj_files) if isinstance(traj_files, str) else list(traj_files)
        if len(files) < 2:
            raise ValueError(
                f"A comparison needs at least two ensembles; {len(files)} given. "
                f"Pass one trajectory file per ensemble."
            )

        missing = [path for path in files if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                "Trajectory file(s) not found: " + ", ".join(missing)
            )
        if not os.path.exists(topology):
            raise FileNotFoundError(f"Topology file not found: {topology}")

        self.traj_files = files
        self.topology = topology
        self.output_dir = output_dir
        self.verbose = verbose
        self.random_state = random_state

        self.ensembles: list[Any] | None = None
        self.ensembles_data: dict[str, list[np.ndarray]] = {}
        self.comparison_results: dict[str, list[ComparisonResult]] = {}
        self.dimred_results: dict[str, dict[str, dict[str, Any]]] = {}
        self.correspondences: dict[int, Any] = {}

        configure_logging(verbose)

    # -- representation ---------------------------------------------------

    def compute_ensemble_representation(self, measure: str) -> list[np.ndarray]:
        """Compute and cache the representation matrices for one measure."""
        spec = resolve_measure(measure)
        logger.info("Computing %s representation", spec.name.upper())
        if self.ensembles is not None:
            reps = [
                compute_representation(e.trajectory, spec.name) for e in self.ensembles
            ]
        else:
            reps = compute_ensemble_representation(
                self.traj_files, self.topology, spec.name, self.verbose
            )
        self.ensembles_data[spec.name] = reps
        return reps

    # -- comparison -------------------------------------------------------

    def compare_ensembles(
        self,
        methods: str | Sequence[str] = "cbcn",
        ref: int = 0,
        x_num: int = 100,
        s_num: int = 5,
        dimred: str | Sequence[str] | None = None,
        alpha: float = 0.05,
        legacy: bool = False,
        metric: str = "jsd",
    ) -> dict[str, list[ComparisonResult]]:
        """Run the study: represent, compare, plot, and record.

        Parameters
        ----------
        methods
            Measures to use, as a list or comma-separated string.
        ref
            Index of the reference ensemble, into ``traj_files``.
        x_num
            Grid points per estimated density.
        s_num
            Resamples per ensemble for the noise floor and significance test.
        dimred
            Techniques to project with, or ``None`` to skip. Skipping is the
            default because MDS over a long trajectory is expensive and the
            projection is a visualisation rather than part of the measurement.
        alpha
            False-discovery rate for the per-residue test.
        legacy
            Reproduce version 2.0's statistics exactly.
        metric
            Per-feature distance: ``jsd`` (default), ``wasserstein`` or ``ks``.
            The permutation null, the false-discovery correction and the noise
            floor are computed under whichever is chosen, so a Wasserstein
            comparison gets a Wasserstein floor rather than a borrowed one.

        Returns
        -------
        dict
            Measure name to the list of :class:`ComparisonResult`, one per
            non-reference ensemble.
        """
        requested = split_list_arg(methods)
        if not requested:
            raise ValueError("No measures requested.")
        specs = [resolve_measure(name) for name in requested]

        if not 0 <= ref < len(self.traj_files):
            raise ValueError(
                f"Reference index {ref} is out of range for {len(self.traj_files)} "
                f"ensembles (valid: 0 to {len(self.traj_files) - 1})."
            )

        techniques = [t.lower() for t in split_list_arg(dimred)]
        unknown = [t for t in techniques if t not in DIMRED_TECHNIQUES]
        if unknown:
            raise ValueError(
                f"Unknown dimensionality reduction technique(s): {', '.join(unknown)}. "
                f"Available: {', '.join(DIMRED_TECHNIQUES)}."
            )

        overall: dict[str, list[ComparisonResult]] = {}

        for spec in specs:
            logger.info("Processing measure %s", spec.name.upper())
            reps = self.compute_ensemble_representation(spec.name)

            for index, rep in enumerate(reps):
                save_matrix_data_and_plot(
                    rep, spec.name, index, self.output_dir, self.verbose
                )

            grid_min = float(min(np.min(rep) for rep in reps))
            grid_max = float(max(np.max(rep) for rep in reps))
            logger.info(
                "%s grid: [%.4g, %.4g]%s",
                spec.name.upper(), grid_min, grid_max,
                " (circular: overridden to [-pi, pi])" if spec.circular else "",
            )

            reference = reps[ref]
            comparisons: list[ComparisonResult] = []

            for index, rep in enumerate(reps):
                if index == ref:
                    continue

                left, right, feature_index = self._align_columns(
                    reference, rep, ref, index, spec.name
                )
                result = dissimilarity(
                    left, right, grid_min, grid_max,
                    x_num=x_num, s_num=s_num,
                    circular=spec.circular,
                    weights_ref=(
                        None if self.ensembles is None else self.ensembles[ref].weights
                    ),
                    weights=(
                        None if self.ensembles is None else self.ensembles[index].weights
                    ),
                    metric=metric,
                    alpha=alpha,
                    random_state=self.random_state,
                    legacy=legacy,
                    ensemble_index=index,
                    reference_index=ref,
                    measure=spec.name,
                )
                result.feature_index = feature_index
                comparisons.append(result)
                plot_local_dissimilarity(
                    spec.name, index, result.local_dissimilarity,
                    self.output_dir, self.verbose, color="k",
                    raw_local_diss=result.raw_local_dissimilarity,
                    feature_index=feature_index,
                )

            plot_combined_local_dissimilarity(
                spec.name, comparisons, self.output_dir, self.verbose
            )
            plot_global_dissimilarity_bar(
                spec.name, comparisons, self.output_dir, self.verbose
            )
            plot_global_dissimilarity_line(
                spec.name, comparisons, self.output_dir, self.verbose, color="k"
            )

            overall[spec.name] = comparisons
            self.comparison_results[spec.name] = comparisons

            if techniques:
                method_dir = get_method_output_dir(self.output_dir, spec.name)
                projections: dict[str, dict[str, Any]] = {}
                for technique in techniques:
                    try:
                        reduced, labels, figure = dimensionality_reduction_plot(
                            reps, technique, method_dir, self.verbose
                        )
                    except ValueError as error:
                        # A refused MDS should not lose the comparison that
                        # already succeeded; it is a visualisation.
                        logger.warning("Skipping %s: %s", technique, error)
                        continue
                    projections[technique] = {
                        "reduced_data": reduced, "labels": labels, "figure": figure
                    }
                self.dimred_results[spec.name] = projections

            self._write_manifest(
                spec.name, comparisons, x_num, s_num, alpha, legacy, metric
            )

        return overall

    def _align_columns(self, reference, other, ref_index, other_index, measure):
        """Reduce two representations to columns describing the same residues.

        For a study built from filenames there is one topology, so the columns
        already correspond and this returns them untouched. For a study built
        from ensembles the molecules may differ, and the correspondence decides
        which columns survive.

        Returns the two matrices and the position of each surviving feature on
        the *reference* ensemble, one-based -- which is what the per-residue
        figures are indexed by. Numbering them 1..n after reconciliation would
        put the label of one residue under the value of another, and the plot
        would look entirely reasonable.
        """
        if self.ensembles is None:
            return reference, other, None

        from ..ingest import feature_residues, reconcile

        key = (ref_index, other_index)
        correspondence = self.correspondences.get(key)
        if correspondence is None:
            correspondence = reconcile(
                self.ensembles[ref_index], self.ensembles[other_index]
            )
            self.correspondences[key] = correspondence

        reference_topology = self.ensembles[ref_index].topology
        other_topology = self.ensembles[other_index].topology

        if correspondence.is_identical and reference.shape[1] == other.shape[1]:
            return reference, other, None

        take_ref, take_other = correspondence.columns_for(
            measure, reference_topology, other_topology
        )
        if take_ref.size == 0:
            raise ValueError(
                f"No {measure} feature of {self.ensembles[ref_index].label} has a "
                f"counterpart in {self.ensembles[other_index].label}, even though "
                f"{correspondence.n_aligned} residues correspond. The measure is "
                f"defined on windows that the differences between these molecules "
                f"break; try a per-residue measure such as cacn or sasa."
            )

        dropped = reference.shape[1] - take_ref.size
        if dropped:
            logger.info(
                "%s vs %s: %d of %d %s columns comparable (%d dropped)",
                self.ensembles[ref_index].label,
                self.ensembles[other_index].label,
                take_ref.size, reference.shape[1], measure, dropped,
            )

        windows = feature_residues(reference_topology, measure)
        # A window measure spans several residues; label it by its first.
        index = np.array([windows[i][0] + 1 for i in take_ref], dtype=int)
        return reference[:, take_ref], other[:, take_other], index

    # -- manifest ---------------------------------------------------------

    def _write_manifest(
        self,
        measure: str,
        comparisons: Sequence[ComparisonResult],
        x_num: int,
        s_num: int,
        alpha: float,
        legacy: bool,
        metric: str = "jsd",
    ) -> str:
        """Record what was run, so the result can be reproduced rather than
        merely admired."""
        from .. import __version__

        out_dir = get_method_output_dir(self.output_dir, measure)
        path = os.path.join(out_dir, "manifest.json")
        payload = {
            "prothon_version": __version__,
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "measure": measure,
            "measure_description": MEASURES[measure].description,
            "circular": MEASURES[measure].circular,
            # A study over Ensemble objects has no single shared topology --
            # that is the point of it. Provenance for those lives on each
            # ensemble instead, under "ensembles" below.
            "topology": os.path.abspath(self.topology) if self.topology else None,
            "trajectories": (
                None
                if self.ensembles is not None
                else [os.path.abspath(p) for p in self.traj_files]
            ),
            "reference_index": comparisons[0].reference_index if comparisons else None,
            "ensembles": (
                [e.to_dict() for e in self.ensembles] if self.ensembles else None
            ),
            "correspondences": [
                {
                    "reference_index": a,
                    "ensemble_index": b,
                    "n_aligned": c.n_aligned,
                    "identity": c.identity,
                    "coverage": c.coverage,
                    "substitutions": [str(x) for x in c.substitutions],
                    "unmatched_reference": c.unmatched_a.tolist(),
                    "unmatched_other": c.unmatched_b.tolist(),
                    "alignment": [
                        {"reference": al.gapped_a, "other": al.gapped_b}
                        for al in c.alignments
                    ],
                }
                for (a, b), c in sorted(self.correspondences.items())
            ] or None,
            "parameters": {
                "x_num": x_num,
                "s_num": s_num,
                "alpha": alpha,
                "metric": metric,
                "legacy_statistics": legacy,
                "random_state": self.random_state,
            },
            "results": [c.to_dict() for c in comparisons],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Wrote %s", path)
        return path

    # -- accessors --------------------------------------------------------

    def get_representation_data(self, measure: str) -> list[np.ndarray] | None:
        """Cached representation matrices for a measure, or ``None``."""
        return self.ensembles_data.get(measure.strip().lower())

    def get_comparison_results(self, measure: str) -> list[ComparisonResult] | None:
        """Comparison results for a measure, or ``None``."""
        return self.comparison_results.get(measure.strip().lower())

    def get_dimred_results(self, measure: str) -> dict[str, dict[str, Any]] | None:
        """Projections for a measure, keyed by technique, or ``None``."""
        return self.dimred_results.get(measure.strip().lower())

    def summary(self) -> str:
        """A short human-readable account of what was found.

        States the noise floor beside every dissimilarity, and says plainly
        where a difference is smaller than the sampling can resolve.
        """
        if not self.comparison_results:
            return "No comparisons have been run yet."

        lines: list[str] = []
        for measure, comparisons in self.comparison_results.items():
            lines.append(f"{measure.upper()} (reference: ensemble {comparisons[0].reference_index})")
            for c in comparisons:
                verdict = (
                    f"{c.n_significant}/{c.local_dissimilarity.size} residues differ"
                    if c.resolved
                    else "not resolvable at this sampling"
                )
                lines.append(
                    f"  ensemble {c.ensemble_index}: "
                    f"d = {c.global_dissimilarity:.4f} "
                    f"(floor {c.noise_floor:.4f}) — {verdict}"
                )
            lines.append("")
        return "\n".join(lines).rstrip()

    # -- replotting -------------------------------------------------------

    def replot_global_dissimilarity(self, measure: str, plot_type: str = "line", **kwargs):
        """Rebuild the global dissimilarity figure with custom styling.

        Returns the figure without writing it; the original saved figure is
        left alone.
        """
        results = self.get_comparison_results(measure)
        if results is None:
            raise ValueError(
                f"No comparison results for {measure!r}. Run compare_ensembles first."
            )
        return replot_global_dissimilarity(measure, results, plot_type=plot_type, **kwargs)

    def replot_local_dissimilarity(self, measure: str, ensemble_index: int, **kwargs):
        """Rebuild a per-residue figure with custom styling.

        Pass ``raw=True`` to plot the unmasked values.
        """
        results = self.get_comparison_results(measure)
        if results is None:
            raise ValueError(
                f"No comparison results for {measure!r}. Run compare_ensembles first."
            )
        match = next(
            (r for r in results if r.ensemble_index == ensemble_index), None
        )
        if match is None:
            available = ", ".join(str(r.ensemble_index) for r in results)
            raise ValueError(
                f"No result for ensemble {ensemble_index} under {measure!r}. "
                f"Available: {available}."
            )
        values = (
            match.raw_local_dissimilarity
            if kwargs.pop("raw", False)
            else match.local_dissimilarity
        )
        return replot_local_dissimilarity(measure, values, ensemble_index, **kwargs)
