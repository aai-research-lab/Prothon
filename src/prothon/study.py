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
import warnings
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .compare.dissimilarity import ComparisonResult, dissimilarity
from .plot.figures import (
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
from .represent.order_parameters import (
    ORDER_PARAMETERS,
    compute_representation,
    resolve_order_parameter,
)
from .utils import configure_logging, get_logger, split_list_arg

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
        ensembles,
        output_dir: str | None = None,
        verbose: bool = False,
        random_state: int | None = None,
    ) -> Prothon:
        """The 2.x name for ``Prothon(ensembles=...)``.

        Kept because it appears in published scripts. The constructor now
        takes sources and loaded ensembles alike, so there is one way in.
        """
        return cls(
            ensembles=ensembles, output_dir=output_dir,
            verbose=verbose, random_state=random_state,
        )

    def __init__(
        self,
        ensembles=None,
        topology=None,
        order_parameters: str | list[str] | None = None,
        output_dir: str | None = None,
        verbose: bool = False,
        random_state: int | None = None,
        cache_dir: str | None = None,
        chains=None,
        study=None,
        *,
        traj_files=None,
    ) -> None:
        """Set up a comparison.

        Parameters
        ----------
        order_parameters
            The order parameter, or several, this study is about. Set here it
            becomes the default for every method, so ``study.compare()`` needs
            no argument. Any method still takes one to override it for a single
            call. Defaults to ``"cbcn"``.
        ensembles
            Two or more sources, or already-loaded
            :class:`~prothon.ingest.Ensemble` objects, or a mixture. A source
            is a trajectory, a directory of structures, a glob, a multi-model
            PDB, or a PED accession -- see
            :func:`~prothon.ingest.sources.resolve`.
        topology
            One topology shared by every source, or a list with one per
            source in the same order. A list is what comparison across
            different molecules needs, since a mutant has its own topology;
            ``None`` in a list means that source carries its own, which a PED
            accession and a multi-model PDB both do.

            Not required at all when every ensemble is already loaded.
        chains
            Keep only these chains, as a letter, an index, or several of
            either -- one selection shared by every ensemble, or one per
            ensemble. A complex is often compared one chain at a time, since
            the rest of the system is a different molecule.
        random_state
            Seed for the resampling behind the noise floor and the p-values.
            Set it and a rerun gives the same numbers.
        study
            The :class:`~prothon.config.Study` this run came from, if any.
            Recorded in the manifest, so a result found later carries the
            question it answered rather than only the answer.
        traj_files
            The name this argument had in 2.x. Accepted, and warns.
        """
        from .ingest.sources import resolve_all

        if traj_files is not None:
            if ensembles is not None:
                raise TypeError(
                    "Give either ensembles= or traj_files=, not both. "
                    "traj_files is the old name for the same thing."
                )
            warnings.warn(
                "traj_files= is now ensembles=, and takes any source rather "
                "than only trajectory files. The old name will be removed in "
                "4.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            ensembles = traj_files

        if ensembles is None:
            raise TypeError("Prothon needs ensembles= : two or more sources.")

        loaded = resolve_all(ensembles, topology, cache_dir=cache_dir, chains=chains)
        if len(loaded) < 2:
            raise ValueError(
                f"A comparison needs at least two ensembles; {len(loaded)} "
                f"given. Each source is one ensemble, and they are never "
                f"concatenated."
            )

        self.ensembles = loaded
        self.traj_files = [e.label for e in loaded]
        # A list of topologies belongs to the ensembles rather than to the
        # study, and each one records its own in its provenance.
        self.topology = topology if isinstance(topology, (str, type(None))) else None
        self.output_dir = output_dir
        self.verbose = verbose
        self.random_state = random_state
        # Private, because `Prothon.order_parameters()` is the registry and an
        # instance attribute of that name would shadow it: `study.order_
        # parameters()` would raise "str object is not callable".
        self._order_parameters = order_parameters
        self.study = study

        self.ensembles_data: dict[str, list[np.ndarray]] = {}
        self.comparison_results: dict[str, list[ComparisonResult]] = {}
        self.dimred_results: dict[str, dict[str, dict[str, Any]]] = {}
        self.correspondences: dict[tuple[int, int], Any] = {}
        self.distinguishability_results: dict[str, dict[str, list]] = {}
        self.coverage_results: dict[str, list] = {}

        configure_logging(verbose)

    # -- one import -------------------------------------------------------
    #
    # `from prothon import Prothon` should be the whole of what a user needs
    # to import. Everything else is reachable from the class or from an
    # instance: the registries as attributes, the other ways of starting a
    # study as classmethods, and the analyses as methods. Anybody who wants
    # the underlying functions can still import them; nobody has to.

    @classmethod
    def from_config(cls, path, **overrides) -> Prothon:
        """Start from a study written in a file.

        >>> Prothon.from_config("study.yml")          # doctest: +SKIP
        """
        from .config.study import Study

        study = Study.from_file(path)
        study.settings.update(overrides)
        return study.run()

    @staticmethod
    def load(source, topology=None, label=None, **kwargs):
        """Load one ensemble from any source, without starting a study.

        A trajectory, a directory of structures, a glob, a multi-model PDB, or
        a PED accession. Rarely needed -- the constructor takes sources
        directly -- but useful when an ensemble needs adjusting before it is
        compared.

        >>> Prothon.load("PED00024")                  # doctest: +SKIP
        """
        from .ingest.sources import resolve

        return resolve(source, topology=topology, label=label, **kwargs)

    @staticmethod
    def order_parameters() -> dict:
        """Every local order parameter, by name."""
        from .represent.order_parameters import ORDER_PARAMETERS

        return dict(ORDER_PARAMETERS)

    @staticmethod
    def metrics() -> dict:
        """Every per-residue distance, by name."""
        from .compare.distance import METRICS

        return dict(METRICS)

    @staticmethod
    def observables() -> dict:
        """Every observable that can be computed and scored against
        measurements."""
        from .validate.observables import OBSERVABLES

        return dict(OBSERVABLES)

    # -- analyses ---------------------------------------------------------

    def _parameters(self, order_parameters=None):
        """Whichever was given: the call's, the study's, or the default."""
        if order_parameters is not None:
            return order_parameters
        return self._order_parameters or "cbcn"

    def compare(self, order_parameters=None, **kwargs):
        """Compare the ensembles.

        Uses the order parameters the study was built with unless given others
        for this call. A shorter name for :meth:`compare_ensembles`, which is
        what published scripts call.
        """
        return self.compare_ensembles(self._parameters(order_parameters), **kwargs)

    def rank(self, order_parameters=None, ref: int = 0, **kwargs):
        """Rank every other ensemble against the reference.

        The benchmark view: ordered by the margin above each ensemble's own
        noise floor rather than by raw distance, with coverage and fidelity
        beside each row.
        """
        from .batch.benchmark import benchmark

        others = [e for i, e in enumerate(self.ensembles) if i != ref]
        if not others:
            raise ValueError("Nothing to compare against the reference.")
        return benchmark(
            self.ensembles[ref], others,
            order_parameters=order_parameters,
            random_state=self.random_state,
            output_dir=self.output_dir,
            **kwargs,
        )

    def validate(
        self,
        observable: str,
        experimental,
        uncertainty,
        index: int = 0,
        **kwargs,
    ):
        """Score one ensemble against experimental measurements.

        Reported beside a floor obtained from the ensemble itself, because a
        perfect ensemble does not score a reduced chi-squared of one.

        Parameters
        ----------
        observable
            ``rg``, ``end_to_end`` or ``j_hn_ha``. Predictions from an
            external tool are scored with
            :func:`~prothon.validate.score.score_observable` directly.
        index
            Which ensemble to score.
        """
        import numpy as np

        from .validate.observables import (
            end_to_end,
            j_coupling_hn_ha,
            radius_of_gyration,
        )
        from .validate.score import score_observable

        compute = {
            "rg": lambda t: radius_of_gyration(t)[:, None],
            "end_to_end": lambda t: end_to_end(t)[:, None],
            "j_hn_ha": lambda t: j_coupling_hn_ha(t)[0],
        }
        if observable not in compute:
            raise ValueError(
                f"Unknown observable {observable!r}. Available: "
                f"{', '.join(sorted(compute))}. Predictions from an external "
                f"tool go to prothon.validate.score_observable directly."
            )

        ensemble = self.ensembles[index]
        return score_observable(
            compute[observable](ensemble.trajectory),
            np.atleast_1d(experimental),
            np.atleast_1d(uncertainty),
            observable=f"{observable} [{ensemble.label}]",
            weights=ensemble.weights,
            random_state=self.random_state,
            **kwargs,
        )

    def save_config(self, path: str = "prothon.yml") -> str:
        """Write this study to a file, so it can be re-run and committed.

        Defaults to ``prothon.yml`` in the working directory, because a study
        written beside the data it describes almost always wants the obvious
        name, and asking for one every time is a step that earns nothing.
        """
        from .config.study import WHERE, Study

        study = self.study or Study(
            ensembles=[
                {WHERE: e.provenance.get("path", e.label), "label": e.label}
                for e in self.ensembles
            ],
            output_dir=self.output_dir,
            settings=(
                {"random_state": self.random_state}
                if self.random_state is not None
                else {}
            ),
        )
        return study.save(path)

    @property
    def shares_topology(self) -> bool:
        """Whether every ensemble has the same number of atoms.

        When they do, representation columns already correspond and no
        reconciliation is needed. When they do not, a residue correspondence
        is built from a sequence alignment.
        """
        return len({e.trajectory.n_atoms for e in self.ensembles}) == 1

    # -- representation ---------------------------------------------------

    def compute_ensemble_representation(self, order_parameter: str) -> list[np.ndarray]:
        """Compute and cache the representation matrices for one parameter."""
        spec = resolve_order_parameter(order_parameter)
        logger.info("Computing %s representation", spec.name.upper())
        reps = [
            compute_representation(e.trajectory, spec.name) for e in self.ensembles
        ]
        self.ensembles_data[spec.name] = reps
        return reps

    # -- comparison -------------------------------------------------------

    def compare_ensembles(
        self,
        order_parameters: str | Sequence[str] | None = None,
        ref: int = 0,
        x_num: int = 100,
        s_num: int = 5,
        dimred: str | Sequence[str] | None = None,
        alpha: float = 0.05,
        legacy: bool = False,
        metric: str = "jsd",
        n_permutations: int = 100,
        block_permutation: bool | None = None,
        sample_size: int = 1000,
        correlation_time_frames: float | None = None,
        n_jobs: int = 1,
        *,
        measures=None,
        methods=None,
    ) -> dict[str, list[ComparisonResult]]:
        """Run the study: represent, compare, plot, and record.

        Parameters
        ----------
        order_parameters
            Which local order parameters to use, as a list or comma-separated
            string. ``measures=`` and ``methods=`` are accepted as the 2.x
            names and warn.
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
        order_parameters = self._parameters(order_parameters)
        for old, value in (("measures", measures), ("methods", methods)):
            if value is not None:
                warnings.warn(
                    f"{old}= is now order_parameters=. 'measure' collided with "
                    f"'metric', which means something else here. The old name "
                    f"will be removed in 4.0.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                order_parameters = value

        requested = split_list_arg(order_parameters)
        if not requested:
            raise ValueError("No order parameters requested.")
        specs = [resolve_order_parameter(name) for name in requested]

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
                    weights_ref=self.ensembles[ref].weights,
                    weights=self.ensembles[index].weights,
                    metric=metric,
                    n_permutations=n_permutations,
                    block_permutation=block_permutation,
                    sample_size=sample_size,
                    correlation_time_frames=correlation_time_frames,
                    n_jobs=n_jobs,
                    alpha=alpha,
                    random_state=self.random_state,
                    legacy=legacy,
                    ensemble_index=index,
                    reference_index=ref,
                    order_parameter=spec.name,
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
        if self.shares_topology and reference.shape[1] == other.shape[1]:
            return reference, other, None

        from .ingest import feature_residues, reconcile

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

    # -- whole-ensemble comparison ----------------------------------------

    def distinguishability(
        self,
        order_parameter: str | None = None,
        method: str = "c2st",
        ref: int = 0,
        random_state: int | None = None,
        **kwargs,
    ) -> list[Any]:
        """Ask whether each ensemble is distinguishable from the reference.

        The per-residue metrics compare each feature on its own, and so cannot
        see a difference that lives in the relationship between features -- two
        loops that visit the same positions but no longer at the same time give
        an identical profile at every residue and are a different ensemble.
        Both methods here read the joint distribution.

        Parameters
        ----------
        measure
            Which representation to compare on.
        method
            ``c2st`` (default) trains a classifier and reports how separable
            the ensembles are, with the features it leaned on. ``mmd`` runs a
            kernel two-sample test, which gives a calibrated p-value and no
            indication of where the difference is.
        ref
            Reference ensemble index.

        Returns
        -------
        list of EnsembleComparison
        """
        order_parameter = self._parameters(order_parameter)
        from .compare.joint import distinguishability as compare
        from .represent.order_parameters import resolve_order_parameter

        spec = resolve_order_parameter(order_parameter)
        reps = self.get_representation_data(spec.name)
        if reps is None:
            reps = self.compute_ensemble_representation(spec.name)
        if not 0 <= ref < len(reps):
            raise ValueError(f"Reference index {ref} is out of range.")

        results = []
        for index, rep in enumerate(reps):
            if index == ref:
                continue
            left, right, feature_index = self._align_columns(
                reps[ref], rep, ref, index, spec.name
            )
            extra = dict(kwargs)
            if method.strip().lower() == "c2st":
                extra["feature_index"] = feature_index
            result = compare(
                left, right, method,
                weights_a=self.ensembles[ref].weights,
                weights_b=self.ensembles[index].weights,
                circular=spec.circular,
                random_state=self.random_state if random_state is None else random_state,
                order_parameter=spec.name,
                **extra,
            )
            result.metadata["ensemble_index"] = index
            result.metadata["reference_index"] = ref
            results.append(result)
            logger.info("%s", result.summary().replace("\n", "; "))

        self.distinguishability_results.setdefault(spec.name, {})[method] = results
        return results

    def coverage_and_fidelity(
        self,
        order_parameter: str | None = None,
        ref: int = 0,
        **kwargs,
    ) -> list[Any]:
        """Split each difference into what is missed and what is invented.

        A single dissimilarity says two ensembles differ. A model that never
        opens a cryptic pocket and one that opens pockets no physics produces
        are both wrong, score alike on any symmetric distance, and need
        opposite work. This says which, at every residue.

        The reference is the ensemble being matched -- molecular dynamics, or
        an experimentally derived ensemble -- and the others are assessed
        against it. The two roles are not interchangeable.
        """
        order_parameter = self._parameters(order_parameter)
        from .compare.coverage import precision_recall
        from .represent.order_parameters import resolve_order_parameter

        spec = resolve_order_parameter(order_parameter)
        reps = self.get_representation_data(spec.name)
        if reps is None:
            reps = self.compute_ensemble_representation(spec.name)
        if not 0 <= ref < len(reps):
            raise ValueError(f"Reference index {ref} is out of range.")

        results = []
        for index, rep in enumerate(reps):
            if index == ref:
                continue
            left, right, feature_index = self._align_columns(
                reps[ref], rep, ref, index, spec.name
            )
            result = precision_recall(
                left, right,
                weights_ref=self.ensembles[ref].weights,
                weights=self.ensembles[index].weights,
                circular=spec.circular,
                feature_index=feature_index,
                random_state=self.random_state,
                order_parameter=spec.name,
                **kwargs,
            )
            result.metadata["ensemble_index"] = index
            result.metadata["reference_index"] = ref
            results.append(result)
            logger.info("ensemble %d: %s", index, result.summary().replace("\n", "; "))

        self.coverage_results.setdefault(spec.name, []).extend(results)
        return results

    # -- manifest ---------------------------------------------------------

    def _write_manifest(
        self,
        order_parameter: str,
        comparisons: Sequence[ComparisonResult],
        x_num: int,
        s_num: int,
        alpha: float,
        legacy: bool,
        metric: str = "jsd",
    ) -> str:
        """Record what was run, so the result can be reproduced rather than
        merely admired."""
        from . import __version__

        out_dir = get_method_output_dir(self.output_dir, order_parameter)
        path = os.path.join(out_dir, "manifest.json")
        payload = {
            "prothon_version": __version__,
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "order_parameter": order_parameter,
            "description": ORDER_PARAMETERS[order_parameter].description,
            "circular": ORDER_PARAMETERS[order_parameter].circular,
            # A study over Ensemble objects has no single shared topology --
            # that is the point of it. Provenance for those lives on each
            # ensemble instead, under "ensembles" below.
            "topology": os.path.abspath(self.topology) if self.topology else None,
            "reference_index": comparisons[0].reference_index if comparisons else None,
            "ensembles": [e.to_dict() for e in self.ensembles],
            "study": None if self.study is None else self.study.to_dict(),
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
            "distinguishability": {
                method: [r.to_dict() for r in items]
                for method, items in self.distinguishability_results.get(
                    order_parameter, {}
                ).items()
            }
            or None,
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
        for name, comparisons in self.comparison_results.items():
            spec = resolve_order_parameter(name)
            lines.append(f"{name.upper()} (reference: ensemble {comparisons[0].reference_index})")
            for c in comparisons:
                if not c.p_values_reported:
                    verdict = (
                        f"no p-value: correlation time {c.correlation_time:.0f} "
                        f"frames leaves {c.n_blocks} independent blocks"
                    )
                elif spec.is_global:
                    # One column describing the whole molecule. "1/1 residues
                    # differ" would be true and useless.
                    verdict = "differs" if c.n_significant else "no significant difference"
                elif c.resolved:
                    verdict = (
                        f"{c.n_significant}/{c.local_dissimilarity.size} residues differ"
                    )
                else:
                    verdict = "not resolvable at this sampling"
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
