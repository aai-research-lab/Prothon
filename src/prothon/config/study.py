"""A study, and the one object every interface builds.

A command line is a good way to ask a question once and a poor way to record
one. The flags that produced a figure live in a shell history that nobody
reads, on one machine, and the study cannot be re-run by someone else without
being reconstructed from the paper.

A configuration file is the study: a thing that can be committed beside the
manuscript, diffed when it changes, and handed to somebody who has the data but
not the terminal session.

**A study is the thing, and each interface is a way of writing one down.** The
command line parses flags into a study; a file is read into one; Python
constructs one directly; a form would fill one in. All of them then run the
same object, which is what keeps them from drifting -- a setting reachable from
one interface and not another is a bug that cannot happen when there is only
one place for settings to live.

    prothon compare -e wt.xtc mut.xtc -t top.pdb    # flags become a study
    prothon compare --config study.yml              # a file becomes one
    Study.from_file("study.yml").run()              # Python runs one

It also runs the other way. A command line typed once can be written down::

    prothon compare -e wt.xtc mut.xtc -t top.pdb --save-config study.yml

so the study behind a figure can be committed beside the manuscript rather
than reconstructed from memory.

**It is not merely the flags in a file.** Three things a study expresses that a
command line cannot:

*A topology per ensemble.* ``--topology`` is one path for every source, which
is right when comparing conditions of one system and wrong for everything else.
A mutant has its own topology, and so does an ortholog.

*A label per ensemble.* Figures and tables read better with "wild type" than
with "sim_run3_final.xtc", and a label survives the file being moved.

*Weights per ensemble.* A reweighted simulation carries per-frame weights in a
separate file, and there is no sensible flag for that.

**Every key is checked against the schema.** A configuration file that silently
ignores a key it does not recognise is a file that lies: a misspelled
``random_seed`` would leave the study unseeded and say nothing, and the run
would look fine. Unknown keys are refused, with the closest known name offered.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from ..utils import get_logger
from .schema import parameters_for

logger = get_logger("config.study")

__all__ = ["Study", "load_study"]

#: Keys allowed at the top level of a configuration file.
TOP_LEVEL = {"ensembles", "reference", "compare", "output_dir", "description"}

#: Keys allowed for one ensemble. ``source`` is the older name for
#: ``ensemble`` and still works.
ENSEMBLE_KEYS = {
    "ensemble", "source", "topology", "label", "weights", "stride", "chains",
}

#: Where an ensemble's conformations come from, and the name it used to have.
WHERE = "ensemble"
WHERE_LEGACY = "source"


def _where(entry: dict):
    """The source of one ensemble, under either name."""
    return entry.get(WHERE, entry.get(WHERE_LEGACY))


def _suggest(name: str, known) -> str:
    import difflib

    close = difflib.get_close_matches(name, known, n=2, cutoff=0.5)
    return f" Did you mean {' or '.join(close)}?" if close else ""


@dataclass
class Study:
    """A comparison, as read from a file.

    Attributes
    ----------
    ensembles
        One entry per ensemble, each with at least a ``source``.
    reference
        A label, an index, or a source of its own.
    settings
        Everything under ``compare:``, validated against the schema.
    """

    ensembles: list[dict[str, Any]] = field(default_factory=list)
    reference: Any = 0
    settings: dict[str, Any] = field(default_factory=dict)
    output_dir: str | None = None
    description: str | None = None
    path: str | None = None

    @classmethod
    def from_file(cls, path) -> Study:
        """Read a study from a YAML file."""
        return load_study(path)

    @classmethod
    def from_arguments(cls, args) -> Study:
        """Build a study from parsed command-line arguments.

        This is what makes the command line a way of *writing* a study rather
        than a second way of running one: every flag lands in the same object a
        file would produce, so the two cannot come to offer different settings.
        """
        sources = args.ensembles or []
        if isinstance(sources, str):
            sources = [sources]
        flattened: list = []
        for item in sources:
            text = str(item)
            flattened += (
                [p for p in text.split(",") if p.strip()] if "," in text else [item]
            )

        ensembles = [{WHERE: item} for item in flattened]
        topology = getattr(args, "topology", None)
        if topology:
            # One topology for every ensemble, or one each in the same order.
            # The list form is what a mutant against a wild type needs.
            if isinstance(topology, str):
                paths = [topology] * len(ensembles)
            elif len(topology) == 1:
                paths = list(topology) * len(ensembles)
            elif len(topology) == len(ensembles):
                paths = list(topology)
            else:
                raise ValueError(
                    f"{len(topology)} topologies for {len(ensembles)} "
                    f"ensembles. Give one, or one per ensemble in the same "
                    f"order, or none for sources that carry their own."
                )
            for entry, path in zip(ensembles, paths):
                if path is not None:
                    entry["topology"] = path

        chains = getattr(args, "chains", None)
        if chains:
            picks = (
                [chains] * len(ensembles)
                if isinstance(chains, str)
                else list(chains) * len(ensembles)
                if len(chains) == 1
                else list(chains)
            )
            if len(picks) != len(ensembles):
                raise ValueError(
                    f"{len(chains)} chain selections for {len(ensembles)} "
                    f"ensembles. Give one, or one per ensemble."
                )
            for entry, pick in zip(ensembles, picks):
                entry["chains"] = pick

        # A flag not given is a flag not written down. argparse reports an
        # unset `store_true` as False while the schema declares its default as
        # None, so comparing against the schema alone would record every
        # boolean flag as an explicit false and produce a file full of
        # settings nobody chose.
        specs = {p.name: p for p in parameters_for("compare")}
        # `chains` lands on each ensemble entry above, not in the settings:
        # it says which part of a molecule an ensemble is, not how to compare.
        skip = {"ensembles", "topology", "reference", "output_dir", "config",
                "json", "verbose", "save_config", "chains"}
        settings = {}
        for name, spec in specs.items():
            if name in skip:
                continue
            value = getattr(args, name, None)
            if value is None:
                continue
            unset = False if spec.action == "store_true" else spec.default
            if value != unset:
                settings[name] = value
        # A reference may name a source of its own rather than one of the
        # ensembles being compared -- "everything against this one thing"
        # should not require putting that thing in the list and counting. It
        # is prepended, and becomes ensemble 0.
        reference = getattr(args, "reference", 0) or 0
        labels = [str(e[WHERE]) for e in ensembles]
        if not str(reference).isdigit() and str(reference) not in labels:
            entry = {WHERE: reference}
            topology = getattr(args, "topology", None)
            if topology:
                entry["topology"] = (
                    topology if isinstance(topology, str) else topology[0]
                )
            ensembles = [entry, *ensembles]
            reference = 0

        return cls(
            ensembles=ensembles,
            reference=reference,
            settings=settings,
            output_dir=getattr(args, "output_dir", None),
        )

    def merged_with(self, args) -> Study:
        """This study, with anything given explicitly on the command line.

        A flag wins over the file, so a study re-runs with a different seed or
        output directory without being edited. Whether a flag was *given* is
        decided by comparing it with the schema default, since argparse cannot
        say otherwise.
        """
        specs = {p.name: p for p in parameters_for("compare")}
        # How a study was reached is not part of what it says. Recording
        # `config` or `save_config` as settings would make a rewritten file
        # point at the file it came from.
        # `output_dir` is a property of the study rather than of the
        # comparison, and the ones below say how a study was reached rather
        # than what it says.
        skip = {"config", "save_config", "json", "verbose", "output_dir",
                "ensembles", "topology", "reference", "chains"}
        settings = {k: v for k, v in self.settings.items() if k not in skip}
        for name, spec in specs.items():
            if name in skip:
                continue
            value = getattr(args, name, None)
            if value is None:
                continue
            unset = False if spec.action == "store_true" else spec.default
            if value != unset:
                settings[name] = value
        return Study(
            ensembles=list(self.ensembles),
            reference=self.reference,
            settings=settings,
            output_dir=getattr(args, "output_dir", None) or self.output_dir,
            description=self.description,
            path=self.path,
        )

    def resolve(self, cache_dir: str | None = None) -> list:
        """Load every ensemble this study names."""
        return resolve_ensembles(self, cache_dir)

    def run(self, cache_dir: str | None = None):
        """Load the ensembles and run the comparison.

        Returns the :class:`~prothon.Prothon` object, so results, summaries and
        further analyses are all reachable from it.
        """
        from ..study import Prothon

        settings = dict(self.settings)
        for key in ("report", "json", "verbose", "config", "save_config",
                    "output_dir", "ensembles", "topology", "reference",
                    "chains"):
            settings.pop(key, None)

        block = None
        if settings.pop("no_block_permutation", False):
            block = False
        elif settings.pop("block_permutation", False):
            block = True

        dimred = settings.pop("dimred", None)
        if dimred is not None and str(dimred).lower() in {"none", ""}:
            dimred = None

        order_parameters = settings.pop("order_parameters", "cbcn")
        if isinstance(order_parameters, (list, tuple)):
            order_parameters = ",".join(order_parameters)

        comparison = Prothon(
            ensembles=self.resolve(cache_dir),
            output_dir=self.output_dir,
            random_state=settings.pop("random_state", None),
            study=self,
        )
        comparison.compare_ensembles(
            order_parameters=order_parameters,
            ref=self.reference_index(),
            dimred=dimred,
            block_permutation=block,
            legacy=settings.pop("legacy_statistics", False),
            **settings,
        )
        return comparison

    def save(self, path) -> str:
        """Write this study to a YAML file.

        The other direction of the same idea: a command line typed once becomes
        a study that can be committed beside the manuscript.
        """
        import yaml

        path = os.fspath(path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                "# Written by Prothon. Run with: prothon compare --config "
                f"{os.path.basename(path)}\n"
            )
            yaml.safe_dump(
                self.to_dict(for_file=True), handle, sort_keys=False, indent=2
            )
        logger.info("Wrote %s", path)
        return path

    @property
    def labels(self) -> list[str]:
        return [
            e.get("label") or os.path.basename(str(_where(e)).rstrip("/"))
            for e in self.ensembles
        ]

    def reference_index(self) -> int:
        """Which ensemble is the reference.

        A label is resolved against the ensembles; an index is used directly.
        A source that is not one of the ensembles is not resolved here --
        :func:`load_study` prepends it, so by this point it is index 0.
        """
        if isinstance(self.reference, int):
            index = self.reference
        elif str(self.reference).isdigit():
            index = int(self.reference)
        else:
            labels = self.labels
            if self.reference in labels:
                index = labels.index(self.reference)
            else:
                # It may name a source rather than a label.
                sources = [str(_where(e)) for e in self.ensembles]
                if self.reference not in sources:
                    raise ValueError(
                        f"The reference {self.reference!r} is not one of the "
                        f"ensembles ({', '.join(labels)}), and is not an index."
                    )
                index = sources.index(self.reference)

        if not 0 <= index < len(self.ensembles):
            raise ValueError(
                f"Reference index {index} is out of range for "
                f"{len(self.ensembles)} ensembles."
            )
        return index

    def to_dict(self, for_file: bool = False) -> dict[str, Any]:
        """The study as a mapping, in the shape a file has.

        What :meth:`save` writes and what the manifest records, so a result
        found later carries the question it answered and can be re-run from it.
        """
        payload: dict[str, Any] = {}
        if self.description:
            payload["description"] = self.description
        payload["ensembles"] = self.ensembles
        if self.reference not in (0, "0"):
            payload["reference"] = self.reference
        if self.settings:
            payload["compare"] = self.settings
        if self.output_dir:
            payload["output_dir"] = self.output_dir
        # Where a study was read from belongs in the record of a run, not in a
        # file that would then point at a different file.
        if self.path and not for_file:
            payload["path"] = os.path.abspath(self.path)
        return payload


def _validate(raw: dict[str, Any], path: str) -> None:
    """Refuse anything the schema does not know about."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} should hold a mapping of settings, not "
            f"{type(raw).__name__}. See the documentation for the shape."
        )

    for key in raw:
        if key not in TOP_LEVEL:
            raise ValueError(
                f"{path}: unknown top-level key {key!r}. Expected one of "
                f"{', '.join(sorted(TOP_LEVEL))}.{_suggest(key, TOP_LEVEL)}"
            )

    if "ensembles" not in raw:
        raise ValueError(f"{path}: no ensembles. A study needs at least two.")
    if not isinstance(raw["ensembles"], list):
        raise ValueError(
            f"{path}: 'ensembles' should be a list, one entry per ensemble."
        )
    if len(raw["ensembles"]) < 2:
        raise ValueError(
            f"{path}: {len(raw['ensembles'])} ensemble(s). A comparison needs "
            f"at least two, and each entry is one ensemble -- they are never "
            f"concatenated."
        )

    for i, entry in enumerate(raw["ensembles"]):
        if isinstance(entry, str):
            continue  # a bare source is allowed
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path}: ensemble {i} should be a path or a mapping with an "
                f"'ensemble' key, not {type(entry).__name__}."
            )
        if WHERE not in entry and WHERE_LEGACY not in entry:
            raise ValueError(
                f"{path}: ensemble {i} has no 'ensemble' key. That is the "
                f"trajectory, directory, glob, multi-model PDB or PED "
                f"accession that holds the conformations."
            )
        for key in entry:
            if key not in ENSEMBLE_KEYS:
                raise ValueError(
                    f"{path}: ensemble {i} has unknown key {key!r}. Expected "
                    f"one of {', '.join(sorted(ENSEMBLE_KEYS - {WHERE_LEGACY}))}."
                    f"{_suggest(key, ENSEMBLE_KEYS)}"
                )

    known = {p.name for p in parameters_for("compare")}
    for key in raw.get("compare") or {}:
        if key not in known:
            raise ValueError(
                f"{path}: unknown setting {key!r} under 'compare'. A setting "
                f"that is silently ignored is worse than one that is refused: "
                f"a misspelled 'random_state' would leave the study unseeded "
                f"and say nothing.{_suggest(key, known)}"
            )


def load_study(path: str | os.PathLike) -> Study:
    """Read a study from a YAML file.

    Raises
    ------
    ValueError
        For an unknown key, a missing source, or fewer than two ensembles.
        Each names what was wrong and what was expected.
    """
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - declared dependency
        raise ValueError(
            "Reading a study from a file needs PyYAML, which should have been "
            "installed with Prothon. `pip install pyyaml`."
        ) from error

    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such configuration file: {path}")

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    _validate(raw, path)

    ensembles = [
        {WHERE: entry} if isinstance(entry, str) else dict(entry)
        for entry in raw["ensembles"]
    ]
    study = Study(
        ensembles=ensembles,
        reference=raw.get("reference", 0),
        settings=dict(raw.get("compare") or {}),
        output_dir=raw.get("output_dir"),
        description=raw.get("description"),
        path=path,
    )

    logger.info(
        "%s: %d ensembles, reference %r", path, len(study.ensembles), study.reference
    )
    return study


def resolve_ensembles(study: Study, cache_dir: str | None = None):
    """Load every ensemble the study names.

    Each entry may carry its own topology, which is the thing a single
    ``--topology`` flag cannot express and the main reason to write a study
    down rather than type it.
    """
    import numpy as np

    from ..ingest.sources import resolve

    loaded = []
    for entry in study.ensembles:
        weights = entry.get("weights")
        if isinstance(weights, str):
            weights = np.loadtxt(weights).ravel()
        ensemble = resolve(
            _where(entry),
            topology=entry.get("topology"),
            label=entry.get("label"),
            cache_dir=cache_dir,
            stride=entry.get("stride"),
            chains=entry.get("chains"),
        )
        if weights is not None:
            ensemble.weights = ensemble._validate_weights(weights, ensemble.n_frames)
        loaded.append(ensemble)
    return loaded
