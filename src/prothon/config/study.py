"""A study, written down.

A command line is a good way to ask a question once and a poor way to record
one. The flags that produced a figure live in a shell history that nobody
reads, on one machine, and the study cannot be re-run by someone else without
being reconstructed from the paper.

A configuration file is the study: a thing that can be committed beside the
manuscript, diffed when it changes, and handed to somebody who has the data but
not the terminal session.

    prothon compare --config study.yml

**It is not merely the flags in a file.** Three things a config expresses that
a command line cannot:

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

#: Keys allowed for one ensemble.
ENSEMBLE_KEYS = {"source", "topology", "label", "weights", "stride"}


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

    @property
    def labels(self) -> list[str]:
        return [
            e.get("label") or os.path.basename(str(e["source"]).rstrip("/"))
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
            if self.reference not in labels:
                raise ValueError(
                    f"The reference {self.reference!r} is not one of the "
                    f"ensembles ({', '.join(labels)}), and is not an index."
                )
            index = labels.index(self.reference)

        if not 0 <= index < len(self.ensembles):
            raise ValueError(
                f"Reference index {index} is out of range for "
                f"{len(self.ensembles)} ensembles."
            )
        return index

    def to_dict(self) -> dict[str, Any]:
        """The study as it was read, for the manifest.

        A run records the study that produced it, so a result found later
        carries the question it answered.
        """
        return {
            "path": None if self.path is None else os.path.abspath(self.path),
            "description": self.description,
            "ensembles": self.ensembles,
            "reference": self.reference,
            "settings": self.settings,
            "output_dir": self.output_dir,
        }


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
                f"{path}: ensemble {i} should be a source or a mapping with a "
                f"'source' key, not {type(entry).__name__}."
            )
        if "source" not in entry:
            raise ValueError(
                f"{path}: ensemble {i} has no 'source'. That is the trajectory, "
                f"directory, glob, multi-model PDB or PED accession that holds "
                f"the conformations."
            )
        for key in entry:
            if key not in ENSEMBLE_KEYS:
                raise ValueError(
                    f"{path}: ensemble {i} has unknown key {key!r}. Expected "
                    f"one of {', '.join(sorted(ENSEMBLE_KEYS))}."
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
        {"source": entry} if isinstance(entry, str) else dict(entry)
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
            entry["source"],
            topology=entry.get("topology"),
            label=entry.get("label"),
            cache_dir=cache_dir,
            stride=entry.get("stride"),
        )
        if weights is not None:
            ensemble.weights = ensemble._validate_weights(weights, ensemble.n_frames)
        loaded.append(ensemble)
    return loaded
