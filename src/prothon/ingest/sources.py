"""Turning "where the conformations are" into an ensemble.

An ensemble arrives from several places, and which place should not change how
it is asked for. A trajectory, a directory of structures a generative model
emitted, a multi-model PDB, a deposited entry -- each is a source, and
:func:`resolve` takes any of them.

    resolve("md.xtc", topology="top.pdb")     a trajectory
    resolve("bioemu_out/")                    a directory of PDBs
    resolve("samples/*.pdb")                  a glob
    resolve("nmr_entry.pdb")                  a multi-model PDB
    resolve("PED00024")                       a deposited entry
    resolve("PED00001:e002")                  one ensemble within an entry

This is what lets the command line and the Python API take the same argument.
``prothon compare --ensembles wt.xtc PED00024`` and
``Prothon(ensembles=["wt.xtc", "PED00024"])`` are the same request, because
both call this.

The kind of a source is decided by inspection rather than by a flag, and where
inspection is ambiguous the error says what was tried.
"""

from __future__ import annotations

import os
import re
from glob import glob

from ..utils import get_logger
from .ensemble import Ensemble

logger = get_logger("ingest.sources")

__all__ = ["describe_source", "resolve", "resolve_all"]

#: A PED accession, optionally naming one ensemble within the entry.
_PED = re.compile(r"^(PED\d+|PED\d{1,5})(?::(e\d+))?$", re.IGNORECASE)

#: Extensions mdtraj reads that need a topology alongside them.
_NEEDS_TOPOLOGY = {
    ".xtc", ".trr", ".dcd", ".nc", ".netcdf", ".binpos", ".lh5", ".h5",
    ".arc", ".xyz", ".lammpstrj", ".tng", ".mdcrd", ".crd",
}


def describe_source(source: str) -> str:
    """What kind of thing a source string names, for messages and help."""
    text = str(source).strip()
    if _PED.match(text):
        return "a PED accession"
    if os.path.isdir(text):
        return "a directory of structures"
    if any(c in text for c in "*?["):
        return "a glob of structures"
    extension = os.path.splitext(text)[1].lower()
    if extension in _NEEDS_TOPOLOGY:
        return "a trajectory"
    if extension in {".pdb", ".pdb.gz", ".cif", ".mmcif"}:
        return "a structure file"
    return "a file"


def resolve(
    source,
    topology: str | None = None,
    label: str | None = None,
    cache_dir: str | os.PathLike | None = None,
    stride: int | None = None,
    chains=None,
) -> Ensemble:
    """Load whatever a source names.

    Parameters
    ----------
    source
        A path, a directory, a glob, a PED accession, or an
        :class:`~prothon.ingest.Ensemble` -- which is returned unchanged, so a
        caller may mix already-loaded ensembles with sources to load.
    topology
        Used only by sources that need one. A trajectory without a topology is
        refused with a message saying so, rather than failing later inside a
        parser.
    label
        Name for figures and tables. Defaults to something derived from the
        source.
    cache_dir
        Where to keep downloaded entries.
    chains
        Keep only these chains: a PDB chain letter, an index, or several of
        either. A complex is often compared one chain at a time.

    Returns
    -------
    Ensemble
    """
    def _chains(ensemble):
        return ensemble if chains is None else ensemble.select_chains(chains)

    if isinstance(source, Ensemble):
        return _chains(source)

    text = str(source).strip()
    if not text:
        raise ValueError("An empty string is not a source.")

    ped = _PED.match(text)
    if ped:
        from .ped import ped_ensemble

        accession, ensemble_id = ped.group(1), ped.group(2) or "e001"
        return _chains(
            ped_ensemble(accession, ensemble_id, label=label, cache_dir=cache_dir)
        )

    if os.path.isdir(text) or any(c in text for c in "*?["):
        return _chains(Ensemble.from_pdb_models(text, label=label))

    if not os.path.exists(text):
        raise FileNotFoundError(
            f"No such source: {text!r}. Expected a trajectory, a directory of "
            f"structures, a glob, a multi-model PDB, or a PED accession such "
            f"as PED00024."
        )

    extension = os.path.splitext(text)[1].lower()
    if extension in _NEEDS_TOPOLOGY:
        if topology is None:
            raise ValueError(
                f"{os.path.basename(text)} is a trajectory and needs a "
                f"topology. Pass --topology, or use a multi-model PDB or a PED "
                f"accession, which carry their own."
            )
        return _chains(
            Ensemble.from_trajectory(text, topology, label=label, stride=stride)
        )

    # A structure file. With a topology it is a trajectory of one frame; on its
    # own it is a multi-model PDB, which is the usual case.
    if topology is not None and extension not in {".pdb", ".cif", ".mmcif"}:
        return _chains(
            Ensemble.from_trajectory(text, topology, label=label, stride=stride)
        )
    return _chains(Ensemble.from_pdb_models(text, label=label))


def resolve_all(
    sources,
    topology=None,
    cache_dir: str | os.PathLike | None = None,
    stride: int | None = None,
    chains=None,
) -> list[Ensemble]:
    """Resolve several sources, keeping them separate.

    Separate on purpose: each source is one ensemble, and joining two of them
    would average away the difference a comparison exists to measure. Use
    :meth:`Ensemble.from_files` to join replicates of a single condition.

    Parameters
    ----------
    topology
        One topology shared by every source, or one per source. A list is what
        comparison across different molecules needs -- a mutant has its own
        topology, and so does an ortholog -- and a single path is right when
        comparing conditions of one system.

        ``None`` in a list means that source carries its own, which a PED
        accession and a multi-model PDB both do::

            resolve_all(["wt.xtc", "mut.xtc"], ["wt.pdb", "mut.pdb"])
            resolve_all(["md.xtc", "PED00024"], ["top.pdb", None])

    Notes
    -----
    A comma-separated string is accepted for the command line's benefit, so
    ``--ensembles wt.xtc,mut.xtc`` and ``--ensembles wt.xtc mut.xtc`` are the
    same request.
    """
    if isinstance(sources, (str, Ensemble)):
        sources = [sources]

    flattened = []
    for item in sources:
        if isinstance(item, str) and "," in item:
            flattened += [part for part in item.split(",") if part.strip()]
        else:
            flattened.append(item)

    if not flattened:
        raise ValueError("No ensembles given.")

    if isinstance(topology, str) or topology is None:
        topologies = [topology] * len(flattened)
    else:
        topologies = list(topology)
        if len(topologies) == 1:
            topologies *= len(flattened)
        elif len(topologies) != len(flattened):
            raise ValueError(
                f"{len(topologies)} topologies for {len(flattened)} ensembles. "
                f"Give one topology, or one per ensemble in the same order, or "
                f"none at all for sources that carry their own."
            )

    if chains is None or isinstance(chains, (str, int)):
        wanted = [chains] * len(flattened)
    else:
        wanted = list(chains)
        if len(wanted) == 1:
            wanted *= len(flattened)
        elif len(wanted) != len(flattened):
            raise ValueError(
                f"{len(wanted)} chain selections for {len(flattened)} "
                f"ensembles. Give one, or one per ensemble in the same order."
            )

    resolved = []
    for item, top, chain in zip(flattened, topologies, wanted):
        if not isinstance(item, Ensemble):
            logger.info("Loading %s (%s)", item, describe_source(item))
        resolved.append(
            resolve(item, top, cache_dir=cache_dir, stride=stride, chains=chain)
        )
    return resolved


def expand_globs(sources) -> list[str]:
    """Expand any glob that a shell did not, keeping directories intact.

    A shell expands ``*.pdb`` before Prothon sees it, but a quoted glob or one
    from a config file arrives whole.
    """
    out = []
    for item in sources:
        text = str(item)
        if not os.path.isdir(text) and any(c in text for c in "*?["):
            matched = sorted(glob(text))
            out += matched or [text]
        else:
            out.append(text)
    return out
