"""Loading ensembles from the Protein Ensemble Database.

PED holds structural ensembles determined from experiment -- NMR, SAXS,
paramagnetic relaxation enhancement, often combined with restrained molecular
dynamics. Comparing a model against one of these asks a different question from
comparing it against a simulation: not whether it reproduces someone else's
force field, but whether it reproduces what the measurements support.

    Lazar, T.; et al. PED in 2021: a major update of the protein ensemble
    database for intrinsically disordered proteins. Nucleic Acids Res. 2021,
    49, D404-D411.

Three details of the API decide the shape of this module, and each of them
would produce a wrong answer if guessed at.

**The download is a gzipped tar, not a gzipped PDB.** The endpoint name says
``ensemble-pdb`` and the payload is ``ensemble.tar`` compressed; decompressing
it and handing the result to a PDB parser interleaves tar headers with
coordinates.

**The model count in the metadata is not authoritative for parsing.** PED00024
reports 576 models and a naive line count finds 575, because the first
``MODEL`` record shares a line with a tar header. The count comes from the
loaded trajectory, and the metadata count is recorded beside it so a
disagreement is visible rather than silent.

**An entry can hold several ensembles.** PED00001 has e001, e002 and e003 --
separate determinations of the same protein, not parts of one. They are
returned separately, because merging them would average over the very
differences a deposition took care to distinguish.

**On weights.** PED conformers are equally weighted: the entry API exposes no
populations, and none of the entries surveyed carried them. An ensemble loaded
here therefore has uniform weights, which is a fact about the database rather
than a default this module chose.
"""

from __future__ import annotations

import gzip
import io
import os
import tarfile
import urllib.error
import urllib.request
from typing import Any

import mdtraj as md

from ..quiet import quiet_c_output
from ..utils import get_logger
from .ensemble import Ensemble

logger = get_logger("ingest.ped")

__all__ = ["PED_API", "ped_entry", "ped_ensemble", "ped_ensembles"]

#: Base URL of the PED REST API.
PED_API = "https://deposition.proteinensemble.org/api/v1"

#: Seconds to wait on a request before giving up.
TIMEOUT = 120.0


def _get(url: str, timeout: float = TIMEOUT) -> bytes:
    """Fetch a URL, turning an HTTP error into a message that names the entry."""
    request = urllib.request.Request(url, headers={"User-Agent": "prothon"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise ValueError(
                f"PED returned 404 for {url}. Check the accession and the "
                f"ensemble identifier; entry PED00001 holds e001, e002 and e003, "
                f"and identifiers are not interchangeable between entries."
            ) from error
        raise ValueError(f"PED returned {error.code} for {url}.") from error
    except urllib.error.URLError as error:
        raise ValueError(
            f"Could not reach PED at {url}: {error.reason}. This needs network "
            f"access; download the entry by hand and use "
            f"Ensemble.from_pdb_models if that is not available."
        ) from error


def _normalise(accession: str) -> str:
    """Accept ``24``, ``PED24`` or ``PED00024`` and return the canonical form."""
    text = str(accession).strip().upper()
    if text.startswith("PED"):
        text = text[3:]
    if not text.isdigit():
        raise ValueError(
            f"Not a PED accession: {accession!r}. Expected something like "
            f"'PED00024', 'PED24' or 24."
        )
    return f"PED{int(text):05d}"


def ped_entry(accession: str) -> dict[str, Any]:
    """Metadata for one PED entry, as the API returns it.

    Useful on its own: ``entry["ensembles"]`` lists what an accession holds,
    which is how to find out that PED00001 contains three separate ensembles
    before downloading any of them.
    """
    import json

    identifier = _normalise(accession)
    payload = _get(f"{PED_API}/entries/{identifier}")
    return json.loads(payload)


def _extract_pdb(payload: bytes, identifier: str) -> str:
    """Pull the PDB text out of the gzipped tar the API returns.

    Falls back to treating the payload as a plain gzipped PDB, and then as
    plain text, so that a change in what PED serves shows up as a parse error
    on real coordinates rather than as a confusing tar failure.
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            if not members:
                raise ValueError(f"The archive for {identifier} holds no files.")
            if len(members) > 1:
                names = ", ".join(m.name for m in members[:5])
                logger.warning(
                    "%s: the archive holds %d files (%s); using the first.",
                    identifier, len(members), names,
                )
            handle = archive.extractfile(members[0])
            if handle is None:  # pragma: no cover - defensive
                raise ValueError(f"Could not read {members[0].name}.")
            return handle.read().decode("utf-8", errors="replace")
    except tarfile.ReadError:
        pass

    try:
        return gzip.decompress(payload).decode("utf-8", errors="replace")
    except (OSError, gzip.BadGzipFile):
        return payload.decode("utf-8", errors="replace")


def ped_ensemble(
    accession: str,
    ensemble_id: str = "e001",
    label: str | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> Ensemble:
    """Load one ensemble from a PED entry.

    Parameters
    ----------
    accession
        ``PED00024``, ``PED24`` or ``24``.
    ensemble_id
        Which ensemble within the entry. An entry may hold several separate
        determinations; :func:`ped_entry` lists them.
    label
        Name for figures and tables. Defaults to ``PED00024/e001``.
    cache_dir
        Where to keep the downloaded PDB. Entries run to tens of megabytes and
        a benchmark may load the same one repeatedly, so caching is worth the
        directory.

    Returns
    -------
    Ensemble
        With uniform weights, because PED does not publish populations.

    Examples
    --------
    >>> alpha_synuclein = ped_ensemble("PED00024")        # doctest: +SKIP
    >>> alpha_synuclein.n_frames                          # doctest: +SKIP
    576
    """
    identifier = _normalise(accession)
    name = f"{identifier}{ensemble_id}.pdb"

    path = None
    if cache_dir:
        path = os.path.join(os.fspath(cache_dir), name)
        os.makedirs(os.fspath(cache_dir), exist_ok=True)

    if path and os.path.exists(path):
        logger.info("%s/%s: from cache", identifier, ensemble_id)
        text = open(path, encoding="utf-8", errors="replace").read()
    else:
        url = f"{PED_API}/entries/{identifier}/ensembles/{ensemble_id}/ensemble-pdb"
        logger.info("%s/%s: downloading", identifier, ensemble_id)
        text = _extract_pdb(_get(url), f"{identifier}/{ensemble_id}")
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)

    # mdtraj reads a multi-model PDB from a file, not from a string, so a
    # temporary file is needed when nothing is being cached.
    if path:
        with quiet_c_output():
            trajectory = md.load(path)
    else:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            temporary = os.path.join(directory, name)
            with open(temporary, "w", encoding="utf-8") as handle:
                handle.write(text)
            with quiet_c_output():
                trajectory = md.load(temporary)

    ensemble = Ensemble(
        trajectory=trajectory,
        label=label or f"{identifier}/{ensemble_id}",
        provenance={
            "kind": "ped",
            "accession": identifier,
            "ensemble_id": ensemble_id,
            "api": PED_API,
            "n_models_loaded": int(trajectory.n_frames),
        },
    )
    logger.info(
        "%s/%s: %d conformations, %d residues",
        identifier, ensemble_id, ensemble.n_frames,
        trajectory.topology.n_residues,
    )
    return ensemble


def ped_ensembles(
    accession: str,
    cache_dir: str | os.PathLike | None = None,
) -> list[Ensemble]:
    """Every ensemble in a PED entry, as separate ensembles.

    Separate on purpose. An entry may hold several determinations of the same
    protein -- PED00001 holds three -- and merging them would average over
    exactly the differences the deposition distinguished. Compare them against
    each other if that is the question.
    """
    entry = ped_entry(accession)
    identifier = entry.get("entry_id", _normalise(accession))
    listed = entry.get("ensembles", [])
    if not listed:
        raise ValueError(f"{identifier} lists no ensembles.")

    ensembles = []
    for item in listed:
        ensemble_id = item["ensemble_id"]
        ensemble = ped_ensemble(identifier, ensemble_id, cache_dir=cache_dir)
        # The metadata count and the parsed count should agree. Where they do
        # not, the discrepancy is recorded rather than resolved: the parsed
        # count is what the comparison will use, and a silent disagreement
        # between the two is worth being able to see afterwards.
        expected = item.get("models")
        ensemble.provenance["n_models_reported"] = expected
        if expected is not None and int(expected) != ensemble.n_frames:
            logger.warning(
                "%s/%s: PED reports %s models, %d were parsed.",
                identifier, ensemble_id, expected, ensemble.n_frames,
            )
        ensembles.append(ensemble)
    return ensembles
