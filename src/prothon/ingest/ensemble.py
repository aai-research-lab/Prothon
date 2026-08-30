"""What a conformational ensemble is, once it is more than a trajectory file.

Version 2.1 took a list of filenames and a shared topology. That is enough for
molecular dynamics of one system and nothing else. An ensemble now arrives from
several places -- a trajectory, a directory of PDB files emitted by a
generative model, a multi-model PDB from NMR, a deposited entry -- and carries
things a trajectory has no field for:

**Weights.** A deposited ensemble stores a probability per conformer, and a
reweighted simulation produces one per frame. Treating those frames as equally
likely throws away the part of the answer that was hardest to obtain.

**Provenance.** Where the conformations came from, and what was done to them on
the way in. A run that cannot say which file, entry or model produced a number
cannot be repeated.

**A sequence.** Not the same thing as a topology. It is what makes comparison
across different molecules possible, and it has to survive a topology that also
holds waters, ions and a ligand.

**A quality record.** Chain breaks, nonstandard residues, frames with different
atom counts. Version 2.0 discovered these at the point of failure, several
frames down a NumPy stack; they are established here, once, where the file is
still in hand and the message can name it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from glob import glob
from typing import Any

import mdtraj as md
import numpy as np

from ..quiet import quiet_c_output
from ..utils import get_logger
from .sequence import THREE_TO_ONE, chain_sequences, sequence_of

logger = get_logger("ingest.ensemble")

__all__ = ["Ensemble", "EnsembleQuality"]

#: Consecutive alpha carbons sit about 0.38 nm apart. Beyond this they are not
#: bonded, and the chain the topology claims is continuous is not.
CHAIN_BREAK_NM = 0.45


@dataclass(frozen=True)
class EnsembleQuality:
    """What is wrong with an ensemble, established at the point of loading."""

    n_frames: int
    n_atoms: int
    n_residues: int
    n_protein_chains: int
    chain_breaks: tuple[int, ...] = ()
    nonstandard_residues: tuple[str, ...] = ()
    residues_without_ca: int = 0

    def warnings(self) -> list[str]:
        """Everything worth saying out loud, in the order it matters."""
        notes: list[str] = []
        if self.chain_breaks:
            where = ", ".join(str(r + 1) for r in self.chain_breaks[:6])
            more = "" if len(self.chain_breaks) <= 6 else f" (+{len(self.chain_breaks) - 6} more)"
            notes.append(
                f"{len(self.chain_breaks)} chain break(s) after residue {where}{more}. "
                f"Contact numbers and virtual angles across a break are computed "
                f"between residues that are not bonded."
            )
        if self.nonstandard_residues:
            notes.append(
                f"nonstandard residue(s): {', '.join(sorted(set(self.nonstandard_residues)))}. "
                f"These align as 'any residue' and may not correspond as intended."
            )
        if self.residues_without_ca:
            notes.append(
                f"{self.residues_without_ca} protein residue(s) have no alpha carbon."
            )
        return notes


@dataclass
class Ensemble:
    """A set of conformations of one molecule, with what is known about them.

    Parameters
    ----------
    trajectory
        The conformations. One frame per conformation.
    label
        Short name, used in figures, reports and error messages. A comparison
        that says "ensemble 1" makes the reader hold the mapping in their head.
    weights
        Probability per frame. ``None`` means every conformation is equally
        likely, which is right for unbiased simulation and wrong for a
        deposited ensemble that stored otherwise.
    provenance
        Where this came from, recorded for the manifest.

    Examples
    --------
    >>> wt = Ensemble.from_trajectory("wt.xtc", "wt.pdb", label="wild type")
    >>> samples = Ensemble.from_pdb_models("bioemu_out/*.pdb", label="BioEmu")
    """

    trajectory: md.Trajectory
    label: str = "ensemble"
    weights: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trajectory.n_frames == 0:
            raise ValueError(f"Ensemble {self.label!r} holds no conformations.")
        if self.weights is not None:
            self.weights = self._validate_weights(self.weights, self.trajectory.n_frames)

    @staticmethod
    def _validate_weights(weights, n_frames: int) -> np.ndarray:
        """Check and normalise per-frame weights.

        Normalising silently is right -- the scale of a weight vector carries
        no information -- but a negative weight is not a scaling problem, it is
        a sign that whatever produced the file did something else.
        """
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size != n_frames:
            raise ValueError(
                f"{w.size} weights for {n_frames} frames. A weight belongs to "
                f"exactly one conformation."
            )
        if not np.all(np.isfinite(w)):
            raise ValueError("Weights contain non-finite values.")
        if np.any(w < 0):
            raise ValueError(
                "Weights contain negative values. A probability cannot be negative; "
                "check whether these are log-weights, which need exponentiating first."
            )
        total = w.sum()
        if total <= 0:
            raise ValueError("Weights sum to zero; no conformation carries any weight.")
        return w / total

    # -- constructors -----------------------------------------------------

    @classmethod
    def from_trajectory(
        cls,
        path: str,
        topology: str,
        label: str | None = None,
        stride: int | None = None,
        weights=None,
    ) -> Ensemble:
        """Load one trajectory file as one ensemble."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Trajectory not found: {path}")
        if not os.path.exists(topology):
            raise FileNotFoundError(f"Topology not found: {topology}")

        with quiet_c_output():
            traj = md.load(path, top=topology, stride=stride)
        logger.info("%s: %d frames from %s", label or path, traj.n_frames, path)
        return cls(
            trajectory=traj,
            label=label or os.path.splitext(os.path.basename(path))[0],
            weights=weights,
            provenance={
                "kind": "trajectory",
                "path": os.path.abspath(path),
                "topology": os.path.abspath(topology),
                "stride": stride,
            },
        )

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str],
        topology: str,
        label: str = "ensemble",
        stride: int | None = None,
        weights=None,
    ) -> Ensemble:
        """Join several files into *one* ensemble.

        For continuing replicates of a single condition. Two different
        conditions are two ensembles, and joining them averages away the
        difference the study exists to measure -- so this is never done
        implicitly.
        """
        with quiet_c_output():
            parts = [md.load(p, top=topology, stride=stride) for p in paths]
        widths = {p.n_atoms for p in parts}
        if len(widths) > 1:
            raise ValueError(
                f"The files hold different numbers of atoms ({sorted(widths)}); "
                f"they cannot be one ensemble."
            )
        traj = parts[0] if len(parts) == 1 else md.join(parts)
        return cls(
            trajectory=traj,
            label=label,
            weights=weights,
            provenance={
                "kind": "trajectory-set",
                "paths": [os.path.abspath(p) for p in paths],
                "topology": os.path.abspath(topology),
                "stride": stride,
                "frames_per_file": [p.n_frames for p in parts],
            },
        )

    @classmethod
    def from_pdb_models(
        cls,
        pattern: str,
        label: str | None = None,
        weights=None,
    ) -> Ensemble:
        """Load a multi-model PDB, or a directory or glob of single-model PDBs.

        How generative models and structure predictors emit ensembles, and how
        NMR entries are deposited. Files are loaded in sorted order so that a
        run is reproducible; a directory listing is not.
        """
        if os.path.isdir(pattern):
            files = sorted(glob(os.path.join(pattern, "*.pdb")))
        else:
            files = sorted(glob(pattern)) or ([pattern] if os.path.exists(pattern) else [])
        if not files:
            raise FileNotFoundError(f"No PDB files matched {pattern!r}.")

        with quiet_c_output():
            if len(files) == 1:
                traj = md.load(files[0])
            else:
                first = md.load(files[0])
                frames = [first]
                for path in files[1:]:
                    model = md.load(path)
                    if model.n_atoms != first.n_atoms:
                        raise ValueError(
                            f"{os.path.basename(path)} has {model.n_atoms} atoms "
                            f"and {os.path.basename(files[0])} has {first.n_atoms}. "
                            f"Every conformation in an ensemble must be the same "
                            f"molecule."
                        )
                    frames.append(model)
                traj = md.join(frames)

        logger.info(
            "%s: %d conformations from %d file(s)",
            label or pattern, traj.n_frames, len(files),
        )
        return cls(
            trajectory=traj,
            label=label or os.path.basename(str(pattern).rstrip("/*")) or "ensemble",
            weights=weights,
            provenance={
                "kind": "pdb-models",
                "pattern": str(pattern),
                "n_files": len(files),
                "files": [os.path.abspath(f) for f in files[:20]],
                "files_truncated": len(files) > 20,
            },
        )

    def select_chains(self, chains) -> Ensemble:
        """One or more chains of this ensemble, as an ensemble of their own.

        A complex is often compared one chain at a time: a bound peptide
        against its free form, or one protomer of a dimer against another. The
        rest of the system is a different molecule and averaging over it is
        not what the question asked.

        Parameters
        ----------
        chains
            A chain letter as written in the PDB (``"A"``), an integer index
            (``0``), or several of either (``"A,B"`` or ``[0, 1]``).

        Notes
        -----
        MDTraj's ``chainid`` selector takes the integer index, and a letter
        passed to it matches nothing and returns an empty selection rather
        than failing -- so letters are resolved here against
        ``chain.chain_id`` and an unknown one is refused with the available
        ones named.
        """
        topology = self.trajectory.topology
        available = {
            (c.chain_id or "").strip(): c.index
            for c in topology.chains
            if (c.chain_id or "").strip()
        }

        if isinstance(chains, (str, int)):
            chains = [chains]
        wanted = []
        for item in chains:
            for part in (
                str(item).split(",") if isinstance(item, str) else [item]
            ):
                part = str(part).strip()
                if not part:
                    continue
                if part.isdigit():
                    index = int(part)
                elif part in available:
                    index = available[part]
                else:
                    named = ", ".join(sorted(available)) or "none are labelled"
                    raise ValueError(
                        f"{self.label}: no chain {part!r}. Chains present: "
                        f"{named}. An index also works: 0 to "
                        f"{topology.n_chains - 1}."
                    )
                if not 0 <= index < topology.n_chains:
                    raise ValueError(
                        f"{self.label}: chain index {index} is out of range; "
                        f"there are {topology.n_chains}."
                    )
                wanted.append(index)

        if not wanted:
            raise ValueError(f"{self.label}: no chains selected.")

        atoms = topology.select(
            " or ".join(f"chainid {i}" for i in sorted(set(wanted)))
        )
        if atoms.size == 0:
            raise ValueError(
                f"{self.label}: chains {sorted(set(wanted))} hold no atoms."
            )

        label = f"{self.label} chain {'+'.join(str(i) for i in sorted(set(wanted)))}"
        selected = Ensemble(
            trajectory=self.trajectory.atom_slice(atoms),
            weights=self.weights,
            label=label,
            provenance={**self.provenance, "chains": sorted(set(wanted))},
        )
        logger.info(
            "%s: %d of %d residues", label,
            selected.trajectory.topology.n_residues, topology.n_residues,
        )
        return selected

    @classmethod
    def from_ped(
        cls,
        accession: str,
        ensemble_id: str = "e001",
        label: str | None = None,
        cache_dir=None,
    ) -> Ensemble:
        """Load an ensemble from the Protein Ensemble Database.

        An entry may hold several separate determinations; this takes one.
        :func:`~prothon.ingest.ped.ped_ensembles` returns them all.

        >>> Ensemble.from_ped("PED00024")            # doctest: +SKIP
        <Ensemble 'PED00024/e001': 576 frames, 140 residues>
        """
        from .ped import ped_ensemble

        return ped_ensemble(accession, ensemble_id, label=label, cache_dir=cache_dir)

    # -- properties -------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return int(self.trajectory.n_frames)

    @property
    def topology(self):
        return self.trajectory.topology

    @property
    def sequence(self) -> str:
        """One-letter sequence of every protein residue, in order."""
        return sequence_of(self.trajectory.topology)[0]

    @property
    def frame_weights(self) -> np.ndarray:
        """Weights, uniform where none were supplied."""
        if self.weights is None:
            return np.full(self.n_frames, 1.0 / self.n_frames)
        return self.weights

    # -- operations -------------------------------------------------------

    def subsample(self, n: int, random_state=None) -> Ensemble:
        """Take ``n`` conformations without replacement.

        Weights come along and are renormalised, so a subsample of a weighted
        ensemble is still a weighted ensemble. Selection is uniform rather than
        weighted: drawing in proportion to weight would fold the weights into
        the sample and leave nothing to distinguish a likely conformation from
        one that merely appears often.
        """
        if n >= self.n_frames:
            return self
        rng = np.random.default_rng(random_state)
        keep = np.sort(rng.choice(self.n_frames, n, replace=False))
        weights = None if self.weights is None else self.weights[keep]
        return Ensemble(
            trajectory=self.trajectory[keep],
            label=self.label,
            weights=weights,
            provenance={**self.provenance, "subsampled_to": int(n)},
        )

    def quality(self) -> EnsembleQuality:
        """Inspect the ensemble, once, where the message can still be useful."""
        top = self.trajectory.topology
        nonstandard = tuple(
            r.name for r in top.residues
            if r.is_protein and r.name.strip().upper() not in THREE_TO_ONE
        )

        without_ca = 0
        breaks: list[int] = []
        for chain in top.chains:
            alphas = [
                next((a.index for a in r.atoms if a.name == "CA"), None)
                for r in chain.residues
                if r.is_protein or r.name.strip().upper() in THREE_TO_ONE
            ]
            without_ca += sum(1 for a in alphas if a is None)
            present = [a for a in alphas if a is not None]
            if len(present) < 2:
                continue
            pairs = np.array([[present[i], present[i + 1]] for i in range(len(present) - 1)])
            # The first frame is enough: a break is topology, not dynamics.
            distances = md.compute_distances(self.trajectory[0], pairs, periodic=False)[0]
            for position in np.nonzero(distances > CHAIN_BREAK_NM)[0]:
                breaks.append(int(top.atom(int(pairs[position, 0])).residue.index))

        record = EnsembleQuality(
            n_frames=self.n_frames,
            n_atoms=int(self.trajectory.n_atoms),
            n_residues=int(top.n_residues),
            n_protein_chains=len(chain_sequences(top)),
            chain_breaks=tuple(sorted(breaks)),
            nonstandard_residues=nonstandard,
            residues_without_ca=without_ca,
        )
        for note in record.warnings():
            logger.warning("%s: %s", self.label, note)
        return record

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the manifest."""
        return {
            "label": self.label,
            "n_frames": self.n_frames,
            "n_atoms": int(self.trajectory.n_atoms),
            "n_residues": int(self.trajectory.topology.n_residues),
            "sequence_length": len(self.sequence),
            "weighted": self.weights is not None,
            "provenance": self.provenance,
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        weighted = ", weighted" if self.weights is not None else ""
        return (
            f"<Ensemble {self.label!r}: {self.n_frames} frames, "
            f"{len(self.sequence)} residues{weighted}>"
        )
