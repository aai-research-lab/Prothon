"""Shared fixtures: small synthetic ensembles with known properties.

The tests need trajectories, and shipping real ones would put megabytes in the
repository for no gain. These are built from scratch: a short peptide whose
geometry is controlled, so a test can assert that two ensembles which differ
only in a named region show dissimilarity in that region and not elsewhere.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

N_RESIDUES = 12


def _build_topology(n_residues: int = N_RESIDUES) -> md.Topology:
    """A poly-alanine chain with N, CA, CB, C and O on every residue."""
    topology = md.Topology()
    chain = topology.add_chain()
    for _ in range(n_residues):
        residue = topology.add_residue("ALA", chain)
        for name, element in (
            ("N", md.element.nitrogen),
            ("CA", md.element.carbon),
            ("CB", md.element.carbon),
            ("C", md.element.carbon),
            ("O", md.element.oxygen),
        ):
            topology.add_atom(name, element, residue)
    for i in range(topology.n_atoms - 1):
        if topology.atom(i).residue.index == topology.atom(i + 1).residue.index:
            topology.add_bond(topology.atom(i), topology.atom(i + 1))
    return topology


def _make_trajectory(
    n_frames: int,
    seed: int,
    spread: float = 0.05,
    compact_from: int | None = None,
    compaction: float = 0.0,
) -> md.Trajectory:
    """An extended chain with Gaussian jitter, optionally pulled inward.

    ``compact_from`` and ``compaction`` bend the tail of the chain toward the
    origin, which raises the contact number of exactly those residues. That
    gives a test a difference whose location is known in advance.
    """
    rng = np.random.default_rng(seed)
    topology = _build_topology()
    n_atoms = topology.n_atoms

    coordinates = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    for atom in topology.atoms:
        residue = atom.residue.index
        offset = {"N": -0.12, "CA": 0.0, "CB": 0.15, "C": 0.12, "O": 0.2}[atom.name]
        base = np.array([residue * 0.38, offset, 0.0])
        if compact_from is not None and residue >= compact_from:
            pull = compaction * (residue - compact_from)
            base = base - np.array([pull * 0.38, 0.0, -pull * 0.2])
        coordinates[:, atom.index, :] = base

    coordinates += rng.normal(0.0, spread, coordinates.shape).astype(np.float32)
    return md.Trajectory(coordinates, topology)


@pytest.fixture(scope="session")
def topology_file(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("prothon") / "top.pdb"
    _make_trajectory(1, seed=0)[0].save_pdb(str(path))
    return str(path)


@pytest.fixture(scope="session")
def ensemble_files(tmp_path_factory) -> list[str]:
    """Three ensembles: two statistically identical, one compacted at the tail."""
    directory = tmp_path_factory.mktemp("ensembles")
    specs = [
        ("a.dcd", dict(seed=1)),
        ("b.dcd", dict(seed=2)),
        ("c.dcd", dict(seed=3, compact_from=7, compaction=0.55)),
    ]
    paths = []
    for name, kwargs in specs:
        path = directory / name
        _make_trajectory(120, **kwargs).save_dcd(str(path))
        paths.append(str(path))
    return paths


@pytest.fixture
def identical_matrices() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(400, 6))
    return base, rng.normal(size=(400, 6))


@pytest.fixture
def shifted_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Two samples from clearly different distributions in every column."""
    rng = np.random.default_rng(11)
    return rng.normal(0.0, 1.0, (400, 6)), rng.normal(6.0, 1.0, (400, 6))
