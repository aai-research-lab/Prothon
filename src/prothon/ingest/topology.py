"""Structural identity for molecular topologies.

An atom count is a shape check, not a molecule identity check. Two mutants,
two isomers, or two complexes whose chains occur in a different order can all
have the same number of atoms while assigning a representation column to a
different physical feature. The fast path in a comparison therefore uses the
complete, deterministic record below.

The fingerprint deliberately excludes coordinates and frame count. It records
only the topology: ordered chains, ordered residues, ordered atoms, and bond
connectivity. Equality of two fingerprints is exact structural equality rather
than equality of a shortened hash; ``hexdigest`` is provided only for compact
provenance and display.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

__all__ = ["TopologyFingerprint", "same_topology", "topology_fingerprint"]


def _text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _integer(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bond_value(value) -> str | int | float | bool | None:
    """Turn optional MDTraj bond metadata into a stable scalar."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    named = getattr(value, "name", None)
    return str(named if named is not None else value)


@dataclass(frozen=True)
class TopologyFingerprint:
    """An immutable, lossless identity record for an MDTraj topology.

    ``chains`` retains file order as well as chain IDs, residue names and
    numbering, atom names and elements, and their global indices. ``bonds`` is
    sorted by endpoint so the order in which a parser happened to discover
    bonds does not affect identity.
    """

    schema: str
    chains: tuple
    bonds: tuple

    def hexdigest(self) -> str:
        """A compact deterministic representation suitable for a manifest."""
        payload = {
            "schema": self.schema,
            "chains": self.chains,
            "bonds": self.bonds,
        }
        encoded = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def topology_fingerprint(topology: Any) -> TopologyFingerprint:
    """Return the complete deterministic identity of a molecular topology.

    A trajectory or :class:`~prothon.ingest.Ensemble` may be supplied in place
    of its topology. Atom and residue indices are recorded explicitly and also
    implied by tuple order; the redundancy makes accidental reordering visible
    even when a third-party topology object exposes unusual index values.
    """
    top = getattr(topology, "topology", topology)

    chains = []
    for chain_position, chain in enumerate(top.chains):
        residues = []
        for residue_position, residue in enumerate(chain.residues):
            atoms = []
            for atom_position, atom in enumerate(residue.atoms):
                element = getattr(atom, "element", None)
                element_symbol = _text(getattr(element, "symbol", element))
                atoms.append(
                    (
                        atom_position,
                        int(atom.index),
                        str(atom.name),
                        element_symbol,
                    )
                )
            residues.append(
                (
                    residue_position,
                    int(residue.index),
                    str(residue.name),
                    _integer(getattr(residue, "resSeq", None)),
                    _text(getattr(residue, "segment_id", None)),
                    tuple(atoms),
                )
            )
        chains.append(
            (
                chain_position,
                int(chain.index),
                _text(getattr(chain, "chain_id", None)),
                tuple(residues),
            )
        )

    bonds = []
    for bond in top.bonds:
        atom_a = getattr(bond, "atom1", None)
        atom_b = getattr(bond, "atom2", None)
        if atom_a is None or atom_b is None:
            atom_a, atom_b = bond[0], bond[1]
        endpoints = tuple(sorted((int(atom_a.index), int(atom_b.index))))
        bonds.append(
            (
                *endpoints,
                _bond_value(getattr(bond, "type", None)),
                _bond_value(getattr(bond, "order", None)),
            )
        )

    return TopologyFingerprint(
        schema="prothon-topology-v1",
        chains=tuple(chains),
        bonds=tuple(sorted(bonds, key=repr)),
    )


def same_topology(left: Any, right: Any) -> bool:
    """Whether two objects have exactly the same structural topology."""
    if left is right:
        return True
    return topology_fingerprint(left) == topology_fingerprint(right)
