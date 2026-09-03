"""Getting ensembles in, and making two of them comparable.

  - :mod:`ensemble`  -- the :class:`Ensemble` type: sources, weights,
    provenance, quality
  - :mod:`ped`       -- loading ensembles from the Protein Ensemble Database
  - :mod:`sequence`  -- sequences from topologies, and affine-gap alignment
  - :mod:`reconcile` -- the residue correspondence between two ensembles, and
    the representation columns that follow from it

The reconciliation layer is what lets Prothon compare ensembles that are not
the same molecule -- a mutant against its wild type, a construct against a
longer one, coarse-grained against all-atom. Methods that require superposition
cannot ask those questions, because there is no common frame to superpose into.
"""

from .ensemble import Ensemble, EnsembleQuality
from .ped import ped_ensemble, ped_ensembles, ped_entry
from .reconcile import (
    Correspondence,
    Substitution,
    feature_identity,
    feature_residues,
    reconcile,
    residue_identity,
)
from .sequence import Alignment, align, chain_sequences, sequence_of
from .topology import TopologyFingerprint, same_topology, topology_fingerprint

__all__ = [
    "Alignment",
    "Correspondence",
    "Ensemble",
    "EnsembleQuality",
    "Substitution",
    "TopologyFingerprint",
    "ped_ensemble",
    "ped_ensembles",
    "ped_entry",
    "align",
    "chain_sequences",
    "feature_identity",
    "feature_residues",
    "reconcile",
    "residue_identity",
    "same_topology",
    "sequence_of",
    "topology_fingerprint",
]
