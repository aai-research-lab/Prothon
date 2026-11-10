"""Local order parameters, and the ensemble matrices built from them.

Four words, four levels, and they are worth keeping apart:

    order parameter   the local quantity -- a contact number at one residue
    representation    the (frames x features) matrix built from one of them
    metric            the distance between two distributions of it
    observable        something an experiment measures

An earlier version of this package called the first of these a *measure*,
which collides with *metric* -- a metric is a measure of distance -- and reads
as a distinction without a difference in ``--measures cbcn --metric jsd``.
"Local order parameter" is the term of the paper this implements and of the
original code's own docstrings.

Each measure turns a trajectory into an ``(n_frames, n_features)`` matrix: one
row per conformation, one column per residue (or per angle). That matrix is the
ensemble's representation, and everything downstream -- density estimation,
Jensen-Shannon distance, dimensionality reduction -- works on it rather than on
coordinates.

**Why a measure declares its domain.** A torsion angle lives on a circle: -179
degrees and +179 degrees are two degrees apart, not 358. A contact number lives
on the positive half-line. Estimating a density needs to know which, and the
call site is the wrong place to remember it -- so each measure carries the fact
with it in :data:`ORDER_PARAMETERS`, and :mod:`prothon.compare.dissimilarity` reads it.
Version 2.0 estimated torsion densities on a linear grid, which put spurious
mass at the wraparound and understated the dissimilarity of any residue sampling
both sides of it.

**Why the contact-number loop was rewritten.** The original computed, for each
atom, a fresh list of the pairs containing it and called ``compute_distances``
once per atom -- a Python-level scan of every pair, N times over. The distances
are the same distances; computing them once and accumulating into both partners
gives identical numbers at a fraction of the cost. Pairs are processed in
blocks so that a long trajectory of a large protein does not have to hold the
whole ``(n_frames, n_pairs)`` distance matrix at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mdtraj as md
import numpy as np

from ..quiet import quiet_c_output
from ..utils import get_logger

logger = get_logger("representation")

__all__ = [
    "ORDER_PARAMETERS",
    "OrderParameter",
    "compute_ensemble_representation",
    "compute_representation",
    "compute_asph",
    "compute_caba",
    "compute_cacn",
    "compute_cata",
    "compute_cbcn",
    "compute_chi1",
    "compute_hsea",
    "compute_hsed",
    "compute_hseu",
    "compute_nu",
    "compute_ree",
    "compute_rg",
    "compute_rsa",
    "compute_sasa",
    "describe_order_parameter",
    "load_ensemble",
    "resolve_order_parameter",
]

#: Steepness of the smooth contact cutoff, in nm^-1. From the 2023 paper.
CONTACT_STEEPNESS = 50.0

#: Contact cutoff distance, in nm.
CONTACT_CUTOFF = 1.0

#: Residues closer than this in sequence are always in contact and carry no
#: information about the fold, so they are excluded from contact counts.
MIN_SEQUENCE_SEPARATION = 3

#: Memory budget for one block of pair distances. The block is sized from this
#: rather than fixed in pairs, because the allocation is pairs x frames: a
#: block of 4096 pairs costs 30 MB over a thousand frames and 6.5 GB over two
#: hundred thousand, so a fixed pair count stops chunking anything useful
#: exactly when chunking starts to matter.
_BLOCK_BYTES = 128 * 1024 * 1024

#: Floor on the block size, so that a very long trajectory still processes a
#: sensible number of pairs at a time rather than one.
_MIN_PAIR_BLOCK = 64

#: Retained as the block size used when the frame count is unknown, and as the
#: value the tests force down to exercise the blocked path.
_PAIR_BLOCK = 4096

#: Maximum solvent-accessible area per residue type, in Angstrom^2, from
#: Tien et al. (2013) Table 1 (ALLOWED-region values), plus the two scales
#: Tien et al. compare against. Keyed by three-letter residue name.
#:
#: The "theoretical" column -- the largest accessible area each residue type
#: reaches in an idealized Gly-X-Gly tripeptide over sterically allowed
#: backbone and side-chain conformations -- is the default (`RSA_SCALE`). It
#: is computed geometrically, as compute_sasa is, rather than taken from
#: observed structures, and it is the scale Tien et al. found few real
#: residues exceed. See docs/order_parameters.md.
#:
#: Glycine and alanine are present, unlike the C-beta based measures: RSA
#: needs no side-chain direction, only a reference area, so both residue
#: types are ordinary columns rather than excluded ones.
RSA_MAX_ASA: dict[str, dict[str, float]] = {
    "ALA": {"theoretical": 129.0, "empirical": 121.0, "miller": 113.0, "rose": 118.1},
    "ARG": {"theoretical": 274.0, "empirical": 265.0, "miller": 241.0, "rose": 256.0},
    "ASN": {"theoretical": 195.0, "empirical": 187.0, "miller": 158.0, "rose": 165.5},
    "ASP": {"theoretical": 193.0, "empirical": 187.0, "miller": 151.0, "rose": 158.7},
    "CYS": {"theoretical": 167.0, "empirical": 148.0, "miller": 140.0, "rose": 146.1},
    "GLU": {"theoretical": 223.0, "empirical": 214.0, "miller": 183.0, "rose": 186.2},
    "GLN": {"theoretical": 225.0, "empirical": 214.0, "miller": 189.0, "rose": 193.2},
    "GLY": {"theoretical": 104.0, "empirical": 97.0, "miller": 85.0, "rose": 88.1},
    "HIS": {"theoretical": 224.0, "empirical": 216.0, "miller": 194.0, "rose": 202.5},
    "ILE": {"theoretical": 197.0, "empirical": 195.0, "miller": 182.0, "rose": 181.0},
    "LEU": {"theoretical": 201.0, "empirical": 191.0, "miller": 180.0, "rose": 193.1},
    "LYS": {"theoretical": 236.0, "empirical": 230.0, "miller": 211.0, "rose": 225.8},
    "MET": {"theoretical": 224.0, "empirical": 203.0, "miller": 204.0, "rose": 203.4},
    "PHE": {"theoretical": 240.0, "empirical": 228.0, "miller": 218.0, "rose": 222.8},
    "PRO": {"theoretical": 159.0, "empirical": 154.0, "miller": 143.0, "rose": 146.8},
    "SER": {"theoretical": 155.0, "empirical": 143.0, "miller": 122.0, "rose": 129.8},
    "THR": {"theoretical": 172.0, "empirical": 163.0, "miller": 146.0, "rose": 152.5},
    "TRP": {"theoretical": 285.0, "empirical": 264.0, "miller": 259.0, "rose": 266.3},
    "TYR": {"theoretical": 263.0, "empirical": 255.0, "miller": 229.0, "rose": 236.8},
    "VAL": {"theoretical": 174.0, "empirical": 165.0, "miller": 160.0, "rose": 164.5},
}

#: Default reference scale for :func:`compute_rsa`. See RSA_MAX_ASA docstring.
RSA_SCALE = "theoretical"

#: Angstrom^2 -> nm^2, to match the units compute_sasa already returns.
_ANGSTROM2_TO_NM2 = 0.01

#: Half-sphere search radius, in nm (13 Angstrom), matching Hamelryck (2005)
#: and the de facto standard used by downstream HSE implementations. This is
#: a different length scale from CONTACT_CUTOFF: cbcn/cacn define direct
#: steric contact at ~1 nm, while HSE defines a local neighborhood sphere
#: whose two halves are compared to each other, not thresholded against a
#: contact/no-contact boundary. See docs/order_parameters.md.
HSE_RADIUS = 1.3

#: chi1 dihedral atom names, N-CA-CB-Xg, per residue type. Where Cbeta has
#: two branching heavy atoms (Ile, Thr, Val), the higher-priority one (CG1 or
#: OG1) is used, per standard rotamer-library convention. Glycine (no Cbeta)
#: and alanine (Cbeta is a terminal methyl, no further heavy atom) have no
#: entry and therefore no chi1 -- deliberately, not an oversight; see
#: docs/order_parameters.md.
CHI1_ATOMS: dict[str, tuple[str, str, str, str]] = {
    "CYS": ("N", "CA", "CB", "SG"),
    "ASP": ("N", "CA", "CB", "CG"),
    "GLU": ("N", "CA", "CB", "CG"),
    "PHE": ("N", "CA", "CB", "CG"),
    "HIS": ("N", "CA", "CB", "CG"),
    "ILE": ("N", "CA", "CB", "CG1"),
    "LYS": ("N", "CA", "CB", "CG"),
    "LEU": ("N", "CA", "CB", "CG"),
    "MET": ("N", "CA", "CB", "CG"),
    "ASN": ("N", "CA", "CB", "CG"),
    "PRO": ("N", "CA", "CB", "CG"),
    "GLN": ("N", "CA", "CB", "CG"),
    "ARG": ("N", "CA", "CB", "CG"),
    "SER": ("N", "CA", "CB", "OG"),
    "THR": ("N", "CA", "CB", "OG1"),
    "VAL": ("N", "CA", "CB", "CG1"),
    "TRP": ("N", "CA", "CB", "CG"),
    "TYR": ("N", "CA", "CB", "CG"),
}


def _pair_block(n_frames: int) -> int:
    """How many pairs to measure at once, given the trajectory length.

    Each pair costs ``n_frames`` float64 distances plus MDTraj's float32 copy.
    Sizing the block from a memory budget keeps peak memory flat as the
    trajectory grows, where a fixed pair count makes it grow linearly.
    """
    if n_frames <= 0:
        return _PAIR_BLOCK
    per_pair = n_frames * 8 * 1.5  # float64 result plus the float32 source
    return int(max(_MIN_PAIR_BLOCK, min(_PAIR_BLOCK, _BLOCK_BYTES // per_pair)))


@dataclass(frozen=True)
class OrderParameter:
    """One local order parameter, and the facts needed to use it correctly.

    Parameters
    ----------
    name
        Short identifier, as used on the command line and in output paths.
    description
        One line, shown by ``prothon info`` and in the generated report.
    units
        Physical units of the values, for axis labels. Empty where the
        quantity is dimensionless.
    circular
        Whether the values live on a circle. Torsions do; contact numbers,
        areas and bond angles do not. Density estimation branches on this.
    per_residue
        Whether column ``i`` corresponds to residue ``i``. Angles and torsions
        are defined on windows of consecutive residues, so their columns are
        offset from the residue index and are labelled accordingly.
    scope
        ``local`` for a quantity with one column per residue or per window,
        ``global`` for one describing the whole molecule -- a radius of
        gyration is one number per conformation, so its representation has a
        single column and there is nothing to plot per residue.
    """

    name: str
    description: str
    units: str
    circular: bool
    per_residue: bool
    scope: str = "local"

    @property
    def is_global(self) -> bool:
        return self.scope == "global"


#: Every order parameter Prothon knows. The single source of truth: the CLI
#: choices, the config validator and the report all read this rather than
#: keeping lists of their own.
ORDER_PARAMETERS: dict[str, OrderParameter] = {
    "cbcn": OrderParameter(
        "cbcn",
        "C-beta contact number, with a smooth cutoff",
        "contacts",
        circular=False,
        per_residue=True,
    ),
    "cacn": OrderParameter(
        "cacn",
        "C-alpha contact number, with a smooth cutoff",
        "contacts",
        circular=False,
        per_residue=True,
    ),
    "caba": OrderParameter(
        "caba",
        "Virtual C-alpha-C-alpha-C-alpha bond angle",
        "rad",
        circular=False,
        per_residue=False,
    ),
    "cata": OrderParameter(
        "cata",
        "Virtual C-alpha torsion angle",
        "rad",
        circular=True,
        per_residue=False,
    ),
    "rg": OrderParameter(
        "rg",
        "Protein radius of gyration",
        "nm",
        circular=False,
        per_residue=False,
        scope="global",
    ),
    "ree": OrderParameter(
        "ree",
        "Single-chain end-to-end distance",
        "nm",
        circular=False,
        per_residue=False,
        scope="global",
    ),
    "asph": OrderParameter(
        "asph",
        "Protein C-alpha asphericity: 0 for a sphere, 1 for a rod",
        "",
        circular=False,
        per_residue=False,
        scope="global",
    ),
    "nu": OrderParameter(
        "nu",
        "Single-chain Flory exponent: 0.33 compact, 0.5 ideal, 0.588 expanded",
        "",
        circular=False,
        per_residue=False,
        scope="global",
    ),
    "sasa": OrderParameter(
        "sasa",
        "Per-protein-residue solvent accessible surface area",
        "nm^2",
        circular=False,
        per_residue=True,
    ),
    "rsa": OrderParameter(
        "rsa",
        "Relative solvent accessibility (SASA normalized by residue-type maximum)",
        "",
        circular=False,
        per_residue=True,
    ),
    "hseu": OrderParameter(
        "hseu",
        "Half-sphere exposure, side-chain-facing hemisphere neighbor count",
        "contacts",
        circular=False,
        per_residue=True,
    ),
    "hsed": OrderParameter(
        "hsed",
        "Half-sphere exposure, opposite hemisphere neighbor count",
        "contacts",
        circular=False,
        per_residue=True,
    ),
    "hsea": OrderParameter(
        "hsea",
        "Half-sphere exposure anisotropy, (hseu - hsed) / (hseu + hsed)",
        "",
        circular=False,
        per_residue=True,
    ),
    "chi1": OrderParameter(
        "chi1",
        "Side-chain chi1 torsion angle (N-CA-CB-Xg)",
        "rad",
        circular=True,
        per_residue=True,
    ),
}


def resolve_order_parameter(name: str) -> OrderParameter:
    """Look up an order parameter by name, suggesting alternatives when unknown.

    A typo that reaches the density estimator produces a confusing failure
    several frames down the stack, so it is caught here with a message that
    names what was meant.
    """
    key = str(name).strip().lower()
    if key in ORDER_PARAMETERS:
        return ORDER_PARAMETERS[key]
    import difflib

    close = difflib.get_close_matches(key, ORDER_PARAMETERS, n=2, cutoff=0.5)
    hint = f" Did you mean {' or '.join(close)}?" if close else ""
    raise ValueError(
        f"Unknown order parameter {name!r}. Available: "
        f"{', '.join(sorted(ORDER_PARAMETERS))}.{hint}"
    )


def describe_order_parameter(name: str) -> str:
    """One-line description, for help text and reports."""
    spec = resolve_order_parameter(name)
    units = f" ({spec.units})" if spec.units else ""
    return f"{spec.name}: {spec.description}{units}"


def load_ensemble(file: str, topology: str) -> md.Trajectory:
    """Load one trajectory file.

    ``mdtraj.load`` dispatches on extension and handles every format Prothon
    accepts, including the DCD files the 2023 paper used; the explicit DCD
    branch the original carried is no longer needed.
    """
    with quiet_c_output():
        return md.load(file, top=topology)


def _selected_atoms(traj: md.Trajectory, selection: str, label: str) -> np.ndarray:
    """Resolve a protein atom selection, failing with a usable message when empty.

    A coarse-grained model with no C-beta atoms, or a topology whose atom names
    do not follow PDB convention, otherwise produces an empty array that fails
    much later inside NumPy with nothing pointing back at the cause.
    """
    protein = {residue.index for residue in _protein_residues(traj.topology)}
    indices = np.asarray(
        [
            index
            for index in traj.topology.select(selection)
            if traj.topology.atom(int(index)).residue.index in protein
        ],
        dtype=int,
    )
    if len(indices) == 0:
        raise ValueError(
            f"No {label} atoms found among protein residues in the topology "
            f"(selection: {selection!r}). "
            f"Check that the topology uses standard PDB atom naming, or choose a "
            f"measure that does not require {label} atoms."
        )
    return indices


def _protein_residues(topology) -> list:
    """Protein residues, including force-field and modified residue names."""
    # Runtime import avoids a module-import cycle: reconciliation imports this
    # module to resolve order parameters, while sequence handling imports no
    # representation code at runtime.
    from ..ingest.sequence import is_amino_acid

    top = getattr(topology, "topology", topology)
    return [residue for residue in top.residues if is_amino_acid(residue)]


def _protein_atom_indices(topology, atom_name: str | None = None) -> np.ndarray:
    """Indices of protein atoms, optionally restricted by atom name."""
    indices = [
        atom.index
        for residue in _protein_residues(topology)
        for atom in residue.atoms
        if atom_name is None or atom.name == atom_name
    ]
    return np.asarray(indices, dtype=int)


def _residue_numbers_are_adjacent(left, right) -> bool:
    """Whether file numbering supports a contiguous residue window."""
    first = getattr(left, "resSeq", None)
    second = getattr(right, "resSeq", None)
    if first is None or second is None:
        return True
    difference = int(second) - int(first)
    # Equal sequence numbers can be adjacent insertion-code residues. A jump
    # larger than one is an unresolved segment and must break a local window.
    return difference in (0, 1)


def _peptide_bonds(topology) -> set[tuple[int, int]]:
    """Directed residue pairs joined by a C--N peptide bond."""
    top = getattr(topology, "topology", topology)
    pairs: set[tuple[int, int]] = set()
    for first, second in top.bonds:
        names = (first.name, second.name)
        if set(names) != {"C", "N"}:
            continue
        left, right = first.residue, second.residue
        if left.chain.index != right.chain.index or left.index == right.index:
            continue
        pairs.add(tuple(sorted((left.index, right.index))))
    return pairs


def _protein_ca_windows(topology, width: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """CA atom and residue windows that never cross a chain or chain break."""
    top = getattr(topology, "topology", topology)
    protein = _protein_residues(top)
    by_chain: dict[int, list] = {}
    for residue in protein:
        by_chain.setdefault(residue.chain.index, []).append(residue)

    peptide_bonds = _peptide_bonds(top)
    atom_windows: list[tuple[int, ...]] = []
    residue_windows: list[tuple[int, ...]] = []
    for residues in by_chain.values():
        records = []
        for residue in residues:
            ca = [atom.index for atom in residue.atoms if atom.name == "CA"]
            if len(ca) == 1:
                records.append((residue, ca[0]))
        chain_bonds = {
            pair
            for pair in peptide_bonds
            if top.residue(pair[0]).chain.index == residues[0].chain.index
        }
        for start in range(len(records) - width + 1):
            window = records[start : start + width]
            residue_window = [record[0] for record in window]
            numbered = all(
                _residue_numbers_are_adjacent(left, right)
                for left, right in zip(residue_window, residue_window[1:])
            )
            bonded = all(
                tuple(sorted((left.index, right.index))) in peptide_bonds
                for left, right in zip(residue_window, residue_window[1:])
            )
            # Programmatically built topologies often carry no peptide bonds.
            # When this chain has any, they are authoritative and expose a
            # break even if residue numbering happens to remain consecutive.
            if not numbered or (chain_bonds and not bonded):
                continue
            atom_windows.append(tuple(record[1] for record in window))
            residue_windows.append(tuple(residue.index for residue in residue_window))
    return np.asarray(atom_windows, dtype=int), residue_windows


def _feature_residue_windows(topology, order_parameter: str) -> list[tuple[int, ...]]:
    """Single source of truth for representation-column residue identity."""
    top = getattr(topology, "topology", topology)
    if order_parameter in ("cbcn", "cacn"):
        atom_name = "CB" if order_parameter == "cbcn" else "CA"
        return [
            (top.atom(int(index)).residue.index,)
            for index in _protein_atom_indices(top, atom_name)
        ]
    if order_parameter in ("caba", "cata"):
        width = 3 if order_parameter == "caba" else 4
        return _protein_ca_windows(top, width)[1]
    if order_parameter in ("sasa", "rsa"):
        return [(residue.index,) for residue in _protein_residues(top)]
    if order_parameter in ("hseu", "hsed", "hsea"):
        # Every protein residue gets a column: HSE uses the real Cβ direction
        # where present and Hamelryck's pseudo-Cβ (flanking Cα) elsewhere, so
        # unlike cbcn it is not restricted to residues that carry a Cβ.
        return [(residue.index,) for residue in _protein_residues(top)]
    if order_parameter == "chi1":
        return _chi1_atom_quads(top)[1]
    raise ValueError(f"No feature-to-residue map defined for {order_parameter!r}.")


def _protein_chain_atoms(topology, atom_name: str, chain=None) -> np.ndarray:
    """Atoms from one protein chain, refusing an ambiguous multichain default."""
    top = getattr(topology, "topology", topology)
    protein = _protein_residues(top)
    records = []
    for candidate in top.chains:
        residues = [
            residue
            for residue in protein
            if residue.chain.index == candidate.index
        ]
        if not residues:
            continue
        atoms = [
            atom.index
            for residue in residues
            for atom in residue.atoms
            if atom.name == atom_name
        ]
        records.append((candidate, np.asarray(atoms, dtype=int)))

    if not records:
        raise ValueError("No protein residues found in the topology.")
    if chain is None:
        if len(records) != 1:
            raise ValueError(
                f"This topology has {len(records)} protein chains. Select one chain "
                f"before computing a chain-level descriptor, or pass chain=."
            )
        indices = records[0][1]
        if indices.size == 0:
            raise ValueError(f"No protein {atom_name} atoms found in the topology.")
        return indices

    if isinstance(chain, (int, np.integer)) and not isinstance(chain, (bool, np.bool_)):
        matches = [atoms for candidate, atoms in records if candidate.index == int(chain)]
    else:
        requested = str(chain).strip()
        matches = [
            atoms
            for candidate, atoms in records
            if (getattr(candidate, "chain_id", None) or "").strip() == requested
        ]
    if len(matches) != 1:
        available = ", ".join(
            str(getattr(candidate, "chain_id", None) or candidate.index).strip()
            for candidate, _ in records
        )
        raise ValueError(f"No unique protein chain {chain!r}; available: {available}.")
    if matches[0].size == 0:
        raise ValueError(f"Protein chain {chain!r} has no {atom_name} atoms.")
    return matches[0]


def _contact_number(
    traj: md.Trajectory,
    atom_indices: np.ndarray,
    steepness: float = CONTACT_STEEPNESS,
    cutoff: float = CONTACT_CUTOFF,
    min_separation: int = MIN_SEQUENCE_SEPARATION,
) -> np.ndarray:
    """Smooth contact number per selected atom, for every frame.

    Each pair contributes ``1 / (1 + exp(steepness * (d - cutoff)))`` to both of
    its partners: a sigmoid that is ~1 well inside the cutoff and ~0 well
    outside, so the count is differentiable rather than a step. The exponent is
    clipped before exponentiation, since at 50 nm^-1 a distance a few nm beyond
    the cutoff overflows a float64 otherwise.
    """
    n_atoms = len(atom_indices)
    if n_atoms < 2:
        raise ValueError(
            f"Contact numbers need at least two atoms; the selection matched {n_atoms}."
        )

    selected_residues = [
        traj.topology.atom(int(atom)).residue for atom in atom_indices
    ]
    chain_indices = np.asarray(
        [residue.chain.index for residue in selected_residues], dtype=int
    )
    protein_residue_indices = {
        residue.index for residue in _protein_residues(traj.topology)
    }
    chain_positions: dict[int, int] = {}
    for chain in traj.topology.chains:
        position = 0
        for residue in chain.residues:
            if residue.index in protein_residue_indices:
                chain_positions[residue.index] = position
                position += 1
    positions = np.asarray(
        [chain_positions[residue.index] for residue in selected_residues], dtype=int
    )
    i_local, j_local = np.triu_indices(n_atoms, k=1)
    same_chain = chain_indices[i_local] == chain_indices[j_local]
    sequence_separation = np.abs(positions[i_local] - positions[j_local])
    # Different chains have no sequence separation. They are physical contact
    # candidates and are retained regardless of their topology residue index.
    keep = ~same_chain | (sequence_separation >= min_separation)
    i_local, j_local = i_local[keep], j_local[keep]

    if i_local.size == 0:
        raise ValueError(
            f"No eligible atom pairs: within-chain pairs must be separated by "
            f"at least {min_separation} residues, and no inter-chain pairs exist."
        )

    # Accumulate as (n_atoms, n_frames) so np.add.at indexes rows, then
    # transpose once at the end. Indexing rows is both clearer and faster than
    # indexing a column axis.
    accumulated = np.zeros((n_atoms, traj.n_frames), dtype=np.float64)

    block = _pair_block(traj.n_frames)
    logger.debug(
        "contact numbers: %d pairs in blocks of %d over %d frames",
        i_local.size, block, traj.n_frames,
    )
    for start in range(0, i_local.size, block):
        stop = start + block
        block_i, block_j = i_local[start:stop], j_local[start:stop]
        pairs = np.column_stack([atom_indices[block_i], atom_indices[block_j]])
        # mdtraj returns float32. exp() of a float32 overflows above ~88,
        # and the clip below allows 700, so the promotion is not cosmetic:
        # in float32 every well-separated pair raises an overflow warning and
        # arrives at the right answer by accident.
        distances = md.compute_distances(traj, pairs, periodic=False).astype(np.float64)
        exponent = np.clip(steepness * (distances - cutoff), -700.0, 700.0)
        weights = (1.0 / (1.0 + np.exp(exponent))).T  # (n_pairs, n_frames)
        np.add.at(accumulated, block_i, weights)
        np.add.at(accumulated, block_j, weights)

    return accumulated.T


def compute_cbcn(traj: md.Trajectory, **kwargs) -> np.ndarray:
    """C-beta contact number per residue. Glycines have no C-beta and are
    absent from the resulting columns."""
    indices = _selected_atoms(traj, "name CB", "C-beta")
    logger.debug("cbcn: %d C-beta atoms", len(indices))
    return _contact_number(traj, indices, **kwargs)


def compute_cacn(traj: md.Trajectory, **kwargs) -> np.ndarray:
    """C-alpha contact number per residue."""
    indices = _selected_atoms(traj, "name CA", "C-alpha")
    logger.debug("cacn: %d C-alpha atoms", len(indices))
    return _contact_number(traj, indices, **kwargs)


def compute_caba(traj: md.Trajectory) -> np.ndarray:
    """Virtual bond angles over consecutive C-alpha triples, in radians."""
    triples, _ = _protein_ca_windows(traj.topology, 3)
    if len(triples) == 0:
        raise ValueError(
            "Virtual bond angles need at least 3 C-alpha atoms in one "
            "contiguous, unbroken protein chain; no valid windows were found."
        )
    return md.compute_angles(traj, triples, periodic=False)


def compute_cata(traj: md.Trajectory) -> np.ndarray:
    """Virtual torsion angles over consecutive C-alpha quadruples, in radians.

    Values wrap at +/- pi. Downstream density estimation must treat them as
    circular; :data:`ORDER_PARAMETERS` records that.
    """
    quads, _ = _protein_ca_windows(traj.topology, 4)
    if len(quads) == 0:
        raise ValueError(
            "Virtual torsions need at least 4 C-alpha atoms in one contiguous, "
            "unbroken protein chain; no valid windows were found."
        )
    return md.compute_dihedrals(traj, quads, periodic=False)


def _selection_indices(
    traj: md.Trajectory,
    selection: str | None,
    *,
    atom_name: str | None = None,
) -> np.ndarray:
    """Resolve an explicit selection or the default protein-only atoms."""
    if selection is None:
        indices = _protein_atom_indices(traj.topology, atom_name)
        description = "protein residues"
    else:
        indices = np.asarray(traj.topology.select(selection), dtype=int)
        description = repr(selection)
    if indices.size == 0:
        raise ValueError(f"The atom selection {description} matched no atoms.")
    return indices


def _gyration_eigenvalues(traj: md.Trajectory, selection: str | None = None):
    """Eigenvalues of the gyration tensor per frame, ascending.

    Everything about a conformation's overall shape follows from these: the
    radius of gyration is the square root of their sum, and the asphericity is
    a ratio of their symmetric functions.
    """
    indices = _selection_indices(traj, selection, atom_name="CA")
    if indices.size < 3:
        raise ValueError(
            f"Shape needs at least three atoms; {selection!r} matched "
            f"{indices.size}."
        )
    coords = traj.xyz[:, indices, :]
    centred = coords - coords.mean(axis=1, keepdims=True)
    tensor = np.einsum("fia,fib->fab", centred, centred) / centred.shape[1]
    return np.sort(np.linalg.eigvalsh(tensor), axis=1)


def compute_rg(traj: md.Trajectory, selection: str | None = None) -> np.ndarray:
    """Radius of gyration per frame, in nm. One column.

    Mass weighted, which is what a SAXS-derived radius of gyration
    corresponds to.
    """
    indices = _selection_indices(traj, selection)
    selected = traj if indices.size == traj.n_atoms else traj.atom_slice(indices)
    return md.compute_rg(selected).astype(np.float64)[:, None]


def compute_ree(
    traj: md.Trajectory,
    selection: str | None = None,
    chain=None,
) -> np.ndarray:
    """End-to-end distance per frame, in nm. One column.

    The default is one protein chain and refuses an ambiguous complex. An
    explicit selection may deliberately define a different pair of endpoints.
    """
    if selection is not None and chain is not None:
        raise ValueError("Pass either selection or chain, not both.")
    indices = (
        _protein_chain_atoms(traj.topology, "CA", chain)
        if selection is None
        else _selection_indices(traj, selection)
    )
    if indices.size < 2:
        raise ValueError("An end-to-end distance needs at least two selected atoms.")
    pair = np.array([[indices[0], indices[-1]]])
    return md.compute_distances(traj, pair, periodic=False).astype(np.float64)


def compute_asph(
    traj: md.Trajectory,
    selection: str | None = None,
) -> np.ndarray:
    """Asphericity per frame, dimensionless. One column.

    Zero for a sphere, one for a rod, a quarter for a flat disc. Built from
    the gyration tensor eigenvalues, so it says how the mass is distributed
    rather than how much of it there is -- two conformations with the same
    radius of gyration can have very different asphericity.
    """
    values = _gyration_eigenvalues(traj, selection)
    trace = values.sum(axis=1)
    pairs = (
        values[:, 0] * values[:, 1]
        + values[:, 1] * values[:, 2]
        + values[:, 0] * values[:, 2]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        result = 1.0 - 3.0 * pairs / trace**2
    return np.clip(np.nan_to_num(result), 0.0, 1.0)[:, None]


def compute_nu(
    traj: md.Trajectory,
    min_separation: int = 3,
    chain=None,
) -> np.ndarray:
    """Flory scaling exponent per frame, dimensionless. One column.

    Fitted from the internal scaling profile of each conformation: the
    root-mean-square distance between alpha carbons separated by ``s`` in
    sequence goes as ``s**nu``, and the exponent is the slope of that on log
    axes.

    About 0.33 for a compact globule, 0.5 for an ideal chain and 0.588 for a
    self-avoiding walk in good solvent -- the numbers a paper on a disordered
    protein reports.

    Fitted per conformation rather than once over the ensemble, because a
    comparison needs a distribution rather than a point. The per-frame value
    is noisy on a short chain, which is a property of the quantity rather than
    of the fit: the spread is roughly 0.15 at thirty residues and 0.10 at a
    hundred and twenty, and it is that spread two ensembles are compared on.
    """
    indices = _protein_chain_atoms(traj.topology, "CA", chain)
    n = indices.size
    if n < 2 * min_separation + 4:
        raise ValueError(
            f"A scaling exponent needs a chain of at least "
            f"{2 * min_separation + 4} residues; this has {n}. Use rg or ree "
            f"on something this short."
        )
    coords = traj.xyz[:, indices, :].astype(np.float64)
    separations = np.arange(min_separation, max(min_separation + 2, n // 2))

    profile = np.empty((traj.n_frames, separations.size))
    for k, s in enumerate(separations):
        delta = coords[:, s:, :] - coords[:, :-s, :]
        profile[:, k] = np.sqrt((delta**2).sum(-1).mean(axis=1))

    logs = np.log(separations)
    centred = logs - logs.mean()
    log_profile = np.log(np.maximum(profile, 1e-12))
    slope = (
        centred @ (log_profile - log_profile.mean(axis=1, keepdims=True)).T
    ) / (centred @ centred)
    return slope.astype(np.float64)[:, None]


def compute_sasa(
    traj: md.Trajectory,
    report_selection: str | None = None,
) -> np.ndarray:
    """Solvent accessible area of selected residues, in nm^2.

    Every atom remains an occluder, retaining shielding by ligands and binding
    partners. Only protein residues are reported by default; an explicit atom
    selection chooses the residues to report without changing the occluders.
    """
    atom_areas = np.asarray(md.shrake_rupley(traj, mode="atom"), dtype=np.float64)
    if report_selection is None:
        residues = _protein_residues(traj.topology)
    else:
        selected = _selection_indices(traj, report_selection)
        selected_residues = {
            traj.topology.atom(int(index)).residue.index for index in selected
        }
        residues = [
            residue
            for residue in traj.topology.residues
            if residue.index in selected_residues
        ]
    if not residues:
        raise ValueError("No residues were selected for the SASA representation.")
    result = np.empty((traj.n_frames, len(residues)), dtype=np.float64)
    for column, residue in enumerate(residues):
        atoms = [atom.index for atom in residue.atoms]
        result[:, column] = atom_areas[:, atoms].sum(axis=1)
    return result


def compute_rsa(
    traj: md.Trajectory,
    report_selection: str | None = None,
    scale: str = RSA_SCALE,
) -> np.ndarray:
    """Relative solvent accessibility per residue, dimensionless.

    RSA is SASA normalized by a residue-type reference maximum, which makes
    exposure comparable across residue types the way raw SASA is not: a fully
    exposed glycine and a fully exposed tryptophan report very different
    absolute areas despite both being maximally solvent-exposed, and RSA
    removes that confound. See docs/order_parameters.md for the full
    derivation.

    Built directly on :func:`compute_sasa` rather than recomputing
    accessibility, so the numerator always matches whatever occluders and
    algorithm SASA already uses.

    Parameters
    ----------
    report_selection
        Passed through to :func:`compute_sasa` unchanged.
    scale
        Which reference maximum to normalize by: one of the keys in every
        entry of :data:`RSA_MAX_ASA` (``"theoretical"``, ``"empirical"``,
        ``"miller"``, ``"rose"``). Defaults to :data:`RSA_SCALE`.

    Notes
    -----
    Values are typically in [0, ~1.05]; a small fraction of real
    conformations exceed 1.0 even under the corrected Tien et al. scale
    (protrusion past a locally flexible neighborhood), and this is expected
    behavior, not an error -- values are not clipped.

    Chain-terminating residues carry extra accessible surface from their free
    amino/carboxyl groups that this reference does not account for (Tien
    et al. explicitly excluded terminal residues from their derivation and
    note that no reliable normalization exists for them); their RSA may
    therefore run high and should be interpreted with that caveat rather than
    dropped from the representation.

    Non-standard residues without a table entry are returned as NaN rather
    than defaulted to a fallback area, so a silent normalization error does
    not masquerade as a real exposure value.
    """
    scales = RSA_MAX_ASA["ALA"].keys()
    if scale not in scales:
        raise ValueError(
            f"Unknown RSA scale {scale!r}; choose one of {', '.join(sorted(scales))}."
        )
    if report_selection is None:
        residues = _protein_residues(traj.topology)
    else:
        selected = _selection_indices(traj, report_selection)
        selected_residues = {
            traj.topology.atom(int(index)).residue.index for index in selected
        }
        residues = [
            residue
            for residue in traj.topology.residues
            if residue.index in selected_residues
        ]
    if not residues:
        raise ValueError("No residues were selected for the RSA representation.")

    sasa = compute_sasa(traj, report_selection=report_selection)

    max_asa = np.full(len(residues), np.nan, dtype=np.float64)
    unknown: set[str] = set()
    for column, residue in enumerate(residues):
        name = residue.name.upper()
        entry = RSA_MAX_ASA.get(name)
        if entry is None:
            unknown.add(name)
            continue
        max_asa[column] = entry[scale] * _ANGSTROM2_TO_NM2
    if unknown:
        logger.warning(
            "rsa: %d residue type(s) have no reference maximum accessibility "
            "and are reported as NaN: %s",
            len(unknown), ", ".join(sorted(unknown)),
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        return sasa / max_asa[None, :]


def _hse_reference_vectors(
    traj: md.Trajectory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Per-residue Cα index and hemisphere-defining direction vector.

    Uses the real Cα->Cβ vector where a Cβ exists, and Hamelryck's (2005)
    pseudo-Cβ direction -- built from the two flanking Cα atoms -- everywhere
    it does not (glycine, and any other residue missing a side chain). This
    is what lets HSE cover every residue, glycine included, rather than
    dropping it the way cbcn does: a residue only fails to get a direction if
    it is glycine *and* sits at a chain terminus, where neither a real Cβ nor
    a full pair of flanking Cα atoms exists.

    Returns
    -------
    own_ca : (n_residues,) int array
        Atom index of each protein residue's own Cα -- also the neighbor
        search set, since HSE looks for neighboring Cα atoms among exactly
        these.
    vectors : (n_frames, n_residues, 3) float64 array
        Hemisphere-defining direction per residue per frame. NaN for a
        residue with no direction available.
    has_direction : (n_residues,) bool array
        Whether each residue got a direction (real or pseudo-Cβ).
    residues : list
        The protein residues, in the same order as the columns above.
    """
    top = traj.topology
    residues = _protein_residues(top)
    n_res = len(residues)

    own_ca = np.full(n_res, -1, dtype=int)
    cb_index_by_residue: dict[int, int] = {}
    for k, residue in enumerate(residues):
        for atom in residue.atoms:
            if atom.name == "CA" and own_ca[k] < 0:
                own_ca[k] = atom.index
            elif atom.name == "CB":
                cb_index_by_residue[residue.index] = atom.index
    if np.any(own_ca < 0):
        missing = [
            str(residues[k]) for k in range(n_res) if own_ca[k] < 0
        ]
        raise ValueError(
            f"HSE needs a Cα atom on every protein residue; missing on: "
            f"{', '.join(missing)}."
        )

    pseudo_atoms, pseudo_windows = _protein_ca_windows(top, 3)
    pseudo_by_residue = {
        residue_window[1]: pseudo_atoms[k]
        for k, residue_window in enumerate(pseudo_windows)
    }

    n_frames = traj.n_frames
    vectors = np.full((n_frames, n_res, 3), np.nan, dtype=np.float64)
    has_direction = np.zeros(n_res, dtype=bool)

    beta_cols = [
        k for k, residue in enumerate(residues) if residue.index in cb_index_by_residue
    ]
    if beta_cols:
        cb_idx = np.array(
            [cb_index_by_residue[residues[k].index] for k in beta_cols], dtype=int
        )
        ca_idx = own_ca[beta_cols]
        vectors[:, beta_cols, :] = (
            traj.xyz[:, cb_idx, :].astype(np.float64)
            - traj.xyz[:, ca_idx, :].astype(np.float64)
        )
        has_direction[beta_cols] = True

    alpha_cols = [
        k
        for k, residue in enumerate(residues)
        if residue.index not in cb_index_by_residue
        and residue.index in pseudo_by_residue
    ]
    if alpha_cols:
        windows = np.array(
            [pseudo_by_residue[residues[k].index] for k in alpha_cols], dtype=int
        )
        prev_xyz = traj.xyz[:, windows[:, 0], :].astype(np.float64)
        own_xyz = traj.xyz[:, windows[:, 1], :].astype(np.float64)
        next_xyz = traj.xyz[:, windows[:, 2], :].astype(np.float64)
        u1 = prev_xyz - own_xyz
        u1 /= np.linalg.norm(u1, axis=-1, keepdims=True)
        u2 = next_xyz - own_xyz
        u2 /= np.linalg.norm(u2, axis=-1, keepdims=True)
        vectors[:, alpha_cols, :] = -(u1 + u2)
        has_direction[alpha_cols] = True

    return own_ca, vectors, has_direction, residues


def _half_sphere_exposure(
    traj: md.Trajectory,
    radius: float = HSE_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    """HSEu and HSEd per residue per frame: see compute_hseu/compute_hsed.

    No sequence-separation exclusion is applied -- an immediately adjacent
    backbone Cα is well within a 13 Angstrom sphere and is a genuine spatial
    neighbor, so it is counted like any other. This matches Hamelryck's
    (2005) original definition and is a deliberate difference from cbcn/cacn,
    which exclude near-sequence pairs because they define direct steric
    contact rather than a local neighborhood sphere; see docs/order_parameters.md.

    Frame-chunked to bound peak memory. The work is a dense
    O(n_residues^2) pairwise comparison per frame: simple and exact, and
    fine for single domains, but a spatial neighbour search would be the
    natural replacement if very large systems need it.
    """
    own_ca, vectors, has_direction, residues = _hse_reference_vectors(traj)
    n_frames = traj.n_frames
    n_res = own_ca.size

    own_coords = traj.xyz[:, own_ca, :].astype(np.float64)  # (F, R, 3): also the
    # neighbor search set -- HSE looks among exactly the same residues' Cα atoms.

    hseu = np.full((n_frames, n_res), np.nan, dtype=np.float64)
    hsed = np.full((n_frames, n_res), np.nan, dtype=np.float64)

    # Each frame in a chunk holds the (R, R, 3) displacement array plus the
    # (R, R) distance and projection arrays, all float64, so budget for
    # five R x R float64 planes per frame rather than one.
    per_frame_bytes = max(1, 5 * n_res * n_res * 8)
    chunk = max(1, min(n_frames, _BLOCK_BYTES // per_frame_bytes))

    logger.debug(
        "hse: %d residues, chunks of %d frame(s) over %d frames",
        n_res, chunk, n_frames,
    )
    for start in range(0, n_frames, chunk):
        stop = min(start + chunk, n_frames)
        own_block = own_coords[start:stop]        # (c, R, 3)
        diff = own_block[:, None, :, :] - own_block[:, :, None, :]  # (c, R, R, 3)
        dist = np.linalg.norm(diff, axis=-1)                        # (c, R, R)
        side = np.einsum("crnk,crk->crn", diff, vectors[start:stop])  # (c, R, R)

        not_self = dist > 1e-8
        eligible = (dist < radius) & not_self

        hseu[start:stop] = np.sum(eligible & (side > 0), axis=2)
        hsed[start:stop] = np.sum(eligible & (side <= 0), axis=2)

    hseu[:, ~has_direction] = np.nan
    hsed[:, ~has_direction] = np.nan
    return hseu, hsed


def compute_hseu(traj: md.Trajectory, radius: float = HSE_RADIUS) -> np.ndarray:
    """Half-sphere exposure, side-chain-facing hemisphere, per residue.

    Counts neighboring Cα atoms within `radius` of a residue's own Cα that
    lie on the same side as its side chain (real Cβ direction where present,
    Hamelryck's (2005) pseudo-Cβ direction for glycine). See
    docs/order_parameters.md for the full derivation and the reasoning
    behind treating this as a refinement of cbcn rather than a new axis.

    NaN for a residue with no direction available (glycine at a chain
    terminus, where neither a real nor a pseudo-Cβ can be built).
    """
    hseu, _ = _half_sphere_exposure(traj, radius=radius)
    return hseu


def compute_hsed(traj: md.Trajectory, radius: float = HSE_RADIUS) -> np.ndarray:
    """Half-sphere exposure, hemisphere opposite the side chain, per residue.

    See :func:`compute_hseu`; this is the same neighbor sphere, the other
    half of it.
    """
    _, hsed = _half_sphere_exposure(traj, radius=radius)
    return hsed


def compute_hsea(traj: md.Trajectory, radius: float = HSE_RADIUS) -> np.ndarray:
    """Half-sphere exposure anisotropy, (HSEu - HSEd) / (HSEu + HSEd).

    Bounded in [-1, 1]: +1 means every neighbor in the sphere is on the
    side-chain side, -1 means every neighbor is on the opposite side, 0 is an
    even split. A single-column summary of the same information HSEu/HSEd
    carry separately, kept because it is bounded and therefore directly
    comparable in shape to PROTHON's other bounded order parameters (e.g.
    asph). NaN wherever the residue's total neighbor count is zero, or where
    HSEu/HSEd themselves are NaN.
    """
    hseu, hsed = _half_sphere_exposure(traj, radius=radius)
    total = hseu + hsed
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (hseu - hsed) / total, np.nan)


def _chi1_atom_quads(topology) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Atom-index quads and residue identities for every chi1-bearing residue
    that actually has all four required atoms.

    Single source of truth for both compute_chi1 and its
    _feature_residue_windows entry: whether a residue gets a column depends
    not just on its name (glycine and alanine never have chi1) but on
    whether the required atoms are actually present, so both call this
    rather than duplicating the filter.
    """
    top = getattr(topology, "topology", topology)
    residues = _protein_residues(top)

    quads: list[tuple[int, int, int, int]] = []
    residue_windows: list[tuple[int, ...]] = []
    missing: list[str] = []
    for residue in residues:
        atom_names = CHI1_ATOMS.get(residue.name.upper())
        if atom_names is None:
            continue  # GLY, ALA, or an unrecognized residue type: no chi1.
        by_name = {atom.name: atom.index for atom in residue.atoms}
        indices = [by_name.get(name) for name in atom_names]
        if any(index is None for index in indices):
            missing.append(str(residue))
            continue
        quads.append(tuple(indices))
        residue_windows.append((residue.index,))

    if missing:
        logger.warning(
            "chi1: %d residue(s) are missing a required atom and are excluded: %s",
            len(missing), ", ".join(missing),
        )
    return np.asarray(quads, dtype=int), residue_windows


def compute_chi1(traj: md.Trajectory) -> np.ndarray:
    """Side-chain chi1 torsion angle per residue, in radians, on (-pi, pi].

    Defined as the dihedral N-CA-CB-Xg, where Xg is the fourth heavy atom
    bonded to Cbeta (CG1/OG1 preferred over CG2/OG2 for the three branched
    residues); see :data:`CHI1_ATOMS`. Circular: PROTHON's density
    estimation and two-sample comparison must use their circular branch for
    this measure, never the linear one -- :data:`ORDER_PARAMETERS` records
    ``circular=True`` so the rest of the package does this automatically.

    Glycine (no Cbeta) and alanine (Cbeta has no further heavy atom) have no
    chi1 and are excluded entirely, the same way cbcn excludes glycine.
    Any other residue missing its required atom for a given topology (e.g.
    an unresolved side chain in a crystal structure) is also excluded, with
    a logged warning, rather than silently producing a wrong angle.
    """
    quads, residue_windows = _chi1_atom_quads(traj.topology)
    if quads.size == 0:
        raise ValueError(
            "No residue in this topology has a complete chi1 (need at least "
            "one of Cys/Asp/Glu/Phe/His/Ile/Lys/Leu/Met/Asn/Pro/Gln/Arg/Ser/"
            "Thr/Val/Trp/Tyr with all four required atoms present)."
        )
    logger.debug("chi1: %d residue(s) with a defined chi1", len(residue_windows))
    return md.compute_dihedrals(traj, quads, periodic=False)


_COMPUTE = {
    "cbcn": compute_cbcn,
    "cacn": compute_cacn,
    "caba": compute_caba,
    "cata": compute_cata,
    "rg": compute_rg,
    "ree": compute_ree,
    "asph": compute_asph,
    "nu": compute_nu,
    "sasa": compute_sasa,
    "rsa": compute_rsa,
    "hseu": compute_hseu,
    "hsed": compute_hsed,
    "hsea": compute_hsea,
    "chi1": compute_chi1,
}


def compute_representation(traj: md.Trajectory, order_parameter: str) -> np.ndarray:
    """Measure one already-loaded trajectory.

    The file-based path loads and releases each trajectory in turn, so peak
    memory is set by the largest ensemble rather than their total. An
    :class:`~prothon.ingest.Ensemble` already holds its frames, so it needs a
    way in that does not go back to disk.
    """
    spec = resolve_order_parameter(order_parameter)
    matrix = np.asarray(_COMPUTE[spec.name](traj), dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"{spec.name} produced a {matrix.ndim}-dimensional result; "
            f"expected a 2-D (frames, features) matrix."
        )
    return matrix


def compute_ensemble_representation(
    traj_files: Sequence[str],
    topology: str,
    order_parameter: str,
    verbose: bool = False,
) -> list[np.ndarray]:
    """Build the representation matrix for each trajectory in turn.

    Parameters
    ----------
    traj_files
        One filename per ensemble. Each is loaded, measured and released
        before the next, so peak memory is set by the largest single ensemble
        rather than by their total.
    topology
        Topology file, shared by all the trajectories.
    order_parameter
        One of the keys of :data:`ORDER_PARAMETERS`.
    verbose
        Retained for backward compatibility; logging is configured centrally
        by :func:`prothon.utils.configure_logging`.

    Returns
    -------
    list of ndarray
        One ``(n_frames, n_features)`` matrix per input file.

    Raises
    ------
    ValueError
        If the order parameter is unknown, or if two ensembles produce different
        numbers of features -- which means the topologies disagree and any
        per-feature comparison between them would silently compare different
        residues.
    """
    spec = resolve_order_parameter(order_parameter)
    compute = _COMPUTE[spec.name]

    representations: list[np.ndarray] = []
    for path in traj_files:
        logger.info("Loading %s", path)
        traj = load_ensemble(path, topology)
        logger.info(
            "  %d frames, %d atoms, %d residues",
            traj.n_frames,
            traj.n_atoms,
            traj.topology.n_residues,
        )
        matrix = np.asarray(compute(traj), dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(
                f"{spec.name} produced a {matrix.ndim}-dimensional result for "
                f"{path}; expected a 2-D (frames, features) matrix."
            )
        representations.append(matrix)
        del traj

    widths = {rep.shape[1] for rep in representations}
    if len(widths) > 1:
        raise ValueError(
            f"The ensembles produced different numbers of {spec.name} features "
            f"({sorted(widths)}). They do not describe the same residues, so a "
            f"per-residue comparison between them would be meaningless. Check "
            f"that every trajectory matches the topology supplied."
        )

    return representations


# The 2.x names. "measure" collided with "metric", which means something else
# here, so the registry took the term the paper and the original code used.
Measure = OrderParameter
MEASURES = ORDER_PARAMETERS
resolve_measure = resolve_order_parameter
describe_measure = describe_order_parameter
