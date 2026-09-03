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
    "compute_nu",
    "compute_ree",
    "compute_rg",
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
    if order_parameter == "sasa":
        return [(residue.index,) for residue in _protein_residues(top)]
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
