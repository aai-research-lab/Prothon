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
with it in :data:`ORDER_PARAMETERS`, and :mod:`prothon.core.dissimilarity` reads it.
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
    "compute_caba",
    "compute_cacn",
    "compute_cata",
    "compute_cbcn",
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
    """

    name: str
    description: str
    units: str
    circular: bool
    per_residue: bool


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
    "sasa": OrderParameter(
        "sasa",
        "Per-residue solvent accessible surface area",
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
    """Resolve an atom selection, failing with a usable message when empty.

    A coarse-grained model with no C-beta atoms, or a topology whose atom names
    do not follow PDB convention, otherwise produces an empty array that fails
    much later inside NumPy with nothing pointing back at the cause.
    """
    indices = traj.topology.select(selection)
    if len(indices) == 0:
        raise ValueError(
            f"No {label} atoms found in the topology (selection: {selection!r}). "
            f"Check that the topology uses standard PDB atom naming, or choose a "
            f"measure that does not require {label} atoms."
        )
    return np.asarray(indices, dtype=int)


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

    residues = np.array(
        [traj.topology.atom(int(a)).residue.index for a in atom_indices], dtype=int
    )
    i_local, j_local = np.triu_indices(n_atoms, k=1)
    keep = np.abs(residues[i_local] - residues[j_local]) >= min_separation
    i_local, j_local = i_local[keep], j_local[keep]

    if i_local.size == 0:
        raise ValueError(
            f"No atom pairs separated by at least {min_separation} residues. "
            f"The chain is too short for a contact-number representation."
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
    indices = _selected_atoms(traj, "name CA", "C-alpha")
    if len(indices) < 3:
        raise ValueError(
            f"Virtual bond angles need at least 3 C-alpha atoms; found {len(indices)}."
        )
    triples = np.array(
        [indices[i : i + 3] for i in range(len(indices) - 2)], dtype=int
    )
    return md.compute_angles(traj, triples, periodic=False)


def compute_cata(traj: md.Trajectory) -> np.ndarray:
    """Virtual torsion angles over consecutive C-alpha quadruples, in radians.

    Values wrap at +/- pi. Downstream density estimation must treat them as
    circular; :data:`ORDER_PARAMETERS` records that.
    """
    indices = _selected_atoms(traj, "name CA", "C-alpha")
    if len(indices) < 4:
        raise ValueError(
            f"Virtual torsions need at least 4 C-alpha atoms; found {len(indices)}."
        )
    quads = np.array([indices[i : i + 4] for i in range(len(indices) - 3)], dtype=int)
    return md.compute_dihedrals(traj, quads, periodic=False)


def compute_sasa(traj: md.Trajectory) -> np.ndarray:
    """Solvent accessible surface area per residue, in nm^2 (Shrake-Rupley)."""
    return md.shrake_rupley(traj, mode="residue")


_COMPUTE = {
    "cbcn": compute_cbcn,
    "cacn": compute_cacn,
    "caba": compute_caba,
    "cata": compute_cata,
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
