"""Quantities an experiment measures, computed from an ensemble.

Comparing two ensembles says how they differ. It cannot say which is right.
Answering that means predicting what an experiment would have measured and
checking against what it did measure -- which is the difference between a
difference detector and an arbiter.

**Only what can be computed exactly lives here.** Each observable below is a
closed-form function of the coordinates, so a value from this module is a
consequence of the ensemble rather than of somebody's regression:

    radius of gyration      from the atomic positions
    end-to-end distance     from the terminal alpha carbons
    pairwise distance       between any two atom selections
    3J(HN,HA)               Karplus, from the backbone phi angle
    FRET efficiency         from the donor-acceptor distance
    PRE distance            from the sixth-power average

**Chemical shifts, SAXS profiles and RDCs are deliberately absent.** A chemical
shift needs an empirical predictor trained on a database -- SPARTA+, SHIFTX2,
UCBShift; a SAXS profile needs an explicit solvent layer -- CRYSOL, FoXS,
Pepsi-SAXS; an RDC needs an alignment tensor fitted to the data it is being
compared against. Reimplementing any of them badly would be worse than not
having them. Compute them with the established tool and bring the numbers in
through :func:`~prothon.validate.score.score_observable`, which takes predicted
values from anywhere.

**Averaging is not always over the value.** A PRE reports a distance, but the
relaxation goes as the inverse sixth power, so the ensemble average is
``<r^-6>^(-1/6)`` and not ``<r>``. For a broad distribution those differ by
several angstroms, and the sixth-power average is dominated by the closest
approach. FRET efficiency is averaged the same way, over the efficiency rather
than over the distance. Each observable declares how it averages.

    Vuister, G. W.; Bax, A. Quantitative J correlation. J. Am. Chem. Soc.
    1993, 115, 7772-7777.

    Iwahara, J.; Schwieters, C. D.; Clore, G. M. Ensemble approach for NMR
    structure refinement against PRE data. J. Am. Chem. Soc. 2004, 126,
    5879-5896.
"""

from __future__ import annotations

from dataclasses import dataclass

import mdtraj as md
import numpy as np

from ..represent.order_parameters import _protein_chain_atoms, _selection_indices
from ..sampling.statistics import validate_weights
from ..utils import get_logger

logger = get_logger("validate.observables")

__all__ = [
    "KARPLUS_VUISTER_BAX",
    "Observable",
    "average_observable",
    "end_to_end",
    "fret_efficiency",
    "j_coupling_hn_ha",
    "pairwise_distance",
    "pre_distance",
    "radius_of_gyration",
]

#: Karplus coefficients for 3J(HN,HA), from Vuister and Bax 1993. The most
#: widely used parameterisation; others exist and change the predicted coupling
#: by a few tenths of a hertz, which is comparable to experimental uncertainty.
KARPLUS_VUISTER_BAX = (6.51, -1.76, 1.60)


@dataclass(frozen=True)
class Observable:
    """One measurable quantity, and how an ensemble average of it is taken.

    Attributes
    ----------
    averaging
        ``linear`` for a quantity whose ensemble average is the mean of the
        per-conformation values. ``r6`` for a distance reported through an
        inverse-sixth-power interaction, where the average is
        ``<r^-6>^(-1/6)`` -- the two differ by several angstroms on a broad
        distribution, and using the wrong one is not a rounding error.
    """

    name: str
    description: str
    units: str
    averaging: str = "linear"


def average_observable(values, weights=None, averaging: str = "linear"):
    """Average per-conformation values the way the observable requires.

    Parameters
    ----------
    values
        ``(n_frames,)`` or ``(n_frames, n_observables)``.
    weights
        Per-frame weights, or ``None`` for uniform.
    averaging
        ``linear`` or ``r6``.
    """
    values = np.asarray(values, dtype=np.float64)
    normalised = validate_weights(weights, values.shape[0])
    if normalised is None:
        w = np.full(values.shape[0], 1.0 / values.shape[0])
    else:
        w = normalised

    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False

    if averaging == "linear":
        averaged = w @ values
    elif averaging == "r6":
        if np.any(values <= 0):
            raise ValueError(
                "A sixth-power average needs positive distances; some are zero "
                "or negative."
            )
        averaged = (w @ values**-6.0) ** (-1.0 / 6.0)
    else:
        raise ValueError(f"Unknown averaging {averaging!r}; use 'linear' or 'r6'.")

    return averaged[0] if squeeze else averaged


# ---------------------------------------------------------------------------
# Global shape
# ---------------------------------------------------------------------------
def radius_of_gyration(
    traj: md.Trajectory,
    mass_weighted: bool = True,
    selection: str | None = None,
) -> np.ndarray:
    """Radius of gyration per frame, in nm, protein-only by default.

    Mass weighted by default, which is what a SAXS-derived Rg corresponds to.
    Pass an explicit MDTraj selection to measure a complex or another subset.
    """
    indices = _selection_indices(traj, selection)
    selected = traj if indices.size == traj.n_atoms else traj.atom_slice(indices)
    return (
        md.compute_rg(selected)
        if mass_weighted
        else md.compute_rg(selected, masses=np.ones(selected.n_atoms))
    )


def end_to_end(
    traj: md.Trajectory,
    selection: str | None = None,
    chain=None,
) -> np.ndarray:
    """Distance between one protein chain's terminal alpha carbons.

    A multichain default is ambiguous and refused. Pass a chain index or ID,
    or an explicit atom selection when different endpoints are intentional.
    """
    if selection is not None and chain is not None:
        raise ValueError("Pass either selection or chain, not both.")
    indices = (
        _protein_chain_atoms(traj.topology, "CA", chain)
        if selection is None
        else _selection_indices(traj, selection)
    )
    if len(indices) < 2:
        description = "the selected protein chain" if selection is None else repr(selection)
        raise ValueError(
            f"End-to-end distance needs at least two atoms; {description} "
            f"matched {len(indices)}."
        )
    pair = np.array([[indices[0], indices[-1]]])
    return md.compute_distances(traj, pair, periodic=False)[:, 0]


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------
def _resolve_pair(traj: md.Trajectory, first, second) -> np.ndarray:
    """Turn two atom selections or indices into one pair."""
    def one(spec, which):
        if isinstance(spec, (int, np.integer)):
            return int(spec)
        indices = traj.topology.select(spec)
        if len(indices) == 0:
            raise ValueError(f"The {which} selection {spec!r} matched no atoms.")
        if len(indices) > 1:
            logger.warning(
                "The %s selection %r matched %d atoms; using the first.",
                which, spec, len(indices),
            )
        return int(indices[0])

    return np.array([[one(first, "first"), one(second, "second")]])


def pairwise_distance(traj: md.Trajectory, first, second) -> np.ndarray:
    """Distance between two atoms, per frame, in nm."""
    return md.compute_distances(traj, _resolve_pair(traj, first, second),
                                periodic=False)[:, 0]


def pre_distance(traj: md.Trajectory, first, second, weights=None) -> float:
    """Effective distance reported by a PRE measurement, in nm.

    The paramagnetic relaxation enhancement goes as the inverse sixth power of
    the distance, so the quantity an experiment reports is the sixth-power
    average ``<r^-6>^(-1/6)``. On a distribution with any breadth this is much
    closer to the shortest approach than to the mean distance, which is what
    makes PRE sensitive to rare compact states -- and what makes averaging it
    linearly wrong rather than approximate.
    """
    return float(
        average_observable(
            pairwise_distance(traj, first, second), weights, averaging="r6"
        )
    )


def fret_efficiency(traj: md.Trajectory, donor, acceptor, r0: float) -> np.ndarray:
    """FRET efficiency per frame, from the dye-to-dye distance.

    ``E = 1 / (1 + (r / R0)^6)``. Averaging is over the efficiency rather than
    over the distance, because that is what the photons report; take the mean
    of what this returns, not the efficiency of the mean distance.

    Parameters
    ----------
    r0
        Forster radius in nm. Dye-pair specific; around 5.4 nm for Alexa
        488/594.
    """
    if r0 <= 0:
        raise ValueError(f"The Forster radius must be positive; got {r0}.")
    distances = pairwise_distance(traj, donor, acceptor)
    return 1.0 / (1.0 + (distances / r0) ** 6)


# ---------------------------------------------------------------------------
# Couplings
# ---------------------------------------------------------------------------
def j_coupling_hn_ha(
    traj: md.Trajectory, coefficients: tuple[float, float, float] = KARPLUS_VUISTER_BAX
) -> tuple[np.ndarray, np.ndarray]:
    """3J(HN,HA) per residue per frame, in Hz, from the backbone phi angle.

    The Karplus relation ``A cos^2(phi - 60) + B cos(phi - 60) + C``, which
    gives about 4 Hz in a helix and about 9 Hz in a sheet.

    Returns
    -------
    couplings, residues
        ``(n_frames, n_angles)`` and the zero-based MDTraj residue index of
        each angle. Public result APIs convert these once, through
        :func:`prothon.ingest.residue_identity`, to one-based stable indices
        and chain-aware display labels.
    """
    indices, phi = md.compute_phi(traj)
    if phi.shape[1] == 0:
        raise ValueError("No backbone phi angles found; is this a protein?")
    a, b, c = coefficients
    theta = phi - np.radians(60.0)
    couplings = a * np.cos(theta) ** 2 + b * np.cos(theta) + c
    # compute_phi returns the four atoms of each dihedral; the coupling belongs
    # to the residue of the third, which carries the amide proton.
    residues = np.array(
        [traj.topology.atom(int(row[2])).residue.index for row in indices]
    )
    return couplings, residues


#: Everything this module computes, with how each is averaged.
OBSERVABLES: dict[str, Observable] = {
    "rg": Observable("rg", "Protein radius of gyration", "nm", "linear"),
    "end_to_end": Observable(
        "end_to_end", "Single-chain end-to-end distance", "nm", "linear"
    ),
    "distance": Observable("distance", "Distance between two atoms", "nm", "linear"),
    "pre": Observable("pre", "PRE effective distance", "nm", "r6"),
    "fret": Observable("fret", "FRET efficiency", "", "linear"),
    "j_hn_ha": Observable("j_hn_ha", "3J(HN,HA) coupling", "Hz", "linear"),
}

#: Observables Prothon does not compute, and what to use instead. Named
#: explicitly so that their absence is a decision rather than an oversight.
NOT_COMPUTED: dict[str, str] = {
    "chemical_shift": "SPARTA+, SHIFTX2 or UCBShift — an empirical predictor "
                      "trained on a database, not a function of the coordinates",
    "saxs": "CRYSOL, FoXS or Pepsi-SAXS — needs an explicit solvent layer",
    "rdc": "PALES or a singular-value fit — needs an alignment tensor fitted "
           "to the data being compared against",
}
