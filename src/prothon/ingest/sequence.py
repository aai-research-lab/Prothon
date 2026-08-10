"""Sequences, and how two of them line up.

Prothon compares ensembles through *local* order parameters, which is what
makes it possible to compare ensembles that do not share a coordinate frame --
and therefore ensembles that do not share a sequence. A wild type against a
point mutant, an ortholog against an ortholog, a construct that resolves a loop
against one that does not, a coarse-grained model against an all-atom one.
Methods that require superposition cannot ask these questions at all.

What they need is a map: which residue of ensemble A corresponds to which
residue of ensemble B. That is a sequence alignment problem, and this module
solves it without adding a dependency, following the same principle as the rest
of the project -- the science a package rests on should be readable inside it.

**Residue names are not sequence.** ``mdtraj`` resolves ``ALA`` and CHARMM's
``HSD`` but returns ``None`` for AMBER's ``HIE``, ``HIP`` and ``CYX``, so a
sequence read straight from an AMBER-prepared topology comes back with holes in
it and every alignment built on that sequence is wrong. Protonation- and
force-field-specific names are therefore resolved here, explicitly, rather than
delegated.

**End gaps are free by default.** The common case is two constructs of the same
protein where one resolves ten more residues at a terminus. Charging the full
affine penalty for that overhang biases the alignment toward absorbing it into
internal gaps, which is precisely the wrong answer.

    Needleman, S. B.; Wunsch, C. D. A general method applicable to the search
    for similarities in the amino acid sequence of two proteins.
    J. Mol. Biol. 1970, 48, 443-453.

    Gotoh, O. An improved algorithm for matching biological sequences.
    J. Mol. Biol. 1982, 162, 705-708.

    Henikoff, S.; Henikoff, J. G. Amino acid substitution matrices from protein
    blocks. Proc. Natl. Acad. Sci. USA 1992, 89, 10915-10919.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..utils import get_logger

logger = get_logger("ingest.sequence")

__all__ = [
    "Alignment",
    "align",
    "chain_sequences",
    "residue_letter",
    "is_amino_acid",
    "sequence_of",
]

#: Three-letter residue name to one-letter code, including the names force
#: fields use for particular protonation and bonding states. An AMBER topology
#: says HIE where a PDB says HIS, and CYX for a cystine sulfur; CHARMM says HSD.
#: All of them are the same amino acid as far as a sequence is concerned, and
#: treating them as unknown is how an alignment silently goes wrong.
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Histidine protonation states: AMBER, CHARMM, GROMACS.
    "HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H",
    # Cysteine: disulfide-bonded and deprotonated.
    "CYX": "C", "CYM": "C", "CYS2": "C",
    # Carboxyl and amine protonation variants.
    "ASH": "D", "GLH": "E", "LYN": "K", "ARN": "R", "TYM": "Y",
    # Terminal-capped forms written by some builders.
    "NALA": "A", "CALA": "A",
    # Selenomethionine and selenocysteine, common in crystal structures.
    "MSE": "M", "SEC": "U", "PYL": "O",
}

#: What an unrecognised protein residue becomes. Aligning as "any residue"
#: rather than dropping it keeps the numbering intact, which matters more than
#: scoring that one position well.
UNKNOWN_LETTER = "X"

#: BLOSUM62. Rows and columns in this order.
_BLOSUM_ORDER = "ARNDCQEGHILKMFPSTWYVBZX*"

_BLOSUM62_ROWS = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4
-2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4
-1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
 0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4
-4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1
"""


def _blosum_matrix() -> np.ndarray:
    rows = [r.split() for r in _BLOSUM62_ROWS.strip().splitlines()]
    return np.array([[int(v) for v in row] for row in rows], dtype=np.int32)


_BLOSUM = _blosum_matrix()
_INDEX = {letter: i for i, letter in enumerate(_BLOSUM_ORDER)}
_UNKNOWN_INDEX = _INDEX["X"]

#: BLAST's defaults for BLOSUM62. A gap of length k costs
#: ``GAP_OPEN + (k - 1) * GAP_EXTEND``.
GAP_OPEN = -11.0
GAP_EXTEND = -1.0

#: Above this product of lengths, the dynamic program is slow enough to be
#: worth warning about. It is quadratic and written in Python.
_LARGE_PROBLEM = 4_000_000

#: Below this fraction of the shorter sequence being aligned at all, an
#: alignment describes a fragment rather than a molecule. Free end gaps are
#: what make this necessary: they let the aligner ignore an overhang, which is
#: right for two constructs of one protein and, for two unrelated sequences,
#: degenerates into finding the best-scoring handful of positions. Two
#: unrelated 40-residue sequences align on two columns at 50% identity, which
#: clears any identity floor while covering a twentieth of the molecule.
MINIMUM_COVERAGE = 0.5

#: Below this fraction of identical aligned positions, an alignment of two
#: protein sequences is not reliable evidence that the positions correspond.
#: Comparing ensembles across such a map produces a per-residue profile whose
#: residues are not the ones named.
TWILIGHT_IDENTITY = 0.25


def residue_letter(residue) -> str:
    """One-letter code for an mdtraj residue.

    Falls back to mdtraj's own ``code`` before giving up, so a residue this
    module has never heard of but mdtraj knows still resolves.
    """
    name = residue.name.strip().upper()
    if name in THREE_TO_ONE:
        return THREE_TO_ONE[name]
    code = getattr(residue, "code", None)
    return code if code else UNKNOWN_LETTER


def is_amino_acid(residue) -> bool:
    """Whether a residue belongs in the sequence.

    Name first, then mdtraj's own judgement, then the backbone. The last test
    is what catches a modified residue nobody has heard of -- a phosphotyrosine
    written as PTR, a crosslink, a non-canonical amino acid in a designed
    protein. mdtraj calls those non-protein, so a sequence built on its
    judgement alone would drop them silently and shift every residue index
    after the drop. A residue carrying N, CA and C is a residue in the chain.
    """
    name = residue.name.strip().upper()
    if name in THREE_TO_ONE or residue.is_protein:
        return True
    return {"N", "CA", "C"} <= {a.name for a in residue.atoms}


def sequence_of(topology, chain_index: int | None = None) -> tuple[str, np.ndarray]:
    """Sequence of a topology, with the residue index behind each letter.

    Returning the indices alongside the letters is the point: an alignment maps
    positions in a string, and a correspondence needs residue indices in a
    topology. Recovering one from the other afterwards means assuming the
    sequence contained every residue, which is false wherever a topology holds
    waters, ions or a ligand.

    Parameters
    ----------
    topology
        An ``mdtraj.Topology``.
    chain_index
        Restrict to one chain. ``None`` takes every protein residue in order.

    Returns
    -------
    sequence, residue_indices
    """
    letters: list[str] = []
    indices: list[int] = []
    for residue in topology.residues:
        if chain_index is not None and residue.chain.index != chain_index:
            continue
        if not is_amino_acid(residue):
            continue  # water, ion, ligand: not part of the sequence
        letters.append(residue_letter(residue))
        indices.append(residue.index)
    return "".join(letters), np.array(indices, dtype=int)


def chain_sequences(topology) -> list[tuple[str, np.ndarray]]:
    """One ``(sequence, residue_indices)`` per chain that holds any protein."""
    out = []
    for chain in topology.chains:
        sequence, indices = sequence_of(topology, chain.index)
        if sequence:
            out.append((sequence, indices))
    return out


@dataclass(frozen=True)
class Alignment:
    """Two sequences lined up, and where they agree.

    Attributes
    ----------
    gapped_a, gapped_b
        The aligned strings, with ``-`` for gaps. Equal length.
    columns
        ``(k, 2)`` array of positions ``(i, j)`` into the *ungapped* sequences,
        one row per column where both sequences have a residue.
    identity
        Identical residues divided by aligned columns.
    score
        Alignment score under BLOSUM62 with affine gaps.
    """

    gapped_a: str
    gapped_b: str
    columns: np.ndarray
    identity: float
    score: float

    @property
    def n_aligned(self) -> int:
        return int(self.columns.shape[0])

    def formatted(self, width: int = 60) -> str:
        """The alignment as text, with a match line, wrapped."""
        marks = "".join(
            "|" if x == y else (" " if "-" in (x, y) else ".")
            for x, y in zip(self.gapped_a, self.gapped_b)
        )
        blocks = []
        for start in range(0, len(self.gapped_a), width):
            stop = start + width
            blocks.append(
                f"  a {self.gapped_a[start:stop]}\n"
                f"    {marks[start:stop]}\n"
                f"  b {self.gapped_b[start:stop]}"
            )
        return "\n\n".join(blocks)


def _score_matrix(a: str, b: str) -> np.ndarray:
    """Substitution scores for every pair of positions, as an array."""
    ia = np.array([_INDEX.get(c, _UNKNOWN_INDEX) for c in a], dtype=int)
    ib = np.array([_INDEX.get(c, _UNKNOWN_INDEX) for c in b], dtype=int)
    return _BLOSUM[np.ix_(ia, ib)].astype(np.float64)


def align(
    a: str,
    b: str,
    gap_open: float = GAP_OPEN,
    gap_extend: float = GAP_EXTEND,
    free_ends: bool = True,
) -> Alignment:
    """Globally align two protein sequences with affine gap penalties.

    Gotoh's three-matrix formulation: ``M`` for a column where both sequences
    have a residue, ``Ix`` for a gap in ``b``, ``Iy`` for a gap in ``a``.
    Separating them is what makes one long gap cheaper than several short ones,
    which is the difference between recognising a missing loop and scattering
    it across the alignment.

    Parameters
    ----------
    a, b
        One-letter sequences.
    gap_open, gap_extend
        A gap of length k costs ``gap_open + (k - 1) * gap_extend``. Defaults
        are BLAST's for BLOSUM62.
    free_ends
        Charge nothing for gaps at either end. On by default because the usual
        case is two constructs of one protein differing by a terminal overhang,
        and charging for it distorts the interior.

    Returns
    -------
    Alignment
    """
    if not a or not b:
        raise ValueError("Cannot align an empty sequence.")

    n, m = len(a), len(b)
    if n * m > _LARGE_PROBLEM:
        warnings.warn(
            f"Aligning {n} against {m} residues is {n * m:,} cells of a "
            f"quadratic dynamic program written in Python, and will be slow. "
            f"Aligning chain by chain is usually what was meant.",
            UserWarning,
            stacklevel=2,
        )

    substitution = _score_matrix(a, b)
    neg = -np.inf

    match = np.full((n + 1, m + 1), neg)
    gap_in_b = np.full((n + 1, m + 1), neg)
    gap_in_a = np.full((n + 1, m + 1), neg)
    from_match = np.zeros((n + 1, m + 1), dtype=np.int8)
    from_gap_b = np.zeros((n + 1, m + 1), dtype=np.int8)
    from_gap_a = np.zeros((n + 1, m + 1), dtype=np.int8)

    match[0, 0] = 0.0
    for i in range(1, n + 1):
        gap_in_b[i, 0] = 0.0 if free_ends else gap_open + (i - 1) * gap_extend
        from_gap_b[i, 0] = 1
    for j in range(1, m + 1):
        gap_in_a[0, j] = 0.0 if free_ends else gap_open + (j - 1) * gap_extend
        from_gap_a[0, j] = 2

    for i in range(1, n + 1):
        row_sub = substitution[i - 1]
        for j in range(1, m + 1):
            diagonal = (match[i - 1, j - 1], gap_in_b[i - 1, j - 1], gap_in_a[i - 1, j - 1])
            best = int(np.argmax(diagonal))
            match[i, j] = diagonal[best] + row_sub[j - 1]
            from_match[i, j] = best

            open_b = match[i - 1, j] + gap_open
            extend_b = gap_in_b[i - 1, j] + gap_extend
            if open_b >= extend_b:
                gap_in_b[i, j], from_gap_b[i, j] = open_b, 0
            else:
                gap_in_b[i, j], from_gap_b[i, j] = extend_b, 1

            open_a = match[i, j - 1] + gap_open
            extend_a = gap_in_a[i, j - 1] + gap_extend
            if open_a >= extend_a:
                gap_in_a[i, j], from_gap_a[i, j] = open_a, 0
            else:
                gap_in_a[i, j], from_gap_a[i, j] = extend_a, 2

    # Where the traceback starts. With free ends a trailing overhang costs
    # nothing, so the best alignment may finish anywhere on the last row or
    # column rather than in the corner.
    layers = (match, gap_in_b, gap_in_a)
    if free_ends:
        best_i, best_j, best_state, best_score = n, m, 0, neg
        for state, layer in enumerate(layers):
            for j in range(m + 1):
                if layer[n, j] > best_score:
                    best_score, best_i, best_j, best_state = layer[n, j], n, j, state
            for i in range(n + 1):
                if layer[i, m] > best_score:
                    best_score, best_i, best_j, best_state = layer[i, m], i, m, state
    else:
        best_state = int(np.argmax([layer[n, m] for layer in layers]))
        best_i, best_j = n, m
        best_score = layers[best_state][n, m]

    out_a: list[str] = []
    out_b: list[str] = []
    columns: list[tuple[int, int]] = []

    # Trailing overhang, unpenalised, emitted as gaps so the strings stay level.
    for i in range(n, best_i, -1):
        out_a.append(a[i - 1])
        out_b.append("-")
    for j in range(m, best_j, -1):
        out_a.append("-")
        out_b.append(b[j - 1])

    i, j, state = best_i, best_j, best_state
    while i > 0 or j > 0:
        if state == 0:
            if i == 0 or j == 0:
                state = 1 if j == 0 else 2
                continue
            out_a.append(a[i - 1])
            out_b.append(b[j - 1])
            columns.append((i - 1, j - 1))
            state = int(from_match[i, j])
            i, j = i - 1, j - 1
        elif state == 1:
            out_a.append(a[i - 1])
            out_b.append("-")
            state = int(from_gap_b[i, j])
            i -= 1
        else:
            out_a.append("-")
            out_b.append(b[j - 1])
            state = int(from_gap_a[i, j])
            j -= 1

    gapped_a = "".join(reversed(out_a))
    gapped_b = "".join(reversed(out_b))
    pairs = np.array(sorted(columns), dtype=int).reshape(-1, 2)

    identical = sum(1 for x, y in pairs if a[x] == b[y])
    identity = identical / len(pairs) if len(pairs) else 0.0

    logger.debug(
        "aligned %d x %d: %d columns, %.1f%% identity, score %.0f",
        n, m, len(pairs), 100 * identity, best_score,
    )
    return Alignment(gapped_a, gapped_b, pairs, float(identity), float(best_score))
