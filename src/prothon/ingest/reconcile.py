"""Making two ensembles comparable when they are not the same molecule.

A per-residue dissimilarity profile is only meaningful if column *k* of one
ensemble's representation describes the same position as column *k* of the
other's. Version 2.1 guaranteed that by refusing anything else: the ensembles
had to share a topology, and differing feature counts were an error.

That is the right default and the wrong limit. The questions worth asking are
mostly about ensembles that differ -- a mutant against its wild type, a
construct that resolves a loop against one that does not, a coarse-grained
model against an all-atom one, an ortholog against an ortholog. Methods built
on superposition cannot ask them at all, because there is no common coordinate
frame to superpose into. Local order parameters have no such difficulty: what
they need is a map between residues, and a sequence alignment is that map.

Two things this module insists on.

**The map is between residues, and columns are derived from it.** A column is
not always a residue. ``sasa`` has one column per residue, but ``cbcn`` has one
per C-beta -- so glycines are absent, and a mutation to or from glycine shifts
every column after it. ``caba`` and ``cata`` are windows of three and four
consecutive alpha carbons, so a column exists only where the whole window does.
Mapping columns directly would be right for one of them and quietly wrong for
the rest.

**A weak alignment is refused, not reported.** Below roughly 25% identity an
alignment of two protein sequences is not evidence that the positions
correspond. Comparing across such a map still produces a per-residue profile;
it is simply a profile of residues that are not the ones named, which is worse
than no answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..represent.order_parameters import _feature_residue_windows, resolve_order_parameter
from ..utils import get_logger
from .sequence import (
    MINIMUM_COVERAGE,
    TWILIGHT_IDENTITY,
    Alignment,
    align,
    chain_sequences,
    sequence_of,
)

if TYPE_CHECKING:  # pragma: no cover
    from .ensemble import Ensemble

logger = get_logger("ingest.reconcile")

__all__ = [
    "Correspondence",
    "Substitution",
    "feature_identity",
    "feature_residues",
    "residue_identity",
    "reconcile",
]


@dataclass(frozen=True)
class Substitution:
    """One position where the two ensembles hold different residues."""

    residue_a: int
    residue_b: int
    letter_a: str
    letter_b: str

    def __str__(self) -> str:
        # One-based, because that is how a mutation is named in a paper.
        return f"{self.letter_a}{self.residue_a + 1}{self.letter_b}"


def feature_residues(topology, order_parameter: str) -> list[tuple[int, ...]]:
    """Residue indices behind each column of a representation.

    Returns one tuple per column: a single residue for the per-residue
    parameters, and the window of consecutive alpha carbons for the angular
    ones.

    This has to stay in step with the ``compute_*`` functions it describes, and
    a test asserts that the count it returns matches the width of the matrix
    they produce.
    """
    spec = resolve_order_parameter(order_parameter)
    return _feature_residue_windows(topology, spec.name)


def residue_identity(topology, residue_indices) -> tuple[np.ndarray, np.ndarray]:
    """Return stable indices and readable labels for topology residues.

    MDTraj residue indices are zero-based implementation details. Public
    Prothon results use one-based *global* topology indices as their stable,
    machine-readable key. Display labels are chain-local and chain-qualified
    for a multichain topology, where residue 1 can legitimately occur more
    than once.

    This is the one conversion boundary used by representations and computed
    observables alike. In particular, callers must pass the zero-based residue
    indices returned by :func:`prothon.validate.observables.j_coupling_hn_ha`
    here rather than adding one independently.
    """
    top = getattr(topology, "topology", topology)
    residues = list(top.residues)
    requested = np.asarray(residue_indices, dtype=int).ravel()
    if np.any(requested < 0) or np.any(requested >= len(residues)):
        raise ValueError(
            f"Residue indices must be between 0 and {len(residues) - 1}."
        )

    chains = list(top.chains)
    multichain = len(chains) > 1
    local_position: dict[int, int] = {}
    chain_label: dict[int, str] = {}
    for chain in chains:
        chain_residues = list(chain.residues)
        identifier = (getattr(chain, "chain_id", None) or "").strip()
        # An unnamed chain still needs an unambiguous public label.
        identifier = identifier or str(chain.index + 1)
        for position, residue in enumerate(chain_residues, start=1):
            local_position[residue.index] = position
            chain_label[residue.index] = identifier

    labels = []
    for raw_index in requested:
        index = int(raw_index)
        if multichain:
            labels.append(f"{chain_label[index]}:{local_position[index]}")
        else:
            labels.append(str(index + 1))
    return requested + 1, np.asarray(labels, dtype=str)


def feature_identity(
    topology,
    order_parameter: str,
    columns=None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return the residue index and display label behind each feature column.

    Local windowed parameters are indexed by the first residue, preserving the
    convention used by existing result files, while their display label names
    the complete window. Global parameters have no residue identity and return
    ``(None, None)``.
    """
    spec = resolve_order_parameter(order_parameter)
    if spec.is_global:
        return None, None

    windows = feature_residues(topology, spec.name)
    selected = np.arange(len(windows), dtype=int)
    if columns is not None:
        selected = np.asarray(columns, dtype=int).ravel()
    chosen = [windows[int(column)] for column in selected]
    starts, _ = residue_identity(topology, [window[0] for window in chosen])

    labels: list[str] = []
    for window in chosen:
        _, parts = residue_identity(topology, window)
        if len(parts) == 1:
            labels.append(str(parts[0]))
        elif all(":" not in part for part in parts):
            labels.append(f"{parts[0]}-{parts[-1]}")
        else:
            first_chain = parts[0].split(":", 1)[0]
            last_chain = parts[-1].split(":", 1)[0]
            if first_chain == last_chain:
                labels.append(f"{parts[0]}-{parts[-1].split(':', 1)[1]}")
            else:
                labels.append("/".join(parts))
    return starts, np.asarray(labels, dtype=str)


@dataclass
class Correspondence:
    """Which residue of one ensemble is which residue of the other.

    Attributes
    ----------
    pairs
        ``(k, 2)`` array of corresponding residue indices.
    identity
        Fraction of corresponding positions holding the same residue.
    coverage
        Fraction of the shorter sequence that found a counterpart.
    substitutions
        Positions that correspond but differ. For a point mutant this is the
        mutation, named as a paper would name it.
    unmatched_a, unmatched_b
        Residues present in one ensemble with no counterpart in the other.
    """

    pairs: np.ndarray
    identity: float
    coverage: float
    substitutions: list[Substitution]
    unmatched_a: np.ndarray
    unmatched_b: np.ndarray
    labels: tuple[str, str] = ("a", "b")
    alignments: list[Alignment] = field(default_factory=list)

    @property
    def n_aligned(self) -> int:
        return int(self.pairs.shape[0])

    @property
    def is_identical(self) -> bool:
        """Whether protein residues correspond one-to-one with equal identities.

        This says nothing about atom names, elements, bonds, ligands or chain
        identity. Only a matching topology fingerprint proves that the full
        molecular topologies are identical.
        """
        return (
            not self.substitutions
            and self.unmatched_a.size == 0
            and self.unmatched_b.size == 0
        )

    def residue_map(self) -> dict[int, int]:
        return {int(i): int(j) for i, j in self.pairs}

    def columns_for(
        self, order_parameter: str, topology_a, topology_b
    ) -> tuple[np.ndarray, np.ndarray]:
        """Column indices to take from each representation so they line up.

        A column survives only if every residue in its window has a counterpart
        *and* those counterparts form a window that exists on the other side.
        The second condition is what a deletion breaks: three consecutive alpha
        carbons in one ensemble may map to three residues in the other that are
        no longer consecutive, and the virtual bond angle between them is not
        the same quantity.
        """
        mapping = self.residue_map()
        features_a = feature_residues(topology_a, order_parameter)
        features_b = feature_residues(topology_b, order_parameter)
        lookup_b = {window: index for index, window in enumerate(features_b)}

        take_a: list[int] = []
        take_b: list[int] = []
        for index, window in enumerate(features_a):
            if not all(residue in mapping for residue in window):
                continue
            counterpart = tuple(mapping[residue] for residue in window)
            match = lookup_b.get(counterpart)
            if match is not None:
                take_a.append(index)
                take_b.append(match)

        logger.debug(
            "%s: %d of %d columns comparable",
            order_parameter, len(take_a), len(features_a),
        )
        return np.array(take_a, dtype=int), np.array(take_b, dtype=int)

    def summary(self) -> str:
        left, right = self.labels
        lines = [f"{left} vs {right}: {self.n_aligned} residues correspond, "
                 f"{self.identity:.1%} identity, {self.coverage:.1%} coverage"]
        if self.substitutions:
            named = ", ".join(str(s) for s in self.substitutions[:8])
            more = "" if len(self.substitutions) <= 8 else f" (+{len(self.substitutions) - 8} more)"
            lines.append(f"  substitutions: {named}{more}")
        if self.unmatched_a.size:
            lines.append(f"  only in {left}: {self.unmatched_a.size} residues")
        if self.unmatched_b.size:
            lines.append(f"  only in {right}: {self.unmatched_b.size} residues")
        if self.is_identical:
            lines.append("  protein sequences correspond one-to-one")
        return "\n".join(lines)


def reconcile(
    ensemble_a: Ensemble,
    ensemble_b: Ensemble,
    min_identity: float = TWILIGHT_IDENTITY,
    min_coverage: float = MINIMUM_COVERAGE,
    per_chain: bool = True,
) -> Correspondence:
    """Build the residue correspondence between two ensembles.

    Parameters
    ----------
    ensemble_a, ensemble_b
        The ensembles to reconcile.
    min_identity
        Refuse below this fraction of identical corresponding positions. The
        default is the twilight zone, where sequence similarity stops being
        evidence of positional equivalence.
    min_coverage
        Refuse unless this fraction of the shorter sequence is aligned.
        Identity on its own is not a sufficient guard: with free end gaps, two
        unrelated sequences align on a handful of positions at high identity.
        Lower it deliberately to compare a domain against the protein
        containing it.
    per_chain
        Align chain against chain. When both structures carry the same complete
        set of unique chain IDs, pair those IDs even if file order differs;
        otherwise retain chain order. Concatenating the chains of a complex
        into one string invites the aligner to slide one chain against another,
        which is cheap in score and nonsense as a map.

    Raises
    ------
    ValueError
        If the chain counts differ under ``per_chain``, or the alignment comes
        back below ``min_identity``. Both cases are refusals rather than
        warnings: the alternative is a per-residue profile whose residues are
        not the ones it names.
    """
    top_a = ensemble_a.trajectory.topology
    top_b = ensemble_b.trajectory.topology
    labels = (ensemble_a.label, ensemble_b.label)

    if per_chain:
        chains_a = chain_sequences(top_a)
        chains_b = chain_sequences(top_b)
        if len(chains_a) != len(chains_b):
            raise ValueError(
                f"{labels[0]} has {len(chains_a)} protein chain(s) and {labels[1]} "
                f"has {len(chains_b)}. Prothon will not guess which chain "
                f"corresponds to which; select matching chains first, or pass "
                f"per_chain=False to align them as one sequence."
            )
        blocks = _pair_chains(top_a, top_b, chains_a, chains_b)
    else:
        blocks = [(_flatten(top_a), _flatten(top_b))]

    pairs: list[tuple[int, int]] = []
    substitutions: list[Substitution] = []
    alignments: list[Alignment] = []
    matched_a: set[int] = set()
    matched_b: set[int] = set()

    for (seq_a, idx_a), (seq_b, idx_b) in blocks:
        alignment = align(seq_a, seq_b)
        alignments.append(alignment)
        for position_a, position_b in alignment.columns:
            residue_a = int(idx_a[position_a])
            residue_b = int(idx_b[position_b])
            pairs.append((residue_a, residue_b))
            matched_a.add(residue_a)
            matched_b.add(residue_b)
            if seq_a[position_a] != seq_b[position_b]:
                substitutions.append(
                    Substitution(residue_a, residue_b, seq_a[position_a], seq_b[position_b])
                )

    if not pairs:
        raise ValueError(
            f"No residues of {labels[0]} correspond to any of {labels[1]}. "
            f"These do not appear to be the same protein."
        )

    pair_array = np.array(sorted(pairs), dtype=int)
    identity = 1.0 - len(substitutions) / len(pairs)

    shorter = min(
        sum(len(seq) for seq, _ in (chains_a if per_chain else [blocks[0][0]])),
        sum(len(seq) for seq, _ in (chains_b if per_chain else [blocks[0][1]])),
    ) if per_chain else min(len(blocks[0][0][0]), len(blocks[0][1][0]))
    coverage = len(pairs) / shorter if shorter else 0.0

    if coverage < min_coverage:
        raise ValueError(
            f"Only {coverage:.1%} of the shorter sequence aligns between {labels[0]} "
            f"and {labels[1]} ({len(pairs)} of {shorter} residues), below the "
            f"{min_coverage:.0%} floor. An alignment covering this little describes a "
            f"fragment, not a molecule, and its identity says nothing: two unrelated "
            f"sequences align on a couple of positions at high identity. Pass "
            f"min_coverage= to compare a domain against the protein containing it."
        )

    if identity < min_identity:
        raise ValueError(
            f"{labels[0]} and {labels[1]} align at {identity:.1%} identity, below the "
            f"{min_identity:.0%} floor. Below roughly a quarter, a sequence alignment "
            f"is not evidence that the positions correspond, and a per-residue "
            f"comparison across this map would name residues it is not describing. "
            f"Pass min_identity= to override if the correspondence is known by "
            f"other means."
        )

    all_a = {r for seq, idx in chain_sequences(top_a) for r in map(int, idx)}
    all_b = {r for seq, idx in chain_sequences(top_b) for r in map(int, idx)}

    correspondence = Correspondence(
        pairs=pair_array,
        identity=float(identity),
        coverage=float(coverage),
        substitutions=substitutions,
        unmatched_a=np.array(sorted(all_a - matched_a), dtype=int),
        unmatched_b=np.array(sorted(all_b - matched_b), dtype=int),
        labels=labels,
        alignments=alignments,
    )
    logger.info("%s", correspondence.summary().replace("\n", "; "))
    return correspondence


def _flatten(topology) -> tuple[str, np.ndarray]:
    """Every protein residue of a topology as one sequence."""
    return sequence_of(topology)


def _pair_chains(top_a, top_b, chains_a, chains_b):
    """Pair uniquely named protein chains by ID, or retain declared order.

    PDB writers are free to emit chains in a different order. When both
    topologies provide the same complete set of unique chain IDs, those IDs
    are stronger identity evidence than file position. If IDs are absent or
    differ between structures, the documented order-based rule remains: a
    chain named A in one file may legitimately be named X in another.
    """

    def records(topology, sequences):
        result = []
        sequence_index = 0
        for chain in topology.chains:
            sequence, indices = sequence_of(topology, chain.index)
            if not sequence:
                continue
            # ``sequences`` is passed in so this helper cannot silently drift
            # from the same chain extraction used by the caller.
            assert np.array_equal(indices, sequences[sequence_index][1])
            chain_id = (getattr(chain, "chain_id", None) or "").strip()
            result.append((chain_id or None, sequences[sequence_index]))
            sequence_index += 1
        return result

    records_a = records(top_a, chains_a)
    records_b = records(top_b, chains_b)
    ids_a = [chain_id for chain_id, _ in records_a]
    ids_b = [chain_id for chain_id, _ in records_b]
    named_a = all(chain_id is not None for chain_id in ids_a)
    named_b = all(chain_id is not None for chain_id in ids_b)
    unique_a = len(set(ids_a)) == len(ids_a)
    unique_b = len(set(ids_b)) == len(ids_b)

    if named_a and named_b and unique_a and unique_b and set(ids_a) == set(ids_b):
        by_id_b = dict(records_b)
        return [(sequence, by_id_b[chain_id]) for chain_id, sequence in records_a]
    return list(zip(chains_a, chains_b))
